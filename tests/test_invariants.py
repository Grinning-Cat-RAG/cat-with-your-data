"""Stateful, property-based tests of the business invariants, with fault injection (requires ``hypothesis``).

Invariants:

- isolation: a user in a conversation sees exactly the shared datasets and its own ones in the conversation (its own
  win on name clashes), never the datasets of another conversation, nor those of another user in the same conversation
  (the chat id is chosen by the client);
- conversations: deleting a conversation (as the core does, removing ``<agent>/<chat>``) removes the datasets of all
  its users and nothing else, whatever the chat id (also ``cat_with_your_data``, ``system``, ``_chats``); the shared
  datasets survive the deletion of any conversation;
- freshness: querying the datasets of a conversation returns the content of the latest successful upload of each
  visible dataset, whatever the clocks of the instances;
- atomicity: a failed upload (storage errors, invalid files, uploads interrupted in the middle of the write) or a failed
  removal leaves the visible datasets unchanged;
- expiration: the cleanup never removes the shared datasets, the datasets of the conversation running it, nor those
  of a conversation that uploaded or used its datasets within the TTL;
- snapshot: a request keeps reading the content it started with until it ends, whatever is uploaded, removed or
  expired meanwhile on any instance;
- read-only: the queries of the agent can neither modify the datasets nor open other files (ATTACH);
- tenants: the datasets of another agent, even with the same chat ids and the same file manager, are never visible,
  and destroying another agent never touches the datasets of this one;
- Rabbit Hole: a dataset uploaded through the Rabbit Hole is queryable as soon as its description is ingested, and the
  description of a dataset that could not be stored never reaches the memory;
- local disk: with a remote file manager, nothing is written on the local disk of the instance;
- replies: the reply contains every chart drawn by the agent (inline), the answer of the agent survives a failure of
  the final LLM call, the answer is stored once in the history (by the core) and the history never contains base64
  images;
- numbers: the numbers of a CSV file separated by ";" are read back unchanged, in European or in the usual format.
"""
import contextlib
from types import SimpleNamespace
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

m = support = None
#: "c1@u2" is the conversation c1 of the user u2; the other conversations are of the user u1
SCOPES = ("shared", "c1", "c2", "c1@u2")
CHATS = SCOPES[1:]
NAMES = ("a.csv", "b.csv", "a.sqlite", "c.sqlite")


def setUpModule():
    global m, support
    import support as support_module
    support = support_module
    m = support.load()


def _hypothesis():
    # hypothesis keeps a cache in the working directory (the root of the core): use a temporary one
    # the same folder at every run (a new one each time would be left behind)
    os.environ.setdefault("HYPOTHESIS_STORAGE_DIRECTORY", os.path.join(tempfile.gettempdir(), "cat-plugins-hypothesis"))
    try:
        import hypothesis  # noqa: F401
    except ImportError:  # pragma: no cover - hypothesis is a test-only dependency
        raise unittest.SkipTest("hypothesis is not installed")
    from hypothesis import HealthCheck, settings, strategies as st
    from hypothesis.stateful import RuleBasedStateMachine, invariant, precondition, rule, run_state_machine_as_test
    return HealthCheck, settings, st, RuleBasedStateMachine, invariant, precondition, rule, run_state_machine_as_test


def chat_and_user(scope: str):
    chat, _, user = scope.partition("@")
    return chat, user or "u1"


def content_of(name: str, rows) -> bytes:
    if name.endswith(".csv"):
        return ("v\n" + "".join(f"{v}\n" for v in rows)).encode()
    return support.sqlite_bytes({f"s_{Path(name).stem}": (["v"], [(v,) for v in rows])})


def table_of(name: str) -> str:
    return m.datasets.table_name_from(name) if name.endswith(".csv") else f"s_{Path(name).stem}"


def read_all(engine, names):
    with engine.connect() as conn:
        return {
            name: [row[0] for row in conn.exec_driver_sql(f"SELECT v FROM {table_of(name)} ORDER BY rowid")]
            for name in names
        }


def build_datasets_machine():
    HealthCheck, settings, st, RuleBasedStateMachine, invariant, precondition, rule, _ = _hypothesis()
    rows_strategy = st.lists(st.integers(-5, 5), min_size=1, max_size=4)

    class DatasetsMachine(RuleBasedStateMachine):
        def __init__(self):
            super().__init__()
            self.agent = f"agent-{time.monotonic_ns()}"
            self.other = f"other:{time.monotonic_ns()}"  # another tenant, on the same file manager
            self.fm = m.fakes.ObjectStoreFileManager()
            self.model = {scope: {} for scope in SCOPES}  # scope -> {name: rows}
            self.other_model = {scope: {} for scope in SCOPES}
            self.snapshots = []  # (engine, expected content) of the requests still running
            self.recent = set()  # conversations that uploaded or used their datasets since time last passed
            self.interrupted = set()  # (scope, name) with the copy of an interrupted upload
            self.clock = m.fakes.FakeClock()
            self.patches = [mock.patch.object(m.datasets, "time", self.clock),
                            mock.patch.object(m.query_agent, "time", self.clock)]
            for patch in self.patches:
                patch.start()

        def store(self, scope, agent=None):
            if scope == "shared":
                return m.datasets.DatasetStore(self.fm, agent or self.agent)
            return m.datasets.DatasetStore(self.fm, agent or self.agent, *chat_and_user(scope))

        def visible(self, chat):
            return {**self.model["shared"], **self.model[chat]}

        def workspace_engine(self, chat):
            chat_id, user_id = chat_and_user(chat)
            cat = m.fakes.make_cat(agent_key=self.agent, chat_id=chat_id, user_id=user_id, file_manager=self.fm)
            return m.query_agent.QueryCatAgent(cat)._uploaded_datasets_engine()

        # ------------------------------------------------------------------------------------------------------------
        @rule(scope=st.sampled_from(SCOPES), name=st.sampled_from(NAMES), rows=rows_strategy,
              skew=st.sampled_from([0, -3600, 3600]))
        def upload(self, scope, name, rows, skew):
            # the upload may be received by an instance whose clock is late or ahead
            self.clock.offset += skew
            try:
                self.store(scope).add(name, content_of(name, rows), shared=scope == "shared")
            finally:
                self.clock.offset -= skew
            self.model[scope][name] = rows
            self.recent.add(scope)
            self.interrupted.discard((scope, name))  # the previous copies are removed

        @rule(scope=st.sampled_from(SCOPES), name=st.sampled_from(NAMES), rows=rows_strategy,
              fault=st.sampled_from(["write-error", "write-refused", "validation", "empty", "wrong-kind", "interrupted"]))
        def failed_upload(self, scope, name, rows, fault):
            store = self.store(scope)
            content = content_of(name, rows)
            folder = store._scope_dir(scope == "shared")
            if fault == "interrupted":
                # the instance died in the middle of the write (after recording the activity, as the store does): a
                # partial copy with a valid name, never visible
                real_write = self.fm.write_file

                def die_while_writing(data, stored, where):
                    if "--" in stored:
                        real_write(data[:-1], stored, where)
                        raise SystemExit("the instance died")
                    return real_write(data, stored, where)

                with mock.patch.object(self.fm, "write_file", die_while_writing):
                    try:
                        store.add(name, content, shared=scope == "shared")
                    except SystemExit:
                        pass
                self.interrupted.add((scope, name))
                self.recent.add(scope)
                return
            patches = {
                "write-error": mock.patch.object(self.fm, "_write_file", side_effect=OSError("storage down")),
                "write-refused": mock.patch.object(self.fm, "write_file", return_value=False),
                "validation": mock.patch.object(m.datasets, "sqlite_tables" if name.endswith(".sqlite") else "read_csv",
                                                side_effect=m.datasets.DatasetError("invalid")),
            }
            if fault == "empty":
                content = b""
            if fault == "wrong-kind":
                name = name.replace(".csv", ".txt") if name.endswith(".csv") else name
                content = b"not a sqlite file" if name.endswith(".sqlite") else content
            with patches.get(fault, contextlib.nullcontext()):
                try:
                    store.add(name, content, shared=scope == "shared")
                except m.datasets.DatasetError:
                    pass
                else:  # pragma: no cover - every fault makes the upload fail
                    raise AssertionError(f"the upload should have failed ({fault})")

        @rule(scope=st.sampled_from(SCOPES), name=st.sampled_from(NAMES), fault=st.booleans())
        def remove(self, scope, name, fault):
            store = self.store(scope)
            if fault:
                with mock.patch.object(self.fm, "_remove_file", side_effect=OSError("storage down")):
                    try:
                        store.remove(name, shared=scope == "shared")
                    except OSError:
                        pass
                return
            removed = store.remove(name, shared=scope == "shared")
            assert removed == (name in self.model[scope] or (scope, name) in self.interrupted)
            self.model[scope].pop(name, None)
            self.interrupted.discard((scope, name))

        @rule(scope=st.sampled_from(SCOPES), name=st.sampled_from(NAMES), rows=rows_strategy)
        def other_tenant_uploads(self, scope, name, rows):
            self.store(scope, self.other).add(name, content_of(name, rows), shared=scope == "shared")
            self.other_model[scope][name] = rows

        @rule(chat=st.sampled_from(["c1", "c2", "cat_with_your_data", "system", "_chats", "c1@u2"]))
        def core_deletes_conversation(self, chat):
            # as the core does (cat/core_plugins/conversation_history/endpoints.py): the folder of the conversation
            # goes, for every user of the conversation (and so the snapshots of the requests running are unaffected)
            chat_id = chat_and_user(chat)[0]
            self.fm.remove_folder(f"{self.agent}/{chat_id}")
            for scope in CHATS:
                if chat_and_user(scope)[0] == chat_id:
                    self.model[scope] = {}
                    self.interrupted = {(s, n) for s, n in self.interrupted if s != scope}

        @rule()
        def other_tenant_is_destroyed(self):
            # as the core does (CheshireCat.destroy): the folder of the agent, then the hooks
            self.fm.remove_folder(self.other)
            support.run(m.query_cat.after_cheshire_cat_destroy.function(self.other, SimpleNamespace(file_manager=self.fm)))
            self.other_model = {scope: {} for scope in SCOPES}

        @rule(scope=st.sampled_from(SCOPES), name=st.sampled_from(NAMES), rows=rows_strategy, fail=st.booleans())
        def rabbit_hole_upload(self, scope, name, rows, fail):
            from langchain_core.documents import Document
            from cat import StrayCat

            plugin = SimpleNamespace(load_settings=mock.AsyncMock(
                return_value=support.default_settings(chat_datasets_ttl_hours=0)))
            if scope == "shared":
                cat = SimpleNamespace(agent_key=self.agent, file_manager=self.fm)
            else:
                cat = mock.MagicMock(spec=StrayCat)
                chat_id, user_id = chat_and_user(scope)
                cat.agent_key, cat.id, cat.file_manager = self.agent, chat_id, self.fm
                cat.user = SimpleNamespace(id=user_id)
                cat.notifier = SimpleNamespace(send_notification=mock.AsyncMock(), send_error=mock.AsyncMock())
            cat.mad_hatter = SimpleNamespace(get_plugin=lambda: plugin)
            docs = [Document(page_content=f"Dataset '{name}'", metadata={
                m.parsers.KIND_KEY: "sqlite" if name.endswith(".sqlite") else "csv", m.parsers.NAME_KEY: name,
                m.parsers.PAYLOAD_KEY: content_of(name, rows)})]
            failing = mock.patch.object(self.fm, "_write_file", side_effect=OSError("storage down"))
            with failing if fail else contextlib.nullcontext():
                result = support.run(m.query_cat.before_rabbithole_splits_documents.function(docs, cat))
            if fail:
                assert result == [], "the description of a dataset not stored never reaches the memory"
                if scope != "shared":
                    cat.notifier.send_error.assert_awaited_once()
                return
            assert len(result) == 1 and "can be queried" in result[0].page_content
            assert not {m.parsers.KIND_KEY, m.parsers.NAME_KEY, m.parsers.PAYLOAD_KEY} & set(result[0].metadata)
            self.model[scope][name] = rows
            self.recent.add(scope)
            self.interrupted.discard((scope, name))

        @rule(chat=st.sampled_from(CHATS))
        def open_request(self, chat):
            visible = self.visible(chat)
            engine = self.workspace_engine(chat)
            if not visible:
                assert engine is None
                return
            self.recent.add(chat)
            self.snapshots.append((engine, visible))
            assert read_all(engine, visible) == visible

        @precondition(lambda self: self.snapshots)
        @rule(data=st.data())
        def close_request(self, data):
            engine, expected = self.snapshots.pop(data.draw(st.integers(0, len(self.snapshots) - 1)))
            assert read_all(engine, expected) == expected
            engine.dispose()

        @rule()
        def time_passes(self):
            self.clock.offset += 10 * 3600
            self.recent.clear()

        @rule(scope=st.sampled_from(SCOPES))
        def cleanup(self, scope):
            self.store(scope).cleanup_expired(5)
            for chat in CHATS:
                if chat in self.recent or chat == scope:
                    continue  # protected: the isolation invariant checks that nothing was removed
                self.model[chat] = {}
                self.interrupted = {(s, n) for s, n in self.interrupted if s != chat}

        # ------------------------------------------------------------------------------------------------------------
        @invariant()
        def other_tenant_is_isolated(self):
            for chat in CHATS:
                listed = {d.name for d in self.store(chat, self.other).list_datasets()}
                assert listed == set(self.other_model["shared"]) | set(self.other_model[chat]), (chat, listed)

        @invariant()
        def isolation(self):
            for chat in CHATS:
                listed = {d.name: d.scope for d in self.store(chat).list_datasets()}
                expected = {name: "shared" for name in self.model["shared"]}
                expected.update({name: "chat" for name in self.model[chat]})
                assert listed == expected, (chat, listed, expected)

        @invariant()
        def freshness_and_read_only(self):
            for chat in CHATS:
                visible = self.visible(chat)
                store = self.store(chat)
                workspace = store.workspace()
                if not visible:
                    assert workspace is None
                    continue
                engine = m.data_engine.engine_from_bytes(workspace)
                try:
                    assert read_all(engine, visible) == visible
                    db = m.data_engine.sql_database(engine, cache=False)
                    name = next(iter(visible))
                    assert "Error" in db.run_no_throw(f"DELETE FROM {table_of(name)}")
                    assert "Error" in db.run_no_throw("ATTACH DATABASE ':memory:' AS other")
                finally:
                    engine.dispose()

        @invariant()
        def requests_keep_their_snapshot(self):
            for engine, expected in self.snapshots:
                assert read_all(engine, expected) == expected

        def teardown(self):
            for engine, _ in self.snapshots:
                engine.dispose()
            for patch in self.patches:
                patch.stop()

    DatasetsMachine.TestCase.settings = settings(
        max_examples=int(os.environ.get("CWYD_EXAMPLES", "60")), stateful_step_count=25, deadline=None, database=None,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
    )
    return DatasetsMachine


def build_replies_machine():
    HealthCheck, settings, st, RuleBasedStateMachine, invariant, _, rule, _ = _hypothesis()
    f = m.fakes

    class RepliesMachine(RuleBasedStateMachine):
        def __init__(self):
            super().__init__()
            self.agent = f"agent-replies-{time.monotonic_ns()}"
            support.put_file(self.agent, "sales.csv", b"region,amount\nN,1\nS,2\nE,3\n")
            self.history = []

        @rule(charts=st.integers(0, 4), answer=st.sampled_from(["", "N leads", "braces {x}"]),
              final=st.sampled_from(["ok", "empty", "llm-error", "exception"]),
              react=st.booleans())
        def ask(self, charts, answer, final, react):
            chart = {"sql": "SELECT region, amount FROM sales", "chart_type": "bar", "x": "region", "y": ["amount"]}
            if react:
                script = [f.react_step("draw_chart", {**chart, "title": f"C{i}"}) for i in range(charts)]
                script.append(f.react_final(answer or "done"))
                llm = f.ReactChatModel(script=script)
                answer = answer or "done"
            else:
                script = [f.tool_call("draw_chart", {**chart, "title": f"C{i}"}, str(i)) for i in range(charts)]
                script.append(f.AIMessage(content=answer))
                llm = f.ScriptedChatModel(script=script)
            workflow = {
                "ok": f.Workflow(output="final text"), "empty": f.Workflow(output=""),
                "llm-error": f.Workflow(output="x", with_llm_error=True), "exception": f.Workflow(error=RuntimeError("down")),
            }[final]
            settings_ = support.default_settings(ds_type="CSV", host="sales.csv")
            cat = f.make_cat(llm, settings_, workflow=workflow, history=self.history, agent_key=self.agent)
            before = len(cat.working_memory.history)
            output = support.run(m.query_cat.agent_fast_reply.function(cat))
            assert len(cat.working_memory.history) == before, "the plugin never stores the answer itself"
            if output is not None:
                # the core runs the hooks of the turn on the answer (the conversation history stores it)
                sent = support.core_sends(cat, output)
                assert sent.text == output.output, "the user gets the charts"
                assert len(cat.working_memory.history) == before + 1, "the answer is stored once"
            self.history = cat.working_memory.history

            drawn = min(charts, m.chart_tool.MAX_CHARTS_PER_ANSWER)
            if not drawn and not answer:
                assert output is None
                return
            text = output.output
            assert text.count("data:image/png;base64,") == drawn, (text[:200], drawn)
            expected_text = "final text" if final == "ok" else answer
            assert text.startswith(expected_text) if expected_text else text.startswith("![")
            saved = self.history[-1].content.text
            assert "base64," not in saved
            assert saved.count("[chart: ") == drawn

        @invariant()
        def history_has_no_images(self):
            assert all("base64," not in item.content.text for item in self.history)

    RepliesMachine.TestCase.settings = settings(
        max_examples=int(os.environ.get("CWYD_EXAMPLES", "60")) // 3 or 1, stateful_step_count=6, deadline=None, database=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    return RepliesMachine


class DatasetsInvariantsTest(unittest.TestCase):
    def test_datasets_state_machine(self):
        *_, run_state_machine_as_test = _hypothesis()
        machine = build_datasets_machine()
        run_state_machine_as_test(machine, settings=machine.TestCase.settings)


class RepliesInvariantsTest(unittest.TestCase):
    def test_replies_state_machine(self):
        *_, run_state_machine_as_test = _hypothesis()
        machine = build_replies_machine()
        run_state_machine_as_test(machine, settings=machine.TestCase.settings)


class NumbersPropertyTest(unittest.TestCase):
    def test_semicolon_numbers_round_trip(self):
        HealthCheck, settings, st, *_ = _hypothesis()
        from hypothesis import given

        def european(value: float) -> str:
            text = f"{value:,.2f}"  # 1,234.50
            return text.replace(",", "_").replace(".", ",").replace("_", ".")

        @settings(max_examples=int(os.environ.get("CWYD_EXAMPLES", "60")) * 3, deadline=None, database=None)
        @given(values=st.lists(st.decimals(min_value=-10**7, max_value=10**7, places=2, allow_nan=False), min_size=1, max_size=8),
               fmt=st.sampled_from(["european", "usual"]))
        def check(values, fmt):
            values = [float(v) for v in values]
            written = [european(v) if fmt == "european" else f"{v:.2f}" for v in values]
            df = m.datasets.read_csv(("name;value\n" + "".join(f"r{i};{w}\n" for i, w in enumerate(written))).encode(), "x.csv")
            assert [round(float(v), 2) for v in df["value"]] == [round(v, 2) for v in values], (written, df["value"].tolist())

        check()


if __name__ == "__main__":
    unittest.main()
