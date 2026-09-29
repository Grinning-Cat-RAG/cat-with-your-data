"""Stateful, property-based tests of the business invariants, with fault injection (requires ``hypothesis``).

Invariants:

- isolation: a conversation sees exactly the shared datasets and its own ones (its own win on name clashes), never
  the datasets of another conversation;
- freshness: querying the datasets of a conversation returns the content of the latest successful upload of each
  visible dataset, also when a dataset is uploaded again with the same name, size and timestamp;
- atomicity: a failed upload or removal (disk errors, invalid files) leaves the visible datasets unchanged;
- expiration: the cleanup never removes the shared datasets, the datasets of the conversation running it, nor those
  of a conversation that uploaded or used its datasets within the TTL;
- snapshot: a request that opened the datasets keeps reading the same content until it ends, whatever is uploaded
  or removed meanwhile;
- read-only: the queries of the agent can neither modify the datasets nor open other files (ATTACH);
- replies: the reply contains every chart drawn by the agent, the answer of the agent survives a failure of the final
  LLM call, and the history never contains base64 images;
- numbers: the numbers of a CSV file separated by ";" are read back unchanged, in European or in the usual format.
"""
import contextlib
import os
import shutil
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

m = support = None
SCOPES = ("shared", "c1", "c2")
NAMES = ("a.csv", "b.csv", "a.sqlite", "c.sqlite")


def setUpModule():
    global m, support
    import support as support_module
    support = support_module
    m = support.load()


def _hypothesis():
    # hypothesis keeps a cache in the working directory (the root of the core): use a temporary one
    os.environ.setdefault("HYPOTHESIS_STORAGE_DIRECTORY", tempfile.mkdtemp(prefix="cwyd-hypothesis-"))
    try:
        import hypothesis  # noqa: F401
    except ImportError:  # pragma: no cover - hypothesis is a test-only dependency
        raise unittest.SkipTest("hypothesis is not installed")
    from hypothesis import HealthCheck, settings, strategies as st
    from hypothesis.stateful import RuleBasedStateMachine, invariant, precondition, rule, run_state_machine_as_test
    return HealthCheck, settings, st, RuleBasedStateMachine, invariant, precondition, rule, run_state_machine_as_test


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
            self.model = {scope: {} for scope in SCOPES}  # scope -> {name: rows}
            self.snapshots = []  # (engine, expected content) of the requests still open
            self.recent = set()  # conversations that uploaded or used their datasets since time last passed

        def store(self, scope):
            return m.datasets.DatasetStore(self.agent, None if scope == "shared" else scope)

        def visible(self, chat):
            return {**self.model["shared"], **self.model[chat]}

        # ------------------------------------------------------------------------------------------------------------
        @rule(scope=st.sampled_from(SCOPES), name=st.sampled_from(NAMES), rows=rows_strategy,
              same_timestamp=st.booleans())
        def upload(self, scope, name, rows, same_timestamp):
            store = self.store(scope)
            folder = store._scope_dir(scope == "shared") / "files"
            previous = [p.stat() for p in folder.glob(f"*--{name}")] if folder.is_dir() else []
            store.add(name, content_of(name, rows), shared=scope == "shared")
            if same_timestamp and previous:
                # the new copy gets the timestamp of the previous one (coarse clocks, fast clients)
                current, = folder.glob(f"*--{name}")
                os.utime(current, ns=(previous[0].st_atime_ns, previous[0].st_mtime_ns))
            self.model[scope][name] = rows
            self.recent.add(scope)

        @rule(scope=st.sampled_from(SCOPES), name=st.sampled_from(NAMES), rows=rows_strategy,
              fault=st.sampled_from(["write", "replace", "validation", "empty", "wrong-kind"]))
        def failed_upload(self, scope, name, rows, fault):
            store = self.store(scope)
            content = content_of(name, rows)
            patches = {
                "write": mock.patch.object(m.datasets.Path, "write_bytes", side_effect=OSError("disk full")),
                "replace": mock.patch.object(m.datasets.Path, "replace", side_effect=OSError("disk full")),
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
                except (OSError, m.datasets.DatasetError):
                    pass
                else:  # pragma: no cover - every fault makes the upload fail
                    raise AssertionError(f"the upload should have failed ({fault})")

        @rule(scope=st.sampled_from(SCOPES), name=st.sampled_from(NAMES), fault=st.booleans())
        def remove(self, scope, name, fault):
            store = self.store(scope)
            if fault:
                with mock.patch.object(m.datasets.Path, "unlink", side_effect=OSError("busy")):
                    try:
                        store.remove(name, shared=scope == "shared")
                    except OSError:
                        pass
                return
            removed = store.remove(name, shared=scope == "shared")
            assert removed == (name in self.model[scope])
            self.model[scope].pop(name, None)

        @rule(chat=st.sampled_from(("c1", "c2")))
        def open_request(self, chat):
            visible = self.visible(chat)
            # the path of a request of the query agent
            agent = m.query_agent.QueryCatAgent(m.fakes.make_cat(agent_key=self.agent, chat_id=chat))
            engine = agent._uploaded_datasets_engine()
            if not visible:
                assert engine is None
                return
            if self.model[chat] or m.datasets.DatasetStore(self.agent, chat).chat_dir.is_dir():
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
            past = time.time() - 10 * 3600
            for path in m.datasets.DatasetStore(self.agent).agent_dir.rglob("*"):
                os.utime(path, (past, past))
            self.recent.clear()

        @rule(scope=st.sampled_from(SCOPES))
        def cleanup(self, scope):
            store = self.store(scope)
            store.cleanup_expired(5)
            for chat in ("c1", "c2"):
                if chat in self.recent or chat == scope:
                    continue  # protected: the isolation invariant checks that nothing was removed
                if not m.datasets.DatasetStore(self.agent, chat).chat_dir.is_dir():
                    self.model[chat] = {}

        @rule(chat=st.sampled_from(("c1", "c2")))
        def failed_workspace_build(self, chat):
            store = m.datasets.DatasetStore(self.agent, chat)
            with mock.patch.object(m.datasets, "read_csv", side_effect=m.datasets.DatasetError("broken")):
                try:
                    store.workspace_path()
                except m.datasets.DatasetError:
                    pass

        # ------------------------------------------------------------------------------------------------------------
        @invariant()
        def isolation(self):
            for chat in ("c1", "c2"):
                listed = {d.name: d.scope for d in m.datasets.DatasetStore(self.agent, chat).list_datasets()}
                expected = {name: "shared" for name in self.model["shared"]}
                expected.update({name: "chat" for name in self.model[chat]})
                assert listed == expected, (chat, listed, expected)

        @invariant()
        def freshness_and_read_only(self):
            for chat in ("c1", "c2"):
                visible = self.visible(chat)
                path = m.datasets.DatasetStore(self.agent, chat).workspace_path()
                if not visible:
                    assert path is None
                    continue
                engine = m.data_engine.engine_from_sqlite_file(path)
                try:
                    assert read_all(engine, visible) == visible
                    db = m.data_engine.sql_database(engine, cache=False)
                    name = next(iter(visible))
                    assert "Error" in db.run_no_throw(f"DELETE FROM {table_of(name)}")
                    assert "Error" in db.run_no_throw(f"ATTACH DATABASE '{path}' AS other")
                finally:
                    engine.dispose()

        def teardown(self):
            for engine, _ in self.snapshots:
                engine.dispose()
            shutil.rmtree(m.datasets.DatasetStore(self.agent).agent_dir, ignore_errors=True)

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
            tmp = Path(tempfile.mkdtemp())
            self.csv = tmp / "sales.csv"
            self.csv.write_text("region,amount\nN,1\nS,2\nE,3\n")
            self.history = []

        @rule(charts=st.integers(0, 4), answer=st.sampled_from(["", "N leads", "braces {x}"]),
              final=st.sampled_from(["ok", "empty", "llm-error", "exception"]),
              delivery=st.sampled_from(["inline", "url"]), react=st.booleans())
        def ask(self, charts, answer, final, delivery, react):
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
            settings_ = support.default_settings(ds_type="CSV", host=str(self.csv), chart_delivery=delivery)
            cat = f.make_cat(llm, settings_, workflow=workflow, history=self.history, agent_key="agent-replies")
            output = support.run(m.query_cat.agent_fast_reply.function(cat))
            self.history = cat.working_memory.history

            drawn = min(charts, m.chart_tool.MAX_CHARTS_PER_ANSWER)
            if not drawn and not answer:
                assert output is None
                return
            text = output.output
            marker = "data:image/png;base64," if delivery == "inline" else "/custom/cat-with-your-data/charts/"
            assert text.count(marker) == drawn, (text[:200], drawn)
            expected_text = "final text" if final == "ok" else answer
            assert text.startswith(expected_text) if expected_text else text.startswith("![")
            saved = self.history[-1].content.text
            assert "base64," not in saved
            assert saved.count("[chart: ") == (drawn if delivery == "inline" else 0)

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
