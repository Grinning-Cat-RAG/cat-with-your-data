"""Hooks of the plugin (fast reply, Rabbit Hole parsers and documents) and the dataset parser."""
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

m = support = f = None

CHART = {"sql": "SELECT region, amount FROM sales", "chart_type": "bar", "x": "region", "y": ["amount"], "title": "T"}


def setUpModule():
    global m, support, f
    import support as support_module
    support = support_module
    m = support.load()
    f = m.fakes


class FastReplyTest(unittest.TestCase):
    def setUp(self):
        self.agent_key = f"a-{time.monotonic_ns()}"
        support.put_file(self.agent_key, "sales.csv", b"region,amount\nN,1\nS,2\n")
        self.settings = support.default_settings(ds_type="CSV", host="sales.csv")

    def reply(self, llm, workflow=None, settings=None):
        cat = f.make_cat(llm, settings or self.settings, workflow=workflow or f.Workflow(), agent_key=self.agent_key)
        output = support.run(m.query_cat.agent_fast_reply.function(cat))
        if output is not None:
            # the answer goes through the hooks of the core, as the ones of the agent
            sent = support.core_sends(cat, output)
            self.assertEqual(sent.text, output.output, "the user gets the charts")
        return output, cat

    def chart_llm(self, answer="N and S"):
        return f.ScriptedChatModel(script=[f.tool_call("draw_chart", CHART), f.AIMessage(content=answer)])

    def test_answer_with_chart(self):
        output, cat = self.reply(self.chart_llm(), f.Workflow(output="Here it is"))
        self.assertTrue(output.output.startswith("Here it is\n\n![T](data:image/png;base64,"))
        # regression: the base64 chart was saved in the history, which the core sends to the LLM
        self.assertEqual(cat.working_memory.history[-1].content.text, "Here it is\n\n[chart: T]")
        self.assertEqual(cat.state["saved_history"], [("assistant", "Here it is\n\n[chart: T]")])
        self.assertEqual(len(cat.working_memory.history), 1, "the answer is stored once (by the core)")

    def test_final_llm_failure_returns_the_agent_answer(self):
        # regression: an exception of the final LLM call lost both the answer and the chart
        output, cat = self.reply(self.chart_llm(), f.Workflow(error=RuntimeError("LLM down")))
        self.assertTrue(output.output.startswith("N and S\n\n![T](data:image/png;base64,"))
        self.assertEqual(cat.working_memory.history[-1].content.text, "N and S\n\n[chart: T]")

    def test_final_llm_error_flag_returns_the_agent_answer(self):
        output, _ = self.reply(self.chart_llm(), f.Workflow(output="sorry", with_llm_error=True))
        self.assertTrue(output.output.startswith("N and S\n\n![T]"))
        output, _ = self.reply(f.ScriptedChatModel(script=[f.AIMessage(content="plain")]), f.Workflow(output="  "))
        self.assertEqual(output.output, "plain")

    def test_no_datasource(self):
        output, cat = self.reply(f.ScriptedChatModel(script=[]), settings=support.default_settings())
        self.assertIsNone(output)
        self.assertEqual(cat.working_memory.history, [])


class RabbitHoleParsersTest(unittest.TestCase):
    def test_parsers_are_wrapped(self):
        text_parser = object()
        cat = f.make_cat(settings=support.default_settings(max_upload_size_mb=2))
        handlers = support.run(m.query_cat.rabbithole_instantiates_parsers.function({"text/plain": text_parser}, cat))
        self.assertIs(handlers["text/plain"].fallback, text_parser)
        self.assertIsNone(handlers["text/csv"].fallback)
        self.assertEqual(handlers["application/x-sqlite3"].max_bytes, 2 * 1024 * 1024)

    def test_capture_disabled(self):
        cat = f.make_cat(settings=support.default_settings(capture_rabbithole_uploads=False))
        self.assertEqual(support.run(m.query_cat.rabbithole_instantiates_parsers.function({"a": 1}, cat)), {"a": 1})

    def test_settings_failure(self):
        cat = f.make_cat()
        cat.state["settings"] = RuntimeError("redis down")
        handlers = support.run(m.query_cat.rabbithole_instantiates_parsers.function({}, cat))
        self.assertIn("text/csv", handlers)


class DatasetParserTest(unittest.TestCase):
    def parse(self, content, name, fallback=None, max_bytes=None):
        from langchain_core.documents.base import Blob
        parser = m.parsers.DatasetBlobParser(fallback=fallback, max_bytes=max_bytes)
        return list(parser.lazy_parse(Blob(data=content, path=name, mimetype="text/plain")))

    def fallback(self):
        from langchain_core.documents import Document
        parser = mock.Mock()
        parser.lazy_parse.side_effect = lambda blob: iter([Document(page_content="as text")])
        return parser

    def test_dataset(self):
        doc, = self.parse(b"a,b\n1,2\n", "x.csv")
        self.assertIn("Dataset 'x.csv' (CSV)", doc.page_content)
        self.assertEqual(doc.metadata[m.parsers.PAYLOAD_KEY], b"a,b\n1,2\n")
        doc, = self.parse(support.sqlite_bytes({"t": (["v"], [(1,)])}), "x.bin")
        self.assertEqual(doc.metadata[m.parsers.KIND_KEY], "sqlite")

    def test_other_files_go_to_the_original_parser(self):
        self.assertEqual([d.page_content for d in self.parse(b"hello", "notes.txt", self.fallback())], ["as text"])
        self.assertEqual(self.parse(b"hello", "notes.txt"), [])
        self.assertEqual([d.page_content for d in self.parse(b'a,b\n"x,1\n', "bad.csv", self.fallback())], ["as text"])

    def test_too_large(self):
        # regression: the description said "available for data analysis" but the dataset was not registered
        with self.assertRaises(m.datasets.DatasetError) as error:
            self.parse(b"a,b\n1,2\n", "big.csv", self.fallback(), max_bytes=5)
        self.assertIn("exceeds the maximum dataset size", str(error.exception))


class SplitDocumentsTest(unittest.TestCase):
    def docs(self, *items):
        from langchain_core.documents import Document
        return [Document(page_content=text, metadata=dict(metadata)) for text, metadata in items]

    def dataset_doc(self, name="sales.csv", payload=b"a,b\n1,2\n", kind="csv"):
        return (f"Dataset '{name}'", {m.parsers.KIND_KEY: kind, m.parsers.NAME_KEY: name, m.parsers.PAYLOAD_KEY: payload})

    def stray_cat(self, notifier=None):
        from cat import StrayCat
        cat = mock.MagicMock(spec=StrayCat)
        cat.agent_key = f"agent-{time.monotonic_ns()}"
        cat.id = "chat-9"
        cat.user = SimpleNamespace(id="user-9")
        plugin = SimpleNamespace(load_settings=mock.AsyncMock(return_value=support.default_settings()))
        cat.mad_hatter = SimpleNamespace(get_plugin=lambda: plugin)
        cat.notifier = notifier or SimpleNamespace(send_notification=mock.AsyncMock())
        cat.file_manager = m.file_manager
        return cat

    def split(self, docs, cat):
        return support.run(m.query_cat.before_rabbithole_splits_documents.function(docs, cat))

    def test_other_documents_untouched(self):
        docs = self.docs(("text", {"source": "a.txt"}))
        self.assertIs(self.split(docs, self.stray_cat()), docs)

    def test_chat_dataset(self):
        cat = self.stray_cat()
        docs = self.split(self.docs(self.dataset_doc(), ("text", {})), cat)
        self.assertEqual(docs[0].metadata, {})
        self.assertIn("can be queried in this conversation (tables: sales)", docs[0].page_content)
        cat.notifier.send_notification.assert_awaited_once()
        store = m.datasets.DatasetStore(m.file_manager, cat.agent_key, "chat-9", "user-9")
        self.assertEqual([d.name for d in store.list_datasets()], ["sales.csv"])

    def test_agent_dataset_without_chat(self):
        cat = SimpleNamespace(agent_key=f"agent-{time.monotonic_ns()}", mad_hatter=self.stray_cat().mad_hatter,
                              file_manager=m.file_manager)
        docs = self.split(self.docs(self.dataset_doc(name=None)), cat)
        self.assertIn("in every conversation", docs[0].page_content)
        self.assertEqual([d.scope for d in m.datasets.DatasetStore(m.file_manager, cat.agent_key).list_datasets()], ["shared"])

    def test_payload_never_reaches_the_memory(self):
        # regression: an unexpected error escaping the hook restored the documents with the raw bytes
        cat = self.stray_cat(notifier=SimpleNamespace(send_notification=mock.AsyncMock(side_effect=RuntimeError("ws closed"))))
        outcomes = [RuntimeError("unexpected"), m.datasets.DatasetInfo("b.csv", "csv", "chat", 1, 0.0, {"b": ["x"]}),
                    m.datasets.DatasetError("The file is empty.")]
        with mock.patch.object(m.datasets.DatasetStore, "add", side_effect=outcomes, autospec=True):
            docs = self.split(self.docs(self.dataset_doc("a.csv"), self.dataset_doc("b.csv"), self.dataset_doc("c.csv", payload=None)), cat)
        for doc in docs:
            self.assertFalse({m.parsers.KIND_KEY, m.parsers.NAME_KEY, m.parsers.PAYLOAD_KEY} & set(doc.metadata))
        # the datasets that could not be stored are not described in the memory (the notifier here cannot send errors)
        self.assertEqual([d.page_content.splitlines()[0] for d in docs], ["Dataset 'b.csv'"])
        self.assertIn("can be queried", docs[0].page_content)


class EndpointsTest(unittest.TestCase):
    def info(self, chat_id=None, permissions=None, agent=True, file_manager=None, user_id="u1", key=None):
        from fastapi import UploadFile  # noqa: F401 - FastAPI is a dependency of the core
        key = key or f"agent-{time.monotonic_ns()}"
        plugin = SimpleNamespace(load_settings=mock.AsyncMock(return_value=support.default_settings(max_upload_size_mb=1)))
        cheshire_cat = SimpleNamespace(
            agent_key=key, mad_hatter=SimpleNamespace(get_plugin=lambda: plugin), file_manager=file_manager or m.file_manager,
        ) if agent else None
        return SimpleNamespace(
            cheshire_cat=cheshire_cat,
            stray_cat=SimpleNamespace(id=chat_id) if chat_id else None,
            user=SimpleNamespace(id=user_id, permissions=permissions if permissions is not None else {"UPLOAD": ["WRITE"]}),
        )

    def upload(self, info, name="s.csv", content=b"a\n1\n", size=None):
        import io
        from fastapi import UploadFile
        file = UploadFile(io.BytesIO(content), filename=name, size=len(content) if size is None else size)
        return support.run(m.endpoints.upload_dataset.function(file=file, info=info))

    def test_upload_list_delete_in_a_chat(self):
        info = self.info(chat_id="c1", permissions={})
        response = self.upload(info)
        self.assertEqual((response.name, response.scope, response.tables), ("s.csv", "chat", {"s": ["a"]}))
        listed = support.run(m.endpoints.list_datasets.function(info=info))
        self.assertEqual([d.name for d in listed.datasets], ["s.csv"])
        self.assertTrue(support.run(m.endpoints.delete_dataset.function(name="s.csv", info=info)).deleted)
        with self.assertRaises(m.endpoints.CustomNotFoundException):
            support.run(m.endpoints.delete_dataset.function(name="s.csv", info=info))

    def test_shared_datasets_need_upload_permission(self):
        with self.assertRaises(m.endpoints.CustomForbiddenException):
            self.upload(self.info(permissions={"CHAT": ["WRITE"]}))
        with self.assertRaises(m.endpoints.CustomForbiddenException):
            support.run(m.endpoints.delete_dataset.function(name="x.csv", info=self.info(permissions={})))
        self.assertEqual(self.upload(self.info()).scope, "shared")

    def test_validation_errors(self):
        with self.assertRaises(m.endpoints.CustomValidationException):
            self.upload(self.info(chat_id="c"), size=2 * 1024 * 1024)
        with self.assertRaises(m.endpoints.CustomValidationException):
            self.upload(self.info(chat_id="c"), name="notes.txt")
        # regression: without an agent the endpoints failed with an AttributeError (HTTP 500)
        for call in (lambda i: self.upload(i), lambda i: support.run(m.endpoints.list_datasets.function(info=i)),
                     lambda i: support.run(m.endpoints.delete_dataset.function(name="x", info=i))):
            with self.assertRaises(m.endpoints.CustomValidationException):
                call(self.info(agent=False))

    def test_datasets_of_the_users_of_a_conversation(self):
        # the chat id is chosen by the client: two users may use the same one, and never see each other's datasets
        key = f"agent-{time.monotonic_ns()}"
        self.upload(self.info(chat_id="c1", permissions={}, user_id="u1", key=key), name="mine.csv")
        other = self.info(chat_id="c1", permissions={}, user_id="u2", key=key)
        self.assertEqual(support.run(m.endpoints.list_datasets.function(info=other)).datasets, [])
        with self.assertRaises(m.endpoints.CustomNotFoundException):
            support.run(m.endpoints.delete_dataset.function(name="mine.csv", info=other))

    def test_conversations_whose_id_is_not_a_folder_name(self):
        for chat_id in ("a/b", "..", "."):
            with self.subTest(chat_id=chat_id):
                info = self.info(chat_id=chat_id, permissions={})
                with self.assertRaises(m.endpoints.CustomValidationException):
                    self.upload(info)
                with self.assertRaises(m.endpoints.CustomValidationException):
                    support.run(m.endpoints.delete_dataset.function(name="s.csv", info=info))
                self.assertEqual(support.run(m.endpoints.list_datasets.function(info=info)).datasets, [])

    def test_settings_failure_uses_the_defaults(self):
        info = self.info(chat_id="c")
        info.cheshire_cat.mad_hatter.get_plugin().load_settings.side_effect = RuntimeError("redis down")
        self.assertEqual(self.upload(info).name, "s.csv")

    def test_file_manager_without_storage(self):
        # the default file manager of the core (Dummy) keeps nothing: the upload is refused with an explicit message
        with self.assertRaises(m.endpoints.CustomValidationException) as error:
            self.upload(self.info(chat_id="c", file_manager=m.fakes.DummyFileManager()))
        self.assertIn("configure a file manager", str(error.exception))

    def test_charts_are_delivered_only_inline(self):
        self.assertFalse(hasattr(m.endpoints, "get_chart"))
        self.assertNotIn("chart_delivery", m.settings.MySettings.model_fields)
        self.assertNotIn("public_base_url", m.settings.MySettings.model_fields)


class SettingsTest(unittest.TestCase):
    def test_examples_are_valid(self):
        # regression: ds_type "" and port "" (file datasources, uploaded datasets only) could not be saved
        for path in sorted((support.PLUGIN_DIR / "settings_examples").glob("*.json")):
            with self.subTest(example=path.name):
                import json
                m.settings.MySettings.model_validate(json.loads(path.read_text()))
        m.settings.MySettings.model_validate(m.settings.MySettings().model_dump(mode="json"))

    def test_bounds(self):
        from pydantic import ValidationError
        for field, value in (("chart_max_rows", 0), ("thought_max_rows", 0), ("max_upload_size_mb", 0),
                             ("chat_datasets_ttl_hours", -1)):
            with self.subTest(field=field), self.assertRaises(ValidationError):
                m.settings.MySettings(**{field: value})

    def test_crypt_round_trip(self):
        crypto = SimpleNamespace(encrypt=lambda v: f"enc:{v}", decrypt=lambda v: v[4:] if v.startswith("enc:") else 1 / 0)
        encrypted = m.crypt.encrypt_secrets({"password": "pw", "username": "u", "host": ""}, crypto)
        self.assertEqual(encrypted, {"password": "enc:pw", "username": "u", "host": ""})
        # regression: SECRET_SETTINGS was a string, so the password was encrypted but never decrypted
        self.assertEqual(m.crypt.decrypt_secrets(encrypted, crypto), ({"password": "pw", "username": "u", "host": ""}, []))
        self.assertEqual(m.crypt.decrypt_secrets({"password": "plain"}, crypto), ({"password": ""}, ["password"]))
        self.assertEqual(m.crypt.decrypt_secrets({"password": ""}, crypto), ({"password": ""}, []))

    def test_load_and_save_hooks(self):
        store = {}

        async def get_setting(agent_id, plugin_id):
            return store.get((agent_id, plugin_id))

        async def update_setting(agent_id, plugin_id, value):
            store[(agent_id, plugin_id)] = dict(value)
            return dict(value)

        crypto = SimpleNamespace(encrypt=lambda v: f"enc:{v}", decrypt=lambda v: v[4:] if v.startswith("enc:") else 1 / 0)
        cruds = SimpleNamespace(get_setting=get_setting, update_setting=update_setting)
        with mock.patch.object(m.settings, "crud_plugins", cruds), mock.patch.object(m.settings, "StringCrypto", lambda: crypto):
            loaded = support.run(m.settings.load_settings.function("p", "a"))
            self.assertEqual(loaded["charts"], "on_request")
            saved = support.run(m.settings.save_settings.function("p", {"ds_type": "CSV", "password": "pw"}, "a"))
            self.assertEqual(store[("a", "p")]["password"], "enc:pw")
            self.assertEqual((saved["password"], saved["chart_max_rows"]), ("pw", 1000))  # old settings get the new fields
            store[("a", "p")]["password"] = "not encrypted"
            self.assertEqual(support.run(m.settings.load_settings.function("p", "a"))["password"], "")
        self.assertEqual(m.settings.settings_schema.function()["title"], "MySettings")


if __name__ == "__main__":
    unittest.main()
