"""The query agent, run with scripted LLMs (ReAct and tool calling) on every kind of datasource."""
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

m = support = f = None

CHART = {"sql": "SELECT region, SUM(amount) AS total FROM sales GROUP BY region ORDER BY total DESC",
         "chart_type": "bar", "x": "region", "y": ["total"], "title": "Sales by region"}


def setUpModule():
    global m, support, f
    import support as support_module
    support = support_module
    m = support.load()
    f = m.fakes


class AgentTestCase(unittest.TestCase):
    def store(self, chat="chat-1"):
        return m.datasets.DatasetStore(m.file_manager, self.agent_key, chat)

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.csv = self.tmp / "sales.csv"
        self.csv.write_text("region;amount\nNorth;1.200,5\nSouth;300\nNorth;50\n")
        self.agent_key = f"agent-{time.monotonic_ns()}"

    def cat(self, llm, settings=None, **kwargs):
        kwargs.setdefault("agent_key", self.agent_key)
        return f.make_cat(llm, settings if settings is not None else support.default_settings(ds_type="CSV", host=str(self.csv)), **kwargs)

    def run_agent(self, cat):
        return support.run(m.query_agent.QueryCatAgent(cat).run())


class DatasourceTest(AgentTestCase):
    def test_tool_calling_on_csv_with_chart(self):
        llm = f.ScriptedChatModel(script=[
            f.tool_call("sql_db_list_tables", {"tool_input": ""}),
            f.tool_call("draw_chart", CHART, "2"),
            f.AIMessage(content="North leads with 1250.5"),
        ])
        result = self.run_agent(self.cat(llm))
        self.assertTrue(result.thought.startswith("North leads with 1250.5\n\n(The chart \"Sales by region\""))
        self.assertEqual(result.answer, "North leads with 1250.5")
        self.assertTrue(result.chart_markdown.startswith("![Sales by region](data:image/png;base64,"))
        tool_message = [msg for msg in llm.prompts[-1] if type(msg).__name__ == "ToolMessage"][-1]
        self.assertIn("| North    |  1250.5 |", tool_message.content)  # European numbers of the ";" CSV
        system = llm.prompts[0][0].content
        self.assertIn("You can also draw charts with the draw_chart tool", system)
        self.assertIn('If the question does not seem related to the database', system)

    def test_react_on_csv_with_chart(self):
        llm = f.ReactChatModel(script=[
            f.react_step("sql_db_list_tables", ""),
            f.react_step("draw_chart", CHART),
            f.react_final("North leads"),
        ])
        result = self.run_agent(self.cat(llm))
        self.assertEqual(result.answer, "North leads")
        self.assertIsNotNone(result.chart_markdown)
        prompt = llm.prompts[0][0].content
        self.assertIn('Example of input: {"sql"', prompt)  # tool descriptions with braces do not break the template

    def test_charts_disabled(self):
        llm = f.ScriptedChatModel(script=[f.AIMessage(content="ok")])
        result = self.run_agent(self.cat(llm, support.default_settings(ds_type="CSV", host=str(self.csv), charts="disabled")))
        self.assertEqual((result.thought, result.chart_markdown), ("ok", None))
        self.assertNotIn("draw_chart", llm.prompts[0][0].content)

    def test_configured_sqlite_and_uploaded_datasets(self):
        db = support.sqlite_file(self.tmp / "conf.sqlite", {"conf": (["v"], [(1,)])})
        settings = support.default_settings(ds_type="SQLite", host=str(db))
        llm = f.ScriptedChatModel(script=[f.tool_call("sql_db_list_tables", {"tool_input": ""}), f.AIMessage(content="a")])
        self.run_agent(self.cat(llm, settings))
        self.assertIn("conf", str(llm.prompts[-1][-1].content))

        self.store().add("up.csv", b"x\n1\n")
        llm = f.ScriptedChatModel(script=[f.tool_call("sql_db_list_tables", {"tool_input": ""}), f.AIMessage(content="b")])
        self.run_agent(self.cat(llm, settings))
        self.assertEqual(llm.prompts[-1][-1].content, "up")  # the uploaded datasets win

        settings["use_uploaded_datasets"] = False
        llm = f.ScriptedChatModel(script=[f.tool_call("sql_db_list_tables", {"tool_input": ""}), f.AIMessage(content="c")])
        self.run_agent(self.cat(llm, settings))
        self.assertEqual(llm.prompts[-1][-1].content, "conf")

    def test_uploaded_datasets_without_configured_datasource(self):
        self.assertIsNone(self.run_agent(self.cat(f.ScriptedChatModel(script=[]), support.default_settings())))
        self.store().add("up.sqlite", support.sqlite_bytes({"t": (["v"], [(1,)])}))
        llm = f.ScriptedChatModel(script=[f.tool_call("sql_db_query", {"query": "SELECT v FROM t"}), f.AIMessage(content="one")])
        self.assertEqual(self.run_agent(self.cat(llm, support.default_settings())).answer, "one")

    def test_uploaded_dataset_replaced_while_reading(self):
        self.store().add("up.sqlite", support.sqlite_bytes({"t": (["v"], [(1,)])}))
        real, calls = m.datasets.DatasetStore.workspace, []

        def flaky(store):
            calls.append(True)
            if len(calls) < 3:
                raise FileNotFoundError("replaced on another instance")
            return real(store)

        with mock.patch.object(m.datasets.DatasetStore, "workspace", flaky), \
                mock.patch.object(m.query_agent, "time", m.fakes.FakeClock()):
            llm = f.ScriptedChatModel(script=[f.tool_call("sql_db_query", {"query": "SELECT v FROM t"}), f.AIMessage(content="one")])
            self.assertEqual(self.run_agent(self.cat(llm)).answer, "one")
        self.assertEqual(len(calls), 3)
        # always failing: the configured datasource is used
        with mock.patch.object(m.datasets.DatasetStore, "workspace", side_effect=FileNotFoundError("gone")), \
                mock.patch.object(m.query_agent, "time", m.fakes.FakeClock()):
            llm = f.ScriptedChatModel(script=[f.tool_call("sql_db_list_tables", {"tool_input": ""}), f.AIMessage(content="csv")])
            self.assertEqual(self.run_agent(self.cat(llm)).answer, "csv")
        self.assertEqual(llm.prompts[-1][-1].content, "sales")

    def test_broken_uploaded_datasets_fall_back_to_the_configured_datasource(self):
        self.store().add("up.csv", b"x\n1\n")
        with mock.patch.object(m.datasets.DatasetStore, "workspace", side_effect=OSError("storage down")):
            llm = f.ScriptedChatModel(script=[f.tool_call("sql_db_list_tables", {"tool_input": ""}), f.AIMessage(content="ok")])
            self.assertEqual(self.run_agent(self.cat(llm)).answer, "ok")
        self.assertEqual(llm.prompts[-1][-1].content, "sales")  # the configured CSV, not the uploaded dataset

    def test_tabular_json(self):
        path = self.tmp / "orders.json"
        path.write_text(json.dumps({"orders": [{"id": 1, "total": 10}, {"id": 2, "total": 5}]}))
        llm = f.ScriptedChatModel(script=[f.tool_call("sql_db_query", {"query": "SELECT SUM(total) FROM orders"}), f.AIMessage(content="15")])
        self.assertEqual(self.run_agent(self.cat(llm, support.default_settings(ds_type="JSON", host=str(path)))).answer, "15")
        self.assertIn("[(15,)]", llm.prompts[-1][-1].content)

    def test_non_tabular_json_uses_the_json_agent(self):
        path = self.tmp / "conf.json"
        path.write_text(json.dumps({"config": {"name": "cat"}}))
        llm = f.ReactChatModel(script=[f.react_step("json_spec_list_keys", "data"), f.react_final("the key is config")])
        result = self.run_agent(self.cat(llm, support.default_settings(ds_type="JSON", host=str(path))))
        self.assertEqual((result.thought, result.answer, result.chart_markdown), ("the key is config", "the key is config", None))

    def test_unreadable_json(self):
        path = self.tmp / "bad.json"
        path.write_text("{not json")
        self.assertIsNone(self.run_agent(self.cat(f.ReactChatModel(script=[]), support.default_settings(ds_type="JSON", host=str(path)))))

    def test_invalid_datasources(self):
        for settings in (support.default_settings(ds_type="CSV", host=str(self.tmp / "missing.csv")),
                         support.default_settings(ds_type="Unknown"),
                         support.default_settings(ds_type="")):
            with self.subTest(settings=settings["ds_type"]):
                self.assertIsNone(self.run_agent(self.cat(f.ScriptedChatModel(script=[]), settings)))

    def test_sql_connection_string(self):
        settings = support.default_settings(ds_type="PostgreSQL", host="db", port=5432, database="d",
                                            username="us er", password="p@ss w:rd/+%")
        with mock.patch.object(m.query_agent, "engine_from_uri", side_effect=lambda uri: uri) as engine, \
                mock.patch.object(m.query_agent.QueryCatAgent, "_get_reasoning_sql_agent", return_value=None):
            self.run_agent(self.cat(f.ScriptedChatModel(script=[]), settings))
        from sqlalchemy.engine import make_url
        uri = engine.call_args[0][0]
        url = make_url(uri)
        # regression: quote_plus turned the spaces into "+", which SQLAlchemy does not decode
        self.assertEqual((url.username, url.password), ("us er", "p@ss w:rd/+%"))
        self.assertEqual(m.query_agent._mask_password(uri), "postgresql+psycopg2://us%20er:***@db:5432/d")

    def test_agent_failures(self):
        # the explicit tool_calling setting with an LLM without tools: the agent cannot be created
        settings = support.default_settings(ds_type="CSV", host=str(self.csv), agent_type="tool_calling")
        self.assertIsNone(self.run_agent(self.cat(f.ReactChatModel(script=[]), settings)))

        class Broken(f.ScriptedChatModel):
            def _generate(self, *args, **kwargs):
                raise RuntimeError("LLM down")

        self.assertIsNone(self.run_agent(self.cat(Broken(script=[]))))

    def test_agent_type_selection(self):
        agent = m.query_agent.QueryCatAgent(self.cat(f.ReactChatModel(script=[])))
        agent.settings = support.default_settings(agent_type="react")
        self.assertEqual(agent._agent_type(), m.query_agent.AgentType.ZERO_SHOT_REACT_DESCRIPTION)
        agent.settings = support.default_settings(agent_type="auto")
        self.assertEqual(agent._agent_type(), m.query_agent.AgentType.ZERO_SHOT_REACT_DESCRIPTION)
        agent.large_language_model = f.ScriptedChatModel(script=[])
        self.assertEqual(agent._agent_type(), "tool-calling")
        agent.settings = support.default_settings(agent_type=m.settings.SqlAgentType.REACT)
        self.assertEqual(agent._agent_type(), m.query_agent.AgentType.ZERO_SHOT_REACT_DESCRIPTION)

    def test_executor_output_shapes(self):
        agent = m.query_agent.QueryCatAgent(self.cat(f.ScriptedChatModel(script=[])))
        agent.settings = support.default_settings()
        for output, expected in (({"output": 42}, "42"), (mock.Mock(output="out"), "out"), ("raw", "raw")):
            executor = mock.Mock()
            executor.ainvoke = mock.AsyncMock(return_value=output)
            self.assertEqual(support.run(agent._execute(executor)), expected)

    def test_backward_compatible_entry_point(self):
        llm = f.ScriptedChatModel(script=[f.AIMessage(content="plain")])
        self.assertEqual(support.run(m.query_agent.QueryCatAgent(self.cat(llm)).get_reasoning_agent()), "plain")
        self.assertIsNone(support.run(m.query_agent.QueryCatAgent(self.cat(llm, support.default_settings())).get_reasoning_agent()))


class UsageTest(AgentTestCase):
    def test_querying_marks_the_datasets_as_used(self):
        # regression: the datasets of a conversation in use were removed once uploaded more than the TTL ago
        with mock.patch.object(m.datasets, "time", m.fakes.FakeClock(-100 * 3600)):
            self.store().add("up.csv", b"x\n1\n")
        llm = f.ScriptedChatModel(script=[f.tool_call("sql_db_query", {"query": "SELECT x FROM up"}), f.AIMessage(content="1")])
        self.assertEqual(self.run_agent(self.cat(llm, support.default_settings())).answer, "1")
        self.store("another-chat").cleanup_expired(1)
        self.assertEqual([d.name for d in self.store().list_datasets()], ["up.csv"])

    def test_nothing_to_say(self):
        llm = f.ScriptedChatModel(script=[f.AIMessage(content="")])
        self.assertIsNone(self.run_agent(self.cat(llm)))

    def test_queries_of_the_agent_are_read_only(self):
        # the sql_db_query tool accepts only a single SELECT (the sessions are read-only too)
        llm = f.ScriptedChatModel(script=[f.tool_call("sql_db_query", {"query": "DELETE FROM sales"}), f.AIMessage(content="no")])
        self.run_agent(self.cat(llm))
        self.assertIn("Error: Only SELECT statements are allowed.", llm.prompts[-1][-1].content)
        llm = f.ReactChatModel(script=[f.react_step("sql_db_query", "SELECT COUNT(*) FROM sales; DROP TABLE sales"), f.react_final("no")])
        self.run_agent(self.cat(llm))
        self.assertIn("Only a single SQL statement is allowed.", str(llm.prompts[-1]))


class FinalOutputTest(AgentTestCase):
    def agent(self, settings=None, history=None, user_message="question"):
        workflow = f.Workflow()
        cat = self.cat(f.ScriptedChatModel(script=[]), settings, history=history, workflow=workflow, user_message=user_message)
        return m.query_agent.QueryCatAgent(cat), workflow

    def test_default_prompt_escapes_braces(self):
        agent, workflow = self.agent(support.default_settings(output_prompt=""), user_message='what is {"a": 1}?')
        output = support.run(agent.get_final_output('the thought has {braces}'))
        task = workflow.tasks[0]
        self.assertIn("- Thought: the thought has {{braces}}", task.system_prompt)
        self.assertEqual(task.user_prompt, 'what is {{"a": 1}}?')
        self.assertEqual(output.output, 'FINAL: what is {{"a": 1}}?')

    def test_custom_prompt_with_history(self):
        chart = m.charts.chart_markdown_inline(b"png", "Old chart")
        history = [f.message("user", "first {question}"), f.message("assistant", f"answer\n\n{chart}"),
                   f.message("assistant", ""), f.message("user", "question")]
        settings = support.default_settings(output_prompt="{prompt_prefix}|{chat_history}|{thought}|{user_message}")
        agent, workflow = self.agent(settings, history)
        support.run(agent.get_final_output("t"))
        prompt = workflow.tasks[0].system_prompt
        self.assertIn("- user: first {{question}}\n- assistant: answer\n\n[chart: Old chart]\n- user: question|t|question", prompt)
        self.assertNotIn("base64", prompt)

    def test_input_prompt_and_conversation(self):
        history = [f.message("user", "sales?"), f.message("assistant", "100"), f.message("user", "as a pie")]
        agent, _ = self.agent(support.default_settings(input_prompt="Q: {user_message}"), history, "as a pie")
        agent.settings = support.default_settings(input_prompt="Q: {user_message}")
        self.assertEqual(agent._get_agent_input(),
                         "Conversation so far:\n- user: sales?\n- assistant: 100\n\nLatest request of the user: Q: as a pie")
        agent.settings = support.default_settings(input_prompt="")
        agent.cat.working_memory.history = []
        self.assertEqual(agent._get_agent_input(), "as a pie")

    def test_save_answer_in_history(self):
        agent, _ = self.agent()
        chart = m.charts.chart_markdown_inline(b"png", "C")
        support.run(agent.save_answer_in_history(f"text\n\n{chart}"))
        self.assertEqual(agent.cat.working_memory.history[-1].content.text, "text\n\n[chart: C]")

        async def broken(**kwargs):
            raise RuntimeError("redis down")

        agent.cat.working_memory.update_history = broken
        support.run(agent.save_answer_in_history("text"))  # logged, not raised


if __name__ == "__main__":
    unittest.main()
