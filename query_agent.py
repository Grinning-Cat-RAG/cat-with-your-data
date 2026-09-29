import asyncio
import json
import re
import sqlite3
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, List
from urllib.parse import quote

from langchain_classic.agents import AgentType
from langchain_community.agent_toolkits import SQLDatabaseToolkit, create_sql_agent, JsonToolkit, create_json_agent
from langchain_community.tools.json.tool import JsonSpec
from langchain_community.agent_toolkits.sql.prompt import SQL_PREFIX
from langchain_core.language_models import BaseChatModel
from sqlalchemy.engine import Engine
from cat import StrayCat, AgenticWorkflowTask, AgenticWorkflowOutput, CatMessage
from cat.log import log
from cat.templates import prompts

from .chart_tool import WHEN_TO_USE, ChartCollector, ChartToolFactory
from .charts import chart_markdown_inline, chart_markdown_url, save_chart, strip_inline_images
from .data_engine import engine_from_csv, engine_from_json, engine_from_sqlite_file, engine_from_uri, sql_database
# the core loader imports and then reloads the plugin modules one by one, in no particular order: the classes of
# `datasets` are looked up at call time, so that `except` and patches always see the current ones
from . import datasets
from .settings import datasources


@dataclass
class DataSource:
    kind: str  # "sql" | "json"
    engine: Engine | None = None
    json_path: str | None = None
    label: str = ""
    # engines built for a single request (uploaded datasets) are disposed after it and their schema is not cached
    per_request: bool = False


@dataclass
class QueryResult:
    thought: str
    chart_markdown: str | None = None
    # the answer of the agent, without the notes for the final prompt: the reply when the final LLM call fails
    answer: str = ""


def _supports_tool_calling(llm: Any) -> bool:
    return isinstance(llm, BaseChatModel) and type(llm).bind_tools is not BaseChatModel.bind_tools


def _value(value: Any) -> Any:
    return value.value if isinstance(value, Enum) else value


def _escape_braces(text: str) -> str:
    """Escape the braces, since the text is going to be used inside a prompt template."""
    return str(text).replace("{", "{{").replace("}", "}}")


def _mask_password(connection_string: str) -> str:
    return re.sub(r"://([^:/@]*):[^@]*@", r"://\1:***@", connection_string)


class QueryCatAgent:
    def __init__(self, cat: StrayCat) -> None:
        self.cat = cat
        self.large_language_model = cat.large_language_model
        self.agentic_workflow = cat.agentic_workflow
        self.settings = None

    # Load configurations
    async def _load_configurations(self):
        # Acquire settings
        settings = await self.cat.mad_hatter.get_plugin().load_settings()

        # If the settings are the same, skip the function
        if self.settings and self.settings == settings:
            return

        # Set settings
        self.settings = settings

    def _setting(self, key: str, default: Any = None) -> Any:
        value = _value((self.settings or {}).get(key, default))
        return default if value in (None, "") else value

    def _get_input_prompt(self) -> str:
        # Get user message
        user_message = self.cat.working_memory.user_message.text

        # Get input prompt from settings
        input_prompt = user_message
        if self.settings["input_prompt"] != '':
            input_prompt = self.settings["input_prompt"].format(
                user_message=user_message
            )

        log.debug("=====================================================")
        log.debug(f"Input prompt:\n{input_prompt}")
        log.debug("=====================================================")

        return input_prompt

    def _history_lines(self, limit: int, include_current: bool = True) -> List[str]:
        history = list(self.cat.working_memory.history or [])
        if not include_current and history and history[-1].who == "user":
            history = history[:-1]
        return [
            f"- {item.who}: {strip_inline_images(item.content.text)}"
            for item in history[-limit:]
            if item.content and item.content.text
        ]

    async def _llm_callbacks(self) -> List:
        return await self.cat.plugin_manager.execute_hook("llm_callbacks", [], caller=self.cat)

    # ----------------------------------------------------------------------------------------------------------------
    # datasource
    # ----------------------------------------------------------------------------------------------------------------
    def _uploaded_datasets_engine(self) -> Engine | None:
        store = datasets.DatasetStore(self.cat.agent_key, self.cat.id)
        failures = 0
        while True:
            if (path := store.workspace_path()) is None:
                return None
            try:
                engine = engine_from_sqlite_file(path)
                store.mark_used()
                return engine
            except sqlite3.OperationalError:
                # the file was replaced by a concurrent upload between the two calls: look for the current one
                failures += 1
                if failures == 3:
                    raise

    async def _resolve_datasource(self) -> DataSource | None:
        # datasets uploaded on the fly (chat and agent level) win over the configured datasource
        if self._setting("use_uploaded_datasets", True):
            try:
                if engine := await asyncio.to_thread(self._uploaded_datasets_engine):
                    return DataSource(kind="sql", engine=engine, label="uploaded datasets", per_request=True)
            except Exception as e:
                log.error(f"[cat-with-your-data] cannot prepare the uploaded datasets: {e}")

        datasource_type = self._setting("ds_type")
        if datasource_type not in datasources:
            return None

        agent_type = datasources[datasource_type]["agent_type"]
        try:
            if agent_type == "sql":
                params = {
                    **self.settings,
                    # URL-encoded (SQLAlchemy decodes them with ``unquote``: a space must be %20, not +)
                    "username": quote(str(self.settings.get("username") or ""), safe=""),
                    "password": quote(str(self.settings.get("password") or ""), safe=""),
                }
                connection_string = datasources[datasource_type]["conn_str"].format(**params)
                log.info(f"Connection string: {_mask_password(connection_string)}")
                return DataSource(kind="sql", engine=engine_from_uri(connection_string), label=datasource_type)
            if agent_type == "csv":
                engine = await asyncio.to_thread(engine_from_csv, self.settings["host"])
                return DataSource(kind="sql", engine=engine, label=Path(self.settings["host"]).name)
            # agent_type == "json": tabular JSON files are queried with SQL (and can be charted), the others with the
            # JSON agent
            try:
                engine = await asyncio.to_thread(engine_from_json, self.settings["host"])
            except Exception as e:
                log.warning(f"[cat-with-your-data] the JSON file cannot be loaded as tables: {e}")
                engine = None
            if engine is not None:
                return DataSource(kind="sql", engine=engine, label="JSON")
            return DataSource(kind="json", json_path=self.settings["host"], label="JSON")
        except Exception as e:
            log.error(f"Failed to create the connection to the datasource: {e}")
        return None

    # ----------------------------------------------------------------------------------------------------------------
    # main entry point
    # ----------------------------------------------------------------------------------------------------------------
    async def run(self) -> QueryResult | None:
        """Answer the user's question from the data, with the charts drawn by the agent (if any)."""
        await self._load_configurations()

        source = await self._resolve_datasource()
        if source is None:
            return None

        if source.kind == "json":
            thought = await self._get_reasoning_json_agent(source.json_path)
            return QueryResult(thought=thought, answer=thought) if thought else None

        collector = ChartCollector()
        try:
            answer = await self._get_reasoning_sql_agent(source.engine, collector, cache_schema=not source.per_request)
        finally:
            if source.per_request:
                source.engine.dispose()

        chart_markdown = "\n\n".join([await self._chart_markdown(c.png, c.spec.title) for c in collector.charts])
        if not answer and not chart_markdown:
            return None

        thought = answer or ""
        if collector.charts:
            titles = ", ".join(f'"{c.spec.title}"' for c in collector.charts)
            thought = (
                f"{thought}\n\n(The chart {titles} is shown to the user right after your answer: "
                "comment the data briefly, do not describe how to draw it and do not include images or links.)"
            ).strip()
        return QueryResult(thought=thought, chart_markdown=chart_markdown or None, answer=answer or "")

    # Execute agent to get a final thought, based on the type (kept for backward compatibility)
    async def get_reasoning_agent(self) -> str | None:
        result = await self.run()
        return result.thought if result else None

    # ----------------------------------------------------------------------------------------------------------------
    # charts
    # ----------------------------------------------------------------------------------------------------------------
    def _store_chart(self, png: bytes) -> str:
        chart_id = save_chart(self.cat.agent_key, png)
        # the expired charts are removed here too: with a configured datasource there may be no uploads at all
        store = datasets.DatasetStore(self.cat.agent_key, self.cat.id)
        store.cleanup_expired(float(self._setting("chat_datasets_ttl_hours", 0)))
        return chart_id

    async def _chart_markdown(self, png: bytes, title: str) -> str:
        if self._setting("chart_delivery", "inline") == "url":
            try:
                chart_id = await asyncio.to_thread(self._store_chart, png)
                return chart_markdown_url(self._setting("public_base_url", ""), self.cat.agent_key, chart_id, title)
            except OSError as e:
                log.error(f"[cat-with-your-data] cannot store the chart, falling back to inline delivery: {e}")
        return chart_markdown_inline(png, title)

    # ----------------------------------------------------------------------------------------------------------------
    # final answer
    # ----------------------------------------------------------------------------------------------------------------
    # Return the final response, based on the user's message and reasoning
    async def get_final_output(self, thought: str) -> AgenticWorkflowOutput:
        user_message = self.cat.working_memory.user_message.text

        # Load configurations
        await self._load_configurations()

        # Get prompt
        prompt_prefix = await self.cat.mad_hatter.execute_hook(
            "agent_prompt_prefix", prompts.MAIN_PROMPT, caller=self.cat
        )

        # Get chat history (the inline charts are replaced by a placeholder)
        chat_history = "\n".join(self._history_lines(10))

        # values going into a prompt template: braces must be escaped
        safe_user_message = _escape_braces(user_message)
        safe_thought = _escape_braces(thought)
        safe_chat_history = _escape_braces(chat_history)

        # Default output Prompt
        output_prompt = f"""{prompt_prefix}
You have elaborated the user's question, you have searched for the answer and now you have the solution in your Thought; 
reply to the user briefly, precisely and based on the context of the dialogue.
- Human: {safe_user_message}
- Thought: {safe_thought}
- AI:"""

        # Set output prompt from settings
        if self.settings["output_prompt"]:
            output_prompt = self.settings["output_prompt"].format(
                prompt_prefix=prompt_prefix,
                user_message=safe_user_message,
                thought=safe_thought,
                chat_history=safe_chat_history,
            )

        # Invoke LLM and get a final and contextual response
        log.debug("=====================================================")
        log.debug(f"Output prompt:\n{output_prompt}")
        log.debug("=====================================================")

        agent_input = AgenticWorkflowTask(system_prompt=output_prompt, user_prompt=safe_user_message)
        return await self.agentic_workflow.run(
            task=agent_input,
            llm=self.large_language_model,
            callbacks=await self._llm_callbacks(),
        )

    async def save_answer_in_history(self, text: str) -> None:
        """The fast reply skips the core hook saving the answer: save it, so that follow-up questions have context.

        The inline charts are replaced by a short placeholder: the core sends the latest messages of the history to the
        LLM, and a base64 image is tens of thousands of tokens of noise.
        """
        try:
            await self.cat.working_memory.update_history(
                who="assistant", content=CatMessage(text=strip_inline_images(text))
            )
        except Exception as e:
            log.warning(f"[cat-with-your-data] cannot save the answer in the conversation history: {e}")

    # ----------------------------------------------------------------------------------------------------------------
    # reasoning agents
    # ----------------------------------------------------------------------------------------------------------------
    def _get_agent_input(self) -> str:
        """The input prompt, preceded by the recent conversation (needed by follow-ups like "now as a pie chart")."""
        conversation = self._history_lines(6, include_current=False)
        if not conversation:
            return self._get_input_prompt()
        return (
            "Conversation so far:\n" + "\n".join(conversation)
            + f"\n\nLatest request of the user: {self._get_input_prompt()}"
        )

    async def _execute(self, agent_executor) -> str | None:
        # Get final thought, after agent reasoning steps
        try:
            final_thought = await agent_executor.ainvoke(
                {"input": self._get_agent_input()},
                config={"callbacks": await self._llm_callbacks()},
            )
            if isinstance(final_thought, dict) and "output" in final_thought:
                return str(final_thought["output"])
            return getattr(final_thought, "output", getattr(final_thought, "content", str(final_thought)))
        except Exception as e:
            log.error(f"Failed to execute the agent: {e}")
            return None

    def _agent_type(self) -> Any:
        setting = self._setting("agent_type", "auto")
        if setting == "tool_calling" or (setting == "auto" and _supports_tool_calling(self.large_language_model)):
            return "tool-calling"
        return AgentType.ZERO_SHOT_REACT_DESCRIPTION

    # Execute sql agent (SQL datasources, CSV and tabular JSON files, uploaded datasets), with the chart tool
    async def _get_reasoning_sql_agent(
        self, engine: Engine, collector: ChartCollector | None = None, cache_schema: bool = True
    ) -> str | None:
        try:
            db = await asyncio.to_thread(sql_database, engine, cache_schema)
            agent_type = self._agent_type()

            extra_tools = []
            prefix = SQL_PREFIX
            mode = self._setting("charts", "on_request")
            if collector is not None and mode in WHEN_TO_USE:
                factory = ChartToolFactory(
                    engine=engine,
                    collector=collector,
                    mode=mode,
                    max_rows=max(1, int(self._setting("chart_max_rows", 1000))),
                    summary_rows=max(1, int(self._setting("thought_max_rows", 30))),
                )
                extra_tools.append(factory.build(structured=agent_type == "tool-calling"))
                prefix = SQL_PREFIX.replace("\n\nIf the question does not seem", f"\n{factory.prefix_addendum}\nIf the question does not seem")

            # Create SQL Agent
            agent_executor = create_sql_agent(
                llm=self.large_language_model,
                toolkit=SQLDatabaseToolkit(db=db, llm=self.large_language_model),
                verbose=True,
                agent_type=agent_type,
                prefix=prefix,
                extra_tools=extra_tools,
                agent_executor_kwargs={"handle_parsing_errors": True},
            )
        except Exception as e:
            log.error(f"Failed to create the SQL agent: {e}")
            return None

        return await self._execute(agent_executor)

    # Execute json agent
    async def _get_reasoning_json_agent(self, json_file_path: str) -> str | None:
        # Create JSON agent
        try:
            # Get json data
            data = json.loads(Path(json_file_path).read_text(encoding="utf-8"))

            # Create JSON toolkit
            json_spec = JsonSpec(dict_=data, max_value_length=4000)
            json_toolkit = JsonToolkit(spec=json_spec)

            agent_executor = create_json_agent(
                llm=self.large_language_model,
                toolkit=json_toolkit,
                verbose=True
            )
        except Exception as e:
            log.error(f"Failed to create JSON agent: {e}")
            return None

        return await self._execute(agent_executor)
