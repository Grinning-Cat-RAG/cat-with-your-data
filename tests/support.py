"""Shared fixtures of the test suite.

Run from the root of grinning-cat-core, with its virtual environment and the plugin dependencies installed:
``python -m unittest discover -s cat/plugins/cat-with-your-data/tests``.

The Cat imports every ``.py`` file of the plugin, tests included: at import time this module (like every test module)
needs only the standard library, and everything else is imported lazily by ``load()``.
"""
import asyncio
import json
import sqlite3
import sys
import tempfile
import warnings
from pathlib import Path
from types import SimpleNamespace

PLUGIN_DIR = Path(__file__).resolve().parents[1]
PACKAGE = f"cat.plugins.{PLUGIN_DIR.name}"

modules = SimpleNamespace()


def load() -> SimpleNamespace:
    """Import the plugin modules (once) and redirect the plugin data folder to a temporary directory."""
    if getattr(modules, "loaded", False):
        return modules

    warnings.filterwarnings("ignore")
    modules.data_root = Path(tempfile.mkdtemp(prefix="cwyd-tests-"))
    modules.plugin = load_plugin()
    for name in ("settings", "crypt", "datasets", "data_engine", "charts", "chart_tool", "parsers", "query_agent",
                 "query_cat", "endpoints"):
        setattr(modules, name, sys.modules[f"{PACKAGE}.{name}"])
    modules.fakes = _build_fakes()
    modules.loaded = True
    return modules


def load_plugin(order=None):
    """Import (or reload) the plugin with the core loader, as in production; ``order`` sorts the files to load.

    The security scan forbids dynamic imports in the plugin files. The data folder of the core is never touched: the
    plugin modules bind ``get_data_path`` when loaded, so it is patched only while loading (other tests running in the
    same process, e.g. the core ones under pytest, are not affected).
    """
    from unittest import mock

    import cat.utils
    from cat.looking_glass.mad_hatter.plugin import Plugin

    plugin = Plugin(str(PLUGIN_DIR))
    if order is not None:
        plugin._py_files = sorted(plugin._py_files, key=order)
    with mock.patch.object(cat.utils, "get_data_path", lambda: str(modules.data_root)):
        plugin._load_decorated_functions()
    assert sys.modules[f"{PACKAGE}.datasets"].get_data_path() == str(modules.data_root)
    return plugin


def run(coroutine):
    return asyncio.run(coroutine)


def sqlite_file(path: Path, tables: dict) -> Path:
    """SQLite file with the given tables: {name: (columns, rows)}."""
    if path.exists():
        path.unlink()
    conn = sqlite3.connect(path)
    try:
        for name, (columns, rows) in tables.items():
            quoted = lambda value: '"' + value.replace('"', '""') + '"'  # noqa: E731
            conn.execute(f"CREATE TABLE {quoted(name)} ({', '.join(quoted(c) for c in columns)})")
            placeholders = ", ".join("?" for _ in columns)
            conn.executemany(f"INSERT INTO {quoted(name)} VALUES ({placeholders})", rows)
        conn.commit()
    finally:
        conn.close()
    return path


def sqlite_bytes(tables: dict) -> bytes:
    with tempfile.TemporaryDirectory() as tmp:
        return sqlite_file(Path(tmp) / "db.sqlite", tables).read_bytes()


def default_settings(**overrides) -> dict:
    settings = modules.settings.MySettings().model_dump(mode="json")
    settings.update(overrides)
    return settings


def _build_fakes() -> SimpleNamespace:
    from langchain_core.language_models import BaseChatModel
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, ChatResult

    from cat import AgenticWorkflowOutput

    class ReactChatModel(BaseChatModel):
        """Chat model replying with the scripted messages, in order, without tool calling (ReAct agents)."""
        script: list
        prompts: list = []

        @property
        def _llm_type(self) -> str:
            return "scripted"

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            self.prompts.append(messages)
            if not self.script:
                return ChatResult(generations=[ChatGeneration(message=AIMessage(content="Final Answer: no more"))])
            return ChatResult(generations=[ChatGeneration(message=self.script.pop(0))])

    class ScriptedChatModel(ReactChatModel):
        """Same, with tool calling."""
        def bind_tools(self, tools, **kwargs):
            return self

    class Workflow:
        """Stand-in of the core agentic workflow: it echoes the prompts, so that tests can check them."""
        def __init__(self, output=None, error=None, with_llm_error=False):
            self.tasks = []
            self.output = output
            self.error = error
            self.with_llm_error = with_llm_error

        async def run(self, task, llm, callbacks=None):
            self.tasks.append(task)
            if self.error is not None:
                raise self.error
            text = self.output if self.output is not None else f"FINAL: {task.user_prompt}"
            return AgenticWorkflowOutput(output=text, with_llm_error=self.with_llm_error)

    def make_cat(llm=None, settings=None, user_message="question", chat_id="chat-1", agent_key="agent-1",
                 history=None, workflow=None, notifier=None):
        working_memory = SimpleNamespace(
            user_message=SimpleNamespace(text=user_message), history=list(history or []),
        )

        async def update_history(who, content):
            working_memory.history.append(SimpleNamespace(who=who, content=content))

        working_memory.update_history = update_history
        state = {"settings": dict(settings if settings is not None else default_settings())}

        async def load_settings(*args):
            if isinstance(state["settings"], Exception):
                raise state["settings"]
            return dict(state["settings"])

        plugin = SimpleNamespace(load_settings=load_settings)

        async def execute_hook(name, *args, caller=None):
            return args[0] if args else None

        manager = SimpleNamespace(get_plugin=lambda: plugin, execute_hook=execute_hook)
        return SimpleNamespace(
            large_language_model=llm, agentic_workflow=workflow or Workflow(), mad_hatter=manager,
            plugin_manager=manager, working_memory=working_memory, agent_key=agent_key, id=chat_id,
            notifier=notifier, state=state,
        )

    def message(who, text):
        return SimpleNamespace(who=who, content=SimpleNamespace(text=text))

    def tool_call(name, args, call_id="1"):
        return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])

    def react_step(tool, tool_input):
        payload = tool_input if isinstance(tool_input, str) else json.dumps(tool_input)
        return AIMessage(content=f"Thought: I use {tool}\nAction: {tool}\nAction Input: {payload}")

    def react_final(text):
        return AIMessage(content=f"Thought: I know the answer\nFinal Answer: {text}")

    return SimpleNamespace(
        ScriptedChatModel=ScriptedChatModel, ReactChatModel=ReactChatModel, Workflow=Workflow, make_cat=make_cat,
        message=message, tool_call=tool_call, react_step=react_step, react_final=react_final, AIMessage=AIMessage,
    )
