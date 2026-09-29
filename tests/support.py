"""Shared fixtures of the test suite.

Run from the root of grinning-cat-core, with its virtual environment and the plugin dependencies installed:
``python -m unittest discover -s cat/plugins/cat-with-your-data/tests``.

The Cat imports every ``.py`` file of the plugin, tests included: at import time this module (like every test module)
needs only the standard library, and everything else is imported lazily by ``load()``.
"""
import asyncio
import json
import os
import sqlite3
import sys
import time
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
    # removed when the interpreter exits
    modules.data_folder = tempfile.TemporaryDirectory(prefix="cwyd-tests-")
    modules.data_root = Path(modules.data_folder.name)
    modules.plugin = load_plugin()
    for name in ("settings", "crypt", "datasets", "data_engine", "charts", "chart_tool", "parsers", "query_agent",
                 "query_cat", "endpoints"):
        setattr(modules, name, sys.modules[f"{PACKAGE}.{name}"])
    modules.fakes = _build_fakes()
    modules.file_manager = modules.fakes.ObjectStoreFileManager()
    modules.loaded = True
    return modules


def load_plugin(order=None):
    """Import (or reload) the plugin with the core loader, as in production; ``order`` sorts the files to load.

    The security scan forbids dynamic imports in the plugin files. The data folder of the core is never touched: it is
    patched only while loading (other tests running in the same process, e.g. the core ones under pytest, are not
    affected).
    """
    from unittest import mock

    import cat.utils
    from cat.looking_glass.mad_hatter.plugin import Plugin

    plugin = Plugin(str(PLUGIN_DIR))
    if order is not None:
        plugin._py_files = sorted(plugin._py_files, key=order)
    with mock.patch.object(cat.utils, "get_data_path", lambda: str(modules.data_root)):
        plugin._load_decorated_functions()
    return plugin


def temporary_folder(test) -> Path:
    """A temporary folder removed at the end of the test."""
    folder = tempfile.TemporaryDirectory(prefix="cwyd-test-")
    test.addCleanup(folder.cleanup)
    return Path(folder.name)


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


def configured_file(name: str, content: bytes):
    """A file datasource with the given content (its version is the content)."""
    import hashlib

    return modules.datasets.ConfiguredFile(name, f"test::{name}::{hashlib.sha256(content).hexdigest()}", lambda: content)


def put_file(agent_key: str, path: str, content: bytes, file_manager=None) -> str:
    """Store a file in the folder of the agent, in the file manager (e.g. a file uploaded in the memory of the agent)."""
    folder, _, name = f"{agent_key}/{path}".rpartition("/")
    (file_manager or modules.file_manager).write_file(content, name, folder)
    return path


def core_sends(cat, output):
    """What the core does with the answer of an ``agent_fast_reply`` hook: the conversation history stores it (unless
    the LLM failed), then the hook of the plugin runs; the history saved in the database is in ``cat.state``."""
    from unittest import mock

    from cat import CatMessage

    message = CatMessage(text=output.output)
    if not output.with_llm_error:
        run(cat.working_memory.update_history(who="assistant", content=message))

    async def set_messages(agent_id, user_id, chat_id, history):
        cat.state["saved_history"] = [(item.who, item.content.text) for item in history]

    with mock.patch.object(modules.query_cat.crud_conversations, "set_messages", set_messages):
        return run(modules.query_cat.before_cat_sends_message.function(message, output, cat))


def default_settings(**overrides) -> dict:
    settings = modules.settings.MySettings().model_dump(mode="json")
    settings.update(overrides)
    return settings


def _build_fakes() -> SimpleNamespace:
    from langchain_core.language_models import BaseChatModel
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, ChatResult

    from datetime import datetime

    from cat import AgenticWorkflowOutput
    from cat.core_plugins.base_plugin.file_managers.custom import LocalFileManager
    from cat.services.factory.file_manager import BaseFileManager, DummyFileManager, FileResponse

    class ObjectStoreFileManager(BaseFileManager):
        """File manager with the semantics of an object storage (S3): atomic writes, no folders, no renames."""
        def __init__(self):
            super().__init__()
            self.objects = {}
            self.hooks = {}  # operation -> callable(path), run before the operation (to simulate the other instances)

        def _hook(self, operation, path):
            if callable(hook := self.hooks.get(operation)):
                hook(path)

        def _eq(self, other):
            return self is other

        def _upload_file(self, file_path, destination_path):  # pragma: no cover - not used by the plugin
            raise NotImplementedError

        def _download_file_to_local(self, file_path, local_path):  # pragma: no cover - not used by the plugin
            raise NotImplementedError

        def _download_file(self, file_path):
            self._hook("download", file_path)
            return self.objects.get(os.path.normpath(file_path))

        def _read_file(self, file_path):  # pragma: no cover - not used by the plugin
            return self.objects[os.path.normpath(file_path)]

        def _write_file(self, file_content, file_path):
            self._hook("write", file_path)
            content = file_content.encode("utf-8") if isinstance(file_content, str) else bytes(file_content)
            self.objects[os.path.normpath(file_path)] = content

        def _remove_file(self, file_path):
            self._hook("remove", file_path)
            return self.objects.pop(os.path.normpath(file_path), None) is not None

        def _remove_folder(self, remote_root_dir):
            prefix = os.path.normpath(remote_root_dir) + os.sep
            for key in [k for k in self.objects if k.startswith(prefix)]:
                del self.objects[key]
            return True

        def _list_files(self, remote_root_dir):
            folder = os.path.normpath(remote_root_dir)
            return [
                FileResponse(path=key, name=os.path.basename(key), hash="", size=len(value),
                             last_modified=datetime.now().strftime("%Y-%m-%d"))
                for key, value in list(self.objects.items())
                if os.path.dirname(key) == folder
            ]

        def _clone_folder(self, remote_root_dir_from, remote_root_dir_to):  # pragma: no cover - not used by the plugin
            raise NotImplementedError

    def local_file_manager():
        """The file manager of the core storing on a (shared) folder."""
        manager = LocalFileManager()
        manager._root_dir = tempfile.mkdtemp(prefix="cwyd-storage-", dir=modules.data_root)
        return manager

    class FakeClock:
        """Replaces the ``time`` module of the plugin: the current time is shifted by ``offset`` seconds."""
        def __init__(self, offset=0.0):
            self.offset = offset

        def time(self):
            return time.time() + self.offset

        def time_ns(self):
            return time.time_ns() + int(self.offset * 1e9)

        def monotonic(self):
            return time.monotonic()

        def sleep(self, seconds):
            self.offset += seconds

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
                 history=None, workflow=None, notifier=None, file_manager=None, user_id="user-1"):
        working_memory = SimpleNamespace(
            user_message=SimpleNamespace(text=user_message), history=list(history or []),
        )

        async def update_history(who, content):
            # the core stores a copy of the message
            working_memory.history.append(SimpleNamespace(who=who, content=SimpleNamespace(text=content.text)))

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
            user=SimpleNamespace(id=user_id),
            notifier=notifier, state=state, file_manager=file_manager or modules.file_manager,
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
        ObjectStoreFileManager=ObjectStoreFileManager, local_file_manager=local_file_manager,
        DummyFileManager=DummyFileManager, FakeClock=FakeClock,
        ScriptedChatModel=ScriptedChatModel, ReactChatModel=ReactChatModel, Workflow=Workflow, make_cat=make_cat,
        message=message, tool_call=tool_call, react_step=react_step, react_final=react_final, AIMessage=AIMessage,
    )
