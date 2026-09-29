import asyncio
from typing import Dict, List

from langchain_core.documents import Document

from cat import hook, log, AgenticWorkflowOutput, StrayCat
from cat.db.cruds import conversations as crud_conversations

# sibling modules are looked up at call time
from . import datasets
from .charts import strip_inline_images
from .parsers import (
    CSV_MIME_TYPES,
    KIND_KEY,
    NAME_KEY,
    PAYLOAD_KEY,
    SQLITE_MIME_TYPES,
    DatasetBlobParser,
)
from .query_agent import QueryCatAgent


# priority 0: this hook must run after the other `agent_fast_reply` hooks (e.g. the core "memory" plugin, which
# replies with a canned message when no declarative memory is recalled), since the last non-empty reply wins
@hook(priority=0)
async def agent_fast_reply(cat) -> AgenticWorkflowOutput | None:
    # a message blocked by guard-plugin is not worked on: its reply is the block
    if getattr(cat.working_memory, "guard_blocked", None) == id(cat.working_memory.user_message):
        return None

    # Instantiate query agent
    query_agent = QueryCatAgent(cat)

    # Get the thought (and the chart, if needed) from the data
    result = await query_agent.run()
    if not result:
        return None

    # Get a final and contextual response; if the LLM fails, the answer of the agent is returned as it is
    try:
        output = await query_agent.get_final_output(result.thought)
    except Exception as e:
        log.error(f"[cat-with-your-data] the final answer cannot be generated, returning the agent answer: {e}")
        output = None
    text = (output.output or "").strip() if output else ""
    if not text or (output and output.with_llm_error):
        text = result.answer

    # Attach the chart to the answer
    if result.chart_markdown:
        text = f"{text}\n\n{result.chart_markdown}".strip()

    # the answer goes through the hooks of the turn, as the ones of the agent: the core stores it in the history
    return AgenticWorkflowOutput(output=text)


# priority 0: after the core hook (priority 1) storing the answer in the conversation history
@hook(priority=0)
async def before_cat_sends_message(message, agent_output, cat):
    """The charts are kept out of the conversation history (the user gets them in the answer).

    The core sends the latest messages of the history to the LLM, and a chart (a base64 image) is tens of thousands of
    tokens of noise: the stored answer has a short placeholder instead.
    """
    try:
        history = cat.working_memory.history or []
        if history and history[-1].who == "assistant":
            text = history[-1].content.text or ""
            if (compact := strip_inline_images(text)) != text:
                history[-1].content.text = compact
                await crud_conversations.set_messages(cat.agent_key, cat.user.id, cat.id, history)
    except Exception as e:  # noqa: BLE001 - the answer is sent anyway
        log.warning(f"[cat-with-your-data] cannot remove the charts from the conversation history: {e}")
    return message


@hook
async def after_cheshire_cat_destroy(agent_id: str, cat) -> None:
    """The agent was destroyed: its shared datasets are removed (the core removes the folders of the conversations)."""
    try:
        cat.file_manager.remove_folder(datasets.shared_dir_of(agent_id))
    except Exception as e:  # noqa: BLE001 - the destruction of the agent goes on
        log.warning(f"[cat-with-your-data] cannot remove the shared datasets of the destroyed agent {agent_id}: {e}")


async def _plugin_settings(cat) -> Dict:
    try:
        # the agent is explicit: otherwise the core looks for it in the call stack, and may find the wrong one
        return await cat.mad_hatter.get_plugin().load_settings(cat.agent_key)
    except Exception as e:
        log.warning(f"[cat-with-your-data] cannot load the settings: {e}")
        return {}


# priority 1: runs after the core plugin (priority 999), so it can wrap the parsers registered there
@hook(priority=1)
async def rabbithole_instantiates_parsers(file_handlers: Dict, cat) -> Dict:
    settings = await _plugin_settings(cat)
    if not settings.get("capture_rabbithole_uploads", True):
        return file_handlers

    max_bytes = int(settings.get("max_upload_size_mb") or 100) * 1024 * 1024

    # CSV files are detected as text/plain by the Rabbit Hole: the parser checks the extension and delegates to the
    # original parser anything that is not a dataset
    for mime_type in CSV_MIME_TYPES:
        file_handlers[mime_type] = DatasetBlobParser(fallback=file_handlers.get(mime_type), max_bytes=max_bytes)
    for mime_type in SQLITE_MIME_TYPES:
        file_handlers[mime_type] = DatasetBlobParser(max_bytes=max_bytes)

    return file_handlers


@hook(priority=10)
async def before_rabbithole_splits_documents(docs: List[Document], cat) -> List[Document]:
    captured = [d for d in docs if isinstance(d.metadata, dict) and KIND_KEY in d.metadata]
    if not captured:
        return docs

    settings = await _plugin_settings(cat)
    chat_id, user_id = (cat.id, cat.user.id) if isinstance(cat, StrayCat) else (None, None)
    store = datasets.DatasetStore(cat.file_manager, cat.agent_key, chat_id, user_id)
    failed: List[Document] = []

    for doc in captured:
        # the raw bytes are always removed: they must never reach the chunker and the vector memory
        kind = doc.metadata.pop(KIND_KEY)
        name = doc.metadata.pop(NAME_KEY, None) or doc.metadata.get("source") or f"dataset.{kind}"
        payload = doc.metadata.pop(PAYLOAD_KEY, None)

        try:
            info = await asyncio.to_thread(store.add, name, payload, chat_id is None)
        except Exception as e:  # noqa: BLE001 - an exception escaping the hook would restore the docs with the bytes
            # the description of a dataset that cannot be queried must not go into the memory: the document is
            # dropped (without other documents the ingestion fails) and the user is told why
            log.error(f"[cat-with-your-data] cannot register the dataset '{name}': {e}")
            failed.append(doc)
            if isinstance(cat, StrayCat):
                try:
                    await cat.notifier.send_error(f"The dataset '{name}' cannot be stored: {e}")
                except Exception as notify_error:  # the websocket may be closed
                    log.debug(f"[cat-with-your-data] notification not sent: {notify_error}")
            continue

        tables = ", ".join(info.tables.keys())
        scope = "in this conversation" if info.scope == "chat" else "in every conversation"
        doc.page_content += f"\nThe dataset can be queried {scope} (tables: {tables})."
        log.info(f"[cat-with-your-data] dataset '{info.name}' registered from the Rabbit Hole ({info.scope})")

        if isinstance(cat, StrayCat):
            try:
                await cat.notifier.send_notification(
                    f"Dataset '{info.name}' is ready: ask me anything about it, even charts (tables: {tables})."
                )
            except Exception as e:  # the websocket may be closed
                log.debug(f"[cat-with-your-data] notification not sent: {e}")

    ttl = settings.get("chat_datasets_ttl_hours", 72)
    await asyncio.to_thread(store.cleanup_expired, float(ttl or 0))
    return [doc for doc in docs if not any(doc is f for f in failed)]
