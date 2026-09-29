import asyncio
from typing import Dict, List

from langchain_core.documents import Document

from cat import hook, log, AgenticWorkflowOutput, StrayCat

# the core loader imports and then reloads the plugin modules one by one, in no particular order: the classes of
# `datasets` are looked up at call time, so that `except` and patches always see the current ones
from . import datasets
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

    await query_agent.save_answer_in_history(text)
    return AgenticWorkflowOutput(output=text)


async def _plugin_settings(cat) -> Dict:
    try:
        return await cat.mad_hatter.get_plugin().load_settings()
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
    chat_id = cat.id if isinstance(cat, StrayCat) else None
    store = datasets.DatasetStore(cat.file_manager, cat.agent_key, chat_id)

    for doc in captured:
        # the raw bytes are always removed: they must never reach the chunker and the vector memory
        kind = doc.metadata.pop(KIND_KEY)
        name = doc.metadata.pop(NAME_KEY, None) or doc.metadata.get("source") or f"dataset.{kind}"
        payload = doc.metadata.pop(PAYLOAD_KEY, None)

        try:
            info = await asyncio.to_thread(store.add, name, payload, chat_id is None)
        except Exception as e:  # noqa: BLE001 - an exception escaping the hook would restore the docs with the bytes
            log.error(f"[cat-with-your-data] cannot register the dataset '{name}': {e}")
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
    return docs
