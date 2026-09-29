"""Rabbit Hole parser that turns the uploaded CSV/SQLite files into queryable datasets.

The parser runs in the ingestion executor and does not know the conversation the file was uploaded to. So it only
describes the dataset (the description is what goes into the Cat's memory) and attaches the raw bytes to the parsed
document; the ``before_rabbithole_splits_documents`` hook, which receives the right StrayCat, stores the dataset in
the proper scope and removes the bytes before chunking.
"""
from typing import Iterator

from langchain_core.document_loaders import BaseBlobParser
from langchain_core.documents.base import Blob, Document

from cat import log

# sibling modules are looked up at call time
from . import datasets

PAYLOAD_KEY = "cat_with_your_data_payload"
KIND_KEY = "cat_with_your_data_kind"
NAME_KEY = "cat_with_your_data_name"

SQLITE_MIME_TYPES = ("application/x-sqlite3", "application/vnd.sqlite3")
CSV_MIME_TYPES = ("text/csv", "text/tab-separated-values", "text/plain")


class DatasetBlobParser(BaseBlobParser):
    def __init__(self, fallback: BaseBlobParser | None = None, max_bytes: int | None = None):
        self.fallback = fallback
        self.max_bytes = max_bytes

    def _fallback(self, blob: Blob) -> Iterator[Document]:
        if self.fallback is not None:
            yield from self.fallback.lazy_parse(blob)

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        name = str(blob.path or blob.source or "")
        content = blob.as_bytes()
        kind = datasets.detect_kind(name, content)
        if kind is None:
            yield from self._fallback(blob)
            return

        # the ingestion fails with this message, which is notified to the user: a description of a dataset that
        # cannot be queried must not end up in the memory
        if self.max_bytes and len(content) > self.max_bytes:
            raise datasets.DatasetError(
                f"'{name}' exceeds the maximum dataset size of {self.max_bytes // (1024 * 1024)} MB "
                "(max_upload_size_mb setting of Cat With Your Data)"
            )

        try:
            description = datasets.describe_dataset(name, content, kind)
        except datasets.DatasetError as e:
            log.warning(f"[cat-with-your-data] '{name}' is not a valid {kind} dataset: {e}")
            yield from self._fallback(blob)
            return

        yield Document(page_content=description, metadata={KIND_KEY: kind, NAME_KEY: name, PAYLOAD_KEY: content})
