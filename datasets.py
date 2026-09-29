"""Storage of the datasets (CSV and SQLite files) uploaded on the fly, in the file manager of the agent.

The datasets are stored with the file manager configured for the agent (local folder, S3, ...), the same service that
the core uses for the uploaded files; the plugin never writes on the local disk of the instance (pod). Paths, relative
to the root of the file manager:

- ``<agent>/<chat>/cat_with_your_data/``: the datasets of a conversation (removed with the conversation);
- ``<agent>/cat_with_your_data/``: the datasets shared by every conversation of the agent (removed with the agent);
- ``<agent>/cat_with_your_data/_chats/<chat>``: the last activity of every conversation with datasets, used to remove
  the datasets of the idle conversations.

The file managers offer neither renames nor locks, and they may be shared by several instances of the Cat, so the
stored files are immutable objects: every upload is written under a new name, ``<timestamp>-<checksum>--<name>``, the
most recent copy of a name wins (its timestamp is always later than the one of the existing copies, whatever the clock
of the instance) and the older copies are removed. A request downloads the files it needs and works on them in memory
(a partially written file is recognized by its checksum, and ignored until complete): what is removed or uploaded
meanwhile, on any instance, never affects it.

All the datasets visible in a conversation are exposed to the agents as a single, read-only SQLite database (the
"workspace"), built in memory: every CSV file becomes a table, every SQLite file brings its own tables. When the
conversation sees a single SQLite file and nothing else, that file is used as-is, so views, indexes and types are
preserved. The workspaces are cached by every instance: their content is identified by the (immutable) stored names.
"""
import csv
import hashlib
import io
import re
import sqlite3
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from cat import log

PLUGIN_DATA_DIR = "cat_with_your_data"
CHATS_INDEX = "_chats"

CSV_EXTENSIONS = {".csv", ".tsv"}
SQLITE_EXTENSIONS = {".sqlite", ".sqlite3", ".db", ".db3"}
SQLITE_MAGIC = b"SQLite format 3\x00"

_SAFE_ID = re.compile(r"^[A-Za-z0-9_\-.]{1,128}$")
_MARKER_WIDTH = 20  # last activity of a conversation: seconds, e.g. 0001790000000.123456
_STORED_NAME = re.compile(r"^(\d{20})-([0-9a-f]{16})--(.+)$")

#: workspaces cached by this instance (stored names -> serialized SQLite database), at most this many bytes
WORKSPACE_CACHE_BYTES = 256 * 1024 * 1024
_workspaces: "OrderedDict[Tuple[str, ...], bytes]" = OrderedDict()
_workspaces_lock = threading.Lock()


class DatasetError(Exception):
    """Raised when an uploaded dataset is not valid."""


class _IncompleteCopy(Exception):
    """A stored file whose content does not match its checksum: being written, or its upload was interrupted."""


@dataclass
class DatasetInfo:
    name: str
    kind: str  # "csv" | "sqlite"
    scope: str  # "chat" | "shared"
    size: int
    uploaded_at: float
    tables: Dict[str, List[str]] = field(default_factory=dict)  # table name -> column names

    def to_dict(self) -> Dict:
        return asdict(self)


def _safe_segment(value: str) -> str:
    """Return a string that can be safely used as a folder name (no path traversal)."""
    value = str(value or "")
    if _SAFE_ID.match(value) and value not in {".", ".."}:
        return value
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:32]


def sanitize_filename(filename: str) -> str:
    name = Path(str(filename or "")).name  # drop any directory component
    stem, ext = Path(name).stem, Path(name).suffix.lower()
    stem = re.sub(r"[^A-Za-z0-9_\-]+", "_", stem).strip("_") or f"dataset_{uuid.uuid4().hex[:8]}"
    return f"{stem[:80]}{ext}"


def table_name_from(value: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_]+", "_", Path(str(value)).stem).strip("_").lower() or "dataset"
    if name[0].isdigit():
        name = f"t_{name}"
    return name[:60]


def detect_kind(filename: str, content: bytes) -> str | None:
    """Detect the type of dataset from the magic number (SQLite) or from the extension (CSV)."""
    if content[:16] == SQLITE_MAGIC:
        return "sqlite"
    ext = Path(str(filename or "")).suffix.lower()
    if ext in CSV_EXTENSIONS:
        return "csv"
    return None


def quote_identifier(name: str) -> str:
    """SQL identifier quoted for SQLite (the names come from the uploaded files)."""
    return '"' + str(name).replace('"', '""') + '"'


# --------------------------------------------------------------------------------------------------------------------
# CSV helpers
# --------------------------------------------------------------------------------------------------------------------
def _sniff_separator(sample: str, ext: str) -> str:
    if ext == ".tsv":
        return "\t"
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        return ","


_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}([ T]\d{2}:\d{2}(:\d{2})?)?")
_SLASH_DATE = re.compile(r"^(\d{1,2})[/.\-](\d{1,2})[/.\-](\d{4})(\s+\d{1,2}:\d{2}(:\d{2})?)?$")


def _normalize_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Convert columns holding dd/mm/yyyy (or mm/dd/yyyy) dates to ISO strings, so SQL can sort and group them."""
    for col in df.columns:
        if not (df[col].dtype == object or pd.api.types.is_string_dtype(df[col])):
            continue
        values = df[col].dropna().astype(str).str.strip()
        if values.empty:
            continue
        sample = values.head(500)
        matches = sample.str.match(_SLASH_DATE)
        if matches.mean() < 0.95:
            continue
        parts = sample[matches].str.extract(_SLASH_DATE)
        first, second = parts[0].astype(int), parts[1].astype(int)
        dayfirst = not (second > 12).any() or (first > 12).any()
        parsed = pd.to_datetime(df[col], dayfirst=dayfirst, errors="coerce")
        if parsed.notna().sum() < 0.95 * df[col].notna().sum():
            continue
        has_time = (parsed.dropna().dt.normalize() != parsed.dropna()).any()
        df[col] = parsed.dt.strftime("%Y-%m-%d %H:%M:%S" if has_time else "%Y-%m-%d")
    return df


_EU_NUMBER = re.compile(r"^[+-]?(\d{1,3}(\.\d{3})+|\d+)(,\d+)?$")
_EU_THOUSANDS = re.compile(r"^[+-]?\d{1,3}(\.\d{3})+$")


def _infer_semicolon_numbers(df: pd.DataFrame) -> pd.DataFrame:
    """Numbers of the files separated by ";" (read as text), column by column.

    A column is read with "," as decimal mark and "." as thousands separator when all its values are written that way
    and at least one of them uses the comma, or all the values with a dot are thousands groups (``1.200``); otherwise
    the usual format (``1.5``) is tried. Columns that are not numeric stay text.
    """
    for col in df.columns:
        values = df[col].dropna().astype(str).str.strip()
        values = values[values != ""]
        if values.empty:
            continue
        dotted = values[values.str.contains(".", regex=False)]
        european = values.str.match(_EU_NUMBER).all() and (
            values.str.contains(",", regex=False).any() or (not dotted.empty and dotted.str.match(_EU_THOUSANDS).all())
        )
        text = df[col].astype("string").str.strip()
        if european:
            text = text.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
        converted = pd.to_numeric(text.replace("", pd.NA), errors="coerce")
        if converted.notna().sum() == len(values):
            df[col] = converted
    return df


def _decode(raw: bytes) -> str:
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1")  # every byte is a latin-1 character: it never fails


def read_csv(source: Path | bytes, filename: str | None = None) -> pd.DataFrame:
    """Read a CSV file detecting separator and encoding."""
    ext = Path(str(filename or (source if isinstance(source, Path) else ""))).suffix.lower()
    text = _decode(source.read_bytes() if isinstance(source, Path) else source)
    separator = _sniff_separator(text[:65536], ext)
    # ";" is the separator of the CSV files exported with European locales, which often (not always) use "," as
    # decimal mark: the numbers are read as text and converted column by column
    semicolon = separator == ";"
    try:
        df = pd.read_csv(io.StringIO(text), sep=separator, low_memory=False, **({"dtype": str} if semicolon else {}))
    except Exception as e:  # noqa: BLE001
        raise DatasetError(f"Invalid CSV file: {e}") from e
    df.columns = [str(c).strip() or f"column_{i}" for i, c in enumerate(df.columns)]
    if semicolon:
        df = _infer_semicolon_numbers(df)
    return _normalize_dates(df)


# --------------------------------------------------------------------------------------------------------------------
# SQLite helpers
# --------------------------------------------------------------------------------------------------------------------
def _open_sqlite(content: bytes) -> sqlite3.Connection:
    """In-memory SQLite database with the given content (a serialized database, i.e. the bytes of a SQLite file)."""
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.deserialize(content)  # not validated here: an invalid file fails in the first query, as DatasetError
    return conn


def _tables_of(conn: sqlite3.Connection, counts: bool = False) -> Dict[str, List[str]]:
    """Tables and views of a SQLite database, with their columns (and, if ``counts``, the number of rows as last item)."""
    try:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table', 'view') AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
        tables = {
            name: [r[1] for r in conn.execute(f"PRAGMA table_info({quote_identifier(name)})").fetchall()]
            for (name,) in rows
        }
        if counts:
            for name, columns in tables.items():
                columns.append(conn.execute(f"SELECT COUNT(*) FROM {quote_identifier(name)}").fetchone()[0])
        return tables
    except sqlite3.DatabaseError as e:
        raise DatasetError(f"Invalid SQLite file: {e}") from e


def sqlite_tables(content: bytes) -> Dict[str, List[str]]:
    """Tables and views of a SQLite file (its bytes), with their columns."""
    conn = _open_sqlite(content)
    try:
        return _tables_of(conn)
    finally:
        conn.close()


def describe_dataset(name: str, content: bytes, kind: str, max_rows: int = 5) -> str:
    """Human (and LLM) readable description of a dataset, used as a memory for the Cat."""
    lines = [f"Dataset '{name}' ({kind.upper()}) uploaded by the user and available for data analysis and charts."]
    if kind == "csv":
        df = read_csv(content, name)
        lines.append(f"Table '{table_name_from(name)}': {len(df)} rows, {len(df.columns)} columns.")
        lines.append("Columns: " + ", ".join(f"{c} ({t})" for c, t in df.dtypes.astype(str).items()))
        lines.append("First rows:\n" + df.head(max_rows).to_markdown(index=False))
        return "\n".join(lines)

    conn = _open_sqlite(content)
    try:
        for table, columns in _tables_of(conn, counts=True).items():
            *columns, count = columns
            lines.append(f"Table '{table}': {count} rows; columns: {', '.join(columns)}.")
    finally:
        conn.close()
    return "\n".join(lines)


def _checksum(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()[:16]


def build_workspace(files: List[Tuple[str, bytes]]) -> bytes:
    """Serialized SQLite database with the tables of the given datasets ((name, content), in this order)."""
    conn = sqlite3.connect(":memory:")
    used_names: set = set()

    def unique(name: str) -> str:
        candidate, i = name, 2
        while candidate.lower() in used_names:
            candidate, i = f"{name}_{i}", i + 1
        used_names.add(candidate.lower())
        return candidate

    try:
        for name, content in files:
            if Path(name).suffix in SQLITE_EXTENSIONS:
                conn.execute("ATTACH DATABASE ':memory:' AS src")
                try:
                    conn.deserialize(content, name="src")
                    tables = [
                        r[0] for r in conn.execute(
                            "SELECT name FROM src.sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
                        ).fetchall()
                    ]
                    for table in tables:
                        target = table if table.lower() not in used_names else f"{table_name_from(name)}_{table}"
                        conn.execute(
                            f"CREATE TABLE {quote_identifier(unique(target))} AS SELECT * FROM src.{quote_identifier(table)}"
                        )
                    conn.commit()
                finally:
                    conn.execute("DETACH DATABASE src")
            else:
                read_csv(content, name).to_sql(unique(table_name_from(name)), conn, index=False, if_exists="replace")
        conn.commit()
        return conn.serialize()
    except sqlite3.DatabaseError as e:
        raise DatasetError(f"Invalid SQLite file: {e}") from e
    finally:
        conn.close()


def _cache_workspace(key: Tuple[str, ...], content: bytes) -> None:
    with _workspaces_lock:
        _workspaces[key] = content
        _workspaces.move_to_end(key)
        while sum(len(v) for v in _workspaces.values()) > WORKSPACE_CACHE_BYTES and len(_workspaces) > 1:
            _workspaces.popitem(last=False)


def _cached_workspace(key: Tuple[str, ...]) -> bytes | None:
    with _workspaces_lock:
        if key in _workspaces:
            _workspaces.move_to_end(key)
            return _workspaces[key]
        return None


# --------------------------------------------------------------------------------------------------------------------
# Store
# --------------------------------------------------------------------------------------------------------------------
class DatasetStore:
    """Datasets visible in a conversation (chat scope) or in the whole agent (shared scope), in a file manager."""

    def __init__(self, file_manager, agent_key: str, chat_id: str | None = None):
        self.file_manager = file_manager
        self.agent_key = agent_key
        self.chat_id = chat_id
        agent = _safe_segment(agent_key)
        self.shared_dir = f"{agent}/{PLUGIN_DATA_DIR}"
        self.index_dir = f"{self.shared_dir}/{CHATS_INDEX}"
        self.chat_segment = _safe_segment(chat_id) if chat_id else None
        # the folder of the conversation is the one of the core: the datasets are removed with the conversation
        self.chat_dir = f"{agent}/{self.chat_segment}/{PLUGIN_DATA_DIR}" if chat_id else None

    # ---------------------------------------------------------------------------------------------------------------
    def _scope_dir(self, shared: bool) -> str:
        if shared or self.chat_dir is None:
            return self.shared_dir
        return self.chat_dir

    def _stored_files(self, folder: str) -> Dict[str, List[str]]:
        """Stored copies of every dataset of a folder, by dataset name, from the oldest to the most recent."""
        files: Dict[str, List[str]] = {}
        for item in sorted(self.file_manager.list_files(folder), key=lambda f: f.name):
            if match := _STORED_NAME.match(item.name):
                files.setdefault(match.group(3), []).append(item.name)
        return files

    def _visible_files(self) -> List[Tuple[str, List[Tuple[str, str]]]]:
        """(dataset name, its stored copies as (scope, path), from the least to the most preferred) by name.

        The copies of the conversation come after the shared ones, since they win on name clashes; the copies of a scope
        are sorted from the oldest.
        """
        dirs = [(self.shared_dir, "shared")]
        if self.chat_dir is not None:
            dirs.append((self.chat_dir, "chat"))

        files: Dict[str, List[Tuple[str, str]]] = {}
        for folder, scope in dirs:
            for name, copies in self._stored_files(folder).items():
                files.setdefault(name, []).extend((scope, f"{folder}/{stored}") for stored in copies)
        return [(name, files[name]) for name in sorted(files)]

    def _download(self, path: str) -> bytes:
        """Content of a stored file: FileNotFoundError if it was removed, _IncompleteCopy if its checksum is wrong."""
        content = self.file_manager.download_file(path)
        if content is None:
            raise FileNotFoundError(f"{path} was removed")
        if _checksum(content) != _STORED_NAME.match(Path(path).name).group(2):
            raise _IncompleteCopy(path)
        return content

    def _content(self, copies: List[Tuple[str, str]]) -> Tuple[Tuple[str, str, bytes] | None, bool]:
        """(scope, path, content) of the most preferred complete copy (None if there is none), and whether a more
        preferred copy was skipped because incomplete: an upload still running (on any instance) is not visible yet,
        and an interrupted one never is.

        FileNotFoundError if a copy was removed meanwhile (replaced on another instance: the listing is stale): the
        caller can retry.
        """
        skipped = False
        for scope, path in reversed(copies):
            try:
                return (scope, path, self._download(path)), skipped
            except _IncompleteCopy:
                log.debug(f"[cat-with-your-data] {path} is incomplete (being written, or interrupted): ignored")
                skipped = True
        return None, skipped

    def _new_stored_name(self, folder: str, name: str, content: bytes) -> str:
        # later than every existing copy: the order of the uploads does not depend on the clocks of the instances
        latest = max((int(n[:20]) for n in self._stored_files(folder).get(name, [])), default=0)
        return f"{max(time.time_ns(), latest + 1):020d}-{_checksum(content)}--{name}"

    # ---------------------------------------------------------------------------------------------------------------
    def add(self, filename: str, content: bytes, shared: bool = False, max_bytes: int | None = None) -> DatasetInfo:
        if not content:
            raise DatasetError("The file is empty.")
        if max_bytes and len(content) > max_bytes:
            raise DatasetError(f"The file exceeds the maximum allowed size of {max_bytes // (1024 * 1024)} MB.")

        kind = detect_kind(filename, content)
        if kind is None:
            raise DatasetError("Only CSV (.csv, .tsv) and SQLite files are supported.")

        # CSV files are recognized by their extension, SQLite files by their content (any extension)
        name = sanitize_filename(filename)
        if kind == "sqlite" and Path(name).suffix not in SQLITE_EXTENSIONS:
            name = f"{Path(name).stem}.sqlite"

        if kind == "csv":
            tables = {table_name_from(name): list(map(str, read_csv(content, name).columns))}
        else:
            tables = sqlite_tables(content)
            if not tables:
                raise DatasetError("The SQLite file does not contain any table.")

        # the activity is recorded first: if this instance dies in the middle of the write, the incomplete copy is
        # removed with the idle conversation
        if not shared:
            self.mark_used()
        folder = self._scope_dir(shared)
        stored = self._new_stored_name(folder, name, content)
        if not self.file_manager.write_file(content, stored, folder):
            raise DatasetError("The dataset cannot be stored: see the log of the Cat.")
        copies = self._stored_files(folder).get(name, [])
        # a more recent copy, stored by a concurrent upload, may have replaced this one already
        if stored not in copies and not any(copy > stored for copy in copies):
            raise DatasetError(
                "The file manager of the agent does not keep the files: configure a file manager to upload datasets."
            )

        # the previous copies are not needed anymore (the requests using them already downloaded them); a more recent
        # copy, stored by a concurrent upload, is kept: it wins
        for old in copies:
            if old < stored:
                self.file_manager.remove_file(f"{folder}/{old}")

        log.info(f"[cat-with-your-data] agent {self.agent_key}: dataset '{name}' stored in {folder}")
        return DatasetInfo(
            name=name,
            kind=kind,
            scope="shared" if (shared or self.chat_dir is None) else "chat",
            size=len(content),
            uploaded_at=int(stored[:20]) / 1e9,
            tables=tables,
        )

    def remove(self, name: str, shared: bool = False) -> bool:
        folder = self._scope_dir(shared)
        copies = self._stored_files(folder).get(sanitize_filename(name), [])
        for stored in copies:
            self.file_manager.remove_file(f"{folder}/{stored}")
        return bool(copies)

    def list_datasets(self, with_tables: bool = False) -> List[DatasetInfo]:
        result = []
        for name, copies in self._visible_files():
            kind = "sqlite" if Path(name).suffix in SQLITE_EXTENSIONS else "csv"
            try:
                if (found := self._content(copies)[0]) is None:
                    continue
            except FileNotFoundError as e:  # removed or replaced meanwhile (on another instance)
                log.warning(f"[cat-with-your-data] cannot read dataset {name}: {e}")
                continue
            scope, path, content = found
            tables = {}
            if with_tables:
                tables = sqlite_tables(content) if kind == "sqlite" else {
                    table_name_from(name): list(map(str, read_csv(content, name).columns))
                }
            result.append(DatasetInfo(name, kind, scope, len(content), int(Path(path).name[:20]) / 1e9, tables))
        return result

    def mark_used(self) -> None:
        """Record the last activity of the conversation: its datasets expire when idle for a while."""
        if self.chat_segment is not None:
            # fixed width: a marker read while being written (in place) is recognized by its length
            self.file_manager.write_file(f"{time.time():0{_MARKER_WIDTH}.6f}", self.chat_segment, self.index_dir)

    # ---------------------------------------------------------------------------------------------------------------
    def workspace(self) -> bytes | None:
        """Serialized SQLite database exposing all the datasets visible in the current scope (None without datasets).

        FileNotFoundError if a dataset was removed or replaced meanwhile (on any instance): the caller can retry.
        """
        files = self._visible_files()
        if not files:
            return None

        # stored files are immutable: their names identify the content
        key = tuple(path for _, copies in files for _, path in copies)
        if (cached := _cached_workspace(key)) is not None:
            return cached

        contents, incomplete = [], False
        for name, copies in files:
            found, skipped = self._content(copies)
            incomplete = incomplete or skipped
            if found is not None:
                contents.append((name, found[2]))
        if not contents:
            return None
        if len(contents) == 1 and Path(contents[0][0]).suffix in SQLITE_EXTENSIONS:
            workspace = contents[0][1]  # a single SQLite file: used as it is
        else:
            workspace = build_workspace(contents)
        # without the copies being written: it would become stale as soon as they are complete
        if not incomplete:
            _cache_workspace(key, workspace)
        return workspace

    # ---------------------------------------------------------------------------------------------------------------
    def cleanup_expired(self, ttl_hours: float) -> None:
        """Remove the datasets of the conversations idle (no uploads, no questions) for more than ``ttl_hours`` hours.

        The datasets of the current conversation and the shared ones are never removed; ``0`` disables the cleanup.
        Only the datasets uploaded before the threshold are removed, so an upload running meanwhile on another instance
        is never lost.
        """
        if not ttl_hours or ttl_hours <= 0:
            return

        threshold = time.time() - ttl_hours * 3600
        for marker in self.file_manager.list_files(self.index_dir):
            chat_segment = marker.name
            if chat_segment == self.chat_segment:
                continue
            try:
                if self._last_activity(chat_segment) >= threshold:
                    continue
                folder = f"{_safe_segment(self.agent_key)}/{chat_segment}/{PLUGIN_DATA_DIR}"
                for copies in self._stored_files(folder).values():
                    for stored in copies:
                        if int(stored[:20]) / 1e9 < threshold:
                            self.file_manager.remove_file(f"{folder}/{stored}")
                # used again meanwhile (on another instance): the marker stays
                if self._last_activity(chat_segment) < threshold:
                    self.file_manager.remove_file(f"{self.index_dir}/{chat_segment}")
            except Exception as e:  # noqa: BLE001 - the cleanup of a conversation never stops the others
                log.warning(f"[cat-with-your-data] cleanup of the conversation {chat_segment} failed: {e}")

    def _last_activity(self, chat_segment: str) -> float:
        content = self.file_manager.download_file(f"{self.index_dir}/{chat_segment}")
        if content is None:
            return float("inf")  # removed meanwhile: nothing to do
        text = content.decode("utf-8", errors="replace")
        if len(text) != _MARKER_WIDTH:
            return float("inf")  # being written (on another instance): certainly recent
        return float(text)
