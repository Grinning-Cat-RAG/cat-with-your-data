"""Storage of the datasets (CSV and SQLite files) uploaded on the fly.

Datasets live on the local disk, under ``<cat data folder>/cat_with_your_data/<agent>/<scope>/files``, where ``scope``
is either the chat id (datasets visible only in that conversation) or ``_shared`` (datasets visible to every
conversation of the agent).

All the datasets visible in a conversation are exposed to the agents as a single, read-only SQLite database (the
"workspace"): every CSV file becomes a table, every SQLite file brings its own tables. When the conversation sees a
single SQLite file and nothing else, that file is used as-is (read-only), so views, indexes and types are preserved.

Stored files are immutable: every upload is saved under a new name (``<timestamp>-<random>--<name>``, the most recent
wins) and the older copies are removed. So the set of the stored names identifies the content of a scope: the
workspace is named after it and rebuilt when it changes, without relying on sizes or timestamps.
"""
import csv
import hashlib
import io
import json
import re
import shutil
import sqlite3
import tempfile
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Tuple
from urllib.parse import quote

import pandas as pd

from cat import log
from cat.utils import get_data_path

PLUGIN_DATA_DIR = "cat_with_your_data"
SHARED_SCOPE = "_shared"
CHARTS_SCOPE = "_charts"
FILES_DIR = "files"
LAST_USED_FILE = ".last_used"
WORKSPACE_PREFIX = "_workspace_"

CSV_EXTENSIONS = {".csv", ".tsv"}
SQLITE_EXTENSIONS = {".sqlite", ".sqlite3", ".db", ".db3"}
SQLITE_MAGIC = b"SQLite format 3\x00"

_SAFE_ID = re.compile(r"^[A-Za-z0-9_\-.]{1,128}$")
_STORED_NAME = re.compile(r"^\d{20}-[0-9a-f]{8}--(.+)$")


class DatasetError(Exception):
    """Raised when an uploaded dataset is not valid."""


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


def sqlite_uri(path: Path, mode: str = "ro") -> str:
    """SQLite URI filename (to be opened with ``uri=True``); "?", "#" and "%" of the path are escaped."""
    return f"file:{quote(path.as_posix())}?mode={mode}"


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
def sqlite_connect_ro(path: Path, check_same_thread: bool = True) -> sqlite3.Connection:
    return sqlite3.connect(sqlite_uri(path), uri=True, check_same_thread=check_same_thread)


def sqlite_tables(path: Path) -> Dict[str, List[str]]:
    """Tables and views of a SQLite file, with their columns."""
    try:
        conn = sqlite_connect_ro(path)
        try:
            rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table', 'view') AND name NOT LIKE 'sqlite_%' ORDER BY name"
            ).fetchall()
            return {
                name: [r[1] for r in conn.execute(f"PRAGMA table_info({quote_identifier(name)})").fetchall()]
                for (name,) in rows
            }
        finally:
            conn.close()
    except sqlite3.DatabaseError as e:
        raise DatasetError(f"Invalid SQLite file: {e}") from e


def describe_dataset(name: str, content: bytes, kind: str, max_rows: int = 5) -> str:
    """Human (and LLM) readable description of a dataset, used as a memory for the Cat."""
    lines = [f"Dataset '{name}' ({kind.upper()}) uploaded by the user and available for data analysis and charts."]
    if kind == "csv":
        df = read_csv(content, name)
        lines.append(f"Table '{table_name_from(name)}': {len(df)} rows, {len(df.columns)} columns.")
        lines.append("Columns: " + ", ".join(f"{c} ({t})" for c, t in df.dtypes.astype(str).items()))
        lines.append("First rows:\n" + df.head(max_rows).to_markdown(index=False))
        return "\n".join(lines)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "dataset.sqlite"
        path.write_bytes(content)
        tables = sqlite_tables(path)
        conn = sqlite_connect_ro(path)
        try:
            for table, columns in tables.items():
                count = conn.execute(f"SELECT COUNT(*) FROM {quote_identifier(table)}").fetchone()[0]
                lines.append(f"Table '{table}': {count} rows; columns: {', '.join(columns)}.")
        finally:
            conn.close()
    return "\n".join(lines)


# --------------------------------------------------------------------------------------------------------------------
# Store
# --------------------------------------------------------------------------------------------------------------------
class DatasetStore:
    """Datasets visible in a conversation (chat scope) or in the whole agent (shared scope)."""

    def __init__(self, agent_key: str, chat_id: str | None = None):
        self.agent_key = agent_key
        self.chat_id = chat_id
        self.agent_dir = Path(get_data_path()) / PLUGIN_DATA_DIR / _safe_segment(agent_key)
        self.shared_dir = self.agent_dir / SHARED_SCOPE
        self.chat_dir = self.agent_dir / f"chat_{_safe_segment(chat_id)}" if chat_id else None

    # ---------------------------------------------------------------------------------------------------------------
    def _scope_dir(self, shared: bool) -> Path:
        if shared or self.chat_dir is None:
            return self.shared_dir
        return self.chat_dir

    def _workspace_dir(self) -> Path:
        return self.chat_dir or self.shared_dir

    @staticmethod
    def _stored_files(folder: Path) -> Dict[str, List[Path]]:
        """Stored copies of every dataset of a folder, by dataset name, from the oldest to the most recent."""
        files: Dict[str, List[Path]] = {}
        if folder.is_dir():
            for path in sorted(folder.iterdir()):
                if path.is_file() and (match := _STORED_NAME.match(path.name)):
                    files.setdefault(match.group(1), []).append(path)
        return files

    def _visible_files(self) -> List[Tuple[Path, str, str]]:
        """(stored file, scope, dataset name) of the datasets visible in the current scope; chat ones win on clashes."""
        dirs = [(self.shared_dir, "shared")]
        if self.chat_dir is not None:
            dirs.append((self.chat_dir, "chat"))

        files: Dict[str, Tuple[Path, str, str]] = {}
        for folder, scope in dirs:
            for name, copies in self._stored_files(folder / FILES_DIR).items():
                files[name] = (copies[-1], scope, name)
        return [files[name] for name in sorted(files)]

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

        folder = self._scope_dir(shared) / FILES_DIR
        folder.mkdir(parents=True, exist_ok=True)
        tmp_path = folder / f".{uuid.uuid4().hex}.upload"
        tmp_path.write_bytes(content)
        try:
            if kind == "csv":
                df = read_csv(tmp_path, name)
                tables = {table_name_from(name): list(map(str, df.columns))}
            else:
                tables = sqlite_tables(tmp_path)
                if not tables:
                    raise DatasetError("The SQLite file does not contain any table.")
            final_path = folder / f"{time.time_ns():020d}-{uuid.uuid4().hex[:8]}--{name}"
            tmp_path.replace(final_path)
        finally:
            tmp_path.unlink(missing_ok=True)

        # the previous copies are not needed anymore (the requests reading them keep them open until they finish); a
        # more recent copy, stored by a concurrent upload, is kept: it wins
        for old in self._stored_files(folder).get(name, []):
            if old.name < final_path.name:
                old.unlink(missing_ok=True)

        log.info(f"[cat-with-your-data] agent {self.agent_key}: dataset '{name}' stored in {folder}")
        stat = final_path.stat()
        return DatasetInfo(
            name=name,
            kind=kind,
            scope="shared" if (shared or self.chat_dir is None) else "chat",
            size=stat.st_size,
            uploaded_at=stat.st_mtime,
            tables=tables,
        )

    def remove(self, name: str, shared: bool = False) -> bool:
        copies = self._stored_files(self._scope_dir(shared) / FILES_DIR).get(sanitize_filename(name), [])
        for path in copies:
            path.unlink(missing_ok=True)
        return bool(copies)

    def list_datasets(self, with_tables: bool = False) -> List[DatasetInfo]:
        result = []
        for path, scope, name in self._visible_files():
            kind = "sqlite" if Path(name).suffix in SQLITE_EXTENSIONS else "csv"
            tables = {}
            if with_tables:
                try:
                    tables = sqlite_tables(path) if kind == "sqlite" else {
                        table_name_from(name): list(map(str, read_csv(path, name).columns))
                    }
                except DatasetError as e:
                    log.warning(f"[cat-with-your-data] cannot read dataset {name}: {e}")
            stat = path.stat()
            result.append(DatasetInfo(name, kind, scope, stat.st_size, stat.st_mtime, tables))
        return result

    def mark_used(self) -> None:
        """Record that the datasets of the conversation are in use: the expiration counts from the last use."""
        if self.chat_dir is not None and self.chat_dir.is_dir():
            try:
                (self.chat_dir / LAST_USED_FILE).touch()
            except OSError as e:  # the datasets can still be queried
                log.warning(f"[cat-with-your-data] cannot record the use of the datasets of {self.chat_dir}: {e}")

    # ---------------------------------------------------------------------------------------------------------------
    def workspace_path(self) -> Path | None:
        """SQLite file (to be opened read-only) exposing all the datasets visible in the current scope."""
        files = self._visible_files()
        if not files:
            return None

        # a single SQLite file: use it directly
        if len(files) == 1 and Path(files[0][2]).suffix in SQLITE_EXTENSIONS:
            return files[0][0]

        workspace_dir = self._workspace_dir()
        workspace_dir.mkdir(parents=True, exist_ok=True)
        # stored files are immutable: their names identify the content
        fingerprint = hashlib.sha256(json.dumps([str(p) for p, _, _ in files]).encode("utf-8")).hexdigest()[:16]
        workspace = workspace_dir / f"{WORKSPACE_PREFIX}{fingerprint}.sqlite"
        if not workspace.is_file():
            self._build_workspace([(p, name) for p, _, name in files], workspace)
            for old in workspace_dir.glob(f"{WORKSPACE_PREFIX}*.sqlite"):
                if old != workspace:
                    old.unlink(missing_ok=True)
        return workspace

    @staticmethod
    def _build_workspace(files: List[Tuple[Path, str]], workspace: Path) -> None:
        tmp_path = workspace.with_name(f".{uuid.uuid4().hex}.sqlite")
        # URI filenames also for ATTACH, which opens the source files read-only
        conn = sqlite3.connect(sqlite_uri(tmp_path, "rwc"), uri=True)
        used_names: set = set()

        def unique(name: str) -> str:
            candidate, i = name, 2
            while candidate.lower() in used_names:
                candidate, i = f"{name}_{i}", i + 1
            used_names.add(candidate.lower())
            return candidate

        try:
            for path, name in files:
                if Path(name).suffix in SQLITE_EXTENSIONS:
                    conn.execute("ATTACH DATABASE ? AS src", (sqlite_uri(path),))
                    try:
                        tables = [
                            r[0] for r in conn.execute(
                                "SELECT name FROM src.sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
                            ).fetchall()
                        ]
                        for table in tables:
                            target = table if table.lower() not in used_names else f"{table_name_from(name)}_{table}"
                            target = unique(target)
                            conn.execute(
                                f"CREATE TABLE {quote_identifier(target)} AS SELECT * FROM src.{quote_identifier(table)}"
                            )
                        conn.commit()
                    finally:
                        conn.execute("DETACH DATABASE src")
                else:
                    df = read_csv(path, name)
                    df.to_sql(unique(table_name_from(name)), conn, index=False, if_exists="replace")
            conn.commit()
        except Exception:
            conn.close()
            tmp_path.unlink(missing_ok=True)
            raise
        conn.close()
        tmp_path.replace(workspace)

    # ---------------------------------------------------------------------------------------------------------------
    def cleanup_expired(self, ttl_hours: float) -> None:
        """Remove the chat datasets not uploaded nor used, and the charts not created, for more than ``ttl_hours`` hours.

        The datasets of the current conversation and the shared ones are never removed; ``0`` disables the cleanup.
        """
        if not ttl_hours or ttl_hours <= 0 or not self.agent_dir.is_dir():
            return

        threshold = time.time() - ttl_hours * 3600
        for folder in self.agent_dir.iterdir():
            if not folder.is_dir() or (folder.name != CHARTS_SCOPE and not folder.name.startswith("chat_")):
                continue
            try:
                if folder.name == CHARTS_SCOPE:
                    for chart in folder.iterdir():
                        if chart.is_file() and chart.stat().st_mtime < threshold:
                            chart.unlink()
                    continue
                if self.chat_dir is not None and folder == self.chat_dir:
                    continue
                newest = max((p.stat().st_mtime for p in folder.rglob("*")), default=folder.stat().st_mtime)
                if newest < threshold:
                    shutil.rmtree(folder, ignore_errors=True)
            except OSError as e:
                log.warning(f"[cat-with-your-data] cleanup of {folder} failed: {e}")
