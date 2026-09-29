"""Helpers to expose every datasource as a SQLAlchemy engine and to run read-only queries on it."""
import json
import re
import sqlite3
import threading
import time
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Callable, Dict, Tuple

import pandas as pd
from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.pool import NullPool, StaticPool

from .datasets import read_csv, sqlite_connect_ro, table_name_from

_MAX_CACHED_ENGINES = 16
# cache key -> (engine, function releasing the resources of the engine)
_engines: "OrderedDict[str, Tuple[Engine, Callable[[], None]]]" = OrderedDict()
_engines_lock = threading.Lock()


def _get_cached(key: str) -> Engine | None:
    with _engines_lock:
        if key not in _engines:
            return None
        _engines.move_to_end(key)
        return _engines[key][0]


def _put_cached(key: str, engine: Engine, release: Callable[[], None]) -> Engine:
    """Cache the engine and return the cached one (another thread may have built the same engine meanwhile)."""
    evicted = []
    with _engines_lock:
        if key in _engines:
            evicted.append(release)
            engine = _engines[key][0]
        else:
            _engines[key] = (engine, release)
        while len(_engines) > _MAX_CACHED_ENGINES:
            _, (_, old_release) = _engines.popitem(last=False)
            evicted.append(old_release)
    for release_engine in evicted:
        release_engine()
    return engine


def _cached(key: str, factory: Callable[[], Tuple[Engine, Callable[[], None]]]) -> Engine:
    return _get_cached(key) or _put_cached(key, *factory())


def _harden_sqlite(engine: Engine, query_only: bool) -> Engine:
    """No ATTACH on the plugin's SQLite connections (it would open any file of the host) and, if required, no writes."""
    @event.listens_for(engine, "connect")
    def _on_connect(dbapi_connection, _connection_record):
        dbapi_connection.setlimit(sqlite3.SQLITE_LIMIT_ATTACHED, 0)
        if query_only:
            dbapi_connection.execute("PRAGMA query_only = ON")

    return engine


_SCHEMA_TTL_SECONDS = 600
_databases: Dict[int, Tuple[float, SQLDatabase]] = {}


def sql_database(engine: Engine, cache: bool = True) -> SQLDatabase:
    """LangChain wrapper of the engine; it reflects the schema, so it is cached for a few minutes when ``cache``."""
    if not cache:
        return SQLDatabase(engine, sample_rows_in_table_info=3)

    now = time.monotonic()
    with _engines_lock:
        cached = _databases.get(id(engine))
        if cached and now - cached[0] < _SCHEMA_TTL_SECONDS and cached[1]._engine is engine:
            return cached[1]

    db = SQLDatabase(engine, sample_rows_in_table_info=3)
    with _engines_lock:
        for key in [k for k, (ts, _) in _databases.items() if now - ts >= _SCHEMA_TTL_SECONDS]:
            _databases.pop(key, None)
        _databases[id(engine)] = (now, db)
    return db


def engine_from_uri(uri: str) -> Engine:
    """Engine for a SQLAlchemy URL (configured SQL datasources); engines are cached to reuse the pools."""
    def factory():
        engine = create_engine(uri, pool_pre_ping=True)
        if engine.dialect.name == "sqlite":
            _harden_sqlite(engine, query_only=False)
        return engine, engine.dispose

    return _cached(f"uri::{uri}", factory)


def engine_from_sqlite_file(path: Path) -> Engine:
    """Read-only engine on a SQLite file (uploaded datasets), for a single request: dispose it at the end.

    The file is opened (and read) here, and the request uses only this connection: the open file stays readable until
    the request ends, even if a new upload or a new workspace removes it meanwhile. The agent runs its tools one at a
    time, so the connection is never used concurrently.
    """
    connection = sqlite_connect_ro(path, check_same_thread=False)
    try:
        connection.execute("SELECT COUNT(*) FROM sqlite_master").fetchone()
    except sqlite3.Error:
        connection.close()
        raise
    engine = create_engine("sqlite://", creator=lambda: connection, poolclass=StaticPool)
    return _harden_sqlite(engine, query_only=True)


def _memory_engine(frames: Dict[str, pd.DataFrame]) -> Tuple[Engine, Callable[[], None]]:
    """Read-only, in-memory SQLite database holding the frames, shared by all the connections of the engine.

    Every connection is a distinct SQLite connection to the same shared-cache database (so concurrent requests never
    share a connection); the database lives as long as the ``keeper`` connection, which is closed on release.
    """
    uri = f"file:cwyd_{uuid.uuid4().hex}?mode=memory&cache=shared"
    keeper = sqlite3.connect(uri, uri=True, check_same_thread=False)
    try:
        for name, df in frames.items():
            df.to_sql(name, keeper, index=False, if_exists="replace")
        keeper.commit()
    except Exception:
        keeper.close()
        raise

    engine = _harden_sqlite(
        create_engine(
            "sqlite://",
            creator=lambda: sqlite3.connect(uri, uri=True, check_same_thread=False),
            poolclass=NullPool,
        ),
        query_only=True,
    )

    def release():
        engine.dispose()
        keeper.close()

    return engine, release


def _file_key(prefix: str, path: Path) -> str:
    stat = path.stat()
    return f"{prefix}::{path.resolve()}::{stat.st_size}::{stat.st_mtime_ns}::{stat.st_ino}"


def engine_from_csv(path: str) -> Engine:
    """In-memory SQLite engine holding the content of a CSV file (one table, named after the file)."""
    file_path = Path(path)
    return _cached(
        _file_key("csv", file_path),
        lambda: _memory_engine({table_name_from(file_path.name): read_csv(file_path)}),
    )


def json_to_frames(data) -> Dict[str, pd.DataFrame]:
    """Best-effort conversion of JSON data into tables: a list of records, or an object of lists of records."""
    if isinstance(data, list) and data and all(isinstance(x, dict) for x in data):
        return {"data": pd.json_normalize(data)}
    if isinstance(data, dict):
        frames = {
            table_name_from(key): pd.json_normalize(value)
            for key, value in data.items()
            if isinstance(value, list) and value and all(isinstance(x, dict) for x in value)
        }
        if frames:
            return frames
        if data and all(not isinstance(v, (dict, list)) for v in data.values()):
            return {"data": pd.json_normalize(data)}
    return {}


def engine_from_json(path: str) -> Engine | None:
    """In-memory SQLite engine for tabular JSON files; None when the JSON cannot be represented as tables."""
    file_path = Path(path)
    key = _file_key("json", file_path)
    if (engine := _get_cached(key)) is not None:
        return engine

    frames = json_to_frames(json.loads(file_path.read_text(encoding="utf-8")))
    if not frames:
        return None
    for name, df in frames.items():
        # nested values cannot be stored in SQLite
        frames[name] = df.map(lambda v: json.dumps(v) if isinstance(v, (list, dict)) else v)
    return _put_cached(key, *_memory_engine(frames))


# --------------------------------------------------------------------------------------------------------------------
# read-only queries
# --------------------------------------------------------------------------------------------------------------------
_FORBIDDEN_KEYWORDS = re.compile(
    r"\b(insert|update|delete|merge|upsert|drop|alter|create|truncate|grant|revoke|attach|detach|pragma|vacuum|"
    r"reindex|call|exec|execute|into|load_extension|outfile|dumpfile)\b",
    re.IGNORECASE,
)


def _strip_literals(sql: str) -> str:
    """Remove comments, string literals and quoted identifiers, keeping only the SQL keywords."""
    sql = re.sub(r"--[^\n]*", " ", sql)
    sql = re.sub(r"/\*.*?\*/", " ", sql, flags=re.DOTALL)
    sql = re.sub(r"'(?:[^']|'')*'", "''", sql)
    sql = re.sub(r'"(?:[^"]|"")*"', '""', sql)
    sql = re.sub(r"`[^`]*`", "``", sql)
    sql = re.sub(r"\[[^\]]*\]", "[]", sql)
    return sql


def clean_sql(sql: str) -> str:
    sql = (sql or "").strip()
    sql = re.sub(r"^```(?:sql)?\s*|\s*```$", "", sql, flags=re.IGNORECASE).strip()
    return sql.rstrip(";").strip()


def validate_read_only(sql: str) -> str:
    """Return the cleaned statement, raising ValueError if it is not a single read-only SELECT."""
    sql = clean_sql(sql)
    if not sql:
        raise ValueError("Empty SQL statement.")

    bare = _strip_literals(sql)
    if ";" in bare:
        raise ValueError("Only a single SQL statement is allowed.")
    if not re.match(r"^\s*(select|with)\b", bare, re.IGNORECASE):
        raise ValueError("Only SELECT statements are allowed.")
    if match := _FORBIDDEN_KEYWORDS.search(bare):
        raise ValueError(f"Keyword '{match.group(1).upper()}' is not allowed in a read-only query.")
    return sql


def run_select(engine: Engine, sql: str, max_rows: int) -> Tuple[pd.DataFrame, bool]:
    """Run a read-only SELECT and return (dataframe, truncated). The transaction is always rolled back."""
    sql = validate_read_only(sql)
    with engine.connect() as conn:
        try:
            # no parameters: the drivers with the "pyformat" style (psycopg2, mysql-connector, pymssql) would
            # otherwise read the "%" of the statement (e.g. LIKE 'A%') as placeholders
            result = conn.execution_options(no_parameters=True).exec_driver_sql(sql)
            columns = list(result.keys())
            rows = result.fetchmany(max_rows + 1)
        finally:
            conn.rollback()

    truncated = len(rows) > max_rows
    df = pd.DataFrame([tuple(r) for r in rows[:max_rows]], columns=columns)
    return df, truncated
