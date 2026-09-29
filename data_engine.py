"""Helpers to expose every datasource as a SQLAlchemy engine and to run read-only queries on it."""
import json
import re
import sqlite3
import threading
import time
import uuid
from collections import OrderedDict
from typing import Callable, Dict, Tuple

import pandas as pd
import sqlglot
from langchain_community.utilities import SQLDatabase
from sqlglot import exp
from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine, make_url
from sqlalchemy.pool import NullPool, StaticPool

# sibling modules are looked up at call time
from . import datasets
from .datasets import read_csv, table_name_from

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


def _harden_sqlite(engine: Engine) -> Engine:
    """No ATTACH on the plugin's SQLite connections (it would open any file of the host) and no writes."""
    @event.listens_for(engine, "connect")
    def _on_connect(dbapi_connection, _connection_record):
        dbapi_connection.setlimit(sqlite3.SQLITE_LIMIT_ATTACHED, 0)
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


def _mysql_read_only(dbapi_connection, _connection_record):
    cursor = dbapi_connection.cursor()
    try:
        cursor.execute("SET SESSION TRANSACTION READ ONLY")
    finally:
        cursor.close()


def engine_from_uri(uri: str) -> Engine:
    """Engine for a SQLAlchemy URL (configured SQL datasources); engines are cached to reuse the pools.

    The sessions are read-only where the database allows it (PostgreSQL, MySQL) and the statements of the agent are
    checked (a single read-only SELECT, see ``validate_read_only``); only a database user with read-only privileges
    guarantees that nothing can be changed: configure one.
    """
    def factory():
        backend = make_url(uri).get_backend_name()
        kwargs = {"connect_args": {"options": "-c default_transaction_read_only=on"}} if backend == "postgresql" else {}
        engine = create_engine(uri, pool_pre_ping=True, **kwargs)
        if backend == "mysql":
            event.listen(engine, "connect", _mysql_read_only)
        return engine, engine.dispose

    return _cached(f"uri::{uri}", factory)


def engine_from_bytes(content: bytes) -> Engine:
    """Read-only engine on an in-memory SQLite database (uploaded datasets), for a single request: dispose it at the end.

    The request uses only this connection (the agent runs its tools one at a time, so it is never used concurrently).
    """
    connection = sqlite3.connect(":memory:", check_same_thread=False)
    try:
        connection.deserialize(content)
        connection.execute("SELECT COUNT(*) FROM sqlite_master").fetchone()
    except sqlite3.DatabaseError as e:
        connection.close()
        raise datasets.DatasetError(f"Invalid SQLite database: {e}") from e
    engine = create_engine("sqlite://", creator=lambda: connection, poolclass=StaticPool)
    return _harden_sqlite(engine)


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
    )

    def release():
        engine.dispose()
        keeper.close()

    return engine, release


def engine_from_csv(file: "datasets.ConfiguredFile") -> Engine:
    """In-memory SQLite engine holding the content of a CSV file (one table, named after the file); cached by version."""
    return _cached(
        f"csv::{file.version}",
        lambda: _memory_engine({table_name_from(file.name): read_csv(file.load(), file.name)}),
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


def read_json(file: "datasets.ConfiguredFile"):
    return json.loads(file.load().decode("utf-8-sig"))


def engine_from_json(file: "datasets.ConfiguredFile") -> Engine | None:
    """In-memory SQLite engine for tabular JSON files, cached by version; None when the JSON cannot be represented as
    tables."""
    key = f"json::{file.version}"
    if (engine := _get_cached(key)) is not None:
        return engine

    frames = json_to_frames(read_json(file))
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


_DOLLAR_QUOTE = re.compile(r"\$(?:[A-Za-z_][A-Za-z0-9_]*)?\$")


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


#: SQLAlchemy dialect -> sqlglot dialect
_SQLGLOT_DIALECTS = {"postgresql": "postgres", "mysql": "mysql", "mariadb": "mysql", "oracle": "oracle",
                     "mssql": "tsql", "sqlite": "sqlite"}

#: functions acting outside the data, even in a SELECT: waits, locks, files of the server, other servers, sequences,
#: settings, extensions
_FORBIDDEN_FUNCTIONS = {
    "pg_sleep", "pg_sleep_for", "pg_sleep_until", "pg_read_file", "pg_read_binary_file", "pg_ls_dir", "pg_stat_file",
    "pg_terminate_backend", "pg_cancel_backend", "pg_reload_conf", "pg_rotate_logfile", "pg_advisory_lock",
    "pg_advisory_xact_lock", "pg_try_advisory_lock", "pg_try_advisory_xact_lock", "lo_import", "lo_export", "lo_get",
    "lo_put", "lo_from_bytea", "dblink", "dblink_exec", "dblink_connect", "dblink_send_query", "set_config", "nextval",
    "setval", "txid_current", "query_to_xml", "query_to_xml_and_xmlschema", "cursor_to_xml",
    "load_file", "sleep", "benchmark", "get_lock", "release_lock", "release_all_locks", "master_pos_wait",
    "source_pos_wait", "load_extension", "readfile", "writefile", "edit", "fts3_tokenizer",
    "openrowset", "openquery", "opendatasource", "openxml",
}
#: packages and procedures of Oracle and SQL Server reaching the network, the files or the server
_FORBIDDEN_PREFIXES = ("dbms_", "utl_", "xp_", "sp_", "sys.dbms_", "sys.utl_")


def _sqlglot_dialect(dialect: str | None) -> str | None:
    return _SQLGLOT_DIALECTS.get((dialect or "").lower())


def _check_statement(sql: str, dialect: str | None) -> None:
    """ValueError unless the statement, read as the database of the datasource reads it, is a single query without
    side effects."""
    try:
        statements = [s for s in sqlglot.parse(sql, read=_sqlglot_dialect(dialect)) if s is not None]
    except sqlglot.errors.SqlglotError as e:
        raise ValueError(f"The statement cannot be read: {e}") from e
    if len(statements) != 1:
        raise ValueError("Only a single SQL statement is allowed.")
    statement, = statements
    if not isinstance(statement, exp.Query):
        raise ValueError("Only SELECT statements are allowed.")
    for node in statement.walk():
        if isinstance(node, (exp.DML, exp.DDL, exp.Command, exp.Into, exp.Lock, exp.Set, exp.Pragma, exp.Transaction)):
            raise ValueError(f"'{node.key.upper()}' is not allowed in a read-only query.")
        names = []
        if isinstance(node, exp.Func):
            names.append(node.name if isinstance(node, exp.Anonymous) else node.sql_name())
        if isinstance(node, (exp.Identifier, exp.Dot)):
            names.append(node.name if isinstance(node, exp.Identifier) else node.sql(dialect=_sqlglot_dialect(dialect)))
        for name in (n.lower() for n in names if n):
            if name in _FORBIDDEN_FUNCTIONS or name.startswith(_FORBIDDEN_PREFIXES):
                raise ValueError(f"'{name}' is not allowed in a read-only query.")


def validate_read_only(sql: str, dialect: str | None = None) -> str:
    """Return the cleaned statement, raising ValueError if it is not a single read-only SELECT without side effects.

    ``dialect`` is the one of the database (SQLAlchemy name, e.g. ``postgresql``): the statement is parsed as that
    database reads it. The checks cannot foresee every function of every database: only a database user with read-only
    privileges guarantees that nothing can be changed.
    """
    sql = clean_sql(sql)
    if not sql:
        raise ValueError("Empty SQL statement.")

    # quoting that the databases read differently (PostgreSQL dollar quoting, backslash escapes of MySQL and of
    # PostgreSQL E'' strings): a literal here could be statements there
    if _DOLLAR_QUOTE.search(sql) or "\\" in sql:
        raise ValueError("Dollar quoting and backslashes are not allowed.")
    bare = _strip_literals(sql)
    if "#" in bare:
        raise ValueError("Comments starting with # are not allowed.")
    if ";" in bare:
        raise ValueError("Only a single SQL statement is allowed.")
    if not re.match(r"^\s*(select|with)\b", bare, re.IGNORECASE):
        raise ValueError("Only SELECT statements are allowed.")
    if match := _FORBIDDEN_KEYWORDS.search(bare):
        raise ValueError(f"Keyword '{match.group(1).upper()}' is not allowed in a read-only query.")
    _check_statement(sql, dialect)
    return sql


def run_select(engine: Engine, sql: str, max_rows: int) -> Tuple[pd.DataFrame, bool]:
    """Run a read-only SELECT and return (dataframe, truncated). The transaction is always rolled back."""
    sql = validate_read_only(sql, engine.dialect.name)
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
