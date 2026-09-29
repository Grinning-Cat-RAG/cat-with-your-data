"""Engines on the datasources and read-only queries."""
import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

m = support = None


def setUpModule():
    global m, support
    import support as support_module
    support = support_module
    m = support.load()


class Files(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def csv(self, text="region,amount\nN,1\nS,2\n", name="sales.csv") -> Path:
        path = self.tmp / name
        path.write_text(text)
        return path

    def secret_db(self) -> Path:
        return support.sqlite_file(self.tmp / "secret.db", {"s": (["x"], [("SECRET",)])})


class SqliteHardeningTest(Files):
    def assert_no_attach(self, engine):
        db = m.data_engine.sql_database(engine, cache=False)
        result = db.run_no_throw(f"ATTACH DATABASE '{self.secret_db()}' AS o")
        self.assertIn("Error", result)
        self.assertIn("Error", db.run_no_throw("SELECT x FROM o.s"))

    def test_uploaded_dataset_cannot_attach_other_files(self):
        # regression: the agent could ATTACH (and read) any SQLite file of the host, e.g. other agents' datasets
        self.assert_no_attach(m.data_engine.engine_from_bytes(support.sqlite_bytes({"t": (["v"], [(1,)])})))

    def test_configured_sqlite_is_read_only(self):
        path = support.sqlite_file(self.tmp / "conf.sqlite", {"t": (["v"], [(1,)])})
        engine = m.data_engine.engine_from_uri(f"sqlite:///{path}")
        self.assert_no_attach(engine)
        db = m.data_engine.sql_database(engine, cache=False)
        self.assertIn("Error", db.run_no_throw("CREATE TABLE w (x)"))
        self.assertEqual(db.run("SELECT v FROM t"), "[(1,)]")

    def test_csv_engine_cannot_attach_nor_write(self):
        # regression: DROP TABLE on the cached in-memory engine persisted until the CSV file changed
        engine = m.data_engine.engine_from_csv(str(self.csv()))
        self.assert_no_attach(engine)
        db = m.data_engine.sql_database(engine, cache=False)
        self.assertIn("Error", db.run_no_throw("DROP TABLE sales"))
        self.assertIn("Error", db.run_no_throw("INSERT INTO sales VALUES ('X', 9)"))
        self.assertEqual(db.run("SELECT COUNT(*) FROM sales"), "[(2,)]")

    def test_uploaded_dataset_is_read_only(self):
        db = m.data_engine.sql_database(m.data_engine.engine_from_bytes(support.sqlite_bytes({"t": (["v"], [(1,)])})), cache=False)
        self.assertIn("Error", db.run_no_throw("DELETE FROM t"))
        self.assertEqual(db.run("SELECT COUNT(*) FROM t"), "[(1,)]")

    def test_invalid_uploaded_database(self):
        with self.assertRaises(m.datasets.DatasetError):
            m.data_engine.engine_from_bytes(m.datasets.SQLITE_MAGIC + b"\x00" * 100)


class ReadOnlySessionsTest(Files):
    def test_postgresql_sessions_are_read_only(self):
        import psycopg2

        with mock.patch.object(psycopg2, "connect", side_effect=RuntimeError("no server")) as connect:
            engine = m.data_engine.engine_from_uri(f"postgresql+psycopg2://u:p@localhost:1/ro{id(self)}")
            with self.assertRaises(Exception):
                engine.connect()
        self.assertIn("default_transaction_read_only=on", connect.call_args.kwargs["options"])

    def test_mysql_sessions_are_read_only(self):
        engine = m.data_engine.engine_from_uri(f"mysql+mysqlconnector://u:p@localhost:1/ro{id(self)}")
        self.assertTrue(m.data_engine.event.contains(engine, "connect", m.data_engine._mysql_read_only))
        connection = mock.Mock()
        m.data_engine._mysql_read_only(connection, None)
        connection.cursor.return_value.execute.assert_called_once_with("SET SESSION TRANSACTION READ ONLY")
        connection.cursor.return_value.close.assert_called_once()


class ConcurrencyTest(Files):
    def test_in_memory_engine_under_concurrent_queries(self):
        # regression: the single connection shared by all the threads (StaticPool) returned wrong results
        engine = m.data_engine.engine_from_csv(str(self.csv("n\n" + "\n".join(str(i) for i in range(200)) + "\n")))
        errors = []

        def worker():
            for _ in range(60):
                try:
                    df, _ = m.data_engine.run_select(engine, "SELECT SUM(n) AS s, COUNT(*) AS c FROM sales", 10)
                    if df.to_dict("records") != [{"s": 19900, "c": 200}]:
                        errors.append(df.to_dict("records"))
                except Exception as e:  # noqa: BLE001
                    errors.append(repr(e))

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(errors, [])


class CacheTest(Files):
    def test_csv_engine_is_cached_until_the_file_changes(self):
        path = self.csv()
        first = m.data_engine.engine_from_csv(str(path))
        self.assertIs(m.data_engine.engine_from_csv(str(path)), first)
        path.write_text("region,amount\nN,1\nS,2\nE,3\n")
        second = m.data_engine.engine_from_csv(str(path))
        self.assertIsNot(second, first)
        self.assertEqual(m.data_engine.sql_database(second, cache=False).run("SELECT COUNT(*) FROM sales"), "[(3,)]")

    def test_eviction_releases_engines(self):
        released = []
        with mock.patch.object(m.data_engine, "_MAX_CACHED_ENGINES", 2), \
                mock.patch.object(m.data_engine, "_engines", m.data_engine.OrderedDict()):
            for i in range(3):
                engine = m.data_engine.create_engine("sqlite://")
                m.data_engine._put_cached(f"k{i}", engine, lambda i=i: released.append(i))
            self.assertEqual(released, [0])
            self.assertIsNone(m.data_engine._get_cached("k0"))
            self.assertIsNotNone(m.data_engine._get_cached("k1"))
            # an engine built concurrently for a key already cached is released, the cached one is returned
            cached = m.data_engine._get_cached("k2")
            other = m.data_engine.create_engine("sqlite://")
            self.assertIs(m.data_engine._put_cached("k2", other, lambda: released.append("dup")), cached)
            self.assertEqual(released, [0, "dup"])

    def test_memory_engine_release_closes_the_database(self):
        engine, release = m.data_engine._memory_engine({"t": m.datasets.pd.DataFrame({"v": [1]})})
        self.assertEqual(m.data_engine.sql_database(engine, cache=False).run("SELECT v FROM t"), "[(1,)]")
        release()
        # the shared in-memory database is gone with its keeper connection
        self.assertIn("Error", m.data_engine.sql_database(engine, cache=False).run_no_throw("SELECT v FROM t"))

    def test_memory_engine_load_failure_closes_the_keeper(self):
        frame = mock.Mock()
        frame.to_sql.side_effect = ValueError("bad frame")
        with self.assertRaises(ValueError):
            m.data_engine._memory_engine({"t": frame})

    def test_uri_engines_are_cached(self):
        uri = f"sqlite:///{self.tmp / 'c.sqlite'}"
        self.assertIs(m.data_engine.engine_from_uri(uri), m.data_engine.engine_from_uri(uri))
        # no connection is opened until the first query
        postgres = m.data_engine.engine_from_uri("postgresql+psycopg2://u:p@localhost:1/d")
        self.assertEqual(postgres.dialect.name, "postgresql")

    def test_schema_cache(self):
        engine = m.data_engine.engine_from_csv(str(self.csv()))
        db = m.data_engine.sql_database(engine)
        self.assertIs(m.data_engine.sql_database(engine), db)
        self.assertIsNot(m.data_engine.sql_database(engine, cache=False), db)
        with mock.patch.object(m.data_engine.time, "monotonic", return_value=m.data_engine.time.monotonic() + 10_000):
            self.assertIsNot(m.data_engine.sql_database(engine), db)


class JsonTest(Files):
    def write(self, data) -> str:
        path = self.tmp / f"d{id(data)}.json"
        path.write_text(json.dumps(data))
        return str(path)

    def test_json_to_frames(self):
        frames = m.data_engine.json_to_frames
        self.assertEqual(list(frames([{"a": 1}, {"a": 2}])), ["data"])
        self.assertEqual(sorted(frames({"Orders": [{"a": 1}], "meta": {"x": 1}})), ["orders"])
        self.assertEqual(list(frames({"a": 1, "b": "x"})), ["data"])
        self.assertEqual(frames({"a": {"b": 1}}), {})
        self.assertEqual(frames([1, 2]), {})
        self.assertEqual(frames({}), {})

    def test_engine_from_json(self):
        path = self.write([{"id": 1, "tags": ["a", "b"], "info": {"x": 1}}])
        engine = m.data_engine.engine_from_json(path)
        self.assertIs(m.data_engine.engine_from_json(path), engine)
        db = m.data_engine.sql_database(engine, cache=False)
        self.assertEqual(db.run('SELECT tags, "info.x" FROM data'), "[('[\"a\", \"b\"]', 1)]")
        self.assertIsNone(m.data_engine.engine_from_json(self.write({"a": {"b": 1}})))


class ReadOnlyQueryTest(Files):
    def test_validate_read_only(self):
        valid = m.data_engine.validate_read_only
        self.assertEqual(valid("```sql\nSELECT 1;\n```"), "SELECT 1")
        self.assertEqual(valid("with x as (select 1) select * from x"), "with x as (select 1) select * from x")
        self.assertEqual(valid("SELECT 'drop; table' AS \"insert\" -- delete\n"), "SELECT 'drop; table' AS \"insert\" -- delete")
        for invalid in ("", "   ", "DELETE FROM t", "SELECT 1; DROP TABLE t", "SELECT * INTO x FROM t",
                        "PRAGMA table_info(t)", "WITH d AS (DELETE FROM t RETURNING *) SELECT * FROM d",
                        "select load_extension('x')", "ATTACH 'x' AS y"):
            with self.subTest(sql=invalid), self.assertRaises(ValueError):
                valid(invalid)

    def test_run_select_truncates(self):
        engine = m.data_engine.engine_from_csv(str(self.csv()))
        df, truncated = m.data_engine.run_select(engine, "SELECT * FROM sales", 1)
        self.assertEqual((len(df), truncated), (1, True))
        df, truncated = m.data_engine.run_select(engine, "SELECT * FROM sales", 5)
        self.assertEqual((len(df), truncated), (2, False))

    def test_run_select_passes_no_parameters_to_the_driver(self):
        # regression: with parameters (even empty), pyformat drivers (psycopg2...) read "%" as a placeholder
        engine = m.data_engine.engine_from_csv(str(self.csv("n\nA%\nB\n")))
        with mock.patch.object(engine.dialect, "do_execute", side_effect=AssertionError("parameters passed")):
            df, _ = m.data_engine.run_select(engine, "SELECT n FROM sales WHERE n LIKE 'A%'", 10)
        self.assertEqual(df["n"].tolist(), ["A%"])


if __name__ == "__main__":
    unittest.main()
