"""Storage of the uploaded datasets, CSV reading and workspace building."""
import os
import sqlite3
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

m = support = None


def setUpModule():
    global m, support
    import support as support_module
    support = support_module
    m = support.load()


def query(path: Path, sql: str):
    conn = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    try:
        return conn.execute(sql).fetchall()
    finally:
        conn.close()


class NamesTest(unittest.TestCase):
    def test_safe_segment(self):
        safe = m.datasets._safe_segment
        self.assertEqual(safe("agent-1.x_y"), "agent-1.x_y")
        for dangerous in ("..", ".", "a/b", "../x", "", "a" * 200, "x\x00"):
            segment = safe(dangerous)
            self.assertRegex(segment, r"^[a-f0-9]{32}$")

    def test_sanitize_filename(self):
        sanitize = m.datasets.sanitize_filename
        self.assertEqual(sanitize("../../etc/Sales Data.CSV"), "Sales_Data.csv")
        self.assertEqual(sanitize("a" * 200 + ".db"), "a" * 80 + ".db")
        self.assertRegex(sanitize("../.."), r"^dataset_[a-f0-9]{8}$")
        self.assertRegex(sanitize(None), r"^dataset_[a-f0-9]{8}$")

    def test_table_name_from(self):
        name = m.datasets.table_name_from
        self.assertEqual(name("Sales 2024.csv"), "sales_2024")
        self.assertEqual(name("2024.csv"), "t_2024")
        self.assertEqual(name("---.csv"), "dataset")
        self.assertEqual(len(name("x" * 100)), 60)

    def test_detect_kind(self):
        detect = m.datasets.detect_kind
        self.assertEqual(detect("x.bin", m.datasets.SQLITE_MAGIC + b"rest"), "sqlite")
        self.assertEqual(detect("x.TSV", b"a\tb"), "csv")
        self.assertIsNone(detect("x.txt", b"a,b"))
        self.assertIsNone(detect(None, b""))

    def test_quote_identifier(self):
        self.assertEqual(m.datasets.quote_identifier('we"ird'), '"we""ird"')


class ReadCsvTest(unittest.TestCase):
    def read(self, text: str, name="x.csv", encoding="utf-8"):
        return m.datasets.read_csv(text.encode(encoding), name)

    def test_comma_separated(self):
        df = self.read("name,price\na,1.5\nb,2\n")
        self.assertEqual(df["price"].tolist(), [1.5, 2.0])

    def test_semicolon_with_dot_decimals(self):
        # regression: "1.5" was read as 15 (the dot was taken as thousands separator)
        df = self.read("name;price\na;1.5\nb;2.25\n")
        self.assertEqual(df["price"].tolist(), [1.5, 2.25])

    def test_semicolon_with_european_numbers(self):
        df = self.read("name;price;qty\na;1.200,5;1.000\nb;2,5;300\n")
        self.assertEqual(df["price"].tolist(), [1200.5, 2.5])
        self.assertEqual(df["qty"].tolist(), [1000, 300])

    def test_semicolon_text_and_missing_values(self):
        df = self.read("name;code;empty;note\na;001;;x\nb;;;y\n")
        self.assertEqual(df["name"].tolist(), ["a", "b"])
        self.assertEqual(df["code"].tolist()[0], 1)
        self.assertTrue(df["empty"].isna().all())
        self.assertEqual(df["note"].tolist(), ["x", "y"])

    def test_tab_and_encodings(self):
        df = m.datasets.read_csv("città\tvalore\nRoma\t3\n".encode("cp1252"), "x.tsv")
        self.assertEqual(df.columns.tolist(), ["città", "valore"])
        df = self.read("﻿a,b\n1,2\n")
        self.assertEqual(df.columns.tolist(), ["a", "b"])

    def test_latin1_fallback(self):
        # 0x81 is invalid both in UTF-8 and in cp1252
        df = m.datasets.read_csv(b"a,b\n\x81,1\n", "x.csv")
        self.assertEqual(df["a"].tolist(), ["\x81"])

    def test_invalid_csv(self):
        with self.assertRaises(m.datasets.DatasetError):
            self.read('a,b\n"unterminated,1\n')

    def test_blank_column_names(self):
        df = self.read(" ,b\n1,2\n")
        self.assertEqual(df.columns.tolist(), ["column_0", "b"])

    def test_sniffer_fallback(self):
        self.assertEqual(m.datasets._sniff_separator("", ".csv"), ",")
        self.assertEqual(m.datasets._sniff_separator("a;b", ".tsv"), "\t")

    def test_dates(self):
        df = self.read("d,e,f,g\n01/02/2024,02/13/2024,x,01/02/2024 10:30\n13/02/2024,03/14/2024,y,13/02/2024 11:00\n")
        self.assertEqual(df["d"].tolist(), ["2024-02-01", "2024-02-13"])
        self.assertEqual(df["e"].tolist(), ["2024-02-13", "2024-03-14"])
        self.assertEqual(df["f"].tolist(), ["x", "y"])
        self.assertEqual(df["g"].tolist(), ["2024-02-01 10:30:00", "2024-02-13 11:00:00"])

    def test_dates_not_normalized(self):
        # empty column, invalid dates, mixed values: left as they are
        df = self.read("a,b,c\n,31/02/2024,01/02/2024\n,30/02/2024,hello\n")
        self.assertEqual(df["b"].tolist(), ["31/02/2024", "30/02/2024"])
        self.assertEqual(df["c"].tolist(), ["01/02/2024", "hello"])


class SqliteHelpersTest(unittest.TestCase):
    def test_tables_with_quoted_names(self):
        # regression: a valid SQLite file with a table named 'we"ird' was rejected as invalid
        with tempfile.TemporaryDirectory() as tmp:
            path = support.sqlite_file(Path(tmp) / "q.db", {'we"ird': (["v"], [(1,)])})
            self.assertEqual(m.datasets.sqlite_tables(path), {'we"ird': ["v"]})
            description = m.datasets.describe_dataset("q.db", path.read_bytes(), "sqlite")
            self.assertIn("Table 'we\"ird': 1 rows", description)

    def test_invalid_sqlite(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.db"
            path.write_bytes(m.datasets.SQLITE_MAGIC + b"\x00" * 100)
            with self.assertRaises(m.datasets.DatasetError):
                m.datasets.sqlite_tables(path)

    def test_describe_csv(self):
        text = m.datasets.describe_dataset("Sales.csv", b"a,b\n1,x\n", "csv")
        self.assertIn("Table 'sales': 1 rows, 2 columns.", text)
        self.assertIn("First rows:", text)


class StoreTest(unittest.TestCase):
    def setUp(self):
        self.agent = f"agent-{time.monotonic_ns()}"

    def store(self, chat="chat"):
        return m.datasets.DatasetStore(self.agent, chat)

    def test_add_validations(self):
        store = self.store()
        with self.assertRaises(m.datasets.DatasetError):
            store.add("a.csv", b"")
        with self.assertRaises(m.datasets.DatasetError):
            store.add("a.csv", b"a,b\n1,2\n", max_bytes=3)
        with self.assertRaises(m.datasets.DatasetError):
            store.add("a.txt", b"a,b\n1,2\n")
        with tempfile.TemporaryDirectory() as tmp:
            path = support.sqlite_file(Path(tmp) / "e.sqlite", {"t": (["v"], [])})
            conn = sqlite3.connect(path)
            conn.execute("DROP TABLE t")
            conn.commit()
            conn.close()
            with self.assertRaises(m.datasets.DatasetError) as error:
                store.add("a.sqlite", path.read_bytes())
            self.assertIn("does not contain any table", str(error.exception))
        with self.assertRaises(m.datasets.DatasetError):
            store.add("a.csv", b'a,b\n"x,1\n')
        self.assertEqual(store.list_datasets(), [])
        # no temporary file is left behind
        self.assertEqual([p.name for p in (store.chat_dir / "files").iterdir()], [])
        # files not written by the store are ignored
        (store.chat_dir / "files" / "foreign.csv").write_bytes(b"a\n1\n")
        self.assertEqual(store.list_datasets(), [])

    def test_add_renames_extensions(self):
        store = self.store()
        info = store.add("dump.bin", support.sqlite_bytes({"t": (["v"], [(1,)])}))
        self.assertEqual((info.name, info.kind, info.scope, info.tables), ("dump.sqlite", "sqlite", "chat", {"t": ["v"]}))
        info = store.add("data.csv", b"a\n1\n")
        self.assertEqual(info.name, "data.csv")

    def test_scopes_and_listing(self):
        shared = m.datasets.DatasetStore(self.agent)
        info = shared.add("shared.csv", b"a\n1\n")
        self.assertEqual(info.scope, "shared")
        self.store("c1").add("mine.csv", b"a\n2\n")
        self.store("c1").add("shared.csv", b"b\n3\n")  # the chat dataset wins on name clashes
        names = {(d.name, d.scope) for d in self.store("c1").list_datasets(with_tables=True)}
        self.assertEqual(names, {("mine.csv", "chat"), ("shared.csv", "chat")})
        self.assertEqual([d.scope for d in self.store("c2").list_datasets()], ["shared"])
        listed = self.store("c1").list_datasets(with_tables=True)
        self.assertEqual({d.name: d.tables for d in listed}["shared.csv"], {"shared": ["b"]})

    def test_listing_skips_unreadable_tables(self):
        store = self.store()
        store.add("x.csv", b"a\n1\n")
        with mock.patch.object(m.datasets, "read_csv", side_effect=m.datasets.DatasetError("broken")):
            self.assertEqual(store.list_datasets(with_tables=True)[0].tables, {})
        store.add("y.sqlite", support.sqlite_bytes({"t": (["v"], [(1,)])}))
        tables = {d.name: d.tables for d in store.list_datasets(with_tables=True)}
        self.assertEqual(tables["y.sqlite"], {"t": ["v"]})

    def test_remove(self):
        store = self.store()
        store.add("x.csv", b"a\n1\n")
        self.assertFalse(store.remove("y.csv"))
        self.assertTrue(store.remove("x.csv"))
        self.assertEqual(store.list_datasets(), [])

    def test_workspace_single_sqlite_is_used_as_is(self):
        store = self.store()
        self.assertIsNone(store.workspace_path())
        store.add("only.sqlite", support.sqlite_bytes({"t": (["v"], [(1,)])}))
        stored, = (store.chat_dir / "files").iterdir()
        self.assertEqual(store.workspace_path(), stored)
        self.assertTrue(stored.name.endswith("--only.sqlite"))

    def test_workspace_merges_datasets(self):
        store = self.store()
        store.add("sales.csv", b"region,amount\nN,1\nS,2\n")
        store.add("db.sqlite", support.sqlite_bytes({"sales": (["x"], [(1,)]), 'q"t': (["v"], [(5,)])}))
        store.add("other.sqlite", support.sqlite_bytes({"extra": (["x"], [(1,)])}))
        path = store.workspace_path()
        tables = {name for (name,) in query(path, "SELECT name FROM sqlite_master WHERE type = 'table'")}
        # files are processed by name: db.sqlite first, so the CSV table gets a suffix
        self.assertEqual(tables, {"sales", "sales_2", 'q"t', "extra"})
        self.assertEqual(query(path, "SELECT SUM(amount) FROM sales_2"), [(3,)])
        self.assertEqual(query(path, 'SELECT v FROM "q""t"'), [(5,)])
        # unchanged datasets: the same workspace is reused
        self.assertEqual(store.workspace_path(), path)

    def test_workspace_follows_reuploads(self):
        # regression: a dataset uploaded again with the same name was served from the previous copy
        store = self.store()
        store.add("a.csv", b"v\n1\n")
        store.add("b.csv", b"v\n1\n")
        first = store.workspace_path()
        before = {p.name: p.stat() for p in (store.chat_dir / "files").iterdir()}
        store.add("a.csv", b"v\n2\n")  # same name and size...
        # ...and same timestamp (coarse clocks, or a fast client)
        stat = next(v for k, v in before.items() if k.endswith("--a.csv"))
        stored = next((store.chat_dir / "files").glob("*--a.csv"))
        os.utime(stored, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        self.assertEqual(len(list((store.chat_dir / "files").glob("*--a.csv"))), 1, "the old copy is removed")
        second = store.workspace_path()
        self.assertNotEqual(first, second)
        self.assertFalse(first.exists())
        self.assertEqual(query(second, "SELECT v FROM a"), [(2,)])

    def test_concurrent_uploads_of_the_same_dataset(self):
        # regression: two uploads with the same name, running together, removed each other's copy
        store = self.store()
        folder = store.chat_dir / "files"
        real_replace = m.datasets.Path.replace

        def replace_then_other_upload(path, target):
            result = real_replace(path, target)
            if target.name.endswith("--a.csv") and not getattr(replace_then_other_upload, "done", False):
                replace_then_other_upload.done = True
                # the other upload stores its (more recent) copy before this one removes the previous copies
                (folder / f"{m.datasets.time.time_ns() + 10**9:020d}-00000000--a.csv").write_bytes(b"v\n2\n")
            return result

        with mock.patch.object(m.datasets.Path, "replace", replace_then_other_upload):
            store.add("a.csv", b"v\n1\n")
        copies = sorted(folder.glob("*--a.csv"))
        self.assertTrue(copies, "no copy of the dataset is left")
        self.assertEqual(copies[-1].read_bytes(), b"v\n2\n")
        self.assertEqual([d.name for d in store.list_datasets()], ["a.csv"])

    def test_workspace_build_failure_leaves_nothing(self):
        store = self.store()
        store.add("a.csv", b"v\n1\n")
        store.add("b.csv", b"v\n1\n")
        with mock.patch.object(m.datasets, "read_csv", side_effect=m.datasets.DatasetError("boom")):
            with self.assertRaises(m.datasets.DatasetError):
                store.workspace_path()
        self.assertEqual(list(store.chat_dir.glob("*.sqlite")), [])

    def test_cleanup_expired(self):
        agent_store = m.datasets.DatasetStore(self.agent)
        agent_store.cleanup_expired(1)  # no agent folder yet
        old, current, recent = self.store("old"), self.store("current"), self.store("recent")
        old.add("x.csv", b"a\n1\n")
        current.add("y.csv", b"a\n1\n")
        recent.add("z.csv", b"a\n1\n")
        m.datasets.DatasetStore(self.agent).add("s.csv", b"a\n1\n")
        charts = agent_store.agent_dir / m.datasets.CHARTS_SCOPE
        charts.mkdir(parents=True)
        (charts / "old.png").write_bytes(b"x")
        (charts / "new.png").write_bytes(b"x")
        (agent_store.agent_dir / "stray-file").write_bytes(b"x")
        past = time.time() - 10 * 3600
        for path in [*old.chat_dir.rglob("*"), *current.chat_dir.rglob("*"), charts / "old.png"]:
            os.utime(path, (past, past))

        agent_store.cleanup_expired(0)  # disabled
        self.assertTrue(old.chat_dir.exists())

        current.cleanup_expired(1)
        self.assertFalse(old.chat_dir.exists())
        self.assertTrue(current.chat_dir.exists(), "the current conversation is never removed")
        self.assertTrue(recent.chat_dir.exists())
        self.assertTrue(agent_store.shared_dir.exists(), "shared datasets are never removed")
        self.assertEqual([p.name for p in charts.iterdir()], ["new.png"])

    def test_datasets_in_use_are_not_expired(self):
        # regression: the datasets of a conversation still in use were removed when uploaded more than the TTL ago
        active, other = self.store("active"), self.store("other")
        active.add("x.csv", b"a\n1\n")
        past = time.time() - 10 * 3600
        for path in active.chat_dir.rglob("*"):
            os.utime(path, (past, past))
        active.mark_used()
        other.cleanup_expired(1)
        self.assertEqual([d.name for d in active.list_datasets()], ["x.csv"])
        m.datasets.DatasetStore(self.agent).mark_used()  # no conversation: nothing to mark
        with mock.patch.object(m.datasets.Path, "touch", side_effect=OSError("read-only file system")):
            active.mark_used()  # logged, not raised

    def test_cleanup_errors_are_logged(self):
        store = self.store("old")
        store.add("x.csv", b"a\n1\n")
        past = time.time() - 10 * 3600
        for path in store.chat_dir.rglob("*"):
            os.utime(path, (past, past))
        with mock.patch.object(m.datasets.Path, "rglob", side_effect=OSError("gone")):
            m.datasets.DatasetStore(self.agent).cleanup_expired(1)
        self.assertTrue(store.chat_dir.exists())


if __name__ == "__main__":
    unittest.main()
