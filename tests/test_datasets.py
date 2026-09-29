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


def query(workspace: bytes, sql: str):
    conn = sqlite3.connect(":memory:")
    try:
        conn.deserialize(workspace)
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
        content = support.sqlite_bytes({'we"ird': (["v"], [(1,)])})
        self.assertEqual(m.datasets.sqlite_tables(content), {'we"ird': ["v"]})
        description = m.datasets.describe_dataset("q.db", content, "sqlite")
        self.assertIn("Table 'we\"ird': 1 rows", description)

    def test_invalid_sqlite(self):
        with self.assertRaises(m.datasets.DatasetError):
            m.datasets.sqlite_tables(m.datasets.SQLITE_MAGIC + b"\x00" * 100)
        with self.assertRaises(m.datasets.DatasetError):
            m.datasets.build_workspace([("a.sqlite", m.datasets.SQLITE_MAGIC + b"\x00" * 100)])

    def test_describe_csv(self):
        text = m.datasets.describe_dataset("Sales.csv", b"a,b\n1,x\n", "csv")
        self.assertIn("Table 'sales': 1 rows, 2 columns.", text)
        self.assertIn("First rows:", text)

    def test_workspace_cache_is_bounded(self):
        with mock.patch.object(m.datasets, "WORKSPACE_CACHE_BYTES", 10), \
                mock.patch.object(m.datasets, "_workspaces", m.datasets.OrderedDict()):
            m.datasets._cache_workspace(("a",), b"x" * 8)
            m.datasets._cache_workspace(("b",), b"y" * 8)
            self.assertIsNone(m.datasets._cached_workspace(("a",)))
            self.assertEqual(m.datasets._cached_workspace(("b",)), b"y" * 8)
            m.datasets._cache_workspace(("c",), b"z" * 20)  # larger than the budget: kept alone
            self.assertEqual(list(m.datasets._workspaces), [("c",)])


class StoreTest(unittest.TestCase):
    """The store on a file manager with the semantics of an object storage (S3)."""

    def file_manager(self):
        return m.fakes.ObjectStoreFileManager()

    def setUp(self):
        self.agent = f"agent-{time.monotonic_ns()}"
        self.fm = self.file_manager()

    def store(self, chat="chat"):
        return m.datasets.DatasetStore(self.fm, self.agent, chat)

    def shared(self):
        return m.datasets.DatasetStore(self.fm, self.agent)

    def stored_names(self, folder):
        return sorted(f.name for f in self.fm.list_files(folder))

    def test_add_validations(self):
        store = self.store()
        with self.assertRaises(m.datasets.DatasetError):
            store.add("a.csv", b"")
        with self.assertRaises(m.datasets.DatasetError):
            store.add("a.csv", b"a,b\n1,2\n", max_bytes=3)
        with self.assertRaises(m.datasets.DatasetError):
            store.add("a.txt", b"a,b\n1,2\n")
        with self.assertRaises(m.datasets.DatasetError):
            store.add("a.csv", b'a,b\n"x,1\n')
        empty = sqlite3.connect(":memory:")
        empty.execute("CREATE TABLE t (v)")
        empty.execute("DROP TABLE t")
        with self.assertRaises(m.datasets.DatasetError) as error:
            store.add("a.sqlite", empty.serialize())
        self.assertIn("does not contain any table", str(error.exception))
        self.assertEqual(store.list_datasets(), [])
        self.assertEqual(self.stored_names(store.chat_dir), [], "nothing is written for invalid files")

    def test_layout(self):
        store = self.store()
        store.add("x.csv", b"a\n1\n")
        stored, = self.stored_names(store.chat_dir)
        self.assertRegex(stored, r"^\d{20}-[0-9a-f]{16}--x\.csv$")
        # the folder of the conversation is the one of the core, removed with the conversation
        self.assertEqual(store.chat_dir, f"{self.agent}/chat/cat_with_your_data")
        self.assertEqual(self.stored_names(store.index_dir), ["chat"])
        self.assertEqual(m.datasets.DatasetStore(self.fm, self.agent, "a/../b").chat_dir.count("/"), 2)

    def test_file_manager_errors(self):
        store = self.store()
        with mock.patch.object(self.fm, "write_file", return_value=False):
            with self.assertRaises(m.datasets.DatasetError):
                store.add("x.csv", b"a\n1\n")
        # the default file manager of the core (Dummy) accepts the writes and keeps nothing
        dummy = m.datasets.DatasetStore(m.fakes.DummyFileManager(), self.agent, "chat")
        with self.assertRaises(m.datasets.DatasetError) as error:
            dummy.add("x.csv", b"a\n1\n")
        self.assertIn("configure a file manager", str(error.exception))

    def test_add_renames_extensions(self):
        store = self.store()
        info = store.add("dump.bin", support.sqlite_bytes({"t": (["v"], [(1,)])}))
        self.assertEqual((info.name, info.kind, info.scope, info.tables), ("dump.sqlite", "sqlite", "chat", {"t": ["v"]}))
        info = store.add("data.csv", b"a\n1\n")
        self.assertEqual((info.name, info.size), ("data.csv", 4))
        self.assertAlmostEqual(info.uploaded_at, time.time(), delta=60)

    def test_scopes_and_listing(self):
        info = self.shared().add("shared.csv", b"a\n1\n")
        self.assertEqual(info.scope, "shared")
        self.store("c1").add("mine.csv", b"a\n2\n")
        self.store("c1").add("shared.csv", b"b\n3\n")  # the chat dataset wins on name clashes
        names = {(d.name, d.scope) for d in self.store("c1").list_datasets(with_tables=True)}
        self.assertEqual(names, {("mine.csv", "chat"), ("shared.csv", "chat")})
        self.assertEqual([d.scope for d in self.store("c2").list_datasets()], ["shared"])
        listed = self.store("c1").list_datasets(with_tables=True)
        self.assertEqual({d.name: d.tables for d in listed}["shared.csv"], {"shared": ["b"]})
        self.store("c1").add("y.sqlite", support.sqlite_bytes({"t": (["v"], [(1,)])}))
        tables = {d.name: d.tables for d in self.store("c1").list_datasets(with_tables=True)}
        self.assertEqual(tables["y.sqlite"], {"t": ["v"]})

    def test_files_not_written_by_the_store_are_ignored(self):
        store = self.store()
        self.fm.write_file(b"a\n1\n", "foreign.csv", store.chat_dir)
        self.assertEqual(store.list_datasets(), [])
        self.assertIsNone(store.workspace())

    def test_remove(self):
        store = self.store()
        store.add("x.csv", b"a\n1\n")
        self.assertFalse(store.remove("y.csv"))
        self.assertTrue(store.remove("x.csv"))
        self.assertEqual(store.list_datasets(), [])

    def test_workspace_single_sqlite_is_used_as_is(self):
        store = self.store()
        self.assertIsNone(store.workspace())
        content = support.sqlite_bytes({"t": (["v"], [(1,)])})
        store.add("only.sqlite", content)
        self.assertEqual(store.workspace(), content)

    def test_workspace_merges_datasets(self):
        store = self.store()
        store.add("sales.csv", b"region,amount\nN,1\nS,2\n")
        store.add("db.sqlite", support.sqlite_bytes({"sales": (["x"], [(1,)]), 'q"t': (["v"], [(5,)])}))
        store.add("other.sqlite", support.sqlite_bytes({"extra": (["x"], [(1,)])}))
        workspace = store.workspace()
        tables = {name for (name,) in query(workspace, "SELECT name FROM sqlite_master WHERE type = 'table'")}
        # datasets are processed by name: db.sqlite first, so the CSV table gets a suffix
        self.assertEqual(tables, {"sales", "sales_2", 'q"t', "extra"})
        self.assertEqual(query(workspace, "SELECT SUM(amount) FROM sales_2"), [(3,)])
        self.assertEqual(query(workspace, 'SELECT v FROM "q""t"'), [(5,)])
        # unchanged datasets: the workspace comes from the cache
        with mock.patch.object(m.datasets, "build_workspace", side_effect=AssertionError("rebuilt")):
            self.assertIs(store.workspace(), workspace)

    def test_workspace_follows_reuploads(self):
        # regression: a dataset uploaded again with the same name was served from the previous copy
        store = self.store()
        store.add("a.csv", b"v\n1\n")
        store.add("b.csv", b"v\n1\n")
        first = store.workspace()
        store.add("a.csv", b"v\n2\n")  # same name and size
        self.assertEqual(query(store.workspace(), "SELECT v FROM a"), [(2,)])
        self.assertEqual(query(first, "SELECT v FROM a"), [(1,)])
        self.assertEqual(len([n for n in self.stored_names(store.chat_dir) if n.endswith("--a.csv")]), 1)

    def test_concurrent_uploads_of_the_same_dataset(self):
        # regression: two uploads with the same name, running together, removed each other's copy
        store, other = self.store(), self.store()
        real_write, raced = self.fm._write_file, []

        def write(content, path):
            real_write(content, path)
            if path.endswith("--a.csv") and not raced:
                raced.append(True)
                # the other upload stores its (more recent) copy right after this one
                with mock.patch.object(m.datasets, "time", m.fakes.FakeClock(60)):
                    other.add("a.csv", b"v\n2\n")

        with mock.patch.object(self.fm, "_write_file", write):
            store.add("a.csv", b"v\n1\n")
        self.assertEqual([d.name for d in store.list_datasets()], ["a.csv"])
        self.assertEqual(query(store.workspace(), "SELECT v FROM a"), [(2,)])

    def test_upload_order_does_not_depend_on_the_clock(self):
        # regression: an instance with the clock ahead made the later uploads of the other instances lose
        store = self.store()
        with mock.patch.object(m.datasets, "time", m.fakes.FakeClock(3600)):
            store.add("x.csv", b"v\n1\n")
        store.add("x.csv", b"v\n2\n")
        self.assertEqual(query(store.workspace(), "SELECT v FROM x"), [(2,)])

    def test_cleanup_expired(self):
        self.shared().cleanup_expired(1)  # no conversations yet
        old, current, recent = self.store("old"), self.store("current"), self.store("recent")
        with mock.patch.object(m.datasets, "time", m.fakes.FakeClock(-10 * 3600)):
            old.add("x.csv", b"a\n1\n")
            current.add("y.csv", b"a\n1\n")
        recent.add("z.csv", b"a\n1\n")
        self.shared().add("s.csv", b"a\n1\n")

        self.shared().cleanup_expired(0)  # disabled
        self.assertEqual(len(old.list_datasets()), 2)

        current.cleanup_expired(1)
        self.assertEqual([d.name for d in old.list_datasets()], ["s.csv"])
        self.assertNotIn("old", self.stored_names(old.index_dir))
        self.assertEqual({d.name for d in current.list_datasets()}, {"s.csv", "y.csv"}, "the current one is kept")
        self.assertEqual({d.name for d in recent.list_datasets()}, {"s.csv", "z.csv"})

    def test_recent_uploads_of_an_idle_conversation_are_kept(self):
        # e.g. the instance storing the upload died before recording the activity of the conversation
        store = self.store("idle")
        with mock.patch.object(m.datasets, "time", m.fakes.FakeClock(-10 * 3600)):
            store.add("old.csv", b"a\n1\n")
            store.add("new.csv", b"a\n2\n")
        new, = [n for n in self.stored_names(store.chat_dir) if n.endswith("--new.csv")]
        self.fm.remove_file(f"{store.chat_dir}/{new}")
        self.fm.write_file(b"a\n2\n", store._new_stored_name(store.chat_dir, "new.csv", b"a\n2\n"), store.chat_dir)
        self.store("other").cleanup_expired(1)
        self.assertEqual([d.name for d in store.list_datasets()], ["new.csv"])

    def test_datasets_in_use_are_not_expired(self):
        # regression: the datasets of a conversation still in use were removed when uploaded more than the TTL ago
        active = self.store("active")
        with mock.patch.object(m.datasets, "time", m.fakes.FakeClock(-10 * 3600)):
            active.add("x.csv", b"a\n1\n")
        active.mark_used()
        self.store("other").cleanup_expired(1)
        self.assertEqual([d.name for d in active.list_datasets()], ["x.csv"])
        self.shared().mark_used()  # no conversation: nothing to mark

    def test_cleanup_errors(self):
        store = self.store("old")
        with mock.patch.object(m.datasets, "time", m.fakes.FakeClock(-10 * 3600)):
            store.add("x.csv", b"a\n1\n")
        with mock.patch.object(self.fm, "remove_file", side_effect=OSError("denied")):
            self.store("other").cleanup_expired(1)  # logged
        self.assertEqual([d.name for d in store.list_datasets()], ["x.csv"])
        # markers removed meanwhile, or being written (on another instance)
        self.fm.write_file("not a number", "old", store.index_dir)
        self.store("other").cleanup_expired(1)
        self.assertEqual([d.name for d in store.list_datasets()], ["x.csv"])
        with mock.patch.object(self.fm, "download_file", return_value=None):
            self.store("other").cleanup_expired(1)
        self.assertEqual([d.name for d in store.list_datasets()], ["x.csv"])

    def test_incomplete_copies(self):
        # regression: an incomplete copy was waited for, by its timestamp; the timestamp may be ahead of the clock
        store = self.store()
        with mock.patch.object(m.datasets, "time", m.fakes.FakeClock(3600)):
            store.add("x.csv", b"v\n1\n")
        store.add("z.csv", b"v\n9\n")
        folder = store.chat_dir
        # a more recent copy being written (or interrupted): not visible, the previous copy is used
        stored = store._new_stored_name(folder, "x.csv", b"v\n2\n")
        self.fm.write_file(b"v\n", stored, folder)
        self.assertEqual(query(store.workspace(), "SELECT v FROM x"), [(1,)])
        self.assertEqual([d.size for d in store.list_datasets() if d.name == "x.csv"], [4])
        # the write completes: the new content is visible (the previous workspace was not cached)
        self.fm.write_file(b"v\n2\n", stored, folder)
        self.assertEqual(query(store.workspace(), "SELECT v FROM x"), [(2,)])
        # no complete copy at all
        store.remove("x.csv")
        store.remove("z.csv")
        self.fm.write_file(b"v\n", store._new_stored_name(folder, "y.csv", b"v\n3\n"), folder)
        self.assertIsNone(store.workspace())
        self.assertEqual(store.list_datasets(), [])

    def test_interrupted_uploads_expire_with_the_conversation(self):
        # the activity is recorded before the write: a copy left by an instance that died is removed by the cleanup
        store = self.store("idle")
        real_write = self.fm.write_file

        def die_while_writing(content, name, folder):
            if "--" in name:
                real_write(content[:-1], name, folder)
                raise SystemExit("the instance died")
            return real_write(content, name, folder)

        with mock.patch.object(m.datasets, "time", m.fakes.FakeClock(-10 * 3600)), \
                mock.patch.object(self.fm, "write_file", die_while_writing), self.assertRaises(SystemExit):
            store.add("x.csv", b"v\n1\n")
        self.assertEqual(len(self.stored_names(store.chat_dir)), 1)
        self.store("other").cleanup_expired(1)
        self.assertEqual(self.stored_names(store.chat_dir), [])


class LocalStoreTest(StoreTest):
    """The same, on the local file manager of the core (on a folder shared by the instances)."""

    def file_manager(self):
        return m.fakes.local_file_manager()


if __name__ == "__main__":
    unittest.main()
