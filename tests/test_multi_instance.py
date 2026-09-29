"""Several instances (pods) of the Cat sharing the file manager of the agent (a shared folder, S3...).

The file managers offer neither renames nor locks; the operations of "another instance" are run while an operation is
in progress, by wrapping the methods of the file manager.
"""
import sqlite3
import time
import unittest
from unittest import mock

m = support = None


def setUpModule():
    global m, support
    import support as support_module
    support = support_module
    m = support.load()


def rows(engine, sql):
    with engine.connect() as conn:
        return conn.exec_driver_sql(sql).fetchall()


class MultiInstanceTest(unittest.TestCase):
    """On a file manager with the semantics of an object storage (S3)."""

    def file_manager(self):
        return m.fakes.ObjectStoreFileManager()

    def setUp(self):
        self.agent = f"agent-{time.monotonic_ns()}"
        self.fm = self.file_manager()

    def store(self, chat="chat"):
        return m.datasets.DatasetStore(self.fm, self.agent, chat)

    def agent_engine(self, chat="chat"):
        cat = m.fakes.make_cat(agent_key=self.agent, chat_id=chat, file_manager=self.fm)
        with mock.patch.object(m.query_agent, "time", m.fakes.FakeClock()):
            return m.query_agent.QueryCatAgent(cat)._uploaded_datasets_engine()

    def on_first(self, operation, suffix, action):
        """Run ``action`` (the other instance) the first time ``operation`` is done on a path ending with ``suffix``."""
        real, done = getattr(self.fm, operation), []

        def wrapper(path, *args, **kwargs):
            if not done and str(path).endswith(suffix):
                done.append(True)
                action()
            return real(path, *args, **kwargs)

        return mock.patch.object(self.fm, operation, wrapper)

    # ----------------------------------------------------------------------------------------------------------------
    def test_requests_in_progress_keep_their_data(self):
        store = self.store("idle")
        with mock.patch.object(m.datasets, "time", m.fakes.FakeClock(-10 * 3600)):
            store.add("x.sqlite", support.sqlite_bytes({"t": (["v"], [(1,)])}))
            store.add("y.csv", b"v\n1\n")
        engine = self.agent_engine("idle")
        # on other instances: a dataset is uploaded again, another one is removed, the conversation expires
        store.add("x.sqlite", support.sqlite_bytes({"t": (["v"], [(2,)])}))
        store.remove("y.csv")
        with mock.patch.object(m.datasets, "time", m.fakes.FakeClock(10 * 3600)):
            self.store("other").cleanup_expired(1)
        self.assertEqual(self.store("idle").list_datasets(), [])
        self.assertEqual(rows(engine, "SELECT v FROM t"), [(1,)])
        self.assertEqual(rows(engine, "SELECT v FROM y"), [(1,)])
        engine.dispose()

    def test_dataset_replaced_while_downloading(self):
        # the listing is stale: the copy was replaced (and removed) by another instance before the download
        store = self.store()
        store.add("a.csv", b"v\n1\n")
        store.add("b.csv", b"v\n1\n")
        with self.on_first("download_file", "--a.csv", lambda: self.store().add("a.csv", b"v\n2\n")):
            engine = self.agent_engine()
        self.assertEqual(rows(engine, "SELECT v FROM a"), [(2,)])
        engine.dispose()

    def test_dataset_removed_while_listing(self):
        store = self.store()
        store.add("a.csv", b"v\n1\n")
        store.add("b.csv", b"v\n1\n")
        with self.on_first("download_file", "--a.csv", lambda: self.store().remove("a.csv")):
            self.assertEqual([d.name for d in store.list_datasets()], ["b.csv"])

    def test_dataset_being_written(self):
        # a file manager may write in place: another instance may read a copy not complete yet
        store = self.store()
        store.add("a.csv", b"v\n1\n")
        store.add("b.csv", b"v\n1\n")
        real, calls = self.fm.download_file, []

        def download(path):
            calls.append(path)
            content = real(path)
            return content[:-1] if len(calls) == 1 else content  # the first read sees a partial file

        with mock.patch.object(self.fm, "download_file", download):
            engine = self.agent_engine()
            # the copy is not complete yet: not visible to this request, and not cached
            self.assertEqual(rows(engine, "SELECT name FROM sqlite_master WHERE type = 'table'"), [("b",)])
            engine.dispose()
            engine = self.agent_engine()
        self.assertEqual(rows(engine, "SELECT v FROM a"), [(1,)])
        engine.dispose()

    def test_cleanup_racing_an_upload(self):
        # an instance removes an expired conversation while another instance stores a new dataset in it
        idle = self.store("idle")
        with mock.patch.object(m.datasets, "time", m.fakes.FakeClock(-10 * 3600)):
            idle.add("old.csv", b"v\n1\n")
        with self.on_first("remove_file", "--old.csv", lambda: self.store("idle").add("new.csv", b"v\n2\n")):
            self.store("other").cleanup_expired(1)
        self.assertEqual([d.name for d in idle.list_datasets()], ["new.csv"])
        self.assertIn("idle", [f.name for f in self.fm.list_files(idle.index_dir)], "the conversation is active")

    def test_cleanup_racing_a_question(self):
        # a question arrives (on another instance) while the conversation is expiring: it stays active from now on
        idle = self.store("idle")
        with mock.patch.object(m.datasets, "time", m.fakes.FakeClock(-10 * 3600)):
            idle.add("old.csv", b"v\n1\n")
        with self.on_first("remove_file", "--old.csv", self.store("idle").mark_used):
            self.store("other").cleanup_expired(1)
        self.assertIn("idle", [f.name for f in self.fm.list_files(idle.index_dir)])

    def test_activity_being_recorded(self):
        # regression: a marker read while being written (in place) looked like a very old activity
        active = self.store("active")
        with mock.patch.object(m.datasets, "time", m.fakes.FakeClock(-10 * 3600)):
            active.add("x.csv", b"v\n1\n")
        active.mark_used()
        real = self.fm.download_file

        def partial(path):
            content = real(path)
            return content[:6] if path.endswith("/_chats/active") else content

        with mock.patch.object(self.fm, "download_file", partial):
            self.store("other").cleanup_expired(1)
        self.assertEqual([d.name for d in active.list_datasets()], ["x.csv"])

    def test_two_instances_expiring_the_same_conversation(self):
        idle = self.store("idle")
        with mock.patch.object(m.datasets, "time", m.fakes.FakeClock(-10 * 3600)):
            idle.add("old.csv", b"v\n1\n")
        with self.on_first("remove_file", "--old.csv", lambda: self.store("third").cleanup_expired(1)):
            self.store("other").cleanup_expired(1)
        self.assertEqual(idle.list_datasets(), [])
        self.assertEqual([f.name for f in self.fm.list_files(idle.index_dir)], [])

    def test_workspaces_are_cached_by_instance(self):
        store = self.store()
        store.add("a.csv", b"v\n1\n")
        store.add("b.csv", b"v\n2\n")
        for _ in range(2):  # two instances, each with its own cache
            with mock.patch.object(m.datasets, "_workspaces", m.datasets.OrderedDict()):
                engine = self.agent_engine()
                self.assertEqual(rows(engine, "SELECT v FROM b"), [(2,)])
                engine.dispose()

    def test_nothing_is_written_on_the_local_disk(self):
        # the instances do not share their local disks: with a remote file manager, the plugin writes nothing locally
        content = support.sqlite_bytes({"t": (["v"], [(1,)])})  # the test helper uses local files
        real_connect = sqlite3.connect

        def connect(database, *args, **kwargs):
            if database != ":memory:":
                raise AssertionError(f"SQLite file opened: {database}")
            return real_connect(database, *args, **kwargs)

        forbidden = AssertionError("local file written")
        with mock.patch("tempfile.mkdtemp", side_effect=forbidden), \
                mock.patch("tempfile.mkstemp", side_effect=forbidden), \
                mock.patch("tempfile.TemporaryDirectory", side_effect=forbidden), \
                mock.patch.object(m.datasets.sqlite3, "connect", connect), \
                mock.patch.object(m.data_engine.sqlite3, "connect", connect), \
                mock.patch("pathlib.Path.write_bytes", side_effect=forbidden), \
                mock.patch("pathlib.Path.write_text", side_effect=forbidden), \
                mock.patch("pathlib.Path.mkdir", side_effect=forbidden):
            m.datasets.describe_dataset("x.sqlite", content, "sqlite")
            store = self.store()
            store.add("x.sqlite", content)
            store.add("y.csv", b"v\n1\n")
            store.list_datasets(with_tables=True)
            engine = self.agent_engine()
            self.assertEqual(rows(engine, "SELECT v FROM y"), [(1,)])
            engine.dispose()
            store.cleanup_expired(1)


class LocalMultiInstanceTest(MultiInstanceTest):
    """The same, on the local file manager of the core (on a folder shared by the instances)."""

    def file_manager(self):
        return m.fakes.local_file_manager()

    def test_nothing_is_written_on_the_local_disk(self):
        self.skipTest("the local file manager of the core writes on its (shared) folder by design")


if __name__ == "__main__":
    unittest.main()
