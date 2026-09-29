"""The plugin as the core sees it: security scan, hooks, priorities, endpoints and settings overrides."""
import unittest
import unittest.mock

m = support = None


def setUpModule():
    global m, support
    import support as support_module
    support = support_module
    m = support.load()


class PluginLoadingTest(unittest.TestCase):
    def test_every_file_passes_the_security_scan(self):
        # the core scans (and imports) every .py file of the plugin, tests included
        from cat.looking_glass.mad_hatter.plugin_extractor import PluginExtractor
        self.assertTrue(PluginExtractor._is_safe_plugin(str(support.PLUGIN_DIR)))

    def test_hooks_endpoints_and_overrides(self):
        plugin = m.plugin  # loaded by the core loader in support.load()
        self.assertEqual(
            sorted((h.name, h.priority) for h in plugin.hooks),
            [("after_cheshire_cat_destroy", 1), ("agent_fast_reply", 0), ("before_cat_sends_message", 0),
             ("before_rabbithole_splits_documents", 10), ("rabbithole_instantiates_parsers", 1)],
        )
        self.assertEqual(
            sorted((e.name, tuple(sorted(e.methods))) for e in plugin.endpoints),
            [("/custom/cat-with-your-data/datasets", ("GET",)),
             ("/custom/cat-with-your-data/datasets", ("POST",)),
             ("/custom/cat-with-your-data/datasets/{name}", ("DELETE",))],
        )
        self.assertEqual(sorted(plugin.overrides), ["load_settings", "save_settings", "settings_schema"])
        self.assertIs(plugin.settings_model(), m.settings.MySettings)


class ReloadOrderTest(unittest.TestCase):
    def test_modules_reloaded_before_datasets(self):
        # regression: the core loader reloads the modules in the order of the file system; a module reloaded before
        # datasets.py kept the previous DatasetError, so its `except DatasetError` missed the errors
        from langchain_core.documents import Document
        from langchain_core.documents.base import Blob
        support.load_plugin(order=lambda path: path.endswith("/datasets.py"))

        fallback = unittest.mock.Mock()
        fallback.lazy_parse.side_effect = lambda blob: iter([Document(page_content="as text")])
        parser = m.parsers.DatasetBlobParser(fallback=fallback)
        docs = list(parser.lazy_parse(Blob(data=b'a,b\n"x,1\n', path="bad.csv")))
        self.assertEqual([d.page_content for d in docs], ["as text"])


if __name__ == "__main__":
    unittest.main()
