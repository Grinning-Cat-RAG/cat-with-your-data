"""Chart rendering, storage and the draw_chart tool."""
import base64
import json
import tempfile
import unittest
from pathlib import Path

m = support = None
PNG = b"\x89PNG\r\n\x1a\n"


def setUpModule():
    global m, support
    import support as support_module
    support = support_module
    m = support.load()


def frame(**columns):
    return m.datasets.pd.DataFrame(columns)


class ChartSpecTest(unittest.TestCase):
    def test_aliases_and_defaults(self):
        spec = m.charts.ChartSpec
        self.assertEqual(spec(chart_type="Histogram").chart_type, "hist")
        self.assertEqual(spec(chart_type="unknown").chart_type, "bar")
        self.assertEqual(spec(chart_type=None).chart_type, "bar")
        self.assertEqual(spec(y="v").y, ["v"])
        self.assertEqual(spec(y=None).y, [])
        self.assertEqual(spec(y=["a", "", None, "b"]).y, ["a", "b"])


class RenderTest(unittest.TestCase):
    def render(self, df, **spec):
        png = m.charts.render_chart(df, m.charts.ChartSpec(**spec))
        self.assertTrue(png.startswith(PNG))
        return png

    def test_every_chart_type(self):
        df = frame(region=["N", "S", "E"], sales=[10, 20, 30], cost=[5, 5, 5])
        for kind in ("bar", "barh", "line", "area", "pie", "scatter", "hist"):
            with self.subTest(kind=kind):
                self.render(df, chart_type=kind, x="region", y=["sales"], title=f"{kind} chart")

    def test_multiple_series_and_stacking(self):
        df = frame(region=["N", "S"], sales=[10, 20], cost=[5, 7])
        for kind in ("bar", "barh", "area", "line", "scatter"):
            for stacked in (False, True):
                with self.subTest(kind=kind, stacked=stacked):
                    self.render(df, chart_type=kind, x="region", y=["sales", "cost"], stacked=stacked)

    def test_columns_are_resolved(self):
        df = frame(Region=["N", "S"], Sales=[1, 2], code=["1", "2"])
        self.render(df, chart_type="bar", x="region", y=["SALES", "missing"])
        self.render(df, chart_type="bar")  # x and y inferred
        self.render(df, chart_type="scatter", x="Sales", y=["Sales"])
        self.render(frame(a=[1, 2], b=[3, 4]), chart_type="bar")  # only numeric columns: x is the first one
        self.render(frame(v=[1, 2, 3, 4, 5]), chart_type="hist")
        self.render(frame(v=[1, 2, 3, 4, 5]), chart_type="hist", x="v")

    def test_errors(self):
        render = m.charts.render_chart
        spec = m.charts.ChartSpec
        with self.assertRaises(m.charts.ChartError):
            render(frame(), spec())
        with self.assertRaises(m.charts.ChartError):
            render(None, spec())
        with self.assertRaises(m.charts.ChartError):
            render(frame(a=["x", "y"], b=["z", "w"]), spec(chart_type="bar"))
        with self.assertRaises(m.charts.ChartError):
            render(frame(a=["x", "y"]), spec(chart_type="hist", x="a"))
        with self.assertRaises(m.charts.ChartError):
            render(frame(a=["x", "y"], b=[0, -1]), spec(chart_type="pie", x="a", y=["b"]))

    def test_time_series(self):
        dates = ["2024-01-03", "2024-01-01", "2024-01-02"]
        self.render(frame(day=dates, v=[3, 1, 2]), chart_type="line", x="day", y=["v"])
        self.render(frame(day=["2024-01-01 10:30", "2024-01-02 11:00"], v=[1, 2]), chart_type="scatter", x="day", y=["v"])
        self.render(frame(n=[1, 2, 3], v=[1, 2, 3]), chart_type="area", x="n", y=["v"])
        self.render(frame(k=["a"] * 50, v=list(range(50))), chart_type="line", x="k", y=["v"])

    def test_large_inputs_are_limited(self):
        many = [f"category number {i}" for i in range(60)]
        png = self.render(frame(c=many, v=list(range(60))), chart_type="bar", x="c", y=["v"])
        self.render(frame(c=many, v=list(range(1, 61))), chart_type="pie", x="c", y=["v"])
        self.render(frame(c=many, v=list(range(60))), chart_type="bar", x="c", y=["v"], title="")
        self.assertGreater(len(png), 1000)

    def test_labels_and_numbers(self):
        label = m.charts._label
        self.assertEqual(label(m.datasets.pd.Timestamp("2024-01-02")), "2024-01-02")
        self.assertEqual(label(m.datasets.pd.Timestamp("2024-01-02 10:30")), "2024-01-02 10:30")
        self.assertEqual(label(None), "(n/a)")
        self.assertEqual(label(float("nan")), "(n/a)")
        self.assertTrue(label("x" * 50).endswith("…"))
        fmt = m.charts._number_formatter()
        self.assertEqual([fmt(v, None) for v in (2_500_000_000, 1_000_000, 1_500, 12, 2.5)], ["2.5B", "1M", "1.5k", "12", "2.5"])

    def test_maybe_datetime(self):
        maybe = m.charts._maybe_datetime
        numbers = frame(v=[1, 2])["v"]
        self.assertIs(maybe(numbers), numbers)
        text = frame(v=["x", "y"])["v"]
        self.assertIs(maybe(text), text)
        self.assertTrue(m.datasets.pd.api.types.is_datetime64_any_dtype(maybe(frame(v=["2024-01-01", "2024-02-01"])["v"])))
        weird = frame(v=[object(), object()])["v"]
        self.assertIs(maybe(weird), weird)
        mixed_timezones = frame(v=["2024-01-01T00:00+01:00", "2024-01-01T00:00+02:00"])["v"]
        self.assertIs(maybe(mixed_timezones), mixed_timezones)


class DeliveryTest(unittest.TestCase):
    def test_markdown(self):
        inline = m.charts.chart_markdown_inline(PNG, "Sales [2024]\nby region")
        self.assertTrue(inline.startswith("![Sales  2024  by region](data:image/png;base64,"))
        self.assertEqual(base64.b64decode(inline.split(",", 1)[1][:-1]), PNG)
        text = f"Answer\n\n{inline}\n\n{inline}"
        self.assertEqual(m.charts.strip_inline_images(text), "Answer\n\n[chart: Sales  2024  by region]\n\n[chart: Sales  2024  by region]")
        self.assertEqual(m.charts.strip_inline_images(None), "")


class ChartToolTest(unittest.TestCase):
    def setUp(self):
        tmp = support.temporary_folder(self)
        path = tmp / "sales.csv"
        path.write_text("region,amount\nN,10\nS,20\nE,5\n")
        self.engine = m.data_engine.engine_from_csv(support.configured_file(path.name, path.read_bytes()))
        self.collector = m.chart_tool.ChartCollector()

    def factory(self, mode="on_request", summary_rows=30, max_rows=1000):
        return m.chart_tool.ChartToolFactory(
            engine=self.engine, collector=self.collector, mode=mode, max_rows=max_rows, summary_rows=summary_rows,
        )

    def args(self, **overrides):
        args = {"sql": "SELECT region, amount FROM sales ORDER BY amount DESC", "chart_type": "bar", "x": "region",
                "y": ["amount"], "title": "Sales"}
        args.update(overrides)
        return args

    def test_prefix_addendum(self):
        self.assertIn("ONLY IF the user explicitly asks", self.factory().prefix_addendum)
        self.assertIn("comparisons", self.factory("auto").prefix_addendum)

    def test_structured_tool(self):
        tool = self.factory(summary_rows=2).build(structured=True)
        result = tool.invoke(self.args())
        self.assertIn('The bar chart "Sales" was drawn from 3 rows', result)
        self.assertIn("(1 more rows)", result)
        self.assertEqual(len(self.collector.charts), 1)
        self.assertEqual(self.collector.charts[0].rows, 3)

    def test_structured_tool_async_and_single_column(self):
        # regression: "y" given as a string aborted the whole agent with a validation error
        tool = self.factory().build(structured=True)
        result = support.run(tool.ainvoke(self.args(y="amount")))
        self.assertIn("was drawn", result)
        self.assertIn("was drawn", tool.invoke(self.args(y=None)))  # the numeric columns are found automatically

    def test_invalid_arguments_go_back_to_the_agent(self):
        tool = self.factory().build(structured=True)
        result = tool.invoke({"chart_type": "bar"})  # no sql
        self.assertIsInstance(result, str)
        self.assertEqual(self.collector.charts, [])

    def test_errors_go_back_to_the_agent(self):
        tool = self.factory().build(structured=True)
        self.assertIn("Error: Only SELECT", tool.invoke(self.args(sql="DELETE FROM sales")))
        self.assertIn("Error:", tool.invoke(self.args(sql="SELECT region FROM sales")))  # nothing numeric
        self.assertEqual(self.collector.charts, [])

    def test_truncated_rows(self):
        result = self.factory(max_rows=2).build(structured=True).invoke(self.args())
        self.assertIn("drawn from 2+ rows", result)

    def test_at_most_three_charts(self):
        tool = self.factory().build(structured=True)
        for _ in range(3):
            self.assertIn("was drawn", tool.invoke(self.args()))
        self.assertIn("at most 3 charts", tool.invoke(self.args()))
        self.assertEqual(len(self.collector.charts), 3)

    def test_text_tool(self):
        tool = self.factory().build(structured=False)
        self.assertIn("Example of input", tool.description)
        payload = "```json\n" + json.dumps(self.args()) + "\n```"
        self.assertIn("was drawn", tool.invoke(payload))
        self.assertIn("was drawn", support.run(tool.ainvoke(json.dumps(self.args()))))
        for invalid in ("not json", "[1, 2]", "{broken"):
            with self.subTest(payload=invalid):
                self.assertIn("Error: invalid input", tool.invoke(invalid))
        self.assertEqual(len(self.collector.charts), 2)

    def test_parse_text_input(self):
        parse = m.chart_tool._parse_text_input
        self.assertEqual(parse('text before {"a": "}"} after'), {"a": "}"})
        with self.assertRaises(ValueError):
            parse('["a"]')
        with self.assertRaises(ValueError):
            parse('{"a": 1} and {"b": 2}')


if __name__ == "__main__":
    unittest.main()
