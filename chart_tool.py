"""The ``draw_chart`` tool given to the LangChain SQL agent.

The agent explores the database with the standard tools of ``SQLDatabaseToolkit`` and, when a chart is needed, calls
this tool with a read-only query and the chart specification. The tool runs the query, renders the chart and keeps it
aside (the agent only sees a textual summary of the plotted data), so that the plugin can attach it to the answer.
"""
import asyncio
import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List

from langchain_core.tools import BaseTool, StructuredTool, Tool
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.engine import Engine

from cat import log

from .charts import ChartSpec, render_chart
from .data_engine import clean_sql, run_select

TOOL_NAME = "draw_chart"
MAX_CHARTS_PER_ANSWER = 3

WHEN_TO_USE = {
    "on_request": "Use this tool ONLY IF the user explicitly asks for a chart, plot, graph, diagram, histogram or any "
                  "other kind of visualization (in any language), or asks to change a chart drawn before. "
                  "Never use it otherwise.",
    "auto": "Use this tool when the user asks for a visualization, or when the answer involves comparisons, "
            "rankings, trends or distributions over several values that are clearly easier to understand visually. "
            "Do NOT use it for single values, yes/no answers or short lists.",
}

ARGUMENTS = (
    '"sql": ONE read-only SELECT (or WITH ... SELECT) statement returning exactly the data to plot, already '
    "aggregated, sorted and limited (at most 40 categories or 500 points; the usual limit on the number of results "
    "does not apply here), with short and simple column aliases without spaces. Check the schema of the tables first. "
    '"chart_type": one of "bar", "barh", "line", "area", "pie", "scatter", "hist" (line/area for trends over time, '
    "bar/barh for comparisons among categories, pie only for parts of a whole with few slices, scatter for the "
    'relation between two numeric variables, hist for the distribution of one numeric variable). '
    '"x": the output column for the x axis, the categories or the pie slices (null for hist). '
    '"y": the list of the numeric output columns to plot (a single column for pie and hist). '
    '"title", "x_label", "y_label": short labels in the language of the user. '
    '"stacked": true only for stacked bar or area charts.'
)

PREFIX_ADDENDUM = """
You can also draw charts with the draw_chart tool: WHEN_TO_USE
The chart is shown to the user together with your final answer: in the final answer comment the data briefly, do not \
describe how to draw the chart and do not include images or links. If draw_chart returns an error, fix the input and \
call it again.
"""


@dataclass
class DrawnChart:
    png: bytes
    spec: ChartSpec
    sql: str
    rows: int


@dataclass
class ChartCollector:
    """Charts drawn by the agent during the current turn."""
    charts: List[DrawnChart] = field(default_factory=list)


class DrawChartInput(BaseModel):
    sql: str = Field(description="Read-only SELECT statement returning the data to plot")
    chart_type: str = Field(description="bar, barh, line, area, pie, scatter or hist")
    x: str | None = Field(default=None, description="Column for the x axis, categories or pie slices (null for hist)")
    y: List[str] = Field(default_factory=list, description="Numeric columns to plot")

    # LLMs often pass a single column as a string instead of a list
    @field_validator("y", mode="before")
    @classmethod
    def _y(cls, value):
        if value is None:
            return []
        return [value] if isinstance(value, str) else value
    title: str = Field(default="", description="Chart title, in the language of the user")
    x_label: str | None = Field(default=None, description="Label of the x axis")
    y_label: str | None = Field(default=None, description="Label of the y axis")
    stacked: bool = Field(default=False, description="True only for stacked bar or area charts")


def _parse_text_input(text: str) -> Dict[str, Any]:
    """Arguments of the ReAct agents, which pass the tool input as a (JSON) string."""
    text = re.sub(r"```(?:json)?", "", str(text or ""), flags=re.IGNORECASE).strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("the input must be a JSON object")
    return json.loads(text[start:end + 1])  # from "{" to "}": if it is valid JSON, it is an object


class ChartToolFactory:
    def __init__(
        self,
        engine: Engine,
        collector: ChartCollector,
        mode: str,
        max_rows: int,
        summary_rows: int,
    ):
        self.engine = engine
        self.collector = collector
        self.mode = mode
        self.max_rows = max_rows
        self.summary_rows = summary_rows

    @property
    def prefix_addendum(self) -> str:
        return PREFIX_ADDENDUM.replace("WHEN_TO_USE", WHEN_TO_USE[self.mode])

    def _draw(self, args: Dict[str, Any]) -> str:
        if len(self.collector.charts) >= MAX_CHARTS_PER_ANSWER:
            return f"Error: at most {MAX_CHARTS_PER_ANSWER} charts can be drawn for a single answer."

        try:
            sql = clean_sql(str(args.get("sql") or ""))
            spec = ChartSpec(
                chart_type=args.get("chart_type"),
                x=args.get("x"),
                y=args.get("y"),
                title=str(args.get("title") or ""),
                x_label=args.get("x_label"),
                y_label=args.get("y_label"),
                stacked=bool(args.get("stacked")),
            )
            df, truncated = run_select(self.engine, sql, self.max_rows)
            png = render_chart(df, spec)
        except Exception as e:  # the agent reads the error and tries again
            log.warning(f"[cat-with-your-data] draw_chart failed: {e}")
            return f"Error: {e}. Fix the input and call {TOOL_NAME} again."

        self.collector.charts.append(DrawnChart(png=png, spec=spec, sql=sql, rows=len(df)))

        table = df.head(self.summary_rows).to_markdown(index=False)
        if len(df) > self.summary_rows:
            table += f"\n... ({len(df) - self.summary_rows} more rows)"
        rows = f"{len(df)}{'+' if truncated else ''}"
        return (
            f"The {spec.chart_type} chart \"{spec.title}\" was drawn from {rows} rows and it will be shown to the "
            f"user together with your final answer. Plotted data:\n{table}"
        )

    # ReAct agents: single string input (a JSON object)
    def _draw_from_text(self, text: str) -> str:
        try:
            args = _parse_text_input(text)
        except (ValueError, json.JSONDecodeError) as e:
            return f"Error: invalid input ({e}). The input must be a JSON object with the keys described."
        return self._draw(args)

    async def _adraw_from_text(self, text: str) -> str:
        return await asyncio.to_thread(self._draw_from_text, text)

    # tool-calling agents: structured input
    def _draw_structured(self, **kwargs) -> str:
        return self._draw(kwargs)

    async def _adraw_structured(self, **kwargs) -> str:
        return await asyncio.to_thread(self._draw, kwargs)

    def build(self, structured: bool) -> BaseTool:
        description = f"Draw a chart from the database and show it to the user. {WHEN_TO_USE[self.mode]}"
        if structured:
            return StructuredTool.from_function(
                func=self._draw_structured,
                coroutine=self._adraw_structured,
                name=TOOL_NAME,
                description=f"{description} Arguments: {ARGUMENTS}",
                args_schema=DrawChartInput,
                # invalid arguments go back to the agent, which fixes them, instead of aborting the whole answer
                handle_validation_error=True,
            )

        example = (
            '{"sql": "SELECT region, SUM(amount) AS total FROM sales GROUP BY region ORDER BY total DESC", '
            '"chart_type": "bar", "x": "region", "y": ["total"], "title": "Sales by region", '
            '"x_label": "Region", "y_label": "Total", "stacked": false}'
        )
        return Tool(
            name=TOOL_NAME,
            func=self._draw_from_text,
            coroutine=self._adraw_from_text,
            description=(
                f"{description} Input: a JSON object (on a single line) with these keys: {ARGUMENTS} "
                f"Example of input: {example}"
            ),
        )
