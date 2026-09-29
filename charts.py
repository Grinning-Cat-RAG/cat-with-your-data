"""Chart rendering (matplotlib, headless) and delivery of the generated images, inline in the answer."""
import base64
import io
import re
import textwrap
from typing import List, Literal

import matplotlib

matplotlib.use("Agg")  # headless backend: the Cat runs on a server

import pandas as pd  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402
from pydantic import BaseModel, Field, field_validator  # noqa: E402

ChartType = Literal["bar", "barh", "line", "area", "pie", "scatter", "hist"]

MAX_CATEGORIES = 40
MAX_PIE_SLICES = 10


class ChartSpec(BaseModel):
    chart_type: ChartType = "bar"
    x: str | None = None
    y: List[str] = Field(default_factory=list)
    title: str = ""
    x_label: str | None = None
    y_label: str | None = None
    stacked: bool = False

    @field_validator("chart_type", mode="before")
    @classmethod
    def _chart_type(cls, value):
        value = str(value or "bar").strip().lower()
        aliases = {"column": "bar", "columns": "bar", "horizontal_bar": "barh", "histogram": "hist", "donut": "pie",
                   "scatterplot": "scatter", "timeseries": "line", "stacked_bar": "bar"}
        value = aliases.get(value, value)
        return value if value in {"bar", "barh", "line", "area", "pie", "scatter", "hist"} else "bar"

    @field_validator("y", mode="before")
    @classmethod
    def _y(cls, value):
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        return [str(v) for v in value if v]


class ChartError(Exception):
    pass


# --------------------------------------------------------------------------------------------------------------------
# data preparation
# --------------------------------------------------------------------------------------------------------------------
def _find_column(df: pd.DataFrame, name: str | None) -> str | None:
    if not name:
        return None
    if name in df.columns:
        return name
    lowered = {str(c).lower(): c for c in df.columns}
    return lowered.get(str(name).lower())


def _is_numeric(series: pd.Series) -> bool:
    converted = pd.to_numeric(series, errors="coerce")
    return converted.notna().sum() >= max(1, int(0.8 * series.notna().sum()))


def _resolve_columns(df: pd.DataFrame, spec: ChartSpec) -> tuple[str | None, List[str]]:
    x = _find_column(df, spec.x)
    y = [
        c for c in (_find_column(df, name) for name in spec.y)
        if c is not None and (c != x or spec.chart_type == "scatter")
    ]

    if spec.chart_type == "hist":
        if not y:
            candidates = [c for c in df.columns if _is_numeric(df[c])]
            y = candidates[:1] or ([x] if x else [])
        return None, y[:1]

    if x is None:
        non_numeric = [c for c in df.columns if not _is_numeric(df[c])]
        x = non_numeric[0] if non_numeric else df.columns[0]
    if not y:
        y = [c for c in df.columns if c != x and _is_numeric(df[c])]
    if not y:
        raise ChartError("The query result does not contain numeric values to plot.")
    if spec.chart_type == "pie":
        y = y[:1]
    return x, y


def _maybe_datetime(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series) or pd.api.types.is_numeric_dtype(series):
        return series
    try:
        parsed = pd.to_datetime(series, errors="coerce", format="mixed")
    except (TypeError, ValueError):
        return series
    return parsed if parsed.notna().mean() > 0.9 else series


def _label(value) -> str:
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d") if value == value.normalize() else value.strftime("%Y-%m-%d %H:%M")
    text = "" if value is None or (isinstance(value, float) and pd.isna(value)) else str(value)
    return textwrap.shorten(text, width=28, placeholder="…") if text else "(n/a)"


def _number_formatter():
    def fmt(value, _pos):
        magnitude = abs(value)
        for divisor, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "k")):
            if magnitude >= divisor:
                return f"{value / divisor:.1f}{suffix}".replace(".0", "")
        return f"{value:.2f}".rstrip("0").rstrip(".") if value % 1 else f"{int(value)}"
    return FuncFormatter(fmt)


# --------------------------------------------------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------------------------------------------------
def render_chart(df: pd.DataFrame, spec: ChartSpec) -> bytes:
    """Render the chart described by ``spec`` from ``df`` and return the PNG bytes."""
    if df is None or df.empty:
        raise ChartError("The query returned no data.")

    df = df.copy()
    x, y = _resolve_columns(df, spec)
    for col in y:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    fig = Figure(figsize=(10, 5.6), dpi=110, layout="constrained")
    ax = fig.add_subplot()
    kind = spec.chart_type
    notes: List[str] = []

    if kind == "hist":
        values = df[y[0]].dropna()
        if values.empty:
            raise ChartError("No numeric values for the histogram.")
        ax.hist(values, bins=min(30, max(5, int(len(values) ** 0.5))), edgecolor="white")
        ax.set_xlabel(spec.x_label or y[0])
        ax.set_ylabel(spec.y_label or "count")

    elif kind == "pie":
        data = df[[x, y[0]]].dropna()
        data = data[data[y[0]] > 0].sort_values(y[0], ascending=False)
        if data.empty:
            raise ChartError("No positive values for the pie chart.")
        if len(data) > MAX_PIE_SLICES:
            others = data.iloc[MAX_PIE_SLICES - 1:][y[0]].sum()
            data = pd.concat([
                data.iloc[:MAX_PIE_SLICES - 1],
                pd.DataFrame({x: ["Others"], y[0]: [others]}),
            ])
        ax.pie(
            data[y[0]], labels=[_label(v) for v in data[x]], autopct="%1.1f%%", startangle=90, counterclock=False,
            wedgeprops={"edgecolor": "white"},
        )
        ax.axis("equal")

    elif kind == "scatter":
        xs = pd.to_numeric(df[x], errors="coerce") if _is_numeric(df[x]) else _maybe_datetime(df[x])
        for col in y:
            ax.scatter(xs, df[col], label=col, alpha=0.75)
        ax.set_xlabel(spec.x_label or x)
        ax.set_ylabel(spec.y_label or (y[0] if len(y) == 1 else ""))
        ax.yaxis.set_major_formatter(_number_formatter())

    elif kind in ("line", "area"):
        xs = _maybe_datetime(df[x])
        data = df.assign(**{x: xs}).sort_values(x) if pd.api.types.is_datetime64_any_dtype(xs) else df
        is_categorical = not (pd.api.types.is_datetime64_any_dtype(xs) or pd.api.types.is_numeric_dtype(xs))
        positions = range(len(data)) if is_categorical else data[x]
        if kind == "area" and spec.stacked and len(y) > 1:
            ax.stackplot(positions, *[data[c].fillna(0) for c in y], labels=y, alpha=0.8)
        else:
            for col in y:
                if kind == "area":
                    ax.fill_between(positions, data[col].fillna(0), alpha=0.35)
                ax.plot(positions, data[col], marker="o" if len(data) <= 40 else None, label=col, linewidth=2)
        if is_categorical:
            step = max(1, len(data) // 20)
            ax.set_xticks(list(positions)[::step], [_label(v) for v in data[x]][::step])
        ax.set_xlabel(spec.x_label or x)
        ax.set_ylabel(spec.y_label or (y[0] if len(y) == 1 else ""))
        ax.yaxis.set_major_formatter(_number_formatter())
        if not is_categorical:
            fig.autofmt_xdate()

    else:  # bar / barh
        data = df
        if len(data) > MAX_CATEGORIES:
            notes.append(f"showing the first {MAX_CATEGORIES} of {len(data)} categories")
            data = data.head(MAX_CATEGORIES)
        labels = [_label(v) for v in data[x]]
        positions = list(range(len(data)))
        width = 0.8 / (1 if spec.stacked else len(y))
        bottom = pd.Series(0.0, index=data.index)
        for i, col in enumerate(y):
            values = data[col].fillna(0)
            if spec.stacked:
                args = {"left": bottom} if kind == "barh" else {"bottom": bottom}
                offsets = positions
            else:
                args = {}
                offsets = [p - 0.4 + width * (i + 0.5) for p in positions]
            if kind == "barh":
                bars = ax.barh(offsets, values, height=width, label=col, **args)
            else:
                bars = ax.bar(offsets, values, width=width, label=col, **args)
            if len(y) == 1 and len(data) <= 25:
                ax.bar_label(bars, labels=[_number_formatter()(v, None) for v in values], padding=2, fontsize=8)
            bottom = bottom + values
        if kind == "barh":
            ax.set_yticks(positions, labels)
            ax.invert_yaxis()
            ax.set_xlabel(spec.y_label or (y[0] if len(y) == 1 else ""))
            ax.set_ylabel(spec.x_label or x)
            ax.xaxis.set_major_formatter(_number_formatter())
        else:
            rotate = len(data) > 6 or max((len(label) for label in labels), default=0) > 10
            ax.set_xticks(positions, labels, rotation=45 if rotate else 0, ha="right" if rotate else "center")
            ax.set_xlabel(spec.x_label or x)
            ax.set_ylabel(spec.y_label or (y[0] if len(y) == 1 else ""))
            ax.yaxis.set_major_formatter(_number_formatter())

    if kind != "pie":
        ax.grid(axis="x" if kind == "barh" else "y", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
        if len(y) > 1:
            ax.legend(frameon=False)

    title = spec.title or ""
    if notes:
        title = f"{title}\n({'; '.join(notes)})" if title else f"({'; '.join(notes)})"
    if title:
        ax.set_title(title, fontsize=13, fontweight="bold")

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png")
    return buffer.getvalue()


# --------------------------------------------------------------------------------------------------------------------
# delivery
# --------------------------------------------------------------------------------------------------------------------
def chart_markdown_inline(png: bytes, title: str) -> str:
    alt = re.sub(r"[\[\]\n]", " ", title or "chart").strip()
    return f"![{alt}](data:image/png;base64,{base64.b64encode(png).decode('ascii')})"


_DATA_URI_IMAGE = re.compile(r"!\[([^\]]*)\]\(data:image/[a-z]+;base64,[A-Za-z0-9+/=]+\)")


def strip_inline_images(text: str) -> str:
    """Replace the inline (base64) images with a short placeholder, to keep them out of the prompts."""
    return _DATA_URI_IMAGE.sub(lambda m: f"[chart: {m.group(1)}]", text or "")
