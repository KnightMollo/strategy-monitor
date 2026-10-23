"""Shared Artificial-Analysis-inspired design system, mirrored in strategy-monitor/lib/aa_theme.py.

Colors/fonts live in .streamlit/config.toml; this module adds layout CSS and small HTML components.
"""
from __future__ import annotations

import html
import itertools
import re
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any

import streamlit as st

INK = "#000000"
MUTED = "#6B6B6B"
FAINT = "#949494"
LINE = "#D9D9D9"
SOFT = "#F2F2F2"
PLUM = "#741A66"
UP = "#1E9E5A"
DOWN = "#D43C3C"
SERIF = '"Newsreader", "Victor Serif", Georgia, Cambria, serif'
SANS = '"Inter", "Suisse Intl", ui-sans-serif, system-ui, sans-serif'
SERIES = {
    "plum": PLUM,
    "violet": "#7C4DFF",
    "blue": "#2B44D6",
    "sky": "#3D8BF0",
    "green": "#4C9F59",
    "orange": "#EE7A35",
    "clay": "#C27A5D",
    "yellow": "#E0B800",
    "gray": FAINT,
    "ink": INK,
}
# tone -> (text/accent color, tinted background)
TONES = {
    "up": (UP, "#EAF6EF"),
    "down": (DOWN, "#FBECEC"),
    "warn": ("#A15C00", "#FFF4E0"),
    "hedge": (PLUM, "#F6EAF4"),
    "info": ("#2B44D6", "#EEF0FC"),
    "hold": (INK, SOFT),
    "neutral": (MUTED, SOFT),
}
STRATEGY_COLORS = {
    "Permanent Portfolio": SERIES["green"], "PP": SERIES["green"],
    "Hold'em": SERIES["blue"], "TMT": SERIES["orange"], "GEX Filter": SERIES["plum"],
    "Dragon Scout": SERIES["clay"], "SPX BWB": SERIES["violet"],
    "Portfolio": INK, "Total (weighted)": INK, "Tracked Total": INK,
    "SPY": FAINT, "Risk-Free": "#BDBDBD",
}
ASSET_COLORS = {
    "GLDM": SERIES["yellow"], "GLD": SERIES["yellow"], "SLV": "#A7A9AC",
    "SPTL": SERIES["blue"], "TLT": SERIES["blue"], "SGOV": FAINT, "BOXX": FAINT,
    "VTI": PLUM, "Cash": "#E5E5E5",
}
_HOLDING_TONES = {"SGOV": "neutral", "BOXX": "neutral", "CASH": "neutral", "UVXY": "hedge", "SQQQ": "down", "TLT": "info"}


def holding_tone(ticker: str | None) -> str:
    """Banner tone for a next-hold ticker: long risk is green, cash gray, hedges plum, shorts red."""
    if not ticker:
        return "neutral"
    return _HOLDING_TONES.get(str(ticker).upper(), "up")
_CARD_IDS = itertools.count()

_CSS = """
<style>
[data-testid="stDecoration"], [data-testid="stAppDeployButton"] { display: none; }
header[data-testid="stHeader"] { background: rgba(255, 255, 255, 0.92); backdrop-filter: blur(6px); }
[data-testid="stMainBlockContainer"] { max-width: 1360px; padding-top: 2.75rem; padding-bottom: 4rem; }
[data-testid="stMarkdownContainer"] p { line-height: 1.55; }

h1 { font-weight: 400 !important; letter-spacing: -0.025em; line-height: 1.05 !important; padding-bottom: 0.25rem !important; }
h2, h3 { font-weight: 400 !important; letter-spacing: -0.015em; }
[data-testid="stHeading"] h2::before, [data-testid="stHeading"] h3::before {
    content: ""; display: inline-block; width: 0.55em; height: 0.55em; margin-right: 0.45em;
    background: __INK__; vertical-align: 0.04em;
}
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p { color: __MUTED__; }
hr { border-color: __LINE__ !important; margin: 1.75rem 0 !important; }

[data-testid="stMetric"] { border: 1px solid __LINE__; border-radius: 8px; padding: 12px 16px; background: #FFFFFF; }
[data-testid="stMetricLabel"] p { color: __MUTED__; font-size: 0.8rem; }
[data-testid="stMetricValue"] { font-weight: 500; font-variant-numeric: tabular-nums; letter-spacing: -0.01em; font-size: 1.75rem; }
[class*="st-key-aa-card"] [data-testid="stMetric"] { border: none; padding: 0; }

[data-baseweb="tab-list"] { background: __SOFT__; border-radius: 9999px; padding: 4px; gap: 2px; width: fit-content; }
[data-baseweb="tab"] { border-radius: 9999px; padding: 6px 18px !important; height: auto; }
[data-baseweb="tab"][aria-selected="true"] { background: __INK__; }
[data-baseweb="tab"][aria-selected="true"] p { color: #FFFFFF; }
[data-baseweb="tab-highlight"], [data-baseweb="tab-border"] { display: none; }

[data-testid="stRadioGroup"] { gap: 4px !important; }
[data-testid="stRadioOption"] { background: __SOFT__; border-radius: 9999px; padding: 5px 16px; margin: 0 !important; }
[data-testid="stRadioOption"] > div > div > div:first-child:not([data-testid]) { display: none; }
[data-testid="stRadioOption"][data-selected="true"] { background: __INK__; }
[data-testid="stRadioOption"][data-selected="true"] p { color: #FFFFFF; }
[data-testid="stSidebar"] [data-testid="stRadioGroup"] { width: 100%; }
[data-testid="stSidebar"] [data-testid="stRadioOption"] { width: 100%; padding: 7px 14px; }
[data-testid="stSidebar"] [data-testid="stRadioOption"]:not([data-selected="true"]) { background: transparent; }
[data-testid="stSidebar"] [data-testid="stRadioOption"]:not([data-selected="true"]):hover { background: __SOFT__; }

[data-testid^="stBaseButton-segmented_control"] { border-radius: 9999px !important; }
[data-testid="stBaseButton-segmented_controlActive"] { background: __INK__ !important; border-color: __INK__ !important; }
[data-testid="stBaseButton-segmented_controlActive"] p { color: #FFFFFF !important; }

[data-testid^="stBaseButton-secondary"] { background: __SOFT__; border-color: transparent; }
[data-testid^="stBaseButton-secondary"]:hover { background: #E7E7E7; border-color: transparent; color: __INK__; }
[data-testid="stExpander"] summary p { font-weight: 500; }

[data-testid="stSidebarNavLink"] { border-radius: 9999px; }
[data-testid="stSidebarNavLink"][aria-current="page"] { background: __INK__; }
[data-testid="stSidebarNavLink"][aria-current="page"] p,
[data-testid="stSidebarNavLink"][aria-current="page"] span { color: #FFFFFF !important; }
[data-testid="stNavSectionHeader"] p { color: __FAINT__; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.08em; }
[data-testid="stSidebarHeader"]::before {
    content: "__WORDMARK__"; display: inline-flex; align-items: center; white-space: nowrap;
    background: __INK__; color: #FFFFFF; font-family: __SERIF__; font-size: 1.1rem;
    padding: 0.4rem 1rem; border-radius: 9999px;
}

.aa-sub { display: flex; justify-content: space-between; gap: 1rem; flex-wrap: wrap; color: __MUTED__; font-size: 0.95rem; margin: -0.25rem 0 1.25rem 0; }
.aa-sub .aa-meta { color: __FAINT__; font-size: 0.85rem; }
.aa-card-title { display: flex; align-items: center; gap: 0.5rem; margin: 0; font-family: __SERIF__; font-size: 1.35rem;
    font-weight: 400; color: __INK__; line-height: 1.2; }
.aa-square { width: 0.65rem; height: 0.65rem; flex: none; display: inline-block; }
.aa-subtitle { color: __FAINT__; font-size: 0.8rem; margin: 0.15rem 0 0.75rem 0; }

.aa-kpis { display: grid; gap: 12px; margin: 0.25rem 0 0.75rem 0; }
.aa-kpi { border: 1px solid __LINE__; border-radius: 8px; padding: 12px 16px; background: #FFFFFF; min-width: 0; }
.aa-kpis.flat { gap: 18px; }
.aa-kpis.flat .aa-kpi { border: none; padding: 0; }
.aa-kpi-label { color: __MUTED__; font-size: 0.78rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.aa-kpi-value { font-size: 1.45rem; font-weight: 500; letter-spacing: -0.015em; font-variant-numeric: tabular-nums;
    line-height: 1.25; color: __INK__; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.aa-kpis.compact .aa-kpi-value { font-size: 1.2rem; }
.aa-kpi-sub { color: __FAINT__; font-size: 0.78rem; font-variant-numeric: tabular-nums; }

.aa-banner { border-radius: 8px; padding: 12px 16px; margin: 0.25rem 0 0.75rem 0; border-left: 3px solid var(--aa-tone); background: var(--aa-tint); }
.aa-banner-top { display: flex; align-items: center; gap: 0.6rem; flex-wrap: wrap; }
.aa-banner-head { font-weight: 600; color: __INK__; font-size: 1rem; }
.aa-banner-detail { color: __MUTED__; font-size: 0.86rem; margin-top: 0.3rem; line-height: 1.5; }
.aa-badge { display: inline-flex; align-items: center; border-radius: 9999px; padding: 2px 10px; font-size: 0.72rem; font-weight: 600;
    letter-spacing: 0.04em; text-transform: uppercase; color: #FFFFFF; background: var(--aa-tone); white-space: nowrap; }

.aa-rows { margin: 0.25rem 0 0.5rem 0; }
.aa-row { display: flex; justify-content: space-between; gap: 1rem; padding: 7px 0; border-bottom: 1px solid #EEEEEE; font-size: 0.88rem; }
.aa-row:last-child { border-bottom: none; }
.aa-row-k { color: __MUTED__; }
.aa-row-v { color: __INK__; text-align: right; font-variant-numeric: tabular-nums; }

.aa-table { width: 100%; border-collapse: collapse; font-size: 0.88rem; font-variant-numeric: tabular-nums; margin: 0.25rem 0 0.75rem 0; border: none !important; }
.aa-table th { text-align: right; color: __MUTED__; font-weight: 500; padding: 8px 12px; border: none !important; border-bottom: 1px solid __LINE__ !important; white-space: nowrap; background: transparent !important; }
.aa-table td { text-align: right; padding: 9px 12px; border: none !important; border-bottom: 1px solid #EEEEEE !important; white-space: nowrap; }
.aa-table th:first-child, .aa-table td:first-child { text-align: left; }
.aa-table tr.aa-total td { border-top: 1px solid __INK__ !important; border-bottom: none !important; font-weight: 600; }
.aa-name { display: inline-flex; align-items: center; gap: 0.45rem; }
.aa-up { color: __UP__; } .aa-down { color: __DOWN__; }
</style>
"""


def _html(markup: str) -> None:
    # "$" pairs would otherwise be rendered as LaTeX by Streamlit markdown.
    st.markdown(markup.replace("$", "&#36;"), unsafe_allow_html=True)


def esc(value: Any) -> str:
    return html.escape(str(value))


def apply_theme(wordmark: str) -> None:
    css = _CSS
    for token, value in (("__INK__", INK), ("__MUTED__", MUTED), ("__FAINT__", FAINT), ("__LINE__", LINE),
                         ("__SOFT__", SOFT), ("__UP__", UP), ("__DOWN__", DOWN), ("__SERIF__", SERIF),
                         ("__WORDMARK__", re.sub(r'["\\<>]', "", wordmark))):
        css = css.replace(token, value)
    st.markdown(css, unsafe_allow_html=True)


def page_header(title: str, subtitle: str | None = None, meta: str | None = None) -> None:
    st.title(title, anchor=False)
    if subtitle or meta:
        _html(f'<div class="aa-sub"><span>{esc(subtitle or "")}</span><span class="aa-meta">{esc(meta or "")}</span></div>')


def card_title(title: str, color: str = INK, subtitle: str | None = None) -> None:
    """Colored square + serif title (+ muted subtitle), the AA chart-card header."""
    markup = (f'<div class="aa-card-title"><span class="aa-square" style="background:{esc(color)}"></span>'
              f"{esc(title)}</div>")
    markup += f'<div class="aa-subtitle">{esc(subtitle)}</div>' if subtitle else '<div style="height:0.6rem"></div>'
    _html(markup)


@contextmanager
def card(title: str | None = None, color: str = INK, subtitle: str | None = None) -> Iterator[Any]:
    with st.container(border=True, key=f"aa-card-{next(_CARD_IDS)}") as container:
        if title:
            card_title(title, color, subtitle)
        yield container


def tone_of(value: float | None) -> str | None:
    if value is None:
        return None
    return "up" if value > 0 else "down" if value < 0 else None


def money(value: float | None, decimals: int = 0, signed: bool = False) -> str:
    if value is None:
        return "N/A"
    sign = "+" if signed and value > 0 else "-" if value < 0 else ""
    return f"{sign}${abs(value):,.{decimals}f}"


def colored(text: str, tone: str | None) -> str:
    """HTML span colored by tone; text must already be safe."""
    return f'<span class="aa-{tone}">{text}</span>' if tone in ("up", "down") else text


def kpis(items: Sequence[Mapping[str, Any]], *, columns: int | None = None, flat: bool = False,
         compact: bool = False) -> None:
    """items: dicts with label, value, optional sub and tone ('up'/'down')."""
    count = columns or max(len(items), 1)
    classes = "aa-kpis" + (" flat" if flat else "") + (" compact" if compact else "")
    cells = []
    for item in items:
        value = colored(esc(item["value"]), item.get("tone"))
        sub = f'<div class="aa-kpi-sub">{esc(item["sub"])}</div>' if item.get("sub") else ""
        cells.append(f'<div class="aa-kpi"><div class="aa-kpi-label">{esc(item["label"])}</div>'
                     f'<div class="aa-kpi-value">{value}</div>{sub}</div>')
    _html(f'<div class="{classes}" style="grid-template-columns:repeat({count},minmax(0,1fr))">{"".join(cells)}</div>')


def banner(headline: str, detail: str | None = None, *, tone: str = "hold", label: str | None = None) -> None:
    """Action/status callout: colored left rule, optional pill label, bold headline, muted detail."""
    color, tint = TONES.get(tone, TONES["hold"])
    badge = f'<span class="aa-badge">{esc(label)}</span>' if label else ""
    body = f'<div class="aa-banner-detail">{esc(detail)}</div>' if detail else ""
    _html(f'<div class="aa-banner" style="--aa-tone:{color};--aa-tint:{tint}">'
          f'<div class="aa-banner-top">{badge}<span class="aa-banner-head">{esc(headline)}</span></div>{body}</div>')


def rows(pairs: Iterable[Sequence[Any]]) -> None:
    """Hairline key/value list for secondary facts; items are (key, value) or (key, value, tone)."""
    items = "".join(
        f'<div class="aa-row"><span class="aa-row-k">{esc(pair[0])}</span>'
        f'<span class="aa-row-v">{colored(esc(pair[1]), pair[2] if len(pair) > 2 else None)}</span></div>'
        for pair in pairs
    )
    _html(f'<div class="aa-rows">{items}</div>')


def table(header: Sequence[str], body: Sequence[Sequence[Any]], *, total: Sequence[Any] | None = None,
          colors: Mapping[str, str] | None = None) -> None:
    """Compact summary table. Cells may be plain values or (text, tone) tuples; first column may get a color square."""
    def cell(value: Any) -> str:
        if isinstance(value, tuple):
            return colored(esc(value[0]), value[1])
        return esc(value)

    def first(value: Any) -> str:
        square = (f'<span class="aa-square" style="background:{esc(colors[value])}"></span>'
                  if colors and value in colors else "")
        return f'<span class="aa-name">{square}{esc(value)}</span>'

    lines = ["<table class='aa-table'><thead><tr>" + "".join(f"<th>{esc(h)}</th>" for h in header) + "</tr></thead><tbody>"]
    for row in body:
        lines.append("<tr><td>" + first(row[0]) + "</td>" + "".join(f"<td>{cell(v)}</td>" for v in row[1:]) + "</tr>")
    if total is not None:
        lines.append("<tr class='aa-total'><td>" + esc(total[0]) + "</td>" + "".join(f"<td>{cell(v)}</td>" for v in total[1:]) + "</tr>")
    lines.append("</tbody></table>")
    _html("".join(lines))


def style_figure(figure: Any, *, height: int | None = None) -> Any:
    """AA chart look: white canvas, dotted grid, top-left horizontal legend."""
    figure.update_layout(
        font={"family": SANS, "color": INK, "size": 12},
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        colorway=list(SERIES.values()),
        margin={"t": 36, "b": 36, "l": 48, "r": 16},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0, "title": None},
        hoverlabel={"bgcolor": "#FFFFFF", "bordercolor": LINE, "font": {"color": INK, "family": SANS}},
        title={"font": {"family": SERIF, "size": 18, "color": INK}, "x": 0, "xanchor": "left"},
    )
    if height:
        figure.update_layout(height=height)
    axis = {"gridcolor": "#E5E5E5", "griddash": "dot", "zeroline": False, "linecolor": LINE, "tickfont": {"color": MUTED}}
    figure.update_xaxes(**axis)
    figure.update_yaxes(**axis)
    return figure


def chart(figure: Any, *, height: int | None = None, key: str | None = None) -> None:
    st.plotly_chart(style_figure(figure, height=height), theme=None, width="stretch", key=key,
                    config={"displayModeBar": False})
