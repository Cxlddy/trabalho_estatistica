"""
app.py — Dashboard principal para análise de dados do TCC.
Tecnologias: Streamlit, Plotly, Pandas.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from utils import (
    fetch_data,
    clean_dataframe,
    classify_columns,
    compute_kpis,
    generate_insights,
    get_type_label,
    get_type_color,
)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURAÇÃO DA PÁGINA
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Dashboard TCC",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# DESIGN SYSTEM
# ─────────────────────────────────────────────────────────────────────────────

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] { font-family: 'Inter', system-ui, sans-serif; }

.block-container {
    padding: 1.75rem 2.25rem 3rem 2.25rem !important;
    max-width: 1440px !important;
}

/* ── Sidebar shell ─────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: #070E1A !important;
    border-right: 1px solid rgba(255,255,255,0.05) !important;
    min-width: 280px !important;
}
[data-testid="stSidebar"] > div:first-child { padding: 0 !important; }
[data-testid="stSidebar"] section[data-testid="stSidebarContent"] {
    padding: 0 !important; gap: 0 !important;
}

/* ── Sidebar: cabeçalho ────────────────────────────────────────────────── */
.sb-head {
    padding: 1.6rem 1.4rem 1.3rem;
    background: linear-gradient(160deg, #0F172A 0%, #070E1A 100%);
    border-bottom: 1px solid rgba(255,255,255,0.04);
    position: relative;
}
.sb-head::after {
    content: '';
    position: absolute;
    inset: 0 0 auto 0;
    height: 2px;
    background: linear-gradient(90deg, #6366F1 0%, #818CF8 60%, transparent 100%);
}
.sb-head-eyebrow {
    font-size: 0.58rem; font-weight: 700; letter-spacing: 0.2em;
    text-transform: uppercase; color: #6366F1; margin-bottom: 0.45rem;
}
.sb-head-title {
    font-size: 0.95rem; font-weight: 600; color: #F1F5F9;
    margin: 0 0 0.3rem; letter-spacing: -0.01em;
}
.sb-head-sub { font-size: 0.72rem; color: #2D3F57; line-height: 1.5; margin: 0; }

/* ── Sidebar: corpo ────────────────────────────────────────────────────── */
.sb-body { padding: 0.5rem 1.4rem 1.6rem; }

/* ── Sidebar: seção ────────────────────────────────────────────────────── */
.sb-section {
    display: flex; align-items: center; gap: 0.55rem;
    margin: 1.4rem 0 0.7rem; padding-bottom: 0.45rem;
    border-bottom: 1px solid rgba(255,255,255,0.04);
}
.sb-dot { width: 5px; height: 5px; border-radius: 50%; flex-shrink: 0; }
.sb-section-text {
    font-size: 0.6rem; font-weight: 700; letter-spacing: 0.15em;
    text-transform: uppercase; color: #2D3F57;
}

/* ── Sidebar: slider label ─────────────────────────────────────────────── */
.sb-sl-label {
    font-size: 0.73rem; font-weight: 500; color: #3D526B;
    margin-bottom: 0.15rem; display: block;
}

/* ── Sidebar: contador ─────────────────────────────────────────────────── */
.sb-counter {
    margin: 1.4rem 0 1rem;
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 10px;
    padding: 1rem 1.1rem 0.85rem;
}
.sb-counter-lbl {
    font-size: 0.58rem; font-weight: 700; letter-spacing: 0.15em;
    text-transform: uppercase; color: #2D3F57; margin-bottom: 0.5rem;
}
.sb-counter-row { display: flex; align-items: baseline; gap: 0.3rem; margin-bottom: 0.65rem; }
.sb-val { font-family: 'JetBrains Mono',monospace; font-size: 1.65rem; font-weight: 500; color: #818CF8; line-height: 1; }
.sb-of  { font-size: 0.72rem; color: #2D3F57; }
.sb-pct { font-family: 'JetBrains Mono',monospace; font-size: 0.7rem; color: #3D526B; margin-left: auto; }
.sb-track { height: 3px; background: rgba(255,255,255,0.05); border-radius: 100px; overflow: hidden; }
.sb-fill  { height: 100%; border-radius: 100px; background: linear-gradient(90deg, #6366F1, #818CF8); }

/* ── Sidebar: botão ────────────────────────────────────────────────────── */
[data-testid="stSidebar"] .stButton > button {
    width: 100% !important; background: transparent !important;
    border: 1px solid rgba(255,255,255,0.07) !important; color: #2D3F57 !important;
    font-size: 0.72rem !important; font-weight: 500 !important;
    padding: 0.55rem 1rem !important; border-radius: 7px !important;
    letter-spacing: 0.04em !important; transition: all 0.18s !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    border-color: rgba(99,102,241,0.4) !important;
    color: #818CF8 !important; background: rgba(99,102,241,0.06) !important;
}

/* ── Sidebar: inputs ───────────────────────────────────────────────────── */
[data-testid="stSidebar"] label {
    font-size: 0.73rem !important; font-weight: 500 !important; color: #3D526B !important;
}
[data-testid="stSidebar"] div[data-baseweb="select"] > div {
    background: rgba(255,255,255,0.025) !important;
    border-color: rgba(255,255,255,0.07) !important;
    border-radius: 8px !important; font-size: 0.78rem !important;
}
[data-testid="stSidebar"] div[data-baseweb="select"] > div:focus-within {
    border-color: rgba(99,102,241,0.45) !important;
    box-shadow: 0 0 0 2px rgba(99,102,241,0.1) !important;
}
[data-testid="stSidebar"] .stSlider > div > div > div { background: rgba(99,102,241,0.2) !important; }
[data-testid="stSidebar"] .stSlider > div > div > div > div { background: #6366F1 !important; }

/* ── Header principal ──────────────────────────────────────────────────── */
.dash-header {
    background: linear-gradient(140deg, #0F172A 0%, #070E1A 100%);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 14px; padding: 2.2rem 2.8rem;
    margin-bottom: 2rem; position: relative; overflow: hidden;
}
.dash-header::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, #6366F1 0%, #8B5CF6 50%, #06B6D4 100%);
}
.dash-header::after {
    content: ''; position: absolute; top: -80px; right: -80px;
    width: 280px; height: 280px;
    background: radial-gradient(circle, rgba(99,102,241,0.07) 0%, transparent 70%);
    pointer-events: none;
}
.dash-eyebrow {
    display: inline-flex; align-items: center; gap: 0.5rem;
    font-size: 0.6rem; font-weight: 700; letter-spacing: 0.18em;
    text-transform: uppercase; color: #6366F1;
    background: rgba(99,102,241,0.08); border: 1px solid rgba(99,102,241,0.18);
    padding: 0.28rem 0.8rem; border-radius: 100px; margin-bottom: 1rem;
}
.dash-pulse { width: 5px; height: 5px; border-radius: 50%; background: #6366F1; animation: dpulse 2s ease infinite; }
@keyframes dpulse { 0%,100% { opacity:1; transform:scale(1); } 50% { opacity:.35; transform:scale(.65); } }
.dash-title { font-size: 1.8rem; font-weight: 700; color: #F8FAFC; margin: 0 0 0.55rem; letter-spacing: -0.025em; line-height: 1.15; }
.dash-desc  { font-size: 0.875rem; color: #3D526B; line-height: 1.65; max-width: 680px; margin: 0; }

/* ── KPI Cards ─────────────────────────────────────────────────────────── */
.kpi-card {
    background: #0B1422; border: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px; padding: 1.2rem 1.4rem 1.3rem;
    transition: border-color .2s, transform .2s; cursor: default;
}
.kpi-card:hover { border-color: rgba(99,102,241,0.3); transform: translateY(-1px); }
.kpi-lbl { font-size: 0.6rem; font-weight: 700; letter-spacing: 0.15em; text-transform: uppercase; color: #1E3352; margin-bottom: 0.55rem; }
.kpi-val { font-family: 'JetBrains Mono',monospace; font-size: 1.9rem; font-weight: 500; line-height: 1; margin-bottom: 0.3rem; }
.kpi-sub { font-size: 0.68rem; color: #1E3352; }

/* ── Títulos de seção ──────────────────────────────────────────────────── */
.sec-wrap { margin-bottom: 1.3rem; }
.sec-title { font-size: 1rem; font-weight: 600; color: #E2E8F0; margin: 0 0 0.2rem; letter-spacing: -0.015em; }
.sec-sub   { font-size: 0.77rem; color: #1E3352; margin: 0; }

/* ── Filtros ativos ────────────────────────────────────────────────────── */
.af-bar {
    display: flex; align-items: center; gap: 0.55rem; flex-wrap: wrap;
    padding: 0.65rem 1rem; background: rgba(99,102,241,0.04);
    border: 1px solid rgba(99,102,241,0.12); border-radius: 9px; margin-bottom: 1.2rem;
}
.af-lbl { font-size: 0.6rem; font-weight: 700; letter-spacing: 0.14em; text-transform: uppercase; color: #6366F1; }
.af-pill {
    font-size: 0.68rem; color: #818CF8; background: rgba(99,102,241,0.09);
    border: 1px solid rgba(99,102,241,0.18); padding: 0.16rem 0.6rem;
    border-radius: 100px; font-family: 'JetBrains Mono',monospace;
}

/* ── Abas ──────────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid rgba(255,255,255,0.05) !important;
    gap: 0 !important; margin-bottom: 1.4rem !important;
}
.stTabs [data-baseweb="tab"] {
    font-size: 0.78rem !important; font-weight: 500 !important;
    color: #2D3F57 !important; padding: 0.65rem 1.25rem !important;
    background: transparent !important; border-radius: 0 !important;
    transition: color .15s !important;
}
.stTabs [data-baseweb="tab"]:hover { color: #64748B !important; }
.stTabs [aria-selected="true"] { color: #818CF8 !important; border-bottom: 2px solid #6366F1 !important; }

/* ── Chart wrap ────────────────────────────────────────────────────────── */
.chart-wrap {
    background: #070E1A; border: 1px solid rgba(255,255,255,0.05);
    border-radius: 12px; padding: 1.3rem 1.4rem 0.4rem;
}
.chart-eyebrow { font-size: 0.58rem; font-weight: 700; letter-spacing: 0.15em; text-transform: uppercase; color: #1E3352; margin-bottom: 0.2rem; }
.chart-title   { font-size: 0.88rem; font-weight: 600; color: #94A3B8; margin: 0 0 0.12rem; }
.chart-sub     { font-size: 0.7rem; color: #1A2E44; margin: 0 0 0.15rem; }

/* ── Insight cards ─────────────────────────────────────────────────────── */
.ins-card {
    background: #070E1A; border: 1px solid rgba(255,255,255,0.05);
    border-radius: 11px; padding: 1.1rem 1.2rem 1rem;
    position: relative; transition: border-color .2s;
}
.ins-card:hover { border-color: rgba(99,102,241,0.2); }
.ins-accent { position: absolute; left:0; top:0; bottom:0; width:3px; border-radius:11px 0 0 11px; }
.ins-lbl    { font-size: 0.57rem; font-weight: 700; letter-spacing: 0.14em; text-transform: uppercase; color: #1A2E44; margin-bottom: 0.4rem; padding-left: 0.75rem; }
.ins-val    { font-size: 1.12rem; font-weight: 600; color: #E2E8F0; margin-bottom: 0.28rem; padding-left: 0.75rem; word-break: break-word; }
.ins-detail { font-size: 0.7rem; color: #2D3F57; line-height: 1.5; padding-left: 0.75rem; }

/* ── Tabela metodologia ────────────────────────────────────────────────── */
.meth-table { width: 100%; border-collapse: collapse; font-size: 0.82rem; }
.meth-table th {
    background: #040810; color: #1A2E44; font-size: 0.6rem; font-weight: 700;
    letter-spacing: 0.12em; text-transform: uppercase;
    padding: 0.72rem 1.1rem; text-align: left;
    border-bottom: 1px solid rgba(255,255,255,0.04);
}
.meth-table td { padding: 0.62rem 1.1rem; color: #64748B; border-bottom: 1px solid rgba(255,255,255,0.025); vertical-align: middle; }
.meth-table tr:hover td { background: rgba(99,102,241,0.025); }
.meth-table code {
    font-family: 'JetBrains Mono',monospace; font-size: 0.75rem;
    color: #818CF8; background: rgba(99,102,241,0.07);
    padding: 0.1rem 0.45rem; border-radius: 4px;
}
.type-pill {
    display: inline-block; font-size: 0.64rem; font-weight: 600;
    font-family: 'JetBrains Mono',monospace;
    padding: 0.22rem 0.65rem; border-radius: 100px; letter-spacing: 0.02em;
}

/* ── Divisor ───────────────────────────────────────────────────────────── */
.divider { border: none; border-top: 1px solid rgba(255,255,255,0.04); margin: 2.2rem 0; }

/* ── Caixas de alerta ──────────────────────────────────────────────────── */
.info-box {
    background: rgba(6,182,212,0.04); border: 1px solid rgba(6,182,212,0.13);
    border-radius: 9px; padding: 0.9rem 1.2rem; color: #22D3EE; font-size: 0.81rem; line-height: 1.55;
}
.warn-box {
    background: rgba(245,158,11,0.04); border: 1px solid rgba(245,158,11,0.15);
    border-radius: 9px; padding: 0.9rem 1.2rem; color: #FCD34D; font-size: 0.81rem; line-height: 1.55;
}

/* ── Rodapé ────────────────────────────────────────────────────────────── */
.dash-footer {
    text-align: center; color: #1A2A3F; font-size: 0.7rem;
    padding: 1.5rem 0 0.5rem; border-top: 1px solid rgba(255,255,255,0.03);
    margin-top: 1rem; letter-spacing: 0.03em;
}

/* ── Dropdowns / selects ───────────────────────────────────────────────── */
div[data-baseweb="select"] > div {
    background: #0B1422 !important; border-color: rgba(255,255,255,0.08) !important;
    border-radius: 8px !important; font-size: 0.8rem !important;
}
div[data-baseweb="select"] > div:focus-within {
    border-color: rgba(99,102,241,0.4) !important;
    box-shadow: 0 0 0 2px rgba(99,102,241,0.1) !important;
}
div[data-baseweb="menu"] {
    background: #0F172A !important; border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 10px !important;
}
div[role="option"] { font-size: 0.79rem !important; }

/* ── Expander ──────────────────────────────────────────────────────────── */
[data-testid="stExpander"] {
    border: 1px solid rgba(255,255,255,0.05) !important;
    border-radius: 11px !important; background: #070E1A !important;
}
[data-testid="stExpander"] summary {
    font-size: 0.82rem !important; font-weight: 500 !important;
    color: #3D526B !important; padding: 0.9rem 1.1rem !important;
}

/* ── Scrollbar ─────────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.06); border-radius: 10px; }
::-webkit-scrollbar-thumb:hover { background: rgba(99,102,241,0.35); }

/* ── Dataframe ─────────────────────────────────────────────────────────── */
[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }

/* ── Spinner ───────────────────────────────────────────────────────────── */
.stSpinner > div { border-top-color: #6366F1 !important; }
</style>
"""

# ─────────────────────────────────────────────────────────────────────────────
# PLOTLY THEME
# ─────────────────────────────────────────────────────────────────────────────

_F = "Inter, system-ui, sans-serif"
_M = "JetBrains Mono, monospace"

PBASE = dict(
    font=dict(family=_F, color="#3D526B", size=11),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=8, r=8, t=36, b=8),
    colorway=["#6366F1","#8B5CF6","#06B6D4","#10B981","#F59E0B","#EF4444","#EC4899","#0EA5E9"],
    xaxis=dict(
        gridcolor="rgba(255,255,255,0.03)", linecolor="rgba(255,255,255,0.05)",
        tickfont=dict(color="#2D3F57", size=10, family=_M),
        title_font=dict(color="#3D526B", size=11), zeroline=False,
    ),
    yaxis=dict(
        gridcolor="rgba(255,255,255,0.03)", linecolor="rgba(255,255,255,0.05)",
        tickfont=dict(color="#2D3F57", size=10, family=_M),
        title_font=dict(color="#3D526B", size=11), zeroline=False,
    ),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#3D526B", size=10), itemsizing="constant"),
    hoverlabel=dict(
        bgcolor="#0F172A", bordercolor="rgba(255,255,255,0.08)",
        font_color="#CBD5E1", font_size=11, font_family=_F,
    ),
)

PAL = ["#6366F1","#8B5CF6","#06B6D4","#10B981","#F59E0B","#EF4444","#EC4899","#0EA5E9"]


def _rgb(h: str) -> str:
    h = h.lstrip("#")
    return f"{int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)}"


def _cfg() -> dict:
    return {
        "displayModeBar": True, "displaylogo": False,
        "modeBarButtonsToRemove": [
            "zoom2d","pan2d","select2d","lasso2d","zoomIn2d","zoomOut2d",
            "autoScale2d","resetScale2d","hoverClosestCartesian",
            "hoverCompareCartesian","toggleSpikelines",
        ],
        "toImageButtonOptions": {"format": "png", "scale": 2},
    }


# ─────────────────────────────────────────────────────────────────────────────
# GRÁFICOS
# ─────────────────────────────────────────────────────────────────────────────

def _empty(msg="Sem dados suficientes para exibir o gráfico.") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=msg, x=0.5, y=0.5, showarrow=False,
        font=dict(color="#2D3F57", size=12, family=_F),
        xref="paper", yref="paper", align="center",
    )
    fig.update_layout(**PBASE, height=260)
    return fig


def plot_bar(df: pd.DataFrame, col: str) -> go.Figure:
    counts = df[col].value_counts().reset_index()
    counts.columns = [col, "n"]
    counts = counts.sort_values("n", ascending=True).tail(25)
    if counts.empty:
        return _empty()
 
    max_n = int(counts["n"].max())
    denom = max_n if max_n > 0 else 1
    colors = [f"rgba(99,102,241,{0.3 + 0.7 * (v / denom):.2f})" for v in counts["n"]]
    bar_height = max(260, len(counts) * 44)
 
    fig = go.Figure(go.Bar(
        x=counts["n"],
        y=counts[col].astype(str),
        orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=counts["n"],
        textposition="outside",
        textfont=dict(color="#2D3F57", size=10, family=_M),
        hovertemplate="<b>%{y}</b><br>Contagem: %{x}<extra></extra>",
    ))
 
    # Aplicar PBASE sem xaxis/yaxis para evitar conflito de chaves duplicadas
    base = {k: v for k, v in PBASE.items() if k not in ("xaxis", "yaxis")}
    fig.update_layout(**base, height=bar_height, bargap=0.3)
 
    # Aplicar eixos separadamente via update_xaxes / update_yaxes
    fig.update_xaxes(
        **PBASE["xaxis"],
        showgrid=False,
        range=[0, max(max_n * 1.3, 2)],
    )
    fig.update_yaxes(
        **PBASE["yaxis"],
        categoryorder="total ascending",
    )
    return fig


def plot_pie(df: pd.DataFrame, col: str) -> go.Figure:
    counts = df[col].value_counts().reset_index()
    counts.columns = [col, "n"]
    if counts.empty:
        return _empty()

    # Com apenas 1 categoria não há proporção — mostrar mensagem informativa
    if len(counts) < 2:
        return _empty(
            f"Apenas uma categoria: <b>{counts.iloc[0][col]}</b><br>"
            f"({int(counts.iloc[0]['n'])} registro(s)). "
            "São necessárias pelo menos 2 categorias para o gráfico de proporção."
        )

    if len(counts) > 8:
        top    = counts.head(7).copy()
        outros = pd.DataFrame([{col: "Outros", "n": counts.iloc[7:]["n"].sum()}])
        top    = pd.concat([top, outros], ignore_index=True)
    else:
        top = counts.copy()

    total = int(top["n"].sum())

    # Com 2–3 categorias, labels externas se sobrepõem — usar "inside" ou "none"
    n_cats    = len(top)
    text_pos  = "inside" if n_cats <= 3 else "outside"
    text_info = "percent+label" if n_cats <= 3 else "label"

    fig = go.Figure(go.Pie(
        labels=top[col].astype(str), values=top["n"],
        hole=0.54,
        marker=dict(colors=PAL[:n_cats], line=dict(color="#070E1A", width=2)),
        textposition=text_pos,
        textinfo=text_info,
        insidetextorientation="radial",
        textfont=dict(size=10, color="#E2E8F0" if text_pos == "inside" else "#3D526B", family=_F),
        hovertemplate="<b>%{label}</b><br>%{value} resposta(s) (%{percent})<extra></extra>",
        sort=False,
        automargin=True,
    ))
    fig.add_annotation(
        text=f"<b>{total}</b><br><span style='font-size:9px;color:#2D3F57'>total</span>",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=14, color="#94A3B8", family=_F), align="center",
    )
    fig.update_layout(**PBASE, height=340,
                      legend=dict(**PBASE["legend"], orientation="v", x=1.02, y=0.5))
    return fig


def plot_histogram(df: pd.DataFrame, col: str) -> go.Figure:
    s = df[col].dropna()
    if s.empty:
        return _empty()

    n = len(s)

    # Com 1 único valor numérico, histograma não faz sentido — mostrar card simples
    if n == 1 or s.nunique() == 1:
        val = s.iloc[0]
        return _empty(f"Apenas 1 valor registrado: <b>{val}</b><br>São necessários mais registros para o histograma.")

    # Regra de Sturges com mínimo 2 e máximo 40 bins
    nbins = max(2, min(40, int(np.ceil(np.log2(n) + 1))))

    fig = go.Figure(go.Histogram(
        x=s, nbinsx=nbins,
        marker=dict(color="rgba(99,102,241,0.7)", line=dict(color="#070E1A", width=0.8)),
        hovertemplate="Intervalo: %{x}<br>Frequência: %{y}<extra></extra>",
    ))
    fig.update_layout(**PBASE, height=300, bargap=0.04,
                      xaxis_title=col, yaxis_title="Frequência")
    return fig


def plot_boxplot(df: pd.DataFrame, col: str, group_col: str = None) -> go.Figure:
    base = {k: v for k, v in PBASE.items() if k not in ("xaxis", "yaxis")}
 
    if group_col and group_col != col:
        groups = df[group_col].value_counts().head(8).index.tolist()
        if not groups:
            return _empty()
 
        fig = go.Figure()
        traces_added = 0
        for i, grp in enumerate(groups):
            sub = df[df[group_col] == grp][col].dropna()
            if sub.empty:
                continue
            c   = PAL[i % len(PAL)]
            pts = "all" if len(sub) <= 5 else "outliers"
            fig.add_trace(go.Box(
                y=sub, name=str(grp),
                marker=dict(color=c, size=5, opacity=0.75),
                line=dict(color=c, width=1.5),
                fillcolor=f"rgba({_rgb(c)},0.12)",
                boxpoints=pts, jitter=0.4,
                hovertemplate=f"<b>{grp}</b><br>%{{y}}<extra></extra>",
            ))
            traces_added += 1
 
        if traces_added == 0:
            return _empty()
 
        fig.update_layout(**base, height=340)
        fig.update_xaxes(**PBASE["xaxis"])
        fig.update_yaxes(**PBASE["yaxis"], title_text=col)
        return fig
 
    else:
        s = df[col].dropna()
        if s.empty:
            return _empty()
 
        n   = len(s)
        pts = "all" if n <= 10 else "outliers"
 
        # Com apenas 1 ponto, Plotly não desenha caixa — usar scatter
        if n == 1:
            fig = go.Figure(go.Scatter(
                x=[col], y=[s.iloc[0]],
                mode="markers",
                marker=dict(color="#6366F1", size=12, symbol="circle"),
                hovertemplate=f"{col}: %{{y}}<extra></extra>",
            ))
            fig.update_layout(**base, height=300)
            fig.update_xaxes(**PBASE["xaxis"], showticklabels=False)
            fig.update_yaxes(**PBASE["yaxis"], title_text=col)
            return fig
 
        fig = go.Figure(go.Box(
            y=s,
            marker=dict(color="#6366F1", size=5, opacity=0.7),
            line=dict(color="#6366F1", width=1.5),
            fillcolor="rgba(99,102,241,0.1)",
            boxpoints=pts, jitter=0.35, name=col, showlegend=False,
            hovertemplate="%{y}<extra></extra>",
        ))
        fig.update_layout(**base, height=300)
        fig.update_xaxes(**PBASE["xaxis"])
        fig.update_yaxes(**PBASE["yaxis"], title_text=col)
        return fig
 


def plot_grouped_bar(df: pd.DataFrame, cat_col: str, num_col: str) -> go.Figure:
    agg = (
        df.groupby(cat_col, observed=True)[num_col]
        .agg(["mean", "count"]).reset_index()
        .rename(columns={"mean": "média", "count": "n"})
        .sort_values("média", ascending=False)
        .head(15)
    )
    if agg.empty or agg["n"].sum() == 0:
        return _empty("Sem dados suficientes para o agrupamento.")
 
    max_v = float(agg["média"].max())
    min_v = float(agg["média"].min())
    span  = max_v - min_v
    denom = span if span > 0 else 1
    colors = [
        f"rgba(99,102,241,{0.3 + 0.7 * max(0.0, min(1.0, (v - min_v) / denom)):.2f})"
        for v in agg["média"]
    ]
    text_pos = "outside" if len(agg) > 1 and max_v > 0 else "auto"
 
    fig = go.Figure(go.Bar(
        x=agg[cat_col].astype(str),
        y=agg["média"],
        marker=dict(color=colors, line=dict(width=0)),
        text=agg["média"].round(2),
        textposition=text_pos,
        textfont=dict(color="#2D3F57", size=9, family=_M),
        customdata=agg["n"],
        hovertemplate="<b>%{x}</b><br>Média: %{y:.2f}<br>N: %{customdata}<extra></extra>",
    ))
 
    y_max = max_v if max_v > 0 else 1
    y_min = min(0.0, min_v)
 
    # Aplicar PBASE sem xaxis/yaxis para evitar conflito de chaves duplicadas
    base = {k: v for k, v in PBASE.items() if k not in ("xaxis", "yaxis")}
    fig.update_layout(**base, height=320, bargap=0.28)
 
    # Aplicar eixos separadamente via update_xaxes / update_yaxes
    fig.update_xaxes(**PBASE["xaxis"], title_text=cat_col)
    fig.update_yaxes(**PBASE["yaxis"], title_text=f"Média de {num_col}", range=[y_min, y_max * 1.28])
 
    return fig


def plot_heatmap(df: pd.DataFrame, num_cols: list):
    if len(num_cols) < 2:
        return None

    # Remover colunas com variância zero (constantes) — corr() gera NaN
    valid = [c for c in num_cols if df[c].dropna().nunique() > 1]
    if len(valid) < 2:
        return None

    # Precisa de pelo menos 2 linhas para calcular correlação
    if len(df.dropna(subset=valid)) < 2:
        return None

    corr = df[valid].corr().round(3)

    # Substituir NaN restantes por 0 para não quebrar o heatmap
    corr = corr.fillna(0)

    fig = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns.tolist(), y=corr.index.tolist(),
        colorscale=[[0, "#0E7490"], [0.5, "#070E1A"], [1, "#4F46E5"]],
        zmin=-1, zmax=1,
        text=corr.values, texttemplate="%{text:.2f}",
        textfont=dict(size=10, color="#94A3B8", family=_M),
        hovertemplate="<b>%{y} × %{x}</b><br>r = %{z:.3f}<extra></extra>",
        showscale=True,
        colorbar=dict(tickfont=dict(color="#2D3F57", size=9, family=_M), outlinewidth=0, thickness=10),
    ))
    fig.update_layout(**PBASE, height=max(300, len(valid) * 54))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# COMPONENTES
# ─────────────────────────────────────────────────────────────────────────────

def section_header(title: str, sub: str = ""):
    st.markdown(
        f'<div class="sec-wrap"><p class="sec-title">{title}</p>'
        + (f'<p class="sec-sub">{sub}</p>' if sub else "")
        + "</div>",
        unsafe_allow_html=True,
    )


def chart_header(eyebrow: str, title: str, sub: str = ""):
    st.markdown(
        f'<div class="chart-eyebrow">{eyebrow}</div>'
        f'<p class="chart-title">{title}</p>'
        + (f'<p class="chart-sub">{sub}</p>' if sub else ""),
        unsafe_allow_html=True,
    )


def type_pill(vtype: str) -> str:
    c  = get_type_color(vtype)
    lb = get_type_label(vtype)
    r  = _rgb(c)
    return (f'<span class="type-pill" style="background:rgba({r},0.09);'
            f'color:{c};border:1px solid rgba({r},0.22);">{lb}</span>')


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

def build_sidebar(df: pd.DataFrame, classification: dict) -> pd.DataFrame:
    with st.sidebar:
        st.markdown("""
        <div class="sb-head">
            <div class="sb-head-eyebrow">Painel de Controle</div>
            <h2 class="sb-head-title">Filtros e Seleções</h2>
            <p class="sb-head-sub">Refine os dados exibidos em todas as visualizações e na tabela.</p>
        </div>
        <div class="sb-body">
        """, unsafe_allow_html=True)

        df_f = df.copy()
        cat_cols = [
            c for c, t in classification.items()
            if t in ("categorical_nominal","categorical_ordinal","numeric_discrete")
            and 2 <= df[c].nunique() <= 20
        ]
        num_cols = [
            c for c, t in classification.items()
            if t in ("numeric_continuous","numeric_discrete") and df[c].nunique() > 1
        ]
        active = []

        # Categóricas
        if cat_cols:
            st.markdown("""
            <div class="sb-section">
                <span class="sb-dot" style="background:#6366F1;"></span>
                <span class="sb-section-text">Variáveis Categóricas</span>
            </div>
            """, unsafe_allow_html=True)
            for col in cat_cols[:5]:
                opts = sorted(df[col].dropna().unique().tolist(), key=str)
                sel  = st.multiselect(
                    col, options=opts, default=[],
                    key=f"fc_{col}", placeholder=f"Todas ({len(opts)})",
                    help=f"{len(opts)} categorias disponíveis",
                )
                if sel:
                    df_f   = df_f[df_f[col].isin(sel)]
                    active.append(col)

        # Numéricas
        if num_cols:
            st.markdown("""
            <div class="sb-section">
                <span class="sb-dot" style="background:#06B6D4;"></span>
                <span class="sb-section-text">Variáveis Numéricas</span>
            </div>
            """, unsafe_allow_html=True)
            for col in num_cols[:3]:
                cd   = df[col].dropna()
                if cd.empty: continue
                cmin = float(cd.min()); cmax = float(cd.max())
                if cmin >= cmax: continue
                st.markdown(f'<span class="sb-sl-label">{col}</span>', unsafe_allow_html=True)
                rng = st.slider(col, min_value=cmin, max_value=cmax,
                                value=(cmin, cmax), key=f"fn_{col}",
                                label_visibility="collapsed", format="%.3g")
                if rng != (cmin, cmax):
                    df_f   = df_f[df_f[col].between(rng[0], rng[1])]
                    active.append(col)

        # Contador
        total    = len(df)
        filtered = len(df_f)
        pct      = filtered / max(total, 1) * 100
        st.markdown(f"""
        <div class="sb-counter">
            <div class="sb-counter-lbl">Registros selecionados</div>
            <div class="sb-counter-row">
                <span class="sb-val">{filtered:,}</span>
                <span class="sb-of">de {total:,}</span>
                <span class="sb-pct">{pct:.1f}%</span>
            </div>
            <div class="sb-track"><div class="sb-fill" style="width:{pct:.2f}%;"></div></div>
        </div>
        """, unsafe_allow_html=True)

        if active:
            if st.button("↺  Limpar todos os filtros", use_container_width=True):
                st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)
    return df_f


# ─────────────────────────────────────────────────────────────────────────────
# SEÇÃO: VISUALIZAÇÕES
# ─────────────────────────────────────────────────────────────────────────────

def render_visualizations(df: pd.DataFrame, classification: dict):
    # Categóricas: aceitar mesmo com 1 valor único (o gráfico de barras funciona; pizza mostrará aviso)
    cat_cols = [
        c for c, t in classification.items()
        if t in ("categorical_nominal", "categorical_ordinal")
        and df[c].dropna().size > 0
    ]
    num_cols = [
        c for c, t in classification.items()
        if t in ("numeric_continuous", "numeric_discrete")
        and not df[c].dropna().empty
    ]

    tabs = st.tabs(["  Categóricas  ","  Numéricas  ","  Comparações  ","  Correlação  "])

    # TAB 1
    with tabs[0]:
        if not cat_cols:
            st.markdown('<div class="info-box">Nenhuma variável categórica encontrada.</div>', unsafe_allow_html=True)
        else:
            sel = st.selectbox("Variável", cat_cols, key="v_cat", label_visibility="collapsed")
            l, r = st.columns(2, gap="medium")
            with l:
                st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
                chart_header("Frequência", f"Distribuição de {sel}", "Contagem absoluta por categoria")
                st.plotly_chart(plot_bar(df, sel), use_container_width=True, config=_cfg())
                st.markdown("</div>", unsafe_allow_html=True)
            with r:
                st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
                chart_header("Proporção", f"Composição de {sel}", "Participação relativa de cada categoria")
                st.plotly_chart(plot_pie(df, sel), use_container_width=True, config=_cfg())
                st.markdown("</div>", unsafe_allow_html=True)

    # TAB 2
    with tabs[1]:
        if not num_cols:
            st.markdown('<div class="info-box">Nenhuma variável numérica encontrada.</div>', unsafe_allow_html=True)
        else:
            sel = st.selectbox("Variável", num_cols, key="v_num", label_visibility="collapsed")
            l, r = st.columns(2, gap="medium")
            with l:
                st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
                chart_header("Distribuição", f"Histograma — {sel}", "Frequência por intervalo de valores")
                st.plotly_chart(plot_histogram(df, sel), use_container_width=True, config=_cfg())
                st.markdown("</div>", unsafe_allow_html=True)
            with r:
                st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
                chart_header("Dispersão", f"Boxplot — {sel}", "Mediana, quartis e valores atípicos")
                st.plotly_chart(plot_boxplot(df, sel), use_container_width=True, config=_cfg())
                st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            section_header("Estatísticas Descritivas")
            desc   = df[sel].describe().round(4)
            labels = {"count":"Registros","mean":"Média","std":"Desvio-Padrão",
                      "min":"Mínimo","25%":"Q1","50%":"Mediana","75%":"Q3","max":"Máximo"}
            sc = st.columns(len(desc))
            for i, (stat, val) in enumerate(desc.items()):
                with sc[i]:
                    st.markdown(
                        f'<div class="kpi-card" style="padding:.85rem 1rem;">'
                        f'<div class="kpi-lbl">{labels.get(stat,stat)}</div>'
                        f'<div class="kpi-val" style="font-size:1.1rem;color:#818CF8;">{val:,.3g}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

    # TAB 3
    with tabs[2]:
        if not cat_cols or not num_cols:
            st.markdown('<div class="info-box">São necessárias pelo menos uma variável categórica e uma numérica para gerar comparações.</div>', unsafe_allow_html=True)
        else:
            c1, c2 = st.columns(2, gap="medium")
            with c1: grp = st.selectbox("Agrupamento (categórica)", cat_cols, key="cmp_c")
            with c2: val = st.selectbox("Valor (numérica)", num_cols, key="cmp_n")
            l, r = st.columns(2, gap="medium")
            with l:
                st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
                chart_header("Médias", f"{val} por {grp}", "Valor médio agrupado por categoria")
                st.plotly_chart(plot_grouped_bar(df, grp, val), use_container_width=True, config=_cfg())
                st.markdown("</div>", unsafe_allow_html=True)
            with r:
                st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
                chart_header("Dispersão", f"{val} por {grp}", "Distribuição e variabilidade por grupo")
                st.plotly_chart(plot_boxplot(df, val, grp), use_container_width=True, config=_cfg())
                st.markdown("</div>", unsafe_allow_html=True)

    # TAB 4
    with tabs[3]:
        # Filtrar apenas colunas com variância real (>1 valor único) e ≥2 linhas válidas
        valid_num = [c for c in num_cols if df[c].dropna().nunique() > 1]
        if len(valid_num) < 2:
            st.markdown(
                '<div class="info-box">São necessárias pelo menos 2 variáveis numéricas '
                'com valores distintos para calcular correlações.</div>',
                unsafe_allow_html=True,
            )
        elif len(df.dropna(subset=valid_num)) < 2:
            st.markdown(
                '<div class="info-box">São necessários pelo menos 2 registros completos '
                'para calcular correlações.</div>',
                unsafe_allow_html=True,
            )
        else:
            hm = plot_heatmap(df, valid_num)
            if hm:
                st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
                chart_header("Pearson", "Mapa de Correlação", "Coeficiente de Pearson entre variáveis numéricas (−1 a +1)")
                st.plotly_chart(hm, use_container_width=True, config=_cfg())
                st.markdown("</div>", unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
                section_header("Pares Ordenados por Força de Correlação")
                corr_m = df[valid_num].corr().fillna(0)
                pairs  = [
                    {"Variável A": valid_num[i], "Variável B": valid_num[j],
                     "Correlação": round(corr_m.iloc[i, j], 4)}
                    for i in range(len(valid_num)) for j in range(i+1, len(valid_num))
                ]
                if pairs:
                    st.dataframe(
                        pd.DataFrame(pairs).sort_values("Correlação", key=abs, ascending=False),
                        use_container_width=True, hide_index=True,
                    )


# ─────────────────────────────────────────────────────────────────────────────
# SEÇÃO: INSIGHTS
# ─────────────────────────────────────────────────────────────────────────────

def render_insights(df: pd.DataFrame, classification: dict):
    insights = generate_insights(df, classification)
    if not insights:
        st.markdown('<div class="info-box">Não foi possível gerar insights automáticos com os dados atuais.</div>', unsafe_allow_html=True)
        return

    for row_s in range(0, len(insights), 3):
        row  = insights[row_s:row_s+3]
        cols = st.columns(3, gap="small")
        for i, ins in enumerate(row):
            ac = "#6366F1" if ins["type"] == "numeric" else "#06B6D4"
            with cols[i]:
                st.markdown(
                    f'<div class="ins-card">'
                    f'<div class="ins-accent" style="background:{ac};"></div>'
                    f'<div class="ins-lbl">{ins["title"]}</div>'
                    f'<div class="ins-val">{ins["value"]}</div>'
                    f'<div class="ins-detail">{ins["detail"]}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )


# ─────────────────────────────────────────────────────────────────────────────
# SEÇÃO: METODOLOGIA
# ─────────────────────────────────────────────────────────────────────────────

def render_methodology(df: pd.DataFrame, classification: dict):
    rows = ""
    for col, vt in classification.items():
        nu = df[col].nunique()
        cp = round((1 - df[col].isnull().sum() / max(len(df),1)) * 100, 1)
        rows += (
            f"<tr><td><code>{col}</code></td><td>{type_pill(vt)}</td>"
            f"<td style='text-align:center;font-family:\"JetBrains Mono\",monospace;font-size:.75rem;color:#3D526B;'>{nu}</td>"
            f"<td style='text-align:center;font-family:\"JetBrains Mono\",monospace;font-size:.75rem;color:#3D526B;'>{cp}%</td></tr>"
        )
    st.markdown(
        f'<table class="meth-table"><thead><tr>'
        f'<th>Variável</th><th>Tipo Classificado</th>'
        f'<th style="text-align:center;">Valores Únicos</th>'
        f'<th style="text-align:center;">Completude</th>'
        f'</tr></thead><tbody>{rows}</tbody></table>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    with st.spinner("Conectando à planilha e carregando dados..."):
        raw_df = fetch_data()

    if raw_df.empty:
        st.markdown('<div class="warn-box"><strong>Atenção:</strong> Não foi possível carregar os dados da planilha. Veja o diagnóstico abaixo.</div>', unsafe_allow_html=True)
        with st.expander("Diagnóstico da conexão", expanded=True):
            from utils import CSV_EXPORT_URL, SHEET_ID, SHEET_GID
            st.markdown("**URL de exportação CSV:**")
            st.code(CSV_EXPORT_URL)
            st.markdown("""
            **Possíveis causas:**
            - Planilha não compartilhada → *Compartilhar → Qualquer pessoa com o link → Leitor*
            - `gid` incorreto → copie o número após `gid=` na URL da planilha
            - Sem acesso à internet ou rede bloqueando `docs.google.com`
            """)
            st.caption(f"SHEET_ID = {SHEET_ID} · SHEET_GID = {SHEET_GID}")
        st.stop()

    df             = clean_dataframe(raw_df)
    classification = classify_columns(df)
    kpis           = compute_kpis(df, classification)
    df_f           = build_sidebar(df, classification)

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="dash-header">
        <div class="dash-eyebrow">
            <span class="dash-pulse"></span>
            Trabalho de Conclusão de Curso
        </div>
        <h1 class="dash-title">Dashboard de Análise de Dados</h1>
        <p class="dash-desc">
            Plataforma de análise exploratória estruturada sobre dados coletados via Google Forms.
            Os dados são carregados em tempo real a partir do Google Sheets e processados com
            classificação automática de variáveis, visualizações interativas e geração de
            insights estatísticos.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── KPIs ──────────────────────────────────────────────────────────────────
    section_header("Visão Geral", "Métricas extraídas do conjunto de dados carregado")
    cards  = [
        ("Registros Totais", f"{kpis['total_records']:,}",  "respostas coletadas",   "#6366F1"),
        ("Variáveis",        f"{kpis['total_columns']}",     "campos analisados",     "#8B5CF6"),
        ("Completude",       f"{kpis['completeness']}%",     "dados preenchidos",     "#10B981"),
        ("Var. Numéricas",   f"{kpis['numeric_cols']}",      "contínuas + discretas", "#06B6D4"),
        ("Var. Categóricas", f"{kpis['categorical_cols']}",  "nominais + ordinais",   "#F59E0B"),
    ]
    for col_w, (lbl, val, sub, ac) in zip(st.columns(5, gap="small"), cards):
        with col_w:
            st.markdown(
                f'<div class="kpi-card" style="border-bottom:2px solid rgba({_rgb(ac)},0.25);">'
                f'<div class="kpi-lbl">{lbl}</div>'
                f'<div class="kpi-val" style="color:{ac};">{val}</div>'
                f'<div class="kpi-sub">{sub}</div></div>',
                unsafe_allow_html=True,
            )

    if len(df_f) < len(df):
        st.markdown("<br>", unsafe_allow_html=True)
        n_ocultos = len(df) - len(df_f)
        st.markdown(
            f'<div class="af-bar">'
            f'<span class="af-lbl">Filtros ativos</span>'
            f'<span class="af-pill">{len(df_f):,} de {len(df):,} registros</span>'
            f'<span class="af-pill">{n_ocultos:,} registros ocultados</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Metodologia ───────────────────────────────────────────────────────────
    with st.expander("Metodologia — Classificação de Variáveis", expanded=False):
        section_header("Classificação Automática de Variáveis",
                        "Cada variável foi classificada com base no tipo de dado e no número de valores únicos.")
        render_methodology(df, classification)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
            <strong>Critérios de classificação:</strong>
            Numérica Contínua (&gt; 20 valores únicos) ·
            Numérica Discreta (≤ 20 valores únicos) ·
            Categórica Nominal (texto com até 30 categorias, sem ordem definida) ·
            Categórica Ordinal (detectada por palavras-chave e padrões de escala Likert) ·
            Texto Livre (média superior a 4 palavras e alta cardinalidade) ·
            Data (reconhecimento automático de formatos de data).
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ── Visualizações ─────────────────────────────────────────────────────────
    section_header("Análise Exploratória",
                    "Visualizações interativas segmentadas por tipo de variável — selecione a aba desejada")
    render_visualizations(df_f, classification)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ── Insights ──────────────────────────────────────────────────────────────
    section_header("Insights Automáticos",
                    "Padrões e estatísticas relevantes identificados no conjunto de dados filtrado")
    render_insights(df_f, classification)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ── Tabela ────────────────────────────────────────────────────────────────
    section_header("Tabela de Dados",
                    f"{len(df_f):,} linhas × {len(df_f.columns)} colunas — dados filtrados")
    sc, dc = st.columns([4,1], gap="small")
    with sc:
        search = st.text_input("Busca", placeholder="Buscar em todos os campos...",
                                key="tbl_s", label_visibility="collapsed")
    with dc:
        st.download_button("Exportar CSV",
                           data=df_f.to_csv(index=False).encode("utf-8"),
                           file_name="dados_filtrados.csv", mime="text/csv",
                           use_container_width=True)

    display = df_f
    if search:
        mask    = display.astype(str).apply(lambda c: c.str.contains(search, case=False, na=False)).any(axis=1)
        display = display[mask]
        if display.empty:
            st.markdown(f'<div class="info-box">Nenhum resultado encontrado para "<strong>{search}</strong>".</div>', unsafe_allow_html=True)

    st.dataframe(display, use_container_width=True, height=420)

    # ── Rodapé ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="dash-footer">
        Dashboard desenvolvido com Streamlit &amp; Plotly &nbsp;·&nbsp;
        Dados coletados via Google Forms e Google Sheets &nbsp;·&nbsp;
        Trabalho de Conclusão de Curso
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()