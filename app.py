"""
app.py — Dashboard principal para análise de dados do TCC.
Tecnologias: Streamlit, Plotly, Pandas.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils import (
    fetch_data,
    clean_dataframe,
    classify_columns,
    compute_kpis,
    generate_insights,
    get_type_label,
    get_type_color,
)

# ---------------------------------------------------------------------------
# CONFIGURAÇÃO DA PÁGINA
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Dashboard TCC",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CSS CUSTOMIZADO
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
<style>
    /* Importar fonte */
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

    /* Reset e base */
    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }

    /* Remove padding padrão do Streamlit */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        padding-left: 2rem;
        padding-right: 2rem;
        max-width: 1400px;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #1E293B;
        border-right: 1px solid #334155;
    }
    [data-testid="stSidebar"] .block-container {
        padding: 1.5rem 1rem;
    }

    /* Header da sidebar */
    .sidebar-header {
        padding: 0.75rem 0;
        border-bottom: 1px solid #334155;
        margin-bottom: 1.5rem;
    }
    .sidebar-header h2 {
        color: #6366F1;
        font-size: 0.9rem;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin: 0;
    }

    /* Seção label na sidebar */
    .sidebar-section-label {
        color: #64748B;
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin: 1.25rem 0 0.5rem 0;
    }

    /* Cabeçalho principal */
    .main-header {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 2rem 2.5rem;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    }
    .main-header::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, #6366F1, #8B5CF6, #06B6D4);
    }
    .main-header h1 {
        color: #F1F5F9;
        font-size: 1.75rem;
        font-weight: 700;
        margin: 0 0 0.5rem 0;
        letter-spacing: -0.02em;
    }
    .main-header p {
        color: #94A3B8;
        font-size: 0.95rem;
        margin: 0;
        line-height: 1.6;
    }
    .header-badge {
        display: inline-block;
        background: rgba(99, 102, 241, 0.15);
        border: 1px solid rgba(99, 102, 241, 0.3);
        color: #6366F1;
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        padding: 0.25rem 0.75rem;
        border-radius: 100px;
        margin-bottom: 1rem;
    }

    /* KPI Cards */
    .kpi-card {
        background: #1E293B;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 1.25rem 1.5rem;
        position: relative;
        overflow: hidden;
        transition: border-color 0.2s;
    }
    .kpi-card:hover {
        border-color: #6366F1;
    }
    .kpi-card .kpi-label {
        color: #64748B;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }
    .kpi-card .kpi-value {
        color: #F1F5F9;
        font-size: 2rem;
        font-weight: 700;
        line-height: 1;
        font-family: 'IBM Plex Mono', monospace;
    }
    .kpi-card .kpi-sub {
        color: #64748B;
        font-size: 0.78rem;
        margin-top: 0.4rem;
    }
    .kpi-accent {
        position: absolute;
        bottom: 0; left: 0;
        height: 3px;
        width: 100%;
    }

    /* Section titles */
    .section-title {
        color: #E2E8F0;
        font-size: 1.05rem;
        font-weight: 600;
        margin: 0 0 0.25rem 0;
        letter-spacing: -0.01em;
    }
    .section-subtitle {
        color: #64748B;
        font-size: 0.82rem;
        margin: 0 0 1.25rem 0;
    }
    .section-divider {
        border: none;
        border-top: 1px solid #1E293B;
        margin: 2rem 0;
    }

    /* Tabela de metodologia */
    .methodology-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.85rem;
    }
    .methodology-table th {
        background: #0F172A;
        color: #64748B;
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        padding: 0.65rem 1rem;
        text-align: left;
        border-bottom: 1px solid #334155;
    }
    .methodology-table td {
        padding: 0.6rem 1rem;
        color: #CBD5E1;
        border-bottom: 1px solid #1E293B;
        vertical-align: middle;
    }
    .methodology-table tr:hover td {
        background: rgba(99, 102, 241, 0.04);
    }
    .type-badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 4px;
        font-size: 0.72rem;
        font-weight: 500;
        font-family: 'IBM Plex Mono', monospace;
    }

    /* Insight cards */
    .insight-card {
        background: #1E293B;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 1rem 1.25rem;
        margin-bottom: 0.75rem;
    }
    .insight-card .insight-title {
        color: #94A3B8;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.3rem;
    }
    .insight-card .insight-value {
        color: #F1F5F9;
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 0.2rem;
    }
    .insight-card .insight-detail {
        color: #64748B;
        font-size: 0.78rem;
    }

    /* Chart containers */
    .chart-container {
        background: #1E293B;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 1.25rem;
        margin-bottom: 1.25rem;
    }
    .chart-title {
        color: #CBD5E1;
        font-size: 0.88rem;
        font-weight: 600;
        margin-bottom: 0.15rem;
    }
    .chart-subtitle {
        color: #64748B;
        font-size: 0.75rem;
        margin-bottom: 1rem;
    }

    /* Streamlit widget overrides */
    .stSelectbox label, .stMultiSelect label {
        color: #94A3B8 !important;
        font-size: 0.78rem !important;
        font-weight: 500 !important;
    }
    div[data-baseweb="select"] > div {
        background-color: #0F172A !important;
        border-color: #334155 !important;
    }
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #0F172A;
        gap: 0;
        border-bottom: 1px solid #334155;
    }
    .stTabs [data-baseweb="tab"] {
        color: #64748B;
        font-size: 0.82rem;
        font-weight: 500;
        padding: 0.6rem 1.25rem;
    }
    .stTabs [aria-selected="true"] {
        color: #6366F1 !important;
        border-bottom: 2px solid #6366F1 !important;
    }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: #0F172A; }
    ::-webkit-scrollbar-thumb { background: #334155; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #6366F1; }

    /* Alert/Info boxes */
    .info-box {
        background: rgba(6, 182, 212, 0.08);
        border: 1px solid rgba(6, 182, 212, 0.25);
        border-radius: 8px;
        padding: 1rem 1.25rem;
        color: #67E8F9;
        font-size: 0.85rem;
        margin-bottom: 1rem;
    }
    .warning-box {
        background: rgba(245, 158, 11, 0.08);
        border: 1px solid rgba(245, 158, 11, 0.25);
        border-radius: 8px;
        padding: 1rem 1.25rem;
        color: #FCD34D;
        font-size: 0.85rem;
        margin-bottom: 1rem;
    }
</style>
"""

# ---------------------------------------------------------------------------
# PLOTLY THEME
# ---------------------------------------------------------------------------

PLOTLY_LAYOUT = dict(
    font=dict(family="IBM Plex Sans, sans-serif", color="#94A3B8", size=12),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=10, r=10, t=40, b=10),
    colorway=["#6366F1", "#8B5CF6", "#06B6D4", "#10B981", "#F59E0B", "#EF4444", "#EC4899"],
    xaxis=dict(
        gridcolor="#1E293B",
        linecolor="#334155",
        tickcolor="#334155",
        tickfont=dict(color="#64748B", size=11),
        title_font=dict(color="#94A3B8"),
    ),
    yaxis=dict(
        gridcolor="#1E293B",
        linecolor="#334155",
        tickcolor="#334155",
        tickfont=dict(color="#64748B", size=11),
        title_font=dict(color="#94A3B8"),
    ),
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94A3B8", size=11),
    ),
    hoverlabel=dict(
        bgcolor="#1E293B",
        bordercolor="#334155",
        font_color="#E2E8F0",
        font_size=12,
    ),
)

COLOR_PALETTE = [
    "#6366F1", "#8B5CF6", "#06B6D4", "#10B981",
    "#F59E0B", "#EF4444", "#EC4899", "#0EA5E9",
]


def apply_theme(fig: go.Figure) -> go.Figure:
    fig.update_layout(**PLOTLY_LAYOUT)
    return fig


# ---------------------------------------------------------------------------
# COMPONENTES DE UI
# ---------------------------------------------------------------------------

def render_kpi_card(label: str, value, sub: str = "", accent_color: str = "#6366F1"):
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            {"<div class='kpi-sub'>" + sub + "</div>" if sub else ""}
            <div class="kpi-accent" style="background:{accent_color};"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section(title: str, subtitle: str = ""):
    st.markdown(f'<p class="section-title">{title}</p>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<p class="section-subtitle">{subtitle}</p>', unsafe_allow_html=True)


def render_chart_header(title: str, subtitle: str = ""):
    st.markdown(
        f'<p class="chart-title">{title}</p>'
        + (f'<p class="chart-subtitle">{subtitle}</p>' if subtitle else ""),
        unsafe_allow_html=True,
    )


def render_type_badge(var_type: str) -> str:
    color = get_type_color(var_type)
    label = get_type_label(var_type)
    return (
        f'<span class="type-badge" '
        f'style="background:rgba({_hex_to_rgb(color)},0.12);'
        f'color:{color};border:1px solid rgba({_hex_to_rgb(color)},0.3);">'
        f'{label}</span>'
    )


def _hex_to_rgb(hex_color: str) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"{r},{g},{b}"


# ---------------------------------------------------------------------------
# GRÁFICOS
# ---------------------------------------------------------------------------

def plot_bar(df: pd.DataFrame, col: str) -> go.Figure:
    counts = df[col].value_counts().reset_index()
    counts.columns = [col, "Contagem"]
    counts = counts.sort_values("Contagem", ascending=True).tail(20)

    fig = go.Figure(
        go.Bar(
            x=counts["Contagem"],
            y=counts[col].astype(str),
            orientation="h",
            marker=dict(
                color=counts["Contagem"],
                colorscale=[[0, "#334155"], [1, "#6366F1"]],
                showscale=False,
            ),
            text=counts["Contagem"],
            textposition="outside",
        )
    )

    layout = PLOTLY_LAYOUT.copy()
    layout["yaxis"] = {
        **layout.get("yaxis", {}),
        "categoryorder": "total ascending"
    }

    fig.update_layout(
        **layout,
        height=max(300, len(counts) * 36),
    )

    return fig


def plot_pie(df: pd.DataFrame, col: str) -> go.Figure:
    counts = df[col].value_counts().reset_index()
    counts.columns = [col, "Contagem"]
    top = counts.head(8)
    if len(counts) > 8:
        outros = pd.DataFrame(
            [{col: "Outros", "Contagem": counts.iloc[8:]["Contagem"].sum()}]
        )
        top = pd.concat([top, outros], ignore_index=True)

    fig = go.Figure(
        go.Pie(
            labels=top[col].astype(str),
            values=top["Contagem"],
            hole=0.45,
            marker=dict(colors=COLOR_PALETTE, line=dict(color="#0F172A", width=2)),
            textfont=dict(color="#E2E8F0", size=11),
        )
    )
    fig.update_layout(**PLOTLY_LAYOUT, height=380)
    return fig


def plot_histogram(df: pd.DataFrame, col: str) -> go.Figure:
    series = df[col].dropna()
    nbins = min(40, max(10, int(np.sqrt(len(series)))))

    fig = go.Figure(
        go.Histogram(
            x=series,
            nbinsx=nbins,
            marker=dict(
                color="#6366F1",
                opacity=0.85,
                line=dict(color="#0F172A", width=0.5),
            ),
        )
    )
    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=340,
        bargap=0.05,
        xaxis_title=col,
        yaxis_title="Frequencia",
    )
    return fig


def plot_boxplot(df: pd.DataFrame, col: str, group_col: str = None) -> go.Figure:
    if group_col and group_col != col:
        groups = df[group_col].value_counts().head(8).index.tolist()
        fig = go.Figure()
        for i, grp in enumerate(groups):
            subset = df[df[group_col] == grp][col].dropna()
            fig.add_trace(
                go.Box(
                    y=subset,
                    name=str(grp),
                    marker_color=COLOR_PALETTE[i % len(COLOR_PALETTE)],
                    line_color=COLOR_PALETTE[i % len(COLOR_PALETTE)],
                    fillcolor=f"rgba({_hex_to_rgb(COLOR_PALETTE[i % len(COLOR_PALETTE)])},0.2)",
                )
            )
        fig.update_layout(**PLOTLY_LAYOUT, height=380, yaxis_title=col)
    else:
        series = df[col].dropna()
        fig = go.Figure(
            go.Box(
                y=series,
                marker_color="#6366F1",
                line_color="#6366F1",
                fillcolor="rgba(99,102,241,0.15)",
                boxpoints="outliers",
                pointpos=0,
            )
        )
        fig.update_layout(**PLOTLY_LAYOUT, height=340, yaxis_title=col)
    return fig


def plot_grouped_bar(df: pd.DataFrame, cat_col: str, num_col: str) -> go.Figure:
    agg = (
        df.groupby(cat_col)[num_col]
        .mean()
        .reset_index()
        .sort_values(num_col, ascending=False)
        .head(15)
    )
    fig = go.Figure(
        go.Bar(
            x=agg[cat_col].astype(str),
            y=agg[num_col],
            marker=dict(
                color=agg[num_col],
                colorscale=[[0, "#334155"], [1, "#6366F1"]],
                showscale=False,
                line=dict(color="#0F172A", width=0.5),
            ),
            text=agg[num_col].round(2),
            textposition="outside",
            textfont=dict(color="#94A3B8", size=10),
        )
    )
    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=360,
        xaxis_title=cat_col,
        yaxis_title=f"Media de {num_col}",
    )
    return fig


def plot_correlation_heatmap(df: pd.DataFrame, num_cols: list) -> go.Figure:
    if len(num_cols) < 2:
        return None
    corr = df[num_cols].corr()
    fig = go.Figure(
        go.Heatmap(
            z=corr.values,
            x=corr.columns.tolist(),
            y=corr.index.tolist(),
            colorscale=[
                [0, "#0EA5E9"], [0.5, "#1E293B"], [1, "#6366F1"]
            ],
            zmin=-1, zmax=1,
            text=corr.round(2).values,
            texttemplate="%{text}",
            textfont=dict(size=10, color="#E2E8F0"),
        )
    )
    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=max(300, len(num_cols) * 50),
    )
    return fig


# ---------------------------------------------------------------------------
# SIDEBAR — FILTROS
# ---------------------------------------------------------------------------

def build_sidebar(df: pd.DataFrame, classification: dict) -> pd.DataFrame:
    with st.sidebar:
        st.markdown(
            '<div class="sidebar-header"><h2>Filtros e Controles</h2></div>',
            unsafe_allow_html=True,
        )

        df_filtered = df.copy()
        cat_cols = [
            col for col, t in classification.items()
            if t in ("categorical_nominal", "categorical_ordinal", "numeric_discrete")
            and df[col].nunique() <= 20
        ]
        num_cols = [
            col for col, t in classification.items()
            if t in ("numeric_continuous", "numeric_discrete")
        ]

        if cat_cols:
            st.markdown('<p class="sidebar-section-label">Variaveis Categoricas</p>', unsafe_allow_html=True)
            for col in cat_cols[:4]:
                options = sorted(df[col].dropna().unique().tolist(), key=str)
                selected = st.multiselect(
                    col,
                    options=options,
                    default=[],
                    key=f"filter_{col}",
                )
                if selected:
                    df_filtered = df_filtered[df_filtered[col].isin(selected)]

        if num_cols:
            st.markdown('<p class="sidebar-section-label">Variaveis Numericas</p>', unsafe_allow_html=True)
            for col in num_cols[:3]:
                col_data = df_filtered[col].dropna()
                if col_data.empty:
                    continue
                col_min = float(col_data.min())
                col_max = float(col_data.max())
                if col_min == col_max:
                    continue
                selected_range = st.slider(
                    col,
                    min_value=col_min,
                    max_value=col_max,
                    value=(col_min, col_max),
                    key=f"slider_{col}",
                )
                df_filtered = df_filtered[
                    (df_filtered[col] >= selected_range[0])
                    & (df_filtered[col] <= selected_range[1])
                ]

        st.markdown('<p class="sidebar-section-label">Amostragem</p>', unsafe_allow_html=True)
        total = len(df)
        filtered = len(df_filtered)
        pct = filtered / max(total, 1) * 100
        st.markdown(
            f'<div style="color:#94A3B8;font-size:0.82rem;">'
            f'{filtered} de {total} registros ({pct:.1f}%)</div>',
            unsafe_allow_html=True,
        )

        if st.button("Limpar Filtros", use_container_width=True):
            st.rerun()

    return df_filtered


# ---------------------------------------------------------------------------
# SECAO: VISUALIZACOES
# ---------------------------------------------------------------------------

def render_visualizations(df: pd.DataFrame, classification: dict):
    cat_cols = [
        col for col, t in classification.items()
        if t in ("categorical_nominal", "categorical_ordinal")
        and df[col].nunique() >= 2
    ]
    num_cols = [
        col for col, t in classification.items()
        if t in ("numeric_continuous", "numeric_discrete")
    ]

    tab_labels = ["Categoricas", "Numericas", "Comparacoes", "Correlacao"]
    tabs = st.tabs(tab_labels)

    # ---- TAB 1: Categóricas ----
    with tabs[0]:
        if not cat_cols:
            st.markdown('<div class="info-box">Nenhuma variavel categorica encontrada no dataset.</div>', unsafe_allow_html=True)
        else:
            selected_cat = st.selectbox(
                "Selecionar variavel",
                cat_cols,
                key="cat_select",
            )
            col_a, col_b = st.columns(2)
            with col_a:
                with st.container():
                    render_chart_header(
                        f"Distribuicao de {selected_cat}",
                        "Frequencia absoluta por categoria",
                    )
                    st.plotly_chart(
                        plot_bar(df, selected_cat),
                        use_container_width=True,
                        config={"displayModeBar": False},
                    )
            with col_b:
                with st.container():
                    render_chart_header(
                        f"Proporcao de {selected_cat}",
                        "Participacao relativa de cada categoria",
                    )
                    st.plotly_chart(
                        plot_pie(df, selected_cat),
                        use_container_width=True,
                        config={"displayModeBar": False},
                    )

    # ---- TAB 2: Numéricas ----
    with tabs[1]:
        if not num_cols:
            st.markdown('<div class="info-box">Nenhuma variavel numerica encontrada no dataset.</div>', unsafe_allow_html=True)
        else:
            selected_num = st.selectbox(
                "Selecionar variavel",
                num_cols,
                key="num_select",
            )
            col_a, col_b = st.columns(2)
            with col_a:
                render_chart_header(
                    f"Histograma — {selected_num}",
                    "Distribuicao de frequencias",
                )
                st.plotly_chart(
                    plot_histogram(df, selected_num),
                    use_container_width=True,
                    config={"displayModeBar": False},
                )
            with col_b:
                render_chart_header(
                    f"Boxplot — {selected_num}",
                    "Mediana, quartis e valores atipicos",
                )
                st.plotly_chart(
                    plot_boxplot(df, selected_num),
                    use_container_width=True,
                    config={"displayModeBar": False},
                )

            # Estatísticas descritivas
            st.markdown("<br>", unsafe_allow_html=True)
            render_section("Estatisticas Descritivas")
            desc = df[selected_num].describe().round(3)
            stat_cols = st.columns(len(desc))
            labels_map = {
                "count": "Registros", "mean": "Media",
                "std": "Desvio-Padrao", "min": "Minimo",
                "25%": "Q1", "50%": "Mediana",
                "75%": "Q3", "max": "Maximo",
            }
            for i, (stat, val) in enumerate(desc.items()):
                with stat_cols[i]:
                    render_kpi_card(
                        labels_map.get(stat, stat),
                        f"{val:,.2f}",
                        accent_color="#8B5CF6",
                    )

    # ---- TAB 3: Comparações ----
    with tabs[2]:
        if not cat_cols or not num_cols:
            st.markdown(
                '<div class="info-box">E necessario ter pelo menos uma variavel categorica e uma numerica para comparacoes.</div>',
                unsafe_allow_html=True,
            )
        else:
            col_a, col_b = st.columns(2)
            with col_a:
                group_col = st.selectbox("Variavel de agrupamento (categorica)", cat_cols, key="grp_cat")
            with col_b:
                value_col = st.selectbox("Variavel de valor (numerica)", num_cols, key="grp_num")

            col_c, col_d = st.columns(2)
            with col_c:
                render_chart_header(
                    f"Media de {value_col} por {group_col}",
                    "Agrupamento por categoria",
                )
                st.plotly_chart(
                    plot_grouped_bar(df, group_col, value_col),
                    use_container_width=True,
                    config={"displayModeBar": False},
                )
            with col_d:
                render_chart_header(
                    f"Boxplot de {value_col} por {group_col}",
                    "Dispersao por categoria",
                )
                st.plotly_chart(
                    plot_boxplot(df, value_col, group_col),
                    use_container_width=True,
                    config={"displayModeBar": False},
                )

    # ---- TAB 4: Correlação ----
    with tabs[3]:
        if len(num_cols) < 2:
            st.markdown('<div class="info-box">Sao necessarias pelo menos 2 variaveis numericas para calcular correlacoes.</div>', unsafe_allow_html=True)
        else:
            heatmap = plot_correlation_heatmap(df, num_cols)
            if heatmap:
                render_chart_header(
                    "Mapa de Correlacao",
                    "Coeficiente de Pearson entre variaveis numericas (-1 a +1)",
                )
                st.plotly_chart(
                    heatmap,
                    use_container_width=True,
                    config={"displayModeBar": False},
                )

                # Interpretação
                corr_matrix = df[num_cols].corr()
                pairs = []
                for i in range(len(num_cols)):
                    for j in range(i + 1, len(num_cols)):
                        pairs.append({
                            "Variavel A": num_cols[i],
                            "Variavel B": num_cols[j],
                            "Correlacao": round(corr_matrix.iloc[i, j], 4),
                        })
                if pairs:
                    corr_df = pd.DataFrame(pairs).sort_values(
                        "Correlacao", key=abs, ascending=False
                    )
                    st.dataframe(
                        corr_df,
                        use_container_width=True,
                        hide_index=True,
                    )


# ---------------------------------------------------------------------------
# SECAO: INSIGHTS
# ---------------------------------------------------------------------------

def render_insights(df: pd.DataFrame, classification: dict):
    insights = generate_insights(df, classification)

    if not insights:
        st.markdown('<div class="info-box">Nao foi possivel gerar insights automaticos com os dados atuais.</div>', unsafe_allow_html=True)
        return

    cols_per_row = 3
    rows = [insights[i:i+cols_per_row] for i in range(0, len(insights), cols_per_row)]

    for row in rows:
        cols = st.columns(cols_per_row)
        for i, insight in enumerate(row):
            accent = "#6366F1" if insight["type"] == "numeric" else "#06B6D4"
            with cols[i]:
                st.markdown(
                    f"""
                    <div class="insight-card" style="border-left: 3px solid {accent};">
                        <div class="insight-title">{insight['title']}</div>
                        <div class="insight-value">{insight['value']}</div>
                        <div class="insight-detail">{insight['detail']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


# ---------------------------------------------------------------------------
# SECAO: METODOLOGIA
# ---------------------------------------------------------------------------

def render_methodology(df: pd.DataFrame, classification: dict):
    rows_html = ""
    for col, var_type in classification.items():
        n_unique = df[col].nunique()
        n_null = df[col].isnull().sum()
        completeness = round((1 - n_null / max(len(df), 1)) * 100, 1)
        badge = render_type_badge(var_type)
        rows_html += f"""
        <tr>
            <td style="font-family:'IBM Plex Mono',monospace;color:#6366F1;font-size:0.82rem;">{col}</td>
            <td>{badge}</td>
            <td style="text-align:center;">{n_unique}</td>
            <td style="text-align:center;">{completeness}%</td>
        </tr>
        """

    st.markdown(
        f"""
        <table class="methodology-table">
            <thead>
                <tr>
                    <th>Variavel</th>
                    <th>Tipo Classificado</th>
                    <th style="text-align:center;">Valores Unicos</th>
                    <th style="text-align:center;">Completude</th>
                </tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
        </table>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # --- Carregar dados ---
    with st.spinner("Conectando a API e carregando dados..."):
        raw_df = fetch_data()

    if raw_df.empty:
        st.markdown(
            '<div class="warning-box">'
            'Nao foi possivel carregar os dados da API. '
            'Veja o diagnostico abaixo.'
            '</div>',
            unsafe_allow_html=True,
        )
        with st.expander("Diagnostico da conexao", expanded=True):
            from utils import CSV_EXPORT_URL, SHEET_ID, SHEET_GID

            st.markdown(f"**URL de exportacao CSV:**")
            st.code(CSV_EXPORT_URL, language="text")
            st.markdown(
                """
                **Possiveis causas do erro:**
                - A planilha nao esta compartilhada publicamente
                  → No Google Sheets: *Compartilhar → Qualquer pessoa com o link → Leitor*
                - O `gid` da aba esta incorreto
                  → Abra a planilha no navegador e copie o numero apos `gid=` na URL
                - Sem acesso a internet ou rede corporativa bloqueando `docs.google.com`
                  → Tente abrir a URL acima diretamente no navegador para verificar
                """
            )
            st.markdown(
                f"**SHEET_ID:** `{SHEET_ID}`  \n"
                f"**SHEET_GID:** `{SHEET_GID}`"
            )
        st.stop()

    df = clean_dataframe(raw_df)
    classification = classify_columns(df)
    kpis = compute_kpis(df, classification)

    # --- Sidebar com filtros ---
    df_filtered = build_sidebar(df, classification)

    # =========================================================================
    # HEADER
    # =========================================================================
    st.markdown(
        """
        <div class="main-header">
            <div class="header-badge">Trabalho de Conclusao de Curso</div>
            <h1>Dashboard de Analise de Dados</h1>
            <p>
                Plataforma de analise exploratoria estruturada sobre dados coletados via Google Forms.
                Os dados sao consumidos em tempo real por uma API construida com Google Apps Script
                e processados com classificacao automatica de variaveis, visualizacoes interativas
                e geracao de insights estatisticos.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # =========================================================================
    # KPIs
    # =========================================================================
    render_section("Visao Geral do Dataset", "Metricas gerais extraidas do conjunto de dados atual")

    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        render_kpi_card("Registros Totais", f"{kpis['total_records']:,}", "respostas coletadas", "#6366F1")
    with k2:
        render_kpi_card("Variaveis", f"{kpis['total_columns']}", "campos analisados", "#8B5CF6")
    with k3:
        render_kpi_card("Completude", f"{kpis['completeness']}%", "dados preenchidos", "#10B981")
    with k4:
        render_kpi_card("Var. Numericas", f"{kpis['numeric_cols']}", "continuas + discretas", "#06B6D4")
    with k5:
        render_kpi_card("Var. Categoricas", f"{kpis['categorical_cols']}", "nominais + ordinais", "#F59E0B")

    if len(df_filtered) < len(df):
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            f'<div class="info-box">Filtros ativos: exibindo <strong>{len(df_filtered)}</strong> de '
            f'<strong>{len(df)}</strong> registros.</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # =========================================================================
    # METODOLOGIA
    # =========================================================================
    with st.expander("Metodologia — Classificacao de Variaveis", expanded=False):
        render_section(
            "Classificacao Automatica de Variaveis",
            "Cada variavel foi classificada com base em tipo de dado e numero de valores unicos."
        )
        render_methodology(df, classification)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            """
            <div class="info-box">
                <strong>Criterios de classificacao:</strong>
                Numerica Continua (numerica com mais de 20 valores unicos) |
                Numerica Discreta (numerica com ate 20 valores unicos) |
                Categorica Nominal (string com ate 30 categorias, sem ordem) |
                Categorica Ordinal (detectada por palavras-chave e padroes de escala) |
                Texto Livre (media de palavras maior que 4 e muitos valores unicos) |
                Data (reconhecimento de formatos de data).
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # =========================================================================
    # VISUALIZACOES
    # =========================================================================
    render_section(
        "Analise Exploratoria",
        "Visualizacoes interativas segmentadas por tipo de variavel",
    )
    render_visualizations(df_filtered, classification)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # =========================================================================
    # INSIGHTS AUTOMATICOS
    # =========================================================================
    render_section(
        "Insights Automaticos",
        "Padroes e estatisticas relevantes identificados no conjunto de dados filtrado",
    )
    render_insights(df_filtered, classification)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # =========================================================================
    # TABELA INTERATIVA
    # =========================================================================
    render_section(
        "Tabela de Dados",
        f"Visualizacao dos registros filtrados — {len(df_filtered)} linhas x {len(df_filtered.columns)} colunas",
    )

    search_col, download_col = st.columns([3, 1])
    with search_col:
        search_term = st.text_input(
            "Buscar na tabela",
            placeholder="Digite para filtrar linhas...",
            key="table_search",
            label_visibility="collapsed",
        )
    with download_col:
        csv_data = df_filtered.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Exportar CSV",
            data=csv_data,
            file_name="dados_filtrados.csv",
            mime="text/csv",
            use_container_width=True,
        )

    display_df = df_filtered
    if search_term:
        mask = display_df.astype(str).apply(
            lambda col: col.str.contains(search_term, case=False, na=False)
        ).any(axis=1)
        display_df = display_df[mask]

    st.dataframe(
        display_df,
        use_container_width=True,
        height=400,
    )

    # =========================================================================
    # RODAPÉ
    # =========================================================================
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style="text-align:center;color:#334155;font-size:0.78rem;padding:1rem 0;border-top:1px solid #1E293B;">
            Dashboard desenvolvido com Streamlit e Plotly &mdash;
            Dados coletados via Google Forms &middot; API Google Apps Script &mdash;
            Trabalho de Conclusao de Curso
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()