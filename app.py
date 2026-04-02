"""
app.py — Dashboard principal de análise de dados.
Tecnologias: Streamlit, Altair, Pandas.
"""

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

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
    page_title="Painel Analítico",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CSS CUSTOMIZADO
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.5/font/bootstrap-icons.css">
<style>
    /* Importar fonte */
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

    /* Reset e base */
    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
        background-color: #0B1220;
        color: #E2E8F0;
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
# ALTAIR THEME & COLORS
# ---------------------------------------------------------------------------

COLOR_PALETTE = [
    "#6366F1", "#8B5CF6", "#06B6D4", "#10B981",
    "#F59E0B", "#EF4444", "#EC4899", "#0EA5E9",
]

# Configurar tema dark para Altair
alt.themes.enable("dark")


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
    r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
    return (
        f'<span class="type-badge" '
        f'style="background:rgba({r},{g},{b},0.12);'
        f'color:{color};border:1px solid rgba({r},{g},{b},0.3);">'
        f'{label}</span>'
    )


# ---------------------------------------------------------------------------
# GRÁFICOS COM ALTAIR
# ---------------------------------------------------------------------------

def plot_bar(df: pd.DataFrame, col: str) -> alt.Chart:
    counts = df[col].fillna("(nulo)").astype(str).value_counts().reset_index()
    counts.columns = [col, "Contagem"]
    counts = counts.sort_values("Contagem", ascending=True).tail(20)

    return alt.Chart(counts).mark_bar(color="#6366F1", opacity=0.85).encode(
        x=alt.X("Contagem:Q", title="Contagem"),
        y=alt.Y(f"{col}:N", title=col, sort="-x"),
        tooltip=[alt.Tooltip(f"{col}:N"), alt.Tooltip("Contagem:Q")],
    ).properties(
        height=max(300, len(counts) * 35),
        width="container"
    ).interactive()


def plot_pie(df: pd.DataFrame, col: str) -> alt.Chart:
    counts = df[col].fillna("(nulo)").astype(str).value_counts().reset_index()
    counts.columns = [col, "Contagem"]
    top = counts.head(8)
    if len(counts) > 8:
        outros = pd.DataFrame(
            [{col: "Outros", "Contagem": counts.iloc[8:]["Contagem"].sum()}]
        )
        top = pd.concat([top, outros], ignore_index=True)
    
    return alt.Chart(top).encode(
        theta=alt.Theta("Contagem:Q"),
        color=alt.Color(f"{col}:N", scale=alt.Scale(range=COLOR_PALETTE)),
        tooltip=[alt.Tooltip(f"{col}:N"), alt.Tooltip("Contagem:Q")],
    ).mark_arc(innerRadius=60, opacity=0.85).properties(
        height=380,
        width="container",
    )


def plot_histogram(df: pd.DataFrame, col: str) -> alt.Chart:
    series = df[col].dropna()
    if series.empty:
        return alt.Chart(pd.DataFrame({col: [], "count": []})).mark_bar().encode()

    nbins = min(40, max(10, int(np.sqrt(len(series)))))
    data = pd.DataFrame({col: series})

    return alt.Chart(data).mark_bar(color="#6366F1", opacity=0.85).encode(
        x=alt.X(f"{col}:Q", bin=alt.Bin(maxbins=nbins), title=col),
        y=alt.Y("count():Q", title="Frequência"),
        tooltip=[alt.Tooltip(f"{col}:Q", format=".2f"), "count()"],
    ).properties(
        height=340,
        width="container"
    ).interactive()


def plot_boxplot(df: pd.DataFrame, col: str, group_col: str = None) -> alt.Chart:
    if group_col and group_col != col:
        groups = df[group_col].value_counts().head(8).index.tolist()
        data = df[df[group_col].isin(groups)].dropna(subset=[col])
        data[group_col] = data[group_col].astype(str)

        return alt.Chart(data).mark_boxplot(opacity=0.7, size=40).encode(
            x=alt.X(f"{group_col}:N", title=group_col),
            y=alt.Y(f"{col}:Q", title=col),
            color=alt.Color(f"{group_col}:N", scale=alt.Scale(range=COLOR_PALETTE), legend=None),
            tooltip=[alt.Tooltip(f"{group_col}:N"), alt.Tooltip(f"{col}:Q", format=".2f")],
        ).properties(
            height=380,
            width="container"
        ).interactive()
    else:
        series = df[col].dropna()
        data = pd.DataFrame({col: series})

        return alt.Chart(data).mark_boxplot(opacity=0.85, color="#6366F1", size=40).encode(
            y=alt.Y(f"{col}:Q", title=col),
            tooltip=[alt.Tooltip(f"{col}:Q", format=".2f")],
        ).properties(
            height=340,
            width="container"
        ).interactive()


def plot_grouped_bar(df: pd.DataFrame, cat_col: str, num_col: str) -> alt.Chart:
    agg = (
        df.groupby(cat_col)[num_col]
        .mean()
        .reset_index()
        .sort_values(num_col, ascending=False)
        .head(15)
    )
    agg[cat_col] = agg[cat_col].astype(str)

    return alt.Chart(agg).mark_bar(color="#6366F1", opacity=0.85).encode(
        x=alt.X(f"{cat_col}:N", title=cat_col, sort="-y"),
        y=alt.Y(f"{num_col}:Q", title=f"Média de {num_col}"),
        tooltip=[alt.Tooltip(f"{cat_col}:N"), alt.Tooltip(f"{num_col}:Q", format=".2f")],
    ).properties(
        height=360,
        width="container"
    ).interactive()


def plot_correlation_heatmap(df: pd.DataFrame, num_cols: list) -> alt.Chart:
    if len(num_cols) < 2:
        return None

    corr = df[num_cols].corr().reset_index().melt(id_vars="index")
    corr.columns = ["Variável A", "Variável B", "Correlação"]

    return alt.Chart(corr).mark_rect(opacity=0.9).encode(
        x=alt.X("Variável B:N", title=""),
        y=alt.Y("Variável A:N", title=""),
        color=alt.Color(
            "Correlação:Q",
            scale=alt.Scale(scheme="redblue"),
            title="Correlação"
        ),
        tooltip=[alt.Tooltip("Variável A:N"), alt.Tooltip("Variável B:N"), alt.Tooltip("Correlação:Q", format=".3f")],
    ).properties(
        height=max(300, len(num_cols) * 50),
        width=max(300, len(num_cols) * 50),
    )


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
            st.markdown('<p class="sidebar-section-label">Variáveis Categóricas</p>', unsafe_allow_html=True)
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
            st.markdown('<p class="sidebar-section-label">Variáveis Numéricas</p>', unsafe_allow_html=True)
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

    tab_labels = ["Categóricas", "Numéricas", "Comparações", "Correlação"]
    tabs = st.tabs(tab_labels)

    # ---- TAB 1: Categóricas ----
    with tabs[0]:
        if not cat_cols:
            st.markdown('<div class="info-box">Nenhuma variável categórica encontrada no conjunto de dados.</div>', unsafe_allow_html=True)
        else:
            selected_cat = st.selectbox(
                "Selecionar variável",
                cat_cols,
                key="cat_select",
            )
            col_a, col_b = st.columns(2)
            with col_a:
                with st.container():
                    render_chart_header(
                        f"Distribuição de {selected_cat}",
                        "Frequência absoluta por categoria",
                    )
                    st.altair_chart(
                        plot_bar(df, selected_cat),
                        use_container_width=True,
                    )
            with col_b:
                with st.container():
                    render_chart_header(
                        f"Proporção de {selected_cat}",
                        "Participação relativa de cada categoria",
                    )
                    st.altair_chart(
                        plot_pie(df, selected_cat),
                        use_container_width=True,
                    )

    # ---- TAB 2: Numéricas ----
    with tabs[1]:
        if not num_cols:
            st.markdown('<div class="info-box">Nenhuma variável numérica encontrada no conjunto de dados.</div>', unsafe_allow_html=True)
        else:
            selected_num = st.selectbox(
                "Selecionar variável",
                num_cols,
                key="num_select",
            )
            col_a, col_b = st.columns(2)
            with col_a:
                render_chart_header(
                    f"Histograma — {selected_num}",
                    "Distribuição de frequências",
                )
                st.altair_chart(
                    plot_histogram(df, selected_num),
                    use_container_width=True,
                )
            with col_b:
                render_chart_header(
                    f"Boxplot — {selected_num}",
                    "Mediana, quartis e valores atipicos",
                )
                st.altair_chart(
                    plot_boxplot(df, selected_num),
                    use_container_width=True,
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
                '<div class="info-box">É necessário ter pelo menos uma variável categórica e uma variável numérica para comparações.</div>',
                unsafe_allow_html=True,
            )
        else:
            col_a, col_b = st.columns(2)
            with col_a:
                group_col = st.selectbox("Variável de agrupamento (categórica)", cat_cols, key="grp_cat")
            with col_b:
                value_col = st.selectbox("Variável de valor (numérica)", num_cols, key="grp_num")

            col_c, col_d = st.columns(2)
            with col_c:
                render_chart_header(
                    f"Média de {value_col} por {group_col}",
                    "Agrupamento por categoria",
                )
                st.altair_chart(
                    plot_grouped_bar(df, group_col, value_col),
                    use_container_width=True,
                )
            with col_d:
                render_chart_header(
                    f"Boxplot de {value_col} por {group_col}",
                    "Dispersão por categoria",
                )
                st.altair_chart(
                    plot_boxplot(df, value_col, group_col),
                    use_container_width=True,
                )

    # ---- TAB 4: Correlação ----
    with tabs[3]:
        if len(num_cols) < 2:
            st.markdown('<div class="info-box">São necessárias ao menos duas variáveis numéricas para calcular correlações.</div>', unsafe_allow_html=True)
        else:
            heatmap = plot_correlation_heatmap(df, num_cols)
            if heatmap:
                render_chart_header(
                    "Mapa de Correlação",
                    "Coeficiente de Pearson entre variáveis numéricas (-1 a +1)",
                )
                st.altair_chart(
                    heatmap,
                    use_container_width=True,
                )

                # Interpretação
                corr_matrix = df[num_cols].corr()
                pairs = []
                for i in range(len(num_cols)):
                    for j in range(i + 1, len(num_cols)):
                        pairs.append({
                            "Variável A": num_cols[i],
                            "Variável B": num_cols[j],
                            "Correlação": round(corr_matrix.iloc[i, j], 4),
                        })
                if pairs:
                    corr_df = pd.DataFrame(pairs).sort_values(
                        "Correlação", key=abs, ascending=False
                    )
                    st.dataframe(
                        corr_df,
                        use_container_width=True,
                        hide_index=True,
                    )


# ---------------------------------------------------------------------------
# SEÇÃO: INSIGHTS
# ---------------------------------------------------------------------------

def render_insights(df: pd.DataFrame, classification: dict):
    insights = generate_insights(df, classification)

    if not insights:
        st.markdown('<div class="info-box">Não foi possível gerar insights automáticos com os dados atuais.</div>', unsafe_allow_html=True)
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
# SEÇÃO: METODOLOGIA
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
                    <th>Variável</th>
                    <th>Tipo Classificado</th>
                    <th style="text-align:center;">Valores Únicos</th>
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
            'Não foi possível carregar os dados da API. '
            'Veja o diagnóstico abaixo.'
            '</div>',
            unsafe_allow_html=True,
        )
        with st.expander("Diagnóstico da conexão", expanded=True):
            from utils import CSV_EXPORT_URL, SHEET_ID, SHEET_GID

            st.markdown(f"**URL de exportação CSV:**")
            st.code(CSV_EXPORT_URL, language="text")
            st.markdown(
                """
                **Possíveis causas do erro:**
                - A planilha não está compartilhada publicamente
                  → No Google Sheets: *Compartilhar → Qualquer pessoa com o link → Leitor*
                - O `gid` da aba está incorreto
                  → Abra a planilha no navegador e copie o número após `gid=` na URL
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
            <div class="header-badge"><i class="bi bi-speedometer2"></i> Dashboard Analítico</div>
            <h1>Painel de Análise de Dados</h1>
            <p>
                Painel de análise exploratória com visualizações modernas e métricas automáticas.
                Dados importados em tempo real a partir do Google Sheets, classificados e apresentados
                com estatísticas claras, comparações e insights relevantes.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # =========================================================================
    # KPIs
    # =========================================================================
    render_section("Visão Geral do Conjunto de Dados", "Métricas gerais extraídas do conjunto de dados atual")

    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        render_kpi_card("Registros Totais", f"{kpis['total_records']:,}", "respostas coletadas", "#6366F1")
    with k2:
        render_kpi_card("Variáveis", f"{kpis['total_columns']}", "campos analisados", "#8B5CF6")
    with k3:
        render_kpi_card("Completude", f"{kpis['completeness']}%", "percentual de preenchimento", "#10B981")
    with k4:
        render_kpi_card("Variáveis Numéricas", f"{kpis['numeric_cols']}", "contínuas + discretas", "#06B6D4")
    with k5:
        render_kpi_card("Variáveis Categóricas", f"{kpis['categorical_cols']}", "nominais + ordinais", "#F59E0B")

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
    with st.expander("Metodologia — Classificação de Variáveis", expanded=False):
        render_section(
            "Classificação Automática de Variáveis",
            "Cada variável foi classificada com base em tipo de dado e número de valores únicos."
        )
        render_methodology(df, classification)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            """
            <div class="info-box">
                <strong>Critérios de classificação:</strong>
                Numérica Contínua (numérica com mais de 20 valores únicos) |
                Numérica Discreta (numérica com até 20 valores únicos) |
                Categórica Nominal (texto com até 30 categorias, sem ordem) |
                Categórica Ordinal (detecção por palavras-chave e padrões de escala) |
                Texto Livre (média de palavras maior que 4 e muitos valores únicos) |
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
        "Análise Exploratória",
        "Visualizações interativas segmentadas por tipo de variável",
    )
    render_visualizations(df_filtered, classification)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # =========================================================================
    # INSIGHTS AUTOMATICOS
    # =========================================================================
    render_section(
        "Insights Automáticos",
        "Padrões e estatísticas relevantes identificados no conjunto de dados filtrado",
    )
    render_insights(df_filtered, classification)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # =========================================================================
    # TABELA INTERATIVA
    # =========================================================================
    render_section(
        "Tabela de Dados",
        f"Visualização dos registros filtrados — {len(df_filtered)} linhas x {len(df_filtered.columns)} colunas",
    )

    _, download_col = st.columns([3, 1])
    with download_col:
        csv_data = df_filtered.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Exportar CSV",
            data=csv_data,
            file_name="dados_filtrados.csv",
            mime="text/csv",
            use_container_width=True,
        )

    display_df = df_filtered.copy()

    # Substituir DataHora numérico/nao-desejado por formato legível
    if "DataHoraFormatado" in display_df.columns:
        display_df = display_df.drop(columns=["DataHora"], errors="ignore")
        display_df = display_df.rename(columns={"DataHoraFormatado": "HorarioDia"})

    # Remover colunas duplicadas residuais para não quebrar o Streamlit
    display_df = display_df.loc[:, ~display_df.columns.duplicated()]

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
            Dashboard desenvolvido com Streamlit e Altair &mdash;
            Dados coletados via Google Forms &middot; API Google Apps Script
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()