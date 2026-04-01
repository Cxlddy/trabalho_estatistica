"""
utils.py — Funções utilitárias para o dashboard de análise de dados do TCC.
Versão corrigida e robusta.
"""

import pandas as pd
import numpy as np
import streamlit as st
import re
import io
import urllib.request
import urllib.error
import ssl
from typing import Optional


# ---------------------------------------------------------------------------
# 1. CONFIGURAÇÃO DA FONTE DE DADOS
# ---------------------------------------------------------------------------

SHEET_ID  = "1WTH6-XNneOKLjmk_u4QaDtLWbFIHDUP_3phnqlbs1dU"
SHEET_GID = "951156165"

CSV_EXPORT_URL = (
    f"https://docs.google.com/spreadsheets/d/{SHEET_ID}"
    f"/export?format=csv&gid={SHEET_GID}"
)

_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)


# ---------------------------------------------------------------------------
# 2. FETCH E CACHE
# ---------------------------------------------------------------------------

@st.cache_data(ttl=60, show_spinner=False)
def fetch_data() -> pd.DataFrame:
    raw_bytes = _fetch_csv_bytes()
    if raw_bytes is None:
        return pd.DataFrame()

    try:
        return pd.read_csv(io.BytesIO(raw_bytes), encoding="utf-8", on_bad_lines="skip")
    except UnicodeDecodeError:
        return pd.read_csv(io.BytesIO(raw_bytes), encoding="latin-1", on_bad_lines="skip")
    except Exception as e:
        st.error(f"Erro ao interpretar o CSV: {e}")
        return pd.DataFrame()


def _fetch_csv_bytes() -> Optional[bytes]:
    for verify_ssl in (True, False):
        try:
            req = urllib.request.Request(
                CSV_EXPORT_URL,
                headers={"User-Agent": _UA, "Accept": "text/csv,*/*"},
            )

            ctx = (
                ssl.create_default_context()
                if verify_ssl
                else ssl._create_unverified_context()
            )

            with urllib.request.urlopen(req, context=ctx, timeout=30) as resp:
                data = resp.read()

                if b"<!DOCTYPE" in data[:200] or b"accounts.google.com" in data[:500]:
                    st.error(
                        "A planilha requer login. Configure como: "
                        "'Qualquer pessoa com o link → Leitor'."
                    )
                    return None

                return data

        except urllib.error.HTTPError as e:
            if not verify_ssl:
                st.error(f"Erro HTTP: {e.code} {e.reason}")
                return None
        except Exception as e:
            if not verify_ssl:
                st.error(f"Erro de conexão: {e}")
                return None

    return None


# ---------------------------------------------------------------------------
# 3. LIMPEZA DE DADOS
# ---------------------------------------------------------------------------

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.copy()

    df.dropna(axis=1, how="all", inplace=True)
    df.dropna(axis=0, how="all", inplace=True)

    df.columns = [str(c).strip() for c in df.columns]

    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].apply(
                lambda x: x.strip() if isinstance(x, str) else x
            )
            df[col] = df[col].replace(
                ["", "N/A", "n/a", "NA", "-", "--"], np.nan
            )

        converted = pd.to_numeric(df[col], errors="coerce")
        valid_ratio = converted.notna().sum() / (df[col].notna().sum() or 1)

        if valid_ratio > 0.85:
            df[col] = converted
            continue

        if df[col].dtype == object:
            date_converted = _try_parse_dates(df[col])
            if date_converted is not None:
                df[col] = date_converted

    return df


def _try_parse_dates(series: pd.Series) -> Optional[pd.Series]:
    try:
        return pd.to_datetime(series, errors="coerce", dayfirst=True)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 4. CLASSIFICAÇÃO DE VARIÁVEIS
# ---------------------------------------------------------------------------

ORDINAL_KEYWORDS = [
    "nivel", "nível", "grau", "escolaridade", "frequencia", "frequência",
    "satisfação", "satisfacao", "concordo", "discordo",
    "nunca", "sempre", "baixo", "médio", "alto"
]

ORDINAL_PATTERNS = [
    r"^\d+\s*[-–]\s*.+",
    r"^(nunca|raramente|às vezes|sempre)$",
    r"^(ruim|regular|bom|ótimo)$",
    r"^(baixo|médio|alto)$",
    r"^\d+$",
]

CONTINUOUS_THRESHOLD = 20
CATEGORICAL_THRESHOLD = 30
FREE_TEXT_AVG_WORDS = 4


def classify_columns(df: pd.DataFrame) -> dict:
    return {col: _classify_single(df[col]) for col in df.columns}


def _classify_single(series: pd.Series) -> str:
    series_clean = series.dropna()
    n_unique = series_clean.nunique()

    if pd.api.types.is_datetime64_any_dtype(series):
        return "date"

    if pd.api.types.is_numeric_dtype(series):
        return "numeric_continuous" if n_unique > CONTINUOUS_THRESHOLD else "numeric_discrete"

    if series.dtype == object:
        avg_words = series_clean.apply(lambda x: len(str(x).split())).mean()

        if avg_words > FREE_TEXT_AVG_WORDS and n_unique > CATEGORICAL_THRESHOLD:
            return "free_text"

        if _is_ordinal(series_clean):
            return "categorical_ordinal"

        if n_unique <= CATEGORICAL_THRESHOLD:
            return "categorical_nominal"

        return "free_text"

    return "categorical_nominal"


def _is_ordinal(series: pd.Series) -> bool:
    col_name_lower = str(series.name).lower()

    for kw in ORDINAL_KEYWORDS:
        if kw in col_name_lower:
            return True

    sample = series.dropna().astype(str).str.lower().head(100)

    matches = sum(
        any(re.fullmatch(pat, val.strip()) for pat in ORDINAL_PATTERNS)
        for val in sample
    )

    return (matches / (len(sample) or 1)) > 0.5


def get_type_label(var_type: str) -> str:
    return {
        "numeric_continuous": "Numerica Continua",
        "numeric_discrete": "Numerica Discreta",
        "categorical_nominal": "Categorica Nominal",
        "categorical_ordinal": "Categorica Ordinal",
        "free_text": "Texto Livre",
        "date": "Data",
    }.get(var_type, var_type)


def get_type_color(var_type: str) -> str:
    return {
        "numeric_continuous": "#6366F1",
        "numeric_discrete": "#8B5CF6",
        "categorical_nominal": "#06B6D4",
        "categorical_ordinal": "#0EA5E9",
        "free_text": "#64748B",
        "date": "#10B981",
    }.get(var_type, "#94A3B8")


# ---------------------------------------------------------------------------
# 5. KPIs E INSIGHTS
# ---------------------------------------------------------------------------

def compute_kpis(df: pd.DataFrame, classification: dict) -> dict:
    return {
        "total_records": len(df),
        "total_columns": len(df.columns),
        "completeness": round(
            (1 - df.isnull().sum().sum() / max(df.size, 1)) * 100, 1
        ),
        "numeric_cols": sum(
            t in ("numeric_continuous", "numeric_discrete")
            for t in classification.values()
        ),
        "categorical_cols": sum(
            t in ("categorical_nominal", "categorical_ordinal")
            for t in classification.values()
        ),
    }


def generate_insights(df: pd.DataFrame, classification: dict) -> list:
    insights = []

    for col, var_type in classification.items():
        series = df[col].dropna()
        if series.empty:
            continue

        if var_type in ("numeric_continuous", "numeric_discrete"):
            mean_val = series.mean()
            median_val = series.median()
            std_val = series.std()

            insights.append({
                "title": f"Média de {col}",
                "value": f"{mean_val:.2f}",
                "detail": f"Mediana: {median_val:.2f} | Desvio: {std_val:.2f}",
                "type": "numeric",
            })

        elif var_type in ("categorical_nominal", "categorical_ordinal"):
            mode_val = series.mode()
            if not mode_val.empty:
                freq = (series == mode_val.iloc[0]).sum()
                pct = freq / len(series) * 100

                insights.append({
                    "title": f"Mais frequente em {col}",
                    "value": str(mode_val.iloc[0]),
                    "detail": f"{freq} ocorrências ({pct:.1f}%)",
                    "type": "categorical",
                })

    return insights[:12]