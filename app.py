"""
app.py — Streamlit Deployment App
Online Course Recommendation System
=====================================
Run with:  streamlit run app.py

Models available:
  1. Content-Based KNN (PCA features + cosine similarity)
  2. User-Based Collaborative Filtering (KNN on user-item matrix)
  3. SVD Matrix Factorization
  4. Hybrid (CF + CBF weighted blend)
  5. TF-IDF Content-Based (NLP)

Layout of this file
  1. Imports, page config, design system (CSS tokens + components)
  2. Matplotlib theme + small UI helpers
  3. Data loading, preprocessing and the five recommenders  (unchanged logic)
  4. Header / navigation and the four pages

Pair this file with .streamlit/config.toml (light base theme) so that native
widgets, tables and sliders pick up the same palette as the custom CSS.
"""

import warnings
warnings.filterwarnings("ignore")

import html
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from pathlib import Path

from scipy import stats
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler, OrdinalEncoder
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Course Atlas",
    page_icon="C",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ═══════════════════════════════════════════════════════════════════════════════
# DESIGN SYSTEM
# One token block + shared component rules. Everything below reads the tokens;
# change a colour here and it changes everywhere (charts read the same values
# from the PALETTE dict further down).
#
# Contrast (measured against the #F7F5F0 background):
#   ink 13.4:1 · text-2 6.5:1 · text-3 5.0:1 · teal-strong 5.6:1 · gold-ink 5.7:1
# The brand teal (#4F8078, 4.1:1) is used for fills / rules only, never as small
# text. Gold (#D9A441) is decorative only (brand mark, selected-item marker).
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Newsreader:opsz,wght@6..72,500;6..72,600&display=swap');

    :root {
        /* Colour */
        --ink: #172B3A;
        --bg: #F7F5F0;
        --bg-2: #EFECE3;
        --surface: #FCFBF8;
        --line: #DDD8CB;
        --line-strong: #8E8979;
        --text: #172B3A;
        --text-2: #4A5A66;
        --text-3: #5E6C75;
        --teal: #4F8078;
        --teal-strong: #3B6B63;
        --teal-hover: #2F5A53;
        --sage: #DCE8DF;
        --sage-soft: #E9EFE7;
        --gold: #D9A441;
        --gold-ink: #7F5A12;
        --success: #2F6B4F;  --success-bg: #E3EFE6;
        --warning: #7A4E08;  --warning-bg: #F6EAD0;
        --error:   #9B3328;  --error-bg:   #F6E1DD;
        --focus: #3B6B63;

        /* Type */
        --font-ui: "Inter", "Segoe UI", system-ui, -apple-system, "Helvetica Neue", Arial, sans-serif;
        --font-display: "Newsreader", Georgia, "Times New Roman", serif;

        /* Spacing scale (4px base) */
        --s1: 0.25rem; --s2: 0.5rem; --s3: 0.75rem; --s4: 1rem;
        --s5: 1.5rem;  --s6: 2rem;   --s7: 3rem;   --s8: 4rem;

        --radius: 6px;
    }

    /* ── Foundation ─────────────────────────────────────────────────────── */
    html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
        background: var(--bg) !important;
        color: var(--text);
    }
    .stApp, .stApp p, .stApp li, .stApp label, .stApp button, .stApp input,
    .stApp textarea, .stApp [data-baseweb="select"], .stApp [data-testid="stMarkdownContainer"],
    .stApp [data-testid="stCaptionContainer"] {
        font-family: var(--font-ui);
    }
    .stApp p, .stApp li { line-height: 1.6; }

    [data-testid="stHeader"] { background: transparent !important; }
    [data-testid="stAppDeployButton"],
    [data-testid="stSidebar"],
    [data-testid="stSidebarCollapsedControl"],
    [data-testid="collapsedControl"] { display: none !important; }

    /* One content container: full width on laptops, capped on very wide screens.
       Top padding clears Streamlit's fixed header so the brand is never clipped. */
    [data-testid="stMainBlockContainer"] {
        width: 100%;
        max-width: 1520px !important;
        margin: 0 auto;
        padding: var(--s8) clamp(1.25rem, 3.2vw, 3.5rem) var(--s7) !important;
    }

    h1, h2, h3, h4 {
        font-family: var(--font-display) !important;
        color: var(--ink) !important;
        font-weight: 600 !important;
        letter-spacing: -0.01em !important;
    }
    h3 { font-size: 1.25rem !important; line-height: 1.3 !important; padding-top: 0 !important; }
    a { color: var(--teal-strong); }

    [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {
        color: var(--text-3) !important;
        font-size: 0.84rem;
    }

    /* Short, subtle entrance for each page. Each page lives in its own keyed
       container, so it remounts (and fades in) only when the page changes. */
    [class*="st-key-atlas_page_"] { animation: atlas-in 160ms ease-out both; }
    @keyframes atlas-in { from { opacity: 0; transform: translateY(4px); } to { opacity: 1; transform: none; } }
    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after { animation: none !important; transition: none !important; scroll-behavior: auto !important; }
    }

    /* ── Top bar: brand + navigation ────────────────────────────────────── */
    .st-key-atlas_topbar {
        border-bottom: 1px solid var(--line);
        margin-bottom: var(--s6);
    }
    .st-key-atlas_topbar > div { gap: 0 !important; }
    .atlas-brand {
        display: flex; align-items: center; gap: 0.7rem;
        padding: 0 0 0.7rem;
        white-space: nowrap;
    }
    .atlas-mark {
        position: relative; flex: 0 0 30px; width: 30px; height: 30px;
        background-color: var(--ink);
        background-image:
            linear-gradient(rgba(220,232,223,0.30) 1px, transparent 1px),
            linear-gradient(90deg, rgba(220,232,223,0.30) 1px, transparent 1px);
        background-size: 10px 10px;
        background-position: -1px -1px;
        border-radius: 6px;
    }
    .atlas-mark::after {
        content: ""; position: absolute; left: 62%; top: 34%;
        width: 8px; height: 8px; margin: -4px 0 0 -4px;
        background: var(--gold); border-radius: 50%;
        box-shadow: 0 0 0 2px var(--ink);
    }
    .atlas-wordmark {
        font-family: var(--font-display);
        font-size: 1.6rem; font-weight: 600; line-height: 1.1;
        letter-spacing: -0.015em; color: var(--ink);
    }

    /* Navigation = native segmented control restyled as an underline tab bar.
       Streamlit's own DOM is targeted (button[kind=...]) and scoped to the
       keyed container, so the rules are isolated to this one place. */
    .st-key-atlas_nav [data-testid="stButtonGroup"] { gap: 0 !important; flex-wrap: wrap; }
    .st-key-atlas_nav [data-testid="stButtonGroup"] button {
        background: transparent !important;
        border: 0 !important;
        border-bottom: 2px solid transparent !important;
        border-radius: 0 !important;
        box-shadow: none !important;
        margin: 0 !important;
        min-height: 2.9rem;
        padding: 0.55rem 1.1rem !important;
        color: var(--text-2) !important;
        font-weight: 500;
        transition: color 120ms ease, background-color 120ms ease, border-color 120ms ease;
    }
    .st-key-atlas_nav [data-testid="stButtonGroup"] button p,
    .st-key-atlas_nav [data-testid="stButtonGroup"] button div { color: inherit !important; font-size: 0.95rem; }
    .st-key-atlas_nav [data-testid="stButtonGroup"] button:hover {
        color: var(--ink) !important;
        background: var(--sage-soft) !important;
    }
    .st-key-atlas_nav [data-testid="stButtonGroup"] button[kind$="Active"],
    .st-key-atlas_nav [data-testid="stButtonGroup"] button[aria-pressed="true"],
    .st-key-atlas_nav [data-testid="stButtonGroup"] button[aria-checked="true"] {
        color: var(--ink) !important;
        font-weight: 650;
        border-bottom-color: var(--teal) !important;
        background: transparent !important;
    }

    /* ── Page header + section headers ──────────────────────────────────── */
    .page-head { margin: 0 0 var(--s6); }
    .eyebrow {
        color: var(--teal-strong);
        font-size: 0.72rem; font-weight: 650;
        letter-spacing: 0.14em; text-transform: uppercase;
        margin-bottom: var(--s2);
    }
    .page-title {
        font-family: var(--font-display);
        font-size: clamp(2rem, 3.2vw, 2.75rem);
        font-weight: 600; line-height: 1.1; letter-spacing: -0.02em;
        color: var(--ink); margin: 0 0 var(--s3);
    }
    .page-intro {
        color: var(--text-2);
        max-width: 64ch; font-size: 1.04rem; line-height: 1.65; margin: 0;
    }
    .section-head {
        display: flex; align-items: baseline; justify-content: space-between; gap: var(--s4);
        flex-wrap: wrap;
        margin: var(--s7) 0 var(--s4); padding-bottom: var(--s2);
        border-bottom: 1px solid var(--line);
    }
    .section-head h2 {
        margin: 0 !important; padding: 0 !important;
        font-size: 1.45rem !important; line-height: 1.2 !important;
    }
    .section-note { color: var(--text-3); font-size: 0.84rem; }

    .small-label {
        color: var(--teal-strong);
        font-size: 0.7rem; font-weight: 650;
        letter-spacing: 0.12em; text-transform: uppercase;
    }

    /* ── Content blocks (no boxes: rules, whitespace, type) ─────────────── */
    .route { border-top: 2px solid var(--ink); padding-top: var(--s4); margin-bottom: var(--s3); }
    .route-title {
        font-family: var(--font-display); font-size: 1.5rem; font-weight: 600;
        color: var(--ink); line-height: 1.2; margin: var(--s2) 0 var(--s2);
    }
    .route-text { color: var(--text-2); max-width: 46ch; line-height: 1.6; margin: 0 0 var(--s2); }

    .course-item { border-top: 1px solid var(--line-strong); padding: var(--s3) 0 var(--s2); }
    .course-item-title {
        color: var(--ink); font-weight: 600; font-size: 1rem; line-height: 1.4;
        margin: var(--s1) 0 var(--s2); overflow-wrap: anywhere;
    }
    .course-item-meta { color: var(--text-2); font-size: 0.86rem; line-height: 1.7; }
    .course-item-meta strong, .result-meta strong { color: var(--ink); font-weight: 650; }

    .selected-course {
        border-left: 3px solid var(--gold);
        padding: var(--s1) 0 var(--s1) var(--s4);
    }
    .selected-title {
        font-family: var(--font-display); font-size: 1.6rem; font-weight: 600;
        line-height: 1.2; color: var(--ink); margin: var(--s2) 0; overflow-wrap: anywhere;
    }

    .result-list { list-style: none; margin: 0; padding: 0; border-top: 2px solid var(--ink); }
    .result-row {
        display: grid; grid-template-columns: 2.75rem minmax(0, 1fr) auto;
        align-items: center; gap: var(--s4);
        padding: var(--s3) 0; border-bottom: 1px solid var(--line);
    }
    .result-rank {
        font-family: var(--font-display); font-size: 1.55rem; font-weight: 600;
        color: var(--teal-strong); text-align: left; font-variant-numeric: tabular-nums;
    }
    .result-title { color: var(--ink); font-weight: 600; font-size: 1.02rem; line-height: 1.35; overflow-wrap: anywhere; }
    .result-meta { color: var(--text-2); font-size: 0.86rem; line-height: 1.6; margin-top: 2px; }
    .result-score { text-align: right; min-width: 6rem; }
    .result-score-value {
        display: block; color: var(--ink); font-weight: 650; font-size: 1.1rem;
        font-variant-numeric: tabular-nums;
    }
    .result-score-label { display: block; color: var(--text-3); font-size: 0.72rem; letter-spacing: 0.06em; text-transform: uppercase; }

    .empty-state {
        border: 1px dashed var(--line-strong); border-radius: var(--radius);
        padding: var(--s5); margin-top: var(--s5); color: var(--text-2);
    }
    .empty-state strong { display: block; color: var(--ink); font-size: 1rem; margin-bottom: var(--s1); }

    /* ── Native components, restyled once ───────────────────────────────── */
    [data-testid="stMetric"] {
        background: transparent !important; border: 0 !important; border-radius: 0 !important;
        border-top: 1px solid var(--line-strong) !important;
        padding: var(--s3) 0 var(--s2) !important;
    }
    [data-testid="stMetricLabel"], [data-testid="stMetricLabel"] p {
        color: var(--text-2) !important; font-size: 0.74rem !important; font-weight: 600 !important;
        letter-spacing: 0.08em; text-transform: uppercase;
    }
    [data-testid="stMetricValue"], [data-testid="stMetricValue"] div {
        font-family: var(--font-display) !important; color: var(--ink) !important;
        font-size: 2.2rem !important; font-weight: 600 !important; line-height: 1.15 !important;
    }

    /* Inputs, selects */
    .stApp label p, .stApp [data-testid="stWidgetLabel"] p {
        color: var(--text-2) !important; font-size: 0.84rem !important; font-weight: 600 !important;
    }
    .stApp [data-baseweb="select"] > div,
    .stApp [data-baseweb="input"],
    .stApp [data-baseweb="base-input"] {
        background: var(--surface) !important;
        border-color: var(--line-strong) !important;
        border-radius: var(--radius) !important;
        min-height: 2.65rem;
        color: var(--text) !important;
    }
    .stApp [data-baseweb="select"] span, .stApp [data-baseweb="select"] div { color: var(--text); }
    [data-baseweb="popover"] [data-baseweb="menu"], [data-baseweb="popover"] ul {
        background: var(--surface) !important;
        border: 1px solid var(--line-strong); border-radius: var(--radius);
    }
    [data-baseweb="popover"] li { color: var(--text) !important; }
    [data-baseweb="popover"] li:hover, [data-baseweb="popover"] li[aria-selected="true"] {
        background: var(--sage-soft) !important;
    }
    .stApp [role="slider"] { background-color: var(--teal-strong) !important; }

    /* Buttons: one primary (filled teal), one secondary (outlined ink) */
    .stApp button[kind="primary"], .stApp button[kind="primaryFormSubmit"] {
        background: var(--teal-strong) !important; color: #FFFFFF !important;
        border: 1px solid var(--teal-strong) !important;
        border-radius: var(--radius) !important; min-height: 2.75rem; font-weight: 600;
        box-shadow: none !important;
        transition: background-color 120ms ease, border-color 120ms ease;
    }
    .stApp button[kind="primary"] p, .stApp button[kind="primary"] div { color: #FFFFFF !important; }
    .stApp button[kind="primary"]:hover, .stApp button[kind="primaryFormSubmit"]:hover {
        background: var(--teal-hover) !important; border-color: var(--teal-hover) !important;
    }
    .stApp button[kind="secondary"], .stApp button[kind="secondaryFormSubmit"] {
        background: transparent !important; color: var(--ink) !important;
        border: 1px solid var(--ink) !important;
        border-radius: var(--radius) !important; min-height: 2.75rem; font-weight: 600;
        box-shadow: none !important;
        transition: background-color 120ms ease;
    }
    .stApp button[kind="secondary"] p, .stApp button[kind="secondary"] div { color: var(--ink) !important; }
    .stApp button[kind="secondary"]:hover { background: var(--sage-soft) !important; }
    .stApp button:disabled { opacity: 0.45 !important; cursor: not-allowed; }

    /* Tabs */
    [data-testid="stTabs"] [role="tablist"] { border-bottom: 1px solid var(--line); gap: var(--s2); }
    [data-testid="stTabs"] button[role="tab"] { color: var(--text-2) !important; font-weight: 500; }
    [data-testid="stTabs"] button[role="tab"] p { color: inherit !important; font-size: 0.95rem; }
    [data-testid="stTabs"] button[role="tab"][aria-selected="true"] { color: var(--ink) !important; font-weight: 650; }
    [data-testid="stTabs"] [data-baseweb="tab-highlight"] { background-color: var(--teal) !important; height: 2px; }
    [data-testid="stTabs"] [data-baseweb="tab-border"] { background-color: var(--line) !important; }

    /* Expanders, tables, alerts, spinner */
    [data-testid="stExpander"] details {
        border: 1px solid var(--line) !important; border-radius: var(--radius) !important;
        background: transparent !important;
    }
    [data-testid="stExpander"] summary p { color: var(--ink) !important; font-weight: 600; }
    [data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: var(--radius); overflow: hidden; }
    [data-testid="stAlert"] {
        border-radius: var(--radius) !important; border: 1px solid var(--line) !important;
        background: var(--surface) !important; color: var(--text) !important;
    }
    [data-testid="stAlert"] p { color: inherit !important; }
    [data-testid="stSpinner"] p, [data-testid="stSpinner"] { color: var(--text-2) !important; }
    [data-testid="stImage"] img, [data-testid="stImageContainer"] img { max-width: 100%; }

    /* Keyboard focus: always visible, never colour-only reliance */
    .stApp button:focus-visible, .stApp a:focus-visible, .stApp [role="tab"]:focus-visible,
    .stApp [role="slider"]:focus-visible, .stApp input:focus-visible,
    .stApp [data-baseweb="select"]:focus-within {
        outline: 2px solid var(--focus) !important; outline-offset: 2px !important;
    }

    /* ── Narrow screens ─────────────────────────────────────────────────── */
    @media (max-width: 760px) {
        [data-testid="stMainBlockContainer"] { padding-left: 1rem !important; padding-right: 1rem !important; }
        .atlas-wordmark { font-size: 1.35rem; }
        .st-key-atlas_nav [data-testid="stButtonGroup"] button { padding: 0.5rem 0.65rem !important; }
        .result-row { grid-template-columns: 2rem minmax(0, 1fr); }
        .result-score { grid-column: 2; text-align: left; display: flex; align-items: baseline; gap: var(--s2); }
        [data-testid="stMetricValue"], [data-testid="stMetricValue"] div { font-size: 1.8rem !important; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Chart theme (mirrors the CSS tokens so charts match the page) ─────────────
PALETTE = {
    "bg": "#F7F5F0", "surface": "#FCFBF8", "ink": "#172B3A",
    "text2": "#4A5A66", "grid": "#E4DFD2", "axis": "#8E8979",
    "teal": "#4F8078", "teal_strong": "#3B6B63", "sage": "#DCE8DF", "gold": "#B8841F",
}
plt.rcParams.update({
    "figure.facecolor": PALETTE["bg"], "savefig.facecolor": PALETTE["bg"],
    "axes.facecolor": PALETTE["surface"], "axes.edgecolor": PALETTE["axis"],
    "axes.labelcolor": PALETTE["text2"], "axes.titlecolor": PALETTE["ink"],
    "axes.titlesize": 12.5, "axes.titleweight": "semibold", "axes.titlelocation": "left",
    "axes.titlepad": 10, "axes.labelsize": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": PALETTE["grid"], "grid.linewidth": 0.8,
    "axes.axisbelow": True,
    "xtick.color": PALETTE["text2"], "ytick.color": PALETTE["text2"],
    "text.color": PALETTE["ink"], "legend.frameon": False,
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"],
})


def _stretch_kwargs() -> dict:
    """Full-width argument that works on old and new Streamlit (use_container_width is being retired)."""
    try:
        major, minor = (int(p) for p in st.__version__.split(".")[:2])
    except Exception:
        return {"use_container_width": True}
    return {"width": "stretch"} if (major, minor) >= (1, 50) else {"use_container_width": True}


STRETCH = _stretch_kwargs()


def show_figure(fig):
    """Render a Matplotlib figure, then release it (figures otherwise pile up across reruns)."""
    st.pyplot(fig, **STRETCH)
    plt.close(fig)


# ── Small UI helpers (all dynamic text is HTML-escaped) ───────────────────────
def esc(value) -> str:
    return html.escape(str(value), quote=True)


def money(value) -> str:
    # &#36; instead of "$": Streamlit's markdown would otherwise read pairs of $ as LaTeX.
    return f"&#36;{float(value):,.0f}"


def has_certificate(value) -> bool:
    return str(value).strip().lower() in ["yes", "true", "1"]


def page_header(eyebrow: str, title: str, intro: str) -> None:
    st.markdown(
        f'<header class="page-head"><div class="eyebrow">{esc(eyebrow)}</div>'
        f'<h1 class="page-title">{esc(title)}</h1><p class="page-intro">{esc(intro)}</p></header>',
        unsafe_allow_html=True,
    )


def section_head(title: str, note: str = "") -> None:
    note_html = f'<span class="section-note">{esc(note)}</span>' if note else ""
    st.markdown(
        f'<div class="section-head"><h2>{esc(title)}</h2>{note_html}</div>',
        unsafe_allow_html=True,
    )


def empty_state(title: str, text: str) -> None:
    st.markdown(
        f'<div class="empty-state"><strong>{esc(title)}</strong>{esc(text)}</div>',
        unsafe_allow_html=True,
    )


def course_meta(course) -> str:
    cert = "Certificate" if has_certificate(course["certification_offered"]) else "No certificate"
    return (
        f"{esc(course['difficulty_level'])} · {cert} · "
        f"Rating <strong>{float(course['rating']):.1f}</strong> · "
        f"{money(course['course_price'])} · ID {esc(course['course_id'])}"
    )


def render_ranked_results(result: pd.DataFrame, score_col: str, score_label: str) -> None:
    """Ranked list of recommendations. Order and scores come straight from the model output."""
    rows = []
    for rank, (_, course) in enumerate(result.iterrows(), start=1):
        score = float(course.get(score_col, 0))
        rows.append(
            '<li class="result-row">'
            f'<span class="result-rank">{rank}</span>'
            f'<div><div class="result-title">{esc(course["course_name"])}</div>'
            f'<div class="result-meta">{course_meta(course)}</div></div>'
            f'<div class="result-score"><span class="result-score-value">{score:.3f}</span>'
            f'<span class="result-score-label">{esc(score_label)}</span></div>'
            "</li>"
        )
    st.markdown(f'<ol class="result-list">{"".join(rows)}</ol>', unsafe_allow_html=True)


def results_table(result: pd.DataFrame, score_col: str) -> None:
    with st.expander("View results as a table"):
        display_cols = [
            "course_id", "course_name", "difficulty_level",
            "certification_offered", "rating", "course_price", score_col,
        ]
        display_cols = [c for c in display_cols if c in result.columns]
        st.dataframe(
            result[display_cols].style.format(
                {score_col: "{:.4f}", "rating": "{:.2f}", "course_price": "${:.0f}"}
            ),
            **STRETCH,
        )



# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_DIR = "models"
NUMERICAL_FEATURES = [
    "course_duration_hours", "rating", "enrollment_numbers",
    "course_price", "feedback_score", "time_spent_hours",
    "previous_courses_taken",
]
ALPHA = 0.7  # hybrid weight: CF contribution


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING & PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_and_preprocess(uploaded_file) -> dict:
    df = pd.read_csv(uploaded_file)

    # Temporary sample for memory-limited testing
    max_users = 500
    max_courses = 300

    selected_users = df["user_id"].drop_duplicates().head(max_users)
    selected_courses = df["course_id"].drop_duplicates().head(max_courses)

    df = df[
        df["user_id"].isin(selected_users)
        & df["course_id"].isin(selected_courses)
    ].copy()

    df_processed = df.copy()

    st.caption(
        f"Running on a test sample: {len(df_processed):,} records, "
        f"{df_processed['user_id'].nunique():,} users and "
        f"{df_processed['course_id'].nunique():,} courses."
    )

    # Outlier capping
    for col in NUMERICAL_FEATURES:
        q1, q3 = df_processed[col].quantile([0.25, 0.75])
        iqr = q3 - q1

        df_processed[col] = df_processed[col].clip(
            lower=q1 - 1.5 * iqr,
            upper=q3 + 1.5 * iqr
        )

    # Continue with the existing encoding section below

    # ── Encoding ──────────────────────────────────────────────────────────────
    label_encoders = {}
    for col in ["certification_offered", "study_material_available"]:
        le = LabelEncoder()
        df_processed[col + "_encoded"] = le.fit_transform(df_processed[col])
        label_encoders[col] = le

    oe = OrdinalEncoder(categories=[["Beginner", "Intermediate", "Advanced"]])
    df_processed["difficulty_level_encoded"] = oe.fit_transform(
        df_processed[["difficulty_level"]]
    )

    # ── Scaling ───────────────────────────────────────────────────────────────
    scaler_standard = StandardScaler()
    scaler_minmax = MinMaxScaler()
    df_processed[["scaled_" + c for c in NUMERICAL_FEATURES]] = (
        scaler_standard.fit_transform(df_processed[NUMERICAL_FEATURES])
    )
    df_processed[["normalized_" + c for c in NUMERICAL_FEATURES]] = (
        scaler_minmax.fit_transform(df_processed[NUMERICAL_FEATURES])
    )

    # ── Feature Engineering ───────────────────────────────────────────────────
    df_processed["engagement_score"] = (
        df_processed["time_spent_hours"] / df_processed["course_duration_hours"]
    ) * df_processed["rating"]

    df_processed["price_per_hour"] = (
        df_processed["course_price"] / df_processed["course_duration_hours"]
    )

    enroll = df_processed["enrollment_numbers"]
    df_processed["popularity_score"] = (enroll - enroll.min()) / (enroll.max() - enroll.min())

    df_processed["quality_indicator"] = (
        df_processed["rating"] + df_processed["feedback_score"] / 2
    )

    df_processed["completion_ratio"] = (
        df_processed["time_spent_hours"] / df_processed["course_duration_hours"]
    )

    df_processed["experience_level"] = pd.cut(
        df_processed["previous_courses_taken"],
        bins=[-1, 2, 5, 10, 20],
        labels=["Beginner", "Intermediate", "Advanced", "Expert"],
    )

    df_processed["value_for_money"] = df_processed["rating"] / (
        df_processed["course_price"] + 1
    )

    # ── Aggregations ──────────────────────────────────────────────────────────
    user_stats = df_processed.groupby("user_id").agg(
        user_avg_rating=("rating", "mean"),
        user_rating_std=("rating", "std"),
        user_course_count=("rating", "count"),
        user_avg_time=("time_spent_hours", "mean"),
        user_total_time=("time_spent_hours", "sum"),
        user_avg_price=("course_price", "mean"),
    ).reset_index()
    df_processed = df_processed.merge(user_stats, on="user_id", how="left")

    course_stats = df_processed.groupby("course_id").agg(
        course_avg_rating=("rating", "mean"),
        course_rating_count=("rating", "count"),
        course_avg_enrollment=("enrollment_numbers", "mean"),
        course_avg_feedback=("feedback_score", "mean"),
    ).reset_index()
    df_processed = df_processed.merge(course_stats, on="course_id", how="left")

    # ── Interaction Features ──────────────────────────────────────────────────
    df_processed["rating_experience_interaction"] = (
        df_processed["rating"] * df_processed["previous_courses_taken"]
    )
    df_processed["price_certification_interaction"] = (
        df_processed["course_price"] * df_processed["certification_offered_encoded"]
    )
    df_processed["duration_difficulty_interaction"] = (
        df_processed["course_duration_hours"] * (df_processed["difficulty_level_encoded"] + 1)
    )

    # ── PCA ───────────────────────────────────────────────────────────────────
    pca_features = [c for c in df_processed.columns if c.startswith("scaled_")]
    X_pca = df_processed[pca_features].values

    pca_full = PCA()
    pca_full.fit(X_pca)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    n_opt = int(np.argmax(cumvar >= 0.95)) + 1

    pca = PCA(n_components=n_opt)
    X_pca_transformed = pca.fit_transform(X_pca)
    pca_cols = [f"PC{i+1}" for i in range(n_opt)]
    df_pca = pd.DataFrame(X_pca_transformed, columns=pca_cols)
    for col in pca_cols:
        df_processed[col] = df_pca[col].values

    # ── K-Means clustering ────────────────────────────────────────────────────
    X_cluster = df_pca[pca_cols[:3]].values if n_opt >= 3 else df_pca[pca_cols[:2]].values
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df_processed["cluster"] = kmeans.fit_predict(X_cluster)

    # ── Course Catalog ────────────────────────────────────────────────────────
    course_catalog = (
        df_processed.groupby("course_id")
        .agg(
            course_name=("course_name", "first"),
            certification_offered=("certification_offered", "first"),
            difficulty_level=("difficulty_level", "first"),
            rating=("rating", "mean"),
            course_price=("course_price", "mean"),
            feedback_score=("feedback_score", "mean"),
            enrollment_numbers=("enrollment_numbers", "mean"),
        )
        .reset_index()
    )

    # ── User-Item Matrix ──────────────────────────────────────────────────────
    user_item_matrix = df_processed.pivot_table(
        index="user_id", columns="course_id", values="rating", fill_value=0
    )

    # ── Model 1: Content-Based KNN ────────────────────────────────────────────
    X_courses = df_pca.values
    knn_model = NearestNeighbors(n_neighbors=10, metric="cosine", algorithm="brute")
    knn_model.fit(X_courses)
    course_index_map = df_processed[["course_id"]].reset_index(drop=True)

    # ── Model 2: User-Based KNN ───────────────────────────────────────────────
    user_knn = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=10)
    user_knn.fit(user_item_matrix)

    # ── Model 3: SVD ──────────────────────────────────────────────────────────
    svd_model = TruncatedSVD(n_components=20, random_state=42)
    user_features = svd_model.fit_transform(user_item_matrix)
    predicted_ratings = np.dot(user_features, svd_model.components_)
    predicted_ratings_df = pd.DataFrame(
        predicted_ratings,
        index=user_item_matrix.index,
        columns=user_item_matrix.columns,
    )

    # ── Model 5: TF-IDF ───────────────────────────────────────────────────────
    course_df = (
        df_processed.drop_duplicates(subset=["course_name"])[
            ["course_id", "course_name", "difficulty_level",
             "certification_offered", "rating", "experience_level"]
        ]
        .reset_index(drop=True)
    )
    course_df["course_text"] = (
        course_df["course_name"].astype(str) + " " +
        course_df["difficulty_level"].astype(str) + " " +
        course_df["certification_offered"].astype(str) + " " +
        course_df["experience_level"].astype(str)
    )
    tfidf = TfidfVectorizer(stop_words="english", max_features=2000)
    tfidf_matrix = tfidf.fit_transform(course_df["course_text"])
    tfidf_course_index = pd.Series(course_df.index, index=course_df["course_name"])

    return {
        "df": df,
        "df_processed": df_processed,
        "df_pca": df_pca,
        "pca": pca,
        "course_catalog": course_catalog,
        "user_item_matrix": user_item_matrix,
        "knn_model": knn_model,
        "X_courses": X_courses,
        "course_index_map": course_index_map,
        "user_knn": user_knn,
        "predicted_ratings_df": predicted_ratings_df,
        "tfidf_matrix": tfidf_matrix,
        "tfidf_course_index": tfidf_course_index,
        "course_df": course_df,
        "pca_full": pca_full,
        "kmeans": kmeans,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# RECOMMENDATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def recommend_knn(course_id: int, top_n: int, state: dict) -> pd.DataFrame:
    """Model 1: Content-Based KNN on PCA features."""
    cmap = state["course_index_map"]
    if course_id not in cmap["course_id"].values:
        return pd.DataFrame()
    idx = cmap[cmap["course_id"] == course_id].index[0]
    distances, indices = state["knn_model"].kneighbors(
        state["X_courses"][idx].reshape(1, -1), n_neighbors=top_n + 1
    )
    sim_indices = indices[0][1:]
    sim_dists = distances[0][1:]
    rec_ids = cmap.iloc[sim_indices]["course_id"].values
    result = state["course_catalog"][state["course_catalog"]["course_id"].isin(rec_ids)].copy()
    sim_map = {cmap.iloc[i]["course_id"]: round(1 - d, 4)
               for i, d in zip(sim_indices, sim_dists)}
    result["cosine_similarity"] = result["course_id"].map(sim_map)
    result["rank"] = result["course_id"].apply(lambda x: list(rec_ids).index(x))
    return result.sort_values("rank").drop(columns="rank").reset_index(drop=True)


def recommend_user_based(user_id: int, top_n: int, state: dict) -> pd.DataFrame:
    """Model 2: User-Based Collaborative Filtering."""
    uim = state["user_item_matrix"]
    if user_id not in uim.index:
        return pd.DataFrame()
    user_vec = uim.loc[user_id].values.reshape(1, -1)
    distances, indices = state["user_knn"].kneighbors(user_vec)
    similar_users = uim.iloc[indices[0][1:]]
    mean_scores = similar_users.mean(axis=0)
    already_seen = uim.loc[user_id] > 0
    recs = mean_scores[~already_seen].sort_values(ascending=False).head(top_n)
    result = state["course_catalog"][state["course_catalog"]["course_id"].isin(recs.index)].copy()
    result["predicted_score"] = result["course_id"].map(recs.to_dict())
    return result.sort_values("predicted_score", ascending=False).reset_index(drop=True)


def recommend_svd(user_id: int, top_n: int, state: dict) -> pd.DataFrame:
    """Model 3: SVD Matrix Factorization."""
    prd = state["predicted_ratings_df"]
    uim = state["user_item_matrix"]
    if user_id not in prd.index:
        return pd.DataFrame()
    user_preds = prd.loc[user_id]
    already_taken = uim.loc[user_id] > 0
    recs = user_preds[~already_taken].sort_values(ascending=False).head(top_n)
    result = state["course_catalog"][state["course_catalog"]["course_id"].isin(recs.index)].copy()
    result["predicted_rating"] = result["course_id"].map(recs.to_dict())
    return result.sort_values("predicted_rating", ascending=False).reset_index(drop=True)


def recommend_hybrid(user_id: int, top_n: int, alpha: float, state: dict) -> pd.DataFrame:
    """Model 4: Hybrid CF + CBF."""
    uim = state["user_item_matrix"]
    cmap = state["course_index_map"]
    X = state["X_courses"]
    knn = state["user_knn"]
    catalog = state["course_catalog"]

    if user_id not in uim.index:
        return pd.DataFrame()

    # CF scores
    user_vec = uim.loc[user_id].values.reshape(1, -1)
    distances, indices = knn.kneighbors(user_vec)
    similar_users = uim.iloc[indices[0][1:]]
    cf_scores = similar_users.mean(axis=0)

    # CBF scores
    rated = uim.loc[user_id][uim.loc[user_id] > 0].index.tolist()
    cbf_scores_dict: dict = {}
    for cid in rated:
        if cid not in cmap["course_id"].values:
            continue
        idx = cmap[cmap["course_id"] == cid].index[0]
        dists, inds = state["knn_model"].kneighbors(X[idx].reshape(1, -1), n_neighbors=11)
        for d, i in zip(dists[0][1:], inds[0][1:]):
            ncid = cmap.iloc[i]["course_id"]
            cbf_scores_dict[ncid] = cbf_scores_dict.get(ncid, 0) + (1 - d)

    cbf_series = pd.Series(cbf_scores_dict)
    if len(cbf_series) > 0:
        cbf_series = (cbf_series - cbf_series.min()) / (cbf_series.max() - cbf_series.min() + 1e-9)
    if len(cf_scores) > 0:
        cf_norm = (cf_scores - cf_scores.min()) / (cf_scores.max() - cf_scores.min() + 1e-9)
    else:
        cf_norm = cf_scores

    all_ids = set(cf_norm.index) | set(cbf_series.index)
    hybrid: dict = {}
    for cid in all_ids:
        cf_val = float(cf_norm.get(cid, 0))
        cbf_val = float(cbf_series.get(cid, 0))
        hybrid[cid] = alpha * cf_val + (1 - alpha) * cbf_val

    already_seen = set(uim.loc[user_id][uim.loc[user_id] > 0].index)
    hybrid_filtered = {k: v for k, v in hybrid.items() if k not in already_seen}
    top_ids = sorted(hybrid_filtered, key=hybrid_filtered.get, reverse=True)[:top_n]

    result = catalog[catalog["course_id"].isin(top_ids)].copy()
    result["hybrid_score"] = result["course_id"].map(hybrid_filtered)
    return result.sort_values("hybrid_score", ascending=False).reset_index(drop=True)


def recommend_tfidf(course_id: int, top_n: int, state: dict) -> pd.DataFrame:
    """Model 5: TF-IDF Content-Based."""
    dp = state["df_processed"]
    row = dp[dp["course_id"] == course_id]
    if row.empty:
        return pd.DataFrame()
    course_name = row["course_name"].iloc[0]
    ci = state["tfidf_course_index"]
    if course_name not in ci.index:
        return pd.DataFrame()
    idx = ci[course_name]
    cosine_sim = linear_kernel(state["tfidf_matrix"][idx], state["tfidf_matrix"]).flatten()
    sim_indices = cosine_sim.argsort()[::-1][1: top_n + 1]
    rec_names = state["course_df"].iloc[sim_indices]["course_name"].values
    rec_sims = cosine_sim[sim_indices]
    result = state["course_catalog"][
        state["course_catalog"]["course_id"].isin(
            state["course_df"].iloc[sim_indices]["course_id"].values
        )
    ].copy()
    name_to_sim = dict(zip(rec_names, rec_sims))
    result["tfidf_similarity"] = result["course_id"].apply(
        lambda cid: float(
            name_to_sim.get(
                state["df_processed"][state["df_processed"]["course_id"] == cid]["course_name"].iloc[0]
                if not state["df_processed"][state["df_processed"]["course_id"] == cid].empty
                else "", 0
            )
        )
    )
    return result.sort_values("tfidf_similarity", ascending=False).reset_index(drop=True)


# ── Course Atlas header and navigation ────────────────────────────────────────
PAGES = ["Overview", "Insights", "Similar courses", "For a learner"]
NAV_KEY = "course_atlas_navigation"

# Set before the widget exists (allowed), so first load and bad states both land on Overview.
if st.session_state.get(NAV_KEY) not in PAGES:
    st.session_state[NAV_KEY] = PAGES[0]


def go_to(page_name: str) -> None:
    """on_click callback: callbacks run before the widget is built, so changing its state is allowed."""
    st.session_state[NAV_KEY] = page_name


def keep_page_selected() -> None:
    """The segmented control lets users click the active tab to deselect it; put it back."""
    if st.session_state.get(NAV_KEY) is None:
        st.session_state[NAV_KEY] = st.session_state.get("_atlas_last_page", PAGES[0])


with st.container(key="atlas_topbar"):
    brand_col, nav_col = st.columns([1, 2.6], vertical_alignment="bottom")
    with brand_col:
        st.markdown(
            '<div class="atlas-brand"><span class="atlas-mark" aria-hidden="true"></span>'
            '<span class="atlas-wordmark">Course Atlas</span></div>',
            unsafe_allow_html=True,
        )
    with nav_col:
        with st.container(key="atlas_nav"):
            st.segmented_control(
                "Main navigation",
                options=PAGES,
                label_visibility="collapsed",
                key=NAV_KEY,
                on_change=keep_page_selected,
            )

page = st.session_state.get(NAV_KEY) or PAGES[0]
st.session_state["_atlas_last_page"] = page

# Load the project's bundled dataset automatically.
# Keep the CSV beside app.py so the app works from any terminal working directory.
DATASET_PATH = Path(__file__).resolve().parent / "online_course_recommendation_v2.csv"

state = None
if DATASET_PATH.exists():
    try:
        with st.spinner("Loading course data and preparing recommendation models…"):
            state = load_and_preprocess(str(DATASET_PATH))
    except Exception as exc:
        st.error("The bundled dataset could not be loaded.")
        st.code(f"{type(exc).__name__}: {exc}")
        st.stop()
else:
    st.error(
        "Dataset file not found. Place "
        "`online_course_recommendation_v2.csv` in the same folder as `app.py`."
    )
    st.stop()

# ── PAGE: OVERVIEW ────────────────────────────────────────────────────────────
if page == "Overview":
    with st.container(key="atlas_page_overview"):
        df = state["df"]
        catalog = state["course_catalog"]

        intro_col, stats_col = st.columns([1.6, 1], gap="large")
        with intro_col:
            page_header(
                "Course Atlas · Discovery",
                "Find a course worth your time.",
                "Explore related courses or use learner activity to surface options you may want to study next.",
            )
        with stats_col:
            st.metric("Courses loaded", f"{catalog['course_id'].nunique():,}")
            s1, s2 = st.columns(2)
            s1.metric("Learners loaded", f"{df['user_id'].nunique():,}")
            s2.metric("Interactions loaded", f"{len(df):,}")

        section_head("Two ways to explore", "Choose a route to get started")
        left, right = st.columns(2, gap="large")
        with left:
            st.markdown(
                '<div class="route"><div class="small-label">Course-based</div>'
                '<div class="route-title">Find related courses</div>'
                '<p class="route-text">Start with a course you know and compare similar options '
                "using content features or text similarity.</p></div>",
                unsafe_allow_html=True,
            )
            st.button(
                "Browse similar courses", key="home_to_similar", type="primary",
                on_click=go_to, args=("Similar courses",),
            )
        with right:
            st.markdown(
                '<div class="route"><div class="small-label">Learner-based</div>'
                '<div class="route-title">Explore learner recommendations</div>'
                "<p class=\"route-text\">Use a learner's course history to generate recommendations "
                "from the existing collaborative and hybrid models.</p></div>",
                unsafe_allow_html=True,
            )
            st.button(
                "View learner recommendations", key="home_to_learner",
                on_click=go_to, args=("For a learner",),
            )

        section_head("Highest-rated courses", "Top four by average rating in the loaded data")
        preview = catalog.sort_values("rating", ascending=False).head(4)
        cols = st.columns(4, gap="large")
        for col, (_, course) in zip(cols, preview.iterrows()):
            with col:
                st.markdown(
                    f'<div class="course-item"><div class="small-label">{esc(course["difficulty_level"])}</div>'
                    f'<div class="course-item-title">{esc(course["course_name"])}</div>'
                    f'<div class="course-item-meta">Rating <strong>{float(course["rating"]):.1f}</strong><br>'
                    f'{"Certificate available" if has_certificate(course["certification_offered"]) else "No certificate"}'
                    f' · {money(course["course_price"])}<br>ID {esc(course["course_id"])}</div></div>',
                    unsafe_allow_html=True,
                )


# ── PAGE: INSIGHTS ────────────────────────────────────────────────────────────
elif page == "Insights":
    with st.container(key="atlas_page_insights"):
        page_header(
            "Exploratory analysis",
            "Dataset insights",
            "A closer look at course ratings, engagement and the shape of the dataset.",
        )

        df = state["df"]
        a, b, c = st.columns(3, gap="large")
        a.metric("Records", f"{len(df):,}")
        b.metric("Unique courses", f"{df['course_id'].nunique():,}")
        c.metric("Unique learners", f"{df['user_id'].nunique():,}")

        st.write("")
        tab1, tab2, tab3, tab4 = st.tabs(
            ["Numeric distribution", "Correlations", "Course categories", "PCA"]
        )
        with tab1:
            col_sel = st.selectbox("Feature", NUMERICAL_FEATURES)
            fig, axes = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={"width_ratios": [2.2, 1]})
            axes[0].hist(df[col_sel].dropna(), bins=35, color=PALETTE["teal"], edgecolor=PALETTE["surface"])
            axes[0].axvline(df[col_sel].mean(), color=PALETTE["gold"], linestyle="--", linewidth=2, label="Mean")
            axes[0].axvline(df[col_sel].median(), color=PALETTE["ink"], linestyle=":", linewidth=2, label="Median")
            axes[0].set_title(f"{col_sel.replace('_', ' ').title()} distribution")
            axes[0].grid(axis="x", visible=False)
            axes[0].legend()
            axes[1].boxplot(
                df[col_sel].dropna(), vert=True, patch_artist=True,
                boxprops=dict(facecolor=PALETTE["sage"], color=PALETTE["teal_strong"]),
                medianprops=dict(color=PALETTE["gold"], linewidth=2.5),
                whiskerprops=dict(color=PALETTE["teal_strong"]),
                capprops=dict(color=PALETTE["teal_strong"]),
                flierprops=dict(marker="o", markerfacecolor="none", markeredgecolor=PALETTE["text2"], markersize=4),
            )
            axes[1].set_title("Spread and outliers")
            axes[1].grid(axis="x", visible=False)
            axes[1].set_xticks([])
            fig.tight_layout()
            show_figure(fig)
            x, y, z = st.columns(3, gap="large")
            x.metric("Mean", f"{df[col_sel].mean():.2f}")
            y.metric("Median", f"{df[col_sel].median():.2f}")
            z.metric("Missing values", f"{df[col_sel].isna().sum():,}")

        with tab2:
            st.subheader("How numeric features move together")
            st.caption("Pearson correlation between the numeric course and learner features. Values near ±1 move together; near 0 are unrelated.")
            corr = df[NUMERICAL_FEATURES].corr()
            pretty = {c: c.replace("_", " ") for c in NUMERICAL_FEATURES}
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(
                corr.rename(index=pretty, columns=pretty), annot=True, fmt=".2f",
                cmap=sns.diverging_palette(178, 55, s=55, l=60, sep=20, center="light", as_cmap=True),
                center=0, vmin=-1, vmax=1, linewidths=1.5, linecolor=PALETTE["bg"],
                ax=ax, square=True, cbar_kws={"shrink": 0.8},
                annot_kws={"fontsize": 9, "color": PALETTE["ink"]},
            )
            ax.grid(False)
            ax.tick_params(axis="x", rotation=35, length=0)
            ax.tick_params(axis="y", rotation=0, length=0)
            for spine in ax.spines.values():
                spine.set_visible(False)
            fig.tight_layout()
            show_figure(fig)

        with tab3:
            cat_sel = st.selectbox(
                "Category",
                ["difficulty_level", "certification_offered", "study_material_available"],
            )
            vc = df[cat_sel].value_counts()
            fig, ax = plt.subplots(figsize=(10, 4))
            bars = ax.bar(vc.index.astype(str), vc.values, color=PALETTE["teal"], width=0.6)
            ax.bar_label(bars, fmt="%d", padding=3, color=PALETTE["ink"], fontsize=10)
            ax.set_title(f"Records by {cat_sel.replace('_', ' ')}")
            ax.set_ylabel("Records")
            ax.grid(axis="x", visible=False)
            ax.margins(y=0.12)
            fig.tight_layout()
            show_figure(fig)

        with tab4:
            st.subheader("Principal component variance")
            st.caption("Cumulative share of variance explained as components are added; the dashed line marks 95%.")
            ev = state["pca_full"].explained_variance_ratio_
            cv = np.cumsum(ev)
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(range(1, len(cv) + 1), cv, marker="o", color=PALETTE["teal_strong"], linewidth=2)
            ax.axhline(0.95, color=PALETTE["gold"], linestyle="--", linewidth=2, label="95% variance")
            ax.set_xlabel("Number of components")
            ax.set_ylabel("Cumulative explained variance")
            ax.set_title("Explained variance by component count")
            ax.legend()
            fig.tight_layout()
            show_figure(fig)


# ── PAGE: SIMILAR COURSES ─────────────────────────────────────────────────────
elif page == "Similar courses":
    with st.container(key="atlas_page_similar"):
        page_header(
            "Course discovery",
            "Find similar courses",
            "Choose a starting course and compare it with courses that share similar features or descriptions.",
        )

        catalog = state["course_catalog"].copy()
        catalog["course_label"] = catalog.apply(
            lambda r: f"{r['course_name']}  ·  {r['difficulty_level']}  ·  ID {r['course_id']}",
            axis=1,
        )
        control_col, summary_col = st.columns([1, 1.5], gap="large")
        with control_col:
            st.subheader("Set your starting point")
            model_choice = st.selectbox(
                "Recommendation method",
                ["Content similarity (KNN)", "Text similarity (TF-IDF)"],
            )
            top_n = st.slider("Number of results", 3, 15, 5)
            course_label = st.selectbox("Starting course", catalog["course_label"].tolist())
            course_id = catalog.loc[catalog["course_label"] == course_label, "course_id"].iloc[0]
            run_recommendations = st.button(
                "Find similar courses", type="primary", **STRETCH
            )

        sel = catalog[catalog["course_id"] == course_id].iloc[0]
        with summary_col:
            st.subheader("Starting course")
            st.markdown(
                f'<div class="selected-course"><div class="small-label">Selected · {esc(sel["difficulty_level"])}</div>'
                f'<div class="selected-title">{esc(sel["course_name"])}</div>'
                f'<div class="result-meta">{course_meta(sel)}</div></div>',
                unsafe_allow_html=True,
            )

        if run_recommendations:
            with st.spinner("Finding matching courses…"):
                if model_choice == "Content similarity (KNN)":
                    result = recommend_knn(course_id, top_n, state)
                    score_col = "cosine_similarity"
                else:
                    result = recommend_tfidf(course_id, top_n, state)
                    score_col = "tfidf_similarity"

            st.session_state["atlas_similar_result"] = result
            st.session_state["atlas_similar_score_col"] = score_col
            st.session_state["atlas_similar_course_id"] = course_id
            st.session_state["atlas_similar_method"] = model_choice
            st.session_state["atlas_similar_course_name"] = str(sel["course_name"])

        if st.session_state.get("atlas_similar_course_id") == course_id and "atlas_similar_result" in st.session_state:
            result = st.session_state["atlas_similar_result"]
            score_col = st.session_state["atlas_similar_score_col"]
            method_used = st.session_state.get("atlas_similar_method", "")
            section_head(
                f"Similar to {st.session_state.get('atlas_similar_course_name', sel['course_name'])}",
                f"{method_used} · ranked by similarity score" if method_used else "Ranked by the selected model",
            )
            if result.empty:
                st.warning("No recommendations were returned for this course. Try another starting course.")
            else:
                render_ranked_results(result, score_col, "Similarity")
                results_table(result, score_col)
        else:
            empty_state(
                "No results yet",
                "Choose a starting course, then select “Find similar courses” to see ranked matches here.",
            )


# ── PAGE: LEARNER RECOMMENDATIONS ────────────────────────────────────────────
elif page == "For a learner":
    with st.container(key="atlas_page_learner"):
        page_header(
            "Personal recommendations",
            "Recommendations for a learner",
            "Use the course history in the dataset to find options the selected learner has not rated yet.",
        )

        uim = state["user_item_matrix"]
        control_col, profile_col = st.columns([1, 1.5], gap="large")
        with control_col:
            st.subheader("Choose a learner")
            model_choice = st.selectbox(
                "Recommendation method",
                [
                    "Similar learners (KNN)",
                    "SVD factorization",
                    "Hybrid model",
                ],
            )
            # The weight is only meaningful for the hybrid model. It lives outside the button
            # block so it stays visible and its value is read when the button is pressed.
            alpha_val = ALPHA
            if model_choice == "Hybrid model":
                alpha_val = st.slider(
                    "Collaborative filtering weight",
                    0.0, 1.0, ALPHA, key="atlas_hybrid_alpha",
                    help="Share of the hybrid score that comes from similar learners; the remainder comes from content similarity.",
                )
            top_n = st.slider("Number of results", 3, 15, 5)
            user_ids = sorted(uim.index.tolist())
            user_id = st.selectbox("Learner ID", user_ids)
            user_history = uim.loc[user_id]
            taken = user_history[user_history > 0]
            generate = st.button("Generate recommendations", type="primary", **STRETCH)

        with profile_col:
            st.subheader("Selected learner")
            a, b = st.columns(2, gap="large")
            a.metric("Learner", str(user_id))
            b.metric("Courses rated", f"{len(taken):,}")
            with st.expander("View this learner's rating history"):
                rated_courses = taken.reset_index()
                rated_courses.columns = ["course_id", "rating"]
                merged = rated_courses.merge(state["course_catalog"], on="course_id", how="left")
                st.dataframe(
                    merged[["course_id", "course_name", "difficulty_level", "rating_x"]].rename(
                        columns={"rating_x": "learner_rating"}
                    ),
                    **STRETCH,
                )

        if generate:
            with st.spinner("Generating recommendations…"):
                if model_choice == "Similar learners (KNN)":
                    result = recommend_user_based(user_id, top_n, state)
                    score_col = "predicted_score"
                elif model_choice == "SVD factorization":
                    result = recommend_svd(user_id, top_n, state)
                    score_col = "predicted_rating"
                else:
                    result = recommend_hybrid(user_id, top_n, alpha_val, state)
                    score_col = "hybrid_score"

            st.session_state["atlas_user_result"] = result
            st.session_state["atlas_user_score_col"] = score_col
            st.session_state["atlas_user_id"] = user_id
            st.session_state["atlas_user_method"] = (
                f"{model_choice} · CF weight {alpha_val:.2f}" if model_choice == "Hybrid model" else model_choice
            )

        if st.session_state.get("atlas_user_id") == user_id and "atlas_user_result" in st.session_state:
            result = st.session_state["atlas_user_result"]
            score_col = st.session_state["atlas_user_score_col"]
            section_head(
                f"Suggested for learner {user_id}",
                st.session_state.get("atlas_user_method", "Based on the selected recommendation method"),
            )
            if result.empty:
                st.warning("No recommendations were returned for this learner.")
            else:
                render_ranked_results(result, score_col, "Model score")
                st.caption("Scores come from each method's own scale, so they are not comparable between methods.")
                results_table(result, score_col)
        else:
            empty_state(
                "No results yet",
                "Choose a method and learner, then select “Generate recommendations” to see ranked suggestions here.",
            )
