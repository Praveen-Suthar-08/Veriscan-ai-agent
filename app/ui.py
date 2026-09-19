"""UI components, custom CSS, and evidence visualizers for VeriScan Streamlit app."""

from __future__ import annotations
import io
from typing import List, Dict, Optional, Tuple, Any
import cv2  # type: ignore
import numpy as np  # type: ignore
from PIL import Image, ImageDraw, ImageFont  # type: ignore
import streamlit as st  # type: ignore

from veriscan.schemas import Finding, DISCLAIMER

# Custom Cyber-Command Design System with universal dark override & zero space wastage
CUSTOM_CSS = """
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');
    
    /* Force consistent high-contrast dark theme across all elements */
    html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stSidebar"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
        background-color: #070A13 !important;
        color: #F8FAFC !important;
    }

    /* Keyboard focus outlines for accessibility (WCAG AA) */
    *:focus-visible {
        outline: 2px solid #6366F1 !important;
        outline-offset: 2px !important;
    }

    /* Clean sidebar styling — hidden scrollbar, fits viewport */
    section[data-testid="stSidebar"] {
        background-color: #0B0F1D !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
        overflow-y: auto !important;
        overflow-x: hidden !important;
        scrollbar-width: none !important;
        -ms-overflow-style: none !important;
    }
    section[data-testid="stSidebar"]::-webkit-scrollbar {
        display: none !important;
    }

    /* Main container padding */
    .block-container {
        padding-top: 2.2rem !important;
        padding-bottom: 1.5rem !important;
        padding-left: 1.5rem !important;
        padding-right: 1.5rem !important;
        max-width: 98% !important;
    }

    /* Sidebar inner content: compact but not clipping text */
    section[data-testid="stSidebar"] > div:first-child,
    [data-testid="stSidebarContent"],
    [data-testid="stSidebarUserContent"] {
        padding-top: 0.4rem !important;
        padding-bottom: 0.5rem !important;
        padding-left: 0.8rem !important;
        padding-right: 0.8rem !important;
    }

    /* Gap between widget blocks — balanced, not too tight */
    section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
        gap: 0.45rem !important;
    }
    section[data-testid="stSidebar"] .element-container {
        margin-bottom: 4px !important;
        margin-top: 0 !important;
    }
    /* Collapse Streamlit spacer wrappers */
    section[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"] {
        padding-top: 0 !important;
        padding-bottom: 0 !important;
    }
    section[data-testid="stSidebar"] hr {
        margin-top: 6px !important;
        margin-bottom: 6px !important;
        border-color: rgba(255, 255, 255, 0.1) !important;
    }

    /* Radio nav — readable labels with proper tap targets */
    section[data-testid="stSidebar"] [data-testid="stRadio"] {
        margin-bottom: 4px !important;
    }
    section[data-testid="stSidebar"] [data-testid="stRadio"] label {
        padding-top: 4px !important;
        padding-bottom: 4px !important;
        line-height: 1.35 !important;
    }
    section[data-testid="stSidebar"] [data-testid="stRadio"] p {
        font-size: 13px !important;
        font-weight: 600 !important;
        margin: 0 !important;
    }

    /* Selectbox — compact height, label visible */
    section[data-testid="stSidebar"] .stSelectbox {
        margin-bottom: 2px !important;
    }
    section[data-testid="stSidebar"] .stSelectbox label {
        margin-bottom: 2px !important;
    }
    /* Text input — compact */
    section[data-testid="stSidebar"] .stTextInput {
        margin-bottom: 2px !important;
    }
    section[data-testid="stSidebar"] .stTextInput label {
        margin-bottom: 2px !important;
    }
    /* Checkbox */
    section[data-testid="stSidebar"] .stCheckbox {
        margin-bottom: 2px !important;
    }
    section[data-testid="stSidebar"] .stCheckbox label {
        line-height: 1.4 !important;
    }
    section[data-testid="stSidebar"] div[data-baseweb="select"] > div {
        min-height: 34px !important;
        padding-top: 3px !important;
        padding-bottom: 3px !important;
    }
    section[data-testid="stSidebar"] .stTextInput > div > div > input {
        padding-top: 5px !important;
        padding-bottom: 5px !important;
        font-size: 12px !important;
    }
    /* All label text: readable, not clipped, wraps properly */
    section[data-testid="stSidebar"] label p {
        font-size: 11px !important;
        font-weight: 600 !important;
        margin-bottom: 2px !important;
        line-height: 1.35 !important;
        white-space: normal !important;
        overflow: visible !important;
    }

    /* Standard text colors - WCAG AA Contrast Compliant */
    p, span, label, div {
        color: #CBD5E1;
    }
    h1, h2, h3, h4, h5, strong, b {
        color: #F8FAFC !important;
    }
    .helper-text {
        color: #94A3B8 !important;
        font-size: 12px;
    }

    /* Code & Mono */
    code, pre, .mono-font {
        font-family: 'JetBrains Mono', monospace !important;
        background: rgba(0, 0, 0, 0.45) !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 4px !important;
        padding: 2px 6px !important;
        color: #E2E8F0 !important;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0B0F1D !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
    }

    /* Form Controls & Inputs */
    div[data-baseweb="select"] > div, .stTextInput > div > div > input, .stFileUploader {
        background-color: #111827 !important;
        color: #F8FAFC !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        border-radius: 8px !important;
    }

    /* Animated status dot */
    .pulse-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #10B981;
        box-shadow: 0 0 8px #10B981;
        animation: pulse 2s infinite;
        display: inline-block;
    }
    @keyframes pulse {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
        70% { transform: scale(1.05); box-shadow: 0 0 0 8px rgba(16, 185, 129, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
    }

    /* Glass Panels */
    .glass-panel {
        background: rgba(17, 24, 39, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 14px;
        backdrop-filter: blur(12px);
    }

    /* 6-Stage How It Works Strip */
    .how-it-works-strip {
        display: grid;
        grid-template-columns: repeat(6, 1fr);
        gap: 8px;
        background: rgba(17, 24, 39, 0.65);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 8px 12px;
        margin-bottom: 16px;
    }
    .how-step {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 4px 6px;
    }
    .how-step-num {
        width: 22px;
        height: 22px;
        border-radius: 50%;
        background: rgba(79, 70, 229, 0.25);
        border: 1px solid #4F46E5;
        color: #A5B4FC;
        font-size: 11px;
        font-weight: 700;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
    }
    .how-step-text {
        font-size: 11px;
        font-weight: 700;
        color: #F1F5F9;
        line-height: 1.2;
    }
    .how-step-sub {
        font-size: 9px;
        color: #94A3B8;
        display: block;
    }

    /* Triage HUD Card */
    .triage-hud {
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 14px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        backdrop-filter: blur(16px);
        box-shadow: 0 6px 24px rgba(0, 0, 0, 0.3);
    }
    .triage-hud-green {
        background: linear-gradient(135deg, rgba(6, 78, 59, 0.5) 0%, rgba(15, 23, 42, 0.85) 100%);
        border: 1px solid rgba(16, 185, 129, 0.4);
        border-left: 5px solid #10B981;
    }
    .triage-hud-amber {
        background: linear-gradient(135deg, rgba(120, 53, 15, 0.5) 0%, rgba(15, 23, 42, 0.85) 100%);
        border: 1px solid rgba(245, 158, 11, 0.4);
        border-left: 5px solid #F59E0B;
    }
    .triage-hud-red {
        background: linear-gradient(135deg, rgba(127, 29, 29, 0.55) 0%, rgba(15, 23, 42, 0.85) 100%);
        border: 1px solid rgba(239, 68, 68, 0.45);
        border-left: 5px solid #EF4444;
    }

    /* Metric HUD Stat Badges */
    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 10px;
        margin-bottom: 14px;
    }
    .kpi-card {
        background: rgba(17, 24, 39, 0.75);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 10px;
        text-align: center;
        backdrop-filter: blur(12px);
    }
    .kpi-val {
        font-size: 20px;
        font-weight: 800;
        margin: 2px 0;
        font-family: 'JetBrains Mono', monospace;
    }
    .kpi-lbl {
        font-size: 10px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #94A3B8;
    }

    /* Verdict Chips — Strictly restricted colors */
    .chip {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }
    .chip-MATCH {
        background: rgba(16, 185, 129, 0.15);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.4);
    }
    .chip-MINOR_VARIANT {
        background: rgba(6, 182, 212, 0.15);
        color: #22D3EE;
        border: 1px solid rgba(6, 182, 212, 0.4);
    }
    .chip-MISMATCH {
        background: rgba(239, 68, 68, 0.15);
        color: #F87171;
        border: 1px solid rgba(239, 68, 68, 0.4);
    }
    .chip-LOW_CONFIDENCE {
        background: rgba(245, 158, 11, 0.15);
        color: #FBBF24;
        border: 1px solid rgba(245, 158, 11, 0.4);
    }
    .chip-MISSING {
        background: rgba(148, 163, 184, 0.15);
        color: #94A3B8;
        border: 1px solid rgba(148, 163, 184, 0.3);
    }

    /* Flag Card UI */
    .finding-card {
        background: rgba(17, 24, 39, 0.85);
        border: 1px solid rgba(255, 255, 255, 0.09);
        border-radius: 12px;
        padding: 16px 18px;
        margin-bottom: 14px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.25);
    }
    .finding-card:hover {
        border-color: rgba(99, 102, 241, 0.45);
    }
    .finding-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 10px;
    }
    .finding-field-name {
        font-size: 15px;
        font-weight: 700;
        color: #F8FAFC !important;
    }

    /* Prominent Single-Accent Primary Action Buttons */
    div.stButton > button {
        border-radius: 8px;
        font-weight: 600;
        font-size: 13px;
        transition: all 0.2s ease;
        padding: 8px 16px;
        border: 1px solid rgba(255, 255, 255, 0.15);
    }
    div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #2563EB 0%, #4F46E5 100%) !important;
        border: 1px solid rgba(255, 255, 255, 0.25) !important;
        color: #FFFFFF !important;
        box-shadow: 0 4px 14px rgba(79, 70, 229, 0.35) !important;
    }
    div.stButton > button[kind="primary"]:hover {
        box-shadow: 0 6px 20px rgba(79, 70, 229, 0.55) !important;
        transform: translateY(-1px);
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background-color: rgba(17, 24, 39, 0.6);
        padding: 4px;
        border-radius: 8px;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px;
        color: #94A3B8 !important;
        font-weight: 600;
        font-size: 13px;
        padding: 6px 14px;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(99, 102, 241, 0.25) !important;
        color: #FFFFFF !important;
        border: 1px solid rgba(99, 102, 241, 0.5) !important;
    }

    /* SLIM FULL-WIDTH FOOTER DISCLAIMER STRIP (REPLACES FLOATING OVERLAY) */
    .footer-disclaimer-strip {
        margin-top: 36px;
        border-top: 1px solid rgba(245, 158, 11, 0.3);
        background: rgba(15, 23, 42, 0.85);
        border-radius: 8px;
        padding: 10px 16px;
        display: flex;
        align-items: center;
        gap: 14px;
        font-size: 11px;
        color: #CBD5E1;
        line-height: 1.45;
    }
    .footer-disclaimer-title {
        font-weight: 700;
        color: #FBBF24;
        white-space: nowrap;
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .footer-disclaimer-text {
        font-size: 11px;
        color: #E2E8F0;
    }

    /* Scenario Card Styling */
    .scenario-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 12px;
        margin-bottom: 14px;
    }
    .scenario-card {
        background: rgba(17, 24, 39, 0.75);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 12px;
        cursor: pointer;
        transition: all 0.2s ease;
    }
    .scenario-card:hover {
        border-color: #6366F1;
        background: rgba(30, 41, 59, 0.85);
    }
    .scenario-card-active {
        border: 2px solid #4F46E5 !important;
        background: rgba(79, 70, 229, 0.15) !important;
        box-shadow: 0 0 14px rgba(79, 70, 229, 0.25);
    }
    .doc-tag {
        display: inline-block;
        background: rgba(255, 255, 255, 0.08);
        border-radius: 4px;
        padding: 2px 6px;
        font-size: 10px;
        margin: 2px 2px 0 0;
        color: #CBD5E1;
    }

    /* Narrow width responsiveness */
    @media (max-width: 900px) {
        .how-it-works-strip {
            grid-template-columns: repeat(2, 1fr) !important;
        }
        .scenario-grid {
            grid-template-columns: repeat(2, 1fr) !important;
        }
        .kpi-grid {
            grid-template-columns: repeat(2, 1fr) !important;
        }
        .footer-disclaimer-strip {
            flex-direction: column !important;
            align-items: flex-start !important;
        }
    }
</style>
"""


def apply_global_styles():
    """Inject global CSS rules."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def render_sidebar_brand():
    """Render top-left sidebar branding with website title and agent status with zero top gap."""
    st.sidebar.markdown("""
    <div style="margin-top: -6px; margin-bottom: 5px; border-bottom: 1px solid rgba(255, 255, 255, 0.08); padding-bottom: 5px;">
        <div style="font-size: 16px; font-weight: 800; color: #FFFFFF; display: flex; align-items: center; gap: 6px; letter-spacing: -0.4px;">
            <span>🛡️ VeriScan</span>
        </div>
        <div style="font-size: 10px; font-weight: 500; color: #818CF8; margin-top: 1px; line-height: 1.2;">
            Document &amp; Identity Consistency-Checking Agent
        </div>
        <div style="margin-top: 4px; background: rgba(16, 185, 129, 0.12); border: 1px solid rgba(16, 185, 129, 0.35); border-radius: 5px; padding: 3px 6px; display: flex; align-items: center; gap: 6px;">
            <span class="pulse-dot"></span>
            <div style="display: flex; align-items: center; gap: 6px;">
                <span style="font-size: 9px; font-weight: 700; color: #6EE7B7; text-transform: uppercase;">Agent:</span>
                <span style="font-size: 10px; font-weight: 700; color: #10B981;">Ready</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def trigger_scroll_to_top():
    """Inject smooth scroll to top on screen change."""
    st.components.v1.html("""
    <script>
        window.parent.document.querySelector('.main').scrollTo({top: 0, behavior: 'smooth'});
        window.scrollTo({top: 0, behavior: 'smooth'});
    </script>
    """, height=0)


def render_how_it_works_strip():
    """Render the 6-stage autonomous agent pipeline strip under the header."""
    st.markdown("""
    <div class="how-it-works-strip">
        <div class="how-step">
            <div class="how-step-num">1</div>
            <div>
                <span class="how-step-text">Quality & Deskew</span>
                <span class="how-step-sub">Blur, Skew, DPI</span>
            </div>
        </div>
        <div class="how-step">
            <div class="how-step-num">2</div>
            <div>
                <span class="how-step-text">Multi-OCR</span>
                <span class="how-step-sub">Word BBoxes</span>
            </div>
        </div>
        <div class="how-step">
            <div class="how-step-num">3</div>
            <div>
                <span class="how-step-text">Field Extract</span>
                <span class="how-step-sub">Label Proximity</span>
            </div>
        </div>
        <div class="how-step">
            <div class="how-step-num">4</div>
            <div>
                <span class="how-step-text">Normalization</span>
                <span class="how-step-sub">Explainable Rules</span>
            </div>
        </div>
        <div class="how-step">
            <div class="how-step-num">5</div>
            <div>
                <span class="how-step-text">Consistency</span>
                <span class="how-step-sub">Similarity + Gating</span>
            </div>
        </div>
        <div class="how-step">
            <div class="how-step-num">6</div>
            <div>
                <span class="how-step-text">Review Report</span>
                <span class="how-step-sub">Evidence & Audit</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_footer_disclaimer():
    """Render statutory disclaimer as a slim full-width footer strip on every page."""
    st.markdown(f"""
    <div class="footer-disclaimer-strip">
        <div class="footer-disclaimer-title">
            ⚖️ Screening aid, human review required
        </div>
        <div class="footer-disclaimer-text">
            {DISCLAIMER}
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_triage_banner(triage: str, risk_score: float, findings_count: int, is_cached: bool = False):
    """Render triage rating and heuristic risk score banner with command HUD styling."""
    cls_name = f"triage-hud-{triage.lower()}"
    triage_labels = {
        "GREEN": "CONSISTENT / PASS PRE-SCREENING",
        "AMBER": "ATTENTION REQUIRED / MINOR VARIANTS",
        "RED": "HIGH RISK / DISCREPANCIES DETECTED"
    }
    label = triage_labels.get(triage, triage)
    badge_colors = {
        "GREEN": "#10B981",
        "AMBER": "#F59E0B",
        "RED": "#EF4444"
    }
    color = badge_colors.get(triage, "#3B82F6")

    cached_badge = (
        '<span style="background: rgba(99, 102, 241, 0.25); border: 1px solid #4F46E5; color: #A5B4FC; font-size: 10px; font-weight: 700; padding: 2px 8px; border-radius: 4px; margin-left: 8px;">⚡ OCR from cache</span>'
        if is_cached else ''
    )

    st.markdown(f"""
    <div class="triage-hud {cls_name}">
        <div>
            <div style="font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.8px; color: {color}; display: flex; align-items: center;">
                Automated Screening Result {cached_badge}
            </div>
            <div style="font-size: 24px; font-weight: 800; margin-top: 2px; color: #FFFFFF; letter-spacing: -0.5px;">
                {triage} — {label}
            </div>
            <div style="font-size: 13px; margin-top: 2px; color: #CBD5E1;">
                Flagged Discrepancies: <strong style="color: #FFFFFF;">{findings_count}</strong>
            </div>
        </div>
        <div style="text-align: right; background: rgba(0, 0, 0, 0.35); padding: 10px 18px; border-radius: 10px; border: 1px solid rgba(255, 255, 255, 0.08);" title="Heuristic risk score computed across findings: 1 - prod(1 - w_i * conf_i). Not a statistical probability.">
            <div style="font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.8px; color: #94A3B8;">
                Case Risk Score ℹ️
            </div>
            <div style="font-size: 30px; font-weight: 800; color: {color}; font-family: 'JetBrains Mono', monospace; line-height: 1.1;">
                {risk_score:.2f}
            </div>
            <div style="font-size: 10px; color: #94A3B8;">*Heuristic score [0.0 - 1.0]</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_kpi_summary(triage: str, risk_score: float, findings_count: int, consistent_count: int, doc_count: int):
    """Render 4-card metric strip with key figures."""
    triage_colors = {"GREEN": "#10B981", "AMBER": "#F59E0B", "RED": "#EF4444"}
    color = triage_colors.get(triage, "#3B82F6")

    st.markdown(f"""
    <div class="kpi-grid">
        <div class="kpi-card">
            <div class="kpi-lbl">Screening Triage</div>
            <div class="kpi-val" style="color: {color};">{triage}</div>
            <div style="font-size: 10px; color: #94A3B8;">Pre-Screening Status</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-lbl">Risk Score</div>
            <div class="kpi-val" style="color: {color};">{risk_score:.2f}</div>
            <div style="font-size: 10px; color: #94A3B8;">Empirical Indicator</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-lbl">Discrepancies Flagged</div>
            <div class="kpi-val" style="color: {'#EF4444' if findings_count > 0 else '#10B981'};">{findings_count}</div>
            <div style="font-size: 10px; color: #94A3B8;">Requires Inspection</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-lbl">Reconciled Consistent</div>
            <div class="kpi-val" style="color: #34D399;">{consistent_count}</div>
            <div style="font-size: 10px; color: #94A3B8;">Across {doc_count} Documents</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def draw_bounding_boxes_on_image(
    image: np.ndarray,
    highlight_bbox: Optional[List[int]] = None,
    label: str = "FIELD",
    is_mismatch: bool = True
) -> Image.Image:
    """
    Draw clean high-contrast visual bounding boxes on an image for reviewer inspection.
    """
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)

    if highlight_bbox and len(highlight_bbox) == 4:
        x, y, w, h = highlight_bbox
        pad = 4
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(pil_img.width, x + w + pad), min(pil_img.height, y + h + pad)

        box_color = (239, 68, 68) if is_mismatch else (16, 185, 129)
        bg_tag = (239, 68, 68, 230) if is_mismatch else (16, 185, 129, 230)

        # Draw box outline
        for i in range(3):
            draw.rectangle([x0 - i, y0 - i, x1 + i, y1 + i], outline=box_color)

        # Draw label tag
        tag_text = f" {label.upper()} "
        draw.rectangle([x0, max(0, y0 - 18), x0 + len(tag_text) * 8 + 4, y0], fill=bg_tag)
        draw.text((x0 + 2, max(0, y0 - 16)), tag_text, fill=(255, 255, 255))

    return pil_img
