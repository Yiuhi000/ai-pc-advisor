"""
app.py
======
Phần mềm Tư vấn và Xây dựng Cấu hình Máy tính (AI PC Builder Advisor)
Môn học: Trí tuệ nhân tạo - Biểu diễn Tri thức & Giải thuật (CS 417)
Giao diện phong cách chuyên nghiệp (PCPartPicker / GearVN), Light Theme, 2 Tab duy nhất.
100% Tiếng Việt có dấu, không sử dụng Emoji.
"""

import base64
import re
import csv
import io
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

# Thêm thư mục hiện tại vào sys.path
sys.path.insert(0, str(Path(__file__).parent))

from ai_engine import (
    BranchAndBoundOptimizer,
    BruteForceOptimizer,
    BuildResult,
    ForwardChainingEngine,
    load_data,
    suggest_upgrades,
)
from knowledge_base import (
    RULES,
    USAGE_PROFILES,
    calculate_min_psu_wattage,
    check_r03_ram_capacity,
)
from nlp_assistant import (
    extract_upgrade_from_history,
    generate_build_review,
    generate_conversational_reply,
    get_gemini_api_key,
    is_upgrade_acceptance,
    is_upgrade_inquiry_after_acceptance,
    parse_user_request,
)

# ===========================================================================
# CẤU HÌNH TRANG STREAMLIT
# ===========================================================================
st.set_page_config(
    page_title="AI PC Builder Advisor",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ===========================================================================
# HÀM ĐỌC ẢNH VÀ CHUẨN BỊ TÀI NGUYÊN BANNER
# ===========================================================================
def get_image_base64(rel_path: str) -> str:
    """Đọc file ảnh và chuyển sang chuỗi base64."""
    p = Path(__file__).parent / rel_path
    if p.exists():
        with open(p, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    return ""

PC_CASE_B64 = get_image_base64("pc_case_banner.png")
ROCKET_B64 = get_image_base64("rocket_promo.png")
ROBOT_B64 = get_image_base64("robot_avatar.png")
BANNER_BG_B64 = (
    get_image_base64("assets/banner_bg.png")
    or get_image_base64("assets/banner_bg.jpg")
    or get_image_base64("banner_bg.png")
    or get_image_base64("pc_case_banner.png")
)

# ===========================================================================
# QUẢN LÝ THEME (DARK / LIGHT) & CSS VARIABLES ĐỒNG BỘ
# ===========================================================================
st.session_state.setdefault("theme", "dark")
current_theme = st.session_state.get("theme", "dark")
is_dark = (current_theme == "dark")

COMMON_CSS = """
/* 1. An han thanh header va dai trang tri mac dinh cua Streamlit */
header[data-testid="stHeader"] {
    display: none !important;
    height: 0 !important;
}
div[data-testid="stToolbar"] {
    display: none !important;
}
div[data-testid="stDecoration"] {
    display: none !important;
}
#MainMenu, footer {
    visibility: hidden !important;
    display: none !important;
}
[data-testid="stSidebar"] {
    display: none !important;
}

/* 2. Keo noi dung trang sat len tren */
.block-container, div[data-testid="stMainBlockContainer"] {
    padding-top: 1rem !important;
    margin-top: 0 !important;
    padding-bottom: 2rem !important;
    padding-left: 1.5rem !important;
    padding-right: 1.5rem !important;
    max-width: 1440px !important;
}
section[data-testid="stMain"] > div:first-child {
    padding-top: 0 !important;
}

/* Typography baseline */
.stApp {
    font-family: 'Inter', 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif !important;
    font-size: 15px !important;
}

/* Top Header Bar */
.top-header-left {
    display: flex;
    align-items: center;
    gap: 14px;
}
.top-logo-box {
    background: rgba(37, 99, 235, 0.22);
    border: 1px solid rgba(56, 189, 248, 0.4);
    border-radius: 10px;
    width: 42px;
    height: 42px;
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: 0 0 15px rgba(56, 189, 248, 0.25);
}
.top-app-title {
    font-size: 1.45rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    line-height: 1.2;
}
.top-app-subtitle {
    font-size: 0.92rem;
    margin-top: 3px;
}

/* Hero Banner Layout */
.hero-banner {
    width: 100%;
    min-height: 130px;
    border-radius: 16px;
    position: relative;
    overflow: hidden;
    padding: 28px 36px;
    border: 1px solid rgba(96, 165, 250, 0.25);
    display: flex;
    align-items: center;
    justify-content: space-between;
    box-shadow: 0 4px 25px rgba(30, 58, 138, 0.35);
    margin-bottom: 14px;
    box-sizing: border-box;
}
.hero-content {
    position: relative;
    z-index: 2;
    display: flex;
    flex-direction: column;
    justify-content: center;
}
.hero-title {
    font-size: 28px !important;
    font-weight: 700 !important;
    color: #FFFFFF !important;
    letter-spacing: -0.01em !important;
    margin: 0 0 6px 0 !important;
    line-height: 1.25 !important;
    text-shadow: 0 2px 10px rgba(0, 0, 0, 0.5);
}
.hero-subtitle {
    font-size: 14px !important;
    color: #CBD5E1 !important;
    font-weight: 500 !important;
    margin: 0 !important;
    text-shadow: 0 1px 6px rgba(0, 0, 0, 0.4);
}

/* Ep trang toan bo chu trong hero-banner o ca 2 che do (Item 3) */
.stApp .hero-banner h1,
.stApp .hero-banner h2,
.stApp .hero-banner p,
.stApp .hero-banner span {
    color: #FFFFFF !important;
}

/* Hero badge styling */
.hero-badge {
    position: absolute;
    right: 24px;
    bottom: 20px;
    border-radius: 9999px;
    padding: 6px 14px;
    font-size: 0.85rem;
    font-weight: 700;
    display: flex !important;
    align-items: center;
    gap: 8px;
    z-index: 3;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.25);
    opacity: 1 !important;
}

/* Labels */
label[data-testid="stWidgetLabel"] p, label[data-testid="stWidgetLabel"] span {
    font-size: 0.95rem !important;
    font-weight: 600 !important;
}

/* Buttons chung */
.stButton > button, div[data-testid="stFormSubmitButton"] > button {
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    padding: 4px 6px !important;
    min-height: 32px !important;
    white-space: nowrap !important;
    transition: all 0.18s ease-in-out !important;
}
.stButton > button[kind="primary"], div[data-testid="stFormSubmitButton"] > button[kind="primary"] {
    background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%) !important;
    border: 1px solid #3B82F6 !important;
    color: #FFFFFF !important;
    box-shadow: 0 0 15px rgba(37, 99, 235, 0.45) !important;
}

/* Cum nut tren cung */
.auto-complete-btn-wrap button {
    background: linear-gradient(90deg, #0EA5E9 0%, #2563EB 100%) !important;
    border: 1px solid rgba(56, 189, 248, 0.5) !important;
    box-shadow: 0 4px 15px rgba(14, 165, 233, 0.35) !important;
    height: 44px !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    color: #FFFFFF !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
}
.auto-complete-btn-wrap button::before {
    content: "";
    display: inline-block;
    width: 18px;
    height: 18px;
    margin-right: 8px;
    background-color: #FFFFFF;
    -webkit-mask: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Cpath d='M7.5 5.6L5 7l1.4-2.5L5 2l2.5 1.4L10 2 8.6 4.5 10 7 7.5 5.6zm12 9.8L22 14l-2.5 1.4L17 14l1.4 2.5L17 19l2.5-1.4L22 19l-1.4-2.5L22 14l-2.5 1.4zM22 2l-2.5 1.4L17 2l1.4 2.5L17 7l2.5-1.4L22 7l-1.4-2.5L22 2zm-7.63 7.82l-9.19 9.19c-.39.39-1.02.39-1.41 0l-1.41-1.41c-.39-.39-.39-1.02 0-1.41l9.19-9.19 2.82 2.82zm2.12-2.12c.78-.78.78-2.05 0-2.83l-.71-.71c-.78-.78-2.05-.78-2.83 0l-1.41 1.41 2.82 2.82 2.13-2.13z'/%3E%3C/svg%3E") no-repeat center / contain;
    mask: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Cpath d='M7.5 5.6L5 7l1.4-2.5L5 2l2.5 1.4L10 2 8.6 4.5 10 7 7.5 5.6zm12 9.8L22 14l-2.5 1.4L17 14l1.4 2.5L17 19l2.5-1.4L22 19l-1.4-2.5L22 14l-2.5 1.4zM22 2l-2.5 1.4L17 2l1.4 2.5L17 7l2.5-1.4L22 7l-1.4-2.5L22 2zm-7.63 7.82l-9.19 9.19c-.39.39-1.02.39-1.41 0l-1.41-1.41c-.39-.39-.39-1.02 0-1.41l9.19-9.19 2.82 2.82zm2.12-2.12c.78-.78.78-2.05 0-2.83l-.71-.71c-.78-.78-2.05-.78-2.83 0l-1.41 1.41 2.82 2.82 2.13-2.13z'/%3E%3C/svg%3E") no-repeat center / contain;
}
.reset-btn-wrap button {
    height: 44px !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
}
.reset-btn-wrap button::before {
    content: "";
    display: inline-block;
    width: 17px;
    height: 17px;
    margin-right: 6px;
    background-color: currentColor;
    -webkit-mask: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Cpath d='M17.65 6.35A7.958 7.958 0 0 0 12 4c-4.42 0-7.99 3.58-7.99 8s3.57 8 7.99 8c3.73 0 6.84-2.55 7.73-6h-2.08A5.99 5.99 0 0 1 12 18c-3.31 0-6-2.69-6-6s2.69-6 6-6c1.66 0 3.14.69 4.22 1.78L13 11h7V4l-2.35 2.35z'/%3E%3C/svg%3E") no-repeat center / contain;
    mask: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Cpath d='M17.65 6.35A7.958 7.958 0 0 0 12 4c-4.42 0-7.99 3.58-7.99 8s3.57 8 7.99 8c3.73 0 6.84-2.55 7.73-6h-2.08A5.99 5.99 0 0 1 12 18c-3.31 0-6-2.69-6-6s2.69-6 6-6c1.66 0 3.14.69 4.22 1.78L13 11h7V4l-2.35 2.35z'/%3E%3C/svg%3E") no-repeat center / contain;
}

/* Tabs: An duong ke mac dinh */
div[data-baseweb="tab-highlight"], div[data-baseweb="tab-border"] {
    display: none !important;
}

/* Tối ưu hiển thị dropdown cột Số lượng (SL) trong bảng linh kiện */
div[class*="st-key-select_qty"] div.react-aria-ComboBox div[role="group"],
div[class*="st-key-select_qty"] div[data-baseweb="select"] > div {
    padding-left: 6px !important;
    padding-right: 2px !important;
    min-height: 32px !important;
    height: 32px !important;
}
div[class*="st-key-select_qty"] input,
div[class*="st-key-select_qty"] div[data-baseweb="select"] span {
    padding-left: 2px !important;
    padding-right: 0px !important;
    text-align: center !important;
    font-size: 0.92rem !important;
    font-weight: 600 !important;
    overflow: visible !important;
}
div[class*="st-key-select_qty"] button {
    padding: 0 2px !important;
    min-width: 18px !important;
    width: 18px !important;
}
div[class*="st-key-select_qty"] svg,
div[class*="st-key-select_qty"] [data-baseweb="select"] svg {
    width: 14px !important;
    height: 14px !important;
    margin-left: 0 !important;
}

/* Bang linh kien */
.cat-icon-box {
    background: linear-gradient(135deg, #1D4ED8 0%, #3B82F6 100%) !important;
    border: 1px solid rgba(96, 165, 250, 0.4) !important;
    border-radius: 8px !important;
    width: 34px !important;
    height: 34px !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    flex-shrink: 0 !important;
    box-shadow: 0 2px 8px rgba(29, 78, 216, 0.35) !important;
}
.spec-badges-container {
    margin-top: 5px;
    display: flex;
    flex-wrap: wrap;
    gap: 5px;
    align-items: center;
}
.spec-badge {
    padding: 2px 8px;
    border-radius: 9999px;
    font-size: 0.82rem;
    font-weight: 500;
    display: inline-block;
    white-space: nowrap;
    line-height: 1.4;
}
.brand-badge {
    font-size: 0.82rem;
    padding: 2px 8px;
    border-radius: 4px;
    font-weight: 600;
    margin-left: 6px;
}
.total-price-display {
    font-size: 2.1rem !important;
    font-weight: 800 !important;
    margin: 4px 0 6px 0 !important;
    line-height: 1.2 !important;
    letter-spacing: -0.02em;
}

/* Bong bong chat */
.chat-msg-row {
    display: flex;
    margin-bottom: 12px;
    width: 100%;
}
.chat-msg-user-row {
    justify-content: flex-end;
}
.chat-msg-ai-row {
    justify-content: flex-start;
    align-items: flex-start;
    gap: 10px;
}
.chat-msg-user {
    border-radius: 14px 14px 2px 14px;
    padding: 10px 14px;
    font-size: 14.5px !important;
    line-height: 1.6 !important;
    font-weight: 400 !important;
    max-width: 82%;
    word-break: break-word;
    opacity: 1 !important;
}
.chat-msg-ai {
    border-radius: 14px 14px 14px 2px;
    padding: 12px 16px;
    font-size: 14.5px !important;
    line-height: 1.6 !important;
    font-weight: 400 !important;
    max-width: 86%;
    word-break: break-word;
    opacity: 1 !important;
}
.chat-msg-user *,
.chat-msg-ai * {
    font-size: 14.5px !important;
    line-height: 1.6 !important;
}
/* Co chu o nhap tin nhan & nut goi y nhanh */
[data-testid="stTextInputRootElement"] input,
.stApp input[placeholder*="Nhắn tin"] {
    font-size: 14px !important;
}
.stApp div[data-testid="stColumn"] button {
    font-size: 13.5px !important;
}
.chat-ai-avatar {
    width: 34px;
    height: 34px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
}
.quick-prompts-wrap button {
    border-radius: 9999px !important;
    font-size: 0.84rem !important;
    font-weight: 500 !important;
    padding: 5px 12px !important;
    transition: all 0.2s ease !important;
}

/* Buttons icon masks: send, csv, pdf, expander */
div[data-testid="stFormSubmitButton"] button::before {
    content: "";
    display: inline-block;
    width: 15px;
    height: 15px;
    margin-right: 6px;
    background-color: #FFFFFF;
    -webkit-mask: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Cpath d='M2.01 21L23 12 2.01 3 2 10l15 2-15 2z'/%3E%3C/svg%3E") no-repeat center / contain;
    mask: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Cpath d='M2.01 21L23 12 2.01 3 2 10l15 2-15 2z'/%3E%3C/svg%3E") no-repeat center / contain;
}
.csv-dl-wrap button::before {
    content: "";
    display: inline-block;
    width: 15px;
    height: 15px;
    margin-right: 6px;
    background-color: currentColor;
    -webkit-mask: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Cpath d='M19 9h-4V3H9v6H5l7 7 7-7zM5 18v2h14v-2H5z'/%3E%3C/svg%3E") no-repeat center / contain;
    mask: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Cpath d='M19 9h-4V3H9v6H5l7 7 7-7zM5 18v2h14v-2H5z'/%3E%3C/svg%3E") no-repeat center / contain;
}
.html-dl-wrap button::before {
    content: "";
    display: inline-block;
    width: 15px;
    height: 15px;
    margin-right: 6px;
    background-color: currentColor;
    -webkit-mask: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Cpath d='M19 8H5c-1.66 0-3 1.34-3 3v6h4v4h12v-4h4v-6c0-1.66-1.34-3-3-3zm-3 11H8v-5h8v5zm3-7c-.55 0-1-.45-1-1s.45-1 1-1 1 .45 1 1-.45 1-1 1zm-1-9H6v4h12V3z'/%3E%3C/svg%3E") no-repeat center / contain;
    mask: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Cpath d='M19 8H5c-1.66 0-3 1.34-3 3v6h4v4h12v-4h4v-6c0-1.66-1.34-3-3-3zm-3 11H8v-5h8v5zm3-7c-.55 0-1-.45-1-1s.45-1 1-1 1 .45 1 1-.45 1-1 1zm-1-9H6v4h12V3z'/%3E%3C/svg%3E") no-repeat center / contain;
}
details[data-testid="stExpander"] summary span p::before {
    content: "";
    display: inline-block;
    width: 16px;
    height: 16px;
    margin-right: 8px;
    vertical-align: -2px;
    background-color: currentColor;
    -webkit-mask: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Cpath d='M14 2H6c-1.1 0-1.99.9-1.99 2L4 20c0 1.1.89 2 1.99 2H18c1.1 0 2-.9 2-2V8l-6-6zm2 16H8v-2h8v2zm0-4H8v-2h8v2zm-3-5V3.5L18.5 9H13z'/%3E%3C/svg%3E") no-repeat center / contain;
    mask: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Cpath d='M14 2H6c-1.1 0-1.99.9-1.99 2L4 20c0 1.1.89 2 1.99 2H18c1.1 0 2-.9 2-2V8l-6-6zm2 16H8v-2h8v2zm0-4H8v-2h8v2zm-3-5V3.5L18.5 9H13z'/%3E%3C/svg%3E") no-repeat center / contain;
}

/* Thanh tien do Hieu nang */
.perf-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 7px;
    font-size: 0.86rem;
}
.perf-label {
    width: 65px;
    font-weight: 600;
}
.perf-track {
    flex: 1;
    height: 7px;
    border-radius: 9999px;
    margin: 0 10px;
    overflow: hidden;
}
.perf-fill {
    height: 100%;
    border-radius: 9999px;
    background: linear-gradient(90deg, #2563EB, #38BDF8);
}
.perf-val {
    width: 36px;
    text-align: right;
    font-weight: 700;
}

/* An iframe ho tro cuon */
div[data-testid="stCustomComponentV1"] {
    position: fixed !important;
    top: 0 !important;
    left: 0 !important;
    width: 0 !important;
    height: 0 !important;
    opacity: 0 !important;
    pointer-events: none !important;
    border: none !important;
    margin: 0 !important;
    padding: 0 !important;
    z-index: -9999 !important;
}
"""

if BANNER_BG_B64:
    COMMON_CSS += f"""
    .hero-banner {{
        background-image: linear-gradient(90deg, rgba(11, 24, 68, 0.96) 0%, rgba(20, 42, 112, 0.82) 45%, rgba(15, 23, 42, 0.15) 100%), url("data:image/png;base64,{BANNER_BG_B64}") !important;
        background-size: cover !important;
        background-position: right center !important;
        min-height: 130px !important;
        border-radius: 16px !important;
        padding: 28px 36px !important;
        position: relative !important;
        overflow: hidden !important;
    }}
    """
else:
    COMMON_CSS += """
    .hero-banner {
        background: linear-gradient(135deg, #0F2046 0%, #1E3A8A 55%, #4C1D95 100%) !important;
        min-height: 130px !important;
        border-radius: 16px !important;
        padding: 28px 36px !important;
        position: relative !important;
        overflow: hidden !important;
    }
    """

LIGHT_THEME_CSS = """
/* Hero badge trong Light Mode */
.hero-badge {
    background: #FFFFFF !important;
    color: #0F172A !important;
    border: 1px solid #CBD5E1 !important;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.15) !important;
}
.hero-badge span {
    color: #0F172A !important;
    -webkit-text-fill-color: #0F172A !important;
    font-weight: 700 !important;
}

/* Khung Tu van AI sac net trong Light Mode */
.stApp div[class*="st-emotion-cache-700m8t"]:has([data-testid="stLayoutWrapper"] div[class*="st-emotion-cache-1ez5325"]),
.stApp div[class*="st-emotion-cache-700m8t"]:has(div:has-text("Tư vấn AI")) {
    background-color: #FFFFFF !important;
    border: 1.5px solid #CBD5E1 !important;
    border-radius: 14px !important;
    box-shadow: 0 4px 14px rgba(15, 23, 42, 0.08) !important;
    opacity: 1 !important;
}
.stApp div[class*="st-emotion-cache-1ez5325"] {
    background-color: #F8FAFC !important;
    border: 1px solid #CBD5E1 !important;
    border-radius: 10px !important;
    opacity: 1 !important;
}
.chat-msg-ai {
    background-color: #FFFFFF !important;
    border: 1px solid #CBD5E1 !important;
    color: #0F172A !important;
    opacity: 1 !important;
}
.chat-msg-ai, .chat-msg-ai p, .chat-msg-ai span, .chat-msg-ai div {
    color: #0F172A !important;
    -webkit-text-fill-color: #0F172A !important;
}
.chat-msg-user {
    background-color: #2563EB !important;
    border: none !important;
    color: #FFFFFF !important;
    opacity: 1 !important;
}
.chat-msg-user, .chat-msg-user p, .chat-msg-user span {
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
}
:root, .stApp {
    color-scheme: light !important;
    --bg-page: #EEF4FB;
    --bg-card: #FFFFFF;
    --border-card: #DCE7F5;
    --shadow-card: 0 4px 16px rgba(15, 23, 42, 0.05);
    --text-main: #0F172A;
    --text-sub: #475569;
    --text-price: #2563EB;
    --bg-badge: #EFF6FF;
    --text-badge: #1E40AF;
    --border-badge: #DBEAFE;
    --brand-badge-bg: #EFF6FF;
    --brand-badge-color: #2563EB;
    --brand-badge-border: #BFDBFE;
    --bg-review-box: #FFFBEB;
    --border-review-box: #FDE68A;
    --text-review-box: #475569;
    --text-review-title: #B45309;
    --bg-tab: #E2EBF6;
    --border-tab: #DCE7F5;
    --text-tab: #334155;
    --hr-line: #DCE7F5;
    --bg-track: #E2E8F0;
    --perf-val-color: #0F172A;
    --bg-info-card: #F8FAFC;
    --border-info-card: #DCE7F5;
}
.stApp {
    background-color: #EEF4FB !important;
    color: #0F172A !important;
}
.top-app-title {
    color: #0F172A !important;
}
.top-app-subtitle {
    color: #475569 !important;
}

/* 1. Ép sáng toàn bộ Input, NumberInput, TextInput, Selectbox trong Light Mode */
.stApp input,
.stApp textarea,
.stApp [data-testid="stNumberInputField"],
.stApp [data-testid="stTextInputField"],
.stApp input[role="combobox"] {
    background-color: transparent !important;
    color: #0F172A !important;
    -webkit-text-fill-color: #0F172A !important;
}

.stApp [data-testid="stNumberInputContainer"],
.stApp [data-testid="stTextInputRootElement"],
.stApp div.react-aria-TextField,
.stApp div.react-aria-ComboBox div[role="group"],
.stApp div[data-testid="stSelectbox"] div[role="group"],
.stApp div[data-testid="stSelectbox"] > div,
.stApp div[data-baseweb="input"],
.stApp div[data-baseweb="base-input"],
.stApp div[data-baseweb="select"] > div {
    background-color: #FFFFFF !important;
    border: 1px solid #CBD5E1 !important;
    color: #0F172A !important;
    border-radius: 8px !important;
}

/* Nút bấm +/- trong NumberInput & mũi tên Selectbox */
.stApp [data-testid="stNumberInput"] button,
.stApp [data-testid="stSelectbox"] button,
.stApp div.react-aria-ComboBox button {
    background-color: #F1F5F9 !important;
    color: #0F172A !important;
    border: none !important;
}
.stApp [data-testid="stNumberInput"] button svg,
.stApp [data-testid="stSelectbox"] button svg,
.stApp div.react-aria-ComboBox button svg {
    fill: #334155 !important;
    color: #334155 !important;
}

/* 2. Checkbox trong Light Mode (Cả React Aria và BaseWeb) */
.stApp [data-testid="stCheckbox"] label > div:first-of-type,
.stApp [data-testid="stCheckbox"] label div[class*="e1hn2pbo5"],
.stApp [data-testid="stCheckbox"] div[class*="e1hn2pbo5"],
.stApp [data-testid="stCheckbox"] label > span:first-child {
    background-color: #FFFFFF !important;
    border: 1.5px solid #CBD5E1 !important;
    border-radius: 4px !important;
}
.stApp [data-testid="stCheckbox"] label svg polyline {
    stroke: #FFFFFF !important;
}
.stApp [data-testid="stCheckbox"] label:has(input:checked) > div:first-of-type,
.stApp [data-testid="stCheckbox"] label:has(input:checked) > span:first-child {
    background-color: #2563EB !important;
    border-color: #2563EB !important;
}
.stApp [data-testid="stCheckbox"] label p {
    color: #0F172A !important;
    font-weight: 500 !important;
}

/* 3. Dropdown Menu popover Selectbox */
div[role="listbox"],
div[data-baseweb="popover"] ul,
div[data-rac][role="listbox"],
ul[data-baseweb="menu"] {
    background-color: #FFFFFF !important;
    border: 1px solid #CBD5E1 !important;
}
div[role="listbox"] [role="option"],
div[data-rac][role="listbox"] div[role="option"],
ul[data-baseweb="menu"] li {
    background-color: #FFFFFF !important;
    color: #0F172A !important;
}
div[role="listbox"] [role="option"]:hover,
div[data-rac][role="listbox"] div[role="option"]:hover,
div[role="listbox"] [role="option"][data-focused="true"],
ul[data-baseweb="menu"] li:hover {
    background-color: #EFF6FF !important;
    color: #2563EB !important;
}

/* 4. Tab trong Light Mode */
.stApp div[role="tablist"],
.stApp div[data-baseweb="tab-list"] {
    background: #E2EBF6 !important;
    border: 1px solid #DCE7F5 !important;
    border-radius: 12px !important;
    padding: 5px !important;
    gap: 6px !important;
    width: fit-content !important;
    margin-bottom: 12px !important;
}
.stApp div[data-testid="stTab"],
.stApp div[role="tab"],
.stApp button[data-baseweb="tab"] {
    background: transparent !important;
    border: none !important;
    border-radius: 8px !important;
    opacity: 1 !important;
    color: #334155 !important;
}
.stApp div[data-testid="stTab"] p,
.stApp div[data-testid="stTab"] span,
.stApp div[data-testid="stTab"] div,
.stApp div[role="tab"] p,
.stApp div[role="tab"] span,
.stApp div[role="tab"] div,
.stApp button[data-baseweb="tab"] p,
.stApp button[data-baseweb="tab"] span {
    color: #334155 !important;
    -webkit-text-fill-color: #334155 !important;
    font-weight: 600 !important;
    opacity: 1 !important;
}
.stApp div[data-testid="stTab"][aria-selected="true"],
.stApp div[role="tab"][aria-selected="true"],
.stApp button[data-baseweb="tab"][aria-selected="true"] {
    background: #2563EB !important;
    border: 1px solid #2563EB !important;
    border-radius: 8px !important;
    box-shadow: 0 2px 8px rgba(37, 99, 235, 0.25) !important;
    opacity: 1 !important;
    color: #FFFFFF !important;
}
.stApp div[data-testid="stTab"][aria-selected="true"] p,
.stApp div[data-testid="stTab"][aria-selected="true"] span,
.stApp div[data-testid="stTab"][aria-selected="true"] div,
.stApp div[role="tab"][aria-selected="true"] p,
.stApp div[role="tab"][aria-selected="true"] span,
.stApp div[role="tab"][aria-selected="true"] div,
.stApp button[data-baseweb="tab"][aria-selected="true"] p,
.stApp button[data-baseweb="tab"][aria-selected="true"] span,
.stApp button[data-baseweb="tab"][aria-selected="true"] * {
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
    font-weight: 700 !important;
    opacity: 1 !important;
}

/* Nút chuyển chế độ Sáng/Tối & nút secondary trong Light Mode */
.stApp div[class*="st-key-btn_theme_toggle"] button,
.stApp button[key="btn_theme_toggle"],
.stApp button[kind="secondary"],
.stApp .stButton > button:not([kind="primary"]) {
    background-color: #FFFFFF !important;
    border: 1px solid #CBD5E1 !important;
    border-radius: 9999px !important;
}
.stApp div[class*="st-key-btn_theme_toggle"] button p,
.stApp div[class*="st-key-btn_theme_toggle"] button span,
.stApp button[key="btn_theme_toggle"] p,
.stApp button[key="btn_theme_toggle"] span,
.stApp button[kind="secondary"] p,
.stApp button[kind="secondary"] span,
.stApp .stButton > button:not([kind="primary"]) p,
.stApp .stButton > button:not([kind="primary"]) span {
    color: #0F172A !important;
    -webkit-text-fill-color: #0F172A !important;
    font-weight: 600 !important;
    display: inline-block !important;
}

/* 5. Card container và Khung Chat trong Light Mode */
.stApp div[class*="st-emotion-cache-700m8t"],
.stApp div[data-testid="stVerticalBlockBorderWrapper"],
.stApp [data-testid="stVerticalBlock"]:has(> [data-testid="stLayoutWrapper"] > div[class*="st-emotion-cache-1ez5325"]) {
    background-color: #FFFFFF !important;
    border: 1px solid #DCE7F5 !important;
    border-radius: 14px !important;
    box-shadow: 0 4px 16px rgba(15, 23, 42, 0.05) !important;
}
.stApp div[class*="st-emotion-cache-1ez5325"] {
    background-color: #F8FAFC !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 10px !important;
}

/* Bảng linh kiện Light Mode */
.table-header-wrap {
    background-color: #F4F8FD !important;
    border: 1px solid #DCE7F5 !important;
    border-radius: 8px !important;
    padding: 6px 8px !important;
    margin-bottom: 4px !important;
}
.spec-badge {
    background-color: #EFF6FF !important;
    border: 1px solid #DBEAFE !important;
    color: #1E40AF !important;
}
.status-pill-ok {
    background-color: #ECFDF5 !important;
    border: 1px solid #A7F3D0 !important;
    color: #065F46 !important;
}
.status-pill-warn {
    background-color: #FFFBEB !important;
    border: 1px solid #FDE68A !important;
    color: #92400E !important;
}
.status-pill-err {
    background-color: #FEF2F2 !important;
    border: 1px solid #FECACA !important;
    color: #991B1B !important;
}
.stat-pill {
    background-color: #EFF6FF !important;
    border: 1px solid #DBEAFE !important;
    color: #1E40AF !important;
}
.price-display {
    color: #2563EB !important;
}
.hero-banner {
    background: linear-gradient(135deg, #0F2046 0%, #1E3A8A 55%, #4C1D95 100%);
}
"""

DARK_THEME_CSS = """
/* Hero badge trong Dark Mode */
.hero-badge {
    background: rgba(15, 23, 42, 0.9) !important;
    color: #FFFFFF !important;
    border: 1px solid rgba(147, 197, 253, 0.35) !important;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.35) !important;
}
.hero-badge span {
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
    font-weight: 700 !important;
}

/* Khung Tu van AI sac net trong Dark Mode */
div[class*="st-emotion-cache-700m8t"]:has([data-testid="stLayoutWrapper"] div[class*="st-emotion-cache-1ez5325"]),
div[class*="st-emotion-cache-700m8t"]:has(div:has-text("Tư vấn AI")) {
    background-color: #16223A !important;
    border: 1.5px solid rgba(96, 165, 250, 0.35) !important;
    border-radius: 14px !important;
    opacity: 1 !important;
}
div[class*="st-emotion-cache-1ez5325"] {
    background-color: #0F172A !important;
    border: 1px solid rgba(148, 163, 184, 0.25) !important;
    border-radius: 10px !important;
    opacity: 1 !important;
}
.chat-msg-ai {
    background-color: #1E293B !important;
    border: 1px solid rgba(148, 163, 184, 0.3) !important;
    color: #F8FAFC !important;
    opacity: 1 !important;
}
.chat-msg-ai, .chat-msg-ai p, .chat-msg-ai span, .chat-msg-ai div {
    color: #F8FAFC !important;
    -webkit-text-fill-color: #F8FAFC !important;
}
.chat-msg-user {
    background-color: #2563EB !important;
    border: none !important;
    color: #FFFFFF !important;
    opacity: 1 !important;
}
.chat-msg-user, .chat-msg-user p, .chat-msg-user span {
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
}
:root, .stApp {
    color-scheme: dark !important;
    --bg-page: #0B1120;
    --bg-card: #111A2E;
    --border-card: rgba(59, 130, 246, 0.22);
    --shadow-card: 0 4px 20px rgba(0, 0, 0, 0.35);
    --text-main: #F1F5F9;
    --text-sub: #94A3B8;
    --text-price: #38BDF8;
    --bg-badge: #1E293B;
    --text-badge: #CBD5E1;
    --border-badge: #334155;
    --brand-badge-bg: rgba(37, 99, 235, 0.25);
    --brand-badge-color: #60A5FA;
    --brand-badge-border: rgba(59, 130, 246, 0.35);
    --bg-review-box: #131C31;
    --border-review-box: rgba(245, 158, 11, 0.25);
    --text-review-box: #F1F5F9;
    --text-review-title: #F59E0B;
    --bg-tab: #131C31;
    --border-tab: #1E293B;
    --text-tab: #94A3B8;
    --hr-line: rgba(148, 163, 184, 0.14);
    --bg-track: #1E293B;
    --perf-val-color: #E2E8F0;
    --bg-info-card: #131C31;
    --border-info-card: #1E293B;
}
.stApp {
    background-color: #0B1120 !important;
    color: #F1F5F9 !important;
}
.top-app-title {
    color: #F1F5F9 !important;
}
.top-app-subtitle {
    color: #94A3B8 !important;
}

/* Inputs & Widgets Dark Mode */
div[data-baseweb="input"],
div[data-baseweb="select"] > div {
    background-color: #0B1120 !important;
    border: 1px solid #1E293B !important;
    color: #F1F5F9 !important;
    border-radius: 8px !important;
}
div[data-baseweb="input"] input,
div[data-baseweb="select"] span {
    color: #F1F5F9 !important;
    -webkit-text-fill-color: #F1F5F9 !important;
    background-color: transparent !important;
}
div[data-baseweb="input"] button {
    color: #F1F5F9 !important;
    background-color: #131C31 !important;
}
div[data-baseweb="select"] svg {
    fill: #94A3B8 !important;
}
div[data-testid="stCheckbox"] label span {
    color: #F1F5F9 !important;
}

/* Tabs Dark Mode */
div[data-baseweb="tab-list"] {
    background: #111A2E !important;
    border: 1px solid rgba(59, 130, 246, 0.22) !important;
    border-radius: 12px !important;
    padding: 6px !important;
    gap: 8px !important;
    width: fit-content !important;
    margin-bottom: 12px !important;
}
button[data-baseweb="tab"] {
    background: transparent !important;
    border: 1px solid transparent !important;
    border-radius: 8px !important;
    color: #94A3B8 !important;
}
button[data-baseweb="tab"] * {
    color: #94A3B8 !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    background: linear-gradient(90deg, #1D4ED8 0%, #2563EB 100%) !important;
    border: 1px solid rgba(96, 165, 250, 0.55) !important;
    border-radius: 8px !important;
    color: #FFFFFF !important;
    font-weight: 600 !important;
    box-shadow: 0 2px 10px rgba(37, 99, 235, 0.4) !important;
}
button[data-baseweb="tab"][aria-selected="true"] * {
    color: #FFFFFF !important;
}

/* Nut chuyen che do Sang/Toi Dark Mode */
button[key="btn_theme_toggle"],
button[kind="secondary"] {
    background-color: #131C31 !important;
    border: 1px solid rgba(148, 163, 184, 0.3) !important;
}
button[key="btn_theme_toggle"] p,
button[key="btn_theme_toggle"] span,
button[kind="secondary"] p,
button[kind="secondary"] span {
    color: #F1F5F9 !important;
    -webkit-text-fill-color: #F1F5F9 !important;
    font-weight: 600 !important;
    display: inline-block !important;
}

/* Card Dark Mode */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: #111A2E !important;
    border: 1px solid rgba(59, 130, 246, 0.22) !important;
    border-radius: 12px !important;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.35) !important;
    padding: 14px 18px !important;
    margin-bottom: 12px !important;
}

/* Bang linh kien Dark Mode */
.spec-badge {
    background-color: #1E293B !important;
    color: #CBD5E1 !important;
    border: 1px solid #334155 !important;
}
.brand-badge {
    background: rgba(37, 99, 235, 0.25) !important;
    color: #60A5FA !important;
    border: 1px solid rgba(59, 130, 246, 0.35) !important;
}
.reset-btn-wrap button {
    background: #111A2E !important;
    border: 1px solid rgba(59, 130, 246, 0.22) !important;
    color: #F1F5F9 !important;
}
.reset-btn-wrap button p, .reset-btn-wrap button span {
    color: #F1F5F9 !important;
    -webkit-text-fill-color: #F1F5F9 !important;
}
.total-price-display {
    color: #38BDF8 !important;
}
.perf-label {
    color: #94A3B8 !important;
}
.perf-track {
    background: #1E293B !important;
}
.perf-val {
    color: #E2E8F0 !important;
}

/* Chat bong bong Dark Mode */
.chat-msg-user {
    background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%) !important;
    color: #FFFFFF !important;
}
.chat-msg-user p {
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
}
.chat-msg-ai {
    background-color: #1E293B !important;
    border: 1px solid #334155 !important;
    color: #F1F5F9 !important;
}
.chat-msg-ai p {
    color: #F1F5F9 !important;
    -webkit-text-fill-color: #F1F5F9 !important;
}
.chat-ai-avatar {
    background: #111A2E !important;
    border: 1px solid rgba(59, 130, 246, 0.3) !important;
}
.quick-prompts-wrap button {
    background: #1E293B !important;
    border: 1px solid #334155 !important;
    color: #94A3B8 !important;
}
.quick-prompts-wrap button p, .quick-prompts-wrap button span {
    color: #94A3B8 !important;
    -webkit-text-fill-color: #94A3B8 !important;
}
"""

# 1. Tach biet hoan toan CSS cua Dark Mode va Light Mode (Item 1)
if st.session_state.get("theme") == "light":
    st.markdown("<style>\n" + COMMON_CSS + "\n" + LIGHT_THEME_CSS + "\n</style>", unsafe_allow_html=True)
else:
    st.markdown("<style>\n" + COMMON_CSS + "\n" + DARK_THEME_CSS + "\n</style>", unsafe_allow_html=True)


# HỆ THỐNG BIỂU TƯỢNG SVG (SVG ICON SYSTEM)
# ===========================================================================
SVG_PATHS = {
    "ai_chip": '<path d="M4 4h16v16H4V4zm2 2v12h12V6H6zm3-4h2v2H9V2zm4 0h2v2h-2V2zm-8 7H2v2h3V9zm0 4H2v2h3v-2zm17-4h3v2h-3V9zm0 4h3v2h-3v-2zM9 20h2v2H9v-2zm4 0h2v2h-2v-2z"/>',
    "cpu": '<path d="M4 4h16v16H4V4zm2 2v12h12V6H6zm3-4h2v2H9V2zm4 0h2v2h-2V2zm-8 7H2v2h3V9zm0 4H2v2h3v-2zm17-4h3v2h-3V9zm0 4h3v2h-3v-2zM9 20h2v2H9v-2zm4 0h2v2h-2v-2z"/>',
    "cpus": '<path d="M4 4h16v16H4V4zm2 2v12h12V6H6zm3-4h2v2H9V2zm4 0h2v2h-2V2zm-8 7H2v2h3V9zm0 4H2v2h3v-2zm17-4h3v2h-3V9zm0 4h3v2h-3v-2zM9 20h2v2H9v-2zm4 0h2v2h-2v-2z"/>',
    "mainboard": '<path d="M3 3h18v18H3V3zm2 2v14h14V5H5zm2 2h3v3H7V7zm5 0h5v2h-5V7zm0 4h5v2h-5v-2zm-5 2h3v3H7v-3zm5 2h5v2h-5v-2z"/>',
    "mainboards": '<path d="M3 3h18v18H3V3zm2 2v14h14V5H5zm2 2h3v3H7V7zm5 0h5v2h-5V7zm0 4h5v2h-5v-2zm-5 2h3v3H7v-3zm5 2h5v2h-5v-2z"/>',
    "ram": '<path d="M2 7h20v10H2V7zm2 2v6h16V9H4zm2 1h2v4H6v-4zm4 0h2v4h-2v-4zm4 0h2v4h-2v-4zm4 0h2v4h-2v-4z"/>',
    "rams": '<path d="M2 7h20v10H2V7zm2 2v6h16V9H4zm2 1h2v4H6v-4zm4 0h2v4h-2v-4zm4 0h2v4h-2v-4zm4 0h2v4h-2v-4z"/>',
    "gpu": '<path d="M2 6h20v12H2V6zm2 2v8h16V8H4zm3 2a2 2 0 1 1 0 4 2 2 0 0 1 0-4zm8 0a2 2 0 1 1 0 4 2 2 0 0 1 0-4z"/>',
    "gpus": '<path d="M2 6h20v12H2V6zm2 2v8h16V8H4zm3 2a2 2 0 1 1 0 4 2 2 0 0 1 0-4zm8 0a2 2 0 1 1 0 4 2 2 0 0 1 0-4z"/>',
    "vga": '<path d="M2 6h20v12H2V6zm2 2v8h16V8H4zm3 2a2 2 0 1 1 0 4 2 2 0 0 1 0-4zm8 0a2 2 0 1 1 0 4 2 2 0 0 1 0-4z"/>',
    "ssd": '<path d="M4 5h16v14H4V5zm2 2v10h12V7H6zm2 2h8v2H8V9zm0 4h4v2H8v-2z"/>',
    "storages": '<path d="M4 5h16v14H4V5zm2 2v10h12V7H6zm2 2h8v2H8V9zm0 4h4v2H8v-2z"/>',
    "hdd": '<path d="M4 3h16a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2zm0 2v14h16V5H4zm8 2a5 5 0 1 1 0 10 5 5 0 0 1 0-10zm0 2a3 3 0 1 0 0 6 3 3 0 0 0 0-6zm4 8a1 1 0 1 1 0-2 1 1 0 0 1 0 2z"/>',
    "hdds": '<path d="M4 3h16a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2zm0 2v14h16V5H4zm8 2a5 5 0 1 1 0 10 5 5 0 0 1 0-10zm0 2a3 3 0 1 0 0 6 3 3 0 0 0 0-6zm4 8a1 1 0 1 1 0-2 1 1 0 0 1 0 2z"/>',
    "psu": '<path d="M16 7V3h-2v4h-4V3H8v4H6v6c0 3.31 2.69 6 6 6v3h2v-3c3.31 0 6-2.69 6-6V7h-4z"/>',
    "psus": '<path d="M16 7V3h-2v4h-4V3H8v4H6v6c0 3.31 2.69 6 6 6v3h2v-3c3.31 0 6-2.69 6-6V7h-4z"/>',
    "case": '<path d="M5 2h14a2 2 0 0 1 2 2v16a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2zm0 2v16h14V4H5zm2 2h10v2H7V6zm0 4h10v2H7v-2zm5 7a1.5 1.5 0 1 1 0-3 1.5 1.5 0 0 1 0 3z"/>',
    "cases": '<path d="M5 2h14a2 2 0 0 1 2 2v16a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2zm0 2v16h14V4H5zm2 2h10v2H7V6zm0 4h10v2H7v-2zm5 7a1.5 1.5 0 1 1 0-3 1.5 1.5 0 0 1 0 3z"/>',
    "cooler": '<path d="M12 2a10 10 0 1 0 10 10A10 10 0 0 0 12 2zm0 18a8 8 0 1 1 8-8 8 8 0 0 1-8 8zm-1-13h2v3.17a3 3 0 0 1 1.73 1l2.25-2.24 1.41 1.41-2.24 2.25A3 3 0 0 1 15.17 11H18.3v2h-3.17a3 3 0 0 1-1 1.73l2.25 2.25-1.41 1.41-2.25-2.24A3 3 0 0 1 13 15.17V18.3h-2v-3.17a3 3 0 0 1-1.73-1l-2.25 2.24-1.41-1.41 2.24-2.25A3 3 0 0 1 8.83 13H5.7v-2h3.17a3 3 0 0 1 1-1.73L7.62 7.02l1.41-1.41 2.25 2.24A3 3 0 0 1 11 8.83V5.7z"/>',
    "coolers": '<path d="M12 2a10 10 0 1 0 10 10A10 10 0 0 0 12 2zm0 18a8 8 0 1 1 8-8 8 8 0 0 1-8 8zm-1-13h2v3.17a3 3 0 0 1 1.73 1l2.25-2.24 1.41 1.41-2.24 2.25A3 3 0 0 1 15.17 11H18.3v2h-3.17a3 3 0 0 1-1 1.73l2.25 2.25-1.41 1.41-2.25-2.24A3 3 0 0 1 13 15.17V18.3h-2v-3.17a3 3 0 0 1-1.73-1l-2.25 2.24-1.41-1.41 2.24-2.25A3 3 0 0 1 8.83 13H5.7v-2h3.17a3 3 0 0 1 1-1.73L7.62 7.02l1.41-1.41 2.25 2.24A3 3 0 0 1 11 8.83V5.7z"/>',
    "cube": '<path d="M12 2L2 7l10 5 10-5-10-5z M2 17l10 5 10-5M2 12l10 5 10-5"/>',
    "sparkles": '<path d="M12 2l2.4 7.2L22 12l-7.6 2.8L12 22l-2.4-7.2L2 12l7.6-2.8z"/>',
    "robot": '<path d="M12 2a2 2 0 0 1 2 2c0 .74-.4 1.39-1 1.73V7h4a3 3 0 0 1 3 3v8a3 3 0 0 1-3 3H7a3 3 0 0 1-3-3v-8a3 3 0 0 1 3-3h4V5.73c-.6-.34-1-.99-1-1.73a2 2 0 0 1 2-2zM9 11a1.5 1.5 0 1 0 0 3 1.5 1.5 0 0 0 0-3zm6 0a1.5 1.5 0 1 0 0 3 1.5 1.5 0 0 0 0-3zm-6 5h6v1H9v-1z"/>',
    "send": '<path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/>',
    "chart": '<path d="M5 9.2h3V19H5zM10.6 5h2.8v14h-2.8zm5.6 8H19v6h-2.8z"/>',
    "lightning": '<path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z"/>',
    "shield": '<path d="M12 1L3 5v6c0 5.55 3.84 10.74 9 12 5.16-1.26 9-6.45 9-12V5l-9-4zm-2 16l-4-4 1.41-1.41L10 14.17l6.59-6.59L18 9l-8 8z"/>',
    "info": '<path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-6h2v6zm0-8h-2V7h2v2z"/>',
    "star": '<path d="M12 17.27L18.18 21l-1.64-7.03L22 9.24l-7.19-.61L12 2 9.19 8.63 2 9.24l5.46 4.73L5.82 21z"/>',
    "check_circle": '<path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/>',
    "download": '<path d="M19 9h-4V3H9v6H5l7 7 7-7zM5 18v2h14v-2H5z"/>',
    "document": '<path d="M19 8H5c-1.66 0-3 1.34-3 3v6h4v4h12v-4h4v-6c0-1.66-1.34-3-3-3zm-3 11H8v-5h8v5zm3-7c-.55 0-1-.45-1-1s.45-1 1-1 1 .45 1 1-.45 1-1 1zm-1-9H6v4h12V3z"/>',
    "rocket": '<path d="M12 2.5s4 3.5 4 8.5c0 2.2-.8 4.2-2 5.5l1.5 2.5-3.5-1-3.5 1 1.5-2.5c-1.2-1.3-2-3.3-2-5.5 0-5 4-8.5 4-8.5z M12 6.5a1.5 1.5 0 1 0 0 3 1.5 1.5 0 0 0 0-3z"/>',
    "wand": '<path d="M7.5 5.6L5 7l1.4-2.5L5 2l2.5 1.4L10 2 8.6 4.5 10 7 7.5 5.6zm12 9.8L22 14l-2.5 1.4L17 14l1.4 2.5L17 19l2.5-1.4L22 19l-1.4-2.5L22 14l-2.5 1.4zM22 2l-2.5 1.4L17 2l1.4 2.5L17 7l2.5-1.4L22 7l-1.4-2.5L22 2zm-7.63 7.82l-9.19 9.19c-.39.39-1.02.39-1.41 0l-1.41-1.41c-.39-.39-.39-1.02 0-1.41l9.19-9.19 2.82 2.82zm2.12-2.12c.78-.78.78-2.05 0-2.83l-.71-.71c-.78-.78-2.05-.78-2.83 0l-1.41 1.41 2.82 2.82 2.13-2.13z"/>',
    "wallet": '<path d="M21 18v1c0 1.1-.9 2-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h14c1.1 0 2 .9 2 2v1h-9a2 2 0 0 0-2 2v8a2 2 0 0 0 2 2h9zm-9-2h10V8H12v8zm4-2.5a1.5 1.5 0 1 1 0-3 1.5 1.5 0 0 1 0 3z"/>',
    "target": '<path d="M12 2a10 10 0 1 0 10 10A10 10 0 0 0 12 2zm0 18a8 8 0 1 1 8-8 8 8 0 0 1-8 8zm0-14a6 6 0 1 0 6 6 6 6 0 0 0-6-6zm0 10a4 4 0 1 1 4-4 4 4 0 0 1-4 4zm0-6a2 2 0 1 0 2 2 2 2 0 0 0-2-2z"/>',
    "layers": '<path d="M11.99 18.54l-7.37-5.73L3 14.07l9 7 9-7-1.63-1.27-7.38 5.74zM12 16l7.36-5.73L21 9.07l-9-7-9 7 1.63 1.2L12 16z"/>',
    "gear": '<path d="M19.14 12.94c.04-.3.06-.61.06-.94 0-.32-.02-.64-.07-.94l2.03-1.58a.49.49 0 0 0 .12-.61l-1.92-3.32a.49.49 0 0 0-.59-.22l-2.39.96c-.5-.38-1.03-.7-1.62-.94l-.36-2.54a.48.48 0 0 0-.48-.41h-3.84c-.24 0-.43.17-.47.41l-.36 2.54c-.59.24-1.13.57-1.62.94l-2.39-.96c-.22-.08-.47 0-.59.22L2.74 8.87c-.12.21-.08.47.12.61l2.03 1.58c-.05.3-.09.63-.09.94s.02.64.07.94l-2.03 1.58a.49.49 0 0 0-.12.61l1.92 3.32c.12.22.37.29.59.22l2.39-.96c.5.38 1.03.7 1.62.94l.36 2.54c.05.24.24.41.48.41h3.84c.24 0 .44-.17.47-.41l.36-2.54c.59-.24 1.13-.56 1.62-.94l2.39.96c.22.08.47 0 .59-.22l1.92-3.32c.12-.22.07-.47-.12-.61l-2.01-1.58zM12 15.6c-1.98 0-3.6-1.62-3.6-3.6s1.62-3.6 3.6-3.6 3.6 1.62 3.6 3.6-1.62 3.6-3.6 3.6z"/>',
    "rotate": '<path d="M17.65 6.35A7.958 7.958 0 0 0 12 4c-4.42 0-7.99 3.58-7.99 8s3.57 8 7.99 8c3.73 0 6.84-2.55 7.73-6h-2.08A5.99 5.99 0 0 1 12 18c-3.31 0-6-2.69-6-6s2.69-6 6-6c1.66 0 3.14.69 4.22 1.78L13 11h7V4l-2.35 2.35z"/>',
}

def get_svg_icon(name: str, color: str = "#FFFFFF", size: int = 18) -> str:
    """Trả về chuỗi thẻ SVG với tên icon, màu sắc và kích thước tương ứng."""
    path = SVG_PATHS.get(name, SVG_PATHS["cpu"])
    return f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="{color}" style="flex-shrink:0;">{path}</svg>'

CAT_ICONS = {
    "cpus": get_svg_icon("cpu", "#FFFFFF", 18),
    "mainboards": get_svg_icon("mainboard", "#FFFFFF", 18),
    "rams": get_svg_icon("ram", "#FFFFFF", 18),
    "gpus": get_svg_icon("gpu", "#FFFFFF", 18),
    "storages": get_svg_icon("ssd", "#FFFFFF", 18),
    "hdds": get_svg_icon("hdd", "#FFFFFF", 18),
    "psus": get_svg_icon("psu", "#FFFFFF", 18),
    "cases": get_svg_icon("case", "#FFFFFF", 18),
    "coolers": get_svg_icon("cooler", "#FFFFFF", 18),
}

def generate_html_quotation(current_selected_dict, categories_list, tot_cost, psu_rec, current_usage) -> str:
    """Tạo tài liệu báo giá HTML định dạng sẵn chuẩn in A4 / PDF."""
    rows_h = ""
    for c_name, c_key, _, _ in categories_list:
        it = current_selected_dict.get(
            {"cpus": "cpu", "mainboards": "mainboard", "rams": "ram", "gpus": "gpu",
             "psus": "psu", "cases": "case", "storages": "storage", "coolers": "cooler", "hdds": "hdd"}.get(c_key)
        )
        if it and it.get("id") not in ("gpu_none", "hdd_none", "cooler_stock"):
            q = st.session_state.get(f"qty_{c_key}", 1)
            u_p = it.get("price", 0)
            rows_h += f"""
            <tr>
                <td style="padding: 10px 12px; border-bottom: 1px solid #E2E8F0;"><b>{c_name}</b></td>
                <td style="padding: 10px 12px; border-bottom: 1px solid #E2E8F0;">{it['name']}</td>
                <td style="padding: 10px 12px; border-bottom: 1px solid #E2E8F0;">{get_item_brand(it)}</td>
                <td style="padding: 10px 12px; border-bottom: 1px solid #E2E8F0; text-align: center;">{q}</td>
                <td style="padding: 10px 12px; border-bottom: 1px solid #E2E8F0; text-align: right;">{u_p:,.0f} đ</td>
                <td style="padding: 10px 12px; border-bottom: 1px solid #E2E8F0; text-align: right; font-weight: bold; color: #2563EB;">{u_p * q:,.0f} đ</td>
            </tr>
            """
        elif it and it.get("id") in ("gpu_none", "cooler_stock"):
            rows_h += f"""
            <tr>
                <td style="padding: 10px 12px; border-bottom: 1px solid #E2E8F0;"><b>{c_name}</b></td>
                <td style="padding: 10px 12px; border-bottom: 1px solid #E2E8F0;">{it['name']}</td>
                <td style="padding: 10px 12px; border-bottom: 1px solid #E2E8F0;">Tiêu chuẩn</td>
                <td style="padding: 10px 12px; border-bottom: 1px solid #E2E8F0; text-align: center;">1</td>
                <td style="padding: 10px 12px; border-bottom: 1px solid #E2E8F0; text-align: right;">0 đ</td>
                <td style="padding: 10px 12px; border-bottom: 1px solid #E2E8F0; text-align: right; font-weight: bold; color: #2563EB;">0 đ</td>
            </tr>
            """
            
    doc = f"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<title>Báo giá Cấu hình Máy tính - AI PC Builder Advisor</title>
<style>
    body {{ font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; margin: 30px; color: #1E293B; background: #FFFFFF; line-height: 1.5; }}
    .header {{ border-bottom: 2px solid #2563EB; padding-bottom: 15px; margin-bottom: 20px; display: flex; justify-content: space-between; align-items: flex-end; }}
    .title {{ font-size: 24px; font-weight: 800; color: #1E293B; margin: 0; }}
    .subtitle {{ font-size: 13px; color: #64748B; margin-top: 4px; }}
    .meta-box {{ background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 12px 18px; margin-bottom: 20px; display: flex; gap: 30px; }}
    .meta-item {{ font-size: 13px; }}
    .meta-label {{ color: #64748B; font-weight: 500; }}
    .meta-value {{ font-weight: 700; color: #0F172A; }}
    table {{ width: 100%; border-collapse: collapse; margin-bottom: 25px; }}
    th {{ background: #F1F5F9; color: #475569; font-size: 12px; font-weight: 700; text-align: left; padding: 10px 12px; border-bottom: 2px solid #CBD5E1; text-transform: uppercase; }}
    .total-box {{ float: right; width: 340px; background: #EFF6FF; border: 1px solid #BFDBFE; border-radius: 8px; padding: 15px; margin-top: 10px; }}
    .total-row {{ display: flex; justify-content: space-between; align-items: center; }}
    .total-title {{ font-size: 14px; font-weight: 700; color: #1E3A8A; }}
    .total-price {{ font-size: 22px; font-weight: 800; color: #2563EB; }}
    .footer {{ clear: both; margin-top: 50px; text-align: center; font-size: 12px; color: #94A3B8; border-top: 1px solid #E2E8F0; padding-top: 15px; }}
    @media print {{ body {{ margin: 0; }} .no-print {{ display: none; }} }}
</style>
</head>
<body>
<div class="header">
    <div>
        <h1 class="title">AI PC Builder Advisor</h1>
        <div class="subtitle">Bảng báo giá cấu hình máy tính tối ưu hóa bằng Trí tuệ nhân tạo (CS 417)</div>
    </div>
    <div style="text-align: right; font-size: 12px; color: #64748B;">
        Ngày lập: {time.strftime('%d/%m/%Y %H:%M')}<br>
        Hệ thống: Trí tuệ nhân tạo (Branch & Bound Heuristic)
    </div>
</div>

<div class="meta-box">
    <div class="meta-item"><span class="meta-label">Nhu cầu sử dụng:</span> <span class="meta-value">{current_usage}</span></div>
    <div class="meta-item"><span class="meta-label">Nguồn khuyến nghị:</span> <span class="meta-value">{'>= ' + str(psu_rec) + 'W' if psu_rec else 'Chưa tính'}</span></div>
    <div class="meta-item"><span class="meta-label">Trạng thái:</span> <span class="meta-value" style="color: #10B981;">Tương thích hoàn toàn</span></div>
</div>

<table>
    <thead>
        <tr>
            <th style="width: 22%;">Danh mục</th>
            <th style="width: 38%;">Tên linh kiện</th>
            <th style="width: 14%;">Hãng sản xuất</th>
            <th style="width: 6%; text-align: center;">SL</th>
            <th style="width: 10%; text-align: right;">Đơn giá</th>
            <th style="width: 10%; text-align: right;">Thành tiền</th>
        </tr>
    </thead>
    <tbody>
        {rows_h}
    </tbody>
</table>

<div class="total-box">
    <div class="total-row">
        <span class="total-title">TỔNG CHI PHÍ:</span>
        <span class="total-price">{tot_cost:,.0f} đ</span>
    </div>
</div>

<div class="footer">
    Bảng báo giá được tạo tự động bởi AI PC Builder Advisor. Để in hoặc lưu thành file PDF, nhấn tổ hợp phím <b>Ctrl + P</b> trên trình duyệt.
</div>
</body>
</html>"""
    return doc

# ===========================================================================
# TẢI DỮ LIỆU
# ===========================================================================
@st.cache_data
def get_data():
    return load_data(str(Path(__file__).parent / "data.json"))

DATA = get_data()

# ===========================================================================
# DANH SÁCH THƯƠNG HIỆU & HÀM TIỆN ÍCH
# ===========================================================================
KNOWN_BRANDS = [
    "ASUS", "Gigabyte", "MSI", "ASRock", "Kingston", "Corsair", "G.Skill",
    "Crucial", "TeamGroup", "NVIDIA", "AMD", "Intel", "Seasonic", "Deepcool",
    "Cooler Master", "Thermalright", "Noctua", "be quiet!", "NZXT", "Lian Li",
    "Xigmatek", "VSP", "Mik", "SAMA", "Seagate", "Western Digital", "WD",
    "Toshiba", "Samsung", "ID-Cooling", "Phanteks", "Thermaltake", "Antec",
    "Super Flower", "FSP", "ARCTIC", "Jonsbo", "Fractal", "EVGA"
]

def fmt_vnd(amount: int) -> str:
    """Định dạng số tiền VNĐ."""
    return f"{amount:,.0f} VNĐ".replace(",", ".")

def parse_budget_int(val: Any, default: int = 20_000_000) -> int:
    """Helper chuyển đổi an toàn mọi giá trị ngân sách (int, float, chuỗi có dấu phẩy) sang int."""
    if isinstance(val, (int, float)):
        return int(val)
    if isinstance(val, str):
        digits = re.sub(r"\D", "", str(val))
        if digits:
            try:
                return int(digits)
            except ValueError:
                pass
    return default

def get_item_brand(item: Optional[Dict]) -> str:
    """Trích xuất hãng sản xuất của linh kiện."""
    if not item:
        return ""
    if "brand" in item and item["brand"]:
        return item["brand"]
    item_id = item.get("id", "")
    if item_id in ("gpu_none", "hdd_none"):
        return "Tiêu chuẩn"
    if item_id == "cooler_stock":
        return "Kèm CPU"
    name = item.get("name", "")
    for b in KNOWN_BRANDS:
        if b.lower() in name.lower():
            return b
    return name.split()[0] if name else "Khác"

def get_item_specs_summary(item: Optional[Dict], cat_key: str) -> str:
    """Tạo chuỗi thông số kỹ thuật chính của sản phẩm."""
    if not item:
        return ""
    item_id = item.get("id", "")
    if item_id == "gpu_none":
        return "Sử dụng đồ họa tích hợp iGPU sẵn trong CPU"
    if item_id == "hdd_none":
        return "Không trang bị thêm ổ cứng phụ HDD"
    if item_id == "cooler_stock":
        return "Tản nhiệt đi kèm theo hộp CPU (TDP tối đa 65W)"

    if cat_key == "cpus":
        igpu_str = "Có iGPU" if item.get("has_igpu") else "Không iGPU"
        return f"Socket {item.get('socket')} | {item.get('ram_type')} | TDP {item.get('tdp')}W | Điểm hiệu năng: {item.get('score', 0):,} | {igpu_str}"
    elif cat_key == "mainboards":
        return f"Socket {item.get('socket')} | Chuẩn RAM: {item.get('ram_type')} | Kích thước: {item.get('form_factor')} | Tối đa {item.get('max_ram_gb')}GB RAM"
    elif cat_key == "rams":
        is_kit = item.get("is_kit") or item.get("sticks", 1) >= 2 or "2x" in item.get("name", "").lower()
        kit_str = f" | Kit {item.get('sticks', 2)} thanh (Dual Channel)" if is_kit else ""
        return f"Chuẩn {item.get('ram_type')} | Dung lượng: {item.get('capacity_gb')}GB{kit_str} | Tốc độ: {item.get('speed_mhz')}MHz"
    elif cat_key == "gpus":
        return f"VRAM: {item.get('vram_gb')}GB | TDP: {item.get('tdp')}W | Chiều dài: {item.get('length_mm')}mm | Điểm đồ họa: {item.get('score', 0):,}"
    elif cat_key == "storages":
        return f"Chuẩn kết nối: {item.get('type')} | Dung lượng: {item.get('capacity_gb')}GB"
    elif cat_key == "hdds":
        return f"Dung lượng: {item.get('capacity_gb')}GB | Tốc độ quay: {item.get('rpm')} RPM"
    elif cat_key == "psus":
        return f"Công suất danh định: {item.get('wattage')}W | Chứng nhận hiệu suất: {item.get('efficiency')}"
    elif cat_key == "cases":
        ff = ", ".join(item.get("supported_form_factors", []))
        return f"Hỗ trợ bo mạch: {ff} | Chiều dài GPU tối đa: {item.get('max_gpu_length_mm')}mm"
    elif cat_key == "coolers":
        t = item.get("type", "Air")
        size = f" {item.get('aio_size_mm')}mm" if item.get("aio_size_mm") else ""
        h = f" | Chiều cao: {item.get('height_mm')}mm" if item.get("height_mm") else ""
        return f"Loại tản: {t}{size} | Khả năng tản nhiệt tối đa: {item.get('tdp_support')}W{h}"
    return ""

def get_item_specs_badges_html(item: Optional[Dict], cat_key: str) -> str:
    """Tạo chuỗi HTML chứa các thẻ Pill Badges nhỏ gọn mô tả thông số kỹ thuật chính."""
    if not item:
        return ""
    item_id = item.get("id", "")
    if item_id == "gpu_none":
        return "<div class='spec-badges-container'><span class='spec-badge'>iGPU tích hợp CPU</span></div>"
    if item_id == "hdd_none":
        return "<div class='spec-badges-container'><span class='spec-badge'>Không dùng HDD phụ</span></div>"
    if item_id == "cooler_stock":
        return "<div class='spec-badges-container'><span class='spec-badge'>Tản Stock kèm CPU (<= 65W)</span></div>"

    badges = []
    if cat_key == "cpus":
        badges.append(f"Socket {item.get('socket')}")
        badges.append(f"{item.get('ram_type')}")
        badges.append(f"TDP {item.get('tdp')}W")
        badges.append(f"Điểm {item.get('score', 0):,}")
        badges.append("Có iGPU" if item.get("has_igpu") else "Không iGPU")
    elif cat_key == "mainboards":
        badges.append(f"Socket {item.get('socket')}")
        badges.append(f"{item.get('ram_type')}")
        badges.append(f"{item.get('form_factor')}")
        badges.append(f"Max {item.get('max_ram_gb')}GB RAM")
    elif cat_key == "rams":
        is_kit = item.get("is_kit") or item.get("sticks", 1) >= 2 or "2x" in item.get("name", "").lower()
        badges.append(f"{item.get('ram_type')}")
        badges.append(f"{item.get('capacity_gb')}GB")
        if is_kit:
            badges.append("Kit Dual Channel")
        badges.append(f"{item.get('speed_mhz')}MHz")
    elif cat_key == "gpus":
        badges.append(f"VRAM {item.get('vram_gb')}GB")
        badges.append(f"TDP {item.get('tdp')}W")
        badges.append(f"Dài {item.get('length_mm')}mm")
        badges.append(f"Điểm {item.get('score', 0):,}")
    elif cat_key == "storages":
        badges.append(f"Chuẩn {item.get('type')}")
        badges.append(f"{item.get('capacity_gb')}GB")
    elif cat_key == "hdds":
        badges.append(f"{item.get('capacity_gb')}GB")
        badges.append(f"{item.get('rpm')} RPM")
    elif cat_key == "psus":
        badges.append(f"{item.get('wattage')}W")
        badges.append(f"{item.get('efficiency')}")
    elif cat_key == "cases":
        ff = "/".join(item.get("supported_form_factors", []))
        badges.append(f"Bo mạch: {ff}")
        badges.append(f"GPU <= {item.get('max_gpu_length_mm')}mm")
    elif cat_key == "coolers":
        t = item.get("type", "Air")
        size = f" {item.get('aio_size_mm')}mm" if item.get("aio_size_mm") else ""
        badges.append(f"{t}{size}")
        badges.append(f"TDP <= {item.get('tdp_support')}W")

    badges_html = " ".join([f"<span class='spec-badge'>{b}</span>" for b in badges if b])
    return f"<div class='spec-badges-container'>{badges_html}</div>"

def find_item_by_id(category: str, item_id: Optional[str]) -> Optional[Dict]:
    """Tìm đối tượng linh kiện theo mã định danh."""
    if not item_id:
        return None
    for item in DATA.get(category, []):
        if item.get("id") == item_id:
            return item
    return None

def format_chat_content_to_html(text: str) -> str:
    """Chuyển đổi nội dung tin nhắn chat sang HTML chuẩn đẹp, hỗ trợ in đậm, in nghiêng và danh sách gạch đầu dòng xuống dòng rõ ràng."""
    if not text:
        return ""
    # Chuyển đổi in đậm **nội dung** -> <b>nội dung</b>
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    # Chuyển đổi in nghiêng *nội dung* -> <i>nội dung</i>
    t = re.sub(r"(?<!\*)\*([^*\n]+?)\*(?!\*)", r"<i>\1</i>", t)

    lines = t.replace("<br>", "\n").split("\n")
    html_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            html_lines.append("<div style='height: 8px;'></div>")
            continue
        if stripped.startswith("- ") or stripped.startswith("* "):
            content = stripped[2:].strip()
            bullet_html = (
                f"<div style='display:flex; align-items:baseline; margin: 3px 0 3px 6px;'>"
                f"<span style='color:#3B82F6; font-size:10px; margin-right:8px; line-height:1; flex-shrink:0;'>&#9679;</span>"
                f"<div>{content}</div>"
                f"</div>"
            )
            html_lines.append(bullet_html)
        else:
            html_lines.append(f"<div>{stripped}</div>")
    return "".join(html_lines)

# ===========================================================================
# CẤU HÌNH DANH MỤC LINH KIỆN (CATEGORIES)
# ===========================================================================
CATEGORIES = [
    ("Bộ vi xử lý", "cpus", "sel_cpus", False),
    ("Bo mạch chủ", "mainboards", "sel_mainboards", False),
    ("Bộ nhớ RAM", "rams", "sel_rams", True),
    ("Card đồ họa", "gpus", "sel_gpus", False),
    ("Ổ cứng SSD", "storages", "sel_storages", True),
    ("Ổ cứng HDD", "hdds", "sel_hdds", True),
    ("Nguồn máy tính", "psus", "sel_psus", False),
    ("Vỏ máy (Case)", "cases", "sel_cases", False),
    ("Tản nhiệt", "coolers", "sel_coolers", False),
]

# ===========================================================================
# KHỞI TẠO TRẠNG THÁI (SESSION STATE)
# ===========================================================================
for _, cat_key, state_key, _ in CATEGORIES:
    if state_key not in st.session_state:
        st.session_state[state_key] = None

for cat_key in ["rams", "storages", "hdds"]:
    if f"qty_{cat_key}" not in st.session_state:
        st.session_state[f"qty_{cat_key}"] = 1

if "budget" not in st.session_state:
    st.session_state.budget = 20_000_000

if "usage" not in st.session_state:
    st.session_state.usage = "Tất cả / Đa dụng (Cân bằng)"

if "need_discrete_gpu" not in st.session_state:
    st.session_state.need_discrete_gpu = True

if "need_hdd" not in st.session_state:
    st.session_state.need_hdd = False

if "auto_filter_compat" not in st.session_state:
    st.session_state.auto_filter_compat = True

if "last_stats" not in st.session_state:
    st.session_state.last_stats = None

if "opened_dialog" not in st.session_state:
    st.session_state.opened_dialog = None

if "quick_ai_feedback" not in st.session_state:
    st.session_state.quick_ai_feedback = None

if "gemini_api_key" not in st.session_state:
    st.session_state.gemini_api_key = ""

DEFAULT_WELCOME_MESSAGE = {
    "role": "assistant",
    "content": "Chào bạn! Bạn đang muốn dựng PC tầm giá bao nhiêu và phục vụ nhu cầu gì để mình tư vấn cấu hình tối ưu nhất nhé?",
}

if "chat_history" not in st.session_state or not st.session_state.chat_history:
    st.session_state.chat_history = [DEFAULT_WELCOME_MESSAGE]

# ===========================================================================
# POPUP DIALOG CHỌN LINH KIỆN (@st.dialog) - PHONG CÁCH GEARVN / PHONG VŨ
# ===========================================================================
@st.dialog("Chọn sản phẩm", width="large")
def pick_component_dialog(cat_key: str, cat_name: str, available_items: List[Dict]):
    st.markdown(f"#### Danh sách {cat_name}")
    st.caption("Tìm kiếm, lọc theo thương hiệu và chọn linh kiện phù hợp với cấu hình của bạn.")

    # Thanh công cụ 3 cột bên trong Popup
    col_search, col_brand, col_sort = st.columns([1.5, 1.2, 1.3])

    with col_search:
        kw = st.text_input(
            "Tìm kiếm theo tên sản phẩm",
            placeholder="Nhập tên sản phẩm...",
            key=f"dlg_kw_{cat_key}",
        )

    # Lấy danh sách hãng có trong tập sản phẩm
    brands = sorted(list(set(get_item_brand(i) for i in available_items if get_item_brand(i))))
    brand_options = ["Tất cả hãng"] + brands

    with col_brand:
        selected_brand = st.selectbox(
            "Hãng sản xuất",
            options=brand_options,
            key=f"dlg_brand_{cat_key}",
        )

    with col_sort:
        sort_choice = st.selectbox(
            "Sắp xếp theo",
            options=[
                "Mặc định",
                "Giá: Thấp đến Cao",
                "Giá: Cao đến Thấp",
                "Điểm hiệu năng: Cao đến Thấp",
            ],
            key=f"dlg_sort_{cat_key}",
        )

    # Lọc danh sách theo từ khóa tìm kiếm
    items = list(available_items)
    if kw and kw.strip():
        k_val = kw.strip().lower()
        items = [i for i in items if k_val in i.get("name", "").lower()]

    # Lọc theo hãng sản xuất
    if selected_brand != "Tất cả hãng":
        items = [i for i in items if get_item_brand(i) == selected_brand]

    # Sắp xếp
    if sort_choice == "Giá: Thấp đến Cao":
        items.sort(key=lambda x: x.get("price", 0))
    elif sort_choice == "Giá: Cao đến Thấp":
        items.sort(key=lambda x: x.get("price", 0), reverse=True)
    elif sort_choice == "Điểm hiệu năng: Cao đến Thấp":
        items.sort(key=lambda x: x.get("score", 0) if "score" in x else -x.get("price", 0), reverse=True)

    st.markdown(
        f"<div style='font-size: 0.92rem; color: var(--text-sub); margin-top: 6px; margin-bottom: 12px;'>"
        f"Tìm thấy <b>{len(items)}</b> sản phẩm phù hợp:"
        f"</div>",
        unsafe_allow_html=True,
    )

    if not items:
        st.info("Không tìm thấy sản phẩm nào phù hợp với bộ lọc tìm kiếm hiện tại.")
    else:
        # Hiển thị từng sản phẩm dạng thẻ bo viền
        for item in items:
            with st.container(border=True):
                info_col, act_col = st.columns([3.8, 1.2])
                with info_col:
                    brand_tag = get_item_brand(item)
                    st.markdown(
                        f"<div style='font-weight: 700; font-size: 1.02rem; color: var(--text-main); line-height: 1.35;'>"
                        f"{item['name']} "
                        f"<span class='brand-badge'>{brand_tag}</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                    specs = get_item_specs_summary(item, cat_key)
                    if specs:
                        st.markdown(f"<div style='font-size:0.86rem; color:var(--text-sub); margin-top:3px;'>{specs}</div>", unsafe_allow_html=True)
                    st.markdown(
                        f"<div style='font-size: 1.15rem; font-weight: 700; color: var(--text-price); margin-top: 4px;'>"
                        f"{fmt_vnd(item.get('price', 0))}"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

                with act_col:
                    st.write("")
                    is_active = (st.session_state.get(f"sel_{cat_key}") == item["id"])
                    if is_active:
                        st.button("Đang chọn", key=f"dlg_btn_act_{item['id']}", disabled=True, use_container_width=True)
                    else:
                        if st.button("Chọn sản phẩm", key=f"dlg_btn_pick_{item['id']}", type="primary", use_container_width=True):
                            st.session_state[f"sel_{cat_key}"] = item["id"]
                            st.session_state.opened_dialog = None
                            st.rerun()

    st.divider()
    if st.button("Đóng cửa sổ", use_container_width=True):
        st.session_state.opened_dialog = None
        st.rerun()


# ===========================================================================
# PHẦN ĐẦU TRANG (HEADER & HERO BANNER)
# ===========================================================================
head_c_left, head_c_right = st.columns([3.8, 1.2], gap="small")
with head_c_left:
    st.markdown(
        f"""
        <div class="top-header-left">
            <div class="top-logo-box">
                {get_svg_icon("ai_chip", "#38BDF8" if is_dark else "#2563EB", 22)}
            </div>
            <div>
                <div class="top-app-title">AI PC Builder Advisor</div>
                <div class="top-app-subtitle">Hệ thống tư vấn và xây dựng cấu hình máy tính theo đúng Trí tuệ Nhân tạo</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with head_c_right:
    theme_col, deploy_col = st.columns([1.5, 1.1], gap="small")
    with theme_col:
        theme_btn_label = "Chế độ tối" if not is_dark else "Chế độ sáng"
        theme_btn_help = "Chuyển sang giao diện Sáng (Light Mode)" if is_dark else "Chuyển sang giao diện Tối (Dark Mode)"
        if st.button(theme_btn_label, key="btn_theme_toggle", help=theme_btn_help, use_container_width=True):
            st.session_state.theme = "light" if is_dark else "dark"
            st.rerun()
    with deploy_col:
        deploy_bg = "#1E293B" if is_dark else "#2563EB"
        deploy_border = "1px solid rgba(148, 163, 184, 0.3)" if is_dark else "none"
        st.markdown(
            f"""
            <div style="background: {deploy_bg}; border: {deploy_border}; color: #FFFFFF; border-radius: 9999px; padding: 6px 14px; font-weight: 600; font-size: 0.88rem; display: flex; align-items: center; justify-content: center; gap: 6px; box-shadow: 0 2px 8px rgba(37, 99, 235, 0.25); cursor: pointer; margin-top: 1px;">
                {get_svg_icon("send", "#FFFFFF", 14)}
                <span>Deploy</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

badge_icon_color = "#FFFFFF" if is_dark else "#0F172A"
st.markdown(
    f"""
    <div class="hero-banner">
        <div class="hero-content">
            <h1 class="hero-title">Xây dựng &amp; Tư vấn cấu hình PC</h1>
            <div class="hero-subtitle">Cấu hình tối ưu • Hiệu năng vượt trội • Phù hợp ngân sách</div>
        </div>
        <div class="hero-badge">
            {get_svg_icon("academic", badge_icon_color, 15)}
            <span>CS 417 - Trí tuệ nhân tạo</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_build, tab_benchmark = st.tabs([
    "Xây dựng & Tư vấn cấu hình PC",
    "Cơ sở dữ liệu & Thử nghiệm thuật toán",
])


with tab_build:
    # ── BỐ TRÍ LẠI THANH ĐIỀU KHIỂN PHÍA TRÊN (TÁCH 2 CỤM 65% | 35%) ──
    col_left, col_right = st.columns([64, 36], gap="large")

    with col_left:

        with st.container(border=True):
            ctl_c1, ctl_c2, ctl_c3 = st.columns([1.3, 1.2, 1.35], gap="small")

            with ctl_c1:
                budget_val = max(6_000_000, min(120_000_000, parse_budget_int(st.session_state.budget)))
                c_wall_col = "#10B981" if not is_dark else "#34D399"
                st.markdown(
                    f"<div style='font-size:0.92rem; font-weight:600; color:var(--text-main); margin-bottom: 4px; display:flex; align-items:center; gap:6px;'>"
                    f"{get_svg_icon('wallet', c_wall_col, 16)} <span>Ngân sách dự tính (VNĐ)</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                b_inp_col, b_m_col, b_p_col = st.columns([0.72, 0.14, 0.14], gap="small")
                with b_inp_col:
                    budget_str = st.text_input(
                        "Ngân sách dự tính (VNĐ)",
                        label_visibility="collapsed",
                        value=f"{budget_val:,}",
                        key=f"input_budget_{st.session_state.get('budget_version', 0)}",
                    )
                    clean_digits = "".join(c for c in str(budget_str) if c.isdigit())
                    if clean_digits:
                        parsed_val = max(6_000_000, min(120_000_000, int(clean_digits)))
                        if parsed_val != budget_val:
                            st.session_state.budget = parsed_val
                            st.session_state["budget_version"] = st.session_state.get("budget_version", 0) + 1
                            st.rerun()
                with b_m_col:
                    if st.button("－", key="btn_budget_minus", use_container_width=True, help="Giảm 500.000 VNĐ"):
                        st.session_state.budget = max(6_000_000, budget_val - 500_000)
                        st.session_state["budget_version"] = st.session_state.get("budget_version", 0) + 1
                        st.rerun()
                with b_p_col:
                    if st.button("＋", key="btn_budget_plus", use_container_width=True, help="Tăng 500.000 VNĐ"):
                        st.session_state.budget = min(120_000_000, budget_val + 500_000)
                        st.session_state["budget_version"] = st.session_state.get("budget_version", 0) + 1
                        st.rerun()

            with ctl_c2:
                usage_options = list(USAGE_PROFILES.keys())
                usage_idx = usage_options.index(st.session_state.usage) if st.session_state.usage in usage_options else 0
                c_tgt_col = "#06B6D4" if not is_dark else "#38BDF8"
                st.markdown(
                    f"<div style='font-size:0.92rem; font-weight:600; color:var(--text-main); margin-bottom: 4px; display:flex; align-items:center; gap:6px;'>"
                    f"{get_svg_icon('target', c_tgt_col, 16)} <span>Nhu cầu sử dụng</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                usage_input = st.selectbox(
                    "Nhu cầu sử dụng",
                    label_visibility="collapsed",
                    options=usage_options,
                    index=usage_idx,
                    key=f"select_usage_{st.session_state.get('usage_version', 0)}",
                )
                if usage_input != st.session_state.usage:
                    st.session_state.usage = usage_input
                    if usage_input == "Văn phòng / Học tập":
                        st.session_state.need_discrete_gpu = False
                        st.session_state["vga_version"] = st.session_state.get("vga_version", 0) + 1
                        st.session_state.need_hdd = True
                        st.session_state["hdd_version"] = st.session_state.get("hdd_version", 0) + 1
                    elif usage_input in ["Gaming", "Đồ họa / Lập trình AI", "Tất cả / Đa dụng (Cân bằng)"]:
                        st.session_state.need_discrete_gpu = True
                        st.session_state["vga_version"] = st.session_state.get("vga_version", 0) + 1
                else:
                    st.session_state.usage = usage_input

            with ctl_c3:
                c_lay_col = "#3B82F6" if not is_dark else "#60A5FA"
                st.markdown(
                    f"<div style='font-size:0.92rem; font-weight:600; color:var(--text-main); margin-bottom: 4px; display:flex; align-items:center; gap:6px;'>"
                    f"{get_svg_icon('layers', c_lay_col, 16)} <span>Tùy chọn linh kiện</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                need_vga = st.checkbox(
                    "Card đồ họa rời (VGA)",
                    value=st.session_state.need_discrete_gpu,
                    help="Nếu bỏ tích, hệ thống sẽ chỉ dùng đồ họa tích hợp iGPU trong CPU để tối ưu ngân sách.",
                    key=f"cb_need_vga_{st.session_state.get('vga_version', 0)}",
                )
                st.session_state.need_discrete_gpu = need_vga

                need_hdd = st.checkbox(
                    "Ổ cứng HDD phụ",
                    value=st.session_state.need_hdd,
                    help="Nếu tích chọn, hệ thống bắt buộc phân bổ ngân sách để trang bị thêm 1 ổ HDD lưu trữ phụ.",
                    key=f"cb_need_hdd_{st.session_state.get('hdd_version', 0)}",
                )
                st.session_state.need_hdd = need_hdd

            st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
            btn_c1, btn_c2 = st.columns([0.65, 0.35], gap="small")
            with btn_c1:
                st.markdown('<div class="auto-complete-btn-wrap">', unsafe_allow_html=True)
                btn_auto_build = st.button("Tự động hoàn thiện", type="primary", use_container_width=True, key="btn_auto_complete_top")
                st.markdown('</div>', unsafe_allow_html=True)
            with btn_c2:
                st.markdown('<div class="reset-btn-wrap">', unsafe_allow_html=True)
                btn_reset = st.button("Đặt lại", use_container_width=True, key="btn_reset_top")
                st.markdown('</div>', unsafe_allow_html=True)
        # ── THỰC HIỆN ĐỌC TOÀN BỘ LINH KIỆN ĐANG CHỌN TỪ SESSION STATE ──
        current_selected = {
            "cpu": find_item_by_id("cpus", st.session_state.sel_cpus),
            "mainboard": find_item_by_id("mainboards", st.session_state.sel_mainboards),
            "ram": find_item_by_id("rams", st.session_state.sel_rams),
            "gpu": find_item_by_id("gpus", st.session_state.sel_gpus),
            "psu": find_item_by_id("psus", st.session_state.sel_psus),
            "case": find_item_by_id("cases", st.session_state.sel_cases),
            "storage": find_item_by_id("storages", st.session_state.sel_storages),
            "cooler": find_item_by_id("coolers", st.session_state.sel_coolers),
            "hdd": find_item_by_id("hdds", st.session_state.sel_hdds),
        }
        st.session_state["selected_parts"] = current_selected

        # ── XỬ LÝ NÚT ĐẶT LẠI ──
        if btn_reset:
            for _, cat_key, state_key, _ in CATEGORIES:
                st.session_state[state_key] = None
            for cat_key in ["rams", "storages", "hdds"]:
                st.session_state[f"qty_{cat_key}"] = 1
            st.session_state["selected_parts"] = {}
            st.session_state.last_stats = None
            st.session_state.opened_dialog = None
            st.session_state.chat_history = [DEFAULT_WELCOME_MESSAGE]
            st.session_state["pending_ai_query"] = None
            st.rerun()

        # ── XỬ LÝ NÚT TỰ ĐỘNG CHỌN CẤU HÌNH (BRANCH & BOUND) ──
        if btn_auto_build:
            # Đồng bộ và luôn đọc giá trị mới nhất trực tiếp từ st.session_state
            cur_budget = parse_budget_int(st.session_state.get(f"input_budget_{st.session_state.get('budget_version', 0)}", st.session_state.budget))
            cur_usage = st.session_state.get(f"select_usage_{st.session_state.get('usage_version', 0)}", st.session_state.usage)
            cur_need_vga = st.session_state.get(f"cb_need_vga_{st.session_state.get('vga_version', 0)}", st.session_state.need_discrete_gpu)
            cur_need_hdd = st.session_state.get(f"cb_need_hdd_{st.session_state.get('hdd_version', 0)}", st.session_state.need_hdd)

            st.session_state.budget = cur_budget
            st.session_state.usage = cur_usage
            st.session_state.need_discrete_gpu = cur_need_vga
            st.session_state.need_hdd = cur_need_hdd

            # Khóa toàn bộ linh kiện đã chọn trước đó (khác None)
            locked = {k: v for k, v in current_selected.items() if v is not None}

            # Nếu người dùng đã chọn đủ cả 9 linh kiện
            if len(locked) == 9:
                st.info("Cấu hình đã đầy đủ linh kiện. Bạn có thể xem kết quả kiểm tra tương thích và hiệu năng bên dưới.")
            else:
                optimizer = BranchAndBoundOptimizer(DATA)
                with st.spinner("Thuật toán Branch & Bound đang tìm kiếm cấu hình tối ưu..."):
                    best_builds, stats = optimizer.optimize(
                        budget=cur_budget,
                        usage=cur_usage,
                        locked_parts=locked,
                        top_k=1,
                        require_discrete_gpu=cur_need_vga,
                        require_hdd=cur_need_hdd,
                        prefer_dual_ram=True,
                        strict_budget=st.session_state.get("strict_budget", False),
                    )

                if best_builds:
                    b = best_builds[0]
                    st.session_state.sel_cpus = b.cpu.get("id")
                    st.session_state.sel_mainboards = b.mainboard.get("id")
                    st.session_state.sel_rams = b.ram.get("id")
                    st.session_state.sel_gpus = b.gpu.get("id")
                    st.session_state.sel_psus = b.psu.get("id")
                    st.session_state.sel_cases = b.case.get("id")
                    st.session_state.sel_storages = b.storage.get("id")
                    st.session_state.sel_coolers = b.cooler.get("id") if b.cooler else "cooler_stock"
                    st.session_state.sel_hdds = b.hdd.get("id") if b.hdd else "hdd_none"
                    st.session_state.last_stats = stats
                    st.rerun()
                else:
                    # Giữ nguyên toàn bộ linh kiện đã chọn, tuyệt đối không ghi đè và cảnh báo rõ ràng
                    if locked:
                        locked_names = [f"**{p.get('name', k)}**" for k, p in locked.items() if isinstance(p, dict)]
                        locked_str = ", ".join(locked_names)
                        st.warning(
                            f"Không thể tự động hoàn thiện trong ngân sách {fmt_vnd(cur_budget)} khi giữ cố định {locked_str}. "
                            f"Vui lòng tăng ngân sách hoặc đổi linh kiện để đảm bảo tính tương thích."
                        )
                    else:
                        min_b = optimizer.find_minimum_budget(cur_usage)
                        st.warning(
                            f"Ngân sách {fmt_vnd(cur_budget)} chưa đủ để lập cấu hình hợp lệ cho '{cur_usage}'. "
                            f"Ngân sách tối thiểu đề xuất là: {fmt_vnd(min_b)}."
                        )

        # Tính toán phụ tải ổ cứng
        qty_ram = st.session_state.get("qty_rams", 1)
        qty_ssd = st.session_state.get("qty_storages", 1)
        qty_hdd = st.session_state.get("qty_hdds", 1) if (current_selected["hdd"] and current_selected["hdd"]["id"] != "hdd_none") else 0
        extra_storage_count = max(0, qty_ssd - 1) + qty_hdd

        engine = ForwardChainingEngine(DATA)
        fc_result = engine.infer(
            current_selected,
            usage=st.session_state.usage,
            validate_errors=True,
            ram_qty=qty_ram,
            extra_storage_count=extra_storage_count,
        )
        filtered = fc_result.filtered_options

        # Kiểm tra bổ sung R03 cho số lượng thanh RAM và giới hạn bo mạch chủ
        if current_selected["mainboard"] and current_selected["ram"]:
            r03_ok, r03_msg = check_r03_ram_capacity(
                current_selected["mainboard"], current_selected["ram"], ram_qty=qty_ram
            )
            if not r03_ok and r03_msg not in fc_result.errors:
                fc_result.errors.append(r03_msg)

        # Bố cục chính 2:1 (~65% bên trái, ~35% bên phải)
        top_bar_c1, top_bar_c2 = st.columns([2, 1.2])
        with top_bar_c1:
            st.markdown(
                f"""<div style='font-size: 1.05rem; font-weight: 700; color: var(--text-main); display: flex; align-items: center; gap: 8px;'>
                    <div style='width: 28px; height: 28px; border-radius: 50%; background: rgba(6, 182, 212, 0.15); border: 1px solid rgba(6, 182, 212, 0.35); display: flex; align-items: center; justify-content: center; flex-shrink: 0;'>
                        {get_svg_icon('cube', '#06B6D4', 16)}
                    </div>
                    Danh sách linh kiện xây dựng
                </div>""",
                unsafe_allow_html=True,
            )
        with top_bar_c2:
            st.session_state.auto_filter_compat = st.toggle(
                "Tự động lọc tương thích",
                value=st.session_state.auto_filter_compat,
                help="Bật: Popup chỉ hiển thị linh kiện tương thích với lựa chọn hiện tại. Tắt: hiển thị toàn bộ để thử nghiệm phát hiện xung đột.",
            )

        # Tiêu đề bảng sản phẩm
        st.markdown('<div class="table-header-wrap">', unsafe_allow_html=True)
        h_col1, h_col2, h_col3, h_col4, h_col5 = st.columns([1.75, 3.35, 1.05, 1.45, 1.4], gap="small")
        with h_col1:
            st.markdown("<span style='font-size:0.92rem; font-weight:700; color:var(--text-sub);'>Danh mục</span>", unsafe_allow_html=True)
        with h_col2:
            st.markdown("<span style='font-size:0.92rem; font-weight:700; color:var(--text-sub);'>Linh kiện đã chọn</span>", unsafe_allow_html=True)
        with h_col3:
            st.markdown("<span style='font-size:0.92rem; font-weight:700; color:var(--text-sub); display:block; text-align:center;'>SL</span>", unsafe_allow_html=True)
        with h_col4:
            st.markdown("<span style='font-size:0.92rem; font-weight:700; color:var(--text-sub); display:block; text-align:right;'>Thành tiền</span>", unsafe_allow_html=True)
        with h_col5:
            st.markdown("<span style='font-size:0.92rem; font-weight:700; color:var(--text-sub); display:block; text-align:center;'>Thao tác</span>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("<hr style='margin: 4px 0 8px 0; border: none; border-top: 1px solid var(--hr-line);'>", unsafe_allow_html=True)

        # Render 9 dòng linh kiện
        for cat_name, cat_key, state_key, allows_qty in CATEGORIES:
            item_id = st.session_state.get(state_key)
            chosen_item = find_item_by_id(cat_key, item_id)
            icon_svg = CAT_ICONS.get(cat_key, "")

            r_c1, r_c2, r_c3, r_c4, r_c5 = st.columns([1.75, 3.35, 1.05, 1.45, 1.4], gap="small")

            with r_c1:
                st.markdown(
                    f"""<div style='display:flex; align-items:center; gap:6px; padding-top:2px;'>
                        <div class='cat-icon-box'>{icon_svg}</div>
                        <span style='font-weight:700; font-size:0.92rem; color:var(--text-main);'>{cat_name}</span>
                    </div>""",
                    unsafe_allow_html=True,
                )

            with r_c2:
                if chosen_item:
                    brand_name = get_item_brand(chosen_item)
                    st.markdown(
                        f"<div style='font-weight: 600; font-size: 0.98rem; color:var(--text-main); line-height: 1.35;'>"
                        f"{chosen_item['name']} "
                        f"<span class='brand-badge'>{brand_name}</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                    specs_badges = get_item_specs_badges_html(chosen_item, cat_key)
                    if specs_badges:
                        st.markdown(specs_badges, unsafe_allow_html=True)
                else:
                    st.markdown("<div style='color: var(--text-sub); font-style: italic; font-size: 0.9rem; padding-top: 4px;'>Chưa chọn linh kiện</div>", unsafe_allow_html=True)

            with r_c3:
                if allows_qty and chosen_item and chosen_item.get("id") not in ("hdd_none",):
                    cur_qty = st.session_state.get(f"qty_{cat_key}", 1)
                    q_val = st.selectbox(
                        f"SL_{cat_key}",
                        options=[1, 2, 3, 4],
                        index=max(0, cur_qty - 1),
                        key=f"select_qty_{cat_key}",
                        label_visibility="collapsed",
                    )
                    if q_val != st.session_state.get(f"qty_{cat_key}"):
                        st.session_state[f"qty_{cat_key}"] = q_val
                        st.rerun()
                else:
                    st.markdown("<div style='text-align: center; color: var(--text-sub); font-weight: 600; padding-top: 4px; font-size: 0.95rem;'>1</div>", unsafe_allow_html=True)
                    st.session_state[f"qty_{cat_key}"] = 1

            with r_c4:
                item_qty = st.session_state.get(f"qty_{cat_key}", 1)
                unit_p = chosen_item.get("price", 0) if chosen_item else 0
                line_total = unit_p * item_qty
                st.markdown(
                    f"<div style='font-weight: 700; color: var(--text-price); font-size: 1.02rem; padding-top: 4px; text-align: right; white-space: nowrap;'>{fmt_vnd(line_total)}</div>",
                    unsafe_allow_html=True,
                )

            with r_c5:
                act_a, act_b = st.columns([1.4, 1.0], gap="small")
                with act_a:
                    btn_label = "Đổi" if chosen_item else "Chọn"
                    btn_type = "secondary" if chosen_item else "primary"
                    if st.button(btn_label, key=f"btn_dlg_open_{cat_key}", type=btn_type, use_container_width=True):
                        st.session_state.opened_dialog = cat_key
                        st.rerun()
                with act_b:
                    if chosen_item:
                        if st.button("X", key=f"btn_clear_{cat_key}", help=f"Bỏ chọn {cat_name}", use_container_width=True):
                            st.session_state[state_key] = None
                            st.session_state[f"qty_{cat_key}"] = 1
                            st.rerun()

            st.markdown("<hr style='margin: 4px 0 6px 0; border: none; border-top: 1px solid var(--hr-line);'>", unsafe_allow_html=True)

        # Mở popup dialog nếu người dùng vừa bấm nút Chọn/Đổi
        if st.session_state.opened_dialog:
            active_cat = st.session_state.opened_dialog
            active_cat_name = next((c[0] for c in CATEGORIES if c[1] == active_cat), "Linh kiện")

            # Xác định danh sách sản phẩm hiển thị trong Popup
            if st.session_state.auto_filter_compat:
                popup_items = filtered.get(active_cat, DATA.get(active_cat, []))
            else:
                popup_items = DATA.get(active_cat, [])

            pick_component_dialog(active_cat, active_cat_name, popup_items)

        # Expander chi tiết suy diễn và thống kê thuật toán
        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        with st.expander("Chi tiết suy diễn & thống kê thuật toán", expanded=False):
            if st.session_state.last_stats:
                st_info = st.session_state.last_stats
                sc1, sc2, sc3, sc4 = st.columns(4)
                with sc1:
                    st.metric("Thời gian xử lý", f"{st_info.execution_time_ms:.1f} ms")
                with sc2:
                    st.metric("Số nút đã duyệt", f"{st_info.nodes_visited:,}")
                with sc3:
                    st.metric("Số nhánh đã cắt", f"{st_info.pruned_branches:,}")
                with sc4:
                    st.metric("Cấu hình hợp lệ", f"{st_info.valid_builds_found:,}")
                st.divider()

            st.markdown("**Nhật ký kích hoạt tập luật suy diễn:**")
            if fc_result.logs:
                log_html = "<div class='log-container'>"
                for line in fc_result.logs:
                    log_html += f"<div>{line}</div>"
                log_html += "</div>"
                st.markdown(log_html, unsafe_allow_html=True)
            else:
                st.write("Chưa có thông tin suy diễn.")

        # ── CARD DƯỚI CÙNG: PHÂN BỔ CHI PHÍ LINH KIỆN (TOÀN CHIỀU RỘNG) ──
        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown(
                f"""<div style='font-size: 0.98rem; font-weight: 700; color: var(--text-main); display: flex; align-items: center; gap: 6px; margin-bottom: 8px;'>
                    {get_svg_icon("chart", "#38BDF8", 16)}
                    Phân bổ chi phí linh kiện
                </div>""",
                unsafe_allow_html=True,
            )
            
            # Tính toán chi phí chi tiết từng nhóm
            c_cpu = current_selected["cpu"].get("price", 0) if current_selected.get("cpu") else 0
            c_mb = current_selected["mainboard"].get("price", 0) if current_selected.get("mainboard") else 0
            c_ram = (current_selected["ram"].get("price", 0) * st.session_state.get("qty_rams", 1)) if current_selected.get("ram") else 0
            c_ssd = (current_selected["storage"].get("price", 0) * st.session_state.get("qty_storages", 1)) if current_selected.get("storage") else 0
            c_hdd = (current_selected["hdd"].get("price", 0) * st.session_state.get("qty_hdds", 1)) if (current_selected.get("hdd") and current_selected["hdd"].get("id") != "hdd_none") else 0
            c_gpu = (current_selected["gpu"].get("price", 0) if current_selected.get("gpu") and current_selected["gpu"].get("id") != "gpu_none" else 0)
            c_psu = (current_selected["psu"].get("price", 0) if current_selected.get("psu") else 0)
            c_case = (current_selected["case"].get("price", 0) if current_selected.get("case") else 0)
            c_cooler = (current_selected["cooler"].get("price", 0) if current_selected.get("cooler") and current_selected["cooler"].get("id") != "cooler_stock" else 0)
            c_other = c_psu + c_case + c_cooler

            t_cost = c_cpu + c_mb + c_ram + c_ssd + c_hdd + c_gpu + c_other
            
            if t_cost > 0:
                donut_labels = ["CPU", "Mainboard", "RAM", "SSD"]
                donut_values = [c_cpu, c_mb, c_ram, c_ssd]
                donut_colors = ["#2563EB", "#06B6D4", "#8B5CF6", "#EC4899"]
                
                if c_gpu > 0:
                    donut_labels.append("VGA")
                    donut_values.append(c_gpu)
                    donut_colors.append("#3B82F6")
                if c_hdd > 0:
                    donut_labels.append("HDD")
                    donut_values.append(c_hdd)
                    donut_colors.append("#F59E0B")
                if c_other > 0:
                    donut_labels.append("Khác")
                    donut_values.append(c_other)
                    donut_colors.append("#64748B")

                d_col_left, d_col_right = st.columns([1.0, 1.8], gap="medium")
                with d_col_left:
                    fig_donut = go.Figure(data=[go.Pie(
                        labels=donut_labels,
                        values=donut_values,
                        hole=0.62,
                        textinfo="none",
                        marker=dict(colors=donut_colors),
                        hoverinfo="label+percent",
                    )])
                    total_m = f"{t_cost / 1_000_000:.2f}M"
                    fig_donut.add_annotation(
                        text=f"<b>{total_m}</b><br><span style='font-size:10px;'>VNĐ</span>",
                        x=0.5, y=0.5,
                        font_size=13,
                        font_color="#0F172A" if not is_dark else "#F1F5F9",
                        showarrow=False,
                    )
                    fig_donut.update_layout(
                        showlegend=False,
                        height=160,
                        margin=dict(t=0, b=0, l=0, r=0),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                    )
                    st.plotly_chart(fig_donut, use_container_width=True, config={"displayModeBar": False})

                with d_col_right:
                    legend_items = []
                    for l_idx, l_name in enumerate(donut_labels):
                        val = donut_values[l_idx]
                        pct = (val / t_cost * 100) if t_cost > 0 else 0
                        col = donut_colors[l_idx]
                        legend_items.append(
                            f"""<div style='display:flex; justify-content:space-between; align-items:center; font-size:0.88rem; margin-bottom:5px;'>
                                <span><span style='color:{col}; font-weight:bold; margin-right:6px;'>●</span><span style='color:var(--text-sub);'>{l_name}:</span> <b style='color:var(--text-main);'>{pct:.1f}%</b></span>
                                <span style='color:var(--text-sub); font-size:0.82rem;'>({fmt_vnd(val)})</span>
                            </div>"""
                        )
                    st.markdown("<div style='padding-top: 8px;'>" + "".join(legend_items) + "</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div style='color: #64748B; font-size: 0.88rem; text-align: center; padding: 30px 0;'>Chưa có dữ liệu linh kiện để phân bổ chi phí.</div>", unsafe_allow_html=True)

    # ========================================================================
    # CỘT PHẢI: TỔNG KẾT CẤU HÌNH & KIỂM TRA TƯƠNG THÍCH
    # ========================================================================
    with col_right:
        # ── KHUNG TRỢ LÝ TƯ VẤN AI (CỘT PHẢI) ──
        with st.container(border=True):
            st.markdown(
                f"""<div style='font-weight: 700; font-size: 0.98rem; color:var(--text-main); display: flex; align-items: center; gap: 8px; margin-bottom: 8px;'>
                    <div style='width: 26px; height: 26px; border-radius: 6px; background: linear-gradient(135deg, rgba(245, 158, 11, 0.2) 0%, rgba(139, 92, 246, 0.2) 100%); border: 1px solid rgba(245, 158, 11, 0.4); display: flex; align-items: center; justify-content: center; flex-shrink: 0;'>
                        {get_svg_icon('sparkles', '#FBBF24', 16)}
                    </div>
                    Tư vấn AI
                </div>""",
                unsafe_allow_html=True,
            )

            # Vùng hiển thị bong bóng chat (chiều cao cố định ~340px)
            AI_AVATAR_TAG = f"<img src='data:image/png;base64,{ROBOT_B64}' style='width: 30px; height: 30px; object-fit: contain; border-radius: 50%;' alt='AI' />" if ROBOT_B64 else get_svg_icon('robot', '#38BDF8' if is_dark else '#2563EB', 16)
            with st.container(height=340, border=True):
                st.markdown("<div id='chat-box-sentinel' style='display:none;'></div>", unsafe_allow_html=True)
                for msg in st.session_state.chat_history:
                    if msg["role"] == "user":
                        u_content = format_chat_content_to_html(msg["content"])
                        st.markdown(
                            f"<div class='chat-msg-row chat-msg-user-row'>"
                            f"<div class='chat-msg-user'>{u_content}</div>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )
                    else:
                        a_content = format_chat_content_to_html(msg["content"])
                        st.markdown(
                            f"<div class='chat-msg-row chat-msg-ai-row'>"
                            f"<div class='chat-ai-avatar'>{AI_AVATAR_TAG}</div>"
                            f"<div class='chat-msg-ai'>{a_content}</div>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )

                # Nếu có tin nhắn mới đang chờ Trợ lý AI phản hồi (chạy hiệu ứng chữ và tự động cuộn xuống)
                if st.session_state.get("pending_ai_query"):
                    user_text = st.session_state["pending_ai_query"]
                    with st.spinner("Trợ lý đang suy nghĩ..."):
                        parsed = parse_user_request(user_text, DATA, conversation_history=st.session_state.chat_history)

                        if parsed.get("intent") == "upgrade" or is_upgrade_acceptance(user_text) or is_upgrade_inquiry_after_acceptance(user_text):
                            pending_up = st.session_state.get("pending_upgrade_suggestion")
                            if not pending_up:
                                pending_up = extract_upgrade_from_history(st.session_state.chat_history, DATA)

                            if pending_up:
                                target_cat = pending_up["category"]
                                state_k = {
                                    "cpu": "sel_cpus", "mainboard": "sel_mainboards", "ram": "sel_rams",
                                    "gpu": "sel_gpus", "storage": "sel_storages", "psu": "sel_psus",
                                    "case": "sel_cases", "cooler": "sel_coolers", "hdd": "sel_hdds"
                                }.get(target_cat, f"sel_{target_cat}s")

                                st.session_state[state_k] = pending_up["target_id"]
                                added_c = pending_up.get("added_cost", 0)
                                st.session_state.budget = parse_budget_int(st.session_state.budget) + added_c
                                st.session_state["budget_version"] = st.session_state.get("budget_version", 0) + 1
                                st.session_state["pending_upgrade_suggestion"] = None

                                if is_upgrade_inquiry_after_acceptance(user_text):
                                    ai_reply = (
                                        f"Rất xin lỗi bạn về sự chậm trễ này! Mình vừa cập nhật đồng bộ ngay sang bảng danh sách linh kiện bên trái rồi nhé: "
                                        f"đã chuyển vi xử lý từ {pending_up.get('current_name', 'cũ')} sang **{pending_up['upgrade_name']}** (bù thêm {fmt_vnd(added_c)}).\n\n"
                                        f"Bảng cấu hình bên trái hiện đã được đổi mới hoàn tất, giúp tăng mạnh hiệu năng và khai thác trọn vẹn sức mạnh của card đồ họa. "
                                        f"Bạn kiểm tra lại bảng danh sách linh kiện bên trái xem đã hoàn toàn ưng ý chưa nhé!"
                                    )
                                else:
                                    ai_reply = (
                                        f"Tuyệt vời! Mình đã cập nhật ngay linh kiện mới trên bảng cấu hình bên trái cho bạn: "
                                        f"nâng cấp từ {pending_up.get('current_name', 'cũ')} lên **{pending_up['upgrade_name']}** (bù thêm {fmt_vnd(added_c)}).\n\n"
                                        f"Sự nâng cấp này giúp tăng mạnh hiệu năng xử lý và hạn chế tối đa nghẽn cổ chai. "
                                        f"Bạn kiểm tra lại bảng danh sách linh kiện bên trái xem đã ưng ý chưa để mình hỗ trợ tiếp nhé!"
                                    )
                            else:
                                ai_reply = "Mình đã ghi nhận yêu cầu của bạn. Bạn muốn nâng cấp cụ thể linh kiện nào (CPU, RAM, Mainboard hay Card đồ họa) để mình hỗ trợ thay đổi ngay nhé?"

                        elif parsed.get("intent") == "consult_no_budget":
                            # Người dùng hỏi nhu cầu/game mà chưa có ngân sách cụ thể:
                            # KHÔNG lấy lại ngân sách cũ, KHÔNG chạy optimizer, giữ nguyên bảng linh kiện hiện tại bên trái
                            if parsed.get("usage"):
                                st.session_state.usage = parsed["usage"]
                                st.session_state["usage_version"] = st.session_state.get("usage_version", 0) + 1
                                if parsed["usage"] == "Văn phòng / Học tập":
                                    st.session_state.need_discrete_gpu = False
                                    st.session_state["vga_version"] = st.session_state.get("vga_version", 0) + 1
                                    st.session_state.need_hdd = True
                                    st.session_state["hdd_version"] = st.session_state.get("hdd_version", 0) + 1
                                elif parsed["usage"] in ["Gaming", "Đồ họa / Lập trình AI"]:
                                    st.session_state.need_discrete_gpu = True
                                    st.session_state["vga_version"] = st.session_state.get("vga_version", 0) + 1

                            ai_reply = generate_conversational_reply(
                                user_query=user_text,
                                parsed=parsed,
                                build=None,
                                budget=0,
                                usage=st.session_state.usage,
                                conversation_history=st.session_state.chat_history,
                                catalog=DATA,
                            )

                        elif parsed.get("intent") == "build":
                            if parsed.get("budget") and parsed["budget"] >= 6_000_000:
                                st.session_state.budget = parsed["budget"]
                                st.session_state["budget_version"] = st.session_state.get("budget_version", 0) + 1
                            if parsed["usage"]:
                                st.session_state.usage = parsed["usage"]
                                st.session_state["usage_version"] = st.session_state.get("usage_version", 0) + 1
                                if parsed["usage"] == "Văn phòng / Học tập":
                                    if parsed["need_vga"] is None:
                                        st.session_state.need_discrete_gpu = False
                                        st.session_state["vga_version"] = st.session_state.get("vga_version", 0) + 1
                                    if parsed["need_hdd"] is None:
                                        st.session_state.need_hdd = True
                                        st.session_state["hdd_version"] = st.session_state.get("hdd_version", 0) + 1
                                elif parsed["usage"] in ["Gaming", "Đồ họa / Lập trình AI", "Tất cả / Đa dụng (Cân bằng)"]:
                                    if parsed["need_vga"] is None:
                                        st.session_state.need_discrete_gpu = True
                                        st.session_state["vga_version"] = st.session_state.get("vga_version", 0) + 1
                            if parsed["need_vga"] is not None:
                                st.session_state.need_discrete_gpu = parsed["need_vga"]
                                st.session_state["vga_version"] = st.session_state.get("vga_version", 0) + 1
                            if parsed["need_hdd"] is not None:
                                st.session_state.need_hdd = parsed["need_hdd"]
                                st.session_state["hdd_version"] = st.session_state.get("hdd_version", 0) + 1

                            strict_budget_val = parsed.get("strict_budget", False)
                            st.session_state["strict_budget"] = strict_budget_val

                            opt_ai = BranchAndBoundOptimizer(DATA)
                            best_builds_ai, _ = opt_ai.optimize(
                                budget=parse_budget_int(st.session_state.budget),
                                usage=st.session_state.usage,
                                locked_parts=parsed["locked_parts"],
                                top_k=1,
                                require_discrete_gpu=st.session_state.need_discrete_gpu,
                                require_hdd=st.session_state.need_hdd,
                                cpu_brand=parsed["cpu_brand"],
                                gpu_brand=parsed["gpu_brand"],
                                target_ram_gb=parsed["target_ram_gb"],
                                min_storage_gb=parsed.get("min_storage_gb"),
                                min_total_storage_gb=parsed.get("min_total_storage_gb"),
                                prefer_dual_ram=parsed.get("prefer_dual_ram", True),
                                strict_budget=strict_budget_val,
                            )

                            min_needed = None
                            chosen_build = None
                            if best_builds_ai:
                                chosen_build = best_builds_ai[0]
                                st.session_state.sel_cpus = chosen_build.cpu.get("id")
                                st.session_state.sel_mainboards = chosen_build.mainboard.get("id")
                                st.session_state.sel_rams = chosen_build.ram.get("id")
                                st.session_state.sel_gpus = chosen_build.gpu.get("id")
                                st.session_state.sel_psus = chosen_build.psu.get("id")
                                st.session_state.sel_cases = chosen_build.case.get("id")
                                st.session_state.sel_storages = chosen_build.storage.get("id")
                                st.session_state.sel_coolers = chosen_build.cooler.get("id") if chosen_build.cooler else "cooler_stock"
                                st.session_state.sel_hdds = chosen_build.hdd.get("id") if chosen_build.hdd else "hdd_none"
                            else:
                                min_needed = opt_ai.find_minimum_budget(
                                    usage=st.session_state.usage,
                                    locked_parts=parsed["locked_parts"],
                                    cpu_brand=parsed["cpu_brand"],
                                    gpu_brand=parsed["gpu_brand"],
                                    require_discrete_gpu=st.session_state.need_discrete_gpu,
                                    require_hdd=st.session_state.need_hdd,
                                    target_ram_gb=parsed["target_ram_gb"],
                                    min_storage_gb=parsed.get("min_storage_gb"),
                                    min_total_storage_gb=parsed.get("min_total_storage_gb"),
                                    prefer_dual_ram=parsed.get("prefer_dual_ram", True),
                                )

                            upgrade_list = suggest_upgrades(chosen_build, DATA, parse_budget_int(st.session_state.budget), st.session_state.usage) if chosen_build else []
                            st.session_state["pending_upgrade_suggestion"] = upgrade_list[0] if upgrade_list else None
                            ai_reply = generate_conversational_reply(
                                user_query=user_text,
                                parsed=parsed,
                                build=chosen_build,
                                budget=parse_budget_int(st.session_state.budget),
                                usage=st.session_state.usage,
                                conversation_history=st.session_state.chat_history,
                                min_needed_budget=min_needed,
                                upgrade_suggestions=upgrade_list,
                                catalog=DATA,
                            )
                        else:
                            # intent == "chat": Giữ nguyên toàn bộ cấu hình bên trái không thay đổi gì
                            ai_reply = generate_conversational_reply(
                                user_query=user_text,
                                parsed=parsed,
                                build=None,
                                budget=parse_budget_int(st.session_state.budget),
                                usage=st.session_state.usage,
                                conversation_history=st.session_state.chat_history,
                            )

                    # Hiệu ứng chạy chữ ra (streaming typewriter effect)
                    stream_holder = st.empty()
                    words = ai_reply.split(" ")
                    accumulated = ""
                    chunk_size = 2 if len(words) > 30 else 1
                    for i in range(0, len(words), chunk_size):
                        chunk = " ".join(words[i:i + chunk_size])
                        accumulated += (" " + chunk if accumulated else chunk)
                        acc_html = format_chat_content_to_html(accumulated)
                        stream_holder.markdown(
                            f"<div class='chat-msg-row chat-msg-ai-row'>"
                            f"<div class='chat-ai-avatar'>{AI_AVATAR_TAG}</div>"
                            f"<div class='chat-msg-ai'>{acc_html}</div>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )
                        time.sleep(0.02)

                    st.session_state.chat_history.append({"role": "assistant", "content": ai_reply})
                    st.session_state["pending_ai_query"] = None
                    st.rerun()

                st.markdown("<div id='chat-bottom-anchor'></div>", unsafe_allow_html=True)

            # Auto-scroll cho khung chat (TUYỆT ĐỐI KHÔNG dùng scrollIntoView để tránh kéo cả trang chính)
            components.html(
                """
                <script>
                (function() {
                    try {
                        var doc = window.parent.document;
                        var win = window.parent;

                        function scrollOnlyChatBox() {
                            var sentinel = doc.getElementById('chat-box-sentinel');
                            var container = null;
                            if (sentinel) {
                                container = sentinel.closest('[data-testid="stVerticalBlock"]');
                            }
                            if (!container) {
                                var rows = doc.querySelectorAll('.chat-msg-row');
                                if (rows.length > 0) {
                                    var last = rows[rows.length - 1];
                                    container = last.closest('[data-testid="stVerticalBlock"]');
                                }
                            }
                            if (container) {
                                var curY = win.scrollY;
                                var curX = win.scrollX;
                                // TUYỆT ĐỐI KHÔNG dùng scrollIntoView; chỉ cập nhật scrollTop của riêng container
                                container.scrollTop = container.scrollHeight;
                                // Bảo vệ vị trí trang chính không bị dịch chuyển dù chỉ 1 pixel
                                if (win.scrollY !== curY) {
                                    win.scrollTo(curX, curY);
                                }
                            }
                        }

                        scrollOnlyChatBox();
                        setTimeout(scrollOnlyChatBox, 40);
                        setTimeout(scrollOnlyChatBox, 120);
                        setTimeout(scrollOnlyChatBox, 250);
                        setTimeout(scrollOnlyChatBox, 500);

                        // Quan sát thay đổi nội dung (khi AI chạy chữ) để cuộn bám đáy tự động
                        var sentinel = doc.getElementById('chat-box-sentinel');
                        if (sentinel) {
                            var c = sentinel.closest('[data-testid="stVerticalBlock"]');
                            if (c) {
                                var obs = new MutationObserver(function() {
                                    scrollOnlyChatBox();
                                });
                                obs.observe(c, { childList: true, subtree: true, characterData: true });
                            }
                        }
                    } catch(e) {}
                })();
                </script>
                """,
                height=0,
            )

            # Form nhập câu chat (dùng st.form để không tự cuộn màn hình)
            with st.form("quick_ai_form", clear_on_submit=True):
                f_c1, f_c2 = st.columns([3.6, 1.4], gap="small")
                with f_c1:
                    ai_query = st.text_input(
                        "Nhập yêu cầu tư vấn",
                        placeholder="Nhắn tin với trợ lý AI...",
                        label_visibility="collapsed",
                    )
                with f_c2:
                    btn_ai_submit = st.form_submit_button("Gửi", type="primary", use_container_width=True)

            if btn_ai_submit and ai_query.strip():
                user_text = ai_query.strip()
                st.session_state.chat_history.append({"role": "user", "content": user_text})
                st.session_state["pending_ai_query"] = user_text
                st.rerun()

            # 4 Nút gợi ý nhanh (Quick Prompts) xếp 2x2 gọn đẹp dạng pill
            st.markdown('<div class="quick-prompts-wrap">', unsafe_allow_html=True)
            qp_r1_c1, qp_r1_c2 = st.columns(2, gap="small")
            with qp_r1_c1:
                if st.button("Văn phòng 15tr", key="qp_btn_vp", use_container_width=True):
                    st.session_state.chat_history.append({"role": "user", "content": "Tư vấn cấu hình Văn phòng 15 triệu"})
                    st.session_state["pending_ai_query"] = "Tư vấn cấu hình Văn phòng 15 triệu"
                    st.rerun()
            with qp_r1_c2:
                if st.button("Gaming 20tr", key="qp_btn_gm", use_container_width=True):
                    st.session_state.chat_history.append({"role": "user", "content": "Tư vấn cấu hình Gaming 20 triệu"})
                    st.session_state["pending_ai_query"] = "Tư vấn cấu hình Gaming 20 triệu"
                    st.rerun()

            qp_r2_c1, qp_r2_c2 = st.columns(2, gap="small")
            with qp_r2_c1:
                if st.button("Chơi GTA 5", key="qp_btn_gta", use_container_width=True):
                    st.session_state.chat_history.append({"role": "user", "content": "Cần 1 bộ PC chơi GTA 5"})
                    st.session_state["pending_ai_query"] = "Cần 1 bộ PC chơi GTA 5"
                    st.rerun()
            with qp_r2_c2:
                if st.button("Lập trình AI 35tr", key="qp_btn_ai", use_container_width=True):
                    st.session_state.chat_history.append({"role": "user", "content": "Tư vấn máy Lập trình AI 35 triệu"})
                    st.session_state["pending_ai_query"] = "Tư vấn máy Lập trình AI 35 triệu"
                    st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<div style='height: 4px;'></div>", unsafe_allow_html=True)

        # Tính tổng chi phí có tính theo số lượng
        total_cost = 0
        for _, cat_key, state_key, _ in CATEGORIES:
            item = current_selected.get(
                {"cpus": "cpu", "mainboards": "mainboard", "rams": "ram", "gpus": "gpu",
                 "psus": "psu", "cases": "case", "storages": "storage", "coolers": "cooler", "hdds": "hdd"}.get(cat_key)
            )
            if item:
                qty = st.session_state.get(f"qty_{cat_key}", 1)
                total_cost += item.get("price", 0) * qty

        cpu_item = current_selected.get("cpu")
        gpu_item = current_selected.get("gpu")

        rec_psu = 0
        if cpu_item and gpu_item:
            rec_psu = calculate_min_psu_wattage(cpu_item, gpu_item, extra_storage_count=extra_storage_count)

        has_errors = len(fc_result.errors) > 0
        is_complete = all(
            current_selected.get(k) is not None
            for k in ["cpu", "mainboard", "ram", "gpu", "psu", "case", "storage"]
        )

        with st.container(border=True):
            st.markdown(
                f"""<div style='font-size: 1.05rem; font-weight: 700; color: var(--text-main); display: flex; align-items: center; gap: 8px;'>
                    <div style='width: 28px; height: 28px; border-radius: 6px; background: rgba(37, 99, 235, 0.18); border: 1px solid rgba(59, 130, 246, 0.35); display: flex; align-items: center; justify-content: center; flex-shrink: 0;'>
                        {get_svg_icon('chart', '#38BDF8', 16)}
                    </div>
                    Tổng kết cấu hình
                </div>""",
                unsafe_allow_html=True,
            )
            st.markdown(f'<div class="total-price-display">{fmt_vnd(total_cost)}</div>', unsafe_allow_html=True)

            budget_int = parse_budget_int(st.session_state.budget)
            diff = budget_int - total_cost
            pct_budget = (total_cost / budget_int * 100) if budget_int > 0 else 0
            bar_fill_width = min(100.0, pct_budget)

            if total_cost <= budget_int:
                bar_color = "linear-gradient(90deg, #3B82F6, #10B981)"
                status_text = f"Tiết kiệm: {fmt_vnd(diff)} (-{(diff/budget_int*100):.2f}%)" if diff > 0 else "Khớp hoàn toàn ngân sách"
                status_color = "#15803D" if not is_dark else "#34D399"
                savings_bg = "#DCFCE7" if not is_dark else "rgba(16, 185, 129, 0.2)"
                savings_border = "#86EFAC" if not is_dark else "rgba(16, 185, 129, 0.35)"
                savings_text = "#15803D" if not is_dark else "#34D399"
                c_icon_col = "#15803D" if not is_dark else "#34D399"
                compat_badge = f'<div style="color: {c_icon_col}; font-weight: 600; font-size: 0.88rem; display: flex; align-items: center; gap: 6px;">{get_svg_icon("check_circle", c_icon_col, 15)} <span>Cấu hình phù hợp với ngân sách</span></div>'
            elif total_cost <= int(budget_int * 1.05):
                bar_color = "linear-gradient(90deg, #3B82F6, #F59E0B)"
                status_text = f"Vượt nhẹ: {fmt_vnd(abs(diff))} (+{pct_budget - 100:.1f}%)"
                status_color = "#854D0E" if not is_dark else "#FBBF24"
                savings_bg = "#FEF3C7" if not is_dark else "rgba(245, 158, 11, 0.2)"
                savings_border = "#FCD34D" if not is_dark else "rgba(245, 158, 11, 0.35)"
                savings_text = "#854D0E" if not is_dark else "#FBBF24"
                c_icon_col = "#D97706" if not is_dark else "#FBBF24"
                compat_badge = f'<div style="color: {c_icon_col}; font-weight: 600; font-size: 0.88rem; display: flex; align-items: center; gap: 6px;">{get_svg_icon("info", c_icon_col, 15)} <span>Vượt nhẹ trong biên độ linh hoạt (+5%)</span></div>'
            else:
                bar_color = "linear-gradient(90deg, #F59E0B, #EF4444)"
                status_text = f"Vượt ngân sách: {fmt_vnd(abs(diff))} (+{pct_budget - 100:.1f}%)"
                status_color = "#991B1B" if not is_dark else "#F87171"
                savings_bg = "#FEE2E2" if not is_dark else "rgba(239, 68, 68, 0.2)"
                savings_border = "#FCA5A5" if not is_dark else "rgba(239, 68, 68, 0.35)"
                savings_text = "#991B1B" if not is_dark else "#F87171"
                c_icon_col = "#DC2626" if not is_dark else "#F87171"
                compat_badge = f'<div style="color: {c_icon_col}; font-weight: 600; font-size: 0.88rem; display: flex; align-items: center; gap: 6px;">{get_svg_icon("alert_triangle", c_icon_col, 15)} <span>Vượt quá ngân sách</span></div>'

            # Thanh tiến độ ngân sách trực quan
            st.markdown(
                f"""
                <div style="margin: 4px 0 10px 0;">
                    <div style="display: inline-block; background: {savings_bg}; border: 1px solid {savings_border}; color: {savings_text}; border-radius: 9999px; padding: 4px 12px; font-weight: 700; font-size: 0.86rem; margin-bottom: 8px;">
                        {status_text}
                    </div>
                    <div style="background-color: var(--bg-track); border-radius: 9999px; height: 9px; width: 100%; overflow: hidden; margin-bottom: 6px;">
                        <div style="background: {bar_color}; width: {bar_fill_width}%; height: 100%; border-radius: 9999px; transition: width 0.3s ease;"></div>
                    </div>
                    <div style="display: flex; justify-content: space-between; font-size: 0.86rem; color: var(--text-sub);">
                        <span>Chi phí: <b style="color: var(--text-main);">{fmt_vnd(total_cost)}</b> / {fmt_vnd(budget_int)}</span>
                    </div>
                    <div style="margin-top: 8px;">
                        {compat_badge}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Tính toán Hiệu năng dự kiến
            c_score = cpu_item.get("score", 15000) if cpu_item else 0
            cpu_p = min(99, max(45, int(c_score / 33000 * 100))) if cpu_item else 0

            g_score = gpu_item.get("score", 0) if (gpu_item and gpu_item.get("id") != "gpu_none") else 0
            gpu_p = min(99, max(30, int(g_score / 35000 * 100))) if g_score > 0 else 0

            ram_it = current_selected.get("ram")
            if ram_it:
                q_r = st.session_state.get("qty_rams", 1)
                tot_r = ram_it.get("capacity_gb", 16) * q_r
                ram_p = 98 if tot_r >= 32 else (93 if tot_r >= 16 else 75)
            else:
                ram_p = 0

            ssd_it = current_selected.get("storage")
            if ssd_it:
                storage_p = 94 if "NVMe" in ssd_it.get("type", "") or "nvme" in ssd_it.get("name", "").lower() else 82
            else:
                storage_p = 0

            # Section Hiệu năng dự kiến
            st.markdown(
                f"""
                <div style="margin: 14px 0 12px 0;">
                    <div style="font-size: 0.92rem; font-weight: 700; color: var(--text-main); margin-bottom: 8px;">Hiệu năng dự kiến</div>
                    <div class="perf-row">
                        <span class="perf-label">CPU</span>
                        <div class="perf-track"><div class="perf-fill" style="width: {cpu_p}%;"></div></div>
                        <span class="perf-val">{cpu_p}%</span>
                    </div>
                    <div class="perf-row">
                        <span class="perf-label">VGA</span>
                        <div class="perf-track"><div class="perf-fill" style="width: {gpu_p}%;"></div></div>
                        <span class="perf-val">{gpu_p}%</span>
                    </div>
                    <div class="perf-row">
                        <span class="perf-label">RAM</span>
                        <div class="perf-track"><div class="perf-fill" style="width: {ram_p}%;"></div></div>
                        <span class="perf-val">{ram_p}%</span>
                    </div>
                    <div class="perf-row">
                        <span class="perf-label">Ổ cứng</span>
                        <div class="perf-track"><div class="perf-fill" style="width: {storage_p}%;"></div></div>
                        <span class="perf-val">{storage_p}%</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # 2 Thẻ Info Cards
            compat_txt = "Tương thích hoàn toàn" if (is_complete and not has_errors) else ("Xung đột linh kiện" if has_errors else "Đang lựa chọn")
            compat_c = ("#15803D" if not is_dark else "#10B981") if (is_complete and not has_errors) else (("#DC2626" if not is_dark else "#EF4444") if has_errors else ("#475569" if not is_dark else "#94A3B8"))

            info_card_1_bg = "#EFF6FF" if not is_dark else "var(--bg-info-card)"
            info_card_1_border = "#BFDBFE" if not is_dark else "var(--border-info-card)"
            info_card_1_title_color = "#475569" if not is_dark else "var(--text-sub)"
            info_card_1_val_color = "#1E40AF" if not is_dark else "var(--text-main)"

            info_card_2_bg = "#ECFDF5" if not is_dark else "var(--bg-info-card)"
            info_card_2_border = "#A7F3D0" if not is_dark else "var(--border-info-card)"
            info_card_2_title_color = "#475569" if not is_dark else "var(--text-sub)"

            st.markdown(
                f"""
                <div style="display: flex; gap: 8px; margin: 12px 0 14px 0;">
                    <div style="flex: 1; background: {info_card_1_bg}; border: 1px solid {info_card_1_border}; border-radius: 8px; padding: 10px 12px; display: flex; align-items: center; gap: 10px;">
                        <div style="width: 32px; height: 32px; border-radius: 8px; background: rgba(245, 158, 11, 0.15); border: 1px solid rgba(245, 158, 11, 0.3); display: flex; align-items: center; justify-content: center; flex-shrink: 0;">
                            {get_svg_icon('lightning', '#F59E0B', 18)}
                        </div>
                        <div>
                            <div style="font-size: 0.78rem; color: {info_card_1_title_color}; font-weight: 500;">Nguồn đề xuất</div>
                            <div style="font-size: 0.95rem; font-weight: 700; color: {info_card_1_val_color};">{'>= ' + str(rec_psu) + 'W' if rec_psu else 'Chưa tính'}</div>
                        </div>
                    </div>
                    <div style="flex: 1; background: {info_card_2_bg}; border: 1px solid {info_card_2_border}; border-radius: 8px; padding: 10px 12px; display: flex; align-items: center; gap: 10px;">
                        <div style="width: 32px; height: 32px; border-radius: 8px; background: rgba(16, 185, 129, 0.15); border: 1px solid rgba(16, 185, 129, 0.3); display: flex; align-items: center; justify-content: center; flex-shrink: 0;">
                            {get_svg_icon('shield', compat_c, 18)}
                        </div>
                        <div>
                            <div style="font-size: 0.78rem; color: {info_card_2_title_color}; font-weight: 500;">Trạng thái</div>
                            <div style="font-size: 0.92rem; font-weight: 700; color: {compat_c};">{compat_txt}</div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Danh sách lỗi vi phạm ràng buộc cứng
            if fc_result.errors:
                st.markdown("<span style='color: #F87171; font-weight: 700; font-size: 0.92rem;'>Lỗi tương thích:</span>", unsafe_allow_html=True)
                for err in fc_result.errors:
                    st.markdown(f"<div style='font-size: 0.88rem; color: #F87171; margin-bottom: 4px;'>• {err}</div>", unsafe_allow_html=True)

            # Danh sách cảnh báo khuyến nghị mềm
            if fc_result.warnings:
                st.markdown("<span style='color: #FBBF24; font-weight: 700; font-size: 0.92rem;'>Lưu ý khuyến nghị:</span>", unsafe_allow_html=True)
                for warn in fc_result.warnings:
                    st.markdown(f"<div style='font-size: 0.88rem; color: #FBBF24; margin-bottom: 4px;'>• {warn}</div>", unsafe_allow_html=True)

            # ── LƯU Ý & NHẬN XÉT TỪ AI ──
            q_dict = {
                "rams": st.session_state.get("qty_rams", 1),
                "storages": st.session_state.get("qty_storages", 1),
                "hdds": st.session_state.get("qty_hdds", 1),
            }
            ai_review = generate_build_review(
                current_selected, q_dict, parse_budget_int(st.session_state.budget), st.session_state.usage, fc_result
            )
            st.markdown(
                f"""
                <div style="background: var(--bg-review-box); border: 1px solid var(--border-review-box); border-radius: 8px; padding: 12px 14px; margin-top: 10px;">
                    <div style="font-size: 0.92rem; font-weight: 700; color: var(--text-review-title); display: flex; align-items: center; gap: 6px; margin-bottom: 6px;">
                        {get_svg_icon('info', '#B45309' if not is_dark else '#F59E0B', 16)}
                        Lưu ý & Nhận xét chi tiết:
                    </div>
                    <div style="font-size: 0.88rem; line-height: 1.55; color: var(--text-review-box);">
                        {ai_review.replace(chr(10) + chr(10), '<br><br>')}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Gợi ý nâng cấp đáng giá từ AI
            if is_complete and not has_errors:
                upgrades_display = suggest_upgrades(current_selected, DATA, parse_budget_int(st.session_state.budget), st.session_state.usage)
                if upgrades_display:
                    st.session_state["pending_upgrade_suggestion"] = upgrades_display[0]
                    up_header_col = "#0F172A" if not is_dark else "#F59E0B"
                    up_icon_type = "check_circle" if not is_dark else "star"
                    up_icon_col = "#10B981" if not is_dark else "#F59E0B"
                    st.markdown(
                        f"""
                        <div style="font-size: 0.92rem; font-weight: 700; color: {up_header_col}; display: flex; align-items: center; gap: 6px; margin: 12px 0 6px 0;">
                            {get_svg_icon(up_icon_type, up_icon_col, 16)}
                            Gợi ý nâng cấp đáng giá:
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    for idx, up in enumerate(upgrades_display):
                        st.markdown(
                            f"""<div style='font-size: 0.88rem; background: var(--bg-info-card); border: 1px solid var(--border-info-card); padding: 10px 12px; border-radius: 6px; margin-top: 6px; color: var(--text-main);'>
                                <div style='display:flex; align-items:center; gap:6px; margin-bottom:4px;'>
                                    {get_svg_icon('check_circle', '#10B981', 15)}
                                    <span><b>Bù thêm {fmt_vnd(up['added_cost'])}:</b> Nâng từ <i>{up['current_name']}</i> lên <b style='color:var(--text-price);'>{up['upgrade_name']}</b></span>
                                </div>
                                <div style='color: var(--text-sub); font-size: 0.84rem; padding-left: 21px;'>• {up['benefit']}</div>
                            </div>""",
                            unsafe_allow_html=True,
                        )
                        btn_apply = st.button(
                            f"Áp dụng nâng lên {up['upgrade_name']} (+{fmt_vnd(up['added_cost'])})",
                            key=f"btn_apply_upgrade_{idx}",
                            use_container_width=True,
                        )
                        if btn_apply:
                            target_cat = up["category"]
                            state_k = {
                                "cpu": "sel_cpus", "mainboard": "sel_mainboards", "ram": "sel_rams",
                                "gpu": "sel_gpus", "storage": "sel_storages", "psu": "sel_psus",
                                "case": "sel_cases", "cooler": "sel_coolers", "hdd": "sel_hdds"
                            }.get(target_cat, f"sel_{target_cat}s")
                            st.session_state[state_k] = up["target_id"]
                            st.session_state.budget = parse_budget_int(st.session_state.budget) + up["added_cost"]
                            st.session_state["budget_version"] = st.session_state.get("budget_version", 0) + 1
                            st.session_state["pending_upgrade_suggestion"] = None
                            st.session_state.chat_history.append({"role": "user", "content": f"Đồng ý nâng cấp lên {up['upgrade_name']}"})
                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "content": f"Mình đã cập nhật nâng cấp sang {up['upgrade_name']} trên bảng linh kiện rồi bạn nhé!"
                            })
                            st.rerun()

            # Nút Tải bảng cấu hình (CSV) và Tải báo giá (HTML)
            st.markdown("<hr style='margin: 12px 0; border: none; border-top: 1px solid var(--hr-line);'>", unsafe_allow_html=True)
            csv_rows = [
                ["Danh mục", "Tên linh kiện", "Hãng sản xuất", "Số lượng", "Đơn giá (VNĐ)", "Thành tiền (VNĐ)"]
            ]
            for cat_name, cat_key, state_key, _ in CATEGORIES:
                it = current_selected.get(
                    {"cpus": "cpu", "mainboards": "mainboard", "rams": "ram", "gpus": "gpu",
                     "psus": "psu", "cases": "case", "storages": "storage", "coolers": "cooler", "hdds": "hdd"}.get(cat_key)
                )
                if it and it.get("id") not in ("gpu_none", "hdd_none", "cooler_stock"):
                    q = st.session_state.get(f"qty_{cat_key}", 1)
                    u_price = it.get("price", 0)
                    csv_rows.append([cat_name, it["name"], get_item_brand(it), str(q), f"{u_price:,}", f"{u_price * q:,}"])
                elif it and it.get("id") in ("gpu_none", "cooler_stock"):
                    csv_rows.append([cat_name, it["name"], "Tiêu chuẩn", "1", "0", "0"])

            csv_rows.append([])
            csv_rows.append(["Tổng chi phí", "", "", "", "", f"{total_cost:,} VNĐ"])
            csv_rows.append(["Công suất nguồn khuyến nghị", f"{rec_psu}W" if rec_psu else "Chưa tính", "", "", "", ""])
            csv_rows.append(["Nhu cầu sử dụng", st.session_state.usage, "", "", "", ""])

            output_str = io.StringIO()
            writer = csv.writer(output_str)
            writer.writerows(csv_rows)
            csv_bytes = output_str.getvalue().encode("utf-8-sig")

            html_report = generate_html_quotation(current_selected, CATEGORIES, total_cost, rec_psu, st.session_state.usage)

            dl_c1, dl_c2 = st.columns(2)
            with dl_c1:
                st.markdown('<div class="csv-dl-wrap">', unsafe_allow_html=True)
                st.download_button(
                    label="Tải bảng cấu hình (CSV)",
                    data=csv_bytes,
                    file_name="cau_hinh_may_tinh.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
                st.markdown('</div>', unsafe_allow_html=True)
            with dl_c2:
                st.markdown('<div class="html-dl-wrap">', unsafe_allow_html=True)
                st.download_button(
                    label="Tải báo giá (PDF)",
                    data=html_report.encode("utf-8"),
                    file_name="bao_gia_cau_hinh_pc.html",
                    mime="text/html",
                    use_container_width=True,
                )
                st.markdown('</div>', unsafe_allow_html=True)


# ============================================================================
# TAB 2: CƠ SỞ TRI THỨC & THỰC NGHIỆM THUẬT TOÁN
# ============================================================================
with tab_benchmark:
    st.markdown("##### Cơ sở tri thức - Tập luật suy diễn (R01 - R10)")
    st.caption("Hệ thống quy tắc biểu diễn tri thức theo cấu trúc luật IF-THEN trong Hệ chuyên gia.")

    rules_table = []
    for r in RULES:
        rules_table.append({
            "Mã luật": r["rule_id"],
            "Tên quy tắc": r["name"],
            "Phân loại": "Ràng buộc cứng (Hard)" if r["category"] == "Hard" else "Ràng buộc mềm (Soft)",
            "Biểu thức logic": r["logic_expr"],
            "Mô tả hành động (THEN)": r["action_desc"],
        })

    df_rules = pd.DataFrame(rules_table)
    st.dataframe(
        df_rules,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Mã luật": st.column_config.TextColumn("Mã luật", width="small"),
            "Tên quy tắc": st.column_config.TextColumn("Tên quy tắc", width="medium"),
            "Phân loại": st.column_config.TextColumn("Phân loại", width="small"),
            "Biểu thức logic": st.column_config.TextColumn("Biểu thức logic", width="medium"),
            "Mô tả hành động (THEN)": st.column_config.TextColumn("Mô tả hành động", width="large"),
        },
    )

    st.divider()

    st.markdown("##### Thực nghiệm so sánh thuật toán (Benchmark)")
    st.caption("Đối chiếu trực tiếp hiệu năng giữa thuật toán Vét cạn (Brute-Force) và thuật toán Nhánh & Cận (Branch & Bound).")

    bm_c1, bm_c2, bm_c3, bm_c4 = st.columns([1.2, 1.5, 1.5, 1.2])
    with bm_c1:
        bm_size = st.selectbox(
            "Số linh kiện mỗi loại",
            options=[3, 4, 5],
            index=1,
            key="bm_size_select",
        )
    with bm_c2:
        bm_budget = st.number_input(
            "Ngân sách thực nghiệm (VNĐ)",
            min_value=10_000_000,
            max_value=60_000_000,
            value=25_000_000,
            step=1_000_000,
            format="%d",
            key="bm_budget_select",
        )
    with bm_c3:
        bm_usage = st.selectbox(
            "Nhu cầu thực nghiệm",
            options=list(USAGE_PROFILES.keys()),
            key="bm_usage_select",
        )
    with bm_c4:
        st.write("")
        st.write("")
        btn_run_bm = st.button("Chạy so sánh tốc độ", type="primary", use_container_width=True)

    if btn_run_bm:
        with st.spinner(f"Đang chạy so sánh trên tập dữ liệu {bm_size}^7 tổ hợp..."):
            subset = {
                "cpus": DATA.get("cpus", [])[:bm_size],
                "mainboards": DATA.get("mainboards", [])[:bm_size],
                "rams": DATA.get("rams", [])[:bm_size],
                "gpus": DATA.get("gpus", [])[:bm_size],
                "psus": DATA.get("psus", [])[:bm_size],
                "cases": DATA.get("cases", [])[:bm_size],
                "storages": DATA.get("storages", [])[:bm_size],
            }

            bf_opt = BruteForceOptimizer(subset)
            bf_builds, bf_stats = bf_opt.optimize(
                budget=int(bm_budget),
                usage=bm_usage,
                items_per_category=bm_size,
                top_k=1,
            )

            bb_opt = BranchAndBoundOptimizer(subset)
            bb_builds, bb_stats = bb_opt.optimize(
                budget=int(bm_budget),
                usage=bm_usage,
                top_k=1,
            )

            st.session_state.bm_result = {
                "bf_stats": bf_stats,
                "bb_stats": bb_stats,
                "bf_score": bf_builds[0].fitness_score if bf_builds else 0.0,
                "bb_score": bb_builds[0].fitness_score if bb_builds else 0.0,
            }

    if "bm_result" in st.session_state:
        res = st.session_state.bm_result
        bf_s = res["bf_stats"]
        bb_s = res["bb_stats"]

        speedup = bf_s.execution_time_ms / max(bb_s.execution_time_ms, 0.001)
        node_reduction = bf_s.nodes_visited / max(bb_s.nodes_visited, 1)

        # Bảng dữ liệu so sánh
        bm_df = pd.DataFrame([
            {
                "Thuật toán": "Brute-Force Backtracking",
                "Số nút đã duyệt": f"{bf_s.nodes_visited:,}",
                "Số nhánh cắt tỉa": "--",
                "Cấu hình hợp lệ": f"{bf_s.valid_builds_found:,}",
                "Thời gian chạy (ms)": f"{bf_s.execution_time_ms:.2f}",
                "Tốc độ tương đối": "1.0x (Chuẩn)",
                "Điểm tối ưu đạt được": f"{res['bf_score']:.4f}",
            },
            {
                "Thuật toán": "Branch & Bound (Heuristic)",
                "Số nút đã duyệt": f"{bb_s.nodes_visited:,}",
                "Số nhánh cắt tỉa": f"{bb_s.pruned_branches:,}",
                "Cấu hình hợp lệ": f"{bb_s.valid_builds_found:,}",
                "Thời gian chạy (ms)": f"{bb_s.execution_time_ms:.2f}",
                "Tốc độ tương đối": f"{speedup:.1f}x nhanh hơn",
                "Điểm tối ưu đạt được": f"{res['bb_score']:.4f}",
            },
        ])
        st.dataframe(bm_df, use_container_width=True, hide_index=True)

        is_exact = abs(res["bf_score"] - res["bb_score"]) < 0.001
        if is_exact:
            st.success(
                f"Branch & Bound tìm ra đúng nghiệm tối ưu toàn cục (Điểm = {res['bb_score']:.4f}), "
                f"nhanh hơn {speedup:.1f} lần và giảm {node_reduction:.1f} lần số nút cần duyệt so với Brute-Force."
            )
        else:
            st.info("Kết quả đã được tính toán dựa trên tập dữ liệu thực nghiệm.")

        # 2 biểu đồ cột đơn giản (Số nút duyệt & Thời gian thực thi)
        chart_c1, chart_c2 = st.columns(2)

        with chart_c1:
            fig_nodes = go.Figure(data=[go.Bar(
                x=["Brute-Force", "Branch & Bound"],
                y=[bf_s.nodes_visited, bb_s.nodes_visited],
                marker_color=["#EF4444", "#10B981"],
                text=[f"{bf_s.nodes_visited:,}", f"{bb_s.nodes_visited:,}"],
                textposition="auto",
            )])
            chart_text_color = "#94A3B8" if is_dark else "#475569"
            chart_title_color = "#F1F5F9" if is_dark else "#0F172A"
            fig_nodes.update_layout(
                title=dict(text="Số nút đã duyệt (ít hơn là tốt hơn)", font=dict(size=14, color=chart_title_color)),
                yaxis_title="Số nút",
                height=320,
                margin=dict(t=40, b=20, l=20, r=20),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color=chart_text_color),
            )
            st.plotly_chart(fig_nodes, use_container_width=True)

        with chart_c2:
            fig_time = go.Figure(data=[go.Bar(
                x=["Brute-Force", "Branch & Bound"],
                y=[bf_s.execution_time_ms, bb_s.execution_time_ms],
                marker_color=["#EF4444", "#2563EB"],
                text=[f"{bf_s.execution_time_ms:.2f} ms", f"{bb_s.execution_time_ms:.2f} ms"],
                textposition="auto",
            )])
            fig_time.update_layout(
                title=dict(text="Thời gian thực thi (ms) (nhanh hơn là tốt hơn)", font=dict(size=14, color=chart_title_color)),
                yaxis_title="Thời gian (ms)",
                height=320,
                margin=dict(t=40, b=20, l=20, r=20),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color=chart_text_color),
            )
            st.plotly_chart(fig_time, use_container_width=True)# ── 2 CARD DƯỚI CÙNG: PHÂN BỔ CHI PHÍ & PROMO BANNER ──
        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
        bot_c1, bot_c2 = st.columns([1.0, 1.0], gap="small")

        with bot_c1:
            with st.container(border=True):
                st.markdown(
                    f"""<div style='font-size: 0.98rem; font-weight: 700; color: var(--text-main); display: flex; align-items: center; gap: 6px; margin-bottom: 8px;'>
                        {get_svg_icon("chart", "#38BDF8", 16)}
                        Phân bổ chi phí linh kiện
                    </div>""",
                    unsafe_allow_html=True,
                )
                
                # Tính toán chi phí chi tiết từng nhóm
                c_cpu = current_selected["cpu"].get("price", 0) if current_selected.get("cpu") else 0
                c_mb = current_selected["mainboard"].get("price", 0) if current_selected.get("mainboard") else 0
                c_ram = (current_selected["ram"].get("price", 0) * st.session_state.get("qty_rams", 1)) if current_selected.get("ram") else 0
                c_ssd = (current_selected["storage"].get("price", 0) * st.session_state.get("qty_storages", 1)) if current_selected.get("storage") else 0
                c_hdd = (current_selected["hdd"].get("price", 0) * st.session_state.get("qty_hdds", 1)) if (current_selected.get("hdd") and current_selected["hdd"].get("id") != "hdd_none") else 0
                c_gpu = (current_selected["gpu"].get("price", 0) if current_selected.get("gpu") and current_selected["gpu"].get("id") != "gpu_none" else 0)
                c_psu = (current_selected["psu"].get("price", 0) if current_selected.get("psu") else 0)
                c_case = (current_selected["case"].get("price", 0) if current_selected.get("case") else 0)
                c_cooler = (current_selected["cooler"].get("price", 0) if current_selected.get("cooler") and current_selected["cooler"].get("id") != "cooler_stock" else 0)
                c_other = c_psu + c_case + c_cooler

                t_cost = c_cpu + c_mb + c_ram + c_ssd + c_hdd + c_gpu + c_other
                
                if t_cost > 0:
                    donut_labels = ["CPU", "Mainboard", "RAM", "SSD"]
                    donut_values = [c_cpu, c_mb, c_ram, c_ssd]
                    donut_colors = ["#2563EB", "#06B6D4", "#8B5CF6", "#EC4899"]
                    
                    if c_gpu > 0:
                        donut_labels.append("VGA")
                        donut_values.append(c_gpu)
                        donut_colors.append("#3B82F6")
                    if c_hdd > 0:
                        donut_labels.append("HDD")
                        donut_values.append(c_hdd)
                        donut_colors.append("#F59E0B")
                    if c_other > 0:
                        donut_labels.append("Khác")
                        donut_values.append(c_other)
                        donut_colors.append("#64748B")

                    d_col_left, d_col_right = st.columns([1.1, 1.4])
                    with d_col_left:
                        fig_donut = go.Figure(data=[go.Pie(
                            labels=donut_labels,
                            values=donut_values,
                            hole=0.62,
                            textinfo="none",
                            marker=dict(colors=donut_colors),
                            hoverinfo="label+percent",
                        )])
                        total_m = f"{t_cost / 1_000_000:.2f}M"
                        fig_donut.add_annotation(
                            text=f"<b>{total_m}</b><br><span style='font-size:10px;'>VNĐ</span>",
                            x=0.5, y=0.5,
                            font_size=13,
                            font_color="#0F172A" if not is_dark else "#F1F5F9",
                            showarrow=False,
                        )
                        fig_donut.update_layout(
                            showlegend=False,
                            height=150,
                            margin=dict(t=0, b=0, l=0, r=0),
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                        )
                        st.plotly_chart(fig_donut, use_container_width=True, config={"displayModeBar": False})

                    with d_col_right:
                        legend_items = []
                        for l_idx, l_name in enumerate(donut_labels):
                            val = donut_values[l_idx]
                            pct = (val / t_cost * 100) if t_cost > 0 else 0
                            col = donut_colors[l_idx]
                            legend_items.append(
                                f"""<div style='display:flex; justify-content:space-between; align-items:center; font-size:0.85rem; margin-bottom:4px;'>
                                    <span><span style='color:{col}; font-weight:bold; margin-right:5px;'>●</span><span style='color:var(--text-sub);'>{l_name}:</span> <b style='color:var(--text-main);'>{pct:.1f}%</b></span>
                                    <span style='color:var(--text-sub); font-size:0.78rem;'>({fmt_vnd(val)})</span>
                                </div>"""
                            )
                        st.markdown("<div style='padding-top: 6px;'>" + "".join(legend_items) + "</div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div style='color: #64748B; font-size: 0.88rem; text-align: center; padding: 30px 0;'>Chưa có dữ liệu linh kiện để phân bổ chi phí.</div>", unsafe_allow_html=True)

        with bot_c2:
            with st.container(border=True):
                st.markdown(
                    f"""
                    <div style="height: 100%; min-height: 175px; display: flex; align-items: center; position: relative; overflow: hidden; border-radius: 12px; background: linear-gradient(135deg, #0F2046 0%, #1E3A8A 55%, #4C1D95 100%); padding: 20px 24px; box-shadow: 0 4px 20px rgba(30, 58, 138, 0.35);">
                        <div style="position: absolute; right: -25px; bottom: -25px; width: 140px; height: 140px; background: radial-gradient(circle, rgba(56, 189, 248, 0.35) 0%, transparent 70%); pointer-events: none;"></div>
                        <div style="display: flex; align-items: center; gap: 18px; z-index: 2; width: 100%;">
                            <div style="display: flex; align-items: center; justify-content: center; width: 56px; height: 56px; border-radius: 14px; background: rgba(56, 189, 248, 0.18); border: 1px solid rgba(56, 189, 248, 0.4); flex-shrink: 0; box-shadow: 0 0 16px rgba(56, 189, 248, 0.35);">
                                {get_svg_icon("rocket", "#38BDF8", 28)}
                            </div>
                            <div style="flex: 1;">
                                <div style="font-size: 1.35rem; font-weight: 800; color: #FFFFFF !important; letter-spacing: -0.01em; line-height: 1.25; font-style: italic;">
                                    Build Smart <span style="color: #38BDF8 !important;">• Play Better</span>
                                </div>
                                <div style="font-size: 0.9rem; color: #E2E8F0 !important; font-style: normal; margin-top: 6px; font-weight: 500;">
                                    Cấu hình tối ưu cho mọi nhu cầu của bạn!
                                </div>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    
