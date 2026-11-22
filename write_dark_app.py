from pathlib import Path

orig = Path("ai_pc_advisor/app.py").read_text(encoding="utf-8")

# Check markers
m1 = "# CSS TÙY CHỈNH - GIAO DIỆN SÁNG (LIGHT THEME) HIỆN ĐẠI & TỐI GIẢN CHUYÊN NGHIỆP"
m2 = "# TẢI DỮ LIỆU"
m3 = "# PHẦN ĐẦU TRANG (HEADER)"
m4 = "# TAB 1: XÂY DỰNG & TƯ VẤN CẤU HÌNH PC"
m5 = "col_parts, col_summary = st.columns([1.98, 1.02], gap=\"medium\")"
m6 = "with col_summary:"
m7 = "active_items = []"
m8 = "with tab_benchmark:"

markers = [m1, m2, m3, m4, m5, m6, m7, m8]
for m in markers:
    print(m, "-> found:", m in orig)
