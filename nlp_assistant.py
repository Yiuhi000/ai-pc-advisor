"""
nlp_assistant.py
================
Module Trợ lý Chat AI Xử lý Ngôn ngữ Tự nhiên & Tự động Nhận xét Cấu hình PC.
100% Thuần Python, không sử dụng API Key hay LLM bên ngoài.
Xử lý bằng Regex, từ điển ánh xạ ngữ nghĩa tiếng Việt và luật chuyên gia.
Tuân thủ nguyên tắc: Không Emoji, Tiếng Việt chuẩn xác tự nhiên.
"""

from datetime import datetime
import math
import re
from typing import Any, Dict, List, Optional, Tuple

from knowledge_base import (
    USAGE_PROFILES,
    calculate_min_psu_wattage,
    check_r08_bottleneck,
    is_pcie3_mainboard,
    is_vrm_incompatible,
)


# ===========================================================================
# CƠ CHẾ QUẢN LÝ GOOGLE GEMINI API KEY
# ===========================================================================
def get_gemini_api_key() -> Optional[str]:
    """
    Lấy Google Gemini API Key theo thứ tự ưu tiên:
    1. Key người dùng nhập tay trong st.session_state["gemini_api_key"].
    2. Key cấu hình trên máy chủ từ st.secrets hoặc biến môi trường os.getenv("GEMINI_API_KEY").
       Bỏ qua nếu là chuỗi mẫu 'DAN_API_KEY_CUA_BAN_VAO_DAY' hoặc chuỗi rỗng.
    3. Trả về None nếu không có Key hợp lệ (chuyển sang chế độ chuyên gia nội bộ).
    """
    dummy_patterns = {"", "DAN_API_KEY_CUA_BAN_VAO_DAY", "YOUR_GEMINI_API_KEY", "NONE", "NULL"}

    # 1. Kiểm tra session_state nếu người dùng nhập tay
    try:
        import streamlit as st
        user_key = st.session_state.get("gemini_api_key", "")
        if user_key and isinstance(user_key, str):
            user_key = user_key.strip()
            if user_key and user_key not in dummy_patterns:
                return user_key
    except Exception:
        pass

    # 2. Kiểm tra st.secrets
    try:
        import streamlit as st
        secret_key = st.secrets.get("GEMINI_API_KEY", "")
        if secret_key and isinstance(secret_key, str):
            secret_key = secret_key.strip()
            if secret_key and secret_key not in dummy_patterns:
                return secret_key
    except Exception:
        pass

    # 3. Kiểm tra biến môi trường os.getenv
    import os
    env_key = os.getenv("GEMINI_API_KEY", "")
    if env_key and isinstance(env_key, str):
        env_key = env_key.strip()
        if env_key and env_key not in dummy_patterns:
            return env_key

    return None


SUPPORTED_GEMINI_MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-flash-lite-latest",
    "gemini-3.1-flash-lite",
    "gemini-3.8-flash",
]


def call_gemini_api(prompt: str, api_key: Optional[str] = None, timeout: int = 10) -> Optional[str]:
    """
    Gọi Google Gemini API trực tiếp bằng HTTP REST request (sử dụng thư viện requests).
    Hỗ trợ tự động fallback qua các model mới nhất của Google (gemini-3.5-flash-lite, gemini-flash-lite-latest,...).
    In chi tiết lỗi ra terminal nếu gặp sự cố kết nối hoặc xác thực.
    """
    key = api_key or get_gemini_api_key()
    if not key:
        print("[Gemini API] Không tìm thấy API Key (cả st.session_state và st.secrets đều trống). Chuyển sang fallback.")
        return None

    import requests

    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [
            {
                "parts": [{"text": prompt}]
            }
        ],
        "generationConfig": {
            "temperature": 0.4,
            "maxOutputTokens": 1000,
        }
    }

    for model_name in SUPPORTED_GEMINI_MODELS:
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={key}"
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
            if resp.status_code == 200:
                data = resp.json()
                candidates = data.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    for part in parts:
                        txt = part.get("text", "")
                        if txt and txt.strip():
                            return txt.strip()
            else:
                print(f"[Gemini API Error] Model '{model_name}' - HTTP {resp.status_code}: {resp.text[:300]}")
        except Exception as e:
            print(f"[Gemini API Exception] Model '{model_name}' gặp ngoại lệ: {e}")
            continue

    print("[Gemini API] Không gọi được model nào thành công. Chuyển sang chế độ dự phòng nội bộ.")
    return None


# ===========================================================================
# CÁC HÀM TIỆN ÍCH HỖ TRỢ ĐỊNH DẠNG
# ===========================================================================
def fmt_vnd(amount: int) -> str:
    """Định dạng số tiền VNĐ."""
    return f"{amount:,.0f} VNĐ".replace(",", ".")


# ===========================================================================
# PHẦN 1: BỘ PHÂN TÍCH YÊU CẦU NGƯỜI DÙNG (NLP REQUEST PARSER)
# ===========================================================================
def extract_budget_from_text(text: str) -> Optional[int]:
    """
    Trích xuất số tiền ngân sách từ chuỗi tiếng Việt.
    Nhận diện: '20tr', '20 triệu', '20.000.000', '15m', 'dưới 30 củ', '18tr5', '22,5 tr'...
    """
    t = text.lower()

    # Dạng thập phân tắt: '18tr5', '18 củ 5', '22tr5' -> 18.5 triệu
    m_split = re.search(r'\b(\d+)\s*(?:tr|triệu|củ|cu)\s*(\d+)\b', t)
    if m_split:
        base = int(m_split.group(1))
        dec = float('0.' + m_split.group(2))
        return int((base + dec) * 1_000_000)

    # Dạng số đầy đủ có dấu chấm: '20.000.000', '15.500.000'
    m_dots = re.search(r'(\d{1,3}(?:\.\d{3}){2,3})', t)
    if m_dots:
        return int(m_dots.group(1).replace('.', ''))

    # Dạng số nguyên liền: '20000000', '15000000'
    m_raw = re.search(r'\b(\d{7,9})\b', t)
    if m_raw:
        return int(m_raw.group(1))

    # Dạng số kèm đơn vị rõ ràng: '20tr', '20 triệu', '15m', '30 củ', '25 trđ'
    m_unit = re.search(r'\b(\d+[\.,]?\d*)\s*(?:triệu|trđ|vnd|vnđ|tr\b|củ\b|cu\b|m\b)', t)
    if m_unit:
        val = float(m_unit.group(1).replace(',', '.'))
        return int(val * 1_000_000)

    # Dạng ước lượng: 'tầm 25', 'dưới 30', 'khoảng 15', 'tối đa 22'
    m_approx = re.search(r'(?:tầm|khoảng|dưới|tối đa|mức|ngân sách)\s*(\d+[\.,]?\d*)', t)
    if m_approx:
        val = float(m_approx.group(1).replace(',', '.'))
        if 5 <= val <= 150:  # Trong ngưỡng 5 - 150 triệu
            return int(val * 1_000_000)

    return None


def extract_strict_budget_flag(text: str) -> bool:
    """
    Xác định chế độ ngân sách cứng hay linh hoạt từ câu chat.
    - True (ngân sách cứng): Khi có các từ như: 'tối đa', 'không quá', 'dưới', 'cứng', 'max', 'kịch kim', 'đúng', 'chẵn'.
    - False (ngân sách linh hoạt): Mặc định hoặc khi có từ 'tầm', 'khoảng', 'cỡ', 'quanh', 'loanh quanh', 'xấp xỉ'.
    """
    t = text.lower()
    strict_patterns = [
        r'\btối đa\b', r'\btoi da\b', r'\bkhông quá\b', r'\bkhong qua\b',
        r'\bk quá\b', r'\bk qua\b', r'\bdưới\b', r'\bduoi\b',
        r'\bcứng\b', r'\bcung\b', r'\bmax\b', r'\bkịch kim\b', r'\bkich kim\b',
        r'\bđúng\b', r'\bdung\b', r'\bchẵn\b', r'\bchan\b'
    ]
    return any(re.search(pat, t) for pat in strict_patterns)


def extract_usage_from_text(text: str) -> Optional[str]:
    """
    Ánh xạ nhu cầu sử dụng từ từ khóa tiếng Việt.
    """
    t = text.lower()

    # Gaming (Nhận diện từ hành vi chơi game, tên game hoặc các từ khóa gaming)
    gaming_patterns = [
        r'\b(?:chơi|choi|chiến|chien|game|play)\b',
        r'\b(?:watch\s*dogs?|watchdogs?)\b',
        r'\b(?:elden\s*ring|god\s*of\s*war|rdr2|red\s*dead|forza|assassin|spiderman|spider\s*man|hogwarts|palworld)\b',
    ]
    for p in gaming_patterns:
        if re.search(p, t):
            return "Gaming"

    gaming_keywords = [
        "chơi game", "gaming", "gamming", "gameing", "gemming", "choi game",
        "chơi gem", "danh game", "đánh game", "game", "gamer", "valorant",
        "gta", "cs2", "csgo", "lol", "liên minh", "lien minh", "pubg",
        "genshin", "cyberpunk", "fps", "fifa", "wukong", "black myth", "chiến game", "dota",
        "watchdog", "watchdogs", "watch dog", "watch dogs"
    ]
    for kw in gaming_keywords:
        if kw in t:
            return "Gaming"

    # Đồ họa / Lập trình AI
    graphic_keywords = [
        "đồ họa", "do hoa", "render", "edit video", "dựng phim", "dung phim",
        "photoshop", "premiere", "after effect", "illustrator", "ai",
        "lập trình", "lap trinh", "machine learning", "deep learning",
        "blender", "3d", "autocad", "solidworks", "code"
    ]
    for kw in graphic_keywords:
        if kw in t:
            return "Đồ họa / Lập trình AI"

    # Văn phòng / Học tập
    office_keywords = [
        "văn phòng", "van phong", "học tập", "hoc tap", "word", "excel",
        "kế toán", "ke toan", "lướt web", "luot web", "xem phim",
        "soạn thảo", "cơ bản", "nhẹ nhàng", "day hoc", "dạy học"
    ]
    for kw in office_keywords:
        if kw in t:
            return "Văn phòng / Học tập"

    # Đa dụng / Cân bằng
    balanced_keywords = [
        "đa dụng", "da dung", "cân bằng", "can bang", "toàn diện",
        "tất cả", "tat ca", "tổng hợp", "tong hop", "ưu tiên hiệu năng",
        "u tien hieu nang", "hiệu năng cao", "hieu nang cao", "hiệu năng",
        "hieu nang", "mạnh mẽ", "manh me", "cấu hình mạnh", "cau hinh manh"
    ]
    for kw in balanced_keywords:
        if kw in t:
            return "Tất cả / Đa dụng (Cân bằng)"

    return None


# ===========================================================================
# CƠ SỞ TRI THỨC VỀ GAME & ỨNG DỤNG (3 MỐC NGÂN SÁCH THỰC TẾ)
# ===========================================================================
GAME_APP_PROFILES: Dict[str, Dict[str, Any]] = {
    "gta": {
        "name": "GTA 5",
        "keywords": ["gta", "gta 5", "gta v", "gta5", "gtav", "fivem", "five m", "roleplay"],
        "usage": "Gaming",
        "analysis": "GTA 5 cần CPU khỏe và đủ RAM tải map",
        "tiers": [
            {
                "tier": 1,
                "label": "Phương án 1 (11-13tr)",
                "range_str": "11 – 13 triệu",
                "budget": 12_000_000,
                "desc": "Mượt Full HD High 60-80 FPS."
            },
            {
                "tier": 2,
                "label": "Phương án 2 (20-24tr)",
                "range_str": "20 – 24 triệu",
                "budget": 22_000_000,
                "desc": "Max Setting, mượt mod đông người."
            },
            {
                "tier": 3,
                "label": "Phương án 3 (32-40tr)",
                "range_str": "32 – 40 triệu",
                "budget": 35_000_000,
                "desc": "Đồ họa 2K-4K siêu nét, mod nặng."
            }
        ]
    },
    "watchdogs": {
        "name": "Watch Dogs 2",
        "keywords": ["watch dog", "watch dogs", "watchdog", "watchdogs", "watchdog2", "watchdogs2", "watch dogs 2"],
        "usage": "Gaming",
        "analysis": "Watch Dogs 2 cần CPU đa nhân và card VRAM tốt",
        "tiers": [
            {
                "tier": 1,
                "label": "Phương án 1 (14-16tr)",
                "range_str": "14 – 16 triệu",
                "budget": 15_000_000,
                "desc": "Chiến mượt Full HD mức Medium-High."
            },
            {
                "tier": 2,
                "label": "Phương án 2 (26-30tr)",
                "range_str": "26 – 30 triệu",
                "budget": 28_000_000,
                "desc": "Mượt Max Setting Full HD và 2K."
            },
            {
                "tier": 3,
                "label": "Phương án 3 (42-50tr)",
                "range_str": "42 – 50 triệu",
                "budget": 45_000_000,
                "desc": "Đồ họa 2K-4K, khung hình cực cao."
            }
        ]
    },
    "esports": {
        "name": "Game eSports (Valorant, CS2, LOL, FO4)",
        "keywords": ["valorant", "cs2", "cs:go", "csgo", "lol", "liên minh", "lien minh", "fo4", "fc online", "fifa", "dota"],
        "usage": "Gaming",
        "analysis": "Game eSports ưu tiên CPU đơn nhân để max FPS",
        "tiers": [
            {
                "tier": 1,
                "label": "Phương án 1 (7-9tr)",
                "range_str": "7 – 9 triệu",
                "budget": 8_000_000,
                "desc": "Chiến tốt 100-140 FPS Full HD."
            },
            {
                "tier": 2,
                "label": "Phương án 2 (15-18tr)",
                "range_str": "15 – 18 triệu",
                "budget": 16_000_000,
                "desc": "Đạt 200-300+ FPS màn tần số cao."
            },
            {
                "tier": 3,
                "label": "Phương án 3 (25-30tr)",
                "range_str": "25 – 30 triệu",
                "budget": 27_000_000,
                "desc": "Mượt mọi giải đấu, stream đa nhiệm."
            }
        ]
    },
    "aaa": {
        "name": "Game bom tấn AAA (Wukong, Cyberpunk, PUBG)",
        "keywords": ["wukong", "black myth", "cyberpunk", "pubg", "genshin", "aaa", "game nang", "game nặng"],
        "usage": "Gaming",
        "analysis": "Game AAA bom tấn cần GPU mạnh mẽ VRAM lớn",
        "tiers": [
            {
                "tier": 1,
                "label": "Phương án 1 (18-22tr)",
                "range_str": "18 – 22 triệu",
                "budget": 20_000_000,
                "desc": "Chơi tốt Full HD 60 FPS Medium."
            },
            {
                "tier": 2,
                "label": "Phương án 2 (28-35tr)",
                "range_str": "28 – 35 triệu",
                "budget": 30_000_000,
                "desc": "Mượt mà 2K Ray Tracing và DLSS."
            },
            {
                "tier": 3,
                "label": "Phương án 3 (45-55tr)",
                "range_str": "45 – 55 triệu",
                "budget": 48_000_000,
                "desc": "Chiến đỉnh cao 4K Max Settings."
            }
        ]
    },
    "do_hoa": {
        "name": "Đồ họa, dựng phim & Render",
        "keywords": ["đồ họa", "do hoa", "render", "edit video", "dựng phim", "dung phim", "premiere", "photoshop", "after effects", "3d", "blender"],
        "usage": "Đồ họa / Lập trình AI",
        "analysis": "Đồ họa cần CPU khỏe và GPU tăng tốc render",
        "tiers": [
            {
                "tier": 1,
                "label": "Phương án 1 (14-17tr)",
                "range_str": "14 – 17 triệu",
                "budget": 15_000_000,
                "desc": "Thiết kế 2D, dựng video Full HD."
            },
            {
                "tier": 2,
                "label": "Phương án 2 (25-30tr)",
                "range_str": "25 – 30 triệu",
                "budget": 27_000_000,
                "desc": "Dựng video 4K và render 3D nhanh."
            },
            {
                "tier": 3,
                "label": "Phương án 3 (42-52tr)",
                "range_str": "42 – 52 triệu",
                "budget": 46_000_000,
                "desc": "Xử lý mượt 3D và dựng phim."
            }
        ]
    },
    "lap_trinh": {
        "name": "Lập trình & Trí tuệ Nhân tạo (AI)",
        "keywords": ["lập trình", "lap trinh", "code", "ai", "machine learning", "deep learning", "docker"],
        "usage": "Đồ họa / Lập trình AI",
        "analysis": "Lập trình và AI cần CPU đa luồng, RAM lớn",
        "tiers": [
            {
                "tier": 1,
                "label": "Phương án 1 (12-14tr)",
                "range_str": "12 – 14 triệu",
                "budget": 13_000_000,
                "desc": "Lập trình web, app, database mượt."
            },
            {
                "tier": 2,
                "label": "Phương án 2 (24-28tr)",
                "range_str": "24 – 28 triệu",
                "budget": 26_000_000,
                "desc": "Chạy Docker, máy ảo và model AI."
            },
            {
                "tier": 3,
                "label": "Phương án 3 (42-52tr)",
                "range_str": "42 – 52 triệu",
                "budget": 46_000_000,
                "desc": "Huấn luyện AI chuyên sâu mượt mà."
            }
        ]
    },
    "van_phong": {
        "name": "Văn phòng & Học tập",
        "keywords": ["văn phòng", "van phong", "học tập", "hoc tap", "word", "excel", "kế toán", "ke toan", "lướt web"],
        "usage": "Văn phòng / Học tập",
        "analysis": "Văn phòng cần chạy bền bỉ và ổn định",
        "tiers": [
            {
                "tier": 1,
                "label": "Phương án 1 (6-8tr)",
                "range_str": "6 – 8 triệu",
                "budget": 7_000_000,
                "desc": "Mượt Word, Excel và lướt web."
            },
            {
                "tier": 2,
                "label": "Phương án 2 (14-17tr)",
                "range_str": "14 – 17 triệu",
                "budget": 15_000_000,
                "desc": "Mở nhiều tab, file Excel lớn."
            },
            {
                "tier": 3,
                "label": "Phương án 3 (24-28tr)",
                "range_str": "24 – 28 triệu",
                "budget": 25_000_000,
                "desc": "Tốc độ cực nhanh, siêu êm ái."
            }
        ]
    }
}


def detect_game_or_app(text: str) -> Optional[Dict[str, Any]]:
    """Nhận diện tên game hoặc phần mềm người dùng đề cập (cả danh mục có sẵn và trích xuất động)."""
    t = text.lower()
    for prof_key, prof in GAME_APP_PROFILES.items():
        for kw in prof["keywords"]:
            if len(kw) <= 4:
                if re.search(r'\b' + re.escape(kw) + r'\b', t):
                    return prof
            else:
                if kw in t:
                    return prof

    # Nhận diện động tên game sau các từ khóa hành động: chơi, chiến, game, play
    m_game = re.search(r'\b(?:chơi|choi|chiến|chien|game|play)\s+([a-z0-9\s\.\-_:]+)', t)
    if m_game:
        raw_name = m_game.group(1).strip()
        # Loại bỏ các từ đệm ở cuối như: đi, nhé, nha, mượt, ngon, tốt, được, ko, không, ạ, a...
        clean_name = re.sub(r'\s+(?:đi|di|nhé|nhe|nha|mượt|muot|ngon|tốt|tot|được|duoc|ko|không|khong|ạ|a)$', '', raw_name).strip()
        if clean_name and len(clean_name) >= 2:
            display_name = clean_name.title()
            return {
                "name": f"Game {display_name}",
                "keywords": [clean_name],
                "usage": "Gaming",
                "analysis": f"{display_name} cần CPU đa nhân và card đồ họa rời để duy trì khung hình ổn định.",
                "tiers": [
                    {
                        "tier": 1,
                        "label": "Phương án 1 (14 – 16 triệu)",
                        "range_str": "14 – 16 triệu",
                        "budget": 15_000_000,
                        "desc": "Chiến mượt Full HD mức thiết lập trung bình - cao 60 FPS."
                    },
                    {
                        "tier": 2,
                        "label": "Phương án 2 (26 – 30 triệu)",
                        "range_str": "26 – 30 triệu",
                        "budget": 28_000_000,
                        "desc": "Trải nghiệm mượt mà Max Setting Full HD và 2K."
                    },
                    {
                        "tier": 3,
                        "label": "Phương án 3 (42 – 50 triệu)",
                        "range_str": "42 – 50 triệu",
                        "budget": 45_000_000,
                        "desc": "Chiến đỉnh cao 4K Max Settings mọi tựa game bom tấn."
                    }
                ]
            }

    return None


def extract_tier_selection(text: str) -> Optional[int]:
    """Trích xuất lựa chọn phương án/mức ngân sách khi người dùng chọn theo gợi ý của AI (ví dụ: phương án 2, pa 2, mức 2, làm phương án 1...)."""
    t = text.lower().strip()
    # Mẫu 1: phương án 1, pa 2, mức 1, tầm 2, mốc 3, bộ 2, option 2...
    m = re.search(r'(?:phương án|phuong an|pa|mức|muc|tầm|tam|mốc|moc|bộ|bo|option|opt)\s*([123])\b', t)
    if m:
        return int(m.group(1))

    # Mẫu 2: chữ số tiếng Việt: phương án một, phương án hai, mức một, tầm hai...
    word_map = {'một': 1, 'mot': 1, 'hai': 2, 'ba': 3}
    m_w = re.search(r'(?:phương án|phuong an|pa|mức|muc|tầm|tam|mốc|moc)\s*(một|mot|hai|ba)\b', t)
    if m_w:
        return word_map[m_w.group(1)]

    # Mẫu 3: chọn 2, làm 2, lấy 2, chốt 2, số 2...
    m_act = re.search(r'\b(?:chọn|chon|làm|lam|lấy|lay|chốt|chot|số|so)\s*([123])\b', t)
    if m_act:
        return int(m_act.group(1))

    # Mẫu 4: chỉ có số 1, 2, 3 kèm hậu tố: 2 đi, 2 nhé, 2 nha, 2 thôi...
    m_end = re.search(r'^\s*([123])\s*(?:nhé|nha|đi|di|thôi|luôn)?\s*$', t)
    if m_end:
        return int(m_end.group(1))

    return None


def resolve_budget_from_tier(
    tier_idx: int,
    conversation_history: Optional[List[Dict[str, str]]] = None,
    data: Optional[Dict] = None,
) -> Optional[int]:
    """
    Xác định số tiền ngân sách tương ứng với phương án lựa chọn (Phương án 1, 2, 3)
    dựa vào tin nhắn gợi ý trước đó trong lịch sử hoặc bảng hồ sơ game.
    """
    if conversation_history:
        # 1. Quét ngược trong tin nhắn của trợ lý AI để tìm khoảng giá của phương án đó
        for msg in reversed(conversation_history):
            if msg.get("role") != "assistant":
                continue
            content = msg.get("content", "")
            pat = r'(?:Phương án|PA|Mức|Mốc|Tầm)\s*' + str(tier_idx) + r'[^\n\d]+(\d+)[\s–\-]+(\d+)\s*(?:triệu|tr\b)'
            m = re.search(pat, content, re.IGNORECASE)
            if m:
                low = float(m.group(1))
                high = float(m.group(2))
                avg_tr = (low + high) / 2
                return int(avg_tr * 1_000_000)

        # 2. Quét tin nhắn của user để xem game profile nào được nhắc đến
        for msg in reversed(conversation_history):
            if msg.get("role") != "user":
                continue
            u_txt = msg.get("content", "")
            prof = detect_game_or_app(u_txt)
            if prof and "tiers" in prof:
                for t in prof["tiers"]:
                    if t.get("tier") == tier_idx:
                        return t.get("budget")

    # 3. Mặc định theo Gaming thông thường nếu không tìm thấy
    default_gaming_tiers = {1: 11_000_000, 2: 15_000_000, 3: 22_500_000}
    return default_gaming_tiers.get(tier_idx)


def normalize_text_for_matching(s: str) -> str:
    """Loại bỏ khoảng trắng, dấu gạch nối và ký tự đặc biệt, chuyển về chữ thường."""
    s = s.lower()
    s = re.sub(r'\br([3579])\b', r'ryzen\1', s)
    s = re.sub(r'\br([3579])(\d{4})', r'ryzen\1\2', s)
    return re.sub(r'[^a-z0-9]', '', s)


def match_specific_components(text: str, data: Dict[str, List[Dict]]) -> Tuple[Dict[str, Dict], List[str]]:
    """
    Nhận diện linh kiện cụ thể (CPU, GPU) được người dùng nhắc đến để khóa vào cấu hình.
    Hỗ trợ cả tên dính liền (rtx5060, rtx4060, i513600k, rx7600, r55600x).
    Nếu người dùng chỉ định rõ model mà dữ liệu không có, sinh cảnh báo chuyên môn.
    """
    locked: Dict[str, Dict] = {}
    warnings: List[str] = []

    # 1. Nhận diện GPU qua Regex bắt model cụ thể
    gpu_matches = re.findall(
        r'\b(rtx\s*\d{3,4}(?:\s*ti|\s*super)?|gtx\s*\d{3,4}(?:\s*ti|\s*super)?|rx\s*\d{3,4}(?:\s*xt)?|rtx\d{3,4}(?:ti|super)?|gtx\d{3,4}(?:ti|super)?|rx\d{3,4}(?:xt)?)\b',
        text,
        re.IGNORECASE,
    )
    if gpu_matches:
        for gm in gpu_matches:
            norm_gm = normalize_text_for_matching(gm)
            found = next((g for g in data.get("gpus", []) if norm_gm in normalize_text_for_matching(g["name"])), None)
            if found:
                locked["gpu"] = found
                break
            else:
                warnings.append(
                    f"Lưu ý: Dữ liệu hiện tại chưa có {gm.strip().upper()}, "
                    f"hệ thống tự động đề xuất GPU tương đương tốt nhất trong ngân sách."
                )

    # Nếu regex chưa bắt được GPU, quét thêm danh sách gpus theo từ khóa
    if "gpu" not in locked:
        t_clean = ' ' + text.lower().replace('-', ' ') + ' '
        for gpu in data.get("gpus", []):
            if gpu["id"] == "gpu_none":
                continue
            g_name = gpu["name"].lower().replace("-", " ")
            parts = [p for p in g_name.split() if p in ["rtx", "gtx", "rx", "radeon", "geforce"]]
            for p in parts:
                idx = g_name.find(p)
                candidate_str = g_name[idx:]
                if re.search(r'\b' + re.escape(candidate_str) + r'\b', t_clean):
                    locked["gpu"] = gpu
                    break
            if "gpu" in locked:
                break

    # 2. Nhận diện CPU qua Regex bắt model cụ thể
    cpu_matches = re.findall(
        r'\b(i[3579]\s*[-]?\s*\d{4,5}[a-z]*|i[3579]\d{4,5}[a-z]*|ryzen\s*[3579]\s*[-]?\s*\d{4}[a-z0-9]*|r[3579]\s*[-]?\s*\d{4}[a-z0-9]*)\b',
        text,
        re.IGNORECASE,
    )
    if cpu_matches:
        for cm in cpu_matches:
            norm_cm = normalize_text_for_matching(cm)
            found = next((c for c in data.get("cpus", []) if norm_cm in normalize_text_for_matching(c["name"])), None)
            if found:
                locked["cpu"] = found
                break
            else:
                warnings.append(
                    f"Lưu ý: Dữ liệu hiện tại chưa có {cm.strip().upper()}, "
                    f"hệ thống tự động đề xuất CPU tương đương tốt nhất trong ngân sách."
                )

    # 2b. Nhận diện CPU theo thế hệ / đời: ví dụ "i5 đời 10", "i5 gen 10", "i3 đời 10", "i7 đời 10"
    if "cpu" not in locked:
        gen_match = re.search(r'\b(i[3579])\s*(?:đời|doi|gen|thế hệ|the he)\s*(\d{1,2})\b', text.lower())
        if gen_match:
            tier = gen_match.group(1).lower()
            gen_num = gen_match.group(2)
            found = next(
                (c for c in data.get("cpus", [])
                 if tier in c["name"].lower() and (f"-{gen_num}" in c["name"] or f" {gen_num}" in c["name"])),
                None
            )
            if found:
                locked["cpu"] = found

    # Nếu regex chưa bắt được CPU, quét thêm danh sách cpus theo mã số
    if "cpu" not in locked:
        t_clean = ' ' + text.lower().replace('-', ' ') + ' '
        for cpu in data.get("cpus", []):
            c_name = cpu["name"].lower().replace("-", " ")
            parts = c_name.split()
            model_code = parts[-1]
            if len(model_code) >= 4 and any(ch.isdigit() for ch in model_code):
                if re.search(r'\b' + re.escape(model_code) + r'\b', t_clean):
                    locked["cpu"] = cpu
                    break

    return locked, warnings


def is_upgrade_acceptance(text: str) -> bool:
    """Kiểm tra người dùng có đang đồng ý / chấp nhận phương án nâng cấp linh kiện hay không."""
    t = text.lower().strip()
    clean_t = re.sub(r'[!?,.]+$', '', t).strip()

    # Các cụm từ đồng ý nâng cấp / thêm tiền rõ ràng
    specific_patterns = [
        r'\b(?:ok|oke|oki|okay|được|duoc|chốt|chot|đồng ý|dong y)\s*(?:thêm|nâng|nang|lên|len)\b',
        r'\b(?:thêm|them)\s*(?:đi|di|luôn|luon|nhé|nhe|nha|ạ|a)\b',
        r'^\s*thêm\b', r'\bthêm\s*$',
        r'\b(?:bù thêm|bu them|thêm tiền|them tien)\b',
        r'\b(?:nâng luôn|nang luon|lên luôn|len luon|nâng đi|nang di|lên đi|len di|nâng cấp đi|nang cap di)\b',
        r'\b(?:đồng ý|dong y|đồng ý nâng|đồng ý nâng cấp|dong y nang cap)\b',
        r'\b(?:lấy con đó|lay con do|lên con đó|len con do|đổi sang con đó|doi sang con do)\b',
        r'\b(?:triển luôn|trien luon|nhất trí|nhat tri|duyệt|duyet)\b',
    ]
    if any(re.search(pat, t) for pat in specific_patterns):
        return True

    # Các từ đồng ý ngắn nếu câu ngắn <= 3 từ: "ok", "oke", "được", "chốt", "oke bạn"
    short_agreements = {
        "ok", "oke", "oki", "okay", "được", "duoc", "chốt", "chot", "đồng ý", "dong y",
        "oke thêm", "ok thêm", "oke them", "ok them", "thêm", "them", "thêm luôn",
        "oke bạn", "ok bạn", "oke shop", "ok shop", "được bạn", "được nhé", "ok nhé", "oke nhé",
        "chốt nhé", "chốt luôn", "chốt đơn", "chot don", "lấy luôn", "lay luon"
    }
    if clean_t in short_agreements:
        return True

    return False


def is_upgrade_inquiry_after_acceptance(text: str) -> bool:
    """Kiểm tra người dùng có đang thắc mắc tại sao đã đồng ý nâng cấp mà linh kiện chưa đổi."""
    t = text.lower().strip()
    inquiry_patterns = [
        r'\b(?:đồng ý|dong y)\s*(?:r|rồi|roi)\s*mà\s*(?:nó\s*)?(?:ko|không|chưa)\s*đổi\b',
        r'\b(?:sao|sao lại)\s*(?:chưa|không|ko)\s*(?:thấy\s*)?(?:đổi|cập nhật|thay)\b',
        r'\b(?:sao|sao lại)\s*(?:vẫn là|vẫn dùng|vẫn để)\b',
        r'\b(?:chưa|ko|không)\s*(?:đổi|thay|cập nhật)\s*(?:à|hả|sao)\b',
        r'\bđổi\s*(?:cho tôi|giúp tôi|hộ tôi|đi|luôn)\b',
    ]
    return any(re.search(pat, t) for pat in inquiry_patterns)


def suggest_upgrades(
    selected_build: Any,
    catalog: Dict[str, List[Dict]],
    budget: int = 0,
    usage: str = "Gaming",
) -> List[Dict[str, Any]]:
    """
    Tự động tìm 1 đến 2 phương án nâng cấp đáng giá nhất có tương thích hoàn toàn
    với dàn máy hiện tại và chi phí chênh lệch hợp lý (bù thêm từ 100.000 VNĐ đến 3.500.000 VNĐ).
    Khớp chính xác với ngữ cảnh nhu cầu sử dụng (Văn phòng, Gaming, Đồ họa/AI).
    """
    if not selected_build:
        return []

    def get_part(cat: str) -> Dict:
        if hasattr(selected_build, cat):
            p = getattr(selected_build, cat)
            return p if isinstance(p, dict) else {}
        elif isinstance(selected_build, dict):
            p = selected_build.get(cat, {})
            return p if isinstance(p, dict) else {}
        return {}

    cpu = get_part("cpu")
    mb = get_part("mainboard")
    ram = get_part("ram")
    gpu = get_part("gpu")
    psu = get_part("psu")
    case = get_part("case")
    storage = get_part("storage")

    if not cpu or not mb:
        return []

    is_office = (usage == "Văn phòng / Học tập" or "văn phòng" in usage.lower() or "học tập" in usage.lower())
    is_creative = ("đồ họa" in usage.lower() or "ai" in usage.lower() or "lập trình" in usage.lower())

    upgrades: List[Dict[str, Any]] = []

    # 1.1 Kiểm tra nâng RAM từ 16GB lên 32GB Dual Channel
    ram_cap = ram.get("capacity_gb", 0)
    ram_price = ram.get("price", 0)
    if ram_cap <= 16 and ram_price > 0:
        cands_ram = [
            r for r in catalog.get("rams", [])
            if r.get("ram_type") == ram.get("ram_type")
            and r.get("capacity_gb", 0) == 32
            and (r.get("is_kit", False) or r.get("sticks", 1) >= 2 or "2x" in r.get("name", "").lower())
            and r.get("capacity_gb", 0) <= mb.get("max_ram_gb", 128)
            and r.get("price", 0) > ram_price
        ]
        if cands_ram:
            best_ram = min(cands_ram, key=lambda r: r.get("price", 0))
            diff = best_ram["price"] - ram_price
            if 500_000 <= diff <= 3_500_000:
                if is_office:
                    benefit = "mở nhiều tab trình duyệt, file Excel nặng và đa nhiệm mượt mà hơn"
                elif is_creative:
                    benefit = "tối ưu đa nhiệm mượt mà cho các project đồ họa và tác vụ nặng"
                else:
                    benefit = "tối ưu đa nhiệm mượt mà và không lo tràn RAM khi mở nhiều tác vụ hay chơi game nặng"
                upgrades.append({
                    "category": "ram",
                    "target_id": best_ram["id"],
                    "current_name": ram.get("name", "RAM 16GB"),
                    "upgrade_name": best_ram["name"],
                    "added_cost": diff,
                    "benefit": benefit,
                })

    # 1.2 Kiểm tra nâng SSD từ chuẩn SATA lên chuẩn NVMe tốc độ cao
    storage_price = storage.get("price", 0)
    storage_name = storage.get("name", "").upper()
    is_storage_sata = "SATA" in storage_name or "SATA" in str(storage.get("interface", "")).upper()
    if is_storage_sata and storage_price > 0:
        cands_nvme = [
            s for s in catalog.get("storages", [])
            if ("NVME" in s.get("name", "").upper() or "NVME" in str(s.get("interface", "")).upper())
            and s.get("capacity_gb", 0) >= storage.get("capacity_gb", 0)
            and s.get("price", 0) > storage_price
        ]
        if cands_nvme:
            cands_nvme.sort(key=lambda s: (s.get("price", 0) - storage_price))
            best_nvme = cands_nvme[0]
            diff = best_nvme["price"] - storage_price
            if 100_000 <= diff <= 2_500_000:
                if is_office:
                    benefit = "nâng cấp lên chuẩn SSD NVMe tốc độ cao hơn, giúp sao chép tài liệu lớn và mở ứng dụng tức thì"
                else:
                    benefit = "nâng cấp lên chuẩn SSD NVMe tốc độ cao hơn, giúp load game và khởi động project nhanh hơn rõ rệt"
                upgrades.append({
                    "category": "storage",
                    "target_id": best_nvme["id"],
                    "current_name": storage.get("name", "SSD SATA"),
                    "upgrade_name": best_nvme["name"],
                    "added_cost": diff,
                    "benefit": benefit,
                })

    # 1.3 Kiểm tra nâng Mainboard nếu đang dùng bo mạch chủ PCIe 3.0 hoặc H610
    mb_price = mb.get("price", 0)
    if (is_pcie3_mainboard(mb) or "H610" in mb.get("name", "").upper()) and mb_price > 0:
        cands_mb = [
            m for m in catalog.get("mainboards", [])
            if m.get("socket") == cpu.get("socket")
            and m.get("ram_type") == cpu.get("ram_type")
            and not is_pcie3_mainboard(m)
            and not is_vrm_incompatible(cpu, m)
            and (not case or m.get("form_factor") in case.get("supported_form_factors", ["ATX", "mATX", "ITX"]))
            and m.get("price", 0) > mb_price
        ]
        if cands_mb:
            best_mb = min(cands_mb, key=lambda m: m.get("price", 0))
            diff = best_mb["price"] - mb_price
            if 500_000 <= diff <= 3_500_000:
                if is_office:
                    benefit = "nâng lên bo mạch chủ cứng cáp hơn với dàn pha điện VRM ổn định, mở rộng khả năng tản nhiệt và vận hành bền bỉ lâu dài"
                else:
                    benefit = "nâng cấp lên nền tảng hỗ trợ PCIe 4.0 và dàn pha điện VRM cứng cáp hơn, giải phóng 100% băng thông VGA và SSD NVMe"
                upgrades.append({
                    "category": "mainboard",
                    "target_id": best_mb["id"],
                    "current_name": mb.get("name", "Bo mạch chủ hiện tại"),
                    "upgrade_name": best_mb["name"],
                    "added_cost": diff,
                    "benefit": benefit,
                })

    # 2. Kiểm tra nâng CPU cùng Socket
    cpu_price = cpu.get("price", 0)
    cpu_score = cpu.get("score", 0)
    cands_cpu = [
        c for c in catalog.get("cpus", [])
        if c.get("socket") == cpu.get("socket")
        and c.get("ram_type") == cpu.get("ram_type")
        and c.get("score", 0) > cpu_score
        and not is_vrm_incompatible(c, mb)
        and c.get("price", 0) > cpu_price
    ]
    if cands_cpu:
        cands_cpu.sort(key=lambda c: (c.get("price", 0) - cpu_price))
        for c in cands_cpu:
            diff = c["price"] - cpu_price
            if 500_000 <= diff <= 3_500_000:
                c_u = c.get("name", "").upper()
                if is_office:
                    benefit = "tăng mạnh hiệu năng xử lý đa nhân và đơn nhân, xử lý dữ liệu và bảng tính lớn mượt mà, phản hồi tức thì"
                elif "X3D" in c_u:
                    benefit = "trang bị công nghệ bộ nhớ đệm 3D V-Cache đỉnh cao, tăng mạnh FPS và giữ khung hình cực kỳ ổn định trong game"
                elif any(k in c_u for k in ["13600K", "14600K", "13700K"]):
                    if is_creative:
                        benefit = "bổ sung nhiều nhân E-Core thế hệ mới, tối ưu vượt trội cho tác vụ render đồ họa và đa nhiệm nặng"
                    else:
                        benefit = "bổ sung nhiều nhân E-Core thế hệ mới, tối ưu vượt trội cho các tác vụ tính toán và đa nhiệm nặng"
                else:
                    if is_creative:
                        benefit = "tăng mạnh hiệu năng xử lý đa nhân, giúp render video nhanh hơn và biên dịch code mượt mà"
                    else:
                        benefit = "tăng mạnh hiệu năng xử lý đa nhân và đơn nhân, giảm thiểu nghẽn cổ chai cho card đồ họa khi chơi game"
                upgrades.append({
                    "category": "cpu",
                    "target_id": c["id"],
                    "current_name": cpu.get("name", "CPU hiện tại"),
                    "upgrade_name": c["name"],
                    "added_cost": diff,
                    "benefit": benefit,
                })
                break

    # 3. Kiểm tra nâng GPU (chỉ nâng GPU nếu không phải Văn phòng / Học tập hoặc máy có GPU rời)
    gpu_price = gpu.get("price", 0)
    gpu_score = gpu.get("score", 0)
    if not is_office and gpu.get("id") != "gpu_none" and gpu_price > 0:
        cands_gpu = [
            g for g in catalog.get("gpus", [])
            if g.get("id") != "gpu_none"
            and g.get("score", 0) > gpu_score
            and (not case or g.get("length_mm", 0) <= case.get("max_gpu_length_mm", 400))
            and g.get("price", 0) > gpu_price
        ]
        if cands_gpu:
            cands_gpu.sort(key=lambda g: (g.get("price", 0) - gpu_price))
            for g in cands_gpu:
                diff = g["price"] - gpu_price
                if 500_000 <= diff <= 3_500_000:
                    min_psu = calculate_min_psu_wattage(cpu, g)
                    if psu.get("wattage", 0) >= min_psu:
                        if is_creative:
                            benefit = "tăng vượt bậc sức mạnh đồ họa với dung lượng VRAM lớn, tăng tốc render video và huấn luyện mô hình AI"
                        else:
                            benefit = "tăng vượt bậc sức mạnh xử lý đồ họa, nâng cao chất lượng hình ảnh và FPS khi chơi game độ phân giải cao"
                        upgrades.append({
                            "category": "gpu",
                            "target_id": g["id"],
                            "current_name": gpu.get("name", "VGA hiện tại"),
                            "upgrade_name": g["name"],
                            "added_cost": diff,
                            "benefit": benefit,
                        })
                        break

    # 4. Kiểm tra nâng PSU
    psu_price = psu.get("price", 0)
    psu_eff = psu.get("efficiency", "")
    if ("Gold" not in psu_eff or psu.get("wattage", 0) < 650) and psu_price > 0:
        cands_psu = [
            p for p in catalog.get("psus", [])
            if p.get("wattage", 0) >= max(psu.get("wattage", 0), 650)
            and ("Gold" in p.get("efficiency", "") or p.get("wattage", 0) > psu.get("wattage", 0))
            and p.get("price", 0) > psu_price
        ]
        if cands_psu:
            best_psu = min(cands_psu, key=lambda p: p.get("price", 0))
            diff = best_psu["price"] - psu_price
            if 500_000 <= diff <= 1_500_000:
                upgrades.append({
                    "category": "psu",
                    "target_id": best_psu["id"],
                    "current_name": psu.get("name", "Nguồn hiện tại"),
                    "upgrade_name": best_psu["name"],
                    "added_cost": diff,
                    "benefit": "nâng lên chuẩn nguồn 80 Plus Gold cao cấp với công suất dư dả, giúp dàn máy vận hành êm ái, tiết kiệm điện và bền bỉ tối đa",
                })

    return upgrades[:2]



def extract_upgrade_from_history(conversation_history: List[Dict[str, str]], catalog: Dict[str, List[Dict]]) -> Optional[Dict[str, Any]]:
    """Tìm phương án nâng cấp gần nhất được Trợ lý AI đề xuất trong lịch sử hội thoại."""
    if not conversation_history:
        return None

    cat_map = {
        "cpus": "cpu", "mainboards": "mainboard", "rams": "ram",
        "gpus": "gpu", "psus": "psu", "cases": "case",
        "storages": "storage", "coolers": "cooler", "hdds": "hdd"
    }

    # Duyệt ngược từ tin nhắn gần nhất của assistant
    for msg in reversed(conversation_history):
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content", "")
        m = re.search(r'nâng từ\s+(.+?)\s+lên\s+(.+?)\s+sẽ giúp', content, re.IGNORECASE)
        curr_name = ""
        if m:
            curr_name = m.group(1).replace("**", "").strip()
            target_name = m.group(2).replace("**", "").strip()
        else:
            m = re.search(r'nâng lên\s+(.+?)\s+sẽ giúp', content, re.IGNORECASE)
            if m:
                target_name = m.group(1).replace("**", "").strip()
            else:
                m = re.search(r'nâng từ\s+(.+?)\s+lên\s+([A-Za-z0-9\s\.\-\/\(\)]+?)(?:[\.,\n]|$)', content, re.IGNORECASE)
                if m:
                    curr_name = m.group(1).replace("**", "").strip()
                    target_name = m.group(2).replace("**", "").strip()
                else:
                    m = re.search(r'nâng lên\s+([A-Za-z0-9\s\.\-\/\(\)]+?)(?:[\.,\n]|$)', content, re.IGNORECASE)
                    if m:
                        target_name = m.group(1).replace("**", "").strip()

        if m:

            cost_m = re.search(r'thêm(?:\s+khoảng)?\s*([\d\.\,]+)\s*(?:triệu|trđ|vnd|vnđ|tr\b|củ\b)', content, re.IGNORECASE)
            added_cost = 0
            if cost_m:
                val = float(cost_m.group(1).replace('.', '').replace(',', '.'))
                added_cost = int(val * 1_000_000) if val < 1000 else int(val)

            # Tìm trong catalog
            for cat_key, items in catalog.items():
                for it in items:
                    if it.get("name", "").lower() == target_name.lower() or target_name.lower() in it.get("name", "").lower():
                        return {
                            "category": cat_map.get(cat_key, "cpu"),
                            "target_id": it["id"],
                            "current_name": curr_name,
                            "upgrade_name": it["name"],
                            "added_cost": added_cost or it.get("price", 0),
                            "benefit": "Nâng cấp tối ưu sức mạnh và độ ổn định",
                        }
    return None


def classify_intent(text: str, parsed: Dict[str, Any]) -> str:
    """
    Phân loại ý định người dùng thành 'build' (xây dựng/chỉnh sửa cấu hình PC),
    'upgrade' (đồng ý nâng cấp linh kiện được gợi ý),
    hoặc 'chat' (chào hỏi, hỏi đáp kiến thức, trò chuyện thông thường).

    Quy tắc bắt buộc:
    - Nếu người dùng đồng ý nâng cấp ('oke thêm', 'đồng ý', 'chốt nâng...') hoặc hỏi tại sao chưa đổi: trả về 'upgrade'.
    - Chỉ trả về 'build' khi tin nhắn có ý định rõ ràng về xây dựng hoặc chỉnh sửa
      cấu hình PC (nhắc đến ngân sách/tiền, tên linh kiện, hãng CPU/VGA, hoặc nhu cầu
      dùng máy kèm yêu cầu tư vấn/lên cấu hình).
    - Với mọi tin nhắn chào hỏi, bắt chuyện, hoặc trò chuyện bình thường: BẮT BUỘC trả về 'chat'.
    """
    t = text.lower().strip()
    clean_t = re.sub(r'[!?,.]+$', '', t).strip()

    # 0. Ý định đồng ý nâng cấp hoặc thắc mắc đã đồng ý nâng cấp -> 'upgrade'
    if is_upgrade_acceptance(text) or is_upgrade_inquiry_after_acceptance(text):
        return "upgrade"

    # 1. Các câu chào hỏi / bắt chuyện thuần túy -> LUÔN LÀ 'chat'
    pure_chat_greetings = {
        "chào", "chao", "hi", "hello", "xin chào", "xin chao", "alo", "hé lô",
        "helo", "hai", "chào bạn", "chao ban", "chào shop", "chao shop",
        "bạn là ai", "ban la ai", "bạn tên gì", "ban ten gi", "bạn khỏe không",
        "ban khoe khong", "bot à", "bot a", "ai đấy", "ai day", "ê", "e", "hey"
    }
    if clean_t in pure_chat_greetings:
        return "chat"

    # 2. Câu hỏi so sánh hoặc hỏi lý thuyết kiến thức chung mà không kèm ngân sách -> 'chat'
    comparison_patterns = [
        r'nên chọn\s+.*\s+hay\b',
        r'nên mua\s+.*\s+hay\b',
        r'nên dùng\s+.*\s+hay\b',
        r'\b(?:intel|inter|amd|amđ)\s+hay\s+(?:intel|inter|amd|amđ)\b',
        r'\b(?:nvidia|nvida|amd|amđ)\s+hay\s+(?:nvidia|nvida|amd|amđ)\b',
        r'\b(?:ddr4|ddr5)\s+hay\s+(?:ddr4|ddr5)\b',
        r'\bso sánh\b',
        r'\bkhác gì\b',
        r'\btốt hơn\b',
        r'\bmạnh hơn\b',
        r'\bcó nên mua\b',
        r'\bcó nên dùng\b',
        r'\bcó bền không\b',
        r'\bcó nóng không\b',
        r'\bcó mạnh không\b',
        r'\bchơi được không\b',
    ]
    if any(re.search(pat, t) for pat in comparison_patterns):
        if not parsed.get("budget"):
            return "chat"

    # 3. Yêu cầu tư vấn chung chưa nói rõ ngân sách hay cấu hình cụ thể -> 'chat'
    general_requests = {
        "tư vấn giúp mình", "tu van giup minh", "tư vấn cho mình", "tu van cho minh",
        "mình cần tư vấn", "minh can tu van", "cần tư vấn", "can tu van",
        "tư vấn pc", "tu van pc", "giúp mình với", "giup minh voi", "hỗ trợ mình", "ho tro minh",
        "tư vấn giúp", "tu van giup", "tư vấn nhé", "tu van nhe"
    }
    if clean_t in general_requests:
        return "chat"

    # 4. Ý định BUILD rõ ràng:
    # 4.0 Người dùng đã chọn mốc ngân sách được tư vấn từ trước (Mức 1, Mức 2, Mức 3...)
    if parsed.get("selected_tier_idx"):
        return "build"

    # 4.1 Có ngân sách/tiền cụ thể (VD: 20tr, 15 triệu, 30 củ...)
    if parsed.get("budget") is not None:
        return "build"

    # 4.2 Có tên model linh kiện cụ thể được nhận diện (i5-13600k, rtx 4060, b660...)
    if parsed.get("locked_parts"):
        return "build"

    # 4.3 Trạng thái TƯ VẤN CHƯA CÓ NGÂN SÁCH (consult_no_budget):
    # Người dùng hỏi theo tên Game, Phần mềm hoặc Nhu cầu nhưng chưa đưa ra số tiền cụ thể
    if parsed.get("budget") is None and not parsed.get("locked_parts"):
        if parsed.get("app_game_profile") or (parsed.get("usage") and parsed["usage"] != "Tất cả / Đa dụng (Cân bằng)"):
            return "consult_no_budget"

        request_pc_patterns = [
            r'\b(?:cho|làm|lấy|cần|muốn|xin|tư vấn|build|dựng|lắp)\s+(?:tôi\s+|mình\s+|em\s+)?(?:1\s+|một\s+)?(?:bộ|dàn|cấu hình|máy|pc)\b',
            r'\b(?:bộ|dàn)\s+(?:pc|máy)\b',
            r'\b(?:pc|máy|dàn|cấu hình)\s+(?:để\s+)?(?:chơi|choi|chiến|làm|học|render|edit|code)\b',
            r'\b(?:chơi|choi|chiến)\s+[a-z0-9_]+\b'
        ]
        if any(re.search(pat, t) for pat in request_pc_patterns):
            return "consult_no_budget"

    # 4.4 Có từ khóa chỉnh sửa / nâng cấp cấu hình hiện tại
    modify_keywords = [
        "nâng lên", "nang len", "tăng lên", "tang len", "hạ xuống", "ha xuong",
        "giảm xuống", "giam xuong", "thay vga", "thay card", "thay cpu", "thay chip",
        "thay ram", "thay ssd", "thay hdd", "thay nguồn", "thay nguon", "thay main",
        "đổi vga", "doi vga", "đổi card", "doi card", "đổi cpu", "doi cpu",
        "đổi sang", "doi sang", "thêm hdd", "them hdd", "bỏ hdd", "bo hdd",
        "thêm ssd", "them ssd", "nâng cấp", "nang cap"
    ]
    if any(k in t for k in modify_keywords):
        return "build"

    # 4.5 Có từ khóa dựng máy kèm theo nhu cầu hoặc hãng linh kiện
    build_keywords = [
        "build", "dựng pc", "dung pc", "lên cấu hình", "len cau hinh",
        "lên dàn", "len dan", "lắp máy", "lap may", "lắp pc", "lap pc",
        "mua pc", "mua máy", "mua may", "cấu hình", "cau hinh", "dàn máy",
        "dan may", "dàn pc", "dan pc", "làm một bộ", "lam mot bo", "làm 1 dàn", "lam 1 dan",
        "cần 1 bộ", "can 1 bo", "cần bộ", "can bo", "mua 1 bộ", "mua 1 bo",
        "cho tôi bộ pc", "cho minh bo pc", "cho mình bộ pc", "cho tôi bộ", "cho mình bộ",
        "cho xin bộ", "cho 1 bộ", "cho mot bo", "bộ pc", "bo pc", "bộ máy", "bo may",
        "làm bộ", "lam bo", "lấy bộ", "lay bo"
    ]
    has_build_kw = any(k in t for k in build_keywords)
    if has_build_kw:
        # Nếu có từ khóa build hoặc nhu cầu nhưng KHÔNG có ngân sách cụ thể -> Chuyển sang tư vấn các mốc giá
        if parsed.get("budget") is None and not parsed.get("locked_parts"):
            return "consult_no_budget"
        if parsed.get("usage") and parsed["usage"] != "Tất cả / Đa dụng (Cân bằng)":
            return "build"
        if parsed.get("cpu_brand") or parsed.get("gpu_brand") or parsed.get("target_ram_gb"):
            return "build"
        if parsed.get("need_vga") is not None or parsed.get("need_hdd") is not None:
            return "build"

    # Nếu câu chat có nhắc đến nhu cầu sử dụng nhưng chưa có ngân sách -> consult_no_budget
    if parsed.get("usage") and parsed["usage"] != "Tất cả / Đa dụng (Cân bằng)" and parsed.get("budget") is None:
        return "consult_no_budget"

    # 4.6 Người dùng nói rõ cấu hình mong muốn (RAM >= 16/32GB, có card rời...)
    if parsed.get("target_ram_gb") and any(w in t for w in ["cần", "lắp", "dùng", "muốn"]):
        return "build"

    # Còn lại là trò chuyện
    return "chat"


def _parse_single_text(text: str, data: Dict[str, List[Dict]]) -> Dict[str, Any]:
    """Trích xuất các tiêu chí kỹ thuật từ một câu chat đơn lẻ."""
    t = text.lower()
    locked_parts, unmatched_warnings = match_specific_components(text, data)

    res: Dict[str, Any] = {
        "budget": extract_budget_from_text(text),
        "strict_budget": extract_strict_budget_flag(text),
        "usage": extract_usage_from_text(text),
        "need_vga": None,
        "need_hdd": None,
        "cpu_brand": None,
        "gpu_brand": None,
        "target_ram_gb": None,
        "min_storage_gb": None,
        "min_total_storage_gb": None,
        "locked_parts": locked_parts,
        "unmatched_warnings": unmatched_warnings,
    }

    # Nhận diện yêu cầu có/không Card đồ họa rời (VGA)
    no_vga_patterns = [
        "không cần vga", "không cần card", "không vga", "không card",
        "k cần vga", "k cần card", "k vga", "k card", "onboard",
        "igpu", "ko can vga", "ko vga", "đồ họa tích hợp", "khong vga"
    ]
    has_vga_patterns = [
        "cần vga", "cần card", "card rời", "vga rời", "phải có card",
        "phải có vga", "lắp vga", "lắp card", "co card roi", "co vga"
    ]

    for p in no_vga_patterns:
        if p in t:
            res["need_vga"] = False
            break
    if res["need_vga"] is None:
        for p in has_vga_patterns:
            if p in t:
                res["need_vga"] = True
                break

    # Nhận diện yêu cầu có/không HDD
    no_hdd_patterns = [
        "không hdd", "không cần hdd", "ko hdd", "k cần hdd", "bỏ hdd", "khong hdd", "ko can hdd", "k hdd", "không lấy hdd"
    ]
    need_hdd_patterns = [
        "có hdd", "cần hdd", "thêm hdd", "lưu trữ nhiều", "ổ hdd",
        "chứa dữ liệu", "o hdd", "kèm hdd", "lap hdd", "lắp hdd",
        "có cả ssd và hdd", "cả ssd và hdd", "ssd và hdd", "ssd va hdd"
    ]
    if any(p in t for p in no_hdd_patterns):
        res["need_hdd"] = False
    elif re.search(r'\bhdd\b', t) or any(p in t for p in need_hdd_patterns):
        res["need_hdd"] = True

    # Nhận diện dung lượng lưu trữ tổng: 1T, 2T, 2TB, 1TB, 500GB...
    tb_match = re.search(r'\b(\d+)\s*(?:t|tb)\b', t)
    if tb_match:
        val_tb = int(tb_match.group(1))
        res["min_total_storage_gb"] = val_tb * 1000
    else:
        gb_match = re.search(r'\b(500|512|1000|2000|4000)\s*(?:gb|g)\b', t)
        if gb_match and any(w in t for w in ["ssd", "hdd", "ổ", "o cung", "lưu trữ", "luu tru", "dung lượng", "dung luong"]):
            res["min_total_storage_gb"] = int(gb_match.group(1))

    # Nhận diện hãng CPU mong muốn (kể cả lỗi chính tả/gõ tắt: inter, in-tel, intet, amđ...)
    if "cpu" not in res["locked_parts"]:
        if re.search(r'\b(?:intel|inter|in-tel|intet|core|i[3579])\b', t):
            res["cpu_brand"] = "Intel"
        elif re.search(r'\b(?:amd|amđ|ryzen|r[3579])\b', t):
            res["cpu_brand"] = "AMD"

    # Nhận diện hãng GPU mong muốn (kể cả lỗi chính tả: nvida, nividia, amđ...)
    if "gpu" not in res["locked_parts"] and res["need_vga"] is not False:
        if re.search(r'\b(?:nvidia|nvida|nividia|geforce|rtx|gtx)\b', t):
            res["gpu_brand"] = "NVIDIA"
        elif re.search(r'\b(?:amd|amđ|radeon|rx)\b', t):
            res["gpu_brand"] = "AMD"

    # Nhận diện dung lượng RAM cụ thể
    ram_match = re.search(r'\b(8|16|32|64)\s*(?:gb|g)\b', t)
    if ram_match:
        res["target_ram_gb"] = int(ram_match.group(1))

    # Nhận diện từ khóa bộ nhớ cao / nhiều RAM
    high_mem_patterns = [
        "bộ nhớ cao", "bo nho cao", "ram cao", "nhiều ram", "nhieu ram",
        "bộ nhớ lớn", "bo nho lon", "nhiều bộ nhớ", "nhieu bo nho"
    ]
    if any(p in t for p in high_mem_patterns):
        if not res["target_ram_gb"] or res["target_ram_gb"] < 32:
            res["target_ram_gb"] = 32
        res["min_storage_gb"] = 1000

    # Nhận diện yêu cầu ưu tiên Dual Channel RAM
    dual_patterns = [
        "dual ram", "ram dual", "chạy dual", "chay dual", "kênh đôi", "kenh doi",
        "2 thanh ram", "2 thanh", "hai thanh", "hai thanh ram", "kit ram", "kit 2 thanh",
        "thay vì thanh 64gb", "thay vi thanh 64gb", "thay vì 1 thanh", "thay vi 1 thanh"
    ]
    if any(p in t for p in dual_patterns):
        res["prefer_dual_ram"] = True

    return res


def parse_user_request(
    text: str,
    data: Dict[str, List[Dict]],
    conversation_history: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    """
    Phân tích câu hỏi tự nhiên của người dùng và trích xuất các tiêu chí cấu hình.
    Kế thừa và tổng hợp toàn bộ các tiêu chí người dùng đã nêu trong 6-8 tin nhắn gần nhất.
    """
    # 1. Thu thập chuỗi hội thoại của người dùng (tối đa 8 tin nhắn gần nhất)
    user_turns: List[str] = []
    if conversation_history:
        recent = conversation_history[-8:]
        for m in recent:
            if m.get("role") == "user" and m.get("content", "").strip():
                user_turns.append(m["content"].strip())

    if not user_turns or user_turns[-1] != text.strip():
        user_turns.append(text.strip())

    # 2. Duyệt qua từng lượt hỏi để kế thừa và tích lũy tiêu chí theo thời gian
    parsed: Dict[str, Any] = {
        "budget": None,
        "strict_budget": False,
        "usage": None,
        "need_vga": None,
        "need_hdd": None,
        "cpu_brand": None,
        "gpu_brand": None,
        "target_ram_gb": None,
        "min_storage_gb": None,
        "min_total_storage_gb": None,
        "prefer_dual_ram": False,
        "locked_parts": {},
        "unmatched_warnings": [],
        "extracted_summary": "",
        "intent": "chat",
    }

    for turn_txt in user_turns:
        curr = _parse_single_text(turn_txt, data)
        if curr["budget"] is not None:
            parsed["budget"] = curr["budget"]
            parsed["strict_budget"] = curr.get("strict_budget", False)
        if curr["usage"] is not None and curr["usage"] != "Tất cả / Đa dụng (Cân bằng)":
            parsed["usage"] = curr["usage"]
        elif curr["usage"] is not None and parsed["usage"] is None:
            parsed["usage"] = curr["usage"]
        if curr["need_vga"] is not None:
            parsed["need_vga"] = curr["need_vga"]
        if curr["need_hdd"] is not None:
            parsed["need_hdd"] = curr["need_hdd"]
        if curr["cpu_brand"] is not None:
            parsed["cpu_brand"] = curr["cpu_brand"]
            # Nếu người dùng đổi hãng CPU, xóa linh kiện CPU cũ không khớp
            if "cpu" in parsed["locked_parts"]:
                if parsed["locked_parts"]["cpu"].get("brand") != curr["cpu_brand"]:
                    del parsed["locked_parts"]["cpu"]
        if curr["gpu_brand"] is not None:
            parsed["gpu_brand"] = curr["gpu_brand"]
        if curr["target_ram_gb"] is not None:
            parsed["target_ram_gb"] = curr["target_ram_gb"]
        if curr["min_storage_gb"] is not None:
            parsed["min_storage_gb"] = curr["min_storage_gb"]
        if curr["min_total_storage_gb"] is not None:
            parsed["min_total_storage_gb"] = curr["min_total_storage_gb"]
        if curr.get("prefer_dual_ram"):
            parsed["prefer_dual_ram"] = True
        for p_k, p_val in curr["locked_parts"].items():
            parsed["locked_parts"][p_k] = p_val
        if curr["unmatched_warnings"]:
            parsed["unmatched_warnings"] = curr["unmatched_warnings"]

    if extract_strict_budget_flag(text):
        parsed["strict_budget"] = True

    # 2b. Kiểm tra xem người dùng có chọn mốc ngân sách (Mức 1, Mức 2, Mức 3...) hay không
    tier_idx = extract_tier_selection(text)
    if tier_idx is not None:
        parsed["selected_tier_idx"] = tier_idx
        resolved_budget = resolve_budget_from_tier(tier_idx, conversation_history, data)
        if resolved_budget:
            parsed["budget"] = resolved_budget

    # 2c. Nhận diện game hoặc phần mềm từ câu hiện tại hoặc lịch sử
    app_game = detect_game_or_app(text)
    if not app_game and conversation_history:
        for msg in reversed(conversation_history):
            if msg.get("role") == "user":
                app_game = detect_game_or_app(msg.get("content", ""))
                if app_game:
                    break
    if app_game:
        parsed["app_game_profile"] = app_game
        if not parsed.get("usage") or parsed["usage"] == "Tất cả / Đa dụng (Cân bằng)":
            parsed["usage"] = app_game.get("usage", "Gaming")
        if app_game.get("usage") == "Gaming":
            if parsed.get("need_vga") is None:
                parsed["need_vga"] = True
        elif app_game.get("usage") == "Văn phòng / Học tập":
            if parsed.get("need_vga") is None:
                parsed["need_vga"] = False
            if parsed.get("need_hdd") is None:
                parsed["need_hdd"] = True

    # 3. Xây dựng đoạn tóm tắt tiêu chí nhận diện được
    summary_parts = []
    if parsed["budget"]:
        budget_tag = f"Ngân sách: {fmt_vnd(parsed['budget'])}"
        if parsed.get("strict_budget"):
            budget_tag += " (Tối đa / Cố định)"
        summary_parts.append(budget_tag)
    if parsed["usage"]:
        summary_parts.append(f"Nhu cầu: {parsed['usage']}")
    if parsed["cpu_brand"]:
        summary_parts.append(f"Hãng CPU: {parsed['cpu_brand']}")
    if parsed["gpu_brand"]:
        summary_parts.append(f"Hãng GPU: {parsed['gpu_brand']}")
    if parsed["target_ram_gb"]:
        ram_tag = f"RAM: {parsed['target_ram_gb']}GB"
        if parsed.get("prefer_dual_ram"):
            ram_tag += " (Ưu tiên Dual Channel)"
        summary_parts.append(ram_tag)
    elif parsed.get("prefer_dual_ram"):
        summary_parts.append("RAM: Ưu tiên Dual Channel (Kit 2 thanh)")
    if parsed.get("min_total_storage_gb"):
        val_t = parsed["min_total_storage_gb"] // 1000 if parsed["min_total_storage_gb"] >= 1000 else f"{parsed['min_total_storage_gb']}GB"
        tag = f"{val_t}TB" if isinstance(val_t, int) else val_t
        if parsed.get("need_hdd"):
            summary_parts.append(f"Lưu trữ: Tối thiểu {tag} (cả SSD và HDD)")
        else:
            summary_parts.append(f"Lưu trữ: Tối thiểu {tag}")
    elif parsed.get("min_storage_gb"):
        summary_parts.append("SSD: Tối thiểu 1TB")
    if parsed["need_vga"] is False:
        summary_parts.append("Dùng iGPU tích hợp (không VGA rời)")
    elif parsed["need_vga"] is True:
        summary_parts.append("Yêu cầu VGA rời")
    if parsed["need_hdd"] is True and not parsed.get("min_total_storage_gb"):
        summary_parts.append("Trang bị thêm ổ cứng HDD")

    for part_type, item in parsed["locked_parts"].items():
        summary_parts.append(f"Cố định {part_type.upper()}: {item['name']}")

    parsed["intent"] = classify_intent(text, parsed)
    parsed["should_build"] = bool(parsed["intent"] == "build" or parsed["intent"] == "upgrade")

    if parsed["intent"] == "upgrade":
        parsed["extracted_summary"] = "Đồng ý áp dụng nâng cấp linh kiện được gợi ý."
    elif parsed["intent"] == "consult_no_budget":
        if parsed.get("app_game_profile"):
            parsed["extracted_summary"] = f"Tư vấn các phương án cho {parsed['app_game_profile']['name']}."
        elif parsed.get("usage"):
            parsed["extracted_summary"] = f"Tư vấn các phương án cho nhu cầu {parsed['usage']}."
        else:
            parsed["extracted_summary"] = "Tư vấn các phương án tham khảo."
    elif parsed["intent"] == "chat":
        parsed["extracted_summary"] = "Trò chuyện & tư vấn thông thường."
    else:
        parsed["extracted_summary"] = ", ".join(summary_parts) if summary_parts else "Tự động tối ưu tổng thể."
    return parsed


# ===========================================================================
# PHẦN 2: BỘ SINH NHẬN XÉT CẤU HÌNH TỰ ĐỘNG (BUILD REVIEW GENERATOR)
# ===========================================================================
def generate_build_review(
    selected_parts: Dict[str, Optional[Dict]],
    quantities: Dict[str, int],
    budget: int,
    usage: str,
    eval_result: Any,
) -> str:
    """
    Sinh bài đánh giá, nhận xét cấu hình tự nhiên, mạch lạc bằng Tiếng Việt gồm 3 ý chính:
    1. Đánh giá tương thích và độ cân bằng phần cứng (R01 - R10).
    2. Khả năng đáp ứng thực tế theo nhu cầu sử dụng.
    3. Nhận xét phân bổ ngân sách và đề xuất tối ưu.
    """
    cpu = selected_parts.get("cpu")
    mb = selected_parts.get("mainboard")
    ram = selected_parts.get("ram")
    gpu = selected_parts.get("gpu")
    psu = selected_parts.get("psu")
    case = selected_parts.get("case")
    storage = selected_parts.get("storage")
    cooler = selected_parts.get("cooler")
    hdd = selected_parts.get("hdd")

    # Kiểm tra xem cấu hình có linh kiện nào chưa
    active_parts = [p for p in selected_parts.values() if p]
    if not active_parts:
        return (
            "Hiện tại cấu hình chưa có linh kiện nào được chọn. "
            "Bạn có thể nhắn tin yêu cầu tư vấn (ví dụ: 'Build PC Gaming 25 triệu chip Intel, RAM 32GB') "
            "hoặc tự chọn linh kiện từ bảng bên trái để hệ thống nhận xét chi tiết."
        )

    # Tính tổng tiền và mức chênh lệch ngân sách
    total_cost = 0
    for cat_name, cat_key, state_key, _ in [
        ("CPU", "cpus", "cpu", False),
        ("Mainboard", "mainboards", "mainboard", False),
        ("RAM", "rams", "ram", True),
        ("GPU", "gpus", "gpu", False),
        ("SSD", "storages", "storage", True),
        ("HDD", "hdds", "hdd", True),
        ("PSU", "psus", "psu", False),
        ("Case", "cases", "case", False),
        ("Cooler", "coolers", "cooler", False),
    ]:
        item = selected_parts.get(state_key)
        if item:
            qty = quantities.get(cat_key, 1)
            total_cost += item.get("price", 0) * qty

    diff = budget - total_cost

    # Nếu có Google Gemini API Key hợp lệ, thử sinh nhận xét bằng mô hình Gemini
    active_key = get_gemini_api_key()
    if active_key:
        compat_info = "; ".join(eval_result.errors) if (eval_result and eval_result.errors) else "Hoàn toàn tương thích kỹ thuật"
        gemini_prompt = (
            f"Dưới đây là cấu hình PC người dùng đã chọn:\n"
            f"- Ngân sách dự tính: {fmt_vnd(budget)}\n"
            f"- Tổng chi phí linh kiện: {fmt_vnd(total_cost)}\n"
            f"- Nhu cầu sử dụng: {usage}\n"
            f"- CPU: {cpu['name'] if cpu else 'Chưa chọn'}\n"
            f"- Bo mạch chủ: {mb['name'] if mb else 'Chưa chọn'}\n"
            f"- RAM: {ram['name'] if ram else 'Chưa chọn'} (Số lượng: {quantities.get('rams', 1)})\n"
            f"- Card đồ họa (VGA): {gpu['name'] if gpu else 'Chưa chọn'}\n"
            f"- Ổ cứng SSD: {storage['name'] if storage else 'Chưa chọn'}\n"
            f"- Ổ cứng HDD: {hdd['name'] if hdd else 'Không dùng'}\n"
            f"- Bộ nguồn: {psu['name'] if psu else 'Chưa chọn'}\n"
            f"- Tản nhiệt: {cooler['name'] if cooler else 'Chưa chọn'}\n"
            f"- Trạng thái tương thích: {compat_info}\n\n"
            f"QUY TẮC BẮT BUỘC:\n"
            f"1. TUYỆT ĐỐI KHÔNG mở đầu bằng bất kỳ câu chào hỏi hay rào đón nào như 'Chào bạn', 'Với tư cách là chuyên gia...'. Hãy đi thẳng vào nội dung nhận xét.\n"
            f"2. BẮT BUỘC trình bày ngắn gọn, mạch lạc đúng 3 gạch đầu dòng:\n"
            f"- Tương thích & Cân bằng: Đánh giá tính tương thích phần cứng và độ cân bằng giữa CPU và GPU (tránh nghẽn cổ chai).\n"
            f"- Hiệu năng thực tế: Khả năng đáp ứng thực tế cho nhu cầu {usage}.\n"
            f"- Tối ưu ngân sách: Nhận xét phân bổ chi phí và đề xuất tối ưu.\n"
            f"3. TUYỆT ĐỐI KHÔNG dùng bất kỳ emoji nào."
        )
        gemini_review = call_gemini_api(gemini_prompt, api_key=active_key, timeout=5)
        if gemini_review and len(gemini_review.strip()) > 30:
            clean_rev = re.sub(r'[\U00010000-\U0010ffff]', '', gemini_review).strip()
            # Bỏ bất kỳ câu rào đón nào còn sót lại
            clean_rev = re.sub(r'^(?:chào bạn[^\n]*\n*|với tư cách[^\n]*\n*|dưới đây là[^\n]*\n*)+', '', clean_rev, flags=re.IGNORECASE).strip()
            return clean_rev

    paragraphs = []

    # -----------------------------------------------------------------------
    # Ý 1: Đánh giá tương thích và độ cân bằng CPU - GPU
    # -----------------------------------------------------------------------
    p1 = []
    if eval_result and eval_result.errors:
        p1.append(
            f"Cảnh báo tương thích: Cấu hình hiện tại đang gặp {len(eval_result.errors)} xung đột kỹ thuật: "
            + "; ".join(eval_result.errors)
            + ". Bạn cần đổi linh kiện tương ứng để máy có thể khởi động bình thường."
        )
    else:
        compat_notes = []
        if cpu and mb:
            compat_notes.append(f"Socket {cpu['socket']}")
        if mb and ram:
            compat_notes.append(f"chuẩn {ram['ram_type']}")
        if mb and case:
            compat_notes.append(f"kích thước {mb['form_factor']}")

        spec_str = f" ({', '.join(compat_notes)})" if compat_notes else ""
        p1.append(
            f"Toàn bộ linh kiện đã chọn phối hợp chuẩn xác{spec_str}. "
        )

        # Đánh giá cấu hình RAM (Dual Channel vs Single)
        if ram:
            is_dual = (
                ram.get("is_kit", False)
                or ram.get("sticks", 1) >= 2
                or "2x" in ram.get("name", "").lower()
                or quantities.get("rams", 1) >= 2
            )
            if is_dual:
                p1.append(
                    "Cấu hình sử dụng RAM chạy kênh đôi (Dual Channel) giúp nhân đôi băng thông truyền tải dữ liệu, "
                    "giảm thiểu hiện tượng giật khung hình (1% low FPS) và tối ưu độ ổn định khi xử lý đa nhiệm. "
                )
            else:
                p1.append(
                    "Cấu hình hiện đang chạy RAM kênh đơn (Single Channel). "
                    "Nếu muốn tối đa hóa băng thông bộ nhớ và chống tụt FPS khi chơi game, bạn nên ưu tiên chọn kit 2 thanh (Dual Channel). "
                )

        # Đánh giá công suất nguồn
        if cpu and gpu and psu:
            extra_drives = max(0, quantities.get("storages", 1) - 1) + (quantities.get("hdds", 1) if hdd and hdd.get("id") != "hdd_none" else 0)
            rec_w = calculate_min_psu_wattage(cpu, gpu, extra_storage_count=extra_drives)
            if psu["wattage"] >= rec_w:
                p1.append(f"Bộ nguồn {psu['name']} ({psu['wattage']}W) đáp ứng thoải mái công suất khuyến nghị ({rec_w}W). ")

        # Đánh giá độ cân bằng CPU - GPU
        if cpu and gpu:
            if gpu["id"] == "gpu_none":
                p1.append(
                    f"Cấu hình sử dụng nhân đồ họa tích hợp iGPU trong CPU {cpu['name']}, "
                    "vận hành rất êm ái, tiết kiệm điện năng và không phát sinh nhiệt lượng lớn."
                )
            else:
                is_balanced, level, _ = check_r08_bottleneck(cpu, gpu)
                if level == "cpu_bottleneck":
                    p1.append(
                        f"Lưu ý cân bằng: CPU {cpu['name']} có phần hơi thấp so với Card đồ họa {gpu['name']}. "
                        "Trong các tác vụ hoặc tựa game ăn nhiều CPU, Card đồ họa có thể chưa phát huy được tối đa 100% công suất."
                    )
                elif level == "gpu_bottleneck":
                    p1.append(
                        f"Về độ cân bằng: CPU {cpu['name']} có sức mạnh vượt trội so với Card đồ họa {gpu['name']}. "
                        "Cấu hình này sẽ tối ưu vượt bậc cho các tác vụ xử lý tính toán, lập trình, render CPU hơn là trải nghiệm game đồ họa nặng."
                    )
                else:
                    p1.append(
                        f"Sự kết hợp giữa vi xử lý {cpu['name']} và card đồ họa {gpu['name']} "
                        "đạt tỷ lệ cân đối tối ưu, phát huy trọn vẹn sức mạnh phần cứng mà không lo nghẽn cổ chai."
                    )

    # -----------------------------------------------------------------------
    # Ý 2: Khả năng đáp ứng thực tế theo nhu cầu sử dụng
    # -----------------------------------------------------------------------
    p2 = []
    gpu_score = gpu.get("score", 0) if gpu else 0
    cpu_score = cpu.get("score", 0) if cpu else 0

    if usage == "Gaming":
        if gpu and gpu.get("id") == "gpu_none":
            p2.append(
                "Do chưa trang bị Card đồ họa rời, máy đáp ứng tốt các tựa game eSports nhẹ "
                "(Liên Minh Huyền Thoại, Đột Kích, CS:GO cũ) ở mức thiết lập đồ họa cơ bản. "
                "Nếu muốn chiến mượt Valorant, PUBG hay GTA V, bạn nên trang bị thêm VGA rời."
            )
        elif gpu_score >= 35000:
            p2.append(
                f"Sức mạnh đồ họa đỉnh cao từ {gpu['name']} cho phép bạn trải nghiệm "
                "mọi tựa game bom tấn AAA (Cyberpunk 2077, Black Myth: Wukong, Forza Horizon 5) ở độ phân giải 4K "
                "với thiết lập đồ họa Ultra Settings và bật Ray Tracing mượt mà."
            )
        elif gpu_score >= 25000:
            p2.append(
                f"Card đồ họa {gpu['name']} chiến xuất sắc toàn bộ các tựa game hiện đại "
                "ở độ phân giải 2K (1440p) thiết lập Max Settings, duy trì chỉ số khung hình FPS cao và ổn định."
            )
        elif gpu_score >= 12000:
            p2.append(
                f"Cấu hình cân mượt mà mọi tựa game eSports thịnh hành (Valorant, CS2, LOL, PUBG, FC Online) "
                "ở độ phân giải Full HD (1080p) với tần số quét 144Hz - 240Hz, đồng thời chơi tốt game AAA ở mức High Settings."
            )
        else:
            p2.append(
                "Máy đáp ứng tốt các tựa game eSports cơ bản ở thiết lập đồ họa trung bình, "
                "mang lại trải nghiệm giải trí ổn định trong phân khúc giá phổ thông."
            )

    elif usage == "Đồ họa / Lập trình AI":
        ram_gb = (ram.get("capacity_gb", 8) * quantities.get("rams", 1)) if ram else 8
        p2.append(
            f"Vi xử lý {cpu['name'] if cpu else 'hiện tại'} cùng dung lượng RAM {ram_gb}GB "
            "đáp ứng hoàn hảo các tác vụ render video 2D/3D (Premiere Pro, After Effects, DaVinci Resolve), "
            "thiết kế đồ họa Photoshop, dựng hình Blender và biên dịch mã nguồn, chạy máy ảo đa nhiệm mượt mà."
        )
        if gpu and gpu.get("vram_gb", 0) >= 12:
            p2.append(
                f" Bộ nhớ VRAM {gpu['vram_gb']}GB của {gpu['name']} hỗ trợ đắc lực cho việc tăng tốc render CUDA "
                "và huấn luyện các mô hình AI/Machine Learning cục bộ."
            )

    elif usage == "Văn phòng / Học tập":
        p2.append(
            "Tốc độ xử lý đa nhiệm cực kỳ ấn tượng. Ổ cứng SSD giúp mở Windows và phần mềm "
            "trong 3 - 5 giây, mở các bảng tính Excel hàng trăm nghìn dòng, chạy đồng thời nhiều tab trình duyệt và ứng dụng họp trực tuyến "
            "hoàn toàn trơn tru, không có hiện tượng giật lag."
        )

    else:  # Tất cả / Đa dụng (Cân bằng)
        p2.append(
            "Cấu hình mang tính toàn diện cao, vừa phục vụ xuất sắc công việc học tập, "
            "văn phòng, chỉnh sửa đồ họa đa phương tiện bán chuyên, vừa sẵn sàng chiến tốt các tựa game giải trí phổ biến."
        )

    # -----------------------------------------------------------------------
    # Ý 3: Đánh giá ngân sách và đề xuất tối ưu
    # -----------------------------------------------------------------------
    p3 = []
    if diff > 1_500_000:
        p3.append(
            f"Cấu hình hiện tại có tổng chi phí {fmt_vnd(total_cost)}, vẫn còn dư {fmt_vnd(diff)} so với ngân sách {fmt_vnd(budget)}. "
            "Gợi ý: Bạn có thể tận dụng số tiền dư này để nâng cấp dung lượng RAM lên mức cao hơn hoặc nâng ổ cứng SSD lên 1TB "
            "để mở rộng không gian cài game và lưu trữ dữ liệu."
        )
    elif 0 <= diff <= 1_500_000:
        p3.append(
            f"Tổng chi phí {fmt_vnd(total_cost)} bám rất sát mức ngân sách dự kiến {fmt_vnd(budget)} (chênh lệch chỉ {fmt_vnd(diff)}). "
            "Tỷ lệ phân bổ chi phí giữa CPU, Bo mạch chủ và Card đồ họa đạt hiệu quả đầu tư tối đa trên từng đồng chi tiêu."
        )
    else:
        p3.append(
            f"Cấu hình hiện tại đạt {fmt_vnd(total_cost)}, vượt ngân sách dự kiến {fmt_vnd(abs(diff))}. "
            "Nếu cần tiết kiệm chi phí về đúng ngưỡng ngân sách, bạn có thể cân nhắc chuyển sang vỏ máy hoặc tản nhiệt kinh tế hơn "
            "mà vẫn bảo toàn nguyên vẹn hiệu năng cốt lõi của CPU và Card đồ họa."
        )

    paragraphs = [
        "- Tương thích & Cân bằng: " + "".join(p1),
        "- Hiệu năng thực tế: " + "".join(p2),
        "- Tối ưu ngân sách: " + "".join(p3),
    ]

    return "\n\n".join(paragraphs)


# ===========================================================================
# PHẦN 3: BỘ SINH PHẢN HỒI HỘI THOẠI (CHAT RESPONSE BUILDER)
# ===========================================================================
def generate_conversational_reply(
    user_query: str,
    parsed: Dict[str, Any],
    build: Optional[Any],
    budget: int,
    usage: str,
    conversation_history: Optional[List[Dict[str, str]]] = None,
    min_needed_budget: Optional[int] = None,
    upgrade_suggestions: Optional[List[Dict[str, Any]]] = None,
    catalog: Optional[Dict[str, List[Dict]]] = None,
) -> str:
    """
    Sinh câu phản hồi tự nhiên, thân thiện như chuyên viên tư vấn Build PC tại cửa hàng (xưng hô 'mình - bạn').
    Xử lý các tình huống lệch tầm giá, thiếu ngân sách tối thiểu hoặc nghẽn cổ chai nặng.
    Tuyệt đối không in các nhãn kỹ thuật khô khan.
    Ưu tiên gọi Gemini AI nếu có Key hợp lệ; fallback sang bộ sinh hội thoại tự nhiên nội bộ.
    """
    active_key = get_gemini_api_key()

    # Tự động tính toán gợi ý nâng cấp nếu chưa có
    if upgrade_suggestions is None and build is not None and catalog is not None:
        try:
            upgrade_suggestions = suggest_upgrades(build, catalog, budget, usage)
        except Exception:
            upgrade_suggestions = []

    # Kiểm tra nghẽn cổ chai nếu đã lên được build
    bottleneck_detected = False
    if build and build.cpu and build.gpu:
        _, b_level, _ = check_r08_bottleneck(build.cpu, build.gpu)
        if b_level == "cpu_bottleneck":
            bottleneck_detected = True

    # Tạo mô tả dung lượng lưu trữ
    if build:
        ram_cap = build.ram.get("capacity_gb", 16)
        ssd_cap = build.storage.get("capacity_gb", 500)
        has_hdd = build.hdd and build.hdd.get("id") != "hdd_none"
        is_ssd_sata = "SATA" in build.storage.get("name", "").upper() or "SATA" in str(build.storage.get("interface", "")).upper()
        if is_ssd_sata:
            ssd_perf_desc = f"ổ SSD SATA {ssd_cap}GB (khởi động hệ điều hành và mở ứng dụng ổn định)"
        else:
            ssd_perf_desc = f"ổ SSD NVMe {ssd_cap}GB tốc độ cao"

        if has_hdd:
            storage_desc = f"{ssd_perf_desc} kết hợp cùng HDD {build.hdd.get('name', 'lưu trữ')}"
        else:
            storage_desc = ssd_perf_desc
    else:
        storage_desc = "ổ cứng"

    # Tạo tóm tắt linh kiện yêu cầu chính (dùng cho TH không đủ ngân sách)
    comp_list = []
    if parsed.get("cpu_brand"):
        comp_list.append(parsed["cpu_brand"])
    if parsed.get("locked_parts", {}).get("gpu"):
        g_name = parsed["locked_parts"]["gpu"]["name"].replace("NVIDIA GeForce ", "")
        comp_list.append(g_name)
    elif parsed.get("gpu_brand"):
        comp_list.append(parsed["gpu_brand"])
    if parsed.get("min_total_storage_gb"):
        val_t = parsed["min_total_storage_gb"] // 1000 if parsed["min_total_storage_gb"] >= 1000 else f"{parsed['min_total_storage_gb']}GB"
        tag = f"{val_t}TB" if isinstance(val_t, int) else val_t
        comp_list.append(f"{tag} lưu trữ")
    comp_str = " + ".join(comp_list) if comp_list else "tổ hợp linh kiện theo yêu cầu"

    b_val = budget or 0
    min_needed = min_needed_budget or (b_val + 2_500_000)
    min_needed_tr = f"{min_needed / 1_000_000:,.1f}".replace(".", ",").rstrip("0").rstrip(",") + " triệu"
    budget_tr = f"{b_val / 1_000_000:,.1f}".replace(".", ",").rstrip("0").rstrip(",") + " triệu"

    if active_key:
        history_text = ""
        if conversation_history:
            recent = conversation_history[-8:]
            hist_lines = []
            for m in recent:
                r = "Khách hàng" if m.get("role") == "user" else "Chuyên viên tư vấn"
                hist_lines.append(f"{r}: {m.get('content', '')}")
            history_text = "\nLịch sử trao đổi trước đó:\n" + "\n".join(hist_lines) + "\n"

        warnings_text = ""
        if parsed.get("unmatched_warnings"):
            warnings_text = " (Lưu ý: " + "; ".join(parsed["unmatched_warnings"]) + ")"

        now_dt = datetime.now()
        days_vi = ["Thứ Hai", "Thứ Ba", "Thứ Tư", "Thứ Năm", "Thứ Sáu", "Thứ Bảy", "Chủ Nhật"]
        now_full_str = f"{days_vi[now_dt.weekday()]}, ngày {now_dt.strftime('%d/%m/%Y')}, lúc {now_dt.strftime('%H:%M:%S')}"

        # TH1: Trò chuyện / chào hỏi / hỏi đáp thông thường (intent == 'chat')
        if parsed.get("intent") == "chat":
            gemini_prompt = (
                "Bạn là một chuyên viên tư vấn PC thân thiện, chu đáo tại cửa hàng máy tính uy tín.\n"
                f"- Thời gian thực hiện tại của hệ thống: {now_full_str} (Năm {now_dt.year}, Tháng {now_dt.month}).\n"
                f"{history_text}\n"
                f"Khách hàng vừa nhắn tin: \"{user_query}\"\n\n"
                "QUY TẮC BẮT BUỘC:\n"
                "1. Xưng hô tự nhiên là 'mình' và gọi khách là 'bạn'.\n"
                "2. Văn phong thân thiện, gần gũi, ngắn gọn chỉ 1-2 câu như người thật đang nhắn tin chat trực tiếp.\n"
                "3. Trả lời đúng trọng tâm câu hỏi của khách hàng: nếu khách hỏi ngày, giờ, thứ, tháng, năm hay thời gian, hãy trả lời chính xác dựa theo thời gian thực hiện tại của hệ thống; nếu chào hỏi niềm nở hoặc giải đáp ngắn gọn, ghi nhận nhu cầu/phần cứng khách mong muốn và hỏi mức ngân sách dự kiến để bắt đầu lên cấu hình.\n"
                "4. TUYỆT ĐỐI KHÔNG dùng bất kỳ emoji hay icon cảm xúc nào.\n"
                "5. TUYỆT ĐỐI KHÔNG tự bịa hoặc nhắc đến mức ngân sách mặc định khi khách không hề nhắc đến số tiền đó.\n"
                "6. Tự động nhận diện lỗi chính tả phổ biến (inter = Intel, amđ = AMD, 2t = 2TB lưu trữ) và trả lời bình thường.\n"
            )

        # TH1b: Tư vấn theo Nhu cầu / Game / Phần mềm khi khách chưa nêu ngân sách cụ thể (intent == 'consult_no_budget')
        elif parsed.get("intent") == "consult_no_budget":
            prof = parsed.get("app_game_profile")
            target_name = prof["name"] if prof else parsed.get("usage", "Gaming")
            analysis_hint = prof["analysis"] if prof else "Cần cân đối hiệu năng và tối ưu chi phí theo nhu cầu thực tế."
            tiers_hint = ""
            if prof and "tiers" in prof:
                t_lines = []
                for t in prof["tiers"]:
                    t_lines.append(f"- {t['label']}: {t['desc']}")
                tiers_hint = "\nCác phương án gợi ý tham khảo:\n" + "\n".join(t_lines) + "\n"

            gemini_prompt = (
                "Bạn là chuyên viên tư vấn phần cứng của phần mềm AI PC Builder Advisor.\n"
                f"{history_text}\n"
                f"Khách hàng vừa hỏi: \"{user_query}\"\n"
                f"Nhu cầu / Game hoặc phần mềm khách quan tâm: {target_name}.\n"
                f"Đặc thù kỹ thuật thực tế: {analysis_hint}\n"
                f"{tiers_hint}\n"
                "QUY TẮC BẮT BUỘC VỀ BỐ CỤC VÀ VĂN PHONG:\n"
                "1. TUYỆT ĐỐI CẤM chào hỏi rườm rà. TUYỆT ĐỐI CẤM liệt kê tên linh kiện cụ thể trong từng phương án.\n"
                "2. BẮT BUỘC trả lời đúng 5 dòng (tổng dưới 60 từ), mỗi dòng phương án tối đa 12 từ theo đúng mẫu sau:\n"
                "   [1 câu ngắn dưới 18 từ nêu đúng yêu cầu cốt lõi của game/nhu cầu]:\n"
                "   - **Phương án 1 (X – Y triệu):** [Trải nghiệm ngắn gọn <= 10 từ].\n"
                "   - **Phương án 2 (X – Y triệu):** [Trải nghiệm ngắn gọn <= 10 từ].\n"
                "   - **Phương án 3 (X – Y triệu):** [Trải nghiệm ngắn gọn <= 10 từ].\n\n"
                "   Bạn chọn phương án nào để mình lên cấu hình chi tiết?\n\n"
                "VÍ DỤ MẪU CHUẨN CHO GTA 5:\n"
                "GTA 5 cần CPU ổn định và đủ RAM để tải bản đồ mượt mà không khựng giật:\n"
                "- **Phương án 1 (11 – 13 triệu):** Chiến mượt Full HD mức High 60–80 FPS ổn định.\n"
                "- **Phương án 2 (20 – 24 triệu):** Chơi mượt bản mod và server đông người, Max Setting.\n"
                "- **Phương án 3 (32 – 40 triệu):** Trải nghiệm đồ họa đỉnh cao 2K–4K và mod nặng.\n\n"
                "Bạn chọn phương án nào để mình lên cấu hình chi tiết?\n\n"
                "3. TUYỆT ĐỐI KHÔNG dùng bất kỳ emoji hay icon cảm xúc nào.\n"
                "4. Dùng xưng hô 'mình' và 'bạn'."
            )

        # TH2: Có ý định build nhưng không tìm thấy cấu hình phù hợp (Trường hợp 1: Không đủ tiền lắp tối thiểu)
        elif not build:
            gemini_prompt = (
                "Bạn là một kỹ thuật viên tư vấn Build PC chuyên nghiệp tại cửa hàng máy tính uy tín.\n"
                f"{history_text}\n"
                f"Khách hàng vừa yêu cầu: \"{user_query}\"\n"
                f"Ngân sách khách hàng đưa ra: {fmt_vnd(budget)} ({budget_tr}).\n"
                f"Hệ thống kiểm tra thấy mức ngân sách này không đủ để lắp tối thiểu bộ linh kiện yêu cầu: {comp_str} (tối thiểu cần khoảng {min_needed_tr}).\n\n"
                "QUY TẮC BẮT BUỘC CỦA KỸ THUẬT VIÊN:\n"
                "1. Xưng hô tự nhiên là 'mình' và gọi khách là 'bạn'.\n"
                f"2. Nêu rõ thẳng thắn và lịch sự: Với {budget_tr} thì hơi căng để gánh trọn bộ {comp_str} (tối thiểu cần khoảng {min_needed_tr}).\n"
                "3. Đặt câu hỏi gợi ý giải pháp thực tế: Hỏi khách muốn tăng ngân sách lên mức này hay hạ bớt card đồ họa để cấu hình cân đối hơn.\n"
                "4. TUYỆT ĐỐI KHÔNG dùng bất kỳ emoji nào.\n"
                "5. Độ dài vừa phải 2-3 câu ngắn gọn.\n"
            )

        # TH3: Lắp vừa tiền nhưng bị nghẽn cổ chai nặng (Trường hợp 2)
        elif bottleneck_detected:
            gemini_prompt = (
                "Bạn là một kỹ thuật viên tư vấn Build PC chuyên nghiệp tại cửa hàng máy tính uy tín.\n"
                f"{history_text}\n"
                f"Khách hàng vừa yêu cầu: \"{user_query}\"\n"
                f"Cấu hình hệ thống đã cố gắng tối ưu vào ngân sách {fmt_vnd(budget)}:\n"
                f"- CPU: {build.cpu['name']}\n"
                f"- Card đồ họa (VGA): {build.gpu['name']}\n"
                f"- Bộ nhớ RAM: {build.ram['name']}\n"
                f"- Lưu trữ: {storage_desc}\n"
                f"- Tổng chi phí: {fmt_vnd(build.total_price)}\n\n"
                "CẢNH BÁO KỸ THUẬT BẮT BUỘC (NGHẼN CỔ CHAI CPU NẶNG THEO LUẬT R08):\n"
                f"Cấu hình này Card đồ họa {build.gpu['name']} quá mạnh trong khi vi xử lý {build.cpu['name']} quá yếu.\n"
                "Bạn PHẢI CẢNH BÁO THẲNG THẮN VÀ TƯ VẤN NHƯ KỸ THUẬT VIÊN THỰC THỤ:\n"
                f"- Xác nhận: Mình đã cố gắng ép đủ {build.gpu['name']} và {storage_desc} vào đúng {budget_tr}.\n"
                f"- Cảnh báo: Ngân sách còn lại chỉ đủ lắp chip {build.cpu['name']} nên sẽ bị nghẽn cổ chai khá nặng.\n"
                "- Lời khuyên tối ưu: Theo mình bạn nên cân nhắc đổi sang dòng card đồ họa phù hợp hơn để nâng CPU lên dòng cao hơn, giúp cấu hình cân đối và mượt mà hơn.\n"
                "QUY TẮC:\n"
                "- Xưng hô 'mình - bạn', văn phong người thật, ngắn gọn 2-3 câu.\n"
                "- TUYỆT ĐỐI KHÔNG dùng bất kỳ emoji nào.\n"
            )

        # TH4: Cấu hình cân đối, tối ưu chuẩn
        else:
            upgrade_info = ""
            if upgrade_suggestions:
                up_lines = []
                for u in upgrade_suggestions:
                    up_lines.append(f"- Nâng từ {u['current_name']} lên {u['upgrade_name']}: Bù thêm {fmt_vnd(u['added_cost'])}, mang lại lợi ích: {u['benefit']}")
                upgrade_info = "\nPhương án nâng cấp cấu hình đáng giá tính toán được từ hệ thống (nếu khách có thể thêm tiền):\n" + "\n".join(up_lines) + "\n"

            user_budget_info = f"- Ngân sách khách hàng: {fmt_vnd(budget)} ({budget_tr})\n"
            gemini_prompt = (
                "Bạn là một chuyên viên tư vấn Build PC chuyên nghiệp, nhiệt tình và am hiểu sâu sắc về phần cứng tại cửa hàng máy tính uy tín.\n"
                "Hãy trả lời khách hàng dựa trên thông tin cấu hình vừa được hệ thống tối ưu bên dưới và toàn bộ ngữ cảnh trao đổi trước đó.\n"
                f"{history_text}\n"
                f"Tin nhắn mới nhất của khách hàng: \"{user_query}\"\n"
                f"Thông tin cấu hình tối ưu vừa lập trên bảng linh kiện:\n"
                f"{user_budget_info}"
                f"- Nhu cầu: {usage}\n"
                f"- Tổng chi phí dàn máy: {fmt_vnd(build.total_price)}\n"
                f"- CPU: {build.cpu['name']}\n"
                f"- Bo mạch chủ: {build.mainboard['name']}\n"
                f"- RAM: {build.ram['name']}\n"
                f"- Card đồ họa (VGA): {build.gpu['name']}\n"
                f"- Ổ cứng SSD: {build.storage['name']}\n"
                f"- Ổ cứng HDD: {build.hdd['name'] if build.hdd else 'Không dùng'}\n"
                f"- Bộ nguồn: {build.psu['name']}\n"
                f"- Vỏ máy: {build.case['name']}\n"
                f"- Tản nhiệt: {build.cooler['name'] if build.cooler else 'Mặc định'}\n"
                f"{upgrade_info}"
                f"{warnings_text}\n\n"
                "QUY TẮC BẮT BUỘC: ĐỊNH DẠNG XUỐNG DÒNG RÕ RÀNG TỪNG LINH KIỆN (CẤM GỘP CHUNG 1 ĐOẠN VĂN XUÔI):\n"
                "Bắt buộc trình bày theo đúng mẫu 4 linh kiện cốt lõi sau:\n"
                f"Đã dựng xong dàn **{usage}** tổng **{fmt_vnd(build.total_price)}** ở bảng bên trái:\n"
                "- **CPU:** [Tên CPU]\n"
                "- **VGA:** [Tên VGA hoặc 'Dùng iGPU tích hợp']\n"
                "- **RAM:** [Dung lượng & chuẩn RAM, ví dụ: 16GB DDR4]\n"
                "- **Lưu trữ:** [SSD / HDD nếu có. Nếu là SSD SATA thì mô tả 'SSD SATA {dung lượng} (khởi động hệ điều hành và mở ứng dụng ổn định)', cấm nói 'tốc độ siêu cao']\n\n"
                "**Nhận xét:** [1 câu ngắn về hiệu năng thực tế].\n"
                "**Gợi ý nâng cấp:** [Chỉ hiện nếu tổng chi phí < 30 triệu và không phải cấu hình cao cấp; viết 1 câu ngắn đúng nhu cầu. Ở Văn phòng / Học tập: cấm nhắc game nặng hay render 3D].\n\n"
                "Bạn có muốn đổi linh kiện nào hay chỉnh ngân sách không?\n\n"
                "QUY TẮC BỔ SUNG:\n"
                "1. TUYỆT ĐỐI CẤM LIỆT KÊ LẠI TOÀN BỘ 9 LINH KIỆN (không cần lặp lại Mainboard, Nguồn, Case, Tản nhiệt).\n"
                "2. Xưng hô tự nhiên là 'mình' - 'bạn'.\n"
                "3. TUYỆT ĐỐI KHÔNG dùng bất kỳ emoji hay icon cảm xúc nào.\n"
            )

        gemini_res = call_gemini_api(gemini_prompt, api_key=active_key, timeout=8)
        if gemini_res and len(gemini_res.strip()) > 15:
            clean_res = re.sub(r'[\U00010000-\U0010ffff]', '', gemini_res).strip()
            return clean_res

    # CHẾ ĐỘ NỘI BỘ (FALLBACK) - VĂN PHONG TỰ NHIÊN NHƯ NGƯỜI THẬT
    if parsed.get("intent") == "chat":
        now_dt = datetime.now()
        days_vi = ["Thứ Hai", "Thứ Ba", "Thứ Tư", "Thứ Năm", "Thứ Sáu", "Thứ Bảy", "Chủ Nhật"]
        now_full_str = f"{days_vi[now_dt.weekday()]}, ngày {now_dt.strftime('%d/%m/%Y')}"

        t_low = user_query.lower().strip()
        clean_q = re.sub(r'[!?,.]+$', '', t_low).strip()

        # Câu hỏi về thời gian, ngày giờ, thứ, tháng, năm
        if any(w in clean_q for w in ["mấy giờ", "may gio", "bây giờ là mấy giờ", "thời gian", "mấy h", "may h"]):
            return f"Bây giờ là {now_dt.strftime('%H:%M')} ({now_full_str}) bạn nhé. Bạn cần mình tư vấn cấu hình PC hay hỗ trợ gì thêm không?"
        elif any(w in clean_q for w in ["hôm nay ngày mấy", "ngày bao nhiêu", "hôm nay là ngày", "ngay may", "ngay bao nhieu", "thứ mấy", "thu may"]):
            return f"Hôm nay là {days_vi[now_dt.weekday()]}, ngày {now_dt.strftime('%d/%m/%Y')} bạn nhé. Bạn đang muốn lên dàn PC tầm giá bao nhiêu?"
        elif any(w in clean_q for w in ["tháng mấy", "thang may"]):
            return f"Hiện tại đang là tháng {now_dt.month} năm {now_dt.year} bạn nhé. Bạn đang dự tính lắp máy phục vụ nhu cầu gì thế?"
        elif any(w in clean_q for w in ["năm mấy", "năm bao nhiêu", "nam may", "nam bao nhieu"]):
            return f"Năm nay là năm {now_dt.year} bạn nhé. Bạn cần tư vấn dàn máy nào nhắn mình nhé."
        elif any(w in clean_q for w in ["chào", "chao", "hi", "hello", "alo", "hé lô", "hai", "xin chào"]):
            return "Chào bạn! Mình sẵn sàng hỗ trợ rồi đây. Bạn đang định dựng PC tầm giá bao nhiêu hay cần mình tư vấn gì về linh kiện?"
        elif ("intel" in t_low or "inter" in t_low) and ("amd" in t_low or "amđ" in t_low):
            return "Cả Intel và AMD đều rất mạnh: Intel tối ưu tốt cho đa nhiệm và làm việc đồ họa, còn AMD cực kỳ tối ưu cho chơi game và tiết kiệm điện. Bạn đang định đầu tư tầm ngân sách bao nhiêu để mình tư vấn cấu hình phù hợp nhất nhé?"
        elif parsed.get("usage") or parsed.get("cpu_brand") or parsed.get("need_hdd"):
            reqs = []
            if parsed.get("usage"):
                reqs.append(f"nhu cầu {parsed['usage'].lower()}")
            if parsed.get("cpu_brand"):
                reqs.append(f"chip {parsed['cpu_brand']}")
            if parsed.get("need_hdd"):
                reqs.append("có cả SSD và HDD")
            req_str = " ".join(reqs)
            return f"Chào bạn! Mình đã ghi nhận bạn cần dàn máy {req_str}. Bạn dự tính ngân sách khoảng bao nhiêu để mình tối ưu cấu hình chuẩn nhất cho bạn nhé?"
        elif any(w in t_low for w in ["tư vấn", "giúp", "hỗ trợ"]):
            return "Chào bạn! Bạn chia sẻ giúp mình ngân sách dự kiến và nhu cầu sử dụng chính (chơi game, đồ họa, văn phòng...) để mình tư vấn cấu hình chuẩn nhất cho bạn nhé!"
        elif any(w in t_low for w in ["cảm ơn", "cam on", "thank"]):
            return "Không có gì đâu bạn! Nếu bạn cần tìm hiểu thêm hay muốn tinh chỉnh cấu hình nào, cứ nhắn cho mình nhé."
        elif any(w in t_low for w in ["khỏe không", "khoe khong"]):
            return "Mình khỏe, cảm ơn bạn nhé! Bạn đang muốn tìm hiểu linh kiện nào hay cần lên dàn PC tầm giá bao nhiêu để mình hỗ trợ ngay?"
        else:
            return "Chào bạn! Mình sẵn sàng hỗ trợ mọi thắc mắc về linh kiện và xây dựng cấu hình PC. Bạn đang muốn tìm hiểu về phần cứng nào hay cần lên dàn máy tầm giá bao nhiêu?"

    # Fallback cho TH tư vấn khi chưa có ngân sách cụ thể (chuẩn 5 dòng, dưới 60 từ)
    if parsed.get("intent") == "consult_no_budget":
        prof = parsed.get("app_game_profile")
        if not prof:
            u_key = "van_phong" if parsed.get("usage") == "Văn phòng / Học tập" else ("do_hoa" if parsed.get("usage") == "Đồ họa / Lập trình AI" else "gta")
            prof = GAME_APP_PROFILES.get(u_key, GAME_APP_PROFILES["gta"])

        line1 = f"{prof.get('analysis', f'{prof.get("name")} cần cấu hình cân đối để hoạt động mượt mà.')}:"
        tier_items = []
        for t in prof.get("tiers", []):
            label = t.get("label", f"Phương án {t.get('tier', 1)}")
            desc = t.get("desc", "").strip()
            tier_items.append(f"- **{label}:** {desc}")
        line_tiers = chr(10).join(tier_items)
        line5 = "Bạn chọn phương án nào để mình lên cấu hình chi tiết?"
        return f"{line1}\n{line_tiers}\n\n{line5}"

    # Fallback cho TH không đủ ngân sách tối thiểu (Trường hợp 1)
    if not build:
        return (
            f"Với {budget_tr} thì hơi căng để gánh trọn bộ {comp_str} (tối thiểu cần khoảng {min_needed_tr}). "
            f"Bạn muốn tăng ngân sách lên mức này hay đổi sang card đồ họa phân khúc thấp hơn để cấu hình cân đối?"
        )

    # Fallback cho TH nghẽn cổ chai nặng (Trường hợp 2)
    if bottleneck_detected:
        return (
            f"Mình đã cố gắng ép đủ {build.gpu['name']} và {storage_desc} vào đúng {budget_tr}, "
            f"nhưng ngân sách còn lại chỉ đủ lắp chip {build.cpu['name']} nên sẽ bị nghẽn cổ chai khá nặng. "
            f"Theo mình bạn nên cân nhắc đổi sang dòng card đồ họa phù hợp hơn để nâng CPU lên dòng cao hơn, giúp cấu hình cân đối và mượt mà hơn."
        )

    # Fallback cho cấu hình chuẩn cân đối - Rút gọn 80-110 từ tránh Wall of Text
    cpu_name = build.cpu.get("name", "CPU")
    gpu_name = build.gpu.get("name", "VGA")
    has_discrete_gpu = build.gpu and build.gpu.get("id") != "gpu_none"
    has_hdd = build.hdd and build.hdd.get("id") != "hdd_none"
    is_ssd_sata = "SATA" in build.storage.get("name", "").upper() or "SATA" in str(build.storage.get("interface", "")).upper()
    is_office = (usage == "Văn phòng / Học tập" or "văn phòng" in usage.lower() or "học tập" in usage.lower())
    is_creative = ("đồ họa" in usage.lower() or "ai" in usage.lower() or "lập trình" in usage.lower())

    # 1. Câu mở đầu ngắn chốt nhu cầu và tổng chi phí
    p_open = f"Đã dựng xong dàn **{usage}** tổng **{fmt_vnd(build.total_price)}** ở bảng bên trái:"

    # 2. Bắt buộc 4 linh kiện cốt lõi, mỗi linh kiện trên 1 dòng riêng biệt
    line_cpu = f"- **CPU:** {cpu_name}"
    line_vga = f"- **VGA:** {gpu_name}" if has_discrete_gpu else "- **VGA:** Dùng iGPU tích hợp"
    ram_type = build.ram.get("ram_type", "DDR4")
    line_ram = f"- **RAM:** {ram_cap}GB {ram_type}"

    if is_ssd_sata:
        ssd_short = f"SSD SATA {ssd_cap}GB"
    else:
        ssd_short = f"SSD NVMe {ssd_cap}GB"

    if has_hdd:
        hdd_cap = build.hdd.get("capacity_gb", 1000)
        hdd_short = f"HDD {hdd_cap // 1000}TB" if hdd_cap >= 1000 else f"HDD {hdd_cap}GB"
        line_storage = f"- **Lưu trữ:** {ssd_short} + {hdd_short}"
    else:
        line_storage = f"- **Lưu trữ:** {ssd_short}"

    comp_block = f"{line_cpu}\n{line_vga}\n{line_ram}\n{line_storage}"

    # 3. Nhận xét: 1 câu ngắn về hiệu năng thực tế
    if is_office:
        review_text = "**Nhận xét:** Vận hành êm ái, tối ưu cho xử lý văn phòng, mở nhiều tab trình duyệt và file Excel nặng mượt mà."
    elif is_creative:
        review_text = "**Nhận xét:** Hiệu năng mạnh mẽ, đáp ứng trơn tru các tác vụ thiết kế đồ họa, render video và đa nhiệm."
    else: # Gaming / Đa dụng
        review_text = "**Nhận xét:** Hiệu năng ấn tượng, chiến mượt mà các tựa game yêu thích ở thiết lập đồ họa cao và khung hình ổn định."

    # 4. Gợi ý nâng cấp: Chỉ hiện nếu ngân sách < 30tr và không phải mốc cao cấp; viết 1 câu ngắn đúng nhu cầu
    upgrade_note = ""
    if build.total_price < 30_000_000 and upgrade_suggestions:
        best_up = upgrade_suggestions[0]
        b_txt = best_up.get("benefit", "")
        if len(b_txt) > 1 and b_txt[0].isupper() and not b_txt[1].isupper():
            b_txt = b_txt[0].lower() + b_txt[1:]
        upgrade_note = f"**Gợi ý nâng cấp:** Thêm {fmt_vnd(best_up['added_cost'])}, nâng lên **{best_up['upgrade_name']}** giúp {b_txt}."

    warning_note = ""
    if parsed.get("unmatched_warnings"):
        warning_note = "*" + "; ".join(parsed["unmatched_warnings"]) + "*"

    # 5. Câu hỏi chốt ngắn 1 dòng
    p_close = "Bạn có muốn đổi linh kiện nào hay chỉnh ngân sách không?"

    parts_reply = [p_open, comp_block, review_text]
    if warning_note:
        parts_reply.append(warning_note)
    if upgrade_note:
        parts_reply.append(upgrade_note)
    parts_reply.append(p_close)

    return "\n\n".join(parts_reply)


def build_assistant_chat_response(
    parsed: Dict[str, Any],
    build: Optional[Any],
    review_text: str,
) -> str:
    """
    Sinh câu trả lời hoàn chỉnh của Trợ lý AI sau khi xử lý yêu cầu người dùng.
    """
    lines = []
    lines.append("Chào bạn, tôi đã phân tích yêu cầu cấu hình của bạn:")
    lines.append(f"- Tiêu chí trích xuất: {parsed['extracted_summary']}")

    if parsed.get("unmatched_warnings"):
        for warn in parsed["unmatched_warnings"]:
            lines.append(f"- *{warn}*")

    if build:
        lines.append(f"- Cấu hình tối ưu được đề xuất: Tổng chi phí **{fmt_vnd(build.total_price)}**.")
        cpu_name = build.cpu.get("name", "CPU")
        gpu_name = build.gpu.get("name", "VGA")
        ram_name = build.ram.get("name", "RAM")
        storage_name = build.storage.get("name", "SSD")
        lines.append(f"  - CPU & VGA: {cpu_name} kết hợp cùng {gpu_name}.")
        lines.append(f"  - Bộ nhớ & Lưu trữ: {ram_name} đi cùng {storage_name}.")
        lines.append("")
        lines.append("---")
        lines.append("### Nhận xét & Đánh giá chuyên môn:")
        lines.append(review_text)
    else:
        lines.append(
            "Rất tiếc, với mức ngân sách và các điều kiện ràng buộc bạn đưa ra, "
            "hệ thống chưa tìm thấy tổ hợp linh kiện hoàn toàn tương thích và đủ chi phí. "
            "Bạn vui lòng tăng nhẹ ngân sách hoặc nới lỏng bớt một số linh kiện cố định để tôi tìm kiếm lại nhé!"
        )

    return "\n".join(lines)
