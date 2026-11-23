"""
knowledge_base.py
=================
Module Cơ sở Tri thức (Knowledge Base)
Môn: Trí tuệ nhân tạo - Biểu diễn & Giải thuật (CS 417)

Triển khai:
  - Biểu diễn tri thức bằng Frames (dictionary có cấu trúc)
  - Tập luật suy diễn dạng IF-THEN (Production Rules R01 - R10)
  - Phân loại: Luật cứng (Hard Constraint) và Luật mềm (Soft Heuristic)
  - Không sử dụng ký tự biểu cảm (emoji) để đảm bảo giao diện chuyên nghiệp
"""

import math
from typing import Any, Dict, List, Optional, Tuple


# ===========================================================================
# PHẦN 1: BIỂU DIỄN TRI THỨC BẰNG FRAMES (TẬP LUẬT R01 - R10)
# ===========================================================================

RULES: List[Dict[str, str]] = [
    {
        "rule_id": "R01",
        "name": "Tương thích Socket CPU và Bo mạch chủ (Kèm giới hạn VRM/Chipset)",
        "category": "Hard",
        "condition_desc": "CPU được chọn sử dụng chuẩn Socket và mức công suất tiêu thụ nhất định",
        "action_desc": "Lọc và chỉ giữ lại Bo mạch chủ cùng chuẩn Socket; cấm ghép CPU Intel dòng K/TDP cao với H610 và CPU AMD Ryzen 9/TDP >= 105W với A520.",
        "logic_expr": "CPU.socket == Mainboard.socket AND NOT is_vrm_incompatible(CPU, Mainboard)",
    },
    {
        "rule_id": "R02",
        "name": "Tương thích chuẩn bộ nhớ RAM (DDR4 / DDR5)",
        "category": "Hard",
        "condition_desc": "CPU và Bo mạch chủ yêu cầu chuẩn bộ nhớ RAM cụ thể",
        "action_desc": "Lọc và chỉ giữ lại thanh RAM có cùng chuẩn ram_type với cả CPU và Bo mạch chủ.",
        "logic_expr": "CPU.ram_type == RAM.ram_type AND Mainboard.ram_type == RAM.ram_type",
    },
    {
        "rule_id": "R03",
        "name": "Giới hạn dung lượng RAM tối đa của Bo mạch chủ",
        "category": "Hard",
        "condition_desc": "Bo mạch chủ có thông số dung lượng RAM tối đa hỗ trợ (max_ram_gb)",
        "action_desc": "Loại bỏ các cấu hình có tổng dung lượng RAM vượt quá khả năng tiếp nhận của Bo mạch chủ.",
        "logic_expr": "RAM.capacity_gb <= Mainboard.max_ram_gb",
    },
    {
        "rule_id": "R04",
        "name": "Tương thích kích thước Bo mạch chủ và Vỏ máy (Form Factor)",
        "category": "Hard",
        "condition_desc": "Bo mạch chủ có kích thước chuẩn (ATX, mATX, ITX)",
        "action_desc": "Chỉ chấp nhận Vỏ máy có danh sách hỗ trợ chứa kích thước của Bo mạch chủ đã chọn.",
        "logic_expr": "Mainboard.form_factor IN Case.supported_form_factors",
    },
    {
        "rule_id": "R05",
        "name": "Giới hạn chiều dài Card đồ họa theo Vỏ máy",
        "category": "Hard",
        "condition_desc": "Card đồ họa rời có chiều dài vật lý (length_mm)",
        "action_desc": "Loại bỏ Vỏ máy có khoảng trống lắp card đồ họa nhỏ hơn chiều dài của Card đồ họa đã chọn.",
        "logic_expr": "GPU.length_mm <= Case.max_gpu_length_mm (nếu có card rời)",
    },
    {
        "rule_id": "R06",
        "name": "Công suất danh định của Bộ nguồn (PSU)",
        "category": "Hard",
        "condition_desc": "Hệ thống tính toán tổng công suất tiêu thụ tối đa của CPU và GPU",
        "action_desc": "Tính công suất tối thiểu = (CPU.tdp + GPU.tdp + 100W phụ tải) * 1.25 dự phòng; lọc giữ PSU có công suất >= mức này.",
        "logic_expr": "PSU.wattage >= ceil((CPU.tdp + GPU.tdp + 100) * 1.25)",
    },
    {
        "rule_id": "R07",
        "name": "Khả năng xuất tín hiệu hình ảnh hiển thị",
        "category": "Hard",
        "condition_desc": "Cấu hình không trang bị Card đồ họa rời (gpu_none)",
        "action_desc": "Bắt buộc CPU phải có nhân đồ họa tích hợp (has_igpu == True) để máy có thể xuất hình ảnh lên màn hình.",
        "logic_expr": "IF GPU.id == 'gpu_none' THEN CPU.has_igpu == True",
    },
    {
        "rule_id": "R08",
        "name": "Cân bằng hiệu năng CPU và Card đồ họa (Chống nghẽn cổ chai)",
        "category": "Soft",
        "condition_desc": "Hệ thống trang bị Card đồ họa rời và cả hai linh kiện đều có điểm hiệu năng",
        "action_desc": "Chuẩn hóa điểm hiệu năng về thang 0-100; cảnh báo nếu CPU quá yếu kìm hãm GPU hoặc GPU quá yếu so với CPU.",
        "logic_expr": "0.55 <= (CPU_score_norm / GPU_score_norm) <= 2.22",
    },
    {
        "rule_id": "R09",
        "name": "Trọng số tối ưu hóa và ngưỡng bão hòa theo nhu cầu",
        "category": "Soft",
        "condition_desc": "Người dùng lựa chọn mục đích sử dụng (Văn phòng, Gaming, Đồ họa - Lập trình)",
        "action_desc": "Áp dụng bộ trọng số chuyên biệt; áp dụng ngưỡng bão hòa cho RAM và ổ cứng để tránh phân bổ ngân sách lãng phí.",
        "logic_expr": "Fitness = w_cpu * CPU_norm + w_gpu * GPU_norm + w_ram * RAM_norm + w_storage * Storage_norm",
    },
    {
        "rule_id": "R10",
        "name": "Công suất giải nhiệt của Tản nhiệt (Cooler TDP)",
        "category": "Hard",
        "condition_desc": "CPU có mức tỏa nhiệt TDP và Tản nhiệt có chỉ số công suất làm mát tối đa (tdp_support)",
        "action_desc": "Bắt buộc tản nhiệt phải có tdp_support >= CPU.tdp. CPU có TDP > 65W không được sử dụng tản nhiệt mặc định (stock).",
        "logic_expr": "Cooler.tdp_support >= CPU.tdp (và CPU.tdp <= 65W nếu dùng cooler_stock)",
    },
    {
        "rule_id": "R11",
        "name": "Ưu tiên SSD chuẩn NVMe cho phân khúc từ 12 triệu trở lên",
        "category": "Soft",
        "condition_desc": "Ngân sách từ 12 triệu trở lên hoặc hệ thống đã có HDD phụ để lưu trữ dữ liệu lớn",
        "action_desc": "Bắt buộc ưu tiên SSD chuẩn NVMe tốc độ cao thay vì SSD SATA để tối ưu tốc độ đọc ghi, phản hồi ứng dụng và sao chép dữ liệu.",
        "logic_expr": "Budget >= 12_000_000 hoặc has_hdd => Storage.type == 'NVMe'",
    },
]

# ===========================================================================
# PHẦN 2: HỒ SƠ TRỌNG SỐ THEO NHU CẦU (Luật R09)
# ===========================================================================

USAGE_PROFILES: Dict[str, Dict[str, Any]] = {
    "Văn phòng / Học tập": {
        "w_cpu": 0.45,
        "w_gpu": 0.00,  # 0% mặc định (tối đa 5% nếu người dùng tự chọn VGA rời)
        "w_ram": 0.20,
        "w_storage": 0.15,
        "w_durability": 0.20,  # Độ bền & Êm ái (Main, PSU, Case, Tản) 20%
        "min_ram_gb": 16,
        "require_discrete_gpu": False,
        "prefer_igpu_only": True,
        "ram_saturation_gb": 16,
        "storage_saturation_gb": 512,
        "description": "CPU 45% | RAM & Lưu trữ 35% | Độ bền & Êm ái 20% | GPU 0% (tối đa 5% nếu bật VGA rời).",
    },
    "Gaming": {
        "w_cpu": 0.30,
        "w_gpu": 0.50,
        "w_ram": 0.10,
        "w_storage": 0.05,
        "w_durability": 0.05,  # Lưu trữ, Nguồn & Tản nhiệt 10% (Storage 5%, Nguồn & Tản 5%)
        "min_ram_gb": 16,
        "require_discrete_gpu": True,
        "prefer_igpu_only": False,
        "ram_saturation_gb": 32,
        "storage_saturation_gb": 1000,
        "description": "GPU 50% | CPU 30% (X3D / đơn nhân cao) | RAM 10% | Lưu trữ, Nguồn & Tản nhiệt 10%.",
    },
    "Đồ họa / Lập trình AI": {
        "w_cpu": 0.35,
        "w_gpu": 0.35,
        "w_ram": 0.20,
        "w_storage": 0.05,
        "w_durability": 0.05,  # Lưu trữ NVMe & Nguồn/Tản 10% (Storage 5%, Nguồn & Tản 5%)
        "min_ram_gb": 16,
        "require_discrete_gpu": True,
        "prefer_igpu_only": False,
        "ram_saturation_gb": 64,
        "storage_saturation_gb": 2000,
        "description": "CPU đa nhân 35% | GPU NVIDIA & VRAM 35% | RAM dung lượng lớn 20% | Lưu trữ NVMe & Nguồn/Tản 10%.",
    },
    "Tất cả / Đa dụng (Cân bằng)": {
        "w_cpu": 0.35,
        "w_gpu": 0.35,
        "w_ram": 0.15,
        "w_storage": 0.08,
        "w_durability": 0.07,  # Lưu trữ, Nguồn, Case & Tản 15% (Storage 8%, Nguồn, Case & Tản 7%)
        "min_ram_gb": 16,
        "require_discrete_gpu": True,
        "prefer_igpu_only": False,
        "ram_saturation_gb": 32,
        "storage_saturation_gb": 1000,
        "description": "CPU 35% | GPU 35% (VGA <= 45% ngân sách) | RAM 15% | Lưu trữ, Nguồn, Case & Tản 15%.",
    },
}

# ===========================================================================
# PHẦN 3: CÁC HÀM KIỂM TRA RÀNG BUỘC (R01 - R10)
# ===========================================================================

def is_vrm_incompatible(cpu: Optional[Dict], mainboard: Optional[Dict]) -> bool:
    """
    Ràng buộc VRM / Chipset:
    1. Cấm ghép CPU Intel dòng K (hoặc KF, KS), Core i7/i9 hoặc có TDP >= 105W
       với bo mạch chủ phổ thông chipset H410, H510, H610 hoặc A520 do mạch VRM không đủ tải.
    2. Cấm ghép CPU dòng Ryzen 9 hoặc CPU AMD có TDP >= 105W (Ryzen 7 5800X3D, Ryzen 7 7700X...)
       với bo mạch chủ chipset A520 (bắt buộc dùng từ dòng B550 / X570 / B650 trở lên).
    3. Cấm ghép CPU Intel thế hệ 11 (Rocket Lake) với bo mạch chủ chipset H410 do phần cứng không tương thích.
    """
    if not cpu or not mainboard:
        return False
    cpu_name = cpu.get("name", "").strip().upper()
    cpu_brand = cpu.get("brand", "").strip().upper()
    cpu_tdp = cpu.get("tdp", 0)
    mb_name = mainboard.get("name", "").upper()

    # Intel Thế hệ 11 cấm H410
    is_gen_11 = any(k in cpu_name for k in ["11400", "11700", "11900", "11600", "11500", "11TH"]) or "I5-11" in cpu_name or "I7-11" in cpu_name or "I9-11" in cpu_name
    if is_gen_11 and "H410" in mb_name:
        return True

    # Intel: dòng K / KF / KS, Core i7, Core i9 hoặc TDP >= 105W cấm H410 / H510 / H610 / A520
    is_k_series = (
        cpu_name.endswith("K")
        or cpu_name.endswith("KF")
        or cpu_name.endswith("KS")
        or " K " in cpu_name
        or " KF " in cpu_name
    )
    is_intel_i7_i9 = ("I7" in cpu_name or "I9" in cpu_name) and (cpu_brand == "INTEL" or "CORE" in cpu_name)
    is_intel_high_load = is_k_series or is_intel_i7_i9 or (cpu_tdp >= 105 and (cpu_brand == "INTEL" or "CORE" in cpu_name))
    is_entry_mb = any(chip in mb_name for chip in ["H410", "H510", "H610", "A520", "H310"])

    if is_intel_high_load and is_entry_mb:
        return True

    # AMD: Dòng Ryzen 9 hoặc CPU AMD có TDP >= 105W cấm A520 (bắt buộc B550 / X570 / B650 trở lên)
    is_amd = (cpu_brand == "AMD") or ("RYZEN" in cpu_name)
    is_ryzen_9 = "RYZEN 9" in cpu_name or "R9" in cpu_name
    if is_amd and (is_ryzen_9 or cpu_tdp >= 105) and "A520" in mb_name:
        return True

    return False


def is_pcie3_mainboard(mainboard: Dict) -> bool:
    """
    Kiểm tra bo mạch chủ đời cũ chỉ hỗ trợ chuẩn PCIe 3.0:
    B450, A520 (AM4); H410, H510, B460, H310 (LGA1200/LGA1151).
    Cấu hình từ 18 triệu trở lên bắt buộc dùng bo mạch chủ hỗ trợ PCIe 4.0 trở lên
    (như B550, B660, B760, B650, Z690, Z790).
    """
    if not mainboard:
        return False
    mb_name = mainboard.get("name", "").upper()
    pcie3_chipsets = ["B450", "A520", "H410", "H510", "B460", "H310"]
    return any(chip in mb_name for chip in pcie3_chipsets)


def check_r01_socket(cpu: Dict, mainboard: Dict) -> Tuple[bool, str]:
    """R01: Kiểm tra tương thích socket CPU và Bo mạch chủ kèm giới hạn VRM/Chipset."""
    if cpu["socket"] != mainboard["socket"]:
        return (
            False,
            f"[R01] Khong tuong thich Socket: CPU '{cpu['name']}' ({cpu['socket']}) "
            f"khong the lap vao Bo mach chu '{mainboard['name']}' ({mainboard['socket']}).",
        )
    if is_vrm_incompatible(cpu, mainboard):
        cpu_name = cpu.get("name", "").upper()
        mb_name = mainboard.get("name", "").upper()
        if any(k in cpu_name for k in ["11400", "11700", "11900", "11600", "11500"]) and "H410" in mb_name:
            return (
                False,
                f"[R01] Khong tuong thich Chipset: CPU Intel the he 11 '{cpu['name']}' "
                f"khong duoc ho tro boi Bo mach chu chipset H410 '{mainboard['name']}'. Bắt buộc dùng chipset H510/B560/Z590 trở lên.",
            )
        return (
            False,
            f"[R01] Khong tuong thich VRM / Chipset: Bo mach chu '{mainboard['name']}' (chipset H610/A520/H410) "
            f"khong du pha dien (VRM) de ho tro on dinh cho CPU '{cpu['name']}' (yeu cau chipset B550/B660/X570 tro len do TDP {cpu.get('tdp', 0)}W cao hoac thuoc dong K/Ryzen 9).",
        )
    return (True, f"[R01] Tuong thich Socket va VRM: {cpu['socket']}.")


def check_r02_ram_type(cpu: Dict, mainboard: Dict, ram: Dict) -> Tuple[bool, str]:
    """R02: Kiểm tra chuẩn RAM DDR4/DDR5."""
    if ram["ram_type"] != cpu["ram_type"]:
        return (
            False,
            f"[R02] Khong khop chuan RAM voi CPU: CPU '{cpu['name']}' yeu cau "
            f"'{cpu['ram_type']}' nhung RAM duoc chon la '{ram['ram_type']}'.",
        )
    if ram["ram_type"] != mainboard["ram_type"]:
        return (
            False,
            f"[R02] Khong khop chuan RAM voi Bo mach chu: Bo mach chu '{mainboard['name']}' "
            f"ho tro '{mainboard['ram_type']}' nhung RAM la '{ram['ram_type']}'.",
        )
    return (True, f"[R02] Tuong thich chuan RAM: {ram['ram_type']}.")


def check_r03_ram_capacity(mainboard: Dict, ram: Dict, ram_qty: int = 1) -> Tuple[bool, str]:
    """R03: Dung lượng RAM và số khe không vượt giới hạn Bo mạch chủ."""
    total_ram = ram["capacity_gb"] * ram_qty
    if total_ram > mainboard["max_ram_gb"]:
        return (
            False,
            f"[R03] Vượt giới hạn dung lượng RAM: Bo mạch chủ '{mainboard['name']}' "
            f"hỗ trợ tối đa {mainboard['max_ram_gb']}GB nhưng tổng dung lượng RAM ({ram['capacity_gb']}GB x {ram_qty}) là {total_ram}GB.",
        )
    if ram_qty > 4:
        return (
            False,
            f"[R03] Vượt quá số khe RAM tối đa: Bo mạch chủ hỗ trợ tối đa 4 khe cắm nhưng bạn đã chọn {ram_qty} thanh.",
        )
    return (True, f"[R03] Dung lượng RAM {total_ram}GB ({ram_qty} thanh) hợp lệ.")


def check_r04_form_factor(mainboard: Dict, case: Dict) -> Tuple[bool, str]:
    """R04: Kích thước Bo mạch chủ phù hợp Vỏ máy."""
    if mainboard["form_factor"] not in case["supported_form_factors"]:
        return (
            False,
            f"[R04] Khong vua kich thuoc Vo may: Bo mach chu '{mainboard['name']}' "
            f"chuan {mainboard['form_factor']} khong vua Vo may '{case['name']}' "
            f"(ho tro: {', '.join(case['supported_form_factors'])}).",
        )
    return (True, f"[R04] Bo mach chu {mainboard['form_factor']} lap vua Vo may.")


def check_r05_gpu_length(gpu: Dict, case: Dict) -> Tuple[bool, str]:
    """R05: Chiều dài card đồ họa không vượt kích thước Vỏ máy."""
    if gpu["id"] == "gpu_none":
        return (True, "[R05] Khong su dung card do hoa roi.")
    if gpu["length_mm"] > case["max_gpu_length_mm"]:
        return (
            False,
            f"[R05] Card do hoa qua dai: Card '{gpu['name']}' dai {gpu['length_mm']}mm "
            f"vuot qua chieu dai toi da {case['max_gpu_length_mm']}mm cua Vo may '{case['name']}'.",
        )
    return (True, f"[R05] Card do hoa {gpu['length_mm']}mm vua van trong Vo may.")


def check_r06_psu_wattage(cpu: Dict, gpu: Dict, psu: Dict) -> Tuple[bool, str]:
    """R06: Công suất PSU đủ cấp cho CPU, GPU và phụ tải."""
    min_wattage = math.ceil((cpu["tdp"] + gpu["tdp"] + 100) * 1.25)
    if psu["wattage"] < min_wattage:
        return (
            False,
            f"[R06] Nguon PSU thieu cong suat: Tong cong suat yeu cau la {min_wattage}W "
            f"nhung Nguon '{psu['name']}' chi dat {psu['wattage']}W.",
        )
    return (True, f"[R06] Nguon PSU {psu['wattage']}W dap ung du yeu cau ({min_wattage}W).")


def check_r07_display_output(cpu: Dict, gpu: Dict) -> Tuple[bool, str]:
    """R07: Khả năng xuất tín hiệu hình ảnh."""
    if gpu["id"] == "gpu_none" and not cpu["has_igpu"]:
        return (
            False,
            f"[R07] Khong co tin hieu hinh anh: CPU '{cpu['name']}' khong tich hop do hoa "
            f"va he thong khong co card roi. May se khong the xuat hinh anh.",
        )
    return (True, "[R07] He thong co kha nang xuat tin hieu hinh anh.")


def check_r08_bottleneck(
    cpu: Dict, gpu: Dict, cpu_max_score: int = 50000, gpu_max_score: int = 40000
) -> Tuple[bool, str, str]:
    """R08: Cảnh báo nghẽn cổ chai CPU và GPU."""
    if gpu["id"] == "gpu_none":
        return (True, "ok", "[R08] Khong ap dung kiem tra nghen co chai do khong dung card roi.")

    cpu_norm = (cpu["score"] / cpu_max_score) * 100
    gpu_norm = (gpu["score"] / gpu_max_score) * 100

    if gpu_norm == 0:
        return (True, "ok", "[R08] Bo qua kiem tra nghen co chai.")

    ratio_cpu_gpu = cpu_norm / gpu_norm

    # Ngưỡng nghẽn cổ chai nặng: Chỉ cảnh báo khi CPU thực sự quá yếu (như Core i3, Ryzen 3)
    # kìm hãm card đồ họa tầm trung/cao cấp (tỷ lệ chuẩn hóa < 0.38).
    # Các dòng vi xử lý như i5-13500, i5-13400F, Ryzen 5 5600X đi cùng RTX 4070 / RTX 5060 / RTX 5070
    # đều đạt tỷ lệ >= 0.40 và được tính là cân đối tốt cho Gaming.
    if ratio_cpu_gpu < 0.38:
        return (
            False,
            "cpu_bottleneck",
            f"[R08] Canh bao nghen co chai: CPU '{cpu['name']}' ({cpu_norm:.1f}/100) "
            f"yeu hon dang ke so voi Card do hoa '{gpu['name']}' ({gpu_norm:.1f}/100).",
        )
    elif (gpu_norm / cpu_norm) < 0.45:
        return (
            False,
            "gpu_weak",
            f"[R08] Canh bao card do hoa qua yeu: Card do hoa '{gpu['name']}' ({gpu_norm:.1f}/100) "
            f"chua tuong xung voi CPU '{cpu['name']}' ({cpu_norm:.1f}/100).",
        )
    return (True, "ok", f"[R08] Can bang hieu nang CPU va GPU tot.")


def check_r10_cooler_tdp(cpu: Dict, cooler: Dict) -> Tuple[bool, str]:
    """R10: Khả năng tản nhiệt của Cooler đối với CPU."""
    if cooler is None or cooler.get("id") == "cooler_stock":
        if cpu["tdp"] > 65:
            return (
                False,
                f"[R10] Tan nhiet khong du cong suat: CPU '{cpu['name']}' co TDP {cpu['tdp']}W "
                f"khong the dung tan nhiet mac dinh (chi dap ung <=65W). Can trang bi tan roi.",
            )
        return (True, f"[R10] Tan nhiet mac dinh phu hop voi CPU {cpu['tdp']}W.")

    if cooler.get("tdp_support", 0) < cpu["tdp"]:
        return (
            False,
            f"[R10] Tan nhiet yeu: Tan nhiet '{cooler['name']}' ho tro toi da "
            f"{cooler['tdp_support']}W nhung CPU '{cpu['name']}' yeu cau {cpu['tdp']}W.",
        )
    return (True, f"[R10] Tan nhiet '{cooler['name']}' ({cooler['tdp_support']}W) lam mat tot cho CPU ({cpu['tdp']}W).")


def calculate_min_psu_wattage(cpu: Dict, gpu: Dict, extra_storage_count: int = 0) -> int:
    """Tính công suất PSU tối thiểu theo luật R06 (cộng thêm 10W cho mỗi ổ cứng phụ/gắn thêm)."""
    extra_w = max(0, extra_storage_count) * 10
    return math.ceil((cpu["tdp"] + gpu["tdp"] + 100 + extra_w) * 1.25)


def calculate_tier_min_psu_wattage(
    cpu: Dict,
    gpu: Dict,
    budget: float = 0,
    extra_storage_count: int = 0,
    usage: str = ""
) -> int:
    """
    Tính công suất PSU tối thiểu theo luật tương xứng tải & phân khúc an toàn:
    - Cơ sở tính: (cpu.tdp + gpu.tdp + 100 + extra_w) * 1.25.
    - Không bao giờ dùng nguồn < 500W cho hệ thống có card rời tầm trung trở lên (TDP >= 115W)
      hoặc CPU hiệu năng cao (TDP >= 105W, Ryzen 7/9, Core i7/i9).
    - Phân khúc theo 4 Nhu cầu:
      + Văn phòng / Học tập: < 12M (450W - 500W), 12 - 22M (500W - 550W), > 22M (550W).
      + Gaming: < 15M (500W - 550W), 15 - 30M (550W - 650W), > 30M (650W - 750W).
      + Đồ họa / Lập trình AI: < 18M (550W), 18 - 32M (650W), > 32M (750W+).
      + Tất cả / Đa dụng: < 16M (500W - 550W), 16 - 28M (550W - 650W), > 28M (650W - 750W).
    """
    base_w = calculate_min_psu_wattage(cpu, gpu, extra_storage_count=extra_storage_count)
    cpu_tdp = cpu.get("tdp", 0)
    cpu_name = cpu.get("name", "").upper()
    gpu_id = gpu.get("id", "")
    gpu_name = gpu.get("name", "").upper()
    gpu_tdp = gpu.get("tdp", 0)

    is_high_cpu = (
        cpu_tdp >= 105
        or "I7" in cpu_name
        or "I9" in cpu_name
        or "RYZEN 7" in cpu_name
        or "RYZEN 9" in cpu_name
        or cpu_name.endswith("K")
        or cpu_name.endswith("KF")
    )
    is_mid_gpu = (gpu_id != "gpu_none" and gpu_tdp >= 115)

    floor_w = 400
    if is_mid_gpu or is_high_cpu:
        floor_w = max(floor_w, 500)

    # Văn phòng: 12-22M tối thiểu 500W, >22M tối thiểu 550W
    if "VĂN PHÒNG" in usage.upper():
        if budget >= 22_000_000:
            floor_w = max(floor_w, 550)
        elif budget >= 12_000_000:
            floor_w = max(floor_w, 500)
        else:
            floor_w = max(floor_w, 450)

    # Gaming: <15M (500W), 15-30M (550W-650W), >30M (650W-750W)
    elif "GAMING" in usage.upper():
        if budget > 30_000_000:
            floor_w = max(floor_w, 650)
        elif budget >= 15_000_000:
            floor_w = max(floor_w, 550)
        else:
            floor_w = max(floor_w, 500)

    # Đồ họa / Lập trình AI: <18M (550W), 18-32M (650W), >32M (750W)
    elif "ĐỒ HỌA" in usage.upper() or "AI" in usage.upper():
        if budget > 32_000_000:
            floor_w = max(floor_w, 750)
        elif budget >= 18_000_000:
            floor_w = max(floor_w, 650)
        else:
            floor_w = max(floor_w, 550)

    # Tất cả / Đa dụng: <16M (500W), 16-28M (550W), >28M (650W)
    elif "TẤT CẢ" in usage.upper() or "ĐA DỤNG" in usage.upper():
        if budget > 28_000_000:
            floor_w = max(floor_w, 650)
        elif budget >= 16_000_000:
            floor_w = max(floor_w, 550)
        else:
            floor_w = max(floor_w, 500)

    is_25m_gpu = any(g in gpu_name for g in ["4060", "5060", "7600", "3060", "6700", "4060 TI", "5060 TI"])
    if is_25m_gpu:
        floor_w = max(floor_w, 550)

    is_35m_gpu = any(g in gpu_name for g in ["4070", "5070", "4080", "4090", "7800", "7900"])
    if is_35m_gpu or budget >= 35_000_000:
        floor_w = max(floor_w, 650)

    is_extreme_gpu = any(g in gpu_name for g in ["4070 TI", "5070 TI", "4080", "4090", "7900", "7900 XT", "7900 XTX"])
    if is_extreme_gpu:
        floor_w = max(floor_w, 750)

    return max(base_w, floor_w)


def compute_ram_norm(ram: Dict, usage: str, budget: int = 0, ram_qty: int = 1) -> float:
    """
    Tính điểm chuẩn hóa của RAM (0-100) theo quy tắc Dual Channel và ngưỡng bão hòa:
    - Ưu tiên tuyệt đối RAM Kit 2 thanh (Dual Channel: 2x8GB, 2x16GB, 2x32GB) hơn 1 thanh đơn.
    - Văn phòng / Học tập, Gaming, Đa dụng (Cân bằng):
      + Ngân sách <= 22 triệu: Khóa trần điểm RAM ở mức 16GB (2x8GB), không tự nhảy lên 32GB để dồn tiền cho CPU/GPU.
      + Ngân sách > 22 triệu: Khóa trần điểm RAM ở mức 32GB (2x16GB), không tự nhảy lên 64GB.
    - Đồ họa / Lập trình AI:
      + < 15 triệu: Tối ưu 16GB (2x8GB).
      + 15 - 30 triệu: Tối ưu 32GB (2x16GB).
      + >= 30 triệu: Tối ưu 64GB (2x32GB).
    """
    is_dual = (
        ram.get("is_kit", False)
        or ram.get("sticks", 1) >= 2
        or "2x" in ram.get("name", "").lower()
        or ram_qty >= 2
    )
    ram_cap = ram.get("capacity_gb", 8) * (ram_qty if (ram_qty >= 2 and not ("2x" in ram.get("name", "").lower() or ram.get("is_kit", False))) else 1)

    if usage in ["Văn phòng / Học tập", "Gaming", "Tất cả / Đa dụng (Cân bằng)"]:
        if budget > 0 and budget <= 22_000_000:
            # Ngân sách <= 22 triệu: Khóa trần ở 16GB (2x8GB)
            if ram_cap == 16:
                base_norm = 100.0
            elif ram_cap > 16:
                # Phạt mua thừa RAM ở tầm giá phổ thông / tầm trung
                base_norm = 85.0
            else:
                base_norm = (ram_cap / 16.0) * 70.0
        elif budget > 22_000_000:
            # Ngân sách > 22 triệu: Khóa trần ở 32GB (2x16GB)
            if ram_cap == 32:
                base_norm = 100.0
            elif ram_cap > 32:
                # Phạt mua thừa 64GB RAM cho Gaming/Văn phòng
                base_norm = 80.0
            elif ram_cap == 16:
                base_norm = 88.0
            else:
                base_norm = (ram_cap / 32.0) * 55.0
        else:
            # budget == 0 (mặc định)
            if usage == "Văn phòng / Học tập":
                base_norm = 100.0 if ram_cap == 16 else (95.0 if ram_cap == 32 else 75.0)
            else:
                base_norm = 100.0 if ram_cap == 32 else (90.0 if ram_cap == 16 else 80.0)

    elif usage == "Đồ họa / Lập trình AI":
        if budget > 0 and budget < 15_000_000:
            base_norm = 100.0 if ram_cap == 16 else (90.0 if ram_cap == 32 else 70.0)
        elif budget >= 30_000_000 or budget == 0:
            # >= 30 triệu: Tối ưu 64GB (2x32GB)
            if ram_cap >= 64:
                base_norm = 100.0
            elif ram_cap >= 32:
                base_norm = 90.0
            else:
                base_norm = 70.0
        else:
            # 15 - 30 triệu: Tối ưu 32GB (2x16GB)
            if ram_cap == 32:
                base_norm = 100.0
            elif ram_cap > 32:
                base_norm = 92.0
            elif ram_cap == 16:
                base_norm = 80.0
            else:
                base_norm = 50.0
    else:
        base_norm = min(100.0, (ram_cap / 32.0) * 100.0)

    # Ưu tiên tuyệt đối RAM Kit 2 thanh (Dual Channel) hơn 1 thanh đơn
    if is_dual:
        ram_norm = min(100.0, base_norm + 5.0)
    else:
        ram_norm = base_norm * 0.75

    return round(ram_norm, 2)


def compute_gpu_norm(gpu: Dict, usage: str, gpu_max_score: int = 40000) -> float:
    """Tính điểm chuẩn hóa của Card đồ họa (0-100) theo từng nhu cầu."""
    gpu_norm = min((gpu["score"] / gpu_max_score) * 100, 100.0)
    gpu_l = gpu.get("name", "").lower()
    if usage == "Gaming":
        gpu_norm = min(100.0, gpu_norm * 1.05)
    elif usage == "Đồ họa / Lập trình AI":
        is_nv = "nvidia" in gpu_l or "geforce" in gpu_l or "rtx" in gpu_l
        vram = gpu.get("vram_gb", 0)
        if is_nv:
            cuda_bonus = 10.0
            if vram >= 8:
                cuda_bonus = 15.0
            if vram >= 12:
                cuda_bonus = 25.0 if vram >= 16 else 20.0
            gpu_norm = min(100.0, gpu_norm + cuda_bonus)
    elif usage == "Văn phòng / Học tập":
        gpu_norm = min(30.0, gpu_norm)
    return round(gpu_norm, 2)


def compute_storage_norm(
    storage: Dict,
    usage: str,
    budget: int = 0,
    need_hdd: bool = False,
    hdd: Optional[Dict] = None,
) -> float:
    """
    Tính điểm chuẩn hóa của ổ cứng SSD và phối hợp với HDD theo từng nhu cầu:
    - Gaming & Lập trình AI: Ưu tiên tuyệt đối SSD dung lượng lớn (1TB - 2TB NVMe). Nếu có tích chọn HDD
      thì HDD chỉ chọn cỡ ~1/3 so với SSD (HDD 500GB) để không chiếm dụng ngân sách của SSD và GPU.
    - Tất cả / Đa dụng (Cân bằng): Tỷ lệ cân bằng 50/50 giữa SSD và HDD (SSD 500GB + HDD 500GB, hoặc SSD 1TB + HDD 1TB).
    - Văn phòng / Học tập: Giữ nguyên (SSD chính 256GB - 512GB cài Win + HDD 1TB - 2TB lưu trữ dữ liệu).
    """
    storage_cap = storage.get("capacity_gb", 256)
    hdd_cap = hdd.get("capacity_gb", 0) if (hdd and need_hdd) else 0
    st_type = storage.get("type", "NVMe")
    is_sata = (st_type == "SATA")

    # Luật R11: Siết ưu tiên SSD NVMe khi budget >= 12.000.000 VNĐ hoặc có HDD
    nvme_mod = 0.0
    if budget >= 12_000_000 or (need_hdd and hdd and hdd.get("id") != "hdd_none"):
        if is_sata:
            nvme_mod = -40.0
        else:
            nvme_mod = 10.0

    if usage in ["Gaming", "Đồ họa / Lập trình AI"]:
        # Gaming & AI: Luôn ưu tiên SSD dung lượng cao
        if storage_cap >= 2000:
            ssd_score = 100.0
        elif storage_cap >= 1000:
            ssd_score = 100.0 if budget < 35_000_000 else 90.0
        elif storage_cap >= 500:
            ssd_score = 80.0 if budget < 18_000_000 else 40.0
        else:
            ssd_score = 40.0 if budget < 15_000_000 else 10.0

        if need_hdd and hdd_cap > 0:
            # Nếu HDD cỡ ~1/3 hoặc nhỏ hơn SSD (hdd_cap <= storage_cap * 0.6): thưởng điểm tỷ lệ chuẩn
            if hdd_cap <= storage_cap * 0.6:
                return min(100.0, max(5.0, round(ssd_score + 10.0 + nvme_mod, 2)))
            else:
                # Nếu HDD lớn lấn át SSD: trừ điểm nặng vì sai tôn chỉ ưu tiên SSD của Gaming/AI
                return min(100.0, max(5.0, round(ssd_score - 30.0 + nvme_mod, 2)))
        return min(100.0, max(5.0, round(ssd_score + nvme_mod, 2)))

    elif usage == "Tất cả / Đa dụng (Cân bằng)":
        if need_hdd and hdd_cap > 0:
            # Tỷ lệ 50/50: SSD và HDD tương đương nhau
            ratio = min(storage_cap, hdd_cap) / max(storage_cap, hdd_cap)
            base_score = min(100.0, (storage_cap / 1000.0) * 80.0 + 20.0)
            return min(100.0, max(5.0, round(base_score * ratio + nvme_mod, 2)))
        else:
            target_cap = 1000 if budget <= 28_000_000 else 2000
            base_score = min(100.0, (storage_cap / float(target_cap)) * 100.0)
            return min(100.0, max(5.0, round(base_score + nvme_mod, 2)))

    else:  # Văn phòng / Học tập
        if need_hdd:
            if storage_cap <= 512:
                base_score = min(100.0, (storage_cap / 512.0) * 100.0)
            else:
                if budget > 30_000_000 and storage_cap == 1000:
                    base_score = 100.0
                else:
                    base_score = 90.0
            return min(100.0, max(5.0, round(base_score + nvme_mod, 2)))
        else:
            target_cap = 1000 if budget < 28_000_000 else 2000
            base_score = min(100.0, (storage_cap / float(target_cap)) * 100.0)
            return min(100.0, max(5.0, round(base_score + nvme_mod, 2)))


def filter_gpus_by_budget_floor(
    gpus: List[Dict],
    budget: int,
    usage: str,
    need_discrete_gpu: bool,
) -> List[Dict]:
    """
    Phân tách sàn VGA rõ ràng cho 4 Nhu cầu x 3 Phân khúc ngân sách:
    - Văn phòng / Học tập (need_discrete_gpu = False): Bắt buộc dùng gpu_none (0đ).
    - Gaming:
      + < 15 triệu: GTX 1650 / 1660 Super / RX 580 / RTX 2060.
      + 15 - 30 triệu: Cấm VGA đời cũ dưới RTX 2060. Từ 20 triệu trở lên tối thiểu RTX 3060 / RX 7600 / RTX 4060 / RTX 5060.
      + > 30 triệu: Tối thiểu RTX 4070 / RTX 4070 Ti / RTX 5070.
    - Đồ họa / Lập trình AI:
      + Ưu tiên GPU NVIDIA (nhân CUDA, VRAM lớn).
      + < 18 triệu: RTX 2060 / RTX 3060 12GB.
      + 18 - 32 triệu: RTX 3060 12GB / RTX 4060 Ti / RTX 5060.
      + > 32 triệu: RTX 4070 / 4070 Ti / 5070 12GB+ / 4080 / 4090.
    - Tất cả / Đa dụng (Cân bằng):
      + Phân bổ hài hòa, không dồn quá 45% ngân sách cho riêng VGA.
    """
    if not need_discrete_gpu:
        return [g for g in gpus if g.get("id") == "gpu_none"]

    real_gpus = [g for g in gpus if g.get("id") != "gpu_none"]
    if not real_gpus:
        return gpus

    if usage == "Gaming":
        if budget < 15_000_000:
            valid = [g for g in real_gpus if g.get("id") != "gpu_gt1030"]
            return valid if valid else real_gpus

        elif 15_000_000 <= budget < 30_000_000:
            forbidden = ["gt 1030", "1050 ti", "1650", "rx 580", "6500 xt"]
            if budget >= 20_000_000:
                valid = [
                    g for g in real_gpus
                    if not any(kw in g.get("name", "").lower() for kw in forbidden)
                    and g.get("score", 0) >= 14000
                ]
            else:
                valid = [
                    g for g in real_gpus
                    if not any(kw in g.get("name", "").lower() for kw in forbidden)
                    and g.get("score", 0) >= 11500
                ]
            return valid if valid else real_gpus

        else:  # budget >= 30_000_000
            valid = [
                g for g in real_gpus
                if (
                    any(kw in g.get("name", "").lower() for kw in ["4070", "5070", "4080", "4090", "7800", "7900"])
                    or g.get("score", 0) >= 25000
                )
                and g.get("price", 0) >= 14_000_000
            ]
            if not valid:
                valid = [g for g in real_gpus if g.get("score", 0) >= 23000]
            return valid if valid else real_gpus

    elif usage == "Đồ họa / Lập trình AI":
        nv_gpus = [g for g in real_gpus if any(k in g.get("name", "").lower() for k in ["rtx", "geforce", "gtx", "nvidia"])]
        pool = nv_gpus if nv_gpus else real_gpus

        if budget < 15_000_000:
            valid = [g for g in pool if g.get("id") != "gpu_gt1030"]
            return valid if valid else pool
        elif 15_000_000 <= budget < 18_000_000:
            valid = [g for g in pool if g.get("score", 0) >= 11500 or g.get("vram_gb", 0) >= 6]
            return valid if valid else pool
        elif 18_000_000 <= budget <= 32_000_000:
            valid = [
                g for g in pool
                if g.get("score", 0) >= 14000
                and g.get("vram_gb", 0) >= 8
            ]
            return valid if valid else pool
        else:  # budget > 32_000_000
            valid = [
                g for g in pool
                if (
                    any(kw in g.get("name", "").lower() for kw in ["4070", "5070", "4080", "4090"])
                    or g.get("score", 0) >= 25000
                )
                and g.get("vram_gb", 0) >= 12
                and g.get("price", 0) >= 14_000_000
            ]
            if not valid:
                valid = [g for g in pool if g.get("score", 0) >= 20000]
            return valid if valid else pool

    elif usage == "Tất cả / Đa dụng (Cân bằng)":
        max_gpu_price = int(budget * 0.45) if budget >= 15_000_000 else budget
        if budget < 16_000_000:
            valid = [g for g in real_gpus if g.get("id") != "gpu_gt1030" and g.get("price", 0) <= max_gpu_price]
        elif 16_000_000 <= budget <= 28_000_000:
            forbidden = ["gt 1030", "1050 ti", "1650", "rx 580", "6500 xt"]
            valid = [
                g for g in real_gpus
                if not any(kw in g.get("name", "").lower() for kw in forbidden)
                and g.get("score", 0) >= 11500
                and g.get("price", 0) <= max_gpu_price
            ]
        else:  # budget > 28_000_000
            valid = [
                g for g in real_gpus
                if g.get("score", 0) >= 20000
                and g.get("price", 0) <= max_gpu_price
            ]
        return valid if valid else real_gpus

    else:
        return real_gpus


def compute_durability_norm(
    mainboard: Optional[Dict] = None,
    psu: Optional[Dict] = None,
    case: Optional[Dict] = None,
    cooler: Optional[Dict] = None,
    usage: str = "",
    budget: int = 0,
    gpu: Optional[Dict] = None,
    cpu: Optional[Dict] = None,
) -> float:
    """Tính điểm chuẩn hóa chất lượng phần cứng kèm theo (Mainboard, PSU, Case, Cooler)."""
    if not mainboard and not psu and not case and not cooler:
        return 85.0

    scores = []
    if mainboard:
        mb_name = mainboard.get("name", "").upper()
        mb_price = mainboard.get("price", 0)
        # Luật 4: Tránh để Mainboard chiếm quá ~22% tổng ngân sách ở hồ sơ Văn phòng
        if usage == "Văn phòng / Học tập" and budget > 0 and mb_price > budget * 0.22:
            scores.append(65.0)
        elif any(c in mb_name for c in ["Z790", "Z890", "X670", "X570"]):
            scores.append(100.0)
        elif any(c in mb_name for c in ["B760", "B650", "B550", "B660"]):
            scores.append(90.0)
        elif any(c in mb_name for c in ["B450", "B560", "B460"]):
            scores.append(80.0)
        else:
            scores.append(70.0)

    if psu:
        eff = psu.get("efficiency", "")
        psu_watt = psu.get("wattage", 0)
        psu_price = psu.get("price", 0)
        # Luật 2: Chặn lãng phí công suất Nguồn (PSU) khi không dùng VGA rời
        is_no_gpu = (gpu is not None and gpu.get("id") == "gpu_none") or (
            usage == "Văn phòng / Học tập" and (not gpu or gpu.get("id") == "gpu_none")
        )
        if is_no_gpu and (psu_watt >= 600 or psu_price > 1_100_000):
            scores.append(50.0)  # Phạt nặng nguồn thừa thãi khi không có VGA rời
        elif "Gold" in eff or "Platinum" in eff:
            scores.append(100.0)
        elif "Bronze" in eff:
            scores.append(85.0)
        else:
            scores.append(70.0)

    if case:
        case_price = case.get("price", 0)
        if case_price >= 800_000:
            scores.append(100.0)
        elif case_price >= 500_000:
            scores.append(85.0)
        else:
            scores.append(70.0)

    if cooler:
        cid = cooler.get("id", "")
        cprice = cooler.get("price", 0)
        tdp_sup = cooler.get("tdp_support", 0)
        is_office_mid_cpu = (
            usage == "Văn phòng / Học tập"
            and budget >= 13_000_000
            and cpu is not None
            and (
                cpu.get("price", 0) >= 4_500_000
                or cpu.get("score", 0) >= 18_000
                or cpu.get("cores", 0) >= 10
                or any(k in cpu.get("name", "").upper() for k in ["13500", "13600", "14500", "14600", "12700", "10700"])
            )
        )
        if is_office_mid_cpu:
            if cid == "cooler_stock":
                scores.append(40.0)  # Trừ điểm tản Stock 0đ khi CPU tầm trung / ngân sách từ 13M
            elif cprice <= 800_000:
                scores.append(100.0)  # Thưởng tối đa cho tản khí rời phổ thông (400k-700k) êm ái
            else:
                scores.append(85.0)
        elif cid != "cooler_stock" and tdp_sup >= 200:
            scores.append(100.0)
        elif cid != "cooler_stock":
            scores.append(90.0)
        else:
            scores.append(70.0)

    return sum(scores) / len(scores) if scores else 85.0


def compute_fitness_score(
    cpu: Dict,
    gpu: Dict,
    ram: Dict,
    storage: Dict,
    usage: str,
    ram_qty: int = 1,
    cpu_max_score: int = 50000,
    gpu_max_score: int = 40000,
    budget: int = 0,
    need_hdd: bool = False,
    mainboard: Optional[Dict] = None,
    psu: Optional[Dict] = None,
    case: Optional[Dict] = None,
    cooler: Optional[Dict] = None,
    hdd: Optional[Dict] = None,
) -> float:
    """
    Tính điểm tổng hợp cấu hình (Luật R09).
    Áp dụng trọng số và ngưỡng bão hòa theo 4 Nhu cầu chuẩn:
    1. Văn phòng & Học tập: CPU 45% | RAM & Lưu trữ 35% | Độ bền & Êm ái 20% | GPU 0% (tối đa 5% nếu dùng VGA rời).
    2. Gaming: GPU 50% | CPU 30% | RAM 10% | Lưu trữ, Nguồn & Tản nhiệt 10%.
    3. Đồ họa / Lập trình AI: CPU đa nhân 35% | GPU NVIDIA & VRAM 35% | RAM dung lượng lớn 20% | Lưu trữ NVMe & Nguồn/Tản 10%.
    4. Tất cả / Đa dụng (Cân bằng): CPU 35% | GPU 35% | RAM 15% | Lưu trữ, Nguồn, Case & Tản 15%.
    """
    profile = USAGE_PROFILES.get(usage, USAGE_PROFILES["Tất cả / Đa dụng (Cân bằng)"])

    # 1. Điểm CPU (chuẩn hóa về 0-100)
    cpu_norm = min((cpu["score"] / cpu_max_score) * 100, 100.0)
    cpu_l = cpu.get("name", "").lower()

    if usage == "Gaming":
        if "x3d" in cpu_l:
            cpu_norm = min(100.0, cpu_norm + 15.0)
        elif any(k in cpu_l for k in ["13600k", "14600k", "13700k", "14700k", "7600x", "9600x"]):
            cpu_norm = min(100.0, cpu_norm + 8.0)
        if budget >= 18_000_000 and cpu.get("socket") == "LGA1200":
            cpu_norm = max(20.0, cpu_norm - 15.0)

    elif usage == "Đồ họa / Lập trình AI":
        if any(k in cpu_l for k in ["5950x", "13900k", "14900k", "7950x", "9900x"]):
            cpu_norm = min(100.0, cpu_norm + 15.0)
        elif any(k in cpu_l for k in ["5700x", "3700x", "13500", "13600k", "14600k", "5900x", "7700x", "9700x", "12700"]):
            cpu_norm = min(100.0, cpu_norm + 10.0)

    # 2. Điểm GPU (chuẩn hóa về 0-100)
    gpu_norm = compute_gpu_norm(gpu, usage, gpu_max_score=gpu_max_score)

    # 3. Điểm RAM (áp dụng ngưỡng bão hòa và ưu tiên Dual Channel)
    ram_norm = compute_ram_norm(ram, usage, budget=budget, ram_qty=ram_qty)

    # 4. Điểm Ổ cứng SSD (phối hợp với HDD)
    storage_norm = compute_storage_norm(storage, usage, budget=budget, need_hdd=need_hdd, hdd=hdd)

    # 5. Điểm Độ bền & Linh kiện kèm theo (Main, PSU, Case, Cooler)
    durability_norm = compute_durability_norm(
        mainboard=mainboard,
        psu=psu,
        case=case,
        cooler=cooler,
        usage=usage,
        budget=budget,
        gpu=gpu,
        cpu=cpu,
    )

    # Phân bổ trọng số
    if usage == "Văn phòng / Học tập":
        if gpu.get("id") != "gpu_none":
            w_cpu = 0.45
            w_ram = 0.20
            w_storage = 0.15
            w_dur = 0.15
            w_gpu = 0.05
        else:
            w_cpu = 0.45
            w_ram = 0.20
            w_storage = 0.15
            w_dur = 0.20
            w_gpu = 0.00
        score = (
            w_cpu * cpu_norm
            + w_ram * ram_norm
            + w_storage * storage_norm
            + w_dur * durability_norm
            + w_gpu * gpu_norm
        )
    elif usage == "Gaming":
        score = (
            0.50 * gpu_norm
            + 0.30 * cpu_norm
            + 0.10 * ram_norm
            + 0.05 * storage_norm
            + 0.05 * durability_norm
        )
    elif usage == "Đồ họa / Lập trình AI":
        score = (
            0.35 * cpu_norm
            + 0.35 * gpu_norm
            + 0.20 * ram_norm
            + 0.05 * storage_norm
            + 0.05 * durability_norm
        )
    else:  # Tất cả / Đa dụng (Cân bằng)
        score = (
            0.35 * cpu_norm
            + 0.35 * gpu_norm
            + 0.15 * ram_norm
            + 0.08 * storage_norm
            + 0.07 * durability_norm
        )

    return round(score, 4)


def validate_full_build(
    cpu: Dict,
    mainboard: Dict,
    ram: Dict,
    gpu: Dict,
    psu: Dict,
    case: Dict,
    storage: Dict,
    cooler: Optional[Dict] = None,
    hdd: Optional[Dict] = None,
    ram_qty: int = 1,
    extra_storage_count: int = 0,
    budget: int = 0,
) -> Tuple[List[str], List[str]]:
    """Kiểm tra toàn bộ ràng buộc cứng và mềm cho cấu hình hoàn chỉnh."""
    errors: List[str] = []
    warnings: List[str] = []

    checks = [
        check_r01_socket(cpu, mainboard),
        check_r02_ram_type(cpu, mainboard, ram),
        check_r03_ram_capacity(mainboard, ram, ram_qty=ram_qty),
        check_r04_form_factor(mainboard, case),
        check_r05_gpu_length(gpu, case),
        check_r07_display_output(cpu, gpu),
    ]

    # Kiểm tra PSU có tính thêm ổ cứng phụ
    min_w = calculate_min_psu_wattage(cpu, gpu, extra_storage_count=extra_storage_count)
    if psu["wattage"] < min_w:
        errors.append(
            f"[R06] Nguồn PSU thiếu công suất: Tổng công suất yêu cầu là {min_w}W "
            f"nhưng Nguồn '{psu['name']}' chỉ đạt {psu['wattage']}W."
        )
    else:
        tier_min_w = calculate_tier_min_psu_wattage(cpu, gpu, extra_storage_count=extra_storage_count)
        if psu["wattage"] < tier_min_w:
            warnings.append(
                f"[R06] Khuyến nghị công suất nguồn: Hệ thống CPU '{cpu['name']}' và VGA '{gpu['name']}' "
                f"nên trang bị nguồn từ {tier_min_w}W trở lên để hoạt động bền bỉ, an toàn tải."
            )

    if cooler is not None:
        checks.append(check_r10_cooler_tdp(cpu, cooler))
        cpu_tdp = cpu.get("tdp", 0)
        cpu_name = cpu.get("name", "").upper()
        is_high_heat = (
            cpu_tdp >= 105
            or cpu_name.endswith("K")
            or cpu_name.endswith("KF")
            or "RYZEN 7" in cpu_name and cpu_tdp >= 105
            or "RYZEN 9" in cpu_name
            or "CORE I7" in cpu_name
            or "CORE I9" in cpu_name
        )
        if is_high_heat and cooler.get("tdp_support", 0) < 200:
            warnings.append(
                f"[R10] Khuyến nghị tản nhiệt: CPU '{cpu['name']}' tỏa nhiều nhiệt, "
                f"nên trang bị tản nhiệt khí tháp đôi hoặc tản nhiệt nước AIO (từ 200W trở lên)."
            )

    for ok, msg in checks:
        if not ok:
            errors.append(msg)

    # Kiểm tra luật mềm R08
    is_balanced, _, bt_msg = check_r08_bottleneck(cpu, gpu)
    if not is_balanced:
        warnings.append(bt_msg)

    # Kiểm tra luật mềm lưu trữ HDD/SSD và ngân sách từ 12 triệu (Luật R11)
    if storage.get("type") == "SATA":
        if (hdd is not None and hdd.get("id") != "hdd_none") or budget >= 12_000_000:
            warnings.append(
                f"[R11] Khuyến nghị lưu trữ: Hệ thống sử dụng SSD SATA '{storage['name']}'. "
                f"Nên ưu tiên SSD chuẩn NVMe tốc độ cao hơn cho cấu hình từ 12 triệu hoặc khi đã có HDD lưu trữ."
            )

    return errors, warnings
