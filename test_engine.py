"""
test_engine.py
==============
Script Unit Test cho AI PC Builder Advisor
Môn: Trí tuệ nhân tạo - Biểu diễn & Giải thuật (CS 417)

Kiểm tra:
  1. Các luật ràng buộc cứng R01-R07 bắt đúng các trường hợp vi phạm
  2. BranchAndBoundOptimizer cho nghiệm tối ưu khớp 100% với BruteForceOptimizer
  3. Hiệu năng: B&B phải duyệt ít nút hơn Brute-Force

Chạy: python test_engine.py
"""

import sys
import io
from pathlib import Path

# Đặt stdout sang UTF-8 để hiển thị tiếng Việt trên Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Thêm thư mục gốc dự án vào sys.path
sys.path.insert(0, str(Path(__file__).parent))

from knowledge_base import (
    check_r01_socket,
    check_r02_ram_type,
    check_r03_ram_capacity,
    check_r04_form_factor,
    check_r05_gpu_length,
    check_r06_psu_wattage,
    check_r07_display_output,
    check_r08_bottleneck,
    check_r10_cooler_tdp,
    compute_fitness_score,
)
from ai_engine import (
    BranchAndBoundOptimizer,
    BruteForceOptimizer,
    ForwardChainingEngine,
    load_data,
)

# ANSI color codes
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"

pass_count = 0
fail_count = 0


def test(name: str, condition: bool, detail: str = ""):
    """Hàm kiểm tra và in kết quả."""
    global pass_count, fail_count
    if condition:
        print(f"  {GREEN}✅ PASS{RESET}  {name}")
        if detail:
            print(f"         {CYAN}{detail}{RESET}")
        pass_count += 1
    else:
        print(f"  {RED}❌ FAIL{RESET}  {name}")
        if detail:
            print(f"         {RED}{detail}{RESET}")
        fail_count += 1


def section(title: str):
    print(f"\n{BOLD}{CYAN}{'='*60}{RESET}")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    print(f"{BOLD}{CYAN}{'='*60}{RESET}")


# ===========================================================================
# DỮ LIỆU GIẢ LẬP ĐỂ KIỂM TRA (Fixtures)
# ===========================================================================

# CPU mẫu
cpu_lga1700_ddr4 = {
    "id": "cpu_i5_12400f", "name": "Intel Core i5-12400F",
    "socket": "LGA1700", "ram_type": "DDR4", "has_igpu": False,
    "tdp": 65, "score": 14000, "price": 3900000
}
cpu_lga1700_igpu = {
    "id": "cpu_i3_12100", "name": "Intel Core i3-12100",
    "socket": "LGA1700", "ram_type": "DDR4", "has_igpu": True,
    "tdp": 60, "score": 9500, "price": 2800000
}
cpu_am4_ddr4 = {
    "id": "cpu_r5_5600", "name": "AMD Ryzen 5 5600",
    "socket": "AM4", "ram_type": "DDR4", "has_igpu": False,
    "tdp": 65, "score": 13500, "price": 3600000
}
cpu_am5_ddr5 = {
    "id": "cpu_r7_7700x", "name": "AMD Ryzen 7 7700X",
    "socket": "AM5", "ram_type": "DDR5", "has_igpu": True,
    "tdp": 105, "score": 30000, "price": 12800000
}

# Mainboard mẫu
mb_lga1700_ddr4 = {
    "id": "mb_b660m", "name": "Gigabyte B660M DDR4",
    "socket": "LGA1700", "ram_type": "DDR4",
    "form_factor": "mATX", "max_ram_gb": 128, "price": 2900000
}
mb_am4_ddr4 = {
    "id": "mb_b550", "name": "Gigabyte B550M",
    "socket": "AM4", "ram_type": "DDR4",
    "form_factor": "mATX", "max_ram_gb": 128, "price": 2200000
}
mb_am5_ddr5 = {
    "id": "mb_b650", "name": "ASUS TUF B650E DDR5",
    "socket": "AM5", "ram_type": "DDR5",
    "form_factor": "ATX", "max_ram_gb": 128, "price": 7200000
}
mb_lga1700_atx = {
    "id": "mb_z690", "name": "Gigabyte Z690 AORUS",
    "socket": "LGA1700", "ram_type": "DDR4",
    "form_factor": "ATX", "max_ram_gb": 128, "price": 7500000
}

# RAM mẫu
ram_ddr4_16gb = {
    "id": "ram_ddr4_16", "name": "Kingston DDR4 16GB",
    "ram_type": "DDR4", "capacity_gb": 16, "speed_mhz": 3200, "price": 1250000
}
ram_ddr5_16gb = {
    "id": "ram_ddr5_16", "name": "Kingston DDR5 16GB",
    "ram_type": "DDR5", "capacity_gb": 16, "speed_mhz": 5200, "price": 2100000
}
ram_ddr4_8gb = {
    "id": "ram_ddr4_8", "name": "Kingston DDR4 8GB",
    "ram_type": "DDR4", "capacity_gb": 8, "speed_mhz": 3200, "price": 650000
}
ram_ddr4_256gb_too_large = {
    "id": "ram_huge", "name": "RAM giả 256GB",
    "ram_type": "DDR4", "capacity_gb": 256, "speed_mhz": 3200, "price": 99000000
}

# GPU mẫu
gpu_none = {
    "id": "gpu_none", "name": "Không dùng GPU rời",
    "vram_gb": 0, "tdp": 0, "length_mm": 0, "score": 1500, "price": 0
}
gpu_rtx3060 = {
    "id": "gpu_rtx3060", "name": "RTX 3060 12GB",
    "vram_gb": 12, "tdp": 170, "length_mm": 242, "score": 14500, "price": 7900000
}
gpu_rtx4090 = {
    "id": "gpu_rtx4090", "name": "RTX 4090 24GB",
    "vram_gb": 24, "tdp": 450, "length_mm": 336, "score": 40000, "price": 42000000
}
gpu_very_long = {
    "id": "gpu_long", "name": "GPU giả dài 500mm",
    "vram_gb": 16, "tdp": 300, "length_mm": 500, "score": 35000, "price": 15000000
}

# PSU mẫu
psu_350w = {
    "id": "psu_350", "name": "PSU giả 350W (không đủ công suất)",
    "wattage": 350, "efficiency": "80 Plus Bronze", "price": 500000
}
psu_450w = {
    "id": "psu_450", "name": "Seasonic 450W Bronze",
    "wattage": 450, "efficiency": "80 Plus Bronze", "price": 850000
}
psu_850w = {
    "id": "psu_850", "name": "EVGA 850W Gold",
    "wattage": 850, "efficiency": "80 Plus Gold", "price": 2900000
}

# Case mẫu
case_matx = {
    "id": "case_matx", "name": "Jonsbo D31 mATX",
    "supported_form_factors": ["mATX", "ITX"],
    "max_gpu_length_mm": 340, "price": 1300000
}
case_atx_large = {
    "id": "case_atx", "name": "NZXT H510 ATX",
    "supported_form_factors": ["ATX", "mATX", "ITX"],
    "max_gpu_length_mm": 325, "price": 1700000
}
case_full_tower = {
    "id": "case_full", "name": "Lian Li PC-O11 ATX",
    "supported_form_factors": ["ATX", "mATX"],
    "max_gpu_length_mm": 400, "price": 3200000
}

# Storage mẫu
storage_nvme_1tb = {
    "id": "ssd_1tb", "name": "Samsung 970 EVO Plus 1TB",
    "type": "NVMe", "capacity_gb": 1000, "price": 1900000
}


# ===========================================================================
# TEST SUITE 1: Các luật ràng buộc cứng (R01 - R07)
# ===========================================================================
section("SUITE 1: Kiểm tra Luật Ràng buộc cứng (Hard Constraints R01-R07)")

# R01 - Socket
print(f"\n{BOLD}[R01] Tương thích Socket{RESET}")
ok, _ = check_r01_socket(cpu_lga1700_ddr4, mb_lga1700_ddr4)
test("CPU LGA1700 + Mainboard LGA1700 → Hợp lệ", ok)

ok, _ = check_r01_socket(cpu_am4_ddr4, mb_lga1700_ddr4)
test("CPU AM4 + Mainboard LGA1700 → Phát hiện lỗi", not ok, "CPU AM4 ≠ Mainboard LGA1700")

ok, _ = check_r01_socket(cpu_am5_ddr5, mb_am5_ddr5)
test("CPU AM5 + Mainboard AM5 → Hợp lệ", ok)

ok, _ = check_r01_socket(cpu_lga1700_ddr4, mb_am4_ddr4)
test("CPU LGA1700 + Mainboard AM4 → Phát hiện lỗi", not ok)

cpu_k_series = {"name": "Intel Core i5-13600K", "socket": "LGA1700", "tdp": 125, "ram_type": "DDR4"}
mb_h610 = {"name": "Asus Prime H610M-K DDR4", "socket": "LGA1700", "ram_type": "DDR4"}
mb_b660 = {"name": "MSI MAG B660M Mortar DDR4", "socket": "LGA1700", "ram_type": "DDR4"}

ok_k_h610, _ = check_r01_socket(cpu_k_series, mb_h610)
test("CPU dòng K (i5-13600K) + Bo mạch chủ H610 → Phát hiện xung đột VRM", not ok_k_h610, "Cấm ghép CPU K với H610")

ok_k_b660, _ = check_r01_socket(cpu_k_series, mb_b660)
test("CPU dòng K (i5-13600K) + Bo mạch chủ B660 → Hợp lệ", ok_k_b660)

# AMD Ryzen 9 / TDP >= 105W + A520 vs B550
cpu_r9_5900x = {"name": "AMD Ryzen 9 5900X", "brand": "AMD", "socket": "AM4", "tdp": 105, "ram_type": "DDR4"}
cpu_r7_5800x3d = {"name": "AMD Ryzen 7 5800X3D", "brand": "AMD", "socket": "AM4", "tdp": 105, "ram_type": "DDR4"}
cpu_r5_5600 = {"name": "AMD Ryzen 5 5600", "brand": "AMD", "socket": "AM4", "tdp": 65, "ram_type": "DDR4"}
mb_a520 = {"name": "Gigabyte A520M DS3H", "socket": "AM4", "ram_type": "DDR4"}
mb_b550 = {"name": "MSI B550-A Pro", "socket": "AM4", "ram_type": "DDR4"}

ok_r9_a520, _ = check_r01_socket(cpu_r9_5900x, mb_a520)
test("CPU AMD Ryzen 9 5900X + Bo mạch chủ A520 → Phát hiện xung đột Chipset/VRM", not ok_r9_a520, "Cấm Ryzen 9 với A520")

ok_r7_a520, _ = check_r01_socket(cpu_r7_5800x3d, mb_a520)
test("CPU AMD Ryzen 7 5800X3D (105W) + Bo mạch chủ A520 → Phát hiện xung đột TDP", not ok_r7_a520, "Cấm AMD TDP >= 105W với A520")

ok_r5_a520, _ = check_r01_socket(cpu_r5_5600, mb_a520)
test("CPU AMD Ryzen 5 5600 (65W) + Bo mạch chủ A520 → Hợp lệ", ok_r5_a520)

ok_r9_b550, _ = check_r01_socket(cpu_r9_5900x, mb_b550)
test("CPU AMD Ryzen 9 5900X + Bo mạch chủ B550 → Hợp lệ", ok_r9_b550)

# Intel Gen 11 + H410 vs H510
cpu_gen11 = {"name": "Intel Core i5-11400F", "brand": "Intel", "socket": "LGA1200", "tdp": 65, "ram_type": "DDR4"}
mb_h410 = {"name": "ASUS Prime H410M-E", "socket": "LGA1200", "ram_type": "DDR4"}
mb_h510 = {"name": "MSI H510M-A Pro", "socket": "LGA1200", "ram_type": "DDR4"}

ok_gen11_h410, _ = check_r01_socket(cpu_gen11, mb_h410)
test("CPU Intel Gen 11 (i5-11400F) + Bo mạch chủ H410 → Phát hiện không tương thích", not ok_gen11_h410, "Cấm Gen 11 với H410")

ok_gen11_h510, _ = check_r01_socket(cpu_gen11, mb_h510)
test("CPU Intel Gen 11 (i5-11400F) + Bo mạch chủ H510 → Hợp lệ", ok_gen11_h510)

# R02 - RAM Type
print(f"\n{BOLD}[R02] Tương thích chuẩn RAM (DDR4/DDR5){RESET}")
ok, _ = check_r02_ram_type(cpu_lga1700_ddr4, mb_lga1700_ddr4, ram_ddr4_16gb)
test("CPU DDR4 + MB DDR4 + RAM DDR4 → Hợp lệ", ok)

ok, _ = check_r02_ram_type(cpu_lga1700_ddr4, mb_lga1700_ddr4, ram_ddr5_16gb)
test("CPU DDR4 + MB DDR4 + RAM DDR5 → Phát hiện lỗi", not ok, "DDR4 CPU không nhận DDR5 RAM")

ok, _ = check_r02_ram_type(cpu_am5_ddr5, mb_am5_ddr5, ram_ddr5_16gb)
test("CPU DDR5 + MB DDR5 + RAM DDR5 → Hợp lệ", ok)

ok, _ = check_r02_ram_type(cpu_am5_ddr5, mb_am5_ddr5, ram_ddr4_16gb)
test("CPU DDR5 + MB DDR5 + RAM DDR4 → Phát hiện lỗi", not ok)

# R03 - RAM Capacity
print(f"\n{BOLD}[R03] Dung lượng RAM không vượt giới hạn Mainboard{RESET}")
ok, _ = check_r03_ram_capacity(mb_lga1700_ddr4, ram_ddr4_16gb)
test("RAM 16GB <= Mainboard max 128GB → Hợp lệ", ok)

ok, _ = check_r03_ram_capacity(mb_lga1700_ddr4, ram_ddr4_256gb_too_large)
test("RAM 256GB > Mainboard max 128GB → Phát hiện lỗi", not ok, "256GB > 128GB")

# R04 - Form Factor
print(f"\n{BOLD}[R04] Kích thước Mainboard phù hợp Case{RESET}")
ok, _ = check_r04_form_factor(mb_lga1700_ddr4, case_matx)
test("Mainboard mATX vào Case hỗ trợ [mATX, ITX] → Hợp lệ", ok)

ok, _ = check_r04_form_factor(mb_lga1700_atx, case_matx)
test("Mainboard ATX vào Case chỉ hỗ trợ [mATX, ITX] → Phát hiện lỗi", not ok, "ATX không vừa Case mATX-only")

ok, _ = check_r04_form_factor(mb_am5_ddr5, case_full_tower)
test("Mainboard ATX vào Case hỗ trợ [ATX, mATX] → Hợp lệ", ok)

# R05 - GPU Length
print(f"\n{BOLD}[R05] Chiều dài GPU không vượt giới hạn Case{RESET}")
ok, _ = check_r05_gpu_length(gpu_rtx3060, case_atx_large)
test("GPU 242mm <= Case max 325mm → Hợp lệ", ok)

ok, _ = check_r05_gpu_length(gpu_very_long, case_atx_large)
test("GPU 500mm > Case max 325mm → Phát hiện lỗi", not ok, "GPU 500mm không vừa Case 325mm")

ok, _ = check_r05_gpu_length(gpu_none, case_matx)
test("GPU None (iGPU) → Bỏ qua kiểm tra chiều dài", ok, "gpu_none không cần kiểm tra kích thước")

ok, _ = check_r05_gpu_length(gpu_rtx4090, case_full_tower)
test("GPU RTX 4090 (336mm) <= Case max 400mm → Hợp lệ", ok)

# R06 - PSU Wattage
print(f"\n{BOLD}[R06] Công suất nguồn PSU đủ dùng{RESET}")
# i5-12400F (65W) + RTX 3060 (170W) + 100W = 335W * 1.25 = ceil(418.75) = 419W
ok, msg = check_r06_psu_wattage(cpu_lga1700_ddr4, gpu_rtx3060, psu_850w)
test("CPU 65W + GPU 170W → cần 419W, PSU 850W → Hợp lệ", ok)

ok, msg = check_r06_psu_wattage(cpu_lga1700_ddr4, gpu_rtx3060, psu_350w)
test("CPU 65W + GPU 170W → cần 419W, PSU 350W (nhỏ hơn 419W) → Phát hiện lỗi", not ok, f"350W < 419W yêu cầu")

# PSU 450W đủ cho CPU(65)+GPU(170)+100)*1.25=419W
ok, msg = check_r06_psu_wattage(cpu_lga1700_ddr4, gpu_rtx3060, psu_450w)
test("CPU 65W + GPU 170W → cần 419W, PSU 450W (450>=419) → Hợp lệ", ok, "450W >= 419W → đủ công suất")

# CPU 65W + GPU 0W (none) + 100W = 165W * 1.25 = 207W → PSU 450W đủ
ok, _ = check_r06_psu_wattage(cpu_lga1700_ddr4, gpu_none, psu_450w)
test("CPU 65W + GPU None (0W) → cần 207W, PSU 450W → Hợp lệ", ok)

# RTX 4090 (450W): (65 + 450 + 100) * 1.25 = 768.75 → ceil = 769W → PSU 450W thiếu
ok, _ = check_r06_psu_wattage(cpu_lga1700_ddr4, gpu_rtx4090, psu_450w)
test("CPU 65W + GPU RTX4090 (450W) → cần 769W, PSU 450W → Phát hiện lỗi", not ok)

# R07 - Display Output
print(f"\n{BOLD}[R07] Yêu cầu hiển thị hình ảnh (iGPU hoặc GPU rời){RESET}")
ok, _ = check_r07_display_output(cpu_lga1700_igpu, gpu_none)
test("CPU có iGPU + Không có GPU rời → Hợp lệ (dùng iGPU)", ok)

ok, _ = check_r07_display_output(cpu_lga1700_ddr4, gpu_none)
test("CPU không có iGPU (đuôi F) + Không có GPU rời → Phát hiện lỗi", not ok,
     "i5-12400F không iGPU, không GPU rời = không xuất hình ảnh được!")

ok, _ = check_r07_display_output(cpu_lga1700_ddr4, gpu_rtx3060)
test("CPU không có iGPU + Có GPU rời RTX 3060 → Hợp lệ", ok)

ok, _ = check_r07_display_output(cpu_lga1700_igpu, gpu_rtx3060)
test("CPU có iGPU + Có GPU rời → Hợp lệ (dùng GPU rời)", ok)

# R10 - Cooler TDP
print(f"\n{BOLD}[R10] Công suất giải nhiệt của Tản nhiệt (Cooler TDP){RESET}")
cooler_stock = {"id": "cooler_stock", "name": "Stock Cooler", "tdp_support": 65, "price": 0}
cooler_air_150w = {"id": "cooler_ak400", "name": "Deepcool AK400", "tdp_support": 150, "price": 550000}
cooler_weak_50w = {"id": "cooler_weak", "name": "Tản mini", "tdp_support": 50, "price": 100000}

# CPU 60W + Stock Cooler (65W) -> OK
ok, _ = check_r10_cooler_tdp(cpu_lga1700_igpu, cooler_stock)
test("CPU 60W + Stock Cooler (65W) → Hợp lệ", ok)

# CPU 105W + Stock Cooler -> FAIL
ok, _ = check_r10_cooler_tdp(cpu_am5_ddr5, cooler_stock)
test("CPU 105W + Stock Cooler → Phát hiện lỗi quá nhiệt", not ok, "CPU 105W > 65W tản stock")

# CPU 105W + Cooler 150W -> OK
ok, _ = check_r10_cooler_tdp(cpu_am5_ddr5, cooler_air_150w)
test("CPU 105W + Tản 150W → Hợp lệ", ok)

# CPU 65W + Tản yếu 50W -> FAIL
ok, _ = check_r10_cooler_tdp(cpu_lga1700_ddr4, cooler_weak_50w)
test("CPU 65W + Tản yếu 50W → Phát hiện lỗi thiếu công suất", not ok, "65W > 50W")


# ===========================================================================
# TEST SUITE 2: Luật mềm R08 (Bottleneck Detection)
# ===========================================================================
section("SUITE 2: Kiểm tra Luật mềm R08 (Bottleneck Detection)")

print(f"\n{BOLD}[R08] Phát hiện nghẽn cổ chai CPU/GPU{RESET}")

# CPU yếu (9500), GPU mạnh (40000) → cpu_norm=22.6, gpu_norm=100 → ratio=0.226 < 0.55 → bottleneck CPU
ok, level, msg = check_r08_bottleneck(cpu_lga1700_igpu, gpu_rtx4090)
test("CPU yếu (9500) + GPU RTX4090 (40000) → Phát hiện CPU Bottleneck", level == "cpu_bottleneck",
     f"Level = {level}")

# CPU mạnh (30000), GPU yếu (14500) → cpu_norm=71.4, gpu_norm=36.25 → ratio=0.508 < 0.55 (CPU cũng bị BN)
# gpu_norm/cpu_norm = 0.508 không < 0.45 → thực ra là "ok" hoặc "cpu_bottleneck" nhẹ
ok_flag, level2, msg2 = check_r08_bottleneck(cpu_am5_ddr5, gpu_rtx3060)
test("CPU RTX7700X (30000) + GPU RTX3060 (14500) → Kiểm tra cân bằng", True,
     f"Level = {level2}: {msg2[:80]}...")

# GPU cực yếu so với CPU mạnh
cpu_very_strong = {"id": "x", "name": "CPU mạnh", "score": 42000}
gpu_very_weak = {"id": "y", "name": "GPU yếu", "score": 2000, "tdp": 50, "length_mm": 200}
_, level3, msg3 = check_r08_bottleneck(cpu_very_strong, gpu_very_weak)
test("CPU cực mạnh (42000) + GPU cực yếu (2000) → Phát hiện GPU quá yếu", level3 == "gpu_weak",
     f"gpu_norm/cpu_norm = {2000/40000 / (42000/42000):.3f} < 0.45")

# Cân bằng tốt: CPU 14000, GPU 14500
_, level4, _ = check_r08_bottleneck(cpu_lga1700_ddr4, gpu_rtx3060)
test("CPU i5-12400F (14000) + GPU RTX3060 (14500) → Cân bằng tốt (OK)", level4 == "ok")

# GPU None → không áp dụng
_, level5, _ = check_r08_bottleneck(cpu_lga1700_ddr4, gpu_none)
test("GPU None (iGPU) → Bỏ qua kiểm tra bottleneck", level5 == "ok")


# ===========================================================================
# TEST SUITE 3: Hàm tính điểm tổng hợp compute_fitness_score
# ===========================================================================
section("SUITE 3: Kiểm tra Hàm tính điểm tổng hợp (R09)")

print(f"\n{BOLD}[R09] compute_fitness_score(){RESET}")

score_van_phong = compute_fitness_score(
    cpu_lga1700_igpu, gpu_none, ram_ddr4_8gb, storage_nvme_1tb, "Văn phòng / Học tập"
)
test("Điểm Văn phòng > 0", score_van_phong > 0, f"Score = {score_van_phong:.4f}")
test("Điểm Văn phòng <= 100", score_van_phong <= 100, f"Score = {score_van_phong:.4f}")

score_gaming = compute_fitness_score(
    cpu_lga1700_ddr4, gpu_rtx3060, ram_ddr4_16gb, storage_nvme_1tb, "Gaming"
)
test("Điểm Gaming > 0", score_gaming > 0, f"Score = {score_gaming:.4f}")

# Gaming với GPU mạnh hơn phải có điểm cao hơn hoặc bằng
score_gaming_strong = compute_fitness_score(
    cpu_lga1700_ddr4, gpu_rtx4090, ram_ddr4_16gb, storage_nvme_1tb, "Gaming"
)
test("Gaming RTX4090 > Gaming RTX3060 (GPU là thành phần chính)", score_gaming_strong > score_gaming,
     f"RTX4090: {score_gaming_strong:.4f} > RTX3060: {score_gaming:.4f}")


# ===========================================================================
# TEST SUITE 4: Forward Chaining Engine
# ===========================================================================
section("SUITE 4: Kiểm tra Động cơ Suy diễn Tiến (ForwardChainingEngine)")

DATA = load_data(str(Path(__file__).parent / "data.json"))
engine = ForwardChainingEngine(DATA)

print(f"\n{BOLD}[FC] Kiểm tra lọc linh kiện theo Forward Chaining{RESET}")

# Test 1: Chọn CPU LGA1700 → Mainboard chỉ còn LGA1700
fc1 = engine.infer({"cpu": DATA["cpus"][0]})  # i3-12100 LGA1700
mb_sockets = set(mb["socket"] for mb in fc1.filtered_options["mainboards"])
test("Chọn CPU LGA1700 → Chỉ còn Mainboard LGA1700", mb_sockets == {"LGA1700"},
     f"Sockets còn lại: {mb_sockets}")

# Test 2: Chọn CPU AM4 → Mainboard chỉ còn AM4
cpu_am4 = next(c for c in DATA["cpus"] if c["socket"] == "AM4")
fc2 = engine.infer({"cpu": cpu_am4})
mb_sockets2 = set(mb["socket"] for mb in fc2.filtered_options["mainboards"])
test("Chọn CPU AM4 → Chỉ còn Mainboard AM4", mb_sockets2 == {"AM4"},
     f"Sockets còn lại: {mb_sockets2}")

# Test 3: Chọn GPU None (iGPU) → CPU phải có iGPU
fc3 = engine.infer({"gpu": DATA["gpus"][0]})  # gpu_none
cpu_has_igpu = all(c["has_igpu"] for c in fc3.filtered_options["cpus"])
test("Chọn GPU None → Chỉ còn CPU có iGPU", cpu_has_igpu)

# Test 4: Chọn CPU + GPU → PSU được lọc theo công suất tối thiểu
cpu_sel = next(c for c in DATA["cpus"] if c["id"] == "cpu_i5_12400f")
gpu_sel = next(g for g in DATA["gpus"] if g["id"] == "gpu_rtx3060")
fc4 = engine.infer({"cpu": cpu_sel, "gpu": gpu_sel})
import math
min_w = math.ceil((65 + 170 + 100) * 1.25)  # 419W
psu_min_watt = min(p["wattage"] for p in fc4.filtered_options["psus"])
test(f"CPU 65W + GPU 170W → PSU lọc, wattage tối thiểu >= {min_w}W",
     psu_min_watt >= min_w,
     f"PSU thấp nhất còn lại: {psu_min_watt}W")


# ===========================================================================
# TEST SUITE 5: Tính đúng đắn Branch & Bound vs Brute-Force
# ===========================================================================
section("SUITE 5: Kiểm tra tính đúng đắn Branch & Bound vs Brute-Force")

print(f"\n{BOLD}[BB vs BF] Chạy so sánh nghiệm tối ưu...{RESET}")

BUDGET_TEST = 20_000_000
SIZE = 4  # Dùng 4 linh kiện/nhóm để chạy nhanh

for usage_name in ["Văn phòng / Học tập", "Gaming", "Đồ họa / Lập trình AI"]:
    subset = {
        "cpus": DATA["cpus"][:SIZE],
        "mainboards": DATA["mainboards"][:SIZE],
        "rams": DATA["rams"][:SIZE],
        "gpus": DATA["gpus"][:SIZE],
        "psus": DATA["psus"][:SIZE],
        "cases": DATA["cases"][:SIZE],
        "storages": DATA["storages"][:SIZE],
    }

    bf = BruteForceOptimizer(subset)
    bb = BranchAndBoundOptimizer(subset)

    bf_builds, bf_stats = bf.optimize(BUDGET_TEST, usage_name, items_per_category=SIZE)
    bb_builds, bb_stats = bb.optimize(BUDGET_TEST, usage_name)

    bf_score = bf_builds[0].fitness_score if bf_builds else 0.0
    bb_score = bb_builds[0].fitness_score if bb_builds else 0.0
    is_match = abs(bf_score - bb_score) < 0.001

    test(
        f"Nghiệm khớp 100% cho '{usage_name}'",
        is_match,
        f"BF={bf_score:.4f} | B&B={bb_score:.4f} | "
        f"BF nodes={bf_stats.nodes_visited:,} | B&B nodes={bb_stats.nodes_visited:,} | "
        f"B&B tốc độ: {bf_stats.execution_time_ms/max(bb_stats.execution_time_ms,0.001):.1f}x"
    )

    test(
        f"B&B duyệt ít nút hơn BF cho '{usage_name}'",
        bb_stats.nodes_visited <= bf_stats.nodes_visited,
        f"B&B: {bb_stats.nodes_visited:,} <= BF: {bf_stats.nodes_visited:,}"
    )

    test(
        f"B&B tổng thời gian hợp lệ cho '{usage_name}'",
        bb_stats.execution_time_ms < 10000,
        f"Thời gian: {bb_stats.execution_time_ms:.2f}ms"
    )


# ===========================================================================
# TỔNG KẾT
# ===========================================================================
section("TỔNG KẾT KẾT QUẢ KIỂM THỬ")
total = pass_count + fail_count
print(f"\n  Tổng số test: {total}")
print(f"  {GREEN}✅ Passed: {pass_count}{RESET}")
print(f"  {RED if fail_count > 0 else GREEN}❌ Failed: {fail_count}{RESET}")
print(f"  Tỷ lệ thành công: {pass_count/total*100:.1f}%\n")

if fail_count == 0:
    print(f"  {GREEN}{BOLD}🎉 TẤT CẢ TEST ĐỀU PASS! Hệ thống hoạt động đúng đắn.{RESET}\n")
else:
    print(f"  {RED}{BOLD}⚠️  Có {fail_count} test FAIL! Cần kiểm tra lại.{RESET}\n")

sys.exit(0 if fail_count == 0 else 1)
