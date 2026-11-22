"""
ai_engine.py
============
Module Động cơ Trí tuệ Nhân tạo
Môn: Trí tuệ nhân tạo - Biểu diễn & Giải thuật (CS 417)

Triển khai:
  1. ForwardChainingEngine   - Động cơ Suy diễn tiến (Forward Chaining)
  2. BranchAndBoundOptimizer - Tìm kiếm tối ưu Nhánh & Cận (Branch & Bound)
  3. BruteForceOptimizer      - Tìm kiếm vét cạn (Backtracking để đối chiếu)
  4. Khử trùng lặp cảnh báo và không dùng ký tự emoji
"""

import json
import math
import time
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from knowledge_base import (
    USAGE_PROFILES,
    calculate_min_psu_wattage,
    calculate_tier_min_psu_wattage,
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
    compute_gpu_norm,
    compute_ram_norm,
    filter_gpus_by_budget_floor,
    is_pcie3_mainboard,
    is_vrm_incompatible,
    validate_full_build,
)


def select_optimal_cooler(coolers_list: List[Dict], cpu: Dict, budget: int = 0, usage: str = "") -> Optional[Dict]:
    """
    Tự động chọn tản nhiệt chuẩn xác theo luật an toàn phần cứng:
    - CPU mát (TDP <= 65W, như Ryzen 7 5700X, Ryzen 5 5600, i5-12400F... không thuộc dòng K/X3D):
      Cấm chọn tản nhiệt quá đắt (giới hạn giá <= 700.000 VNĐ). Ưu tiên tản khí 1 tháp gọn mát
      như ID-Cooling SE-224-XT (480k) hoặc Deepcool AK400 (700k) thay vì tản tháp đôi >= 1.000.000 VNĐ (AK620).
    - CPU TDP >= 105W (Ryzen 7/9, Core i7/i9, dòng K, dòng X3D): Bắt buộc tản nhiệt có tdp_support >= 200W
      (tản khí tháp đôi hoặc tản nhiệt nước AIO), cấm dùng tản stock.
    - Gaming >= 16 triệu: Ưu tiên tản khí rời thay vì Stock Cooler để giữ xung nhịp ổn định khi chơi game dài giờ.
    - Đồ họa / Lập trình AI: CPU >= 105W bắt buộc tản rời >= 200W; phân khúc > 32 triệu ưu tiên tản tháp đôi hoặc AIO.
    - Văn phòng > 22 triệu: Bắt buộc tản nhiệt khí rời êm ái (cấm dùng tản stock).
    - Tất cả / Đa dụng >= 20 triệu: Bắt buộc tản nhiệt rời (tản khỏe từ 28 triệu).
    """
    if not coolers_list:
        return None
    cpu_tdp = cpu.get("tdp", 65)
    cpu_name = cpu.get("name", "").upper()

    is_k_or_x3d = (
        cpu_name.endswith("K")
        or cpu_name.endswith("KF")
        or cpu_name.endswith("KS")
        or " K " in cpu_name
        or " KF " in cpu_name
        or "X3D" in cpu_name
    )

    # 1. CPU mát (TDP <= 65W) và không phải dòng ép xung K/X3D:
    # Cấm chọn tản quá đắt (giới hạn giá <= 700.000 VNĐ).
    if cpu_tdp <= 65 and not is_k_or_x3d:
        need_aftermarket = (
            (usage == "Gaming" and budget >= 16_000_000)
            or (usage == "Văn phòng / Học tập" and budget > 22_000_000)
            or (("TẤT CẢ" in usage.upper() or "ĐA DỤNG" in usage.upper()) and budget >= 20_000_000)
            or (("ĐỒ HỌA" in usage.upper() or "AI" in usage.upper()) and budget >= 18_000_000)
        )
        if need_aftermarket:
            compact_aftermarket = [
                c for c in coolers_list
                if c.get("id") != "cooler_stock" and c.get("price", 0) <= 700_000 and c.get("tdp_support", 0) >= cpu_tdp
            ]
            if compact_aftermarket:
                return deepcopy(min(compact_aftermarket, key=lambda c: c.get("price", 0)))
        stock = next((c for c in coolers_list if c.get("id") == "cooler_stock"), coolers_list[0])
        return deepcopy(stock)

    # 2. CPU tỏa nhiều nhiệt (TDP >= 105W, dòng K/KF/KS, dòng X3D, i7, i9, Ryzen 9)
    is_high_heat_cpu = (
        cpu_tdp >= 105
        or is_k_or_x3d
        or ("RYZEN 7" in cpu_name and cpu_tdp >= 105)
        or "RYZEN 9" in cpu_name
        or "CORE I7" in cpu_name
        or "CORE I9" in cpu_name
        or (("ĐỒ HỌA" in usage.upper() or "AI" in usage.upper()) and (budget > 32_000_000 or cpu_tdp >= 105))
    )
    if is_high_heat_cpu:
        matching = [
            c for c in coolers_list
            if c.get("tdp_support", 0) >= 200 and c.get("id") != "cooler_stock"
        ]
        if not matching:
            matching = [
                c for c in coolers_list
                if c.get("tdp_support", 0) >= cpu_tdp and c.get("id") != "cooler_stock"
            ]
        if budget >= 35_000_000 or (("ĐỒ HỌA" in usage.upper() or "AI" in usage.upper()) and budget > 32_000_000):
            aio_coolers = [
                c for c in matching
                if "aio" in c.get("name", "").lower() or c.get("tdp_support", 0) >= 250
            ]
            if aio_coolers:
                return deepcopy(min(aio_coolers, key=lambda c: c.get("price", 0)))
        return deepcopy(min(matching, key=lambda c: c.get("price", 0)) if matching else coolers_list[0])

    need_aftermarket = (
        (usage == "Gaming" and budget >= 16_000_000)
        or (usage == "Văn phòng / Học tập" and budget > 22_000_000)
        or (("TẤT CẢ" in usage.upper() or "ĐA DỤNG" in usage.upper()) and budget >= 20_000_000)
        or (("ĐỒ HỌA" in usage.upper() or "AI" in usage.upper()) and budget >= 18_000_000)
    )
    if need_aftermarket:
        aftermarket = [
            c for c in coolers_list
            if c.get("id") != "cooler_stock" and c.get("tdp_support", 0) >= cpu_tdp
        ]
        if aftermarket:
            return deepcopy(min(aftermarket, key=lambda c: c.get("price", 0)))

    matching = [
        c for c in coolers_list
        if c.get("tdp_support", 0) >= cpu_tdp and c.get("id") != "cooler_stock"
    ]
    return deepcopy(min(matching, key=lambda c: c.get("price", 0)) if matching else coolers_list[0])


def load_data(path: str = "data.json") -> Dict[str, List[Dict]]:
    """Tải cơ sở dữ liệu linh kiện từ file JSON."""
    data_path = Path(path)
    if not data_path.exists():
        data_path = Path(__file__).parent / "data.json"
    with open(data_path, encoding="utf-8") as f:
        return json.load(f)


# ===========================================================================
# PHẦN 1: ĐỘNG CƠ SUY DIỄN TIẾN (Forward Chaining Engine)
# ===========================================================================

@dataclass
class FCResult:
    """Kết quả trả về từ động cơ suy diễn tiến."""
    derived_facts: Dict[str, Any] = field(default_factory=dict)
    filtered_options: Dict[str, List[Dict]] = field(default_factory=dict)
    logs: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class ForwardChainingEngine:
    """
    Động cơ Suy diễn Tiến (Forward Chaining / Data-Driven Inference).
    Khởi tạo Working Memory từ các sự kiện hiện có, kích hoạt tập luật
    theo chu trình lặp (fixed-point) cho đến khi không sinh thêm sự kiện mới.
    """

    def __init__(self, data: Dict[str, List[Dict]]):
        self.data = data

    def infer(
        self,
        selected: Dict[str, Optional[Dict]],
        usage: str = "Văn phòng / Học tập",
        validate_errors: bool = True,
        ram_qty: int = 1,
        extra_storage_count: int = 0,
    ) -> FCResult:
        result = FCResult()

        cpu = selected.get("cpu")
        mainboard = selected.get("mainboard")
        ram = selected.get("ram")
        gpu = selected.get("gpu")
        psu = selected.get("psu")
        case = selected.get("case")
        storage = selected.get("storage")
        cooler = selected.get("cooler")
        hdd = selected.get("hdd")

        # Khởi tạo Working Memory
        wm_cpu = list(self.data.get("cpus", []))
        wm_mb = list(self.data.get("mainboards", []))
        wm_ram = list(self.data.get("rams", []))
        wm_gpu = list(self.data.get("gpus", []))
        wm_psu = list(self.data.get("psus", []))
        wm_case = list(self.data.get("cases", []))
        wm_storage = list(self.data.get("storages", []))
        wm_cooler = list(self.data.get("coolers", []))
        wm_hdd = list(self.data.get("hdds", []))

        changed = True
        iteration = 0
        while changed:
            changed = False
            iteration += 1
            result.logs.append(f"--- Vong lap suy dien #{iteration} ---")

            # [R01] CPU -> lọc Mainboard theo Socket và VRM
            if cpu is not None:
                socket_val = cpu["socket"]
                if result.derived_facts.get("socket") != socket_val:
                    result.derived_facts["socket"] = socket_val
                    before = len(wm_mb)
                    wm_mb = [mb for mb in wm_mb if mb["socket"] == socket_val and not is_vrm_incompatible(cpu, mb)]
                    removed = before - len(wm_mb)
                    result.logs.append(
                        f"[R01] Suy dien tu CPU '{cpu['name']}' -> Yeu cau Socket = {socket_val} va tuong thich VRM "
                        f"(Loai bo {removed}/{before} Bo mach chu khong phu hop)"
                    )
                    if removed > 0:
                        changed = True

            # [R01 dao] Mainboard -> lọc CPU theo Socket và VRM
            if mainboard is not None:
                mb_socket = mainboard["socket"]
                if result.derived_facts.get("mb_socket") != mb_socket:
                    result.derived_facts["mb_socket"] = mb_socket
                    before = len(wm_cpu)
                    wm_cpu = [c for c in wm_cpu if c["socket"] == mb_socket and not is_vrm_incompatible(c, mainboard)]
                    removed = before - len(wm_cpu)
                    if removed > 0:
                        result.logs.append(
                            f"[R01] Suy dien tu Bo mach chu '{mainboard['name']}' -> Loc CPU socket {mb_socket} va gioi han VRM "
                            f"(Loai bo {removed}/{before} CPU khong tuong thich)"
                        )
                        changed = True

            # [R02] Chuẩn RAM DDR4/DDR5
            ram_type_known = None
            if cpu is not None:
                ram_type_known = cpu["ram_type"]
            elif mainboard is not None:
                ram_type_known = mainboard["ram_type"]

            if ram_type_known and result.derived_facts.get("ram_type") != ram_type_known:
                result.derived_facts["ram_type"] = ram_type_known
                before_ram = len(wm_ram)
                wm_ram = [r for r in wm_ram if r["ram_type"] == ram_type_known]
                removed_ram = before_ram - len(wm_ram)
                result.logs.append(
                    f"[R02] Suy dien chuan RAM = {ram_type_known} -> Loai bo {removed_ram}/{before_ram} RAM khong tuong thich"
                )
                before_mb = len(wm_mb)
                wm_mb = [mb for mb in wm_mb if mb["ram_type"] == ram_type_known]
                removed_mb = before_mb - len(wm_mb)
                if removed_mb > 0:
                    result.logs.append(
                        f"[R02] Loai bo {removed_mb}/{before_mb} Bo mach chu khong ho tro chuan {ram_type_known}"
                    )
                if removed_ram > 0 or removed_mb > 0:
                    changed = True

            # [R03] Dung lượng RAM <= Mainboard max_ram_gb
            if mainboard is not None:
                max_gb = mainboard["max_ram_gb"]
                if result.derived_facts.get("max_ram_gb") != max_gb:
                    result.derived_facts["max_ram_gb"] = max_gb
                    before = len(wm_ram)
                    wm_ram = [r for r in wm_ram if r["capacity_gb"] <= max_gb]
                    removed = before - len(wm_ram)
                    if removed > 0:
                        result.logs.append(
                            f"[R03] Bo mach chu gioi han RAM toi da {max_gb}GB -> Loai bo {removed}/{before} RAM vuot dung luong"
                        )
                        changed = True

            # [R04] Form Factor Mainboard - Case
            if mainboard is not None:
                ff = mainboard["form_factor"]
                if result.derived_facts.get("form_factor") != ff:
                    result.derived_facts["form_factor"] = ff
                    before = len(wm_case)
                    wm_case = [c for c in wm_case if ff in c["supported_form_factors"]]
                    removed = before - len(wm_case)
                    result.logs.append(
                        f"[R04] Bo mach chu chuan {ff} -> Loai bo {removed}/{before} Vo may khong ho tro"
                    )
                    if removed > 0:
                        changed = True

            # [R05] Chiều dài GPU <= Case max_gpu_length_mm
            if gpu is not None and gpu["id"] != "gpu_none":
                gpu_len = gpu["length_mm"]
                if result.derived_facts.get("gpu_length") != gpu_len:
                    result.derived_facts["gpu_length"] = gpu_len
                    before = len(wm_case)
                    wm_case = [c for c in wm_case if c["max_gpu_length_mm"] >= gpu_len]
                    removed = before - len(wm_case)
                    result.logs.append(
                        f"[R05] Card do hoa dai {gpu_len}mm -> Loai bo {removed}/{before} Vo may qua nho"
                    )
                    if removed > 0:
                        changed = True

            # [R06] Công suất nguồn PSU
            if cpu is not None and gpu is not None:
                min_w = calculate_min_psu_wattage(cpu, gpu)
                if result.derived_facts.get("min_psu_wattage") != min_w:
                    result.derived_facts["min_psu_wattage"] = min_w
                    before = len(wm_psu)
                    wm_psu = [p for p in wm_psu if p["wattage"] >= min_w]
                    removed = before - len(wm_psu)
                    result.logs.append(
                        f"[R06] Tong cong suat TDP ({cpu['tdp']}W + {gpu['tdp']}W + 100W) * 1.25 = {min_w}W "
                        f"-> Yeu cau PSU >= {min_w}W (Loai bo {removed}/{before} PSU thieu cong suat)"
                    )
                    if removed > 0:
                        changed = True

            # [R07] Xuất hình ảnh khi không có GPU rời
            if gpu is not None and gpu["id"] == "gpu_none":
                if result.derived_facts.get("r07_checked") is None:
                    result.derived_facts["r07_checked"] = True
                    before = len(wm_cpu)
                    wm_cpu = [c for c in wm_cpu if c.get("has_igpu", False)]
                    removed = before - len(wm_cpu)
                    result.logs.append(
                        f"[R07] Khong dung card do hoa roi -> Yeu cau CPU co iGPU tich hop (Loai bo {removed}/{before} CPU)"
                    )
                    if removed > 0:
                        changed = True

                    if cpu is not None and not cpu.get("has_igpu", False) and validate_errors:
                        result.errors.append(
                            f"[R07] CPU '{cpu['name']}' khong co do hoa tich hop nhung khong chon card roi."
                        )

            # [R10] Công suất tản nhiệt đối với CPU
            if cpu is not None and wm_cooler:
                cpu_tdp = cpu.get("tdp", 65)
                if result.derived_facts.get("cpu_tdp_cooler") != cpu_tdp:
                    result.derived_facts["cpu_tdp_cooler"] = cpu_tdp
                    before = len(wm_cooler)
                    if cpu_tdp > 65:
                        wm_cooler = [
                            c for c in wm_cooler
                            if c.get("id") != "cooler_stock" and c.get("tdp_support", 0) >= cpu_tdp
                        ]
                    else:
                        wm_cooler = [
                            c for c in wm_cooler
                            if c.get("tdp_support", 0) >= cpu_tdp or c.get("id") == "cooler_stock"
                        ]
                    removed = before - len(wm_cooler)
                    if removed > 0:
                        result.logs.append(
                            f"[R10] CPU '{cpu['name']}' ({cpu_tdp}W) -> Yeu cau tan nhiet >= {cpu_tdp}W "
                            f"(Loai bo {removed}/{before} tan nhiet khong du cong suat)"
                        )
                        changed = True

        # Kiểm tra toàn bộ ràng buộc cứng khi đủ linh kiện cơ bản
        if validate_errors and all(
            v is not None for v in [cpu, mainboard, ram, gpu, psu, case, storage]
        ):
            errs, warns = validate_full_build(
                cpu, mainboard, ram, gpu, psu, case, storage, cooler, hdd,
                ram_qty=ram_qty, extra_storage_count=extra_storage_count
            )
            result.errors.extend(errs)
            result.warnings.extend(warns)

        # Kiểm tra cảnh báo nghẽn cổ chai R08 duy nhất một lần
        if cpu is not None and gpu is not None and gpu["id"] != "gpu_none":
            is_balanced, _, bt_msg = check_r08_bottleneck(cpu, gpu)
            if not is_balanced:
                result.warnings.append(bt_msg)
            else:
                result.logs.append(bt_msg)

        # Kiểm tra khuyến nghị RAM theo nhu cầu R09
        if ram is not None and usage in USAGE_PROFILES:
            profile = USAGE_PROFILES[usage]
            if ram["capacity_gb"] < profile["min_ram_gb"]:
                result.warnings.append(
                    f"[R09] Dung luong RAM {ram['capacity_gb']}GB thap hon muc khuyen nghi {profile['min_ram_gb']}GB cho '{usage}'."
                )

        if gpu is not None and usage in USAGE_PROFILES:
            profile = USAGE_PROFILES[usage]
            if profile["require_discrete_gpu"] and gpu["id"] == "gpu_none":
                result.warnings.append(
                    f"[R09] Nhu cau '{usage}' yeu cau card do hoa roi de dat hieu suat on dinh."
                )

        # Khử trùng lặp danh sách thông báo và log
        result.errors = list(dict.fromkeys(result.errors))
        result.warnings = list(dict.fromkeys(result.warnings))
        result.logs = list(dict.fromkeys(result.logs))

        result.filtered_options = {
            "cpus": wm_cpu,
            "mainboards": wm_mb,
            "rams": wm_ram,
            "gpus": wm_gpu,
            "psus": wm_psu,
            "cases": wm_case,
            "storages": wm_storage,
            "coolers": wm_cooler,
            "hdds": wm_hdd,
        }

        return result


# ===========================================================================
# PHẦN 2: THUẬT TOÁN TỐI ƯU NHÁNH & CẬN (Branch & Bound Optimizer)
# ===========================================================================

@dataclass
class BuildResult:
    """Đại diện cho một cấu hình máy tính hoàn chỉnh."""
    cpu: Dict = field(default_factory=dict)
    mainboard: Dict = field(default_factory=dict)
    ram: Dict = field(default_factory=dict)
    gpu: Dict = field(default_factory=dict)
    psu: Dict = field(default_factory=dict)
    case: Dict = field(default_factory=dict)
    storage: Dict = field(default_factory=dict)
    cooler: Dict = field(default_factory=dict)
    hdd: Dict = field(default_factory=dict)
    total_price: int = 0
    fitness_score: float = 0.0

    def to_dict(self) -> Dict:
        d = {
            "cpu": self.cpu,
            "mainboard": self.mainboard,
            "ram": self.ram,
            "gpu": self.gpu,
            "psu": self.psu,
            "case": self.case,
            "storage": self.storage,
            "total_price": self.total_price,
            "fitness_score": self.fitness_score,
        }
        if self.cooler:
            d["cooler"] = self.cooler
        if self.hdd:
            d["hdd"] = self.hdd
        return d


@dataclass
class OptimizerStats:
    """Thống kê chi tiết quá trình tìm kiếm của giải thuật."""
    nodes_visited: int = 0
    pruned_branches: int = 0
    valid_builds_found: int = 0
    execution_time_ms: float = 0.0
    algorithm_name: str = ""


class BranchAndBoundOptimizer:
    """
    Thuật toán Tối ưu Nhánh & Cận (Branch & Bound).
    Phân nhánh theo 7 tầng cốt lõi: CPU -> Mainboard -> RAM -> GPU -> PSU -> Case -> Storage
    Áp dụng Heuristic Value Ordering, Cắt tỉa theo Cận dưới Ngân sách, Cận trên Điểm số và Ràng buộc cứng.
    """

    def __init__(self, data: Dict[str, List[Dict]]):
        self.data = data
        self.cpu_max_score = max((c["score"] for c in data.get("cpus", [])), default=50000)
        self.gpu_max_score = max(
            (g["score"] for g in data.get("gpus", []) if g["id"] != "gpu_none"), default=40000
        )

    def _score_per_price(self, item: Dict) -> float:
        price = item.get("price", 1)
        if price == 0:
            return 0.0
        score = item.get("score", item.get("capacity_gb", 1))
        # Ưu tiên kit Dual Channel khi sắp xếp thứ tự duyệt
        if item.get("is_kit", False) or item.get("sticks", 1) >= 2 or "2x" in item.get("name", "").lower():
            score = score * 1.25
        return score / price

    def _get_min_price(self, candidates: List[Dict]) -> int:
        if not candidates:
            return 0
        return min(c.get("price", 0) for c in candidates)

    def _get_max_norm_score(self, candidates: List[Dict], category: str) -> float:
        if not candidates:
            return 0.0
        return 100.0

    def optimize(
        self,
        budget: int,
        usage: str = "Văn phòng / Học tập",
        locked_parts: Optional[Dict[str, Dict]] = None,
        top_k: int = 3,
        require_discrete_gpu: Optional[bool] = None,
        require_hdd: bool = False,
        cpu_brand: Optional[str] = None,
        gpu_brand: Optional[str] = None,
        target_ram_gb: Optional[int] = None,
        min_storage_gb: Optional[int] = None,
        min_total_storage_gb: Optional[int] = None,
        prefer_dual_ram: bool = False,
        strict_budget: bool = True,
    ) -> Tuple[List[BuildResult], OptimizerStats]:
        if locked_parts is None:
            locked_parts = {}

        start_time = time.perf_counter()
        profile = USAGE_PROFILES.get(usage, USAGE_PROFILES["Văn phòng / Học tập"])
        stats = OptimizerStats(algorithm_name="Branch & Bound")
        max_search_budget = budget if strict_budget else int(budget * 1.05)

        all_cpus = sorted(self.data.get("cpus", []), key=self._score_per_price, reverse=True)
        all_mbs = sorted(self.data.get("mainboards", []), key=lambda x: -x["price"])
        all_rams = sorted(self.data.get("rams", []), key=self._score_per_price, reverse=True)
        all_gpus = sorted(self.data.get("gpus", []), key=self._score_per_price, reverse=True)
        all_psus = sorted(self.data.get("psus", []), key=lambda x: x["wattage"])
        all_cases = sorted(self.data.get("cases", []), key=lambda x: x["price"])
        all_storages = sorted(self.data.get("storages", []), key=self._score_per_price, reverse=True)

        # Cấm Mainboard PCIe 3.0 cho cấu hình từ 18 triệu trở lên
        if budget >= 18_000_000 and "mainboard" not in locked_parts:
            valid_mbs = [mb for mb in all_mbs if not is_pcie3_mainboard(mb)]
            if valid_mbs:
                all_mbs = valid_mbs

        # Xử lý quy tắc chuyên biệt và cờ tùy chọn người dùng
        if require_discrete_gpu is not None:
            is_discrete_gpu = require_discrete_gpu
        else:
            is_discrete_gpu = profile.get("require_discrete_gpu", False)

        if "gpu" in locked_parts:
            if locked_parts["gpu"].get("id") != "gpu_none":
                is_discrete_gpu = True
            else:
                is_discrete_gpu = False

        if "hdd" in locked_parts and locked_parts["hdd"].get("id") != "hdd_none":
            require_hdd = True

        if is_discrete_gpu:
            gpu_list_full = [g for g in all_gpus if g["id"] != "gpu_none"]
            cpu_candidates = all_cpus
        else:
            gpu_list_full = [g for g in all_gpus if g["id"] == "gpu_none"]
            cpu_candidates = [c for c in all_cpus if c.get("has_igpu", False)]

        # Áp dụng mức sàn Card đồ họa (GPU) theo ngân sách khi bật need_vga = True
        if "gpu" not in locked_parts:
            gpu_list_full = filter_gpus_by_budget_floor(gpu_list_full, budget, usage, is_discrete_gpu)

        # =====================================================================
        # ÁP DỤNG MA TRẬN 4 NHU CẦU x 3 PHÂN KHÚC NGÂN SÁCH
        # =====================================================================
        def is_extreme_office_cpu(c_item: Dict) -> bool:
            c_name = c_item.get("name", "").upper()
            if "RYZEN 9" in c_name or "CORE I9" in c_name or "R9" in c_name or "I9" in c_name:
                return True
            if c_name.endswith("K") or c_name.endswith("KF") or c_name.endswith("KS") or " K " in c_name:
                return True
            return False

        if usage == "Văn phòng / Học tập":
            # Bắt buộc dùng CPU có iGPU tích hợp; loại bỏ CPU Workstation/Enthusiast quá tải nhiệt (Ryzen 9, Core i9, dòng K)
            if "cpu" not in locked_parts:
                cpu_candidates = [c for c in cpu_candidates if not is_extreme_office_cpu(c)]
                igpu_cpus = [c for c in cpu_candidates if c.get("has_igpu", False)]
                if igpu_cpus:
                    cpu_candidates = igpu_cpus

                if budget < 12_000_000:
                    pref = [c for c in cpu_candidates if "i3" in c.get("name", "").lower() or ("ryzen" in c.get("name", "").lower() and "g" in c.get("name", "").lower())]
                    if pref:
                        cpu_candidates = pref + [c for c in cpu_candidates if c not in pref]
                elif 12_000_000 <= budget <= 22_000_000:
                    pref = [c for c in cpu_candidates if ("i5" in c.get("name", "").lower() and c.get("has_igpu", False)) or ("ryzen" in c.get("name", "").lower() and "g" in c.get("name", "").lower())]
                    if pref:
                        cpu_candidates = pref + [c for c in cpu_candidates if c not in pref]
                else:  # budget > 22_000_000
                    pref = [c for c in cpu_candidates if c.get("has_igpu", False) and any(k in c.get("name", "").lower() for k in ["9700x", "13700", "12700", "13500"])]
                    if pref:
                        cpu_candidates = pref + [c for c in cpu_candidates if c not in pref]

        elif usage == "Gaming":
            if budget < 15_000_000 and "cpu" not in locked_parts:
                pref_entry = [
                    c for c in cpu_candidates
                    if any(k in c.get("name", "").lower() for k in ["12100f", "10400f", "3600", "5500", "12100", "10400"])
                ]
                if pref_entry:
                    cpu_candidates = pref_entry + [c for c in cpu_candidates if c not in pref_entry]
            elif 15_000_000 <= budget <= 30_000_000 and "cpu" not in locked_parts:
                # Không ghép card mạnh với Core i3 / Ryzen 3 khi ngân sách >= 15 triệu
                no_entry_cpu = [
                    c for c in cpu_candidates
                    if not any(k in c.get("name", "").lower() for k in ["i3", "ryzen 3", "r3"])
                ]
                if no_entry_cpu:
                    cpu_candidates = no_entry_cpu
                pref_mid = [
                    c for c in cpu_candidates
                    if any(k in c.get("name", "").lower() for k in ["12400f", "13400f", "5600", "5700x", "5600x", "12400", "13400"])
                ]
                if budget >= 18_000_000:
                    # Ưu tiên PCIe 4.0 (LGA1700 / AM4 B550 / AM5)
                    pcie4_cpus = [c for c in cpu_candidates if c.get("socket") in ["LGA1700", "AM4", "AM5"] and c.get("socket") != "LGA1200"]
                    if pcie4_cpus:
                        cpu_candidates = pcie4_cpus
                if pref_mid:
                    cpu_candidates = pref_mid + [c for c in cpu_candidates if c not in pref_mid]
            elif budget > 30_000_000 and "cpu" not in locked_parts:
                x3d_k = [
                    c for c in cpu_candidates
                    if "x3d" in c.get("name", "").lower()
                    or any(k in c.get("name", "").lower() for k in ["13600k", "14600k", "13700k", "14700k", "7800x3d", "5800x3d", "7600x", "9600x", "5700x"])
                ]
                if x3d_k:
                    cpu_candidates = x3d_k + [c for c in cpu_candidates if c not in x3d_k]

        elif usage == "Đồ họa / Lập trình AI":
            # Ưu tiên GPU NVIDIA có CUDA và VRAM lớn
            if "gpu" not in locked_parts and not gpu_brand:
                gpu_list_full = sorted(
                    gpu_list_full,
                    key=lambda g: (
                        1 if any(k in g.get("name", "").lower() for k in ["rtx", "geforce", "gtx"]) else 0,
                        g.get("vram_gb", 0),
                        self._score_per_price(g),
                    ),
                    reverse=True,
                )

            if budget < 18_000_000 and "cpu" not in locked_parts:
                pref_entry_ai = [
                    c for c in cpu_candidates
                    if any(k in c.get("name", "").lower() for k in ["12400f", "5600", "3700x", "12400", "5600x"])
                ]
                if pref_entry_ai:
                    cpu_candidates = pref_entry_ai + [c for c in cpu_candidates if c not in pref_entry_ai]
            elif 18_000_000 <= budget <= 32_000_000 and "cpu" not in locked_parts:
                multi_core_mid = [
                    c for c in cpu_candidates
                    if any(k in c.get("name", "").lower() for k in ["13500", "13600k", "5700x", "5900x", "14600k", "7700x", "9700x", "12700"])
                ]
                if multi_core_mid:
                    cpu_candidates = multi_core_mid + [c for c in cpu_candidates if c not in multi_core_mid]
            elif budget > 32_000_000 and "cpu" not in locked_parts:
                multi_core = [
                    c for c in cpu_candidates
                    if any(k in c.get("name", "").lower() for k in ["5900x", "5950x", "13700k", "13900k", "14700k", "14900k", "7900x", "7950x", "9900x"])
                ]
                if multi_core:
                    cpu_candidates = multi_core + [c for c in cpu_candidates if c not in multi_core]

        elif usage == "Tất cả / Đa dụng (Cân bằng)":
            if "cpu" not in locked_parts:
                if budget < 16_000_000:
                    pref_entry_bal = [
                        c for c in cpu_candidates
                        if any(k in c.get("name", "").lower() for k in ["12100f", "11400f", "5500", "12100", "10400f"])
                    ]
                    if pref_entry_bal:
                        cpu_candidates = pref_entry_bal + [c for c in cpu_candidates if c not in pref_entry_bal]
                elif 16_000_000 <= budget <= 28_000_000:
                    pref_mid_bal = [
                        c for c in cpu_candidates
                        if any(k in c.get("name", "").lower() for k in ["12400f", "13400f", "5700x", "12400", "13400", "5600"])
                    ]
                    if pref_mid_bal:
                        cpu_candidates = pref_mid_bal + [c for c in cpu_candidates if c not in pref_mid_bal]
                else:  # budget > 28_000_000
                    pref_high_bal = [
                        c for c in cpu_candidates
                        if any(k in c.get("name", "").lower() for k in ["13600k", "13700", "9700x", "14600k", "7700x", "12700"])
                    ]
                    if pref_high_bal:
                        cpu_candidates = pref_high_bal + [c for c in cpu_candidates if c not in pref_high_bal]

        # Lọc theo hãng CPU nếu có yêu cầu
        if cpu_brand and "cpu" not in locked_parts:
            cb_lower = cpu_brand.lower()
            matching_cpus = [c for c in cpu_candidates if c.get("brand", "").lower() == cb_lower]
            if matching_cpus:
                cpu_candidates = matching_cpus

        # Lọc theo hãng GPU nếu có yêu cầu
        if gpu_brand and "gpu" not in locked_parts:
            gb_lower = gpu_brand.lower()
            if gb_lower == "nvidia":
                matching_gpus = [
                    g for g in gpu_list_full
                    if "geforce" in g.get("name", "").lower()
                    or "rtx" in g.get("name", "").lower()
                    or "gtx" in g.get("name", "").lower()
                    or g["id"] == "gpu_none"
                ]
            elif gb_lower == "amd":
                matching_gpus = [
                    g for g in gpu_list_full
                    if "radeon" in g.get("name", "").lower()
                    or "rx" in g.get("name", "").lower()
                    or g["id"] == "gpu_none"
                ]
            else:
                matching_gpus = gpu_list_full
            if matching_gpus:
                gpu_list_full = matching_gpus

        # ---------------------------------------------------------------------
        # QUY TẮC RAM DUAL CHANNEL & NGƯỠNG BÃO HÒA THEO PHÂN KHÚC
        # ---------------------------------------------------------------------
        def is_dual_ram(r_item: Dict) -> bool:
            return (
                r_item.get("is_kit", False)
                or r_item.get("sticks", 1) >= 2
                or "2x" in r_item.get("name", "").lower()
            )

        if "ram" not in locked_parts:
            if budget >= 25_000_000:
                # Sàn RAM bắt buộc cho phân khúc cao: Từ 25 triệu trở lên, bắt buộc dùng tối thiểu 32GB RAM (2x16GB), cấm tụt xuống 16GB
                r_ge_32 = [r for r in all_rams if r.get("capacity_gb", 0) >= 32]
                if r_ge_32:
                    all_rams = r_ge_32
                if usage == "Đồ họa / Lập trình AI" and budget >= 30_000_000:
                    if not target_ram_gb:
                        target_ram_gb = 64
                else:
                    if not target_ram_gb:
                        target_ram_gb = 32
            elif usage in ["Văn phòng / Học tập", "Gaming", "Tất cả / Đa dụng (Cân bằng)"]:
                if budget <= 22_000_000:
                    # Ngân sách <= 22 triệu: Khóa trần ở 16GB (2x8GB)
                    r_16 = [r for r in all_rams if r.get("capacity_gb", 0) <= 16]
                    if r_16:
                        all_rams = r_16
                    if not target_ram_gb:
                        target_ram_gb = 16
                else:
                    # 22 - 25 triệu: Khóa trần ở 32GB (2x16GB)
                    r_32 = [r for r in all_rams if r.get("capacity_gb", 0) <= 32]
                    if r_32:
                        all_rams = r_32
                    if not target_ram_gb:
                        target_ram_gb = 32

            elif usage == "Đồ họa / Lập trình AI":
                if budget < 15_000_000:
                    r_16 = [r for r in all_rams if r.get("capacity_gb", 0) <= 16]
                    if r_16:
                        all_rams = r_16
                    if not target_ram_gb:
                        target_ram_gb = 16
                else:
                    r_32 = [r for r in all_rams if r.get("capacity_gb", 0) <= 32]
                    if r_32:
                        all_rams = r_32
                    if not target_ram_gb:
                        target_ram_gb = 32

            if target_ram_gb:
                if budget >= 25_000_000:
                    matching_rams = [r for r in all_rams if r.get("capacity_gb", 0) >= target_ram_gb]
                    if matching_rams:
                        all_rams = matching_rams
                else:
                    matching_rams = [r for r in all_rams if r.get("capacity_gb", 0) >= target_ram_gb]
                    if matching_rams:
                        all_rams = matching_rams + [r for r in all_rams if r not in matching_rams]
                    else:
                        matching_rams = [r for r in all_rams if r.get("capacity_gb", 0) == target_ram_gb]
                        if matching_rams:
                            all_rams = matching_rams

            # Ưu tiên tuyệt đối RAM Kit 2 thanh (Dual Channel) hơn 1 thanh đơn
            matching_dual = [r for r in all_rams if is_dual_ram(r)]
            if matching_dual:
                all_rams = matching_dual

        # ---------------------------------------------------------------------
        # QUY TẮC LƯU TRỮ SSD VÀ PHỐI HỢP HDD THEO NHU CẦU
        # ---------------------------------------------------------------------
        if "storage" not in locked_parts:
            if usage in ["Gaming", "Đồ họa / Lập trình AI"]:
                # Gaming và Lập trình AI: Luôn ưu tiên chọn SSD dung lượng lớn (1TB - 2TB NVMe)
                if budget < 18_000_000:
                    pref_ssd = [s for s in all_storages if s.get("capacity_gb", 0) >= 1000]
                    if not pref_ssd:
                        pref_ssd = [s for s in all_storages if s.get("capacity_gb", 0) >= 500]
                elif budget < 28_000_000:
                    pref_ssd = [s for s in all_storages if s.get("capacity_gb", 0) >= 1000]
                else:
                    pref_ssd = [s for s in all_storages if s.get("capacity_gb", 0) >= 2000]
                    if not pref_ssd:
                        pref_ssd = [s for s in all_storages if s.get("capacity_gb", 0) >= 1000]
                if pref_ssd:
                    all_storages = pref_ssd + [s for s in all_storages if s not in pref_ssd]

            elif usage == "Tất cả / Đa dụng (Cân bằng)":
                if require_hdd:
                    # Nhu cầu Tất cả: 50/50 giữa SSD và HDD (500GB + 500GB hoặc 1TB + 1TB)
                    if budget <= 20_000_000:
                        pref_ssd = [s for s in all_storages if s.get("capacity_gb", 0) == 512]
                    else:
                        pref_ssd = [s for s in all_storages if s.get("capacity_gb", 0) == 1000]
                    if pref_ssd:
                        all_storages = pref_ssd + [s for s in all_storages if s not in pref_ssd]
                else:
                    pref_ssd = [s for s in all_storages if s.get("capacity_gb", 0) >= 1000]
                    if pref_ssd:
                        all_storages = pref_ssd + [s for s in all_storages if s not in pref_ssd]

            else:  # Văn phòng / Học tập (Giữ nguyên)
                if require_hdd:
                    if budget <= 30_000_000:
                        ssd_512 = [s for s in all_storages if s.get("capacity_gb", 0) <= 512]
                        if ssd_512:
                            all_storages = ssd_512 + [s for s in all_storages if s not in ssd_512]
                    else:
                        ssd_1tb = [s for s in all_storages if s.get("capacity_gb", 0) <= 1000]
                        if ssd_1tb:
                            all_storages = ssd_1tb + [s for s in all_storages if s not in ssd_1tb]
                else:
                    pref_ssd = [s for s in all_storages if s.get("capacity_gb", 0) >= 500]
                    if pref_ssd:
                        all_storages = pref_ssd + [s for s in all_storages if s not in pref_ssd]

            if min_storage_gb:
                matching_storages = [s for s in all_storages if s.get("capacity_gb", 0) >= min_storage_gb]
                if matching_storages:
                    all_storages = matching_storages

        cpu_list = [locked_parts["cpu"]] if "cpu" in locked_parts else cpu_candidates
        gpu_list = [locked_parts["gpu"]] if "gpu" in locked_parts else gpu_list_full
        if "mainboard" in locked_parts:
            all_mbs = [locked_parts["mainboard"]]
        if "ram" in locked_parts:
            all_rams = [locked_parts["ram"]]
        if "psu" in locked_parts:
            all_psus = [locked_parts["psu"]]
        if "case" in locked_parts:
            all_cases = [locked_parts["case"]]
        if "storage" in locked_parts:
            all_storages = [locked_parts["storage"]]

        min_prices = {
            "mainboard": self._get_min_price(all_mbs),
            "ram": self._get_min_price(all_rams),
            "gpu": self._get_min_price(gpu_list),
            "psu": self._get_min_price(all_psus),
            "case": self._get_min_price(all_cases),
            "storage": self._get_min_price(all_storages),
        }

        w_cpu = profile["w_cpu"]
        w_gpu = profile["w_gpu"]
        w_ram = profile["w_ram"]
        w_storage = profile["w_storage"]
        w_dur = profile.get("w_durability", 0.05)

        best_builds: List[BuildResult] = []
        best_score = -1.0

        def branch(
            level: int,
            remaining_budget: int,
            current: Dict[str, Dict],
            current_score: float,
        ):
            nonlocal best_score
            stats.nodes_visited += 1

            # Tầng 0: CPU
            if level == 0:
                for cpu in cpu_list:
                    if cpu["price"] > remaining_budget:
                        stats.pruned_branches += 1
                        continue

                    min_remaining = sum(min_prices.values())
                    if cpu["price"] + min_remaining > max_search_budget:
                        stats.pruned_branches += 1
                        continue

                    cpu_norm = min((cpu["score"] / self.cpu_max_score) * 100, 100.0)
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
                    cpu_contrib = w_cpu * cpu_norm

                    max_gpu = self._get_max_norm_score(gpu_list, "gpus")
                    upper_bound = cpu_contrib + w_gpu * max_gpu + w_ram * 100.0 + w_storage * 100.0 + w_dur * 100.0
                    if upper_bound <= best_score and len(best_builds) >= top_k:
                        stats.pruned_branches += 1
                        continue

                    current["cpu"] = cpu
                    branch(1, remaining_budget - cpu["price"], current, cpu_contrib)
                return

            # Tầng 1: Bo mạch chủ
            if level == 1:
                cpu = current["cpu"]
                if "mainboard" in locked_parts:
                    locked_mb = locked_parts["mainboard"]
                    if (
                        locked_mb["socket"] == cpu["socket"]
                        and locked_mb["ram_type"] == cpu["ram_type"]
                        and not is_vrm_incompatible(cpu, locked_mb)
                    ):
                        valid_mbs = [locked_mb]
                    else:
                        valid_mbs = []
                else:
                    valid_mbs = [
                        mb for mb in all_mbs
                        if mb["socket"] == cpu["socket"]
                        and mb["ram_type"] == cpu["ram_type"]
                        and not is_vrm_incompatible(cpu, mb)
                    ]
                    if budget >= 18_000_000:
                        valid_mbs = [mb for mb in valid_mbs if not is_pcie3_mainboard(mb)]
                if not valid_mbs:
                    stats.pruned_branches += 1
                    return

                for mb in valid_mbs:
                    cost_so_far = max_search_budget - remaining_budget + mb["price"]
                    min_remaining = sum(
                        min_prices[k] for k in ["ram", "gpu", "psu", "case", "storage"]
                    )
                    if cost_so_far + min_remaining > max_search_budget:
                        stats.pruned_branches += 1
                        continue

                    current["mainboard"] = mb
                    branch(2, remaining_budget - mb["price"], current, current_score)
                return

            # Tầng 2: RAM
            if level == 2:
                cpu = current["cpu"]
                mb = current["mainboard"]
                if "ram" in locked_parts:
                    locked_ram = locked_parts["ram"]
                    if (
                        locked_ram["ram_type"] == cpu["ram_type"]
                        and locked_ram["capacity_gb"] <= mb["max_ram_gb"]
                    ):
                        valid_rams = [locked_ram]
                    else:
                        valid_rams = []
                else:
                    valid_rams = [
                        r for r in all_rams
                        if r["ram_type"] == cpu["ram_type"] and r["capacity_gb"] <= mb["max_ram_gb"]
                    ]
                    if budget >= 25_000_000:
                        valid_rams = [r for r in valid_rams if r.get("capacity_gb", 0) >= 32]
                if not valid_rams:
                    stats.pruned_branches += 1
                    return

                for ram in valid_rams:
                    cost_so_far = max_search_budget - remaining_budget + ram["price"]
                    min_remaining = sum(
                        min_prices[k] for k in ["gpu", "psu", "case", "storage"]
                    )
                    if cost_so_far + min_remaining > max_search_budget:
                        stats.pruned_branches += 1
                        continue

                    ram_norm = compute_ram_norm(ram, usage, budget=budget)
                    ram_contrib = current_score + w_ram * ram_norm
                    max_gpu = self._get_max_norm_score(gpu_list, "gpus")
                    upper_bound = ram_contrib + w_gpu * max_gpu + w_storage * 100.0 + w_dur * 100.0
                    if upper_bound <= best_score and len(best_builds) >= top_k:
                        stats.pruned_branches += 1
                        continue

                    current["ram"] = ram
                    branch(3, remaining_budget - ram["price"], current, ram_contrib)
                return

            # Tầng 3: Card đồ họa
            if level == 3:
                cpu = current["cpu"]
                for gpu in gpu_list:
                    if gpu["id"] == "gpu_none" and not cpu.get("has_igpu", False):
                        stats.pruned_branches += 1
                        continue

                    cost_so_far = max_search_budget - remaining_budget + gpu["price"]
                    min_remaining = sum(min_prices[k] for k in ["psu", "case", "storage"])
                    if cost_so_far + min_remaining > max_search_budget:
                        stats.pruned_branches += 1
                        continue

                    gpu_norm = compute_gpu_norm(gpu, usage, gpu_max_score=self.gpu_max_score)
                    gpu_contrib = current_score + w_gpu * gpu_norm
                    upper_bound = gpu_contrib + w_storage * 100.0 + w_dur * 100.0
                    if upper_bound <= best_score and len(best_builds) >= top_k:
                        stats.pruned_branches += 1
                        continue

                    current["gpu"] = gpu
                    branch(4, remaining_budget - gpu["price"], current, gpu_contrib)
                return

            # Tầng 4: Bộ nguồn PSU
            if level == 4:
                cpu = current["cpu"]
                gpu = current["gpu"]
                min_w = calculate_min_psu_wattage(
                    cpu, gpu, extra_storage_count=1 if (require_hdd or "hdd" in locked_parts) else 0
                )
                if "psu" in locked_parts:
                    locked_psu = locked_parts["psu"]
                    if locked_psu["wattage"] >= min_w:
                        valid_psus = [locked_psu]
                    else:
                        valid_psus = []
                else:
                    min_w_tier = calculate_tier_min_psu_wattage(
                        cpu, gpu, budget=budget, extra_storage_count=1 if require_hdd else 0, usage=usage
                    )
                    valid_psus = [p for p in all_psus if p["wattage"] >= min_w_tier]
                    if not valid_psus:
                        valid_psus = [p for p in all_psus if p["wattage"] >= min_w]
                    if not valid_psus:
                        stats.pruned_branches += 1
                        return

                    # Nếu ở phân khúc cao hoặc theo chuẩn ma trận nhu cầu, ưu tiên nguồn chuẩn Gold
                    need_gold_psu = (
                        (usage == "Văn phòng / Học tập" and budget > 22_000_000)
                        or (usage == "Gaming" and budget > 30_000_000)
                        or (("ĐỒ HỌA" in usage.upper() or "AI" in usage.upper()) and budget > 32_000_000)
                        or (("TẤT CẢ" in usage.upper() or "ĐA DỤNG" in usage.upper()) and budget > 28_000_000)
                        or budget >= 35_000_000
                    )
                    if need_gold_psu:
                        gold_psus = [p for p in valid_psus if "Gold" in p.get("efficiency", "") or "Platinum" in p.get("efficiency", "")]
                        if gold_psus:
                            valid_psus = gold_psus

                if not valid_psus:
                    stats.pruned_branches += 1
                    return

                for psu in valid_psus:
                    cost_so_far = max_search_budget - remaining_budget + psu["price"]
                    min_remaining = sum(min_prices[k] for k in ["case", "storage"])
                    if cost_so_far + min_remaining > max_search_budget:
                        stats.pruned_branches += 1
                        continue

                    current["psu"] = psu
                    branch(5, remaining_budget - psu["price"], current, current_score)
                return

            # Tầng 5: Vỏ máy Case
            if level == 5:
                mb = current["mainboard"]
                gpu = current["gpu"]
                if "case" in locked_parts:
                    locked_case = locked_parts["case"]
                    if (
                        mb["form_factor"] in locked_case["supported_form_factors"]
                        and locked_case["max_gpu_length_mm"] >= gpu["length_mm"]
                    ):
                        valid_cases = [locked_case]
                    else:
                        valid_cases = []
                else:
                    valid_cases = [
                        c for c in all_cases
                        if mb["form_factor"] in c["supported_form_factors"]
                        and c["max_gpu_length_mm"] >= gpu["length_mm"]
                    ]
                    need_case_800k = (
                        (usage == "Văn phòng / Học tập" and budget > 22_000_000)
                        or (usage == "Gaming" and budget > 30_000_000)
                        or (("TẤT CẢ" in usage.upper() or "ĐA DỤNG" in usage.upper()) and budget >= 20_000_000)
                        or (("ĐỒ HỌA" in usage.upper() or "AI" in usage.upper()) and budget >= 30_000_000)
                    )
                    if need_case_800k:
                        cases_800k = [c for c in valid_cases if c.get("price", 0) >= 800_000]
                        if cases_800k:
                            valid_cases = cases_800k

                if not valid_cases:
                    stats.pruned_branches += 1
                    return

                for case in valid_cases:
                    cost_so_far = max_search_budget - remaining_budget + case["price"]
                    if cost_so_far + min_prices["storage"] > max_search_budget:
                        stats.pruned_branches += 1
                        continue

                    current["case"] = case
                    branch(6, remaining_budget - case["price"], current, current_score)
                return

            # Tầng 6: Ổ cứng SSD
            if level == 6:
                for storage in all_storages:
                    if storage["price"] > remaining_budget:
                        stats.pruned_branches += 1
                        continue

                    # Tự động trang bị Tản nhiệt tương thích theo quy chuẩn an toàn phần cứng
                    if "cooler" in locked_parts:
                        cooler_item = deepcopy(locked_parts["cooler"])
                    else:
                        coolers_list = self.data.get("coolers", [])
                        cooler_item = select_optimal_cooler(coolers_list, current["cpu"], budget=budget, usage=usage)

                    # Tự động gán HDD phụ theo yêu cầu
                    if "hdd" in locked_parts:
                        hdd_item = deepcopy(locked_parts["hdd"])
                    else:
                        hdds_list = self.data.get("hdds", [])
                        hdd_item = None
                        if hdds_list:
                            if require_hdd:
                                real_hdds = [h for h in hdds_list if h.get("id") != "hdd_none"]

                                if usage in ["Gaming", "Đồ họa / Lập trình AI"]:
                                    # Gaming và Lập trình AI: Ưu tiên SSD, nếu có tích HDD thì chỉ chọn cỡ 1/3 so với SSD
                                    # Nếu ngân sách từ 18 triệu trở lên, SSD phải từ 1000GB (1TB) trở lên để HDD 500GB đạt tỷ lệ 1/3 - 1/2
                                    if budget >= 18_000_000 and storage.get("capacity_gb", 0) < 1000:
                                        stats.pruned_branches += 1
                                        continue
                                    hdds_one_third = [h for h in real_hdds if h.get("capacity_gb", 0) <= 500]
                                    if not hdds_one_third and storage.get("capacity_gb", 0) >= 2000:
                                        hdds_one_third = [h for h in real_hdds if h.get("capacity_gb", 0) <= 1000]
                                    if hdds_one_third:
                                        real_hdds = hdds_one_third

                                elif usage == "Tất cả / Đa dụng (Cân bằng)":
                                    # Nhu cầu Tất cả thì 50/50: HDD dung lượng tương đương SSD
                                    ssd_cap = storage.get("capacity_gb", 512)
                                    matching_5050 = [h for h in real_hdds if abs(h.get("capacity_gb", 0) - ssd_cap) <= 100]
                                    if matching_5050:
                                        real_hdds = matching_5050

                                else:  # Văn phòng / Học tập (Giữ nguyên)
                                    if budget >= 12_000_000:
                                        hdds_ge_1tb = [h for h in real_hdds if h.get("capacity_gb", 0) >= 1000]
                                        if hdds_ge_1tb:
                                            real_hdds = hdds_ge_1tb
                                    if budget > 22_000_000:
                                        hdds_2tb = [h for h in real_hdds if h.get("capacity_gb", 0) >= 2000]
                                        if hdds_2tb:
                                            real_hdds = hdds_2tb

                                if min_total_storage_gb:
                                    need_hdd_cap = max(0, min_total_storage_gb - storage.get("capacity_gb", 0))
                                    eligible_hdds = [h for h in real_hdds if h.get("capacity_gb", 0) >= need_hdd_cap]
                                    if not eligible_hdds:
                                        stats.pruned_branches += 1
                                        continue
                                    hdd_item = deepcopy(min(eligible_hdds, key=lambda h: h.get("price", 0)))
                                else:
                                    hdd_item = deepcopy(min(real_hdds, key=lambda h: h.get("price", 0)) if real_hdds else hdds_list[0])
                        else:
                            if min_total_storage_gb and storage.get("capacity_gb", 0) < min_total_storage_gb:
                                stats.pruned_branches += 1
                                continue
                            hdd_none = next((h for h in hdds_list if h.get("id") == "hdd_none"), hdds_list[0])
                            hdd_item = deepcopy(hdd_none)

                    extra_cost = (cooler_item.get("price", 0) if cooler_item else 0) + (hdd_item.get("price", 0) if hdd_item else 0)
                    total_price = (max_search_budget - remaining_budget) + storage["price"] + extra_cost
                    if total_price > max_search_budget:
                        stats.pruned_branches += 1
                        continue

                    final_score = compute_fitness_score(
                        current["cpu"],
                        current["gpu"],
                        current["ram"],
                        storage,
                        usage,
                        cpu_max_score=self.cpu_max_score,
                        gpu_max_score=self.gpu_max_score,
                        budget=budget,
                        need_hdd=require_hdd,
                        mainboard=current["mainboard"],
                        psu=current["psu"],
                        case=current["case"],
                        cooler=cooler_item,
                        hdd=hdd_item,
                    )

                    # Phạt nhẹ khi vượt ngân sách trong biên độ linh hoạt (+5%)
                    if total_price > budget:
                        exceeded_ratio = (total_price - budget) / float(budget)
                        final_score = round(final_score - (exceeded_ratio * 10.0), 4)

                    if final_score <= best_score and len(best_builds) >= top_k:
                        stats.pruned_branches += 1
                        continue

                    stats.valid_builds_found += 1
                    build = BuildResult(
                        cpu=deepcopy(current["cpu"]),
                        mainboard=deepcopy(current["mainboard"]),
                        ram=deepcopy(current["ram"]),
                        gpu=deepcopy(current["gpu"]),
                        psu=deepcopy(current["psu"]),
                        case=deepcopy(current["case"]),
                        storage=deepcopy(storage),
                        cooler=cooler_item,
                        hdd=hdd_item,
                        total_price=total_price,
                        fitness_score=final_score,
                    )

                    best_builds.append(build)
                    best_builds.sort(key=lambda b: b.fitness_score, reverse=True)
                    if len(best_builds) > top_k:
                        best_builds.pop()
                    if best_builds:
                        best_score = best_builds[-1].fitness_score

        branch(0, max_search_budget, {}, 0.0)

        end_time = time.perf_counter()
        stats.execution_time_ms = (end_time - start_time) * 1000
        return best_builds, stats

    def find_minimum_budget(
        self,
        usage: str = "Văn phòng / Học tập",
        locked_parts: Optional[Dict[str, Dict]] = None,
        cpu_brand: Optional[str] = None,
        gpu_brand: Optional[str] = None,
        require_discrete_gpu: Optional[bool] = None,
        require_hdd: bool = False,
        target_ram_gb: Optional[int] = None,
        min_storage_gb: Optional[int] = None,
        min_total_storage_gb: Optional[int] = None,
        prefer_dual_ram: bool = False,
    ) -> int:
        locked_parts = locked_parts or {}
        profile = USAGE_PROFILES.get(usage, USAGE_PROFILES.get("Tất cả / Đa dụng (Cân bằng)", {}))
        data = self.data

        cpus = data.get("cpus", [])
        if "cpu" in locked_parts:
            cpus = [locked_parts["cpu"]]
        elif cpu_brand:
            m_cpus = [c for c in cpus if c.get("brand", "").lower() == cpu_brand.lower()]
            if m_cpus:
                cpus = m_cpus

        gpus = data.get("gpus", [])
        if "gpu" in locked_parts:
            gpus = [locked_parts["gpu"]]
        else:
            if require_discrete_gpu is not None:
                if require_discrete_gpu:
                    gpus = [g for g in gpus if g.get("id") != "gpu_none"]
                else:
                    gpus = [g for g in gpus if g.get("id") == "gpu_none"]
            elif profile.get("prefer_igpu_only", False):
                gpus = [g for g in gpus if g.get("id") == "gpu_none"]
            elif profile.get("require_discrete_gpu", False):
                gpus = [g for g in gpus if g.get("id") != "gpu_none"]

            if usage == "Văn phòng / Học tập" and require_discrete_gpu:
                gpus = [g for g in gpus if g.get("id") != "gpu_none" and g.get("price", 0) <= 6_000_000]

            if gpu_brand:
                gb_l = gpu_brand.lower()
                if gb_l == "nvidia":
                    m_gpus = [g for g in gpus if "geforce" in g.get("name", "").lower() or "rtx" in g.get("name", "").lower() or "gtx" in g.get("name", "").lower() or g["id"] == "gpu_none"]
                elif gb_l == "amd":
                    m_gpus = [g for g in gpus if "radeon" in g.get("name", "").lower() or "rx" in g.get("name", "").lower() or g["id"] == "gpu_none"]
                else:
                    m_gpus = gpus
                if m_gpus:
                    gpus = m_gpus

        storages = data.get("storages", [])
        if min_storage_gb:
            m_storages = [s for s in storages if s.get("capacity_gb", 0) >= min_storage_gb]
            if m_storages:
                storages = m_storages

        min_total = float("inf")
        for cpu in sorted(cpus, key=lambda x: x["price"])[:6]:
            valid_mbs = [
                mb for mb in data.get("mainboards", [])
                if mb["socket"] == cpu["socket"]
                and mb["ram_type"] == cpu["ram_type"]
                and not is_vrm_incompatible(cpu, mb)
            ]
            if not valid_mbs:
                continue
            mb = min(valid_mbs, key=lambda x: x["price"])

            valid_rams = [
                r for r in data.get("rams", [])
                if r["ram_type"] == cpu["ram_type"]
                and r["capacity_gb"] <= mb["max_ram_gb"]
                and (not target_ram_gb or r["capacity_gb"] >= target_ram_gb)
                and (not prefer_dual_ram or r.get("is_kit", False) or r.get("sticks", 1) >= 2 or "2x" in r.get("name", "").lower())
            ]
            if not valid_rams:
                valid_rams = [
                    r for r in data.get("rams", [])
                    if r["ram_type"] == cpu["ram_type"]
                    and r["capacity_gb"] <= mb["max_ram_gb"]
                    and (not target_ram_gb or r["capacity_gb"] >= target_ram_gb)
                ]
            if not valid_rams:
                continue
            ram = min(valid_rams, key=lambda x: x["price"])

            # Cooler
            cooler = select_optimal_cooler(data.get("coolers", []), cpu, budget=0)
            if not cooler:
                continue

            for gpu in gpus:
                valid_cases = [
                    c for c in data.get("cases", [])
                    if mb["form_factor"] in c["supported_form_factors"]
                    and c["max_gpu_length_mm"] >= gpu["length_mm"]
                ]
                if not valid_cases:
                    continue
                case = min(valid_cases, key=lambda x: x["price"])

                min_w = calculate_tier_min_psu_wattage(cpu, gpu, budget=0, extra_storage_count=1 if require_hdd else 0, usage=usage)
                valid_psus = [p for p in data.get("psus", []) if p["wattage"] >= min_w]
                if not valid_psus:
                    valid_psus = [p for p in data.get("psus", []) if p["wattage"] >= calculate_min_psu_wattage(cpu, gpu, extra_storage_count=1 if require_hdd else 0)]
                if not valid_psus:
                    continue
                psu = min(valid_psus, key=lambda x: x["price"])

                for ssd in storages:
                    if require_hdd:
                        real_hdds = [h for h in data.get("hdds", []) if h.get("id") != "hdd_none"]
                        needed_hdd = max(0, (min_total_storage_gb or 0) - ssd.get("capacity_gb", 0))
                        valid_hdds = [h for h in real_hdds if h.get("capacity_gb", 0) >= needed_hdd]
                        if not valid_hdds:
                            continue
                        hdd = min(valid_hdds, key=lambda x: x["price"])
                    else:
                        if min_total_storage_gb and ssd.get("capacity_gb", 0) < min_total_storage_gb:
                            continue
                        hdd = next((h for h in data.get("hdds", []) if h.get("id") == "hdd_none"), None)

                    extra_h = hdd["price"] if hdd else 0
                    total = sum([
                        cpu["price"], mb["price"], ram["price"], gpu["price"],
                        psu["price"], case["price"], ssd["price"], cooler.get("price", 0), extra_h
                    ])
                    if total < min_total:
                        min_total = total

        return int(min_total) if min_total != float("inf") else 0


# ===========================================================================
# PHẦN 3: THUẬT TOÁN VÉT CẠN (Brute-Force Backtracking)
# ===========================================================================

class BruteForceOptimizer:
    """
    Thuật toán Vét cạn Backtracking dùng để đối chiếu Benchmark.
    Chỉ kiểm tra tính hợp lệ và ngân sách ở nút lá, không cắt tỉa cận trên/dưới.
    """

    def __init__(self, data: Dict[str, List[Dict]]):
        self.data = data
        self.cpu_max_score = max((c["score"] for c in data.get("cpus", [])), default=50000)
        self.gpu_max_score = max(
            (g["score"] for g in data.get("gpus", []) if g["id"] != "gpu_none"), default=40000
        )

    def optimize(
        self,
        budget: int,
        usage: str = "Văn phòng / Học tập",
        locked_parts: Optional[Dict[str, Dict]] = None,
        top_k: int = 3,
        items_per_category: int = 4,
        require_discrete_gpu: Optional[bool] = None,
        require_hdd: bool = False,
        strict_budget: bool = True,
    ) -> Tuple[List[BuildResult], OptimizerStats]:
        if locked_parts is None:
            locked_parts = {}

        start_time = time.perf_counter()
        profile = USAGE_PROFILES.get(usage, USAGE_PROFILES["Văn phòng / Học tập"])
        stats = OptimizerStats(algorithm_name="Brute-Force Backtracking")
        max_search_budget = budget if strict_budget else int(budget * 1.05)

        n = items_per_category

        if require_discrete_gpu is not None:
            is_discrete_gpu = require_discrete_gpu
        else:
            is_discrete_gpu = profile.get("require_discrete_gpu", False)

        if "gpu" in locked_parts:
            if locked_parts["gpu"].get("id") != "gpu_none":
                is_discrete_gpu = True
            else:
                is_discrete_gpu = False

        if "hdd" in locked_parts and locked_parts["hdd"].get("id") != "hdd_none":
            require_hdd = True

        raw_cpus = self.data.get("cpus", [])
        if usage == "Văn phòng / Học tập" and "cpu" not in locked_parts:
            def is_extreme_office_cpu(c_item: Dict) -> bool:
                c_name = c_item.get("name", "").upper()
                if "RYZEN 9" in c_name or "CORE I9" in c_name or "R9" in c_name or "I9" in c_name:
                    return True
                if c_name.endswith("K") or c_name.endswith("KF") or c_name.endswith("KS") or " K " in c_name:
                    return True
                return False

            raw_cpus = [c for c in raw_cpus if not is_extreme_office_cpu(c)]
        if (profile.get("prefer_igpu_only", False) or not is_discrete_gpu) and "cpu" not in locked_parts:
            raw_cpus = [c for c in raw_cpus if c.get("has_igpu", False)]

        cpu_list = [locked_parts["cpu"]] if "cpu" in locked_parts else raw_cpus[:n]
        raw_mbs = self.data.get("mainboards", [])
        if budget >= 18_000_000 and "mainboard" not in locked_parts:
            valid_mbs = [m for m in raw_mbs if not is_pcie3_mainboard(m)]
            mb_list = valid_mbs[:n] if valid_mbs else raw_mbs[:n]
        else:
            mb_list = raw_mbs[:n]

        raw_rams = self.data.get("rams", [])
        def is_dual_ram(r_item: Dict) -> bool:
            return (
                r_item.get("is_kit", False)
                or r_item.get("sticks", 1) >= 2
                or "2x" in r_item.get("name", "").lower()
            )

        if budget >= 25_000_000:
            r_ge_32 = [r for r in raw_rams if r.get("capacity_gb", 0) >= 32]
            if r_ge_32:
                raw_rams = r_ge_32
        elif usage in ["Văn phòng / Học tập", "Gaming", "Tất cả / Đa dụng (Cân bằng)"]:
            if budget <= 22_000_000:
                r_filtered = [r for r in raw_rams if r.get("capacity_gb", 0) <= 16]
            else:
                r_filtered = [r for r in raw_rams if r.get("capacity_gb", 0) <= 32]
            if r_filtered:
                raw_rams = r_filtered
        elif usage == "Đồ họa / Lập trình AI":
            if budget < 15_000_000:
                r_filtered = [r for r in raw_rams if r.get("capacity_gb", 0) <= 16]
            else:
                r_filtered = [r for r in raw_rams if r.get("capacity_gb", 0) <= 32]
            if r_filtered:
                raw_rams = r_filtered

        matching_dual = [r for r in raw_rams if is_dual_ram(r)]
        if matching_dual:
            raw_rams = matching_dual

        ram_list = [locked_parts["ram"]] if "ram" in locked_parts else raw_rams[:n]

        raw_gpus = self.data.get("gpus", [])
        if "gpu" not in locked_parts:
            raw_gpus = filter_gpus_by_budget_floor(raw_gpus, budget, usage, is_discrete_gpu)

        gpu_list = [locked_parts["gpu"]] if "gpu" in locked_parts else raw_gpus[:n]
        psu_list = self.data.get("psus", [])[:n]
        case_list = self.data.get("cases", [])[:n]
        storage_list = self.data.get("storages", [])[:n]

        best_builds: List[BuildResult] = []

        for cpu in cpu_list:
            for mb in mb_list:
                ok_r01, _ = check_r01_socket(cpu, mb)
                ok_r02_cpu = cpu["ram_type"] == mb["ram_type"]
                if not (ok_r01 and ok_r02_cpu):
                    stats.nodes_visited += 1
                    continue

                for ram in ram_list:
                    ok_r02, _ = check_r02_ram_type(cpu, mb, ram)
                    ok_r03, _ = check_r03_ram_capacity(mb, ram)
                    if not (ok_r02 and ok_r03):
                        stats.nodes_visited += 1
                        continue

                    for gpu in gpu_list:
                        ok_r07, _ = check_r07_display_output(cpu, gpu)
                        if not ok_r07:
                            stats.nodes_visited += 1
                            continue

                        for psu in psu_list:
                            ok_r06, _ = check_r06_psu_wattage(cpu, gpu, psu)
                            if not ok_r06:
                                stats.nodes_visited += 1
                                continue

                            for case in case_list:
                                ok_r04, _ = check_r04_form_factor(mb, case)
                                ok_r05, _ = check_r05_gpu_length(gpu, case)
                                if not (ok_r04 and ok_r05):
                                    stats.nodes_visited += 1
                                    continue

                                for storage in storage_list:
                                    stats.nodes_visited += 1
                                    coolers_list = self.data.get("coolers", [])
                                    cooler_item = select_optimal_cooler(coolers_list, cpu, budget=budget, usage=usage) if coolers_list else None

                                    hdds_list = self.data.get("hdds", [])
                                    hdd_item = None
                                    if hdds_list:
                                        if require_hdd:
                                            real_hdds = [h for h in hdds_list if h.get("id") != "hdd_none"]
                                            if usage in ["Gaming", "Đồ họa / Lập trình AI"]:
                                                if budget >= 18_000_000 and storage.get("capacity_gb", 0) < 1000:
                                                    continue
                                                hdds_one_third = [h for h in real_hdds if h.get("capacity_gb", 0) <= 500]
                                                if not hdds_one_third and storage.get("capacity_gb", 0) >= 2000:
                                                    hdds_one_third = [h for h in real_hdds if h.get("capacity_gb", 0) <= 1000]
                                                if hdds_one_third:
                                                    real_hdds = hdds_one_third
                                            elif usage == "Tất cả / Đa dụng (Cân bằng)":
                                                ssd_cap = storage.get("capacity_gb", 512)
                                                matching_5050 = [h for h in real_hdds if abs(h.get("capacity_gb", 0) - ssd_cap) <= 100]
                                                if matching_5050:
                                                    real_hdds = matching_5050
                                            else:
                                                if budget >= 12_000_000:
                                                    hdds_ge_1tb = [h for h in real_hdds if h.get("capacity_gb", 0) >= 1000]
                                                    if hdds_ge_1tb:
                                                        real_hdds = hdds_ge_1tb
                                                if budget > 22_000_000:
                                                    hdds_2tb = [h for h in real_hdds if h.get("capacity_gb", 0) >= 2000]
                                                    if hdds_2tb:
                                                        real_hdds = hdds_2tb
                                            hdd_item = deepcopy(min(real_hdds, key=lambda h: h.get("price", 0)) if real_hdds else hdds_list[0])
                                        else:
                                            hdd_none = next((h for h in hdds_list if h.get("id") == "hdd_none"), hdds_list[0])
                                            hdd_item = deepcopy(hdd_none)

                                    extra_cost = (cooler_item.get("price", 0) if cooler_item else 0) + (hdd_item.get("price", 0) if hdd_item else 0)
                                    total = sum([
                                        cpu["price"], mb["price"], ram["price"],
                                        gpu["price"], psu["price"], case["price"],
                                        storage["price"]
                                    ]) + extra_cost

                                    if total > max_search_budget:
                                        continue

                                    score = compute_fitness_score(
                                        cpu, gpu, ram, storage, usage,
                                        cpu_max_score=self.cpu_max_score,
                                        gpu_max_score=self.gpu_max_score,
                                        budget=budget,
                                        need_hdd=require_hdd,
                                        mainboard=mb,
                                        psu=psu,
                                        case=case,
                                        cooler=cooler_item,
                                        hdd=hdd_item,
                                    )

                                    if total > budget:
                                        exceeded_ratio = (total - budget) / float(budget)
                                        score = round(score - (exceeded_ratio * 10.0), 4)

                                    build = BuildResult(
                                        cpu=cpu, mainboard=mb, ram=ram, gpu=gpu,
                                        psu=psu, case=case, storage=storage,
                                        total_price=total, fitness_score=score,
                                        cooler=cooler_item,
                                        hdd=hdd_item,
                                    )

                                    best_builds.append(build)
                                    stats.valid_builds_found += 1

        end_time = time.perf_counter()
        stats.execution_time_ms = (end_time - start_time) * 1000

        best_builds.sort(key=lambda b: b.fitness_score, reverse=True)
        return best_builds[:top_k], stats


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

