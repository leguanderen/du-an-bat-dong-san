"""
extract_text.py — Trích xuất thông tin có cấu trúc từ tiêu đề và mô tả tin đăng.

LÝ DO TỒN TẠI CỦA MODULE NÀY
----------------------------
Người đăng tin thường bỏ trống các ô thuộc tính của nhatot nhưng lại viết đầy đủ
mọi thứ trong phần mô tả tự do. Ví dụ với chung cư, ô "Tầng số" chỉ được điền ở
29% số tin — nhưng rất nhiều tin viết "căn hộ tầng 12" ngay trong mô tả.

Module này KHÔNG thay thế giá trị đã có. Nó chỉ VÁ vào chỗ đang trống.
Nguyên tắc: dữ liệu người dùng khai qua ô có cấu trúc luôn đáng tin hơn dữ liệu
ta đoán từ văn bản tự do.

Mọi hàm ở đây đều trả về NaN khi không chắc, thay vì đoán bừa.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from clean_common import strip_accents

# =============================================================================
# TẦNG SỐ (chung cư)
# =============================================================================
# Các cách người Việt hay viết: "tầng 12", "t12", "tang 8", "căn 1205" (tầng 12).
# Cố tình KHÔNG bắt "căn 1205" vì quy ước đánh số căn khác nhau giữa các toà —
# suy ra tầng từ mã căn là đoán mò, dễ sai hơn là để trống.

_RE_TANG = [
    re.compile(r"\btang\s*(\d{1,2})\b"),          # "tầng 12"
    re.compile(r"\bt\s*(\d{1,2})\b(?!\s*t)"),     # "t12" (tránh khớp "5t5")
]
_TANG_MIN, _TANG_MAX = 1, 60


def extract_floor(text: object) -> float:
    """Rút số tầng từ văn bản. NaN nếu không tìm thấy hoặc ngoài khoảng hợp lý."""
    if text is None:
        return np.nan
    flat = strip_accents(text)
    for pat in _RE_TANG:
        m = pat.search(flat)
        if m:
            v = int(m.group(1))
            if _TANG_MIN <= v <= _TANG_MAX:
                return float(v)
    return np.nan


# =============================================================================
# TỔNG SỐ TẦNG (nhà đất)
# =============================================================================
# "nhà 5 tầng", "xây 5T", "5 tang", "4,5 tầng" (tầng lửng tính là 0,5)

_RE_TONG_TANG = [
    re.compile(r"(\d{1,2}(?:[.,]5)?)\s*tang\b"),
    re.compile(r"\bxay\s*(\d{1,2})\s*t\b"),
    re.compile(r"\b(\d{1,2})\s*t\b(?=\s*[,.\-–|]|$)"),
]


def extract_num_floors(text: object) -> float:
    if text is None:
        return np.nan
    flat = strip_accents(text)
    for pat in _RE_TONG_TANG:
        m = pat.search(flat)
        if m:
            try:
                v = float(m.group(1).replace(",", "."))
            except ValueError:
                continue
            if 1 <= v <= 20:
                return v
    return np.nan


# =============================================================================
# MẶT TIỀN / CHIỀU NGANG (nhà đất)
# =============================================================================
# "MT 4m", "mặt tiền 4,15m", "mt: 4.2 m", "4m x 12m" (số đầu là mặt tiền).

_RE_MT = [
    re.compile(r"(?:mat tien|\bmt\b|chieu ngang)\s*:?\s*(\d{1,2}(?:[.,]\d{1,2})?)\s*m"),
    # Mẫu "4m x 12m" / "5 x 12". Phần (?!2) rất quan trọng: nó chặn mẫu
    # "45m2 x 5 tang" (diện tích × số tầng) — nếu không chặn, một căn 45 m² sẽ
    # bị gán mặt tiền 45 m. Đây là lỗi tìm ra khi soi lại kết quả trích xuất:
    # 37 dòng có mặt tiền lớn hơn cả cạnh hình vuông của lô đất.
    re.compile(r"\b(\d{1,2}(?:[.,]\d{1,2})?)\s*m?(?!2|²)\s*[x×]\s*\d{1,3}(?:[.,]\d{1,2})?\s*m(?!2|²)"),
]
_MT_MIN, _MT_MAX = 1.5, 50.0


def extract_frontage(text: object) -> float:
    """Rút mặt tiền (m). Khoảng hợp lý 1,5–50 m; ngoài khoảng coi như đọc nhầm."""
    if text is None:
        return np.nan
    flat = strip_accents(text)
    for pat in _RE_MT:
        m = pat.search(flat)
        if m:
            try:
                v = float(m.group(1).replace(",", "."))
            except ValueError:
                continue
            if _MT_MIN <= v <= _MT_MAX:
                return v
    return np.nan


def extract_depth(text: object) -> float:
    """Rút chiều dài (m) từ mẫu "4m x 12m" — lấy số THỨ HAI."""
    if text is None:
        return np.nan
    flat = strip_accents(text)
    m = re.search(r"\b\d{1,2}(?:[.,]\d{1,2})?\s*m?\s*[x×]\s*(\d{1,3}(?:[.,]\d{1,2})?)\s*m", flat)
    if m:
        try:
            v = float(m.group(1).replace(",", "."))
        except ValueError:
            return np.nan
        if 2.0 <= v <= 200.0:
            return v
    return np.nan


# =============================================================================
# ĐỘ RỘNG ĐƯỜNG / NGÕ (nhà đất)
# =============================================================================

_RE_DUONG = re.compile(
    r"(?:ngo|duong|hem|ngach)\s*(?:truoc nha\s*)?(?:rong\s*)?:?\s*(\d{1,2}(?:[.,]\d{1,2})?)\s*m"
)


def extract_road_width(text: object) -> float:
    if text is None:
        return np.nan
    m = _RE_DUONG.search(strip_accents(text))
    if m:
        try:
            v = float(m.group(1).replace(",", "."))
        except ValueError:
            return np.nan
        if 0.8 <= v <= 60.0:
            return v
    return np.nan


# =============================================================================
# CỜ NHỊ PHÂN — TIỆN ÍCH VÀ ĐẶC ĐIỂM
# =============================================================================
# Các cờ này chỉ mang tính "tin đăng CÓ NHẮC ĐẾN", không phải "bất động sản CÓ".
# Đặt tên cột bắt đầu bằng `nhac_` để không ai nhầm hai chuyện đó với nhau —
# một căn hộ có bể bơi nhưng người đăng không nhắc thì cờ vẫn bằng 0.

BINARY_PATTERNS: dict[str, str] = {
    "nhac_o_to":      r"o ?to|oto|xe hoi|xe hoi",
    "nhac_thang_may": r"thang may",
    "nhac_be_boi":    r"be boi|ho boi",
    "nhac_gym":       r"\bgym\b|phong tap",
    "nhac_truong":    r"truong hoc|truong cap|mam non|dai hoc",
    "nhac_benh_vien": r"benh vien|phong kham",
    "nhac_cho_sieu_thi": r"\bcho\b|sieu thi|vincom|bigc|big c",
    "nhac_cong_vien": r"cong vien|ho dieu hoa",
    "nhac_metro":     r"metro|duong sat tren cao|nhon|cat linh",
    "nhac_kinh_doanh": r"kinh doanh|van phong|cho thue|mat bang",
    "nhac_chinh_chu": r"chinh chu",
    "nhac_view":      r"\bview\b",
    "nhac_lo_goc":    r"lo goc|can goc",
    "nhac_no_hau":    r"no hau",
    "nhac_can_sua":   r"nha nat|can sua|xuong cap",
}


def add_binary_flags(df: pd.DataFrame, text_cols: list[str]) -> pd.DataFrame:
    """Thêm các cờ nhị phân, quét trên phần văn bản đã ghép từ nhiều cột."""
    out = df.copy()
    blob = out[text_cols[0]].fillna("").astype(str)
    for c in text_cols[1:]:
        blob = blob + " " + out[c].fillna("").astype(str)
    flat = blob.map(strip_accents)
    for col, pat in BINARY_PATTERNS.items():
        out[col] = flat.str.contains(pat, regex=True).astype(int)
    return out


# =============================================================================
# GIÁ NHẮC TRONG TIÊU ĐỀ — dùng để ĐỐI CHIẾU, không dùng để thay thế
# =============================================================================

def extract_price_hint(text: object) -> float:
    """Rút giá nhắc trong tiêu đề: "5,4 tỷ", "1t7" (=1,7 tỷ), "12 tỉ", "850tr"."""
    if text is None:
        return np.nan
    t = strip_accents(text)
    m = re.search(r"\b(\d{1,3})\s*t\s*(\d)\b", t)             # "1t7"
    if m:
        return float(f"{m.group(1)}.{m.group(2)}") * 1e9
    m = re.search(r"\b([\d,]+(?:\.\d+)?)\s*t[yi]\b", t)        # "5,4 tỷ"
    if m:
        try:
            return float(m.group(1).replace(",", ".")) * 1e9
        except ValueError:
            pass
    m = re.search(r"\b([\d,]+)\s*(?:tr|trieu)\b", t)           # "850tr"
    if m:
        try:
            return float(m.group(1).replace(",", ".")) * 1e6
        except ValueError:
            pass
    return np.nan


# =============================================================================
# HÀM TIỆN ÍCH: VÁ CỘT THIẾU
# =============================================================================

def validate_dimensions(
    df: pd.DataFrame,
    area_col: str,
    frontage_col: str = "mat_tien_m",
    depth_col: str = "chieu_dai_m",
    max_ratio: float = 3.0,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Kiểm tra kích thước có nhất quán với diện tích không; xoá giá trị vô lý.

    LẬP LUẬN HÌNH HỌC
    Với một lô đất gần chữ nhật, mặt tiền × chiều dài ≈ diện tích. Lô càng rộng
    ngang thì càng nông. Cho phép tỉ lệ ngang:sâu tối đa `max_ratio` (mặc định
    3:1 — đã rất rộng rãi với nhà phố Việt Nam), suy ra:

        mặt tiền ≤ căn bậc hai của (diện tích × max_ratio)

    Ví dụ lô 45 m²: mặt tiền tối đa ≈ √135 ≈ 11,6 m. Một giá trị 45 m là bất
    khả thi và gần như chắc chắn đến từ việc đọc nhầm "45m2 x 5 tầng".

    Giá trị vi phạm bị đặt về NaN thay vì để nguyên — thà thiếu dữ liệu còn hơn
    có dữ liệu sai, vì XGBoost xử lý NaN được nhưng không tự phát hiện giá trị
    phi lý.
    """
    out = df.copy()
    stats = {"mat_tien_bi_xoa": 0, "chieu_dai_bi_xoa": 0}
    area = out[area_col]

    if frontage_col in out.columns:
        bound = np.sqrt(area * max_ratio)
        bad = out[frontage_col].notna() & area.notna() & (out[frontage_col] > bound)
        stats["mat_tien_bi_xoa"] = int(bad.sum())
        out.loc[bad, frontage_col] = np.nan

    if depth_col in out.columns:
        # chiều dài không thể vượt diện tích chia cho mặt tiền tối thiểu khả dĩ (1,5 m)
        bad = out[depth_col].notna() & area.notna() & (out[depth_col] > area / 1.5)
        stats["chieu_dai_bi_xoa"] = int(bad.sum())
        out.loc[bad, depth_col] = np.nan

    return out, stats


def fill_missing_from_text(
    df: pd.DataFrame, target_col: str, source_cols: list[str], extractor
) -> tuple[pd.DataFrame, int]:
    """Chỉ điền vào các ô đang TRỐNG của `target_col`, giá trị đã có giữ nguyên.

    Trả về (df mới, số ô đã vá được) để ghi vào báo cáo làm sạch.
    """
    out = df.copy()
    if target_col not in out.columns:
        out[target_col] = np.nan
    missing = out[target_col].isna()
    if not missing.any():
        return out, 0
    text = out.loc[missing, source_cols[0]].fillna("").astype(str)
    for c in source_cols[1:]:
        text = text + " " + out.loc[missing, c].fillna("").astype(str)
    recovered = text.map(extractor)
    n = int(recovered.notna().sum())
    out.loc[missing, target_col] = recovered
    return out, n
