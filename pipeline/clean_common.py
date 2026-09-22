"""
clean_common.py — Các hàm dùng chung cho việc làm sạch dữ liệu bất động sản nhatot.com

Dùng cho cả hai luồng: chung cư và nhà đất.

NGUYÊN TẮC XUYÊN SUỐT CỦA MODULE NÀY
------------------------------------
1. Không bao giờ xoá dòng một cách âm thầm. Mọi dòng bị loại đều được gắn cờ
   kèm lý do vào cột `ly_do_loai`, và được đếm vào báo cáo làm sạch.
2. Ưu tiên SỬA hơn XOÁ. Một tin ghi sai đơn vị giá vẫn là một quan sát thật;
   xoá đi là mất thông tin, sửa lại là giữ được.
3. Mọi luật suy đoán (heuristic) đều phải kiểm chứng được bằng cách bốc mẫu
   ra soi tay. Hàm nào dùng heuristic đều trả về đủ thông tin để làm việc đó.

Tác giả: pipeline cho khoá luận định giá BĐS Hà Nội
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# =============================================================================
# 1. CHUẨN HOÁ CHUỖI
# =============================================================================

def strip_accents(s: object) -> str:
    """Bỏ dấu tiếng Việt, trả về chữ thường.

    Lưu ý: 'đ'/'Đ' KHÔNG bị chuẩn hoá NFD tách dấu (vì nó là một ký tự riêng
    trong Unicode chứ không phải 'd' + dấu gạch), nên phải thay thủ công trước.
    Đây là lỗi rất hay gặp khi xử lý tiếng Việt.
    """
    if s is None:
        return ""
    s = str(s).replace("đ", "d").replace("Đ", "D")
    nfd = unicodedata.normalize("NFD", s)
    return "".join(c for c in nfd if unicodedata.category(c) != "Mn").lower()


def normalize_space(s: object) -> str:
    """Gộp mọi khoảng trắng liên tiếp thành một dấu cách, cắt hai đầu."""
    return re.sub(r"\s+", " ", str(s)).strip()


# =============================================================================
# 2. PARSE GIÁ VÀ DIỆN TÍCH
# =============================================================================
# nhatot hiển thị giá dạng "6,558 tỷ" hoặc "650 triệu".
# Dấu phẩy là dấu THẬP PHÂN (kiểu Việt Nam), dấu chấm là phân cách hàng nghìn.

_PRICE_UNITS = [
    (r"([\d,]+)\s*ty", 1e9),      # "6,558 tỷ"  -> 6.558 tỷ VND
    (r"([\d,]+)\s*trieu", 1e6),   # "650 triệu" -> 650 triệu VND
]


def parse_price(raw: object) -> float:
    """Chuỗi giá của nhatot -> số VND. Trả về NaN nếu không đọc được."""
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return np.nan
    t = strip_accents(raw).replace(".", "").strip()   # bỏ phân cách hàng nghìn
    for pattern, factor in _PRICE_UNITS:
        m = re.search(pattern, t)
        if m:
            try:
                return float(m.group(1).replace(",", ".")) * factor
            except ValueError:
                return np.nan
    return np.nan


def parse_area(raw: object) -> float:
    """Chuỗi diện tích ("47 m²", "66.12 m²") -> số m². NaN nếu không đọc được."""
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return np.nan
    m = re.search(r"([\d.,]+)", str(raw))
    return _vn_to_float(m.group(1)) if m else np.nan


def _vn_to_float(token: str) -> float:
    """Số viết theo quy ước Việt Nam -> float.

    Quy ước: '.' phân cách hàng nghìn, ',' là dấu thập phân. Nhưng nhatot dùng
    lẫn lộn cả hai kiểu, nên phải suy luận theo hình dạng chuỗi:

        "6,558"    -> 6.558     (chỉ có phẩy  -> thập phân)
        "74.528"   -> 74528.0   (nhóm 3 chữ số sau chấm -> hàng nghìn)
        "48.9"     -> 48.9      (không đủ 3 chữ số      -> thập phân)
        "1.234,56" -> 1234.56   (có cả hai -> chấm là nghìn, phẩy là thập phân)

    Nếu bỏ qua phân biệt này thì "74.528 đ/m²" bị đọc thành 74,5 đồng/m².
    """
    token = token.strip().rstrip(".,")
    if not token:
        return np.nan
    has_dot, has_comma = "." in token, "," in token
    if has_dot and has_comma:
        token = token.replace(".", "").replace(",", ".")
    elif has_comma:
        token = token.replace(",", ".")
    elif has_dot and re.fullmatch(r"\d{1,3}(?:\.\d{3})+", token):
        token = token.replace(".", "")
    try:
        return float(token)
    except ValueError:
        return np.nan


def parse_number(raw: object) -> float:
    """Rút số đầu tiên trong chuỗi ("3 phòng" -> 3, "74.528 đ" -> 74528)."""
    if raw is None:
        return np.nan
    m = re.search(r"([\d.,]+)", str(raw))
    return _vn_to_float(m.group(1)) if m else np.nan


def parse_price_per_m2(raw: object) -> float:
    """Trường "Giá/m²" của nhatot -> số VND/m².

    CẨN THẬN — nhatot đổi đơn vị theo độ lớn, không cố định "triệu/m²":
        "93,04 triệu/m²"  -> 93.04e6
        "1,60 tỷ/m²"      -> 1.60e9      (dùng khi vượt 1 tỷ/m²)
        "850.000 đ/m²"    -> 850000      (dùng khi cực nhỏ)

    Nếu chỉ bắt "triệu" bằng regex rồi nhân 1e6 thì các dòng đơn vị "tỷ/m²" sẽ
    bị bỏ sót — và dễ dẫn tới kết luận sai rằng nhatot "ẩn" giá/m² của tin bất
    thường. Thực tế nhatot vẫn hiển thị đủ, chỉ đổi đơn vị.
    """
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return np.nan
    text = str(raw)
    value = parse_number(text)
    if np.isnan(value):
        return np.nan
    flat = strip_accents(text)
    if "ty/m" in flat or re.search(r"\bty\b", flat):
        return value * 1e9
    if "trieu" in flat:
        return value * 1e6
    return value            # đơn vị "đ/m²" — đã là VND


def cross_check_parse(price: pd.Series, area: pd.Series,
                      stated_ppm2: pd.Series, tol: float = 0.02) -> pd.Series:
    """So `giá ÷ diện tích` với Giá/m² do nhatot công bố. Trả về sai số tương đối.

    MỤC ĐÍCH VÀ GIỚI HẠN — đọc kỹ trước khi diễn giải kết quả:
    Phép này kiểm chứng BỘ PARSER của ta, không kiểm chứng ĐỘ THẬT của dữ liệu.
    nhatot tự tính Giá/m² từ chính giá và diện tích mà người đăng nhập vào, nên
    khi người ta gõ nhầm "55000 tỷ" cho 70 m², nhatot vẫn ngoan ngoãn hiển thị
    "785,71 tỷ/m²" — khớp hoàn hảo với phép chia của ta.
    Nói cách khác: lệch ≈ 0 nghĩa là "ta đọc đúng những gì trang web hiển thị",
    KHÔNG có nghĩa là "con số đó đúng với thực tế". Việc bắt tin sai/ảo phải do
    fix_price_unit_errors() và flag_outliers_grouped() đảm nhiệm.
    """
    calc = price / area
    ok = calc.notna() & stated_ppm2.notna() & (stated_ppm2 > 0)
    rel = pd.Series(np.nan, index=price.index)
    rel[ok] = (calc[ok] - stated_ppm2[ok]).abs() / stated_ppm2[ok]
    return rel


# =============================================================================
# 3. BÓC KHỐI THUỘC TÍNH `dac_diem_tong_hop`
# =============================================================================
# nhatot nối nhãn và giá trị lại thành một chuỗi dài KHÔNG có dấu phân cách:
#   "Loại hìnhNhà ngõ, hẻmDiện tích đất47 m²Giá/m²139,53 triệu/m²..."
#
# Cách bóc: biết trước tập nhãn, tìm mọi vị trí nhãn xuất hiện, rồi lấy phần
# văn bản nằm GIỮA hai nhãn liên tiếp làm giá trị. Cách này bền hơn nhiều so
# với viết regex riêng cho từng trường, vì nó không phụ thuộc thứ tự các trường
# và tự bỏ qua trường vắng mặt.

# Nhãn phải sắp theo ĐỘ DÀI GIẢM DẦN khi ghép regex, để "Diện tích đất" được
# thử trước "Diện tích" — nếu không, "Diện tích" sẽ khớp trước và cắt sai.
BLOB_LABELS_CHUNGCU = [
    "Tình trạng bất động sản", "Loại hình", "Diện tích", "Giá/m²",
    "Giấy tờ pháp lý", "Số phòng ngủ", "Số phòng vệ sinh",
    "Tình trạng nội thất", "Hướng ban công", "Hướng cửa chính",
    "Tầng số", "Block/Tháp",
    # Ba nhãn này `nguon_batdongsan._ghep` CÓ THỂ xuất ra nhưng danh sách cũ
    # không có. Đo trên dữ liệu chung cư hiện tại thì chưa tin nào dính — vì
    # tin chung cư trên batdongsan hầu như không khai ba trường này — nên đây
    # là mìn chưa nổ, không phải lỗi đang gây hại. `kiem_nhan_ghep()` tìm ra.
    # Thêm một nhãn không bao giờ xuất hiện thì vô hại: nó chỉ không khớp.
    "Chiều ngang", "Đường vào", "Mã căn",
]

BLOB_LABELS_NHADAT = [
    "Loại hình", "Diện tích đất", "Diện tích sử dụng", "Diện tích", "Giá/m²",
    "Giấy tờ pháp lý", "Số phòng ngủ", "Số phòng vệ sinh",
    "Tình trạng nội thất", "Hướng cửa chính", "Đặc điểm",
    "Chiều ngang", "Chiều dài", "Tổng số tầng",
    # batdongsan có sẵn trường này; nhatot thì không.
    "Đường vào",
    # THIẾU BA NHÃN NÀY LÀM HỎNG 63% GIÁ TRỊ NỘI THẤT NHÀ ĐẤT
    #
    # Cách bóc ở đây là "lấy phần văn bản nằm giữa hai nhãn liên tiếp". Hệ quả
    # ít ai để ý: một nhãn KHÔNG có trong danh sách thì nó không kết thúc giá
    # trị đứng trước, nên nhãn và giá trị của nó bị dính luôn vào trường trước.
    # `nguon_batdongsan._ghep` xuất ra "Tầng số" và "Hướng ban công", mà danh
    # sách này lại chỉ có "Tổng số tầng" — nên "Nội thất đầy đủ" biến thành
    # "Nội thất đầy đủTầng số5 tầng". Đo ngày 12/09/2026 trên dữ liệu đang
    # chạy:
    #
    #     Tình trạng nội thất   5.100/8.034  bị dính (63%) →  72 loại thay vì 7
    #     Hướng cửa chính       1.882/2.824  bị dính (67%) →  83 loại thay vì 8
    #     Giấy tờ pháp lý         753/11.952 bị dính  (6%) →  44 loại thay vì 7
    #
    # Không có lỗi nào báo ra, vì chuỗi dính vẫn là một chuỗi hợp lệ. Đây là
    # cùng một LOẠI lỗi với việc mất cột `url_goc` và `latitude`: dữ liệu hỏng
    # im lặng. Thêm nhãn vào đây là cách chữa gốc; muốn chặn hẳn thì
    # `kiem_nhan_ghep()` ở dưới so hai danh sách với nhau.
    "Tầng số", "Hướng ban công", "Mã căn",
]

# Khối của nhà đất mở đầu bằng tiêu đề "Đặc điểm bất động sản" — đây KHÔNG phải
# một trường dữ liệu. Nếu không cắt bỏ trước, nhãn "Đặc điểm" sẽ khớp nhầm vào
# tiêu đề này và sinh ra giá trị rác "bất động sản".
BLOB_HEADERS = ["Đặc điểm bất động sản"]


def kiem_nhan_ghep(nhan_xuat, nhan_boc: list[str], ten: str = "") -> None:
    """Chặn ngay lúc chạy: mọi nhãn đem GHÉP phải nằm trong danh sách BÓC.

    Đây là chốt chặn cho một loại lỗi đã xảy ra thật và không hề báo gì:
    `nguon_batdongsan._ghep` ghép "Tầng số" và "Hướng ban công" vào khối
    thuộc tính, nhưng `BLOB_LABELS_NHADAT` không có hai nhãn đó, nên chúng
    không kết thúc giá trị đứng trước và dính luôn vào trường trước nó —
    "Nội thất đầy đủ" thành "Nội thất đầy đủTầng số5 tầng", 63% số tin.

    Không có cách nào để bước bóc TỰ phát hiện: chuỗi dính vẫn là chuỗi hợp
    lệ, vẫn ra một giá trị, chỉ là giá trị sai. Nên phải so hai danh sách với
    nhau ngay lúc ghép, và vỡ to thay vì hỏng im lặng.
    """
    thieu = [x for x in nhan_xuat if x not in set(nhan_boc)]
    if thieu:
        raise ValueError(
            f"{ten or 'Khối thuộc tính'}: ghép nhãn {thieu} nhưng danh sách "
            f"bóc không có. Giá trị của trường ĐỨNG TRƯỚC mỗi nhãn này sẽ bị "
            f"dính thêm nhãn và giá trị của nó. Thêm vào BLOB_LABELS_* rồi "
            f"chạy lại.")


def _build_label_pattern(labels: list[str]) -> re.Pattern:
    ordered = sorted(labels, key=len, reverse=True)
    return re.compile("(" + "|".join(re.escape(x) for x in ordered) + ")")


def parse_attr_blob(blob: object, labels: list[str]) -> dict[str, str]:
    """Bóc khối thuộc tính nối liền thành dict {nhãn: giá trị}.

    Trả về dict rỗng nếu blob trống. Trường vắng mặt đơn giản là không có key.
    """
    if blob is None or (isinstance(blob, float) and np.isnan(blob)):
        return {}
    text = str(blob)
    for header in BLOB_HEADERS:                 # cắt tiêu đề dẫn nếu có
        if text.startswith(header):
            text = text[len(header):]
    pattern = _build_label_pattern(labels)
    parts = pattern.split(text)
    # parts = [phần trước nhãn đầu, nhãn1, giá trị1, nhãn2, giá trị2, ...]
    out: dict[str, str] = {}
    for i in range(1, len(parts) - 1, 2):
        label, value = parts[i], parts[i + 1].strip()
        if value and label not in out:          # giữ lần xuất hiện ĐẦU TIÊN
            out[label] = value
    return out


def expand_attr_blob(df: pd.DataFrame, blob_col: str, labels: list[str],
                     prefix: str = "") -> pd.DataFrame:
    """Bóc cả cột blob thành nhiều cột mới, ghép vào DataFrame."""
    parsed = df[blob_col].map(lambda b: parse_attr_blob(b, labels))
    expanded = pd.DataFrame(list(parsed), index=df.index)
    if prefix:
        expanded.columns = [prefix + c for c in expanded.columns]
    return pd.concat([df, expanded], axis=1)


# =============================================================================
# 4. PARSE ĐỊA CHỈ
# =============================================================================
# Định dạng nhatot rất đều: "<đường/ngõ>, <Phường X>, <Quận Y>, Hà Nội"
# Một số dòng có thêm số nhà ở đầu: "4/16/79, Đường An Dương Vương, Phường ..."

_RE_QUAN = re.compile(r"(Quận\s+[^,]+|Huyện\s+[^,]+|Thị\s+xã\s+[^,]+)")
_RE_PHUONG = re.compile(r"(Phường\s+[^,]+|Xã\s+[^,]+|Thị\s+trấn\s+[^,]+)")


def parse_address(raw: object) -> dict[str, str | float]:
    """Địa chỉ nhatot -> {duong_pho, phuong_xa, quan_huyen}.

    `duong_pho` là toàn bộ phần đứng TRƯỚC cấp phường/xã — tức số nhà, ngõ và
    tên đường gộp lại. Giữ nguyên để dùng làm đầu vào geocoding.
    """
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return {"duong_pho": np.nan, "phuong_xa": np.nan, "quan_huyen": np.nan}
    text = normalize_space(raw)
    m_quan = _RE_QUAN.search(text)
    m_phuong = _RE_PHUONG.search(text)
    quan = normalize_space(m_quan.group(1)) if m_quan else np.nan
    phuong = normalize_space(m_phuong.group(1)) if m_phuong else np.nan
    # phần đường phố = mọi thứ trước vị trí phường/xã (hoặc trước quận nếu thiếu)
    cut = m_phuong.start() if m_phuong else (m_quan.start() if m_quan else len(text))
    duong = normalize_space(text[:cut].rstrip(", ")) or np.nan
    return {"duong_pho": duong, "phuong_xa": phuong, "quan_huyen": quan}


def bu_quan_tu_phuong(df: pd.DataFrame, bang_tra: pd.Series | None = None,
                      cot_phuong: str = "phuong_xa",
                      cot_quan: str = "quan_huyen") -> tuple[pd.DataFrame, int]:
    """Điền quận/huyện còn trống bằng cách tra ngược từ tên phường.

    VÌ SAO CẦN: từ 1/7/2025 cấp quận/huyện bị bãi bỏ, và batdongsan đã bắt đầu
    ghi địa chỉ theo cấu trúc mới — "Đường Tân Mai, Phường Tân Mai, Hà Nội",
    không còn cấp quận. Đợt chung cư trước còn kèm dòng phụ "(Quận Hoàng Mai,
    Hà Nội cũ)" nên vẫn tách được; đợt nhà đất thì không.

    Hậu quả nếu không xử lý: bộ lọc "không xác định được quận/huyện" quét sạch
    **toàn bộ 1.172 tin**, và không một lỗi nào báo ra — báo cáo chỉ ghi lặng lẽ
    rằng còn 0 dòng.

    Cách bù: tên PHƯỜNG MỚI là khoá đủ mạnh, vì mỗi phường mới chỉ thuộc một
    quận cũ (trừ vài phường nằm vắt qua ranh giới, khi đó lấy quận chiếm đa số).
    Bảng tra dựng từ chính dữ liệu sạch đã có.

    Lưu ý về ý nghĩa: `quan_huyen` từ đây là một cấp hành chính ĐÃ BỊ BÃI BỎ,
    giữ lại chỉ để nhóm thô và để tương thích với dữ liệu cũ. Cấp thật sự dùng
    trong model là `phuong_moi`.
    """
    if cot_phuong not in df.columns:
        return df, 0
    if bang_tra is None:
        bang_tra = _bang_phuong_quan()
    if bang_tra is None or bang_tra.empty:
        return df, 0

    out = df.copy()
    if cot_quan not in out.columns:
        out[cot_quan] = np.nan
    # Cột toàn NaN có dtype float64; gán chuỗi vào sẽ ném TypeError ở pandas
    # mới. Ép sang object trước.
    out[cot_quan] = out[cot_quan].astype(object)
    thieu = out[cot_quan].isna() & out[cot_phuong].notna()
    if not thieu.any():
        return out, 0
    dien = out.loc[thieu, cot_phuong].map(bang_tra).astype(object)
    out.loc[thieu, cot_quan] = dien
    return out, int(dien.notna().sum())


def _bang_phuong_quan() -> pd.Series | None:
    """Bảng tra phường -> quận, dựng từ dữ liệu sạch đã có trên đĩa."""
    from pathlib import Path as _P
    goc = _P(__file__).resolve().parent
    for ten in ("data_clean", "../data_clean"):
        thu_muc = goc / ten
        if thu_muc.is_dir():
            break
    else:
        return None
    # Tra bằng CẢ tên phường cũ lẫn tên phường mới. Địa chỉ sau sáp nhập dùng
    # tên MỚI, mà tên mới thường không trùng tên cũ nào — chỉ tra `phuong_xa`
    # thì mất 242/1.172 tin chỉ vì phường của chúng vừa được đặt tên lại.
    khung = []
    for tep in ("nhadat_clean_geo.xlsx", "chungcu_gop_geo.xlsx",
                "chungcu_clean_geo.xlsx"):
        d = thu_muc / tep
        if not d.exists():
            continue
        for cot in ("phuong_xa", "phuong_moi"):
            try:
                # Chọn cột THEO TÊN sau khi đọc. `usecols` trả về cột theo thứ
                # tự trong file chứ không theo thứ tự mình liệt kê, nên gán
                # `x.columns = [...]` là cách đảo ngược bảng tra lúc nào không
                # hay: phường thành quận và ngược lại.
                x = pd.read_excel(d, usecols=[cot, "quan_huyen"])
                x = x[[cot, "quan_huyen"]].rename(columns={cot: "phuong"})
                khung.append(x.dropna())
            except Exception:
                continue
    if not khung:
        return None
    t = pd.concat(khung, ignore_index=True)
    return t.groupby("phuong")["quan_huyen"].agg(
        lambda s: s.mode().iat[0] if len(s.mode()) else np.nan)


def build_geocode_query(row: pd.Series) -> str:
    """Ghép chuỗi truy vấn cho dịch vụ geocoding, bỏ phần trống."""
    parts = [row.get("duong_pho"), row.get("phuong_xa"),
             row.get("quan_huyen"), "Hà Nội", "Việt Nam"]
    return ", ".join(str(p) for p in parts if isinstance(p, str) and p.strip())


# =============================================================================
# 5. SỬA LỖI ĐƠN VỊ GIÁ
# =============================================================================
# Người đăng tin thỉnh thoảng gõ nhầm đơn vị: "2950 tỷ" cho một căn 43 m²
# (đúng ra là 2.950 triệu = 2,95 tỷ), hoặc "3,5 triệu" cho căn 48,9 m²
# (đúng ra là 3,5 tỷ).
#
# LUẬT: nếu giá/m² nằm ngoài khoảng hợp lý, thử nhân hoặc chia 1.000. Nếu một
# trong hai phép đưa giá/m² về ĐÚNG khoảng hợp lý, coi đó là lỗi đơn vị và sửa.
# Nếu cả hai đều không cứu được, để nguyên và gắn cờ ngoại lai.
#
# CẢNH BÁO: đây là suy đoán, không phải sự thật. Luôn chạy `sample_unit_fixes()`
# và soi tay trước khi tin. Hàm ghi lại mọi dòng đã sửa để làm việc đó.

def plausible_ppm2_band(price: pd.Series, area: pd.Series,
                        lo_ppm2: float, hi_ppm2: float,
                        q_lo: float = 0.01, q_hi: float = 0.99) -> tuple[float, float]:
    """Suy ra khoảng giá/m² "trông bình thường" TỪ CHÍNH DỮ LIỆU.

    Cách làm: lấy các dòng đã nằm trong khoảng khả dĩ thô [lo, hi], rồi lấy
    phân vị 1% và 99% của chúng. Kết quả là một khoảng chặt hơn nhiều so với
    ngưỡng cứng, và tự thích nghi với từng bộ dữ liệu (chung cư và nhà đất có
    mặt bằng giá/m² rất khác nhau, không thể dùng chung một ngưỡng).

    Dùng làm khoảng CHẤP NHẬN cho giá trị SAU khi sửa lỗi đơn vị.
    """
    ppm2 = (price / area).replace([np.inf, -np.inf], np.nan).dropna()
    normal = ppm2[(ppm2 >= lo_ppm2) & (ppm2 <= hi_ppm2)]
    if len(normal) < 50:
        return lo_ppm2, hi_ppm2
    return float(normal.quantile(q_lo)), float(normal.quantile(q_hi))


def fix_price_unit_errors(
    df: pd.DataFrame,
    price_col: str = "gia_vnd",
    area_col: str = "dien_tich_m2",
    lo_ppm2: float = 10e6,
    hi_ppm2: float = 1000e6,
    accept_lo: float | None = None,
    accept_hi: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Phát hiện và sửa lỗi đơn vị giá.

    lo_ppm2 / hi_ppm2 — khoảng PHÁT HIỆN: giá/m² nằm ngoài khoảng này bị coi là
        đáng ngờ. Đặt rộng rãi, chỉ để chặn những trường hợp bất khả thi.

    accept_lo / accept_hi — khoảng CHẤP NHẬN: giá trị SAU khi nhân/chia 1000
        phải rơi vào đây thì phép sửa mới được công nhận. Nếu để None thì dùng
        luôn khoảng phát hiện.

        VÌ SAO CẦN HAI KHOẢNG RIÊNG BIỆT — bài học rút ra khi kiểm chứng:
        với khoảng chấp nhận rộng (8–400 triệu/m²), một tin ở Hà Đông ghi
        "16,666 triệu" đã được "sửa" thành 16,67 tỷ tức 370 triệu/m². Con số đó
        vẫn lọt khoảng nhưng hoàn toàn vô lý với Hà Đông (mặt bằng 30–50 triệu),
        và tiêu đề của chính tin đó ghi "giá chỉ 1t7". Tức là phép sửa đã biến
        một giá trị sai thành một giá trị sai khác, nguy hiểm hơn vì trông hợp lệ.
        Dùng khoảng chấp nhận hẹp (phân vị 1–99% của dữ liệu, xem
        plausible_ppm2_band) thì phép sửa đó bị từ chối, dòng được để nguyên và
        rơi vào bước lọc ngoại lai — đúng như mong muốn.

    Trả về (df đã sửa, bảng các dòng đã sửa để soi tay).
    """
    out = df.copy()
    if accept_lo is None:
        accept_lo = lo_ppm2
    if accept_hi is None:
        accept_hi = hi_ppm2

    ppm2 = out[price_col] / out[area_col]
    valid = ppm2.notna() & (out[area_col] > 0)

    too_high = valid & (ppm2 > hi_ppm2)
    too_low = valid & (ppm2 < lo_ppm2)

    # chia 1000 có cứu được không? (trường hợp gõ "tỷ" thay vì "triệu")
    fix_down = too_high & ((ppm2 / 1000).between(accept_lo, accept_hi))
    # nhân 1000 có cứu được không? (trường hợp gõ "triệu" thay vì "tỷ")
    fix_up = too_low & ((ppm2 * 1000).between(accept_lo, accept_hi))

    log = pd.DataFrame({
        "gia_truoc": out.loc[fix_down | fix_up, price_col],
        "dien_tich": out.loc[fix_down | fix_up, area_col],
        "ppm2_truoc": ppm2[fix_down | fix_up],
        "huong_sua": np.where(fix_down[fix_down | fix_up], "chia 1000", "nhân 1000"),
    })

    out.loc[fix_down, price_col] = out.loc[fix_down, price_col] / 1000
    out.loc[fix_up, price_col] = out.loc[fix_up, price_col] * 1000

    log["gia_sau"] = out.loc[log.index, price_col]
    log["ppm2_sau"] = out.loc[log.index, price_col] / out.loc[log.index, area_col]
    return out, log


def extract_price_from_text(text: object) -> float:
    """Rút giá nhắc trong tiêu đề/mô tả để đối chiếu chéo.

    Bắt các dạng người Việt hay viết: "5,4 tỷ", "1t7" (= 1,7 tỷ), "12 tỉ".
    Dùng để XÁC NHẬN giá đã parse, không dùng để thay thế.
    """
    if text is None:
        return np.nan
    t = strip_accents(text)
    m = re.search(r"(\d{1,3})\s*t\s*(\d)\b", t)          # "1t7" -> 1.7 tỷ
    if m:
        return float(f"{m.group(1)}.{m.group(2)}") * 1e9
    m = re.search(r"([\d,]+)\s*t[yi]\b", t)               # "5,4 tỷ" / "12 tỉ"
    if m:
        try:
            return float(m.group(1).replace(",", ".")) * 1e9
        except ValueError:
            return np.nan
    return np.nan


# =============================================================================
# 6. KHỬ TRÙNG LẶP
# =============================================================================
# Ba tầng, chạy theo thứ tự từ chắc chắn nhất đến suy đoán nhất:
#
#   Tầng 1 — ID tin: hai dòng cùng ID trên trang nguồn thì chắc chắn là một.
#            Chỉ dùng được cho nhà đất (chung cư không crawl cột link).
#   Tầng 2 — Khoá nghiệp vụ: cùng mô tả + giá + diện tích. Cùng một căn được
#            đăng lại thành nhiều tin khác ID.
#   Tầng 3 — Gần trùng: mô tả khác nhau chút ít nhưng cùng một tài sản. So bằng
#            cosine trên TF-IDF, và chỉ so TRONG CÙNG KHỐI (quận + phường +
#            diện tích làm tròn) để không phải so tất cả với tất cả.

def dedup_exact(df: pd.DataFrame, subset: list[str], reason: str,
                report: "CleanReport") -> pd.DataFrame:
    """Khử trùng lặp theo khoá chính xác, giữ dòng xuất hiện đầu tiên."""
    before = len(df)
    mask = df.duplicated(subset=subset, keep="first")
    report.log(reason, int(mask.sum()), before)
    return df.loc[~mask].copy()


def dedup_near(
    df: pd.DataFrame,
    text_col: str,
    block_cols: list[str],
    threshold: float = 0.90,
    report: "CleanReport | None" = None,
) -> pd.DataFrame:
    """Khử trùng lặp gần đúng bằng cosine similarity trên TF-IDF.

    threshold=0.90 chọn khá cao có chủ ý: thà bỏ sót vài tin trùng còn hơn xoá
    nhầm hai căn khác nhau trong cùng toà nhà (mô tả của chúng vốn rất giống
    nhau). Nếu muốn siết, hạ xuống 0.85 rồi soi lại mẫu.

    Chỉ so trong cùng khối `block_cols` -> độ phức tạp giảm từ O(n²) toàn cục
    xuống O(sum of block²), chạy được trên máy thường.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    before = len(df)
    drop_idx: set = set()
    for _, block in df.groupby(block_cols, dropna=False):
        if len(block) < 2 or len(block) > 800:   # khối quá lớn thì bỏ qua cho an toàn
            continue
        texts = block[text_col].fillna("").astype(str)
        if (texts.str.len() < 20).all():
            continue
        try:
            tfidf = TfidfVectorizer(min_df=1, ngram_range=(1, 2)).fit_transform(texts)
        except ValueError:                        # khối không có từ nào hợp lệ
            continue
        sim = cosine_similarity(tfidf)
        np.fill_diagonal(sim, 0.0)
        idx = list(block.index)
        for i in range(len(idx)):
            if idx[i] in drop_idx:
                continue
            for j in range(i + 1, len(idx)):
                if sim[i, j] >= threshold:
                    drop_idx.add(idx[j])          # giữ dòng đầu, bỏ dòng sau
    if report is not None:
        report.log(f"Trùng gần đúng (cosine ≥ {threshold})", len(drop_idx), before)
    return df.drop(index=list(drop_idx)).copy()


# =============================================================================
# 7. LỌC NGOẠI LAI
# =============================================================================
# KHÔNG dùng ngưỡng cứng toàn cục. Lý do: nhà mặt phố Hoàn Kiếm 600 triệu/m² là
# giá THẬT, trong khi nhà ngõ Hoàng Mai 600 triệu/m² gần như chắc chắn là tin ảo.
# Cùng một con số, ý nghĩa khác nhau tuỳ vị trí -> phải xét trong từng nhóm.

def flag_outliers_grouped(
    df: pd.DataFrame,
    value_col: str,
    group_cols: list[str],
    k: float = 3.0,
    min_group: int = 15,
) -> pd.Series:
    """Gắn cờ ngoại lai bằng luật IQR trong từng nhóm.

    k=3.0 (thay vì 1.5 như quy ước thống kê thông thường) vì giá bất động sản
    lệch phải rất mạnh — dùng 1.5 sẽ cắt mất nhiều giao dịch cao cấp có thật.
    Nhóm dưới `min_group` mẫu thì không đủ để ước lượng IQR đáng tin, bỏ qua.
    """
    def _flag(g: pd.Series) -> pd.Series:
        if len(g) < min_group:
            return pd.Series(False, index=g.index)
        q1, q3 = g.quantile(0.25), g.quantile(0.75)
        iqr = q3 - q1
        return (g < q1 - k * iqr) | (g > q3 + k * iqr)

    return df.groupby(group_cols, dropna=False)[value_col].transform(_flag).fillna(False)


# =============================================================================
# 8. BÁO CÁO LÀM SẠCH
# =============================================================================
# Bảng "còn bao nhiêu dòng sau mỗi bước" là một mục bắt buộc trong chương
# phương pháp của khoá luận. Lớp này ghi lại tự động để khỏi phải đếm tay.

@dataclass
class CleanReport:
    ten_bo_du_lieu: str
    so_dong_ban_dau: int = 0
    _rows: list[dict] = field(default_factory=list)

    def log(self, buoc: str, so_dong_loai: int, so_dong_truoc: int) -> None:
        self._rows.append({
            "Bước": buoc,
            "Dòng vào": so_dong_truoc,
            "Loại bỏ": so_dong_loai,
            "Còn lại": so_dong_truoc - so_dong_loai,
            "Tỉ lệ loại (%)": round(so_dong_loai / so_dong_truoc * 100, 2) if so_dong_truoc else 0.0,
        })

    def note(self, buoc: str, ghi_chu: str) -> None:
        self._rows.append({"Bước": buoc, "Dòng vào": None, "Loại bỏ": None,
                           "Còn lại": None, "Tỉ lệ loại (%)": None, "Ghi chú": ghi_chu})

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self._rows)

    def render(self) -> str:
        df = self.to_frame()
        head = f"BÁO CÁO LÀM SẠCH — {self.ten_bo_du_lieu}"
        line = "=" * len(head)
        giu_lai = df["Còn lại"].dropna()
        cuoi = int(giu_lai.iloc[-1]) if len(giu_lai) else self.so_dong_ban_dau
        tom_tat = (f"\nBan đầu {self.so_dong_ban_dau} dòng -> giữ lại {cuoi} dòng "
                   f"({cuoi / self.so_dong_ban_dau * 100:.1f}%)")
        return f"{head}\n{line}\n{df.to_string(index=False)}\n{tom_tat}"
