"""
tim_nha.py — Tìm nhà bằng câu tiếng Việt tự nhiên.

Gộp ba tính năng cũ (Săn tin, Ngân sách, So sánh) vào một chỗ. Lý do gộp
không phải để giao diện gọn, mà vì cả ba đều trả lời CÙNG một câu hỏi của
người đi mua nhà: *"với điều kiện của tôi thì nên xem những căn nào?"*

    Ngân sách  ->  "5 tỷ mua được gì, ở đâu"        = lọc theo giá
    Săn tin    ->  "căn nào đang lệch mặt bằng"      = cột `lech_phan_tram`
    So sánh    ->  "vì sao căn này đắt hơn căn kia"   = chọn 2 căn trong kết quả

VÌ SAO TỰ VIẾT BỘ PHÂN TÍCH, KHÔNG GỌI API LLM
-----------------------------------------------
Gọi API của OpenAI hay Gemini thì hiểu câu tự nhiên tốt hơn hẳn. Nhưng với
một luận văn, ba điểm trừ này nặng hơn:

  1. Tốn tiền theo lượt, và app công khai thì ai có link cũng đốt được.
  2. Phải đặt khoá API trong bản triển khai công khai.
  3. Quan trọng nhất: phần "AI" ấy không phải công của mình. Hội đồng hỏi
     "em cài đặt phần hiểu ngôn ngữ thế nào" thì câu trả lời là "em gọi API".

Bộ phân tích ở đây chạy hoàn toàn offline, dựa trên chính dữ liệu đã có: 113
phường mới, 286 tên phường CŨ trước sáp nhập, 27 quận, 52 dự án, hơn 2.000 tên
phố. Nó không hiểu câu lạ, nhưng nó hiểu ĐÚNG những gì nó nhận ra, và luôn
hiện lại cho người dùng thấy nó hiểu gì để sửa. Máy đoán, người xác nhận.

NGUYÊN TẮC XUYÊN SUỐT: THÀ KHÔNG HIỂU CÒN HƠN HIỂU NGƯỢC
--------------------------------------------------------
Bỏ sót một điều kiện thì người dùng thấy kết quả rộng hơn mong đợi và gõ rõ
thêm. Hiểu NGƯỢC một điều kiện thì họ nhận đúng những căn họ không muốn mà
không hiểu tại sao. Nên mọi chỗ mơ hồ trong file này đều nghiêng về không
nhận, và mọi điều kiện đã nhận đều phải hiện lại được bằng tiếng Việt.

CÁCH ĐỌC FILE NÀY
-----------------
    _bo_dau, _chuan, _chuan_so   — chuẩn hoá chữ để so không phụ thuộc dấu
    _phu_dinh_truoc              — có chữ phủ định ngay trước cụm này không
    doc_gia, doc_dien_tich       — bóc số kèm đơn vị ("5 tỷ", "2 tỷ 8", "60m2")
    doc_phong, doc_kich_thuoc    — số phòng; số tầng, mặt tiền, độ rộng ngõ
    doc_dia_diem                 — quận / phường mới / phường cũ / phố / dự án
    doc_dac_diem, doc_sap        — yêu cầu có-không; ý định sắp xếp
    phan_tich                    — gọi hết các hàm trên, trả về dict điều kiện
    tim                          — áp điều kiện lên dữ liệu, xếp hạng, trả kết quả

Bộ kiểm: `_scratch/kiem_tim_nha.py` — 54 câu thật, chạy trước mỗi lần sửa.
"""
from __future__ import annotations

import re
import unicodedata as ud
from functools import lru_cache

import numpy as np
import pandas as pd

import valuation_service as vs


# =============================================================================
# CHUẨN HOÁ CHỮ
# =============================================================================

def _bo_dau(s: object) -> str:
    """Bỏ dấu tiếng Việt và hạ chữ thường.

    Người dùng gõ "cau giay", "Cầu Giấy", "CẦU GIẤY" đều phải khớp. Cách rẻ
    nhất là quy mọi thứ về dạng không dấu rồi so.
    """
    t = ud.normalize("NFD", str(s).lower())
    t = "".join(c for c in t if ud.category(c) != "Mn")
    return t.replace("đ", "d").replace("Đ", "d")


def _chuan(s: object) -> str:
    """Bỏ dấu + nén khoảng trắng + bỏ dấu câu, để so tên địa danh."""
    t = _bo_dau(s)
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    return " ".join(t.split())


def _chuan_so(s: object) -> str:
    """Như `_chuan` nhưng GIỮ LẠI `-`, `~`, `.`, `,`.

    Cần vì "3-5 tỷ" là một khoảng, mà `_chuan` biến dấu gạch thành khoảng
    trắng nên "3 5 ty" không còn khớp mẫu khoảng nữa — bản đầu đọc nhầm
    "3-5 tỷ" thành "không quá 5 tỷ".
    """
    t = _bo_dau(s)
    t = re.sub(r"[^a-z0-9\s.,~-]", " ", t)
    return " ".join(t.split())


# Số viết bằng chữ, dùng cho TÊN ĐỊA DANH: người ta gõ "quận 2 bà trưng" thay
# vì "Hai Bà Trưng", "quận 3 đình" thay vì "Ba Đình". Chỉ dùng cho vòng khớp
# tên, không dùng cho vòng bóc số — nếu không thì "2 phòng ngủ" cũng bị đổi.
_SO_THANH_CHU = {"1": "mot", "2": "hai", "3": "ba", "4": "bon", "5": "nam",
                 "6": "sau", "7": "bay", "8": "tam", "9": "chin"}


def _thay_so_bang_chu(t: str) -> str:
    return re.sub(r"\b([1-9])\b", lambda m: _SO_THANH_CHU[m.group(1)], t)


# =============================================================================
# PHỦ ĐỊNH
# =============================================================================
# Bản trước KHÔNG xử lý phủ định, và đó là lỗi tệ nhất của nó: câu "chung cư
# không cần thang máy" cho ra điều kiện "CÓ thang máy", tức đúng ngược ý người
# dùng. Bỏ sót thì người ta gõ lại; hiểu ngược thì người ta tin vào một danh
# sách sai.

_TU_PHU_DINH = ("khong", "khong can", "khong muon", "khong thich", "khoi",
                "khoi can", "chang can", "chua can", "mien", "tru",
                "ngoai tru", "ko", "k can", "chua co", "khong co")

# Cửa sổ nhìn về trước, tính bằng ký tự. Đủ để bắt "không nhất thiết phải có
# thang máy" nhưng không trùm sang mệnh đề trước đó.
_CUA_SO_PHU_DINH = 26


def _phu_dinh_truoc(t: str, vi_tri: int) -> bool:
    """Ngay trước vị trí này có chữ phủ định nào không?

    Có một cái bẫy phải tránh: "miễn là có thang máy" mang chữ "miễn" nhưng
    KHÔNG phủ định cái đi sau nó. Nên nếu giữa chữ phủ định và cụm đang xét
    có chữ "la" hoặc "sao" ("miễn là", "miễn sao") thì bỏ qua chữ phủ định đó.
    """
    truoc = t[max(0, vi_tri - _CUA_SO_PHU_DINH):vi_tri]
    for w in _TU_PHU_DINH:
        m = None
        for m in re.finditer(rf"\b{re.escape(w)}\b", truoc):
            pass                                  # lấy lần xuất hiện CUỐI
        if m is None:
            continue
        giua = truoc[m.end():]
        if re.search(r"\b(la|sao)\b", giua):
            continue                              # "miễn là", "miễn sao"
        return True
    return False


# =============================================================================
# BÓC SỐ TIỀN
# =============================================================================

# "tỷ" viết được rất nhiều kiểu, và người ta cũng gõ tắt.
#
# ĐÃ BỎ "m" KHỎI ĐÂY, ĐỪNG THÊM LẠI. Trước đây "m" nghĩa là triệu, và hậu quả
# là "mặt tiền 5m" bị đọc thành "giá không quá 5 triệu" = 0,01 tỷ, lọc sạch
# mọi căn. "m" trong câu tiếng Việt về nhà đất gần như luôn là MÉT; muốn nói
# triệu thì người ta viết "tr" hoặc "triệu".
_DV_TIEN = {
    "ty": 1e9, "tỷ": 1e9, "ti": 1e9, "tỉ": 1e9, "b": 1e9,
    "trieu": 1e6, "triệu": 1e6, "tr": 1e6,
}
_DV_TY = ("ty", "tỷ", "ti", "tỉ", "b")
_TU_DUOI = ("duoi", "toi da", "khong qua", "it hon", "nho hon", "max",
            "trong tam", "tam gia")
_TU_TREN = ("tren", "toi thieu", "hon", "lon hon", "tu", "min")
# "khoảng 5 tỷ" là MỘT DẢI quanh 5 tỷ, không phải trần 5 tỷ. Trước đây gộp
# chung vào _TU_DUOI nên "khoảng 5 tỷ" thành "dưới 5 tỷ" — người muốn xem căn
# 5,2 tỷ không bao giờ thấy nó.
#
# HAI CÁI BẪY Ở DANH SÁCH NÀY, đều đã sập một lần:
#
#   "tôi CÓ 5 tỷ"        -> chữ "có" từng nằm trong danh sách này, nên câu
#                           thẳng thắn nhất của người mua thành "khoảng 5 tỷ".
#   "gần trung TÂM 4 tỷ" -> chữ "tâm" của "trung tâm" nằm ngay trước con số.
#
# Nên: (1) bỏ hẳn những từ quá phổ thông ("có", "cũng"); (2) chỉ nhận khi từ
# xấp xỉ đứng SÁT ngay trước con số, không phải "ở đâu đó trong 22 ký tự
# trước"; (3) loại riêng cụm "trung tâm".
_TU_XAP_XI = ("khoang", "tam", "xap xi", "chung", "quanh", "tren duoi")
_XAP_XI_TRU = ("trung tam",)
_BIEN_XAP_XI = 0.10          # ±10% quanh con số người dùng nêu

# Đơn vị đứng SAU phần lẻ thì phần lẻ đó không phải phần lẻ của tiền:
# "3 tỷ 2 ngủ" là "3 tỷ" rồi "2 phòng ngủ", không phải "3,2 tỷ".
_SAU_PHAN_LE = r"(?!\s*(?:pn|phong|ngu|wc|vs|ve sinh|tang|m2|m²|met|m\b|ty|tỷ|ti|tỉ|tr|trieu))"


def doc_gia(cau: str) -> dict:
    """Bóc khoảng giá từ câu. Trả về {'gia_tu': ..., 'gia_den': ...} (VND).

    Xử lý được:
        "5 tỷ"              -> coi là TRẦN (dưới 5 tỷ) — xem ghi chú dưới
        "dưới 5 tỷ"         -> trần 5 tỷ
        "trên 3 tỷ"         -> sàn 3 tỷ
        "3-5 tỷ", "3 đến 5 tỷ", "từ 3 tới 5 tỷ"  -> khoảng
        "khoảng 5 tỷ", "tầm 5 tỷ"                -> dải 4,5–5,5 tỷ
        "2 tỷ 8", "6 tỷ 5", "2 tỷ 85"            -> 2,8 / 6,5 / 2,85 tỷ
        "800 triệu"         -> trần 800 triệu
        "5ty", "5 tỉ", "5b" -> đều nhận

    VÌ SAO "5 TỶ" HIỂU LÀ TRẦN CHỨ KHÔNG PHẢI "ĐÚNG 5 TỶ"
    -----------------------------------------------------
    Người đi mua nói "tôi có 5 tỷ" nghĩa là 5 tỷ là mức tối đa, không phải họ
    muốn đúng một căn giá 5,00 tỷ. Đây là suy đoán về ý định, nên giao diện
    PHẢI hiện lại "đang lọc: giá ≤ 5 tỷ" để người dùng sửa nếu hiểu sai.
    """
    t = _chuan_so(cau)
    dv = "|".join(sorted(_DV_TIEN, key=len, reverse=True))
    ty = "|".join(_DV_TY)

    # (1) khoảng: "3 - 5 ty", "3 den 5 ty", "tu 3 toi 5 ty"
    m = re.search(rf"(\d+[.,]?\d*)\s*(?:{dv})?\s*(?:-|den|tới|toi|~)\s*"
                  rf"(\d+[.,]?\d*)\s*({dv})\b", t)
    if m:
        k = _DV_TIEN[m.group(3)]
        a = float(m.group(1).replace(",", ".")) * k
        b = float(m.group(2).replace(",", ".")) * k
        return {"gia_tu": min(a, b), "gia_den": max(a, b)}

    # (2) kiểu nói tiếng Việt "2 tỷ 8" = 2,8 tỷ. Phần lẻ 1 chữ số là phần
    #     mười, 2 chữ số là phần trăm, 3 chữ số là số triệu ("2 tỷ 850").
    m = re.search(rf"(\d+)\s*({ty})\s*(\d{{1,3}})\b{_SAU_PHAN_LE}", t)
    if m:
        le = m.group(3)
        so = (float(m.group(1)) + float(le) / 10 ** len(le)) * 1e9
        truoc = t[max(0, m.start() - 22):m.start()]
        return _huong_gia(so, truoc)

    # (3) một số kèm đơn vị, xem từ đứng trước để biết sàn, trần hay dải
    m = re.search(rf"(\d+[.,]?\d*)\s*({dv})\b", t)
    if not m:
        return {}
    so = float(m.group(1).replace(",", ".")) * _DV_TIEN[m.group(2)]
    return _huong_gia(so, t[max(0, m.start() - 22):m.start()])


def _ke_ngay_truoc(truoc: str, tu: tuple) -> bool:
    """Một trong các cụm này có đứng SÁT NGAY TRƯỚC con số không?

    VÌ SAO PHẢI "SÁT NGAY TRƯỚC" CHỨ KHÔNG PHẢI "ĐÂU ĐÓ TRONG 22 KÝ TỰ"
    ------------------------------------------------------------------
    Bản trước quét cả một đoạn 22 ký tự trước con số, nên một câu có HAI cụm
    số thì cụm sau bị gán hướng của cụm trước:

        "từ 60 m2 dưới 5 tỷ"  ->  thấy chữ "từ" -> hiểu "giá TỪ 5 tỷ"

    Chữ "từ" ấy thuộc về diện tích, không thuộc về giá. Người dùng nói rõ
    "dưới 5 tỷ" mà hệ thống lọc ngược thành "trên 5 tỷ" — và câu đó là câu
    người mua nhà gõ hằng ngày. Cùng loại lỗi với "gần trung tâm 4 tỷ" bị
    hiểu thành "khoảng 4 tỷ" vì chữ "tâm" của "trung tâm".

    Nên chỉ nhận khi cụm hướng là phần ĐUÔI của đoạn đứng trước con số.
    """
    t = truoc.rstrip()
    return any(re.search(rf"(?:^|\s){re.escape(w)}$", t) for w in tu)


def _xap_xi(truoc: str) -> bool:
    t = truoc.rstrip()
    for w in _XAP_XI_TRU:
        if t.endswith(w):
            return False
    return _ke_ngay_truoc(truoc, _TU_XAP_XI)


def _huong_gia(so: float, truoc: str) -> dict:
    """Con số này là sàn, trần, hay một dải quanh nó? Xét từ đứng SÁT trước."""
    if _xap_xi(truoc):
        return {"gia_tu": so * (1 - _BIEN_XAP_XI),
                "gia_den": so * (1 + _BIEN_XAP_XI)}
    if _ke_ngay_truoc(truoc, _TU_TREN):
        return {"gia_tu": so}
    return {"gia_den": so}          # mặc định: coi là trần


# =============================================================================
# BÓC DIỆN TÍCH
# =============================================================================

# Chữ đứng trước một con số kèm "m" mà làm cho "m" đó KHÔNG phải diện tích:
# "mặt tiền 5m", "ngõ rộng 3m", "đường vào 4m", "sâu 12m".
_TRUOC_KHONG_PHAI_DT = ("mat tien", "mt", "ngo", "hem", "duong vao", "duong",
                        "rong", "dai", "sau", "ngang", "lo gioi", "via he")


def doc_dien_tich(cau: str) -> dict:
    """Bóc diện tích: "60m2", "60 m²", "dưới 60m2", "từ 50 đến 70 m2", "100m".

    Chấp nhận cả "100m" trần trụi (người ta viết tắt m² thành m rất nhiều),
    NHƯNG chỉ khi trước đó không có chữ chỉ kích thước — nếu không thì
    "mặt tiền 5m" thành "diện tích từ 5 m²".
    """
    t = _chuan_so(cau)
    m = re.search(r"(\d+[.,]?\d*)\s*(?:-|den|toi|~)\s*(\d+[.,]?\d*)\s*"
                  r"(?:m2|m²|met|m)\b", t)
    if m:
        a = float(m.group(1).replace(",", "."))
        b = float(m.group(2).replace(",", "."))
        return {"dt_tu": min(a, b), "dt_den": max(a, b)}

    for m in re.finditer(r"(\d+[.,]?\d*)\s*(m2|m²|met vuong|met|m)\b", t):
        truoc = t[max(0, m.start() - 24):m.start()]
        if m.group(2) == "m" and any(
                re.search(rf"\b{re.escape(w)}\b", truoc)
                for w in _TRUOC_KHONG_PHAI_DT):
            continue                    # đây là một chiều dài, không phải diện tích
        so = float(m.group(1).replace(",", "."))
        if _ke_ngay_truoc(truoc, _TU_DUOI):
            return {"dt_den": so}
        # "60 m2" một mình: hiểu là TỪ 60 trở lên. Người mua nói diện tích
        # thường là nói mức tối thiểu họ cần ở được — ngược với tiền.
        return {"dt_tu": so}
    return {}


# =============================================================================
# BÓC SỐ PHÒNG
# =============================================================================

def doc_phong(cau: str) -> dict:
    """Bóc số phòng ngủ và phòng vệ sinh.

    "2PN", "2 phòng ngủ", "2 ngủ", "2 pn 2 wc", "3 phòng ngủ 2 vệ sinh".
    """
    t = _chuan_so(cau)
    ra = {}
    m = re.search(r"(\d+)\s*(?:pn|phong ngu|ngu|phong)\b", t)
    if m:
        ra["so_phong_ngu"] = int(m.group(1))
    m = re.search(r"(\d+)\s*(?:wc|vs|ve sinh|toilet|nha ve sinh)\b", t)
    if m:
        ra["so_phong_vs"] = int(m.group(1))
    return ra


# =============================================================================
# BÓC KÍCH THƯỚC: SỐ TẦNG, MẶT TIỀN, ĐỘ RỘNG NGÕ
# =============================================================================
# Ba thứ này người mua nhà đất nói liên tục ("nhà 4 tầng mặt tiền 5m, ngõ 3m
# ô tô vào") và dữ liệu có sẵn cả ba cột — `tong_so_tang` 91,9%, `mat_tien_m`
# 62,1%, `duong_rong_m` 39,6%. Bản trước bỏ qua hết, và tệ hơn: con số kèm "m"
# bị hiểu thành tiền.

def doc_kich_thuoc(cau: str) -> dict:
    """Bóc số tầng, mặt tiền, độ rộng ngõ/đường."""
    t = _chuan_so(cau)
    ra: dict = {}

    m = re.search(r"(\d+)\s*tang\b", t)
    if m:
        so = int(m.group(1))
        truoc = t[max(0, m.start() - 24):m.start()]
        if _ke_ngay_truoc(truoc, _TU_DUOI):
            ra["tang_den"] = so
        else:
            ra["tang_tu"] = so

    m = re.search(r"(?:mat tien|mt|ngang)\s*(?:rong\s*)?"
                  r"(\d+[.,]?\d*)\s*m?\b", t)
    if m:
        ra["mt_tu"] = float(m.group(1).replace(",", "."))

    m = re.search(r"(?:ngo|hem|duong vao|duong)\s*(?:rong\s*)?"
                  r"(\d+[.,]?\d*)\s*m\b", t)
    if m:
        ra["ngo_tu"] = float(m.group(1).replace(",", "."))
    return ra


# =============================================================================
# BÓC ĐỊA ĐIỂM
# =============================================================================
# TIỀN TỐ PHẢI CẮT RIÊNG CHO TỪNG CẤP — ĐÂY LÀ NGUỒN CỦA MỘT LỖI THẬT
#
# Bản trước cắt chung một danh sách tiền tố cho cả ba cấp, trong đó có
# "phuong ". Hậu quả: tên PHỐ "Phương Mai" bị cắt chữ "phương" (nó là phần của
# tên phố, không phải tiền tố hành chính!) và còn lại khoá "mai"; phố "Phượng
# Đồng" còn lại "dong". Từ đó "Hoàng Mai" khớp phố "mai", "Hà Đông" khớp phố
# "dong" — sai mà chẳng có lỗi nào báo ra. Đếm được 252 khoá ngắn ≤6 ký tự
# sinh ra kiểu đó, tức 252 cái bẫy đang chờ.
_TIEN_TO = {
    "quan":   ("quan ", "huyen ", "thi xa "),
    "phuong": ("phuong ", "xa ", "thi tran "),
    "duong":  ("duong ", "pho ", "ngo ", "ngach ", "hem ", "duong pho "),
    "du_an":  ("kdt ", "khu do thi ", "chung cu ", "toa "),
}

# Khoá tên PHỐ yếu hơn ngưỡng này thì bỏ, vì nó sẽ khớp bừa vào câu thường.
# Tên phố thật ở Hà Nội hầu hết từ hai chữ trở lên ("Trung Kính", "Lê Văn
# Lương"); khoá một chữ như "mai", "dong", "ngo" chỉ là rác sinh ra do cắt
# tiền tố hoặc do dữ liệu bẩn.
_TOI_THIEU_CHU_DUONG = 2
_TOI_THIEU_KY_TU_DUONG = 7


def _cat_tien_to(k: str, cap: str) -> str:
    for tt in _TIEN_TO.get(cap, ()):
        if k.startswith(tt):
            return k[len(tt):]
    return k                        # chỉ cắt MỘT tiền tố, không cắt lặp


@lru_cache(maxsize=4)
def _tu_dien_dia_diem(loai: str) -> dict:
    """Danh sách quận / phường mới / phường cũ / phố / dự án, dạng không dấu.

    Dựng từ CHÍNH dữ liệu chứ không gõ tay: phường nào không có trong dữ liệu
    thì cũng không có căn nào để trả về, nên đưa vào từ điển là vô ích.

    Riêng PHƯỜNG CŨ thì lấy từ bảng tra sáp nhập, vì đó là tên người dùng vẫn
    gọi hằng ngày — "Mỹ Đình", "Dịch Vọng", "Nhân Chính" không còn là phường
    từ 1/7/2025 nhưng không ai gọi tên mới. Không nhận chúng thì người dùng gõ
    đúng tên khu mình muốn mà hệ thống báo không hiểu.
    """
    df = vs.nap(loai)["df"]

    def bang(ds, cap):
        ra, thay = [], {}
        for x in ds:
            k = _cat_tien_to(_chuan(x), cap)
            if len(k) < 3:
                continue
            if cap == "duong" and (len(k.split()) < _TOI_THIEU_CHU_DUONG
                                   and len(k) < _TOI_THIEU_KY_TU_DUONG):
                continue
            ra.append((k, x))
            thay.setdefault(k, x)
        # dài trước ngắn: "tu liem" phải thử sau "bac tu liem"
        return sorted(ra, key=lambda p: -len(p[0]))

    quan = bang(sorted(df["quan_huyen"].dropna().unique()), "quan")
    phuong = bang(sorted(df["phuong_moi"].dropna().unique()), "phuong")
    duong = bang(sorted(df["duong_pho"].dropna().unique()), "duong") \
        if "duong_pho" in df.columns else []

    du_an = []
    if "du_an_clean" in df.columns:
        ten = (df["du_an_clean"].replace("Không xác định", np.nan)
               .dropna().unique())
        du_an = bang(sorted(ten), "du_an")

    return {"quan": quan, "phuong": phuong, "duong": duong,
            "du_an": du_an, "phuong_cu": _bang_phuong_cu(loai)}


@lru_cache(maxsize=4)
def _bang_phuong_cu(loai: str) -> list:
    """Tên phường TRƯỚC sáp nhập -> phường mới, chỉ giữ phường có dữ liệu.

    Bỏ luôn phần số đuôi để "Mỹ Đình" khớp được cả "Mỹ Đình 1" và "Mỹ Đình 2"
    — nhưng CHỈ khi mọi biến thể cùng trỏ về một phường mới. Nếu "X 1" và
    "X 2" nhập vào hai phường khác nhau thì tên gốc "X" là mơ hồ, và đoán bừa
    một trong hai còn tệ hơn không hiểu.
    """
    try:
        b = pd.read_excel(vs.DATA_DIR / "bang_tra_phuong.xlsx",
                          sheet_name="cu_sang_moi")
    except Exception:
        return []
    co = set(vs.nap(loai)["df"]["phuong_moi"].dropna())
    b = b[b["phuong_moi"].isin(co)]

    cap: dict[str, set] = {}
    for _, r in b.iterrows():
        k = _cat_tien_to(_chuan(r["phuong_cu"]), "phuong")
        if len(k) >= 3:
            cap.setdefault(k, set()).add(r["phuong_moi"])
        goc = re.sub(r"\s*\d+$", "", k)          # "my dinh 1" -> "my dinh"
        if goc != k and len(goc) >= 3:
            cap.setdefault(goc, set()).add(r["phuong_moi"])

    ra = [(k, next(iter(v))) for k, v in cap.items() if len(v) == 1]
    return sorted(ra, key=lambda p: -len(p[0]))


# maxsize PHẢI lớn hơn số (nhánh × cấp) = 2 × 5 = 10. Đặt 8 như bản đầu thì
# bộ đệm đầy và mỗi lượt phân tích lại biên dịch lại mẫu 2.299 tên phố: đo
# được 379 ms mỗi câu ở nhánh nhà đất, trong khi chung cư chỉ 4,7 ms. Không
# lỗi nào báo ra, chỉ là chậm — và chậm ở một hộp tìm kiếm thì người dùng
# tưởng hệ thống treo.
@lru_cache(maxsize=32)
def _mau_cap(loai: str, cap: str):
    """Gộp mọi tên của một cấp thành MỘT biểu thức chính quy đã biên dịch.

    Trước đây mỗi cấp chạy một vòng `re.search` cho từng tên — 2.299 lần cho
    riêng tên phố, và cả hàm phân tích mất 352 ms mỗi câu. Gộp thành một mẫu
    rồi quét một lượt là đủ, vì `re` tự xử lý phần chọn nhánh.
    """
    bang = (_tu_dien_dia_diem(loai)[cap] if cap != "phuong_cu"
            else _bang_phuong_cu(loai))
    if not bang:
        return None, {}
    khoa = sorted({k for k, _ in bang}, key=len, reverse=True)
    tra = {}
    for k, goc in bang:
        tra.setdefault(k, goc)
    mau = re.compile(r"\b(?:" + "|".join(re.escape(k) for k in khoa) + r")\b")
    return mau, tra


def _khop_ten(t: str, loai: str, cap: str) -> list:
    """Tên nào của cấp này xuất hiện trong câu, dài trước ngắn sau.

    VÌ SAO PHẢI CÓ BIÊN TỪ
    ----------------------
    Bản đầu so chuỗi con thô, và kết quả là "tôi muốn mua nhà 800 triệu" khớp
    phường "Phượng Trì": tên đó sau khi bỏ tiền tố còn lại "tri", mà "tri"
    nằm trong "trieu". Một chuỗi ba ký tự lọt vào giữa từ khác thì khớp bừa.
    Mẫu gộp ở `_mau_cap` đã kẹp `\\b` hai đầu nên chỉ khớp khi đứng thành từ.
    """
    mau, tra = _mau_cap(loai, cap)
    if mau is None:
        return []
    thay = {m.group(0) for m in mau.finditer(t)}
    return sorted(((k, tra[k]) for k in thay), key=lambda p: -len(p[0]))


def doc_dia_diem(cau: str, loai: str) -> dict:
    """Tìm tên dự án / quận / phường / phố trong câu, giữ lại mức HỢP LÝ.

    VẤN ĐỀ PHẢI XỬ LÝ: MỘT CÁI TÊN, BA CẤP
    --------------------------------------
    Sau sáp nhập, rất nhiều phường mới mang luôn tên quận: "Cầu Giấy",
    "Thanh Xuân", "Tây Hồ", "Hà Đông" vừa là tên quận vừa là tên phường, và
    đôi khi còn là tên phố. Bản đầu nhận cả ba nên câu "5 tỷ, 2 phòng ngủ ở
    Cầu Giấy" bị lọc thành `quận = Cầu Giấy` VÀ `phường = Cầu Giấy` VÀ
    `đường = Cầu Giấy` — từ 77 căn hợp lý còn 1 căn.

    Quy tắc: cùng MỘT cái tên khớp nhiều cấp thì giữ cấp RỘNG NHẤT (quận),
    vì "ở Cầu Giấy" gần như luôn nghĩa là cả quận. Chỉ nhận cấp hẹp khi người
    dùng nói rõ bằng tiền tố — "phố Cầu Giấy", "phường Cầu Giấy".

    Tên KHÁC nhau ở các cấp khác nhau thì giữ cả: "nhà phố Trung Kính, Cầu
    Giấy" là một ý định thu hẹp hợp lệ.
    """
    t = _chuan(cau)
    t_so = _thay_so_bang_chu(t)        # "quan 2 ba trung" -> "quan hai ba trung"

    def khop(cap):
        ra = _khop_ten(t, loai, cap)
        if t_so != t:
            # Chỉ nhận thêm từ bản đổi số sang chữ nếu khoá có từ HAI chữ trở
            # lên. Khoá một chữ dễ trùng với con số vừa đổi ("3" -> "ba") và
            # sinh ra khớp bừa.
            da = {k for k, _ in ra}
            ra += [(k, g) for k, g in _khop_ten(t_so, loai, cap)
                   if k not in da and len(k.split()) >= 2]
        return sorted(ra, key=lambda p: -len(p[0]))

    hit = {c: khop(c) for c in ("du_an", "duong", "phuong", "phuong_cu", "quan")}

    ra: dict = {}

    def co_tien_to(k: str, tien_to: tuple) -> bool:
        """Người dùng có nói rõ "phố X" / "phường X" không?

        "MẶT PHỐ Thanh Xuân" KHÔNG phải nói rõ tên phố — "mặt phố" là LOẠI
        HÌNH nhà. Bản trước không loại trường hợp này nên "nhà mặt phố Thanh
        Xuân" sinh ra thêm điều kiện `đường = Đường Thanh Xuân`, thu hẹp kết
        quả xuống gần như không còn gì. Cùng lỗi với "nhà phố", "shophouse".
        """
        for tt in tien_to:
            if re.search(rf"(?<!mat )(?<!nha )\b{tt}\s+{re.escape(k)}\b", t_so):
                return True
        return False

    ten_quan = {k for k, _ in hit["quan"]}
    ten_phuong = {k for k, _ in hit["phuong"]}
    # Tên phường CŨ cũng phải nằm trong bộ chống trùng: dữ liệu có phố tên
    # "Mỹ Đình", "Dịch Vọng" — mà đó cũng là tên phường trước sáp nhập, và khi
    # người ta gõ "nhà Mỹ Đình" thì họ đang nói KHU, không nói một con phố.
    ten_phuong_cu = {k for k, _ in hit["phuong_cu"]}

    # (0) DỰ ÁN trước tiên: tên dự án là thứ cụ thể nhất người dùng nêu, và
    #     hầu như không trùng tên hành chính ("Times City", "Royal City").
    for k, goc in hit["du_an"]:
        ra["du_an"] = goc
        break

    for k, goc in hit["duong"]:
        # Trùng tên với quận/phường thì chỉ nhận khi có chữ "phố/đường/ngõ".
        if k in ten_quan or k in ten_phuong or k in ten_phuong_cu:
            # KHÔNG nhận "ngõ"/"hẻm" làm tiền tố tên phố: "nhà ngõ Thanh Xuân"
            # là LOẠI HÌNH (nhà trong ngõ) ở Thanh Xuân, không phải một con
            # ngõ tên Thanh Xuân. Phố thật thì người ta nói "phố" hoặc "đường".
            if not co_tien_to(k, ("pho", "duong")):
                continue
        if ra.get("du_an") and k in _chuan(ra["du_an"]):
            continue                   # tên phố nằm trong tên dự án đã nhận
        ra["duong_pho"] = goc
        break

    for k, goc in hit["phuong"]:
        if k in ten_quan and not co_tien_to(k, ("phuong", "xa", "thi tran")):
            continue
        if ra.get("duong_pho") and k == _chuan(ra["duong_pho"]):
            continue
        ra["phuong_moi"] = goc
        break

    # (2b) Tên phường CŨ — chỉ dùng khi chưa nhận được phường mới nào, và tên
    #      đó không trùng tên quận (nếu trùng thì ý người dùng là cả quận).
    if "phuong_moi" not in ra:
        for k, goc in hit["phuong_cu"]:
            if k in ten_quan:
                continue
            if ra.get("duong_pho") and k == _chuan(ra["duong_pho"]):
                continue
            ra["phuong_moi"] = goc
            ra["_phuong_go_cu"] = k     # để giao diện nói "hiểu tên cũ -> mới"
            break

    for k, goc in hit["quan"]:
        ra["quan_huyen"] = goc
        break
    return ra


# =============================================================================
# BÓC ĐẶC ĐIỂM
# =============================================================================

# Mỗi mục: (tên cột trong dữ liệu, các cách người dùng có thể gõ)
#
# MỌI TỪ Ở ĐÂY ĐỀU KHỚP THEO BIÊN TỪ, và đó là lý do danh sách này không còn
# chữ "cho" trần trụi: "nhà CHO gia đình 4 người" từng bị hiểu thành "gần
# chợ, siêu thị", vì "cho" đứng thành một từ riêng nên biên từ cũng không cứu
# được. Từ nào ngắn và trùng từ thường thì phải viết thành cụm.
_CO_KHONG = [
    ("nhac_o_to",       ("o to", "oto", "xe hoi", "do xe", "gara", "garage",
                         "oto vao", "o to vao")),
    ("nhac_thang_may",  ("thang may", "elevator")),
    # KHÔNG để "mat bang" trần trụi ở đây: trong chính ứng dụng này "mặt bằng"
    # còn nghĩa là MẶT BẰNG GIÁ, nên câu "căn nào đang rẻ hơn mặt bằng" bị
    # hiểu thêm thành "tiện kinh doanh" và lọc mất phần lớn kết quả.
    ("nhac_kinh_doanh", ("kinh doanh", "mat bang kinh doanh", "mat bang thuong mai",
                         "mat bang cho thue", "buon ban", "mo shop", "mo quan")),
    ("nhac_truong",     ("truong hoc", "gan truong", "truong cap", "mam non")),
    ("nhac_benh_vien",  ("benh vien", "phong kham")),
    ("nhac_cho_sieu_thi", ("sieu thi", "gan cho", "di cho", "khu cho",
                           "cho dan sinh", "trung tam thuong mai", "ttttm")),
    ("nhac_cong_vien",  ("cong vien", "ho tay", "gan ho", "cay xanh")),
    ("nhac_metro",      ("metro", "duong sat tren cao", "tau dien")),
    ("nhac_be_boi",     ("be boi", "ho boi")),
    ("nhac_gym",        ("gym", "phong tap")),
    ("nhac_lo_goc",     ("lo goc", "can goc", "goc hai mat")),
    ("nhac_view",       ("view", "tam nhin")),
    ("nhac_chinh_chu",  ("chinh chu", "khong qua moi gioi")),
]

_LOAI_HINH = [
    ("Nhà ngõ, hẻm",          ("nha ngo", "trong ngo", "ngo hem", "hem",
                               "nha rieng", "nha trong ngo")),
    ("Nhà mặt phố, mặt tiền", ("mat pho", "mat tien", "mat duong", "nha pho")),
    ("Nhà phố liền kề",       ("lien ke", "shophouse", "nha lien ke")),
    ("Nhà biệt thự",          ("biet thu", "villa")),
]

_PHAP_LY = [
    (("Sổ hồng riêng", "Đã có sổ"), ("so hong", "so do", "co so", "so rieng",
                                     "day du giay to", "phap ly ro")),
    (("Đang chờ sổ",),              ("cho so", "dang cho so")),
]

# "gần trung tâm" là một yêu cầu rất hay gặp mà bản trước bỏ qua hoàn toàn,
# dù cột `kc_trung_tam_km` có đủ 100%. Ngưỡng 5 km KHÔNG đặt tay: đó là phân
# vị 25 của chính dữ liệu (chung cư 5,44 km, nhà đất 4,97 km), tức "một phần
# tư số căn gần trung tâm nhất". Ở ngưỡng này còn 1.150 căn chung cư và 3.670
# căn nhà đất — đủ rộng để còn gì mà xem.
_TU_TRUNG_TAM = ("trung tam", "noi thanh", "trung tam thanh pho", "gan ho guom",
                 "pho co")
NGUONG_TRUNG_TAM_KM = 5.0


def _tim_cum(t: str, tu: tuple) -> int | None:
    """Vị trí đầu tiên trong câu khớp một trong các cụm, theo biên từ."""
    for w in tu:
        m = re.search(rf"\b{re.escape(w)}\b", t)
        if m:
            return m.start()
    return None


def _cum_khop(t: str, tu: tuple) -> str:
    """Cụm nào trong `tu` đã khớp — cần để biết nó dài bao nhiêu ký tự."""
    for w in tu:
        if re.search(rf"\b{re.escape(w)}\b", t):
            return w
    return ""


def doc_dac_diem(cau: str, loai: str) -> dict:
    """Bóc yêu cầu có / KHÔNG, loại hình, pháp lý, gần trung tâm."""
    t = _chuan(cau)
    df = vs.nap(loai)["df"]
    ra: dict = {"co": [], "khong": [], "loai_hinh_tru": []}

    for cot, tu in _CO_KHONG:
        if cot not in df.columns:
            continue
        vi = _tim_cum(t, tu)
        if vi is None:
            continue
        (ra["khong"] if _phu_dinh_truoc(t, vi) else ra["co"]).append(cot)

    if loai == "nhadat":
        for nhan, tu in _LOAI_HINH:
            vi = _tim_cum(t, tu)
            if vi is None:
                continue
            # "MẶT TIỀN 5M" LÀ MỘT SỐ ĐO, KHÔNG PHẢI LOẠI NHÀ.
            #
            # "mặt tiền" vừa là dấu hiệu nhà mặt phố, vừa là tên một kích
            # thước. Câu "nhà 4 tầng mặt tiền 5m trong ngõ" từng bị gán luôn
            # loại hình "Nhà mặt phố, mặt tiền" — trái hẳn chữ "trong ngõ" ở
            # ngay sau, và thu kết quả xuống còn 1 căn. Có số đi liền sau thì
            # đó là số đo.
            if re.match(r"\s*(?:rong\s*)?\d", t[vi + len(_cum_khop(t, tu)):]):
                continue
            if _phu_dinh_truoc(t, vi):
                ra["loai_hinh_tru"].append(nhan)
            elif "loai_hinh" not in ra:
                ra["loai_hinh"] = nhan

    for nhan, tu in _PHAP_LY:
        vi = _tim_cum(t, tu)
        if vi is None or _phu_dinh_truoc(t, vi):
            continue
        co = [x for x in nhan if x in set(df["phap_ly"].dropna())]
        if co:
            ra["phap_ly"] = co
            break

    vi = _tim_cum(t, _TU_TRUNG_TAM)
    if vi is not None and not _phu_dinh_truoc(t, vi):
        ra["kc_den"] = NGUONG_TRUNG_TAM_KM
    return ra


# =============================================================================
# Ý ĐỊNH SẮP XẾP NẰNG TRONG CÂU
# =============================================================================
# Người dùng gõ "nhà nào đang rẻ hơn thị trường" là đang nói CÁCH SẮP, không
# phải một điều kiện lọc. Bản trước bỏ qua, nên câu đó chỉ còn lại "ở Hà Đông"
# và danh sách trả về chẳng liên quan gì tới ý họ.
#
# Thứ tự xét có ý nghĩa: cụm "rẻ hơn thị trường" phải được thử TRƯỚC "rẻ
# nhất", vì cả hai đều chứa chữ "rẻ".
_Y_DINH_SAP = [
    ("lech_thap", ("re hon thi truong", "re hon mat bang", "duoi gia thi truong",
                   "duoi mat bang", "gia tot", "dang re hon", "lech thap",
                   "re hon so")),
    ("lech_cao",  ("cao hon mat bang", "dat hon thi truong", "rao cao",
                   "cao hon thi truong", "bi rao cao", "lech cao")),
    ("re_nhat",   ("re nhat", "gia thap nhat", "thap nhat", "re nhat co the")),
]


def doc_sap(cau: str) -> dict:
    t = _chuan(cau)
    for nhan, tu in _Y_DINH_SAP:
        if _tim_cum(t, tu) is not None:
            return {"sap": nhan}
    return {}


# =============================================================================
# ĐOÁN LOẠI BẤT ĐỘNG SẢN
# =============================================================================

# ĐÃ BỎ "tang" khỏi nhóm chung cư: từ khi bộ phân tích hiểu được số tầng nhà
# đất, câu "nhà 4 tầng trong ngõ" lại mang một dấu hiệu "chung cư" — trong khi
# nó là câu nhà đất rõ ràng nhất có thể.
_TU_CHUNGCU = ("chung cu", "can ho", "apartment", "toa nha", "ban cong",
               "toa", "block")
_TU_NHADAT = ("nha dat", "nha rieng", "tho cu", "nha ngo", "mat pho",
              "lien ke", "biet thu", "ngo", "hem", "mat tien", "san thuong",
              "dien tich dat", "so do")


def doan_loai(cau: str, mac_dinh: str = "chungcu") -> str:
    """Câu hỏi đang nói về chung cư hay nhà đất.

    Đếm số từ khoá mỗi bên. Bằng nhau thì giữ lựa chọn hiện tại của người dùng
    — không tự nhảy nhánh, vì nhảy sai còn tệ hơn không nhảy.
    """
    t = _chuan(cau)
    a = sum(_tim_cum(t, (w,)) is not None for w in _TU_CHUNGCU)
    b = sum(_tim_cum(t, (w,)) is not None for w in _TU_NHADAT)
    if a > b:
        return "chungcu"
    if b > a:
        return "nhadat"

    # HOÀ PHIẾU TỪ KHOÁ THÌ XÉT ĐIỀU KIỆN CHỈ MỘT NHÁNH MỚI CÓ
    #
    # Câu "nhà 4 tầng dưới 6 tỷ Hoàng Mai ô tô vào được" không chứa từ khoá
    # nào trong cả hai danh sách — "nhà 4 tầng" không phải "nhà riêng", "ô tô"
    # không phải "ngõ" — nên nó hoà 0-0 và rơi về nhánh mặc định là chung cư.
    # Một câu nhà đất rõ ràng như thế bị đưa sang nhánh chung cư thì mọi kết
    # quả sau đó đều sai loại.
    #
    # Nhưng SỐ TẦNG, MẶT TIỀN và ĐỘ RỘNG NGÕ là ba thứ chỉ nhánh nhà đất có
    # cột dữ liệu. Bóc được một trong ba mà không có dấu hiệu chung cư nào thì
    # đó là bằng chứng mạnh hơn hẳn phép đếm từ khoá.
    #
    # KHÔNG dùng chữ "nhà" trần trụi làm dấu hiệu, dù nó cũng hay đi với nhà
    # đất: "mua nhà" trong tiếng Việt là mua chỗ ở nói chung, căn hộ cũng là
    # "nhà". Đoán theo nó là đoán bừa trên một từ mơ hồ — thà giữ lựa chọn
    # người dùng đang có.
    if a == 0 and set(doc_kich_thuoc(cau)) & {"tang_tu", "tang_den",
                                              "mt_tu", "ngo_tu"}:
        return "nhadat"
    return mac_dinh


# =============================================================================
# PHÂN TÍCH CẢ CÂU
# =============================================================================

def phan_tich(cau: str, loai: str) -> dict:
    """Câu tiếng Việt -> dict điều kiện. Đây là "bộ não" của tính năng.

    Trả về dict phẳng để giao diện hiện lại được từng điều kiện đã hiểu, và
    cho người dùng bỏ từng cái nếu hiểu sai. KHÔNG bao giờ lọc theo một điều
    kiện mà không nói ra.
    """
    dk: dict = {}
    dk.update(doc_gia(cau))
    dk.update(doc_dien_tich(cau))
    dk.update(doc_phong(cau))
    dk.update(doc_kich_thuoc(cau))
    dk.update(doc_dia_diem(cau, loai))
    dk.update(doc_sap(cau))

    dd = doc_dac_diem(cau, loai)
    for k in ("co", "khong", "loai_hinh_tru"):
        if dd.get(k):
            dk[k] = dd[k]
    for k in ("loai_hinh", "phap_ly", "kc_den"):
        if dd.get(k):
            dk[k] = dd[k]
    # Vừa đòi một loại hình vừa loại chính nó ra thì lời loại trừ thắng: người
    # ta viết "nhà dưới 5 tỷ nhưng không ở trong ngõ" — chữ "ngõ" xuất hiện
    # một lần và nó bị phủ định, nên không được biến thành yêu cầu.
    if dk.get("loai_hinh") and dk["loai_hinh"] in dk.get("loai_hinh_tru", []):
        dk.pop("loai_hinh")
    return dk


# =============================================================================
# ÁP ĐIỀU KIỆN LÊN DỮ LIỆU
# =============================================================================

NHAN_DK = {
    "gia_tu": "giá từ", "gia_den": "giá đến",
    "dt_tu": "diện tích từ", "dt_den": "diện tích đến",
    "so_phong_ngu": "số phòng ngủ", "so_phong_vs": "số phòng vệ sinh",
    "tang_tu": "số tầng từ", "tang_den": "số tầng đến",
    "mt_tu": "mặt tiền từ", "ngo_tu": "độ rộng ngõ từ",
    "quan_huyen": "quận/huyện", "phuong_moi": "phường", "duong_pho": "đường",
    "du_an": "dự án", "kc_den": "khoảng cách tới trung tâm",
    "loai_hinh": "loại hình", "loai_hinh_tru": "loại trừ loại hình",
    "phap_ly": "pháp lý", "co": "yêu cầu thêm", "khong": "không muốn",
}

NHAN_CO = {
    "nhac_o_to": "ô tô vào được", "nhac_thang_may": "có thang máy",
    "nhac_kinh_doanh": "tiện kinh doanh", "nhac_truong": "gần trường học",
    "nhac_benh_vien": "gần bệnh viện", "nhac_cho_sieu_thi": "gần chợ, siêu thị",
    "nhac_cong_vien": "gần công viên, hồ", "nhac_metro": "gần metro",
    "nhac_be_boi": "có bể bơi", "nhac_gym": "có phòng gym",
    "nhac_lo_goc": "lô góc", "nhac_view": "có view",
    "nhac_chinh_chu": "chính chủ",
}

# Cột chứa số tầng khác nhau ở hai nhánh: nhà đất là tổng số tầng của căn,
# chung cư là tầng mà căn hộ nằm ở. Không phải cùng một đại lượng, nhưng câu
# "nhà 4 tầng" / "căn tầng 12" đều dùng chữ "tầng".
COT_TANG = {"nhadat": "tong_so_tang", "chungcu": "tang_so"}


def _bang_lech(loai: str) -> pd.DataFrame:
    """Bảng mức lệch so với mặt bằng, cho từng căn — phần việc của Săn tin cũ.

    Dùng khoảng tin cậy tính NGOÀI MẪU: mỗi căn được đoán bởi mô hình chưa
    từng thấy chính nó. Nếu tính trong mẫu thì mô hình đã nhớ giá của căn đó
    rồi, và "lệch so với dự đoán" mất hết ý nghĩa.
    """
    import tinh_nang as tn
    return tn._khoang_ngoai_mau(loai)


def tim(cau: str, loai: str, dk_them: dict | None = None,
        sap: str | None = None, k: int = 30) -> dict:
    """Tìm các căn khớp điều kiện trong câu.

    `sap` chọn cách xếp hạng:
        "khop_nhat"  — gần giữa khoảng giá/diện tích đã nêu nhất
        "re_nhat"    — giá thấp trước
        "lech_thap"  — rẻ hơn mặt bằng nhiều nhất trước  (Săn tin cũ)
        "lech_cao"   — đắt hơn mặt bằng nhiều nhất trước

    Truyền `sap=None` thì lấy ý định sắp xếp NẰM TRONG CÂU nếu có, không thì
    "khop_nhat". Người dùng bấm đổi cách sắp trên giao diện thì tham số này
    khác None và nó thắng — lựa chọn tay bao giờ cũng thắng máy đoán.

    Trả về dict: {dk, so_khop, ket_qua, bo_qua} — `bo_qua` ghi những điều kiện
    phải nới ra vì không còn căn nào, để giao diện nói cho người dùng biết.
    """
    dk = {**phan_tich(cau, loai), **(dk_them or {})}
    if sap is None:
        sap = dk.get("sap", "khop_nhat")

    g = vs.nap(loai)
    df, cfg = g["df"], g["cfg"]
    cot_dt = cfg.cot_dien_tich
    cot_tang = COT_TANG.get(loai)

    d = df[df["TARGET_gia_vnd"].notna()].copy()
    # Ghép mức lệch so với mặt bằng vào mọi kết quả, không chỉ khi sắp theo nó:
    # đây là thông tin người mua cần thấy ngay cạnh từng căn.
    try:
        lech = _bang_lech(loai)[["lech_phan_tram", "gia_du_doan",
                                 "khoang_duoi", "khoang_tren", "vi_the"]]
        d = d.join(lech, how="left")
    except Exception:
        pass

    def _co_cot(c):
        return c in d.columns

    dieu_kien = [
        ("gia_tu", lambda v: d["TARGET_gia_vnd"] >= v),
        ("gia_den", lambda v: d["TARGET_gia_vnd"] <= v),
        ("dt_tu", lambda v: d[cot_dt] >= v),
        ("dt_den", lambda v: d[cot_dt] <= v),
        ("so_phong_ngu", lambda v: d["so_phong_ngu"] == v),
        ("so_phong_vs", lambda v: d["so_phong_vs"] == v),
        ("tang_tu", lambda v: d[cot_tang] >= v if _co_cot(cot_tang) else None),
        ("tang_den", lambda v: d[cot_tang] <= v if _co_cot(cot_tang) else None),
        ("mt_tu", lambda v: d["mat_tien_m"] >= v if _co_cot("mat_tien_m") else None),
        ("ngo_tu", lambda v: d["duong_rong_m"] >= v
         if _co_cot("duong_rong_m") else None),
        ("kc_den", lambda v: d["kc_trung_tam_km"] <= v
         if _co_cot("kc_trung_tam_km") else None),
        ("du_an", lambda v: d["du_an_clean"] == v if _co_cot("du_an_clean") else None),
        ("duong_pho", lambda v: d["duong_pho"] == v),
        ("phuong_moi", lambda v: d["phuong_moi"] == v),
        ("quan_huyen", lambda v: d["quan_huyen"] == v),
        ("loai_hinh", lambda v: d["loai_hinh"] == v),
        ("loai_hinh_tru", lambda v: ~d["loai_hinh"].isin(v)),
        ("phap_ly", lambda v: d["phap_ly"].isin(v)),
    ]

    mat_na = pd.Series(True, index=d.index)
    bo_qua = []
    for ten, ham in dieu_kien:
        if ten not in dk:
            continue
        try:
            mk = ham(dk[ten])
        except Exception:
            continue
        if mk is None:              # nhánh này không có cột đó
            bo_qua.append(ten)
            continue
        m2 = mat_na & mk.fillna(False)
        # NỚI DẦN: điều kiện nào làm cạn kết quả thì bỏ nó ra và ghi lại.
        # Trả về danh sách rỗng kèm im lặng là cách nhanh nhất làm người dùng
        # bỏ đi; thà trả kết quả gần đúng và nói rõ đã nới chỗ nào.
        if m2.sum() == 0:
            bo_qua.append(ten)
        else:
            mat_na = m2

    for cot in dk.get("co", []):
        if cot in d.columns:
            m2 = mat_na & (d[cot].fillna(0) > 0)
            if m2.sum() == 0:
                bo_qua.append(cot)
            else:
                mat_na = m2

    # "không muốn X" lọc bỏ những căn CÓ nhắc tới X. Lưu ý đây là dữ liệu suy
    # từ mô tả tin rao, nên "không nhắc" không đảm bảo là "không có" — giao
    # diện phải nói rõ chỗ này thay vì hứa chắc.
    for cot in dk.get("khong", []):
        if cot in d.columns:
            m2 = mat_na & ~(d[cot].fillna(0) > 0)
            if m2.sum() == 0:
                bo_qua.append(cot)
            else:
                mat_na = m2

    kq = d[mat_na].copy()
    # THỐNG KÊ CỦA CẢ TẬP KHỚP, tính trước khi cắt còn k căn đầu.
    #
    # Cần cho những câu tiếp kiểu "rẻ hơn nữa": bước giá không đặt tay mà lấy
    # phân vị của CHÍNH tập đang xem (xem `_moc_tuong_doi`). Phải tính trên cả
    # tập, không phải trên k căn đầu — k căn đầu đã bị sắp xếp nên phân vị của
    # nó không nói gì về thị trường.
    thong_ke = {}
    if not kq.empty:
        thong_ke = {
            "gia_q25": float(kq["TARGET_gia_vnd"].quantile(.25)),
            "gia_q75": float(kq["TARGET_gia_vnd"].quantile(.75)),
            "dt_q25": float(kq[cot_dt].quantile(.25)),
            "dt_q75": float(kq[cot_dt].quantile(.75)),
        }
    if kq.empty:
        return {"dk": dk, "so_khop": 0, "ket_qua": kq, "bo_qua": bo_qua,
                "loai": loai, "sap": sap, "thong_ke": thong_ke}

    if sap == "re_nhat":
        kq = kq.sort_values("TARGET_gia_vnd")
    elif sap == "lech_thap" and "lech_phan_tram" in kq.columns:
        kq = kq.sort_values("lech_phan_tram")
    elif sap == "lech_cao" and "lech_phan_tram" in kq.columns:
        kq = kq.sort_values("lech_phan_tram", ascending=False)
    else:
        # "khớp nhất": gần giữa khoảng người dùng nêu nhất. Nếu họ chỉ nêu
        # trần giá thì mốc là chính cái trần — người có 5 tỷ thường muốn xem
        # căn sát 5 tỷ chứ không phải căn 1 tỷ.
        moc_g = dk.get("gia_den") or dk.get("gia_tu")
        moc_d = dk.get("dt_tu") or dk.get("dt_den")
        diem = pd.Series(0.0, index=kq.index)
        if moc_g:
            diem += (kq["TARGET_gia_vnd"] - moc_g).abs() / moc_g
        if moc_d:
            diem += (kq[cot_dt] - moc_d).abs() / moc_d
        kq = kq.assign(_diem=diem).sort_values("_diem").drop(columns="_diem")

    return {"dk": dk, "so_khop": int(mat_na.sum()),
            "ket_qua": kq.head(k).reset_index(drop=True),
            "bo_qua": bo_qua, "loai": loai, "sap": sap,
            "thong_ke": thong_ke}


NHAN_SAP = {
    "khop_nhat": "khớp điều kiện nhất",
    "re_nhat": "giá thấp nhất trước",
    "lech_thap": "rẻ hơn mặt bằng nhiều nhất trước",
    "lech_cao": "cao hơn mặt bằng nhiều nhất trước",
}


def mo_ta_dk(dk: dict, loai: str) -> list[str]:
    """Đổi dict điều kiện thành danh sách câu tiếng Việt để hiện lại.

    Bắt buộc phải có: người dùng cần thấy hệ thống hiểu gì trước khi tin vào
    danh sách kết quả.
    """
    ra = []
    if dk.get("gia_tu") and dk.get("gia_den"):
        ra.append(f"giá {dk['gia_tu']/1e9:.2f}–{dk['gia_den']/1e9:.2f} tỷ")
    elif dk.get("gia_den"):
        ra.append(f"giá không quá {dk['gia_den']/1e9:.2f} tỷ")
    elif dk.get("gia_tu"):
        ra.append(f"giá từ {dk['gia_tu']/1e9:.2f} tỷ")

    if dk.get("dt_tu") and dk.get("dt_den"):
        ra.append(f"diện tích {dk['dt_tu']:.0f}–{dk['dt_den']:.0f} m²")
    elif dk.get("dt_tu"):
        ra.append(f"diện tích từ {dk['dt_tu']:.0f} m²")
    elif dk.get("dt_den"):
        ra.append(f"diện tích tối đa {dk['dt_den']:.0f} m²")

    if dk.get("so_phong_ngu"):
        ra.append(f"{dk['so_phong_ngu']} phòng ngủ")
    if dk.get("so_phong_vs"):
        ra.append(f"{dk['so_phong_vs']} phòng vệ sinh")

    nha_tang = "tổng số tầng" if loai == "nhadat" else "tầng"
    if dk.get("tang_tu") and dk.get("tang_den"):
        ra.append(f"{nha_tang} {dk['tang_tu']:.0f}–{dk['tang_den']:.0f}")
    elif dk.get("tang_tu"):
        ra.append(f"{nha_tang} từ {dk['tang_tu']:.0f}")
    elif dk.get("tang_den"):
        ra.append(f"{nha_tang} tối đa {dk['tang_den']:.0f}")

    if dk.get("mt_tu"):
        ra.append(f"mặt tiền từ {dk['mt_tu']:.1f} m".replace(".", ","))
    if dk.get("ngo_tu"):
        ra.append(f"ngõ/đường rộng từ {dk['ngo_tu']:.1f} m".replace(".", ","))
    if dk.get("kc_den"):
        ra.append(f"cách trung tâm không quá {dk['kc_den']:.0f} km")

    if dk.get("du_an"):
        ra.append(f"dự án {dk['du_an']}")
    for c in ("duong_pho", "phuong_moi", "quan_huyen", "loai_hinh"):
        if dk.get(c):
            ra.append(str(dk[c]))
    if dk.get("loai_hinh_tru"):
        ra.append("không lấy " + ", ".join(dk["loai_hinh_tru"]).lower())
    if dk.get("phap_ly"):
        ra.append(" hoặc ".join(dk["phap_ly"]))
    for c in dk.get("co", []):
        ra.append(NHAN_CO.get(c, c))
    for c in dk.get("khong", []):
        nh = NHAN_CO.get(c, c)
        ra.append("không cần " + nh.replace("có ", "").replace("gần ", "gần "))
    return ra


# =============================================================================
# HỘI THOẠI NHIỀU LƯỢT
# =============================================================================
# Ý TƯỞNG: LƯỢT SAU SỬA ĐIỀU KIỆN CỦA LƯỢT TRƯỚC, KHÔNG TÌM LẠI TỪ ĐẦU
#
# Người đi mua nhà không mô tả xong nhu cầu trong một câu. Họ nói "5 tỷ 2 ngủ
# Cầu Giấy", xem kết quả, rồi nói "rẻ hơn nữa", "thêm chỗ đỗ xe", "thử Thanh
# Xuân xem". Mỗi câu sau là một phép SỬA trên điều kiện đang có — và đó là
# khác biệt duy nhất giữa một hộp tìm kiếm và một cuộc hội thoại.
#
# BA NGUYÊN TẮC, cả ba đều để tránh việc máy tự tiện
#
# 1. KHÔNG ĐOÁN NGẦM ĐÂY LÀ CÂU TIẾP HAY CÂU MỚI. Câu có dấu hiệu sửa đổi
#    ("rẻ hơn", "thêm", "bỏ", "đổi sang") thì nối tiếp; câu tự nó đã đủ điều
#    kiện mới thì thay hẳn. Và luôn NÓI RA mình đang làm gì, kèm nút làm lại.
#
# 2. BƯỚC THAY ĐỔI LẤY TỪ DỮ LIỆU, KHÔNG ĐẶT TAY. "rẻ hơn nữa" không phải
#    "giảm 20%" — nó hạ trần giá xuống phân vị 25 của CHÍNH tập kết quả đang
#    xem. Nhờ vậy mỗi lần nói "rẻ hơn nữa" luôn còn lại khoảng một phần tư số
#    căn, thay vì có lúc mất sạch có lúc chẳng đổi gì. Giảm 20% cố định trên
#    một tập đã hẹp thì lần thứ hai là về không.
#
# 3. HOÀN TÁC ĐƯỢC. Mỗi lượt lưu lại nguyên dict điều kiện, nên bấm "quay lại"
#    là về đúng trạng thái trước. Máy đoán sai mà không lùi được thì người
#    dùng phải gõ lại từ đầu — đó là lúc họ bỏ đi.

_TU_RE_HON = ("re hon", "re nua", "ha gia", "giam gia", "bot tien",
              "it tien hon", "re hon nua", "thap hon")
_TU_DAT_HON = ("dat hon", "tang ngan sach", "them tien", "noi gia",
               "cao hon nua", "dat nua")
_TU_RONG_HON = ("rong hon", "to hon", "rong nua", "rong ra hon", "lon hon")
_TU_HEP_HON = ("nho hon", "hep hon", "bot dien tich", "nho nua")
_TU_LAM_LAI = ("tim lai", "lam lai", "bo het", "xoa het", "tu dau",
               "bat dau lai", "tim moi")
_TU_HOAN_TAC = ("quay lai", "hoan tac", "tra lai", "nhu cu", "lui lai")
_TU_MO_RONG = ("mo rong", "ca quan", "rong ra", "toan quan", "khong bo phuong")
_TU_DOI_SANG = ("doi sang", "doi qua", "thu", "chuyen sang", "sang", "o")
# Chữ báo hiệu BỎ một điều kiện đang có, khác hẳn với phủ định thường:
# "bỏ điều kiện thang máy" = thôi không đòi thang máy nữa;
# "loại căn có thang máy" = lọc bỏ những căn có thang máy.
_TU_BO_DK = ("bo dieu kien", "bo yeu cau", "bo", "thoi khong can",
             "khong can nua", "khong quan trong", "de sau")

# Từ chỉ rõ đây là câu TIẾP, không phải câu mới. Có bất kỳ từ nào trong đây
# thì nối vào điều kiện cũ.
_DAU_HIEU_CAU_TIEP = (_TU_RE_HON + _TU_DAT_HON + _TU_RONG_HON + _TU_HEP_HON
                      + _TU_MO_RONG + _TU_BO_DK
                      + ("them", "va", "con", "nua", "thay vi", "doi sang",
                         "doi qua", "chuyen sang", "ngoai ra", "nhung ma",
                         "van", "giu nguyen", "cung duoc", "thu xem"))


def la_lam_lai(cau: str) -> bool:
    return _tim_cum(_chuan(cau), _TU_LAM_LAI) is not None


def la_hoan_tac(cau: str) -> bool:
    return _tim_cum(_chuan(cau), _TU_HOAN_TAC) is not None


def la_cau_tiep(cau: str, dk_cu: dict | None) -> bool:
    """Câu này là SỬA điều kiện cũ, hay là một tìm kiếm mới hẳn?

    Chưa có điều kiện cũ thì không thể là câu tiếp. Có rồi thì:
      - có dấu hiệu sửa đổi  -> câu tiếp
      - câu rất ngắn (≤ 5 từ) và bóc được ít nhất một điều kiện -> câu tiếp,
        vì "Thanh Xuân thì sao" hay "3 ngủ" là cách người ta nói tiếp
      - còn lại -> câu mới
    """
    if not dk_cu:
        return False
    t = _chuan(cau)
    if _tim_cum(t, _DAU_HIEU_CAU_TIEP) is not None:
        return True
    return len(t.split()) <= 5


def _moc_tuong_doi(thong_ke: dict, khoa: str, mac_dinh: float | None):
    """Mốc mới cho một phép thay đổi tương đối, lấy từ phân vị của tập hiện tại.

    Trả `mac_dinh` (thường là mốc cũ nhân một hệ số) khi chưa có thống kê —
    ví dụ lượt trước không khớp căn nào.
    """
    v = thong_ke.get(khoa)
    return float(v) if v else mac_dinh


def doc_chinh_sua(cau: str, dk_cu: dict, thong_ke: dict, loai: str) -> tuple:
    """Áp câu tiếp lên điều kiện cũ. Trả về (dk_moi, danh sách mô tả thay đổi).

    Mô tả thay đổi là phần BẮT BUỘC, không phải trang trí: người dùng nói "rẻ
    hơn nữa" thì phải đọc được "trần giá: 5,00 tỷ -> 3,80 tỷ" để biết máy hiểu
    đúng mức mình muốn, và sửa nếu không.
    """
    dk = dict(dk_cu)
    t = _chuan(cau)
    doi: list[str] = []

    def _ty(x):
        return f"{x/1e9:.2f} tỷ".replace(".", ",")

    # ---------------------------------------------------------------- giá
    if _tim_cum(t, _TU_RE_HON) is not None:
        cu = dk.get("gia_den")
        moi = _moc_tuong_doi(thong_ke, "gia_q25", cu * 0.8 if cu else None)
        if moi and (not cu or moi < cu):
            dk["gia_den"] = moi
            dk.pop("gia_tu", None)          # nói "rẻ hơn" thì bỏ luôn sàn giá
            doi.append(f"trần giá: {_ty(cu) if cu else 'chưa đặt'} → {_ty(moi)}")
        else:
            doi.append("muốn rẻ hơn, nhưng tập kết quả đã ở mức thấp nhất — "
                       "không hạ thêm được")

    if _tim_cum(t, _TU_DAT_HON) is not None:
        cu = dk.get("gia_den")
        moi = _moc_tuong_doi(thong_ke, "gia_q75", cu * 1.25 if cu else None)
        if moi and (not cu or moi > cu):
            dk["gia_den"] = moi
            doi.append(f"trần giá: {_ty(cu) if cu else 'chưa đặt'} → {_ty(moi)}")

    # ---------------------------------------------------------- diện tích
    if _tim_cum(t, _TU_RONG_HON) is not None:
        cu = dk.get("dt_tu")
        moi = _moc_tuong_doi(thong_ke, "dt_q75", cu * 1.25 if cu else None)
        if moi and (not cu or moi > cu):
            dk["dt_tu"] = moi
            dk.pop("dt_den", None)
            doi.append(f"diện tích tối thiểu: "
                       f"{f'{cu:.0f}' if cu else 'chưa đặt'} → {moi:.0f} m²")

    if _tim_cum(t, _TU_HEP_HON) is not None:
        cu = dk.get("dt_den")
        moi = _moc_tuong_doi(thong_ke, "dt_q25", cu * 0.8 if cu else None)
        if moi and (not cu or moi < cu):
            dk["dt_den"] = moi
            dk.pop("dt_tu", None)
            doi.append(f"diện tích tối đa: "
                       f"{f'{cu:.0f}' if cu else 'chưa đặt'} → {moi:.0f} m²")

    # ------------------------------------------------- mở rộng ra cả quận
    if _tim_cum(t, _TU_MO_RONG) is not None:
        if dk.get("phuong_moi") or dk.get("duong_pho"):
            hep = dk.get("duong_pho") or dk.get("phuong_moi")
            doi.append(f"bỏ giới hạn hẹp ({hep}), tìm cả quận")
            dk.pop("phuong_moi", None)
            dk.pop("_phuong_go_cu", None)
            dk.pop("duong_pho", None)
        elif dk.get("quan_huyen"):
            # Nói rõ thay vì im lặng: người dùng bấm "mở rộng" mà không thấy
            # gì đổi sẽ tưởng hệ thống không hiểu câu, rồi gõ lại mãi.
            doi.append(f"đang tìm cả {dk['quan_huyen']} rồi — không còn giới "
                       f"hạn phường hay phố nào để bỏ. Muốn rộng hơn nữa thì "
                       f"bấm ✕ trên thẻ quận.")
        else:
            doi.append("đang tìm toàn Hà Nội rồi, không giới hạn khu vực nào")

    # ------------------------------------------------------ BỎ điều kiện
    # Phải xét TRƯỚC khi gộp điều kiện mới, vì "bỏ điều kiện thang máy" cũng
    # bóc ra được "thang máy" và nếu gộp trước thì nó vừa bị bỏ vừa được thêm.
    vi_bo = _tim_cum(t, _TU_BO_DK)
    if vi_bo is not None:
        dd = doc_dac_diem(cau, loai)
        can_bo = list(dd.get("co", [])) + list(dd.get("khong", []))
        for cot in can_bo:
            for nhom in ("co", "khong"):
                if cot in dk.get(nhom, []):
                    dk[nhom] = [x for x in dk[nhom] if x != cot]
                    if not dk[nhom]:
                        dk.pop(nhom)
                    doi.append(f"bỏ yêu cầu “{NHAN_CO.get(cot, cot)}”")
        for khoa, nhan in (("gia_den", "trần giá"), ("gia_tu", "sàn giá"),
                           ("dt_tu", "diện tích tối thiểu"),
                           ("dt_den", "diện tích tối đa"),
                           ("so_phong_ngu", "số phòng ngủ"),
                           ("tang_tu", "số tầng"), ("mt_tu", "mặt tiền"),
                           ("ngo_tu", "độ rộng ngõ"),
                           ("phap_ly", "pháp lý"), ("du_an", "dự án"),
                           ("kc_den", "gần trung tâm")):
            tu = {"gia_den": ("gia", "tien", "ngan sach", "tran gia"),
                  "gia_tu": ("san gia",),
                  "dt_tu": ("dien tich",), "dt_den": ("dien tich",),
                  "so_phong_ngu": ("phong ngu", "pn", "ngu"),
                  "tang_tu": ("tang",), "mt_tu": ("mat tien",),
                  "ngo_tu": ("ngo", "duong rong"),
                  "phap_ly": ("phap ly", "so do", "so hong"),
                  "du_an": ("du an",),
                  "kc_den": ("trung tam",)}[khoa]
            if khoa in dk and _tim_cum(t[vi_bo:], tu) is not None:
                dk.pop(khoa, None)
                doi.append(f"bỏ điều kiện {nhan}")
        return dk, doi

    # ------------------------------ gộp những điều kiện MỚI nêu trong câu tiếp
    them = phan_tich(cau, loai)
    them.pop("sap", None)               # ý định sắp xếp xử lý riêng ở giao diện

    # Địa điểm nêu lại thì THAY, không phải thêm: "đổi sang Thanh Xuân" nghĩa
    # là thôi Cầu Giấy. Và đổi quận thì phải bỏ luôn phường/phố của quận cũ,
    # nếu không thì điều kiện tự mâu thuẫn và kết quả về không.
    if them.get("quan_huyen") and them["quan_huyen"] != dk.get("quan_huyen"):
        doi.append(f"khu vực: {dk.get('quan_huyen') or 'cả Hà Nội'} → "
                   f"{them['quan_huyen']}")
        for k in ("phuong_moi", "duong_pho", "_phuong_go_cu"):
            dk.pop(k, None)
    if them.get("phuong_moi") and them["phuong_moi"] != dk.get("phuong_moi"):
        doi.append(f"phường: {dk.get('phuong_moi') or 'không giới hạn'} → "
                   f"{them['phuong_moi']}")
        dk.pop("duong_pho", None)

    for k, v in them.items():
        if k in ("co", "khong", "loai_hinh_tru"):
            cu = list(dk.get(k, []))
            moi = [x for x in v if x not in cu]
            if moi:
                dk[k] = cu + moi
                nhan = ", ".join(NHAN_CO.get(x, x) for x in moi)
                doi.append(("thêm yêu cầu " if k == "co" else
                            "thêm loại trừ ") + nhan)
            continue
        if dk.get(k) != v:
            if k in NHAN_DK and k not in ("quan_huyen", "phuong_moi"):
                cu = dk.get(k)
                doi.append(f"{NHAN_DK[k]}: "
                           f"{_gon(cu) if cu is not None else 'chưa đặt'} → "
                           f"{_gon(v)}")
            dk[k] = v
    return dk, doi


def _gon(v) -> str:
    """Rút gọn một giá trị điều kiện để in trong câu mô tả thay đổi."""
    if isinstance(v, float) and v >= 1e8:
        return f"{v/1e9:.2f} tỷ".replace(".", ",")
    if isinstance(v, float):
        return f"{v:.0f}"
    if isinstance(v, (list, tuple)):
        return ", ".join(str(x) for x in v)
    return str(v)


VI_DU_CAU_TIEP = [
    "Rẻ hơn nữa",
    "Thêm chỗ đỗ xe ô tô",
    "Đổi sang Thanh Xuân",
    "Mở rộng ra cả quận",
    "Bỏ điều kiện thang máy",
    "Rộng hơn",
]


# =============================================================================
# THẺ ĐIỀU KIỆN BỎ ĐƯỢC TỪNG CÁI
# =============================================================================
# `mo_ta_dk` trả về câu chữ để ĐỌC. Hàm dưới trả về (khoá, chữ) để giao diện
# đặt được một nút ✕ cho từng điều kiện — cách sửa nhanh nhất khi máy hiểu
# sai một chỗ mà đúng mọi chỗ khác. Không có nó thì người dùng phải gõ lại cả
# câu chỉ vì một thẻ sai.

def the_dk(dk: dict, loai: str) -> list[tuple[str, str]]:
    """[(khoá xoá được, chữ hiện)] theo đúng thứ tự người ta đọc."""
    ra: list[tuple[str, str]] = []
    if dk.get("gia_tu") or dk.get("gia_den"):
        if dk.get("gia_tu") and dk.get("gia_den"):
            chu = f"giá {dk['gia_tu']/1e9:.2f}–{dk['gia_den']/1e9:.2f} tỷ"
        elif dk.get("gia_den"):
            chu = f"giá ≤ {dk['gia_den']/1e9:.2f} tỷ"
        else:
            chu = f"giá ≥ {dk['gia_tu']/1e9:.2f} tỷ"
        ra.append(("gia", chu.replace(".", ",")))

    if dk.get("dt_tu") or dk.get("dt_den"):
        if dk.get("dt_tu") and dk.get("dt_den"):
            chu = f"{dk['dt_tu']:.0f}–{dk['dt_den']:.0f} m²"
        elif dk.get("dt_tu"):
            chu = f"từ {dk['dt_tu']:.0f} m²"
        else:
            chu = f"tối đa {dk['dt_den']:.0f} m²"
        ra.append(("dt", chu))

    if dk.get("so_phong_ngu"):
        ra.append(("so_phong_ngu", f"{dk['so_phong_ngu']} phòng ngủ"))
    if dk.get("so_phong_vs"):
        ra.append(("so_phong_vs", f"{dk['so_phong_vs']} phòng vệ sinh"))

    if dk.get("tang_tu") or dk.get("tang_den"):
        nh = "tổng số tầng" if loai == "nhadat" else "tầng"
        if dk.get("tang_tu") and dk.get("tang_den"):
            chu = f"{nh} {dk['tang_tu']:.0f}–{dk['tang_den']:.0f}"
        elif dk.get("tang_tu"):
            chu = f"{nh} ≥ {dk['tang_tu']:.0f}"
        else:
            chu = f"{nh} ≤ {dk['tang_den']:.0f}"
        ra.append(("tang", chu))

    if dk.get("mt_tu"):
        ra.append(("mt_tu", f"mặt tiền ≥ {dk['mt_tu']:.1f} m".replace(".", ",")))
    if dk.get("ngo_tu"):
        ra.append(("ngo_tu",
                   f"ngõ rộng ≥ {dk['ngo_tu']:.1f} m".replace(".", ",")))
    if dk.get("kc_den"):
        ra.append(("kc_den", f"cách trung tâm ≤ {dk['kc_den']:.0f} km"))

    for khoa in ("du_an", "duong_pho", "phuong_moi", "quan_huyen", "loai_hinh"):
        if dk.get(khoa):
            ra.append((khoa, str(dk[khoa])))
    if dk.get("loai_hinh_tru"):
        ra.append(("loai_hinh_tru",
                   "không lấy " + ", ".join(dk["loai_hinh_tru"]).lower()))
    if dk.get("phap_ly"):
        ra.append(("phap_ly", " / ".join(dk["phap_ly"])))
    for c in dk.get("co", []):
        ra.append((f"co:{c}", NHAN_CO.get(c, c)))
    for c in dk.get("khong", []):
        ra.append((f"khong:{c}", "không cần " + NHAN_CO.get(c, c)))
    return ra


# Một thẻ có thể gom nhiều khoá trong dict (thẻ "giá" gom cả sàn lẫn trần),
# nên xoá thẻ phải xoá đủ bộ — xoá thiếu thì thẻ biến mất mà bộ lọc vẫn còn.
_GOM_KHOA = {
    "gia": ("gia_tu", "gia_den"),
    "dt": ("dt_tu", "dt_den"),
    "tang": ("tang_tu", "tang_den"),
    "phuong_moi": ("phuong_moi", "_phuong_go_cu"),
    "quan_huyen": ("quan_huyen", "phuong_moi", "_phuong_go_cu", "duong_pho"),
}


def xoa_dk(dk: dict, khoa: str) -> dict:
    """Bỏ một thẻ điều kiện, trả về dict MỚI (không sửa dict cũ).

    Không sửa tại chỗ là có chủ ý: lịch sử hội thoại lưu lại chính các dict
    này để hoàn tác được, nên sửa tại chỗ sẽ làm hỏng cả các lượt trước.
    """
    d = dict(dk)
    if khoa.startswith(("co:", "khong:")):
        nhom, cot = khoa.split(":", 1)
        d[nhom] = [x for x in d.get(nhom, []) if x != cot]
        if not d[nhom]:
            d.pop(nhom, None)
        return d
    for k in _GOM_KHOA.get(khoa, (khoa,)):
        d.pop(k, None)
    return d
