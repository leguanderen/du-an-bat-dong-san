"""
app.py — Giao diện Streamlit của hệ thống định giá bất động sản Hà Nội.

BẢN NÀY THAY GÌ SO VỚI BẢN CŨ
-----------------------------
Giao diện giữ nguyên: vẫn Bento grid nền charcoal, vẫn cùng bố cục, cùng các ô
nhập. Thay phần RUỘT:

  * Không nạp `model_*.pkl` nữa. Mọi thứ đi qua `pipeline/valuation_service.py`,
    module này tự huấn luyện lúc khởi động từ dữ liệu sạch. Nhờ vậy không còn
    cảnh file .pkl và file .xlsx lệch pha nhau.
  * "Khoảng tham khảo ±15%" bị bỏ. Đo lại thì khoảng đó chỉ đúng 58,5% với
    chung cư và 45,6% với nhà đất — tức sai hơn một nửa số lần. Thay bằng
    khoảng CQR có bảo chứng thống kê: rộng hơn nhiều, nhưng đúng 90% số lần.
  * Thêm phần "Vì sao lại là con số này?" (SHAP quy ra phần trăm) và bảng tin
    rao tương đồng — thứ làm người dùng tin kết quả hơn mọi lời giải thích.
  * Thêm tab Bản đồ: bản đồ nhiệt mặt bằng giá và tìm tin theo bán kính.

CÁC Ô NHẬP ĐỀU THỰC SỰ ĐƯỢC DÙNG
--------------------------------
Bản cũ có ô "Giấy tờ pháp lý", "Nội thất", "Hướng" nhưng model nhà đất không hề
nhận chúng làm đặc trưng — người dùng chọn xong mà kết quả không đổi. Đã sửa:
thêm vào cấu hình model và đo lại, MAPE giảm từ 27,66% xuống 26,37% (nhà đất)
và 16,14% xuống 15,46% (chung cư). Ô nào hiện trên màn hình thì ô đó có tác dụng.

CHẠY
----
    streamlit run app.py
"""

from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR / "pipeline"))

import ban_do as bd                     # noqa: E402
import tinh_nang as tn                  # noqa: E402
import valuation_service as vs          # noqa: E402


def scroll_to_top() -> None:
    """Cuộn cửa sổ lên đầu trang (gọi sau khi bấm Định giá)."""
    components.html(
        """
        <script>
            const doc = window.parent.document;
            const el = doc.querySelector('section.main') ||
                       doc.querySelector('[data-testid="stMain"]') ||
                       doc.querySelector('.main');
            if (el) { el.scrollTo({top: 0, behavior: 'smooth'}); }
            window.parent.scrollTo({top: 0, behavior: 'smooth'});
        </script>
        """,
        height=0,
    )


PROJECT_OTHER = "__other__"   # "Dự án khác" -> ước lượng theo phường

INTERIOR_CHOICES = ["Nội thất đầy đủ", "Nội thất cao cấp", "Hoàn thiện cơ bản",
                    "Bàn giao thô", "Không rõ"]
DIRECTION_CHOICES = ["Đông", "Tây", "Nam", "Bắc", "Đông Bắc", "Tây Bắc",
                     "Đông Nam", "Tây Nam", "Không rõ"]
LEGAL_CHOICES = ["Sổ hồng riêng", "Hợp đồng mua bán", "Đang chờ sổ",
                 "Hợp đồng đặt cọc", "Không rõ"]
STATUS_CHOICES = ["Đã bàn giao", "Chưa bàn giao", "Không rõ"]

NHADAT_LOAI_HINH = ["Nhà ngõ, hẻm", "Nhà mặt phố, mặt tiền",
                    "Nhà phố liền kề", "Nhà biệt thự"]
NHADAT_PHAP_LY = ["Đã có sổ", "Sổ chung / công chứng vi bằng", "Đang chờ sổ",
                  "Giấy tờ viết tay", "Không có sổ", "Không rõ"]
NHADAT_NOI_THAT = ["Nội thất đầy đủ", "Nội thất cao cấp", "Hoàn thiện cơ bản",
                   "Bàn giao thô", "Không rõ"]
NHADAT_HUONG = DIRECTION_CHOICES
NHADAT_DAC_DIEM = ["Hẻm xe hơi", "Nhà nở hậu", "Nhà tóp hậu", "Nhà nát",
                   "Nhà chưa hoàn công", "Nhà dính quy hoạch / lộ giới",
                   "Đất chưa chuyển thổ", "Hiện trạng khác"]


def _hoac_none(v: object) -> object:
    """"Không rõ" và chuỗi rỗng -> None, để model hiểu là thiếu dữ liệu chứ
    không phải một hạng mục riêng tên là "Không rõ"."""
    return None if v in (None, "", "Không rõ") else v


# =============================================================================
# NẠP DỮ LIỆU — mọi thứ đi qua valuation_service
# =============================================================================
# cache_resource cho model (đối tượng nặng, không serialize được), cache_data
# cho các bảng tra (bản sao rẻ, an toàn khi nhiều phiên dùng chung).

@st.cache_resource(show_spinner="Đang huấn luyện mô hình từ dữ liệu sạch…")
def nap_model(loai: str):
    return vs.nap(loai)


@st.cache_data(show_spinner="Đang dựng cây Quận → Phường…")
def nap_cay(loai: str) -> dict:
    return vs.cay_dia_diem(loai)


@st.cache_data(show_spinner="Đang dựng lưới giá…")
def nap_luoi(loai: str) -> pd.DataFrame:
    return bd.luoi_gia([loai])



@st.cache_data(show_spinner="Đang nạp bản đồ…")
def nap_ghim(loai: str) -> pd.DataFrame:
    return bd.tin_ghim(loai)


@st.cache_data(show_spinner="Đang tô bản đồ phường…")
def nap_vung_phuong(loai: str) -> dict:
    return bd.vung_phuong(loai)


@st.cache_data(show_spinner="Đang tổng hợp giá theo phường…")
def nap_phuong(loai: str) -> pd.DataFrame:
    return bd.thong_ke_phuong(loai)



def format_vnd(value: float) -> str:
    if value >= 1_000_000_000:
        return f"{value/1_000_000_000:,.2f} tỷ VND ({value:,.0f} VND)"
    if value >= 1_000_000:
        return f"{value/1_000_000:,.0f} triệu VND ({value:,.0f} VND)"
    return f"{value:,.0f} VND"


def format_ty(value: float) -> str:
    return f"{value/1e9:,.2f} tỷ"


# =============================================================================
# UI LAYER — Soft UI Evolution + Bento Grid (nền Charcoal) — GIỮ NGUYÊN
# =============================================================================# =============================================================================
# HỆ MÀU SÁNG — "nền giấy, mực đậm, một màu hành động"
# =============================================================================
# VÌ SAO ĐỔI KHỎI NỀN TỐI
#
# Nền charcoal + vàng kim là ngôn ngữ của bảng điều khiển phân tích: nó nói
# "công cụ cho người chuyên". Nhưng người đi mua nhà đang cần cảm giác ĐƯỢC TƯ
# VẤN, và họ mở máy giữa ban ngày, thường trên điện thoại ngoài đường — nơi
# nền tối phản chiếu mặt họ. Cả hai sàn họ quen dùng đều nền sáng, nên nền tối
# không đọc ra là "chuyên nghiệp hơn", nó đọc ra là "khác lạ".
#
# BA NGUYÊN TẮC, và mỗi nguyên tắc đều có hệ quả trong bảng dưới:
#
# 1. NỀN GIẤY ẤM, KHÔNG TRẮNG TINH. Nền #F7F5F1, thẻ TRẮNG nổi lên trên. Phân
#    tầng bằng độ sáng thay vì bằng đường kẻ, nên trang bớt hẳn nét.
#
# 2. MỘT MÀU HÀNH ĐỘNG DUY NHẤT. `accent` (xanh mực) cho mọi nút, liên kết, ô
#    đang chọn. `tot` và `xau` CHỈ nói "so với mặt bằng" — dùng chúng làm màu
#    nút là lấy mất nghĩa của chúng, đúng lỗi bản vàng kim cũ mắc phải (vàng
#    vừa là nút, vừa là nhãn, vừa là viền thẻ, nên chẳng còn nghĩa gì).
#
# 3. MÀU ĐO ĐƯỢC MỚI DÙNG. Mọi màu chữ dưới đây đạt tối thiểu 4,5:1 trên cả ba
#    nền. Cặp tot/xau đạt ΔE 8,9 với người mù màu đỏ-lục — bảng màu tối cũ chỉ
#    đạt 6,0, tức DƯỚI ngưỡng an toàn 8.
#
# MỘT CÁI BẪY PHẢI GIỮ ĐÚNG: `tot` (#00916E) chỉ đạt 3,98:1 nên nó CHỈ được
# dùng cho THANH biểu đồ (mốc cho hình khối là 3:1). Chữ xanh phải dùng
# `tot_chu` (#0A7157, 5,98:1). Dùng lẫn là chữ mờ không đọc được.

COLOR = {
    # nền và mực
    "bg": "#F7F5F1",              # nền trang — giấy ấm
    "bg_grad": "#FFFFFF",         # giữ khoá cũ để chỗ nào còn dùng không vỡ
    "surface": "#FFFFFF",         # thẻ
    "surface_2": "#EFEDE7",       # nền phụ: hàng xen kẽ, thẻ điều kiện
    "border": "rgba(23,24,28,0.11)",
    "border_manh": "rgba(23,24,28,0.22)",   # viền ô nhập, cần rõ hơn
    "text": "#17181C",            # 17,7:1
    "text_muted": "#5C6169",      # 5,3:1 trở lên — KHÔNG phải xám nhạt

    # một màu hành động
    "accent": "#2B4C8C",          # nút, liên kết, ô đang chọn — chữ trắng 8,4:1
    "accent_dam": "#1E3A6E",      # hover / nhấn
    "accent_soft": "#EBEFF8",     # nền chip đang chọn

    # trạng thái — chỉ nói "so với mặt bằng"
    "success": "#00916E",         # THANH biểu đồ (3,98:1 — không dùng cho chữ)
    "success_chu": "#0A7157",     # chữ và nhãn (5,98:1)
    "success_nen": "#E3F3ED",
    "danger": "#BC4326",          # thanh và chữ đều được (5,3:1)
    "danger_nen": "#FBEAE4",
    "warn": "#8F6108",            # thiếu dữ liệu, chưa xác minh (5,4:1)
    "warn_nen": "#FAF0DC",
}

def _rgb(ma: str) -> list:
    """'#2B4C8C' -> [43, 76, 140] — pydeck nhận màu dạng danh sách RGB."""
    ma = ma.lstrip("#")
    return [int(ma[i:i + 2], 16) for i in (0, 2, 4)]


# =============================================================================
# THANG MÀU BẢN ĐỒ THEO PHƯỜNG
# =============================================================================
# VÌ SAO KHÔNG CÒN LÀ MỘT DANH SÁCH MÀU CHỌN TAY
#
# Bản trước là năm mã màu chọn tay rồi để `_mau_bac` nội suy tuyến tính trong
# sRGB ra đủ số bậc cần. Đo lại thì nó sai hai chỗ, và cả hai chỉ hiện ra sau
# khi đổi sang nền sáng:
#
#   1. BA BẬC NHẠT NHẤT DÍNH VÀO NHAU. Bản đồ phường vẽ BẢY bậc. Nội suy
#      trong sRGB dồn độ sáng về đầu nhạt, nên ba bậc đầu chỉ cách nhau ΔL
#      0,041 và 0,044 — dưới hẳn mốc 0,06. Tức là khoảng một phần ba số
#      phường Hà Nội tô ra thành một vệt không phân biệt được.
#
#   2. "RẺ NHẤT" TRÔNG HỆT "CHƯA ĐỦ DỮ LIỆU". Bậc nhạt nhất sau khi tô mờ lên
#      nền bản đồ chỉ cách ô "chưa đủ dữ liệu" 1,11:1. Đây là lỗi nặng nhất
#      trong cả hệ màu: nó âm thầm biến chỗ THIẾU SỐ LIỆU thành một lời khẳng
#      định về giá. Cả dự án tôi giữ nguyên tắc không bịa độ chính xác, mà
#      chính bảng màu lại bịa hộ.
#
# Nên thang giờ SINH RA TỪ SỐ ĐO, không chọn tay: một sắc (hue 262° — đúng sắc
# của `accent`), chroma cố định, còn độ sáng chia ĐỀU trong OKLab. Chia đều
# trong OKLab nghĩa là bậc nào cũng cách bậc kế tiếp đúng bằng nhau THEO MẮT,
# với bất kỳ số bậc — 7 bậc cho bản đồ phường, 11 bậc cho bề mặt đồng mức.
#
# BỐN MỐC ĐÃ ĐO (nền bản đồ #F8F8F8, lớp tô đục `DUC_VUNG`):
#   · bậc nhạt nhất nổi trên nền bản đồ     2,01:1  (trước: 1,08:1)
#   · ΔL nhỏ nhất giữa hai bậc kề, 7 bậc    0,065   (trước: 0,041)
#   · nhãn chữ trên bậc yếu nhất            4,57:1
#   · "rẻ nhất" tách khỏi "chưa đủ dữ liệu" 2,06:1  (trước: 1,11:1)
#
# CÒN MỘT ĐIỂM PHẢI NÓI THẲNG: với 11 bậc thì ΔL nhỏ nhất chỉ còn 0,038, dưới
# mốc 0,06. Bề mặt đồng mức là thang LIÊN TỤC — người xem đọc xu hướng đậm
# nhạt chứ không tra từng bậc về một con số — nên ở đó tôi nhận mức này. Bản
# đồ phường thì người xem TRA từng phường, nên nó phải đủ 0,06, và nó đủ.
THANG_HUE = 262.0     # cùng sắc với COLOR["accent"] (#2B4C8C ở 262,2°)
THANG_C = 0.100       # chroma — cùng mức với accent, không rực hơn
THANG_L_DAU = 0.735   # OKLab L của bậc nhạt nhất
THANG_L_CUOI = 0.280  # OKLab L của bậc đậm nhất


def _oklch_rgb(L: float, C: float, h_do: float) -> list:
    """OKLCH -> sRGB 0..255. Tự viết vì không muốn thêm gói chỉ để đổi màu."""
    a = C * np.cos(np.radians(h_do))
    b = C * np.sin(np.radians(h_do))
    l_ = (L + 0.3963377774 * a + 0.2158037573 * b) ** 3
    m_ = (L - 0.1055613458 * a - 0.0638541728 * b) ** 3
    s_ = (L - 0.0894841775 * a - 1.2914855480 * b) ** 3
    lin = (4.0767416621 * l_ - 3.3077115913 * m_ + 0.2309699292 * s_,
           -1.2684380046 * l_ + 2.6097574011 * m_ - 0.3413193965 * s_,
           -0.0041960863 * l_ - 0.7034186147 * m_ + 1.7076147010 * s_)
    ra = []
    for c in lin:
        c = min(max(float(c), 0.0), 1.0)
        c = c * 12.92 if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055
        ra.append(int(round(min(max(c, 0.0), 1.0) * 255)))
    return ra


@lru_cache(maxsize=16)
def thang_vung(so_bac: int) -> tuple:
    """`so_bac` màu tô, nhạt -> đậm, độ sáng chia đều trong OKLab."""
    if so_bac <= 1:
        return (tuple(_oklch_rgb(THANG_L_CUOI, THANG_C, THANG_HUE)),)
    return tuple(
        tuple(_oklch_rgb(
            THANG_L_DAU + (THANG_L_CUOI - THANG_L_DAU) * i / (so_bac - 1),
            THANG_C, THANG_HUE))
        for i in range(so_bac))


def _tuong_phan(a, b) -> float:
    """Tỷ số tương phản WCAG giữa hai màu RGB."""
    def s(m):
        v = [x / 255 for x in m]
        v = [x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4
             for x in v]
        return 0.2126 * v[0] + 0.7152 * v[1] + 0.0722 * v[2]
    la, lb = s(a), s(b)
    return (max(la, lb) + 0.05) / (min(la, lb) + 0.05)


def chu_tren(mau_to) -> str:
    """Nhãn trên một bậc: chọn mực hay trắng theo SỐ ĐO, không theo cảm giác.

    Tính trên màu MẮT THẤY (đã tô mờ `DUC_VUNG` lên nền bản đồ), không tính
    trên mã màu tô. Tính nhầm trên mã màu tô thì mấy bậc giữa chọn sai bên và
    nhãn mờ đi đúng ở chỗ bản đồ đông phường nhất.
    """
    nen = [round(DUC_VUNG * c + (1 - DUC_VUNG) * n)
           for c, n in zip(mau_to, NEN_BAN_DO)]
    return (COLOR["text"] if _tuong_phan(nen, _rgb(COLOR["text"]))
            >= _tuong_phan(nen, [255, 255, 255]) else "#FFFFFF")


# Năm bậc dùng cho chú giải, thẻ màu trong tài liệu, và lớp chấm tin rao.
THANG_PHUONG = ["#%02X%02X%02X" % m for m in thang_vung(5)]


def inject_css() -> None:
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Newsreader:opsz,wght@6..72,400;6..72,500;6..72,600&display=swap');
        :root {{
            --surface: {COLOR['surface']}; --surface-2: {COLOR['surface_2']};
            --border: {COLOR['border']}; --border-manh: {COLOR['border_manh']};
            --text: {COLOR['text']}; --muted: {COLOR['text_muted']};
            --accent: {COLOR['accent']}; --accent-dam: {COLOR['accent_dam']};
            --accent-soft: {COLOR['accent_soft']};
            --radius: 16px; --t: 200ms cubic-bezier(.4,0,.2,1);
        }}
        /* Nền PHẲNG, không còn radial-gradient. Gradient trên nền sáng chỉ tạo
           một vệt xám bẩn ở góc, và nó là dấu hiệu rất dễ nhận của giao diện
           dựng vội. */
        .stApp {{
            background: {COLOR['bg']};
            color: var(--text); font-family: 'Plus Jakarta Sans', -apple-system, sans-serif;
        }}
        .block-container {{ padding-top: 2rem; padding-bottom: 3rem; max-width: 1240px; }}

        /* TIÊU ĐỀ VÀ CON SỐ DÙNG PHÔNG SERIF. Đây là chỗ đổi lớn thứ hai sau
           màu: trước đây giá và nhãn cùng một phông, cùng cỡ gần nhau, nên mắt
           không biết nhìn đâu trước. */
        h1,h2,h3,h4, [data-testid="stMarkdownContainer"] h1,
        [data-testid="stMarkdownContainer"] h2, [data-testid="stMarkdownContainer"] h3 {{
            font-family: 'Newsreader', Georgia, serif !important; color: var(--text) !important;
            font-weight: 600 !important; letter-spacing: -0.012em;
        }}
        p, span, label, .stMarkdown {{ color: var(--text); }}

        /* Thẻ trắng nổi trên nền giấy. Đổ bóng NHẸ hơn hẳn bản tối: trên nền
           sáng, bóng đậm đọc ra là bẩn chứ không phải nổi. */
        [data-testid="stVerticalBlockBorderWrapper"] {{
            background: var(--surface) !important;
            border: 1px solid var(--border) !important; border-radius: var(--radius) !important;
            padding: 1.3rem 1.45rem !important;
            box-shadow: 0 1px 2px rgba(23,24,28,0.04), 0 4px 14px rgba(23,24,28,0.045) !important;
            transition: box-shadow var(--t);
        }}
        [data-testid="stVerticalBlockBorderWrapper"]:hover {{
            box-shadow: 0 2px 4px rgba(23,24,28,0.05), 0 8px 22px rgba(23,24,28,0.07) !important;
        }}

        /* Ô nhập: nền TRẮNG viền rõ. Nền xám lõm kiểu bản tối làm ô nhập trên
           nền sáng trông như ô bị vô hiệu hoá. */
        [data-baseweb="select"] > div, [data-testid="stNumberInput"] input,
        [data-testid="stTextInput"] input, [data-testid="stTextArea"] textarea {{
            background: {COLOR['surface']} !important;
            border: 1px solid var(--border-manh) !important;
            border-radius: 10px !important; color: var(--text) !important;
            box-shadow: none !important;
            transition: border-color var(--t), box-shadow var(--t);
        }}
        [data-baseweb="select"] > div:hover, [data-testid="stNumberInput"] input:hover,
        [data-testid="stTextInput"] input:hover {{ border-color: var(--accent) !important; }}
        [data-baseweb="select"] > div:focus-within, [data-testid="stTextInput"] input:focus {{
            border-color: var(--accent) !important;
            box-shadow: 0 0 0 3px {COLOR['accent_soft']} !important;
        }}
        [data-baseweb="select"] svg {{ color: var(--muted) !important; }}
        [data-testid="stNumberInput"] button {{ display: none !important; }}
        [data-testid="stWidgetLabel"] p {{
            color: var(--text) !important; font-weight: 600 !important; font-size: .87rem;
        }}

        /* Nút: một màu đặc, không gradient. Chữ trắng trên xanh mực đạt 8,4:1. */
        .stButton > button {{
            background: var(--accent) !important;
            color: #FFFFFF !important; border: none !important; border-radius: 10px !important;
            font-family: 'Plus Jakarta Sans', sans-serif !important; font-weight: 700 !important;
            padding: 0.68rem 1rem !important; cursor: pointer !important;
            box-shadow: none !important;
            transition: background var(--t);
        }}
        .stButton > button:hover {{ background: var(--accent-dam) !important; }}
        .stButton > button[kind="secondary"] {{
            background: {COLOR['surface']} !important; color: var(--accent) !important;
            border: 1px solid var(--accent) !important;
        }}
        .stButton > button[kind="secondary"]:hover {{ background: {COLOR['accent_soft']} !important; }}

        /* Ô số liệu: con số dùng mực ĐẬM, không dùng màu hành động. Màu hành
           động để dành cho thứ bấm được — tô nó lên một con số tĩnh là mời
           người dùng bấm vào chỗ không bấm được. */
        [data-testid="stMetric"] {{
            background: {COLOR['surface_2']}; border: 1px solid var(--border);
            border-radius: 12px; padding: 0.85rem 1rem; box-shadow: none;
        }}
        [data-testid="stMetricValue"] {{
            color: var(--text) !important; font-family: 'Newsreader', Georgia, serif !important;
            font-weight: 600 !important; font-variant-numeric: tabular-nums;
        }}
        [data-testid="stMetricLabel"] p {{ color: var(--muted) !important; font-weight: 500 !important; }}

        [data-testid="stExpander"] {{
            background: {COLOR['surface']}; border: 1px solid var(--border) !important;
            border-radius: 12px !important;
        }}
        [data-testid="stExpander"] summary {{ color: var(--text) !important; font-weight: 600; }}
        [data-testid="stExpander"] summary:hover {{ color: var(--accent) !important; }}

        [data-baseweb="checkbox"] [data-checked="true"] {{ background: var(--accent) !important; }}
        [data-baseweb="radio"] [data-checked="true"] {{ border-color: var(--accent) !important; }}
        [data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {{
            border-color: var(--accent) !important;
        }}
        [data-testid="stAlert"] {{
            border-radius: 12px !important; border: 1px solid var(--border) !important;
        }}
        [data-testid="stDataFrame"] {{ border-radius: 12px !important; overflow: hidden; }}

        /* DẢI CHỌN TRANG — dạng viên thuốc.
           Bốn tab là thứ người dùng bấm nhiều nhất, nên nó xứng đáng trông
           như thanh tab chứ không phải bốn vòng tròn radio. */
        .st-key-nav_chinh [role="radiogroup"] {{
            display: inline-flex; flex-wrap: wrap; gap: .3rem;
            background: var(--surface); padding: .3rem;
            border: 1px solid var(--border); border-radius: 12px;
            box-shadow: 0 1px 2px rgba(23,24,28,0.04);
        }}
        .st-key-nav_chinh [role="radiogroup"] label {{
            margin: 0 !important; padding: .5rem .95rem; border-radius: 9px;
            cursor: pointer; transition: background var(--t);
        }}
        .st-key-nav_chinh [role="radiogroup"] label p {{
            margin: 0 !important; font-weight: 600 !important; font-size: .92rem;
        }}
        .st-key-nav_chinh [role="radiogroup"] label:hover {{ background: var(--surface-2); }}
        /* Ẩn vòng tròn CHỈ KHI trình duyệt hiểu `:has()`, vì dấu "đang chọn"
           của dạng viên thuốc phụ thuộc hoàn toàn vào `:has()`. Trình duyệt
           cũ không hiểu thì giữ nguyên vòng tròn — thà kém đẹp còn hơn để
           người dùng không biết mình đang đứng ở tab nào. */
        @supports selector(:has(*)) {{
            .st-key-nav_chinh [role="radiogroup"] label > div:first-child {{
                display: none !important;
            }}
            .st-key-nav_chinh [role="radiogroup"] label:has(input:checked) {{
                background: var(--accent) !important;
            }}
            .st-key-nav_chinh [role="radiogroup"] label:has(input:checked) p {{
                color: #FFFFFF !important;
            }}
        }}

        /* KHUNG CHAT — ô nhập trắng viền rõ, bong bóng để trong suốt vì
           `_bong_bong_may` tự dựng nền của nó. */
        [data-testid="stChatMessage"] {{ background: transparent !important; }}
        [data-testid="stChatInput"] {{
            background: {COLOR['surface']} !important;
            border: 1px solid var(--border-manh) !important;
            border-radius: 12px !important; box-shadow: none !important;
        }}
        [data-testid="stChatInput"]:focus-within {{
            border-color: var(--accent) !important;
            box-shadow: 0 0 0 3px {COLOR['accent_soft']} !important;
        }}
        [data-testid="stChatInput"] textarea {{ color: var(--text) !important; }}
        [data-testid="stChatInput"] textarea::placeholder {{ color: var(--muted) !important; }}
        [data-testid="stChatInputSubmitButton"] {{ color: var(--accent) !important; }}

        /* Thanh trên cùng của Streamlit: để trong suốt cho liền nền giấy. */
        [data-testid="stHeader"] {{ background: transparent !important; }}

        /* Danh sách thả xuống: nền trắng, dòng đang trỏ tô xanh nhạt. Nếu
           thiếu khối này thì bản dựng mặc định của Streamlit vẫn tô theo nền
           máy người xem, nên nửa sáng nửa tối. */
        [data-baseweb="popover"] [role="listbox"],
        [data-baseweb="popover"] ul {{
            background: {COLOR['surface']} !important;
            border: 1px solid var(--border) !important; border-radius: 10px !important;
            box-shadow: 0 6px 24px rgba(23,24,28,0.10) !important;
        }}
        [data-baseweb="popover"] li {{ color: var(--text) !important; }}
        [data-baseweb="popover"] li[aria-selected="true"],
        [data-baseweb="popover"] li:hover {{
            background: {COLOR['accent_soft']} !important; color: var(--accent) !important;
        }}

        hr {{ border-color: var(--border) !important; opacity: 1; }}
        a {{ color: var(--accent) !important; }}
        a:hover {{ color: var(--accent-dam) !important; }}
        @media (prefers-reduced-motion: reduce) {{ * {{ transition: none !important; animation: none !important; }} }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def hero_header(subtitle_html: str | None = None) -> None:
    default_sub = ""
    sub = subtitle_html or default_sub
    st.markdown(
        f"""
        <div style="background: {COLOR['surface']};
            border: 1px solid {COLOR['border']}; border-radius: 16px; padding: 1.5rem 1.7rem;
            margin-bottom: 1.3rem;
            box-shadow: 0 1px 2px rgba(23,24,28,0.04), 0 4px 14px rgba(23,24,28,0.045);
            display: flex; align-items: center; gap: 1.1rem;">
            <div style="width:56px;height:56px;flex:0 0 56px;display:flex;align-items:center;
                justify-content:center;background:{COLOR['accent_soft']};border-radius:16px;">
                <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="{COLOR['accent']}"
                     stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
                    <rect x="4" y="3" width="16" height="18" rx="1.5"/><path d="M9 21V8h6v13"/>
                    <line x1="8" y1="7" x2="8.01" y2="7"/><line x1="12" y1="7" x2="12.01" y2="7"/>
                    <line x1="16" y1="7" x2="16.01" y2="7"/><line x1="8" y1="11" x2="8.01" y2="11"/>
                    <line x1="16" y1="11" x2="16.01" y2="11"/>
                </svg>
            </div>
            <div>
                <div style="font-family:Newsreader,Georgia,serif;font-size:1.62rem;font-weight:600;
                    color:{COLOR['text']};line-height:1.12;letter-spacing:-0.012em;">Hệ thống định giá Bất động sản Hà Nội</div>
                <div style="color:{COLOR['text_muted']};font-size:0.82rem;margin-top:0.4rem;">
                     {sub}Theo <b style="color:{COLOR['text']}">giá đang chào bán</b>,
                     không phải giá đã chốt.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_header(icon_path: str, title: str, subtitle: str = "") -> None:
    sub = (
        f'<div style="color:{COLOR["text_muted"]};font-size:0.8rem;margin-top:1px;">{subtitle}</div>'
        if subtitle else ""
    )
    st.markdown(
        f"""
        <div style="display:flex;align-items:center;gap:0.6rem;margin-bottom:0.75rem;">
            <span style="width:34px;height:34px;display:flex;align-items:center;justify-content:center;
                background:{COLOR['accent_soft']};border-radius:11px;">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="{COLOR['accent']}"
                     stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round">{icon_path}</svg>
            </span>
            <div>
                <div style="font-family:'Plus Jakarta Sans',sans-serif;font-weight:700;font-size:1.02rem;
                    color:{COLOR['text']};">{title}</div>{sub}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


ICON_AREA = '<path d="M21 3 3 21"/><path d="M21 9V3h-6"/><path d="M3 15v6h6"/>'
ICON_PIN = '<path d="M20 10c0 6-8 12-8 12s-8-6-8-12a8 8 0 0 1 16 0Z"/><circle cx="12" cy="10" r="3"/>'
ICON_DOC = '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><path d="M14 2v6h6"/><path d="M9 13h6M9 17h4"/>'
ICON_CHART = '<path d="M3 3v18h18"/><path d="M7 15l4-4 3 3 5-6"/>'
ICON_MAP = ('<path d="m3 6 6-3 6 3 6-3v15l-6 3-6-3-6 3z"/>'
            '<path d="M9 3v15"/><path d="M15 6v15"/>')
ICON_INFO = '<circle cx="12" cy="12" r="9"/><path d="M12 16v-4"/><path d="M12 8h.01"/>'


# =============================================================================
# KHỐI HIỂN THỊ KẾT QUẢ — dùng chung cho cả hai nhánh
# =============================================================================

# Mức giữa dùng `warn`, KHÔNG dùng `accent`: màu hành động để dành cho thứ
# bấm được. Và chữ xanh dùng `success_chu` (5,98:1), không dùng `success`
# (3,98:1 — chỉ đủ cho thanh hình khối).
_MAU_TIN_CAY = {"cao": COLOR["success_chu"], "trung binh": COLOR["warn"],
                "thap": COLOR["danger"]}


def _huy_hieu_tin_cay(tc: dict) -> None:
    mau = _MAU_TIN_CAY.get(tc["muc"], COLOR["text_muted"])
    st.markdown(
        f"""<div style="display:inline-flex;align-items:center;gap:.45rem;
             background:{mau}22;border:1px solid {mau}55;border-radius:999px;
             padding:.3rem .8rem;margin:.4rem 0 .2rem;">
            <span style="width:8px;height:8px;border-radius:50%;background:{mau};"></span>
            <span style="color:{mau};font-weight:600;font-size:.86rem;">{tc['nhan']}</span>
            <span style="color:{COLOR['text_muted']};font-size:.82rem;">— {tc['can_cu']}</span>
        </div>""",
        unsafe_allow_html=True,
    )


def _thanh_yeu_to(yeu_to: list[dict], gia_co_so: float | None = None,
                  gia_cuoi: float | None = None) -> None:
    """Thác nước tiền: từ căn trung bình Hà Nội, cộng trừ dần tới giá của bạn.

    VÌ SAO ĐỔI TỪ "THANH PHẦN TRĂM" SANG "THÁC NƯỚC TIỀN"
    -----------------------------------------------------
    Bản trước là một danh sách phần trăm kèm thanh màu, và người dùng nói đúng
    hai điều về nó:

      "nhiều chữ và khó hiểu" — có một đoạn văn dài phía trên, rồi mỗi dòng
      lại một nhãn dài kiểu "Mô tả nhấn mạnh view".

      "tại sao lại cộng trừ như vậy" — và đây là chỗ nặng nhất, vì các phần
      trăm ấy KHÔNG cộng lại thành tổng. Đóng góp SHAP cộng được trong không
      gian logarit, không phải trong phần trăm. Một căn chung cư cho ra ba
      dòng +24,0%, −8,4%, +7,4% — cộng tay được +23,0% trong khi mức lệch thật
      là +22,3%. Người dùng cộng thử thấy không khớp thì mất tin cả bảng.

    Trong ĐỒNG thì cộng khớp tuyệt đối (xem ghi chú trong `explain.explain_one`).
    Nên bảng này là một thác nước: dòng đầu là căn trung bình, các dòng giữa
    cộng trừ, dòng cuối đúng bằng con số hiển thị ở trên. Cộng tay kiểm được —
    đó là toàn bộ mục đích.

    MÀU CHỈ LÀ LỚP THỨ HAI
    ----------------------
    Cặp xanh/đỏ của hệ sáng đạt ΔE 8,9 với người mù màu đỏ-lục, tức là đã
    trên ngưỡng an toàn 8 (bảng tối cũ chỉ đạt 6,0). Nhưng vẫn không dựa vào
    riêng màu: hướng của thanh tự nói lên dấu — chạy SANG PHẢI từ vạch giữa
    là cộng, SANG TRÁI là trừ — cộng thêm dấu +/− ngay trong số. Bỏ hết màu
    đi vẫn đọc được bảng.

    Riêng thanh dùng `success` (#00916E, 3,98:1). Mốc tương phản cho HÌNH KHỐI
    là 3:1 nên nó hợp lệ ở đây, còn mọi CHỮ xanh trong app dùng `success_chu`.
    """
    if not yeu_to:
        st.caption("Không yếu tố nào lệch đủ lớn so với mặt bằng chung.")
        return

    def _ty(v):
        return f"{v/1e9:+.2f}".replace(".", ",")

    lon = max(abs(y.get("dong_gop_vnd") or 0) for y in yeu_to) or 1
    hang = []
    if gia_co_so:
        hang.append(
            f'<div style="display:flex;align-items:center;gap:.6rem;'
            f'padding:.3rem 0;border-bottom:1px solid {COLOR["border"]};">'
            f'<div style="flex:1;font-size:.84rem;color:{COLOR["text_muted"]};">'
            f'Căn trung bình ở Hà Nội</div>'
            f'<div style="width:5.2rem;text-align:right;font-size:.86rem;'
            f'color:{COLOR["text_muted"]};font-variant-numeric:tabular-nums;">'
            f'{gia_co_so/1e9:.2f} tỷ</div></div>')

    for y in yeu_to:
        v = y.get("dong_gop_vnd") or 0
        mau = COLOR["success"] if v > 0 else COLOR["danger"]
        rong = abs(v) / lon * 48          # nửa bề rộng, tính theo %
        lech = f"left:50%;width:{rong:.1f}%" if v > 0 \
            else f"right:50%;width:{rong:.1f}%"
        pct = (f' · {y["dong_gop_pct"]:+.0f}%'
               if y.get("dong_gop_pct") is not None else "")
        hang.append(
            f'<div style="display:flex;align-items:center;gap:.6rem;'
            f'padding:.32rem 0;">'
            f'<div style="flex:1;font-size:.84rem;color:{COLOR["text"]};'
            f'line-height:1.3;">{y["ten"]}'
            f'<span style="color:{COLOR["text_muted"]};font-size:.76rem;">'
            f'{pct}</span></div>'
            f'<div style="position:relative;width:38%;height:14px;">'
            f'<div style="position:absolute;left:50%;top:0;width:1px;height:14px;'
            f'background:{COLOR["border_manh"]};"></div>'
            f'<div style="position:absolute;top:3px;height:8px;{lech};'
            f'background:{mau};border-radius:2px;"></div></div>'
            f'<div style="width:5.2rem;text-align:right;font-size:.86rem;'
            f'font-weight:600;color:{COLOR["text"]};'
            f'font-variant-numeric:tabular-nums;">{_ty(v)} tỷ</div></div>')

    if gia_cuoi:
        hang.append(
            f'<div style="display:flex;align-items:center;gap:.6rem;'
            f'padding:.45rem 0 .2rem;border-top:1px solid {COLOR["text_muted"]};'
            f'margin-top:.2rem;">'
            f'<div style="flex:1;font-size:.88rem;font-weight:600;'
            f'color:{COLOR["text"]};">Giá ước tính cho căn của bạn</div>'
            # Dòng tổng dùng MỰC, không dùng màu hành động: xanh mực để dành
            # cho thứ bấm được. Nhấn bằng phông serif và cỡ lớn hơn là đủ.
            f'<div style="width:5.2rem;text-align:right;font-size:1.02rem;'
            f'font-family:Newsreader,Georgia,serif;font-weight:600;'
            f'color:{COLOR["text"]};'
            f'font-variant-numeric:tabular-nums;">{gia_cuoi/1e9:.2f} tỷ</div>'
            f'</div>')

    st.markdown("".join(hang), unsafe_allow_html=True)
    st.caption("Cột bên phải cộng lại đúng bằng giá ước tính. "
               "Thanh sang phải là làm tăng giá, sang trái là làm giảm.")


def _da_mo_rong_quan(loai: str, dac_diem: dict, k: int = 5) -> bool:
    """Phường người dùng chọn có đủ `k` căn không — để nói đúng phạm vi."""
    p = dac_diem.get("phuong_moi")
    if not p:
        return True
    df = vs.nap(loai)["df"]
    return int((df["phuong_moi"] == p).sum()) < k


def _bang_comparables(cmp_df: pd.DataFrame, cot_dt: str,
                      dac_diem: dict | None = None,
                      mo_rong_quan: bool = False) -> None:
    """Bảng những căn tương tự, KÈM TIÊU CHÍ CHỌN nói thẳng ra.

    Cô hỏi: "lấy căn tương tự theo tiêu chí nào, cùng diện tích trước hay cùng
    địa điểm trước?" Câu hỏi đó phải trả lời được ngay trên giao diện, không
    phải mở code ra mới biết. Ba bước, đúng thứ tự trong `tim_comparables`:

        1. LỌC CỨNG theo phường. Không đủ 5 căn thì mới mở ra cả quận.
        2. Sắp theo chênh lệch diện tích TƯƠNG ĐỐI (chia cho diện tích của bạn).
        3. Đồng hạng thì xét tiếp chênh lệch số phòng ngủ.

    Vị trí đi trước diện tích vì đo được: bỏ nhóm đặc trưng vị trí ra khỏi mô
    hình làm sai số tăng 7,3 điểm (chung cư) và 8,4 điểm (nhà đất) — nhiều hơn
    bất kỳ nhóm nào khác. Một căn cùng phường khác diện tích vẫn đáng so hơn
    một căn cùng diện tích ở quận khác.

    Và GIÁ không tham gia việc chọn. Nếu lọc theo giá thì bảng chỉ còn những
    căn hợp ý mô hình, thành ra tự chứng minh mình đúng. Chọn theo đặc điểm rồi
    giá ra sao hiện vậy, kể cả khi nó ngược với ước lượng.
    """
    if cmp_df is None or cmp_df.empty:
        st.caption("Chưa tìm được căn nào đủ giống để đối chiếu.")
        return

    pham_vi = ("cả quận (phường của bạn không đủ 5 căn)" if mo_rong_quan
               else "cùng phường với bạn")
    st.caption(f"Lấy trong **{pham_vi}**, sắp theo **diện tích gần nhất**. "
               f"Giá không tham gia việc chọn.")

    dt_ban = (dac_diem or {}).get(cot_dt)
    hien = pd.DataFrame({
        "Giá rao": cmp_df["TARGET_gia_vnd"].map(format_ty),
        "Giá/m²": (cmp_df["gia_tren_m2"] / 1e6).round(1).astype(str) + " tr",
        "Diện tích": cmp_df[cot_dt].astype(str) + " m²",
        "Phường": cmp_df["phuong_moi"],
        "Đường": cmp_df.get("duong_pho", pd.Series(dtype=object)),
    })
    # Tên đường CHƯA XÁC MINH thì đánh dấu ngay trong ô, không để nó trông như
    # một sự thật đã kiểm. Kèm cột mức chắc chắn của địa chỉ.
    if "nguon_toa_do" in cmp_df.columns:
        chac = cmp_df.apply(_do_chac_dia_chi, axis=1)
        hien["Địa chỉ"] = [c[0] for c in chac]
        hien["Đường"] = [
            (f"{d} (?)" if isinstance(d, str) and d and not ok else d)
            for d, (_, ok) in zip(hien["Đường"], chac)]
    # Cột "Lệch diện tích" làm tiêu chí sắp xếp hiện ra thành số kiểm được:
    # người xem thấy đúng là bảng tăng dần theo cột này.
    if dt_ban:
        lech = (cmp_df[cot_dt] - dt_ban) / dt_ban * 100
        hien.insert(3, "Lệch diện tích",
                    lech.map(lambda x: f"{x:+.0f}%" if pd.notna(x) else "—"))
    # Cột "Đã lưu" thay cho việc chỉ dán link: tin rao bị sàn xoá sau ~30
    # ngày, nên ngày lưu là thông tin người xem cần để biết căn này còn nói
    # được gì về thị trường hôm nay. Link chỉ hiện khi còn cơ hội mở được.
    if "ngay_quan_sat" in cmp_df.columns:
        hien["Đã lưu"] = pd.to_datetime(
            cmp_df["ngay_quan_sat"], errors="coerce").dt.strftime("%d/%m/%Y")
    if "url_goc" in cmp_df.columns:
        tuoi = (pd.Timestamp.today().normalize()
                - pd.to_datetime(cmp_df.get("ngay_quan_sat"), errors="coerce")
                ).dt.days if "ngay_quan_sat" in cmp_df.columns else None
        u = cmp_df["url_goc"].where(
            cmp_df["url_goc"].astype(str).str.startswith("http"))
        if tuoi is not None:
            u = u.where(tuoi.isna() | (tuoi <= HAN_LINK_CU))
        hien["Tin gốc"] = u
        st.dataframe(hien, hide_index=True, width='stretch',
                     column_config={"Tin gốc": st.column_config.LinkColumn(
                         "Tin gốc", display_text="mở")})
    else:
        st.dataframe(hien, hide_index=True, width='stretch')

    if "Địa chỉ" in hien.columns:
        st.caption(
            "Cột **Địa chỉ** cho biết hệ thống xác định được vị trí căn đó tới "
            "mức nào. **“chỉ biết phường”** nghĩa là không tìm được con đường "
            "người đăng ghi ở trong phường đó — tên đường vẫn hiện (đánh dấu "
            "**(?)**) vì đó là nguyên văn tin rao, nhưng chưa kiểm được, và ghim "
            "của căn đó đặt ở giữa phường. Người đăng ghi sai địa chỉ là chuyện "
            "thường; hệ thống chép lại chứ không sửa hộ.")

    with st.expander("Xem nội dung 5 tin này (bản đã lưu)"):
        st.caption(
            "Tin rao bị sàn xoá sau khoảng 30 ngày, nên link không phải bằng "
            "chứng dùng được lâu. Bằng chứng là nội dung dưới đây — bản chốt "
            "lại đúng lúc thu thập, nằm trong dữ liệu của hệ thống.")
        for i, (_, r) in enumerate(cmp_df.iterrows()):
            if i:
                st.divider()
            _noi_dung_da_luu(r)


_BANG_CHUNG = {
    "đủ tin cậy":           ("●●●", COLOR["success_chu"], "nhiều căn làm chứng"),
    "tạm được":             ("●●○", COLOR["warn"],        "ít căn làm chứng"),
    "quá ít tin, đừng tin": ("●○○", COLOR["text_muted"], "gần như không có căn nào"),
}


# ĐÃ BỎ KHỐI "NẾU ĐỔI MỘT ĐẶC ĐIỂM, GIÁ THAY ĐỔI THẾ NÀO"
#
# Bỏ vì ba lý do, và lý do thứ ba mới là lý do quyết định:
#
# 1. TRÙNG VIỆC. Nó cũng là một phép phân rã của cùng mô hình, trả lời gần
#    đúng câu mà bảng yếu tố ở trên đã trả lời — chỉ khác ở chỗ nó đổi một ô
#    rồi chạy lại. Hai khối cạnh nhau nói cùng một chuyện bằng hai cách thì
#    người dùng không biết tin cái nào.
#
# 2. NGƯỜI DÙNG ĐÃ CÓ CÁCH LÀM ĐIỀU ĐÓ RỒI, trực tiếp hơn: đổi ô trong form
#    rồi bấm định giá lại. Kết quả thật, không phải mô phỏng.
#
# 3. NÓ LÀ KHỐI TỐN NHIỀU CHỮ NHẤT và cũng là khối hỏng nhiều nhất trong dự
#    án này: thang đo bị một tin ngoại lai kéo lệch, sắp xếp dẫn bằng những
#    dòng "0 căn — đừng tin", và một lỗi ValueError vì `_so()` gọi int() trên
#    "8.4". Mỗi lần sửa lại phải giải thích thêm một đoạn. Với một sản phẩm
#    sắp đem bảo vệ, bớt một thứ phải giải thích là bớt một chỗ hỏng.
#
# Phần tính toán vẫn còn ở `tinh_nang.dinh_gia_nguoc` — nếu cần dựng lại thì
# chỉ phải viết lại phần giao diện.


def _nap_tien_ich():
    import tien_ich as ti
    try:
        return ti.nap_tien_ich()
    except FileNotFoundError:
        return None


def _khoi_tien_ich(dac_diem: dict) -> None:
    import tien_ich as ti
    diem = _nap_tien_ich()
    if diem is None:
        st.caption("Chưa có dữ liệu tiện ích. Chạy `python tien_ich.py --tai` một lần.")
        return
    lat, lon = dac_diem.get("latitude"), dac_diem.get("longitude")
    if lat is None or pd.isna(lat):
        st.caption("Chưa xác định được toạ độ nên chưa tính được tiện ích.")
        return

    mot = pd.DataFrame([{"latitude": lat, "longitude": lon}])
    dt = ti.dac_trung_tien_ich(mot, diem)
    w = pd.Series(1 / len(ti.NHOM), index=list(ti.NHOM))
    d = float(ti.cham_diem(dt, w).iloc[0])

    st.markdown(
        f"""<div style="display:flex;align-items:baseline;gap:.6rem;margin-bottom:.5rem;">
          <span style="font-family:Newsreader,Georgia,serif;font-size:2.1rem;font-weight:600;
                color:{COLOR['text']};">{d:.0f}</span>
          <span style="color:{COLOR['text_muted']};">/ 100 điểm tiện ích sống</span>
        </div>""", unsafe_allow_html=True)

    # ĐIỂM NÀY TÍNH QUANH ĐIỂM NÀO — phải nói ra.
    #
    # Nếu người dùng chỉ chọn phường thì mốc là toạ độ trung vị của phường, mà
    # đo được: mốc đó cách vị trí thật của từng căn trong cùng phường trung vị
    # 0,77 km, p90 2,06 km. Nghĩa là hoàn toàn có thể xảy ra chuyện trường học
    # ở cuối phường còn mốc lại ở đầu phường. Để nguyên con số mà không nói gì
    # thì người dùng đọc là "tiện ích quanh nhà tôi" — sai.
    muc = dac_diem.get("_muc_toa_do")
    if muc in ("ghim", "du_an"):
        st.caption("Tính quanh **đúng vị trí bạn chỉ**.")
    elif muc == "duong":
        st.caption("Tính quanh **giữa con phố** — lệch 0,2–0,3 km.")
    else:
        st.caption("⚠️ Tính quanh **giữa phường**, lệch trung bình 0,77 km so "
                   "với nhà bạn. Ghim vị trí ở phần Vị trí để chính xác hơn.")

    ten = {"cho_sieu_thi": "Chợ, siêu thị", "truong_hoc": "Trường học",
           "cong_vien": "Công viên", "metro_ga": "Metro, ga",
           "benh_vien": "Bệnh viện", "dai_hoc": "Đại học"}
    for nhom in ti.NHOM:
        km = float(dt[f"kc_{nhom}_km"].iloc[0])
        n = int(dt[f"dem_{nhom}_1km"].iloc[0])
        rong = max(2, min(100, 100 * np.exp(-km / ti.NHOM[nhom][1])))
        st.markdown(
            f"""<div style="margin:.3rem 0;">
              <div style="display:flex;justify-content:space-between;font-size:.82rem;">
                <span>{ten[nhom]}</span>
                <span style="color:{COLOR['text_muted']};">
                  gần nhất {km:.2f} km · {n} trong 1 km</span>
              </div>
              <div style="height:5px;background:{COLOR['surface_2']};border-radius:99px;
                   margin-top:2px;">
                <div style="height:5px;width:{rong:.0f}%;background:{COLOR['accent']};
                     border-radius:99px;"></div>
              </div>
            </div>""", unsafe_allow_html=True)
    st.caption("Nguồn OpenStreetMap — ngoại thành hay bị chấm thấp vì OSM gắn "
               "thẻ thưa.")


MUC_TOA_DO = {
    "ghim":   ("Ghim trên bản đồ", "chính xác tới điểm bạn chọn"),
    "du_an":  ("Theo dự án",       "chính xác tới toà nhà"),
    "duong":  ("Theo đường phố",   "lệch khoảng 0,2–0,3 km"),
    "phuong": ("Theo phường",      "lệch khoảng 0,75 km, có khi tới 2 km"),
    "quan":   ("Theo quận",        "rất thô, chỉ nên dùng khi chưa rõ phường"),
}


def _chon_vi_tri(loai: str, cay: dict, khoa: str) -> dict:
    """Ba cách đặt vị trí, từ nhanh nhất tới chính xác nhất.

    VÌ SAO PHẢI CÓ CẢ BA
    --------------------
    Đo được: dùng đúng toạ độ thật thay cho mốc phường lấy lại 3,41 điểm MAPE
    ở chung cư và 2,29 điểm ở nhà đất — nhiều hơn mọi cải tiến còn lại cộng
    lại. Nguyên nhân là mốc phường cách vị trí thật trung vị 0,75 km, p90
    2,17 km, mà trong nội thành 2 km đã qua mấy khu giá khác hẳn.

    Nhưng bắt buộc ai cũng phải mò trên bản đồ thì nhiều người bỏ giữa chừng.
    Nên ba mức, người dùng tự chọn mức nào tiện:

        chọn phường   -> nhanh nhất, thô nhất
        chọn đường    -> gõ vài chữ là ra, gần đúng
        ghim bản đồ   -> chính xác nhất, tốn công nhất

    Danh sách đường CHỈ gồm phố mà hệ thống biết toạ độ thật (từ 3 căn trở
    lên). Nhận mọi cái tên rồi âm thầm rơi về mốc phường thì tệ hơn không cho
    chọn, vì người dùng tưởng mình đã khai chính xác.
    """
    quan_list = sorted(cay["cay"].keys())
    c1, c2 = st.columns(2)
    quan = c1.selectbox("Quận / Huyện", quan_list, index=None,
                        placeholder="Chọn Quận/Huyện", key=f"{khoa}_quan")
    ds_phuong = cay["cay"].get(quan, []) if quan else []
    phuong = c2.selectbox("Phường", ds_phuong, index=None,
                          placeholder="Chọn phường",
                          key=f"{khoa}_phuong") if ds_phuong else None

    ra = {"quan_huyen": quan, "phuong_moi": phuong,
          "duong_pho": None, "latitude": None, "longitude": None}
    if not (quan and phuong):
        st.caption("Chọn quận và phường trước, rồi có thể chỉ vị trí "
                   "chính xác hơn nếu muốn.")
        return ra

    ds_duong = vs.duong_trong_phuong(loai, quan, phuong)
    lua = ["Chỉ cần phường là đủ"]
    if ds_duong:
        lua.append(f"Chọn đường phố ({len(ds_duong)} phố)")
    lua.append("Gõ địa chỉ / ghim trên bản đồ")
    cach = st.radio("Muốn chỉ vị trí chính xác hơn không?", lua,
                    key=f"{khoa}_cach", horizontal=True,
                    help="Càng chỉ rõ vị trí, ước lượng càng sát. Đo được: "
                         "biết đúng vị trí thay vì chỉ biết phường giúp giảm "
                         "sai số khoảng 2–3 điểm phần trăm.")

    if cach.startswith("Chọn đường"):
        ra["duong_pho"] = st.selectbox(
            "Đường / phố", ds_duong, index=None,
            placeholder="Gõ vài chữ để tìm", key=f"{khoa}_duong")
        st.caption("Chỉ liệt kê những phố hệ thống biết vị trí thật. "
                   "Không thấy phố của bạn thì dùng cách ghim bản đồ.")

    elif cach.startswith("Gõ địa chỉ"):
        ra.update(_ghim_ban_do(loai, quan, phuong, khoa))

    return ra


def _tim_dia_chi_osm(dia_chi: str, quan: str, phuong: str) -> tuple | None:
    """Đường dự phòng: hỏi Nominatim khi dữ liệu của mình không có phố đó.

    Thử từ HẸP tới RỘNG. Ràng buộc chặt quá thì thất bại oan: gõ một phố ở
    Thanh Xuân trong khi đang chọn Gia Lâm, mà ghép cả phường lẫn quận vào
    truy vấn thì Nominatim trả về rỗng dù nó biết phố đó ở đâu.

    Kết quả CHỈ dùng để kéo bản đồ tới gần, không tự đặt ghim. `geocode.py`
    ghi lại một ca thật: hỏi "Phường Nghĩa Tân" thì Nominatim trả về một chi
    nhánh BIDV ở phường khác. Máy gợi ý, người bấm.
    """
    import urllib.parse
    import urllib.request

    dc = dia_chi.strip()
    thu = [", ".join(x for x in (dc, phuong, quan, "Hà Nội") if x),
           ", ".join(x for x in (dc, quan, "Hà Nội") if x),
           f"{dc}, Hà Nội"]
    for q in thu:
        ts = urllib.parse.urlencode({
            "format": "json", "limit": 1, "countrycodes": "vn",
            "viewbox": "105.27,20.55,106.03,21.40", "bounded": 1, "q": q})
        req = urllib.request.Request(
            f"https://nominatim.openstreetmap.org/search?{ts}",
            headers={"User-Agent": "dinh-gia-bds-hanoi/1.0 (luan van dai hoc)"})
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                kq = json.loads(r.read().decode("utf-8"))
        except Exception:
            return None            # mất mạng hoặc bị chặn: thôi, ghim tay
        if kq:
            return (float(kq[0]["lat"]), float(kq[0]["lon"]),
                    kq[0].get("display_name", ""))
    return None


def _ghim_ban_do(loai: str, quan: str, phuong: str, khoa: str) -> dict:
    """Bản đồ bấm-để-ghim, kèm ô gõ địa chỉ để nhảy tới.

    TRA DỮ LIỆU CỦA MÌNH TRƯỚC, GỌI MẠNG SAU
    ----------------------------------------
    Hệ thống đã biết toạ độ của 540 con phố (nhà đất) và 190 phố (chung cư),
    dựng từ chính những căn có toạ độ chính xác. Tra bảng đó thì tức thì, không
    cần mạng, và chắc chắn là chỗ hệ thống có dữ liệu. Chỉ khi không thấy mới
    hỏi Nominatim.

    NỀN BẢN ĐỒ PHẢI LÀ OPENSTREETMAP
    --------------------------------
    Bản đầu tôi đặt `tiles="CartoDB positron"` cho nhìn nhẹ mắt. Nền đó giờ
    đòi API key nên bản đồ phủ kín chữ "API KEY REQUIRED" và mất hết tên phố —
    tức là không ghim được. OpenStreetMap miễn phí, không cần khoá, và hiện rõ
    tên phố, ngõ, số nhà.
    """
    try:
        import folium
        from streamlit_folium import st_folium
    except ImportError:
        st.info("Cách ghim bản đồ cần thư viện `streamlit-folium`. Cài bằng "
                "`pip install streamlit-folium folium` rồi khởi động lại. "
                "Trong lúc đó bạn vẫn dùng được cách chọn phường hoặc đường.")
        return {}

    goc = vs.bo_sung_toa_do(loai, {"quan_huyen": quan, "phuong_moi": phuong})
    tam_phuong = [goc.get("latitude") or vs.TRUNG_TAM[0],
                  goc.get("longitude") or vs.TRUNG_TAM[1]]
    k_ghim, k_tam, k_ten = f"{khoa}_ghim", f"{khoa}_tam", f"{khoa}_ten"

    c1, c2 = st.columns([3, 1])
    dia_chi = c1.text_input(
        "Gõ địa chỉ để nhảy tới", placeholder="ví dụ: 1 Phố Bùi Xương Trạch",
        key=f"{khoa}_dc", label_visibility="collapsed") or ""
    if c2.button("Tìm", key=f"{khoa}_tim", width='stretch') and dia_chi.strip():
        st.session_state[k_ten] = None
        # (1) tra bảng phố của chính hệ thống, toàn Hà Nội
        hit = vs.tim_duong_toan_thanh(loai, dia_chi)
        if hit:
            h = hit[0]
            st.session_state[k_tam] = [h["latitude"], h["longitude"]]
            st.session_state[k_ten] = {
                "nguon": "trong dữ liệu", "ten": h["duong_pho"],
                "quan": h["quan_huyen"], "phuong": h["phuong_moi"],
                "so_can": h["so_can"]}
        else:
            # (2) không có trong dữ liệu -> hỏi OpenStreetMap
            r = _tim_dia_chi_osm(dia_chi, quan, phuong)
            if r:
                st.session_state[k_tam] = [r[0], r[1]]
                st.session_state[k_ten] = {"nguon": "OpenStreetMap",
                                           "ten": r[2]}
            else:
                st.warning(
                    "Không tra được địa chỉ này — hệ thống chưa có căn nào trên "
                    "phố đó, và OpenStreetMap cũng không trả về kết quả. "
                    "Bạn kéo bản đồ tới đúng chỗ rồi bấm để ghim.")

    tt = st.session_state.get(k_ten)
    if tt and tt.get("nguon") == "trong dữ liệu":
        # So CHỈ theo PHƯỜNG. Quận là nhãn suy ra (quận chiếm đa số tin của
        # phường đó) — một phường mới trải trên nhiều quận cũ, nên so cả quận
        # sẽ báo lệch oan: "Phương Liệt, Hoàng Mai không phải Phương Liệt,
        # Thanh Xuân" trong khi vẫn đúng một phường.
        lech = tt["phuong"] != phuong
        if lech:
            st.warning(
                f"**{tt['ten']}** nằm ở **{tt['phuong']}** "
                f"({tt['quan']}) — không phải {phuong} như bạn đang chọn. "
                f"Bản đồ đã nhảy tới đó. Nếu đúng là căn của bạn, hãy đổi lại "
                f"Phường ở trên cho khớp rồi ghim.")
        else:
            st.success(f"Đã nhảy tới **{tt['ten']}** — hệ thống có "
                       f"{tt['so_can']} căn trên phố này. **Bấm lên bản đồ** "
                       f"để chốt vị trí.")
    elif tt:
        st.caption(f"OpenStreetMap trả về: {tt['ten']}. "
                   f"**Bấm lên bản đồ** để chốt vị trí.")

    da_ghim = st.session_state.get(k_ghim)
    da_nhay = st.session_state.get(k_tam)
    tam = da_ghim or da_nhay or tam_phuong

    m = folium.Map(location=tam, zoom_start=17 if (da_ghim or da_nhay) else 15,
                   tiles=None, control_scale=True)
    _nen_ban_do(m, folium)
    folium.Marker(tam_phuong, tooltip=f"Điểm giữa {phuong}",
                  icon=folium.Icon(color="lightgray", icon="info-sign")).add_to(m)
    if da_ghim:
        folium.Marker(da_ghim, tooltip="Vị trí bạn đã chọn",
                      icon=folium.Icon(color="darkblue", icon="home")).add_to(m)

    kq = st_folium(m, height=380, width=None,
                   returned_objects=["last_clicked"], key=f"{khoa}_map")
    bam = (kq or {}).get("last_clicked")
    if bam:
        st.session_state[k_ghim] = [bam["lat"], bam["lng"]]
        da_ghim = st.session_state[k_ghim]

    if da_ghim:
        c3, c4 = st.columns([3, 1])
        c3.caption(f"📍 Đã ghim tại {da_ghim[0]:.5f}, {da_ghim[1]:.5f}. "
                   "Bấm chỗ khác để đổi.")
        if c4.button("Bỏ ghim", key=f"{khoa}_bo", width='stretch'):
            st.session_state.pop(k_ghim, None)
            st.rerun()
        return {"latitude": da_ghim[0], "longitude": da_ghim[1]}

    st.caption("Gõ tên phố rồi bấm **Tìm** để bản đồ nhảy tới, hoặc tự kéo và "
               "phóng to. Sau đó **bấm lên bản đồ** đúng chỗ căn nhà. Ghim xám "
               "là điểm giữa phường, chỉ để định hướng.")
    return {}


def _hien_muc_toa_do(dac_diem: dict) -> None:
    """Nói rõ hệ thống đang định vị ở mức nào. Người dùng có quyền biết con số
    mình nhận được dựa trên vị trí chính xác đến đâu."""
    muc = dac_diem.get("_muc_toa_do")
    if not muc:
        return
    ten, do_chinh_xac = MUC_TOA_DO.get(muc, (muc, ""))
    mau = (COLOR["success_chu"] if muc in ("ghim", "du_an", "duong")
           else COLOR["warn"])
    st.markdown(
        f"""<div style="font-size:.78rem;color:{COLOR['text_muted']};
             padding:.3rem 0 .1rem;">
          Định vị: <b style="color:{mau};">{ten}</b> — {do_chinh_xac}
        </div>""", unsafe_allow_html=True)


def render_ket_qua(loai: str, dac_diem: dict, cot_dt: str) -> None:
    """Toàn bộ phần kết quả: khoảng CQR, độ tin cậy, giải thích, tin tương đồng."""
    # Bổ sung toạ độ TRƯỚC khi dùng: người dùng chỉ chọn quận/phường, còn
    # `dinh_gia` suy ra toạ độ bên trong rồi trả về dict mới — nếu không lấy
    # bản đã bổ sung ra dùng chung thì khối "Tiện ích quanh đây" tưởng là
    # không có toạ độ và im lặng bỏ qua.
    dac_diem = vs.bo_sung_toa_do(loai, dac_diem)
    kq = vs.dinh_gia(loai, dac_diem)

    # `dac_diem` đã đi qua bo_sung_toa_do ở trên nên mang sẵn `_muc_toa_do`.
    _hien_muc_toa_do(dac_diem)
    dt = dac_diem.get(cot_dt)

    st.metric("Giá bán dự kiến", format_vnd(kq["gia"]))
    st.markdown(
        f"""<div style="background:{COLOR['accent_soft']};border:1px solid {COLOR['accent']}33;
             border-radius:14px;padding:.7rem .9rem;margin:.5rem 0;">
          <div style="color:{COLOR['text_muted']};font-size:.8rem;">
            Khoảng tin cậy {kq['muc_tin_cay']}% — nghĩa là giá thật rơi vào khoảng này
            khoảng {kq['muc_tin_cay']} trên 100 lần
          </div>
          <div style="font-family:Newsreader,Georgia,serif;font-weight:600;
               font-size:1.2rem;color:{COLOR['text']};margin-top:.2rem;
               font-variant-numeric:tabular-nums;">
            {format_ty(kq['khoang_duoi'])} — {format_ty(kq['khoang_tren'])}
            <span style="color:{COLOR['text_muted']};font-weight:500;font-size:.85rem;">
              (±{kq['do_rong_phan_tram']:.0f}%)</span>
          </div>
        </div>""",
        unsafe_allow_html=True,
    )
    _huy_hieu_tin_cay(kq["do_tin_cay"])
    if dt:
        st.metric("Giá/m² dự kiến", f"{kq['gia'] / dt:,.0f} VND/m²")

    with st.expander("Vì sao lại là con số này?", expanded=True):
        gt = kq.get("giai_thich") or {}
        if gt:
            _thanh_yeu_to(gt["yeu_to"], gt.get("gia_co_so"),
                          gt.get("gia_du_doan"))

    with st.expander("Tiện ích quanh đây", expanded=False):
        _khoi_tien_ich(dac_diem)

    with st.expander(f"Những căn tương tự ({len(kq.get('comparables', []))} căn)",
                     expanded=True):
        _bang_comparables(kq.get("comparables"), cot_dt, dac_diem,
                          mo_rong_quan=_da_mo_rong_quan(loai, dac_diem))

    st.session_state["diem_vua_dinh_gia"] = {
        "loai": loai,
        "lat": dac_diem.get("latitude"),
        "lon": dac_diem.get("longitude"),
        "ten": f"{dac_diem.get('phuong_moi') or ''}, {dac_diem.get('quan_huyen') or ''}",
    }


# =============================================================================
# TRANG CHUNG CƯ
# =============================================================================

def render_chungcu() -> None:
    cay = nap_cay("chungcu")
    col_form, col_result = st.columns([1.15, 0.85], gap="large")

    with col_form:
        with st.container(border=True):
            section_header(ICON_AREA, "Diện tích & phòng", "")
            c1, c2, c3 = st.columns(3)
            with c1:
                dien_tich = st.number_input("Diện tích (m²)", min_value=15.0, max_value=500.0,
                                            value=None, step=1.0, placeholder="VD: 70")
            with c2:
                phong_ngu = st.selectbox("Số phòng ngủ", [1, 2, 3, 4, 5],
                                         index=None, placeholder="Chọn")
            with c3:
                phong_vs = st.selectbox("Số phòng vệ sinh", [1, 2, 3],
                                        index=None, placeholder="Chọn",
                                        )
            tang_input = st.number_input(
                "Tầng số", min_value=0, max_value=60, value=None, step=1,
                placeholder="Bỏ trống nếu không rõ")
            tang_so = float(tang_input) if tang_input else None

        with st.container(border=True):
            section_header(ICON_PIN, "Vị trí", "Càng rõ càng sát")
            vt = _chon_vi_tri("chungcu", cay, "cc")
            quan, phuong = vt["quan_huyen"], vt["phuong_moi"]
            du_an = render_project_selector(cay, quan, phuong) if (quan and phuong) \
                else PROJECT_OTHER

        with st.container(border=True):
            # SẮP XẾP FORM THEO ĐÓNG GÓP ĐO ĐƯỢC, KHÔNG THEO CẢM TÍNH.
            #
            # Bỏ từng ô ra khỏi model rồi đo lại MAPE trên 4.887 tin chung cư:
            #
            #     Pháp lý          +0,24 điểm
            #     Phòng ngủ        +0,13
            #     Tình trạng BĐS   +0,05
            #     Phòng vệ sinh    +0,05
            #     Tầng             +0,04
            #     Hướng cửa        −0,01   <- bỏ đi model TỐT HƠN
            #     Hướng ban công   −0,04
            #     Nội thất         −0,05
            #
            # Ba ô cuối không chỉ vô ích mà còn hơi có hại. Bắt người dùng điền
            # 12 ô rồi dùng có 4 là lãng phí thời gian của họ — nên phần ít giá
            # trị gập lại, ai muốn vẫn mở ra điền được.
            section_header(ICON_DOC, "Pháp lý", "Ảnh hưởng giá nhiều nhất")
            giay_to = st.selectbox("Giấy tờ pháp lý", LEGAL_CHOICES,
                                   index=None, placeholder="Chọn")
            st.caption("Vị trí, diện tích và pháp lý đã đủ để định giá.")

            with st.expander("Điền thêm (không bắt buộc)", expanded=False):
                tinh_trang_bds = st.selectbox(
                    "Tình trạng bất động sản", STATUS_CHOICES,
                    index=None, placeholder="Chọn")
                noi_that = st.selectbox(
                    "Tình trạng nội thất", INTERIOR_CHOICES,
                    index=None, placeholder="Chọn",
                    help="Gần như không đổi được kết quả")
                hc1, hc2 = st.columns(2)
                with hc1:
                    huong_ban_cong = st.selectbox(
                        "Hướng ban công", DIRECTION_CHOICES, index=None,
                        placeholder="Chọn")
                with hc2:
                    huong_cua = st.selectbox(
                        "Hướng cửa chính", DIRECTION_CHOICES, index=None,
                        placeholder="Chọn")
                can_goc = st.checkbox("Căn góc")

            predict_clicked = st.button("Định giá ngay", type="primary",
                                        width='stretch')

    with col_result:
        with st.container(border=True):
            section_header(ICON_CHART, "Kết quả định giá", "")
            if predict_clicked and (not quan or not phuong or not dien_tich):
                st.warning("Vui lòng nhập **Diện tích**, **Quận/Huyện** và "
                           "**Phường/Xã** trước khi định giá.")
            elif predict_clicked:
                scroll_to_top()
                dac_diem = {
                    "dien_tich_m2": dien_tich,
                    "so_phong_ngu": phong_ngu,
                    "so_phong_vs": phong_vs,
                    "tang_so": tang_so,
                    "quan_huyen": quan,
                    "phuong_moi": phuong,
                    # đường phố / toạ độ ghim, nếu người dùng có chỉ rõ
                    "duong_pho": vt.get("duong_pho"),
                    "latitude": vt.get("latitude"),
                    "longitude": vt.get("longitude"),
                    "du_an_clean": None if du_an == PROJECT_OTHER else du_an,
                    "phap_ly": _hoac_none(giay_to),
                    "noi_that": _hoac_none(noi_that),
                    "huong_ban_cong": _hoac_none(huong_ban_cong),
                    "huong_cua": _hoac_none(huong_cua),
                    "tinh_trang_bds": _hoac_none(tinh_trang_bds),
                    "nhac_lo_goc": int(can_goc),
                }
                render_ket_qua("chungcu", dac_diem, "dien_tich_m2")
            else:
                st.markdown(
                    "Nhập thông tin căn hộ bên trái, chọn **Quận → Phường → Dự án** "
                    "rồi bấm **Định giá ngay** để xem mức giá ước lượng.")

        with st.container(border=True):
            section_header(ICON_INFO, "Đọc kết quả thế nào cho đúng", "")
            st.markdown(
                "- Con số ở giữa là mức giá dễ gặp nhất; **khoảng vàng** mới là "
                "thứ nên bám vào khi thương lượng.\n"
                "- Khoảng rộng không có nghĩa hệ thống đoán kém. Hai căn giống hệt "
                "nhau ngoài đời đã được chào giá chênh nhau khoảng 20% — hệ thống "
                "chỉ đang nói thật về mức chênh đó.\n"
                "- **Độ tin cậy thấp** nghĩa là khu vực đó có quá ít căn để đối "
                "chiếu; hãy xem thêm danh sách những căn tương tự bên dưới.\n"
                "- Các yếu tố ± cho biết những căn có đặc điểm đó thường đắt hay rẻ "
                "hơn bao nhiêu — không phải lời hứa sửa xong sẽ bán được thêm."
            )


def render_project_selector(cay: dict, quan: str, phuong: str) -> str:
    """Danh sách dự án lọc theo đúng (quận, phường) đang chọn."""
    projects = cay["du_an"].get((quan, phuong), [])
    options = [PROJECT_OTHER] + [p for p in projects if p != "Không xác định"]
    labels = {PROJECT_OTHER: "Dự án khác / không có trong danh sách "
                             "(ước lượng theo phường)"}
    return st.selectbox(
        "Tên dự án", options=options,
        format_func=lambda x: labels.get(x, x),
        help="Danh sách lọc theo Quận/Phường đang chọn. Chọn 'Dự án khác' nếu "
             "căn hộ không thuộc các dự án này.")


# =============================================================================
# TRANG NHÀ ĐẤT
# =============================================================================

def render_nhadat() -> None:
    cay = nap_cay("nhadat")
    col_form, col_result = st.columns([1.15, 0.85], gap="large")

    with col_form:
        with st.container(border=True):
            section_header(ICON_AREA, "Loại hình & diện tích", "")
            loai_hinh = st.selectbox("Loại hình", NHADAT_LOAI_HINH,
                                     help="Ô ảnh hưởng tới giá nhiều nhất",
                                     index=None, placeholder="Chọn loại hình")
            a1, a2 = st.columns(2)
            with a1:
                dien_tich_dat = st.number_input("Diện tích đất (m²)", min_value=5.0,
                                                max_value=2000.0, value=None, step=1.0,
                                                placeholder="VD: 45")
            with a2:
                dt_sd_in = st.number_input("Diện tích sử dụng (m²)", min_value=0.0,
                                           max_value=5000.0, value=None, step=1.0,
                                           placeholder="Bỏ trống nếu không rõ")
            b1, b2 = st.columns(2)
            with b1:
                mt_in = st.number_input("Mặt tiền (m)", min_value=0.0, max_value=100.0,
                                        value=None, step=0.1, placeholder="VD: 4")
            with b2:
                cd_in = st.number_input("Chiều dài (m)", min_value=0.0, max_value=200.0,
                                        value=None, step=0.1,
                                        placeholder="Bỏ trống nếu không rõ")
            dr_in = st.number_input("Độ rộng đường/ngõ trước nhà (m)",
                                    min_value=0.0, max_value=100.0, value=None, step=0.1,
                                    placeholder="Bỏ trống nếu không rõ")
            c1, c2, c3 = st.columns(3)
            with c1:
                pn_in = st.number_input("Số phòng ngủ", min_value=0, max_value=20,
                                        value=None, step=1, placeholder="—")
            with c2:
                vs_in = st.number_input("Số phòng VS", min_value=0, max_value=20,
                                        value=None, step=1, placeholder="—")
            with c3:
                tang_in = st.number_input("Tổng số tầng", min_value=0, max_value=50,
                                          value=None, step=1, placeholder="—")
            f1, f2, f3, f4 = st.columns(4)
            with f1:
                co_oto = st.checkbox("Ô tô vào/đỗ")
            with f2:
                kinh_doanh = st.checkbox("Tiện kinh doanh")
            with f3:
                thang_may = st.checkbox("Có thang máy")
            with f4:
                lo_goc = st.checkbox("Lô góc")

        with st.container(border=True):
            section_header(ICON_PIN, "Vị trí", "Càng rõ càng sát")
            vt = _chon_vi_tri("nhadat", cay, "nd")
            quan, phuong = vt["quan_huyen"], vt["phuong_moi"]

        with st.container(border=True):
            # Đóng góp đo được trên 12.294 tin nhà đất (bỏ từng ô, đo lại):
            #
            #     Loại hình  +0,88     Phòng ngủ      +0,35
            #     Số tầng    +0,57     Mặt tiền       +0,23
            #     Pháp lý    +0,52     Phòng vệ sinh  +0,07
            #     Rộng ngõ   +0,40     Ô tô vào       +0,04
            #                          Hướng cửa      +0,02
            #                          Nội thất       −0,01
            #
            # Khác hẳn chung cư: ở nhà đất phần lớn các ô đều đáng điền, vì
            # loại hình và các thông số hình học mang nhiều thông tin. Chỉ hai
            # ô cuối là gần như vô nghĩa.
            section_header(ICON_DOC, "Pháp lý", "Ảnh hưởng mạnh tới giá — nên điền")
            phap_ly = st.selectbox("Giấy tờ pháp lý", NHADAT_PHAP_LY,
                                   index=None, placeholder="Chọn")
            st.caption("Loại hình, số tầng, pháp lý và độ rộng ngõ ảnh hưởng "
                       "nhiều nhất — điền được bốn ô này thì sát hơn hẳn.")

            with st.expander("Điền thêm (không bắt buộc)", expanded=False):
                noi_that = st.selectbox(
                    "Tình trạng nội thất", NHADAT_NOI_THAT, index=None,
                    placeholder="Chọn",
                    help="Gần như không đổi được kết quả")
                huong = st.selectbox(
                    "Hướng cửa chính", NHADAT_HUONG, index=None,
                    placeholder="Chọn")
                dac_diem_chon = st.multiselect("Đặc điểm (chọn nhiều)",
                                               NHADAT_DAC_DIEM, default=[])

            predict_clicked = st.button("Định giá ngay", type="primary",
                                        width='stretch', key="nd_btn")

    with col_result:
        with st.container(border=True):
            section_header(ICON_CHART, "Kết quả định giá",
                           "Khoảng có bảo chứng thống kê, không phải ±22%")
            if predict_clicked and (not loai_hinh or not dien_tich_dat
                                    or not quan or not phuong):
                st.warning("Vui lòng nhập **Loại hình**, **Diện tích đất**, "
                           "**Quận/Huyện** và **Phường/Xã**.")
            elif predict_clicked:
                scroll_to_top()
                dac_diem = {
                    "dien_tich_dat_m2": dien_tich_dat,
                    "dien_tich_su_dung_m2": dt_sd_in or None,
                    "so_phong_ngu": pn_in or None,
                    "so_phong_vs": vs_in or None,
                    "tong_so_tang": tang_in or None,
                    "mat_tien_m": mt_in or None,
                    "chieu_dai_m": cd_in or None,
                    "duong_rong_m": dr_in or None,
                    "co_oto": int(co_oto), "kinh_doanh": int(kinh_doanh),
                    "thang_may": int(thang_may), "lo_goc": int(lo_goc),
                    "quan_huyen": quan, "phuong_moi": phuong,
                    "duong_pho": vt.get("duong_pho"),
                    "latitude": vt.get("latitude"),
                    "longitude": vt.get("longitude"),
                    "loai_hinh": loai_hinh,
                    "phap_ly": _hoac_none(phap_ly),
                    "noi_that": _hoac_none(noi_that),
                    "huong_cua": _hoac_none(huong),
                }
                for dd in NHADAT_DAC_DIEM:
                    dac_diem[f"dd_{dd}"] = int(dd in dac_diem_chon)
                render_ket_qua("nhadat", dac_diem, "dien_tich_dat_m2")
            else:
                st.markdown(
                    "Nhập thông tin nhà/đất bên trái, chọn **Loại hình → Diện tích → "
                    "Quận → Phường** rồi bấm **Định giá ngay**.")

        with st.container(border=True):
            section_header(ICON_INFO, "Đọc kết quả thế nào cho đúng", "")
            st.markdown(
                "- Nhà đất khó đoán hơn chung cư nhiều: hai căn cùng phường, cùng "
                "diện tích vẫn chênh nhau vì hình thửa, ngõ và hướng — nên khoảng "
                "tin cậy rộng hơn.\n"
                "- **Pháp lý** và **ô tô vào nhà** là hai yếu tố tác động mạnh nhất.\n"
                "- Điền càng đầy đủ, ước lượng càng sát; bỏ trống vẫn dự đoán được."
            )


# =============================================================================
# TRANG BẢN ĐỒ
# =============================================================================

# Thang màu cho các dải giá. Vàng nhạt → đỏ sẫm, MỘT tông chứ không dùng thang
# cầu vồng xanh-lục-đỏ: thang cầu vồng đổi màu nhanh ở quãng xanh lá sang vàng,
# tạo ra đường biên giả khiến người xem tưởng có ranh giới thật ở đó. Một tông
# thì độ đậm tăng đều, mắt đọc ra thứ tự ngay mà không cần tra chú giải.
# NỀN BẢN ĐỒ SÁNG.
#
# "light" là nền Carto sáng mà pydeck gọi được không cần khoá API — đã sập một
# =============================================================================
# NỀN ẢNH CHO BẢN ĐỒ FOLIUM
# =============================================================================
# CHUYỆN ĐÃ XẢY RA (đo ngày 20/09/2026, ngay trên máy người dùng)
#
# Bản đồ ghim hiện ra một mảng xám trơn. Nút phóng to và thước tỷ lệ vẫn vẽ
# bình thường — chúng chỉ là CSS — nên nhìn như "bản đồ hỏng" chứ không ai
# đoán được nguyên nhân nằm ở nền ảnh.
#
# Đo từng máy chủ một bằng trình duyệt của chính người dùng:
#   tile.openstreetmap.org   HỎNG sau ~300 ms   <- bị CHẶN, không phải quá hạn
#   cdnjs, jsdelivr          ĐẠT               (nên Leaflet vẫn chạy)
#   cartocdn, Esri           ĐẠT
#
# MỘT CÁI BẪY PHẢI GHI LẠI, VÌ TÔI SUÝT SẬP VÀO LẦN THỨ HAI
#
# Thấy "cartocdn ĐẠT" thì tưởng đổi sang CARTO là xong. Nhưng phép thử của
# tôi chỉ hỏi "ảnh có tải được không": CARTO trả HTTP 200 kèm một ảnh PNG
# 256×256 hoàn toàn hợp lệ — mà nội dung ảnh là dòng chữ "API KEY REQUIRED"
# in chéo kín ô. Máy báo ĐẠT trong khi bản đồ hỏng sạch. Phải MỞ RA NHÌN mới
# biết. Ghi lại đây để lần sau đừng tin một phép đo chỉ nhìn mã trạng thái.
#
# (Nền của pydeck ở tab Khu vực là hàng VECTOR — `carto.streets/v1/*.mvt`,
# một sản phẩm khác — vẫn miễn phí, vẫn chạy, đã kiểm lại. Chỗ đó không sửa.)
#
# CHỌN GÌ, VÀ VÌ SAO KHÔNG CHỈ ĐỔI MỘT MÁY CHỦ
#
# Nền chính là máy chủ gương của cộng đồng OSM Đức: CÙNG một kiểu vẽ với
# tile.openstreetmap.org — tên ngõ, số nhà, khối nhà đều còn — nên mọi thứ
# khác trong thiết kế giữ nguyên, không phải chỉnh theo.
#
# Nhưng bài học thật của lần này không phải "đổi sang máy chủ khác", mà là
# "một máy chủ ảnh miễn phí thì lúc nào cũng có thể mất hoặc bị chặn". App
# này còn phải chạy cho người xem ở mạng khác nữa, và tôi không đo được mạng
# của họ. Nên có dây chuyền dự phòng chạy NGAY TRONG TRÌNH DUYỆT: nền nào
# hỏng liên tiếp thì tự chuyển sang nền kế tiếp; hỏng hết thì NÓI RA, tuyệt
# đối không để lại một màn xám im lặng như lần này nữa.
NEN_FOLIUM = [
    ("https://tile.openstreetmap.de/{z}/{x}/{y}.png",
     '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
     ' · máy chủ gương FOSSGIS'),
    ("https://a.tile.openstreetmap.fr/osmfr/{z}/{x}/{y}.png",
     '&copy; OpenStreetMap France'),
    # Esri xếp thư mục theo {z}/{y}/{x}, ngược với OSM. Leaflet thay theo TÊN
    # ô chứ không theo thứ tự nên để vậy là đúng, đừng "sửa" cho giống trên.
    ("https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map"
     "/MapServer/tile/{z}/{y}/{x}",
     '&copy; Esri'),
    # Vẫn để nền gốc ở cuối: mạng nào không chặn nó thì nó vẫn là nền tốt
    # nhất, và ngày nào đó chỗ chặn này được gỡ.
    ("https://tile.openstreetmap.org/{z}/{x}/{y}.png",
     '&copy; OpenStreetMap'),
]

# Bao nhiêu ô hỏng LIÊN TIẾP thì đổi nền. Không đổi ngay từ ô đầu: lúc kéo
# bản đồ nhanh, Leaflet huỷ giữa chừng vài yêu cầu và chúng cũng vào
# `tileerror` — đổi nền vì mấy ô đó là đổi oan.
NGUONG_HONG_NEN = 5


def _nen_ban_do(m, folium) -> None:
    """Gắn nền chính + dây chuyền dự phòng cho một bản đồ folium."""
    url0, attr0 = NEN_FOLIUM[0]
    lop = folium.TileLayer(tiles=url0, attr=attr0, name="nen",
                           control=False, max_zoom=19)
    lop.add_to(m)

    # PHẢI GẮN VÀO CHÍNH BẢN ĐỒ, KHÔNG GẮN VÀO `get_root().script`.
    #
    # Thử cách kia trước: `m.get_root().script.add_child(...)` đổ đoạn JS ra
    # vị trí 2808 trong file, trong khi `var map_... = L.map(` mãi vị trí 3804
    # mới có. Tức là JS chạy trước khi bản đồ tồn tại, ném ReferenceError, và
    # dây chuyền dự phòng không bao giờ chạy — im lặng hỏng, đúng kiểu lỗi mà
    # cả phần này sinh ra để tránh. Gắn làm con của bản đồ thì folium xếp nó
    # vào đúng đoạn khởi tạo, sau bản đồ và sau lớp nền.
    js = """
        (function () {
          var lop = %s, ban_do = %s;
          var du = %s, i = 0, hong = 0, het = false;
          lop.on('tileload', function () { hong = 0; });
          lop.on('tileerror', function () {
            if (het || ++hong < %d) { return; }
            hong = 0;
            if (i < du.length) { lop.setUrl(du[i++]); return; }
            het = true;
            var bao = document.createElement('div');
            bao.style.cssText = 'position:absolute;z-index:500;left:50%%;top:50%%;'
              + 'transform:translate(-50%%,-50%%);max-width:84%%;padding:.7rem .9rem;'
              + 'background:#fff;border:1px solid rgba(23,24,28,.22);'
              + 'border-radius:10px;font:13px/1.45 sans-serif;color:#17181C;'
              + 'text-align:center;box-shadow:0 4px 14px rgba(23,24,28,.12)';
            bao.innerHTML = 'Mạng của bạn không tải được ảnh nền bản đồ nào '
              + '(đã thử ' + (du.length + 1) + ' nguồn).<br>'
              + '<b>Ghim vẫn bấm được và toạ độ vẫn đúng</b> — chỉ là không '
              + 'có ảnh nền để nhìn.';
            ban_do.getContainer().appendChild(bao);
          });
        })();
    """ % (lop.get_name(), m.get_name(),
           json.dumps([u for u, _ in NEN_FOLIUM[1:]]), NGUONG_HONG_NEN)

    # Jinja2 nuốt `{{`, `{%` và `{#`. URL nền chứa `{z}/{x}/{y}` — dấu ngoặc
    # ĐƠN nên không sao — nhưng kiểm thẳng ra đây, vì nếu sau này ai thêm một
    # nền có `{{` thì trang sẽ vỡ ở tận lúc chạy chứ không báo gì lúc này.
    for xau in ("{{", "{%", "{#"):
        if xau in js:
            raise ValueError(f"JS nền bản đồ chứa {xau!r} — Jinja2 sẽ nuốt nó")

    from branca.element import MacroElement, Template

    class _DuPhongNen(MacroElement):
        _template = Template("{% macro script(this, kwargs) %}"
                             + js + "{% endmacro %}")

    m.add_child(_DuPhongNen())


# lần vì chọn "CartoDB positron" cho folium: nền đó giờ đòi khoá nên bản đồ
# phủ kín chữ "API KEY REQUIRED". Chuỗi "light"/"dark" là tên rút gọn pydeck
# tự dịch, không phải URL nền, nên nó không vướng chuyện khoá.
#
# Nếu một ngày nền này cũng đòi khoá thì đổi về "road"; đừng dựng URL nền tay.
MAP_STYLE = "light"

THANG_MAU = [_rgb(x) for x in THANG_PHUONG]

# Nền của chính bản đồ (Carto "light"), KHÁC nền trang #F7F5F1. Mọi phép đo
# tương phản của lớp tô phải lấy nền này làm mốc, vì đó là thứ nằm dưới nó.
NEN_BAN_DO = [248, 248, 248]

# Độ đục của lớp vùng giá — 0,88, chứ không phải 0,72 như hằng số này bị bỏ
# quên trước đây (nó được khai báo rồi không ai dùng, còn lớp GeoJson thì ghi
# thẳng `opacity=0.72` vào thân hàm; giờ lớp đó dùng đúng hằng số này).
#
# 0,88 là mức đo được: tô mờ hơn thì thang bị nền bản đồ kéo nhạt lại, ba bậc
# đầu dồn vào nhau còn ΔL 0,04; đục hơn nữa thì mất hết tên phố bên dưới.
DUC_VUNG = 0.88

# Màu cho phường chưa đủ dữ liệu. Trắng ngà, nằm NGOÀI hệ xanh hoàn toàn: giờ
# bậc nhạt nhất đã đủ đậm nên nó tách khỏi ô này 2,06:1 (trước chỉ 1,11:1 —
# "rẻ nhất" và "chưa đủ dữ liệu" trông y hệt nhau). Kèm thêm viền gạch nét
# đứt ở chú giải để không phải dựa vào riêng màu.
MAU_THIEU = [252, 251, 248]


def _mau_bac(bac: int, so_bac: int) -> list[int]:
    """Màu của dải thứ `bac` trong thang `so_bac` bậc.

    Không còn nội suy trong sRGB: `thang_vung` sinh thẳng ra đúng `so_bac` màu
    có độ sáng chia đều trong OKLab. Nội suy sRGB là nguyên nhân ba bậc nhạt
    nhất của bản đồ phường dính vào nhau (xem ghi chú ở `THANG_HUE`).
    """
    t = thang_vung(max(int(so_bac), 1))
    return list(t[min(max(int(bac), 0), len(t) - 1)])


def _chu_giai_vung(vung: dict) -> None:
    """Chú giải bậc thang rời, in ĐÚNG các mốc thật.

    Các bậc chia theo phân vị nên khoảng cách giữa chúng KHÔNG bằng nhau — bậc
    trên cùng trải rộng hơn bậc dưới nhiều lần. Chú giải phải in ra từng mốc
    một, chứ nếu chỉ ghi hai đầu như thang liên tục thì người xem sẽ ngầm hiểu
    là chia đều và đọc sai độ chênh giữa các vùng.
    """
    # Số ô màu suy ra từ chính các bậc đang được vẽ, không lấy từ `so_bac`:
    # bản đồ phường có 7 bậc, còn bề mặt đường đồng mức có 11 (thêm hai dải mở
    # ở hai đầu). Lấy nhầm thì chú giải lệch số ô so với bản đồ.
    bac_co = [f["bac"] for f in vung["features"] if f.get("bac", -1) >= 0]
    n = (max(bac_co) + 1) if bac_co else vung["so_bac"]
    moc = vung.get("moc")
    if not moc:
        moc = list(np.linspace(vung["gia_min"], vung["gia_max"], n + 1)[1:-1])

    o = "".join(
        f'<div style="flex:1;height:11px;background:rgb{tuple(_mau_bac(i, n))};'
        f'{"border-radius:5px 0 0 5px;" if i == 0 else ""}'
        f'{"border-radius:0 5px 5px 0;" if i == n - 1 else ""}"></div>'
        for i in range(n))
    nhan = "".join(
        f'<div style="flex:1;text-align:right;transform:translateX(50%);">'
        f'{m:,.0f}</div>' for m in moc) + '<div style="flex:1;"></div>'

    st.markdown(
        f"""
        <div style="margin:2px 0 10px;">
          <div style="display:flex;gap:2px;">{o}</div>
          <div style="display:flex;gap:2px;font-size:11px;
                      color:{COLOR['text_muted']};margin-top:3px;">{nhan}</div>
          <div style="display:flex;align-items:center;gap:8px;margin-top:8px;
                      font-size:11px;color:{COLOR['text_muted']};">
            <!-- Ô "chưa đủ dữ liệu" có VIỀN NÉT ĐỨT, không chỉ khác màu. Nó
                 tách khỏi bậc nhạt nhất 2,06:1 — đủ thấy, nhưng đây là chỗ
                 tuyệt đối không được để người xem đoán, vì đọc nhầm nó thành
                 "rẻ nhất" là biến chỗ thiếu số liệu thành một mức giá. -->
            <div style="width:16px;height:11px;border-radius:3px;
                        background:rgb{tuple(MAU_THIEU)};
                        border:1px dashed {COLOR['border_manh']};"></div>
            <span>chưa đủ dữ liệu</span>
            <span style="margin-left:auto;">đơn vị: triệu/m² · các bậc chia
              theo phân vị nên khoảng cách không đều nhau</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True)



def _bang_phuong(loai: str) -> None:
    """Bảng xếp hạng mặt bằng giá theo phường — phần chính của trang này."""
    g = nap_phuong(loai)
    tv_tp = g.attrs.get("trung_vi_thanh_pho", float(g["gia_m2"].median()))

    c1, c2, c3 = st.columns([1.1, 1.1, 0.8])
    quan_list = sorted(x for x in g["quan_huyen"].dropna().unique())
    loc_quan = c1.multiselect("Lọc theo Quận / Huyện", quan_list,
                              placeholder="Tất cả", key=f"bp_quan_{loai}")
    tim = c2.text_input("Tìm tên phường", placeholder="ví dụ: Yên Hoà",
                        key=f"bp_tim_{loai}") or ""
    sap = c3.selectbox("Sắp xếp", ["Đắt nhất trước", "Rẻ nhất trước",
                                   "Nhiều căn nhất trước"],
                       key=f"bp_sap_{loai}")

    d = g
    if loc_quan:
        d = d[d["quan_huyen"].isin(loc_quan)]
    if tim.strip():
        d = d[d["phuong_moi"].str.lower().str.contains(tim.strip().lower())]
    d = d.sort_values(
        {"Đắt nhất trước": "gia_m2", "Rẻ nhất trước": "gia_m2",
         "Nhiều căn nhất trước": "so_tin"}[sap],
        ascending=(sap == "Rẻ nhất trước"))

    if d.empty:
        st.info("Không có phường nào khớp. Thử bỏ bớt bộ lọc.")
        return

    bang = pd.DataFrame({
        "#": d["hang"],
        "Phường / Xã": d["phuong_moi"],
        "Quận / Huyện": d["quan_huyen"],
        "Giá/m² (triệu)": (d["gia_m2"] / 1e6).round(1),
        "So với toàn TP": d["so_voi_tp"] / 100,
        "Phần lớn nằm trong khoảng": ((d["q25"] / 1e6).round(0).astype(int).astype(str)
                                 + " – "
                                 + (d["q75"] / 1e6).round(0).astype(int).astype(str)
                                 + " triệu"),
        "Số căn": d["so_tin"],
    })
    st.dataframe(
        bang, hide_index=True, width='stretch', height=420,
        column_config={
            # Thanh ngang cho cột giá: mắt so sánh độ dài nhanh hơn nhiều so
            # với đọc và trừ hai con số.
            "Giá/m² (triệu)": st.column_config.ProgressColumn(
                "Giá/m² (triệu)", format="%.1f", min_value=0.0,
                max_value=float((d["gia_m2"] / 1e6).max())),
            "So với toàn TP": st.column_config.NumberColumn(
                "So với toàn TP", format="percent",
                help=f"Trung vị toàn thành phố: {tv_tp/1e6:.1f} triệu/m²"),
            "Số căn": st.column_config.NumberColumn(
                "Số căn", help="Số căn đang rao bán làm căn cứ. Phường ít căn "
                               "thì con số kém chắc chắn hơn."),
        })
    st.caption(
        f"Trung vị toàn thành phố **{tv_tp/1e6:.1f} triệu/m²**. Bảng chỉ liệt "
        f"kê phường có từ 10 căn trở lên — {g.attrs.get('so_phuong_bi_loai', 0)} "
        f"phường khác có mặt trong dữ liệu nhưng chưa đủ căn để đưa ra một "
        f"con số. Cột \"phần lớn nằm trong khoảng\" mới là thứ nói lên độ chắc: "
        f"khoảng càng rộng thì riêng tên phường càng không đủ đoán giá.")


def _so_sanh_phuong(loai: str) -> None:
    """Đặt hai phường cạnh nhau và trả lời đúng một câu: chỗ nào đắt hơn.

    VÌ SAO BẢN TRƯỚC KHÓ HIỂU
    -------------------------
    Bản trước trả về câu này khi hai khoảng giá chồng nhau:

        "A có trung vị cao hơn B 8%, nhưng hai khoảng giá chồng lên nhau —
         chọn ngẫu nhiên một căn ở mỗi bên thì rất có thể căn ở phường rẻ hơn
         lại đắt hơn. Chênh lệch này chưa đủ chắc để dựa vào."

    Đó là một bài kiểm định giả thuyết viết bằng tiếng Việt. Người dùng hỏi
    "phường nào đắt hơn" và nhận về một bài giảng về phương sai.

    Nhưng KẾT LUẬN thì đúng và không được bỏ: nếu hai phường thật sự không phân
    biệt được thì tuyên bố "đắt hơn 8%" là bịa ra độ chắc chắn không có. Nên
    giữ nguyên phép so sánh, chỉ đổi cách nói — một câu, không giảng bài.
    """
    g = nap_phuong(loai)
    ten = list(g["phuong_moi"])
    c1, c2 = st.columns(2)
    a = c1.selectbox("Phường thứ nhất", ten, index=0, key=f"ss_a_{loai}")
    b = c2.selectbox("Phường thứ hai", ten,
                     index=min(1, len(ten) - 1), key=f"ss_b_{loai}")
    ra = g[g["phuong_moi"] == a].iloc[0]
    rb = g[g["phuong_moi"] == b].iloc[0]

    for col, r in ((c1, ra), (c2, rb)):
        col.metric(r["phuong_moi"], f"{r['gia_m2']/1e6:.1f} triệu/m²",
                   f"{r['so_voi_tp']:+.0f}% so với mặt bằng Hà Nội")
        col.caption(
            f"Đắt thứ **{int(r['hang'])}** trong {len(g)} phường có dữ liệu · "
            f"phần lớn từ {r['q25']/1e6:.0f} đến {r['q75']/1e6:.0f} triệu/m² · "
            f"dựa trên {int(r['so_tin'])} căn")

    chenh = (ra["gia_m2"] / rb["gia_m2"] - 1) * 100
    # Hai khoảng giá phổ biến chồng nhau thì chênh lệch KHÔNG đáng tin: một căn
    # bất kỳ ở phường "rẻ hơn" vẫn rất dễ đắt hơn một căn ở phường kia.
    chong = not (ra["q25"] > rb["q75"] or rb["q25"] > ra["q75"])
    cao, thap, ty = ((a, b, chenh) if chenh > 0 else (b, a, -chenh))

    if abs(chenh) < 1:
        st.info(f"**{a}** và **{b}** cùng một mặt bằng giá.")
    elif chong:
        st.warning(
            f"**{a}** và **{b}** coi như ngang nhau. Trên giấy tờ {cao} nhỉnh "
            f"hơn {ty:.0f}%, nhưng khoảng giá của hai phường đè lên nhau — "
            f"chênh lệch giữa các căn trong cùng một phường còn lớn hơn chênh "
            f"lệch giữa hai phường. Chọn theo tiêu chí khác thì hợp lý hơn.")
    else:
        st.success(
            f"**{cao}** đắt hơn **{thap}** khoảng **{ty:.0f}%** mỗi m². "
            f"Khoảng giá của hai phường tách hẳn nhau, nên chênh lệch này là "
            f"thật chứ không phải ngẫu nhiên.")


def render_ban_do() -> None:
    import pydeck as pdk

    loai = st.radio(
        "Loại bất động sản", ["nhadat", "chungcu"],
        format_func=lambda x: "Nhà đất (giá/m² đất)" if x == "nhadat"
        else "Chung cư (giá/m² sàn)",
        horizontal=True, key="bd_loai")
    st.caption("Giá/m² đất và giá/m² sàn là hai đại lượng khác nhau nên tách "
               "riêng.")

    # TÌM QUANH MỘT ĐIỂM LÊN ĐẦU TIÊN.
    #
    # Trước đây nó nằm cuối trang, dưới bốn khối thống kê. Nhưng đây là khối
    # duy nhất người dùng TƯƠNG TÁC được — trỏ vào một chỗ và xem quanh đó có
    # gì. Bốn khối kia là thông tin nền, đọc lúc nào cũng được. Đặt thứ tương
    # tác được xuống cuối trang là bắt người dùng cuộn qua hết phần nền mới
    # tới được việc họ muốn làm.
    with st.container(border=True):
        section_header(ICON_PIN, "Quanh đây có gì đang bán",
                       "Gõ địa chỉ hoặc bấm lên bản đồ")
        _khoi_ban_do(loai)

    with st.container(border=True):
        section_header(ICON_PIN, "Toàn bộ căn đang bán trên bản đồ",
                       "Bấm một điểm để xem chi tiết")
        _ban_do_ghim(loai, pdk)

    with st.container(border=True):
        section_header(ICON_MAP, "Bản đồ theo ranh giới phường",
                       "Ranh giới hành chính thật, sau sáp nhập 1/7/2025")
        _choropleth(loai, pdk)

    with st.container(border=True):
        section_header(ICON_CHART, "Mặt bằng giá theo phường",
                       "Xếp hạng 113 phường sau sáp nhập 1/7/2025")
        _bang_phuong(loai)

    with st.container(border=True):
        section_header(ICON_CHART, "So sánh hai phường",
                       "Chỗ nào đắt hơn, và hơn bao nhiêu")
        _so_sanh_phuong(loai)


def _ban_do_ghim(loai: str, pdk) -> None:
    """Mỗi căn một điểm ghim, bấm vào thì hiện chi tiết — như các sàn bất động sản.

    Khác choropleth ở câu hỏi nó trả lời: choropleth nói "khu này mặt bằng bao
    nhiêu", bản đồ ghim nói "quanh đây đang có gì bán".
    """
    d = nap_ghim(loai)
    if d.empty:
        st.info("Không có căn nào để hiển thị.")
        return

    c1, c2, c3 = st.columns([1.1, 1.3, 1.0])
    quan_list = sorted(x for x in d["quan_huyen"].dropna().unique())
    quan = c1.selectbox("Quận / Huyện", ["Toàn thành phố"] + quan_list,
                        key=f"gh_quan_{loai}")
    gmin, gmax = float(d["gia_ty"].min()), float(d["gia_ty"].quantile(.99))
    khoang = c2.slider("Khoảng giá (tỷ)", gmin, gmax, (gmin, gmax), 0.5,
                       key=f"gh_gia_{loai}")
    chi_chac = c3.checkbox(
        "Chỉ hiện căn có vị trí chính xác", value=False, key=f"gh_chac_{loai}",
        help="Chỉ giữ căn có toạ độ do nơi đăng công bố hoặc khớp đúng dự án. "
             "Những căn còn lại được đặt theo tên đường/phường, lệch khoảng "
             "khoảng 0,67–0,79 km — ghim của chúng không chỉ đúng căn nhà.")

    x = d
    if quan != "Toàn thành phố":
        x = x[x["quan_huyen"] == quan]
    x = x[(x["gia_ty"] >= khoang[0]) & (x["gia_ty"] <= khoang[1])]
    if chi_chac:
        x = x[x["chinh_xac"]]
    if x.empty:
        st.info("Không có căn nào khớp bộ lọc. Thử nới khoảng giá.")
        return

    x = x.reset_index(drop=True)

    # MẬT ĐỘ CHẤM ĐI THEO PHẠM VI ĐANG XEM.
    #
    # Streamlit không trả khung nhìn của pydeck về Python, nên không biết người
    # dùng đang phóng to ở mức nào để bớt chấm cho đúng kiểu LOD. Thứ thay thế
    # gần nhất: lấy chính phạm vi họ chọn làm mức chi tiết. Toàn thành phố thì
    # lưới 0,8 km (12.294 chấm còn ~600); chọn một quận thì lưới 0,25 km; lọc
    # hẹp dưới 400 tin thì hiện hết.
    n_goc = len(x)
    if n_goc <= 400:
        o_luoi = 0.0
    elif quan == "Toàn thành phố":
        o_luoi = 0.8
    else:
        o_luoi = 0.25
    x = bd.thua_theo_luoi(x, o_luoi)

    lo, hi = x["gia_m2_trieu"].quantile(.10), x["gia_m2_trieu"].quantile(.90)
    t = ((x["gia_m2_trieu"] - lo) / max(hi - lo, 1e-9)).clip(0, 1)
    # TIN CÓ VỊ TRÍ ƯỚC LƯỢNG VẼ THÀNH VÒNG RỖNG, KHÔNG PHẢI CHẤM MỜ.
    #
    # Bản nền tối phân biệt hai loại bằng độ mờ: 235 so với 110. Trên nền sáng
    # cách đó vỡ — chấm xanh nhạt ở alpha 110 trên nền gần trắng thì biến mất
    # hẳn, nên tin vị trí ước lượng không phải "mờ hơn" mà là "không thấy".
    # Đổi sang khác HÌNH: chấm đặc viền trắng = biết đúng chỗ, vòng rỗng viền
    # màu = chỉ biết tới phường. Hình thì nhạt hay đậm vẫn đọc được, và người
    # mù màu cũng phân biệt được.
    n_mau = len(THANG_MAU) - 1
    goc = [list(THANG_MAU[min(int(v * n_mau), n_mau)]) for v in t]
    x["mau"] = [g + [235] if c else g + [0]
                for g, c in zip(goc, x["chinh_xac"])]
    x["vien"] = [[255, 255, 255, 255] if c else g + [255]
                 for g, c in zip(goc, x["chinh_xac"])]
    x["ban_kinh"] = np.where(x["chinh_xac"], 60, 45)

    tam = (x["latitude"].median(), x["longitude"].median())
    lop = pdk.Layer(
        "ScatterplotLayer", x[["latitude", "longitude", "mau", "vien",
                               "ban_kinh", "nhan"]],
        get_position=["longitude", "latitude"],
        get_fill_color="mau", get_radius="ban_kinh",
        radius_min_pixels=3, radius_max_pixels=14,
        pickable=True, auto_highlight=True,
        # VIỀN LẤY THEO TỪNG CHẤM, KHÔNG CỐ ĐỊNH MỘT MÀU.
        #
        # Trên nền tối, chấm màu tự nổi nên viền chỉ cần mờ 120/255, dày 0,5px.
        # Trên nền sáng thì bậc nhạt của thang gần như tan vào nền bản đồ, nên
        # viền phải đặc và dày hơn. Và vì tin vị trí ước lượng giờ vẽ thành
        # vòng rỗng, màu viền phải đi theo từng chấm (xem ghi chú ở `x["vien"]`
        # bên trên) chứ không thể để một giá trị chung.
        filled=True, stroked=True, get_line_color="vien",
        line_width_min_pixels=1.5)

    su_kien = st.pydeck_chart(
        pdk.Deck(layers=[lop], map_style=MAP_STYLE,
                 initial_view_state=pdk.ViewState(
                     latitude=tam[0], longitude=tam[1],
                     zoom=11 if quan == "Toàn thành phố" else 13, pitch=0),
                 tooltip={"text": "{nhan}"}),
        width='stretch', on_select="rerun", selection_mode="single-object",
        key=f"gh_map_{loai}")

    n_chac = int(x["chinh_xac"].sum())
    if o_luoi > 0:
        st.caption(
            f"Đang hiện **{_so(len(x))} chấm đại diện** cho {_so(n_goc)} căn — "
            f"mỗi ô lưới {o_luoi} km giữ một điểm đông căn nhất, để bản đồ "
            f"đỡ rối. **Chọn một quận** ở trên để hiện dày hơn, hoặc thu hẹp "
            f"khoảng giá xuống dưới 400 căn để hiện hết.")
    else:
        st.caption(f"Đang hiện đủ **{_so(len(x))} căn**.")
    st.caption(
        f"Chấm đậm viền rõ là **{_so(n_chac)} căn có vị trí chính xác** (nơi "
        f"đăng tự công bố); chấm mờ là căn đặt theo tên đường hoặc phường, lệch "
        f"khoảng 0,67–0,79 km — ghim chỉ đúng khu vực, không đúng căn nhà.")

    _hien_tin_da_chon(su_kien, x)


def _hien_tin_da_chon(su_kien, x: pd.DataFrame) -> None:
    """Hiện căn ở điểm vừa bấm — và MỌI căn khác dùng chung điểm ghim đó.

    deck.gl chỉ trả về một đối tượng khi bấm, nhưng có điểm ghim mang tới 144
    tin chồng lên nhau (cả một con phố dùng chung một toạ độ mức đường). Chỉ
    hiện đúng tin trên cùng thì người dùng tưởng quanh đó có mỗi thế.
    """
    chon = None
    try:
        doi_tuong = su_kien.selection["objects"]
        for ds in doi_tuong.values():
            if ds:
                chon = ds[0]
                break
    except Exception:
        chon = None

    if not chon:
        st.caption("Bấm vào một điểm trên bản đồ để xem chi tiết.")
        return

    vi, kinh = chon.get("latitude"), chon.get("longitude")
    if vi is None:
        return
    cung = x[(x["latitude"].round(4) == round(float(vi), 4))
             & (x["longitude"].round(4) == round(float(kinh), 4))]
    cung = cung.sort_values("gia_ty")

    if len(cung) > 1:
        st.warning(
            f"**{len(cung)} căn cùng một điểm ghim.** Chúng dùng chung toạ độ "
            f"vì được đặt theo tên đường, không phải vị trí thật của từng căn.")

    # iterrows chứ không itertuples: itertuples đổi tên cột có dấu cách
    # ("Giấy tờ pháp lý" → "_9"), nên khối nguồn không đọc được pháp lý và
    # nội thất. Series giữ đúng tên cột, và `r.ten_hien` vẫn dùng được.
    for _, r in cung.head(12).iterrows():
        with st.container(border=True):
            a, b = st.columns([3, 1])
            a.markdown(f"**{r.ten_hien}**")
            a.caption(
                f"{_so(r.dien_tich)} m² · {r.gia_m2_trieu} triệu/m² · "
                f"{r.phuong_moi or ''} · {r.quan_huyen or ''}")
            b.metric("Giá rao", f"{r.gia_ty} tỷ")
            _khoi_nguon(a, r)
            if not r.chinh_xac:
                a.caption("⚠️ Vị trí trên bản đồ là ước lượng theo tên đường "
                          "hoặc phường, không phải toạ độ thật của căn này.")
    if len(cung) > 12:
        st.caption(f"… và {len(cung) - 12} căn nữa ở cùng điểm ghim.")


def _choropleth(loai: str, pdk) -> None:
    """Tô từng phường theo giá, dùng ranh giới hành chính thật.

    Ranh giới lấy từ Bản đồ hành chính Việt Nam (bản 2025, sau sáp nhập), bóc
    ra bằng `ranh_gioi.py`. Nếu chưa chạy bước đó thì báo cho người dùng biết
    phải làm gì, thay vì để ứng dụng vỡ.
    """
    try:
        v = nap_vung_phuong(loai)
    except FileNotFoundError:
        st.info("Chưa có file ranh giới phường. Chạy `python ranh_gioi.py` "
                "trong thư mục `pipeline` để tạo, rồi khởi động lại ứng dụng.")
        return
    if not v["features"]:
        st.info("Không đủ dữ liệu để tô màu phường nào.")
        return

    n = v["so_bac"]
    gj = {"type": "FeatureCollection", "features": [
        {**f, "properties": {**f["properties"],
                             "mau": _mau_bac(f["bac"], n)}}
        for f in v["features"]]}

    lop = pdk.Layer(
        "GeoJsonLayer", gj, filled=True, stroked=True, pickable=True,
        get_fill_color="properties.mau",
        # VIỀN TRẮNG ĐẶC, KHÔNG PHẢI 150/255 NHƯ BẢN NỀN TỐI.
        #
        # Đây là thứ gánh cho chỗ yếu duy nhất còn lại của thang: bậc nhạt
        # nhất chỉ nổi trên nền bản đồ 2,01:1, tức là vừa đủ mốc chứ không
        # dư. Có viền trắng đặc thì hình của phường vẫn đọc được kể cả khi
        # phần tô nhạt — hình thay màu làm lớp nhận dạng thứ hai.
        get_line_color=[255, 255, 255, 255],
        line_width_min_pixels=1.4, opacity=DUC_VUNG,
        auto_highlight=True, highlight_color=[43, 76, 140, 90])

    st.pydeck_chart(pdk.Deck(
        layers=[lop], map_style=MAP_STYLE,
        initial_view_state=pdk.ViewState(
            latitude=vs.TRUNG_TAM[0], longitude=vs.TRUNG_TAM[1],
            zoom=9.2, pitch=0),
        tooltip={"text": "{nhan}"},
    ), width='stretch')

    _chu_giai_vung(v)
    st.caption(
        f"Tô màu **{v['so_vung_du_tin']}/{v['so_vung']}** phường/xã. Phần xám "
        f"là phường chưa đủ 5 căn — vẫn vẽ ra để hình Hà Nội liền mạch, chứ "
        f"không bỏ trống, vì trên nền tối thì chỗ trống trông y hệt phần không "
        f"thuộc thành phố. Rê chuột lên một phường để xem giá trung vị, khoảng "
        f"khoảng giá phổ biến, và số căn làm căn cứ.")
    st.caption(
        "Mỗi phường được tô theo những căn **nằm trong ranh giới của nó**, "
        "không phải theo tên phường ghi trong nội dung rao. Ở nội thành, nơi "
        "phường chỉ rộng chưa tới 1 km mà vị trí thường lệch khoảng 0,7 km, một "
        "phần có thể rơi sang phường bên cạnh — nên luôn đọc giá cùng với số căn.")


def _ghim_tu_do(loai: str, khoa: str) -> tuple:
    """Chọn một điểm bất kỳ ở Hà Nội: gõ địa chỉ, hoặc bấm lên bản đồ.

    Khác `_ghim_ban_do` ở chỗ KHÔNG đòi chọn quận/phường trước. Ở form định giá
    thì bắt buộc phải biết phường (mô hình cần nó làm đặc trưng), còn ở đây
    người dùng chỉ muốn trỏ vào một chỗ trên bản đồ và xem quanh đó có gì —
    hỏi họ quận trước là đặt một cửa ải không cần thiết.

    Trả về (lat, lon, mô tả ngắn) hoặc (None, None, "").
    """
    try:
        import folium
        from streamlit_folium import st_folium
    except ImportError:
        st.info("Cách ghim bản đồ cần `streamlit-folium`. Cài bằng "
                "`pip install streamlit-folium folium` rồi khởi động lại.")
        return None, None, ""

    k_ghim, k_tam, k_ten = f"{khoa}_ghim", f"{khoa}_tam", f"{khoa}_ten"

    c1, c2 = st.columns([3, 1])
    dia_chi = c1.text_input(
        "Địa chỉ", placeholder="Gõ địa chỉ, ví dụ: Phố Bùi Xương Trạch",
        key=f"{khoa}_dc", label_visibility="collapsed") or ""
    if c2.button("Tìm", key=f"{khoa}_tim", width='stretch') and dia_chi.strip():
        hit = vs.tim_duong_toan_thanh(loai, dia_chi)
        if hit:
            h = hit[0]
            st.session_state[k_tam] = [h["latitude"], h["longitude"]]
            st.session_state[k_ghim] = [h["latitude"], h["longitude"]]
            st.session_state[k_ten] = (f"{h['duong_pho']}, {h['phuong_moi']}")
        else:
            r = _tim_dia_chi_osm(dia_chi, None, None)
            if r:
                st.session_state[k_tam] = [r[0], r[1]]
                st.session_state[k_ghim] = [r[0], r[1]]
                st.session_state[k_ten] = r[2]
            else:
                st.warning("Không tra được địa chỉ này. Bấm thẳng lên bản đồ "
                           "để chọn vị trí.")
    return (st.session_state.get(k_ghim) or [None, None])[0], \
           (st.session_state.get(k_ghim) or [None, None])[1], \
           st.session_state.get(k_ten) or ""


def _khoi_ban_do(loai: str, pdk=None) -> None:
    """Tìm quanh một điểm: trỏ vào một chỗ, xem những căn đang bán quanh đó.

    BA THAY ĐỔI SO VỚI BẢN TRƯỚC, đều theo đúng chỗ người dùng vướng:

    1. CHỌN ĐIỂM BẰNG ĐỊA CHỈ HOẶC BẤM BẢN ĐỒ. Trước đây chỉ có hai ô chọn
       Quận rồi Phường, nên điểm tìm luôn là TÂM PHƯỜNG — cách vị trí thật
       trung vị 0,75 km. Muốn xem quanh nhà mình thì không có cách nào.

    2. MỖI CĂN LÀ MỘT GHIM BẤM ĐƯỢC, mở ra bảng thông tin kèm LINK. Trước đây
       vẽ bằng pydeck nên bấm vào chấm không ra gì. Folium cho đặt popup HTML
       lên từng ghim, nên link tin gốc nằm ngay trong đó.

    3. BỎ CỘT "NGUỒN TOẠ ĐỘ" trong bảng, thay bằng link. Cột đó là thuật ngữ
       nội bộ của pipeline, người đi mua nhà không có việc gì với nó.

    ĐÃ BỎ: LỚP "BỀ MẶT GIÁ LIÊN TỤC"
    --------------------------------
    Khối này trước đây còn vẽ một mặt giá làm mượt phủ kín Hà Nội. Bỏ vì nó vẽ
    ra thứ dữ liệu không có: mặt giá được nội suy từ toạ độ có sai số p90
    khoảng 0,67–0,79 km, nên người xem đọc được "chỗ này đắt hơn chỗ cách
    300 m" — một kết luận dữ liệu không cho phép. Phần tìm quanh một điểm
    không có vấn đề đó: nó chỉ liệt kê những căn thật trong bán kính.
    """
    try:
        import folium
        from streamlit_folium import st_folium
    except ImportError:
        st.info("Phần này cần `streamlit-folium`. Cài bằng "
                "`pip install streamlit-folium folium` rồi khởi động lại.")
        return

    khoa = f"bd_{loai}"
    vi, kinh, ten = _ghim_tu_do(loai, khoa)

    c1, c2, c3 = st.columns([1.2, 1.4, 1.4])
    cay = nap_cay(loai)
    quan = c1.selectbox("Hoặc chọn nhanh theo quận", sorted(cay["cay"].keys()),
                        index=None, placeholder="Chọn quận", key=f"{khoa}_quan")
    if quan and vi is None:
        g = vs.bo_sung_toa_do(loai, {"quan_huyen": quan})
        vi, kinh = g.get("latitude"), g.get("longitude")
        ten = ten or quan
    ban_kinh = c2.slider("Bán kính (km)", 0.3, 5.0, 1.5, 0.1, key=f"{khoa}_bk")
    chi_chac = c3.checkbox(
        "Chỉ căn biết vị trí tới mức đường phố", value=False, key=f"{khoa}_cc",
        help="Nên bật khi bán kính dưới 1 km: căn chỉ biết tên phường lệch "
             "khoảng 0,8 km nên dễ bị xếp nhầm chỗ.")

    if vi is None:
        vi, kinh = vs.TRUNG_TAM
        ten = ten or "Hồ Hoàn Kiếm"
        chua_chon = True
    else:
        chua_chon = False

    quanh = bd.tim_quanh((vi, kinh), ban_kinh, loai=loai,
                         chi_toa_do_chac=chi_chac)

    m = folium.Map(location=[vi, kinh], zoom_start=15 if not chua_chon else 12,
                   tiles=None, control_scale=True)
    _nen_ban_do(m, folium)
    folium.Circle([vi, kinh], radius=ban_kinh * 1000,
                  color=COLOR["accent"], weight=2, fill=True,
                  fill_color=COLOR["accent"], fill_opacity=0.05,
                  dash_array="7 5").add_to(m)
    folium.Marker([vi, kinh], tooltip="Điểm bạn đang xem quanh",
                  icon=folium.Icon(color="darkblue", icon="screenshot")).add_to(m)

    # GHIM TỪNG CĂN, POPUP CÓ LINK. Chỉ vẽ 150 căn gần nhất: folium dựng ghim
    # bằng HTML thật nên vài trăm ghim là trang đứng hình.
    for _, r in quanh.head(150).iterrows():
        u = r.get("url_goc")
        tuoi = _tuoi_tin(r)
        if isinstance(u, str) and u.startswith("http") and \
                (tuoi is None or tuoi <= HAN_LINK_CU):
            lk = (f'<br><a href="{u}" target="_blank" '
                  f'rel="noopener">Mở tin gốc ↗</a>')
        else:
            lk = '<br><span style="color:#888">không còn link tin gốc</span>'
        pop = (f'<b>{_sole(r["TARGET_gia_vnd"]/1e9, 2)} tỷ</b> · '
               f'{_so(r["dien_tich"])} m²<br>'
               f'{r["gia_m2_trieu"] if "gia_m2_trieu" in r else ""}'
               f'{_sole(r["gia_tren_m2"]/1e6, 1)} triệu/m²<br>'
               f'{r.get("duong_pho") or ""} · {r.get("phuong_moi") or ""}<br>'
               f'cách {r["khoang_cach_km"]:.2f} km{lk}')
        folium.CircleMarker(
            [r["latitude"], r["longitude"]], radius=6,
            # Viền TRẮNG quanh mỗi chấm: trên nền bản đồ sáng, chấm không có
            # viền thì chìm vào nền ở những khu nhạt màu.
            color="#FFFFFF", weight=2, fill=True,
            fill_color=COLOR["accent"], fill_opacity=0.95,
            popup=folium.Popup(pop, max_width=260),
            tooltip=f'{_sole(r["TARGET_gia_vnd"]/1e9, 2)} tỷ · '
                    f'{_so(r["dien_tich"])} m²').add_to(m)

    # CHỈ nhận `last_clicked` (bấm vào nền bản đồ), KHÔNG nhận
    # `last_object_clicked`. Nhờ vậy bấm vào một ghim căn nhà chỉ mở popup của
    # nó, không kéo điểm tìm kiếm sang đó — nếu nhận cả hai thì mỗi lần xem
    # một căn là mất luôn vị trí đang tìm.
    kq_map = st_folium(m, height=440, width=None,
                       returned_objects=["last_clicked"], key=f"{khoa}_map")
    bam = (kq_map or {}).get("last_clicked")
    if bam:
        st.session_state[f"{khoa}_ghim"] = [bam["lat"], bam["lng"]]
        st.session_state[f"{khoa}_ten"] = ""
        st.rerun()

    if chua_chon:
        st.caption("**Bấm lên bản đồ** hoặc gõ địa chỉ ở trên để chọn điểm. "
                   "Đang tạm lấy Hồ Hoàn Kiếm.")
    else:
        st.caption(f"Đang xem quanh **{ten or f'{vi:.5f}, {kinh:.5f}'}** · "
                   f"bấm chỗ khác trên bản đồ để đổi điểm · bấm một chấm vàng "
                   f"để xem căn đó.")

    if quanh.empty:
        st.info(f"Không có căn nào trong bán kính {ban_kinh} km. "
                "Thử nới bán kính ra.")
        return

    o = bd.gia_khu_vuc((vi, kinh), nap_luoi(loai))
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Số căn quanh đây", _so(len(quanh)))
    m2.metric("Giá mỗi m² thường gặp",
              f"{quanh['gia_tren_m2'].median()/1e6:,.1f} triệu")
    m3.metric("Giá cả căn thường gặp",
              format_ty(quanh["TARGET_gia_vnd"].median()))
    m4.metric("Diện tích thường gặp",
              f"{quanh['dien_tich'].median():,.0f} m²")
    if o and not o["du_tin_cay"]:
        st.caption(f"⚠️ Chỉ {o['so_tin']} căn quanh đây nên các con số trên "
                   f"chưa chắc.")

    hien = pd.DataFrame({
        "Cách": quanh["khoang_cach_km"].round(2).astype(str) + " km",
        "Giá rao": quanh["TARGET_gia_vnd"].map(format_ty),
        "Giá/m²": (quanh["gia_tren_m2"] / 1e6).round(1).astype(str) + " tr",
        "Diện tích": quanh["dien_tich"].astype(str) + " m²",
        "Phường": quanh["phuong_moi"],
        "Đường": quanh["duong_pho"],
    })
    # LINK thay cho cột "Nguồn toạ độ". Ẩn link đã quá cũ, vì tin rao bị sàn
    # xoá sau khoảng 30 ngày — dán một link chết vào bảng còn tệ hơn bỏ trống.
    if "url_goc" in quanh.columns:
        u = quanh["url_goc"].where(
            quanh["url_goc"].astype(str).str.startswith("http"))
        if "ngay_quan_sat" in quanh.columns:
            tuoi = (pd.Timestamp.today().normalize()
                    - pd.to_datetime(quanh["ngay_quan_sat"],
                                     errors="coerce")).dt.days
            u = u.where(tuoi.isna() | (tuoi <= HAN_LINK_CU))
        hien["Tin gốc"] = u
        st.dataframe(hien.head(200), hide_index=True, width='stretch',
                     column_config={"Tin gốc": st.column_config.LinkColumn(
                         "Tin gốc", display_text="mở")})
    else:
        st.dataframe(hien.head(200), hide_index=True, width='stretch')


# =============================================================================
# TRANG SĂN TIN
# =============================================================================

def _so(n) -> str:
    """Số nguyên với dấu chấm phân cách hàng nghìn.

    Trước đây tôi viết `f"...{n:,}...".replace(",", ".")` cho cả câu — và nó
    đổi luôn mọi dấu phẩy TRONG CÂU VĂN thành dấu chấm, biến "đông tin nhất,
    để bản đồ đỡ rối" thành "...nhất. để bản đồ đỡ rối". Định dạng số phải làm
    riêng cho từng số, không quét cả chuỗi.
    """
    try:
        return f"{int(round(float(n))):,}".replace(",", ".")
    except (TypeError, ValueError):
        return "—"          # None/NaN/chuỗi lạ: không được làm vỡ cả trang


def _sole(x, le: int = 1) -> str:
    """Số THẬP PHÂN kiểu Việt Nam: nghìn ngăn bằng ".", thập phân bằng ",".

    Tách khỏi `_so()` vì `_so()` gọi `int(n)` — đưa "8.4" vào là vỡ ngay
    (`ValueError: invalid literal for int() with base 10: '8.4'`). Tôi đã mắc
    đúng lỗi đó khi viết bảng what-if, và bản thử nghiệm không bắt được vì ở
    đó tôi thay `_so` bằng một hàm giả chỉ đổi "." thành ",". Bài học: khi
    chạy thử một khối giao diện tách rời, phải dùng CHÍNH các hàm thật của
    ứng dụng, không dựng hàm giả cho tiện.
    """
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    return f"{v:,.{le}f}".replace(",", "\x00").replace(".", ",").replace("\x00", ".")


# =============================================================================
# NGUỒN CỦA MỘT CĂN — bằng chứng nằm TRONG ứng dụng, không nằm trên sàn
# =============================================================================
#
# VÌ SAO KHÔNG ĐƯỢC DỰA VÀO LINK
# ------------------------------
# Tin rao có tuổi thọ khoảng 30 ngày rồi sàn xoá. Đo thật ngày 12/09/2026,
# mở 11 link lấy ngẫu nhiên từ chính dữ liệu này:
#
#     nhatot nhà đất  (cào 18/06)   0/5 còn sống — 4 trang lỗi trắng, 1 hết hạn
#     nhatot chung cư (cào 25/08)   2/5 còn sống — 2 hết hạn, 1 lỗi
#     batdongsan      (cào 04/09)   3/3 còn sống — NHƯNG chính trang ghi
#                                   "Ngày hết hạn 13/09 / 14/09 / 16/09"
#
# Tức là đây KHÔNG phải lỗi riêng của một sàn, và cũng không chữa được bằng
# cách cào lại: những link batdongsan hôm nay còn sống sẽ chết trong vài ngày.
# Nên bằng chứng phải là NỘI DUNG TIN ĐÃ LƯU, hiện ngay trong ứng dụng; link
# chỉ là thứ yếu — còn thì tốt, chết thì không mất gì.
#
# Ngoài ra 1.602 tin chung cư nhatot không hề có link: đợt cào 13/06 không lấy
# cột địa chỉ trang. Không cứu được từ dữ liệu đang có, nên phải nói thẳng ra
# chứ không để ô trống cho người dùng tự đoán.

HAN_LINK_MOI = 30      # ngày — trong khoảng này link gần như còn sống
HAN_LINK_CU = 60       # ngày — quá mốc này thì hạ link xuống mục thu gọn

TEN_NGUON = {"nhatot": "Nhà Tốt", "batdongsan": "Batdongsan.com.vn"}


def _lay(r, ten, mac_dinh=None):
    """Đọc một trường của hàng, chịu được cả thiếu cột lẫn NaN.

    Dùng `r.get(ten)` trực tiếp là hỏng ở hai chỗ: cột không tồn tại (nhánh
    chung cư không có `tieu_de`) và cột tồn tại nhưng giá trị là NaN — NaN lọt
    xuống giao diện thành chữ "nan" hiện lên cho người dùng đọc.
    """
    try:
        v = r.get(ten, mac_dinh)
    except Exception:
        v = getattr(r, ten, mac_dinh)
    if v is None:
        return mac_dinh
    try:
        if not isinstance(v, str) and pd.isna(v):
            return mac_dinh
    except (TypeError, ValueError):
        pass
    if isinstance(v, str) and (not v.strip() or v.strip().lower() == "nan"):
        return mac_dinh
    return v


def _tuoi_tin(r) -> int | None:
    """Số ngày từ lúc lưu tin tới hôm nay. None nếu không biết ngày."""
    ngay = _lay(r, "ngay_quan_sat")
    if ngay is None:
        return None
    try:
        t = pd.Timestamp(ngay)
    except (TypeError, ValueError):
        return None
    if pd.isna(t):
        return None
    return int((pd.Timestamp.today().normalize() - t.normalize()).days)


def _ngay_vn(r) -> str:
    ngay = _lay(r, "ngay_quan_sat")
    try:
        return pd.Timestamp(ngay).strftime("%d/%m/%Y")
    except (TypeError, ValueError):
        return "—"


def _ten_nguon(r) -> str:
    n = str(_lay(r, "nguon", "") or "").strip().lower()
    return TEN_NGUON.get(n, n or "không rõ nguồn")


_TRUONG_NGUON = [
    ("dien_tich_dat_m2", "Diện tích đất", "m²"),
    ("dien_tich_m2", "Diện tích", "m²"),
    ("dien_tich_su_dung_m2", "Diện tích sử dụng", "m²"),
    ("so_phong_ngu", "Phòng ngủ", ""),
    ("so_phong_vs", "Phòng vệ sinh", ""),
    ("tong_so_tang", "Số tầng", ""),
    ("mat_tien_m", "Mặt tiền", "m"),
    ("duong_rong_m", "Đường rộng", "m"),
    ("tang_so", "Tầng", ""),
]

# BA TRƯỜNG CHỌN PHẢI CẮT VỀ NHÃN GỐC TRƯỚC KHI HIỆN
#
# Khâu bóc của batdongsan dính nhãn của ô bên cạnh vào giá trị, nên cột thô
# hiện ra thành "Nội thất đầy đủTầng số5 tầng" — 63% số tin nhà đất. Ưu tiên
# cột đã chuẩn hoá (`noi_that`…) nếu người gọi đã chạy `don_truong_chon`;
# không có thì tự cắt ngay tại đây, vì `ban_do.nap` đọc thẳng file xlsx và
# không đi qua bước chuẩn hoá đó.
# =============================================================================
# ĐỘ CHẮC CHẮN CỦA ĐỊA CHỈ MỘT CĂN
# =============================================================================
# VÌ SAO PHẢI HIỆN CÁI NÀY RA
#
# Có tin ghi địa chỉ "Nguyễn Trãi, Phường Định Công, Quận Hoàng Mai" — mà
# Nguyễn Trãi không nằm trong Định Công. Đây là người đăng ghi sai, hệ thống
# chỉ chép lại. Nhưng hệ thống BIẾT là nó khả nghi: nó không tìm được đường
# Nguyễn Trãi trong Định Công, nên đã đặt ghim ở tâm phường (`nguon_toa_do`
# = "phuong") thay vì theo đường. Thông tin đó có sẵn, chỉ là chưa hiện ra —
# nên người xem đọc "Nguyễn Trãi" như một sự thật đã kiểm.
#
# TÔI ĐÃ THỬ MỘT CÁCH KHÁC VÀ BỎ, ghi lại để không ai làm lại:
# dùng những tin có toạ độ đáng tin làm mốc, rồi cờ những cặp (phường, đường)
# có khoảng cách lớn. Đo ra 2,5% tin nhà đất lệch quá 3 km, nghe có vẻ dùng
# được — nhưng riêng "Nguyễn Trãi" chỉ có ĐÚNG MỘT tin đáng tin làm mốc, và
# nó cách tâm Định Công 2,52 km, tức là dưới ngưỡng. Một con đường dài 5 km
# với một điểm mốc thì phép đo khoảng cách vô nghĩa. Cờ dựng trên chứng cứ
# mỏng như vậy sẽ vừa bỏ sót vừa báo oan.
#
# Nên chỉ dùng thứ đã đo được và đã công bố sai số: mức toạ độ.

MUC_DIA_CHI = {
    "nguon_goc":  ("đúng căn",          True),
    "ma_tin":     ("đúng căn",          True),
    "du_an":      ("đúng toà",          True),
    "duong":      ("đúng đường",        True),
    "duong_moi":  ("đúng đường",        True),
    "duong_quan": ("đúng đường",        True),
    "phuong":     ("chỉ biết phường",   False),
    "phuong_moi": ("chỉ biết phường",   False),
    "quan":       ("chỉ biết quận",     False),
}


def _do_chac_dia_chi(r) -> tuple[str, bool]:
    """(nhãn ngắn, tên đường có được xác minh hay không)."""
    return MUC_DIA_CHI.get(str(_lay(r, "nguon_toa_do", "")), ("không rõ", False))


_TRUONG_MUC = [
    ("phap_ly", "Giấy tờ pháp lý", "Pháp lý",
     vs.PHAP_LY_ND + vs.PHAP_LY_CC),
    ("noi_that", "Tình trạng nội thất", "Nội thất", vs.NOI_THAT),
    ("huong_cua", "Hướng cửa chính", "Hướng cửa", vs.HUONG),
]


def _muc_sach(r, cot_moi: str, cot_goc: str, nhan) -> str | None:
    v = _lay(r, cot_moi)
    if isinstance(v, str) and v:
        return v
    v = _lay(r, cot_goc)
    if not isinstance(v, str):
        return None
    sach = vs.chuan_hoa_muc(v, tuple(nhan))
    return sach if isinstance(sach, str) else None


def _noi_dung_da_luu(r) -> None:
    """Hiện nguyên văn những gì tin rao nói, theo bản đã lưu lúc thu thập.

    Đây là phần thay thế cho link chết. Người xem — và cô khi hỏi "5 căn này
    lấy ở đâu ra" — đọc được ngay tiêu đề, địa chỉ, thông số và mô tả gốc mà
    không phải phụ thuộc vào trang của sàn còn sống hay không.
    """
    tieu_de = _lay(r, "tieu_de")
    dia_chi = _lay(r, "dia_chi_goc")
    mo_ta = _lay(r, "mo_ta")

    if tieu_de:
        st.markdown(f"**{str(tieu_de)[:220]}**")
    if dia_chi:
        nhan, xac_minh = _do_chac_dia_chi(r)
        them = ("" if xac_minh else
                f" — hệ thống chỉ xác định được tới mức **{nhan}**, "
                f"phần tên đường/số nhà là nguyên văn tin rao và chưa kiểm được")
        st.caption(f"📍 {str(dia_chi)[:220]}{them}")

    gia = _lay(r, "TARGET_gia_vnd")
    muc = [f"**Giá chào {_sole(float(gia) / 1e9, 2)} tỷ**"] if gia else []
    for cot, nhan, don in _TRUONG_NGUON:
        v = _lay(r, cot)
        if v is None:
            continue
        if isinstance(v, str):
            muc.append(f"{nhan} {v}")
        else:
            muc.append(f"{nhan} {_sole(v, 0 if float(v) == int(float(v)) else 1)}"
                       + (f" {don}" if don else ""))
    for cot_moi, cot_goc, nhan, hop_le in _TRUONG_MUC:
        v = _muc_sach(r, cot_moi, cot_goc, hop_le)
        if v:
            muc.append(f"{nhan} {v}")
    if muc:
        st.markdown(" · ".join(muc))

    if mo_ta:
        t = " ".join(str(mo_ta).split())
        st.markdown(f"> {t[:700]}{'…' if len(t) > 700 else ''}")
    elif not tieu_de and not dia_chi:
        st.caption("Đợt thu thập này chỉ lưu các thông số ở trên, không lưu "
                   "tiêu đề và mô tả.")

    st.caption(f"Nguồn: **{_ten_nguon(r)}** · bản lưu ngày **{_ngay_vn(r)}**. "
               f"Đây là nội dung chốt lại lúc thu thập, không thay đổi sau đó.")


def _khoi_nguon(cot, r, khoa: str = "") -> None:
    """Khối "căn này lấy từ đâu", đặt trong thẻ của từng căn.

    Ba việc theo đúng thứ tự quan trọng: nội dung đã lưu (bằng chứng thật,
    không bao giờ hỏng) — nhãn nguồn và ngày lưu — rồi mới tới link, hạ dần
    theo tuổi tin.
    """
    tuoi = _tuoi_tin(r)
    url = _lay(r, "url_goc")
    co_link = isinstance(url, str) and url.startswith("http")

    dong = f"📄 {_ten_nguon(r)} · đã lưu ngày {_ngay_vn(r)}"
    if tuoi is not None:
        dong += f" ({_so(tuoi)} ngày trước)"
    cot.caption(dong)

    with cot.expander("Xem nội dung tin đã lưu"):
        _noi_dung_da_luu(r)

    if not co_link:
        cot.caption(
            "🔗 Đợt thu thập này không lưu địa chỉ trang, chỉ lưu nội dung "
            "tin — nội dung đó ở ngay trên. Căn này vẫn được dùng để tính "
            "giá bình thường.")
    elif tuoi is None or tuoi <= HAN_LINK_MOI:
        cot.markdown(f"[Mở tin gốc ↗]({url})")
    elif tuoi <= HAN_LINK_CU:
        cot.markdown(
            f"[Mở tin gốc ↗]({url})  \n"
            f"<span style='color:{COLOR['text_muted']};font-size:.78rem;'>"
            f"tin đã lưu {_so(tuoi)} ngày — sàn thường xoá sau ~30 ngày, "
            f"nên trang này có thể đã hết hạn</span>",
            unsafe_allow_html=True)
    else:
        with cot.expander(f"Trang gốc — đã lưu {_so(tuoi)} ngày trước, "
                          f"rất có thể đã bị xoá"):
            st.markdown(f"[{url}]({url})")
            st.caption(
                "Tin rao thường bị sàn xoá sau khoảng 30 ngày. Link để đây "
                "cho đủ dấu vết nguồn, nhưng phần dùng được là **nội dung đã "
                "lưu** ở trên — nó không phụ thuộc trang này còn sống hay không.")


def _gioi_thieu(cau_hoi: str, giai_thich: str, vi_du: str) -> None:
    """Khối mở đầu cho các tab công cụ.

    Người mới mở trang không biết "Săn tin", "Ngân sách", "So sánh" là gì —
    tên tab mô tả TÍNH NĂNG chứ không mô tả CÂU HỎI mà nó trả lời. Nên mỗi tab
    mở đầu bằng đúng câu hỏi người dùng đang có trong đầu, rồi mới đến cách
    dùng và một ví dụ cụ thể.
    """
    st.markdown(
        f"""
        <div style="background:{COLOR['accent_soft']};
                    border-left:3px solid {COLOR['accent']};
                    border-radius:8px;padding:12px 16px;margin-bottom:14px;">
          <div style="font-size:15px;font-weight:700;color:{COLOR['text']};
                      margin-bottom:6px;">{cau_hoi}</div>
          <div style="font-size:13px;color:{COLOR['text_muted']};
                      line-height:1.55;">{giai_thich}</div>
          <div style="font-size:12.5px;color:{COLOR['text_muted']};
                      margin-top:8px;font-style:italic;">Ví dụ: {vi_du}</div>
        </div>
        """, unsafe_allow_html=True)



# =============================================================================
# TRANG NGÂN SÁCH
# =============================================================================


# =============================================================================
# TRANG SO SÁNH
# =============================================================================


# =============================================================================
# ĐIỀU HƯỚNG
# =============================================================================

# =============================================================================
# TRANG TÌM NHÀ  (gộp Săn tin + Ngân sách + So sánh)
# =============================================================================

# Bốn ví dụ này chọn để mỗi cái phô ra một khả năng KHÁC nhau của bộ hiểu
# câu, không phải bốn câu na ná: (1) câu thường nhất; (2) phủ định + tên
# phường trước sáp nhập; (3) số tầng, mặt tiền, độ rộng ngõ; (4) tên dự án +
# ý định sắp xếp nằm trong câu.
VI_DU_TIM = [
    "Tôi có 5 tỷ, muốn 2 phòng ngủ ở Cầu Giấy",
    "Chung cư Mỹ Đình tầm 4 tỷ, không cần thang máy",
    "Nhà 4 tầng mặt tiền 5m, ngõ 3m ô tô vào, dưới 6 tỷ",
    "Times City 2 ngủ, căn nào đang rẻ hơn mặt bằng",
]

SAP_TIM = {
    "khop_nhat": "Khớp điều kiện nhất",
    "lech_thap": "Rẻ hơn mặt bằng nhiều nhất",
    "lech_cao": "Đắt hơn mặt bằng nhiều nhất",
    "re_nhat": "Giá thấp trước",
}


def _the_lech(r) -> str:
    """Nhãn 'so với mặt bằng' cho một căn — phần việc của tab Săn tin cũ.

    Đây là chỗ tính năng phát hiện lệch giá sống tiếp sau khi bỏ tab riêng:
    thay vì một danh sách riêng, mỗi kết quả tìm kiếm tự mang nhãn của nó.
    Người dùng đang xem nhà thì thấy luôn căn nào bất thường, không phải đổi
    tab để biết.
    """
    v = r.get("lech_phan_tram")
    if v is None or pd.isna(v):
        return ""
    if v <= -10:
        mau, chu = COLOR["success_chu"], f"rẻ hơn mặt bằng {abs(v):.0f}%"
    elif v >= 10:
        mau, chu = COLOR["danger"], f"cao hơn mặt bằng {v:.0f}%"
    else:
        mau, chu = COLOR["text_muted"], "đúng mặt bằng"
    return (f'<span style="font-size:.72rem;font-weight:600;color:{mau};'
            f'border:1px solid {mau};border-radius:2px;padding:.1rem .4rem;'
            f'white-space:nowrap;">{chu}</span>')


# Số căn gợi ý ngay trong bong bóng chat. Ba, không phải hăm bốn.
#
# Bản trước đổ cả 24 thẻ có viền xuống dưới khung chat, mỗi thẻ kèm một khối
# "xem nội dung tin đã lưu" — thành ra khung hội thoại chỉ còn là cái mũ nhỏ
# trên đầu một danh sách dài gấp mười lần nó, và người dùng nói đúng: "chưa
# được gọn gàng". Ba căn là đủ để trả lời câu "có căn nào không", còn ai muốn
# duyệt hết thì sang trang riêng — hai việc khác nhau, hai chỗ khác nhau.
SO_GOI_Y = 3


def _dong_can_gon(r, cot_dt) -> str:
    """Một căn gợi ý, gọn trong hai dòng để nằm được TRONG bong bóng chat.

    Khác `_hien_ket_qua_tim` ở chỗ nó không dựng thẻ có viền, không có ô số
    liệu, không có khối nguồn mở ra được — vì cả ba thứ đó cộng lại mới là
    nguyên nhân danh sách dài. Ở đây chỉ còn: tên, mấy số chính, nhãn lệch
    mặt bằng, giá, và link nếu tin còn mới. Muốn xem nguồn thì sang trang
    danh sách đầy đủ.
    """
    ten = (r.get("du_an_clean")
           if isinstance(r.get("du_an_clean"), str)
           and r.get("du_an_clean") != "Không xác định"
           else (str(r.get("tieu_de") or "")[:58] or "Căn đang bán"))

    phu = [f"{_so(r[cot_dt])} m²"]
    if pd.notna(r.get("so_phong_ngu")):
        phu.append(f"{int(r['so_phong_ngu'])} PN")
    if pd.notna(r.get("gia_tren_m2")):
        phu.append(f"{r['gia_tren_m2'] / 1e6:.0f} tr/m²")
    phu.append(str(r.get("phuong_moi") or r.get("quan_huyen") or ""))

    # Link CHỈ hiện khi tin còn trong hạn. Tin quá hạn thì đưa link ra đây là
    # mời người dùng bấm vào một trang đã bị xoá — cả dự án tôi hạ link theo
    # tuổi tin chứ không rải link bừa.
    tuoi = _tuoi_tin(r)
    url = _lay(r, "url_goc")
    lk = ""
    if isinstance(url, str) and url.startswith("http") \
            and (tuoi is None or tuoi <= HAN_LINK_MOI):
        lk = (f' · <a href="{url}" target="_blank" '
              f'style="font-size:.74rem;">tin gốc ↗</a>')

    return (
        f'<div style="display:flex;align-items:baseline;gap:.6rem;'
        f'padding:.42rem 0;border-top:1px solid {COLOR["border"]};">'
        f'<div style="flex:1;min-width:0;">'
        f'<div style="font-size:.86rem;font-weight:600;color:{COLOR["text"]};'
        f'overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">'
        f'{ten}</div>'
        f'<div style="font-size:.76rem;color:{COLOR["text_muted"]};'
        f'margin-top:.1rem;">'
        f'{" · ".join(x for x in phu if x)}{lk}</div></div>'
        f'<div style="text-align:right;white-space:nowrap;">'
        f'<div style="font-family:Newsreader,Georgia,serif;font-size:1rem;'
        f'font-weight:600;color:{COLOR["text"]};'
        f'font-variant-numeric:tabular-nums;">'
        f'{_sole(r["TARGET_gia_vnd"] / 1e9, 2)} tỷ</div>'
        f'<div style="margin-top:.12rem;">{_the_lech(r)}</div>'
        f'</div></div>')


def _hien_ket_qua_tim(kq: dict, loai: str) -> None:
    """Danh sách căn tìm được. Tách khỏi `render_tim_nha` cho hàm đó gọn."""
    import tim_nha as tnh

    if kq["so_khop"] == 0:
        st.info("Không còn căn nào thoả. Gõ **“mở rộng ra cả quận”**, "
                "**“đắt hơn cũng được”**, hoặc bấm ✕ trên một thẻ điều kiện.")
        return

    st.caption(f"Tìm được **{_so(kq['so_khop'])} căn**, hiện "
               f"{len(kq['ket_qua'])} căn đầu · sắp theo "
               f"**{tnh.NHAN_SAP.get(kq.get('sap', ''), '—').lower()}**.")

    cot_dt = vs.nap(loai)["cfg"].cot_dien_tich
    for _, r in kq["ket_qua"].iterrows():
        with st.container(border=True):
            a, b = st.columns([3, 1])
            ten = (r.get("du_an_clean") if isinstance(r.get("du_an_clean"), str)
                   and r.get("du_an_clean") != "Không xác định"
                   else (str(r.get("tieu_de") or "")[:80] or "Căn đang bán"))
            a.markdown(f"**{ten}**")
            phu = [f"{_so(r[cot_dt])} m²"]
            if pd.notna(r.get("so_phong_ngu")):
                phu.append(f"{int(r['so_phong_ngu'])} PN")
            if pd.notna(r.get("gia_tren_m2")):
                phu.append(f"{r['gia_tren_m2']/1e6:.0f} tr/m²")
            phu += [str(r.get("phuong_moi") or ""), str(r.get("quan_huyen") or "")]
            a.caption(" · ".join(x for x in phu if x))
            b.metric("Giá chào", f"{_sole(r['TARGET_gia_vnd']/1e9, 2)} tỷ")

            the = _the_lech(r)
            if the:
                a.markdown(the, unsafe_allow_html=True)
            _khoi_nguon(a, r)

    with st.expander("Đọc danh sách này thế nào"):
        st.markdown(
            "- Nhãn **rẻ / cao hơn mặt bằng** so giá chào với dự đoán của mô "
            "hình **chưa từng thấy căn này**.\n"
            "- Rẻ hơn mặt bằng **không chắc là món tốt** — có thể vướng pháp "
            "lý, nhà cần sửa, hoặc vị trí có vấn đề mà dữ liệu không ghi.")


def _the_dk_html(the: list, dam: bool = True) -> str:
    """Các thẻ điều kiện dạng chữ, để nhúng vào trong một bong bóng chat."""
    if not the:
        return ""
    vien = COLOR["accent"] if dam else COLOR["border"]
    chu = COLOR["accent"] if dam else COLOR["text_muted"]
    nen = COLOR["accent_soft"] if dam else "transparent"
    return "".join(
        f'<span style="display:inline-block;background:{nen};'
        f'border:1px solid {vien};color:{chu};border-radius:99px;'
        f'padding:.1rem .55rem;margin:.12rem .22rem .12rem 0;'
        f'font-size:.76rem;font-weight:600;">{c}</span>'
        for _, c in the)


def _luot_moi(tnh, cau: str, dk: dict, doi: list, tiep: bool,
              loai: str) -> dict:
    """Dựng một lượt hội thoại, CÓ SẴN phần tóm tắt kết quả.

    VÌ SAO TÍNH SẴN RỒI LƯU LẠI
    ---------------------------
    Giao diện chat vẽ lại toàn bộ các lượt cũ ở mỗi lần Streamlit chạy lại.
    Nếu mỗi bong bóng cũ lại gọi `tim()` một lần thì một cuộc mười lượt là
    mười lần lọc toàn bộ dữ liệu, mỗi lần kèm phép ghép bảng mức lệch — trang
    sẽ đứng dần theo độ dài cuộc hội thoại.
    Nên lượt nào tính một lần lúc sinh ra, lưu lại con số; chỉ LƯỢT CUỐI mới
    chạy lại để còn đổi được cách sắp xếp. Đúng như một ứng dụng nhắn tin: tin
    cũ không tải lại.
    """
    kq = tnh.tim("", loai, dk_them=dk, k=24)
    return {"cau": cau, "dk": dk, "doi": doi, "tiep": tiep, "loai": loai,
            "so_khop": kq["so_khop"], "bo_qua": kq["bo_qua"],
            "the": tnh.the_dk(dk, loai), "thong_ke": kq.get("thong_ke", {})}


def _bong_bong_may(tnh, t: dict, cuoi: bool) -> None:
    """Phần trả lời của hệ thống trong một lượt."""
    if not t["the"] and not t["doi"]:
        st.markdown("Mình chưa bóc được điều kiện nào từ câu này. Thử nêu số "
                    "tiền (*5 tỷ*), diện tích (*60 m2*), số phòng "
                    "(*2 phòng ngủ*) hoặc tên quận, phường.")
        return

    if t["doi"]:
        st.markdown("".join(
            f'<div style="font-size:.84rem;margin:0 0 .15rem;">↳ {x}</div>'
            for x in t["doi"]), unsafe_allow_html=True)

    if t["so_khop"]:
        st.markdown(
            f'Tìm được **{_so(t["so_khop"])} căn**.'
            if cuoi else
            f'<span style="color:{COLOR["text_muted"]};font-size:.84rem;">'
            f'{_so(t["so_khop"])} căn</span>', unsafe_allow_html=True)
    else:
        st.markdown("Không còn căn nào thoả. Nói **“mở rộng ra cả quận”** hoặc "
                    "**“đắt hơn cũng được”**, hoặc bấm ✕ trên một thẻ.")

    st.markdown(
        f'<div style="margin-top:.35rem;">{_the_dk_html(t["the"], cuoi)}</div>',
        unsafe_allow_html=True)

    if t["bo_qua"]:
        ten = ", ".join(tnh.NHAN_DK.get(x, tnh.NHAN_CO.get(x, x))
                        for x in t["bo_qua"])
        st.caption(f"⚠️ Không có căn nào thoả hết, mình đã nới: **{ten}**.")
    if cuoi and t["dk"].get("_phuong_go_cu"):
        st.caption(f"ℹ️ **{t['dk']['_phuong_go_cu'].title()}** giờ thuộc "
                   f"**{t['dk']['phuong_moi']}** sau sáp nhập 1/7/2025.")


def render_tim_nha() -> None:
    """Tìm nhà bằng hội thoại, giao diện như một ứng dụng nhắn tin.

    Dùng `st.chat_message` / `st.chat_input` của Streamlit: bong bóng trái là
    hệ thống, bong bóng phải là người dùng, ô gõ dính đáy trang.

    VÌ SAO ĐÁNG ĐỔI SANG DẠNG NÀY
    -----------------------------
    Bản trước là một khung "nhật ký hội thoại" tự vẽ, đặt cạnh ô nhập và bảng
    thẻ điều kiện — ba khối rời nhau nói về cùng một cuộc trao đổi. Người dùng
    phải tự ghép lại trong đầu mới hiểu lượt nào sinh ra kết quả nào.

    Dạng nhắn tin bỏ hẳn việc ghép đó: mỗi lượt là một cặp bong bóng, câu hỏi
    ngay trên câu trả lời, và phần hệ thống hiểu gì nằm trong chính bong bóng
    trả lời chứ không ở một bảng khác. Người dùng đã biết cách đọc dạng này từ
    trước, không phải học gì thêm.

    HAI THỨ GIỮ LẠI TỪ BẢN TRƯỚC, vì chúng không phải trang trí:
      · thẻ điều kiện BỎ ĐƯỢC TỪNG CÁI — cách sửa nhanh nhất khi máy hiểu sai
        đúng một chỗ;
      · mỗi lượt nói rõ ĐỔI GÌ kèm con số cũ và mới — "rẻ hơn nữa" tự nó
        không cho biết mức nào.
    """
    import tim_nha as tnh

    lich: list = st.session_state.setdefault("tn_lich", [])
    loai_hien: str = (lich[-1].get("loai") if lich else None) or \
        st.session_state.get("tn_loai", "chungcu")
    # Số lượt, để `_cach_sap` biết khi nào là lượt MỚI.
    st.session_state["_tn_n"] = len(lich)

    # HAI MÀN, KHÔNG PHẢI HAI KHỐI XẾP DỌC. Danh sách đầy đủ THAY CHỖ khung
    # chat chứ không nằm dưới nó — xem ghi chú ở `_trang_danh_sach`.
    if st.session_state.get("tn_xem_het") and lich and lich[-1]["so_khop"]:
        _trang_danh_sach(tnh, lich[-1], lich[-1].get("loai") or loai_hien)
        return
    st.session_state["tn_xem_het"] = False

    if not lich:
        loai_hien = st.radio(
            "Loại bất động sản", ["chungcu", "nhadat"],
            format_func=lambda x: "Chung cư" if x == "chungcu" else "Nhà đất",
            horizontal=True, key="tn_loai")

    # ------------------------------------------------------- khung hội thoại
    with st.container(border=True):
        with st.chat_message("assistant", avatar="🏠"):
            if lich:
                st.markdown("Mình đang tìm theo những điều kiện dưới đây. "
                            "Nói thêm một câu để sửa.")
            else:
                st.markdown(
                    "Bạn đang tìm nhà thế nào? Gõ bằng câu thường — "
                    "**“tôi có 5 tỷ, 2 phòng ngủ ở Cầu Giấy”**. Xem kết quả "
                    "rồi nói tiếp *“rẻ hơn nữa”*, *“đổi sang Thanh Xuân”* — "
                    "mình sửa trên điều kiện cũ, bạn không phải gõ lại.")

        for i, t in enumerate(lich):
            cuoi = (i == len(lich) - 1)
            with st.chat_message("user", avatar="🧑"):
                st.markdown(t["cau"])
            with st.chat_message("assistant", avatar="🏠"):
                _bong_bong_may(tnh, t, cuoi)
                # Ba căn gợi ý chỉ ở lượt CUỐI. Lượt cũ mà cũng kèm ba căn
                # thì cuộn lại một cuộc mười lượt là ba mươi căn — đúng cái
                # dài mà lần này đang đi sửa.
                if cuoi and t["so_khop"]:
                    _goi_y_trong_chat(tnh, t, t.get("loai") or loai_hien)

        # Thẻ bỏ được: chỉ ở lượt cuối, vì chỉ điều kiện HIỆN TẠI mới sửa được.
        if lich and lich[-1]["the"]:
            st.caption("Bấm để bỏ một điều kiện:")
            the = lich[-1]["the"]
            for r in range(0, len(the), 4):
                cot = st.columns(4)
                for j, (khoa, chu) in enumerate(the[r:r + 4]):
                    if cot[j].button(f"✕ {chu}", key=f"tn_xoa_{khoa}",
                                     width='stretch'):
                        dk = tnh.xoa_dk(lich[-1]["dk"], khoa)
                        lich.append(_luot_moi(tnh, f"Bỏ {chu}", dk,
                                              [f"bỏ {chu}"], True, loai_hien))
                        st.rerun()

        # BA GỢI Ý, KHÔNG PHẢI SÁU. `st.columns(len(goi_y))` với sáu nhãn dài
        # kiểu "Bỏ điều kiện thang máy" cho ra sáu cột hẹp, chữ trong nút gãy
        # ba dòng — chính nó làm đáy khung chat rối nhất.
        goi_y = (tnh.VI_DU_CAU_TIEP if lich else VI_DU_TIM)[:3]
        c = st.columns(len(goi_y))
        for i, vd in enumerate(goi_y):
            if c[i].button(vd, key=f"tn_vd{i}", width='stretch',
                           type="secondary"):
                st.session_state["_tn_cho_gui"] = vd
                st.rerun()

        if lich:
            n1, n2 = st.columns(2)
            if n1.button("↩ Quay lại lượt trước", key="tn_undo",
                         width='stretch', disabled=len(lich) < 2):
                lich.pop()
                st.rerun()
            if n2.button("✦ Tìm mới từ đầu", key="tn_moi", width='stretch'):
                st.session_state["tn_lich"] = []
                st.rerun()

    # HAI ĐƯỜNG VÀO, MỘT CHỖ XỬ LÝ. Người dùng gõ vào ô chat, hoặc bấm một
    # nút gợi ý. Nút không trả chữ về ngay được (nó chạy trước khi ô chat được
    # dựng) nên nó ghi vào session_state rồi chạy lại; ở đây nhặt ra. Gộp hai
    # đường vào cùng một khối xử lý để không có hai nhánh logic song song —
    # hai nhánh thì sớm muộn chúng lệch nhau.
    cau = st.chat_input("Nói với hệ thống…", key="tn_chat") \
        or st.session_state.pop("_tn_cho_gui", None)

    if cau and cau.strip():
        cau = cau.strip()
        dk_hien = lich[-1]["dk"] if lich else {}
        if tnh.la_lam_lai(cau):
            st.session_state["tn_lich"] = []
        elif tnh.la_hoan_tac(cau):
            if lich:
                lich.pop()
        elif tnh.la_cau_tiep(cau, dk_hien):
            tk = lich[-1].get("thong_ke", {}) if lich else {}
            dk, doi = tnh.doc_chinh_sua(cau, dk_hien, tk, loai_hien)
            lich.append(_luot_moi(tnh, cau, dk, doi, True, loai_hien))
        else:
            l2 = tnh.doan_loai(cau, loai_hien)
            lich.append(_luot_moi(tnh, cau, tnh.phan_tich(cau, l2), [],
                                  False, l2))
        st.rerun()


def _cach_sap(tnh, t: dict) -> str:
    """Cách sắp xếp đang dùng, chung cho cả bong bóng chat lẫn trang danh sách.

    Câu nói của người dùng được ưu tiên: "rẻ nhất trước" thì đổi cách sắp ngay
    ở lượt đó. Nhưng chỉ đổi MỘT lần cho mỗi lượt mới — nếu áp lại ở mọi lần
    Streamlit chạy lại thì người dùng bấm tay sang cách khác sẽ bị nhảy về,
    và họ sẽ tưởng nút bị hỏng.
    """
    KHOA = "tn_sap_luu"   # KHÔNG phải khoá của widget — xem ghi chú dưới.
    #
    # Cách sắp xếp lưu ở một khoá RIÊNG, không dùng chung khoá với ô radio ở
    # trang danh sách. Dùng chung thì Streamlit xoá khoá đó ở những lần chạy
    # mà ô radio không được vẽ ra (tức là mọi lần ở khung chat), nên người
    # dùng chọn "rẻ nhất trước" ở trang danh sách, bấm về khung chat, ba căn
    # gợi ý lại nhảy về cách sắp mặc định.
    y = tnh.doc_sap(t["cau"]).get("sap")
    if st.session_state.get("tn_luot_truoc") != st.session_state.get("_tn_n"):
        st.session_state["tn_luot_truoc"] = st.session_state.get("_tn_n")
        if y:
            st.session_state[KHOA] = y
    return st.session_state.get(KHOA) or next(iter(SAP_TIM))


def _goi_y_trong_chat(tnh, t: dict, loai: str) -> None:
    """Ba căn gợi ý + đường sang danh sách đầy đủ, NẰM TRONG bong bóng chat.

    Đây là chỗ trả lời yêu cầu "tất cả gói gọn trong 1 khung chat thôi". Trước
    đây khối này nằm NGOÀI khung, nên trang có hai vùng nói về cùng một câu
    hỏi: khung chat bảo "tìm được 216 căn", rồi bên dưới lại một danh sách
    không có gì nối nó với câu nào đã sinh ra nó.
    """
    sap = _cach_sap(tnh, t)
    kq = tnh.tim("", loai, dk_them=t["dk"], sap=sap, k=SO_GOI_Y)
    if not len(kq["ket_qua"]):
        return

    cot_dt = vs.nap(loai)["cfg"].cot_dien_tich
    st.markdown(
        f'<div style="font-size:.78rem;color:{COLOR["text_muted"]};'
        f'margin:.5rem 0 0;">{SO_GOI_Y} căn '
        f'{tnh.NHAN_SAP.get(sap, "").lower().removesuffix(" trước")}'
        f'</div>'
        + "".join(_dong_can_gon(r, cot_dt)
                  for _, r in kq["ket_qua"].iterrows()),
        unsafe_allow_html=True)

    if t["so_khop"] > SO_GOI_Y:
        if st.button(f"Xem cả {_so(t['so_khop'])} căn →", key="tn_xem_het_nut",
                     type="secondary"):
            st.session_state["tn_xem_het"] = True
            st.rerun()


def _trang_danh_sach(tnh, t: dict, loai: str) -> None:
    """Trang danh sách đầy đủ — thay chỗ khung chat, không nằm dưới nó.

    Người dùng muốn "đọc riêng" thì nó phải là một chỗ riêng thật: khung chat
    biến đi, trang này chiếm toàn bộ, và có đường về. Nếu vẫn xếp dưới khung
    chat thì dù gợi ý có gọn tới đâu, trang vẫn dài y như cũ.
    """
    if st.button("← Về khung chat", key="tn_ve_chat"):
        st.session_state["tn_xem_het"] = False
        st.rerun()

    st.markdown(f"#### {_so(t['so_khop'])} căn thoả điều kiện")
    if t["the"]:
        st.markdown(_the_dk_html(t["the"], True), unsafe_allow_html=True)
    st.caption(f'Từ câu: “{t["cau"]}” — sửa điều kiện thì về khung chat.')

    khoa = list(SAP_TIM)
    dang = _cach_sap(tnh, t)
    sap = st.radio("Sắp xếp", khoa, format_func=SAP_TIM.get, horizontal=True,
                   index=khoa.index(dang) if dang in khoa else 0,
                   key="tn_sap_radio")
    # Ghi lại để khung chat dùng chung một cách sắp — ba căn gợi ý và danh
    # sách đầy đủ phải nói cùng một thứ tự, nếu không thì "căn đầu" ở hai chỗ
    # là hai căn khác nhau và người dùng nghĩ hệ thống sai.
    st.session_state["tn_sap_luu"] = sap
    kq = tnh.tim("", loai, dk_them=t["dk"], sap=sap, k=24)
    _hien_ket_qua_tim(kq, loai)


def main() -> None:
    st.set_page_config(page_title="Định giá Bất động sản Hà Nội",
                       page_icon="🏢", layout="wide")
    inject_css()

    # `key="nav_chinh"` không phải để đọc lại giá trị — nó để CSS bám vào.
    # Streamlit gắn class `st-key-<key>` lên khối bọc của widget, và đó là
    # cách duy nhất có giấy tờ để nhắm CSS vào MỘT widget. Nhờ nó, dải chọn
    # trang này thành dạng viên thuốc mà bốn `st.radio` khác trong app vẫn
    # giữ nguyên hình tròn bình thường.
    mode = st.radio(
        "Chọn trang",
        ["🏢 Chung cư", "🏠 Nhà đất", "🗺️ Khu vực", "💬 Tìm nhà"],
        horizontal=True, label_visibility="collapsed", key="nav_chinh",
    )

    if "Chung cư" in mode:
        hero_header(
            f'Định giá <b style="color:{COLOR["text"]}">căn hộ chung cư</b> — '
            "chọn quận, phường, dự án rồi nhập vài thông số cơ bản.")
        render_chungcu()
    elif "Nhà đất" in mode:
        hero_header(
            f'Định giá <b style="color:{COLOR["text"]}">nhà đất thổ cư</b> — '
            "nhà ngõ hẻm, mặt phố, liền kề, biệt thự.")
        render_nhadat()
    elif "Khu vực" in mode:
        hero_header(
            f'<b style="color:{COLOR["text"]}">Mặt bằng giá theo khu vực</b> — '
            "xếp hạng phường, so sánh hai phường, và xem những căn đang bán "
            "quanh một điểm.")
        render_ban_do()
    else:
        hero_header(
            f'<b style="color:{COLOR["text"]}">Tìm nhà bằng câu nói</b> — '
            "gõ điều kiện của bạn như nói với người quen, hệ thống lọc và "
            "chỉ ra căn nào đang lệch mặt bằng.")
        render_tim_nha()


if __name__ == "__main__":
    main()
