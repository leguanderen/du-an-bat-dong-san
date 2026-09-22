"""
gan_toa_do.py — Gán toạ độ cho dữ liệu cũ bằng cách CHUYỂN từ dữ liệu API.

VÌ SAO KHÔNG DÙNG NOMINATIM NỮA
-------------------------------
`geocode.py` hỏi Nominatim từng địa chỉ một. Cách đó có ba vấn đề đã đo được:

  * Chậm: 1 giây/địa chỉ, 6.192 dòng là gần hai tiếng.
  * Sai vì cải cách hành chính 1/7/2025: OSM đã đổi sang tên phường mới, dữ liệu
    của mình còn tên cũ. Đối chiếu 148 dòng Nominatim trả về "mức phường" với
    toạ độ API cùng phường: lệch TRUNG VỊ 17,6 km — tức là trỏ nhầm hẳn sang
    một đơn vị hành chính khác trùng tên.
  * Ngay cả 563 dòng Nominatim trả về "mức đường" cũng lệch trung vị 0,83 km so
    với toạ độ thật của các tin API trên đúng con đường đó — lớn hơn cả sai số
    p90 của phương pháp trong file này (0,67 km).

Nên module này bỏ hẳn Nominatim và dùng một nguồn tốt hơn: 4.693 tin API vừa
cào có SẴN toạ độ do chính nhatot gắn. Từ đó dựng bảng tra và gán ngược cho tin cũ.

Ý TƯỞNG
-------
Giống như tra sổ địa chỉ: mình không biết nhà anh A ở đâu, nhưng biết anh A ở
phố Vũ Trọng Phụng, phường Thanh Xuân Trung — và trong sổ có 12 người khác cùng
phố cùng phường, toạ độ của họ nằm gọn một chỗ. Lấy trung vị 12 điểm đó làm toạ
độ cho anh A. Sai bao nhiêu thì ĐO ĐƯỢC, không phải đoán (xem `kiem_dinh`).

TÁM MỨC, XẾP THEO ĐỘ TIN CẬY GIẢM DẦN
-------------------------------------
  1. ma_tin     — trùng đúng mã tin trong dữ liệu API -> lấy nguyên toạ độ gốc
  2. du_an      — trùng tên dự án (chung cư)          -> p90 = 0,00 km
  3. duong      — trung vị nhóm quận+phường+đường     -> p90 = 0,67 km
  4. phuong     — trung vị nhóm quận+phường           -> p90 = 0,80 km
  5. duong_quan — trung vị nhóm quận+đường            -> p90 = 2,54 km
  6. quan       — trung vị nhóm quận                  -> p90 = 4,28 km

Mức 5 chỉ để vớt vài chục dòng mất tên phường nhưng còn tên đường — thà biết
"ở đường Minh Khai, quận Hai Bà Trưng" còn hơn chỉ biết "ở quận Hai Bà Trưng".

Cột `nguon_toa_do` ghi rõ mỗi dòng rơi vào mức nào, `sai_so_p90_km` ghi sai số
kỳ vọng của mức đó. Khi cần độ chính xác cao (ví dụ tính khoảng cách tới trường
học trong bán kính 500 m) thì lọc `nguon_toa_do` in ('ma_tin','du_an','duong').

MỘT ĐIỀU PHẢI NÓI THẲNG
-----------------------
86,7% tin API dùng chung toạ độ với ít nhất một tin khác. Nghĩa là bản thân
nhatot cũng KHÔNG gắn toạ độ tới từng căn nhà — họ gắn một điểm ghim đại diện
cho con đường / khu vực. Vậy nên kết quả ở đây là toạ độ MỨC ĐƯỜNG, đủ tốt để
tính khoảng cách tới tiện ích và vẽ bản đồ nhiệt, KHÔNG đủ để chỉ đúng một căn
nhà. Đừng viết trong khoá luận rằng hệ thống định vị được từng bất động sản.

CÁCH CHẠY
---------
    python gan_toa_do.py                  # gán cho cả chung cư và nhà đất
    python gan_toa_do.py --kiem-dinh      # in bảng sai số leave-one-out
    python gan_toa_do.py --loai chungcu   # chỉ một nhánh
"""

from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent


def thu_muc(ten: str) -> Path:
    """Tìm thư mục dữ liệu dù script nằm trong `pipeline/` hay ngang hàng.

    Trên máy bạn, cây thư mục là `Dự án bất động sản/{pipeline, data_clean}`,
    nên từ `pipeline/` phải đi lên một cấp. Nhưng khi chạy thử ở nơi khác thì
    data_clean lại nằm cạnh script. Dò cả hai chỗ thì cả hai đều chạy được, và
    không ai phải sửa hằng số đường dẫn mỗi lần chép file.
    """
    for goc in (BASE_DIR, BASE_DIR.parent):
        if (goc / ten).is_dir():
            return goc / ten
    return BASE_DIR.parent / ten


DATA_RAW = thu_muc("data_raw")
DATA_CLEAN = thu_muc("data_clean")


def thu_muc_ra(ten: str) -> Path:
    """Nơi GHI RA một thư mục kết quả. KHÁC `thu_muc` ở chỗ quan trọng nhất.

    `thu_muc` dò một thư mục ĐÃ CÓ, và khi không thấy thì đoán
    `BASE_DIR.parent / ten`. Đoán như vậy hợp lý khi chỉ ĐỌC — không thấy thì
    đằng nào cũng hỏng, đường dẫn nào cũng thế.

    Nhưng đem đúng hàm đó đi chọn chỗ GHI thì thành lỗi thật, và tôi đã dính:
    `_trien_khai` chưa tồn tại (vừa `rm -rf` nó), nên `thu_muc` trả về
    `BASE_DIR.parent/_trien_khai` — tức NGOÀI thư mục dự án. Gói model 15 MB
    ghi ra đó, mọi phép kiểm vẫn xanh vì cả bên ghi lẫn bên đọc đều dùng chung
    đường dẫn sai ấy. Đem deploy thì gói không nằm trong repo để mà commit,
    còn trên máy người dùng nó rơi ra ngang hàng thư mục dự án.

    Nên hàm này neo vào thứ chắc chắn có: thư mục CHA của `data_clean`. Đó là
    gốc dự án trong cả hai kiểu bố trí (script ở gốc, hay script trong
    `pipeline/`), và nó không phụ thuộc việc thư mục đích đã tồn tại hay chưa.
    """
    for goc in (BASE_DIR, BASE_DIR.parent):
        if (goc / ten).is_dir():
            return goc / ten
    return DATA_CLEAN.parent / ten

# Ngày cào từng nhánh. Dữ liệu cũ không giữ được ngày đăng của từng tin, nên
# đây là mốc thời gian trung thực nhất mình có: "tin này còn treo trên sàn vào
# ngày ấy". Ghi ra cột riêng để sau này so sánh giá theo thời gian không bị
# nhầm tưởng là ngày đăng.
NGAY_QUAN_SAT = {"chungcu": "2026-06-13", "nhadat": "2026-06-18"}

# Số phường/xã của Hà Nội sau sáp nhập 1/7/2025. Nguồn chuẩn là bảng tra dựng
# trong `gazetteer_phuong.dung_bang()` (345 phường cũ -> 113 phường mới); ở đây
# để số cứng vì chỗ dùng nó chỉ là một CẢNH BÁO, không phải phép lọc — dựng cả
# bảng tra chỉ để in một dòng nhắc thì quá đắt.
SO_PHUONG_HA_NOI = 113

# Sai số p90 đo bằng leave-one-out trên chính dữ liệu API (xem `kiem_dinh`).
SAI_SO_P90_KM = {
    "nguon_goc": 0.00,     # toạ độ do chính sàn công bố, không phải ước lượng
    "ma_tin": 0.00,
    "du_an": 0.00,
    "duong": 0.67,
    "phuong": 0.79,
    "duong_moi": 1.17,
    "phuong_moi": 2.19,
    "duong_quan": 2.51,
    "quan": 4.27,
}

MA_CATEGORY = (1010, 1020, 1040)  # chung cư, nhà ở, đất

# Khung bao Hà Nội. Có vài tin ghi địa chỉ Hà Nội nhưng ghim toạ độ ở Đà Nẵng
# hay TP.HCM — người đăng bấm nhầm bản đồ. Một tin bẩn lọt vào nhóm chỉ 2–3 tin
# là kéo trung vị ra giữa Biển Đông, nên phải loại TRƯỚC khi dựng bảng tra.
HN_BOX = (20.53, 21.39, 105.28, 106.03)   # vĩ độ min/max, kinh độ min/max


# =============================================================================
# CHUẨN HOÁ KHOÁ TRA CỨU
# =============================================================================
# Hai nguồn viết địa chỉ khác nhau: "Ngõ 58 Vũ Trọng Phụng" và "Vũ Trọng Phụng"
# là cùng một chỗ. Chuẩn hoá về không dấu, thường, bỏ tiền tố hành chính rồi
# mới so khớp thì tỉ lệ khớp mức đường tăng từ ~64% lên ~68–79%.

def khong_dau(s: object) -> str | None:
    if not isinstance(s, str):
        return None
    s = s.replace("đ", "d").replace("Đ", "D")
    s = "".join(c for c in unicodedata.normalize("NFD", s)
                if unicodedata.category(c) != "Mn").lower()
    return re.sub(r"[^a-z0-9]+", " ", s).strip() or None


_PFX_DUONG = re.compile(
    r"^(duong|pho|ngo|ngach|hem|to dan pho|to|khu do thi|khu|so nha|so|dg)\s+")


def chuan_duong(s: object) -> str | None:
    """"Ngõ 58 Vũ Trọng Phụng" -> "vu trong phung".

    Bóc tiền tố nhiều lần vì địa chỉ hay lồng nhau ("Ngõ 12 ngách 3 Kim Mã"),
    và bỏ số đứng đầu vì số ngõ không giúp định vị — con đường mẹ mới giúp.
    """
    s = khong_dau(s)
    if not s:
        return None
    for _ in range(4):
        t = re.sub(r"^\d+\s*", "", _PFX_DUONG.sub("", s))
        if t == s:
            break
        s = t
    return s or None


def chuan_phuong(s: object) -> str | None:
    s = khong_dau(s)
    return re.sub(r"^(phuong|xa|thi tran|tt)\s+", "", s) or None if s else None


def chuan_quan(s: object) -> str | None:
    s = khong_dau(s)
    return re.sub(r"^(quan|huyen|thi xa|tp|thanh pho)\s+", "", s) or None if s else None


_TU_CHUNG_DU_AN = re.compile(
    r"\b(chung cu|khu do thi moi|khu do thi|kdt|du an|toa nha|toa|cc|can ho|"
    r"khu can ho|block)\b")


def chuan_du_an(s: object) -> str | None:
    """Bỏ những từ ai cũng viết ("chung cư", "toà"), giữ lại phần định danh.

    Nhờ vậy "Chung cư Đại Thanh" khớp được với "Đại Thanh": số dòng khớp dự án
    tăng từ 152 lên 263.
    """
    s = khong_dau(s)
    if not s:
        return None
    return re.sub(r"\s+", " ", _TU_CHUNG_DU_AN.sub(" ", s)).strip() or None


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0088
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = (np.sin((p2 - p1) / 2) ** 2
         + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2)
    return 2 * R * np.arcsin(np.sqrt(a))


# =============================================================================
# NẠP DỮ LIỆU API
# =============================================================================

def nap_api(thu_muc: Path = DATA_RAW) -> pd.DataFrame:
    """Gộp ba nhóm hàng của API thành một bảng tra chung.

    Đọc .csv chứ không đọc .xlsx: bản .xlsx của nhóm 1020 và "tatca" bị hỏng
    (chỉ còn 5 cột đầu có dữ liệu, toàn bộ toạ độ trống). Bản .csv luôn được
    ghi trước và đầy đủ, nên lấy nó làm nguồn.

    Dùng cả nhóm 1040 (đất) dù mô hình không định giá đất: vị trí một con đường
    không phụ thuộc vào việc trên đó bán nhà hay bán đất, nên thêm 608 tin này
    làm bảng tra dày hơn mà không hề trộn dữ liệu vào mô hình.
    """
    khung = []
    for cg in MA_CATEGORY:
        p = thu_muc / f"chotot_api_{cg}.csv"
        if p.exists():
            khung.append(pd.read_csv(p))
        else:
            print(f"  (thiếu {p.name} — bỏ qua)", file=sys.stderr)
    if not khung:
        raise FileNotFoundError(
            f"Không thấy file chotot_api_*.csv nào trong {thu_muc}. "
            "Chạy `python crawl_api.py --chi-xuat` để xuất lại từ kho SQLite.")

    api = pd.concat(khung, ignore_index=True)
    api = api.drop_duplicates(subset="list_id")
    api = api.dropna(subset=["latitude", "longitude"]).copy()

    lo_v, hi_v, lo_k, hi_k = HN_BOX
    trong = (api["latitude"].between(lo_v, hi_v)
             & api["longitude"].between(lo_k, hi_k))
    if (~trong).any():
        print(f"  loại {int((~trong).sum())} tin API ghim toạ độ ngoài Hà Nội")
    api = api[trong].copy()

    api["k_quan"] = api["area_name"].map(chuan_quan)
    api["k_phuong"] = api["ward_name"].map(chuan_phuong)     # ward_name = tên CŨ
    api["k_duong"] = api["street_name"].map(chuan_duong)
    api["k_du_an"] = api["pty_project_name"].map(chuan_du_an)
    # Khoá thứ hai theo tên phường MỚI sau sáp nhập. Cần vì batdongsan chỉ công
    # bố tên mới, không có tên cũ — không có bảng này thì mọi tin batdongsan
    # rơi thẳng xuống mức quận (sai số p90 4,3 km).
    api["k_phuong_moi"] = api["ward_name_v3"].map(chuan_phuong)
    return _loai_ghim_lac(api)


# Một phường/xã của Hà Nội rộng nhất cũng chỉ vài km. Tin nào ghim cách trung
# vị phường của chính nó quá xa thì gần như chắc chắn là bấm nhầm bản đồ.
NGUONG_LAC_KM = 5.0
TIN_TOI_THIEU_DE_XET = 5   # phường quá thưa thì trung vị chưa đáng tin, tha


def _loai_ghim_lac(api: pd.DataFrame) -> pd.DataFrame:
    """Loại tin ghim sai NGAY TRONG phường của nó.

    Khung Hà Nội bắt được tin ghim sang Đà Nẵng, nhưng không bắt được tin ghi
    "Phường Tương Mai, Hoàng Mai" mà ghim xuống Phú Xuyên — vẫn trong thành phố,
    cách 50 km. Loại 8 tin kiểu này thì ô lưới giá ở phía nam thôi hết đỏ oan.
    """
    tam = api.groupby(["k_quan", "k_phuong"])[["latitude", "longitude"]].transform("median")
    cach = haversine_km(api["latitude"], api["longitude"],
                        tam["latitude"], tam["longitude"])
    dem = api.groupby(["k_quan", "k_phuong"])["latitude"].transform("size")
    lac = (cach > NGUONG_LAC_KM) & (dem >= TIN_TOI_THIEU_DE_XET)
    if lac.any():
        print(f"  loại {int(lac.sum())} tin API ghim lệch >"
              f"{NGUONG_LAC_KM:g} km so với phường của chính nó")
    return api[~lac].copy()


def _bang_tra(api: pd.DataFrame, khoa: list[str]) -> pd.DataFrame:
    """Trung vị toạ độ theo nhóm. Trung vị chứ không phải trung bình: một tin
    ghi nhầm toạ độ sang tỉnh khác sẽ kéo trung bình đi hàng chục km, còn trung
    vị thì không nhúc nhích."""
    d = api.dropna(subset=khoa)
    g = d.groupby(khoa)
    out = g[["latitude", "longitude"]].median()
    out["so_tin_tham_chieu"] = g.size()
    return out


def bang_phuong_moi(api: pd.DataFrame) -> pd.DataFrame:
    """(quận cũ, phường cũ) -> tên phường/xã MỚI sau sáp nhập 1/7/2025.

    API trả cả hai: `ward_name` là tên cũ, `ward_name_v3` là tên mới. Đây là
    cách lấy bảng đối chiếu miễn phí, khỏi phải gõ tay 126 phường.
    """
    d = api.dropna(subset=["k_quan", "k_phuong", "ward_name_v3"])
    return (d.groupby(["k_quan", "k_phuong"])["ward_name_v3"]
            .agg(lambda s: s.mode().iat[0])
            .rename("phuong_moi").to_frame())


# =============================================================================
# VÁ ĐỊA GIỚI HỎNG
# =============================================================================

_RE_QUAN = re.compile(r"^(quận|huyện|thị xã)\s+", re.I)
_RE_PHUONG = re.compile(r"^(phường|xã|thị trấn)\s+", re.I)


def _tach(dia_chi: object, mau: re.Pattern) -> str | None:
    """Lấy thành phần khớp mẫu trong chuỗi địa chỉ đầy đủ, tính từ cuối lên."""
    if not isinstance(dia_chi, str):
        return None
    for phan in reversed([p.strip() for p in dia_chi.split(",")]):
        if mau.match(phan):
            return phan
    return None


def va_dia_gioi(out: pd.DataFrame, quan_hop_le: set[str]) -> int:
    """Sửa lại quận/phường khi khoá tra không nằm trong danh sách 30 quận/huyện.

    Bộ bóc địa chỉ ở khâu làm sạch từng nhận nhầm "Quận uỷ" (trụ sở cơ quan)
    thành tên quận, cho ra `quan_huyen = "Quận Uỷ Cầu Giấy"`. Chỉ 2 dòng, nhưng
    thay vì bỏ qua thì lấy lại từ `dia_chi_goc` — chuỗi đó vẫn còn nguyên
    "…, Quận Cầu Giấy, Hà Nội".

    Trả về số dòng đã vá. Đây là lưới an toàn; chỗ sửa gốc nằm ở clean_nhadat.py.
    """
    if "dia_chi_goc" not in out.columns:
        return 0
    hong = out["k_quan"].notna() & ~out["k_quan"].isin(quan_hop_le)
    if not hong.any():
        return 0
    for i in out.index[hong]:
        q = _tach(out.at[i, "dia_chi_goc"], _RE_QUAN)
        if q and chuan_quan(q) in quan_hop_le:
            out.at[i, "k_quan"] = chuan_quan(q)
            # Sửa cả cột hiển thị, không chỉ khoá tra. Nếu chỉ sửa khoá thì toạ
            # độ đúng nhưng bảng thống kê vẫn hiện một "quận" tên là "Quận ủy".
            out.at[i, "quan_huyen"] = q
            p = _tach(out.at[i, "dia_chi_goc"], _RE_PHUONG)
            if p:
                out.at[i, "k_phuong"] = chuan_phuong(p)
                out.at[i, "phuong_xa"] = p
    return int(hong.sum())


# =============================================================================
# GÁN TOẠ ĐỘ
# =============================================================================

def gan(df: pd.DataFrame, api: pd.DataFrame,
        cot_ma_tin: str | None = None) -> pd.DataFrame:
    """Trả về bản sao của `df` có thêm toạ độ và nhãn nguồn.

    Duyệt năm mức từ tin cậy nhất xuống, mỗi mức chỉ điền vào những dòng còn
    trống. Nhờ vậy một dòng luôn nhận mức tốt nhất mà nó với tới được.
    """
    out = df.copy()
    out["k_quan"] = out["quan_huyen"].map(chuan_quan)
    out["k_phuong"] = out["phuong_xa"].map(chuan_phuong)
    out["k_duong"] = out["duong_pho"].map(chuan_duong)
    out["k_phuong_moi"] = (out["phuong_moi"].map(chuan_phuong)
                           if "phuong_moi" in out.columns else None)
    # Ưu tiên tên dự án lấy từ TRƯỜNG ĐỊA CHỈ có cấu trúc (batdongsan) hơn tên
    # suy từ tiêu đề tin — trường có cấu trúc do sàn gắn, ít sai hơn nhiều.
    cot_du_an = "du_an_clean" if "du_an_clean" in out.columns else None
    k = out[cot_du_an].map(chuan_du_an) if cot_du_an else pd.Series(
        [None] * len(out), index=out.index, dtype=object)
    if "du_an_dia_chi" in out.columns:
        k = out["du_an_dia_chi"].map(chuan_du_an).fillna(k)
    out["k_du_an"] = k

    da_va = va_dia_gioi(out, set(api["k_quan"].dropna().unique()))
    if da_va:
        print(f"  đã vá {da_va} dòng có tên quận/huyện hỏng (lấy lại từ dia_chi_goc)")

    # GIỮ toạ độ mà nguồn đã cung cấp sẵn (batdongsan bóc từ khối JS). Trước
    # đây hàm này gán np.nan cho cả cột rồi mới điền — tức là xoá sạch toạ độ
    # chính xác rồi thay bằng ước lượng trung vị phường. Sai âm thầm và tệ.
    if "latitude" in out.columns and out["latitude"].notna().any():
        # TOẠ ĐỘ NGUỒN CUNG CẤP NHƯNG RƠI NGOÀI HÀ NỘI THÌ BỎ, KHÔNG SỬA.
        #
        # Có tin ghi kinh độ 106,853 thay vì 105,853 — lệch một chữ số, đẩy căn
        # nhà ở Đồng Mai sang tận Hải Dương, cách 104 km. Trước đây chỗ này chỉ
        # ĐẾM rồi in ra "toạ độ lạc ra ngoài khung Hà Nội: 1" và vẫn dùng tiếp:
        # bản đồ mọc một ghim ở tỉnh khác, và điểm đó còn làm lệch mọi phép đo
        # lấy nó làm mốc tham chiếu cho tên đường.
        #
        # KHÔNG tự sửa 106 thành 105 cho "hợp lý". Đoán một chữ số là bịa ra dữ
        # liệu mình không có. Bỏ toạ độ đi thì bậc dưới của thang (theo đường,
        # theo phường) tự điền lại — đúng như mọi tin không khai toạ độ.
        lac = out["latitude"].notna() & out["longitude"].notna() & ngoai_khung(out)
        if lac.any():
            print(f"  bỏ {int(lac.sum())} toạ độ nguồn cung cấp nhưng rơi ngoài "
                  f"Hà Nội (để bậc dưới điền lại, KHÔNG đoán sửa chữ số)")
            out.loc[lac, ["latitude", "longitude"]] = np.nan

        co_san = out["latitude"].notna() & out["longitude"].notna()
        out["nguon_toa_do"] = np.where(co_san, "nguon_goc", None)
        print(f"  giữ nguyên {int(co_san.sum())} toạ độ chính xác do nguồn cung cấp")
    else:
        out["latitude"] = np.nan
        out["longitude"] = np.nan
        out["nguon_toa_do"] = pd.Series([None] * len(out), dtype=object)
    out["so_tin_tham_chieu"] = np.nan

    def _dien(nhan: str, khoa: list[str], bang: pd.DataFrame) -> None:
        con_trong = out["latitude"].isna()
        if not con_trong.any() or bang.empty:
            return
        tra = out.loc[con_trong, khoa].join(bang, on=khoa)
        co = tra["latitude"].notna()
        idx = tra.index[co]
        out.loc[idx, "latitude"] = tra.loc[co, "latitude"].values
        out.loc[idx, "longitude"] = tra.loc[co, "longitude"].values
        out.loc[idx, "so_tin_tham_chieu"] = tra.loc[co, "so_tin_tham_chieu"].values
        out.loc[idx, "nguon_toa_do"] = nhan

    # Mức 1: trùng đúng mã tin — tin cũ và tin mới là MỘT, lấy nguyên toạ độ.
    if cot_ma_tin and cot_ma_tin in out.columns:
        bang = (api.assign(_k=api["list_id"].astype(str))
                .set_index("_k")[["latitude", "longitude"]]
                .assign(so_tin_tham_chieu=1))
        bang = bang[~bang.index.duplicated()]
        out["_ma"] = out[cot_ma_tin].astype(str)
        _dien("ma_tin", ["_ma"], bang)
        out = out.drop(columns="_ma")

    if cot_du_an:
        _dien("du_an", ["k_du_an"], _bang_tra(api, ["k_du_an"]))
    _dien("duong", ["k_quan", "k_phuong", "k_duong"],
          _bang_tra(api, ["k_quan", "k_phuong", "k_duong"]))
    # Thứ tự xếp theo SAI SỐ ĐO ĐƯỢC, không theo trực giác: "đường trong phường
    # mới" (p90 1,17 km) thực ra kém hơn "phường cũ" (0,79 km), vì một phường
    # mới gộp 3–5 phường cũ nên cùng một tên đường trải trên vùng rộng hơn.
    _dien("phuong", ["k_quan", "k_phuong"], _bang_tra(api, ["k_quan", "k_phuong"]))
    _dien("duong_moi", ["k_quan", "k_phuong_moi", "k_duong"],
          _bang_tra(api, ["k_quan", "k_phuong_moi", "k_duong"]))
    _dien("phuong_moi", ["k_quan", "k_phuong_moi"],
          _bang_tra(api, ["k_quan", "k_phuong_moi"]))
    _dien("duong_quan", ["k_quan", "k_duong"], _bang_tra(api, ["k_quan", "k_duong"]))
    _dien("quan", ["k_quan"], _bang_tra(api, ["k_quan"]))

    out["sai_so_p90_km"] = out["nguon_toa_do"].map(SAI_SO_P90_KM)
    tra_pm = bang_phuong_moi(api)
    if "phuong_moi" in out.columns:
        # Nguồn đã tự công bố tên phường mới (batdongsan) thì tin nguồn, chỉ vá
        # những dòng còn trống bằng bảng tra suy từ API.
        vá = out.join(tra_pm, on=["k_quan", "k_phuong"], rsuffix="_tra")
        out["phuong_moi"] = out["phuong_moi"].fillna(vá["phuong_moi_tra"])
    else:
        out = out.join(tra_pm, on=["k_quan", "k_phuong"])
    return out.drop(columns=["k_quan", "k_phuong", "k_duong", "k_du_an",
                             "k_phuong_moi"])


# =============================================================================
# KIỂM ĐỊNH
# =============================================================================

def kiem_dinh(api: pd.DataFrame) -> pd.DataFrame:
    """Đo sai số bằng leave-one-out ngay trên dữ liệu API.

    Với mỗi tin đã biết toạ độ thật: giấu nó đi, tính trung vị nhóm từ những
    tin còn lại, rồi đo xem đoán lệch bao nhiêu km. Đây là cách duy nhất trung
    thực để công bố con số sai số — vì mình chưa từng cho nhóm nhìn thấy đáp án.
    """
    hang = []
    muc = [("du_an", ["k_du_an"]),
           ("duong", ["k_quan", "k_phuong", "k_duong"]),
           ("duong_moi", ["k_quan", "k_phuong_moi", "k_duong"]),
           ("phuong", ["k_quan", "k_phuong"]),
           ("phuong_moi", ["k_quan", "k_phuong_moi"]),
           ("duong_quan", ["k_quan", "k_duong"]),
           ("quan", ["k_quan"])]
    for nhan, khoa in muc:
        d = api.dropna(subset=khoa)
        res = []
        for _, sub in d.groupby(khoa):
            if len(sub) < 2:
                continue
            lat, lon = sub["latitude"].values, sub["longitude"].values
            for i in range(len(sub)):
                m = np.ones(len(sub), bool)
                m[i] = False
                res.append(haversine_km(lat[i], lon[i],
                                        np.median(lat[m]), np.median(lon[m])))
        r = np.array(res)
        hang.append({
            "muc": nhan, "so_tin_kiem": len(r),
            "p50_km": round(float(np.median(r)), 2),
            "p90_km": round(float(np.percentile(r, 90)), 2),
            "p95_km": round(float(np.percentile(r, 95)), 2),
            "duoi_0_5km": round(float((r < 0.5).mean()), 4),
            "duoi_1km": round(float((r < 1).mean()), 4),
            "duoi_2km": round(float((r < 2).mean()), 4),
        })
    return pd.DataFrame(hang)


def kiem_dinh_doi_chung(api: pd.DataFrame) -> pd.DataFrame:
    """Kiểm định NGOÀI MẪU thật sự, trên chính dữ liệu cũ.

    244 tin nhà đất cũ vẫn còn treo trên sàn nên xuất hiện lại trong dữ liệu
    API — tức là mình biết toạ độ thật của chúng. Giấu đáp án đi, chạy y hệt
    quy trình dành cho tin không có mã trùng, rồi đối chiếu.

    Khác với `kiem_dinh` (đo trong nội bộ dữ liệu API), phép đo này chạy trên
    đúng loại địa chỉ mà tin cũ có — chữ nghĩa lem nhem, tên phường cũ, ngõ
    ngách lồng nhau — nên nó đại diện đúng cho sai số thực tế.
    """
    df = pd.read_excel(DATA_CLEAN / CAU_HINH["nhadat"]["vao"])
    that = (api.assign(_k=api["list_id"].astype(str)).set_index("_k")
            [["latitude", "longitude"]])
    that = that[~that.index.duplicated()]
    df["_k"] = df["listing_id"].astype(str)
    co = df[df["_k"].isin(that.index)].copy()
    if co.empty:
        return pd.DataFrame()

    out = gan(co, api, cot_ma_tin=None)          # cố tình KHÔNG dùng mã tin
    out = out.join(that.rename(columns={"latitude": "lat_that",
                                        "longitude": "lon_that"}), on="_k")
    out["lech_km"] = haversine_km(out["latitude"], out["longitude"],
                                  out["lat_that"], out["lon_that"])

    hang = []
    for nhan, s in list(out.groupby("nguon_toa_do")) + [("TỔNG", out)]:
        r = s["lech_km"].dropna()
        hang.append({"muc": nhan, "so_tin_kiem": len(r),
                     "p50_km": round(float(r.median()), 2),
                     "p90_km": round(float(r.quantile(.9)), 2),
                     "duoi_1km": round(float((r < 1).mean()), 4),
                     "duoi_2km": round(float((r < 2).mean()), 4)})
    return pd.DataFrame(hang)


def ve_ban_do(cac_bang: dict[str, pd.DataFrame], duong_dan: Path) -> None:
    """Vẽ điểm đã gán lên mặt phẳng lat/lon, tô màu theo mức tin cậy.

    Không phải bản đồ nền, chỉ là đám mây điểm — nhưng nhìn hình dạng Hà Nội
    hiện ra là biết ngay có toạ độ nào lạc sang tỉnh khác không."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mau = {"ma_tin": "#1b5e20", "du_an": "#2e7d32", "duong": "#1565c0",
           "phuong": "#ef6c00", "duong_quan": "#8e24aa", "quan": "#c62828"}
    fig, axes = plt.subplots(1, len(cac_bang), figsize=(6.5 * len(cac_bang), 6.5))
    axes = np.atleast_1d(axes)
    for ax, (ten, d) in zip(axes, cac_bang.items()):
        for nhan, s in d.groupby("nguon_toa_do"):
            ax.scatter(s["longitude"], s["latitude"], s=6, alpha=.55,
                       c=mau.get(nhan, "#666"),
                       label=f"{nhan} ({len(s)})")
        ax.set_title(f"{ten} — {len(d)} tin đã có toạ độ")
        ax.set_xlabel("kinh độ"); ax.set_ylabel("vĩ độ")
        ax.legend(markerscale=2.5, fontsize=8, loc="lower left")
        ax.set_aspect(1 / np.cos(np.radians(21.0)))
    fig.tight_layout()
    fig.savefig(duong_dan, dpi=140)
    plt.close(fig)


def ngoai_khung(d: pd.DataFrame) -> pd.Series:
    lo_v, hi_v, lo_k, hi_k = HN_BOX
    return ~(d["latitude"].between(lo_v, hi_v) & d["longitude"].between(lo_k, hi_k))


def bao_cao(out: pd.DataFrame, ten: str) -> pd.DataFrame:
    v = out["nguon_toa_do"].value_counts(dropna=False)
    bc = pd.DataFrame({"nguon_toa_do": v.index.astype(str), "so_dong": v.values})
    bc["ti_le"] = (bc["so_dong"] / len(out)).round(4)
    bc["sai_so_p90_km"] = bc["nguon_toa_do"].map(SAI_SO_P90_KM)
    bc.insert(0, "nhanh", ten)
    return bc


# =============================================================================
# CHẠY
# =============================================================================

CAU_HINH = {
    "chungcu": {"vao": "chungcu_clean.xlsx", "ra": "chungcu_clean_geo.xlsx",
                "ma_tin": None},
    # Tập chung cư ĐÃ GỘP nhiều nguồn — đây mới là tập bản chạy thật dùng.
    # Bản đầu tôi quên đưa nó vào bảng này, nên `python gan_toa_do.py` chạy
    # xong vẫn để lại `chungcu_gop_geo.xlsx` của lần trước, và mọi thứ phía sau
    # âm thầm dùng dữ liệu cũ mà không có lỗi nào báo ra.
    "chungcu-gop": {"vao": "chungcu_gop.xlsx", "ra": "chungcu_gop_geo.xlsx",
                    "ma_tin": "listing_id"},
    "nhadat": {"vao": "nhadat_clean.xlsx", "ra": "nhadat_clean_geo.xlsx",
               "ma_tin": "listing_id"},
    "nhadat-gop": {"vao": "nhadat_gop.xlsx", "ra": "nhadat_gop_geo.xlsx",
                   "ma_tin": "listing_id"},
}


def chay(loai: str, api: pd.DataFrame) -> pd.DataFrame:
    ch = CAU_HINH[loai]
    df = pd.read_excel(DATA_CLEAN / ch["vao"])
    out = gan(df, api, ch["ma_tin"])
    # Tập gộp đã mang sẵn ngày quan sát RIÊNG cho từng đợt cào; ghi đè bằng
    # một ngày duy nhất là xoá mất thông tin đó, và mọi phép đo theo thời gian
    # sau này sẽ thấy toàn bộ dữ liệu cùng rơi vào một ngày.
    if "ngay_quan_sat" not in out.columns or out["ngay_quan_sat"].isna().all():
        out["ngay_quan_sat"] = pd.Timestamp(NGAY_QUAN_SAT.get(loai, "2026-06-13"))
    out.to_excel(DATA_CLEAN / ch["ra"], index=False)

    bc = bao_cao(out, loai)
    print(f"\n{loai}: {len(out)} dòng -> {ch['ra']}")
    for _, r in bc.iterrows():
        print(f"  {r['nguon_toa_do']:<8} {r['so_dong']:>5}  {r['ti_le']:>6.1%}"
              f"   (p90 ≈ {r['sai_so_p90_km']} km)")
    thieu = out["latitude"].isna().sum()
    print(f"  có toạ độ: {len(out) - thieu}/{len(out)} "
          f"({1 - thieu / len(out):.1%})")
    print(f"  có tên phường mới: {out['phuong_moi'].notna().sum()}/{len(out)}")

    # CHỐT CHẶN: BƯỚC NÀY CHƯA PHẢI BƯỚC CUỐI
    #
    # Khi không tra được phường mới, hàm này chép nguyên tên phường CŨ sang cột
    # `phuong_moi`. Đó là lựa chọn có ý thức (thà có tên cũ hơn để trống), NHƯNG
    # nó bắt buộc phải chạy tiếp `gazetteer_phuong.py --ap-dung` để quy về 113
    # phường sau sáp nhập.
    #
    # Bỏ sót bước đó thì không có lỗi nào báo ra: dữ liệu vẫn đủ dòng, vẫn đủ
    # toạ độ, chỉ có tên phường trộn hai hệ. Hậu quả là mã hoá mục bị chia vụn
    # và bản đồ theo phường mất dữ liệu vì tên không khớp ranh giới nào. Tôi đã
    # bỏ sót đúng bước này một lần và chỉ phát hiện nhờ đếm ra 171 phường.
    #
    # Dấu hiệu nhận biết duy nhất là số phường vượt 113 — nên in nó ra, to.
    so_phuong = int(out["phuong_moi"].nunique())
    if so_phuong > SO_PHUONG_HA_NOI:
        print(f"\n  !!! {so_phuong} tên phường, trong khi Hà Nội sau sáp nhập "
              f"chỉ có {SO_PHUONG_HA_NOI}.")
        print(f"  !!! Dữ liệu đang trộn tên trước và sau sáp nhập. CHƯA DÙNG "
              f"ĐƯỢC.")
        print(f"  !!! Chạy tiếp:  python gazetteer_phuong.py --ap-dung")
    else:
        print(f"  số phường: {so_phuong}/{SO_PHUONG_HA_NOI} (hệ tên sau sáp nhập)")

    lac = int(ngoai_khung(out).sum())
    print(f"  toạ độ lạc ra ngoài khung Hà Nội: {lac}")
    return out, bc


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--loai",
                    choices=["chungcu", "chungcu-gop", "nhadat", "nhadat-gop",
                             "ca-hai"],
                    default="ca-hai")
    ap.add_argument("--kiem-dinh", action="store_true",
                    help="in bảng sai số leave-one-out rồi ghi ra Excel")
    ap.add_argument("--ban-do", action="store_true",
                    help="vẽ đám mây điểm để kiểm tra bằng mắt")
    a = ap.parse_args()

    api = nap_api()
    print(f"Bảng tra dựng từ {len(api)} tin API có toạ độ "
          f"({api['k_duong'].notna().sum()} tin có tên đường).")

    if a.kiem_dinh:
        kd = kiem_dinh(api)
        print("\nSai số leave-one-out (trong nội bộ dữ liệu API):")
        print(kd.to_string(index=False))
        dc = kiem_dinh_doi_chung(api)
        print("\nSai số ngoài mẫu (244 tin nhà đất cũ còn treo trên sàn):")
        print(dc.to_string(index=False))
        with pd.ExcelWriter(DATA_CLEAN / "kiem_dinh_gan_toa_do.xlsx") as w:
            kd.to_excel(w, sheet_name="leave_one_out", index=False)
            dc.to_excel(w, sheet_name="ngoai_mau", index=False)

    loai = ["chungcu", "nhadat"] if a.loai == "ca-hai" else [a.loai]
    ket = {l: chay(l, api) for l in loai}
    pd.concat([v[1] for v in ket.values()], ignore_index=True).to_excel(
        DATA_CLEAN / "bao_cao_gan_toa_do.xlsx", index=False)
    print("\nBáo cáo -> data_clean/bao_cao_gan_toa_do.xlsx")

    if a.ban_do:
        p = DATA_CLEAN / "ban_do_toa_do.png"
        ve_ban_do({k: v[0].dropna(subset=["latitude"]) for k, v in ket.items()}, p)
        print(f"Bản đồ  -> {p.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    main()
