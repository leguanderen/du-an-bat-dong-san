"""
nguon_batdongsan.py — Bộ chuyển đổi dữ liệu batdongsan.com.vn về đúng khuôn
dạng mà `clean_chungcu.py` đang nhận.

VÌ SAO LÀM BỘ CHUYỂN ĐỔI CHỨ KHÔNG VIẾT MỘT LUỒNG LÀM SẠCH THỨ HAI
------------------------------------------------------------------
Viết `clean_batdongsan.py` riêng thì phải chép lại toàn bộ 250 dòng logic đã
được kiểm chứng: sửa lỗi đơn vị giá, khử trùng lặp ba tầng, lọc ngoại lai theo
IQR trong quận, vá tầng số từ mô tả. Chép là hỏng — sáu tháng sau sửa một chỗ
thì quên chỗ kia, và hai nguồn âm thầm được làm sạch theo hai chuẩn khác nhau.

Cách ở đây: mỗi nguồn có một bộ chuyển đổi mỏng, nhiệm vụ duy nhất là dịch
khuôn dạng của nó sang khuôn dạng chung. Sau đó **một** luồng làm sạch chạy cho
tất cả. Thêm nguồn thứ ba sau này chỉ tốn thêm một file mỏng như file này.

BA KHÁC BIỆT PHẢI DỊCH
----------------------
1. Tên nhãn: batdongsan ghi "Số phòng tắm, vệ sinh", nhatot ghi "Số phòng vệ
   sinh"; "Pháp lý" vs "Giấy tờ pháp lý"; "Nội thất" vs "Tình trạng nội thất".

2. Giá trị hạng mục: "Sổ đỏ/ Sổ hồng" và "Sổ hồng riêng" là cùng một thứ, nhưng
   với model thì là hai hạng mục khác nhau — mỗi hạng mục lại chia nhỏ dữ liệu
   ra khi làm target encoding. Phải quy về một tên.

3. Địa chỉ: batdongsan ghi tên phường MỚI (sau sáp nhập 1/7/2025) kèm tên quận
   CŨ trong ngoặc:

       "Khu đô thị Việt Hưng, Phường Việt Hưng, Hà Nội (Quận Long Biên, Hà Nội cũ)"

   Đây thực ra là món quà: nó cho luôn bảng đối chiếu cũ ↔ mới. Nhưng nghĩa là
   cột `phuong_xa` của nguồn này là tên MỚI, còn của nhatot là tên CŨ — không so
   khớp trực tiếp được. Cách xử lý: dùng `phuong_moi` làm khoá địa lý chung cho
   mọi nguồn (dữ liệu nhatot đã có cột này từ `gan_toa_do.py`).

   Đo thử thì khoá mới còn TỐT HƠN: gom về 62 phường thay vì 144, MAPE chung cư
   16,00% vs 16,01%, nhà đất 27,24% vs 27,75% — vì mỗi phường nhiều mẫu hơn nên
   target encoding bớt nhiễu.

KHÔNG KIỂM CHỨNG ĐƯỢC PARSER BẰNG Giá/m²
----------------------------------------
Với nhatot, mình đối chiếu giá × diện tích với Giá/m² do site tự công bố — 0
dòng lệch quá 1%, nên biết chắc parser đọc đúng. batdongsan không công bố
Giá/m² trong khối này. Thay thế: đối chiếu với giá nhắc trong TIÊU ĐỀ tin
("...chỉ 4.2 tỷ"), yếu hơn nhưng vẫn bắt được lỗi đơn vị hệ thống. Hàm
`kiem_chung_tieu_de` làm việc đó và in tỉ lệ khớp.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from extract_text import extract_price_hint
from gan_toa_do import thu_muc

DATA_RAW = thu_muc("data_raw")

# Nhãn batdongsan -> nhãn nhatot. Nhãn nào không có trong bảng này thì bỏ qua.
# Cố tình BỎ "Mức giá": trong nhiều tin nó là "Mức giá điện", không phải giá nhà.
DOI_NHAN = {
    "Diện tích": "Diện tích",
    "Số phòng ngủ": "Số phòng ngủ",
    "Số phòng tắm, vệ sinh": "Số phòng vệ sinh",
    "Hướng nhà": "Hướng cửa chính",
    "Hướng ban công": "Hướng ban công",
    "Pháp lý": "Giấy tờ pháp lý",
    "Nội thất": "Tình trạng nội thất",
    "Số tầng": "Tầng số",
    # HAI TRƯỜNG CHỈ NHÀ ĐẤT MỚI CÓ, và chúng là thứ đáng giá nhất ở nhánh này.
    #
    # Bảng tầm quan trọng đặc trưng cho thấy model nhà đất chưa nắm được yếu tố
    # chính (không đặc trưng nào vượt 13%), và nghi vấn là vị trí vi mô: nhà
    # trong ngõ hay mặt phố, ngõ rộng mấy mét, ô tô vào được không.
    #
    # Ở nhatot hai thứ đó phải suy ra từ mô tả bằng regex: `mat_tien_m` đầy 50%,
    # `duong_rong_m` chỉ 7%. batdongsan cho SẴN dưới dạng trường có cấu trúc —
    # "Mặt tiền" xuất hiện ở 65% tin, "Đường vào" ở 47%. Bỏ qua chúng là vứt đi
    # đúng thứ mình đang thiếu.
    "Mặt tiền": "Chiều ngang",
    "Đường vào": "Đường vào",
}

# Mọi nhãn có thể xuất hiện trong khối — cần đủ để cắt đúng ranh giới giữa các
# trường, kể cả nhãn mình không dùng.
NHAN_BDS = list(DOI_NHAN) + ["Khoảng giá", "Mức giá"]

DOI_PHAP_LY = {
    "Sổ đỏ/ Sổ hồng": "Sổ hồng riêng", "Sổ đỏ": "Sổ hồng riêng",
    "Sổ hồng": "Sổ hồng riêng", "Có sổ": "Sổ hồng riêng",
    "Sổ đỏ/Sổ hồng": "Sổ hồng riêng",
    "Hợp đồng mua bán": "Hợp đồng mua bán",
    "Đang chờ sổ": "Đang chờ sổ", "Giấy tờ khác": np.nan,
}
DOI_NOI_THAT = {
    "Đầy đủ": "Nội thất đầy đủ", "Full": "Nội thất đầy đủ",
    "Cao cấp": "Nội thất cao cấp",
    "Cơ bản": "Hoàn thiện cơ bản",
    "Không nội thất": "Bàn giao thô", "Bàn giao thô": "Bàn giao thô",
    "Nguyên bản": "Bàn giao thô",
}


def _gon(s: object) -> str:
    return re.sub(r"\s+", " ", str(s)).strip() if isinstance(s, str) else ""


def tach_khoi(blob: object) -> dict:
    """Khối đặc điểm của batdongsan -> {nhãn: giá trị}.

    Khối là một chuỗi dài xuống dòng lung tung, nhãn và giá trị nằm sát nhau.
    Cắt bằng cách tìm nhãn kế tiếp làm điểm dừng — không dựa vào xuống dòng,
    vì số lượng dấu xuống dòng thay đổi theo từng tin.
    """
    t = _gon(blob)
    if not t:
        return {}
    dung = "|".join(re.escape(x) for x in sorted(NHAN_BDS, key=len, reverse=True))
    out = {}
    for nhan in NHAN_BDS:
        m = re.search(re.escape(nhan) + r"\s+(.+?)(?=\s+(?:" + dung + r")\s|$)", t)
        if m:
            out[nhan] = m.group(1).strip()
    return out


# Phần đầu địa chỉ batdongsan có cấu trúc rất đều, phần cuối là tên đường:
#   "Khu đô thị Việt Hưng"                        -> chỉ có dự án
#   "The Diamond Residence, Đường Lê Văn Lương"   -> dự án + đường
#   "số 72A, Royal City, Đường Nguyễn Trãi"       -> số nhà + dự án + đường
_RE_DUONG = re.compile(r"^(Đường|Phố|Ngõ|Ngách|Hẻm)\s+", re.I)


def _tach_duong_va_du_an(truoc: str) -> tuple[str, str]:
    """Tách phần đầu địa chỉ thành (tên đường, tên dự án).

    Không tách thì cả cụm "The Diamond Residence, Đường Lê Văn Lương" bị coi là
    một tên đường duy nhất — và nó không khớp với bất kỳ tên đường nào trong
    bảng tra, nên tin rơi thẳng xuống mức phường. Đây chính là lý do chỉ 12,2%
    tin batdongsan có được toạ độ mức đường trước khi sửa.

    Bù lại, tên dự án lấy được ở đây đáng tin hơn hẳn tên suy từ tiêu đề tin —
    nó là trường có cấu trúc do sàn tự gắn, không phải chữ người bán tự viết.
    """
    phan = [x.strip() for x in truoc.split(",") if x.strip()]
    duong = next((x for x in reversed(phan) if _RE_DUONG.match(x)), "")
    con_lai = [x for x in phan if x != duong]
    # Bỏ mẩu chỉ có số nhà ("số 72A", "360 giải phóng") — không phải tên dự án.
    du_an = next((x for x in reversed(con_lai)
                  if not re.match(r"^(số\s*)?\d", x, re.I)), "")
    if not duong and not du_an and phan:
        duong = phan[-1]
    return duong, du_an


def tach_dia_chi(raw: object) -> dict:
    """ "Đường X, Phường Y, Hà Nội (Quận Z, Hà Nội cũ)" -> các phần rời."""
    trong = {"dia_chi": np.nan, "phuong_moi": np.nan, "quan_cu": np.nan,
             "du_an_dia_chi": np.nan}
    t = _gon(raw)
    if not t:
        return trong
    m = re.search(r"\((.+?),\s*Hà Nội\s*cũ\)", t)
    quan = m.group(1).strip() if m else np.nan
    truoc = t[:m.start()].strip().rstrip(", ") if m else t
    truoc = re.sub(r",?\s*Hà Nội\s*$", "", truoc).strip().rstrip(",")
    mp = re.search(r"(Phường|Xã|Thị trấn)\s+[^,]+", truoc)
    phuong = mp.group(0).strip() if mp else np.nan
    dau = truoc[:mp.start()].strip().rstrip(",") if mp else truoc
    duong, du_an = _tach_duong_va_du_an(dau)
    # Dựng lại địa chỉ theo đúng khuôn nhatot để dùng chung bộ parse địa chỉ.
    phan = [p for p in (duong, phuong, quan, "Hà Nội") if isinstance(p, str) and p]
    return {"dia_chi": ", ".join(phan), "phuong_moi": phuong, "quan_cu": quan,
            "du_an_dia_chi": du_an or np.nan}


# =============================================================================
# BÓC TOẠ ĐỘ TỪ KHỐI JAVASCRIPT
# =============================================================================
# Trang chi tiết batdongsan nhúng thẳng toạ độ vào một khối JS tĩnh trong
# `.re__main-content`, dạng:
#
#     ... latitude: 21.0289486097874, longitude: 105.8123456, ...
#
# Khoá KHÔNG có dấu nháy (object literal của JavaScript, không phải JSON), nên
# đừng cố json.loads — chỉ cần bắt bằng biểu thức chính quy.

_RE_LAT = re.compile(r"\blatitude\s*:\s*(-?\d{1,3}(?:\.\d+)?)")
_RE_LON = re.compile(r"\blongitude\s*:\s*(-?\d{1,3}(?:\.\d+)?)")


def boc_toa_do(text: object) -> tuple[float, float]:
    """Khối JS -> (vĩ độ, kinh độ). Không thấy thì trả (nan, nan).

    Chặn luôn giá trị 0/0 và giá trị ngoài Việt Nam: vài trang trả về toạ độ
    mặc định khi người đăng không ghim bản đồ, và 0/0 nằm giữa Đại Tây Dương —
    lọt vào bảng tra là kéo trung vị đi rất xa.
    """
    if not isinstance(text, str):
        return (np.nan, np.nan)
    mla, mlo = _RE_LAT.search(text), _RE_LON.search(text)
    if not mla or not mlo:
        return (np.nan, np.nan)
    lat, lon = float(mla.group(1)), float(mlo.group(1))
    if not (8.0 <= lat <= 24.0 and 102.0 <= lon <= 110.0):
        return (np.nan, np.nan)
    return (lat, lon)


def gop_toa_do(tho: pd.DataFrame, toa_do: pd.DataFrame,
               cot_khoi: str = "toa_do") -> pd.DataFrame:
    """Ghép bảng toạ độ (cào bằng sitemap riêng) vào bảng chính theo mã tin.

    Sitemap toạ độ bật ô "Nhiều" nên mỗi tin ra vài dòng, chỉ một dòng trong đó
    có `latitude`. Hàm này lọc lấy dòng ấy rồi ghép theo mã tin lấy từ URL —
    khoá chắc chắn, không phụ thuộc thứ tự dòng giữa hai lần cào.
    """
    def _ma(s):
        return s.astype(str).str.extract(r"-pr(\d+)")[0]

    t = toa_do.copy()
    cot_url = "click" if "click" in t.columns else "web_scraper_start_url"
    t["_ma"] = _ma(t[cot_url])
    t[["latitude", "longitude"]] = t[cot_khoi].map(boc_toa_do).apply(pd.Series)
    t = (t.dropna(subset=["latitude"]).drop_duplicates("_ma")
          [["_ma", "latitude", "longitude"]])

    out = tho.copy()
    out["_ma"] = _ma(out["click"] if "click" in out.columns
                     else out["web_scraper_start_url"])
    out = out.merge(t, on="_ma", how="left").drop(columns="_ma")
    print(f"  ghép toạ độ chính xác: {out['latitude'].notna().sum()}/{len(out)} tin")
    return out


def bao_cao_thieu_toa_do(tho: pd.DataFrame, duong_dan: Path | None = None) -> pd.DataFrame:
    """Liệt kê những tin KHÔNG bóc được toạ độ, kèm URL để cào lại.

    VÌ SAO CẦN
    ----------
    Không selector nào chắc chắn đúng trên cả 11.000 trang: chỉ cần một trang
    đổi bố cục, hay thiếu thuộc tính `type`, là khối JS lọt lưới. Cố tìm một
    selector hoàn hảo là cuộc chiến không thắng được — và tệ hơn, khi nó lọt
    thì lọt IM LẶNG, dòng đó vẫn có đủ giá, diện tích, chỉ thiếu toạ độ và
    không ai để ý.

    Cách xử lý đúng là chấp nhận sẽ lọt vài phần trăm, nhưng làm cho chúng
    HIỆN RA: hàm này xuất đúng danh sách URL còn thiếu. Cào lại vài chục trang
    đó bằng selector rộng hơn thì rẻ; cào lại cả 11.000 trang thì không.
    """
    if "latitude" not in tho.columns:
        thieu = tho.copy()
    else:
        thieu = tho[tho["latitude"].isna()].copy()
    cot = [c for c in ("click", "url_goc", "listing_id", "ten_du_an", "dia_chi")
           if c in thieu.columns]
    out = thieu[cot] if cot else thieu
    ti_le = len(thieu) / max(len(tho), 1)
    print(f"  thiếu toạ độ: {len(thieu)}/{len(tho)} tin ({ti_le:.1%})")
    if ti_le > 0.10:
        print("  ⚠ trên 10% — nhiều khả năng selector sai chứ không phải "
              "vài trang cá biệt. Kiểm lại selector trước khi cào tiếp.")
    if duong_dan is not None and len(out):
        out.to_excel(duong_dan, index=False)
        print(f"  -> danh sách cần cào lại: {duong_dan}")
    return out


def _dung_huong(v: object) -> object:
    """"Đông - Nam" -> "Đông Nam" (nhatot viết liền)."""
    if not isinstance(v, str):
        return np.nan
    return re.sub(r"\s*-\s*", " ", v).strip() or np.nan


def _dung(v: object, bang: dict) -> object:
    if not isinstance(v, str):
        return np.nan
    return bang.get(v.strip().rstrip("."), np.nan)


def gom_dong(df: pd.DataFrame) -> pd.DataFrame:
    """Gộp nhiều dòng của cùng một tin về một dòng, giữ dòng CÓ toạ độ.

    Selector toạ độ bật ô "Nhiều" nên mỗi tin ra ~3 dòng: cùng giá, cùng địa
    chỉ, chỉ khác nội dung khối JavaScript, và chỉ MỘT dòng trong đó chứa
    `latitude`. Không gộp thì mọi thống kê sau đó đếm mỗi căn hộ ba lần.
    """
    if "click" not in df.columns:
        return df
    d = df.copy()
    d["_ma"] = d["click"].astype(str).str.extract(r"-pr(\d+)")[0]
    if not d["_ma"].duplicated().any():
        return df
    if "toa_do" in d.columns:
        # Sắp dòng có toạ độ lên đầu mỗi nhóm rồi giữ dòng đầu.
        d["_co"] = d["toa_do"].astype(str).str.contains(r"latitude\s*:")
        d = d.sort_values("_co", ascending=False)
    truoc = len(d)
    d = d.drop_duplicates("_ma", keep="first").drop(
        columns=[c for c in ("_ma", "_co") if c in d.columns])
    print(f"  gộp {truoc} dòng -> {len(d)} tin")
    return d.reset_index(drop=True)


# Slug trong URL batdongsan -> nhãn loại hình của nhatot. Xếp slug DÀI trước
# slug ngắn: "ban-nha-mat-pho" phải khớp trước "ban-nha", nếu không mọi tin đều
# rơi vào nhánh đầu tiên.
SLUG_LOAI_HINH = (
    ("ban-nha-biet-thu-lien-ke", "Nhà phố liền kề"),
    ("ban-biet-thu-lien-ke", "Nhà phố liền kề"),
    ("ban-nha-mat-pho", "Nhà mặt phố, mặt tiền"),
    ("ban-shophouse", "Nhà mặt phố, mặt tiền"),
    ("ban-nha-rieng", "Nhà ngõ, hẻm"),
)


def _loai_hinh_tu_url(df: pd.DataFrame, loai: str) -> pd.Series:
    """Đọc loại hình từ slug URL. Chung cư thì cố định vì mục cào là cIds=650."""
    if loai == "chungcu":
        return pd.Series("Chung cư", index=df.index)
    u = df.get("click", pd.Series("", index=df.index)).astype(str)
    ra = pd.Series([None] * len(df), index=df.index, dtype=object)
    for slug, nhan in SLUG_LOAI_HINH:
        khop = u.str.contains(slug, na=False) & ra.isna()
        ra[khop] = nhan
    # Không nhận ra slug thì để "Nhà ngõ, hẻm" — nhóm áp đảo (66% dữ liệu hiện
    # có) và cũng là mặc định an toàn nhất, thay vì vứt tin đi.
    return ra.fillna("Nhà ngõ, hẻm")


def chuyen_doi(df: pd.DataFrame, loai: str = "chungcu") -> pd.DataFrame:
    """Bảng xuất từ extension -> bảng đúng khuôn `clean_*.load_raw`."""
    df = gom_dong(df)
    bat_buoc = {"ten_du_an", "dia_chi", "gia_ban", "dien_tich",
                "dac_diem_tong_hop", "mo_ta"}
    thieu = bat_buoc - set(df.columns)
    if thieu:
        raise ValueError(f"File thiếu cột bắt buộc: {sorted(thieu)}")

    khoi = df["dac_diem_tong_hop"].map(tach_khoi).apply(pd.Series)
    dc = df["dia_chi"].map(tach_dia_chi).apply(pd.Series)

    out = pd.DataFrame(index=df.index)
    out["ten_du_an"] = df["ten_du_an"]
    out["dia_chi"] = dc["dia_chi"]
    out["phuong_moi"] = dc["phuong_moi"]
    out["du_an_dia_chi"] = dc["du_an_dia_chi"]
    out["gia_ban"] = df["gia_ban"]
    out["dien_tich"] = df["dien_tich"]
    out["mo_ta"] = df["mo_ta"]

    # Khối đặc điểm dựng lại theo nhãn nhatot. clean_chungcu bóc lại từ đây, nên
    # chỉ cần nối "nhãn + giá trị" liền nhau đúng như nhatot vẫn xuất.
    gia_tri = {}
    for nhan_bds, nhan_nt in DOI_NHAN.items():
        v = khoi.get(nhan_bds, pd.Series(index=df.index, dtype=object))
        if nhan_nt == "Giấy tờ pháp lý":
            v = v.map(lambda x: _dung(x, DOI_PHAP_LY))
        elif nhan_nt == "Tình trạng nội thất":
            v = v.map(lambda x: _dung(x, DOI_NOI_THAT))
        elif nhan_nt in ("Hướng cửa chính", "Hướng ban công"):
            v = v.map(_dung_huong)
        gia_tri[nhan_nt] = v

    # LOẠI HÌNH suy từ URL của chính tin, không gán cứng.
    #
    # Bản đầu tôi viết `= "Chung cư"` cho mọi dòng, vì lúc đó sitemap chỉ cào
    # mục chung cư. Đến khi dùng lại bộ chuyển đổi này cho nhà đất thì cả 1.172
    # tin nhà riêng đều mang nhãn "Chung cư", và bộ lọc loại hình của
    # clean_nhadat quét sạch — 0 dòng còn lại, không một lỗi nào báo ra.
    #
    # URL tin luôn chứa slug loại hình (.../ban-nha-rieng-duong-x-...), nên suy
    # từ đó vừa đúng vừa tự thích ứng khi cào thêm mục mới.
    gia_tri["Loại hình"] = _loai_hinh_tu_url(df, loai)

    NHAN_GHEP = ("Loại hình", "Diện tích", "Giấy tờ pháp lý", "Số phòng ngủ",
                 "Số phòng vệ sinh", "Tình trạng nội thất", "Hướng ban công",
                 "Hướng cửa chính", "Tầng số", "Chiều ngang", "Đường vào")

    # Chốt chặn: nhãn nào ghép ra thì bước bóc phải biết nhãn đó. Thiếu một
    # nhãn là làm hỏng im lặng trường đứng trước nó (xem kiem_nhan_ghep).
    from clean_common import (BLOB_LABELS_CHUNGCU, BLOB_LABELS_NHADAT,
                              kiem_nhan_ghep)
    kiem_nhan_ghep(NHAN_GHEP,
                   BLOB_LABELS_NHADAT if loai == "nhadat"
                   else BLOB_LABELS_CHUNGCU,
                   f"nguon_batdongsan({loai})")

    def _ghep(i):
        phan = ["Đặc điểm bất động sản"]
        for nhan in NHAN_GHEP:
            v = gia_tri.get(nhan)
            if v is not None and isinstance(v.get(i), str) and v.get(i):
                phan.append(nhan + v.get(i))
        return "".join(phan)

    out["dac_diem_tong_hop"] = [_ghep(i) for i in df.index]

    # Mã tin nằm ở đuôi URL: .../ban-can-ho-...-pr45747194
    if "click" in df.columns:
        out["url_goc"] = df["click"]
        out["listing_id"] = df["click"].astype(str).str.extract(r"-pr(\d+)")[0]
    # Toạ độ chính xác nếu sitemap có lấy khối JS. Cột này thắng mọi ước lượng
    # từ bảng tra — gan_toa_do sẽ thấy latitude đã có sẵn và không đụng vào.
    if "toa_do" in df.columns:
        out[["latitude", "longitude"]] = (df["toa_do"].map(boc_toa_do)
                                          .apply(pd.Series))
        print(f"  bóc được toạ độ chính xác: {out['latitude'].notna().sum()}"
              f"/{len(out)} tin")
    out["nguon"] = "batdongsan"
    return out


def kiem_chung_tieu_de(gia: pd.Series, tieu_de: pd.Series,
                       dung_sai: float = 0.15) -> dict:
    """Đối chiếu giá đã bóc với giá người bán nhắc trong tiêu đề.

    Đây KHÔNG phải kiểm chứng dữ liệu đúng hay sai — người bán vẫn có thể ghi
    một đằng làm một nẻo. Nó kiểm chứng PARSER: nếu bóc sai đơn vị (đọc "4,2 tỷ"
    thành 4,2) thì tỉ lệ khớp sẽ sụp xuống gần 0 và thấy ngay.
    """
    goi_y = tieu_de.map(extract_price_hint)
    co = goi_y.notna() & gia.notna()
    if not co.any():
        return {"so_doi_chieu": 0, "ti_le_khop": float("nan")}
    khop = (gia[co] - goi_y[co]).abs() / goi_y[co] < dung_sai
    return {"so_doi_chieu": int(co.sum()), "ti_le_khop": float(khop.mean())}


def nap(duong_dan: Path) -> pd.DataFrame:
    return chuyen_doi(pd.read_excel(duong_dan))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output", type=Path,
                    help="ghi bảng đã chuyển đổi ra .xlsx để soi bằng mắt")
    a = ap.parse_args()

    tho = pd.read_excel(a.input)
    out = chuyen_doi(tho)
    print(f"Chuyển đổi {len(out)} dòng.")
    for c in ("dia_chi", "phuong_moi", "du_an_dia_chi", "listing_id"):
        if c in out:
            print(f"  {c:12s}: {out[c].notna().sum()}/{len(out)} có giá trị")
    khoi = tho["dac_diem_tong_hop"].map(tach_khoi).apply(pd.Series)
    print("  độ phủ từng trường trong khối đặc điểm:")
    for c in khoi.columns:
        print(f"    {c:24s} {khoi[c].notna().mean()*100:5.1f}%")
    if a.output:
        out.to_excel(a.output, index=False)
        print(f"-> {a.output}")
