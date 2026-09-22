"""
ban_do.py — Hai tính năng bản đồ chạy hoàn toàn bằng dữ liệu của mình.

    (1) TÌM QUANH ĐÂY  — chọn một điểm, lấy mọi tin trong bán kính R km.
    (2) BẢN ĐỒ NHIỆT GIÁ ĐẤT — chia Hà Nội thành ô lưới, mỗi ô một mức giá/m².

Cả hai chỉ cần toạ độ do `gan_toa_do.py` sinh ra, không phụ thuộc nguồn ngoài
nào, nên chạy được ngay cả khi máy không có mạng.

Ô LƯỚI TỰ ĐIỀU CHỈNH KÍCH THƯỚC
-------------------------------
Vẽ bản đồ nhiệt bằng lưới cố định thì hỏng ở hai đầu: ô 500 m ở Hoàn Kiếm có
80 tin (đẹp), nhưng ô 500 m ở Ba Vì có 1 tin — và một tin thì không phải "mức
giá khu vực", nó là ý muốn của đúng một người bán. Ngược lại, lưới 5 km thì
gộp cả Mỹ Đình lẫn Cầu Diễn vào một màu, mất hết chi tiết.

Cách làm ở đây giống cách người ta đọc bản đồ giấy: nhìn ô nhỏ trước, ô nào
quá thưa thì lùi ra ô lớn hơn. Duyệt lần lượt 0,5 km → 1 km → 2 km → 4 km, mỗi
điểm nhận mức lưới NHỎ NHẤT mà ô của nó còn đủ `n_toi_thieu` tin. Cột `muc_km`
ghi rõ ô đó thuộc mức nào, để khi trình bày còn nói được "vùng này màu nhạt vì
dữ liệu thưa" thay vì để người xem tưởng là giá thật.

MỘT CẢNH BÁO KHI ĐỌC BẢN ĐỒ
---------------------------
Toạ độ của mình là toạ độ MỨC ĐƯỜNG (xem đầu file gan_toa_do.py): nhiều tin
cùng một con phố dùng chung một điểm ghim. Nên một ô 500 m rất dễ bị một con
phố chi phối. Đừng đọc bản đồ này như bản đồ giá từng thửa đất — nó là bản đồ
mặt bằng giá theo khu vực.

CÁCH CHẠY
---------
    python ban_do.py                      # xuất lưới giá + GeoJSON + ảnh xem thử
    python ban_do.py --quanh 21.0287 105.8524 --ban-kinh 1.5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from gan_toa_do import HN_BOX, haversine_km, thu_muc, thu_muc_ra

BASE_DIR = Path(__file__).resolve().parent
DATA_CLEAN = thu_muc("data_clean")

TEP = {"chungcu": "chungcu_gop_geo.xlsx", "nhadat": "nhadat_gop_geo.xlsx"}
COT_DIEN_TICH = {"chungcu": "dien_tich_m2", "nhadat": "dien_tich_dat_m2"}

HO_HOAN_KIEM = (21.0287, 105.8524)
MUC_LUOI_KM = (0.5, 1.0, 2.0, 4.0)
N_TOI_THIEU = 5

# Những mức toạ độ đủ chính xác để ghim một điểm lên bản đồ mà không nói dối.
# `nguon_goc` là toạ độ sàn tự công bố, `ma_tin`/`du_an` khớp đúng đối tượng.
# Các mức còn lại là ước lượng theo tên đường hoặc tên phường.
TOA_DO_GHIM_DUOC = ("nguon_goc", "ma_tin", "du_an")

# Ở vĩ độ 21°, một độ kinh tuyến ngắn hơn một độ vĩ tuyến khoảng 7%. Bỏ qua
# chuyện này thì ô lưới thành hình chữ nhật méo, tính diện tích sai.
_KM_MOI_DO_VI = 111.32
_HE_SO_KINH = float(np.cos(np.radians(21.0)))


def nap(loai: str) -> pd.DataFrame:
    """Bảng tin đã gán toạ độ, dùng cho mọi phần bản đồ.

    ƯU TIÊN PARQUET TRONG GÓI TRIỂN KHAI.
    Hàm này chạy lúc NGƯỜI DÙNG MỞ TAB KHU VỰC, không phải lúc dựng dữ liệu —
    và đọc .xlsx mất 1,6 s (chung cư) / 5,4 s (nhà đất). Sau khi
    `chuan_bi_trien_khai.py` đã rút khởi động chính từ 16,6 s xuống 0,35 s thì
    đây lại thành chỗ chậm nhất còn lại, nên nó phải đọc cùng một file parquet.

    Đọc chung một file còn được thêm một thứ quan trọng hơn tốc độ: bản đồ và
    phần định giá nhìn ĐÚNG MỘT bộ dữ liệu. Trước đây hai bên đọc hai file
    khác nhau, và chỉ cần chạy lại pipeline mà quên một bước là bản đồ hiển
    thị một tập tin còn giá lại tính trên tập khác.

    Không có gói thì quay về .xlsx như cũ — đường chậm vẫn chạy đúng.
    """
    pq = thu_muc_ra("_trien_khai") / loai / "du_lieu.parquet"
    if pq.exists():
        try:
            d = pd.read_parquet(pq)
        except Exception:                            # noqa: BLE001
            d = pd.read_excel(DATA_CLEAN / TEP[loai])
    else:
        d = pd.read_excel(DATA_CLEAN / TEP[loai])
    d = d.dropna(subset=["latitude", "longitude"]).copy()
    d["loai"] = loai
    d["dien_tich"] = d[COT_DIEN_TICH[loai]]
    if "gia_tren_m2" not in d.columns:
        d["gia_tren_m2"] = d["TARGET_gia_vnd"] / d["dien_tich"]
    d["gia_tren_m2"] = d["gia_tren_m2"].fillna(d["TARGET_gia_vnd"] / d["dien_tich"])
    return d


# =============================================================================
# (1) TÌM QUANH ĐÂY
# =============================================================================

def tim_quanh(diem: tuple[float, float], ban_kinh_km: float = 1.0,
              loai: str | list[str] = ("chungcu", "nhadat"),
              gia_toi_da: float | None = None,
              dien_tich_toi_thieu: float | None = None,
              chi_toa_do_chac: bool = False,
              sap_xep: str = "khoang_cach") -> pd.DataFrame:
    """Mọi tin nằm trong bán kính `ban_kinh_km` quanh `diem` = (vĩ độ, kinh độ).

    `chi_toa_do_chac=True` chỉ giữ những tin có toạ độ mức đường trở lên. Nên
    bật khi bán kính nhỏ (dưới 1 km): với sai số p90 = 0,8 km của mức phường,
    một tin có thể bị xếp nhầm vào hay ra khỏi vòng tròn 500 m.
    """
    loai = [loai] if isinstance(loai, str) else list(loai)
    d = pd.concat([nap(l) for l in loai], ignore_index=True)

    if chi_toa_do_chac:
        d = d[d["nguon_toa_do"].isin(["ma_tin", "du_an", "duong"])]
    if gia_toi_da is not None:
        d = d[d["TARGET_gia_vnd"] <= gia_toi_da]
    if dien_tich_toi_thieu is not None:
        d = d[d["dien_tich"] >= dien_tich_toi_thieu]

    d = d.copy()
    d["khoang_cach_km"] = haversine_km(d["latitude"], d["longitude"], *diem)
    d = d[d["khoang_cach_km"] <= ban_kinh_km]

    khoa = {"khoang_cach": "khoang_cach_km", "gia": "TARGET_gia_vnd",
            "gia_m2": "gia_tren_m2"}.get(sap_xep, "khoang_cach_km")
    return d.sort_values(khoa).reset_index(drop=True)


# =============================================================================
# (2) BẢN ĐỒ NHIỆT
# =============================================================================

def _o_luoi(lat: pd.Series, lon: pd.Series, canh_km: float):
    """Trả về (chỉ số hàng, chỉ số cột) của ô lưới chứa mỗi điểm."""
    b_vi = canh_km / _KM_MOI_DO_VI
    b_kinh = b_vi / _HE_SO_KINH
    return (np.floor(lat / b_vi).astype("Int64"),
            np.floor(lon / b_kinh).astype("Int64"))


def luoi_gia(loai: str | list[str] = ("chungcu", "nhadat"),
             n_toi_thieu: int = N_TOI_THIEU,
             muc_km: tuple[float, ...] = MUC_LUOI_KM,
             chi_toa_do_chac: bool = False) -> pd.DataFrame:
    """Bảng ô lưới: tâm ô, giá/m² trung vị, số tin, mức lưới đã dùng.

    Mỗi tin được gán vào ô ở mức lưới nhỏ nhất còn đủ dày. Kết quả là một bảng
    ô KHÔNG chồng lấn (mỗi tin thuộc đúng một ô), nên tô màu lên bản đồ không
    bị đè nhau.
    """
    loai = [loai] if isinstance(loai, str) else list(loai)
    d = pd.concat([nap(l) for l in loai], ignore_index=True)
    if chi_toa_do_chac:
        d = d[d["nguon_toa_do"].isin(["ma_tin", "du_an", "duong"])]
    d = d.dropna(subset=["gia_tren_m2"]).reset_index(drop=True)

    chua_xep = np.ones(len(d), bool)
    o = []
    for canh in muc_km:
        if not chua_xep.any():
            break
        sub = d[chua_xep]
        r, c = _o_luoi(sub["latitude"], sub["longitude"], canh)
        tam = pd.DataFrame({"r": r.values, "c": c.values,
                            "gia_m2": sub["gia_tren_m2"].values,
                            "vi": sub["latitude"].values,
                            "kinh": sub["longitude"].values})
        dem = tam.groupby(["r", "c"])["gia_m2"].transform("size")
        du = (dem >= n_toi_thieu).values

        # Ở mức lưới thô nhất thì nhận hết, kể cả ô thưa — thà có một ô ghi rõ
        # "chỉ 2 tin" còn hơn để trống một mảng bản đồ không lời giải thích.
        if canh == muc_km[-1]:
            du = np.ones(len(tam), bool)
        if not du.any():
            continue

        g = tam[du].groupby(["r", "c"])
        b_vi = canh / _KM_MOI_DO_VI
        b_kinh = b_vi / _HE_SO_KINH
        bang = pd.DataFrame({
            "muc_km": canh,
            "so_tin": g.size(),
            "gia_m2_trung_vi": g["gia_m2"].median(),
            "gia_m2_q25": g["gia_m2"].quantile(.25),
            "gia_m2_q75": g["gia_m2"].quantile(.75),
        }).reset_index()
        bang["vi_min"] = bang["r"] * b_vi
        bang["vi_max"] = (bang["r"] + 1) * b_vi
        bang["kinh_min"] = bang["c"] * b_kinh
        bang["kinh_max"] = (bang["c"] + 1) * b_kinh
        bang["vi_tam"] = (bang["vi_min"] + bang["vi_max"]) / 2
        bang["kinh_tam"] = (bang["kinh_min"] + bang["kinh_max"]) / 2
        o.append(bang)

        idx = np.where(chua_xep)[0][du]
        chua_xep[idx] = False

    out = pd.concat(o, ignore_index=True)
    out["gia_m2_trieu"] = (out["gia_m2_trung_vi"] / 1e6).round(2)
    out["du_tin_cay"] = out["so_tin"] >= n_toi_thieu
    tt = haversine_km(out["vi_tam"], out["kinh_tam"], *HO_HOAN_KIEM)
    out["kc_trung_tam_km"] = tt.round(2)
    # Sắp ô LỚN lên trước: ô 4 km và ô 0,5 km nằm trong nó chồng nhau về hình
    # học (dù mỗi tin chỉ thuộc một ô). Vẽ theo thứ tự này thì ô nhỏ nằm trên,
    # ô lớn làm nền — cả matplotlib lẫn folium đều tô theo đúng thứ tự bảng.
    return out.sort_values(["muc_km", "gia_m2_trung_vi"],
                           ascending=[False, False]).reset_index(drop=True)


def diem_nhiet(loai: str, k_toi_thieu: int = N_TOI_THIEU,
               ban_kinh_km: float = 1.0,
               chi_toa_do_chac: bool = False) -> pd.DataFrame:
    """Điểm cho lớp bản đồ nhiệt vẽ bằng vệt màu mềm (HeatmapLayer).

    Khác với `luoi_gia`: ở đây KHÔNG gộp thành ô. Mỗi tin là một điểm, trọng số
    là giá/m² của chính nó, và việc làm mượt để deck.gl lo. Nhờ vậy vùng đắt
    loang ra thành mảng màu mềm thay vì mấy chục hình vuông cạnh sắc.

    MỘT ĐIỀU BẮT BUỘC PHẢI LÀM, nếu không bản đồ sẽ nói dối
    -------------------------------------------------------
    HeatmapLayer với `aggregation="MEAN"` tô màu theo giá TRUNG BÌNH của các
    tin quanh mỗi điểm ảnh — kể cả khi quanh đó chỉ có đúng một tin. Một căn
    biệt thự 300 triệu/m² đứng lẻ ở Sóc Sơn sẽ tạo ra một vệt đỏ rực trông y hệt
    vệt đỏ của Hoàn Kiếm, trong khi bằng chứng đằng sau nó là một người rao.

    Nên trước khi vẽ, bỏ những tin ĐỨNG LẺ: tin nào có dưới `k_toi_thieu` tin
    khác trong bán kính `ban_kinh_km` thì không được góp mặt vào lớp nhiệt.
    Đây đúng là quy tắc mà ô lưới cũ thể hiện bằng cách lùi sang mức lưới thô
    hơn — chỉ khác là ở đây nó thành "vùng này không tô màu" thay vì "vùng này
    tô một ô 4 km nhạt".

    ĐẾM GHIM RIÊNG BIỆT, KHÔNG ĐẾM TIN
    ----------------------------------
    Chỗ này tôi làm sai ở bản đầu và nó hiện ra thành lỗi nhìn thấy được: trên
    bản đồ mọc lên những đốm TRÒN HOÀN HẢO ở vùng ven. Một trường giá thật
    không bao giờ tạo ra hình tròn hoàn hảo — đó là dấu vết của đúng một điểm
    ghim được làm mượt bằng nhân Gauss.

    Nguyên nhân: toạ độ của mình phần lớn ở mức ĐƯỜNG, nên vài chục tin cùng
    một con phố dùng chung y hệt một điểm. Đếm theo TIN thì một điểm ghim có 30
    tin dễ dàng vượt ngưỡng "5 hàng xóm", trong khi về mặt không gian nó chỉ là
    MỘT mẩu bằng chứng, không phải 30.

    Nên điều kiện đúng là: quanh đây phải có ít nhất `k_toi_thieu` ĐIỂM GHIM
    KHÁC NHAU. Sửa chỗ này thì các đốm tròn biến mất, và nhờ đó mới hạ được mức
    làm mượt xuống để bản đồ chi tiết hơn mà vẫn không bịa.

    Số tin bị loại được trả về ở thuộc tính `.attrs["so_bi_loai"]` để giao diện
    nói thẳng với người xem, thay vì im lặng giấu đi một phần dữ liệu.
    """
    from scipy.spatial import cKDTree

    d = nap(loai)
    if chi_toa_do_chac:
        d = d[d["nguon_toa_do"].isin(["ma_tin", "du_an", "duong"])]
    d = d.dropna(subset=["gia_tren_m2", "latitude", "longitude"])
    d = d.reset_index(drop=True)
    if d.empty:
        out = d.assign(gia_m2_trieu=[])
        out.attrs["so_bi_loai"] = 0
        out.attrs["so_ve"] = 0
        return out

    # Gom các tin dùng chung một điểm ghim. Làm tròn 4 chữ số ≈ 11 m: dưới mức
    # đó thì hai toạ độ khác nhau cũng không mang thêm thông tin không gian nào.
    khoa = list(zip(d["latitude"].round(4), d["longitude"].round(4)))
    ghim, chi_so = np.unique(np.array(khoa), axis=0, return_inverse=True)

    # Quy về toạ độ km phẳng để bán kính tính bằng km có nghĩa như nhau ở mọi
    # hướng — dùng thẳng độ thì 1 độ kinh ngắn hơn 1 độ vĩ khoảng 7%.
    xy_ghim = np.column_stack([
        ghim[:, 1] * _KM_MOI_DO_VI * _HE_SO_KINH,
        ghim[:, 0] * _KM_MOI_DO_VI,
    ])
    cay = cKDTree(xy_ghim)
    # −1 vì mỗi ghim luôn tự đếm chính nó.
    so_ghim_quanh = np.array([len(v) for v in
                              cay.query_ball_point(xy_ghim, r=ban_kinh_km)]) - 1
    du = (so_ghim_quanh >= k_toi_thieu)[chi_so]

    out = d[du].copy()
    out["gia_m2_trieu"] = (out["gia_tren_m2"] / 1e6).round(2)
    giu = ["latitude", "longitude", "gia_m2_trieu", "gia_tren_m2"]
    if "nguon_toa_do" in out.columns:
        giu.append("nguon_toa_do")
    out = out[giu]
    out.attrs["so_bi_loai"] = int((~du).sum())
    out.attrs["so_ve"] = int(du.sum())
    out.attrs["so_ghim"] = int(len(np.unique(chi_so[du])))
    return out.reset_index(drop=True)


def sigma_tu_toa_do(d: pd.DataFrame, san: float = 0.30) -> float:
    """Mức làm mượt suy ra từ chính độ bất định của toạ độ.

    Đây là chỗ đáng nói nhất trong cả tính năng bản đồ. Câu hỏi "vẽ chi tiết
    đến đâu" nhìn thì tưởng là câu hỏi thẩm mỹ, nhưng thật ra nó có một câu trả
    lời đúng: **chi tiết đến đúng mức mà toạ độ đỡ được, không hơn.**

    Mỗi tin mang theo một mức sai số vị trí đã đo được (`SAI_SO_P90_KM` trong
    `gan_toa_do.py`): tin do sàn tự công bố toạ độ thì sai số 0 km, tin gán
    theo tên đường thì p90 = 0,67 km, gán theo phường thì 0,79 km. Lấy trung
    bình các mức đó trên đúng tập đang vẽ, ta được bán kính mà "vị trí của một
    tin" thực sự trải ra. Làm mượt nhỏ hơn con số đó là vẽ ra những mấu nhọn
    chỉ tồn tại vì điểm ghim bị đặt sai chỗ.

    Hệ quả tự nhiên và đúng đắn: bản đồ chung cư sắc nét hơn bản đồ nhà đất
    (0,34 km so với 0,73 km), vì một phần ba tin chung cư có toạ độ thật do
    batdongsan công bố. Sự chênh lệch đó là thứ KIẾM ĐƯỢC bằng dữ liệu, không
    phải một lựa chọn tuỳ hứng — và nếu sau này cào thêm được tin có toạ độ
    chính xác thì bản đồ tự sắc nét lên mà không phải sửa dòng nào.
    """
    from gan_toa_do import SAI_SO_P90_KM

    if "nguon_toa_do" not in d.columns or d.empty:
        return 0.70
    e = d["nguon_toa_do"].map(SAI_SO_P90_KM).astype(float)
    if not e.notna().any():
        return 0.70
    return max(float(e.mean()), san)


def _vong_cua(geom: dict) -> list[list[np.ndarray]]:
    """Đưa Polygon và MultiPolygon về cùng một dạng: danh sách các đa giác,
    mỗi đa giác là [viền ngoài, lỗ, lỗ...] dưới dạng mảng numpy (n, 2)."""
    if geom["type"] == "Polygon":
        tho = [geom["coordinates"]]
    elif geom["type"] == "MultiPolygon":
        tho = geom["coordinates"]
    else:
        return []
    return [[np.asarray(v, dtype=float) for v in dg] for dg in tho]


def _trong_vong(px: np.ndarray, py: np.ndarray, vong: np.ndarray) -> np.ndarray:
    """Bắn tia: điểm nằm trong vòng kín nếu tia ngang cắt biên số lẻ lần.

    Vector hoá theo cả điểm lẫn cạnh. Gọi hàm này sau khi đã lọc bằng hộp bao,
    nếu không ma trận (số điểm × số cạnh) sẽ rất lớn.
    """
    x1, y1 = vong[:-1, 0], vong[:-1, 1]
    x2, y2 = vong[1:, 0], vong[1:, 1]
    cat = (y1[None, :] > py[:, None]) != (y2[None, :] > py[:, None])
    dy = np.where(y2 - y1 == 0, np.nan, y2 - y1)
    with np.errstate(divide="ignore", invalid="ignore"):
        xg = (x2 - x1)[None, :] * (py[:, None] - y1[None, :]) / dy[None, :] \
             + x1[None, :]
    return (np.nansum(cat & (px[:, None] < xg), axis=1) % 2) == 1


def _gan_diem_vao_vung(lon: np.ndarray, lat: np.ndarray,
                       features: list) -> tuple[np.ndarray, np.ndarray]:
    """Mỗi điểm rơi vào đa giác nào. Trả về (chỉ số điểm, chỉ số đa giác).

    VÌ SAO TỰ VIẾT THAY VÌ DÙNG shapely
    ------------------------------------
    shapely là thư viện đúng cho việc này và `ranh_gioi.py` vẫn dùng nó. Nhưng
    `ranh_gioi.py` chỉ chạy một lần để sinh ra file GeoJSON, còn hàm này chạy
    mỗi khi ai đó mở ứng dụng. Bắt người dùng cài một thư viện có phần mở rộng
    biên dịch (GEOS) chỉ để tô màu bản đồ là một rào cản thật: trên Python mới
    ra, bản dựng sẵn thường chưa có, và pip sẽ cố biên dịch rồi hỏng.

    Phép kiểm tra điểm-trong-đa-giác là bắn tia — khoảng hai chục dòng numpy.
    Đổi lại, ứng dụng chỉ cần numpy và pandas, những thứ vốn đã phải có.
    """
    trong_vung = []
    for i, f in enumerate(features):
        for dg in _vong_cua(f["geometry"]):
            ngoai = dg[0]
            # Lọc hộp bao trước: rẻ, và loại ngay phần lớn điểm.
            x0, y0 = ngoai[:, 0].min(), ngoai[:, 1].min()
            x1, y1 = ngoai[:, 0].max(), ngoai[:, 1].max()
            ung = np.where((lon >= x0) & (lon <= x1)
                           & (lat >= y0) & (lat <= y1))[0]
            if not len(ung):
                continue
            nam = _trong_vong(lon[ung], lat[ung], ngoai)
            # Trừ đi các lỗ: điểm nằm trong lỗ thì không thuộc đa giác.
            for lo in dg[1:]:
                if nam.any():
                    nam &= ~_trong_vong(lon[ung], lat[ung], lo)
            if nam.any():
                trong_vung.append((ung[nam], i))

    if not trong_vung:
        return np.array([], int), np.array([], int)
    i_tin = np.concatenate([t for t, _ in trong_vung])
    i_vung = np.concatenate([np.full(len(t), v) for t, v in trong_vung])
    return i_tin, i_vung


def tin_ghim(loai: str, quan: str | None = None,
             gia_tu: float | None = None, gia_den: float | None = None,
             dt_tu: float | None = None, dt_den: float | None = None,
             chi_toa_do_chac: bool = False) -> pd.DataFrame:
    """Từng tin rao thành một điểm ghim để bấm vào xem — không phải bản đồ giá.

    Đây là tính năng KHÁC HẲN choropleth, dù cùng vẽ trên bản đồ. Choropleth
    trả lời "khu này mặt bằng bao nhiêu"; bản đồ ghim trả lời "quanh đây đang
    có gì bán". Người đi mua nhà hỏi cả hai câu, nhưng không cùng lúc.

    VẤN ĐỀ TRUNG THỰC PHẢI XỬ LÝ, KHÔNG ĐƯỢC LỜ ĐI
    ----------------------------------------------
    Một điểm ghim trên bản đồ ngầm hứa với người xem rằng "căn nhà nằm ở đúng
    chỗ này". Với dữ liệu của mình, lời hứa đó chỉ đúng một phần:

        chung cư : 55,7% tin có toạ độ chính xác tới toà nhà
        nhà đất  : 62,7%   (đo lại 01/09/2026 sau khi có dữ liệu batdongsan)

    Phần còn lại là toạ độ suy ra từ tên đường (sai số p90 = 0,67 km) hoặc tên
    phường (0,79 km). Ghim chúng như ghim toạ độ thật là nói dối bằng hình ảnh
    — người xem sẽ tưởng căn nhà ở đúng góc phố đó.

    Hai việc phải làm, và cả hai đều nằm trong hàm này:

    1. Cột `chinh_xac` phân biệt hai loại, để giao diện vẽ khác nhau: ghim đặc
       cho toạ độ thật, ghim rỗng mờ cho toạ độ ước lượng.

    2. Cột `so_tin_cung_ghim` đếm bao nhiêu tin dùng chung đúng một điểm. Con
       số này gây sốc: có điểm ghim mang **144 tin chung cư** và **114 tin nhà
       đất** — cả một con phố dồn vào một chấm. Không hiện ra thì người dùng
       bấm vào thấy đúng một tin và tưởng quanh đó chỉ có thế.

    KHÔNG rải ngẫu nhiên (jitter) các ghim trùng nhau cho đẹp. Làm vậy là bịa
    ra độ chính xác không có, và còn tệ hơn để chúng chồng lên nhau — ít nhất
    chồng lên nhau thì còn nhìn ra là dữ liệu thô.
    """
    d = nap(loai).dropna(subset=["latitude", "longitude", "gia_tren_m2"]).copy()

    if quan:
        d = d[d["quan_huyen"] == quan]
    if gia_tu is not None:
        d = d[d["TARGET_gia_vnd"] >= gia_tu]
    if gia_den is not None:
        d = d[d["TARGET_gia_vnd"] <= gia_den]
    if dt_tu is not None:
        d = d[d["dien_tich"] >= dt_tu]
    if dt_den is not None:
        d = d[d["dien_tich"] <= dt_den]

    d["chinh_xac"] = d["nguon_toa_do"].isin(TOA_DO_GHIM_DUOC)
    if chi_toa_do_chac:
        d = d[d["chinh_xac"]]
    if d.empty:
        return d

    khoa = list(zip(d["latitude"].round(4), d["longitude"].round(4)))
    dem = pd.Series(khoa).value_counts()
    d["so_tin_cung_ghim"] = [int(dem[k]) for k in khoa]

    d["gia_m2_trieu"] = (d["gia_tren_m2"] / 1e6).round(1)
    d["gia_ty"] = (d["TARGET_gia_vnd"] / 1e9).round(2)

    ten = (d["du_an_clean"].replace("Không xác định", np.nan)
           if "du_an_clean" in d.columns else pd.Series(np.nan, index=d.index))
    if "tieu_de" in d.columns:
        ten = ten.fillna(d["tieu_de"].astype(str).str.slice(0, 60))
    ten = ten.fillna(d.get("duong_pho")).fillna("Tin rao")
    d["ten_hien"] = ten

    them = np.where(d["so_tin_cung_ghim"] > 1,
                    " · " + d["so_tin_cung_ghim"].astype(str) + " tin cùng ghim",
                    "")
    d["nhan"] = (d["ten_hien"].astype(str) + " · "
                 + d["gia_ty"].astype(str) + " tỷ · "
                 + d["dien_tich"].round(0).astype("Int64").astype(str) + " m² · "
                 + d["gia_m2_trieu"].astype(str) + " tr/m²"
                 + np.where(d["chinh_xac"], "", " · vị trí ước lượng")
                 + them)
    return d.reset_index(drop=True)


def thua_theo_luoi(d: pd.DataFrame, o_km: float,
                   cot_uu_tien: str = "so_tin_cung_ghim") -> pd.DataFrame:
    """Giảm mật độ ghim bằng cách chia lưới, mỗi ô giữ một tin đại diện.

    VÌ SAO KHÔNG LÀM ĐÚNG KIỂU "THU NHỎ THÌ BỚT CHẤM"
    -------------------------------------------------
    Cách chuẩn của các sàn là đổi mật độ theo mức phóng to (level of detail).
    Làm được vậy cần biết người dùng đang xem ở mức zoom nào — mà Streamlit chỉ
    trả về sự kiện BẤM từ pydeck, không trả về khung nhìn. Không có JS tự viết
    thì Python không bao giờ biết bản đồ đang ở zoom mấy.

    Nên ở đây dùng thứ thay thế gần nhất mà vẫn trung thực: **mật độ đi theo
    phạm vi người dùng chọn**. Xem toàn thành phố thì lưới thô, chọn một quận
    thì lưới mịn, chọn phường thì hiện hết. Người dùng thu hẹp phạm vi cũng
    chính là lúc họ muốn thấy chi tiết, nên hiệu quả gần giống LOD thật.

    Trong mỗi ô giữ tin có `so_tin_cung_ghim` lớn nhất — tức điểm ghim đại diện
    cho nhiều tin nhất, chứ không phải một tin ngẫu nhiên.
    """
    if d.empty or o_km <= 0:
        return d
    b_vi = o_km / _KM_MOI_DO_VI
    b_kinh = b_vi / _HE_SO_KINH
    g = d.assign(
        _r=np.floor(d["latitude"] / b_vi).astype(int),
        _c=np.floor(d["longitude"] / b_kinh).astype(int))
    cot = cot_uu_tien if cot_uu_tien in g.columns else "gia_ty"
    g = g.sort_values(cot, ascending=False).drop_duplicates(["_r", "_c"])
    return g.drop(columns=["_r", "_c"]).reset_index(drop=True)


def vung_phuong(loai: str, n_toi_thieu: int = 5,
                so_bac: int = 7) -> dict:
    """Choropleth: tô mỗi PHƯỜNG THẬT theo giá, dùng ranh giới hành chính thật.

    Đây là thứ thay cho bề mặt vẽ bằng đường đồng mức. Khác biệt không nằm ở
    thẩm mỹ mà ở chỗ người xem nhận ra được mình đang nhìn đâu: hình dạng phường
    là hình dạng người ta thấy trên mọi bản đồ hành chính, còn một vệt loang
    thì không neo vào cái gì cả.

    TÔ MÀU THEO TIN NẰM BÊN TRONG, KHÔNG PHẢI THEO TÊN
    --------------------------------------------------
    Có hai đường để nối một đa giác với một mức giá:
      (a) đặt tên cho đa giác, rồi tra giá theo tên;
      (b) hỏi thẳng "những tin nào rơi vào bên trong đa giác này".

    Ở đây chọn (b). Lý do: tên của đa giác phải suy ra bằng bỏ phiếu từ chính
    các tin bên trong (file GIS chỉ có mã xã), mà phép bỏ phiếu đó có hai đa
    giác cùng ra một tên. Nối theo tên thì hai phường sẽ tranh nhau một mức
    giá. Nối theo hình học thì mỗi đa giác có bộ tin của riêng nó, không bao
    giờ mập mờ.

    Cái giá phải trả, nói thẳng: toạ độ tin rao sai số p90 khoảng 0,67 km ở mức
    đường, trong khi phường nội thành có bán kính chưa tới 1 km. Nên ở khu
    trung tâm, một phần tin bị rơi sang phường bên cạnh. Đó là lý do cột
    `so_tin` phải luôn hiện ra cùng con số giá, và các phường quá ít tin thì
    không tô màu.
    """
    tep = DATA_CLEAN / "ranh_gioi_phuong_hanoi.geojson"
    if not tep.exists():
        raise FileNotFoundError(
            f"Chưa có {tep}. Chạy `python ranh_gioi.py` trước.")
    gj = json.loads(tep.read_text(encoding="utf-8"))

    d = nap(loai).dropna(subset=["gia_tren_m2", "latitude", "longitude"])
    i_tin, i_vung = _gan_diem_vao_vung(
        d["longitude"].to_numpy(float), d["latitude"].to_numpy(float),
        gj["features"])

    thuoc = pd.DataFrame({
        "i_vung": i_vung,
        "gia_m2": d["gia_tren_m2"].to_numpy()[i_tin],
        "gia": d["TARGET_gia_vnd"].to_numpy()[i_tin],
    })
    dem_tho = thuoc["i_vung"].value_counts()
    g = thuoc.groupby("i_vung").agg(
        so_tin=("gia_m2", "size"),
        tv=("gia_m2", "median"),
        q25=("gia_m2", lambda s: s.quantile(.25)),
        q75=("gia_m2", lambda s: s.quantile(.75)),
        gia_tin=("gia", "median"),
    )
    g = g[g["so_tin"] >= n_toi_thieu]

    if g.empty:
        return {"type": "FeatureCollection", "features": [], "so_bac": so_bac,
                "gia_min": 0.0, "gia_max": 1.0, "so_vung": 0,
                "so_vung_du_tin": 0}

    tr = g["tv"] / 1e6

    # CHIA BẬC THEO PHÂN VỊ, KHÔNG CHIA ĐỀU THEO GIÁ TRỊ.
    #
    # Giá bất động sản lệch phải rất mạnh: trung vị các phường chạy từ 36 đến
    # 175 triệu/m², nhưng một nửa số phường dồn trong quãng 62–86. Chia đều
    # khoảng giá thì 21 trên 57 phường rơi vào CÙNG một bậc màu, trong khi hai
    # bậc trên cùng mỗi bậc chỉ có 2 phường. Kết quả là gần như cả Hà Nội một
    # màu, chỉ vài phường trung tâm nổi lên — đúng thứ nhìn thấy trên bản đồ.
    #
    # Chia theo phân vị thì mỗi bậc có 7–9 phường, thang màu được dùng hết, và
    # người xem phân biệt được các quận ngoài chứ không phải chỉ thấy "trung
    # tâm đỏ, còn lại vàng nhạt".
    #
    # Cái giá phải trả và phải nói rõ trong chú giải: khoảng cách giữa các bậc
    # KHÔNG bằng nhau. Bậc trên cùng trải rộng hơn bậc dưới nhiều lần, nên
    # không được đọc màu như một thang tuyến tính.
    moc = np.quantile(tr, np.linspace(0, 1, so_bac + 1)[1:-1])
    lo, hi = float(tr.min()), float(tr.max())

    ft = []
    for i, f in enumerate(gj["features"]):
        if i not in g.index:
            # Phường không đủ tin: VẪN VẼ, chỉ tô xám mờ.
            #
            # Bản trước bỏ hẳn chúng ra, và trên nền tối thì "không có dữ liệu"
            # trông y hệt "không thuộc Hà Nội" — người xem thấy một lỗ thủng
            # giữa bản đồ mà không biết đó là gì. Vẽ ra thì hình Hà Nội liền
            # mạch, và chỗ trống tự nói lên rằng mình thiếu dữ liệu ở đó.
            ft.append({
                "type": "Feature",
                "geometry": f["geometry"],
                "bac": -1,
                "nhan": (f"{f['properties'].get('phuong_moi') or 'Chưa rõ tên'}"
                         f" · chưa đủ dữ liệu "
                         f"({int(dem_tho.get(i, 0))} căn)"),
                "properties": {
                    "phuong_moi": f["properties"].get("phuong_moi"),
                    "gia_m2_trieu": None,
                    "so_tin": int(dem_tho.get(i, 0)),
                },
            })
            continue
        r = g.loc[i]
        tv_tr = r["tv"] / 1e6
        ft.append({
            "type": "Feature",
            "geometry": f["geometry"],
            "bac": int(np.searchsorted(moc, tv_tr)),
            "nhan": (
                f"{f['properties'].get('phuong_moi') or 'Chưa rõ tên'} · "
                f"{tv_tr:,.0f} triệu/m² · phần lớn từ "
                f"{r['q25']/1e6:,.0f} đến {r['q75']/1e6:,.0f} · "
                f"dựa trên {int(r['so_tin'])} căn").replace(",", "."),
            "properties": {
                "phuong_moi": f["properties"].get("phuong_moi"),
                "gia_m2_trieu": round(float(tv_tr), 1),
                "so_tin": int(r["so_tin"]),
            },
        })

    return {
        "type": "FeatureCollection", "features": ft,
        "so_bac": so_bac, "gia_min": round(lo, 1), "gia_max": round(hi, 1),
        "moc": [round(float(x), 1) for x in moc],
        "so_vung": len(gj["features"]),
        "so_vung_du_tin": int((g["so_tin"] >= n_toi_thieu).sum()),
        "don_vi": " triệu/m²",
    }


def thong_ke_phuong(loai: str, n_toi_thieu: int = 10) -> pd.DataFrame:
    """Mặt bằng giá theo PHƯỜNG MỚI — bảng, không phải bản đồ.

    Vì sao thứ này quan trọng hơn bề mặt giá vẽ bằng đường đồng mức: bản đồ
    nhiệt không có mốc định vị nào. Không tên phường, không tên phố, không "bạn
    đang ở đây". Một mảng cam trên nền tối chỉ có nghĩa với người đã thuộc địa
    lý Hà Nội — mà người đi mua nhà thì thường không thuộc, đó chính là lý do
    họ tra cứu.

    Bảng theo phường trả lời thẳng câu người ta thật sự hỏi: "khu này giá bao
    nhiêu, đắt hơn khu kia bao nhiêu". Không cần đọc thang màu, không cần biết
    hình dạng quận nào nằm ở đâu.

    Ba cột luôn đi cùng nhau và không được tách rời:
      - TRUNG VỊ: mức giá đại diện, dùng trung vị chứ không phải trung bình vì
        vài căn mặt phố đắt sẽ kéo trung bình lên 6,8% (đo trên dữ liệu này).
      - KHOẢNG q25–q75: nửa số tin nằm trong đó. Thiếu cột này thì người xem
        tưởng mọi căn trong phường đều quanh mức trung vị, trong khi nhà đất
        chênh nhau tới 8,5 lần trong bán kính 500 m.
      - SỐ TIN: căn cứ. Một phường 11 tin và một phường 200 tin không đáng tin
        như nhau, và con số phải hiện ra chứ không được giấu.
    """
    d = nap(loai).dropna(subset=["gia_tren_m2", "phuong_moi"])
    g = d.groupby("phuong_moi").agg(
        so_tin=("gia_tren_m2", "size"),
        gia_m2=("gia_tren_m2", "median"),
        q25=("gia_tren_m2", lambda s: s.quantile(.25)),
        q75=("gia_tren_m2", lambda s: s.quantile(.75)),
        gia_tin=("TARGET_gia_vnd", "median"),
        vi=("latitude", "median"),
        kinh=("longitude", "median"),
    ).reset_index()

    # Quận của phường: lấy quận xuất hiện nhiều nhất trong phường đó. Sau sáp
    # nhập, một vài phường mới nằm vắt qua hai quận cũ nên không có ánh xạ
    # một-một; lấy đa số là cách trung thực nhất mà vẫn dùng được để lọc.
    if "quan_huyen" in d.columns:
        quan = (d.groupby("phuong_moi")["quan_huyen"]
                .agg(lambda s: s.mode().iat[0] if len(s.mode()) else None))
        g["quan_huyen"] = g["phuong_moi"].map(quan)

    g = g[g["so_tin"] >= n_toi_thieu].copy()
    tv_thanh_pho = float(d["gia_tren_m2"].median())
    g["so_voi_tp"] = (g["gia_m2"] / tv_thanh_pho - 1) * 100
    g["do_trai"] = (g["q75"] - g["q25"]) / g["gia_m2"] * 100
    g = g.sort_values("gia_m2", ascending=False).reset_index(drop=True)
    g["hang"] = np.arange(1, len(g) + 1)
    g.attrs["trung_vi_thanh_pho"] = tv_thanh_pho
    g.attrs["so_phuong_bi_loai"] = int(d["phuong_moi"].nunique() - len(g))
    return g


def _mat_gia(d: pd.DataFrame, n_luoi: int = 420, sigma_km: float | None = None,
             nguong_mat_do: float = 0.18, san_sigma: float = 0.30):
    """Mặt giá làm mượt.

    Trả về (lưới kinh, lưới vĩ, giá đại diện, độ phân tán, mặt nạ).

    Cách làm là trung bình có trọng số theo nhân Gauss. Với mỗi điểm trên lưới,
    giá gán cho nó là trung bình giá các tin quanh đó, tin càng gần càng nặng:

        giá(ô) = Σ(giá_tin · trọng_số) / Σ(trọng_số)

    Chia cho tổng trọng số là chỗ dễ làm sai nhất. Nếu chỉ lấy tử số thì chỗ
    nào đông tin sẽ sáng rực lên — kể cả khi toàn tin rẻ — và bản đồ giá biến
    thành bản đồ MẬT ĐỘ. Phép chia đúng nghĩa "trung bình", nên một khu có 200
    tin rẻ vẫn ra màu rẻ.

    LÀM MƯỢT TRÊN LOG, KHÔNG PHẢI TRÊN GIÁ
    --------------------------------------
    Phân phối giá lệch phải rất mạnh: trong bán kính 500 m, căn đắt nhất gấp
    căn rẻ nhất 8,5 lần với nhà đất (p90 là 24,7 lần). Với phân phối như thế,
    trung bình cộng bị vài căn đắt kéo lên và KHÔNG còn đại diện cho "một căn
    bình thường quanh đây" — đo trên dữ liệu này, nó nằm cao hơn trung vị 6,8%
    (p90 là 17,6%). Tức bản đồ cũ nói giá cao hơn thực tế một cách có hệ thống.

    Thứ ta muốn là TRUNG VỊ, nhưng trung vị không lọc Gauss được. May là với
    phân phối xấp xỉ log-chuẩn, trung bình HÌNH HỌC — exp của trung bình log —
    bám rất sát trung vị. Kiểm trên chính dữ liệu này: lệch trung vị −0,2%
    (nhà đất) và −1,6% (chung cư), so với +6,8% của trung bình cộng.

    Nên: lọc Gauss trên log(giá) rồi mũ ngược lại. Vẫn rẻ, vẫn dùng được toàn
    bộ bộ máy sigma-theo-từng-tin, mà hết thiên lệch lên trên.

    ĐỘ PHÂN TÁN — con số thứ hai, quan trọng không kém
    -------------------------------------------------
    Một con số giá cho cả một vùng luôn giấu đi câu hỏi "giấu bao nhiêu?".
    Cùng mức trung vị 200 triệu/m², một vùng có thể toàn nhà ngõ giá sát nhau,
    vùng khác trộn nhà mặt phố 400 với nhà ngõ 100. Hai vùng đó không nên trông
    giống hệt nhau trên bản đồ.

    Nên hàm này trả thêm độ lệch chuẩn của log giá, quy ra phần trăm dễ đọc:
    `exp(sd) − 1`. Con số đó nói "quanh đây giá dao động cỡ ±X% quanh mức trung
    vị", và nó chính là bản đồ cho biết **ở đâu thì riêng vị trí đã đủ đoán
    giá, ở đâu thì không**.

    MỖI TIN ĐƯỢC LÀM MƯỢT THEO SAI SỐ CỦA CHÍNH NÓ
    ----------------------------------------------
    Đây là chỗ tôi làm sai hai lần trước khi làm đúng.

    Lần một: một sigma cố định 1,2 km cho tất cả — bản đồ thành một cục mờ, mất
    hết chi tiết kể cả ở nơi mình biết toạ độ chính xác tới từng toà nhà.

    Lần hai: một sigma bằng TRUNG BÌNH sai số của cả tập. Với chung cư ra 0,31
    km, vì một phần ba tin có toạ độ thật kéo trung bình xuống. Nhưng hai phần
    ba còn lại vẫn sai 0,67–0,79 km, và làm mượt chúng ở 0,31 km thì bản đồ vỡ
    thành những đốm rời — chi tiết giả sinh ra từ chính sai số vị trí.

    Cái sai chung của cả hai lần là ép mọi tin dùng chung một mức làm mượt,
    trong khi chúng KHÔNG có cùng độ bất định. Cách đúng: mỗi tin toả ra một
    vệt Gauss rộng đúng bằng sai số vị trí của riêng nó. Tin do sàn công bố toạ
    độ đóng góp một chấm sắc nét; tin chỉ biết tên phường toả thành một mảng
    rộng 0,79 km. Cả hai cùng góp vào một phép trung bình, nên nơi nào có nhiều
    tin toạ độ tốt thì bản đồ tự sắc nét lên ở đúng chỗ đó.

    Cài đặt: gom tin theo mức sai số (chỉ khoảng chín mức khác nhau), lọc Gauss
    riêng từng nhóm rồi cộng tử số với mẫu số lại. Chín lượt lọc, vẫn rẻ.

    `san_sigma` là mức làm mượt tối thiểu, áp cả cho tin có sai số toạ độ bằng
    0. Không có nó thì một toà chung cư thành một mấu nhọn rộng đúng một pixel,
    và trên bản đồ nó hiện ra thành vòng tròn đồng tâm li ti — đường đồng mức
    cắt qua một đỉnh dốc đứng. Mà "mặt bằng giá khu vực" của đúng một toà nhà
    thì cũng không có nghĩa: kể cả khi biết chính xác nó ở đâu, một toà không
    đại diện cho khu vực quanh nó. 0,30 km xấp xỉ bán kính đi bộ 4 phút.

    `nguong_mat_do` cắt phần quá thưa thành trong suốt: nội suy vào chỗ không
    có tin nào là bịa ra giá cho vùng mình không có bằng chứng.
    """
    from scipy.ndimage import gaussian_filter
    from gan_toa_do import SAI_SO_P90_KM

    x = d["longitude"].to_numpy()
    y = d["latitude"].to_numpy()
    w = d["gia_m2_trieu"].to_numpy()

    xl = (float(np.quantile(x, .005)), float(np.quantile(x, .995)))
    yl = (float(np.quantile(y, .005)), float(np.quantile(y, .995)))
    rong_km = (xl[1] - xl[0]) * _KM_MOI_DO_VI * _HE_SO_KINH

    ix = np.clip(((x - xl[0]) / (xl[1] - xl[0]) * (n_luoi - 1)).astype(int),
                 0, n_luoi - 1)
    iy = np.clip(((y - yl[0]) / (yl[1] - yl[0]) * (n_luoi - 1)).astype(int),
                 0, n_luoi - 1)

    # Sai số vị trí của từng tin. Ép sigma cố định nếu người gọi yêu cầu (dùng
    # cho các thí nghiệm so sánh), còn mặc định là theo từng tin.
    if sigma_km is not None:
        sai_so = np.full(len(d), float(sigma_km))
    elif "nguon_toa_do" in d.columns:
        sai_so = (d["nguon_toa_do"].map(SAI_SO_P90_KM)
                  .astype(float).fillna(0.79).to_numpy())
    else:
        sai_so = np.full(len(d), 0.70)
    sai_so = np.maximum(sai_so, san_sigma)

    lg = np.log(np.maximum(w, 1e-9))

    Sw = np.zeros((n_luoi, n_luoi))    # Σ log(giá)
    Sw2 = np.zeros((n_luoi, n_luoi))   # Σ log(giá)²  — để tính phương sai
    Sd = np.zeros((n_luoi, n_luoi))    # Σ số tin
    for muc in np.unique(sai_so):
        chon = sai_so == muc
        tw = np.zeros((n_luoi, n_luoi))
        tw2 = np.zeros((n_luoi, n_luoi))
        td = np.zeros((n_luoi, n_luoi))
        np.add.at(tw, (iy[chon], ix[chon]), lg[chon])
        np.add.at(tw2, (iy[chon], ix[chon]), lg[chon] ** 2)
        np.add.at(td, (iy[chon], ix[chon]), 1.0)
        sg = float(muc) / rong_km * n_luoi
        Sw += gaussian_filter(tw, sg)
        Sw2 += gaussian_filter(tw2, sg)
        Sd += gaussian_filter(td, sg)

    an = np.maximum(Sd, 1e-12)
    tb_log = np.where(Sd > 1e-9, Sw / an, np.nan)
    gia = np.exp(tb_log)

    # Phương sai = E[log²] − E[log]². Cắt âm: sai số dấu phẩy động có thể đẩy
    # nó xuống dưới 0 ở những ô mà mọi tin gần như cùng một giá.
    ph_sai = np.maximum(np.where(Sd > 1e-9, Sw2 / an, np.nan) - tb_log ** 2, 0)
    phan_tan = np.exp(np.sqrt(ph_sai)) - 1.0

    nguong = np.quantile(Sd[Sd > 0], nguong_mat_do)
    mat_na = Sd >= nguong
    gia[~mat_na] = np.nan
    phan_tan[~mat_na] = np.nan

    xs = np.linspace(*xl, n_luoi)
    ys = np.linspace(*yl, n_luoi)
    return xs, ys, gia, phan_tan, mat_na


def _tach_lo_thung(duong_dan) -> list[list[list[list[float]]]]:
    """Tách một Path của matplotlib thành các đa giác [viền ngoài, lỗ, lỗ...].

    contourf sinh ra đa giác có lỗ (một vùng giá thấp nằm lọt giữa vùng giá
    cao). GeoJSON và deck.gl đều hiểu lỗ, nhưng phải xếp đúng thứ tự: phần tử
    đầu là viền ngoài, các phần tử sau là lỗ. Xếp sai thì lỗ bị tô đặc và bản
    đồ nói rằng chỗ trũng giá cũng đắt như xung quanh.

    Phân biệt bằng dấu của diện tích có hướng: matplotlib vẽ viền ngoài ngược
    chiều kim đồng hồ (diện tích dương) và lỗ theo chiều kim đồng hồ (âm).
    """
    from matplotlib.path import Path as MPath  # noqa: F401

    vong = [np.asarray(p) for p in duong_dan.to_polygons(closed_only=True)]
    vong = [v for v in vong if len(v) >= 4]
    if not vong:
        return []

    def dien_tich_co_huong(v):
        return float(np.sum(v[:-1, 0] * v[1:, 1] - v[1:, 0] * v[:-1, 1]) / 2)

    dt = [dien_tich_co_huong(v) for v in vong]
    ngoai = [(v, abs(a)) for v, a in zip(vong, dt) if a > 0]
    lo = [v for v, a in zip(vong, dt) if a <= 0]

    if not ngoai:                       # toàn lỗ: coi như không có gì
        return []

    ket = [[v.tolist()] for v, _ in ngoai]
    for h in lo:
        diem = h[0]
        # Gán lỗ cho viền ngoài NHỎ NHẤT chứa nó — nếu chọn viền lớn nhất thì
        # với hai vùng lồng nhau, lỗ sẽ khoét nhầm vào vùng ngoài cùng.
        ung_vien = [(dt_, i) for i, (v, dt_) in enumerate(ngoai)
                    if _trong_da_giac(diem, v)]
        if ung_vien:
            ket[min(ung_vien)[1]].append(h.tolist())
    return ket


def _trong_da_giac(diem, vong) -> bool:
    """Kiểm tra điểm nằm trong đa giác bằng phép bắn tia."""
    x, y = float(diem[0]), float(diem[1])
    vx, vy = vong[:, 0], vong[:, 1]
    x1, y1 = vx[:-1], vy[:-1]
    x2, y2 = vx[1:], vy[1:]
    cat = ((y1 > y) != (y2 > y))
    with np.errstate(divide="ignore", invalid="ignore"):
        xg = (x2 - x1) * (y - y1) / np.where(y2 - y1 == 0, np.nan, y2 - y1) + x1
    return bool(np.sum(cat & (x < xg)) % 2 == 1)


def vung_gia(loai: str, so_dai: int = 9, k_toi_thieu: int = N_TOI_THIEU,
             chi_toa_do_chac: bool = False,
             sigma_km: float | None = None,
             chi_tieu: str = "gia") -> dict:
    """Mặt bằng giá dưới dạng các VÙNG có đường viền, thay cho ô lưới vuông.

    Vì sao không dùng vệt loang liên tục (HeatmapLayer): độ đậm biến thiên liên
    tục thì mắt không bám được vào đâu, cả bản đồ thành một mảng mờ. Cắt cùng
    mặt giá đó thành 7 dải rời rạc có viền thì mắt đọc ra cấu trúc ngay — đúng
    nguyên tắc của bản đồ địa hình: cùng dữ liệu, nhưng đường đồng mức đọc được
    còn chuyển sắc liên tục thì không.

    Trả về một dict kiểu GeoJSON FeatureCollection, mỗi feature là một dải giá,
    kèm `gia_tu` / `gia_den` để giao diện hiện chú giải và tooltip.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = diem_nhiet(loai, k_toi_thieu=k_toi_thieu,
                   chi_toa_do_chac=chi_toa_do_chac)
    xs, ys, gia, phan_tan, _ = _mat_gia(d, sigma_km=sigma_km)

    if chi_tieu == "phan_tan":
        # Độ phân tán quy ra phần trăm cho dễ đọc.
        mat = phan_tan * 100
        hop_le = mat[np.isfinite(mat)]
        lo, hi = (float(np.quantile(hop_le, .10)),
                  float(np.quantile(hop_le, .90))) if hop_le.size else (0.0, 1.0)
        don_vi = "%"
    else:
        mat = gia
        lo, hi = dai_mau(d["gia_m2_trieu"])
        don_vi = " triệu/m²"

    muc = np.linspace(lo, hi, so_dai + 1)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # extend="both": phải phủ cả hai đầu. Để "max" thì những vũng giá thấp hơn
    # mốc dưới của thang màu không được tô, và chúng hiện ra thành các chấm đen
    # lỗ chỗ giữa vùng có màu — trông như dữ liệu thủng, trong khi thật ra chỗ
    # đó có dữ liệu và chỉ là rẻ hơn phân vị 10.
    cs = ax.contourf(xs, ys, np.ma.masked_invalid(mat), levels=muc,
                     extend="both")

    # Với extend="both", contourf trả về len(muc)+1 dải chứ không phải len(muc)-1:
    # một dải cho phần DƯỚI mốc đầu, các dải giữa, và một dải cho phần TRÊN mốc
    # cuối. Đánh nhãn theo chỉ số i mà quên hai dải mở ở hai đầu thì mọi nhãn
    # lệch đúng một bậc — bản đồ vẫn đẹp y hệt, chỉ có tooltip nói sai giá, nên
    # đây là loại lỗi rất dễ lọt.
    dac_trung = []
    duong = cs.get_paths()
    n_dai = len(muc) + 1
    for i, p in enumerate(duong):
        da_giac = _tach_lo_thung(p)
        if not da_giac:
            continue
        tu = float(muc[i - 1]) if i >= 1 else None
        den = float(muc[i]) if i < len(muc) else None
        if tu is None:
            nhan = f"dưới {den:,.0f}{don_vi}"
        elif den is None:
            nhan = f"trên {tu:,.0f}{don_vi}"
        else:
            nhan = f"{tu:,.0f}–{den:,.0f}{don_vi}"
        if chi_tieu == "phan_tan":
            nhan = "giá quanh đây dao động " + nhan
        for dg in da_giac:
            dac_trung.append({
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": dg},
                "properties": {
                    "bac": i,
                    "gia_tu": round(tu, 1) if tu is not None else None,
                    "gia_den": round(den, 1) if den is not None else None,
                    "nhan": nhan.replace(",", "."),
                },
            })
    plt.close(fig)

    return {
        "type": "FeatureCollection",
        "features": dac_trung,
        "so_bac": n_dai - 1,
        "chi_tieu": chi_tieu,
        "don_vi": don_vi,
        "gia_min": round(lo, 1),
        "gia_max": round(hi, 1),
        "so_tin_ve": int(d.attrs.get("so_ve", len(d))),
        "so_tin_bo": int(d.attrs.get("so_bi_loai", 0)),
        "so_ghim": int(d.attrs.get("so_ghim", 0)),
        "sigma_km": (round(float(sigma_km), 2) if sigma_km is not None
                     else round(sigma_tu_toa_do(d), 2)),
        "sigma_theo_tin": sigma_km is None,
    }


def dai_mau(gia_m2_trieu: pd.Series, duoi: float = 0.10,
            tren: float = 0.90) -> tuple[float, float]:
    """Khoảng giá ứng với hai đầu thang màu.

    Cắt ở phân vị 10–90 chứ không dùng min–max: chỉ cần một tin 380 triệu/m² là
    toàn bộ phần còn lại của Hà Nội bị dồn vào một sắc màu duy nhất.
    """
    return (float(gia_m2_trieu.quantile(duoi)),
            float(gia_m2_trieu.quantile(tren)))


def gia_khu_vuc(diem: tuple[float, float], luoi: pd.DataFrame) -> dict | None:
    """Tra mức giá/m² của khu vực chứa `diem`. Trả ô NHỎ NHẤT chứa điểm đó.

    Dùng cho phần "giá tham khảo quanh đây" trong ứng dụng: người dùng bấm lên
    bản đồ, hệ thống trả về mặt bằng giá khu vực kèm số tin làm căn cứ.
    """
    vi, kinh = diem
    trong = luoi[(luoi["vi_min"] <= vi) & (vi < luoi["vi_max"])
                 & (luoi["kinh_min"] <= kinh) & (kinh < luoi["kinh_max"])]
    if trong.empty:
        return None
    r = trong.sort_values("muc_km").iloc[0]
    return {"gia_m2_trieu": float(r["gia_m2_trieu"]),
            "khoang_q25_q75_trieu": (round(r["gia_m2_q25"] / 1e6, 2),
                                     round(r["gia_m2_q75"] / 1e6, 2)),
            "so_tin": int(r["so_tin"]), "muc_km": float(r["muc_km"]),
            "du_tin_cay": bool(r["du_tin_cay"])}


def sang_geojson(luoi: pd.DataFrame, duong_dan: Path) -> None:
    """Xuất ra GeoJSON để Streamlit/folium/QGIS mở được thẳng."""
    feats = []
    for _, r in luoi.iterrows():
        feats.append({
            "type": "Feature",
            "properties": {
                "gia_m2_trieu": float(r["gia_m2_trieu"]),
                "so_tin": int(r["so_tin"]),
                "muc_km": float(r["muc_km"]),
                "kc_trung_tam_km": float(r["kc_trung_tam_km"]),
                "du_tin_cay": bool(r["du_tin_cay"]),
            },
            "geometry": {"type": "Polygon", "coordinates": [[
                [float(r["kinh_min"]), float(r["vi_min"])],
                [float(r["kinh_max"]), float(r["vi_min"])],
                [float(r["kinh_max"]), float(r["vi_max"])],
                [float(r["kinh_min"]), float(r["vi_max"])],
                [float(r["kinh_min"]), float(r["vi_min"])],
            ]]},
        })
    duong_dan.write_text(json.dumps(
        {"type": "FeatureCollection", "features": feats}, ensure_ascii=False),
        encoding="utf-8")


def ve_luoi(luoi: pd.DataFrame, duong_dan: Path) -> None:
    """Ảnh xem thử: mỗi ô một hình chữ nhật tô theo giá/m²."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    v = luoi["gia_m2_trieu"]
    norm = Normalize(v.quantile(.05), v.quantile(.95))
    cmap = plt.get_cmap("YlOrRd")

    fig, ax = plt.subplots(figsize=(9, 9))
    for _, r in luoi.iterrows():
        # Ô mỏng dữ liệu thì vẽ mờ và gạch chéo. Không giấu đi — người xem vẫn
        # thấy có số liệu ở đó — nhưng mắt tự khắc bớt tin, đúng như nên thế.
        thua = r["so_tin"] < N_TOI_THIEU
        ax.add_patch(Rectangle(
            (r["kinh_min"], r["vi_min"]),
            r["kinh_max"] - r["kinh_min"], r["vi_max"] - r["vi_min"],
            facecolor=cmap(norm(r["gia_m2_trieu"])),
            alpha=.35 if thua else 1.0,
            hatch="///" if thua else None,
            edgecolor="white", linewidth=.25))
    lo_v, hi_v, lo_k, hi_k = HN_BOX
    ax.set_xlim(luoi["kinh_min"].min() - .01, luoi["kinh_max"].max() + .01)
    ax.set_ylim(luoi["vi_min"].min() - .01, luoi["vi_max"].max() + .01)
    ax.set_aspect(1 / _HE_SO_KINH)
    ax.set_xlabel("kinh độ"); ax.set_ylabel("vĩ độ")
    ax.set_title(f"Mặt bằng giá theo ô lưới — {len(luoi)} ô, "
                 f"lưới {luoi['muc_km'].min()}–{luoi['muc_km'].max()} km")
    fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                 label="giá/m² trung vị (triệu đồng)", shrink=.8)
    fig.tight_layout()
    fig.savefig(duong_dan, dpi=140)
    plt.close(fig)


# =============================================================================
# CHẠY
# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser(description="Lưới giá và tìm tin quanh một điểm")
    ap.add_argument("--quanh", nargs=2, type=float, metavar=("VI_DO", "KINH_DO"))
    ap.add_argument("--ban-kinh", type=float, default=1.0)
    ap.add_argument("--n-toi-thieu", type=int, default=N_TOI_THIEU)
    a = ap.parse_args()

    if a.quanh:
        d = tim_quanh(tuple(a.quanh), a.ban_kinh)
        print(f"{len(d)} tin trong bán kính {a.ban_kinh} km quanh "
              f"({a.quanh[0]}, {a.quanh[1]})")
        if len(d):
            cot = ["loai", "khoang_cach_km", "TARGET_gia_vnd", "dien_tich",
                   "quan_huyen", "phuong_xa", "duong_pho", "nguon_toa_do"]
            print(d[cot].head(15).to_string(index=False))
            print(f"\ngiá/m² trung vị quanh đây: "
                  f"{d['gia_tren_m2'].median()/1e6:.1f} triệu")
        return

    # KHÔNG gộp hai nhánh vào một bản đồ. Giá/m² của chung cư tính trên diện
    # tích sàn (~50–100 triệu), của nhà đất tính trên diện tích đất (~200–500
    # triệu). Trộn vào một ô thì trung vị chẳng còn nghĩa gì — nó chỉ nói lên
    # ô đó tình cờ có nhiều tin loại nào hơn.
    for ten in ("nhadat", "chungcu"):
        g = luoi_gia([ten], n_toi_thieu=a.n_toi_thieu)
        g.to_excel(DATA_CLEAN / f"luoi_gia_{ten}.xlsx", index=False)
        sang_geojson(g, DATA_CLEAN / f"luoi_gia_{ten}.geojson")
        ve_luoi(g, DATA_CLEAN / f"ban_do_nhiet_{ten}.png")
        thua = int((~g["du_tin_cay"]).sum())
        print(f"{ten:8s}: {len(g):4d} ô ({thua} ô dưới {a.n_toi_thieu} tin) | "
              + " | ".join(f"{k} km: {v}" for k, v in
                           g["muc_km"].value_counts().sort_index().items()))
    print("-> data_clean/luoi_gia_*.xlsx/.geojson, ban_do_nhiet_*.png")


if __name__ == "__main__":
    main()
