"""
tien_ich.py — Chấm điểm Tiện ích Sống 1–100 cho từng bất động sản.

Ý tưởng đơn giản: giá nhà không chỉ nằm ở căn nhà. Hai căn giống hệt nhau,
một căn cách trường tiểu học 300 m và chợ 400 m, căn kia phải đi 3 km mới có
trường — người mua trả tiền cho khác biệt đó. Module này biến khác biệt ấy
thành con số.

BA BƯỚC
-------
1. `--tai`   : tải vị trí tiện ích Hà Nội từ OpenStreetMap (Overpass API),
               lưu vào kho SQLite để lần sau khỏi tải lại.
2. `--dac-trung` : với mỗi tin, tính khoảng cách tới tiện ích GẦN NHẤT của từng
               nhóm và số tiện ích trong bán kính 1 km.
3. `--cham-diem` : quy các khoảng cách đó về một điểm 1–100.

    python tien_ich.py --tai            # chạy một lần, cần mạng
    python tien_ich.py --dac-trung --cham-diem
    python tien_ich.py --cham-diem --trong-so hoc   # trọng số học từ dữ liệu

TRỌNG SỐ: ĐỪNG TỰ BỊA
---------------------
Chỗ dễ sai nhất của mọi "chỉ số tiện ích" là người làm tự gán trọng số theo
cảm tính ("trường học 30%, bệnh viện 20%…"). Hỏi lại "vì sao 30 mà không phải
20" thì không trả lời được — và hội đồng chấm khoá luận sẽ hỏi đúng câu đó.

Ở đây có hai chế độ, khai báo rõ ràng:

  * `deu`  — mọi nhóm trọng số bằng nhau. Không giả vờ biết cái gì quan trọng
             hơn. Minh bạch, dễ bảo vệ, dùng làm mặc định.
  * `hoc`  — hồi quy log(giá/m²) theo các điểm thành phần, có khống chế phường
             và diện tích, rồi lấy hệ số (đã ép không âm, chuẩn hoá về tổng 1)
             làm trọng số.

ĐÃ ĐO: CHẾ ĐỘ `hoc` KHÔNG DÙNG ĐƯỢC — HÃY DÙNG `deu`
----------------------------------------------------
Chạy bootstrap 500 lần (`--kiem-trong-so`) trên dữ liệu chung cư: **không một
trọng số nào phân biệt được với 0.** Trọng số lớn nhất là chợ/siêu thị 0,40
nhưng khoảng tin cậy 95% là [0,00; 0,99] — trùm gần hết miền giá trị, và trong
12% số lần lặp nó bị ép về đúng 0. Metro thì 92% số lần bằng 0.

Lý do: sáu điểm thành phần tương quan rất mạnh (gần chợ thì thường cũng gần
trường, gần bệnh viện — đều là biến thể của "gần trung tâm"). Đa cộng tuyến làm
hồi quy chia hệ số gần như tuỳ tiện giữa chúng.

Nên `deu` là mặc định, và khi bảo vệ hãy nói thẳng: "dữ liệu không đủ để tách
được tiện ích nào quan trọng hơn tiện ích nào, nên tôi để trọng số bằng nhau."
Câu đó chắc chắn hơn nhiều so với một bảng trọng số không lặp lại được.

HÀM SUY GIẢM THEO KHOẢNG CÁCH
-----------------------------
Điểm thành phần dùng dạng suy giảm mũ:  s = exp(−d / d₀).

Chọn dạng này vì nó khớp trực giác đi bộ: từ 100 m ra 400 m là khác biệt lớn,
từ 3 km ra 3,3 km thì gần như chẳng ai quan tâm — hàm mũ giảm nhanh ở gần và
phẳng dần ở xa, còn hàm tuyến tính thì phạt đều nhau, sai về mặt hành vi.
d₀ là "khoảng cách nửa tiện", đặt riêng cho từng nhóm: chợ 0,5 km (đi bộ hằng
ngày), bệnh viện 2 km (thỉnh thoảng, thường đi xe).

CẢNH BÁO VỀ DỮ LIỆU
-------------------
OpenStreetMap ở Việt Nam do cộng đồng đóng góp nên KHÔNG đầy đủ đều nhau: nội
thành gắn thẻ dày, ngoại thành thưa. Một căn ở Ba Vì có thể bị chấm điểm thấp
vì OSM chưa ai gắn cái chợ ngay đầu làng, chứ không phải vì ở đó không có chợ.
Cột `so_tien_ich_1km` để cạnh điểm chính là để nhìn thấy điều đó: điểm thấp mà
tổng số tiện ích quanh vùng cũng gần bằng 0 thì nên nghi ngờ dữ liệu, không
nên kết luận về khu vực.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from gan_toa_do import thu_muc

BASE_DIR = Path(__file__).resolve().parent
DATA_RAW = thu_muc("data_raw")
DATA_CLEAN = thu_muc("data_clean")
KHO = DATA_RAW / "osm_tien_ich.sqlite"

OVERPASS = "https://overpass-api.de/api/interpreter"
# Khung bao Hà Nội, nới rộng 0,05° để tiện ích ngay sát ranh giới vẫn được tính
# (người ở mép Gia Lâm vẫn đi bệnh viện bên Hưng Yên).
KHUNG = (20.48, 105.23, 21.44, 106.08)     # nam, tây, bắc, đông

# Mỗi nhóm: (tên, bộ lọc Overpass, d₀ tính bằng km)
#
# d₀ chọn theo tần suất sử dụng thực tế, không theo cảm hứng:
#   - chợ/siêu thị: đi gần như hằng ngày, thường đi bộ    -> 0,5 km
#   - trường học  : đưa đón hai lượt mỗi ngày             -> 0,8 km
#   - công viên   : đi dạo, không bắt buộc                -> 1,0 km
#   - metro/ga    : đi làm, chấp nhận đi bộ xa hơn        -> 1,2 km
#   - bệnh viện   : thỉnh thoảng, đi xe                   -> 2,0 km
#   - đại học     : ảnh hưởng gián tiếp (cho thuê)        -> 2,0 km
NHOM = {
    "cho_sieu_thi": (["node[shop=supermarket]", "way[shop=supermarket]",
                      "node[amenity=marketplace]", "way[amenity=marketplace]",
                      "node[shop=convenience]"], 0.5),
    "truong_hoc": (["node[amenity=school]", "way[amenity=school]",
                    "node[amenity=kindergarten]", "way[amenity=kindergarten]"], 0.8),
    "cong_vien": (["node[leisure=park]", "way[leisure=park]",
                   "way[leisure=garden]"], 1.0),
    "metro_ga": (["node[railway=station]", "node[railway=subway_entrance]",
                  "node[public_transport=station]"], 1.2),
    "benh_vien": (["node[amenity=hospital]", "way[amenity=hospital]",
                   "node[amenity=clinic]", "way[amenity=clinic]"], 2.0),
    "dai_hoc": (["node[amenity=university]", "way[amenity=university]",
                 "node[amenity=college]", "way[amenity=college]"], 2.0),
}

TEP_TIN = {"chungcu": "chungcu_clean_geo.xlsx", "nhadat": "nhadat_clean_geo.xlsx"}
BAN_KINH_DEM_KM = 1.0


# =============================================================================
# (1) TẢI DỮ LIỆU OSM
# =============================================================================

def _truy_van(bo_loc: list[str]) -> str:
    nam, tay, bac, dong = KHUNG
    than = "".join(f"{f}({nam},{tay},{bac},{dong});" for f in bo_loc)
    return f"[out:json][timeout:180];({than});out center;"


def _goi_overpass(q: str, so_lan: int = 3) -> dict:
    """Gọi Overpass, lùi dần nếu máy chủ bận.

    Overpass là hạ tầng cộng đồng chạy bằng tiền quyên góp. Mã 429/504 nghĩa là
    "đang quá tải, chờ chút" — nên chờ thật, và khai báo User-Agent trung thực
    để quản trị viên biết ai đang gọi.
    """
    yc = urllib.request.Request(
        OVERPASS, data=q.encode("utf-8"),
        headers={"User-Agent": "khoa-luan-dinh-gia-bds/1.0 (sinh vien CNTT)"})
    for lan in range(so_lan):
        try:
            with urllib.request.urlopen(yc, timeout=300) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code in (429, 504) and lan < so_lan - 1:
                cho = 30 * (lan + 1)
                print(f"    máy chủ bận ({e.code}), chờ {cho}s…", flush=True)
                time.sleep(cho)
                continue
            raise
    raise RuntimeError("Overpass không phản hồi")


def _mo_kho() -> sqlite3.Connection:
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(KHO)
    conn.execute("""CREATE TABLE IF NOT EXISTS tien_ich (
        nhom TEXT, osm_id TEXT, ten TEXT, lat REAL, lon REAL,
        PRIMARY KEY (nhom, osm_id))""")
    return conn


def tai_osm(chi_nhom: str | None = None) -> pd.DataFrame:
    conn = _mo_kho()
    for ten, (bo_loc, _) in NHOM.items():
        if chi_nhom and ten != chi_nhom:
            continue
        print(f"  tải {ten}…", end=" ", flush=True)
        d = _goi_overpass(_truy_van(bo_loc))
        hang = []
        for e in d.get("elements", []):
            lat = e.get("lat") or (e.get("center") or {}).get("lat")
            lon = e.get("lon") or (e.get("center") or {}).get("lon")
            if lat is None or lon is None:
                continue
            hang.append((ten, f"{e['type']}/{e['id']}",
                         (e.get("tags") or {}).get("name"), lat, lon))
        conn.executemany("INSERT OR REPLACE INTO tien_ich VALUES (?,?,?,?,?)", hang)
        conn.commit()
        print(f"{len(hang)} điểm")
        time.sleep(5)          # tử tế với máy chủ dùng chung
    out = pd.read_sql("SELECT * FROM tien_ich", conn)
    conn.close()
    out.to_csv(DATA_RAW / "osm_tien_ich.csv", index=False, encoding="utf-8-sig")
    return out


def nap_tien_ich() -> pd.DataFrame:
    """Ưu tiên kho SQLite; không có thì đọc bản .csv đã xuất.

    Bản .csv là bản mang đi được — chép sang máy khác vẫn chạy, trong khi kho
    SQLite thì nặng và dễ quên chép. Cho đọc cả hai để không ai phải tải lại
    OpenStreetMap chỉ vì thiếu một file.
    """
    if KHO.exists():
        conn = sqlite3.connect(KHO)
        d = pd.read_sql("SELECT * FROM tien_ich", conn)
        conn.close()
        return d
    csv = DATA_RAW / "osm_tien_ich.csv"
    if csv.exists():
        return pd.read_csv(csv)
    raise FileNotFoundError(
        f"Chưa có {KHO.name} lẫn {csv.name}. Chạy `python tien_ich.py --tai` "
        "một lần (cần mạng) để tải vị trí tiện ích từ OpenStreetMap.")


# =============================================================================
# (2) KHOẢNG CÁCH TỚI TIỆN ÍCH
# =============================================================================
# Tính khoảng cách kiểu "mỗi tin so với mọi tiện ích" là 6.192 × 40.000 phép
# tính cho mỗi nhóm. Thay vào đó chiếu toạ độ về mặt phẳng km rồi dùng KD-Tree:
# cùng kết quả, nhanh hơn vài trăm lần. Sai số của phép chiếu phẳng trong phạm
# vi một thành phố là dưới 0,1% — không đáng kể so với sai số toạ độ 0,67 km.

_KM_MOI_DO_VI = 111.32
_VI_GOC = 21.0


def _ve_phang(lat, lon) -> np.ndarray:
    return np.column_stack([
        np.asarray(lat, float) * _KM_MOI_DO_VI,
        np.asarray(lon, float) * _KM_MOI_DO_VI * np.cos(np.radians(_VI_GOC)),
    ])


def dac_trung_tien_ich(diem: pd.DataFrame, ti: pd.DataFrame,
                       ban_kinh_km: float = BAN_KINH_DEM_KM) -> pd.DataFrame:
    """Với mỗi dòng của `diem`: khoảng cách tới tiện ích gần nhất mỗi nhóm và
    số tiện ích trong bán kính `ban_kinh_km`."""
    from scipy.spatial import cKDTree

    P = _ve_phang(diem["latitude"], diem["longitude"])
    out = pd.DataFrame(index=diem.index)
    tong_dem = np.zeros(len(diem))

    for nhom in NHOM:
        sub = ti[ti["nhom"] == nhom]
        if sub.empty:
            out[f"kc_{nhom}_km"] = np.nan
            out[f"dem_{nhom}_1km"] = 0
            continue
        cay = cKDTree(_ve_phang(sub["lat"], sub["lon"]))
        d, _ = cay.query(P, k=1)
        dem = np.array([len(x) for x in cay.query_ball_point(P, ban_kinh_km)])
        out[f"kc_{nhom}_km"] = np.round(d, 3)
        out[f"dem_{nhom}_1km"] = dem
        tong_dem += dem

    out["so_tien_ich_1km"] = tong_dem.astype(int)
    return out


# =============================================================================
# (3) CHẤM ĐIỂM
# =============================================================================

def _diem_thanh_phan(dt: pd.DataFrame) -> pd.DataFrame:
    """Khoảng cách -> điểm 0–1 theo hàm suy giảm mũ exp(−d/d₀)."""
    s = pd.DataFrame(index=dt.index)
    for nhom, (_, d0) in NHOM.items():
        d = dt[f"kc_{nhom}_km"]
        s[nhom] = np.exp(-d / d0)
    return s.fillna(0.0)


def hoc_trong_so(df: pd.DataFrame, s: pd.DataFrame) -> pd.Series:
    """Trọng số học từ dữ liệu: hồi quy log(giá/m²) theo điểm thành phần.

    Khống chế phường và diện tích trước, vì nếu không thì mọi hệ số chỉ đang đo
    lại một điều duy nhất — "gần trung tâm thì đắt". Cách khống chế: lấy phần
    dư của log(giá/m²) sau khi trừ trung vị của phường, rồi mới hồi quy.

    Hệ số âm bị ép về 0: một trọng số âm nghĩa là "càng gần bệnh viện càng rẻ",
    có thể đúng về mặt tương quan nhưng không dùng được cho một chỉ số TIỆN ÍCH.
    Ép về 0 và nói rõ là đã ép, thay vì lặng lẽ để nó âm.
    """
    from sklearn.linear_model import Ridge

    y = np.log(df["gia_tren_m2"].replace([np.inf, -np.inf], np.nan))
    ok = y.notna() & s.notna().all(axis=1)
    y = y[ok]
    if "phuong_xa" in df.columns:
        # Dòng khuyết tên phường thì không có trung vị phường để trừ — lùi về
        # trung vị toàn cục thay vì để thành NaN rồi vỡ cả phép hồi quy.
        tv = y.groupby(df.loc[ok, "phuong_xa"].fillna("__khong_ro__")) \
              .transform("median").fillna(y.median())
        y = y - tv
    else:
        y = y - y.median()

    X = s.loc[ok]
    m = Ridge(alpha=1.0).fit((X - X.mean()) / X.std(ddof=0), y)
    w = pd.Series(m.coef_, index=X.columns).clip(lower=0)
    if w.sum() <= 0:
        print("  (không hệ số nào dương — lùi về trọng số đều)", file=sys.stderr)
        return pd.Series(1 / len(s.columns), index=s.columns)
    return w / w.sum()


def trong_so_co_khoang_tin_cay(df: pd.DataFrame, s: pd.DataFrame,
                               n_lap: int = 500) -> pd.DataFrame:
    """Lặp bootstrap để xem trọng số nào ĐO ĐƯỢC, trọng số nào chỉ là nhiễu.

    VÌ SAO CẦN
    ----------
    Sáu điểm thành phần tương quan rất mạnh với nhau: chỗ nào gần chợ thì
    thường cũng gần trường, gần bệnh viện — tất cả đều là biến thể của "gần
    trung tâm". Khi các biến đầu vào chồng nhau như vậy (đa cộng tuyến), hồi
    quy chia hệ số gần như tuỳ tiện giữa chúng: đổi dữ liệu một chút là thứ tự
    đảo lộn. Chạy một lần rồi công bố bảng trọng số là công bố một con số không
    lặp lại được.

    Chạy lại trên 500 mẫu bootstrap thì thấy ngay: trọng số nào có khoảng tin
    cậy KHÔNG chứa 0 là trọng số thật sự đo được; trọng số nào khoảng tin cậy
    trùm qua 0 thì phải nói thẳng là "không phân biệt được với 0", chứ không
    được trình bày như một phát hiện.
    """
    rng = np.random.default_rng(0)
    mau = []
    for _ in range(n_lap):
        i = rng.integers(0, len(df), len(df))
        try:
            mau.append(hoc_trong_so(df.iloc[i].reset_index(drop=True),
                                    s.iloc[i].reset_index(drop=True)))
        except Exception:
            continue
    m = pd.DataFrame(mau)
    out = pd.DataFrame({
        "trong_so": m.mean().round(3),
        "p2_5": m.quantile(.025).round(3),
        "p97_5": m.quantile(.975).round(3),
        "ti_le_bang_0": (m <= 1e-9).mean().round(3),
    })
    out["do_duoc"] = np.where(out["p2_5"] > 0, "có", "không phân biệt được với 0")
    return out.sort_values("trong_so", ascending=False)


def cham_diem(dt: pd.DataFrame, trong_so: pd.Series) -> pd.Series:
    """Điểm 1–100. Không dùng thang 0 vì "0 điểm" hàm ý không có gì cả, trong
    khi thực tế điểm thấp nhất vẫn là một nơi ở được."""
    s = _diem_thanh_phan(dt)
    diem = (s * trong_so).sum(axis=1) * 99 + 1
    return diem.round(1)


# =============================================================================
# CHẠY
# =============================================================================

def xu_ly(loai: str, ti: pd.DataFrame, che_do: str) -> pd.DataFrame:
    df = pd.read_excel(DATA_CLEAN / TEP_TIN[loai])
    if "gia_tren_m2" not in df.columns:
        cot = "dien_tich_m2" if loai == "chungcu" else "dien_tich_dat_m2"
        df["gia_tren_m2"] = df["TARGET_gia_vnd"] / df[cot]

    dt = dac_trung_tien_ich(df, ti)
    s = _diem_thanh_phan(dt)
    w = (hoc_trong_so(df, s) if che_do == "hoc"
         else pd.Series(1 / len(NHOM), index=list(NHOM)))

    out = pd.concat([df, dt], axis=1)
    out["diem_tien_ich"] = cham_diem(dt, w)
    out.to_excel(DATA_CLEAN / f"{loai}_tien_ich.xlsx", index=False)

    print(f"\n{loai}: {len(out)} dòng -> {loai}_tien_ich.xlsx")
    print("  trọng số  : " + ", ".join(f"{k} {v:.2f}" for k, v in w.items()))
    print(f"  điểm      : trung vị {out['diem_tien_ich'].median():.1f}, "
          f"thấp nhất {out['diem_tien_ich'].min():.1f}, "
          f"cao nhất {out['diem_tien_ich'].max():.1f}")
    for nhom in NHOM:
        print(f"  k/c {nhom:14s}: trung vị {dt[f'kc_{nhom}_km'].median():.2f} km")
    return w.rename(loai).to_frame().T


def main() -> None:
    ap = argparse.ArgumentParser(description="Chấm điểm tiện ích sống")
    ap.add_argument("--tai", action="store_true", help="tải OSM (cần mạng)")
    ap.add_argument("--nhom", help="chỉ tải một nhóm")
    ap.add_argument("--cham-diem", action="store_true")
    ap.add_argument("--trong-so", choices=["deu", "hoc"], default="deu")
    ap.add_argument("--kiem-trong-so", action="store_true",
                    help="bootstrap khoảng tin cậy cho trọng số học được")
    ap.add_argument("--loai", choices=["chungcu", "nhadat", "ca-hai"],
                    default="ca-hai")
    a = ap.parse_args()

    if a.tai:
        d = tai_osm(a.nhom)
        print(f"\nKho tiện ích: {len(d)} điểm")
        print(d["nhom"].value_counts().to_string())
    if not a.cham_diem:
        return

    ti = nap_tien_ich()
    print(f"Nạp {len(ti)} điểm tiện ích ({ti['nhom'].nunique()} nhóm).")
    loai = ["chungcu", "nhadat"] if a.loai == "ca-hai" else [a.loai]
    ws = [xu_ly(l, ti, a.trong_so) for l in loai]
    pd.concat(ws).to_excel(DATA_CLEAN / f"trong_so_tien_ich_{a.trong_so}.xlsx")

    if a.kiem_trong_so:
        with pd.ExcelWriter(DATA_CLEAN / "kiem_trong_so_tien_ich.xlsx") as w:
            for l in loai:
                df = pd.read_excel(DATA_CLEAN / TEP_TIN[l])
                cot = "dien_tich_m2" if l == "chungcu" else "dien_tich_dat_m2"
                if "gia_tren_m2" not in df.columns:
                    df["gia_tren_m2"] = df["TARGET_gia_vnd"] / df[cot]
                sc = _diem_thanh_phan(dac_trung_tien_ich(df, ti))
                bang = trong_so_co_khoang_tin_cay(df, sc)
                print(f"\nBootstrap trọng số — {l}:")
                print(bang.to_string())
                bang.to_excel(w, sheet_name=l)


if __name__ == "__main__":
    main()
