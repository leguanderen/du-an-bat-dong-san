"""
ranh_gioi.py — Bóc ranh giới phường/xã Hà Nội từ bộ dữ liệu GIS hành chính.

NGUỒN
-----
Bộ `vietnamese-provinces-database` (thanglequoc), dựng từ Bản đồ hành chính
Việt Nam do Nhà xuất bản Tài nguyên - Môi trường và Bản đồ Việt Nam phát hành.
Trường `gis_server_id` ghi rõ `diaphanhanhchinhcapxa_2025`, tức là ranh giới
SAU sáp nhập 1/7/2025 — đúng cấp mà toàn bộ dự án đang dùng.

VÌ SAO PHẢI TỰ ĐẶT TÊN CHO TỪNG ĐA GIÁC
---------------------------------------
File GIS chỉ có `ward_code`, không có tên phường. Bộ dữ liệu tên nằm ở file
khác, nhưng ta không cần tải thêm: mình đã có 8.500 tin rao kèm toạ độ VÀ tên
phường (do `gazetteer_phuong.py` chuẩn hoá). Nên chỉ việc hỏi ngược lại —
"những tin nào rơi vào bên trong đa giác này, và chúng khai phường nào" — rồi
lấy tên chiếm đa số.

Cách này còn lợi hơn việc tải bảng tên về: nó KIỂM CHỨNG luôn phần chuẩn hoá
địa giới. Nếu tên rút ra từ toạ độ khớp với tên phân tích từ chữ, hai con
đường độc lập cùng chỉ về một chỗ. Chỗ nào lệch là chỗ đáng đi xem lại, và tỉ
lệ khớp là một con số đo được để đưa vào khoá luận.

CÁCH CHẠY
---------
    python ranh_gioi.py            # bóc, gán tên, ghi ra data_clean/
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from gan_toa_do import HN_BOX, thu_muc

BASE_DIR = Path(__file__).resolve().parent
DATA_RAW = thu_muc("data_raw")
DATA_CLEAN = thu_muc("data_clean")

TEP_GEOJSON = "ranh_gioi_phuong_hanoi.geojson"

# Khung Hà Nội, nới rộng một chút so với HN_BOX để không cắt mất xã rìa.
VI_MIN, VI_MAX, KINH_MIN, KINH_MAX = 20.50, 21.42, 105.25, 106.06

# Một bản ghi gis_wards trong file .sql:
#   ('00004','diaphanhanhchinhcapxa_2025.3256',2.97,
#    ST_GeomFromText('POLYGON((...))', 4326),
#    ST_GeomFromText('MULTIPOLYGON(((...)))', 4326))
_RE_WARD = re.compile(
    r"\('(?P<ma>\d{5})','(?P<sv>diaphanhanhchinhcapxa[^']*)',"
    r"(?P<dt>[\d.]+),"
    r"ST_GeomFromText\('(?P<bbox>POLYGON\(\([^']*?\)\))', 4326\),"
    r"ST_GeomFromText\('(?P<geom>MULTIPOLYGON\(\(\(.*?\)\)\))', 4326\)",
    re.DOTALL)


def _tam_bbox(wkt_bbox: str) -> tuple[float, float]:
    """Tâm của hộp bao, dùng để lọc nhanh trước khi phân tích hình học nặng."""
    so = re.findall(r"(-?\d+\.?\d*) (-?\d+\.?\d*)", wkt_bbox)
    xs = [float(a) for a, _ in so]
    ys = [float(b) for _, b in so]
    return (sum(ys) / len(ys), sum(xs) / len(xs))


def doc_ranh_gioi(thu_muc_sql: Path | None = None) -> pd.DataFrame:
    """Đọc mọi file part-*.sql, giữ lại các phường/xã nằm trong khung Hà Nội.

    Lọc bằng hộp bao TRƯỚC khi dựng hình học: cả nước có 3.321 phường/xã, dựng
    hết thành đối tượng shapely rồi mới lọc thì tốn thời gian và bộ nhớ vô ích
    khi ta chỉ cần khoảng 126 cái.
    """
    from shapely import wkt as _wkt

    thu_muc_sql = thu_muc_sql or DATA_RAW
    tep = sorted(thu_muc_sql.glob("postgresql_ImportData_gis-part-*.sql"))
    if not tep:
        raise FileNotFoundError(
            f"Không thấy file postgresql_ImportData_gis-part-*.sql trong "
            f"{thu_muc_sql}")

    ban_ghi = []
    for t in tep:
        s = t.read_text(encoding="utf-8", errors="replace")
        # Chỉ quét phần gis_wards; phần gis_provinces cũng khớp mẫu nhưng
        # không cần và làm chậm.
        i = s.find("-- DATA for gis_wards --")
        if i < 0:
            continue
        for m in _RE_WARD.finditer(s, i):
            vi, kinh = _tam_bbox(m.group("bbox"))
            if not (VI_MIN <= vi <= VI_MAX and KINH_MIN <= kinh <= KINH_MAX):
                continue
            ban_ghi.append({
                "ma_xa": m.group("ma"),
                "dien_tich_km2": float(m.group("dt")),
                "vi_tam": vi, "kinh_tam": kinh,
                "hinh": _wkt.loads(m.group("geom")),
                "tep_nguon": t.name,
            })
    d = pd.DataFrame(ban_ghi).drop_duplicates(subset="ma_xa")
    return d.reset_index(drop=True)


def dat_ten(ranh: pd.DataFrame, loai=("chungcu", "nhadat"),
            n_toi_thieu: int = 3) -> pd.DataFrame:
    """Gán tên phường cho từng đa giác bằng cách bỏ phiếu từ tin rao bên trong.

    Trả về thêm ba cột:
      - `phuong_moi`: tên chiếm đa số trong số tin rơi vào đa giác
      - `so_tin_trong`: bao nhiêu tin đã bỏ phiếu
      - `ty_le_dong_thuan`: tỉ lệ tin khai đúng cái tên đó

    `ty_le_dong_thuan` chính là thước đo chéo giữa hai con đường độc lập —
    phân tích địa chỉ bằng chữ, và vị trí địa lý. Đa giác nào đồng thuận thấp
    là đa giác có tin bị gán nhầm phường hoặc toạ độ lệch, và đáng đi xem lại.
    """
    from shapely.geometry import Point
    from shapely.strtree import STRtree
    import ban_do as bd

    tin = pd.concat([bd.nap(l) for l in loai], ignore_index=True)
    tin = tin.dropna(subset=["latitude", "longitude", "phuong_moi"])

    hinh = list(ranh["hinh"])
    cay = STRtree(hinh)
    diem = [Point(x, y) for x, y in zip(tin["longitude"], tin["latitude"])]

    # `query` với predicate="within" trả về cặp (chỉ số điểm, chỉ số đa giác)
    # cho mọi cặp thực sự lồng nhau — nhanh hơn nhiều so với duyệt tay.
    cap = cay.query(diem, predicate="within")
    thuoc = pd.DataFrame({"i_tin": cap[0], "i_vung": cap[1]})
    thuoc["ten"] = tin["phuong_moi"].to_numpy()[thuoc["i_tin"]]

    ket = []
    for i_vung, nhom in thuoc.groupby("i_vung"):
        dem = nhom["ten"].value_counts()
        ket.append({
            "i_vung": i_vung,
            "phuong_moi": dem.index[0],
            "so_tin_trong": int(dem.sum()),
            "ty_le_dong_thuan": float(dem.iloc[0] / dem.sum()),
        })
    k = pd.DataFrame(ket).set_index("i_vung")

    out = ranh.copy()
    for c in ("phuong_moi", "so_tin_trong", "ty_le_dong_thuan"):
        out[c] = k[c].reindex(range(len(out))).to_numpy() if len(k) else None
    out["so_tin_trong"] = out["so_tin_trong"].fillna(0).astype(int)
    out.loc[out["so_tin_trong"] < n_toi_thieu, "phuong_moi"] = None
    return out


def sang_geojson(ranh: pd.DataFrame, don_gian_m: float = 0.0002) -> dict:
    """Xuất GeoJSON. `don_gian_m` làm thưa đỉnh để trình duyệt vẽ cho nổi.

    0,0002 độ ≈ 20 m — nhỏ hơn nhiều so với sai số toạ độ của tin rao (0,67 km
    ở mức đường), nên việc làm thưa này không làm mất thông tin nào có thật,
    trong khi kích thước file giảm mạnh.
    """
    from shapely.geometry import mapping

    ft = []
    for r in ranh.itertuples():
        h = r.hinh.simplify(don_gian_m, preserve_topology=True)
        ft.append({
            "type": "Feature",
            "geometry": mapping(h),
            "properties": {
                "ma_xa": r.ma_xa,
                "phuong_moi": getattr(r, "phuong_moi", None),
                "dien_tich_km2": round(r.dien_tich_km2, 2),
                "so_tin_trong": int(getattr(r, "so_tin_trong", 0) or 0),
                "ty_le_dong_thuan": (
                    round(float(r.ty_le_dong_thuan), 3)
                    if getattr(r, "ty_le_dong_thuan", None) == r.ty_le_dong_thuan
                    and r.ty_le_dong_thuan is not None else None),
            },
        })
    return {"type": "FeatureCollection", "features": ft}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sql", type=Path, default=None)
    a = ap.parse_args()

    ranh = doc_ranh_gioi(a.sql)
    print(f"Đọc được {len(ranh)} phường/xã trong khung Hà Nội")

    ranh = dat_ten(ranh)
    co_ten = ranh["phuong_moi"].notna()
    print(f"  đặt tên được {int(co_ten.sum())} vùng "
          f"({int((~co_ten).sum())} vùng không đủ 3 tin bên trong)")
    if co_ten.any():
        dt = ranh.loc[co_ten, "ty_le_dong_thuan"]
        print(f"  đồng thuận giữa toạ độ và tên phân tích từ chữ: "
              f"trung vị {dt.median():.1%}, "
              f"{int((dt >= 0.8).sum())}/{int(co_ten.sum())} vùng đạt trên 80%")
        thap = ranh.loc[co_ten & (ranh['ty_le_dong_thuan'] < 0.6),
                        ["phuong_moi", "so_tin_trong", "ty_le_dong_thuan"]]
        if len(thap):
            print(f"  {len(thap)} vùng đồng thuận dưới 60% — đáng xem lại:")
            print(thap.sort_values("ty_le_dong_thuan").head(8).to_string(index=False))

    gj = sang_geojson(ranh)
    ra = DATA_CLEAN / TEP_GEOJSON
    ra.write_text(json.dumps(gj, ensure_ascii=False), encoding="utf-8")
    print(f"Đã ghi {ra}  ({ra.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
