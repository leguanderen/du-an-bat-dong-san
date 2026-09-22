"""
geocode.py — Chuyển địa chỉ thành toạ độ (lat/lon) bằng Nominatim của OpenStreetMap.

CHẠY TRÊN MÁY BẠN, không chạy trong môi trường của Claude (ở đó Nominatim bị chặn).

    python geocode.py --input data_clean/chungcu_clean.xlsx --outdir data_clean
    python geocode.py --input data_clean/nhadat_clean.xlsx  --outdir data_clean

Có thể chạy SONG SONG với crawler — hai việc không đụng nhau.

THỜI GIAN DỰ KIẾN
-----------------
Nominatim giới hạn 1 yêu cầu/giây (điều khoản sử dụng, phải tôn trọng).
    Chung cư: 1.109 địa chỉ duy nhất  -> khoảng 19 phút
    Nhà đất : 2.486 địa chỉ duy nhất  -> khoảng 42 phút
Script chỉ gọi cho địa chỉ DUY NHẤT, không gọi lại cho các dòng trùng địa chỉ.

CÓ THỂ DỪNG GIỮA CHỪNG
----------------------
Mỗi kết quả được ghi ngay vào bộ nhớ đệm `geocode_cache.sqlite`. Nếu máy tắt
hoặc bạn bấm Ctrl+C, chạy lại lệnh cũ là nó tiếp tục từ chỗ dừng, không gọi lại
những địa chỉ đã xong. Bộ đệm dùng chung cho cả hai bộ dữ liệu.

CHIẾN LƯỢC — và ba bài học phải trả giá mới rút ra
--------------------------------------------------
Địa chỉ nhatot có dạng "<đường/ngõ>, <Phường X>, <Quận Y>, Hà Nội".

BÀI HỌC 1 — OSM đã bỏ cấp quận/huyện.
    Bản đầu nhét cả "Quận Thanh Xuân" vào truy vấn. Cấu trúc hiện tại của Hà
    Nội trong OSM là Phường -> Thành phố Hà Nội, không còn Quận ở giữa, nên
    chuỗi đó không khớp cấp hành chính nào và Nominatim khớp bừa: hỏi "Phường
    Nghĩa Tân, Quận Cầu Giấy" thì trả về một chi nhánh BIDV ở phường Nghĩa Đô.
    Kết quả: 65,2% số tin rơi xuống tâm quận.
    -> Bỏ cấp quận khỏi truy vấn, bóc tiền tố "Phường"/"Xã", và kiểm tra
       addresstype để từ chối kết quả là cửa hàng / toà nhà.

BÀI HỌC 2 — số nhà dính vào tên đường làm hỏng truy vấn.
    "83, Hào Nam" thất bại chỉ vì dấu phẩy; tham số `street` của Nominatim
    nhận dạng "<số nhà> <tên đường>" không dấu phẩy.
    -> street_candidates() sinh vài biến thể đã bóc số nhà và rác đầu chuỗi.

BÀI HỌC 3 — khớp đúng tên đường vẫn có thể sai chỗ hàng chục km.
    Sau hai sửa trên, tỉ lệ "cấp đường" tăng vọt nhưng toạ độ vẫn sai: hỏi
    "Nguyễn Trãi" ra một điểm ở Phường Thanh Liệt trong khi tin đăng ở Phường
    Thượng Đình (lệch 4 km); hỏi "Thành Công" ra Xã Liên Minh thay vì Phường
    Thành Công ở Ba Đình (lệch hơn 20 km). Tên đường ở Hà Nội vừa dài vừa
    trùng lặp giữa các khu vực.
    -> Đây là kiểu sai NGUY HIỂM NHẤT vì nó được gắn nhãn "tốt". Cách chặn:
       tra tâm phường trước làm điểm neo, rồi CHỈ NHẬN kết quả cấp đường nếu
       nó nằm trong bán kính MAX_KM_TU_TAM_PHUONG quanh neo đó.

QUY TRÌNH HIỆN TẠI
    1. Tra tâm phường (rẻ: ~140 phường cho 1.109 địa chỉ, hầu hết lấy từ đệm).
    2. Tra cấp đường, ưu tiên truy vấn có kèm tên phường.
    3. Nhận kết quả cấp đường CHỈ KHI cách tâm phường không quá 3 km.
    4. Không đạt thì dùng chính tâm phường; cuối cùng mới tới tâm quận.

Cột `do_chinh_xac` ghi lại mức nào đã dùng, kèm `cach_tam_phuong_km` để bạn tự
kiểm tra. ĐỪNG BỎ QUA CỘT NÀY: điểm ở mức "quan" chỉ là tâm quận, dùng để tính
khoảng cách tới tiện ích sẽ cho kết quả gần như vô nghĩa. Khi huấn luyện, nên
lọc giữ mức "duong" và "phuong", hoặc đưa chính `do_chinh_xac` vào làm đặc
trưng để model tự biết mà giảm tin cậy.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sqlite3
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"

# Điều khoản sử dụng Nominatim yêu cầu: tối đa 1 yêu cầu/giây và một User-Agent
# nhận dạng được. Đừng hạ số này xuống — sẽ bị chặn IP, và cũng là hành vi xấu
# với một dịch vụ miễn phí do cộng đồng vận hành.
RATE_LIMIT_SECONDS = 1.1
USER_AGENT = "khoa-luan-dinh-gia-bds-hanoi/1.0 (lien he: buiduyluongz@gmail.com)"

# Khung bao Hà Nội (viewbox) để Nominatim ưu tiên kết quả trong thành phố.
# Thứ tự: kinh độ trái, vĩ độ trên, kinh độ phải, vĩ độ dưới.
HANOI_VIEWBOX = "105.28,21.39,106.02,20.56"


# =============================================================================
# BỘ NHỚ ĐỆM
# =============================================================================

def open_cache(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS geocode (
            truy_van      TEXT PRIMARY KEY,
            lat           REAL,
            lon           REAL,
            do_chinh_xac  TEXT,
            ten_tra_ve    TEXT,
            thoi_diem     REAL
        )
    """)
    conn.commit()
    return conn


def cache_get(conn: sqlite3.Connection, query: str) -> dict | None:
    row = conn.execute(
        "SELECT lat, lon, do_chinh_xac, ten_tra_ve FROM geocode WHERE truy_van = ?",
        (query,),
    ).fetchone()
    if row is None:
        return None
    return {"lat": row[0], "lon": row[1], "do_chinh_xac": row[2], "ten_tra_ve": row[3]}


def cache_put(conn: sqlite3.Connection, query: str, result: dict) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO geocode VALUES (?,?,?,?,?,?)",
        (query, result.get("lat"), result.get("lon"), result.get("do_chinh_xac"),
         result.get("ten_tra_ve"), time.time()),
    )
    conn.commit()


# =============================================================================
# GỌI NOMINATIM
# =============================================================================

# Loại đối tượng OSM được CHẤP NHẬN cho từng mức truy vấn.
# Không có bộ lọc này, Nominatim sẵn sàng trả về một chi nhánh ngân hàng khi ta
# hỏi về một phường — đúng như đã xảy ra ở lần chạy đầu: hỏi "Phường Nghĩa Tân"
# thì nó trả về "BIDV, 184 Hoàng Quốc Việt, Phường Nghĩa Đô", tức sai cả loại
# đối tượng lẫn phường. Toạ độ như thế còn tệ hơn không có, vì nhìn thì tưởng đúng.
_ADMIN_TYPES = {"suburb", "quarter", "neighbourhood", "village", "town",
                "city_district", "administrative", "hamlet", "municipality",
                "county", "district", "borough", "locality", "city"}

ACCEPT_TYPES = {
    "duong": {"road", "residential", "pedestrian", "street", "highway",
              "house", "house_number", "place", "neighbourhood", "path"},
    "phuong": _ADMIN_TYPES,
    # Cấp quận dùng CHUNG tập loại với cấp phường. Lý do: sau cải cách bỏ cấp
    # quận/huyện, nhiều đơn vị cũ giờ nằm ở cấp xã/phường trong OSM — ví dụ
    # "Huyện Gia Lâm" nay là "Xã Gia Lâm", trả về addresstype='village'.
    # Tập loại hẹp ban đầu không chấp nhận 'village' nên truy vấn thất bại,
    # dù OSM có dữ liệu đúng.
    "quan": _ADMIN_TYPES,
}


def _in_hanoi(lat: float, lon: float) -> bool:
    """Chặn thô: điểm phải nằm trong khung bao Hà Nội."""
    return 20.55 <= lat <= 21.40 and 105.27 <= lon <= 106.03


def nominatim_lookup(query: dict | str, muc: str, retries: int = 3,
                     debug: bool = False) -> dict | None:
    """Một lần gọi Nominatim, có kiểm tra loại đối tượng trả về.

    `query` là chuỗi tự do, hoặc dict tham số có cấu trúc (street/city/county).
    Truy vấn có cấu trúc đáng tin hơn nhiều so với nhồi tất cả vào một chuỗi.
    """
    base = {"format": "json", "limit": 3, "countrycodes": "vn",
            "viewbox": HANOI_VIEWBOX, "bounded": 1, "addressdetails": 1}
    params = {**base, **(query if isinstance(query, dict) else {"q": query})}
    url = f"{NOMINATIM_URL}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})

    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=25) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            if not data:
                return None
            allowed = ACCEPT_TYPES.get(muc, set())
            for hit in data:                      # duyệt tối đa 3 kết quả đầu
                atype = (hit.get("addresstype") or hit.get("type") or "").lower()
                lat, lon = float(hit["lat"]), float(hit["lon"])
                if not _in_hanoi(lat, lon):
                    continue
                if allowed and atype not in allowed:
                    if debug:
                        print(f"      bỏ qua (loại '{atype}'): "
                              f"{hit.get('display_name','')[:60]}", file=sys.stderr)
                    continue
                return {"lat": lat, "lon": lon,
                        "ten_tra_ve": hit.get("display_name", ""),
                        "loai_osm": atype}
            return None
        except Exception as exc:
            if attempt == retries - 1:
                print(f"    ! lỗi sau {retries} lần thử: {exc}", file=sys.stderr)
                return None
            time.sleep(2 ** attempt * 2)      # lùi theo cấp số nhân
    return None


# Tiền tố cấp hành chính — cần bóc ra vì OSM lưu tên trần ("Thượng Đình")
# trong khi dữ liệu nhatot ghi kèm tiền tố ("Phường Thượng Đình").
_PREFIX = re.compile(r"^(Phường|Xã|Thị trấn|Quận|Huyện|Thị xã)\s+", re.IGNORECASE)


def strip_admin_prefix(name: object) -> str:
    if not isinstance(name, str):
        return ""
    return _PREFIX.sub("", name).strip()


# Rác hay đứng đầu phần "đường" trong địa chỉ nhatot, cần bóc trước khi hỏi OSM.
_RAC_DAU = re.compile(
    r"^(?:số nhà|số|lô đất|lô|căn|ô|tổ|khu|kđt|kdt|dự án|toà|tòa|block)\s*[\w/\-.]*\s*[,.]?\s*",
    re.IGNORECASE)
_SO_NHA = re.compile(r"^\d+[a-zA-Z]?\s*[/\-]?\s*[\d a-zA-Z/]*?\s*,\s*")


def street_candidates(duong: str) -> list[str]:
    """Các biến thể tên đường để thử với Nominatim, từ đầy đủ tới rút gọn.

    Địa chỉ nhatot nhồi số nhà, số lô và tên dự án vào chung phần đường:

        "2, Kim Giang"            -> Nominatim không hiểu dấu phẩy
        "83, Hào Nam"             -> tương tự
        "Lô đất HH1, An Khánh"    -> "Lô đất HH1" không phải tên đường
        "Lumiere EverGreen"       -> tên dự án, không phải đường

    Tham số `street` của Nominatim nhận dạng "<số nhà> <tên đường>" KHÔNG dấu
    phẩy. Ba ca đầu ở lần chạy trước thất bại chỉ vì dấu phẩy đó. Hàm này sinh
    ra vài biến thể, thử lần lượt; ca nào thực sự không phải tên đường thì rơi
    xuống cấp phường — đúng như mong muốn.
    """
    if not isinstance(duong, str) or not duong.strip():
        return []
    raw = re.sub(r"\s+", " ", duong).strip(" ,.")
    out: list[str] = []

    def add(x: str) -> None:
        x = re.sub(r"\s+", " ", x.replace(",", " ")).strip(" ,.")
        if len(x) > 2 and x not in out:
            out.append(x)

    # Thứ tự quan trọng: biến thể SẠCH NHẤT thử trước, vì mỗi lần thử tốn một
    # giây và một lệnh gọi tới dịch vụ miễn phí.
    add(_SO_NHA.sub("", _RAC_DAU.sub("", raw)))   # bỏ cả rác đầu lẫn số nhà
    add(_SO_NHA.sub("", raw))                     # "2, Kim Giang"  -> "Kim Giang"
    add(raw)                                      # nguyên bản, phẩy thành cách
    if "," in raw:
        add(raw.rsplit(",", 1)[-1])               # đoạn cuối thường là tên đường
    return out[:3]                                # tối đa 3 biến thể


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Khoảng cách giữa hai điểm trên mặt cầu, đơn vị km."""
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


# Bán kính tối đa cho phép giữa điểm cấp đường và tâm phường của tin đăng.
# Phường ở Hà Nội thường rộng 1–2 km, nên 3 km đã rất rộng rãi.
MAX_KM_TU_TAM_PHUONG = 3.0


def street_queries(duong, phuong) -> list[object]:
    """Các truy vấn cấp đường, XẾP THEO ĐỘ TIN CẬY GIẢM DẦN.

    Truy vấn CÓ KÈM PHƯỜNG phải đứng trước. Bài học từ lần chạy thứ ba: truy
    vấn có cấu trúc chỉ với `street=` cho tỉ lệ khớp cao nhưng khớp SAI CHỖ —
    hỏi "Nguyễn Trãi" thì OSM trả về một điểm ở Phường Thanh Liệt trong khi tin
    đăng ở Phường Thượng Đình, cách nhau 4 km; hỏi "Thành Công" thì ra Xã Liên
    Minh thay vì Phường Thành Công ở Ba Đình, lệch hơn 20 km.
    Tên đường ở Hà Nội vừa dài vừa trùng lặp, nên thiếu ngữ cảnh phường thì
    kết quả trông chính xác mà thực chất vô dụng.
    """
    p = strip_admin_prefix(phuong)
    out: list[object] = []
    for cand in street_candidates(duong):
        if p:
            out.append(f"{cand}, {p}, Hà Nội, Việt Nam")
    for cand in street_candidates(duong):
        out.append({"street": cand, "city": "Hà Nội", "country": "Việt Nam"})
    return out


def ward_queries(phuong) -> list[object]:
    p = strip_admin_prefix(phuong)
    return [f"{p}, Hà Nội, Việt Nam"] if p else []


def district_queries(quan) -> list[object]:
    q = strip_admin_prefix(quan)
    return [f"{q}, Hà Nội, Việt Nam"] if q else []


def _cache_key(muc: str, query: object) -> str:
    """Khoá đệm phải gồm cả MỨC, vì cùng một chuỗi có thể được hỏi ở mức khác
    nhau với bộ lọc loại đối tượng khác nhau."""
    if isinstance(query, dict):
        body = "&".join(f"{k}={v}" for k, v in sorted(query.items()))
        return f"[{muc}][struct] {body}"
    return f"[{muc}] {query}"


def _try(conn, muc, queries, stats, debug) -> dict | None:
    """Thử lần lượt các truy vấn của một mức, trả về kết quả đầu tiên hợp lệ."""
    for query in queries:
        key = _cache_key(muc, query)
        cached = cache_get(conn, key)
        if cached is not None:
            stats["tu_dem"] += 1
            if cached["lat"] is not None:
                return cached
            continue
        time.sleep(RATE_LIMIT_SECONDS)
        stats["goi_api"] += 1
        if debug:
            print(f"    [{muc}] {query}", file=sys.stderr)
        hit = nominatim_lookup(query, muc, debug=debug)
        if hit is None:
            cache_put(conn, key, {"lat": None, "lon": None,
                                  "do_chinh_xac": None, "ten_tra_ve": None})
            continue
        res = {"lat": hit["lat"], "lon": hit["lon"], "do_chinh_xac": muc,
               "ten_tra_ve": hit["ten_tra_ve"]}
        cache_put(conn, key, res)
        if debug:
            print(f"      -> {hit['loai_osm']}: {hit['ten_tra_ve'][:70]}", file=sys.stderr)
        return res
    return None


def geocode_one(conn: sqlite3.Connection, duong, phuong, quan,
                stats: dict, debug: bool = False) -> dict:
    """Geocode một địa chỉ, có NEO KIỂM TRA theo tâm phường.

    Quy trình:
        1. Tra tâm phường trước. Bước này rẻ bất ngờ: chung cư có 1.109 địa chỉ
           duy nhất nhưng chỉ khoảng 140 phường, nên hầu hết lượt tra sau đều
           lấy từ bộ đệm.
        2. Tra cấp đường (ưu tiên truy vấn có kèm tên phường).
        3. CHỈ NHẬN kết quả cấp đường nếu nó nằm trong bán kính cho phép quanh
           tâm phường. Đây là chốt chặn quan trọng nhất — không có nó, một điểm
           sai 20 km vẫn được gắn nhãn "cấp đường (tốt)".
        4. Không đạt thì lùi về chính tâm phường, rồi mới tới tâm quận.
    """
    tam_phuong = _try(conn, "phuong", ward_queries(phuong), stats, debug)

    duong_hit = _try(conn, "duong", street_queries(duong, phuong), stats, debug)
    if duong_hit is not None:
        if tam_phuong is None:
            # Không có neo để kiểm tra -> vẫn nhận nhưng đánh dấu là chưa kiểm chứng
            duong_hit = {**duong_hit, "do_chinh_xac": "duong_chua_kiem"}
            return duong_hit
        d = haversine_km(duong_hit["lat"], duong_hit["lon"],
                         tam_phuong["lat"], tam_phuong["lon"])
        if d <= MAX_KM_TU_TAM_PHUONG:
            return {**duong_hit, "cach_tam_phuong_km": round(d, 2)}
        if debug:
            print(f"      loại điểm cấp đường: cách tâm phường {d:.1f} km "
                  f"(> {MAX_KM_TU_TAM_PHUONG} km)", file=sys.stderr)
        stats["loai_vi_qua_xa"] = stats.get("loai_vi_qua_xa", 0) + 1

    if tam_phuong is not None:
        return tam_phuong

    tam_quan = _try(conn, "quan", district_queries(quan), stats, debug)
    if tam_quan is not None:
        return tam_quan

    return {"lat": None, "lon": None, "do_chinh_xac": "that_bai", "ten_tra_ve": None}


# =============================================================================
# CHẠY CHÍNH
# =============================================================================

def run(input_path: Path, outdir: Path, cache_path: Path, limit: int | None,
        debug: bool = False, retry_failed: bool = False) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_excel(input_path)
    for col in ("duong_pho", "phuong_xa", "quan_huyen"):
        if col not in df.columns:
            raise ValueError(f"File thiếu cột {col} — hãy chạy clean_*.py trước.")

    # Chỉ gọi API cho tổ hợp địa chỉ DUY NHẤT. Đây là chỗ tiết kiệm lớn nhất:
    # nhà đất có 4.584 dòng nhưng chỉ 2.486 địa chỉ khác nhau.
    keys = df[["duong_pho", "phuong_xa", "quan_huyen"]].drop_duplicates()
    if limit:
        keys = keys.head(limit)
    print(f"{len(df)} dòng -> {len(keys)} địa chỉ duy nhất cần tra")
    # Mỗi địa chỉ có thể tốn 1–5 lượt gọi (thử vài biến thể tên đường rồi mới
    # lùi xuống phường/quận), nên đưa ra khoảng thay vì một con số.
    lo_p = len(keys) * RATE_LIMIT_SECONDS / 60
    print(f"Ước tính thời gian: {lo_p:.0f}–{lo_p * 3:.0f} phút "
          f"(tuỳ bao nhiêu địa chỉ khớp ngay lần thử đầu; "
          f"lần chạy sau sẽ nhanh hơn nhiều nhờ bộ đệm)\n")

    conn = open_cache(cache_path)
    if retry_failed:
        n = conn.execute("DELETE FROM geocode WHERE lat IS NULL").rowcount
        conn.commit()
        print(f"Đã xoá {n} mục thất bại khỏi bộ đệm để thử lại.\n")
    stats = {"goi_api": 0, "tu_dem": 0}
    results = []
    t0 = time.time()

    # Nhịp in tiến độ phải co theo quy mô: chạy thử 20 địa chỉ mà 25 dòng mới in
    # một lần thì suốt nửa phút màn hình đứng im, trông y như bị treo.
    step = 25 if len(keys) >= 200 else max(1, len(keys) // 8)
    print("Đang gọi Nominatim, mỗi địa chỉ mất khoảng 1 giây...", flush=True)

    try:
        for i, (_, row) in enumerate(keys.iterrows(), start=1):
            res = geocode_one(conn, row["duong_pho"], row["phuong_xa"],
                              row["quan_huyen"], stats, debug=debug)
            results.append({**row.to_dict(), **res})
            if i % step == 0 or i == len(keys):
                elapsed = time.time() - t0
                con_lai = (len(keys) - i) * (elapsed / i) / 60
                thanh_cong = sum(1 for r in results if r["lat"] is not None)
                print(f"  {i}/{len(keys)}  thành công {thanh_cong}  "
                      f"(gọi API {stats['goi_api']}, từ đệm {stats['tu_dem']})  "
                      f"còn ~{con_lai:.0f} phút", flush=True)
    except KeyboardInterrupt:
        print("\nĐã dừng. Kết quả tra được đã lưu vào bộ đệm — chạy lại lệnh cũ "
              "để tiếp tục từ chỗ này.")

    coords = pd.DataFrame(results)
    merged = df.merge(coords, on=["duong_pho", "phuong_xa", "quan_huyen"], how="left")

    out_path = outdir / (input_path.stem + "_geo.xlsx")
    merged.to_excel(out_path, index=False)

    print("\n" + "=" * 58)
    print(f"Đã ghi: {out_path}")
    tong = len(merged)
    co_toa_do = int(merged["lat"].notna().sum())
    print(f"Có toạ độ: {co_toa_do}/{tong} ({co_toa_do/tong*100:.1f}%)")
    print("\nĐộ chính xác:")
    for muc, n in merged["do_chinh_xac"].value_counts(dropna=False).items():
        if pd.isna(muc):
            ten = ("chưa tra — đang chạy thử với --limit"
                   if limit else "chưa tra (không khớp được địa chỉ)")
        else:
            ten = {"duong": "cấp đường/ngõ  (tốt)",
                   "phuong": "tâm phường     (khá)",
                   "quan": "tâm quận       (KÉM — cân nhắc loại khi tính khoảng cách)",
                   "duong_chua_kiem": "cấp đường (CHƯA kiểm chứng — thiếu neo phường)",
                   "that_bai": "không tra được"}.get(muc, str(muc))
        print(f"  {ten:<55} {n:>5} dòng  ({n/tong*100:.1f}%)")

    if limit:
        print(f"\nĐây là lần chạy THỬ (--limit {limit}). Nếu phần lớn ra mức "
              f"'cấp đường' hoặc 'tâm phường' thì bỏ --limit để chạy đầy đủ.")
    elif co_toa_do == 0:
        print("\nKHÔNG tra được địa chỉ nào. Kiểm tra kết nối mạng, hoặc thử mở "
              "https://nominatim.openstreetmap.org trên trình duyệt xem có vào được không.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Geocode địa chỉ BĐS bằng Nominatim")
    ap.add_argument("--input", required=True, type=Path,
                    help="file *_clean.xlsx do clean_*.py sinh ra")
    ap.add_argument("--outdir", default=Path("data_clean"), type=Path)
    ap.add_argument("--cache", default=Path("data_clean/geocode_cache.sqlite"), type=Path,
                    help="bộ nhớ đệm dùng chung, cho phép dừng và chạy tiếp")
    ap.add_argument("--retry-failed", action="store_true",
                    help="xoá các mục THẤT BẠI trong bộ đệm rồi thử lại — dùng "
                         "sau khi logic truy vấn thay đổi, vì thất bại cũ được "
                         "ghi nhớ và sẽ chặn không cho thử lại")
    ap.add_argument("--debug", action="store_true",
                    help="in ra từng truy vấn và kết quả OSM — dùng khi cần chẩn đoán")
    ap.add_argument("--limit", type=int, default=None,
                    help="chỉ tra N địa chỉ đầu — dùng để thử nghiệm nhanh (vd --limit 20)")
    args = ap.parse_args()
    args.cache.parent.mkdir(parents=True, exist_ok=True)
    run(args.input, args.outdir, args.cache, args.limit, args.debug,
        args.retry_failed)
