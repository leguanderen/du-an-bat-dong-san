"""
crawl_api.py — Thu thập tin bất động sản qua API công khai của Chợ Tốt / Nhà Tốt.

    python crawl_api.py --cg 1010 --outdir data_raw     # căn hộ / chung cư, mua bán
    python crawl_api.py --cg 1020 --outdir data_raw     # nhà ở, mua bán
    python crawl_api.py --cg 1040 --outdir data_raw     # đất, mua bán

VÌ SAO DÙNG API THAY VÌ CÀO HTML
--------------------------------
Cào HTML bằng Web Scraper mất khoảng 58 giây mỗi tin và chỉ lấy được những gì
hiển thị trên màn hình. API trả về JSON có cấu trúc, trong đó có ba thứ mà bản
HTML không có:

    latitude / longitude   -> khỏi cần geocoding (đã tốn 4 vòng sửa mà chưa đạt)
    ward_name_v3           -> tên phường sau sáp nhập, khỏi cần bảng tra tự dựng
    list_time              -> dấu thời gian, tức có trục thời gian thật

Ngoài ra giá và diện tích về dưới dạng SỐ, không phải chuỗi "6,558 tỷ" phải
tự bóc — hết cả lớp lỗi đơn vị lẫn lỗi phân tích chuỗi.

TỐC ĐỘ VÀ HỆ QUẢ CỦA NÓ
-----------------------
Một lần chạy đầy đủ mất vài phút thay vì hàng chục giờ. Điều đáng giá không
phải là lần chạy đầu nhanh hơn, mà là từ nay CHẠY LẠI ĐƯỢC MỖI NGÀY. Kho dữ
liệu tích luỹ dần, khử trùng theo list_id, và mỗi tin được ghi nhận lần đầu
thấy / lần cuối thấy — từ đó suy ra được thời gian tin tồn tại trên sàn, một
tín hiệu thị trường mà bản chụp một lần không bao giờ có.

VỀ VIỆC GỌI CÓ CHỪNG MỰC
------------------------
Đây là API công khai, không có tài liệu chính thức và không công bố giới hạn.
Script này mặc định 1,5 giây một lệnh gọi, tự lùi theo cấp số nhân khi gặp lỗi
429/503, và dừng hẳn nếu bị từ chối liên tiếp thay vì cố đấm. Đừng hạ độ trễ
xuống để chạy nhanh hơn: bạn không tiết kiệm được mấy phút mà lại tạo rủi ro
cho chính mình, và đó cũng không phải cách cư xử đúng với hạ tầng của người khác.

Trong báo cáo nên ghi rõ nguồn dữ liệu (Chợ Tốt / Nhà Tốt), phương thức thu
thập (API công khai), thời điểm thu thập và số lượng bản ghi.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

from gan_toa_do import thu_muc

API_URL = "https://gateway.chotot.com/v1/public/ad-listing"

REGION_HANOI = 12000

# Danh mục CHA của toàn bộ bất động sản. Cào mã này là lấy hết mọi danh mục con.
CG_TAT_CA = 1000

# Mã danh mục đã xác minh bằng cách gọi thử và đối chiếu nội dung trả về.
DANH_MUC = {
    1000: "Tất cả bất động sản (danh mục cha)",
    1010: "Căn hộ / Chung cư",
    1020: "Nhà ở",
    1040: "Đất",
    1050: "Văn phòng / Mặt bằng",
}

# st=s là bán (sell). Bỏ tham số này đi thì ra tin cho thuê.
DEFAULT_ST = "s"

DEFAULT_DELAY = 1.5
MAX_BACKOFF_SECONDS = 120
MAX_LOI_LIEN_TIEP = 5

USER_AGENT = ("khoa-luan-dinh-gia-bds-hanoi/1.0 "
              "(nghien cuu hoc thuat; lien he: buiduyluongz@gmail.com)")


# =============================================================================
# KHO LƯU TRỮ
# =============================================================================
# Lưu NGUYÊN VĂN JSON của từng tin, không cắt gọt gì.
#
# Đây là bài học đắt giá từ chính dự án này: bản tiền xử lý cũ đã "chuẩn hoá"
# rồi vứt mất hai cột `dia_chi` và `mo_ta`, và chỉ cứu được nhờ file thô vẫn
# còn. Luật chuẩn hoá thay đổi liên tục; dữ liệu thô thì không. Giữ thô, xử lý
# ở tầng sau, nơi chạy lại chỉ tốn vài giây.

def open_store(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ads (
            list_id     INTEGER PRIMARY KEY,
            cg          INTEGER,
            json_tho    TEXT,
            lan_dau_thay REAL,
            lan_cuoi_thay REAL,
            so_lan_thay  INTEGER DEFAULT 1
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_cg ON ads(cg)")
    conn.commit()
    return conn


def upsert_ad(conn: sqlite3.Connection, ad: dict, cg: int, now: float) -> str:
    """Ghi một tin. Trả về 'moi' nếu chưa từng thấy, 'cu' nếu đã có.

    Tin đã có thì cập nhật `lan_cuoi_thay` và tăng bộ đếm — đó là cách kho dữ
    liệu tự sinh ra thông tin về thời gian tin tồn tại trên sàn.
    """
    list_id = ad.get("list_id")
    if list_id is None:
        return "bo_qua"
    # Ghi danh mục THẬT của tin (trường `category`), không phải danh mục đã yêu
    # cầu. Quan trọng khi cào bằng danh mục cha cg=1000: mọi tin sẽ về chung một
    # lượt, và chỉ trường này mới cho biết tin nào là căn hộ, tin nào là nhà ở.
    cg = ad.get("category") or cg
    row = conn.execute("SELECT so_lan_thay FROM ads WHERE list_id = ?",
                       (list_id,)).fetchone()
    if row is None:
        conn.execute(
            "INSERT INTO ads (list_id, cg, json_tho, lan_dau_thay, lan_cuoi_thay,"
            " so_lan_thay) VALUES (?,?,?,?,?,1)",
            (list_id, cg, json.dumps(ad, ensure_ascii=False), now, now))
        return "moi"
    conn.execute(
        "UPDATE ads SET json_tho = ?, lan_cuoi_thay = ?, so_lan_thay = so_lan_thay + 1"
        " WHERE list_id = ?",
        (json.dumps(ad, ensure_ascii=False), now, list_id))
    return "cu"


# =============================================================================
# GỌI API — có lùi dần khi bị từ chối
# =============================================================================

def fetch_page(params: dict, delay: float, lan_thu: int = 4) -> list[dict] | None:
    """Lấy một trang. Trả về danh sách tin, hoặc None nếu hỏng hẳn.

    Xử lý 429 (quá nhiều yêu cầu) và 503 (quá tải) bằng cách lùi theo cấp số
    nhân. Đây là phần quan trọng nhất về mặt cư xử: khi máy chủ nói "chậm lại",
    cách duy nhất đúng là chậm lại thật, không phải thử lại ngay.
    """
    url = f"{API_URL}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={
        "User-Agent": USER_AGENT,
        "Accept": "application/json",
    })
    for attempt in range(lan_thu):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            ads = data.get("ads")
            if ads is None:
                for key in ("data", "results", "items"):
                    if isinstance(data.get(key), list):
                        ads = data[key]
                        break
            return ads if isinstance(ads, list) else []
        except urllib.error.HTTPError as e:
            if e.code in (429, 503):
                cho = min(delay * (2 ** (attempt + 2)), MAX_BACKOFF_SECONDS)
                print(f"    máy chủ trả {e.code} — chờ {cho:.0f} giây rồi thử lại",
                      file=sys.stderr)
                time.sleep(cho)
                continue
            if e.code == 404:
                return []
            print(f"    lỗi HTTP {e.code}", file=sys.stderr)
            return None
        except Exception as exc:
            if attempt == lan_thu - 1:
                print(f"    lỗi sau {lan_thu} lần thử: {exc}", file=sys.stderr)
                return None
            time.sleep(min(delay * (2 ** attempt), MAX_BACKOFF_SECONDS))
    return None


def do_kich_thuoc_trang(base: dict, delay: float) -> int:
    """Hỏi thử một trang để biết máy chủ thực sự trả về bao nhiêu tin mỗi lần.

    Không đoán: mỗi API cắt `limit` theo cách riêng, và con số này quyết định
    cần bao nhiêu lệnh gọi.
    """
    ads = fetch_page({**base, "limit": 50, "o": 0}, delay)
    n = len(ads) if ads else 0
    return n if n > 0 else 20


# =============================================================================
# THU THẬP
# =============================================================================

def crawl_category(conn: sqlite3.Connection, cg: int, region: int, st: str,
                   delay: float, max_pages: int | None) -> dict:
    base = {"region_v2": region, "cg": cg, "st": st}
    ten = DANH_MUC.get(cg, str(cg))
    print(f"\n=== {ten} (cg={cg}, st={st}) ===")

    kich_thuoc = do_kich_thuoc_trang(base, delay)
    print(f"Mỗi trang máy chủ trả về {kich_thuoc} tin")

    thong_ke = {"moi": 0, "cu": 0, "trang": 0, "loi": 0}
    da_thay: set[int] = set()
    offset = 0
    loi_lien_tiep = 0
    now = time.time()

    while True:
        if max_pages and thong_ke["trang"] >= max_pages:
            print(f"  dừng: đã đạt giới hạn {max_pages} trang")
            break

        time.sleep(delay)
        ads = fetch_page({**base, "limit": kich_thuoc, "o": offset}, delay)
        thong_ke["trang"] += 1

        if ads is None:
            loi_lien_tiep += 1
            thong_ke["loi"] += 1
            if loi_lien_tiep >= MAX_LOI_LIEN_TIEP:
                print(f"  DỪNG: lỗi {MAX_LOI_LIEN_TIEP} lần liên tiếp. "
                      f"Dữ liệu đã lấy được vẫn nằm trong kho.", file=sys.stderr)
                break
            offset += kich_thuoc
            continue
        loi_lien_tiep = 0

        if not ads:
            print(f"  hết dữ liệu ở offset {offset}")
            break

        # Chặn phân trang sâu: nhiều API âm thầm trả lại đúng trang đầu khi
        # offset vượt ngưỡng. Nếu cả trang toàn tin đã thấy thì coi như hết.
        ids = [a.get("list_id") for a in ads if a.get("list_id")]
        moi_trong_trang = [i for i in ids if i not in da_thay]
        if not moi_trong_trang:
            print(f"  dừng ở offset {offset}: trang lặp lại tin đã lấy "
                  f"(nhiều khả năng máy chủ chặn phân trang sâu)")
            break
        da_thay.update(ids)

        for ad in ads:
            kq = upsert_ad(conn, ad, cg, now)
            if kq in thong_ke:
                thong_ke[kq] += 1
        conn.commit()

        if thong_ke["trang"] % 10 == 0 or offset == 0:
            print(f"  offset {offset:>5}  |  mới {thong_ke['moi']:>5}  "
                  f"đã có {thong_ke['cu']:>5}", flush=True)
        offset += kich_thuoc

    print(f"  xong: {thong_ke['moi']} tin mới, {thong_ke['cu']} tin đã có, "
          f"{thong_ke['trang']} trang, {thong_ke['loi']} lỗi")
    return thong_ke


# =============================================================================
# XUẤT RA BẢNG
# =============================================================================
# Các trường được lấy ra. JSON gốc vẫn nằm nguyên trong kho, nên thêm cột về
# sau chỉ cần chạy lại bước xuất, không phải cào lại.

TRUONG_LAY = [
    "list_id", "list_time", "date", "subject", "body",
    "price", "size", "price_million_per_m2",
    "latitude", "longitude",
    "ward", "ward_name", "ward_name_v3",
    "area", "area_name", "region_name", "region_name_v3",
    "street_name", "unique_street_id", "is_main_street",
    "rooms", "toilets", "floors", "direction",
    "apartment_type", "property_legal_document", "furnishing_sell",
    "category", "category_name", "type", "status",
    "account_id", "account_name", "company_ad", "number_of_images",
    "pty_characteristics", "pty_project_name",
]


# Excel không nhận một số ký tự điều khiển, và mỗi ô tối đa 32.767 ký tự.
# Mô tả tin của người bán thường có emoji, ký tự xuống dòng lạ và đôi khi cả ký
# tự điều khiển vô hình lọt vào khi họ dán từ nơi khác. openpyxl gặp là ném lỗi
# và hỏng cả file. Emoji thì Excel nhận bình thường — thủ phạm là nhóm điều khiển.
#
# Kho SQLite vẫn giữ NGUYÊN VĂN JSON, nên việc dọn ở đây chỉ ảnh hưởng bản xuất
# ra Excel cho dễ xem, không làm mất dữ liệu gốc.
_KY_TU_CAM_XLSX = re.compile(r"[\000-\010\013\014\016-\037]")
XLSX_MAX_O = 32767


def _lam_sach_o(v):
    if not isinstance(v, str):
        return v
    v = _KY_TU_CAM_XLSX.sub("", v)
    if len(v) > XLSX_MAX_O:
        v = v[:XLSX_MAX_O - 24] + " …[đã cắt bớt để vừa ô Excel]"
    return v


def _lam_sach_bang(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == object:
            out[c] = out[c].map(_lam_sach_o)
    return out


def export(conn: sqlite3.Connection, outdir: Path, cg: int | None) -> pd.DataFrame:
    q = "SELECT list_id, cg, json_tho, lan_dau_thay, lan_cuoi_thay, so_lan_thay FROM ads"
    args: tuple = ()
    if cg is not None:
        q += " WHERE cg = ?"
        args = (cg,)
    rows = conn.execute(q, args).fetchall()

    ban_ghi = []
    for list_id, cg_row, js, dau, cuoi, so_lan in rows:
        ad = json.loads(js)
        rec = {k: ad.get(k) for k in TRUONG_LAY}
        rec["cg"] = cg_row
        rec["lan_dau_thay"] = pd.to_datetime(dau, unit="s")
        rec["lan_cuoi_thay"] = pd.to_datetime(cuoi, unit="s")
        rec["so_lan_thay"] = so_lan
        # list_time là epoch mili-giây -> đổi sang ngày giờ đọc được
        lt = ad.get("list_time")
        rec["ngay_dang"] = pd.to_datetime(lt, unit="ms") if lt else pd.NaT
        rec["url"] = f"https://www.nhatot.com/{list_id}.htm"
        ban_ghi.append(rec)

    df = pd.DataFrame(ban_ghi)
    if df.empty:
        print("Kho rỗng — chưa có gì để xuất.")
        return df

    outdir.mkdir(parents=True, exist_ok=True)
    ten = f"chotot_api_{cg}" if cg else "chotot_api_tatca"

    # CSV trước: nó nhận mọi ký tự, nên đây là bản đầy đủ không cắt gọt.
    # utf-8-sig để Excel mở lên không bị vỡ tiếng Việt.
    path_csv = outdir / f"{ten}.csv"
    df.to_csv(path_csv, index=False, encoding="utf-8-sig")

    # XLSX sau: tiện xem bằng mắt, nhưng phải dọn ký tự cấm trước.
    path = outdir / f"{ten}.xlsx"
    try:
        _lam_sach_bang(df).to_excel(path, index=False)
    except Exception as exc:
        print(f"  (không ghi được .xlsx: {exc} — bản .csv vẫn đầy đủ)", file=sys.stderr)
        path = path_csv

    print(f"\nĐã xuất {len(df)} tin -> {path}")
    print(f"  bản đầy đủ (không cắt): {path_csv}")
    print(f"  có toạ độ        : {df['latitude'].notna().sum()} "
          f"({df['latitude'].notna().mean()*100:.1f}%)")
    print(f"  có ngày đăng     : {df['ngay_dang'].notna().sum()} "
          f"({df['ngay_dang'].notna().mean()*100:.1f}%)")
    if "ward_name_v3" in df:
        doi_ten = (df["ward_name"].notna() & df["ward_name_v3"].notna()
                   & (df["ward_name"] != df["ward_name_v3"])).sum()
        print(f"  phường đã đổi tên : {doi_ten} tin "
              f"({df.loc[df['ward_name'] != df['ward_name_v3'], 'ward_name'].nunique()} phường)")
    if df["ngay_dang"].notna().any():
        print(f"  khoảng thời gian  : {df['ngay_dang'].min():%d/%m/%Y} "
              f"-> {df['ngay_dang'].max():%d/%m/%Y}")
    return df


# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Thu thập tin BĐS qua API công khai Chợ Tốt / Nhà Tốt")
    ap.add_argument("--cg", type=int, action="append",
                    help="mã danh mục, lặp lại được. 1010 căn hộ, 1020 nhà ở, "
                         "1040 đất. Bỏ qua thì lấy cả ba.")
    ap.add_argument("--region", type=int, default=REGION_HANOI, help="12000 = Hà Nội")
    ap.add_argument("--st", default=DEFAULT_ST, help="s = bán, bỏ đi thì ra cho thuê")
    # Mặc định phải là thư mục data_raw CỦA DỰ ÁN, không phải "data_raw" tính
    # theo thư mục đang đứng. Chạy từ pipeline/ mà dùng đường dẫn tương đối thì
    # script lặng lẽ tạo pipeline/data_raw/ với một kho SQLite thứ hai — dữ liệu
    # không mất, nhưng nằm sai chỗ và không gộp với kho cũ. Đã xảy ra một lần.
    ap.add_argument("--outdir", default=thu_muc("data_raw"), type=Path)
    ap.add_argument("--store", default=None, type=Path,
                    help="kho SQLite dùng chung, mặc định <outdir>/chotot_store.sqlite")
    ap.add_argument("--delay", type=float, default=DEFAULT_DELAY,
                    help=f"giây giữa hai lệnh gọi (mặc định {DEFAULT_DELAY}). "
                         "Đừng hạ xuống dưới 1.")
    ap.add_argument("--max-pages", type=int, default=None,
                    help="giới hạn số trang mỗi danh mục — dùng để chạy thử")
    ap.add_argument("--chi-xuat", action="store_true",
                    help="không cào, chỉ xuất lại từ kho đã có")
    args = ap.parse_args()

    if args.delay < 1.0:
        print("Độ trễ dưới 1 giây là quá nhanh với một API công khai. "
              "Đặt lại về 1.0.", file=sys.stderr)
        args.delay = 1.0

    args.outdir.mkdir(parents=True, exist_ok=True)
    store_path = args.store or (args.outdir / "chotot_store.sqlite")
    conn = open_store(store_path)
    # Mặc định cào danh mục CHA: một lượt lấy hết, không lo sót danh mục con.
    # Việc tách căn hộ / nhà ở / đất làm sau, dựa vào trường `category` của
    # từng tin — chắc chắn hơn là tự liệt kê mã danh mục con.
    danh_muc = args.cg or [CG_TAT_CA]

    if not args.chi_xuat:
        print(f"Kho dữ liệu: {store_path}")
        print(f"Độ trễ: {args.delay}s | danh mục: {danh_muc}")
        for cg in danh_muc:
            crawl_category(conn, cg, args.region, args.st, args.delay, args.max_pages)

    # Xuất một file cho mỗi danh mục THẬT có trong kho, cộng một file gộp.
    co_trong_kho = [r[0] for r in
                    conn.execute("SELECT DISTINCT cg FROM ads ORDER BY cg")]
    print(f"\nDanh mục có trong kho: "
          f"{ {c: DANH_MUC.get(c, str(c)) for c in co_trong_kho} }")
    for cg in co_trong_kho:
        export(conn, args.outdir, cg)
    export(conn, args.outdir, None)


if __name__ == "__main__":
    main()
