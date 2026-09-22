"""
crawl_batdongsan.py — Thu thập tin bất động sản Hà Nội từ batdongsan.com.vn.

    python crawl_batdongsan.py --nhom chungcu --outdir data_raw --max-pages 2   # thử
    python crawl_batdongsan.py --nhom chungcu --outdir data_raw                 # đầy đủ
    python crawl_batdongsan.py --nhom nhadat  --outdir data_raw                 # cả 3 loại nhà

HAI GIAI ĐOẠN
-------------
1. QUÉT DANH SÁCH — tải các trang /p1, /p2, ... để lấy đường dẫn tin.
   Rẻ: mỗi trang cho khoảng 20 tin, nên 11.345 tin chỉ cần ~568 lượt tải.

2. TẢI CHI TIẾT — mỗi tin một lượt, vì toạ độ và đặc điểm chỉ có ở trang chi tiết.
   Đây là phần tốn thời gian: khoảng 11.345 lượt.

Giai đoạn 2 BỎ QUA những tin đã có trong kho, nên chạy lại chỉ tải tin mới.

THỜI GIAN DỰ KIẾN (độ trễ 1,2 giây)
    chung cư : 568 + 11.345 lượt  ≈ 4 giờ
    nhà đất  : ~830 + 16.500 lượt ≈ 6 giờ
Đó là thời gian MÁY chạy. Bật buổi tối, sáng có dữ liệu. Dừng giữa chừng được.

VÌ SAO CHẤP NHẬN CHẬM HƠN API CỦA NHATOT
    nhatot API   : 1.320 tin chung cư, 4 phút
    batdongsan   : 11.345 tin chung cư, 4 giờ
Gấp 8,6 lần dữ liệu, cùng có toạ độ và ngày đăng, lại thêm ngày hết hạn và
hướng nhà. Bốn giờ máy chạy đổi lấy chừng đó là đáng.

HAI CHẾ ĐỘ TẢI TRANG
--------------------
    --che-do http        urllib thuần. Nhanh, nhẹ. NHƯNG batdongsan trả 403 cho
                         client HTTP tự khai danh tính — site chỉ phục vụ trình duyệt.
    --che-do browser     Playwright điều khiển Chromium THẬT. Chậm hơn nhưng chạy được.

VỀ RANH GIỚI ĐẠO ĐỨC — đọc trước khi dùng
    Chế độ `browser` KHÔNG phải là vượt rào. Nó mở một Chromium thật, tự khai
    đúng là Chromium, và tải trang y như người dùng bấm chuột. Đây cùng hạng với
    extension Web Scraper mà dự án này vốn đã dùng.

    Thứ script này CỐ TÌNH KHÔNG làm: đặt User-Agent Chrome cho một client không
    phải Chrome, giả vân tay TLS, hay tự động vượt CAPTCHA. Đó là khai man danh
    tính để qua một chốt chặn dựng lên đúng để chặn việc mình đang làm — khác
    hẳn với việc dùng trình duyệt thật.

    Dù vậy vẫn nên giữ chừng mực: để độ trễ, lấy mẫu vừa đủ thay vì vét sạch,
    và ghi rõ phương thức thu thập trong báo cáo.

VỀ VIỆC GỌI CÓ CHỪNG MỰC
    robots.txt của batdongsan.com.vn chỉ chặn 3 đường dẫn nội bộ, không chặn
    trang danh sách hay trang tin, và không có Crawl-delay. Script vẫn để mặc
    định 1,2 giây một lượt và tự lùi khi gặp 429/503. Đừng hạ xuống: bạn không
    tiết kiệm được bao nhiêu mà lại tạo tải bất thường lên máy chủ người khác.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sqlite3
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

from parse_batdongsan import parse_detail, parse_list

BASE = "https://batdongsan.com.vn"

# Đường dẫn danh mục — đã kiểm chứng bằng cách tải thử ngày 26/08/2026.
# Muốn thêm danh mục khác: mở trang đó trên trình duyệt, chép phần đuôi URL.
NHOM = {
    "chungcu": [("ban-can-ho-chung-cu-ha-noi", "Căn hộ / Chung cư")],
    "nhadat": [
        ("ban-nha-rieng-ha-noi", "Nhà riêng"),
        ("ban-nha-mat-pho-ha-noi", "Nhà mặt phố"),
        ("ban-nha-biet-thu-lien-ke-ha-noi", "Biệt thự / Liền kề"),
    ],
    "dat": [("ban-dat-ha-noi", "Đất")],
}

DEFAULT_DELAY = 1.2
MAX_BACKOFF = 120
MAX_LOI_LIEN_TIEP = 8

# Tự khai danh tính là chuẩn mực khi thu thập tự động: chủ trang biết ai đang
# gọi và liên hệ được nếu có vấn đề.
USER_AGENT = ("Mozilla/5.0 (compatible; khoa-luan-dinh-gia-bds-hanoi/1.0; "
              "nghien cuu hoc thuat; +mailto:buiduyluongz@gmail.com)")


# =============================================================================
# KHO LƯU TRỮ
# =============================================================================
# Lưu bản ghi đã bóc (JSON nhỏ, ~2 KB/tin) thay vì HTML thô (~230 KB/tin).
# Giữ HTML thô cho cả 11.345 tin sẽ tốn 2,6 GB — không thực tế.
#
# Đánh đổi: nếu sau này luật bóc thay đổi thì phải tải lại. Chấp nhận được vì
# một lần tải lại chỉ mất vài giờ máy chạy. Bù lại, cờ --luu-html giữ HTML thô
# cho N tin đầu để làm mẫu kiểm thử khi sửa parser.

def open_store(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS tin (
            ma_tin      TEXT PRIMARY KEY,
            nhom        TEXT,
            loai        TEXT,
            url         TEXT,
            ban_ghi     TEXT,
            html_gz     BLOB,
            thoi_diem   REAL
        );
        CREATE TABLE IF NOT EXISTS hang_doi (
            ma_tin TEXT PRIMARY KEY,
            nhom   TEXT,
            loai   TEXT,
            url    TEXT
        );
    """)
    conn.commit()
    return conn


def da_co(conn: sqlite3.Connection, ma_tin: str) -> bool:
    return conn.execute("SELECT 1 FROM tin WHERE ma_tin = ?", (ma_tin,)).fetchone() is not None


# =============================================================================
# TẢI TRANG
# =============================================================================

# --- Chế độ trình duyệt: Chromium thật qua Playwright -------------------------
_PW = {"pw": None, "browser": None, "page": None}


def mo_browser(headless: bool = True):
    """Khởi động Chromium thật. Cài trước: pip install playwright &&
    playwright install chromium"""
    if _PW["page"] is not None:
        return _PW["page"]
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("Chưa có Playwright. Cài bằng:\n"
              "    pip install playwright\n"
              "    playwright install chromium", file=sys.stderr)
        raise SystemExit(1)
    _PW["pw"] = sync_playwright().start()
    _PW["browser"] = _PW["pw"].chromium.launch(headless=headless)
    ctx = _PW["browser"].new_context(locale="vi-VN")
    # Chặn ảnh, font, media: không cần cho việc bóc dữ liệu, và giảm đáng kể
    # cả thời gian tải lẫn băng thông tiêu tốn của máy chủ.
    ctx.route("**/*", lambda route: route.abort()
              if route.request.resource_type in ("image", "media", "font")
              else route.continue_())
    _PW["page"] = ctx.new_page()
    return _PW["page"]


def dong_browser():
    for k in ("browser", "pw"):
        try:
            obj = _PW[k]
            if obj:
                obj.close() if k == "browser" else obj.stop()
        except Exception:
            pass
    _PW.update({"pw": None, "browser": None, "page": None})


def fetch_browser(url: str, delay: float, lan_thu: int = 3) -> str | None:
    page = mo_browser()
    for attempt in range(lan_thu):
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=45000)
            return page.content()
        except Exception as exc:
            if attempt == lan_thu - 1:
                print(f"    lỗi trình duyệt: {exc} — {url[:70]}", file=sys.stderr)
                return None
            time.sleep(min(delay * (2 ** attempt), MAX_BACKOFF))
    return None


CHE_DO = {"hien_tai": "http"}


def fetch(url: str, delay: float, lan_thu: int = 4) -> str | None:
    if CHE_DO["hien_tai"] == "browser":
        return fetch_browser(url, delay, min(lan_thu, 3))
    return fetch_http(url, delay, lan_thu)


def fetch_http(url: str, delay: float, lan_thu: int = 4) -> str | None:
    req = urllib.request.Request(url, headers={
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml",
        "Accept-Language": "vi-VN,vi;q=0.9",
    })
    for attempt in range(lan_thu):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            if e.code in (429, 503):
                cho = min(delay * (2 ** (attempt + 2)), MAX_BACKOFF)
                print(f"    máy chủ trả {e.code} — chờ {cho:.0f}s", file=sys.stderr)
                time.sleep(cho)
                continue
            if e.code in (404, 410):
                return None
            print(f"    lỗi HTTP {e.code}: {url[:70]}", file=sys.stderr)
            return None
        except Exception as exc:
            if attempt == lan_thu - 1:
                print(f"    lỗi: {exc} — {url[:70]}", file=sys.stderr)
                return None
            time.sleep(min(delay * (2 ** attempt), MAX_BACKOFF))
    return None


# =============================================================================
# GIAI ĐOẠN 1 — QUÉT DANH SÁCH
# =============================================================================

def quet_danh_sach(conn, duong_dan: str, loai: str, nhom: str,
                   delay: float, max_pages: int | None) -> int:
    print(f"\n--- Quét danh sách: {loai} ({duong_dan}) ---")
    them = 0
    trang = 0
    da_thay: set[str] = set()
    loi_lien_tiep = 0

    while True:
        trang += 1
        if max_pages and trang > max_pages:
            print(f"  dừng: đạt giới hạn {max_pages} trang")
            break

        url = f"{BASE}/{duong_dan}" + ("" if trang == 1 else f"/p{trang}")
        time.sleep(delay)
        h = fetch(url, delay)
        if h is None:
            loi_lien_tiep += 1
            if loi_lien_tiep >= 3:
                print(f"  dừng ở trang {trang}: lỗi liên tiếp")
                break
            continue
        loi_lien_tiep = 0

        items = parse_list(h, BASE)
        moi = [i for i in items if i["ma_tin"] not in da_thay]
        if not moi:
            print(f"  hết dữ liệu ở trang {trang}")
            break
        da_thay.update(i["ma_tin"] for i in moi)

        for it in moi:
            conn.execute(
                "INSERT OR IGNORE INTO hang_doi (ma_tin, nhom, loai, url) VALUES (?,?,?,?)",
                (it["ma_tin"], nhom, loai, it["url"]))
            them += 1
        conn.commit()

        if trang % 20 == 0 or trang == 1:
            print(f"  trang {trang:>4}  |  đã gom {len(da_thay):>6} mã tin", flush=True)

    print(f"  xong: {len(da_thay)} mã tin từ {trang} trang")
    return them


# =============================================================================
# GIAI ĐOẠN 2 — TẢI CHI TIẾT
# =============================================================================

def tai_chi_tiet(conn, delay: float, gioi_han: int | None,
                 luu_html: int) -> dict:
    rows = conn.execute("""
        SELECT h.ma_tin, h.nhom, h.loai, h.url FROM hang_doi h
        LEFT JOIN tin t ON t.ma_tin = h.ma_tin
        WHERE t.ma_tin IS NULL
    """).fetchall()
    if gioi_han:
        rows = rows[:gioi_han]

    tong = len(rows)
    print(f"\n--- Tải chi tiết: {tong} tin chưa có ---")
    if not tong:
        return {"moi": 0, "loi": 0}
    print(f"Ước tính {tong * delay / 60:.0f} phút "
          f"({tong * delay / 3600:.1f} giờ). Ctrl+C dừng được, chạy lại tiếp tục.\n")

    tk = {"moi": 0, "loi": 0}
    t0 = time.time()
    loi_lien_tiep = 0
    try:
        for i, (ma_tin, nhom, loai, url) in enumerate(rows, start=1):
            time.sleep(delay)
            h = fetch(url, delay)
            if h is None:
                tk["loi"] += 1
                loi_lien_tiep += 1
                if loi_lien_tiep >= MAX_LOI_LIEN_TIEP:
                    print(f"  DỪNG: lỗi {MAX_LOI_LIEN_TIEP} lần liên tiếp. "
                          f"Dữ liệu đã tải vẫn nằm trong kho.", file=sys.stderr)
                    break
                continue
            loi_lien_tiep = 0

            rec = parse_detail(h, url)
            rec["nhom"], rec["loai"] = nhom, loai
            blob = (gzip.compress(h.encode("utf-8"))
                    if tk["moi"] < luu_html else None)
            conn.execute(
                "INSERT OR REPLACE INTO tin (ma_tin, nhom, loai, url, ban_ghi,"
                " html_gz, thoi_diem) VALUES (?,?,?,?,?,?,?)",
                (ma_tin, nhom, loai, url,
                 json.dumps(rec, ensure_ascii=False), blob, time.time()))
            tk["moi"] += 1
            if tk["moi"] % 25 == 0:
                conn.commit()

            if i % 100 == 0 or i == tong:
                troi = time.time() - t0
                con = (tong - i) * (troi / i) / 60
                print(f"  {i}/{tong}  ok {tk['moi']}  lỗi {tk['loi']}  "
                      f"còn ~{con:.0f} phút", flush=True)
    except KeyboardInterrupt:
        print("\nĐã dừng. Chạy lại lệnh cũ để tiếp tục từ chỗ này.")
    conn.commit()
    return tk


# =============================================================================
# XUẤT
# =============================================================================

import re as _re
_KY_TU_CAM = _re.compile(r"[\000-\010\013\014\016-\037]")


def _sach(v):
    if not isinstance(v, str):
        return v
    v = _KY_TU_CAM.sub("", v)
    return v[:32743] + " …[cắt]" if len(v) > 32767 else v


def export(conn, outdir: Path, nhom: str | None) -> pd.DataFrame:
    q = "SELECT ban_ghi FROM tin"
    args: tuple = ()
    if nhom:
        q += " WHERE nhom = ?"
        args = (nhom,)
    rows = conn.execute(q, args).fetchall()
    if not rows:
        print("Kho rỗng.")
        return pd.DataFrame()

    df = pd.DataFrame([json.loads(r[0]) for r in rows])
    for c in ("ngay_dang", "ngay_het_han"):
        if c in df:
            df[c + "_dt"] = pd.to_datetime(df[c], format="%d/%m/%Y", errors="coerce")
    if {"ngay_dang_dt", "ngay_het_han_dt"} <= set(df.columns):
        df["so_ngay_hien_thi"] = (df["ngay_het_han_dt"] - df["ngay_dang_dt"]).dt.days

    outdir.mkdir(parents=True, exist_ok=True)
    ten = f"batdongsan_{nhom}" if nhom else "batdongsan_tatca"
    df.to_csv(outdir / f"{ten}.csv", index=False, encoding="utf-8-sig")
    try:
        df.map(_sach).to_excel(outdir / f"{ten}.xlsx", index=False)
    except Exception as exc:
        print(f"  (không ghi được .xlsx: {exc} — bản .csv vẫn đầy đủ)", file=sys.stderr)

    print(f"\nĐã xuất {len(df)} tin -> {outdir / (ten + '.csv')}")
    for col, nhan in [("latitude", "có toạ độ"), ("ngay_dang", "có ngày đăng"),
                      ("ngay_het_han", "có ngày hết hạn"), ("gia_vnd", "có giá"),
                      ("mo_ta", "có mô tả")]:
        if col in df:
            print(f"  {nhan:<18}: {df[col].notna().sum()} ({df[col].notna().mean()*100:.1f}%)")
    if "phuong_moi" in df:
        print(f"  có tên phường mới : {df['phuong_moi'].notna().sum()}")
    if "ngay_dang_dt" in df and df["ngay_dang_dt"].notna().any():
        print(f"  khoảng thời gian  : {df['ngay_dang_dt'].min():%d/%m/%Y}"
              f" -> {df['ngay_dang_dt'].max():%d/%m/%Y}")
    return df


# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser(description="Thu thập BĐS Hà Nội từ batdongsan.com.vn")
    ap.add_argument("--nhom", choices=list(NHOM), default="chungcu")
    ap.add_argument("--outdir", type=Path, default=Path("data_raw"))
    ap.add_argument("--store", type=Path, default=None)
    ap.add_argument("--delay", type=float, default=DEFAULT_DELAY)
    ap.add_argument("--max-pages", type=int, default=None,
                    help="giới hạn số trang danh sách — dùng để chạy thử")
    ap.add_argument("--gioi-han-chi-tiet", type=int, default=None,
                    help="chỉ tải N trang chi tiết — dùng để chạy thử")
    ap.add_argument("--luu-html", type=int, default=50,
                    help="giữ HTML thô cho N tin đầu, làm mẫu kiểm thử parser")
    ap.add_argument("--che-do", choices=["http", "browser"], default="http",
                    help="http = urllib (nhanh, nhưng batdongsan trả 403); "
                         "browser = Chromium thật qua Playwright")
    ap.add_argument("--hien-trinh-duyet", action="store_true",
                    help="chế độ browser: hiện cửa sổ Chromium thay vì chạy ẩn")
    ap.add_argument("--chi-xuat", action="store_true", help="không tải, chỉ xuất lại")
    args = ap.parse_args()

    if args.delay < 0.8:
        print("Độ trễ dưới 0,8 giây là quá nhanh. Đặt lại 0,8.", file=sys.stderr)
        args.delay = 0.8

    args.outdir.mkdir(parents=True, exist_ok=True)
    store = args.store or (args.outdir / "batdongsan_store.sqlite")
    conn = open_store(store)

    CHE_DO["hien_tai"] = args.che_do
    if args.che_do == "browser" and not args.chi_xuat:
        mo_browser(headless=not args.hien_trinh_duyet)

    if not args.chi_xuat:
        print(f"Kho: {store} | độ trễ {args.delay}s | nhóm: {args.nhom} "
              f"| chế độ: {args.che_do}")
        for duong_dan, loai in NHOM[args.nhom]:
            quet_danh_sach(conn, duong_dan, loai, args.nhom, args.delay, args.max_pages)
        tai_chi_tiet(conn, args.delay, args.gioi_han_chi_tiet, args.luu_html)

    dong_browser()

    for nhom in [r[0] for r in conn.execute("SELECT DISTINCT nhom FROM tin")]:
        export(conn, args.outdir, nhom)


if __name__ == "__main__":
    main()
