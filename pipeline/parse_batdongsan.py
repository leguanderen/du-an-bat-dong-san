"""
parse_batdongsan.py — Bóc dữ liệu từ HTML của batdongsan.com.vn.

Tách riêng khỏi phần thu thập để KIỂM CHỨNG ĐƯỢC mà không cần mạng: đưa một
file HTML đã lưu vào, đối chiếu kết quả với những gì nhìn thấy trên màn hình.
Đó là cách duy nhất tử tế để biết parser đúng trước khi cho nó chạy 11.345 lượt.

    python parse_batdongsan.py mau/chitiet.html      # in ra kết quả bóc được

NGUỒN DỮ LIỆU TRONG MỘT TRANG CHI TIẾT
--------------------------------------
1. Khối JavaScript tĩnh nhúng thẳng trong HTML — quý nhất, vì toàn số sạch:

       productId, price, pricePerM2, categoryId, cityCode, districtId,
       streetId, wardId, projectId, productType, area, latitude, longitude

   Đây là JS TĨNH, không phải gọi thêm, nên tải HTML bằng Python là đọc được —
   không cần trình duyệt, không cần chạy JavaScript.

2. `re__address-line-1` / `line-2` — địa chỉ theo cấu trúc CŨ và tên phường
   theo cấu trúc MỚI sau sáp nhập. Giống cách nhatot làm, nên bảng tra tên
   phường cũ → mới lấy được miễn phí từ đây.

3. `re__pr-short-info-item` — khoảng giá, diện tích, ngày đăng, NGÀY HẾT HẠN,
   loại tin, mã tin. Ngày hết hạn là thứ nhatot không có: nó cho biết trực tiếp
   tin nằm trên sàn bao lâu, khỏi phải cào lặp nhiều ngày để suy ra.

4. `re__pr-specs-content-item` — số phòng ngủ, số vệ sinh, hướng nhà, hướng
   ban công, pháp lý, nội thất.

5. `detail-content` — mô tả tự do của người bán.
"""

from __future__ import annotations

import html as ht
import json
import re
import sys
import unicodedata
from pathlib import Path

# =============================================================================
# TIỆN ÍCH
# =============================================================================

def _text(x: str) -> str:
    """Bỏ thẻ HTML, giải mã thực thể, gộp khoảng trắng."""
    return re.sub(r"\s+", " ", ht.unescape(re.sub(r"<[^>]+>", " ", x))).strip()


def _key(label: str) -> str:
    """Chuẩn hoá nhãn thành khoá không dấu, để khớp bền với xuống dòng lung tung.

    Nhãn trong HTML hay bị ngắt dòng giữa chừng ("Ngày\\n   đăng"), nên so khớp
    theo chuỗi nguyên văn là hỏng. Chuẩn hoá về "ngay dang" thì ổn định.
    """
    s = _text(label).replace("đ", "d").replace("Đ", "D")
    s = "".join(c for c in unicodedata.normalize("NFD", s)
                if unicodedata.category(c) != "Mn").lower()
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def parse_vn_number(s: object) -> float | None:
    """"2,3 tỷ" -> 2.3e9 ; "43 m²" -> 43 ; "1 phòng" -> 1"""
    if not isinstance(s, str):
        return None
    t = _key(s)
    m = re.search(r"([\d.,]+)", t)
    if not m:
        return None
    num = m.group(1)
    num = num.replace(".", "") if ("," in num and "." in num) else num
    try:
        v = float(num.replace(",", "."))
    except ValueError:
        return None
    if "ty" in t:
        return v * 1e9
    if "trieu" in t:
        return v * 1e6
    return v


# =============================================================================
# CÁC BỘ BÓC
# =============================================================================

# Khối JS: các khoá số nằm rải trong một object literal.
_JS_KEYS = ["productId", "price", "pricePerM2", "categoryId", "districtId",
            "streetId", "wardId", "projectId", "productType", "area",
            "latitude", "longitude"]


def parse_js_block(h: str) -> dict:
    """Bóc các trường số trong khối JS tĩnh. Lấy lần xuất hiện CUỐI CÙNG.

    Trong trang có nhiều object literal (gợi ý tin liên quan, lịch sử giá…).
    Khối chứa latitude/longitude là khối đầy đủ nhất và nằm sau, nên lấy giá trị
    cuối cùng của mỗi khoá là an toàn nhất.
    """
    out: dict = {}
    for k in _JS_KEYS:
        ms = re.findall(rf"\b{k}\s*:\s*(-?[\d.]+)", h)
        if ms:
            try:
                v = float(ms[-1])
                out[k] = int(v) if v.is_integer() and k != "latitude" \
                    and k != "longitude" and k != "pricePerM2" else v
            except ValueError:
                pass
    m = re.findall(r"\bcityCode\s*:\s*'([^']*)'", h)
    if m:
        out["cityCode"] = m[-1]
    return out


def parse_address(h: str) -> dict:
    """Địa chỉ cũ + tên đơn vị hành chính mới sau sáp nhập."""
    out = {"dia_chi": None, "phuong_moi": None}
    m = re.search(r'class="re__address-line-1"[^>]*>(.*?)</span>', h, re.S)
    if m:
        out["dia_chi"] = _text(m.group(1))
    m = re.search(r'class="re__address-line-2\s*"[^>]*>(.*?)</span>', h, re.S)
    if m:
        # dạng "(Xã Gia Lâm, Hà Nội mới)" -> bỏ ngoặc và đuôi " mới"
        v = _text(m.group(1)).strip("() ")
        out["phuong_moi"] = re.sub(r",?\s*(Hà Nội|TP Hà Nội)\s*mới\s*$", "", v).strip()
    return out


def parse_short_info(h: str) -> dict:
    """Khoảng giá, diện tích, ngày đăng, ngày hết hạn, loại tin, mã tin."""
    out: dict = {}
    for m in re.finditer(
        r'class="re__pr-short-info-item[^"]*"[^>]*>\s*'
        r'<span class="title"[^>]*>(.*?)</span>\s*'
        r'<span class="value"[^>]*>(.*?)</span>', h, re.S
    ):
        out[_key(m.group(1))] = _text(m.group(2))
    return out


def parse_specs(h: str) -> dict:
    """Bảng đặc điểm: phòng ngủ, vệ sinh, hướng, pháp lý, nội thất."""
    out: dict = {}
    for m in re.finditer(
        r'class="re__pr-specs-content-item-title"[^>]*>(.*?)</span>\s*'
        r'<span class="re__pr-specs-content-item-value"[^>]*>(.*?)</span>', h, re.S
    ):
        out[_key(m.group(1))] = _text(m.group(2))
    return out


def parse_description(h: str) -> str | None:
    m = re.search(r'class="[^"]*detail-content[^"]*"[^>]*>(.*?)</div>', h, re.S)
    return _text(m.group(1)) if m else None


def parse_title(h: str) -> str | None:
    m = re.search(r"<h1[^>]*>(.*?)</h1>", h, re.S)
    return _text(m.group(1)) if m else None


# =============================================================================
# HÀM CHÍNH
# =============================================================================

def parse_detail(h: str, url: str | None = None) -> dict:
    """HTML trang chi tiết -> một bản ghi phẳng."""
    js = parse_js_block(h)
    addr = parse_address(h)
    info = parse_short_info(h)
    specs = parse_specs(h)

    rec: dict = {
        "ma_tin": info.get("ma tin") or (str(js["productId"]) if "productId" in js else None),
        "url": url,
        "tieu_de": parse_title(h),
        # --- từ khối JS: số sạch, đáng tin nhất ---
        "gia_vnd": js.get("price"),
        "gia_tren_m2": js.get("pricePerM2"),
        "dien_tich_m2": js.get("area"),
        "latitude": js.get("latitude"),
        "longitude": js.get("longitude"),
        "ward_id": js.get("wardId"),
        "district_id": js.get("districtId"),
        "street_id": js.get("streetId"),
        "project_id": js.get("projectId"),
        "city_code": js.get("cityCode"),
        "product_type": js.get("productType"),
        "category_id": js.get("categoryId"),
        # --- địa chỉ ---
        "dia_chi": addr["dia_chi"],
        "phuong_moi": addr["phuong_moi"],
        # --- thời gian ---
        "ngay_dang": info.get("ngay dang"),
        "ngay_het_han": info.get("ngay het han"),
        "loai_tin": info.get("loai tin"),
        # --- đặc điểm ---
        "so_phong_ngu": parse_vn_number(specs.get("so phong ngu")),
        "so_phong_vs": parse_vn_number(specs.get("so phong tam ve sinh")),
        "huong_nha": specs.get("huong nha"),
        "huong_ban_cong": specs.get("huong ban cong"),
        "phap_ly": specs.get("phap ly"),
        "noi_that": specs.get("noi that"),
        "mo_ta": parse_description(h),
    }
    # Dự phòng: nếu khối JS thiếu, lấy từ phần hiển thị
    if rec["gia_vnd"] is None:
        rec["gia_vnd"] = parse_vn_number(info.get("khoang gia"))
    if rec["dien_tich_m2"] is None:
        rec["dien_tich_m2"] = parse_vn_number(info.get("dien tich"))
    return rec


_RE_LINK = re.compile(r'href="(/[^"?#]*?-pr(\d+))"')


def parse_list(h: str, base: str = "https://batdongsan.com.vn") -> list[dict]:
    """HTML trang danh sách -> danh sách {ma_tin, url}, đã khử trùng."""
    seen: dict[str, str] = {}
    for path, pid in _RE_LINK.findall(h):
        seen.setdefault(pid, base + path)
    return [{"ma_tin": k, "url": v} for k, v in seen.items()]


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Dùng: python parse_batdongsan.py <file.html>")
        raise SystemExit(1)
    p = Path(sys.argv[1])
    h = p.read_text(encoding="utf-8", errors="replace")
    if "re__pr-short-info-item" in h:
        print(json.dumps(parse_detail(h), ensure_ascii=False, indent=2))
    else:
        items = parse_list(h)
        print(f"{len(items)} tin trong trang danh sách")
        for it in items[:10]:
            print(" ", it["ma_tin"], it["url"][:90])
