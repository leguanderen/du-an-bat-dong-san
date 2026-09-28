"""Chặn chuyện app lại ngốn CPU tới mức bị Streamlit Cloud bóp.

Streamlit Cloud đã bóp CPU app này một lần ("Your app has been throttled"),
nguyên nhân là `_khoang_ngoai_mau` huấn luyện 20 model XGBoost ngay lúc người
dùng gõ câu đầu tiên vào chatbot. Mỗi lần app tỉnh dậy sau khi ngủ là một lần
như vậy.

Các mốc dưới đây đặt RỘNG RÃI so với số đo thật (thường nhanh hơn 5-10 lần) —
mục đích là bắt việc một phép tính nặng vô tình quay lại chạy lúc chạy thật,
chứ không phải đo hiệu năng chính xác.
"""
from __future__ import annotations

# --- tìm module dù dự án bố trí kiểu nào -------------------------------------
import sys as _sys
from pathlib import Path as _Path
_GOC = _Path(__file__).resolve().parent.parent
for _d in (_GOC, _GOC / "pipeline", _GOC / "_scratch"):
    if _d.is_dir() and str(_d) not in _sys.path:
        _sys.path.insert(0, str(_d))
# -----------------------------------------------------------------------------

import time

import ban_do as bd
import tim_nha as tnh
import tinh_nang as tn
import valuation_service as vs

loi = 0


def kiem(ten, giay, moc):
    global loi
    ok = giay < moc
    print(("  ✓ " if ok else "  ✗ ") + f"{ten:<46} {giay:6.2f}s  (mốc {moc}s)")
    if not ok:
        loi += 1


print("A. Khởi động lạnh")
for loai in ("chungcu", "nhadat"):
    t = time.time(); vs.nap(loai); kiem(f"vs.nap({loai})", time.time() - t, 3.0)

print("\nB. Khoảng ngoài mẫu phải ĐỌC SẴN, không được huấn luyện lại")
for loai in ("chungcu", "nhadat"):
    tn._khoang_ngoai_mau.cache_clear()
    d = tn._khoang_ngoai_mau(loai)
    co = all(c in d.columns for c in tn.COT_NGOAI_MAU)
    print(("  ✓ " if co else "  ✗ ")
          + f"{loai}: 5 cột ngoài mẫu có sẵn trong gói")
    if not co:
        loi += 1
        print("      -> gói thiếu cột. Chạy `python chuan_bi_trien_khai.py`.")

print("\nC. Lượt chat đầu tiên — chỗ đã làm app bị bóp CPU")
for loai, cau in (("chungcu", "căn hộ 2 phòng ngủ ở Cầu Giấy dưới 3 tỷ"),
                  ("nhadat", "nhà 4 tầng dưới 6 tỷ Hoàng Mai")):
    tn._khoang_ngoai_mau.cache_clear()
    t = time.time(); tnh.tim(cau, loai, k=3)
    kiem(f"lượt chat đầu [{loai}]", time.time() - t, 5.0)

print("\nD. Thao tác lặp lại phải gần như tức thì")
for loai in ("chungcu", "nhadat"):
    t = time.time(); tnh.tim("rẻ hơn nữa", loai, k=3)
    kiem(f"lượt chat tiếp [{loai}]", time.time() - t, 0.5)
    t = time.time(); bd.nap(loai)
    kiem(f"bd.nap có đệm [{loai}]", time.time() - t, 0.2)

print("\nE. Bảng đệm không được bị sửa tại chỗ")
d1 = bd.nap("nhadat")
d1["_cot_ban"] = 1
d2 = bd.nap("nhadat")
ok = "_cot_ban" not in d2.columns
print(("  ✓ " if ok else "  ✗ ") + "sửa bản trả về KHÔNG ảnh hưởng bản trong đệm")
if not ok:
    loi += 1

print("\nsố lỗi:", loi)
_sys.exit(1 if loi else 0)
