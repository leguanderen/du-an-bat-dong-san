"""Đo thang màu bản đồ NGAY TRONG app.py — không đo trên bản sao.

Chạy: python _scratch/kiem_thang_mau.py  (trả mã thoát khác 0 nếu có hạng mục hỏng)

Bốn hạng mục, tất cả đo trên màu MẮT THẤY (đã tô mờ DUC_VUNG lên nền bản đồ):
  1. thang đơn điệu nhạt -> đậm
  2. ΔL giữa hai bậc kề >= 0,060 (bản đồ phường) / 0,036 (bề mặt đồng mức)
  3. bậc nhạt nhất nổi trên nền bản đồ >= 2,00:1
  4. nhãn chữ trên mọi bậc >= 4,50:1
Cộng thêm một hạng mục riêng: "rẻ nhất" phải tách khỏi "chưa đủ dữ liệu"
>= 2,00:1 — đọc nhầm hai ô này là biến chỗ thiếu số liệu thành một mức giá.
"""
# --- tìm module dù dự án bố trí kiểu nào -------------------------------------
# Trên máy phát triển, các module nằm trong `pipeline/` còn `app.py` ở thư mục
# gốc; ở bản chép phẳng thì tất cả nằm cùng một chỗ. Dò cả hai để bộ kiểm chạy
# được ở cả hai nơi — một bộ kiểm không chạy được thì không phải bằng chứng.
import sys as _sys
from pathlib import Path as _Path
_GOC = _Path(__file__).resolve().parent.parent
for _d in (_GOC, _GOC / "pipeline", _GOC / "app_moi", _GOC / "_scratch"):
    if _d.is_dir() and str(_d) not in _sys.path:
        _sys.path.insert(0, str(_d))
# -----------------------------------------------------------------------------
import runpy, sys
sys.argv = ["thu_app.py"]
# nạp streamlit giả của bộ thử rồi lấy chính module app đã import
ns = runpy.run_path(str(_GOC / "_scratch" / "thu_app.py"), run_name="__nap__")
app = sys.modules["app"]

def s(m):
    v = [x / 255 for x in m[:3]]
    v = [x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4 for x in v]
    return 0.2126 * v[0] + 0.7152 * v[1] + 0.0722 * v[2]

def tp(a, b):
    la, lb = s(a), s(b)
    return (max(la, lb) + 0.05) / (min(la, lb) + 0.05)

def oklab_L(m):
    r, g, b = [x / 255 for x in m[:3]]
    f = lambda c: c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = f(r), f(g), f(b)
    l = (0.4122214708*r + 0.5363325363*g + 0.0514459929*b) ** (1/3)
    m_ = (0.2119034982*r + 0.6806995451*g + 0.1073969566*b) ** (1/3)
    sx = (0.0883024619*r + 0.2817188376*g + 0.6299787005*b) ** (1/3)
    return 0.2104542553*l + 0.7936177850*m_ - 0.0040720468*sx

NEN = app.NEN_BAN_DO
A = app.DUC_VUNG
chong = lambda m: [round(A*c + (1-A)*n) for c, n in zip(m[:3], NEN)]

loi = 0
for so_bac, nhan_muc, gap_min in ((7, "bản đồ phường", 0.060),
                                  (11, "bề mặt đồng mức", 0.036)):
    print("=== %d bậc (%s)" % (so_bac, nhan_muc))
    hien = [chong(app._mau_bac(i, so_bac)) for i in range(so_bac)]
    Ls = [oklab_L(h) for h in hien]
    gaps = [Ls[i] - Ls[i+1] for i in range(so_bac - 1)]
    noi = tp(hien[0], NEN)
    nhan = [app.chu_tren(app._mau_bac(i, so_bac)) for i in range(so_bac)]
    tp_nhan = [tp(h, app._rgb(n)) for h, n in zip(hien, nhan)]
    print("  đơn điệu nhạt->đậm : %s" % ("ĐÚNG" if all(g > 0 for g in gaps) else "SAI"))
    print("  ΔL nhỏ nhất        : %.3f  (mốc %.3f) %s"
          % (min(gaps), gap_min, "OK" if min(gaps) >= gap_min else "HỎNG"))
    print("  bậc 0 nổi trên nền : %.2f:1 (mốc 2,00) %s"
          % (noi, "OK" if noi >= 2.0 else "HỎNG"))
    print("  nhãn yếu nhất      : %.2f:1 (mốc 4,50) %s"
          % (min(tp_nhan), "OK" if min(tp_nhan) >= 4.5 else "HỎNG"))
    if not all(g > 0 for g in gaps) or min(gaps) < gap_min \
       or noi < 2.0 or min(tp_nhan) < 4.5:
        loi += 1

b0 = chong(app._mau_bac(0, 7))
t = tp(b0, app.MAU_THIEU)
print('=== "rẻ nhất" vs "chưa đủ dữ liệu": %.2f:1 (mốc 2,00) %s'
      % (t, "OK" if t >= 2.0 else "HỎNG"))
if t < 2.0:
    loi += 1

print("\nTHANG_PHUONG =", app.THANG_PHUONG)
print("nhãn 5 bậc   =", [app.chu_tren(m) for m in app.thang_vung(5)])
print("\nsố hạng mục hỏng:", loi)
sys.exit(1 if loi else 0)
