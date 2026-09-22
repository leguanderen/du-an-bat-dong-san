"""Kiểm GÓI TRIỂN KHAI — chủ yếu kiểm phần từ chối, không phải phần chạy được.

Cả thiết kế này đánh đổi: bỏ "huấn luyện lại mỗi lần khởi động" (an toàn nhưng
mất 16,6 giây) lấy "nạp model đã ghi, có kiểm vân tay" (0,35 giây). Đánh đổi đó
CHỈ đúng nếu phần kiểm vân tay thật sự chặn được model lệch pha. Nên phần lớn
các phép kiểm dưới đây cố tình làm hỏng gói rồi đòi hệ thống phải từ chối nó.

Chạy: python _scratch/kiem_goi_trien_khai.py
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
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

import valuation_service as vs
from intervals import ConformalValuer

loi = 0


def kiem(ten, ok, ct=""):
    global loi
    print(("  ✓ " if ok else "  ✗ ") + ten + ("" if ok else f"   << {ct}"))
    if not ok:
        loi += 1


LOAI = "chungcu"
goc = vs.THU_MUC_GOI / LOAI
if not (goc / vs.TEN_PARQUET).exists():
    print("Chưa có gói. Chạy `python chuan_bi_trien_khai.py` trước.")
    raise SystemExit(2)

df = pd.read_parquet(goc / vs.TEN_PARQUET)
cfg = vs._cau_hinh(LOAI, df)
vt = vs.van_tay(goc / vs.TEN_PARQUET, cfg, vs.ALPHA_MAC_DINH)

print("A. Đường chạy thật")
v = ConformalValuer.nap_tu(goc, vt)
kiem("vân tay khớp thì nạp được", v is not None)
kiem("nạp xong dự đoán ra số hợp lệ",
     v is not None and np.isfinite(v.predict_interval(df.head(20))["gia"]).all())

t = time.time()
vs._nap.cache_clear()
vs.nap(LOAI)
giay = time.time() - t
kiem(f"khởi động lạnh dưới 2 giây ({giay:.2f}s)", giay < 2.0, f"{giay:.2f}s")

print("\nB. Phải TỪ CHỐI khi gói không còn khớp")
with tempfile.TemporaryDirectory() as tmp:
    def ban_sao():
        d = Path(tmp) / f"thu{time.time_ns()}"
        shutil.copytree(goc, d)
        return d

    d = ban_sao()
    kiem("vân tay khác -> từ chối",
         ConformalValuer.nap_tu(d, vt + "x") is None)

    d = ban_sao()
    (d / "manifest.json").unlink()
    kiem("mất manifest -> từ chối", ConformalValuer.nap_tu(d, vt) is None)

    d = ban_sao()
    (d / "q_lo.ubj").unlink()
    kiem("mất một file model -> từ chối", ConformalValuer.nap_tu(d, vt) is None)

    d = ban_sao()
    (d / "point.ubj").write_bytes(b"rac khong phai model")
    kiem("file model hỏng -> từ chối", ConformalValuer.nap_tu(d, vt) is None)

    d = ban_sao()
    mf = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    mf["phien_ban_goi"] = 999
    (d / "manifest.json").write_text(json.dumps(mf), encoding="utf-8")
    kiem("cấu trúc gói đổi -> từ chối", ConformalValuer.nap_tu(d, vt) is None)

    d = ban_sao()
    mh = json.loads((d / "ma_hoa.json").read_text(encoding="utf-8"))
    mh.pop(next(iter(mh)))
    (d / "ma_hoa.json").write_text(json.dumps(mh), encoding="utf-8")
    kiem("thiếu bảng mã hoá một cột -> từ chối",
         ConformalValuer.nap_tu(d, vt) is None)

print("\nC. Vân tay phải đổi khi dữ liệu hoặc cấu hình đổi")
kiem("đổi alpha -> vân tay khác",
     vs.van_tay(goc / vs.TEN_PARQUET, cfg, 0.5) != vt)


class _Gia:
    cot_so = tuple(list(cfg.cot_so)[:-1])
    cot_muc = cfg.cot_muc
    cot_dien_tich = cfg.cot_dien_tich


kiem("bớt một cột đặc trưng -> vân tay khác",
     vs.van_tay(goc / vs.TEN_PARQUET, _Gia, vs.ALPHA_MAC_DINH) != vt)

with tempfile.TemporaryDirectory() as tmp:
    pq2 = Path(tmp) / "khac.parquet"
    df.head(len(df) - 1).to_parquet(pq2, index=False)
    kiem("dữ liệu đổi một dòng -> vân tay khác",
         vs.van_tay(pq2, cfg, vs.ALPHA_MAC_DINH) != vt)

print("\nD. Đường nhanh và đường chậm phải cho CÙNG một con số")
# Đây là phép kiểm đắt nhất nhưng cũng đáng nhất: nếu hai đường lệch nhau thì
# bản trên mạng đang định giá khác bản chạy ở máy, mà không ai nhìn ra.
df_x, nen_x, cfg_x, _ = vs._nap_tu_xlsx(LOAI)
v_x = ConformalValuer(list(cfg_x.cot_so), list(cfg_x.cot_muc),
                      alpha=vs.ALPHA_MAC_DINH).fit(df_x, "TARGET_gia_vnd")
mau = df_x.head(300)
a = v_x.predict_interval(mau)
b = v.predict_interval(mau)
for cot in ("gia", "khoang_duoi", "khoang_tren"):
    lech = float(np.nanmax(np.abs(a[cot].to_numpy() - b[cot].to_numpy())
                           / np.maximum(np.abs(a[cot].to_numpy()), 1.0)))
    kiem(f"cột {cot} trùng khớp (lệch {lech:.1e})", lech < 1e-6, f"{lech:.2e}")
kiem("lượng nới conformal Q trùng khớp",
     abs(float(v.Q) - float(v_x.Q)) < 1e-12, f"{v.Q} vs {v_x.Q}")
kiem("nền láng giềng chọn giống nhau",
     len(vs.chon_nen_lang_gieng(df)) == len(nen_x),
     f"{len(vs.chon_nen_lang_gieng(df))} vs {len(nen_x)}")

print("\nsố lỗi:", loi)
sys.exit(1 if loi else 0)
