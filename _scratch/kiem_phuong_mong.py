"""Cảnh báo phường mỏng dữ liệu: đúng mức, đúng con số, đúng chỗ."""
from __future__ import annotations

# --- tìm module dù dự án bố trí kiểu nào -------------------------------------
import sys as _sys
from pathlib import Path as _Path
_GOC = _Path(__file__).resolve().parent.parent
for _d in (_GOC, _GOC / "pipeline", _GOC / "app_moi", _GOC / "_scratch"):
    if _d.is_dir() and str(_d) not in _sys.path:
        _sys.path.insert(0, str(_d))
# -----------------------------------------------------------------------------

exec(open(_GOC / "_scratch" / "thu_app.py", encoding="utf-8")
     .read().split("loi = 0")[0])
import streamlit as _st, app as _app, valuation_service as vs

ve = []
for t in ("markdown", "caption", "info", "warning"):
    setattr(_st, t, (lambda t_: (lambda *a, **k: ve.append(str(a[0]) if a else "")))(t))

loi = 0
def kiem(n, ok, ct=""):
    global loi
    print(("  ✓ " if ok else "  ✗ ") + n + ("" if ok else f"   << {ct}"))
    if not ok: loi += 1

for loai in ("chungcu", "nhadat"):
    df = vs.nap(loai)["df"]
    g = df.dropna(subset=["phuong_moi"]).groupby("phuong_moi").size().sort_values()
    rat_mong = g[g < _app.NGUONG_PHUONG_RAT_MONG]
    mong = g[(g >= _app.NGUONG_PHUONG_RAT_MONG) & (g < _app.NGUONG_PHUONG_MONG)]
    day = g[g >= _app.NGUONG_PHUONG_MONG]
    print(f"\n=== {loai}: {len(rat_mong)} rất mỏng · {len(mong)} mỏng · "
          f"{len(day)} đủ dày")

    def thu(phuong):
        ve.clear()
        _app._canh_bao_phuong_mong(loai, {"phuong_moi": phuong})
        return "".join(ve)

    if len(rat_mong):
        p, n = rat_mong.index[0], int(rat_mong.iloc[0])
        s = thu(p)
        kiem(f"phường {n} căn -> cảnh báo mạnh", "rất thô" in s, s[:120])
        kiem("  nói đúng số căn thật", f"<b>{n} căn</b>" in s, s[:160])
        kiem("  nói đúng tên phường", f"<b>{p}</b>" in s)
        kiem("  dùng màu đỏ, không phải vàng",
             _app.COLOR["danger"] in s and _app.COLOR["warn"] not in s)
    if len(mong):
        p, n = mong.index[0], int(mong.iloc[0])
        s = thu(p)
        kiem(f"phường {n} căn -> cảnh báo nhẹ", "còn mỏng" in s, s[:120])
        kiem("  nói đúng số căn thật", f"<b>{n} căn</b>" in s, s[:160])
        kiem("  dùng màu vàng, không phải đỏ",
             _app.COLOR["warn"] in s and _app.COLOR["danger"] not in s)
    if len(day):
        p, n = day.index[-1], int(day.iloc[-1])
        kiem(f"phường {n} căn -> KHÔNG cảnh báo", thu(p) == "")

    kiem("chưa chọn phường -> không cảnh báo", thu(None) == "")

print("\nsố lỗi:", loi)
sys.exit(1 if loi else 0)
