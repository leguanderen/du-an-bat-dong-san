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
# -*- coding: utf-8 -*-
"""Chạy giao diện chat qua nhiều lượt, đếm bong bóng và kiểm kết quả."""
import sys, traceback
exec(open(_GOC / "_scratch" / "thu_app.py", encoding="utf-8").read().split("loi = 0")[0])
import streamlit as _st, app as _app, tim_nha as tnh

ra = []
for t in ("markdown","caption","dataframe","info","warning"):
    def mk(t=t):
        def f(*a, **k): ra.append((t, a[0] if a else None))
        return f
    setattr(_st, t, mk())

HOI = ["tôi có 5 tỷ, 2 phòng ngủ ở Cầu Giấy, có thang máy",
       "rẻ hơn nữa", "đổi sang Thanh Xuân", "thêm chỗ đỗ xe ô tô", "quay lại"]

loi = 0
for n, cau in enumerate(HOI, 1):
    globals()["BOM_CAU"] = cau
    goi.clear(); ra.clear()
    for _ in range(4):
        try:
            _app.render_tim_nha(); break
        except _RERUN:
            globals()["BOM_CAU"] = None    # lượt sau không gửi lại
            goi.clear(); ra.clear()
        except Exception as e:
            loi += 1; print(f"  ✗ lượt {n} “{cau}”: {type(e).__name__}: {e}")
            traceback.print_exc(limit=4); break
    lich = _st.session_state.get("tn_lich", [])
    bb = goi.count("chat_message")
    mong = 1 + 2 * len(lich)               # 1 lời mở + 2 bong bóng mỗi lượt
    print(f"\n  lượt {n}: “{cau}”")
    print(f"     lịch sử {len(lich)} lượt · {bb} bong bóng "
          f"({'đúng' if bb == mong else f'LỆCH, cần {mong}'})")
    if bb != mong: loi += 1
    if lich:
        t = lich[-1]
        print(f"     khớp {t['so_khop']} căn · thẻ: {[c for _,c in t['the']]}")
        for x in t["doi"]: print(f"     → {x}")
    # HỢP ĐỒNG MỚI (khung chat gọn):
    #   · trong khung chat KHÔNG được có danh sách đầy đủ. Câu "hiện N căn
    #     đầu" là dấu hiệu `_hien_ket_qua_tim` đã chạy — ở đây nó là LỖI.
    #   · thay vào đó phải có đúng SO_GOI_Y dòng căn gợi ý gọn.
    co_ds = any(isinstance(v, str) and "căn đầu" in v for _, v in ra)
    goi_y = sum(str(v).count("font-variant-numeric:tabular-nums")
                for _, v in ra)
    print(f"     danh sách đầy đủ trong khung chat: "
          f"{'CÓ (SAI)' if co_ds else 'không (đúng)'} "
          f"· {goi_y} dòng căn gợi ý")
    if co_ds:
        globals()["loi"] = globals().get("loi", 0) + 1
    if t["so_khop"] and goi_y < min(_app.SO_GOI_Y, t["so_khop"]):
        print(f"     ✗ thiếu dòng gợi ý: cần {min(_app.SO_GOI_Y, t['so_khop'])}")
        globals()["loi"] = globals().get("loi", 0) + 1

# tốc độ: vẽ lại một cuộc 5 lượt mất bao lâu
import time
globals()["BOM_CAU"] = None
t0 = time.time()
for _ in range(3):
    try: _app.render_tim_nha()
    except _RERUN: pass
print(f"\n  vẽ lại cuộc {len(_st.session_state['tn_lich'])} lượt: "
      f"{(time.time()-t0)/3*1000:.0f} ms/lần")
print("\nsố lỗi:", loi)
