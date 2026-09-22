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
"""Khung chat gọn: ba gợi ý trong bong bóng + trang danh sách riêng."""
import sys, traceback
exec(open(_GOC / "_scratch" / "thu_app.py", encoding="utf-8").read().split("loi = 0")[0])
import streamlit as _st, app as _app, tim_nha as tnh

# Bắt lại mọi thứ được vẽ ra, để kiểm nội dung chứ không chỉ kiểm "không vỡ".
ve = []
for t in ("markdown", "caption", "info", "warning", "dataframe"):
    setattr(_st, t, (lambda t_: (lambda *a, **k: ve.append(
        (t_, a[0] if a else "")) ))(t))

# Ghi lại NHÃN của mọi nút. Bộ thử gốc chỉ ghi tên hàm được gọi, nên không
# kiểm được "nút này có chữ gì" — mà đó đúng là thứ cần kiểm ở đây.
goi_nhan = []
def _nut(nhan=None, *a, **k):
    goi_nhan.append(nhan)
    return False
_st.button = _nut
Col.button = staticmethod(_nut)


def gui(cau=None):
    globals()["BOM_CAU"] = cau
    for _ in range(5):
        ve.clear(); goi_nhan.clear()
        try:
            _app.render_tim_nha(); return
        except _RERUN:
            globals()["BOM_CAU"] = None

def chu():
    return "\n".join(str(x) for _, x in ve)

loi = 0
def kiem(n, ok, ct=""):
    global loi
    print(f"  {'✓' if ok else '✗'} {n}" + ("" if ok else f"   << {ct}"))
    if not ok: loi += 1

try:
    gui("căn hộ 4 tỷ ở Hoàng Mai")
    L = _st.session_state["tn_lich"]
    n = L[-1]["so_khop"]
    print(f"  (lượt đầu: {n} căn khớp)")
    s = chu()
    kiem("lượt đầu có khớp căn", n > _app.SO_GOI_Y, f"chỉ {n} căn")

    # --- ba gợi ý nằm TRONG khung chat
    dong = s.count("tin gốc ↗") + s.count("tỷ</div>")
    kiem("bong bóng cuối có đúng %d dòng căn gợi ý" % _app.SO_GOI_Y,
         s.count("font-variant-numeric:tabular-nums") >= _app.SO_GOI_Y,
         f"{s.count('font-variant-numeric:tabular-nums')} dòng")
    kiem("KHÔNG đổ cả danh sách dài vào khung chat",
         "Đọc danh sách này thế nào" not in s
         and "Xem nội dung tin đã lưu" not in s)
    kiem("có nhắc số căn gợi ý trong bong bóng",
         f"{_app.SO_GOI_Y} căn " in s, s[:200])

    # --- nút sang trang danh sách
    kiem("có nút 'Xem cả N căn'",
         any("Xem cả" in str(x) for x in goi_nhan),
         str(goi_nhan[-8:]))

    # --- trang danh sách đầy đủ
    _st.session_state["tn_xem_het"] = True
    gui()
    s2 = chu()
    kiem("trang danh sách hiện đủ chữ giải thích",
         "Đọc danh sách này thế nào" in s2 or "rẻ / cao hơn mặt bằng" in s2,
         s2[:200])
    kiem("trang danh sách nhắc lại câu đã gõ",
         "căn hộ 4 tỷ ở Hoàng Mai" in s2)
    kiem("trang danh sách KHÔNG vẽ lại bong bóng chat",
         "Bạn đang tìm nhà thế nào" not in s2
         and "Nói thêm một câu để sửa" not in s2)
    kiem("có nút về khung chat",
         any("Về khung chat" in str(x) for x in goi_nhan))

    # --- về lại chat thì cờ tự tắt
    _st.session_state["tn_xem_het"] = False
    gui()
    kiem("về chat thì cờ xem-hết tắt",
         _st.session_state.get("tn_xem_het") is False)

    # --- gõ câu mới lúc đang ở trang danh sách: không được kẹt
    _st.session_state["tn_xem_het"] = True
    gui("rẻ hơn nữa")
    kiem("gõ câu mới không kẹt ở trang danh sách", True)

    # --- lượt cũ KHÔNG kèm gợi ý (nếu không thì cuộc dài lại phình ra)
    _st.session_state["tn_xem_het"] = False
    for _ in range(3): gui("rẻ hơn nữa")
    s3 = chu()
    lich = _st.session_state["tn_lich"]
    kiem("cuộc %d lượt chỉ có gợi ý ở lượt cuối" % len(lich),
         s3.count("font-variant-numeric:tabular-nums") <= _app.SO_GOI_Y + 1,
         f"{s3.count('font-variant-numeric:tabular-nums')} dòng / "
         f"{len(lich)} lượt")

    # --- ba chip gợi ý câu tiếp, không phải sáu
    kiem("chỉ 3 chip gợi ý câu tiếp",
         sum(1 for x in goi_nhan if x in tnh.VI_DU_CAU_TIEP) <= 3,
         str([x for x in goi_nhan if x in tnh.VI_DU_CAU_TIEP]))

    # --- cách sắp xếp phải sống qua chuyến đi sang trang danh sách và về
    _st.session_state["tn_sap_luu"] = "re_nhat"
    _st.session_state["tn_xem_het"] = False
    gui()
    kiem("khung chat dùng đúng cách sắp đã chọn",
         "giá thấp nhất" in chu(), chu()[:160])
    kiem("cách sắp không bị xoá khi rời trang danh sách",
         _st.session_state.get("tn_sap_luu") == "re_nhat",
         str(_st.session_state.get("tn_sap_luu")))

    import time
    t0 = time.time(); gui(); ms = (time.time() - t0) * 1000
    kiem(f"vẽ lại cuộc {len(lich)} lượt: {ms:.0f} ms", ms < 800, f"{ms:.0f} ms")
except Exception as e:
    loi += 1
    print(f"  ✗ VỠ: {type(e).__name__}: {e}")
    traceback.print_exc(limit=6)

print("\nsố lỗi:", loi)
sys.exit(1 if loi else 0)
