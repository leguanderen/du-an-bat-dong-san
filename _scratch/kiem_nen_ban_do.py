"""Kiểm nền bản đồ folium: URL, thứ tự chạy, và dây chuyền dự phòng.

Kiểm trên HTML THẬT mà folium sinh ra, không kiểm trên ý định trong code —
lỗi lần này (JS chạy trước khi bản đồ tồn tại) chỉ lộ ra ở đó.
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
import sys, re
exec(open(_GOC / "_scratch" / "thu_app.py", encoding="utf-8").read().split("loi = 0")[0])
import app, folium

m = folium.Map(location=[20.984, 105.833], zoom_start=17,
               tiles=None, control_scale=True)
app._nen_ban_do(m, folium)
folium.Marker([20.984, 105.833],
              icon=folium.Icon(color="darkblue", icon="home")).add_to(m)
h = m.get_root().render()

loi = 0
def kiem(n, ok, ct=""):
    global loi
    print(("  ✓ " if ok else "  ✗ ") + n + ("" if ok else "   << " + str(ct)))
    if not ok: loi += 1

url_nen = re.search(r'L\.tileLayer\(\s*"([^"]+)"', h).group(1)
kiem("nền chính là máy chủ gương OSM Đức",
     url_nen.startswith("https://tile.openstreetmap.de/"), url_nen)
kiem("nền chính KHÔNG phải tile.openstreetmap.org (đang bị chặn)",
     "tile.openstreetmap.org/{z}" not in url_nen)
kiem("nền chính KHÔNG phải CARTO (trả ảnh 'API KEY REQUIRED')",
     "carto" not in url_nen.lower())
kiem("chỉ MỘT lớp nền, không nhân đôi",
     h.count("L.tileLayer(") == 1, h.count("L.tileLayer("))
kiem("đủ 3 nền dự phòng",
     all(x in h for x in ("openstreetmap.fr", "arcgisonline.com",
                          "tile.openstreetmap.org/{z}")))

ten_map = m.get_name()
i_map = h.index("var %s = L.map(" % ten_map)
i_lop = h.index("L.tileLayer(")
i_js = h.index("lop.on('tileerror'")
kiem("JS dự phòng chạy SAU khi bản đồ được dựng", i_js > i_map,
     f"map@{i_map} js@{i_js}")
kiem("JS dự phòng chạy SAU khi lớp nền được dựng", i_js > i_lop,
     f"lop@{i_lop} js@{i_js}")

ten_lop = re.search(r"var (tile_layer_\w+) = L\.tileLayer\(", h).group(1)
kiem("JS trỏ đúng biến lớp nền", "var lop = %s," % ten_lop in h, ten_lop)
kiem("JS trỏ đúng biến bản đồ", "ban_do = %s;" % ten_map in h, ten_map)

kiem("dấu % trong CSS còn nguyên",
     "left:50%;" in h and "translate(-50%,-50%)" in h and "max-width:84%" in h)
kiem("ngưỡng hỏng nối đúng số", "< %d" % app.NGUONG_HONG_NEN in h)
kiem("có câu báo khi hỏng hết, không để màn xám im lặng",
     "không tải được ảnh nền" in h)
kiem("câu báo nói rõ ghim vẫn dùng được", "toạ độ vẫn đúng" in h)
kiem("Jinja2 không nuốt mất {z}/{x}/{y}",
     "{z}" in h and "{x}" in h and "{y}" in h)

# --- con lăn không được nuốt thao tác cuộn trang ---------------------------
# Bắt được lúc dùng thử bản trên mạng: cuộn trang qua bản đồ thì bản đồ nuốt
# con lăn, zoom tuột từ Hà Nội ra cả Đông Nam Á, trang đứng im.
kiem("mặc định con lăn thuộc về TRANG",
     "scrollWheelZoom.disable()" in h)
kiem("bấm vào bản đồ thì con lăn thuộc về bản đồ",
     "on('click'" in h and "scrollWheelZoom.enable()" in h)
kiem("đưa chuột ra ngoài thì trả con lăn lại cho trang",
     "mouseleave" in h)
kiem("đoạn JS con lăn chạy SAU khi bản đồ được dựng",
     h.index("scrollWheelZoom.disable()") > h.index("= L.map("))

print("\nsố lỗi:", loi)
sys.exit(1 if loi else 0)
