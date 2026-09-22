from __future__ import annotations

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
"""Bộ kiểm bộ hiểu câu tiếng Việt của tab Tìm nhà.

VÌ SAO CẦN FILE NÀY
-------------------
Bộ phân tích câu là thứ dễ hỏng âm thầm nhất trong cả hệ thống: sửa một biểu
thức chính quy cho câu này chạy được thì làm vỡ câu khác, mà chẳng có lỗi nào
báo ra — nó chỉ lặng lẽ hiểu sai. Ba lỗi đã từng xảy ra đúng như vậy:

    "5 tỷ 2PN Cầu Giấy"   -> nhận cả quận LẪN phường LẪN phố cùng tên -> 1 căn
    "mua nhà 800 triệu"   -> khoá "tri" của phường Phượng Trì nằm trong "trieu"
    "3-5 tỷ"              -> đọc thành "dưới 5 tỷ" vì dấu gạch bị bỏ

Nên mỗi hành vi đã sửa phải có một câu ở đây khoá lại. Chạy:

    python _scratch/kiem_tim_nha.py

CÁCH ĐỌC MỘT DÒNG KỲ VỌNG
-------------------------
    (loại, câu, {điều kiện phải có}, {khoá không được xuất hiện})

Kỳ vọng là TẬP CON: chỉ kiểm những khoá được liệt kê, không đòi khớp y nguyên
cả dict. Làm vậy để thêm khả năng hiểu mới không làm đỏ toàn bộ bộ kiểm cũ.
"""

import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

import tim_nha as tnh  # noqa: E402

# Ký hiệu đặc biệt cho giá trị kỳ vọng:
#   ("~", x)        — số thực, cho phép lệch 1%
#   ("co", [...])   — danh sách phải CHỨA các phần tử này
BO_KIEM: list[tuple] = [
    # ------------------------------------------------------------------ giá
    ("chungcu", "tôi có 5 tỷ", {"gia_den": 5e9}, ("gia_tu",)),
    ("chungcu", "dưới 5 tỷ", {"gia_den": 5e9}, ()),
    ("chungcu", "trên 3 tỷ", {"gia_tu": 3e9}, ("gia_den",)),
    ("chungcu", "từ 3 tỷ", {"gia_tu": 3e9}, ()),
    ("nhadat", "nhà 3-5 tỷ", {"gia_tu": 3e9, "gia_den": 5e9}, ()),
    ("nhadat", "nhà từ 3 đến 5 tỷ", {"gia_tu": 3e9, "gia_den": 5e9}, ()),
    ("nhadat", "mua nhà 800 triệu", {"gia_den": 8e8}, ("phuong_moi",)),
    ("nhadat", "nhà 2 tỷ 8", {"gia_den": 2.8e9}, ()),
    ("nhadat", "nhà 6 tỷ 5 ở Hà Đông",
     {"gia_den": 6.5e9, "quan_huyen": "Quận Hà Đông"}, ("duong_pho",)),
    ("chungcu", "căn hộ khoảng 5 tỷ", {"gia_tu": 4.5e9, "gia_den": 5.5e9}, ()),
    ("chungcu", "tầm 3 tỷ", {"gia_tu": 2.7e9, "gia_den": 3.3e9}, ()),
    ("nhadat", "5ty", {"gia_den": 5e9}, ()),
    # HAI CỤM SỐ TRONG MỘT CÂU: chữ "từ" thuộc về diện tích, không thuộc về
    # giá. Bản trước quét cả đoạn 22 ký tự trước con số nên hiểu ngược thành
    # "giá TỪ 5 tỷ" — trong khi người dùng nói rõ "dưới 5 tỷ".
    ("chungcu", "căn hộ từ 60 m2 dưới 5 tỷ",
     {"dt_tu": 60.0, "gia_den": 5e9}, ("gia_tu",)),
    ("nhadat", "nhà trên 50 m2 dưới 4 tỷ",
     {"dt_tu": 50.0, "gia_den": 4e9}, ("gia_tu",)),
    ("nhadat", "nhà từ 4 tầng dưới 6 tỷ",
     {"tang_tu": 4, "gia_den": 6e9}, ("gia_tu", "tang_den")),
    ("nhadat", "nhà 4 tỉ", {"gia_den": 4e9}, ()),

    # --------------------------------------------------- mét KHÔNG phải triệu
    ("nhadat", "nhà 4 tầng mặt tiền 5m",
     {"tang_tu": 4, "mt_tu": 5.0}, ("gia_den", "gia_tu")),
    ("nhadat", "ngõ rộng 5m ô tô tránh",
     {"ngo_tu": 5.0}, ("gia_den", "gia_tu")),
    ("nhadat", "mặt tiền 4m ngõ 3m", {"mt_tu": 4.0, "ngo_tu": 3.0}, ()),

    # ------------------------------------------------------------- diện tích
    ("chungcu", "căn hộ 60m2", {"dt_tu": 60.0}, ()),
    ("chungcu", "căn hộ dưới 60m2", {"dt_den": 60.0}, ("dt_tu",)),
    ("chungcu", "từ 60 đến 80 m2", {"dt_tu": 60.0, "dt_den": 80.0}, ()),
    ("nhadat", "đất 60m2 Đông Anh",
     {"dt_tu": 60.0, "quan_huyen": "Huyện Đông Anh"}, ("phuong_moi",)),

    # ------------------------------------------------------------------ phòng
    ("chungcu", "2 phòng ngủ", {"so_phong_ngu": 2}, ()),
    ("chungcu", "2PN 2WC", {"so_phong_ngu": 2, "so_phong_vs": 2}, ()),
    ("chungcu", "3 ngủ", {"so_phong_ngu": 3}, ()),

    # ------------------------------------------------------------- số tầng
    ("nhadat", "nhà 4 tầng", {"tang_tu": 4}, ()),
    ("nhadat", "nhà 5 tầng ở Hoàng Mai",
     {"tang_tu": 5, "quan_huyen": "Quận Hoàng Mai"}, ("duong_pho",)),
    ("nhadat", "nhà từ 3 tầng trở lên", {"tang_tu": 3}, ()),

    # ------------------------------------------------------------- địa điểm
    ("chungcu", "ở Cầu Giấy", {"quan_huyen": "Quận Cầu Giấy"},
     ("phuong_moi", "duong_pho")),
    ("chungcu", "ở phường Cầu Giấy", {"phuong_moi": "Phường Cầu Giấy"}, ()),
    ("nhadat", "nhà 3-5 tỷ Hoàng Mai ô tô vào được",
     {"quan_huyen": "Quận Hoàng Mai"}, ("duong_pho", "phuong_moi")),
    ("nhadat", "nhà ở Hà Đông", {"quan_huyen": "Quận Hà Đông"},
     ("duong_pho", "phuong_moi")),
    ("chungcu", "căn hộ ở quận 2 bà trưng",
     {"quan_huyen": "Quận Hai Bà Trưng"}, ()),
    ("chungcu", "chung cư quận 3 đình", {}, ()),          # không được vỡ
    ("nhadat", "nhà phố Trung Kính", {"duong_pho": ("co", "Trung Kính")}, ()),

    # ---------------------------------------------- tên phường TRƯỚC sáp nhập
    ("nhadat", "nhà Mỹ Đình dưới 6 tỷ",
     {"gia_den": 6e9, "phuong_moi": ("co", "Từ Liêm")}, ()),
    ("chungcu", "căn hộ Dịch Vọng", {"phuong_moi": ("co", "Cầu Giấy")}, ()),
    # "Nhân Chính" KHÔNG còn là phường sau sáp nhập — nó nhập vào Thanh Xuân.
    # Kỳ vọng ở đây là tên mới, không phải tên người dùng gõ.
    ("nhadat", "nhà Nhân Chính", {"phuong_moi": ("co", "Thanh Xuân")}, ()),

    # ------------------------------------------------------------- tên dự án
    ("chungcu", "chung cư 2 ngủ Times City",
     {"so_phong_ngu": 2, "du_an": "Times City"}, ()),
    ("chungcu", "Vinhomes Ocean Park 3 phòng ngủ",
     {"so_phong_ngu": 3, "du_an": "Vinhomes Ocean Park"}, ()),
    ("chungcu", "Royal City dưới 8 tỷ",
     {"du_an": "Royal City", "gia_den": 8e9}, ()),

    # -------------------------------------------------------------- phủ định
    ("chungcu", "chung cư không cần thang máy dưới 3 tỷ",
     {"gia_den": 3e9, "khong": ("co", "nhac_thang_may")}, ()),
    ("nhadat", "nhà dưới 5 tỷ nhưng không ở trong ngõ",
     {"gia_den": 5e9, "loai_hinh_tru": ("co", "Nhà ngõ, hẻm")},
     ("loai_hinh",)),
    ("nhadat", "nhà có ô tô, không cần kinh doanh",
     {"co": ("co", "nhac_o_to"), "khong": ("co", "nhac_kinh_doanh")}, ()),
    ("chungcu", "chưa có sổ cũng được, miễn là có thang máy",
     {"co": ("co", "nhac_thang_may")}, ()),

    # ------------------------------------------ một cụm mang HAI vai khác nhau
    # "mặt tiền 5m" là SỐ ĐO, không phải yêu cầu nhà mặt phố.
    ("nhadat", "nhà 4 tầng mặt tiền 5m trong ngõ",
     {"mt_tu": 5.0, "loai_hinh": "Nhà ngõ, hẻm"}, ()),
    ("nhadat", "nhà mặt tiền 6m ngõ 4m",
     {"mt_tu": 6.0, "ngo_tu": 4.0}, ("loai_hinh",)),
    # "mặt bằng" ở đây là MẶT BẰNG GIÁ, không phải mặt bằng kinh doanh.
    ("chungcu", "căn nào đang rẻ hơn mặt bằng",
     {"sap": "lech_thap"}, ("co",)),
    ("nhadat", "nhà mặt bằng kinh doanh Cầu Giấy",
     {"co": ("co", "nhac_kinh_doanh")}, ()),

    # --------------------------------------------------------- không khớp bừa
    ("chungcu", "nhà cho gia đình 4 người ngân sách 4 tỷ",
     {"gia_den": 4e9}, ("co",)),          # "cho" trong "cho gia đình"
    ("nhadat", "nhà cần sửa lại", {}, ("co",)),
    ("chungcu", "mua căn hộ", {}, ("gia_den", "quan_huyen", "phuong_moi")),

    # -------------------------------------------------------- gần trung tâm
    ("chungcu", "căn hộ gần trung tâm 4 tỷ",
     {"gia_den": 4e9, "kc_den": 5.0}, ()),
    ("nhadat", "nhà trung tâm thành phố", {"kc_den": 5.0}, ()),

    # -------------------------------------------------- ý định sắp xếp trong câu
    ("nhadat", "nhà nào đang rẻ hơn thị trường ở Hà Đông",
     {"quan_huyen": "Quận Hà Đông", "sap": "lech_thap"}, ("duong_pho",)),
    ("chungcu", "căn hộ rẻ nhất Cầu Giấy",
     {"quan_huyen": "Quận Cầu Giấy", "sap": "re_nhat"}, ()),
    ("chungcu", "căn nào đang bị rao cao hơn mặt bằng",
     {"sap": "lech_cao"}, ()),

    # ------------------------------------------------------------- tổng hợp
    ("chungcu", "tôi có 5 tỷ, 2 phòng ngủ ở Cầu Giấy, có thang máy",
     {"gia_den": 5e9, "so_phong_ngu": 2, "quan_huyen": "Quận Cầu Giấy",
      "co": ("co", "nhac_thang_may")}, ("phuong_moi", "duong_pho")),
    ("nhadat", "nhà mặt phố Thanh Xuân trên 10 tỷ sổ đỏ",
     {"gia_tu": 10e9, "quan_huyen": "Quận Thanh Xuân",
      "loai_hinh": "Nhà mặt phố, mặt tiền"}, ("duong_pho",)),
    ("nhadat", "nhà ngõ Hoàng Mai 4 tầng ô tô đỗ cửa dưới 6 tỷ sổ đỏ",
     {"gia_den": 6e9, "tang_tu": 4, "quan_huyen": "Quận Hoàng Mai",
      "loai_hinh": "Nhà ngõ, hẻm", "co": ("co", "nhac_o_to")},
     ("duong_pho",)),
]


# Đoán nhánh: (câu, nhánh người dùng đang chọn, nhánh phải ra)
BO_KIEM_NHANH = [
    # Câu nhà đất rõ ràng nhưng KHÔNG chứa từ khoá nào trong hai danh sách —
    # phải nhận ra qua số tầng / mặt tiền / độ rộng ngõ.
    ("nhà 4 tầng dưới 6 tỷ Hoàng Mai ô tô vào được", "chungcu", "nhadat"),
    ("nhà 5 tầng mặt tiền 4m", "chungcu", "nhadat"),
    ("ngõ rộng 5m", "chungcu", "nhadat"),
    # Có dấu hiệu chung cư thì số tầng KHÔNG được kéo sang nhà đất
    ("căn hộ tầng 12 toà A", "nhadat", "chungcu"),
    ("chung cư 2 ngủ tầng cao", "nhadat", "chungcu"),
    # Mơ hồ thật thì giữ nguyên lựa chọn của người dùng, không đoán bừa
    ("mua nhà 800 triệu", "chungcu", "chungcu"),
    ("mua nhà 800 triệu", "nhadat", "nhadat"),
    ("5 tỷ ở Cầu Giấy", "chungcu", "chungcu"),
]


def _khop(thuc, ky_vong) -> bool:
    if isinstance(ky_vong, tuple) and ky_vong and ky_vong[0] == "co":
        can = ky_vong[1]
        if isinstance(thuc, (list, tuple, set)):
            return can in thuc
        return isinstance(thuc, str) and can in thuc
    if isinstance(ky_vong, float):
        try:
            return abs(float(thuc) - ky_vong) <= abs(ky_vong) * 0.01 + 1e-9
        except (TypeError, ValueError):
            return False
    return thuc == ky_vong


def chay() -> int:
    hong = 0
    for loai, cau, can, khong_duoc in BO_KIEM:
        dk = tnh.phan_tich(cau, loai)
        loi = []
        for k, v in can.items():
            if k not in dk:
                loi.append(f"thiếu {k} (cần {v})")
            elif not _khop(dk[k], v):
                loi.append(f"{k}={dk[k]!r} nhưng cần {v!r}")
        for k in khong_duoc:
            if k in dk:
                loi.append(f"KHÔNG được có {k}={dk[k]!r}")
        if loi:
            hong += 1
            print(f"  ✗ [{loai}] {cau}")
            for x in loi:
                print(f"        {x}")
            print(f"        hiểu được: {dk}")
        else:
            print(f"  ✓ [{loai}] {cau}")

    # ĐOÁN NHÁNH: kiểm riêng, vì bộ trên luôn TRUYỀN SẴN nhánh vào
    # `phan_tich`, nên nó không bao giờ chạm tới `doan_loai`. Đúng lỗ hổng đó
    # để lọt việc câu "nhà 4 tầng dưới 6 tỷ Hoàng Mai ô tô vào được" bị đưa
    # sang nhánh chung cư — 61/61 câu xanh mà tính năng vẫn sai loại.
    for cau, dang_chon, can in BO_KIEM_NHANH:
        ra = tnh.doan_loai(cau, dang_chon)
        if ra != can:
            hong += 1
            print(f"  ✗ [nhánh] “{cau}” (đang chọn {dang_chon}) "
                  f"-> {ra}, cần {can}")
        else:
            print(f"  ✓ [nhánh] “{cau}” -> {ra}")

    tong = len(BO_KIEM) + len(BO_KIEM_NHANH)
    print(f"\n{tong - hong}/{tong} phép kiểm đúng, {hong} hỏng")
    return hong


if __name__ == "__main__":
    sys.exit(1 if chay() else 0)
