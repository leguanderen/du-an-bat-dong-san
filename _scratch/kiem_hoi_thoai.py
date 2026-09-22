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
"""Bộ kiểm lớp hội thoại nhiều lượt của tab Tìm nhà.

Kiểm một cuộc hội thoại THẬT chứ không kiểm từng câu rời: lỗi của lớp này gần
như luôn là lỗi TÍCH LUỸ — lượt 3 làm mất điều kiện của lượt 1, hoặc đổi quận
mà quên bỏ phường của quận cũ nên kết quả về không. Kiểm câu rời không bao giờ
bắt được những lỗi đó.

Mỗi kịch bản: một chuỗi lượt, mỗi lượt nêu điều kiện phải có / không được có
SAU khi áp câu đó, và số căn khớp phải > 0 (trừ khi nói rõ là được phép cạn).

    python _scratch/kiem_hoi_thoai.py
"""

import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

import tim_nha as tnh  # noqa: E402

# (nhãn kịch bản, loại, [(câu, {phải có}, (không được có,), phải_con_ket_qua)])
KICH_BAN = [
    ("Hạ dần ngân sách", "chungcu", [
        ("tôi có 5 tỷ, 2 phòng ngủ ở Cầu Giấy",
         {"gia_den": 5e9, "so_phong_ngu": 2, "quan_huyen": "Quận Cầu Giấy"},
         (), True),
        # "rẻ hơn nữa" phải HẠ trần giá và GIỮ mọi điều kiện cũ
        ("rẻ hơn nữa",
         {"so_phong_ngu": 2, "quan_huyen": "Quận Cầu Giấy"}, (), True),
        ("rẻ hơn nữa",
         {"so_phong_ngu": 2, "quan_huyen": "Quận Cầu Giấy"}, (), True),
    ]),
    ("Thêm rồi bỏ một yêu cầu", "chungcu", [
        ("căn hộ dưới 4 tỷ ở Hà Đông", {"gia_den": 4e9}, (), True),
        ("thêm thang máy", {"co": ("co", "nhac_thang_may")}, (), True),
        # bỏ đi thì KHÔNG được biến thành "loại căn có thang máy"
        ("bỏ điều kiện thang máy", {"gia_den": 4e9}, ("co", "khong"), True),
    ]),
    ("Đổi quận phải bỏ phường của quận cũ", "chungcu", [
        ("căn hộ 3 tỷ phường Cầu Giấy",
         {"phuong_moi": "Phường Cầu Giấy"}, (), True),
        # Đây là lỗi dễ mắc nhất: giữ lại phường cũ thì điều kiện tự mâu thuẫn
        ("đổi sang Thanh Xuân",
         {"quan_huyen": "Quận Thanh Xuân"}, ("phuong_moi", "duong_pho"), True),
    ]),
    ("Mở rộng ra cả quận", "nhadat", [
        ("nhà dưới 5 tỷ phường Định Công",
         {"phuong_moi": "Phường Định Công"}, (), True),
        ("mở rộng ra cả quận", {"gia_den": 5e9}, ("phuong_moi",), True),
    ]),
    ("Nới diện tích", "chungcu", [
        ("căn hộ từ 60 m2 dưới 5 tỷ", {"dt_tu": 60.0}, (), True),
        ("rộng hơn", {"gia_den": 5e9}, (), True),
    ]),
    ("Câu tiếp rất ngắn", "chungcu", [
        ("căn hộ 4 tỷ ở Hoàng Mai", {"gia_den": 4e9}, (), True),
        ("3 ngủ", {"so_phong_ngu": 3, "gia_den": 4e9}, (), True),
    ]),
    ("Nhà đất nhiều lượt", "nhadat", [
        ("nhà 4 tầng dưới 6 tỷ Hoàng Mai",
         {"tang_tu": 4, "gia_den": 6e9, "quan_huyen": "Quận Hoàng Mai"},
         (), True),
        ("thêm ô tô vào được",
         {"co": ("co", "nhac_o_to"), "tang_tu": 4}, (), True),
        ("mặt tiền 4m", {"mt_tu": 4.0, "tang_tu": 4}, (), True),
        ("bỏ điều kiện số tầng", {"mt_tu": 4.0}, ("tang_tu",), True),
    ]),
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
    for nhan, loai, luot in KICH_BAN:
        print(f"\n=== {nhan} [{loai}] ===")
        dk, tk = {}, {}
        for cau, can, khong_duoc, can_con in luot:
            tiep = tnh.la_cau_tiep(cau, dk)
            if tiep:
                dk, doi = tnh.doc_chinh_sua(cau, dk, tk, loai)
            else:
                dk, doi = tnh.phan_tich(cau, loai), []
            kq = tnh.tim("", loai, dk_them=dk, k=5)
            tk = kq.get("thong_ke", {})

            loi = []
            for k, v in can.items():
                if k not in dk:
                    loi.append(f"thiếu {k} (cần {v})")
                elif not _khop(dk[k], v):
                    loi.append(f"{k}={dk[k]!r} nhưng cần {v!r}")
            for k in khong_duoc:
                if dk.get(k):
                    loi.append(f"KHÔNG được còn {k}={dk[k]!r}")
            if can_con and kq["so_khop"] == 0:
                loi.append("không còn căn nào khớp")

            dau = "tiếp " if tiep else "MỚI  "
            if loi:
                hong += 1
                print(f"  ✗ {dau}“{cau}”")
                for x in loi:
                    print(f"       {x}")
                print(f"       điều kiện: {dk}")
            else:
                print(f"  ✓ {dau}“{cau}”  → {kq['so_khop']} căn"
                      + (f" · đổi: {'; '.join(doi)}" if doi else ""))
    tong = sum(len(l) for _, _, l in KICH_BAN)
    print(f"\n{tong - hong}/{tong} lượt đúng, {hong} lượt hỏng")
    return hong


if __name__ == "__main__":
    sys.exit(1 if chay() else 0)
