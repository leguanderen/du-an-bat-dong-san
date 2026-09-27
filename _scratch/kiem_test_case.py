# -*- coding: utf-8 -*-
"""Bộ test case chatbot Tìm nhà — bản chạy được.

Mỗi test case là một cuộc hội thoại (1 hoặc nhiều lượt). Mỗi lượt ghi:
    câu người dùng gõ
    điều kiện PHẢI hiểu ra        {khoá: giá trị}
    điều kiện KHÔNG ĐƯỢC có       (khoá, ...)
    cờ kiểm thêm                  {"tiep": True/False, "loai": ..., "goi_y": True,
                                   "ghi_chu": True, "hoi_tuoi": True, "con": True}

Chạy:   python _scratch/kiem_test_case.py
Ra:     bảng kết quả trên màn hình + file _scratch/ket_qua_test_case.md
        (dán thẳng vào phụ lục khoá luận được).

Mọi lượt đi qua `tim_nha.buoc_hoi_thoai` — đúng hàm giao diện đang gọi, nên
kết quả ở đây là kết quả người dùng thấy.
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

GOC = Path(__file__).resolve().parent.parent
for d in (GOC, GOC / "pipeline"):
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))

import tim_nha as tnh  # noqa: E402

CC, ND = "chungcu", "nhadat"
TY = 1e9

# (mã, nhóm, loại ban đầu, mô tả, [(câu, phải có, không được có, cờ)])
TEST_CASE = [
    # ------------------------------------------------------------ A. Cơ bản
    ("TC-01", "Cơ bản", CC, "Đủ giá, phòng, quận trong một câu", [
        ("Tìm chung cư 2 phòng ngủ ở Cầu Giấy dưới 4 tỷ",
         {"gia_den": 4 * TY, "so_phong_ngu": 2, "quan_huyen": "Quận Cầu Giấy"},
         (), {"con": True})]),
    ("TC-02", "Cơ bản", CC, "Khoảng giá và khoảng diện tích", [
        ("căn hộ 3-5 tỷ, 60-80m2 ở Thanh Xuân",
         {"gia_tu": 3 * TY, "gia_den": 5 * TY, "dt_tu": 60.0, "dt_den": 80.0,
          "quan_huyen": "Quận Thanh Xuân"}, (), {"con": True})]),
    ("TC-03", "Cơ bản", CC, "Viết tắt 2PN2WC dính liền", [
        ("chung cư 2PN2WC Hà Đông",
         {"so_phong_ngu": 2, "so_phong_vs": 2, "quan_huyen": "Quận Hà Đông"},
         (), {"con": True})]),
    ("TC-04", "Cơ bản", CC, "Tên quận chứa tên phường (Nam Từ Liêm)", [
        ("căn hộ Nam Từ Liêm 2 ngủ",
         {"quan_huyen": "Quận Nam Từ Liêm"}, ("phuong_moi",), {"con": True})]),
    ("TC-05", "Cơ bản", ND, "Nhà đất: tầng, mặt tiền, ngõ, ô tô", [
        ("nhà 4 tầng mặt tiền 5m ngõ 3m ô tô vào, dưới 8 tỷ Đống Đa",
         {"tang_tu": 4, "mt_tu": 5.0, "ngo_tu": 3.0, "gia_den": 8 * TY,
          "co": ("co", "nhac_o_to"), "quan_huyen": "Quận Đống Đa"},
         (), {"loai": ND})]),
    ("TC-06", "Cơ bản", ND, "Loại hình biệt thự kèm diện tích", [
        ("biệt thự 200m2 Long Biên",
         {"loai_hinh": "Nhà biệt thự", "dt_tu": 200.0,
          "quan_huyen": "Quận Long Biên"}, ("mt_tu",), {"loai": ND})]),
    ("TC-07", "Cơ bản", CC, "Phủ định: không cần thang máy", [
        ("chung cư không cần thang máy dưới 2 tỷ",
         {"khong": ("khong", "nhac_thang_may")}, ("co",), {})]),

    # ------------------------------------------------------------ B. Hướng
    ("TC-08", "Hướng", CC, "Hướng chung (chung cư: cửa HOẶC ban công)", [
        ("căn hộ hướng Đông Nam ở Hoàng Mai",
         {"huong": ["Đông Nam"], "quan_huyen": "Quận Hoàng Mai"},
         ("huong_cua", "huong_ban_cong"), {"con": True, "ghi_chu": True})]),
    ("TC-09", "Hướng", CC, "Hướng ban công", [
        ("căn hộ 2PN ban công hướng Nam",
         {"huong_ban_cong": ["Nam"], "so_phong_ngu": 2}, ("huong",),
         {"con": True})]),
    ("TC-10", "Hướng", CC, "Cửa và ban công khác hướng trong một câu", [
        ("cửa chính hướng Tây Bắc, ban công hướng Đông Nam",
         {"huong_cua": ["Tây Bắc"], "huong_ban_cong": ["Đông Nam"]}, (),
         {"con": True})]),
    ("TC-11", "Hướng", ND, "Nhiều hướng nối bằng 'hoặc' / dấu phẩy", [
        ("nhà hướng Nam hoặc Đông Nam, Hai Bà Trưng",
         {"huong": ["Nam", "Đông Nam"], "quan_huyen": "Quận Hai Bà Trưng"},
         (), {"con": True}),
        ]),
    ("TC-12", "Hướng", CC, "Tránh hướng", [
        ("căn hộ Hà Đông 2 ngủ tránh hướng Tây",
         {"huong_tru": ["Tây"]}, ("huong",), {"con": True})]),
    ("TC-13", "Hướng", CC, "Đông tứ trạch → 4 hướng", [
        ("chung cư hợp Đông tứ trạch dưới 5 tỷ",
         {"huong": ["Bắc", "Nam", "Đông", "Đông Nam"],
          "_tu_trach": "Đông tứ trạch"}, (), {"con": True})]),
    ("TC-14", "Hướng", CC, "Hướng cụ thể thắng tứ trạch", [
        ("hướng Bắc, hợp đông tứ trạch",
         {"huong": ["Bắc"]}, ("_tu_trach",), {})]),
    ("TC-15", "Hướng", CC, "Hỏi hợp tuổi theo năm sinh → chỉ cách hỏi", [
        ("tôi sinh năm 1990, tìm căn hướng hợp tuổi",
         {}, ("huong",), {"hoi_tuoi": True})]),
    ("TC-16", "Hướng", ND, "Địa danh chứa chữ chỉ hướng không bị hiểu là hướng", [
        ("nhà Tây Hồ 10 tỷ", {"quan_huyen": "Quận Tây Hồ"}, ("huong",), {}),
        ]),
    ("TC-17", "Hướng", ND, "Hướng đứng sát tên quận", [
        ("nhà hướng Nam Đống Đa",
         {"huong": ["Nam"], "quan_huyen": "Quận Đống Đa"}, ("phuong_moi",), {})]),
    ("TC-18", "Hướng", CC, "Hai hướng liền nhau không bị đọc thành phường cũ", [
        ("căn hộ hướng Nam, Đông Nam",
         {"huong": ["Nam", "Đông Nam"]}, ("phuong_moi",), {})]),
    ("TC-19", "Hướng", CC, "Viết tắt hướng", [
        ("căn hộ hướng ĐN 70m2", {"huong": ["Đông Nam"], "dt_tu": 70.0}, (), {})]),

    # ------------------------------------------------- C. Tầng, nội thất
    ("TC-20", "Tầng, nội thất", CC, "Tầng trung + full đồ", [
        ("căn hộ tầng trung full đồ ở Cầu Giấy",
         {"tang_tu": 10, "tang_den": 20,
          "noi_that": ["Nội thất đầy đủ", "Nội thất cao cấp"],
          "quan_huyen": "Quận Cầu Giấy"}, (), {"con": True, "ghi_chu": True})]),
    ("TC-21", "Tầng, nội thất", CC, "Tầng N trở lên", [
        ("căn hộ tầng 15 trở lên ở Hà Đông", {"tang_tu": 15},
         ("tang_den",), {"con": True})]),
    ("TC-22", "Tầng, nội thất", CC, "Bàn giao thô", [
        ("căn hộ bàn giao thô Long Biên", {"noi_that": ["Bàn giao thô"]}, (),
         {"con": True})]),
    ("TC-23", "Tầng, nội thất", CC, "'Không cần nội thất' không thành 'bàn giao thô'", [
        ("căn hộ không cần nội thất dưới 3 tỷ", {"gia_den": 3 * TY},
         ("noi_that",), {})]),

    # -------------------------------------------------- D. Nhiều lượt
    ("TC-24", "Nhiều lượt", CC, "Thu hẹp bằng câu dài, giữ ngữ cảnh", [
        ("Chung cư Nam Từ Liêm 2 phòng ngủ dưới 5 tỷ",
         {"quan_huyen": "Quận Nam Từ Liêm", "gia_den": 5 * TY}, (), {}),
        ("Chỉ lấy những căn có ban công hướng Nam thôi",
         {"quan_huyen": "Quận Nam Từ Liêm", "gia_den": 5 * TY,
          "so_phong_ngu": 2, "huong_ban_cong": ["Nam"]}, (), {"tiep": True}),
        ("đổi sang hướng Đông Nam",
         {"huong_ban_cong": ["Đông Nam"]}, ("huong",), {"tiep": True}),
        ("bỏ điều kiện hướng ban công",
         {"quan_huyen": "Quận Nam Từ Liêm"}, ("huong_ban_cong",),
         {"tiep": True, "con": True}),
    ]),
    ("TC-25", "Nhiều lượt", CC, "Rẻ hơn nữa + thêm tiện ích", [
        ("căn hộ 5 tỷ Cầu Giấy", {"gia_den": 5 * TY}, (), {}),
        ("rẻ hơn nữa", {"quan_huyen": "Quận Cầu Giấy"}, (), {"tiep": True}),
        ("thêm thang máy", {"co": ("co", "nhac_thang_may")}, (), {"tiep": True}),
    ]),
    ("TC-26", "Nhiều lượt", CC, "Đổi quận thì bỏ phường cũ", [
        ("căn hộ 3 tỷ phường Cầu Giấy", {"phuong_moi": "Phường Cầu Giấy"}, (), {}),
        ("Thanh Xuân thì sao", {"quan_huyen": "Quận Thanh Xuân"},
         ("phuong_moi",), {"tiep": True, "con": True}),
    ]),
    ("TC-27", "Nhiều lượt", CC, "Thêm tầng rồi nới: không mất quận", [
        ("căn hộ Cầu Giấy 3 tỷ", {"quan_huyen": "Quận Cầu Giấy"}, (), {}),
        ("tầng trung thôi", {"quan_huyen": "Quận Cầu Giấy"}, (),
         {"tiep": True, "con": True}),
    ]),
    ("TC-28", "Nhiều lượt", CC, "Mở câu bằng 'tìm cho tôi' = tìm mới", [
        ("căn hộ Hoàng Mai dưới 3 tỷ", {"quan_huyen": "Quận Hoàng Mai"}, (), {}),
        ("tìm cho tôi căn hộ 2 phòng ngủ có thang máy",
         {"so_phong_ngu": 2}, ("quan_huyen", "gia_den"), {"tiep": False}),
    ]),
    ("TC-29", "Nhiều lượt", CC, "Chuyển sang nhà đất: giữ khu vực, giá", [
        ("chung cư Hoàng Mai dưới 3 tỷ", {"quan_huyen": "Quận Hoàng Mai"}, (), {}),
        ("nhà đất thì sao",
         {"quan_huyen": "Quận Hoàng Mai", "gia_den": 3 * TY}, (),
         {"tiep": False, "loai": ND}),
    ]),
    ("TC-30", "Nhiều lượt", ND, "Đang nhà đất, nói 'ban công' không nhảy sang chung cư", [
        ("nhà đất Long Biên 6 tỷ", {"quan_huyen": "Quận Long Biên"}, (), {}),
        ("thêm ban công hướng Nam", {"quan_huyen": "Quận Long Biên"}, (),
         {"tiep": True, "loai": ND}),
    ]),

    # --------------------------------------------- E. Không dấu, viết tắt
    ("TC-31", "Không dấu", CC, "Gõ không dấu", [
        ("can ho 3 phong ngu ha dong duoi 3 ty huong dong nam",
         {"so_phong_ngu": 3, "quan_huyen": "Quận Hà Đông", "gia_den": 3 * TY,
          "huong": ["Đông Nam"]}, (), {})]),
    ("TC-32", "Không dấu", CC, "Tiền viết 'tỷ rưỡi' kiểu nói", [
        ("chung cu 2 ty 5 cau giay", {"gia_den": 2.5 * TY}, (), {})]),

    # --------------------------------------------- F. Trường hợp biên
    ("TC-33", "Biên", ND, "Giá phi thực tế → nói giá thật ở đó", [
        ("biệt thự Hoàn Kiếm giá 1 tỷ", {"loai_hinh": "Nhà biệt thự"}, (),
         {"goi_y": True})]),
    ("TC-34", "Biên", CC, "Ngân sách quá thấp cho số phòng", [
        ("chung cư Ba Đình 2PN dưới 1 tỷ", {"so_phong_ngu": 2}, (),
         {"goi_y": True})]),
    ("TC-35", "Biên", CC, "Câu không có điều kiện nào", [
        ("xin chào", {}, ("gia_den", "quan_huyen"), {})]),
    ("TC-36", "Biên", ND, "'mặt tiền 5m trong ngõ' là số đo, không phải mặt phố", [
        ("nhà mặt tiền 5m trong ngõ Thanh Xuân",
         {"mt_tu": 5.0, "loai_hinh": "Nhà ngõ, hẻm"}, (), {})]),
    ("TC-37", "Biên", ND, "Loại trừ loại hình", [
        ("nhà dưới 5 tỷ nhưng không ở trong ngõ",
         {"loai_hinh_tru": ["Nhà ngõ, hẻm"]}, ("loai_hinh",), {})]),
    ("TC-38", "Biên", CC, "'Tây Hồ' trong câu có hướng", [
        ("căn hộ Tây Hồ hướng Đông",
         {"quan_huyen": "Quận Tây Hồ", "huong": ["Đông"]}, ("co",), {})]),
    ("TC-39", "Biên", CC, "Sắp xếp theo mức lệch giá", [
        ("căn nào đang rẻ hơn thị trường ở Hà Đông",
         {"sap": "lech_thap", "quan_huyen": "Quận Hà Đông"}, ("co",), {})]),
]


def _khop(thuc, ky_vong) -> bool:
    if isinstance(ky_vong, tuple) and len(ky_vong) == 2 and ky_vong[0] in ("co", "khong"):
        return isinstance(thuc, list) and ky_vong[1] in thuc
    if isinstance(ky_vong, list):
        return isinstance(thuc, list) and sorted(thuc) == sorted(ky_vong)
    if isinstance(ky_vong, float):
        try:
            return abs(float(thuc) - ky_vong) <= abs(ky_vong) * 0.01 + 1e-9
        except (TypeError, ValueError):
            return False
    return thuc == ky_vong


def _gon(v) -> str:
    if isinstance(v, float) and v >= 1e8:
        return f"{v/1e9:g} tỷ"
    if isinstance(v, tuple):
        return v[1]
    if isinstance(v, list):
        return "/".join(map(str, v))
    return str(v)


def chay() -> int:
    dong, hong_ca = [], 0
    tong_luot = hong_luot = 0
    for ma, nhom, loai, mo_ta, luot in TEST_CASE:
        dk, tk, ca_hong = {}, {}, False
        print(f"\n{ma} [{nhom}] {mo_ta}")
        for cau, can, cam, co in luot:
            tong_luot += 1
            b = tnh.buoc_hoi_thoai(cau, dk, loai, tk)
            dk, loai = b["dk"], b["loai"]
            kq = tnh.tim("", loai, dk_them=dk, k=5)
            tk = kq.get("thong_ke", {})
            loi = []
            for k, v in can.items():
                if k == "co" or k == "khong":
                    if not _khop(dk.get(k), v):
                        loi.append(f"thiếu {k}={v[1]}")
                elif k not in dk:
                    loi.append(f"thiếu {k}")
                elif not _khop(dk[k], v):
                    loi.append(f"{k}={dk[k]!r}, cần {v!r}")
            for k in cam:
                if dk.get(k):
                    loi.append(f"không được có {k}={dk[k]!r}")
            if "tiep" in co and b["tiep"] != co["tiep"]:
                loi.append("phải là câu " + ("tiếp" if co["tiep"] else "mới"))
            if "loai" in co and loai != co["loai"]:
                loi.append(f"loại {loai}, cần {co['loai']}")
            if co.get("con") and kq["so_khop"] == 0:
                loi.append("không còn căn nào")
            if co.get("goi_y") and not kq.get("goi_y_gia"):
                loi.append("thiếu gợi ý giá thực tế")
            if co.get("ghi_chu") and not kq.get("ghi_chu"):
                loi.append("thiếu ghi chú độ phủ")
            if co.get("hoi_tuoi") and not tnh.hoi_hop_tuoi(cau):
                loi.append("không nhận ra câu hỏi hợp tuổi")

            the = ", ".join(c for _, c in tnh.the_dk(dk, loai)) or "—"
            if kq["bo_qua"]:
                the += " · nới: " + ", ".join(kq["bo_qua"])
            dat = not loi
            hong_luot += not dat
            ca_hong |= not dat
            print(f"  {'✓' if dat else '✗'} {'tiếp' if b['tiep'] else 'mới '} "
                  f"“{cau}” → {the} · {kq['so_khop']} căn")
            for x in loi:
                print(f"       ✗ {x}")
            ky_vong = "; ".join(f"{k}={_gon(v)}" for k, v in can.items()) or "—"
            dong.append((ma, nhom, cau, ky_vong, f"{the} ({kq['so_khop']} căn)",
                         "Đạt" if dat else "Hỏng: " + "; ".join(loi)))
        hong_ca += ca_hong

    n = len(TEST_CASE)
    print(f"\n{n - hong_ca}/{n} test case đạt · {tong_luot - hong_luot}/"
          f"{tong_luot} lượt đúng")

    md = ["# Kết quả test case chatbot Tìm nhà",
          "", f"Chạy lúc {datetime.now():%H:%M %d/%m/%Y} · "
          f"**{n - hong_ca}/{n} test case đạt**, {tong_luot - hong_luot}/"
          f"{tong_luot} lượt đúng.", "",
          "| Mã | Nhóm | Câu người dùng | Mong đợi | Hệ thống hiểu | Kết quả |",
          "|---|---|---|---|---|---|"]
    for r in dong:
        md.append("| " + " | ".join(str(x).replace("|", "/") for x in r) + " |")
    (GOC / "_scratch" / "ket_qua_test_case.md").write_text(
        "\n".join(md) + "\n", encoding="utf-8")
    return hong_ca


if __name__ == "__main__":
    sys.exit(1 if chay() else 0)
