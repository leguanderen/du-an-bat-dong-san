"""
clean_chungcu.py — Luồng làm sạch dữ liệu CHUNG CƯ (nguồn: nhatot.com).

Chạy:  python clean_chungcu.py --input nhatot.xlsx --outdir data_clean

ĐẦU RA
------
  chungcu_clean.xlsx          phân khúc "Chung cư" — dùng để huấn luyện model chính
  chungcu_phan_khuc_khac.xlsx tập thể / cư xá / căn hộ mini — giữ riêng, KHÔNG vứt
  chungcu_bao_cao_lam_sach.xlsx  bảng còn bao nhiêu dòng sau mỗi bước
  chungcu_sua_don_vi.xlsx     các dòng bị sửa lỗi đơn vị, để soi tay kiểm chứng

QUYẾT ĐỊNH THIẾT KẾ CẦN GIẢI THÍCH ĐƯỢC KHI BẢO VỆ
--------------------------------------------------
1) KHÔNG tính target encoding ở đây.
   Bản dữ liệu cũ có sẵn các cột `quan_huyen_encoded`, `phuong_xa_encoded`,
   `du_an_encoded` được tính trên TOÀN BỘ dữ liệu rồi mới chia train/test.
   Đó là rò rỉ mục tiêu (target leakage): với 85/140 phường có dưới 10 mẫu,
   giá trị encoding gần như chính là giá của vài tin trong đó — model được xem
   trước đáp án, và mọi chỉ số đánh giá đều đẹp hơn thực tế.
   Encoding phải được tính TRONG TỪNG FOLD lúc huấn luyện. File sạch này chỉ
   chứa dữ liệu thô đã chuẩn hoá, không chứa bất cứ thứ gì suy ra từ giá.

2) GIỮ LẠI các phân khúc khác thay vì lọc bỏ.
   Script cũ giữ đúng nhóm "Chung cư" và loại 604 dòng tập thể/cư xá/căn hộ mini
   mà không ghi lại ở đâu. Việc tách chúng khỏi model chính là đúng — chúng có
   cơ chế giá khác hẳn — nhưng vứt đi thì mất dữ liệu. Ở đây chúng được xuất ra
   file riêng, vừa để đối chiếu vừa để mở rộng hệ thống sau này.

3) Cột `dia_chi` và `mo_ta` được GIỮ NGUYÊN trong đầu ra.
   Bản `ready_to_train` cũ đã đánh mất chúng. `dia_chi` là đầu vào bắt buộc cho
   geocoding; `mo_ta` là đầu vào cho các đặc trưng văn bản.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from clean_common import (
    bu_quan_tu_phuong,
    BLOB_LABELS_CHUNGCU, CleanReport, build_geocode_query, cross_check_parse,
    dedup_exact, dedup_near, expand_attr_blob, fix_price_unit_errors,
    flag_outliers_grouped, normalize_space, parse_address, parse_area,
    plausible_ppm2_band,
    parse_number, parse_price, parse_price_per_m2,
)
from extract_text import (
    add_binary_flags, extract_floor, extract_price_hint, fill_missing_from_text,
)
from gazetteer import UNKNOWN_PROJECT, normalize_project_name

# Phân khúc được coi là "chung cư thương mại" — dùng cho model chính.
PHAN_KHUC_CHINH = {"Chung cư"}

# Khoảng giá/m² hợp lý cho chung cư Hà Nội (VND/m²).
# Cận dưới 8 triệu: dưới mức này gần như chắc chắn là lỗi nhập hoặc tin ảo.
# Cận trên 400 triệu: cao hơn cả căn hộ hạng sang đắt nhất Hà Nội hiện nay.
PPM2_LO, PPM2_HI = 8e6, 400e6


def load_raw(nguon_du_lieu, nguon: str = "nhatot") -> pd.DataFrame:
    """Nạp dữ liệu thô. Nhận đường dẫn hoặc DataFrame sẵn.

    `nguon="batdongsan"` sẽ chạy bộ chuyển đổi trước, để phần còn lại của luồng
    làm sạch không cần biết dữ liệu đến từ đâu — xem nguon_batdongsan.py.
    """
    df = (nguon_du_lieu if isinstance(nguon_du_lieu, pd.DataFrame)
          else pd.read_excel(nguon_du_lieu))
    if nguon == "batdongsan":
        from nguon_batdongsan import chuyen_doi
        df = chuyen_doi(df)
    required = {"ten_du_an", "dia_chi", "gia_ban", "dien_tich", "dac_diem_tong_hop", "mo_ta"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"File thiếu cột bắt buộc: {sorted(missing)}")
    if "nguon" not in df.columns:
        df["nguon"] = nguon
    # Sitemap có selector kiểu Link thì cột `click` chứa URL tin gốc. Bóc mã tin
    # ra ngay ở đây: đó là khoá khử trùng lặp tuyệt đối giữa các đợt cào, thứ mà
    # nhánh chung cư trước giờ không có nên phải so nội dung.
    if "click" in df.columns and "listing_id" not in df.columns:
        df["url_goc"] = df["click"]
        df["listing_id"] = (df["click"].astype(str)
                            .str.extract(r"/(\d+)\.htm")[0])
    return df


def run(input_path, outdir: Path, nguon: str = "nhatot",
        tien_to: str = "chungcu") -> dict:
    outdir.mkdir(parents=True, exist_ok=True)
    df = load_raw(input_path, nguon)
    rep = CleanReport(f"CHUNG CƯ ({nguon})", so_dong_ban_dau=len(df))

    # ---------------------------------------------------------------- bước 1
    # Bóc khối thuộc tính nối liền thành cột riêng.
    df = expand_attr_blob(df, "dac_diem_tong_hop", BLOB_LABELS_CHUNGCU)

    # ---------------------------------------------------------------- bước 2
    # Chuyển chuỗi sang số.
    df["TARGET_gia_vnd"] = df["gia_ban"].map(parse_price)
    df["dien_tich_m2"] = df["dien_tich"].map(parse_area)
    df["so_phong_ngu"] = df.get("Số phòng ngủ", pd.Series(dtype=object)).map(parse_number)
    df["so_phong_vs"] = df.get("Số phòng vệ sinh", pd.Series(dtype=object)).map(parse_number)
    df["tang_so"] = df.get("Tầng số", pd.Series(dtype=object)).map(parse_number)
    df["gia_m2_nguon"] = df.get("Giá/m²", pd.Series(dtype=object)).map(parse_price_per_m2)

    # ---------------------------------------------------------------- bước 3
    # KIỂM CHỨNG PARSER (không phải kiểm chứng dữ liệu — xem ghi chú trong
    # cross_check_parse). Nếu bước này báo lệch, tức là code đọc sai, phải sửa
    # code chứ không phải loại dòng.
    rel = cross_check_parse(df["TARGET_gia_vnd"], df["dien_tich_m2"], df["gia_m2_nguon"])
    n_lech = int((rel > 0.01).sum())
    rep.note("Kiểm chứng parser",
             f"đối chiếu {int(rel.notna().sum())} dòng với Giá/m² do nguồn công bố; "
             f"{n_lech} dòng lệch >1%")
    if n_lech > len(df) * 0.01:
        raise RuntimeError(
            f"Parser sai: {n_lech} dòng lệch quá 1% so với Giá/m² của nguồn. "
            "Kiểm tra lại parse_price / parse_area trước khi chạy tiếp."
        )
    if int(rel.notna().sum()) == 0:
        # Nguồn không công bố Giá/m² (batdongsan). Kiểm chứng thay thế: đối
        # chiếu với giá người bán nhắc trong tiêu đề. Yếu hơn, nhưng nếu parser
        # đọc sai đơn vị thì tỉ lệ khớp sụp xuống gần 0 và thấy ngay.
        from nguon_batdongsan import kiem_chung_tieu_de
        kc = kiem_chung_tieu_de(df["TARGET_gia_vnd"], df["ten_du_an"])
        rep.note("Kiểm chứng parser (thay thế)",
                 f"nguồn không có Giá/m²; đối chiếu {kc['so_doi_chieu']} tiêu đề "
                 f"có nhắc giá -> khớp {kc['ti_le_khop']*100:.1f}%")
        if kc["so_doi_chieu"] >= 30 and kc["ti_le_khop"] < 0.5:
            raise RuntimeError(
                f"Parser nghi sai: chỉ {kc['ti_le_khop']*100:.0f}% khớp với giá "
                "nhắc trong tiêu đề. Kiểm tra parse_price trước khi chạy tiếp.")

    # ---------------------------------------------------------------- bước 4
    # Loại dòng không có giá hoặc diện tích — không thể dùng cho bất cứ việc gì.
    before = len(df)
    mask = df["TARGET_gia_vnd"].isna() | df["dien_tich_m2"].isna() | (df["dien_tich_m2"] <= 0)
    rep.log("Thiếu giá hoặc diện tích", int(mask.sum()), before)
    df = df.loc[~mask].copy()

    # ---------------------------------------------------------------- bước 5
    # Tách địa chỉ thành đường / phường / quận, giữ lại chuỗi gốc.
    addr = df["dia_chi"].map(parse_address).apply(pd.Series)
    df = pd.concat([df, addr], axis=1)
    df["dia_chi_goc"] = df["dia_chi"].map(normalize_space)
    df["geocode_query"] = df.apply(build_geocode_query, axis=1)

    # Bù quận từ tên phường TRƯỚC khi lọc. Từ 1/7/2025 cấp quận/huyện bị bãi
    # bỏ và batdongsan đã đổi địa chỉ toàn site sang cấu trúc mới — "Đường X,
    # Phường Y, Hà Nội". Không bù thì bộ lọc dưới đây quét sạch mọi đợt cào mới,
    # và báo cáo chỉ ghi lặng lẽ "giữ lại 0 dòng".
    df, n_bu = bu_quan_tu_phuong(df)
    if n_bu:
        rep.note("Bù quận/huyện từ tên phường",
                 f"điền {n_bu} ô trống (địa chỉ sau sáp nhập không còn cấp quận)")

    before = len(df)
    mask = df["quan_huyen"].isna()
    rep.log("Không xác định được quận/huyện", int(mask.sum()), before)
    df = df.loc[~mask].copy()

    # ---------------------------------------------------------------- bước 6
    # Chuẩn hoá tên dự án từ tiêu đề tin.
    df["ten_du_an_goc"] = df["ten_du_an"]
    df["du_an_clean"] = df["ten_du_an"].map(normalize_project_name)
    rep.note("Chuẩn hoá tên dự án",
             f"khớp được {int((df['du_an_clean'] != UNKNOWN_PROJECT).sum())}/{len(df)} tin "
             f"về {df.loc[df['du_an_clean'] != UNKNOWN_PROJECT, 'du_an_clean'].nunique()} dự án")

    # ---------------------------------------------------------------- bước 7
    # Sửa lỗi đơn vị giá TRƯỚC khi lọc ngoại lai — nếu làm ngược lại, các tin
    # ghi nhầm đơn vị sẽ bị loại oan thay vì được cứu.
    df["phan_khuc"] = df.get("Loại hình", pd.Series(dtype=object)).fillna("Không rõ")
    # Khoảng chấp nhận cho giá trị SAU khi sửa được suy ra từ chính dữ liệu
    # (phân vị 1–99% của các dòng trông bình thường), không đặt tay.
    acc_lo, acc_hi = plausible_ppm2_band(
        df["TARGET_gia_vnd"], df["dien_tich_m2"], PPM2_LO, PPM2_HI)
    rep.note("Khoảng giá/m² chấp nhận (suy từ dữ liệu)",
             f"{acc_lo/1e6:.1f} – {acc_hi/1e6:.1f} triệu/m²")
    df, unit_log = fix_price_unit_errors(
        df, price_col="TARGET_gia_vnd", area_col="dien_tich_m2",
        lo_ppm2=PPM2_LO, hi_ppm2=PPM2_HI, accept_lo=acc_lo, accept_hi=acc_hi,
    )
    if len(unit_log):
        unit_log = unit_log.join(df[["ten_du_an_goc", "quan_huyen"]], how="left")
        # đối chiếu với giá nhắc trong tiêu đề để tự đánh giá độ tin cậy của luật
        hint = df.loc[unit_log.index, "ten_du_an_goc"].map(extract_price_hint)
        unit_log["gia_trong_tieu_de"] = hint
        khop = (hint.notna() & ((unit_log["gia_sau"] - hint).abs() / hint < 0.15))
        unit_log["tieu_de_xac_nhan"] = np.where(hint.isna(), "không nhắc giá",
                                                np.where(khop, "KHỚP", "lệch"))
        n_xacnhan = int((unit_log["tieu_de_xac_nhan"] == "KHỚP").sum())
        n_conhac = int((unit_log["tieu_de_xac_nhan"] != "không nhắc giá").sum())
        rep.note("Sửa lỗi đơn vị giá",
                 f"sửa {len(unit_log)} dòng; trong số {n_conhac} dòng có nhắc giá ở "
                 f"tiêu đề thì {n_xacnhan} dòng khớp sau khi sửa")
    else:
        rep.note("Sửa lỗi đơn vị giá", "không có dòng nào cần sửa")

    df["gia_tren_m2"] = df["TARGET_gia_vnd"] / df["dien_tich_m2"]

    # ---------------------------------------------------------------- bước 8
    # Khử trùng lặp. Chung cư KHÔNG có cột URL tin gốc (crawler không lấy), nên
    # không có khoá tuyệt đối — phải dựa vào nội dung. Nếu crawl lại, nhớ thêm
    # selector kiểu Link để có khoá chắc chắn như bên nhà đất.
    before = len(df)
    df = dedup_exact(df, ["mo_ta", "gia_ban", "dien_tich"],
                     "Trùng: mô tả + giá + diện tích", rep)
    df = dedup_exact(df, ["ten_du_an", "dia_chi", "gia_ban", "dien_tich"],
                     "Trùng: tiêu đề + địa chỉ + giá + diện tích", rep)
    df["_dt_lam_tron"] = df["dien_tich_m2"].round(0)
    df = dedup_near(df, "mo_ta", ["quan_huyen", "phuong_xa", "_dt_lam_tron"],
                    threshold=0.90, report=rep)
    df = df.drop(columns=["_dt_lam_tron"])
    rep.note("Tổng khử trùng lặp", f"{before - len(df)} dòng bị loại")

    # ---------------------------------------------------------------- bước 9
    # Vá các ô trống bằng thông tin trong tiêu đề và mô tả.
    df, n_tang = fill_missing_from_text(df, "tang_so", ["mo_ta", "ten_du_an"], extract_floor)
    rep.note("Vá tầng số từ mô tả", f"điền thêm {n_tang} ô đang trống")
    df = add_binary_flags(df, ["mo_ta", "ten_du_an"])

    # --------------------------------------------------------------- bước 10
    # Tách phân khúc. Model chính chỉ dùng "Chung cư"; phần còn lại xuất riêng.
    la_chinh = df["phan_khuc"].isin(PHAN_KHUC_CHINH)
    khac = df.loc[~la_chinh].copy()
    df = df.loc[la_chinh].copy()
    rep.log("Tách phân khúc khác (tập thể, cư xá, căn hộ mini…)", int(len(khac)), len(df) + len(khac))
    rep.note("Phân khúc khác",
             "; ".join(f"{k}: {v}" for k, v in khac["phan_khuc"].value_counts().items())
             or "không có")

    # --------------------------------------------------------------- bước 11
    # Gắn cờ ngoại lai theo IQR TRONG TỪNG QUẬN (không dùng ngưỡng toàn cục).
    df["ngoai_lai"] = flag_outliers_grouped(df, "gia_tren_m2", ["quan_huyen"], k=3.0)
    # cộng thêm chặn cứng ở khoảng bất khả thi, phòng khi nhóm quá nhỏ để tính IQR
    df["ngoai_lai"] |= ~df["gia_tren_m2"].between(PPM2_LO, PPM2_HI)
    before = len(df)
    n_out = int(df["ngoai_lai"].sum())
    rep.log("Ngoại lai giá/m² (IQR×3 trong quận + chặn cứng)", n_out, before)
    ngoai_lai = df.loc[df["ngoai_lai"]].copy()
    df = df.loc[~df["ngoai_lai"]].drop(columns=["ngoai_lai"]).copy()

    # --------------------------------------------------------------- bước 12
    # Chọn và sắp xếp cột đầu ra.
    keep = [
        "TARGET_gia_vnd", "gia_tren_m2", "dien_tich_m2", "so_phong_ngu", "so_phong_vs",
        "tang_so", "quan_huyen", "phuong_xa", "duong_pho", "dia_chi_goc", "geocode_query",
        "du_an_clean", "ten_du_an_goc", "phan_khuc",
        "Giấy tờ pháp lý", "Tình trạng nội thất", "Hướng ban công", "Hướng cửa chính",
        "Tình trạng bất động sản", "Block/Tháp", "mo_ta", "gia_m2_nguon",
        # Cột định danh: chỉ có nếu sitemap lấy được. Giữ lại để khử trùng lặp
        # giữa các lần cào có khoá tuyệt đối, thay vì phải so nội dung.
        "listing_id", "url_goc", "nguon", "phuong_moi", "du_an_dia_chi",
        # Toạ độ chính xác do nguồn cung cấp (batdongsan bóc từ khối JS). Thiếu
        # hai cột này trong danh sách giữ lại thì công sức cào toạ độ mất trắng
        # ngay ở khâu làm sạch — mà không có lỗi nào báo ra.
        "latitude", "longitude",
    ] + [c for c in df.columns if c.startswith("nhac_")]
    keep = [c for c in keep if c in df.columns]
    clean = df[keep].reset_index(drop=True)

    # ------------------------------------------------------------------ xuất
    clean.to_excel(outdir / f"{tien_to}_clean.xlsx", index=False)
    khac[[c for c in keep if c in khac.columns]].to_excel(
        outdir / f"{tien_to}_phan_khuc_khac.xlsx", index=False)
    ngoai_lai[[c for c in keep if c in ngoai_lai.columns]].to_excel(
        outdir / f"{tien_to}_ngoai_lai.xlsx", index=False)
    rep.to_frame().to_excel(outdir / f"{tien_to}_bao_cao_lam_sach.xlsx", index=False)
    if len(unit_log):
        unit_log.to_excel(outdir / f"{tien_to}_sua_don_vi.xlsx", index=False)

    print(rep.render())
    return {"clean": clean, "khac": khac, "ngoai_lai": ngoai_lai,
            "report": rep, "unit_log": unit_log}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Làm sạch dữ liệu chung cư nhatot.com")
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--outdir", default=Path("data_clean"), type=Path)
    ap.add_argument("--nguon", choices=["nhatot", "batdongsan"], default="nhatot")
    ap.add_argument("--tien-to", default="chungcu",
                    help="tiền tố tên file đầu ra (để không đè lên nhau)")
    args = ap.parse_args()
    run(args.input, args.outdir, args.nguon, args.tien_to)
