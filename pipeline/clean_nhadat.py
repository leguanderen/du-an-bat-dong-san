"""
clean_nhadat.py — Luồng làm sạch dữ liệu NHÀ ĐẤT THỔ CƯ (nguồn: nhatot.com).

Chạy:  python clean_nhadat.py --input nhatot_nhadat.xlsx --outdir data_clean

KHÁC BIỆT SO VỚI LUỒNG CHUNG CƯ
-------------------------------
1) Có cột `click` chứa URL tin gốc kèm ID duy nhất -> có KHOÁ TUYỆT ĐỐI để khử
   trùng lặp. Chung cư không có, phải đoán qua nội dung.
2) Có hai loại diện tích (đất và sử dụng). Giá/m² của nhà đất luôn tính theo
   DIỆN TÍCH ĐẤT — đó là quy ước của thị trường, và cũng là cách nhatot tính.
3) Trường "Đặc điểm" chứa nhiều giá trị ngăn bằng xuống dòng ("Hẻm xe hơi\\nNhà
   nở hậu") -> phải tách thành nhiều cột nhị phân.
4) Mặt bằng giá/m² cao hơn chung cư nhiều lần (trung vị ~220 triệu so với ~66
   triệu), nên ngưỡng phải đặt riêng chứ không dùng chung.

VỀ NGOẠI LAI Ở NHÀ ĐẤT — điểm cần cẩn thận nhất
-----------------------------------------------
Nhà mặt phố Hoàn Kiếm (Hàng Chiếu, Phùng Hưng, Bảo Khánh) có giá/m² thật lên
tới 1,5–1,6 tỷ. Nhà ngõ Hoàng Mai cùng mức giá đó thì gần như chắc chắn là tin
ảo. Cùng một con số nhưng ý nghĩa trái ngược -> KHÔNG được dùng ngưỡng cứng
toàn cục. Ở đây lọc theo IQR trong từng nhóm (quận × loại hình).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from clean_common import (
    BLOB_LABELS_NHADAT, CleanReport, bu_quan_tu_phuong, build_geocode_query,
    cross_check_parse,
    dedup_exact, dedup_near, expand_attr_blob, fix_price_unit_errors,
    flag_outliers_grouped, normalize_space, parse_address, parse_area,
    parse_number, parse_price, parse_price_per_m2, plausible_ppm2_band,
    strip_accents,
)
from extract_text import (
    add_binary_flags, extract_depth, extract_frontage, extract_num_floors,
    extract_road_width, fill_missing_from_text, validate_dimensions,
)

# Khoảng phát hiện thô cho giá/m² nhà đất Hà Nội (VND/m²).
# Rộng có chủ ý — khoảng chấp nhận hẹp hơn sẽ được suy ra từ dữ liệu.
PPM2_LO, PPM2_HI = 5e6, 2000e6

# TRẦN GIÁ — RANH GIỚI PHÂN KHÚC, KHÔNG PHẢI LỌC NGOẠI LAI
# --------------------------------------------------------
# Trên batdongsan, mục "nhà đất bán" trộn hai thị trường hoàn toàn khác nhau.
# Đọc thẳng các tin đắt nhất trong tập 14.856 tin thì thấy rõ:
#
#   1.812 tỷ — "Bán toà nhà mặt phố Thành Thái 7.250 m² 30 tầng văn phòng"
#   1.300 tỷ — "Trung Kính 814 m², 32 tầng, chuyển nhượng siêu dự án building"
#     880 tỷ — "Chuyển nhượng khách sạn 5 sao siêu VIP tại Hà Đông"
#
# Đây là bất động sản THƯƠNG MẠI. Giá của chúng do dòng tiền cho thuê và giấy
# phép xây dựng quyết định, không phải do số phòng ngủ hay độ rộng ngõ — tức
# là do những biến mà hệ thống này không hề có. Giữ chúng lại không làm mô
# hình học được gì về nhà ở, chỉ làm sai số tuyệt đối phình ra.
#
# ĐO CỤ THỂ (đánh giá chéo 5-fold, cùng một khung):
#
#   toàn bộ     n=14.856   MAPE 21,39%   MAE 5,96 tỷ   R² 0,780
#   ≤ 200 tỷ    n=14.712   MAPE 21,09%   MAE 4,53 tỷ   R² 0,876
#   ≤ 100 tỷ    n=14.365   MAPE 20,86%   MAE 3,79 tỷ   R² 0,873   <- chọn
#   ≤  50 tỷ    n=13.185   MAPE 20,26%   MAE 2,81 tỷ   R² 0,842
#
# 491 tin (3,3%) kéo R² xuống gần 0,1 điểm. Cắt ở 100 tỷ lấy được gần hết
# phần cải thiện mà vẫn giữ lại biệt thự, nhà phố lớn — vốn vẫn là nhà ở.
# Cắt sâu hơn (50, 30 tỷ) thì MAPE còn giảm nữa nhưng R² tụt lại, vì lúc đó
# đã cắt vào chính phân khúc cần định giá.
#
# Đây là quyết định về PHẠM VI SẢN PHẨM, phải nói ra với người dùng, không
# phải giấu vào bước làm sạch: hệ thống định giá nhà ở, không định giá bất
# động sản thương mại.
GIA_TRAN_NHA_O = 100e9

# Bốn loại hình chính. Giá của chúng khác nhau rất mạnh nên dùng làm nhóm
# khi lọc ngoại lai.
LOAI_HINH_CHINH = [
    "Nhà ngõ, hẻm", "Nhà mặt phố, mặt tiền", "Nhà phố liền kề", "Nhà biệt thự",
]

# Các giá trị có thể xuất hiện trong trường "Đặc điểm" của nhatot.
DAC_DIEM_VALUES = [
    "Hẻm xe hơi", "Nhà nở hậu", "Nhà tóp hậu", "Nhà nát",
    "Nhà chưa hoàn công", "Nhà dính quy hoạch / lộ giới",
    "Đất chưa chuyển thổ", "Hiện trạng khác",
]


def load_raw(nguon_du_lieu, nguon: str = "nhatot") -> pd.DataFrame:
    """Nạp dữ liệu thô. Nhận đường dẫn hoặc DataFrame sẵn.

    `nguon="batdongsan"` chạy bộ chuyển đổi trước, để phần còn lại của luồng
    làm sạch không cần biết dữ liệu đến từ đâu — giống hệt nhánh chung cư.
    """
    df = (nguon_du_lieu if isinstance(nguon_du_lieu, pd.DataFrame)
          else pd.read_excel(nguon_du_lieu))
    if nguon == "batdongsan":
        from nguon_batdongsan import chuyen_doi
        df = chuyen_doi(df, loai="nhadat")
        # batdongsan không có cột tiêu đề riêng; tên tin nằm ở `ten_du_an`.
        if "tieu_de" not in df.columns and "ten_du_an" in df.columns:
            df["tieu_de"] = df["ten_du_an"]
    required = {"tieu_de", "dia_chi", "gia_ban", "dien_tich", "dac_diem_tong_hop", "mo_ta"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"File thiếu cột bắt buộc: {sorted(missing)}")
    if "nguon" not in df.columns:
        df["nguon"] = nguon
    if "click" in df.columns and "listing_id" not in df.columns:
        df["url_goc"] = df["click"]
        df["listing_id"] = (df["click"].astype(str)
                            .str.extract(r"/(\d+)\.htm")[0])
    return df


def split_dac_diem(df: pd.DataFrame) -> pd.DataFrame:
    """Tách trường "Đặc điểm" nhiều giá trị thành các cột nhị phân dd_*."""
    out = df.copy()
    raw = out.get("Đặc điểm", pd.Series("", index=out.index)).fillna("").astype(str)
    flat = raw.map(strip_accents)
    for value in DAC_DIEM_VALUES:
        out[f"dd_{value}"] = flat.str.contains(strip_accents(value), regex=False).astype(int)
    return out


def run(input_path, outdir: Path, nguon: str = "nhatot",
        tien_to: str = "nhadat") -> dict:
    outdir.mkdir(parents=True, exist_ok=True)
    df = load_raw(input_path, nguon)
    rep = CleanReport(f"NHÀ ĐẤT THỔ CƯ ({nguon})", so_dong_ban_dau=len(df))

    # ---------------------------------------------------------------- bước 1
    df = expand_attr_blob(df, "dac_diem_tong_hop", BLOB_LABELS_NHADAT)

    # ---------------------------------------------------------------- bước 2
    df["TARGET_gia_vnd"] = df["gia_ban"].map(parse_price)
    df["dien_tich_dat_m2"] = df["dien_tich"].map(parse_area)
    df["dien_tich_su_dung_m2"] = df.get("Diện tích sử dụng", pd.Series(dtype=object)).map(parse_area)
    df["so_phong_ngu"] = df.get("Số phòng ngủ", pd.Series(dtype=object)).map(parse_number)
    df["so_phong_vs"] = df.get("Số phòng vệ sinh", pd.Series(dtype=object)).map(parse_number)
    # Hai sàn gọi cùng một thứ bằng hai tên: nhatot ghi "Tổng số tầng",
    # batdongsan ghi "Tầng số". Lấy cả hai, nhatot trước.
    #
    # Trước đây chỉ lấy "Tổng số tầng", nên với batdongsan số tầng phải moi từ
    # mô tả bằng `extract_num_floors` ở bước 4. Đo lại ngày 12/09/2026 trên
    # 4.106 tin có CẢ hai nguồn: cách moi từ mô tả lệch so với trường thật
    # 336 tin (8,2%). Trường của sàn đáng tin hơn nên phải ưu tiên, còn cách
    # moi từ mô tả chỉ là lưới đỡ cho tin không khai.
    # CHÚ Ý CÁI BẪY Ở ĐÂY, tôi đã sập đúng một lần:
    # `df.get("X", pd.Series(dtype=object))` khi thiếu cột trả về Series RỖNG
    # (độ dài 0), không phải Series toàn NaN theo index của df. Gán thẳng thì
    # vô hại — cột thành toàn NaN, đúng ý. Nhưng `.fillna(cột_khác)` trên một
    # Series rỗng vẫn trả về rỗng: fillna không nới index ra. Nên bản đầu của
    # đoạn này im lặng xoá sạch số tầng thay vì lấy từ "Tầng số".
    trong = pd.Series(np.nan, index=df.index)
    tang = (df["Tổng số tầng"].map(parse_number)
            if "Tổng số tầng" in df.columns else trong.copy())
    if "Tầng số" in df.columns:
        tang = tang.fillna(df["Tầng số"].map(parse_number))
    df["tong_so_tang"] = tang
    df["mat_tien_m"] = df.get("Chiều ngang", pd.Series(dtype=object)).map(parse_number)
    df["chieu_dai_m"] = df.get("Chiều dài", pd.Series(dtype=object)).map(parse_number)
    df["gia_m2_nguon"] = df.get("Giá/m²", pd.Series(dtype=object)).map(parse_price_per_m2)
    df["loai_hinh"] = df.get("Loại hình", pd.Series(dtype=object)).fillna("Không rõ")
    # "Đường vào" của batdongsan là độ rộng đường/ngõ trước nhà — cùng đại lượng
    # với `duong_rong_m` mà nhánh nhatot phải moi từ mô tả.
    if "Đường vào" in df.columns:
        df["duong_rong_m"] = df["Đường vào"].map(parse_number)

    # ---------------------------------------------------------------- bước 3
    rel = cross_check_parse(df["TARGET_gia_vnd"], df["dien_tich_dat_m2"], df["gia_m2_nguon"])
    n_lech = int((rel > 0.01).sum())
    rep.note("Kiểm chứng parser",
             f"đối chiếu {int(rel.notna().sum())} dòng với Giá/m² do nguồn công bố; "
             f"{n_lech} dòng lệch >1%")
    if n_lech > len(df) * 0.01:
        raise RuntimeError(f"Parser sai: {n_lech} dòng lệch quá 1%. Kiểm tra lại parser.")

    # ---------------------------------------------------------------- bước 4
    before = len(df)
    mask = (df["TARGET_gia_vnd"].isna() | df["dien_tich_dat_m2"].isna()
            | (df["dien_tich_dat_m2"] <= 0))
    rep.log("Thiếu giá hoặc diện tích đất", int(mask.sum()), before)
    df = df.loc[~mask].copy()

    # ---------------------------------------------------------------- bước 5
    addr = df["dia_chi"].map(parse_address).apply(pd.Series)
    df = pd.concat([df, addr], axis=1)
    df["dia_chi_goc"] = df["dia_chi"].map(normalize_space)
    df["geocode_query"] = df.apply(build_geocode_query, axis=1)

    # Bù quận từ tên phường TRƯỚC khi lọc. Địa chỉ batdongsan sau sáp nhập
    # không còn cấp quận, nên nếu lọc thẳng thì mất sạch dữ liệu.
    df, n_bu = bu_quan_tu_phuong(df)
    if n_bu:
        rep.note("Bù quận/huyện từ tên phường",
                 f"điền {n_bu} ô trống (địa chỉ sau sáp nhập không còn cấp quận)")

    before = len(df)
    mask = df["quan_huyen"].isna()
    rep.log("Không xác định được quận/huyện", int(mask.sum()), before)
    df = df.loc[~mask].copy()

    # ---------------------------------------------------------------- bước 6
    # Rút ID tin từ URL. Đây là khoá khử trùng lặp CHẮC CHẮN NHẤT có thể có —
    # hai dòng cùng ID thì chắc chắn là cùng một tin đăng.
    # LƯU Ý: adapter của batdongsan (nguon_batdongsan.chuyen_doi) đã tự đặt
    # `url_goc` + `listing_id` và bỏ cột `click`. Nhánh else ở đây TRƯỚC ĐÂY xoá
    # trắng cả hai cột đó -> mọi tin batdongsan mất link "tin gốc". Chỉ tạo cột
    # khi nó thực sự chưa có.
    if "click" in df.columns and "listing_id" not in df.columns:
        df["listing_id"] = df["click"].astype(str).str.extract(r"/(\d+)\.htm")[0]
        df["url_goc"] = df["click"]
        rep.note("Rút ID tin từ URL",
                 f"lấy được {int(df['listing_id'].notna().sum())}/{len(df)} ID")
    elif "listing_id" in df.columns:
        if "url_goc" not in df.columns:
            df["url_goc"] = np.nan
        rep.note("Rút ID tin từ URL",
                 f"nguồn đã có sẵn ID: {int(df['listing_id'].notna().sum())}/{len(df)} ID, "
                 f"{int(df['url_goc'].notna().sum())}/{len(df)} URL")
    else:
        df["listing_id"] = np.nan
        df["url_goc"] = np.nan
        rep.note("Rút ID tin từ URL", "KHÔNG có cột URL trong dữ liệu — bỏ qua tầng 1")

    # ---------------------------------------------------------------- bước 7
    df = split_dac_diem(df)

    # Khoảng chấp nhận suy từ dữ liệu, không đặt tay.
    acc_lo, acc_hi = plausible_ppm2_band(
        df["TARGET_gia_vnd"], df["dien_tich_dat_m2"], PPM2_LO, PPM2_HI)
    rep.note("Khoảng giá/m² chấp nhận (suy từ dữ liệu)",
             f"{acc_lo/1e6:.1f} – {acc_hi/1e6:.1f} triệu/m²")
    df, unit_log = fix_price_unit_errors(
        df, price_col="TARGET_gia_vnd", area_col="dien_tich_dat_m2",
        lo_ppm2=PPM2_LO, hi_ppm2=PPM2_HI, accept_lo=acc_lo, accept_hi=acc_hi,
    )
    if len(unit_log):
        unit_log = unit_log.join(df[["tieu_de", "quan_huyen", "loai_hinh"]], how="left")
    rep.note("Sửa lỗi đơn vị giá",
             f"sửa {len(unit_log)} dòng" if len(unit_log) else "không có dòng nào cần sửa")

    df["gia_tren_m2"] = df["TARGET_gia_vnd"] / df["dien_tich_dat_m2"]

    # ---------------------------------------------------------------- bước 8
    # Khử trùng lặp ba tầng, từ chắc chắn nhất đến suy đoán nhất.
    before = len(df)
    if df["listing_id"].notna().any():
        df = dedup_exact(df, ["listing_id"], "Trùng ID tin (khoá tuyệt đối)", rep)
    df = dedup_exact(df, ["mo_ta", "gia_ban", "dien_tich"],
                     "Trùng: mô tả + giá + diện tích", rep)
    df["_dt_lam_tron"] = df["dien_tich_dat_m2"].round(0)
    df = dedup_near(df, "mo_ta", ["quan_huyen", "phuong_xa", "_dt_lam_tron"],
                    threshold=0.90, report=rep)
    df = df.drop(columns=["_dt_lam_tron"])
    rep.note("Tổng khử trùng lặp", f"{before - len(df)} dòng bị loại")

    # ---------------------------------------------------------------- bước 9
    # Vá các cột thiếu nặng bằng thông tin trong mô tả và tiêu đề.
    text_cols = ["mo_ta", "tieu_de"]
    for col, fn, ten in [
        ("tong_so_tang", extract_num_floors, "tổng số tầng"),
        ("mat_tien_m", extract_frontage, "mặt tiền"),
        ("chieu_dai_m", extract_depth, "chiều dài"),
        ("duong_rong_m", extract_road_width, "độ rộng đường/ngõ"),
    ]:
        truoc = int(df[col].notna().sum()) if col in df.columns else 0
        df, n = fill_missing_from_text(df, col, text_cols, fn)
        sau = int(df[col].notna().sum())
        rep.note(f"Vá {ten} từ mô tả",
                 f"{truoc} -> {sau} ô có dữ liệu (+{n}, độ phủ {sau/len(df)*100:.1f}%)")

    # Kiểm tra kích thước có nhất quán với diện tích không. Bước này bắt được
    # các giá trị đọc nhầm từ mẫu "45m2 x 5 tầng" (diện tích × số tầng).
    df, dim_stats = validate_dimensions(df, "dien_tich_dat_m2")
    rep.note("Kiểm tra hình học kích thước",
             f"xoá {dim_stats['mat_tien_bi_xoa']} mặt tiền và "
             f"{dim_stats['chieu_dai_bi_xoa']} chiều dài không nhất quán với diện tích")

    df = add_binary_flags(df, text_cols)
    # Ba cờ nghiệp vụ mà model cũ dùng, suy từ văn bản vì nhatot không có ô riêng.
    df["co_oto"] = df["nhac_o_to"]
    df["kinh_doanh"] = df["nhac_kinh_doanh"]
    df["thang_may"] = df["nhac_thang_may"]
    df["lo_goc"] = df["nhac_lo_goc"]

    # --------------------------------------------------------------- bước 10
    # Ngoại lai theo IQR trong nhóm (quận × loại hình).
    df["_nhom"] = df["quan_huyen"].astype(str) + " | " + df["loai_hinh"].astype(str)
    df["ngoai_lai"] = flag_outliers_grouped(df, "gia_tren_m2", ["_nhom"], k=3.0, min_group=15)
    df["ngoai_lai"] |= ~df["gia_tren_m2"].between(PPM2_LO, PPM2_HI)
    before = len(df)
    rep.log("Ngoại lai giá/m² (IQR×3 trong quận×loại hình + chặn cứng)",
            int(df["ngoai_lai"].sum()), before)

    # Ranh giới phân khúc — xem GIA_TRAN_NHA_O ở đầu file. Ghi thành một dòng
    # RIÊNG trong báo cáo, không gộp vào "ngoại lai": đây không phải tin hỏng,
    # mà là tin thuộc thị trường khác.
    thuong_mai = df["TARGET_gia_vnd"] > GIA_TRAN_NHA_O
    if thuong_mai.any():
        rep.log(f"Ngoài phân khúc nhà ở (> {GIA_TRAN_NHA_O/1e9:.0f} tỷ — toà "
                f"văn phòng, khách sạn, chuyển nhượng dự án)",
                int(thuong_mai.sum()), len(df))
        df["ngoai_lai"] |= thuong_mai
    ngoai_lai = df.loc[df["ngoai_lai"]].copy()
    df = df.loc[~df["ngoai_lai"]].drop(columns=["ngoai_lai", "_nhom"]).copy()

    # --------------------------------------------------------------- bước 11
    keep = [
        "TARGET_gia_vnd", "gia_tren_m2", "dien_tich_dat_m2", "dien_tich_su_dung_m2",
        "so_phong_ngu", "so_phong_vs", "tong_so_tang", "mat_tien_m", "chieu_dai_m",
        "duong_rong_m", "co_oto", "kinh_doanh", "thang_may", "lo_goc",
        "loai_hinh", "quan_huyen", "phuong_xa", "duong_pho", "dia_chi_goc",
        "geocode_query", "listing_id", "url_goc", "tieu_de", "mo_ta",
        "Giấy tờ pháp lý", "Tình trạng nội thất", "Hướng cửa chính", "gia_m2_nguon",
        # TOẠ ĐỘ DO NGUỒN CÔNG BỐ — phải giữ lại.
        #
        # Đây là lần thứ hai đúng lỗi này xảy ra: nhánh chung cư cũng từng lọc
        # mất latitude/longitude ở bước chọn cột cuối cùng. Không có lỗi nào
        # báo ra, chỉ là toạ độ chính xác 100% của cả đợt cào biến mất và bước
        # sau lặng lẽ ước lượng lại từ tên đường.
        "latitude", "longitude", "nguon", "phuong_moi",
    ] + [c for c in df.columns if c.startswith(("dd_", "nhac_"))]
    keep = [c for c in keep if c in df.columns]
    clean = df[keep].reset_index(drop=True)

    clean.to_excel(outdir / f"{tien_to}_clean.xlsx", index=False)
    ngoai_lai[[c for c in keep if c in ngoai_lai.columns]].to_excel(
        outdir / "nhadat_ngoai_lai.xlsx", index=False)
    rep.to_frame().to_excel(outdir / "nhadat_bao_cao_lam_sach.xlsx", index=False)
    if len(unit_log):
        unit_log.to_excel(outdir / "nhadat_sua_don_vi.xlsx", index=False)

    print(rep.render())
    return {"clean": clean, "ngoai_lai": ngoai_lai, "report": rep, "unit_log": unit_log}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Làm sạch dữ liệu nhà đất nhatot.com")
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--outdir", default=Path("data_clean"), type=Path)
    args = ap.parse_args()
    run(args.input, args.outdir)
