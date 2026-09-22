"""
gazetteer_phuong.py — Bảng tra đơn vị hành chính CŨ ↔ MỚI sau sáp nhập 1/7/2025.

    python gazetteer_phuong.py            # dựng bảng, xuất ra data_clean
    python gazetteer_phuong.py --ap-dung  # gắn phuong_moi vào mọi tệp sạch

VÌ SAO CẦN MỘT BẢNG TRA CHÍNH THỨC
----------------------------------
Bốn nguồn dữ liệu của dự án nói bốn thứ tiếng về địa chỉ:

  * nhatot (extension, tháng 6)  : chỉ có tên phường CŨ
  * nhatot (API)                 : có cả cũ (`ward_name`) và mới (`ward_name_v3`)
  * batdongsan (extension)       : chỉ có tên phường MỚI, kèm tên quận CŨ
  * dữ liệu cũ đã làm sạch       : tên phường CŨ

Không quy về một hệ thì không gộp được, không vẽ bản đồ theo đơn vị hành chính
được, và target encoding thì chia đôi dữ liệu của cùng một nơi thành hai hạng
mục chỉ vì nó được gọi bằng hai cái tên.

CHỌN TÊN MỚI LÀM CHUẨN, VÌ BA LÝ DO ĐO ĐƯỢC
-------------------------------------------
1. Ánh xạ cũ → mới là MỘT HÀM: 346 phường cũ ứng với 113 phường mới, và **không
   một phường cũ nào bị tách ra làm hai**. Nên đổi cũ sang mới không mất thông
   tin. Chiều ngược lại thì mất — một phường mới gộp 3–5 phường cũ, không suy
   ngược ra được.

2. Mã hoá theo phường mới cho kết quả TỐT HƠN dù ít hạng mục hơn: chung cư
   16,00% so với 16,01%, nhà đất 27,24% so với 27,75%. Gộp lại thì mỗi phường
   nhiều mẫu hơn, target encoding bớt nhiễu.

3. Đây là hệ hành chính đang có hiệu lực. OpenStreetMap đã chuyển sang tên mới
   (chính vì vậy mà geocode bằng tên cũ mới sai 17,6 km).

CẤP QUẬN/HUYỆN THÌ SAO
----------------------
Cải cách 1/7/2025 BỎ hẳn cấp quận/huyện: Hà Nội giờ là thành phố → phường/xã,
không còn cấp trung gian. Nhưng module này vẫn giữ `quan_cu` vì hai lý do thực
dụng: người mua nhà vẫn nghĩ theo quận ("nhà Cầu Giấy"), và 113 phường thả
thẳng vào một ô chọn thì không ai tìm nổi. Giữ làm tầng NHÓM cho giao diện,
nhãn ghi rõ là "quận (cũ)", không trình bày như đơn vị hành chính hiện hành.

BA MỨC TRA, GHI RÕ MỨC NÀO
--------------------------
  0. `da_co`     — cột đã có sẵn từ trước (nguồn công bố, hoặc lần chạy trước).
  1. `tra_bang`  — khớp (quận cũ, phường cũ) trong bảng. Chính xác tuyệt đối.
  2. `gan_nhat`  — không khớp tên thì lấy phường mới có tâm GẦN NHẤT với toạ độ
                   của tin. Dùng được vì mọi dòng đều đã có toạ độ.
  3. `khong_ro`  — không có cả tên lẫn toạ độ.

Cột `nguon_phuong_moi` ghi rõ từng dòng thuộc mức nào, để ai đọc kết quả cũng
biết chỗ nào là tra được và chỗ nào là suy ra.
"""

from __future__ import annotations

import argparse
import re

import numpy as np
import pandas as pd

from gan_toa_do import (DATA_CLEAN, chuan_phuong, chuan_quan, haversine_km,
                        nap_api)

# Quận là đô thị: chỉ chứa Phường. Xã và Thị trấn chỉ thuộc Huyện; Thị xã thì
# có cả hai. Quy tắc này bắt được lỗi bóc địa chỉ nhận nhầm tên PHỐ thành xã —
# "Xã Đàn" ở Đống Đa là một con phố, không phải đơn vị hành chính.
_RE_XA = re.compile(r"^(xã|thị trấn)\b", re.I)


def sai_cap_hanh_chinh(quan: object, phuong: object) -> bool:
    """True nếu cặp quận/phường vi phạm quy tắc hành chính."""
    if not isinstance(quan, str) or not isinstance(phuong, str):
        return False
    return quan.strip().lower().startswith("quận") and bool(_RE_XA.match(phuong.strip()))


# =============================================================================
# DỰNG BẢNG
# =============================================================================

def dung_bang(api: pd.DataFrame | None = None) -> pd.DataFrame:
    """Bảng tra (quận cũ, phường cũ) -> phường mới, kèm tâm và số tin làm chứng."""
    api = nap_api() if api is None else api
    d = api.dropna(subset=["ward_name", "ward_name_v3", "area_name"]).copy()
    d["k_quan"] = d["area_name"].map(chuan_quan)
    d["k_phuong"] = d["ward_name"].map(chuan_phuong)

    g = d.groupby(["k_quan", "k_phuong"])
    bang = pd.DataFrame({
        "quan_cu": g["area_name"].agg(lambda s: s.mode().iat[0]),
        "phuong_cu": g["ward_name"].agg(lambda s: s.mode().iat[0]),
        "phuong_moi": g["ward_name_v3"].agg(lambda s: s.mode().iat[0]),
        "so_ten_moi": g["ward_name_v3"].nunique(),
        "so_tin_lam_chung": g.size(),
        "lat": g["latitude"].median(),
        "lon": g["longitude"].median(),
    }).reset_index()

    nhap_nhang = int((bang["so_ten_moi"] > 1).sum())
    if nhap_nhang:
        print(f"  ⚠ {nhap_nhang} phường cũ ứng với nhiều hơn một phường mới — "
              "đã lấy tên xuất hiện nhiều nhất, nhưng nên soi lại bằng tay.")
    return bang


def tam_phuong_moi(bang: pd.DataFrame) -> pd.DataFrame:
    """Tâm của từng phường MỚI, tính bằng trung vị có trọng số theo số tin."""
    g = bang.groupby("phuong_moi")
    return pd.DataFrame({
        "lat": g.apply(lambda s: np.average(s["lat"], weights=s["so_tin_lam_chung"]),
                       include_groups=False),
        "lon": g.apply(lambda s: np.average(s["lon"], weights=s["so_tin_lam_chung"]),
                       include_groups=False),
        "so_phuong_cu_gop_vao": g.size(),
        "so_tin_lam_chung": g["so_tin_lam_chung"].sum(),
    }).reset_index()


# =============================================================================
# ÁP DỤNG
# =============================================================================

def gan_phuong_moi(df: pd.DataFrame, bang: pd.DataFrame | None = None,
                   tam: pd.DataFrame | None = None) -> pd.DataFrame:
    """Thêm `phuong_moi` và `nguon_phuong_moi` vào một bảng dữ liệu sạch.

    Giữ nguyên giá trị `phuong_moi` mà nguồn đã tự công bố (batdongsan) — nguồn
    biết rõ hơn bảng tra suy ra.
    """
    bang = dung_bang() if bang is None else bang
    tam = tam_phuong_moi(bang) if tam is None else tam
    out = df.copy()

    # --- vá lỗi bóc địa chỉ: "Xã ..." không thể nằm trong một Quận ---
    if {"quan_huyen", "phuong_xa"} <= set(out.columns):
        sai = out.apply(lambda r: sai_cap_hanh_chinh(r["quan_huyen"],
                                                     r["phuong_xa"]), axis=1)
        if sai.any():
            print(f"  vá {int(sai.sum())} dòng có 'Xã ...' nằm trong một Quận "
                  "(tên phố bị nhận nhầm thành đơn vị hành chính)")
            out.loc[sai, "phuong_xa"] = np.nan

    if "phuong_moi" not in out.columns:
        out["phuong_moi"] = np.nan

    # GIÁ TRỊ CÓ SẴN CHỈ ĐƯỢC GIỮ NẾU NÓ THẬT SỰ LÀ MỘT PHƯỜNG MỚI.
    #
    # Bản đầu chỉ kiểm `notna()` rồi gắn nhãn "da_co" và bỏ qua. Nhưng
    # `gan_toa_do` cũng ghi vào cột này và khi không tra được thì nó chép
    # nguyên tên phường CŨ sang. Kết quả: 29% số tin nhà đất mang tên trước sáp
    # nhập — "Phường Nhân Chính", "Phường Dịch Vọng", "Thị trấn Trâu Quỳ" —
    # và bảng chuẩn hoá lặng lẽ bỏ qua chúng vì thấy ô đã có chữ.
    #
    # Hậu quả không có lỗi nào báo ra: đếm ra 207 phường trong khi Hà Nội chỉ
    # có 113. Mã hoá mục bị chia vụn, và bản đồ theo phường mất một phần dữ
    # liệu vì tên không khớp ranh giới nào.
    #
    # Dấu hiệu duy nhất nhận ra được là con số phường vượt quá 113 — nên từ nay
    # đối chiếu thẳng với danh sách chuẩn thay vì tin vào việc ô đã có chữ.
    hop_le = set(bang["phuong_moi"].dropna())
    khong_chuan = out["phuong_moi"].notna() & ~out["phuong_moi"].isin(hop_le)
    if khong_chuan.any():
        print(f"  {int(khong_chuan.sum())} dòng có `phuong_moi` không thuộc "
              f"{len(hop_le)} phường chuẩn (tên trước sáp nhập) — tra lại")
        out.loc[khong_chuan, "phuong_moi"] = np.nan

    out["nguon_phuong_moi"] = np.where(out["phuong_moi"].notna(), "da_co", None)

    # --- mức 1: tra bảng theo tên cũ ---
    tra = bang.set_index(["k_quan", "k_phuong"])["phuong_moi"]
    kq = out.assign(_q=out["quan_huyen"].map(chuan_quan),
                    _p=out["phuong_xa"].map(chuan_phuong))
    khop = pd.MultiIndex.from_frame(kq[["_q", "_p"]]).map(tra)
    dien = out["phuong_moi"].isna() & pd.notna(khop)
    out.loc[dien, "phuong_moi"] = pd.Series(khop, index=out.index)[dien]
    out.loc[dien, "nguon_phuong_moi"] = "tra_bang"

    # --- mức 2: phường mới có tâm gần nhất ---
    con = out["phuong_moi"].isna() & out.get(
        "latitude", pd.Series(np.nan, index=out.index)).notna()
    if con.any():
        lat = out.loc[con, "latitude"].to_numpy()[:, None]
        lon = out.loc[con, "longitude"].to_numpy()[:, None]
        d = haversine_km(lat, lon, tam["lat"].to_numpy(), tam["lon"].to_numpy())
        gan = tam["phuong_moi"].to_numpy()[d.argmin(axis=1)]
        out.loc[con, "phuong_moi"] = gan
        out.loc[con, "nguon_phuong_moi"] = "gan_nhat"
        out.loc[con, "cach_tam_phuong_moi_km"] = d.min(axis=1).round(2)

    out["nguon_phuong_moi"] = out["nguon_phuong_moi"].fillna("khong_ro")
    return out


# =============================================================================
# CHẠY
# =============================================================================

# Phải liệt kê ĐỦ mọi tệp mà bản chạy thật dùng. Thiếu một tệp ở đây thì tệp
# đó giữ nguyên tên phường thô của nguồn, và `phuong_moi` của nó không còn nằm
# trong 113 phường chuẩn — mã hoá mục bị chia vụn, bản đồ mất phường. Không có
# lỗi nào báo ra; dấu hiệu duy nhất là số phường đếm được vượt quá 113.
TEP = {"chungcu": "chungcu_clean_geo.xlsx", "nhadat": "nhadat_clean_geo.xlsx",
       "chungcu_gop": "chungcu_gop_geo.xlsx",
       "nhadat_gop": "nhadat_gop_geo.xlsx"}


def main() -> None:
    ap = argparse.ArgumentParser(description="Bảng tra phường cũ ↔ mới")
    ap.add_argument("--ap-dung", action="store_true",
                    help="gắn phuong_moi vào các tệp trong data_clean")
    a = ap.parse_args()

    bang = dung_bang()
    tam = tam_phuong_moi(bang)
    print(f"\nBảng tra: {len(bang)} phường cũ -> {len(tam)} phường mới")
    print(f"  gộp nhiều nhất: {tam['so_phuong_cu_gop_vao'].max()} phường cũ vào một")
    with pd.ExcelWriter(DATA_CLEAN / "bang_tra_phuong.xlsx") as w:
        bang.to_excel(w, sheet_name="cu_sang_moi", index=False)
        tam.to_excel(w, sheet_name="tam_phuong_moi", index=False)
    print("-> data_clean/bang_tra_phuong.xlsx")

    if not a.ap_dung:
        return
    for ten, tep in TEP.items():
        p = DATA_CLEAN / tep
        if not p.exists():
            continue
        print(f"\n--- {tep} ---")
        d = gan_phuong_moi(pd.read_excel(p), bang, tam)
        print(d["nguon_phuong_moi"].value_counts().to_string())
        print(f"  có phuong_moi: {d['phuong_moi'].notna().sum()}/{len(d)} "
              f"({d['phuong_moi'].notna().mean():.1%})")
        d.to_excel(p, index=False)


if __name__ == "__main__":
    main()
