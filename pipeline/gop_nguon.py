"""
gop_nguon.py — Gộp nhiều đợt cào, nhiều sàn thành một tập huấn luyện.

    python gop_nguon.py --do-luong          # gộp xong đo luôn MAPE trước/sau

VÌ SAO PHẢI CÓ CỘT `nguon`
--------------------------
Đo trên chính dữ liệu này: cùng một quận, cùng một ngày, **giá/m² trung vị của
batdongsan cao hơn nhatot khoảng 37%**. Có quận chênh tới hơn hai lần.

Trộn phẳng hai sàn vào nhau thì model học một mặt bằng giá lai — không đúng với
sàn nào cả, và tệ hơn: tỉ lệ tin của mỗi sàn khác nhau giữa các quận, nên phần
chênh lệch đó bị model gán nhầm thành "quận này đắt hơn". Đó là biến gây nhiễu
kinh điển (confounder).

Cách xử lý ở đây là cách rẻ và trung thực nhất: giữ nguyên cả hai, thêm cột
`nguon` làm đặc trưng. Model tự học được "tin batdongsan thường rao cao hơn
chừng này", và phần còn lại của các hệ số không bị méo theo. Khi định giá cho
người dùng thì đặt `nguon` về một giá trị cố định, coi như hỏi "nếu tin này
đăng trên sàn X thì giá bao nhiêu".

MỘT CẢNH BÁO KHÔNG ĐƯỢC QUÊN
----------------------------
Hai file cào ngày 25/8 lấy từ **trang 48–50** (nhatot) và **trang 28–30**
(batdongsan) của danh sách, không phải mẫu ngẫu nhiên. Chênh lệch đo được vì
thế trộn lẫn ba thứ: khác sàn, khác thời điểm, và khác vị trí trang. Không tách
được ba thứ này nếu chỉ có từng đây dữ liệu.

Nên: dùng cột `nguon` để model khỏi bị đánh lừa thì được, nhưng ĐỪNG viết trong
khoá luận rằng "giá batdongsan cao hơn nhatot 37%" như một kết luận về thị
trường. Muốn kết luận được thì phải cào trang 1–3 của cả hai sàn trong cùng một
ngày rồi so lại.

KHỬ TRÙNG LẶP BA TẦNG
---------------------
1. Trùng mã tin — chắc chắn nhất, nhưng chỉ có nếu sitemap lấy được URL.
2. Trùng khoá nghiệp vụ (giá + diện tích + phường + 120 ký tự đầu mô tả).
3. Trùng gần đúng bằng cosine TF-IDF trong cùng phường + cùng diện tích làm tròn.

Tầng 2 và 3 quan trọng vì cùng một căn hộ đăng trên hai sàn thì mã tin khác
nhau hoàn toàn — chỉ nội dung mới tố cáo được.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from clean_common import dedup_near
from gan_toa_do import thu_muc

DATA_RAW = thu_muc("data_raw")
DATA_CLEAN = thu_muc("data_clean")

# Mỗi đợt: (tên file thô, nguồn, ngày cào). Thêm đợt mới chỉ cần thêm một dòng.
DOT_CHUNGCU = [
    ("nhatot_chungcu_raw.xlsx", "nhatot", "2026-06-13"),
    ("nhatot_chungcu_moi.xlsx", "nhatot", "2026-08-25"),
    ("batdongsan_chungcu1.xlsx", "batdongsan", "2026-08-25"),
    ("batdongsan_chungcu_p1_100.xlsx", "batdongsan", "2026-08-26"),
    ("batdongsan_chungcu_p100_200.xlsx", "batdongsan", "2026-08-27"),
    # Ba đợt cào NHẮM ĐÍCH, không cào dàn trải. Đường cong học tập của chung cư
    # cho thấy cào đều tay cần ~12.200 tin mới giảm được 1 điểm MAPE — không
    # đáng. Nhưng ba quận này gần như trống trên bản đồ (Hoàn Kiếm 20 tin, Ba
    # Đình 119, Hai Bà Trưng 140), và người dùng chọn đúng vào đó thì nhận một
    # con số dựng trên gần như không có gì. Đây là vá ĐỘ PHỦ, không phải vá độ
    # chính xác — hai mục tiêu khác nhau, đo bằng hai thước khác nhau.
    ("batdongsan_chungcu_hai_ba_trung.xlsx", "batdongsan", "2026-09-04"),
    ("batdongsan_chungcu_ba_dinh.xlsx", "batdongsan", "2026-09-04"),
    ("batdongsan_chungcu_hoan_kiem.xlsx", "batdongsan", "2026-09-04"),
]


DOT_NHADAT = [
    ("nhatot_nhadat_raw.xlsx", "nhatot", "2026-06-18"),
    ("batdongsan_nhadat_p1_90.xlsx", "batdongsan", "2026-08-30"),
    ("batdongsan_nhadat_p91_350.xlsx", "batdongsan", "2026-08-31"),
    ("batdongsan_nhadat_p351_650.xlsx", "batdongsan", "2026-09-01"),
    ("batdongsan_nhadat_p651_850.xlsx", "batdongsan", "2026-09-04"),
]


def _lam_sach_mot_dot(tep: str, nguon: str, ngay: str, tam: Path,
                      loai: str = "chungcu") -> pd.DataFrame:
    if loai == "nhadat":
        from clean_nhadat import run
    else:
        from clean_chungcu import run
    duong_dan = DATA_RAW / tep
    if not duong_dan.exists():
        print(f"  (bỏ qua {tep} — không tìm thấy)")
        return pd.DataFrame()
    print(f"\n--- {tep} ({nguon}, {ngay}) ---")
    kq = run(duong_dan, tam, nguon=nguon, tien_to=f"_tam_{Path(tep).stem}")
    d = kq["clean"].copy()
    d["nguon"] = nguon
    d["ngay_quan_sat"] = pd.Timestamp(ngay)
    return d


def _khoa_nghiep_vu(d: pd.DataFrame) -> pd.Series:
    """Khoá nhận dạng một tin không phụ thuộc mã tin của sàn.

    Dùng cột diện tích của đúng nhánh: chung cư là diện tích sàn, nhà đất là
    diện tích đất. Lấy nhầm cột thì khoá thành chuỗi "nan" cho toàn bộ dữ liệu
    và bước khử trùng lặp gộp nhầm những tin chẳng liên quan gì nhau.
    """
    cot_dt = ("dien_tich_m2" if "dien_tich_m2" in d.columns
              else "dien_tich_dat_m2")
    return (d["TARGET_gia_vnd"].round(-6).astype(str) + "|"
            + d[cot_dt].round(1).astype(str) + "|"
            + d["phuong_xa"].astype(str) + "|"
            + d["mo_ta"].astype(str).str.slice(0, 120))


def gop(dot: list[tuple[str, str, str]] = None,
        tam: Path = None,
        loai: str = "chungcu") -> tuple[pd.DataFrame, pd.DataFrame]:
    dot = dot or (DOT_NHADAT if loai == "nhadat" else DOT_CHUNGCU)
    tam = tam or (DATA_CLEAN / "_tam_gop")
    tam.mkdir(parents=True, exist_ok=True)

    phan = [_lam_sach_mot_dot(*d, tam, loai) for d in dot]
    phan = [p for p in phan if len(p)]
    d = pd.concat(phan, ignore_index=True)
    truoc = len(d)

    hang = [{"buoc": "Gộp thô", "so_dong": truoc, "loai": 0}]

    # ---- tầng 1: mã tin ----
    if "listing_id" in d.columns:
        co = d["listing_id"].notna()
        n = int(d.loc[co].duplicated(subset="listing_id").sum())
        d = d.loc[~(co & d.duplicated(subset="listing_id"))].copy()
        hang.append({"buoc": "Trùng mã tin", "so_dong": len(d), "loai": n})

    # ---- tầng 2: khoá nghiệp vụ ----
    n0 = len(d)
    d = d.loc[~_khoa_nghiep_vu(d).duplicated()].copy()
    hang.append({"buoc": "Trùng khoá nghiệp vụ", "so_dong": len(d),
                 "loai": n0 - len(d)})

    # ---- tầng 3: gần đúng ----
    n0 = len(d)
    # Cùng lý do như `_khoa_nghiep_vu`: chọn cột diện tích theo nhánh.
    _c_dt = "dien_tich_m2" if "dien_tich_m2" in d.columns else "dien_tich_dat_m2"
    d["_dt"] = d[_c_dt].round(0)
    d = dedup_near(d, "mo_ta", ["quan_huyen", "phuong_xa", "_dt"], threshold=0.90)
    d = d.drop(columns="_dt")
    hang.append({"buoc": "Trùng gần đúng (cosine ≥ 0.90)", "so_dong": len(d),
                 "loai": n0 - len(d)})

    bao_cao = pd.DataFrame(hang)
    bao_cao["ti_le_loai_%"] = (bao_cao["loai"] / truoc * 100).round(2)
    return d.reset_index(drop=True), bao_cao


def do_do_sau_trang(tep_tho: Path) -> pd.DataFrame:
    """Đo xem vị trí TRANG trong danh sách có liên quan tới giá không.

    VÌ SAO PHẢI ĐO
    --------------
    Cào 3 trang cuối và cào 50 trang đầu cho ra hai mẫu khác nhau, dù cùng sàn
    cùng ngày. Nếu sàn sắp xếp tin theo tiêu chí nào đó dính tới giá thì độ sâu
    trang trở thành biến gây nhiễu: so hai đợt cào có độ sâu khác nhau sẽ ra
    một "biến động thị trường" hoàn toàn giả.

    Đo trên chính dữ liệu nhatot pages 1–50: giá/m² tăng khoảng **0,5% mỗi
    trang** (p = 1,4e-06, đã khống chế diện tích). Nghĩa là mẫu cào sâu 120
    trang sẽ cao hơn mẫu cào 50 trang chừng 19% — đủ để tạo ra một "cú giảm
    giá" tưởng tượng nếu hai đợt cào không cùng độ sâu.

    Kết luận thực dụng: mỗi lần cào phải GIỮ LẠI cột `web_scraper_start_url`,
    và khi so sánh theo thời gian thì so cùng một dải trang.
    """
    import statsmodels.formula.api as smf
    from clean_common import parse_area, parse_price

    d = pd.read_excel(tep_tho)
    if "web_scraper_start_url" not in d.columns:
        raise ValueError("File không có cột web_scraper_start_url — không biết "
                         "mỗi tin lấy từ trang nào để đo.")
    d["trang"] = (d["web_scraper_start_url"].astype(str)
                  .str.extract(r"p(?:age=)?(\d+)")[0].astype(float))
    d["ppm2"] = d["gia_ban"].map(parse_price) / d["dien_tich"].map(parse_area)
    d = d[d["ppm2"].between(8e6, 400e6)].dropna(subset=["trang"]).copy()
    d["y"] = np.log(d["ppm2"])
    d["ldt"] = np.log(d["dien_tich"].map(parse_area))

    m = smf.ols("y ~ trang + ldt", data=d).fit()
    b = float(m.params["trang"])
    lo, hi = m.conf_int().loc["trang"]
    return pd.DataFrame([{
        "so_tin": int(m.nobs),
        "trang_tu": int(d["trang"].min()), "trang_den": int(d["trang"].max()),
        "thay_doi_moi_trang_%": round((np.exp(b) - 1) * 100, 3),
        "KTC_95%": f"[{(np.exp(lo)-1)*100:+.3f}; {(np.exp(hi)-1)*100:+.3f}]",
        "p": f"{m.pvalues['trang']:.1e}",
    }])


def do_luong_cong_bang(d: pd.DataFrame, n_fold: int = 5) -> pd.DataFrame:
    """So sánh ĐÚNG CÁCH: đổi tập huấn luyện, giữ nguyên tập kiểm tra.

    VÌ SAO PHẢI VIẾT RIÊNG HÀM NÀY
    ------------------------------
    Cách đo hiển nhiên — chạy đánh giá chéo trên tập cũ, rồi chạy lại trên tập
    gộp, rồi so hai con số MAPE — là SAI, và sai theo kiểu rất dễ tin.

    Vì ở lần chạy thứ hai, tập kiểm tra cũng đổi: nó có thêm tin batdongsan, mà
    tin batdongsan trải trên khoảng giá rộng hơn nhiều (có căn 40 tỷ, 265 m²).
    MAPE cao hơn khi đó không có nghĩa "thêm dữ liệu làm model tệ đi" — nó chỉ
    có nghĩa "đề thi lần hai khó hơn". Hai con số không so được với nhau.

    Bằng chứng cho thấy đúng là như vậy: khi đo kiểu đó, MAPE tăng 16,34% ->
    17,74% NHƯNG R² lại tăng 0,735 -> 0,803. Một model vừa dở đi vừa giải thích
    được nhiều phương sai hơn là chuyện vô lý — dấu hiệu điển hình của việc đổi
    thước đo giữa chừng.

    Cách đúng: cố định đề thi. Tập kiểm tra LUÔN là tin cũ, chia fold y hệt
    nhau ở cả hai nhánh; chỉ có tập huấn luyện là thay đổi. Lúc đó chênh lệch
    MAPE quy được về đúng một nguyên nhân: phần dữ liệu thêm vào.
    """
    from sklearn.model_selection import KFold
    from train_eval import (RANDOM_STATE, apply_target_encoding,
                            fit_target_encoding, make_model)
    from gan_toa_do import haversine_km
    TRUNG_TAM = (21.0287, 105.8524)

    d = d.copy()
    if "latitude" in d.columns:
        d["kc_trung_tam_km"] = haversine_km(d["latitude"], d["longitude"], *TRUNG_TAM)
    num = [c for c in ["dien_tich_m2", "so_phong_ngu", "so_phong_vs", "tang_so",
                       "latitude", "longitude", "kc_trung_tam_km"] if c in d.columns]
    num += [c for c in d.columns if c.startswith("nhac_")]
    cat = ["quan_huyen", "phuong_moi", "du_an_clean"]
    for c in num:
        if d[c].dtype == object:
            d[c] = pd.to_numeric(d[c], errors="coerce")

    moc = d["ngay_quan_sat"].min()
    cu = d[d["ngay_quan_sat"] == moc].reset_index(drop=True)
    moi = d[d["ngay_quan_sat"] > moc].reset_index(drop=True)
    print(f"    tập kiểm tra cố định: {len(cu)} tin cũ | thêm vào huấn luyện: "
          f"{len(moi)} tin mới")

    def _chuan_bi(tr, te, cot_muc):
        y_tr = np.log1p(tr["TARGET_gia_vnd"].astype(float))
        X_tr, X_te = tr[num].copy(), te[num].copy()
        for c in cot_muc:
            bang, du_phong = fit_target_encoding(tr, c, y_tr)
            X_tr[f"{c}_te"] = apply_target_encoding(tr, c, bang, du_phong)
            X_te[f"{c}_te"] = apply_target_encoding(te, c, bang, du_phong)
        return X_tr, y_tr, X_te

    kf = KFold(n_splits=n_fold, shuffle=True, random_state=RANDOM_STATE)
    ket = {n: [] for n in ("chỉ tin cũ", "cũ + mới", "cũ + mới, có cột nguồn")}
    for tr_i, te_i in kf.split(cu):
        tr_cu, te = cu.iloc[tr_i], cu.iloc[te_i]
        that = te["TARGET_gia_vnd"].astype(float).to_numpy()
        for nhan in ket:
            tr = tr_cu if nhan == "chỉ tin cũ" else pd.concat([tr_cu, moi],
                                                              ignore_index=True)
            cm = cat + (["nguon"] if "nguon" in nhan else [])
            X_tr, y_tr, X_te = _chuan_bi(tr, te, cm)
            m = make_model().fit(X_tr, y_tr)
            du = np.expm1(m.predict(X_te))
            ket[nhan].append(float(np.mean(np.abs(du - that) / that) * 100))

    hang = [{"tập huấn luyện": k,
             "MAPE (%)": round(float(np.mean(v)), 2),
             "độ lệch giữa các fold": round(float(np.std(v)), 2)}
            for k, v in ket.items()]
    goc = hang[0]["MAPE (%)"]
    for h in hang:
        h["thay đổi (điểm %)"] = round(h["MAPE (%)"] - goc, 2)
    return pd.DataFrame(hang)


def do_luong(d: pd.DataFrame) -> pd.DataFrame:
    """MAPE khi chỉ dùng dữ liệu cũ, so với khi dùng toàn bộ.

    So sánh trên CÙNG một quy trình đánh giá chéo, cùng hạt giống — nên chênh
    lệch phản ánh đúng phần dữ liệu thêm vào, không phải may rủi chia fold.
    """
    from train_eval import cross_validate
    from gan_toa_do import haversine_km
    TRUNG_TAM = (21.0287, 105.8524)

    d = d.copy()
    if "latitude" in d.columns:
        d["kc_trung_tam_km"] = haversine_km(d["latitude"], d["longitude"], *TRUNG_TAM)
    num = [c for c in ["dien_tich_m2", "so_phong_ngu", "so_phong_vs", "tang_so",
                       "latitude", "longitude", "kc_trung_tam_km"] if c in d.columns]
    num += [c for c in d.columns if c.startswith("nhac_")]

    cu = d[d["ngay_quan_sat"] == d["ngay_quan_sat"].min()]
    bo = [
        ("chỉ dữ liệu cũ", cu, ["quan_huyen", "phuong_moi", "du_an_clean"]),
        ("toàn bộ, KHÔNG có cột nguồn", d, ["quan_huyen", "phuong_moi", "du_an_clean"]),
        ("toàn bộ, CÓ cột nguồn", d,
         ["quan_huyen", "phuong_moi", "du_an_clean", "nguon"]),
    ]
    hang = []
    for nhan, sub, cat in bo:
        r = cross_validate(sub, "TARGET_gia_vnd", num, cat)
        hang.append({"bộ dữ liệu": nhan, "n": r["n"],
                     "MAPE (%)": round(r["MAPE"], 2),
                     "độ lệch": round(r["MAPE_std"], 2),
                     "R²": round(r["R2"], 3)})
        print("   ", hang[-1], flush=True)
    return pd.DataFrame(hang)


def main() -> None:
    ap = argparse.ArgumentParser(description="Gộp nhiều đợt cào")
    ap.add_argument("--loai", choices=["chungcu", "nhadat"], default="chungcu")
    ap.add_argument("--do-luong", action="store_true",
                    help="đo MAPE trước/sau khi thêm dữ liệu")
    a = ap.parse_args()

    d, bc = gop(loai=a.loai)
    print("\n" + "=" * 60)
    print(bc.to_string(index=False))
    print("\nTheo nguồn và đợt cào:")
    print(d.groupby(["nguon", "ngay_quan_sat"]).size().to_string())

    ten_ra = f"{a.loai}_gop.xlsx"
    d.to_excel(DATA_CLEAN / ten_ra, index=False)
    bc.to_excel(DATA_CLEAN / f"bao_cao_gop_nguon_{a.loai}.xlsx", index=False)
    print(f"\n-> data_clean/{ten_ra} ({len(d)} dòng)")

    if a.do_luong:
        from gan_toa_do import gan, nap_api
        print("\nGán toạ độ cho tập gộp…")
        api = nap_api()
        dg = gan(d, api, cot_ma_tin="listing_id")
        print(dg["nguon_toa_do"].value_counts().to_string())
        dg.to_excel(DATA_CLEAN / "chungcu_gop_geo.xlsx", index=False)
        print("\nĐo MAPE — cách SAI (đổi cả tập kiểm tra), để đối chiếu:")
        sai = do_luong(dg)
        print("\nĐo MAPE — cách ĐÚNG (tập kiểm tra cố định là tin cũ):")
        dung = do_luong_cong_bang(dg)
        print(dung.to_string(index=False))
        with pd.ExcelWriter(DATA_CLEAN / "ablation_gop_nguon.xlsx") as w:
            dung.to_excel(w, sheet_name="tap_kiem_tra_co_dinh", index=False)
            sai.to_excel(w, sheet_name="doi_ca_tap_kiem_tra", index=False)


if __name__ == "__main__":
    main()
