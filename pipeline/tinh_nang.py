"""
tinh_nang.py — Bốn tính năng dựng trên tầng định giá đã có.

    1. san_tin()        — tìm tin đang rao thấp/cao bất thường so với mặt bằng
    2. dinh_gia_nguoc() — "nếu có sổ hồng thì giá đổi bao nhiêu?"
    3. ngan_sach()      — "3 tỷ thì mua được gì, ở phường nào?"
    4. so_sanh()        — đặt hai bất động sản cạnh nhau, giải thích chênh lệch

Không cái nào cần dữ liệu mới. Tất cả là cách trình bày khác của thứ đã có:
khoảng tin cậy CQR, mô hình XGBoost, và bảng 6.200 tin đã gắn toạ độ.

MỘT NGUYÊN TẮC XUYÊN SUỐT BỐN TÍNH NĂNG
---------------------------------------
Mô hình học TƯƠNG QUAN trên tin rao, không phải quan hệ nhân quả, và nó dự đoán
GIÁ RAO chứ không phải giá giao dịch. Cả bốn tính năng dưới đây vì thế đều phải
kèm lời cảnh báo đúng chỗ, không được để giao diện nói to hơn bằng chứng:

  * "Tin này rao thấp hơn mặt bằng 25%" KHÔNG có nghĩa là món hời. Nó có nghĩa
    là mô hình không giải thích được vì sao rẻ thế — và lý do thật có thể là
    nhà nằm sát nghĩa trang, dính quy hoạch, hoặc tin ảo. Đó là danh sách
    ĐÁNG ĐI XEM, không phải danh sách đáng đặt cọc.

  * "Có sổ hồng thì +8%" là chênh lệch mô hình quan sát được giữa các tin có
    và không có sổ, sau khi đã tính các yếu tố khác. Nó KHÔNG hứa rằng làm sổ
    xong thì bán thêm được 8%.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

import valuation_service as vs
from intervals import ConformalValuer
from train_eval import RANDOM_STATE


# =============================================================================
# 1. SĂN TIN HỜI / CẢNH BÁO TIN HỚ
# =============================================================================

@lru_cache(maxsize=4)
def _khoang_ngoai_mau(loai: str, n_fold: int = 5) -> pd.DataFrame:
    """Khoảng tin cậy NGOÀI MẪU cho từng tin trong dữ liệu.

    Đây là chỗ dễ làm sai nhất của tính năng này. Nếu huấn luyện trên toàn bộ
    dữ liệu rồi đi chấm chính những tin đó, mô hình đã NHÌN THẤY giá của chúng
    lúc học — nó sẽ dự đoán sát, khoảng tin cậy ôm lấy giá rao, và gần như
    không tin nào bị coi là bất thường. Tính năng sẽ im lặng vô dụng.

    Nên phải chia fold: mỗi tin được chấm bởi một mô hình CHƯA từng thấy nó.
    Tốn 5 lần huấn luyện (khoảng 30 giây), nhưng đó là điều kiện để con số có
    nghĩa.
    """
    d = vs.nap(loai)
    df, cfg = d["df"].reset_index(drop=True), d["cfg"]

    lo = np.full(len(df), np.nan)
    hi = np.full(len(df), np.nan)
    du = np.full(len(df), np.nan)
    kf = KFold(n_splits=n_fold, shuffle=True, random_state=RANDOM_STATE)
    for tr, te in kf.split(df):
        m = ConformalValuer(list(cfg.cot_so), list(cfg.cot_muc),
                            alpha=vs.ALPHA_MAC_DINH)
        m.fit(df.iloc[tr], "TARGET_gia_vnd")
        kq = m.predict_interval(df.iloc[te])
        lo[te], hi[te], du[te] = (kq["khoang_duoi"].to_numpy(),
                                  kq["khoang_tren"].to_numpy(),
                                  kq["gia"].to_numpy())

    out = df.copy()
    out["gia_du_doan"] = du
    out["khoang_duoi"] = lo
    out["khoang_tren"] = hi
    out["lech_phan_tram"] = (out["TARGET_gia_vnd"] / out["gia_du_doan"] - 1) * 100
    out["vi_the"] = np.select(
        [out["TARGET_gia_vnd"] < out["khoang_duoi"],
         out["TARGET_gia_vnd"] > out["khoang_tren"]],
        ["thap_bat_thuong", "cao_bat_thuong"], default="trong_khoang")
    return out


def san_tin(loai: str = "chungcu", vi_the: str = "thap_bat_thuong",
            k: int = 30, quan: str | None = None, phuong: str | None = None,
            gia_toi_da: float | None = None,
            chi_toa_do_chac: bool = True) -> pd.DataFrame:
    """Tin nằm NGOÀI khoảng tin cậy 90% — thấp bất thường hoặc cao bất thường.

    `chi_toa_do_chac` mặc định BẬT: tin có toạ độ mức phường thì mô hình định
    vị nó kém, nên "lệch so với mặt bằng" rất có thể chỉ là mặt bằng bị tính
    nhầm chỗ. Lọc bớt để danh sách đỡ nhiễu.
    """
    d = _khoang_ngoai_mau(loai).copy()
    if chi_toa_do_chac and "nguon_toa_do" in d.columns:
        d = d[d["nguon_toa_do"].isin(["nguon_goc", "ma_tin", "du_an", "duong"])]
    if quan:
        d = d[d["quan_huyen"] == quan]
    if phuong:
        d = d[d["phuong_moi"] == phuong]
    if gia_toi_da:
        d = d[d["TARGET_gia_vnd"] <= gia_toi_da]

    d = d[d["vi_the"] == vi_the]
    d = d.sort_values("lech_phan_tram", ascending=(vi_the == "thap_bat_thuong"))
    return d.head(k).reset_index(drop=True)


def thong_ke_san_tin(loai: str = "chungcu") -> dict:
    d = _khoang_ngoai_mau(loai)
    v = d["vi_the"].value_counts()
    return {"tong": len(d),
            "thap_bat_thuong": int(v.get("thap_bat_thuong", 0)),
            "cao_bat_thuong": int(v.get("cao_bat_thuong", 0)),
            "trong_khoang": int(v.get("trong_khoang", 0)),
            "do_phu_thuc_te": float(v.get("trong_khoang", 0) / len(d))}


# =============================================================================
# 2. ĐỊNH GIÁ NGƯỢC — "NẾU… THÌ GIÁ ĐỔI BAO NHIÊU?"
# =============================================================================
# Mỗi phương án được định giá lại từ đầu bằng chính mô hình, thay vì đọc hệ số
# SHAP. Khác biệt quan trọng: SHAP nói "yếu tố này đóng góp bao nhiêu vào con
# số hiện tại", còn ở đây là "đổi yếu tố này thì con số thành bao nhiêu" — hai
# câu hỏi khác nhau, và người dùng hỏi câu thứ hai.

PHUONG_AN_CHUNGCU = {
    "phap_ly": ["Sổ hồng riêng", "Hợp đồng mua bán", "Đang chờ sổ"],
    "noi_that": ["Nội thất cao cấp", "Nội thất đầy đủ", "Hoàn thiện cơ bản",
                 "Bàn giao thô"],
    "so_phong_ngu": [1, 2, 3, 4],
    "huong_ban_cong": ["Đông Nam", "Đông", "Nam", "Tây Bắc", "Tây"],
    "tinh_trang_bds": ["Đã bàn giao", "Chưa bàn giao"],
}
PHUONG_AN_NHADAT = {
    "phap_ly": ["Đã có sổ", "Đang chờ sổ", "Giấy tờ viết tay", "Không có sổ"],
    "noi_that": ["Nội thất cao cấp", "Nội thất đầy đủ", "Hoàn thiện cơ bản"],
    "co_oto": [0, 1],
    "kinh_doanh": [0, 1],
    "thang_may": [0, 1],
    "lo_goc": [0, 1],
    "duong_rong_m": [2, 3, 5, 8, 12],
}

NHAN = {
    "phap_ly": "Giấy tờ pháp lý", "noi_that": "Nội thất",
    "so_phong_ngu": "Số phòng ngủ", "huong_ban_cong": "Hướng ban công",
    "tinh_trang_bds": "Tình trạng", "co_oto": "Ô tô vào nhà",
    "kinh_doanh": "Tiện kinh doanh", "thang_may": "Có thang máy",
    "lo_goc": "Lô góc", "duong_rong_m": "Độ rộng đường (m)",
}
_BAT_TAT = {0: "không", 1: "có"}


def _so_tin_ho_tro(loai: str, thuoc_tinh: str, gia_tri, dac_diem: dict) -> int:
    """Có bao nhiêu tin trong khu vực thực sự mang giá trị này.

    VÌ SAO CỘT NÀY QUAN TRỌNG HƠN VẺ NGOÀI
    --------------------------------------
    Chạy thử lần đầu, bảng cho ra "đổi sang bàn giao thô thì +2,7%" và "1 phòng
    ngủ đắt hơn 2 phòng ngủ". Nghe vô lý, và đúng là vô lý — không phải mô hình
    hỏng, mà vì trong phường đó chỉ có dăm ba tin bàn giao thô, tình cờ đều
    thuộc dự án cao cấp. Mô hình học đúng cái nó nhìn thấy; cái nó nhìn thấy thì
    quá ít để nói lên điều gì.

    Không thể sửa bằng cách giấu những dòng đó đi — giấu là gian. Cách trung
    thực là đặt số tin làm chứng ngay cạnh con số, để người đọc tự trừ hao.
    Dưới 10 tin thì con số phần trăm bên cạnh gần như là nhiễu.
    """
    df = vs.nap(loai)["df"]
    if thuoc_tinh not in df.columns:
        return 0
    trong_vung = pd.Series(True, index=df.index)
    if dac_diem.get("phuong_moi") and "phuong_moi" in df.columns:
        trong_vung = df["phuong_moi"] == dac_diem["phuong_moi"]
        if trong_vung.sum() < 30 and dac_diem.get("quan_huyen"):
            trong_vung = df["quan_huyen"] == dac_diem["quan_huyen"]
    elif dac_diem.get("quan_huyen"):
        trong_vung = df["quan_huyen"] == dac_diem["quan_huyen"]
    return int((trong_vung & (df[thuoc_tinh] == gia_tri)).sum())


def dinh_gia_nguoc(loai: str, dac_diem: dict,
                   thuoc_tinh: list[str] | None = None) -> pd.DataFrame:
    """Bảng: đổi từng thuộc tính thì giá thành bao nhiêu, chênh mấy phần trăm."""
    bo = PHUONG_AN_CHUNGCU if loai == "chungcu" else PHUONG_AN_NHADAT
    thuoc_tinh = thuoc_tinh or list(bo)
    goc = vs.dinh_gia(loai, dac_diem, giai_thich=False, so_comparables=0)["gia"]

    hang = []
    for tt in thuoc_tinh:
        if tt not in bo:
            continue
        for gt in bo[tt]:
            if dac_diem.get(tt) == gt:
                continue
            thu = dict(dac_diem)
            thu[tt] = gt
            g = vs.dinh_gia(loai, thu, giai_thich=False, so_comparables=0)["gia"]
            n = _so_tin_ho_tro(loai, tt, gt, dac_diem)
            bat_tat = tt in ("co_oto", "kinh_doanh", "thang_may", "lo_goc")
            dl = dac_diem.get(tt)
            hang.append({
                "ma_thuoc_tinh": tt,
                "thuoc_tinh": NHAN.get(tt, tt),
                # Giá trị HIỆN TẠI của căn. Thiếu cột này thì bảng vô nghĩa:
                # người đọc thấy "Độ rộng đường -> 2 m, -6,3%" mà không biết
                # đang đứng ở đâu, nên không biết -6,3% là đi lên hay đi xuống.
                "dang_la": ("chưa điền" if dl is None or pd.isna(dl)
                            else _BAT_TAT.get(dl, dl) if bat_tat else dl),
                "doi_thanh": _BAT_TAT.get(gt, gt) if bat_tat else gt,
                "gia_goc": goc,
                "gia_moi": g,
                "chenh_phan_tram": round((g / goc - 1) * 100, 1),
                "chenh_tien": g - goc,
                "so_tin_lam_chung": n,
                "do_tin_cay": ("đủ tin cậy" if n >= 30 else
                               "tạm được" if n >= 10 else "quá ít tin, đừng tin"),
            })
    out = pd.DataFrame(hang)
    if len(out):
        # Sắp theo ĐỘ TIN CẬY TRƯỚC, rồi mới tới độ lớn.
        #
        # Trước đây chỉ sắp theo |%|, và kết quả là những dòng dựa trên 0–4 tin
        # luôn leo lên đầu bảng — đúng như dự đoán, vì càng ít dữ liệu thì con
        # số càng dễ vọt ra ngoài. Người dùng đọc từ trên xuống, gặp ngay
        # "Độ rộng đường 12m → +30%" kèm chú thích "0 tin — đừng tin". Nói với
        # người ta "đây là con số quan trọng nhất, mà đừng tin nó" thì vô nghĩa.
        #
        # Vẫn KHÔNG xoá các dòng ít bằng chứng — xoá là giấu, và người dùng có
        # quyền biết mô hình nghĩ gì. Chỉ đẩy xuống dưới, sau các dòng có đủ
        # bằng chứng để đọc.
        thu_tu = {"đủ tin cậy": 0, "tạm được": 1, "quá ít tin, đừng tin": 2}
        out["_hang"] = out["do_tin_cay"].map(thu_tu)
        out = (out.sort_values(["_hang", "chenh_phan_tram"],
                               key=lambda s: s.abs() if s.name == "chenh_phan_tram" else s,
                               ascending=[True, False])
                  .drop(columns="_hang").reset_index(drop=True))
    return out


# =============================================================================
# 3. NGÂN SÁCH NÀY MUA ĐƯỢC GÌ, Ở ĐÂU
# =============================================================================

def ngan_sach(loai: str = "chungcu", ngan_sach_vnd: float = 3e9,
              dien_tich_toi_thieu: float | None = None,
              so_phong_ngu: int | None = None,
              du_dia_phan_tram: float = 0.0) -> pd.DataFrame:
    """Mỗi phường: có bao nhiêu tin vừa túi tiền, chiếm bao nhiêu phần trăm.

    `du_dia_phan_tram` cho phép nới ngân sách (ví dụ 10 = chấp nhận đắt hơn
    10%), vì người mua thật hiếm khi có một con số cứng.
    """
    d = vs.nap(loai)["df"].copy()
    cot_dt = vs.nap(loai)["cfg"].cot_dien_tich
    tran = ngan_sach_vnd * (1 + du_dia_phan_tram / 100)

    hop = pd.Series(True, index=d.index)
    if dien_tich_toi_thieu:
        hop &= d[cot_dt] >= dien_tich_toi_thieu
    if so_phong_ngu:
        hop &= d["so_phong_ngu"].fillna(-1) >= so_phong_ngu
    d["hop_yeu_cau"] = hop
    d["mua_duoc"] = hop & (d["TARGET_gia_vnd"] <= tran)

    g = d.groupby("phuong_moi")
    out = pd.DataFrame({
        "so_tin": g.size(),
        "so_tin_hop_yeu_cau": g["hop_yeu_cau"].sum(),
        "so_tin_mua_duoc": g["mua_duoc"].sum(),
        "gia_trung_vi": g["TARGET_gia_vnd"].median(),
        "dien_tich_trung_vi": g[cot_dt].median(),
        "lat": g["latitude"].median(),
        "lon": g["longitude"].median(),
    }).reset_index()
    out["ti_le_mua_duoc"] = np.where(
        out["so_tin_hop_yeu_cau"] > 0,
        out["so_tin_mua_duoc"] / out["so_tin_hop_yeu_cau"], np.nan)
    out["gia_trung_vi_ty"] = (out["gia_trung_vi"] / 1e9).round(2)
    return out.sort_values(["so_tin_mua_duoc", "ti_le_mua_duoc"],
                           ascending=False).reset_index(drop=True)


def tin_mua_duoc(loai: str = "chungcu", ngan_sach_vnd: float = 3e9,
                 phuong: str | None = None, dien_tich_toi_thieu: float | None = None,
                 so_phong_ngu: int | None = None, k: int = 50) -> pd.DataFrame:
    """Danh sách tin cụ thể nằm trong ngân sách."""
    d = vs.nap(loai)
    df, cot_dt = d["df"].copy(), d["cfg"].cot_dien_tich
    m = df["TARGET_gia_vnd"] <= ngan_sach_vnd
    if phuong:
        m &= df["phuong_moi"] == phuong
    if dien_tich_toi_thieu:
        m &= df[cot_dt] >= dien_tich_toi_thieu
    if so_phong_ngu:
        m &= df["so_phong_ngu"].fillna(-1) >= so_phong_ngu
    return (df[m].sort_values("gia_tren_m2")
            .head(k).reset_index(drop=True))


# =============================================================================
# 4. SO SÁNH HAI BẤT ĐỘNG SẢN
# =============================================================================

def so_sanh(loai: str, a: dict, b: dict,
            ten_a: str = "Bất động sản A",
            ten_b: str = "Bất động sản B") -> dict:
    """Hai bất động sản cạnh nhau, kèm phân rã chênh lệch theo từng yếu tố.

    Phần thú vị là bảng `khac_biet`: lấy đóng góp SHAP của từng bên rồi trừ
    nhau. Nó trả lời đúng câu người dùng hỏi — "vì sao căn kia đắt hơn?" — thay
    vì bắt họ tự đối chiếu hai danh sách yếu tố rời rạc.
    """
    ka = vs.dinh_gia(loai, a, giai_thich=True, so_comparables=0)
    kb = vs.dinh_gia(loai, b, giai_thich=True, so_comparables=0)

    # Nhãn của explain_one có kèm giá trị ("Diện tích 70", "Phường/xã Mỹ Đình
    # 1"), nên nếu ghép theo nhãn nguyên văn thì cùng MỘT yếu tố ở hai bất động
    # sản lại thành hai dòng riêng — bảng chênh lệch đọc thành vô nghĩa. Gom về
    # nhãn gốc trước khi trừ nhau.
    def _khoa(ten: str) -> str:
        from explain import FEATURE_LABELS
        goc = [v for v in FEATURE_LABELS.values() if ten.startswith(v)]
        return max(goc, key=len) if goc else ten

    def _bang(kq):
        out = {}
        for y in (kq.get("giai_thich") or {}).get("yeu_to", []):
            k = _khoa(y["ten"])
            out[k] = (out.get(k, (0.0, ""))[0] + y["dong_gop_pct"], y["ten"])
        return out

    ya, yb = _bang(ka), _bang(kb)
    moi = sorted(set(ya) | set(yb))
    khac = pd.DataFrame([{
        "yeu_to": t,
        f"{ten_a}": ya.get(t, (0.0, "—"))[1],
        f"{ten_a} (%)": round(ya.get(t, (0.0, ""))[0], 1),
        f"{ten_b}": yb.get(t, (0.0, "—"))[1],
        f"{ten_b} (%)": round(yb.get(t, (0.0, ""))[0], 1),
        "chenh_lech_diem": round(yb.get(t, (0.0, ""))[0] - ya.get(t, (0.0, ""))[0], 1),
    } for t in moi])
    if len(khac):
        khac = khac.reindex(khac["chenh_lech_diem"].abs()
                            .sort_values(ascending=False).index).reset_index(drop=True)

    return {
        "a": ka, "b": kb, "ten_a": ten_a, "ten_b": ten_b,
        "chenh_phan_tram": round((kb["gia"] / ka["gia"] - 1) * 100, 1),
        "chenh_tien": kb["gia"] - ka["gia"],
        "khac_biet": khac,
    }


__all__ = ["san_tin", "thong_ke_san_tin", "dinh_gia_nguoc", "ngan_sach",
           "tin_mua_duoc", "so_sanh"]
