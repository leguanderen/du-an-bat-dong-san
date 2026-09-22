"""
explain.py — MODULE D: giải thích kết quả định giá bằng tiếng Việt.

Trả lời câu hỏi mà mọi người dùng đều hỏi và hệ thống hiện tại không trả lời
được: "vì sao lại là con số này?"

VÌ SAO SHAP TRONG KHÔNG GIAN LOG LẠI ĐẸP
----------------------------------------
Model được huấn luyện trên log(giá). SHAP phân rã dự đoán thành tổng các đóng
góp CỘNG trong không gian đó:

    log(giá) = giá_trị_cơ_sở + Σ đóng_góp_i

Lấy hàm mũ hai vế thì phép cộng thành phép NHÂN:

    giá = exp(cơ_sở) × Π exp(đóng_góp_i)

Nghĩa là mỗi đóng góp SHAP quy đổi thẳng thành "yếu tố này làm giá tăng/giảm
bao nhiêu phần trăm" — đúng cách người ta nói về giá nhà ngoài đời ("căn góc
đắt hơn khoảng 5%"), chứ không phải "cộng thêm 180 triệu". Không cần xấp xỉ gì
cả, đây là hệ quả toán học trực tiếp của việc học trên log.

NGUYÊN TẮC KHI SINH CÂU
-----------------------
Chỉ nêu các yếu tố có đóng góp đáng kể (mặc định ≥ 1,5%), tối đa 6 yếu tố. Liệt
kê hết 40 đặc trưng thì không ai đọc, và các đóng góp li ti chủ yếu là nhiễu.
Câu chữ mô tả ĐÓNG GÓP CỦA MODEL, không khẳng định quan hệ nhân quả — model học
tương quan trên tin rao, không phải thí nghiệm có kiểm soát.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# =============================================================================
# TÊN ĐẶC TRƯNG THÂN THIỆN
# =============================================================================
# Khoá là tên cột trong ma trận đặc trưng; giá trị là cách gọi trong câu tiếng Việt.

FEATURE_LABELS: dict[str, str] = {
    "dien_tich_m2": "Diện tích",
    "dien_tich_dat_m2": "Diện tích đất",
    "dien_tich_su_dung_m2": "Diện tích sử dụng",
    "so_phong_ngu": "Số phòng ngủ",
    "so_phong_vs": "Số phòng vệ sinh",
    "tang_so": "Tầng",
    "tong_so_tang": "Tổng số tầng",
    "mat_tien_m": "Mặt tiền",
    "chieu_dai_m": "Chiều dài",
    "duong_rong_m": "Độ rộng đường/ngõ",
    "co_oto": "Ô tô vào nhà",
    "kinh_doanh": "Vị trí tiện kinh doanh",
    "thang_may": "Có thang máy",
    "lo_goc": "Lô góc / căn góc",
    "quan_huyen_te": "Quận/huyện",
    "phuong_xa_te": "Phường/xã",
    "du_an_clean_te": "Dự án",
    "loai_hinh_te": "Loại hình",
    "phap_ly_te": "Pháp lý",
    "noi_that_te": "Nội thất",
    "huong_cua_te": "Hướng cửa chính",
    "huong_ban_cong_te": "Hướng ban công",
    "tinh_trang_bds_te": "Tình trạng bàn giao",
    "kc_trung_tam_km": "Khoảng cách tới trung tâm",
}

# MỌI cột nói về vị trí gộp lại thành MỘT yếu tố khi giải thích.
#
# Ban đầu nhóm này chỉ có ba cột toạ độ, vì "latitude +3,9%" thì người đọc
# không hiểu, và tách vĩ độ khỏi kinh độ còn sai về mặt diễn giải — chúng chỉ
# có nghĩa khi đi cùng nhau.
#
# VÌ SAO PHẢI GỘP CẢ QUẬN VÀ PHƯỜNG VÀO ĐÂY
# ------------------------------------------
# Để riêng, giao diện hiện ra hai dòng NGƯỢC DẤU cho cùng một chuyện:
#
#     Vị trí trên bản đồ   +14,4%
#     Quận/huyện Hai Bà Trưng   -7,4%      <- người dùng đọc là "quận trung
#                                             tâm mà bị trừ điểm?!"
#
# Đó không phải mô hình sai, mà là bản chất của SHAP khi các đặc trưng chồng
# lấn nhau: toạ độ và tên quận nói CÙNG một thông tin. SHAP phải chia phần công
# trạng cho từng cột, và cách chia đó không có ý nghĩa khi đọc riêng lẻ. Toạ độ
# đã ăn hết phần "gần trung tâm" (+14,4%), nên cột tên quận chỉ còn phần DƯ:
# "nhà bán ở Hai Bà Trưng rẻ hơn mức mà toạ độ đó gợi ý". Cộng lại mới là câu
# trả lời thật, và nó dương.
#
# Nguyên tắc chung: đặc trưng nào chồng lấn thông tin với nhau thì đọc riêng
# từng con số là vô nghĩa — phải cộng cả cụm. Giống hệt lý do gộp 60 chiều
# TF-IDF thành một dòng "Nội dung mô tả tin".
#
# `du_an_te` KHÔNG nằm trong nhóm này: tên dự án mang thêm thông tin thương
# hiệu/chất lượng xây dựng chứ không chỉ là toạ độ, và người dùng đọc "Dự án
# Times City +8%" là hiểu ngay.
NHOM_TOA_DO = ("latitude", "longitude", "kc_trung_tam_km",
               "quan_huyen_te", "phuong_moi_te", "phuong_xa_te")

FLAG_LABELS: dict[str, str] = {
    "nhac_o_to": "mô tả nhắc ô tô vào nhà",
    "nhac_thang_may": "mô tả nhắc thang máy",
    "nhac_be_boi": "mô tả nhắc bể bơi",
    "nhac_gym": "mô tả nhắc phòng gym",
    "nhac_truong": "mô tả nhắc gần trường học",
    "nhac_benh_vien": "mô tả nhắc gần bệnh viện",
    "nhac_cho_sieu_thi": "mô tả nhắc gần chợ, siêu thị",
    "nhac_cong_vien": "mô tả nhắc gần công viên, hồ",
    "nhac_metro": "mô tả nhắc gần metro",
    "nhac_kinh_doanh": "mô tả nhắc tiện kinh doanh",
    "nhac_chinh_chu": "rao bán chính chủ",
    "nhac_view": "mô tả nhấn mạnh view",
    "nhac_lo_goc": "lô góc / căn góc",
    "nhac_no_hau": "nhà nở hậu",
    "nhac_can_sua": "nhà cần sửa",
}


def _label_for(col: str, raw_row: pd.Series | None,
               moc: dict | None = None) -> str:
    """Tên hiển thị cho một đặc trưng, kèm giá trị thật và MỐC SO SÁNH.

    VÌ SAO PHẢI HIỆN MỐC
    --------------------
    Nhãn cũ chỉ ghi "Diện tích 50 → −18%", và câu hỏi hiển nhiên là: vậy bao
    nhiêu mét thì thành cộng? Không nói mốc thì con số ± treo lơ lửng, người
    đọc không tự kiểm tra được.

    Mốc của một đặc trưng số là giá trị mà tại đó đóng góp đổi dấu — và nó
    trùng đúng với giá trị trung vị trong dữ liệu (đo được: chung cư ~72 m² so
    với trung vị 75 m²; nhà đất ~55 m² so với trung vị 55 m²). Không phải trùng
    hợp: model học từ dữ liệu đó, nên "mức bình thường" của nó chính là mức phổ
    biến trong dữ liệu.
    """
    if col in FEATURE_LABELS:
        base = FEATURE_LABELS[col]
        if raw_row is not None:
            # với đặc trưng mã hoá mục, hiển thị giá trị gốc chứ không phải số mã hoá
            src = col[:-3] if col.endswith("_te") else col
            if src in raw_row.index and pd.notna(raw_row[src]):
                val = raw_row[src]
                if isinstance(val, (int, float, np.integer, np.floating)):
                    m = (moc or {}).get(src)
                    if m is not None and np.isfinite(m):
                        so_sanh = ("lớn hơn" if val > m else
                                   "nhỏ hơn" if val < m else "đúng bằng")
                        return f"{base} {val:g} — {so_sanh} mức phổ biến {m:g}"
                    return f"{base} {val:g}"
                return f"{base} {val}"
        return base
    if col in FLAG_LABELS:
        return FLAG_LABELS[col].capitalize()
    if col.startswith("dd_"):
        return col[3:]
    if col.startswith("txt_"):
        return "Nội dung phần mô tả"
    return col


# =============================================================================
# TÍNH ĐÓNG GÓP
# =============================================================================

def _gia_tri_shap(model, X: pd.DataFrame) -> tuple[np.ndarray, float]:
    """Trả về (ma trận SHAP, giá trị cơ sở) mà KHÔNG cần thư viện `shap`.

    VÌ SAO KHÔNG DÙNG `shap`
    ------------------------
    Thuật toán TreeSHAP vốn do chính nhóm XGBoost cài đặt bằng C++ ngay trong
    thư viện XGBoost: `Booster.predict(..., pred_contribs=True)`. Gói `shap`
    của Python, khi gặp model XGBoost, cũng chỉ gọi đúng hàm đó rồi bọc lại.
    Nghĩa là thêm `shap` vào yêu cầu cài đặt chỉ để lấy một con số mà XGBoost
    đã tính sẵn — mà `shap` lại nặng, hay lỗi biên dịch trên Python mới, và đã
    làm ứng dụng vỡ trên máy bạn (`ModuleNotFoundError: No module named 'shap'`).

    Đây KHÔNG phải xấp xỉ hay thay thế gần đúng: hai đường cho ra cùng một con
    số tới từng chữ số (đã đối chiếu, lệch < 1e-6). Xem `_scratch/doi_chieu_shap.py`.

    Ma trận `pred_contribs` có `n_dac_trung + 1` cột; cột cuối là số hạng chệch
    (bias) — chính là `expected_value` mà `shap` gọi tên khác.

    Nếu model không phải XGBoost (ví dụ sau này đổi sang thuật toán khác) thì
    mới rơi về `shap`, và chỉ khi đó mới cần cài nó.
    """
    booster = getattr(model, "get_booster", None)
    if booster is not None:
        import xgboost as xgb
        b = booster()
        dm = xgb.DMatrix(X, missing=np.nan,
                         feature_names=list(map(str, X.columns)))
        ct = b.predict(dm, pred_contribs=True)
        return ct[:, :-1], float(ct[0, -1])

    import shap  # đường dự phòng cho model không phải cây XGBoost
    ex = shap.TreeExplainer(model)
    return np.asarray(ex.shap_values(X)), float(ex.expected_value)

def explain_one(
    model,
    X_row: pd.DataFrame,
    raw_row: pd.Series | None = None,
    top_k: int = 6,
    min_effect_pct: float = 1.5,
    group_text: bool = True,
    moc: dict | None = None,
    truong_da_nhap: set | None = None,
) -> dict:
    """Phân rã một dự đoán thành các đóng góp phần trăm.

    Trả về dict gồm:
        gia_co_so   — mức giá "trung bình" mà model xuất phát từ đó (VND)
        gia_du_doan — giá dự đoán cho căn này (VND)
        chenh_lech_pct — cao/thấp hơn mức cơ sở bao nhiêu phần trăm
        yeu_to     — danh sách {ten, dong_gop_pct} đã sắp theo độ lớn
        cau_van    — đoạn văn tiếng Việt sẵn để hiển thị
    """
    sv, base_log = _gia_tri_shap(model, X_row)
    if sv.ndim > 1:
        sv = sv[0]

    contribs = pd.Series(sv, index=X_row.columns)

    # Gom các chiều TF-IDF thành một mục duy nhất — 60 cột "txt_i" riêng lẻ thì
    # vô nghĩa với người đọc, nhưng tổng của chúng thì có nghĩa.
    if group_text:
        txt_cols = [c for c in contribs.index if c.startswith("txt_")]
        if txt_cols:
            total = contribs[txt_cols].sum()
            contribs = contribs.drop(index=txt_cols)
            contribs["txt_tong_hop"] = total

    # Gộp toàn bộ cụm vị trí thành một yếu tố — xem NHOM_TOA_DO.
    toa_do = [c for c in NHOM_TOA_DO if c in contribs.index]
    if len(toa_do) > 1:
        total = contribs[toa_do].sum()
        contribs = contribs.drop(index=toa_do)
        contribs["toa_do_tong_hop"] = total

    # log -> phần trăm: exp(đóng góp) - 1
    effects_pct = (np.exp(contribs) - 1) * 100

    gia_co_so = float(np.expm1(base_log))
    gia_du_doan = float(np.expm1(base_log + contribs.sum()))
    chenh_lech_pct = (gia_du_doan / gia_co_so - 1) * 100

    # CHỈ GIẢI THÍCH NHỮNG GÌ NGƯỜI DÙNG THỰC SỰ ĐIỀN
    #
    # Người dùng phàn nàn đúng: bảng yếu tố hiện "Mô tả nhắc bể bơi −2%" và
    # "Mô tả nhấn mạnh view" trong khi họ không hề khai hai thứ đó. Lý do là
    # các cờ `nhac_*` được suy từ MÔ TẢ của tin rao khi huấn luyện, nhưng ở
    # form định giá người dùng không có ô nào để điền chúng — nên chúng vào
    # model dưới dạng thiếu, và SHAP vẫn gán cho chúng một phần đóng góp.
    #
    # Phần đóng góp ấy không sai về mặt toán: nó là "căn này không khai gì về
    # bể bơi, mà mặt bằng thì có một tỉ lệ tin có khai". Nhưng trình bày nó
    # thành một dòng riêng mang tên "bể bơi" là nói với người dùng về một thứ
    # họ chưa từng nhập — vừa khó hiểu vừa làm loãng những yếu tố thật.
    #
    # Nên gom hết những trường người dùng để trống vào MỘT dòng trung thực.
    # Tổng vẫn khớp, không mất phần nào; chỉ khác chỗ trình bày.
    de_trong: list = []
    if truong_da_nhap is not None:
        luon_giu = {"txt_tong_hop", "toa_do_tong_hop"}
        de_trong = [c for c in effects_pct.index
                    if c not in luon_giu and c not in truong_da_nhap]
        if de_trong:
            tong_trong = float(effects_pct[de_trong].sum())
            effects_pct = effects_pct.drop(index=de_trong)
            if abs(tong_trong) >= min_effect_pct:
                effects_pct["_de_trong"] = tong_trong

    ranked = effects_pct.reindex(effects_pct.abs().sort_values(ascending=False).index)
    ranked = ranked[ranked.abs() >= min_effect_pct].head(top_k)

    TEN_GOM = {"txt_tong_hop": "Nội dung phần mô tả",
               "toa_do_tong_hop": "Vị trí",
               "_de_trong": "Các mục bạn để trống"}

    # ĐỔI TỪNG ĐÓNG GÓP RA ĐỒNG, VÀ ĐỔI THEO ĐƯỜNG TÍCH LUỸ
    #
    # VÌ SAO KHÔNG ĐỂ NGUYÊN PHẦN TRĂM
    # Đóng góp SHAP cộng được với nhau trong KHÔNG GIAN LOGARIT, không phải
    # trong phần trăm. Đổi từng cái sang phần trăm rồi cộng lại thì không ra
    # tổng: thử một căn chung cư, ba dòng hiện ra là +24,0%, −8,4%, +7,4%, cộng
    # lại +23,0% trong khi mức lệch thật là +22,3%. Sai số nhỏ, nhưng nó biến
    # cả bảng thành một phép cộng KHÔNG khớp — và người dùng hỏi đúng câu "tại
    # sao lại cộng trừ như vậy" thì đó là điều tệ nhất có thể trả lời.
    #
    # Trong ĐỒNG thì cộng đúng tuyệt đối. Cách tính: đi dọc đường tích luỹ,
    # cộng dần từng đóng góp log, mỗi bước quy ra giá; phần chênh giữa hai bước
    # liên tiếp là số đồng của yếu tố đó. Các phần chênh này cộng lại bằng đúng
    # (giá dự đoán − giá mốc).
    #
    # ĐÁNH ĐỔI PHẢI NÓI RA: cách chia theo đường tích luỹ PHỤ THUỘC THỨ TỰ. Ở
    # đây luôn xếp theo độ lớn giảm dần nên kết quả nhất quán giữa các lần chạy,
    # và phần trăm (không phụ thuộc thứ tự) vẫn được trả về cạnh đó.
    log_gom = contribs.copy()
    if truong_da_nhap is not None and "_de_trong" in effects_pct.index:
        log_gom = log_gom.drop(index=[c for c in log_gom.index
                                      if c in de_trong], errors="ignore")
        log_gom["_de_trong"] = float(np.log1p(effects_pct["_de_trong"] / 100))

    yeu_to, truoc = [], base_log
    for c in ranked.index:
        buoc = float(log_gom.get(c, np.log1p(ranked[c] / 100)))
        sau = truoc + buoc
        yeu_to.append({
            "ten": TEN_GOM.get(c) or _label_for(c, raw_row, moc),
            "dong_gop_pct": round(float(ranked[c]), 1),
            "dong_gop_vnd": float(np.expm1(sau) - np.expm1(truoc)),
            "khoa": c,
        })
        truoc = sau

    # DÒNG KHÉP SỔ. Chỉ hiện top_k yếu tố lớn nhất, nên phần còn lại phải được
    # ghi ra thành một dòng, nếu không cái thác nước không chạm tới con số cuối
    # và người xem cộng tay sẽ thấy lệch.
    con_lai = float(gia_du_doan - np.expm1(truoc))
    if abs(con_lai) > abs(gia_du_doan) * 0.01:
        yeu_to.append({"ten": "Các yếu tố nhỏ khác",
                       "dong_gop_pct": None,
                       "dong_gop_vnd": con_lai,
                       "khoa": "_con_lai"})
    elif yeu_to:
        # Phần dư dưới 1% thì dồn vào dòng cuối thay vì thêm một dòng "−0,01
        # tỷ" chỉ để làm tròn. Cột đồng PHẢI cộng đúng bằng con số hiển thị —
        # bảng này tồn tại để người dùng cộng tay kiểm được, nên lệch 10 triệu
        # ở hàng cuối là đủ để mất hết tác dụng.
        yeu_to[-1]["dong_gop_vnd"] += con_lai

    return {
        "gia_co_so": gia_co_so,
        "gia_du_doan": gia_du_doan,
        "chenh_lech_pct": round(chenh_lech_pct, 1),
        "yeu_to": yeu_to,
        "cau_van": render_sentence(gia_du_doan, chenh_lech_pct, yeu_to, raw_row),
    }


# =============================================================================
# SINH CÂU TIẾNG VIỆT
# =============================================================================

def format_vnd(value: float) -> str:
    if value >= 1e9:
        return f"{value/1e9:,.2f} tỷ".replace(",", "@").replace(".", ",").replace("@", ".")
    if value >= 1e6:
        return f"{value/1e6:,.0f} triệu".replace(",", ".")
    return f"{value:,.0f} đồng".replace(",", ".")


def render_sentence(gia: float, chenh_lech_pct: float,
                    yeu_to: list[dict], raw_row: pd.Series | None) -> str:
    """Ghép các đóng góp thành đoạn văn người thường đọc được."""
    dia_diem = ""
    if raw_row is not None:
        phuong = raw_row.get("phuong_xa")
        quan = raw_row.get("quan_huyen")
        if isinstance(phuong, str) and isinstance(quan, str):
            dia_diem = f" tại {phuong}, {quan}"

    huong = "cao hơn" if chenh_lech_pct >= 0 else "thấp hơn"
    mo_dau = (f"Bất động sản này được định giá {format_vnd(gia)}"
              f"{dia_diem} — {huong} mặt bằng chung khoảng "
              f"{abs(chenh_lech_pct):.0f}%.")

    co_pct = [y for y in yeu_to if y.get("dong_gop_pct") is not None]
    cong = [y for y in co_pct if y["dong_gop_pct"] > 0]
    tru = [y for y in co_pct if y["dong_gop_pct"] < 0]

    dong = [mo_dau]
    if cong:
        dong.append("Yếu tố làm tăng giá: "
                    + ", ".join(f"{y['ten']} +{y['dong_gop_pct']:.0f}%" for y in cong) + ".")
    if tru:
        dong.append("Yếu tố làm giảm giá: "
                    + ", ".join(f"{y['ten']} {y['dong_gop_pct']:.0f}%" for y in tru) + ".")
    if not cong and not tru:
        dong.append("Không có yếu tố nào lệch đáng kể so với mặt bằng chung.")

    dong.append("Đây là đóng góp mà mô hình học được từ dữ liệu tin rao, "
                "không phải khẳng định về quan hệ nhân quả.")
    return " ".join(dong)


# =============================================================================
# TẦM QUAN TRỌNG TỔNG THỂ (cho báo cáo, không phải cho từng căn)
# =============================================================================

def global_importance(model, X: pd.DataFrame, sample: int = 400,
                      group_text: bool = True) -> pd.DataFrame:
    """Tầm quan trọng trung bình theo |SHAP| trên một mẫu ngẫu nhiên.

    Đáng tin hơn `feature_importances_` sẵn có của XGBoost, vốn đếm số lần chia
    nhánh và thiên vị các đặc trưng nhiều giá trị khác nhau.
    """
    Xs = X.sample(min(sample, len(X)), random_state=42)
    sv, _ = _gia_tri_shap(model, Xs)
    imp = pd.Series(np.abs(sv).mean(axis=0), index=X.columns)
    if group_text:
        txt = [c for c in imp.index if c.startswith("txt_")]
        if txt:
            total = imp[txt].sum()
            imp = imp.drop(index=txt)
            imp["Nội dung phần mô tả"] = total
    imp = imp.sort_values(ascending=False)
    return pd.DataFrame({
        "Đặc trưng": [_label_for(c, None) for c in imp.index],
        "Tầm quan trọng (|SHAP| TB, log)": imp.round(4).to_numpy(),
        "Tương đương % giá": ((np.exp(imp) - 1) * 100).round(1).to_numpy(),
    })
