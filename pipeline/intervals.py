"""
intervals.py — MODULE A: khoảng giá có bảo chứng thống kê.

Thay thế dòng `band = 0.15` trong app.py bằng một khoảng có thể KIỂM CHỨNG được.

VẤN ĐỀ VỚI ±15% CỐ ĐỊNH
-----------------------
Hệ thống hiện tại hiển thị "khoảng tham khảo" bằng cách nhân giá dự đoán với
0,85 và 1,15 — cùng một biên độ cho mọi căn. Nhưng sai số thực phụ thuộc rất
mạnh vào phân khúc và vùng: một căn 2 phòng ngủ ở Hoàng Mai (hàng trăm mẫu
tương tự) và một biệt thự ở huyện ngoại thành (dưới 20 mẫu) không thể có cùng
độ tin cậy. Con số 15% cũng không đến từ đâu cả — không ai kiểm chứng rằng giá
thật rơi vào khoảng đó bao nhiêu phần trăm số lần.

BA PHƯƠNG PHÁP ĐƯỢC SO SÁNH TRONG FILE NÀY
------------------------------------------
1. `co_dinh`  — nhân ±15% (cách hiện tại). Làm mốc đối chứng.

2. `quantile` — huấn luyện ba model XGBoost với hàm mất mát pinball tại các
   phân vị 0,05 / 0,50 / 0,95. Khoảng rộng hẹp khác nhau tuỳ từng căn, nhưng
   KHÔNG có bảo đảm nào về độ phủ — model có thể tự tin thái quá.

3. `cqr` — Conformalized Quantile Regression (Romano và cộng sự, 2019).
   Lấy khoảng từ (2), rồi hiệu chỉnh bằng một tập CALIBRATION riêng biệt:

       điểm bất tuân  E_i = max(q_lo(x_i) − y_i,  y_i − q_hi(x_i))
       Q = phân vị (1−α) của {E_i} (có hiệu chỉnh mẫu hữu hạn)
       khoảng cuối cùng = [q_lo(x) − Q,  q_hi(x) + Q]

   ĐÂY LÀ PHƯƠNG PHÁP ĐƯỢC ĐỀ XUẤT. Nó vừa giữ được tính thích nghi của (2)
   — khoảng hẹp ở nơi dữ liệu dày, rộng ở nơi dữ liệu thưa — vừa có BẢO ĐẢM
   TOÁN HỌC: với giả thiết dữ liệu hoán vị được (exchangeable), độ phủ trên
   dữ liệu mới ≥ 1−α. Đây là điều mà ±15% không bao giờ có được.

LÀM VIỆC TRONG KHÔNG GIAN LOG
-----------------------------
Giá bất động sản lệch phải rất mạnh và sai số mang tính NHÂN chứ không phải
CỘNG (sai 10% ở căn 3 tỷ khác hẳn sai 300 triệu ở căn 30 tỷ). Vì vậy mọi thứ
được học trên log(giá), và khoảng tin cậy được đưa ngược về VND bằng expm1 —
kết quả là khoảng KHÔNG đối xứng quanh giá dự đoán, đúng với bản chất bài toán.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split

from train_eval import (RANDOM_STATE, apply_target_encoding, fit_target_encoding,
                        make_model)

ALPHA = 0.10          # mục tiêu độ phủ 90%
Q_LO, Q_HI = 0.05, 0.95


# =============================================================================
# XÂY MA TRẬN ĐẶC TRƯNG (encoding học từ train, giống train_eval)
# =============================================================================

def _build_matrices(train, others, numeric_cols, categorical_cols, y_train,
                    smoothing=10.0):
    X_train = train[numeric_cols].copy()
    X_others = [o[numeric_cols].copy() for o in others]
    for col in categorical_cols:
        table, fallback = fit_target_encoding(train, col, y_train, m=smoothing)
        X_train[f"{col}_te"] = apply_target_encoding(train, col, table, fallback)
        for o, Xo in zip(others, X_others):
            Xo[f"{col}_te"] = apply_target_encoding(o, col, table, fallback)
    return X_train, X_others


# =============================================================================
# CÁC PHƯƠNG PHÁP TẠO KHOẢNG
# =============================================================================

def _fit_quantile_models(X, y):
    """Ba model tại phân vị dưới / giữa / trên, dùng hàm mất mát pinball."""
    models = {}
    for name, q in (("lo", Q_LO), ("mid", 0.50), ("hi", Q_HI)):
        m = make_model(objective="reg:quantileerror", quantile_alpha=q)
        m.fit(X, y)
        models[name] = m
    return models


def _conformal_offset(y_cal, lo_cal, hi_cal, alpha=ALPHA) -> float:
    """Lượng nới khoảng, tính trên tập calibration.

    Hiệu chỉnh mẫu hữu hạn ceil((n+1)(1-α))/n là phần bắt buộc để có bảo đảm
    độ phủ — bỏ nó đi thì phương pháp mất tính chặt chẽ về mặt lý thuyết.
    """
    scores = np.maximum(lo_cal - y_cal, y_cal - hi_cal)
    n = len(scores)
    level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    return float(np.quantile(scores, level, method="higher"))


# =============================================================================
# ĐÁNH GIÁ: ĐỘ PHỦ VÀ ĐỘ RỘNG
# =============================================================================

def evaluate_intervals(
    df: pd.DataFrame,
    target_col: str,
    numeric_cols: list[str],
    categorical_cols: list[str],
    n_folds: int = 5,
    band_co_dinh: float = 0.15,
    alpha: float = ALPHA,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """So sánh ba phương pháp bằng đánh giá chéo.

    Trả về (bảng tổng hợp, bảng chi tiết từng dòng out-of-fold).

    Trong mỗi fold, phần train còn được chia tiếp thành train-proper (75%) và
    calibration (25%). Tập calibration KHÔNG được dùng để huấn luyện — nếu dùng
    thì điểm bất tuân sẽ lạc quan và bảo đảm độ phủ mất hiệu lực.
    """
    df = df.reset_index(drop=True)
    numeric_cols = [c for c in numeric_cols if c in df.columns]
    categorical_cols = [c for c in categorical_cols if c in df.columns]
    y_log = np.log1p(df[target_col].astype(float))

    rows = []
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    for fold, (tr_idx, te_idx) in enumerate(kf.split(df)):
        tr_all, te = df.iloc[tr_idx], df.iloc[te_idx]
        y_tr_all, y_te = y_log.iloc[tr_idx], y_log.iloc[te_idx]

        tr, cal, y_tr, y_cal = train_test_split(
            tr_all, y_tr_all, test_size=0.25, random_state=RANDOM_STATE)

        X_tr, (X_cal, X_te) = _build_matrices(
            tr, [cal, te], numeric_cols, categorical_cols, y_tr)

        # --- điểm dự đoán (dùng chung cho cả ba phương pháp) ---
        point = make_model()
        point.fit(X_tr, y_tr)
        pred_te = point.predict(X_te)

        # --- quantile ---
        qm = _fit_quantile_models(X_tr, y_tr)
        lo_te, hi_te = qm["lo"].predict(X_te), qm["hi"].predict(X_te)
        lo_cal, hi_cal = qm["lo"].predict(X_cal), qm["hi"].predict(X_cal)

        # --- conformal: nới khoảng bằng tập calibration ---
        Q = _conformal_offset(y_cal.to_numpy(), lo_cal, hi_cal, alpha=alpha)

        actual = np.expm1(y_te.to_numpy())
        pred = np.expm1(pred_te)

        block = pd.DataFrame({
            "fold": fold,
            "idx": te_idx,
            "gia_that": actual,
            "gia_du_doan": pred,
            "co_dinh_lo": pred * (1 - band_co_dinh),
            "co_dinh_hi": pred * (1 + band_co_dinh),
            "quantile_lo": np.expm1(lo_te),
            "quantile_hi": np.expm1(hi_te),
            "cqr_lo": np.expm1(lo_te - Q),
            "cqr_hi": np.expm1(hi_te + Q),
            "Q_conformal": Q,
        })
        rows.append(block)

    detail = pd.concat(rows, ignore_index=True)

    summary = []
    for ten, lo_col, hi_col, mo_ta in [
        ("Cố định ±15% (cách hiện tại)", "co_dinh_lo", "co_dinh_hi",
         "không có bảo đảm, không thích nghi"),
        ("Quantile regression", "quantile_lo", "quantile_hi",
         "thích nghi nhưng không có bảo đảm"),
        ("CQR (quantile + conformal)", "cqr_lo", "cqr_hi",
         "thích nghi VÀ có bảo đảm ≥90%"),
    ]:
        phu = ((detail["gia_that"] >= detail[lo_col])
               & (detail["gia_that"] <= detail[hi_col]))
        rong_tuong_doi = (detail[hi_col] - detail[lo_col]) / detail["gia_du_doan"]
        summary.append({
            "Phương pháp": ten,
            "Độ phủ thực tế (%)": round(phu.mean() * 100, 1),
            "Mục tiêu (%)": round((1 - alpha) * 100, 1),
            "Độ rộng TB (± % giá)": round(rong_tuong_doi.mean() / 2 * 100, 1),
            "Độ rộng trung vị": round(rong_tuong_doi.median() / 2 * 100, 1),
            "Ghi chú": mo_ta,
        })
    return pd.DataFrame(summary), detail


def coverage_by_group(detail: pd.DataFrame, df: pd.DataFrame,
                      group_col: str, lo_col="cqr_lo", hi_col="cqr_hi",
                      min_n: int = 20) -> pd.DataFrame:
    """Độ phủ và độ rộng tách theo nhóm — kiểm tra tính thích nghi.

    Đây là bảng chứng minh điều quan trọng nhất: khoảng tin cậy PHẢI hẹp ở nơi
    dữ liệu dày và rộng ở nơi dữ liệu thưa. Nếu độ rộng gần như nhau ở mọi nhóm
    thì phương pháp chưa thực sự thích nghi, chỉ là ±15% viết phức tạp hơn.
    """
    d = detail.copy()
    d[group_col] = df[group_col].to_numpy()[d["idx"].to_numpy()]
    d["phu"] = (d["gia_that"] >= d[lo_col]) & (d["gia_that"] <= d[hi_col])
    d["rong"] = (d[hi_col] - d[lo_col]) / d["gia_du_doan"] / 2 * 100
    g = d.groupby(group_col).agg(
        so_mau=("phu", "size"), do_phu=("phu", "mean"), do_rong=("rong", "mean"))
    g = g[g["so_mau"] >= min_n].copy()
    g["do_phu"] = (g["do_phu"] * 100).round(1)
    g["do_rong"] = g["do_rong"].round(1)
    g = g.rename(columns={"so_mau": "Số mẫu", "do_phu": "Độ phủ (%)",
                          "do_rong": "Độ rộng (± % giá)"})
    return g.sort_values("Số mẫu", ascending=False)


# =============================================================================
# MÔ HÌNH DÙNG CHO ỨNG DỤNG THỰC TẾ
# =============================================================================

class ConformalValuer:
    """Bộ định giá kèm khoảng tin cậy, dùng cho API và giao diện.

    Cách dùng:
        v = ConformalValuer(numeric_cols, categorical_cols).fit(df, "TARGET_gia_vnd")
        v.predict_interval(can_ho_moi)   -> {gia, khoang_duoi, khoang_tren, do_rong}

    Thuộc tính `Q` chính là lượng nới conformal — nên ghi lại vào báo cáo, và
    nên tính lại mỗi lần huấn luyện lại model.
    """

    def __init__(self, numeric_cols: list[str], categorical_cols: list[str],
                 alpha: float = ALPHA, smoothing: float = 10.0,
                 point_estimator: str = "q50"):
        """point_estimator — chọn model nào làm con số hiển thị:

            "q50"         model quantile tại trung vị (MẶC ĐỊNH)
            "binh_phuong" model hồi quy bình phương thông thường

        VÌ SAO MẶC ĐỊNH LÀ q50 — có đo đạc, không phải tuỳ thích:
        Model bình phương chính xác hơn một chút, nhưng nó thuộc một họ khác
        với hai model quantile tạo ra khoảng, nên con số hiển thị đôi khi nằm
        lệch hẳn về một đầu khoảng. Với người dùng, "4,0 tỷ — khoảng 3,5 đến
        7,5 tỷ" trông như hệ thống bị lỗi, dù về mặt thống kê vẫn hợp lệ.

        Đo trên dữ liệu sạch:
                            MAPE (5-fold)      tỉ lệ điểm lệch xa tâm khoảng
            bình phương     17,3% / 28,7%      27,1%
            q50             17,5% / 29,7%      13,7%
                            (chung cư / nhà đất)

        Đổi 0,3 điểm MAPE lấy việc giảm một nửa số lần hiển thị khó hiểu là
        đánh đổi đáng làm cho một sản phẩm. Nếu chỉ quan tâm độ chính xác điểm
        (ví dụ khi lập bảng so sánh trong báo cáo), hãy đặt "binh_phuong".
        Trung vị cũng là đại lượng trung tâm hợp lý hơn cho phân phối lệch phải:
        "một nửa số căn tương tự đắt hơn mức này" là câu người dùng hiểu ngay.
        """
        if point_estimator not in {"q50", "binh_phuong"}:
            raise ValueError("point_estimator phải là 'q50' hoặc 'binh_phuong'")
        self.numeric_cols = list(numeric_cols)
        self.categorical_cols = list(categorical_cols)
        self.alpha = alpha
        self.smoothing = smoothing
        self.point_estimator = point_estimator

    def fit(self, df: pd.DataFrame, target_col: str) -> "ConformalValuer":
        df = df.reset_index(drop=True)
        self.numeric_cols = [c for c in self.numeric_cols if c in df.columns]
        self.categorical_cols = [c for c in self.categorical_cols if c in df.columns]
        y = np.log1p(df[target_col].astype(float))

        tr, cal, y_tr, y_cal = train_test_split(
            df, y, test_size=0.25, random_state=RANDOM_STATE)

        self._enc = {}
        X_tr = tr[self.numeric_cols].copy()
        X_cal = cal[self.numeric_cols].copy()
        for col in self.categorical_cols:
            table, fallback = fit_target_encoding(tr, col, y_tr, m=self.smoothing)
            self._enc[col] = (table, fallback)
            X_tr[f"{col}_te"] = apply_target_encoding(tr, col, table, fallback)
            X_cal[f"{col}_te"] = apply_target_encoding(cal, col, table, fallback)

        self.feature_names_ = list(X_tr.columns)
        self.point_model_ = make_model().fit(X_tr, y_tr)
        self.q_models_ = _fit_quantile_models(X_tr, y_tr)
        self.Q = _conformal_offset(
            y_cal.to_numpy(),
            self.q_models_["lo"].predict(X_cal),
            self.q_models_["hi"].predict(X_cal),
            alpha=self.alpha,
        )
        self.n_train_ = len(tr)
        self.n_cal_ = len(cal)
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        X = frame[self.numeric_cols].copy()
        for col in self.categorical_cols:
            table, fallback = self._enc[col]
            X[f"{col}_te"] = apply_target_encoding(frame, col, table, fallback)
        return X[self.feature_names_]

    @property
    def display_model_(self):
        """Model dùng cho con số hiển thị — xem giải thích ở __init__."""
        return (self.q_models_["mid"] if self.point_estimator == "q50"
                else self.point_model_)

    def predict_interval(self, frame: pd.DataFrame) -> pd.DataFrame:
        X = self.transform(frame)
        gia = np.expm1(self.display_model_.predict(X))
        lo = np.expm1(self.q_models_["lo"].predict(X) - self.Q)
        hi = np.expm1(self.q_models_["hi"].predict(X) + self.Q)
        return pd.DataFrame({
            "gia": gia, "khoang_duoi": lo, "khoang_tren": hi,
            "do_rong_phan_tram": (hi - lo) / gia / 2 * 100,
        }, index=frame.index)

    # =========================================================================
    # GHI RA FILE VÀ NẠP LẠI
    # =========================================================================
    # VÌ SAO TRƯỚC ĐÂY CỐ TÌNH HUẤN LUYỆN LẠI MỖI LẦN KHỞI ĐỘNG
    #
    # Ghi chú cũ trong `valuation_service._nap` nêu hai lý do, và cả hai đều
    # đúng: (1) tránh model lệch pha với dữ liệu — lỗi kinh điển khi file model
    # và file dữ liệu được cập nhật rời nhau; (2) lượng nới conformal Q phải
    # được tính trên đúng dữ liệu đang dùng.
    #
    # Nhưng cái giá của nó đo được: khởi động lạnh 17 giây (chung cư 4,4s +
    # nhà đất 12,2s), trong đó riêng huấn luyện là 8,6s. Trên Streamlit Cloud
    # máy yếu hơn và app NGỦ sau một lúc không ai vào — nghĩa là người mở link
    # ngồi nhìn màn trắng gần một phút. Với một sản phẩm đem đi bảo vệ thì đó
    # là hỏng, không phải chậm.
    #
    # CÁCH GIỮ NGUYÊN BẢO ĐẢM MÀ VẪN NHANH: không bỏ điều kiện, mà KIỂM nó.
    # Lúc ghi, model mang theo "vân tay" của đúng bộ dữ liệu và đúng cấu hình
    # đã sinh ra nó. Lúc nạp, vân tay được tính lại và đem so. Lệch một chút
    # là `nap_tu` TRẢ VỀ None — người gọi tự huấn luyện lại. Model lệch pha
    # vẫn không thể lọt, chỉ khác là giờ ta chứng minh được điều đó thay vì
    # né nó bằng cách huấn luyện lại mọi lúc.
    #
    # BOOSTER GHI BẰNG ĐỊNH DẠNG RIÊNG CỦA XGBOOST, KHÔNG PICKLE.
    # requirements.txt ghi `xgboost>=2.0`, nên bản trên mạng gần như chắc chắn
    # khác bản ở máy này. Pickle qua hai phiên bản khác nhau là kiểu hỏng tệ
    # nhất: có khi nạp được nhưng dự đoán sai lặng lẽ. Định dạng .ubj của
    # XGBoost có cam kết tương thích ngược, và phiên bản vẫn được ghi vào
    # manifest để còn truy được khi có chuyện.

    _PHIEN_BAN_GOI = 1   # tăng lên khi đổi cấu trúc thư mục model
    _TEN_FILE = ("manifest.json", "ma_hoa.json",
                 "point.ubj", "q_lo.ubj", "q_mid.ubj", "q_hi.ubj")

    def _trang_thai(self) -> dict:
        return {
            "phien_ban_goi": self._PHIEN_BAN_GOI,
            "numeric_cols": self.numeric_cols,
            "categorical_cols": self.categorical_cols,
            "feature_names": self.feature_names_,
            "alpha": self.alpha,
            "smoothing": self.smoothing,
            "point_estimator": self.point_estimator,
            "Q": float(self.Q),
            "n_train": int(self.n_train_),
            "n_cal": int(self.n_cal_),
        }

    def luu_ra(self, thu_muc, van_tay: str) -> None:
        """Ghi model ra `thu_muc`, đóng dấu `van_tay` của dữ liệu + cấu hình."""
        import json
        from pathlib import Path

        import xgboost

        thu_muc = Path(thu_muc)
        thu_muc.mkdir(parents=True, exist_ok=True)
        # XOÁ ĐÚNG NHỮNG FILE CỦA MÌNH, không `rmtree` cả thư mục.
        #
        # Bản đầu tôi viết `rmtree(thu_muc)` cho sạch, và nó nuốt luôn file
        # du_lieu.parquet mà `chuan_bi_trien_khai` vừa ghi vào cùng thư mục
        # mấy dòng trước. Một hàm ghi model không có quyền xoá thứ nó không
        # tạo ra — thư mục là của người gọi, không phải của nó.
        for f in self._TEN_FILE:
            (thu_muc / f).unlink(missing_ok=True)

        for ten, m in (("point", self.point_model_),
                       ("q_lo", self.q_models_["lo"]),
                       ("q_mid", self.q_models_["mid"]),
                       ("q_hi", self.q_models_["hi"])):
            m.get_booster().save_model(str(thu_muc / f"{ten}.ubj"))

        # Bảng mã hoá mục: chỉ số với số, không có gì cần pickle. Khoá ép về
        # chuỗi vì JSON không giữ được khoá không phải chuỗi — lúc nạp lại
        # phải ép y hệt, nếu không `frame[col].map(table)` trượt hết.
        ma_hoa = {col: {"bang": {str(k): float(v) for k, v in tab.items()},
                        "du_phong": float(fb)}
                  for col, (tab, fb) in self._enc.items()}
        (thu_muc / "ma_hoa.json").write_text(
            json.dumps(ma_hoa, ensure_ascii=False), encoding="utf-8")

        manifest = dict(self._trang_thai())
        manifest["van_tay"] = van_tay
        manifest["xgboost"] = xgboost.__version__
        (thu_muc / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def nap_tu(cls, thu_muc, van_tay: str) -> "ConformalValuer | None":
        """Nạp model đã ghi, HOẶC trả về None kèm lý do in ra stderr.

        Không bao giờ ném lỗi: người gọi chỉ cần biết "dùng được hay không",
        và mọi đường không dùng được đều phải dẫn tới việc huấn luyện lại chứ
        không phải làm sập app.
        """
        import json
        import sys
        from pathlib import Path

        from xgboost import XGBRegressor

        thu_muc = Path(thu_muc)

        def bo(ly_do: str):
            # Lỗi XGBoost kèm cả chục dòng stack trace C++ — cắt lấy dòng đầu.
            # Log dài tới mức không ai đọc thì cũng như không có log.
            ly_do = ly_do.splitlines()[0][:160]
            print(f"[MODEL] không dùng bản đã ghi ({thu_muc.name}): {ly_do}"
                  f" — huấn luyện lại.", file=sys.stderr)
            return None

        try:
            mf = json.loads((thu_muc / "manifest.json").read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            return bo(f"không đọc được manifest ({e.__class__.__name__})")

        if mf.get("phien_ban_goi") != cls._PHIEN_BAN_GOI:
            return bo("cấu trúc gói đã đổi")
        if mf.get("van_tay") != van_tay:
            return bo("vân tay dữ liệu/cấu hình không khớp")

        try:
            v = cls(mf["numeric_cols"], mf["categorical_cols"],
                    alpha=mf["alpha"], smoothing=mf["smoothing"],
                    point_estimator=mf["point_estimator"])
            v.feature_names_ = list(mf["feature_names"])
            v.Q = float(mf["Q"])
            v.n_train_, v.n_cal_ = int(mf["n_train"]), int(mf["n_cal"])

            mh = json.loads((thu_muc / "ma_hoa.json").read_text(encoding="utf-8"))
            v._enc = {col: (pd.Series(d["bang"], dtype=float), d["du_phong"])
                      for col, d in mh.items()}

            def _nap_mot(ten):
                m = XGBRegressor()
                m.load_model(str(thu_muc / f"{ten}.ubj"))
                return m

            v.point_model_ = _nap_mot("point")
            v.q_models_ = {"lo": _nap_mot("q_lo"), "mid": _nap_mot("q_mid"),
                           "hi": _nap_mot("q_hi")}
        except Exception as e:                      # noqa: BLE001
            return bo(f"{e.__class__.__name__}: {e}")

        # Mã hoá mục phải phủ đúng những cột model chờ đợi. Thiếu một cột thì
        # `transform` ném KeyError mãi tới lúc người dùng bấm định giá.
        thieu = [c for c in v.categorical_cols if c not in v._enc]
        if thieu:
            return bo(f"thiếu bảng mã hoá cho {thieu}")
        return v


def van_tay_cau_hinh(*phan) -> str:
    """Băm cấu hình thành một chuỗi ngắn, để đóng dấu lên model đã ghi.

    Gộp vào đây MỌI thứ mà đổi đi thì model cũ thành sai: danh sách cột, alpha,
    các phân vị, random_state. Nhờ vậy sửa cấu hình trong code là gói model cũ
    tự động bị loại — không phải nhớ xoá tay, mà quên xoá tay đúng là cách
    model lệch pha lọt được vào bản chạy thật.
    """
    import hashlib
    import json

    thanh_phan = list(phan) + [ALPHA, Q_LO, Q_HI, RANDOM_STATE]
    xau = json.dumps(thanh_phan, ensure_ascii=False, sort_keys=True,
                     default=str)
    return hashlib.sha256(xau.encode("utf-8")).hexdigest()[:16]
