"""
train_eval.py — Khung đánh giá dùng chung + bảng nghiên cứu tách biệt (ablation).

Đây là module quan trọng nhất về mặt học thuật trong toàn bộ pipeline, vì nó
quyết định mọi con số trong chương thực nghiệm có đáng tin hay không.

HAI ĐIỀU PHẢI LÀM ĐÚNG
----------------------
1) TARGET ENCODING TÍNH TRONG TỪNG FOLD.
   Cách sai (và là cách bản cũ đang làm): tính giá trung bình theo phường trên
   toàn bộ dữ liệu, lưu thành cột, rồi mới chia train/test. Khi đó giá của
   chính dòng test đã góp phần tạo ra đặc trưng của nó — model được xem trước
   đáp án. Với 85/140 phường có dưới 10 mẫu, mức độ rò rỉ là rất lớn.

   Cách đúng: trong mỗi fold, chỉ dùng phần TRAIN để tính bảng mã hoá, rồi áp
   bảng đó lên phần TEST. Phường chưa từng thấy thì lùi về giá trị trung bình
   toàn cục.

2) LÀM MỊN (SMOOTHING).
   Một phường chỉ có 2 tin thì giá trung bình của nó là con số rất nhiễu.
   Công thức làm mịn kéo nó về phía trung bình toàn cục theo số mẫu:

       mã_hoá = (trung_bình_nhóm × n + trung_bình_toàn_cục × m) / (n + m)

   với m là "số mẫu ảo" (mặc định 10). Nhóm càng ít mẫu, càng bị kéo về trung
   bình chung — đúng với trực giác: ít bằng chứng thì đừng tin nhiều.

CHỈ SỐ ĐÁNH GIÁ
---------------
MAPE là chỉ số chính vì nó dễ diễn giải với người dùng cuối ("sai trung bình
15%") và không bị các căn giá trị lớn chi phối như MAE. Kèm thêm MAE và R² để
đối chiếu. Huấn luyện trên log(giá) vì phân phối giá lệch phải rất mạnh.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from xgboost import XGBRegressor

RANDOM_STATE = 42
N_FOLDS = 5


# =============================================================================
# TARGET ENCODING AN TOÀN
# =============================================================================

def fit_target_encoding(train: pd.DataFrame, col: str, y: pd.Series,
                        m: float = 10.0) -> tuple[pd.Series, float]:
    """Học bảng mã hoá TỪ PHẦN TRAIN. Trả về (bảng tra, giá trị dự phòng)."""
    global_mean = float(y.mean())
    agg = y.groupby(train[col]).agg(["mean", "count"])
    smoothed = (agg["mean"] * agg["count"] + global_mean * m) / (agg["count"] + m)
    return smoothed, global_mean


def apply_target_encoding(frame: pd.DataFrame, col: str,
                          table: pd.Series, fallback: float) -> pd.Series:
    """Áp bảng mã hoá; giá trị chưa từng thấy -> dùng trung bình toàn cục."""
    return frame[col].map(table).astype(float).fillna(fallback)


# =============================================================================
# ĐÁNH GIÁ CHÉO
# =============================================================================

def make_model(**kwargs) -> XGBRegressor:
    params = dict(
        n_estimators=600, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
        reg_lambda=1.0, random_state=RANDOM_STATE, n_jobs=-1, verbosity=0,
    )
    params.update(kwargs)
    return XGBRegressor(**params)


def cross_validate(
    df: pd.DataFrame,
    target_col: str,
    numeric_cols: list[str],
    categorical_cols: list[str],
    text_col: str | None = None,
    dac_trung_theo_fold=None,
    n_text_components: int = 60,
    n_folds: int = N_FOLDS,
    smoothing: float = 10.0,
) -> dict:
    """Đánh giá chéo k-fold, mọi phép biến đổi học TỪ TRAIN trong từng fold.

    text_col: nếu truyền vào, TF-IDF + SVD của cột văn bản sẽ được thêm làm đặc
        trưng. Bộ TF-IDF cũng chỉ được fit trên phần train của fold — nếu fit
        trên toàn bộ dữ liệu thì từ vựng của tập test rò rỉ sang, tuy nhẹ nhưng
        vẫn là rò rỉ.

    dac_trung_theo_fold: hàm (train_df, test_df) -> (bảng_train, bảng_test) để
        sinh đặc trưng phải tính LẠI trong từng fold. Cần cho những đặc trưng
        suy ra từ GIÁ của các dòng khác — ví dụ "giá trung vị của k căn gần
        nhất". Tính một lần trên toàn bộ dữ liệu rồi mới chia fold là rò rỉ
        mục tiêu: mỗi dòng test đã góp giá của chính nó vào đặc trưng của các
        dòng train quanh nó, và ngược lại.
    """
    df = df.reset_index(drop=True)
    numeric_cols = [c for c in numeric_cols if c in df.columns]
    categorical_cols = [c for c in categorical_cols if c in df.columns]

    # Ép các cột đã khai là số về kiểu số. Khi gộp nhiều nguồn, một cột toàn
    # NaN ở nguồn này sẽ làm cả cột thành kiểu object sau khi concat, và
    # XGBoost từ chối thẳng. Ép ở đây thay vì bắt mọi nơi gọi phải nhớ.
    for c in numeric_cols:
        if df[c].dtype == object:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    y_log = np.log1p(df[target_col].astype(float))
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)

    mapes, maes, r2s = [], [], []
    oof_pred = np.full(len(df), np.nan)

    for train_idx, test_idx in kf.split(df):
        tr, te = df.iloc[train_idx], df.iloc[test_idx]
        y_tr, y_te = y_log.iloc[train_idx], y_log.iloc[test_idx]

        X_tr = tr[numeric_cols].copy()
        X_te = te[numeric_cols].copy()

        for col in categorical_cols:
            table, fallback = fit_target_encoding(tr, col, y_tr, m=smoothing)
            X_tr[f"{col}_te"] = apply_target_encoding(tr, col, table, fallback)
            X_te[f"{col}_te"] = apply_target_encoding(te, col, table, fallback)

        if text_col and text_col in df.columns:
            from sklearn.decomposition import TruncatedSVD
            from sklearn.feature_extraction.text import TfidfVectorizer
            texts_tr = tr[text_col].fillna("").astype(str)
            texts_te = te[text_col].fillna("").astype(str)
            tfidf = TfidfVectorizer(max_features=8000, ngram_range=(1, 2), min_df=3)
            M_tr = tfidf.fit_transform(texts_tr)
            M_te = tfidf.transform(texts_te)
            k = min(n_text_components, M_tr.shape[1] - 1)
            svd = TruncatedSVD(n_components=k, random_state=RANDOM_STATE)
            S_tr = svd.fit_transform(M_tr)
            S_te = svd.transform(M_te)
            for i in range(k):
                X_tr[f"txt_{i}"] = S_tr[:, i]
                X_te[f"txt_{i}"] = S_te[:, i]

        if dac_trung_theo_fold is not None:
            them_tr, them_te = dac_trung_theo_fold(tr, te)
            for c in them_tr.columns:
                X_tr[c] = them_tr[c].to_numpy()
                X_te[c] = them_te[c].to_numpy()

        model = make_model()
        model.fit(X_tr, y_tr)
        pred_log = model.predict(X_te)

        pred = np.expm1(pred_log)
        actual = np.expm1(y_te.to_numpy())
        oof_pred[test_idx] = pred

        mapes.append(float(np.mean(np.abs(pred - actual) / actual) * 100))
        maes.append(float(np.mean(np.abs(pred - actual))))
        ss_res = float(np.sum((actual - pred) ** 2))
        ss_tot = float(np.sum((actual - actual.mean()) ** 2))
        r2s.append(1 - ss_res / ss_tot)

    return {
        "MAPE": float(np.mean(mapes)), "MAPE_std": float(np.std(mapes)),
        "MAE": float(np.mean(maes)), "R2": float(np.mean(r2s)),
        "n": len(df), "oof_pred": oof_pred,
    }


# =============================================================================
# ĐƯỜNG CONG HỌC TẬP
# =============================================================================

def learning_curve(
    df: pd.DataFrame, target_col: str, numeric_cols: list[str],
    categorical_cols: list[str], fractions=(0.2, 0.4, 0.6, 0.8, 1.0),
    n_repeats: int = 3,
) -> pd.DataFrame:
    """MAPE theo tỉ lệ dữ liệu được dùng.

    MỤC ĐÍCH: trả lời câu "có nên bỏ công crawl thêm dữ liệu không" bằng số
    thay vì bằng cảm tính. Nếu đường cong còn dốc ở mốc 100% thì thêm dữ liệu
    còn giúp; nếu đã phẳng thì công sức nên dồn sang hướng khác.

    Mỗi mốc lặp `n_repeats` lần với hạt giống khác nhau rồi lấy trung bình, vì
    ở mẫu nhỏ kết quả dao động mạnh — một lần chạy duy nhất dễ gây hiểu nhầm.
    """
    rows = []
    for frac in fractions:
        scores = []
        for rep in range(n_repeats):
            if frac >= 1.0:
                sub = df
                if rep > 0:
                    break                      # dùng toàn bộ thì lặp là vô nghĩa
            else:
                sub = df.sample(frac=frac, random_state=RANDOM_STATE + rep)
            res = cross_validate(sub, target_col, numeric_cols, categorical_cols)
            scores.append(res["MAPE"])
        rows.append({
            "Tỉ lệ dữ liệu": f"{frac*100:.0f}%",
            "Số mẫu": int(len(df) * frac),
            "MAPE (%)": round(float(np.mean(scores)), 2),
            "Độ lệch chuẩn": round(float(np.std(scores)), 2),
        })
    return pd.DataFrame(rows)
