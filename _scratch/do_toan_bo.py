"""Đo lại TOÀN BỘ chỉ số của hệ thống bằng một lệnh.

    python _scratch/do_toan_bo.py            (đầy đủ, ~3 phút)
    python _scratch/do_toan_bo.py --nhanh    (bỏ đánh giá chéo, ~15 giây)

VÌ SAO CẦN FILE NÀY
-------------------
`TRANG_THAI.md` đang ghi số của ngày 01/09 và sai gần hết dòng: nhà đất ghi
12.294 tin trong khi thực tế đã 14.365, MAPE ghi 22,06% trong khi đo lại ra
khác. Sai là vì mỗi con số trước đây đo bằng một lệnh riêng, ở một thời điểm
riêng, rồi chép tay vào file — chép tay thì lần nào cũng sót.

Một lệnh in ra hết, đúng thứ tự các dòng trong TRANG_THAI.md, là để lần sau
cập nhật chỉ còn việc chép cả khối. Và quan trọng hơn cho luận văn: người
chấm hỏi "số này ở đâu ra" thì có một file chạy lại được để chỉ vào.

MỌI CON SỐ ĐỀU ĐO TRÊN out-of-fold, KHÔNG ĐO TRÊN TẬP HUẤN LUYỆN.
"""

from __future__ import annotations

# --- tìm module dù dự án bố trí kiểu nào -------------------------------------
import sys as _sys
from pathlib import Path as _Path
_GOC = _Path(__file__).resolve().parent.parent
for _d in (_GOC, _GOC / "pipeline"):
    if _d.is_dir() and str(_d) not in _sys.path:
        _sys.path.insert(0, str(_d))
# -----------------------------------------------------------------------------

import json
import sys
import time

import numpy as np
import pandas as pd

import ban_do as bd
import valuation_service as vs
from train_eval import cross_validate

NHANH = "--nhanh" in sys.argv

# Ngưỡng nhiễu: ĐỊNH NGHĨA KHOÁ, đừng sửa nếu không ghi lại lý do.
#
# Đây là phần dễ bị "chỉnh cho đẹp" nhất trong cả bài. Nhóm được định nghĩa
# bằng đúng năm trường dưới đây; thêm hay bớt một trường là con số nhảy hẳn —
# đã đo: thêm `loai_hinh` vào thì nhà đất ra 16,46% thay vì 18,51%, lệch 2
# điểm. Trên toàn dải các định nghĩa hợp lý, con số chạy từ 13,47% tới 19,97%,
# tức là một khoảng 6,5 điểm. Nên khi trích dẫn phải LUÔN kèm định nghĩa, và
# không bao giờ được so ngưỡng nhiễu tính theo một định nghĩa với MAPE của một
# cấu hình khác.
NHOM_NHIEU = ["du_an_clean", "phuong_moi", "so_phong_ngu", "so_phong_vs"]
BIN_DIEN_TICH_M2 = 5


def nguong_nhieu(df: pd.DataFrame, cot_dt: str) -> dict:
    """Trung bình KHÔNG trọng số của hệ số biến thiên trong từng nhóm."""
    d = df.dropna(subset=[cot_dt, "TARGET_gia_vnd"]).copy()
    d["_bin"] = (d[cot_dt] // BIN_DIEN_TICH_M2).astype(int)
    khoa = [c for c in NHOM_NHIEU if c in d.columns] + ["_bin"]
    g = d.groupby(khoa)["TARGET_gia_vnd"]
    cv = (g.std(ddof=1) / g.mean())[g.size() >= 2].dropna()
    return {"nguong_pct": float(cv.mean() * 100), "so_nhom": int(len(cv)),
            "so_tin_trong_nhom": int(g.size()[g.size() >= 2].sum())}


def do_mot(loai: str) -> dict:
    t0 = time.time()
    df = vs.nap(loai)["df"]
    cfg = vs.nap(loai)["cfg"]
    cot_dt = cfg.cot_dien_tich
    ra: dict = {"loai": loai, "so_tin": len(df)}

    # --- độ phủ toạ độ ---
    if "nguon_toa_do" in df.columns:
        m = df["nguon_toa_do"]
        ra["toa_do_chinh_xac_pct"] = float(
            m.isin(vs.TOA_DO_GHIM_DUOC).mean() * 100)
        ra["toa_do_theo_muc"] = {
            str(k): int(v) for k, v in m.value_counts().items()}

    # --- phường ---
    g = df.dropna(subset=["phuong_moi"]).groupby("phuong_moi").size()
    ra["so_phuong_co_du_lieu"] = int(len(g))
    ra["so_phuong_du_20"] = int((g >= bd.N_TOI_THIEU_PHUONG).sum())
    ra["so_phuong_duoi_30"] = int((g < 30).sum())
    ra["so_phuong_duoi_10"] = int((g < 10).sum())

    # --- ngưỡng nhiễu ---
    ra["nhieu"] = nguong_nhieu(df, cot_dt)

    # --- khoảng tin cậy (lấy từ chính model đang chạy thật) ---
    v = vs.nap(loai)["valuer"]
    kq = v.predict_interval(df)
    that = df["TARGET_gia_vnd"].astype(float)
    trong = (that >= kq["khoang_duoi"]) & (that <= kq["khoang_tren"])
    ra["ktc"] = {
        "muc_tieu_pct": round((1 - v.alpha) * 100, 1),
        "do_rong_trung_vi_pct": float(kq["do_rong_phan_tram"].median()),
        "Q_conformal": float(v.Q),
        "n_train": int(v.n_train_), "n_hieu_chuan": int(v.n_cal_),
        # ĐỘ PHỦ NÀY LÀ TRÊN CHÍNH TẬP ĐÃ HUẤN LUYỆN nên nó LẠC QUAN — ghi ra
        # để tham khảo, KHÔNG được dùng làm con số báo cáo. Độ phủ thật phải
        # đo out-of-fold, và đó là việc của `intervals.evaluate_intervals`.
        "do_phu_tren_tap_huan_luyen_pct": float(trong.mean() * 100),
    }

    if NHANH:
        ra["giay"] = round(time.time() - t0, 1)
        return ra

    # --- MAPE / R² bằng đánh giá chéo 5 fold, out-of-fold ---
    cv = cross_validate(df, "TARGET_gia_vnd",
                        list(cfg.cot_so), list(cfg.cot_muc))
    du_doan = cv["oof_pred"]
    sai_so = np.abs(du_doan - that) / that * 100
    ra["cv"] = {
        "MAPE": cv["MAPE"], "MAPE_std": cv["MAPE_std"], "R2": cv["R2"],
        "sai_so_trung_vi_pct": float(np.median(sai_so)),
        "trong_10_pct": float((sai_so <= 10).mean() * 100),
        "trong_20_pct": float((sai_so <= 20).mean() * 100),
    }
    ra["cach_nguong_nhieu_diem"] = cv["MAPE"] - ra["nhieu"]["nguong_pct"]
    ra["giay"] = round(time.time() - t0, 1)
    return ra


def main() -> int:
    print("Đo toàn bộ hệ thống" + (" (chế độ nhanh)" if NHANH else "")
          + f" — {time.strftime('%d/%m/%Y %H:%M')}\n")
    ket = [do_mot(l) for l in ("chungcu", "nhadat")]

    def hang(nhan, lay, dinh="{}"):
        o = [dinh.format(lay(k)) if lay(k) is not None else "—" for k in ket]
        print(f"| {nhan:<38} | {o[0]:>14} | {o[1]:>14} |")

    print(f"| {'':<38} | {'Chung cư':>14} | {'Nhà đất':>14} |")
    print(f"|{'-'*40}|{'-'*16}|{'-'*16}|")
    hang("Số tin dùng được", lambda k: k["so_tin"], "{:,}")
    if not NHANH:
        hang("MAPE (5-fold, out-of-fold)", lambda k: k["cv"]["MAPE"], "{:.2f}%")
        hang("  độ lệch giữa các fold", lambda k: k["cv"]["MAPE_std"], "±{:.2f}")
        hang("Sai số trung vị", lambda k: k["cv"]["sai_so_trung_vi_pct"], "{:.2f}%")
        hang("Trong vòng 10%", lambda k: k["cv"]["trong_10_pct"], "{:.1f}%")
        hang("Trong vòng 20%", lambda k: k["cv"]["trong_20_pct"], "{:.1f}%")
        hang("R²", lambda k: k["cv"]["R2"], "{:.3f}")
    hang("Ngưỡng nhiễu (định nghĩa khoá)",
         lambda k: k["nhieu"]["nguong_pct"], "{:.2f}%")
    hang("  số nhóm dùng để tính", lambda k: k["nhieu"]["so_nhom"], "{:,}")
    if not NHANH:
        hang("Khoảng còn cải thiện được",
             lambda k: k["cach_nguong_nhieu_diem"], "{:+.2f} điểm")
    hang("Khoảng tin cậy — mục tiêu",
         lambda k: k["ktc"]["muc_tieu_pct"], "{:.0f}%")
    hang("  độ rộng trung vị", lambda k: k["ktc"]["do_rong_trung_vi_pct"], "±{:.1f}%")
    hang("  lượng nới conformal Q", lambda k: k["ktc"]["Q_conformal"], "{:.4f}")
    hang("Toạ độ ghim được", lambda k: k.get("toa_do_chinh_xac_pct"), "{:.1f}%")
    hang("Phường có dữ liệu", lambda k: k["so_phuong_co_du_lieu"], "{}/113")
    hang(f"  đủ {bd.N_TOI_THIEU_PHUONG} căn để tô bản đồ",
         lambda k: k["so_phuong_du_20"], "{}")
    hang("  dưới 30 căn (cảnh báo nhẹ)", lambda k: k["so_phuong_duoi_30"], "{}")
    hang("  dưới 10 căn (cảnh báo mạnh)", lambda k: k["so_phuong_duoi_10"], "{}")
    hang("Thời gian đo", lambda k: k["giay"], "{:.0f}s")

    ra = _GOC / "data_clean" / "do_toan_bo.json"
    ra.write_text(json.dumps(ket, ensure_ascii=False, indent=2,
                             default=float), encoding="utf-8")
    print(f"\nSố chi tiết: {ra}")
    return 0


if __name__ == "__main__":

    raise SystemExit(main())
