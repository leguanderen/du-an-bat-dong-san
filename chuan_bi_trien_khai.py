"""
chuan_bi_trien_khai.py — Chuẩn bị dữ liệu và model cho bản chạy thật.

    python chuan_bi_trien_khai.py

Chạy bước này MỖI KHI dữ liệu hoặc cấu hình đặc trưng thay đổi, rồi mới deploy.
Quên chạy cũng không sao: app tự phát hiện vân tay không khớp và quay về đường
cũ (đọc .xlsx + huấn luyện lại) — chậm, nhưng không bao giờ sai.

VẤN ĐỀ NÓ GIẢI
--------------
App đọc thẳng .xlsx rồi huấn luyện lại XGBoost mỗi lần khởi động. Đo trên máy
phát triển:

                   đọc xlsx   huấn luyện   tổng
      chung cư        1,6 s       2,6 s     4,4 s
      nhà đất         5,6 s       6,0 s    12,2 s
                                          -------
                                           16,6 s

Streamlit Cloud cho app ngủ sau một lúc không ai vào, nên gần như MỌI người mở
link đều gặp lần khởi động lạnh, trên phần cứng yếu hơn máy này. Bước chuẩn bị
đổi cả hai nửa: .xlsx -> .parquet (đọc nhanh hơn nhiều vì có kiểu dữ liệu sẵn,
không phải bóc XML) và huấn luyện -> nạp model đã ghi.

VÌ SAO KHÔNG PHẢI LÀ "LƯU .PKL" NHƯ HỆ THỐNG CŨ
-----------------------------------------------
Hệ thống cũ nạp .pkl và đó chính là thứ dự án này cố tránh: file model và file
dữ liệu cập nhật rời nhau thì model lệch pha, mà không có gì báo. Ở đây model
mang theo VÂN TAY của đúng bộ dữ liệu + cấu hình đã sinh ra nó; lúc nạp, vân
tay được tính lại và đem so. Lệch là bỏ. Bảo đảm cũ không mất đi, nó chỉ chuyển
từ "né bằng cách luôn huấn luyện lại" sang "kiểm mỗi lần nạp".

Và bước cuối của script này là TỰ KIỂM: nạp lại chính gói vừa ghi, dự đoán trên
200 dòng, so với model vừa huấn luyện trong bộ nhớ. Lệch quá một phần triệu là
báo hỏng và xoá gói đi. Ghi ra một gói sai rồi đem deploy còn tệ hơn là chậm.
"""

from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

# Script này nằm ở thư mục GỐC còn các module nằm trong `pipeline/` — giống
# hệt `app.py`. Ở bản chép phẳng thì tất cả cùng một chỗ, nên dò cả hai.
BASE_DIR = Path(__file__).resolve().parent
for _d in (BASE_DIR / "pipeline", BASE_DIR):
    if _d.is_dir() and str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

import numpy as np                        # noqa: E402
import pandas as pd                       # noqa: E402

import valuation_service as vs            # noqa: E402
from intervals import ConformalValuer     # noqa: E402

# Sai lệch cho phép giữa model trong bộ nhớ và model nạp lại từ file. Đây là
# sai số làm tròn của số thực, không phải "gần đúng là được": cùng một cây,
# cùng một đầu vào thì kết quả phải trùng tới mức này.
DUNG_SAI = 1e-6

# Số dòng đem ra đối chiếu ở bước tự kiểm. 200 là đủ để bắt lỗi bảng mã hoá
# lệch (lỗi đó làm sai hàng loạt, không sai lác đác).
SO_DONG_KIEM = 200


def _ghi_parquet(df: pd.DataFrame, duong_dan) -> None:
    """Ghi .parquet, ép những cột object lẫn kiểu về chuỗi.

    pyarrow từ chối cột object chứa lẫn số với chuỗi — hay gặp ở cột bóc từ
    tin rao. Ép về chuỗi là đúng với ý nghĩa của chúng (chúng là chữ), và
    những cột thật sự là số thì đã được `don_truong_chon` ép sang số từ trước.
    """
    ra = df.copy()
    for c in ra.columns:
        if ra[c].dtype != object:
            continue
        # DUYỆT CẢ CỘT, KHÔNG PHẢI 2000 DÒNG ĐẦU.
        #
        # Bản đầu tôi chỉ nhìn `head(2000)` cho nhanh, và cột `mo_ta` của nhà
        # đất lọt lưới: 2000 dòng đầu toàn chuỗi, nhưng đâu đó phía dưới có
        # một ô là số, nên pyarrow ném ArrowTypeError. Lấy mẫu để quyết định
        # kiểu của cả cột là sai về nguyên tắc — cột chỉ cần MỘT ô lệch kiểu
        # là đã phải ép.
        if len({type(v) for v in ra[c].dropna()}) > 1:
            ra[c] = ra[c].astype("string")
    duong_dan.parent.mkdir(parents=True, exist_ok=True)
    ra.to_parquet(duong_dan, index=False)


def _tu_kiem(loai: str, goi, df: pd.DataFrame, cfg, trong_bo_nho) -> None:
    """Nạp lại gói vừa ghi và đối chiếu dự đoán. Sai thì ném lỗi."""
    pq = goi / vs.TEN_PARQUET
    nap_lai = ConformalValuer.nap_tu(goi, vs.van_tay(pq, cfg, vs.ALPHA_MAC_DINH))
    if nap_lai is None:
        raise RuntimeError(
            f"{loai}: gói vừa ghi mà nạp lại không được — xem lý do in ở trên")

    goc = trong_bo_nho.predict_interval(df.head(SO_DONG_KIEM))

    def so(nhan: str, khac: pd.DataFrame) -> None:
        for cot in ("gia", "khoang_duoi", "khoang_tren"):
            lech = float(np.nanmax(
                np.abs(goc[cot].to_numpy() - khac[cot].to_numpy())
                / np.maximum(np.abs(goc[cot].to_numpy()), 1.0)))
            if not lech < DUNG_SAI:
                raise RuntimeError(
                    f"{loai}: {nhan} cho kết quả khác — cột {cot} lệch "
                    f"{lech:.2e} (cho phép {DUNG_SAI:.0e})")

    # (1) model nạp lại từ file, trên ĐÚNG dữ liệu cũ
    so("model nạp lại", nap_lai.predict_interval(df.head(SO_DONG_KIEM)))

    # (2) VÀ dữ liệu đọc lại từ parquet — bước này mới là bước dễ bỏ sót.
    #
    # Kiểm (1) một mình vẫn để lọt cả một lớp lỗi: parquet đi một vòng có thể
    # đổi kiểu của một cột (số thành chuỗi, ngày thành dấu thời gian), và lúc
    # đó model vẫn chạy, vẫn ra số — chỉ là sai. Bản chạy thật đọc parquet chứ
    # không đọc .xlsx, nên phải đối chiếu đúng thứ nó sẽ đọc.
    dl = pd.read_parquet(pq).head(SO_DONG_KIEM)
    so("dữ liệu đọc lại từ parquet", nap_lai.predict_interval(dl))

    if abs(float(nap_lai.Q) - float(trong_bo_nho.Q)) > 1e-12:
        raise RuntimeError(f"{loai}: lượng nới conformal Q không khớp")


def chuan_bi(loai: str) -> dict:
    t0 = time.time()
    print(f"\n=== {loai}")

    df, nen, cfg, canh_bao = vs._nap_tu_xlsx(loai)
    print(f"  đọc + dọn + láng giềng   {time.time() - t0:5.1f} s  "
          f"({len(df):,} dòng × {df.shape[1]} cột, nền {len(nen):,})"
          .replace(",", "."))

    goi = vs.THU_MUC_GOI / loai
    if goi.exists():
        shutil.rmtree(goi)
    goi.mkdir(parents=True)

    t = time.time()
    pq = goi / vs.TEN_PARQUET
    _ghi_parquet(df, pq)
    mb = pq.stat().st_size / 1048576
    print(f"  ghi parquet              {time.time() - t:5.1f} s  ({mb:.1f} MB)")

    t = time.time()
    valuer = ConformalValuer(list(cfg.cot_so), list(cfg.cot_muc),
                             alpha=vs.ALPHA_MAC_DINH).fit(df, "TARGET_gia_vnd")
    print(f"  huấn luyện               {time.time() - t:5.1f} s  "
          f"(Q = {valuer.Q:.4f}, train {valuer.n_train_}, "
          f"hiệu chuẩn {valuer.n_cal_})")

    valuer.luu_ra(goi, vs.van_tay(pq, cfg, vs.ALPHA_MAC_DINH))

    t = time.time()
    _tu_kiem(loai, goi, df, cfg, valuer)
    print(f"  tự kiểm nạp lại          {time.time() - t:5.1f} s  ✓ khớp")

    return {"loai": loai, "so_dong": len(df), "mb": mb,
            "canh_bao": canh_bao, "Q": float(valuer.Q)}


def _kiem_cho_ghi() -> None:
    """Gói PHẢI nằm trong thư mục dự án, cạnh data_clean.

    Lần chạy đầu tiên nó ghi ra NGOÀI dự án mà không ai biết: hàm dò đường dẫn
    `thu_muc()` khi không thấy thư mục thì đoán `BASE_DIR.parent`, và
    `_trien_khai` lúc ấy chưa tồn tại. Mọi phép kiểm vẫn xanh, vì bên ghi và
    bên đọc dùng chung đúng cái đường dẫn sai đó. Chỉ tới lúc `ls` thư mục dự
    án mới lộ ra là không có gì trong đó — và nếu không lộ ra thì bản deploy
    đã thiếu gói, hoặc trên máy người dùng có một thư mục 15 MB rơi ra ngoài.
    Nên in đường dẫn tuyệt đối ra, và chặn nếu nó không nằm cạnh data_clean.
    """
    goc_du_an = vs.DATA_DIR.parent.resolve()
    dich = vs.THU_MUC_GOI.resolve()
    print(f"  gốc dự án : {goc_du_an}")
    print(f"  ghi gói ra: {dich}")
    if dich.parent != goc_du_an:
        raise SystemExit(
            f"\nDỪNG: gói sẽ ghi ra {dich}, không nằm trong {goc_du_an}.\n"
            f"Ghi ra đó thì nó không vào được repo, và bản trên mạng sẽ chạy "
            f"đường chậm. Kiểm lại `thu_muc_ra` trong gan_toa_do.py.")


def main() -> int:
    print("Chuẩn bị gói triển khai — đọc .xlsx, huấn luyện, ghi ra "
          f"{vs.THU_MUC_GOI.name}/")
    _kiem_cho_ghi()
    ket = []
    for loai in ("chungcu", "nhadat"):
        try:
            ket.append(chuan_bi(loai))
        except Exception as e:                       # noqa: BLE001
            # Xoá gói hỏng đi. Để lại một gói sai thì lần chạy sau app có thể
            # nạp phải nó — mà cả bước này sinh ra là để chuyện đó đừng xảy ra.
            shutil.rmtree(vs.THU_MUC_GOI / loai, ignore_errors=True)
            print(f"\n  ✗ {loai} HỎNG: {e.__class__.__name__}: {e}",
                  file=sys.stderr)
            print("  đã xoá gói hỏng; app sẽ tự quay về đọc .xlsx.",
                  file=sys.stderr)
            return 1

    # Đo lại đúng thứ đang cần cải thiện: khởi động lạnh.
    vs._nap.cache_clear()
    print("\n=== khởi động lạnh sau khi có gói")
    tong = 0.0
    for loai in ("chungcu", "nhadat"):
        t = time.time()
        vs.nap(loai)
        giay = time.time() - t
        tong += giay
        print(f"  {loai:<9} {giay:5.2f} s")
    print(f"  {'tổng':<9} {tong:5.2f} s   (trước khi có gói: 16,6 s)")

    canh = [c for k in ket for c in k["canh_bao"]]
    if canh:
        print(f"\n{len(canh)} cảnh báo dữ liệu — xem stderr ở trên.")
    print("\nXong. Nhớ commit cả thư mục "
          f"{vs.THU_MUC_GOI.name}/ thì bản trên mạng mới nhanh.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
