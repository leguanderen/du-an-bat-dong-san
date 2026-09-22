"""
valuation_service.py — Tầng định giá dùng chung cho giao diện và (sau này) API.

Đây là chỗ `app.py` gọi vào. Mục đích của việc tách riêng file này: logic định
giá không được nằm lẫn trong mã giao diện. Khi dựng FastAPI ở bước sau, nó gọi
đúng module này — không phải chép lại code từ Streamlit.

SO VỚI HỆ THỐNG CŨ
------------------
Bản cũ nạp thẳng model .pkl rồi nhân giá với 0,85 và 1,15 để ra "khoảng tham
khảo". Đo lại thì khoảng đó chỉ phủ 58,5% số tin chung cư và 45,6% nhà đất —
tức sai hơn một nửa số lần với nhà đất.

Bản này thay bằng CQR (conformalized quantile regression), đạt độ phủ 88,0% và
89,7% so với mục tiêu 90%, và kèm giải thích SHAP quy đổi ra phần trăm.

Đổi lại, khoảng rộng hơn nhiều: ±42% và ±61% thay vì ±15%. Đó không phải model
kém đi — đó là mức bất định thật mà con số ±15% đang che giấu.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from explain import explain_one, format_vnd, global_importance
from gan_toa_do import haversine_km, thu_muc, thu_muc_ra
from intervals import ConformalValuer, van_tay_cau_hinh

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = thu_muc("data_clean")

# Dùng bản ĐÃ GÁN TOẠ ĐỘ. Đo bằng đánh giá chéo ghép cặp trên cùng bộ fold:
# thêm toạ độ giảm MAPE chung cư 17,28% -> 16,04% và nhà đất 28,68% -> 27,87%,
# khoảng tin cậy 95% của mức cải thiện là [0,77; 1,71] và [0,23; 1,40] điểm
# phần trăm — tức là không nằm ở phía 0, cải thiện có thật chứ không phải may.
CHUNGCU_FILE = "chungcu_gop_geo.xlsx"
NHADAT_FILE = "nhadat_gop_geo.xlsx"

# Hồ Hoàn Kiếm — mốc "trung tâm" quy ước của Hà Nội.
TRUNG_TAM = (21.0287, 105.8524)
COT_TOA_DO = ("latitude", "longitude", "kc_trung_tam_km")

# Toạ độ đủ chính xác để khái niệm "hàng xóm" có nghĩa. Chỉ nguồn tự công bố
# (batdongsan bóc từ khối JS) mới đạt mức này; các mức còn lại là ước lượng
# theo trung vị nhóm nên nhiều tin dùng chung một điểm.
TOA_DO_CHINH_XAC = ("nguon_goc",)

# Toạ độ đủ tin để dựng bảng tra theo đường phố / dự án: nơi đăng tự công bố,
# suy từ mã tin, hoặc khớp đúng dự án. Ba mức này có sai số p90 = 0 km.
TOA_DO_GHIM_DUOC = ("nguon_goc", "ma_tin", "du_an")

# TẮT đặc trưng "giá k căn gần nhất" trong bản chạy thật, dù nó là ý tưởng tốt.
#
# Hai lý do, cả hai đều đo được:
#
# 1. Trên toàn tập nó chỉ đáng +0,05 điểm MAPE (15,48% -> 15,43%) — vì chỉ
#    1.278/3.929 tin có toạ độ đủ chính xác để hàng xóm có nghĩa, phần còn lại
#    pha loãng hết. Trên riêng nhóm toạ độ chính xác thì nó đáng +0,87 điểm
#    (p = 0,0004), nên đây là chuyện SỐ LƯỢNG dữ liệu chính xác, không phải
#    chuyện ý tưởng sai.
#
# 2. Quan trọng hơn: ở bản chạy thật, đặc trưng này phải tính một lần trên toàn
#    bộ dữ liệu trước khi chia train/hiệu chuẩn. Làm vậy thì phần hiệu chuẩn
#    conformal nhìn model "giỏi" hơn thực tế, và khoảng tin cậy hẹp lại một cách
#    GIẢ TẠO: ±30% thay vì ±42% thật. Khoảng tin cậy là thứ bán được của hệ
#    thống này — thà bỏ 0,05 điểm MAPE còn hơn để nó nói dối.
#
# Bật lại khi nào tỉ lệ tin có toạ độ chính xác vượt quá một nửa, VÀ viết lại
# ConformalValuer để tính đặc trưng này riêng cho phần train và phần hiệu chuẩn.
DUNG_LANG_GIENG = False

# CỘT MÀ GIAO DIỆN CẦN ĐỂ TRƯNG BẰNG CHỨNG, không phải để tính giá.
#
# `tim_comparables` lọc cột trước khi trả về. Danh sách lọc đó viết từ hồi bảng
# chỉ hiện giá và diện tích, nên khi giao diện thêm phần "nội dung tin đã lưu"
# và phần đánh dấu địa chỉ chưa xác minh thì hai phần đó nhận về toàn NaN —
# KHÔNG lỗi nào báo ra, chỉ là ô trống mà người xem tưởng "tin này không có
# mô tả". Tôi phát hiện vì chạy thử đúng căn Định Công/Nguyễn Trãi và thấy
# cột đánh dấu biến mất.
#
# Gom thành một hằng số có tên để hai chỗ không còn lệch nhau: thêm trường mới
# vào phần bằng chứng thì thêm vào đây, không phải đi tìm từng danh sách lọc.
COT_BANG_CHUNG = (
    "nguon_toa_do",                     # địa chỉ xác định được tới mức nào
    "ngay_quan_sat", "nguon",           # tin lưu ngày nào, từ sàn nào
    "dia_chi_goc", "mo_ta",             # nguyên văn tin rao
    "phap_ly", "noi_that", "huong_cua",  # trường chọn đã chuẩn hoá
    "dien_tich_m2", "dien_tich_dat_m2", "dien_tich_su_dung_m2",
    "tong_so_tang", "mat_tien_m", "duong_rong_m", "tang_so",
)

# Mức tin cậy mặc định cho khoảng giá. 0.10 nghĩa là khoảng 90%.
# Có thể hạ xuống 0.20 (khoảng 80%) hoặc 0.50 (khoảng 50%) để khoảng hẹp lại —
# vẫn có bảo chứng thống kê, chỉ là mức đảm bảo thấp hơn. Đây là lựa chọn sản
# phẩm chứ không phải lựa chọn kỹ thuật: khoảng 90% trung thực nhưng rộng,
# khoảng 50% hẹp và dễ đọc nhưng sai 1 trong 2 lần.
ALPHA_MAC_DINH = 0.10


# =============================================================================
# CHUẨN HOÁ GIÁ TRỊ CÁC TRƯỜNG CHỌN
# =============================================================================
# Khâu bóc dữ liệu đôi khi dính chữ của ô bên cạnh vào: "Nội thất đầy đủCăn góc
# Căn góc", "Nội thất cao cấpMã cănBP313". Nếu để nguyên thì mỗi biến thể thành
# một hạng mục riêng, target encoding chia nhỏ dữ liệu ra vô ích. Cắt về đúng
# nhãn gốc bằng cách khớp tiền tố với danh sách nhãn hợp lệ.

NOI_THAT = ("Nội thất cao cấp", "Nội thất đầy đủ", "Hoàn thiện cơ bản",
            "Bàn giao thô")
PHAP_LY_CC = ("Sổ hồng riêng", "Hợp đồng mua bán", "Đang chờ sổ",
              "Hợp đồng đặt cọc")
# HAI SÀN DÙNG HAI BỘ TỪ KHÁC NHAU CHO CÙNG MỘT TRƯỜNG
#
# nhatot ghi "Đã có sổ", "Sổ chung / công chứng vi bằng", "Giấy tờ viết tay";
# batdongsan ghi "Sổ hồng riêng", "Hợp đồng mua bán". Danh sách này ban đầu chỉ
# có bộ từ của nhatot — viết lúc nhà đất chỉ có một nguồn. Khi thêm nguồn
# batdongsan, `chuan_hoa_muc` không khớp được nhãn nào nên trả NaN, và
# 8.641/11.952 tin (72,3%) MẤT TRẮNG trường pháp lý — trong đó 7.625 tin ghi
# "Sổ hồng riêng" hoàn toàn sạch sẽ, không dính lỗi nào. Không một lỗi nào báo
# ra, vì trả NaN là hành vi hợp lệ của hàm đó.
#
# GIỮ HAI NHÃN TÁCH RIÊNG, KHÔNG GỘP "Sổ hồng riêng" VÀO "Đã có sổ": chúng
# không đồng nghĩa ("sổ hồng riêng" là trường hợp mạnh hơn của "đã có sổ"), và
# gộp là tự bịa ra một phép quy đổi giữa hai sàn mà mình không kiểm chứng được.
# Cột `nguon` đã là đặc trưng sẵn, nên để tách không tạo thêm rò rỉ nào.
PHAP_LY_ND = ("Đã có sổ", "Sổ chung / công chứng vi bằng", "Đang chờ sổ",
              "Giấy tờ viết tay", "Không có sổ",
              "Sổ hồng riêng", "Hợp đồng mua bán", "Hợp đồng đặt cọc")
HUONG = ("Đông Bắc", "Đông Nam", "Tây Bắc", "Tây Nam",
         "Đông", "Tây", "Nam", "Bắc")
TINH_TRANG = ("Đã bàn giao", "Chưa bàn giao")


def chuan_hoa_muc(gia_tri: object, nhan_hop_le: tuple[str, ...]) -> object:
    """Cắt giá trị về nhãn gốc; không khớp nhãn nào thì trả NaN.

    Nhãn dài đứng trước nhãn ngắn trong các hằng số trên ("Đông Bắc" trước
    "Đông"), nếu không thì "Đông Bắc" sẽ bị cắt nhầm thành "Đông".
    """
    if not isinstance(gia_tri, str):
        return np.nan
    for v in nhan_hop_le:
        if gia_tri.startswith(v):
            return v
    return np.nan


NGUONG_MAT_CHUAN_HOA = 0.20     # mất quá 20% giá trị là dấu hiệu sai bộ nhãn


def _chuan_cot(df: pd.DataFrame, goc: str, moi: str,
               nhan: tuple[str, ...], canh_bao: list) -> None:
    """Chuẩn hoá một cột mục, và ĐO xem phép chuẩn hoá làm mất bao nhiêu.

    VÌ SAO PHẢI ĐO
    --------------
    `chuan_hoa_muc` trả NaN khi không khớp nhãn nào — hành vi đúng cho một giá
    trị rác, nhưng cũng chính là cách một bộ nhãn THIẾU xoá sạch cả một trường
    mà không báo gì. Đã xảy ra thật: `PHAP_LY_ND` không có "Sổ hồng riêng" nên
    72,3% trường pháp lý nhà đất thành NaN, âm thầm, suốt nhiều đợt gộp nguồn.

    Đây là cùng một loại lỗi với việc mất `url_goc` và `latitude`: thêm một
    nguồn mới có bộ từ vựng khác, code cũ im lặng bỏ qua. Nên chốt chặn không
    phải là sửa cho đúng một lần, mà là ĐO tỉ lệ mất mỗi lần chạy.
    """
    if goc not in df.columns:
        df[moi] = np.nan
        return
    df[moi] = df[goc].map(lambda s: chuan_hoa_muc(s, nhan))
    co = int(df[goc].notna().sum())
    if not co:
        return
    mat = int((df[goc].notna() & df[moi].isna()).sum())
    if mat / co > NGUONG_MAT_CHUAN_HOA:
        vd = df.loc[df[goc].notna() & df[moi].isna(), goc].value_counts()
        canh_bao.append(
            f"{goc}: chuẩn hoá bỏ {mat}/{co} giá trị ({mat/co:.0%}) — bộ nhãn "
            f"có thể thiếu từ của một nguồn mới. Hay gặp nhất: "
            f"{list(vd.head(3).index)}")


def don_truong_chon(df: pd.DataFrame, loai: str) -> list[str]:
    """Chuẩn hoá tại chỗ và đặt tên cột ngắn để dùng làm đặc trưng.

    Trả về danh sách cảnh báo (rỗng là mọi thứ bình thường) thay vì in ra —
    người gọi quyết định hiện ở đâu.
    """
    cb: list[str] = []
    pl = PHAP_LY_CC if loai == "chungcu" else PHAP_LY_ND
    _chuan_cot(df, "Giấy tờ pháp lý", "phap_ly", pl, cb)
    _chuan_cot(df, "Tình trạng nội thất", "noi_that", NOI_THAT, cb)
    _chuan_cot(df, "Hướng cửa chính", "huong_cua", HUONG, cb)
    if loai == "chungcu":
        _chuan_cot(df, "Hướng ban công", "huong_ban_cong", HUONG, cb)
        _chuan_cot(df, "Tình trạng bất động sản", "tinh_trang_bds",
                   TINH_TRANG, cb)
    return cb


@dataclass(frozen=True)
class CauHinh:
    """Mô tả một loại bất động sản: cột nào là số, cột nào là mục."""
    ten: str
    tep: str
    cot_so: tuple[str, ...]
    cot_muc: tuple[str, ...]
    cot_dien_tich: str


def _cot_co(df: pd.DataFrame, ten_bat_dau: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(c for c in df.columns if c.startswith(ten_bat_dau))


def _cot_lang_gieng(df: pd.DataFrame) -> tuple[str, ...]:
    """Cột đặc trưng láng giềng, LIỆT KÊ ĐÍCH DANH chứ không khớp tiền tố.

    Bài học đắt: bản đầu tôi viết `_cot_co(df, ("gia_m2_", ...))` cho gọn. Tiền
    tố đó vô tình khớp luôn cột `gia_m2_nguon` — giá/m² do chính sàn công bố,
    tức là mục tiêu chia cho diện tích. Model được đưa thẳng đáp án, MAPE tụt
    từ 15,75% xuống 7,87% và nhìn như một bước nhảy vọt.

    Khớp tiền tố để chọn đặc trưng là cách rất dễ rò rỉ mục tiêu mà không ai
    thấy: chỉ cần khâu trước thêm một cột tên gần giống là hỏng, mà chẳng có
    lỗi nào báo ra.
    """
    if not DUNG_LANG_GIENG:
        return ()
    from dac_trung_khong_gian import TEN_COT
    return tuple(c for c in TEN_COT if c in df.columns)


def _cot_nguon(df: pd.DataFrame) -> tuple[str, ...]:
    """`nguon` KHÔNG được dùng làm đặc trưng. Đây là quyết định có đo đạc.

    Cột này chiếm 7,4% gain khi để trong model, nhưng lúc chạy thật thì không
    bao giờ có: người đi mua nhà nhập diện tích, số phòng, phường — không ai
    biết (và cũng không nên phải biết) "tin này lấy từ sàn nào".

    Ba kịch bản đo trên tập 3.929 tin chung cư:

        biết nguồn cả khi học lẫn khi đoán      15,48%   <- số đẹp, không thật
        BỎ HẲN cột nguồn                        15,57%   <- đang dùng
        học có nguồn, đoán không có nguồn       16,13%   <- cảnh chạy thật cũ
        học có nguồn, đoán gán 'nhatot'         16,13%
        học có nguồn, đoán gán 'batdongsan'     16,99%

    Để cột này lại làm model tệ đi 0,56 điểm ở nơi duy nhất có người dùng thật,
    đổi lấy 0,09 điểm đẹp hơn trên bảng đánh giá. Đó là đánh đổi sai chiều.

    `nguon` vẫn nằm trong dữ liệu và vẫn dùng cho các phép đo tách biệt — chỉ
    không đi vào model.
    """
    return ()


def cau_hinh_chungcu(df: pd.DataFrame) -> CauHinh:
    return CauHinh(
        ten="Chung cư",
        tep=CHUNGCU_FILE,
        cot_so=("dien_tich_m2", "so_phong_ngu", "so_phong_vs", "tang_so")
                + _cot_co(df, ("nhac_",)) + _cot_co(df, COT_TOA_DO)
                + _cot_lang_gieng(df),
        cot_muc=("quan_huyen", "phuong_moi", "du_an_clean", "phap_ly",
                 "noi_that", "huong_ban_cong", "huong_cua", "tinh_trang_bds")
                + _cot_nguon(df),
        cot_dien_tich="dien_tich_m2",
    )


def cau_hinh_nhadat(df: pd.DataFrame) -> CauHinh:
    return CauHinh(
        ten="Nhà đất",
        tep=NHADAT_FILE,
        cot_so=("dien_tich_dat_m2", "dien_tich_su_dung_m2", "so_phong_ngu",
                "so_phong_vs", "tong_so_tang", "mat_tien_m", "chieu_dai_m",
                "duong_rong_m", "co_oto", "kinh_doanh", "thang_may", "lo_goc")
                + _cot_co(df, ("dd_", "nhac_")) + _cot_co(df, COT_TOA_DO)
                + _cot_lang_gieng(df),
        cot_muc=("quan_huyen", "phuong_moi", "loai_hinh", "phap_ly",
                 "noi_that", "huong_cua")
                + _cot_nguon(df),
        cot_dien_tich="dien_tich_dat_m2",
    )


# =============================================================================
# NẠP VÀ HUẤN LUYỆN
# =============================================================================
# KHỞI ĐỘNG LẠNH: 17 GIÂY -> DƯỚI 1 GIÂY
#
# Bản trước đọc thẳng .xlsx rồi huấn luyện lại XGBoost mỗi lần khởi động. Đo
# ra (máy này, 20/09/2026):
#
#                    đọc xlsx   huấn luyện   tổng
#       chung cư        1,6 s       2,6 s     4,4 s
#       nhà đất         5,6 s       6,0 s    12,2 s
#
# Tức là cả hai nửa đều nặng — sửa mỗi phần model thì vẫn còn 7 giây. Trên
# Streamlit Cloud máy yếu hơn, và app NGỦ sau một lúc không ai vào, nên người
# mở link phải ngồi nhìn màn trắng. Với thứ đem đi bảo vệ thì đó là hỏng.
#
# Nên có bước CHUẨN BỊ chạy trước (xem `chuan_bi_trien_khai.py`): nó đọc xlsx,
# dọn cột, tính đặc trưng láng giềng, huấn luyện, rồi ghi ra `_trien_khai/`
# một file .parquet và bốn file model. Lúc chạy thật chỉ còn đọc parquet và
# nạp model.
#
# BẢO ĐẢM CŨ KHÔNG BỊ BỎ, NÓ ĐƯỢC KIỂM.
#
# Ghi chú cũ ở đây nêu đúng hai lý do để huấn luyện lại mỗi lần: model đừng
# lệch pha với dữ liệu, và Q conformal phải tính trên dữ liệu đang dùng. Cả
# hai vẫn còn nguyên giá trị. Khác ở chỗ giờ chúng được KIỂM chứ không được
# né: model mang theo vân tay của đúng bộ dữ liệu + cấu hình sinh ra nó, lúc
# nạp vân tay được tính lại và đem so, lệch là bỏ bản đã ghi và huấn luyện
# lại ngay tại chỗ (xem `ConformalValuer.nap_tu`). Đường chậm vẫn còn đó,
# chỉ là không phải đi vào nó nữa.

THU_MUC_GOI = thu_muc_ra("_trien_khai")
TEN_PARQUET = "du_lieu.parquet"


def _cau_hinh(loai: str, df: pd.DataFrame):
    if loai == "chungcu":
        return cau_hinh_chungcu(df)
    if loai == "nhadat":
        return cau_hinh_nhadat(df)
    raise ValueError(f"loại không hợp lệ: {loai}")


def _bam_file(duong_dan) -> str:
    """Băm nội dung file theo từng khối — file dữ liệu tới 7,5 MB."""
    import hashlib
    h = hashlib.sha256()
    with open(duong_dan, "rb") as f:
        for khoi in iter(lambda: f.read(1 << 20), b""):
            h.update(khoi)
    return h.hexdigest()[:16]


def van_tay(duong_dan_du_lieu, cfg, alpha: float) -> str:
    """Vân tay của MỘT bộ (dữ liệu, cấu hình) — dùng để đóng dấu lên model."""
    return (_bam_file(duong_dan_du_lieu) + "-"
            + van_tay_cau_hinh(list(cfg.cot_so), list(cfg.cot_muc), alpha))


def _nap_tu_goi(loai: str, alpha: float):
    """Đường NHANH: đọc parquet + nạp model đã ghi.

    Trả (df, nen, cfg, valuer) hoặc None nếu chưa có gói. `valuer` có thể là
    None riêng lẻ — nghĩa là dữ liệu dùng được nhưng model thì không, và
    người gọi chỉ phải huấn luyện lại phần model.
    """
    pq = THU_MUC_GOI / loai / TEN_PARQUET
    if not pq.exists():
        return None
    try:
        df = pd.read_parquet(pq)
    except Exception as e:                          # noqa: BLE001
        print(f"[MODEL] không đọc được {pq.name}: {e.__class__.__name__}: {e}"
              f" — quay về đọc .xlsx.", file=sys.stderr)
        return None

    cfg = _cau_hinh(loai, df)
    # Nền láng giềng suy lại từ chính df, KHÔNG ghi riêng ra file: một bản
    # chép riêng là một bản có thể lệch với df mà không ai biết.
    nen = chon_nen_lang_gieng(df)
    valuer = ConformalValuer.nap_tu(THU_MUC_GOI / loai,
                                    van_tay(pq, cfg, alpha))
    return df, nen, cfg, valuer


def _nap_tu_xlsx(loai: str):
    """Đường CHẬM: đọc .xlsx, dọn cột, tính đặc trưng láng giềng."""
    ten = {"chungcu": CHUNGCU_FILE, "nhadat": NHADAT_FILE}.get(loai)
    if ten is None:
        raise ValueError(f"loại không hợp lệ: {loai}")
    if not (DATA_DIR / ten).exists():
        # .xlsx KHÔNG được đẩy lên bản chạy thật (10 MB, và gói parquet đã
        # thay thế nó ở mọi đường chạy). Nên nếu tới được đây trên máy chủ thì
        # nghĩa là gói cũng hỏng hoặc thiếu — và lỗi phải nói thẳng ra phải
        # làm gì, chứ FileNotFoundError trần thì người đọc log không đoán nổi.
        raise FileNotFoundError(
            f"Không có gói triển khai ở {THU_MUC_GOI / loai}, mà cũng không "
            f"có {DATA_DIR / ten} để dựng lại.\n"
            f"Trên máy phát triển: chạy `python chuan_bi_trien_khai.py` rồi "
            f"commit thư mục {THU_MUC_GOI.name}/.\n"
            f"Nếu đang chạy trên máy chủ: thư mục {THU_MUC_GOI.name}/ chưa "
            f"được đẩy lên, hoặc bị .gitignore loại mất.")
    df = pd.read_excel(DATA_DIR / ten)
    lam_cot_toa_do(df)
    canh_bao = don_truong_chon(df, loai)
    # In ra TERMINAL, không đưa lên giao diện: đây là cảnh báo cho người bảo
    # trì pipeline, không phải thông tin người đi mua nhà cần đọc. Nhưng phải
    # in, vì thứ hỏng im lặng là thứ không bao giờ được sửa.
    for c in canh_bao:
        print(f"[CẢNH BÁO DỮ LIỆU · {loai}] {c}", file=sys.stderr)
    nen = them_lang_gieng(df)
    return df, nen, _cau_hinh(loai, df), canh_bao


@lru_cache(maxsize=4)
def _nap(loai: str, alpha: float) -> dict:
    canh_bao: list = []
    goi = _nap_tu_goi(loai, alpha)
    if goi is not None:
        df, nen, cfg, valuer = goi
    else:
        df, nen, cfg, canh_bao = _nap_tu_xlsx(loai)
        valuer = None

    if valuer is None:
        valuer = ConformalValuer(list(cfg.cot_so), list(cfg.cot_muc),
                                 alpha=alpha).fit(df, "TARGET_gia_vnd")
    # MỐC THAM CHIẾU cho phần giải thích: với mỗi đặc trưng số, giá trị "bình
    # thường" là trung vị trong dữ liệu. Giao diện in nó cạnh con số ± để người
    # đọc biết mốc nào là 0% — trước đây nhãn chỉ ghi "Diện tích 50 → −18%" mà
    # không nói bao nhiêu mét mới thành cộng.
    moc = {c: float(df[c].median()) for c in cfg.cot_so
           if c in df.columns and df[c].notna().any()
           and c not in COT_TOA_DO}

    return {"df": df, "cfg": cfg, "valuer": valuer, "nen_lang_gieng": nen,
            "toa_do_phuong": _bang_toa_do(df), "moc_tham_chieu": moc,
            "canh_bao_du_lieu": canh_bao}


def chon_nen_lang_gieng(df: pd.DataFrame) -> pd.DataFrame:
    """Những tin đủ chính xác để được dùng làm hàng xóm.

    Tách riêng ra vì có HAI chỗ cần đúng cùng một phép chọn: lúc huấn luyện
    (trong `them_lang_gieng`) và lúc nạp lại từ parquet (trong `_nap_tu_goi`).
    Hai chỗ mà viết rời nhau thì chỉ cần lệch một điều kiện là đặc trưng láng
    giềng lúc dự đoán khác lúc huấn luyện — sai lặng lẽ, không có gì báo.
    """
    if "nguon_toa_do" not in df.columns:
        return df.iloc[0:0]
    return df.loc[df["nguon_toa_do"].isin(TOA_DO_CHINH_XAC)]


def them_lang_gieng(df: pd.DataFrame) -> pd.DataFrame:
    """Thêm đặc trưng "giá k căn gần nhất" (tại chỗ) và trả về bảng nền.

    CHỈ tính cho những tin có toạ độ chính xác, và chỉ lấy chính chúng làm hàng
    xóm. Lý do là một thí nghiệm có can thiệp: lấy đúng 1.278 tin toạ độ chính
    xác rồi làm mờ toạ độ về mức đường (giữ nguyên mọi thứ khác), hiệu quả của
    đặc trưng này rơi từ **+0,87 điểm MAPE (p = 0,0004)** xuống **+0,20 điểm
    (p = 0,33)**. Với điểm ghim mức đường, "hàng xóm gần nhất" chỉ là "một tin
    khác cùng con phố" — không mang thêm thông tin nào mà mã hoá phường chưa có.

    Tin không đủ chính xác thì để trống chứ không điền bừa; XGBoost xử lý được
    ô trống, còn điền bừa thì làm nhiễu cả những dòng đang tốt.
    """
    from dac_trung_khong_gian import TEN_COT, gia_lang_gieng

    if "nguon_toa_do" not in df.columns:
        for c in TEN_COT:
            df[c] = np.nan
        return df.iloc[0:0]
    nen = chon_nen_lang_gieng(df)
    chac = df.index.isin(nen.index)
    if len(nen) < 30:
        for c in TEN_COT:
            df[c] = np.nan
        return nen
    lg = gia_lang_gieng(nen, df, loai_tru_chinh_no=False)
    # Dòng nằm trong nền thì phải loại chính nó ra khỏi danh sách hàng xóm.
    lg.loc[chac] = gia_lang_gieng(nen, nen, loai_tru_chinh_no=True).values
    lg.loc[~chac] = np.nan
    for c in lg.columns:
        df[c] = lg[c]
    return nen


def lang_gieng_cho_mot(loai: str, dac_diem: dict) -> dict:
    """Đặc trưng láng giềng cho một yêu cầu định giá."""
    from dac_trung_khong_gian import TEN_COT, gia_lang_gieng

    nen = nap(loai).get("nen_lang_gieng")
    lat = dac_diem.get("latitude")
    if nen is None or len(nen) < 30 or lat is None or pd.isna(lat):
        return {c: np.nan for c in TEN_COT}
    mot = pd.DataFrame([{"latitude": lat, "longitude": dac_diem["longitude"]}])
    return gia_lang_gieng(nen, mot, loai_tru_chinh_no=False).iloc[0].to_dict()


def lam_cot_toa_do(df: pd.DataFrame) -> None:
    """Thêm cột khoảng cách tới trung tâm (tại chỗ)."""
    if "latitude" in df.columns and "kc_trung_tam_km" not in df.columns:
        df["kc_trung_tam_km"] = haversine_km(df["latitude"], df["longitude"],
                                             *TRUNG_TAM)


def _quan_cua_phuong(df: pd.DataFrame) -> dict:
    """Mỗi phường mới thuộc MỘT quận cũ để hiển thị — quận chiếm đa số tin.

    VÌ SAO PHẢI CÓ BẢNG NÀY, VÀ PHẢI DÙNG CHUNG
    -------------------------------------------
    Sau sáp nhập 1/7/2025 một phường mới có thể trải trên nhiều quận cũ. Đo
    được: 21/65 phường chung cư và 12/83 phường nhà đất nằm trên hơn một quận.
    "Phường Hồng Hà" có tin ở cả Tây Hồ (113), Hai Bà Trưng (31) và Hoàn Kiếm (8).

    Trước đây mỗi chỗ tự quyết định lấy quận nào, và hậu quả lộ ra khi chạy
    thử: ô chọn phường gán "Phường Phương Liệt" cho Thanh Xuân (đa số tin),
    còn bảng tra đường phố giữ nguyên quận của từng tin nên ra Hoàng Mai. Kết
    quả là giao diện cảnh báo "phố này thuộc Phương Liệt, Hoàng Mai — không
    phải Phương Liệt, Thanh Xuân", tức là tự mâu thuẫn với chính mình về cùng
    một phường.

    Cấp quận đã bị bãi bỏ; nó chỉ còn là nhãn cho người Hà Nội dễ định vị. Nên
    quy về MỘT bảng duy nhất, và mọi so sánh về vị trí đều so theo PHƯỜNG.
    """
    hop_le = df.dropna(subset=["quan_huyen", "phuong_moi"])
    if hop_le.empty:
        return {}
    dem = hop_le.groupby(["phuong_moi", "quan_huyen"]).size()
    return dem.groupby(level=0).idxmax().map(lambda t: t[1]).to_dict()


def _bang_toa_do(df: pd.DataFrame) -> dict:
    """Bảng tra toạ độ đại diện cho từng phường và từng quận.

    Người dùng ứng dụng chọn "Quận Cầu Giấy / Phường Dịch Vọng", chứ không ai
    nhập vĩ độ kinh độ. Muốn model dùng được đặc trưng toạ độ thì phải suy ra
    toạ độ từ lựa chọn đó — lấy trung vị của chính những tin trong tập huấn
    luyện ở phường ấy. Đây đúng là điều mô hình đã học, nên không có gì lệch.
    """
    if "latitude" not in df.columns:
        return {"phuong": {}, "quan": {}}
    # Khoá theo `phuong_moi` (113 phường sau sáp nhập 1/7/2025), KHÔNG phải
    # `phuong_xa` (345 phường cũ). Giao diện trước đây cho chọn tên phường cũ
    # trong khi bản đồ và bảng xếp hạng dùng tên mới — 68% tên bên chung cư và
    # 83% bên nhà đất không khớp nhau, nên người dùng chọn "Phường Mỹ Đình 1"
    # để định giá rồi không tìm thấy phường đó trên bản đồ. Cả hệ thống giờ
    # dùng một hệ tên duy nhất, là hệ tên đang có hiệu lực.
    #
    # Đổi được vì `phuong_xa` KHÔNG phải đặc trưng của model — đã kiểm: cột mục
    # của cả hai nhánh chỉ có `phuong_moi`. Đây thuần tuý là thay đổi giao diện.
    p = (df.dropna(subset=["latitude", "phuong_moi"])
           .groupby(["quan_huyen", "phuong_moi"])[["latitude", "longitude"]]
           .median().to_dict("index"))
    q = (df.dropna(subset=["latitude", "quan_huyen"])
           .groupby("quan_huyen")[["latitude", "longitude"]]
           .median().to_dict("index"))

    # BẢNG TRA THEO ĐƯỜNG PHỐ — mức chính xác trung gian.
    #
    # Chọn phường thì mốc cách vị trí thật trung vị 0,75 km, p90 2,17 km. Trong
    # nội thành 2 km là qua mấy khu giá khác hẳn nhau, và đo được: dùng toạ độ
    # thật thay cho mốc phường lấy lại 3,41 điểm MAPE (chung cư) và 2,29 điểm
    # (nhà đất). Tên đường thu hẹp khoảng đó lại đáng kể mà người dùng không
    # phải mò trên bản đồ.
    #
    # CHỈ dựng từ những căn có toạ độ CHÍNH XÁC (nơi đăng tự công bố hoặc khớp
    # đúng dự án). Lấy cả căn có toạ độ suy từ tên phường thì bảng này chỉ chép
    # lại mốc phường dưới một cái tên khác — vô nghĩa và còn gây hiểu nhầm là
    # chính xác hơn thực tế.
    d_chac = df[df.get("nguon_toa_do", pd.Series(index=df.index)).isin(
        TOA_DO_GHIM_DUOC)] if "nguon_toa_do" in df.columns else df.iloc[0:0]
    # Khoá là (PHƯỜNG, ĐƯỜNG) — KHÔNG có quận. Xem `_quan_cua_phuong`: một
    # phường mới trải trên nhiều quận cũ, nên đưa quận vào khoá thì cùng một
    # con phố bị tách thành nhiều mục và mỗi mục mang một nhãn quận khác nhau.
    quan_dd = _quan_cua_phuong(df)
    duong = {}
    if len(d_chac) and "duong_pho" in d_chac.columns:
        gd = (d_chac.dropna(subset=["latitude", "duong_pho", "phuong_moi"])
                    .groupby(["phuong_moi", "duong_pho"])
                    .agg(latitude=("latitude", "median"),
                         longitude=("longitude", "median"),
                         n=("latitude", "size")))
        # Dưới 3 căn thì trung vị của một hai điểm không đại diện cho cả con
        # phố — thà để người dùng chọn phường còn hơn hứa chính xác hão.
        duong = {k: {**v, "quan_huyen": quan_dd.get(k[0])}
                 for k, v in gd[gd["n"] >= 3].to_dict("index").items()}

    du_an = {}
    if len(d_chac) and "du_an_clean" in d_chac.columns:
        c = d_chac[d_chac["du_an_clean"].notna()
                   & (d_chac["du_an_clean"] != "Không xác định")]
        gda = (c.dropna(subset=["latitude"])
                .groupby("du_an_clean")
                .agg(latitude=("latitude", "median"),
                     longitude=("longitude", "median"),
                     n=("latitude", "size")))
        du_an = gda[gda["n"] >= 3].to_dict("index")

    return {"phuong": p, "quan": q, "duong": duong, "du_an": du_an,
            "quan_cua_phuong": quan_dd}


def bo_sung_toa_do(loai: str, dac_diem: dict) -> dict:
    """Điền toạ độ cho một yêu cầu định giá nếu người dùng không tự nhập.

    Không sửa `dac_diem` gốc — trả về bản sao, để lời gọi bên ngoài còn dùng
    lại được dict ban đầu.
    """
    d = dict(dac_diem)
    if d.get("latitude") is not None and not pd.isna(d.get("latitude")):
        d["kc_trung_tam_km"] = haversine_km(d["latitude"], d["longitude"], *TRUNG_TAM)
        # Toạ độ đã có sẵn nghĩa là người dùng tự ghim trên bản đồ (hoặc lời
        # gọi từ pipeline). Đánh dấu để giao diện nói đúng mức định vị.
        d.setdefault("_muc_toa_do", "ghim")
        return d

    bang = nap(loai)["toa_do_phuong"]
    q, p, dp = d.get("quan_huyen"), d.get("phuong_moi"), d.get("duong_pho")
    da = d.get("du_an_clean")

    # Dò từ CHÍNH XÁC NHẤT xuống thô nhất, dừng ở cái đầu tiên tìm được.
    diem, muc = None, None
    if da and da in bang.get("du_an", {}):
        diem, muc = bang["du_an"][da], "du_an"
    if diem is None and dp and (p, dp) in bang.get("duong", {}):
        diem, muc = bang["duong"][(p, dp)], "duong"
    if diem is None:
        diem = bang["phuong"].get((q, p))
        muc = "phuong" if diem else None
    if diem is None:
        diem = bang["quan"].get(q)
        muc = "quan" if diem else None

    if diem:
        d["latitude"], d["longitude"] = diem["latitude"], diem["longitude"]
        d["kc_trung_tam_km"] = haversine_km(diem["latitude"], diem["longitude"],
                                            *TRUNG_TAM)
        d["_muc_toa_do"] = muc
    return d


def nap(loai: str, alpha: float = ALPHA_MAC_DINH) -> dict:
    """Nạp dữ liệu + model đã huấn luyện. Có bộ nhớ đệm nên gọi lại là tức thì."""
    return _nap(loai, alpha)


def duong_trong_phuong(loai: str, quan: str | None,
                       phuong: str | None) -> list[str]:
    """Tên đường trong phường đó mà hệ thống BIẾT toạ độ thật.

    Chỉ trả về đường có từ 3 căn toạ độ chính xác trở lên. Danh sách vì thế
    ngắn hơn danh sách phố ngoài đời rất nhiều — đó là chủ ý: ô chọn chỉ nên
    đưa ra những chỗ hệ thống thật sự định vị được, thay vì nhận mọi cái tên
    rồi âm thầm rơi về mốc phường.
    """
    if not phuong:
        return []
    # Lọc theo PHƯỜNG, bỏ qua quận: phường mới trải trên nhiều quận cũ nên lọc
    # thêm theo quận sẽ làm mất phố nằm ở phần lẻ của phường.
    bang = nap(loai)["toa_do_phuong"].get("duong", {})
    return sorted(d for (p, d) in bang if p == phuong)


_TIEN_TO_DUONG = ("duong ", "pho ", "ngo ", "ngach ", "hem ", "so ")


def _bo_dau(s: object) -> str:
    """Bỏ dấu tiếng Việt, hạ chữ thường — để so tên phố không phụ thuộc dấu."""
    import unicodedata as ud
    t = ud.normalize("NFD", str(s).lower())
    return "".join(c for c in t if ud.category(c) != "Mn").replace("đ", "d")


def _chuan_ten_duong(s: str) -> str:
    """Cắt tiền tố và số nhà, còn lại tên phố trần.

    "1 Phố Bùi Xương Trạch" -> "bui xuong trach". Cần vì người dùng gõ tự nhiên
    kèm số nhà và chữ "phố", còn trong dữ liệu cùng một con phố xuất hiện dưới
    nhiều dạng: "Bùi xương trạch", "Phố Bùi Xương Trạch", "Đường Bùi Xương Trạch".
    """
    t = _bo_dau(s).strip()
    t = t.lstrip("0123456789 ,.-/")          # bỏ số nhà đứng đầu
    for tt in _TIEN_TO_DUONG:
        while t.startswith(tt):
            t = t[len(tt):].strip()
    return " ".join(t.split())


def tim_duong_toan_thanh(loai: str, chuoi: str, k: int = 6) -> list[dict]:
    """Tra tên phố trong TOÀN Hà Nội, không giới hạn theo phường đang chọn.

    VÌ SAO KHÔNG GIỚI HẠN THEO PHƯỜNG NGƯỜI DÙNG CHỌN
    -------------------------------------------------
    Ca thật khi chạy thử: người dùng chọn Huyện Gia Lâm / Xã Bát Tràng rồi gõ
    "1 phố bùi xương trạch". Phố đó nằm ở Quận Thanh Xuân. Nếu chỉ tìm trong
    phường đang chọn thì kết quả là "không tìm thấy" — đúng về mặt máy móc,
    nhưng vô ích: hệ thống BIẾT phố đó ở đâu, chỉ là không nói ra.
    Tra toàn thành phố rồi báo "phố này thuộc Thanh Xuân, không phải Gia Lâm"
    vừa giúp người dùng sửa lựa chọn, vừa trung thực hơn.

    Dựa trên bảng toạ độ đường phố dựng từ chính dữ liệu — nhanh, không cần
    mạng, và chỉ trả về phố hệ thống thật sự biết vị trí.
    """
    q = _chuan_ten_duong(chuoi)
    if len(q) < 3:
        return []
    bang = nap(loai)["toa_do_phuong"].get("duong", {})
    ra = []
    for (phuong, duong), v in bang.items():
        quan = v.get("quan_huyen")
        t = _chuan_ten_duong(duong)
        if q == t:
            diem = 0
        elif t.startswith(q) or q.startswith(t):
            diem = 1
        elif q in t or t in q:
            diem = 2
        else:
            continue
        ra.append({"diem": diem, "quan_huyen": quan, "phuong_moi": phuong,
                   "duong_pho": duong, "latitude": v["latitude"],
                   "longitude": v["longitude"], "so_can": int(v["n"])})
    ra.sort(key=lambda r: (r["diem"], -r["so_can"]))
    return ra[:k]


def du_an_co_toa_do(loai: str) -> list[str]:
    """Dự án hệ thống biết toạ độ thật — dùng cho ô chọn dự án."""
    return sorted(nap(loai)["toa_do_phuong"].get("du_an", {}))


def cay_dia_diem(loai: str) -> dict:
    """Cây Quận → Phường (→ Dự án) để dựng các ô chọn phân cấp.

    MỖI PHƯỜNG CHỈ THUỘC ĐÚNG MỘT QUẬN TRONG DANH SÁCH NÀY
    ------------------------------------------------------
    Sau sáp nhập 1/7/2025, một phường mới có thể ghép các mảnh thuộc nhiều quận
    cũ khác nhau. Nhóm thẳng theo (quan_huyen, phuong_moi) thì "Phường Cầu Giấy"
    hiện lên trong danh sách của cả Quận Cầu Giấy lẫn Quận Nam Từ Liêm — người
    dùng mở ô chọn ra thấy tên phường trùng với tên quận khác và không hiểu gì.

    Nên mỗi phường được gán về quận chiếm ĐA SỐ tin của phường đó. Vài tin lẻ ở
    quận khác vẫn nằm trong dữ liệu và vẫn được model dùng — chỉ là không hiện
    trong ô chọn, vì ô chọn cần rõ ràng chứ không cần đầy đủ.

    Cấp quận thực ra đã bị bãi bỏ từ 1/7/2025; giữ nó ở đây chỉ vì người Hà Nội
    vẫn quen định vị theo quận. Đây là chỗ để cân nhắc lại khi thiết kế lại
    phần chọn địa điểm.
    """
    d = nap(loai)
    df = d["df"]
    # DÙNG CHUNG bảng phường -> quận với bảng toạ độ, để hai chỗ không bao giờ
    # nói khác nhau về cùng một phường.
    quan_chinh = d["toa_do_phuong"]["quan_cua_phuong"]

    cay: dict[str, list[str]] = {}
    for phuong, quan in quan_chinh.items():
        cay.setdefault(quan, []).append(phuong)
    cay = {q: sorted(ps) for q, ps in sorted(cay.items())}

    du_an: dict[tuple[str, str], list[str]] = {}
    if "du_an_clean" in df.columns:
        con = df[df["du_an_clean"].notna() & (df["du_an_clean"] != "Không xác định")]
        # Dự án gắn theo PHƯỜNG, rồi phát cho quận chính của phường đó — nếu
        # lọc thêm theo quan_huyen thì dự án nằm ở phần lẻ của phường sẽ biến
        # mất khỏi ô chọn.
        for phuong, nhom in con.dropna(subset=["phuong_moi"]).groupby("phuong_moi"):
            q = quan_chinh.get(phuong)
            if q:
                du_an[(q, phuong)] = sorted(nhom["du_an_clean"].unique())
    return {"cay": cay, "du_an": du_an}


# =============================================================================
# ĐỊNH GIÁ
# =============================================================================

def dinh_gia(loai: str, dac_diem: dict, alpha: float = ALPHA_MAC_DINH,
             giai_thich: bool = True, so_comparables: int = 5) -> dict:
    """Định giá một bất động sản.

    `dac_diem` là dict theo đúng tên cột trong dữ liệu sạch. Cột thiếu -> NaN,
    XGBoost tự xử lý, nên giao diện không cần điền đủ mọi thứ.

    Trả về: gia, khoang_duoi, khoang_tren, do_rong_phan_tram, do_tin_cay,
    giai_thich (dict của explain_one), comparables (DataFrame).
    """
    d = nap(loai, alpha)
    df, cfg, valuer = d["df"], d["cfg"], d["valuer"]
    dac_diem = bo_sung_toa_do(loai, dac_diem)
    dac_diem = {**dac_diem, **lang_gieng_cho_mot(loai, dac_diem)}

    hang = pd.DataFrame([{c: dac_diem.get(c, np.nan)
                          for c in set(cfg.cot_so) | set(cfg.cot_muc)}])
    # Một dòng duy nhất mà có ô None thì pandas cho cả cột thành kiểu object,
    # XGBoost từ chối ngay. Ép về số, thứ gì không ép được thành NaN — đúng
    # nghĩa "người dùng bỏ trống", và XGBoost xử lý NaN được.
    for c in cfg.cot_so:
        hang[c] = pd.to_numeric(hang[c], errors="coerce")
    itv = valuer.predict_interval(hang).iloc[0]

    ket_qua = {
        "loai": cfg.ten,
        "gia": float(itv["gia"]),
        "khoang_duoi": float(itv["khoang_duoi"]),
        "khoang_tren": float(itv["khoang_tren"]),
        "do_rong_phan_tram": float(itv["do_rong_phan_tram"]),
        "muc_tin_cay": int(round((1 - alpha) * 100)),
    }
    dt = dac_diem.get(cfg.cot_dien_tich)
    ket_qua["gia_tren_m2"] = ket_qua["gia"] / dt if dt else None

    ket_qua["do_tin_cay"] = _muc_do_tin_cay(df, dac_diem, itv["do_rong_phan_tram"])

    if giai_thich:
        raw = pd.Series(dac_diem)
        # PHẢI là `display_model_`, không phải `point_model_`. Con số lớn hiển
        # thị trên màn hình do `predict_interval` sinh ra, mà hàm đó dùng
        # `display_model_` (model phân vị 50% của CQR). Trước đây chỗ này giải
        # thích `point_model_` — một model KHÁC — nên câu giải thích nói một
        # giá còn ô "Giá ước tính" hiện một giá khác, lệch tới 44% trên cùng
        # một căn. Giải thích phải giải thích đúng con số đang hiện.
        # Những trường người dùng THỰC SỰ điền — để phần giải thích không nói
        # về các cờ `nhac_*` mà form không có ô nào để nhập. Toạ độ tính là đã
        # nhập khi họ chọn được vị trí, dù ở mức phường.
        da_nhap = {k for k, v in dac_diem.items()
                   if v is not None and not (isinstance(v, float) and np.isnan(v))
                   and not str(k).startswith("_")}
        ket_qua["giai_thich"] = explain_one(
            valuer.display_model_, valuer.transform(hang), raw,
            moc=d["moc_tham_chieu"], truong_da_nhap=da_nhap)

    if so_comparables:
        ket_qua["comparables"] = tim_comparables(loai, dac_diem, so_comparables)
    return ket_qua


def _muc_do_tin_cay(df: pd.DataFrame, dac_diem: dict, do_rong: float) -> dict:
    """Diễn giải độ tin cậy thành câu người dùng hiểu được.

    Hai yếu tố: khoảng rộng bao nhiêu, và có bao nhiêu tin tương tự trong dữ
    liệu. Cái sau quan trọng vì nó trả lời được câu "dựa trên bao nhiêu căn?" —
    thứ mà một con số ±% đơn thuần không nói được.
    """
    phuong = dac_diem.get("phuong_moi")
    quan = dac_diem.get("quan_huyen")
    n_phuong = int((df["phuong_moi"] == phuong).sum()) if phuong else 0
    n_quan = int((df["quan_huyen"] == quan).sum()) if quan else 0

    if n_phuong >= 30 and do_rong < 45:
        muc, nhan = "cao", "Độ tin cậy cao"
    elif n_phuong >= 10 or n_quan >= 60:
        muc, nhan = "trung binh", "Độ tin cậy trung bình"
    else:
        muc, nhan = "thap", "Độ tin cậy thấp"

    if n_phuong >= 5:
        can_cu = f"dựa trên {n_phuong} căn đang bán cùng phường"
    elif n_quan >= 5:
        can_cu = (f"chỉ có {n_phuong} căn cùng phường nên phải mở rộng ra "
                  f"{n_quan} căn trong cả quận")
    else:
        can_cu = "rất ít căn tương tự để đối chiếu"
    return {"muc": muc, "nhan": nhan, "can_cu": can_cu,
            "n_phuong": n_phuong, "n_quan": n_quan}


# =============================================================================
# TÀI SẢN TƯƠNG ĐỒNG
# =============================================================================

def tim_comparables(loai: str, dac_diem: dict, k: int = 5) -> pd.DataFrame:
    """Tìm k tin rao giống nhất — cùng phường trước, hết thì mở ra cùng quận.

    Đây là tính năng làm người dùng TIN kết quả nhất. Một con số do model đưa
    ra thì người ta nghi ngờ; năm căn gần như y hệt đang rao ở mức nào thì
    không cãi được. Nghề môi giới ngoài đời cũng làm đúng như vậy.
    """
    d = nap(loai)
    df, cfg = d["df"], d["cfg"]
    dt_col = cfg.cot_dien_tich
    dt = dac_diem.get(dt_col)

    con = df
    if dac_diem.get("phuong_moi"):
        tmp = df[df["phuong_moi"] == dac_diem["phuong_moi"]]
        con = tmp if len(tmp) >= k else df[df["quan_huyen"] == dac_diem.get("quan_huyen")]
    if con is None or con.empty:
        return pd.DataFrame()

    con = con.copy()
    # Xếp hạng theo độ lệch tương đối về diện tích, rồi tới số phòng ngủ.
    if dt:
        con["_lech_dt"] = (con[dt_col] - dt).abs() / dt
    else:
        con["_lech_dt"] = 0.0
    pn = dac_diem.get("so_phong_ngu")
    con["_lech_pn"] = (con["so_phong_ngu"] - pn).abs().fillna(9) if pn else 0.0
    con = con.sort_values(["_lech_dt", "_lech_pn"]).head(k)

    # KHỬ TRÙNG, đừng bỏ dict.fromkeys đi: `dt_col` là "dien_tich_dat_m2" và
    # cột đó cũng nằm trong COT_BANG_CHUNG. Lấy `con[cot]` với tên cột trùng
    # thì pandas trả về HAI cột cùng tên, và `cmp_df[cot_dt]` sau đó thành một
    # DataFrame hai chiều thay vì một Series — vỡ ở tận giao diện với thông báo
    # "Buffer has wrong number of dimensions", chẳng chỉ về đây chút nào.
    cot = list(dict.fromkeys(
        c for c in ["TARGET_gia_vnd", "gia_tren_m2", dt_col, "so_phong_ngu",
                    "so_phong_vs", "phuong_moi", "quan_huyen", "duong_pho",
                    "du_an_clean", "loai_hinh", "url_goc", "tieu_de",
                    "ten_du_an_goc"] + list(COT_BANG_CHUNG)
        if c in con.columns))
    out = con[cot].reset_index(drop=True)
    out.insert(0, "STT", range(1, len(out) + 1))
    return out


# =============================================================================
# THỐNG KÊ THỊ TRƯỜNG (cho trang tổng quan)
# =============================================================================

def thong_ke_quan(loai: str) -> pd.DataFrame:
    d = nap(loai)
    df = d["df"]
    g = df.groupby("quan_huyen").agg(
        so_tin=("TARGET_gia_vnd", "size"),
        gia_trung_vi=("TARGET_gia_vnd", "median"),
        gia_m2_trung_vi=("gia_tren_m2", "median"),
    ).sort_values("so_tin", ascending=False)
    g["gia_trung_vi_ty"] = (g["gia_trung_vi"] / 1e9).round(2)
    g["gia_m2_trieu"] = (g["gia_m2_trung_vi"] / 1e6).round(1)
    return g.reset_index()[["quan_huyen", "so_tin", "gia_trung_vi_ty", "gia_m2_trieu"]]


def tam_quan_trong(loai: str, n: int = 10) -> pd.DataFrame:
    d = nap(loai)
    v = d["valuer"]
    X = v.transform(d["df"])
    return global_importance(v.point_model_, X).head(n)


__all__ = ["nap", "cay_dia_diem", "dinh_gia", "tim_comparables",
           "thong_ke_quan", "tam_quan_trong", "bo_sung_toa_do",
           "format_vnd", "ALPHA_MAC_DINH", "TRUNG_TAM"]
