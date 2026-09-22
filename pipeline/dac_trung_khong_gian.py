"""
dac_trung_khong_gian.py — Đặc trưng suy từ các bất động sản LÂN CẬN.

Ý tưởng lấy thẳng từ nghề thẩm định giá: muốn biết một căn đáng bao nhiêu, người
ta không tra bảng — họ nhìn xem mấy căn ngay cạnh vừa rao bao nhiêu. Module này
làm đúng việc đó bằng số: với mỗi tin, tìm k tin gần nhất trong không gian rồi
lấy giá/m² trung vị của chúng làm đặc trưng.

VÌ SAO GIỜ MỚI LÀM ĐƯỢC
-----------------------
Trước đây 87% tin nhatot dùng chung toạ độ với tin khác — chúng là ĐIỂM GHIM của
con đường, không phải vị trí căn nhà. "k căn gần nhất" khi đó chỉ là "k căn tình
cờ cùng một điểm ghim", vô nghĩa.

Dữ liệu batdongsan vừa cào có toạ độ tới mức toà nhà (chỉ 12–34% trùng nhau), nên
khái niệm "hàng xóm" mới có nội dung thật.

CHỖ RẤT DỄ RÒ RỈ MỤC TIÊU
-------------------------
Đặc trưng này suy ra từ GIÁ của các dòng khác, nên nếu tính một lần trên toàn bộ
dữ liệu rồi mới chia train/test thì:

  * giá của dòng test đã góp vào đặc trưng của các dòng train quanh nó, và
  * đặc trưng của chính dòng test được tính từ những dòng mà model đã học.

Kết quả là MAPE đẹp giả. Nên nó phải được tính LẠI TRONG TỪNG FOLD, chỉ từ phần
train, và khi tính cho một dòng train thì phải loại chính nó ra khỏi danh sách
hàng xóm. Hàm `tao_ham_theo_fold` đóng gói đúng ràng buộc đó.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_KM_MOI_DO_VI = 111.32
_VI_GOC = 21.0
K_MAC_DINH = (5, 15)
BAN_KINH_DEM_KM = 1.0


def _ve_phang(d: pd.DataFrame) -> np.ndarray:
    """Chiếu (vĩ độ, kinh độ) về mặt phẳng km. Sai số dưới 0,1% trong một thành
    phố — không đáng kể so với sai số toạ độ."""
    return np.column_stack([
        d["latitude"].to_numpy(float) * _KM_MOI_DO_VI,
        d["longitude"].to_numpy(float) * _KM_MOI_DO_VI * np.cos(np.radians(_VI_GOC)),
    ])


def gia_lang_gieng(train: pd.DataFrame, muc_tieu: pd.DataFrame,
                   ks: tuple[int, ...] = K_MAC_DINH,
                   cot_gia_m2: str = "gia_tren_m2",
                   loai_tru_chinh_no: bool = False) -> pd.DataFrame:
    """Giá/m² trung vị của k căn gần nhất, cho từng dòng của `muc_tieu`.

    `loai_tru_chinh_no=True` khi `muc_tieu` chính là `train` — mỗi dòng phải bỏ
    bản thân nó ra, nếu không nó đang tự nhìn giá của mình.
    """
    from scipy.spatial import cKDTree

    hop_le = train["latitude"].notna() & train[cot_gia_m2].notna()
    nen = train.loc[hop_le]
    out = pd.DataFrame(index=muc_tieu.index)
    if len(nen) < max(ks) + 2:
        for k in ks:
            out[f"gia_m2_{k}_gan_nhat"] = np.nan
            out[f"kc_toi_can_thu_{k}_km"] = np.nan
        out[f"so_can_trong_{BAN_KINH_DEM_KM:g}km"] = 0
        return out

    cay = cKDTree(_ve_phang(nen))
    log_gia = np.log(nen[cot_gia_m2].to_numpy(float))
    co = muc_tieu["latitude"].notna().to_numpy()
    P = _ve_phang(muc_tieu.assign(
        latitude=muc_tieu["latitude"].fillna(_VI_GOC),
        longitude=muc_tieu["longitude"].fillna(105.85)))

    k_max = max(ks) + (1 if loai_tru_chinh_no else 0)
    kc, chi_so = cay.query(P, k=k_max)
    if loai_tru_chinh_no:
        # Bỏ hàng xóm gần nhất — chính là bản thân dòng đó (khoảng cách 0).
        kc, chi_so = kc[:, 1:], chi_so[:, 1:]

    for k in ks:
        gia = np.median(log_gia[chi_so[:, :k]], axis=1)
        out[f"gia_m2_{k}_gan_nhat"] = np.where(co, np.exp(gia), np.nan)
        out[f"kc_toi_can_thu_{k}_km"] = np.where(co, kc[:, k - 1], np.nan)

    dem = np.array([len(x) for x in cay.query_ball_point(P, BAN_KINH_DEM_KM)])
    if loai_tru_chinh_no:
        dem = np.maximum(dem - 1, 0)
    out[f"so_can_trong_{BAN_KINH_DEM_KM:g}km"] = np.where(co, dem, 0)
    return out


def tao_ham_theo_fold(ks: tuple[int, ...] = K_MAC_DINH,
                      cot_gia_m2: str = "gia_tren_m2"):
    """Trả về hàm dùng cho tham số `dac_trung_theo_fold` của cross_validate."""
    def _ham(tr: pd.DataFrame, te: pd.DataFrame):
        return (gia_lang_gieng(tr, tr, ks, cot_gia_m2, loai_tru_chinh_no=True),
                gia_lang_gieng(tr, te, ks, cot_gia_m2, loai_tru_chinh_no=False))
    return _ham


TEN_COT = ([f"gia_m2_{k}_gan_nhat" for k in K_MAC_DINH]
           + [f"kc_toi_can_thu_{k}_km" for k in K_MAC_DINH]
           + [f"so_can_trong_{BAN_KINH_DEM_KM:g}km"])
