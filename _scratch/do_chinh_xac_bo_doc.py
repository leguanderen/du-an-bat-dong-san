# -*- coding: utf-8 -*-
"""Đo độ chính xác của bộ phân tích câu trên TIÊU ĐỀ TIN RAO THẬT.

Ý tưởng: tiêu đề tin rao nhà đất là câu tiếng Việt tự nhiên do người thật
viết ("Bán nhà 4 tầng Đống Đa 50m2 giá 6 tỷ"), lại đi kèm đáp án đúng nằm
sẵn trong các cột dữ liệu (quận, giá, số tầng, diện tích, hướng). Cho bộ
phân tích đọc tiêu đề rồi so với cột — được một phép đo khách quan mà không
phải tự gán nhãn tay.

Hai con số cho mỗi trường:
    độ phủ     — bao nhiêu % tiêu đề bộ đọc bóc ra được trường đó
    độ đúng    — trong số bóc ra được, bao nhiêu % khớp dữ liệu

LƯU Ý KHI ĐỌC KẾT QUẢ: "sai" ở đây không phải lúc nào cũng là lỗi bộ đọc.
Tiêu đề và cột dữ liệu đôi khi tự mâu thuẫn (tiêu đề "5 tầng", cột ghi 4;
phố Lạc Long Quân nằm giữa Cầu Giấy và Tây Hồ). Nên file này in ra các ca sai
để người đọc tự phân loại, không chỉ in một con số.

    python _scratch/do_chinh_xac_bo_doc.py [số_tiêu_đề]
"""
from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path

GOC = Path(__file__).resolve().parent.parent
for d in (GOC, GOC / "pipeline"):
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))

import pandas as pd  # noqa: E402
import tim_nha as tnh  # noqa: E402
import valuation_service as vs  # noqa: E402

SAI_SO_GIA = 0.12     # "hơn 14 tỷ" với giá thật 15,5 tỷ vẫn là đọc đúng
SAI_SO_DT = 0.05


def chay(n: int = 2000, hat_giong: int = 1) -> None:
    df = vs.nap("nhadat")["df"]
    mau = df[df["tieu_de"].notna()].sample(min(n, df["tieu_de"].notna().sum()),
                                           random_state=hat_giong)
    dem, sai = Counter(), defaultdict(list)
    for _, r in mau.iterrows():
        d = tnh.phan_tich(r["tieu_de"], "nhadat")

        def xet(ten, doc_duoc, dung, doc, that):
            if doc_duoc:
                dem[ten + "_co"] += 1
                dem[ten + "_dung"] += bool(dung)
                if not dung:
                    sai[ten].append((r["tieu_de"], doc, that))

        xet("Quận", "quan_huyen" in d, d.get("quan_huyen") == r["quan_huyen"],
            d.get("quan_huyen"), r["quan_huyen"])
        g = d.get("gia_den") or d.get("gia_tu")
        gt = r["TARGET_gia_vnd"]
        xet("Giá", bool(g) and pd.notna(gt),
            g and pd.notna(gt) and abs(g - gt) <= SAI_SO_GIA * gt,
            g and f"{g/1e9:.2f} tỷ", pd.notna(gt) and f"{gt/1e9:.2f} tỷ")
        xet("Số tầng", "tang_tu" in d and pd.notna(r["tong_so_tang"]),
            d.get("tang_tu") == r["tong_so_tang"], d.get("tang_tu"),
            r["tong_so_tang"])
        dt = d.get("dt_tu") or d.get("dt_den")
        dtt = r["dien_tich_dat_m2"]
        xet("Diện tích", bool(dt) and pd.notna(dtt),
            dt and pd.notna(dtt) and abs(dt - dtt) <= SAI_SO_DT * dtt, dt, dtt)
        h = d.get("huong") or d.get("huong_cua")
        xet("Hướng", bool(h) and pd.notna(r["huong_cua"]),
            h and r["huong_cua"] in h, h, r["huong_cua"])

    N = len(mau)
    print(f"Đọc {N} tiêu đề tin rao nhà đất thật (chọn ngẫu nhiên, "
          f"random_state={hat_giong}).\n")
    print(f"{'Trường':<11}{'bóc được':>10}{'độ phủ':>9}{'đúng':>8}{'độ đúng':>10}")
    for ten in ("Quận", "Giá", "Diện tích", "Số tầng", "Hướng"):
        co, dung = dem[ten + "_co"], dem[ten + "_dung"]
        print(f"{ten:<11}{co:>10}{co/N:>9.1%}{dung:>8}"
              f"{(dung/co if co else 0):>10.1%}")
    for ten, ds in sai.items():
        print(f"\n--- {ten}: {len(ds)} ca lệch (10 ca đầu) ---")
        for td, doc, that in ds[:10]:
            print(f"  đọc={doc!s:<18} dữ liệu={that!s:<18} “{td[:80]}”")


if __name__ == "__main__":
    chay(int(sys.argv[1]) if len(sys.argv) > 1 else 2000)
