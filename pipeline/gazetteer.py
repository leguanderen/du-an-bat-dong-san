"""
gazetteer.py — Chuẩn hoá tên dự án chung cư Hà Nội.

Tiêu đề tin đăng rất nhiễu ("🏰 CHÍNH CHỦ BÁN CĂN HỘ KING PALACE – NƠI AN CƯ...").
Module này ánh xạ tiêu đề về một tên dự án chuẩn bằng bảng tra alias.

QUY TẮC QUAN TRỌNG: cụm CỤ THỂ phải đứng TRƯỚC cụm CHUNG trong danh sách.
Ví dụ "Imperia Sky Garden" phải nằm trên "Imperia", nếu không mọi tin của
Sky Garden sẽ bị gán nhầm về "Imperia".

Bảng này vốn nằm trong app.py. Tách ra đây để pipeline làm sạch và ứng dụng
dùng CHUNG một nguồn — tránh tình trạng hai nơi chuẩn hoá khác nhau rồi encoding
không khớp. app.py nên sửa lại thành:  from gazetteer import normalize_project_name
"""

from __future__ import annotations

import re

from clean_common import strip_accents

UNKNOWN_PROJECT = "Không xác định"

GAZETTEER: list[tuple[str, list[str]]] = [
    ("Vinhomes Ocean Park",    [r"vinhomes ocean park", r"\bocean park\b"]),
    ("Vinhomes Smart City",    [r"vinhomes\s*smart\s*city", r"smart\s*city", r"smartcity"]),
    ("Vinhomes Gardenia",      [r"vinhomes gardenia", r"\bgardenia\b"]),
    ("Vinhomes Green Bay",     [r"green bay"]),
    ("Times City",             [r"time?s city", r"park hill"]),
    ("Royal City",             [r"royal city"]),
    ("Lumière Evergreen",      [r"lumi[eè]re"]),
    ("Imperia Sky Garden",     [r"imperia sky garden"]),
    ("Imperia Parkland",       [r"imperia parkland"]),
    ("Imperia",                [r"\bimperia\b"]),
    ("HH Linh Đàm",            [r"hh[0-9abc]*\s+linh dam", r"hh.{0,8}linh dam", r"ban dao linh dam"]),
    ("Rice City Sông Hồng",    [r"ri[cs]e city song hong"]),
    ("Rice City Linh Đàm",     [r"rice city"]),
    ("Kim Văn Kim Lũ",         [r"kim van kim lu"]),
    ("KĐT Đại Thanh",          [r"(ct[0-9abz]* )?dai thanh"]),
    ("KĐT Linh Đàm",           [r"linh dam"]),
    ("FLC Garden City Đại Mỗ", [r"flc .*(garden|dai mo)", r"flc garden city"]),
    ("FLC",                    [r"\bflc\b"]),
    ("Ecohome Phúc Lợi",       [r"ecohome phuc loi"]),
    ("Ecohome",                [r"ecohome", r"eco home"]),
    ("Eco Green City",         [r"eco green"]),
    ("Masteri",                [r"masteri"]),
    ("Goldmark City",          [r"goldmark"]),
    ("Gold Season",            [r"gold season"]),
    ("Mipec",                  [r"mipec"]),
    ("KĐT Ngoại Giao Đoàn",    [r"ngoai giao doan", r"\bngoai giao\b"]),
    ("The Pride",              [r"the pride"]),
    ("KĐT Văn Khê",            [r"van khe"]),
    ("The Zei",                [r"the zei"]),
    ("The Sola Park",          [r"the sola"]),
    ("The Matrix One",         [r"the matrix"]),
    ("The Emerald",            [r"the emerald"]),
    ("Roman Plaza",            [r"roman plaza"]),
    ("Five Star Kim Giang",    [r"five star"]),
    ("HPC Landmark 105",       [r"hpc landmark"]),
    ("Hateco",                 [r"hateco"]),
    ("Dream Town",             [r"dream town"]),
    ("Sunshine",               [r"sunshine"]),
    ("Gemek",                  [r"gemek"]),
    ("Feliz Homes",            [r"feliz"]),
    ("TSQ Euroland",           [r"\btsq\b"]),
    ("King Palace",            [r"king palace"]),
    ("Ruby City",              [r"ruby city", r"\bruby\b"]),
    ("Ciputra",                [r"ciputra"]),
    ("Roman / Hồng Hà",        [r"hong ha"]),
    ("Usilk City",             [r"usilk"]),
    ("Gelexia Riverside",      [r"gelexia"]),
    ("Berriver Long Biên",     [r"berriver"]),
    ("Eurowindow River Park",  [r"eurowindow"]),
    ("KĐT Thành Phố Giao Lưu", [r"thanh pho giao luu", r"\btp giao luu\b"]),
    ("X2 Đại Kim",             [r"x2 dai kim"]),
    ("CT2A Thạch Bàn",         [r"ct2a thach ban"]),
]

_COMPILED = [(name, [re.compile(p) for p in pats]) for name, pats in GAZETTEER]


def normalize_project_name(raw: object) -> str:
    """Tiêu đề tin đăng -> tên dự án chuẩn, hoặc UNKNOWN_PROJECT nếu không khớp."""
    if raw is None:
        return UNKNOWN_PROJECT
    text = str(raw).strip()
    if not text or text.lower() in {"nan", "-"}:
        return UNKNOWN_PROJECT
    flat = strip_accents(text)
    for canonical, patterns in _COMPILED:
        for pat in patterns:
            if pat.search(flat):
                return canonical
    return UNKNOWN_PROJECT
