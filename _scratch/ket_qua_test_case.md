# Kết quả test case chatbot Tìm nhà

Chạy lúc 05:44 28/09/2026 · **40/40 test case đạt**, 51/51 lượt đúng.

| Mã | Nhóm | Câu người dùng | Mong đợi | Hệ thống hiểu | Kết quả |
|---|---|---|---|---|---|
| TC-01 | Cơ bản | Tìm chung cư 2 phòng ngủ ở Cầu Giấy dưới 4 tỷ | gia_den=4 tỷ; so_phong_ngu=2; quan_huyen=Quận Cầu Giấy | giá ≤ 4,00 tỷ, 2 phòng ngủ, Quận Cầu Giấy (42 căn) | Đạt |
| TC-02 | Cơ bản | căn hộ 3-5 tỷ, 60-80m2 ở Thanh Xuân | gia_tu=3 tỷ; gia_den=5 tỷ; dt_tu=60.0; dt_den=80.0; quan_huyen=Quận Thanh Xuân | giá 3,00–5,00 tỷ, 60–80 m², Quận Thanh Xuân (22 căn) | Đạt |
| TC-03 | Cơ bản | chung cư 2PN2WC Hà Đông | so_phong_ngu=2; so_phong_vs=2; quan_huyen=Quận Hà Đông | 2 phòng ngủ, 2 phòng vệ sinh, Quận Hà Đông (185 căn) | Đạt |
| TC-04 | Cơ bản | căn hộ Nam Từ Liêm 2 ngủ | quan_huyen=Quận Nam Từ Liêm | 2 phòng ngủ, Quận Nam Từ Liêm (353 căn) | Đạt |
| TC-05 | Cơ bản | nhà 4 tầng mặt tiền 5m ngõ 3m ô tô vào, dưới 8 tỷ Đống Đa | tang_tu=4; mt_tu=5.0; ngo_tu=3.0; gia_den=8 tỷ; co=nhac_o_to; quan_huyen=Quận Đống Đa | giá ≤ 8,00 tỷ, tổng số tầng ≥ 4, mặt tiền ≥ 5,0 m, ngõ rộng ≥ 3,0 m, Quận Đống Đa, ô tô vào được · nới: nhac_o_to (1 căn) | Đạt |
| TC-06 | Cơ bản | biệt thự 200m2 Long Biên | loai_hinh=Nhà biệt thự; dt_tu=200.0; quan_huyen=Quận Long Biên | từ 200 m², Quận Long Biên, Nhà biệt thự (2 căn) | Đạt |
| TC-07 | Cơ bản | chung cư không cần thang máy dưới 2 tỷ | khong=nhac_thang_may | giá ≤ 2,00 tỷ, không cần có thang máy (266 căn) | Đạt |
| TC-08 | Hướng | căn hộ hướng Đông Nam ở Hoàng Mai | huong=Đông Nam; quan_huyen=Quận Hoàng Mai | Quận Hoàng Mai, hướng Đông Nam (109 căn) | Đạt |
| TC-09 | Hướng | căn hộ 2PN ban công hướng Nam | huong_ban_cong=Nam; so_phong_ngu=2 | 2 phòng ngủ, ban công hướng Nam (109 căn) | Đạt |
| TC-10 | Hướng | cửa chính hướng Tây Bắc, ban công hướng Đông Nam | huong_cua=Tây Bắc; huong_ban_cong=Đông Nam | cửa hướng Tây Bắc, ban công hướng Đông Nam (467 căn) | Đạt |
| TC-11 | Hướng | nhà hướng Nam hoặc Đông Nam, Hai Bà Trưng | huong=Nam/Đông Nam; quan_huyen=Quận Hai Bà Trưng | Quận Hai Bà Trưng, hướng Nam, Đông Nam (20 căn) | Đạt |
| TC-12 | Hướng | căn hộ Hà Đông 2 ngủ tránh hướng Tây | huong_tru=Tây | 2 phòng ngủ, Quận Hà Đông, tránh hướng Tây (257 căn) | Đạt |
| TC-13 | Hướng | chung cư hợp Đông tứ trạch dưới 5 tỷ | huong=Bắc/Nam/Đông/Đông Nam; _tu_trach=Đông tứ trạch | giá ≤ 5,00 tỷ, Đông tứ trạch (Bắc, Nam, Đông, Đông Nam) (549 căn) | Đạt |
| TC-14 | Hướng | hướng Bắc, hợp đông tứ trạch | huong=Bắc | hướng Bắc (252 căn) | Đạt |
| TC-15 | Hướng | tôi sinh năm 1990, tìm căn hướng hợp tuổi | — | — (5135 căn) | Đạt |
| TC-16 | Hướng | nhà Tây Hồ 10 tỷ | quan_huyen=Quận Tây Hồ | giá ≤ 10,00 tỷ, Quận Tây Hồ (154 căn) | Đạt |
| TC-17 | Hướng | nhà hướng Nam Đống Đa | huong=Nam; quan_huyen=Quận Đống Đa | Quận Đống Đa, hướng Nam (5 căn) | Đạt |
| TC-18 | Hướng | căn hộ hướng Nam, Đông Nam | huong=Nam/Đông Nam | hướng Nam, Đông Nam (1547 căn) | Đạt |
| TC-19 | Hướng | căn hộ hướng ĐN 70m2 | huong=Đông Nam; dt_tu=70.0 | từ 70 m², hướng Đông Nam (829 căn) | Đạt |
| TC-20 | Tầng, nội thất | căn hộ tầng trung full đồ ở Cầu Giấy | tang_tu=10; tang_den=20; noi_that=Nội thất đầy đủ/Nội thất cao cấp; quan_huyen=Quận Cầu Giấy | tầng 10–20, Quận Cầu Giấy, full nội thất (17 căn) | Đạt |
| TC-21 | Tầng, nội thất | căn hộ tầng 15 trở lên ở Hà Đông | tang_tu=15 | tầng ≥ 15, Quận Hà Đông (38 căn) | Đạt |
| TC-22 | Tầng, nội thất | căn hộ bàn giao thô Long Biên | noi_that=Bàn giao thô | Quận Long Biên, bàn giao thô (1 căn) | Đạt |
| TC-23 | Tầng, nội thất | căn hộ không cần nội thất dưới 3 tỷ | gia_den=3 tỷ | giá ≤ 3,00 tỷ (678 căn) | Đạt |
| TC-24 | Nhiều lượt | Chung cư Nam Từ Liêm 2 phòng ngủ dưới 5 tỷ | quan_huyen=Quận Nam Từ Liêm; gia_den=5 tỷ | giá ≤ 5,00 tỷ, 2 phòng ngủ, Quận Nam Từ Liêm (155 căn) | Đạt |
| TC-24 | Nhiều lượt | Chỉ lấy những căn có ban công hướng Nam thôi | quan_huyen=Quận Nam Từ Liêm; gia_den=5 tỷ; so_phong_ngu=2; huong_ban_cong=Nam | giá ≤ 5,00 tỷ, 2 phòng ngủ, Quận Nam Từ Liêm, ban công hướng Nam (2 căn) | Đạt |
| TC-24 | Nhiều lượt | đổi sang hướng Đông Nam | huong_ban_cong=Đông Nam | giá ≤ 5,00 tỷ, 2 phòng ngủ, Quận Nam Từ Liêm, ban công hướng Đông Nam (39 căn) | Đạt |
| TC-24 | Nhiều lượt | bỏ điều kiện hướng ban công | quan_huyen=Quận Nam Từ Liêm | giá ≤ 5,00 tỷ, 2 phòng ngủ, Quận Nam Từ Liêm (155 căn) | Đạt |
| TC-25 | Nhiều lượt | căn hộ 5 tỷ Cầu Giấy | gia_den=5 tỷ | giá ≤ 5,00 tỷ, Quận Cầu Giấy (132 căn) | Đạt |
| TC-25 | Nhiều lượt | rẻ hơn nữa | quan_huyen=Quận Cầu Giấy | giá ≤ 2,25 tỷ, Quận Cầu Giấy (34 căn) | Đạt |
| TC-25 | Nhiều lượt | thêm thang máy | co=nhac_thang_may | giá ≤ 2,25 tỷ, Quận Cầu Giấy, có thang máy (7 căn) | Đạt |
| TC-26 | Nhiều lượt | căn hộ 3 tỷ phường Cầu Giấy | phuong_moi=Phường Cầu Giấy | giá ≤ 3,00 tỷ, Phường Cầu Giấy, Quận Cầu Giấy (25 căn) | Đạt |
| TC-26 | Nhiều lượt | Thanh Xuân thì sao | quan_huyen=Quận Thanh Xuân | giá ≤ 3,00 tỷ, Quận Thanh Xuân (54 căn) | Đạt |
| TC-27 | Nhiều lượt | căn hộ Cầu Giấy 3 tỷ | quan_huyen=Quận Cầu Giấy | giá ≤ 3,00 tỷ, Quận Cầu Giấy (44 căn) | Đạt |
| TC-27 | Nhiều lượt | tầng trung thôi | quan_huyen=Quận Cầu Giấy | giá ≤ 3,00 tỷ, tầng 10–20, Quận Cầu Giấy · nới: tang (44 căn) | Đạt |
| TC-28 | Nhiều lượt | căn hộ Hoàng Mai dưới 3 tỷ | quan_huyen=Quận Hoàng Mai | giá ≤ 3,00 tỷ, Quận Hoàng Mai (109 căn) | Đạt |
| TC-28 | Nhiều lượt | tìm cho tôi căn hộ 2 phòng ngủ có thang máy | so_phong_ngu=2 | 2 phòng ngủ, có thang máy (179 căn) | Đạt |
| TC-29 | Nhiều lượt | chung cư Hoàng Mai dưới 3 tỷ | quan_huyen=Quận Hoàng Mai | giá ≤ 3,00 tỷ, Quận Hoàng Mai (109 căn) | Đạt |
| TC-29 | Nhiều lượt | nhà đất thì sao | quan_huyen=Quận Hoàng Mai; gia_den=3 tỷ | giá ≤ 3,00 tỷ, Quận Hoàng Mai (65 căn) | Đạt |
| TC-30 | Nhiều lượt | nhà đất Long Biên 6 tỷ | quan_huyen=Quận Long Biên | giá ≤ 6,00 tỷ, Quận Long Biên (107 căn) | Đạt |
| TC-30 | Nhiều lượt | thêm ban công hướng Nam | quan_huyen=Quận Long Biên | giá ≤ 6,00 tỷ, Quận Long Biên, ban công hướng Nam · nới: huong_ban_cong (107 căn) | Đạt |
| TC-40 | Nhiều lượt | chung cư Nam Từ Liêm 2PN dưới 5 tỷ full đồ | noi_that=Nội thất đầy đủ/Nội thất cao cấp | giá ≤ 5,00 tỷ, 2 phòng ngủ, Quận Nam Từ Liêm, full nội thất (101 căn) | Đạt |
| TC-40 | Nhiều lượt | biệt thự Hoàn Kiếm 1 tỷ | quan_huyen=Quận Hoàn Kiếm; loai_hinh=Nhà biệt thự | giá ≤ 1,00 tỷ, Quận Hoàn Kiếm, Nhà biệt thự · nới: loai_hinh, gia_den (46 căn) | Đạt |
| TC-31 | Không dấu | can ho 3 phong ngu ha dong duoi 3 ty huong dong nam | so_phong_ngu=3; quan_huyen=Quận Hà Đông; gia_den=3 tỷ; huong=Đông Nam | giá ≤ 3,00 tỷ, 3 phòng ngủ, Quận Hà Đông, hướng Đông Nam · nới: huong (1 căn) | Đạt |
| TC-32 | Không dấu | chung cu 2 ty 5 cau giay | gia_den=2.5 tỷ | giá ≤ 2,50 tỷ, Quận Cầu Giấy (41 căn) | Đạt |
| TC-33 | Biên | biệt thự Hoàn Kiếm giá 1 tỷ | loai_hinh=Nhà biệt thự | giá ≤ 1,00 tỷ, Quận Hoàn Kiếm, Nhà biệt thự · nới: loai_hinh, gia_den (46 căn) | Đạt |
| TC-34 | Biên | chung cư Ba Đình 2PN dưới 1 tỷ | so_phong_ngu=2 | giá ≤ 1,00 tỷ, 2 phòng ngủ, Quận Ba Đình · nới: so_phong_ngu (1 căn) | Đạt |
| TC-35 | Biên | xin chào | — | — (5135 căn) | Đạt |
| TC-36 | Biên | nhà mặt tiền 5m trong ngõ Thanh Xuân | mt_tu=5.0; loai_hinh=Nhà ngõ, hẻm | mặt tiền ≥ 5,0 m, Quận Thanh Xuân, Nhà ngõ, hẻm (229 căn) | Đạt |
| TC-37 | Biên | nhà dưới 5 tỷ nhưng không ở trong ngõ | loai_hinh_tru=Nhà ngõ, hẻm | giá ≤ 5,00 tỷ, không lấy nhà ngõ, hẻm (164 căn) | Đạt |
| TC-38 | Biên | căn hộ Tây Hồ hướng Đông | quan_huyen=Quận Tây Hồ; huong=Đông | Quận Tây Hồ, hướng Đông (18 căn) | Đạt |
| TC-39 | Biên | căn nào đang rẻ hơn thị trường ở Hà Đông | sap=lech_thap; quan_huyen=Quận Hà Đông | Quận Hà Đông (476 căn) | Đạt |
