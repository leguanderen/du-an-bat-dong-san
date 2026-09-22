# Trạng thái dự án — cập nhật 01/09/2026

## Đang ở đâu

| | Chung cư | Nhà đất |
|---|---|---|
| Số tin dùng được | **4.887** | **12.294** (nhatot 4.579 + batdongsan 7.715) |
| MAPE (cấu hình chạy thật) | **14,81%** | **22,06%** |
| R² | 0,830 | **0,801** |
| Ngưỡng nhiễu không khử được | **11,87%** | **16,38%** |
| **Khoảng còn có thể cải thiện** | **2,9 điểm** | **5,7 điểm** |
| Độ phủ khoảng CQR | 90,1% ở ±36,0% | 90,1% ở ±44,8% |
| Độ phủ nếu dùng ±15% cố định | 62,9% | 50,6% |
| Toạ độ **chính xác** (sàn tự công bố / mã tin / dự án) | **55,7%** (2.722 tin) | **62,7%** (7.715 tin) |
| Phường (mới) phủ được | 66 / 113 | 83 / 113 |




## Đợt 01/09 — nhà đất trang 351–650

+3.669 tin ròng. Tập nhà đất: **12.294 tin**, lần đầu vượt nhánh chung cư.

| | 30/08 | 31/08 | 01/09 |
|---|---|---|---|
| Số tin | 4.584 | 8.641 | **12.294** |
| Toạ độ chính xác | 0% | 47,0% | **62,7%** |
| MAPE | 26,21% | 23,67% | **22,06%** |
| R² | 0,655 | 0,700 | **0,801** |
| Độ rộng khoảng CQR | ±50,4% | ±47,7% | **±44,8%** |
| Phường tô được trên bản đồ | 66 | 71 | **74** |
| Phường trong bảng xếp hạng | 58 | 67 | **70** |

**Giảm tổng cộng 4,15 điểm trong ba ngày**, và R² nhảy từ 0,655 lên 0,801 —
mức tăng lớn hơn nhiều so với phần MAPE, nghĩa là model giờ giải thích được
những căn giá cao mà trước đây nó bó tay.

Ngưỡng nhiễu **giảm** 17,39% → 16,38%: thêm dữ liệu tốt không chỉ giúp model,
nó còn làm các nhóm tin trùng đặc điểm trở nên thuần nhất hơn.

**Đặc trưng láng giềng: đo lại, vẫn tắt.** Điều kiện tôi đặt ra ("bật lại khi
tỉ lệ toạ độ chính xác vượt một nửa") giờ đã thoả — 62,7%. Đo lại: **−0,03
điểm, KTC [−0,25; +0,19], p = 0,16**. Không có tác dụng. Giữ tắt, và giờ có
bằng chứng trực tiếp chứ không chỉ lý do về tính trung thực của khoảng tin cậy.

Đường cong vẫn chưa phẳng (22,56% ở 75% dữ liệu → 22,06% ở 100%), nên cào tiếp
nhà đất vẫn còn lãi.

## Đợt 30–31/08 — nhà đất batdongsan, bước ngoặt của nhánh yếu

Cào trang 1–350. Sau làm sạch và khử trùng lặp: **8.641 tin** (từ 4.584).

| | Trước | Sau |
|---|---|---|
| Số tin nhà đất | 4.584 | **8.641** |
| Toạ độ chính xác | **0%** | **47,0%** (4.061 tin) |
| Độ rộng đường/ngõ có số | 7% | **31%** |
| Mặt tiền có số | 50% | **60%** |
| MAPE | 26,21% | **23,67%** |
| R² | 0,655 | 0,700 |
| Phường phủ trên bản đồ | 66 | **71** |

**Giảm 2,54 điểm — vượt xa dự phóng.** Đường cong cũ ước tính cần 8.752 tin
cho *một* điểm; thực tế 8.641 tin cho **2,54 điểm**. Lý do giống hệt đợt chung
cư: đây không phải dữ liệu trung bình. Nó mang theo toạ độ do sàn công bố và
hai trường `Mặt tiền` / `Đường vào` có cấu trúc — đúng những yếu tố vi mô mà
bảng tầm quan trọng đã chỉ ra là chỗ thiếu. `mat_tien_m` giờ đứng thứ tư trong
bảng tầm quan trọng, trước cả `phuong_moi`.

Khoảng cách hai nhánh thu hẹp từ 11,4 điểm xuống **8,9 điểm**, và khoảng trống
tới ngưỡng nhiễu của nhà đất từ 9,2 xuống **6,3 điểm**.

### Ba lỗi âm thầm bắt được trong đợt này

**1. `Loại hình` gán cứng "Chung cư".** Bộ chuyển đổi batdongsan viết cho nhánh
chung cư nên gán cứng loại hình. Chạy cho nhà đất thì bộ lọc loại hình quét
sạch **toàn bộ 1.172 tin**, báo cáo chỉ ghi "còn 0 dòng". Đã suy loại hình từ
slug URL.

**2. Địa chỉ batdongsan không còn cấp quận/huyện.** Từ 1/7/2025 cấp quận bị bãi
bỏ và batdongsan đã đổi toàn site sang "Đường X, Phường Y, Hà Nội". Bộ lọc
"không xác định được quận" xoá sạch mọi đợt cào mới ở **cả hai nhánh**. Đã viết
`bu_quan_tu_phuong()` tra ngược quận từ tên phường.

**3. Chuẩn hoá địa giới tin vào ô đã có chữ.** `gan_phuong_moi` chỉ kiểm
`notna()` rồi bỏ qua, trong khi `gan_toa_do` khi không tra được thì chép nguyên
tên phường CŨ sang. Kết quả: **29% số tin nhà đất mang tên trước sáp nhập**
(Nhân Chính, Dịch Vọng, Trâu Quỳ…) và hệ thống đếm ra **207 phường trong khi Hà
Nội chỉ có 113**. Mã hoá mục bị chia vụn, bản đồ mất phường. Giờ đối chiếu
thẳng với danh sách 113 thay vì tin vào việc ô đã có chữ.

Cả ba lỗi đều **không ném ra một exception nào**. Dấu hiệu duy nhất nhận ra
được là những con số vô lý: "còn 0 dòng", "207/113 phường".

### Cào nhanh hơn 4 lần, và bài học về chẩn đoán

Extension chậm dần sau vài trang. Tôi đoán sai hai lần — dung lượng dữ liệu,
rồi log DevTools. Đến khi mở Trình quản lý tác vụ Chrome mới thấy: **thẻ đang
cào chiếm 8,98 GB RAM và 99,4% CPU**, trong khi extension chỉ 187 MB. Trang
batdongsan khởi tạo Google Maps mỗi tin và rò rỉ bộ nhớ qua hàng trăm lượt
điều hướng trong cùng một thẻ.

Cách chữa: **tắt JavaScript riêng cho batdongsan.com.vn**. Toạ độ nằm trong
*văn bản* của thẻ `<script>` do máy chủ trả về — chỉ cần đọc, không cần chạy.
Kiểm chứng: mọi cột vẫn đầy 100%, `Mặt tiền` 68%, `Đường vào` 54%, y hệt bản
có JS.

Kèm theo, đổi selector sang `:contains("latitude")` để chỉ khớp đúng khối
script chứa toạ độ: **1 dòng/tin thay vì 3,8**, dữ liệu mỗi tin từ 15,2 KB
xuống 3,7 KB.

## Đợt 27/08 chiều — batdongsan trang 100–200

1.975 tin thô → 1.644 tin sạch, **100% có toạ độ do sàn tự công bố**. Sau khử
trùng lặp còn **+958 tin ròng** (1.046 tin trùng với đợt trước, 17,4%).

| | Trước | Sau |
|---|---|---|
| Số tin chung cư | 3.929 | **4.887** |
| Tin có toạ độ chính xác | 1.278 (32,5%) | **2.722 (55,7%)** |
| MAPE | 15,57% | **14,81%** |
| Ngưỡng nhiễu | 12,13% | 11,87% |
| Khoảng tin cậy CQR | 90,4% ở ±37,4% | 90,1% ở **±36,0%** |

**Cải thiện 0,76 điểm — vượt dự phóng.** Đường cong học tập hôm sáng dự đoán
mốc 4.887 tin cho khoảng 15,2%; thực tế 14,81%. Lý do: đợt này không phải dữ
liệu trung bình, nó **toàn tin có toạ độ chính xác**, đúng thứ đã đo được là
nút thắt thật. Một tin có toạ độ thật đáng giá hơn một tin chỉ có tên đường.

Dự phóng cập nhật (khớp R² = 0,995): **cần 12.198 tin để giảm thêm 1 điểm**
(mốc trước là 8.327 — đường cong đang phẳng dần, đúng như kỳ vọng khi tiến gần
ngưỡng nhiễu 11,87%).

| Số tin | MAPE dự phóng |
|---|---|
| 6.000 | 14,62% |
| 8.000 | 14,26% |
| 12.000 | 13,83% |
| 20.000 | 13,39% |

**Đặc trưng láng giềng vẫn TẮT.** Giờ nó đáng +0,15 điểm (KTC [−0,13; +0,44],
Wilcoxon p = 0,029) — có dấu hiệu nhưng yếu. Lý do giữ tắt không đổi và không
liên quan đến độ lớn hiệu quả: ở bản chạy thật đặc trưng này phải tính ngoài
fold, làm khoảng tin cậy conformal hẹp lại **giả tạo**. Thêm dữ liệu không sửa
được chuyện đó.

**Độ phủ bản đồ gần như không đổi: 58 → 57 phường được tô.** Thêm 958 tin mà
không mở rộng được vùng phủ, vì tin mới rơi vào những phường vốn đã có dữ liệu.
Muốn phủ rộng hơn phải cào **theo quận còn trống**, không phải cào thêm trang.
(Một phường tụt từ 5 xuống 4 tin do khử trùng lặp xáo lại thứ tự — nằm đúng
ranh giới ngưỡng, không phải lỗi.)

## Lỗi đã sửa trong đợt này

`gan_toa_do.py` **không có mục cho tập gộp**. Chạy `python gan_toa_do.py` xong
vẫn để lại `chungcu_gop_geo.xlsx` của lần trước, và mọi bước sau âm thầm dùng
dữ liệu cũ mà không có lỗi nào báo ra. Đã thêm `--loai chungcu-gop`.

Kèm theo: hàm đó ghi đè `ngay_quan_sat` bằng một ngày duy nhất, trong khi tập
gộp mang sẵn ngày riêng cho từng đợt cào. Đã sửa để chỉ điền khi cột còn trống.

## Ba phát hiện mới hôm nay, cả ba đều đổi hướng công việc

### 1. Nhà đất KHÔNG phải là nửa vô vọng — nó là nửa còn nhiều đất nhất

Trước đây tôi ghi "nhà đất bị bỏ lại rất xa" và ngầm hiểu nó khó hơn. Đo lại
với cùng một cái thước thì ngược lại:

| | MAPE | Ngưỡng nhiễu | Cách sàn |
|---|---|---|---|
| Chung cư | 15,57% | 12,13% | **3,4 điểm** |
| Nhà đất | 26,21% | 16,98% | **9,2 điểm** |

Chung cư gần chạm sàn nhiễu của chính nó — mọi nỗ lực thêm chỉ còn 3,4 điểm để
giành. Nhà đất còn 9,2 điểm, tức là nó **đang bị mô hình hoá thiếu**, chứ không
phải "dữ liệu nhà đất vốn nhiễu hơn nên chịu". Công sức đổ vào nhà đất từ giờ
sinh lợi gấp gần ba lần đổ vào chung cư.

> Cảnh báo về cái thước: "8,4% / 19,9%" ghi hôm qua và "12,13% / 16,98%" hôm nay
> là **hai định nghĩa khác nhau**, không phải một sự thay đổi. Định nghĩa dùng
> từ nay: *trung bình không trọng số của hệ số biến thiên (độ lệch chuẩn /
> trung bình) giá trong từng nhóm tin trùng dự án + phường + phòng ngủ + phòng
> vệ sinh + khoảng diện tích 5 m²*. Con số 8,4% cũ tái lập được bằng đúng định
> nghĩa này trên tập nhatot 1.608 tin; con số 19,9% cũ thì không — nó đến từ
> một khoá nhóm tôi không dựng lại được, và gần như chắc chắn là bị thổi lên.

### 2. Cột `nguon` đang làm model tệ đi ở đúng nơi có người dùng thật

`nguon` chiếm 7,4% gain, đứng thứ ba sau diện tích và số phòng ngủ (sau khi bỏ nó, phần gain đó chảy sang diện tích và quận/huyện). Nhưng lúc
người dùng bấm "Định giá", không ai biết "tin này lấy từ sàn nào" — câu hỏi đó
vô nghĩa với người đi mua nhà. Nên `nguon` = NaN, mã hoá tụt về trung bình toàn
cục, một giá trị nằm lửng giữa hai sàn.

| Kịch bản | MAPE |
|---|---|
| Biết nguồn cả khi học lẫn khi đoán | 15,48% ← số đẹp, không thật |
| **Bỏ hẳn cột nguồn** | **15,57%** ← đang dùng |
| Học có nguồn, đoán không có nguồn | 16,13% ← cảnh chạy thật cũ |
| Học có nguồn, đoán gán `'nhatot'` | 16,13% |
| Học có nguồn, đoán gán `'batdongsan'` | 16,99% |

Bỏ cột này làm bảng đánh giá xấu đi 0,09 điểm và làm sản phẩm thật tốt lên 0,56
điểm. Đã bỏ. Đây là loại chênh lệch mà một hệ thống chỉ được đo bằng
cross-validation sẽ không bao giờ nhìn thấy.

### 3. Nút thắt là ĐỘ CHÍNH XÁC toạ độ, không phải số lượng toạ độ

Đặc trưng "giá k căn gần nhất":

| Tập | Cải thiện MAPE | p |
|---|---|---|
| Toàn bộ 3.929 tin | +0,05 điểm | — |
| Riêng 1.278 tin có toạ độ **chính xác** | **+0,87 điểm** | **0,0004** |
| Chính 1.278 tin đó, làm mờ toạ độ về mức đường | +0,20 điểm | 0,33 |

Dòng thứ ba là thí nghiệm quyết định: cùng số tin, cùng model, chỉ hạ độ chính
xác toạ độ xuống — và tác dụng biến mất. Vậy giá trị nằm ở toạ độ *chính xác
tới từng toà nhà*, không phải ở việc có toạ độ nói chung.

Hệ quả cho việc cào: mỗi tin batdongsan có toạ độ thật đáng giá hơn nhiều tin
nhatot chỉ có tên đường.

## Cào thêm bao nhiêu thì đáng?

Khớp `MAPE(n) = sàn + a·n^(−b)` vào đường cong học tập (R² khớp 0,990 và 0,973):

| Số tin | Chung cư | Nhà đất |
|---|---|---|
| hiện tại | 15,57% (3.929) | 26,21% (4.584) |
| 6.000 | 15,02% | 26,08% |
| 8.000 | 14,62% | 25,41% |
| 12.000 | 14,15% | 24,55% |
| 20.000 | 13,68% | 23,59% |
| 50.000 | 13,09% | 22,17% |

Cào từ 3.929 lên **8.327 tin chung cư đổi được đúng 1 điểm MAPE**. Nhà đất cần
8.752 tin cho 1 điểm. Đường cong vẫn còn dốc ở mốc hiện tại — nên câu trả lời
cho "có nên cào tiếp không" là **có**, nhưng biết trước là mua 1 điểm chứ không
phải mua một bước nhảy.

## Số ngoài mẫu, để đưa vào khoá luận

Giữ lại 200 tin trước khi huấn luyện, model không hề thấy chúng:

| | MAPE trung bình | **Trung vị sai lệch** | Độ phủ khoảng |
|---|---|---|---|
| Chung cư | 14,84% | **10,68%** | 89,5% |
| Nhà đất | 25,94% | **15,33%** | 89,0% |

Trung vị thấp hơn trung bình rất nhiều — nghĩa là **một nửa số tin được định
giá sai dưới 10,7%**, còn con số 15,57% bị kéo lên bởi một cái đuôi dài những
tin rao lệch hẳn mặt bằng. Với người dùng, trung vị mới là con số họ cảm nhận.
Bảng cũ `vi_du_dinh_gia_*.xlsx` sinh **trong mẫu** (có dòng dự đoán 6,75 tỷ cho
tin rao 6,75 tỷ) — đã thay bằng bản ngoài mẫu.

## Đặc trưng nào thật sự quan trọng

| Chung cư | | Nhà đất | |
|---|---|---|---|
| Diện tích | 29,7% | Diện tích đất | 12,5% |
| Số phòng ngủ | 11,6% | Pháp lý | 10,1% |
| Quận/huyện | 8,3% | Loại hình | 6,6% |
| Phường mới | 6,6% | Thang máy | 6,5% |
| Số phòng vệ sinh | 4,0% | Quận/huyện | 6,1% |
| Tin nhắc "view" | 3,9% | Phường mới | 5,9% |

Nhà đất phân tán hơn hẳn — không đặc trưng nào vượt 13%. Đó là dấu hiệu khớp
với kết luận ở mục 1: model nhà đất chưa nắm được yếu tố chính, và yếu tố chính
đó nhiều khả năng là **mặt tiền / ngõ / vị trí vi mô** mà dữ liệu hiện tại mô tả
quá thô.

## Kết quả "âm" — vẫn là phần đáng giá nhất

| Phát hiện | Số đo |
|---|---|
| Điểm tiện ích trong model đầy đủ | **0,00 điểm — vô ích** |
| Điểm tiện ích khi bỏ toạ độ và mã hoá phường | +4,73 / +7,78 → nó chỉ là *hàng thay thế* của toạ độ |
| Trọng số tiện ích học từ dữ liệu (bootstrap 500×) | **không trọng số nào phân biệt được với 0** |
| Thêm 1.049 dòng từ nguồn khác | +0,31, KTC [−0,30; +0,89], p = 0,28 — không có ý nghĩa |
| Thêm 2.327 dòng | +0,55, KTC [−0,10; +1,17], p = 0,054 — ở ranh giới |
| Chuẩn hoá mức giá theo nguồn (chia trung vị) | ngưỡng nhiễu **tăng** 12,13% → 13,79% — chênh 37% giữa hai sàn không phải hệ số nhân đều |
| Tuổi tin ảnh hưởng giá (khống chế phường) | 0,0%, p = 1,00 |
| Vị trí trang ảnh hưởng giá | +0,50%/trang, p = 1,4e-06 |
| Phường mới vs phường cũ làm khoá | 16,00 vs 16,01; 27,24 vs 27,75 |

## Hai lỗi tôi tự gây ra và tự bắt được — nên viết vào khoá luận

**Rò rỉ mục tiêu qua khớp tiền tố.** Tôi viết `_cot_co(df, ("gia_m2_", ...))`
cho gọn. Tiền tố đó khớp luôn `gia_m2_nguon` — giá/m² do chính sàn công bố, tức
là mục tiêu chia cho diện tích. MAPE tụt từ 15,75% xuống **7,87%** và trông như
một bước nhảy vọt. Không có lỗi nào báo ra; thứ duy nhất phát hiện được nó là
sự nghi ngờ trước một con số quá đẹp.

**Khoảng tin cậy nói dối.** Đặc trưng láng giềng tính một lần trên toàn bộ dữ
liệu trước khi chia train/hiệu chuẩn làm phần hiệu chuẩn conformal thấy model
"giỏi" hơn thật, và khoảng tin cậy hẹp lại giả tạo: **±30% thay vì ±42% thật**.
Đã tắt đặc trưng này ở bản chạy thật (`DUNG_LANG_GIENG = False`). Khoảng tin cậy
là thứ bán được của hệ thống này — thà bỏ 0,05 điểm MAPE còn hơn để nó nói dối.

## Hạn chế nền tảng, phải nói ở phần đặt vấn đề chứ không giấu xuống cuối

Toàn bộ hệ thống dự đoán **giá RAO**, không phải giá giao dịch. Không sửa được
bằng kỹ thuật vì không có dữ liệu giao dịch công khai ở Việt Nam. Mọi con số
MAPE trong khoá luận đều phải đọc là "sai lệch so với giá người bán *rao*".

## Việc tiếp theo, theo thứ tự đã đổi

1. **Chạy `app.py` thật trong trình duyệt.** Vẫn chưa ai mở. Rủi ro giao hàng
   lớn nhất còn lại, và mục tiêu bạn đặt ra là "sản phẩm dùng được thật".
2. **Kéo nhà đất lên** — giờ là hướng sinh lợi cao nhất (9,2 điểm khoảng trống
   so với 3,4 của chung cư). Hai việc cụ thể: (a) cào nhà đất từ batdongsan để
   có toạ độ chính xác, (b) bóc mặt tiền / độ rộng ngõ kỹ hơn từ mô tả.
3. Cào tiếp chung cư — vẫn đáng, nhưng biết trước là 8.327 tin đổi 1 điểm.
4. Bản đồ choropleth theo phường mới (cần tải ranh giới từ OSM).
5. Tìm kiếm bằng câu tiếng Việt.

## Chạy lại toàn bộ

```
cd pipeline
python gop_nguon.py --do-luong
python gazetteer_phuong.py --ap-dung
python ban_do.py
python tien_ich.py --cham-diem
python _scratch/do_lai_tap_gop.py        # bảng số đầy đủ
python _scratch/vi_du_dinh_gia.py        # ví dụ ngoài mẫu
python _scratch/thu_bo_nguon.py          # kiểm tra cột nguồn
```
