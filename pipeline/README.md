# Pipeline làm sạch dữ liệu — Định giá BĐS Hà Nội

Bộ script làm sạch dữ liệu thô crawl từ nhatot.com, thay cho bước tiền xử lý cũ.

## Cách chạy

```bash
pip install pandas openpyxl scikit-learn xgboost matplotlib

python clean_chungcu.py --input nhatot.xlsx        --outdir data_clean
python clean_nhadat.py  --input nhatot_nhadat.xlsx --outdir data_clean
```

## Các file mã nguồn

| File | Vai trò |
|---|---|
| `clean_common.py` | Hàm dùng chung: parse giá/diện tích/địa chỉ, bóc khối thuộc tính, sửa lỗi đơn vị, khử trùng lặp, lọc ngoại lai, lớp báo cáo |
| `extract_text.py` | Trích xuất tầng, mặt tiền, số tầng, độ rộng ngõ và các cờ tiện ích từ mô tả tự do |
| `gazetteer.py` | Bảng tra chuẩn hoá tên 51 dự án chung cư Hà Nội (tách ra từ `app.py`) |
| `clean_chungcu.py` | Luồng làm sạch chung cư |
| `clean_nhadat.py` | Luồng làm sạch nhà đất |
| `train_eval.py` | Khung đánh giá chéo với target encoding an toàn + đường cong học tập |

## Ba điều quan trọng nhất của pipeline này

**1. Không tính target encoding trong file dữ liệu.**
Bản `ready_to_train` cũ có sẵn ba cột `*_encoded` được tính trên toàn bộ dữ liệu
rồi mới chia train/test — đó là rò rỉ mục tiêu. Encoding giờ được tính trong
từng fold, ở `train_eval.py`. File dữ liệu sạch chỉ chứa dữ liệu thô đã chuẩn hoá.

**2. Ưu tiên sửa hơn xoá, nhưng phải kiểm chứng.**
Lỗi đơn vị giá ("2950 tỷ" cho căn 43 m²) được sửa thay vì loại bỏ. Mọi dòng
được sửa đều ghi ra `*_sua_don_vi.xlsx` để soi tay. Khoảng chấp nhận cho giá
trị sau khi sửa được suy ra từ phân vị 1–99% của chính dữ liệu, không đặt tay —
đặt tay quá rộng thì luật sẽ biến một giá trị sai thành một giá trị sai khác.

**3. Không dùng ngưỡng ngoại lai toàn cục.**
Nhà mặt phố Hoàn Kiếm 1,5 tỷ/m² là thật; nhà ngõ Hoàng Mai cùng giá là tin ảo.
Lọc theo IQR×3 trong từng nhóm (quận, hoặc quận × loại hình).

## Đầu ra

| File | Nội dung |
|---|---|
| `chungcu_clean.xlsx` | 1.608 tin chung cư thương mại đã sạch |
| `chungcu_phan_khuc_khac.xlsx` | 587 tin tập thể / cư xá / căn hộ mini — giữ riêng, không vứt |
| `nhadat_clean.xlsx` | 4.584 tin nhà đất đã sạch |
| `*_ngoai_lai.xlsx` | Các dòng bị gắn cờ ngoại lai, để kiểm tra lại |
| `*_bao_cao_lam_sach.xlsx` | Bảng còn bao nhiêu dòng sau mỗi bước — dùng cho chương phương pháp |
| `bang_ablation.xlsx` | Bảng nghiên cứu tách biệt: từng bước cải tiến đóng góp bao nhiêu |
| `duong_cong_hoc_tap*.xlsx/.png` | MAPE theo số mẫu + ngoại suy |

## Việc còn phải làm

- [ ] `app.py` nên `from gazetteer import normalize_project_name` thay vì giữ bản sao bảng tra.
- [ ] Thêm selector kiểu **Link** vào sitemap chung cư khi crawl lại — hiện chung cư
      không có URL tin gốc nên khử trùng lặp không có khoá tuyệt đối.
- [ ] Geocode `geocode_query` để sinh biến không gian (cột đã sẵn sàng trong cả hai file sạch).
