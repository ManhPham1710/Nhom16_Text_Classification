# NHÓM 16 — PHÂN LOẠI CHỦ ĐỀ VĂN BẢN TIẾNG VIỆT

Ứng dụng học máy phân loại văn bản tiếng Việt thành 10 chủ đề bằng TF-IDF
và các mô hình Multinomial Naive Bayes, Decision Tree, Linear SVM.

## 1. Cấu trúc thư mục

```text
Nhom16_Text_Classification/
├── app.py
├── train.py
├── evaluate.py
├── data_loader.py
├── requirements.txt
├── README.md
├── data/
│   └── raw/
│       ├── train.txt
│       └── test.txt
├── models/
├── results/
└── src/
    ├── __init__.py
    ├── config.py
    └── preprocessing.py
```

## 2. Yêu cầu

- Windows 10/11
- Python 3.10 trở lên (đã kiểm thử mục tiêu tương thích Python 3.13)
- VS Code khuyến nghị

## 3. Cài đặt trên Windows

Mở Terminal tại thư mục dự án:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Nếu PowerShell chặn kích hoạt môi trường, có thể chạy:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## 4. Dữ liệu

Đặt hai tệp `train.txt` và `test.txt` vào `data/raw/`.
Mỗi dòng có định dạng:

```text
__label__the_thao Nội dung văn bản...
```

Không đổi tên nhãn trong dữ liệu. Tập Test không được dùng để chọn mô hình.

## 5. Huấn luyện

```powershell
python train.py
```

Quy trình:
- Chia tập Train thành 80% Train nội bộ và 20% Validation.
- Dùng stratify để giữ tỷ lệ nhãn.
- So sánh 3 mô hình bằng Macro-F1 trên Validation.
- Chọn mô hình tốt nhất.
- Fit lại vectorizer và mô hình được chọn trên toàn bộ tập Train.
- Lưu mô hình, vectorizer và metadata.

## 6. Đánh giá độc lập

Sau khi huấn luyện hoàn tất:

```powershell
python evaluate.py
```

Kết quả được lưu trong `results/`, gồm metrics tổng hợp, classification report,
confusion matrix, confusion matrix chuẩn hóa và các mẫu dự đoán sai.

Chỉ báo cáo số liệu Test sau khi lệnh này chạy thành công. Không chọn lại mô hình
dựa trên kết quả Test.

## 7. Chạy giao diện demo

```powershell
streamlit run app.py
```

Streamlit sẽ hiển thị địa chỉ localhost trong Terminal, thường là
`http://localhost:8501`.

## 8. Các nhãn

| Mã nhãn | Chủ đề |
|---|---|
| the_thao | Thể thao |
| chinh_tri_xa_hoi | Chính trị - Xã hội |
| phap_luat | Pháp luật |
| suc_khoe | Sức khỏe |
| doi_song | Đời sống |
| van_hoa | Văn hóa |
| the_gioi | Thế giới |
| kinh_doanh | Kinh doanh |
| vi_tinh | Vi tính |
| khoa_hoc | Khoa học |

## 9. Thành viên

- Phạm Đức Mạnh
- Đinh Quang Minh
- Đinh Hùng Thiện

Giảng viên hướng dẫn: ThS. Lê Thị Thùy Trang.

## 10. Lưu ý về kết quả

Các chỉ số trong `validation_metrics.csv` là kết quả chọn mô hình.
Các chỉ số trong `test_metrics.json` mới là kết quả đánh giá cuối cùng trên Test.
Không ghi số liệu minh họa hoặc tự suy diễn số liệu khi chưa chạy thực nghiệm.
