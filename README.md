# Phát hiện gian lận thẻ tín dụng sử dụng Deep Autoencoders

Dự án này xây dựng một hệ thống phát hiện bất thường (Anomaly Detection) để nhận diện các giao dịch thẻ tín dụng gian lận sử dụng kiến trúc mạng nơ-ron **Deep Autoencoder**.

Hệ thống hoạt động dựa trên nguyên lý học không giám sát (unsupervised learning): mô hình được huấn luyện **chỉ trên các giao dịch bình thường** để học cách nén và tái tạo lại chúng. Khi gặp một giao dịch gian lận (bất thường), mô hình sẽ không thể tái tạo tốt, dẫn đến lỗi tái tạo (reconstruction error) cao, từ đó phát hiện ra gian lận.

**Link Dataset**: [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

## Cấu trúc Thư mục

```
├── fraud_detection_autoencoder.ipynb  # Notebook chính: Chứa toàn bộ quy trình từ EDA, xử lý dữ liệu đến huấn luyện và đánh giá
├── CreditCard/                        # Thư mục dữ liệu
│   └── creditcard.csv                 # Dữ liệu giao dịch (đã được ẩn danh hóa PCA)
└── README.md                          # Tài liệu mô tả
```

### 1. `fraud_detection_autoencoder.ipynb`
Notebook này thực hiện toàn bộ pipeline của dự án, từ khâu xử lý dữ liệu thô đến đánh giá mô hình.

*   **Mục tiêu:** Xây dựng mô hình Autoencoder để phân tách giao dịch bình thường và gian lận dựa trên lỗi tái tạo.
*   **Quy trình:**
    *   **Khám phá dữ liệu (EDA):**
        *   Kiểm tra sự mất cân bằng dữ liệu nghiêm trọng (Gian lận chỉ chiếm một phần rất nhỏ).
        *   Trực quan hóa phân phối các lớp (Class Distribution).
    *   **Tiền xử lý (Preprocessing):**
        *   *Loại bỏ đặc trưng:* Bỏ cột `Time` vì không đóng góp nhiều cho phương pháp này.
        *   *Chuẩn hóa:* Sử dụng `StandardScaler` để chuẩn hóa cột `Amount` và các đặc trưng `V1-V28` về cùng một thang đo.
        *   *Chia dữ liệu (Data Splitting):* Áp dụng chiến lược chia tách đặc thù cho bài toán phát hiện bất thường:
            *   `Train set`: Chỉ chứa các giao dịch **Bình thường** (để mô hình học "sự bình thường").
            *   `Test set`: Chứa hỗn hợp giao dịch Bình thường còn lại và **Tất cả** giao dịch Gian lận (để đánh giá thực tế).
    *   **Xây dựng Mô hình (Modeling):**
        *   Sử dụng framework **PyTorch**.
        *   *Kiến trúc:* Deep Autoencoder đối xứng.
            *   Encoder: Nén đầu vào (29 chiều) xuống không gian ẩn (10 chiều) qua các lớp trung gian (24, 16).
            *   Decoder: Tái tạo lại đầu vào từ không gian ẩn.
        *   *Hàm kích hoạt:* Tanh.
    *   **Huấn luyện (Training):**
        *   Hàm mất mát: MSE (Mean Squared Error).
        *   Tối ưu hóa: Adam Optimizer.
        *   Mô hình học cách giảm thiểu sai số tái tạo trên tập dữ liệu sạch.
    *   **Đánh giá & Ngưỡng (Evaluation & Thresholding):**
        *   Tính toán **Reconstruction Error (MSE)** trên tập Test.
        *   Trực quan hóa sự khác biệt về phân phối lỗi giữa lớp Bình thường và Gian lận.
        *   Xác định **Ngưỡng (Threshold)** tối ưu để phân loại dựa trên **Precision-Recall Curve** và tối ưu hóa **F1 Score**.

## Yêu cầu cài đặt

Dự án yêu cầu các thư viện Python sau:

```bash
pip install pandas numpy matplotlib seaborn torch scikit-learn tqdm
```
