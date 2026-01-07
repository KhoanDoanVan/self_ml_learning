# 📘 Poisson Distribution trong Machine Learning

## 🧠 1. Định nghĩa cơ bản

**Phân phối Poisson** mô tả **xác suất xảy ra một số lượng sự kiện nhất
định** trong một khoảng thời gian (hoặc không gian) cố định, **khi biết
rằng các sự kiện xảy ra độc lập và với tần suất trung bình λ (lambda)**.

Công thức xác suất:

\[ P(X = k) = `\frac{e^{-\lambda} \lambda^k}{k!}`{=tex} \]

-   ( k ): số sự kiện xảy ra (0, 1, 2, 3, ...)\
-   ( `\lambda `{=tex}): tần suất trung bình\
-   ( e ): hằng số Euler (≈ 2.71828)

------------------------------------------------------------------------

## 📊 2. Khi nào dùng Poisson Distribution?

Phân phối Poisson dùng khi: - Dữ liệu là **số lần xảy ra của sự kiện**
(count data) - Các sự kiện xảy ra **ngẫu nhiên và độc lập** - Tốc độ
trung bình (λ) **ổn định theo thời gian hoặc không gian**

**Ví dụ:** - Số email đến trong 1 giờ\
- Số lỗi (bug) trong 1000 dòng code\
- Số khách hàng đến cửa hàng trong 1 phút\
- Số cuộc gọi tới tổng đài trong 1 giờ

------------------------------------------------------------------------

## 🤖 3. Ứng dụng trong Machine Learning

### 🔹 (a) Modeling Count Data (Poisson Regression)

**Poisson Regression** là một biến thể của **Generalized Linear Model
(GLM)**, dùng khi biến phụ thuộc là **count**.

\[ `\log`{=tex}(`\lambda`{=tex}) = `\beta`{=tex}\_0 + `\beta`{=tex}\_1
x_1 + `\beta`{=tex}\_2 x_2 + ... \]

**Ứng dụng:** - Dự đoán số khách truy cập website\
- Dự đoán số sản phẩm được mua mỗi giờ\
- Dự đoán số yêu cầu đến server\
- Dự đoán số tai nạn giao thông

------------------------------------------------------------------------

### 🔹 (b) Event Forecasting trong Time Series

Dùng để **mô hình hóa tần suất sự kiện hiếm (rare events)** theo thời
gian: - Số lần crash server/ngày\
- Số đơn hàng vượt ngưỡng/giờ\
- Số hành động người dùng trong app

Kết hợp với Bayesian hoặc ARIMA để **dự báo xác suất sự kiện hiếm**.

------------------------------------------------------------------------

### 🔹 (c) Anomaly Detection

Dùng để phát hiện bất thường khi số sự kiện **vượt xa tần suất trung
bình λ**.

**Ví dụ:** - Trung bình 5 lỗi/giờ → đột nhiên 25 lỗi/giờ ⇒ bất thường\
- Log hệ thống đột ngột có nhiều lỗi liên tiếp

------------------------------------------------------------------------

### 🔹 (d) NLP

-   Mô hình hóa **tần suất từ hiếm**
-   Làm baseline cho **word frequency modeling** trong mô hình ngôn ngữ
    xác suất

------------------------------------------------------------------------

### 🔹 (e) Computer Vision

-   Dùng cho **rare object counting** --- mô hình hóa sự xuất hiện ngẫu
    nhiên của vật thể hiếm trong ảnh.

------------------------------------------------------------------------

## 🧩 4. Ưu và Nhược điểm

  -----------------------------------------------------------------------
  Ưu điểm                        Nhược điểm
  ------------------------------ ----------------------------------------
  Đơn giản, dễ hiểu              Giả định mean = variance (λ) → không
                                 đúng nếu overdispersion

  Dễ dùng cho dữ liệu đếm        Không phù hợp cho dữ liệu có quá nhiều
                                 zero

  Có thể mở rộng sang GLM        Không mô hình hóa được sự phụ thuộc giữa
                                 các sự kiện
  -----------------------------------------------------------------------

📍Giải pháp: dùng **Negative Binomial Regression** hoặc **Zero-Inflated
Poisson (ZIP)**.

------------------------------------------------------------------------

## ⚙️ 5. Python Minh Họa

``` python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson

lambda_ = 4
x = np.arange(0, 15)
pmf = poisson.pmf(x, lambda_)

plt.bar(x, pmf)
plt.title("Poisson Distribution (λ=4)")
plt.xlabel("Số sự kiện (k)")
plt.ylabel("Xác suất P(X=k)")
plt.show()
```

------------------------------------------------------------------------

## 💡 Tóm tắt

  -----------------------------------------------------------------------
  Thuộc tính                                  Mô tả
  ------------------------------------------- ---------------------------
  Dạng dữ liệu                                Count data

  Tham số chính                               λ (mean rate)

  Ứng dụng chính                              Count regression, anomaly
                                              detection, event
                                              forecasting

  Liên quan tới                               Negative Binomial,
                                              Exponential, Gamma

  Mô hình ML thường dùng                      Poisson Regression, GLM,
                                              Bayesian Poisson Process
  -----------------------------------------------------------------------
