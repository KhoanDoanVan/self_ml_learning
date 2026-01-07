# Gaussian Distribution vs Gaussian Constant

## 🧩 1. Gaussian Distribution (Phân phối Gaussian)

**Tên khác:** Normal Distribution — Phân phối chuẩn

### 📘 Công thức:

\[
f(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{(x - \mu)^2}{2\sigma^2}}
\]

Trong đó:
- \( \mu \): mean (trung bình)
- \( \sigma \): standard deviation (độ lệch chuẩn)
- \( f(x) \): xác suất (mật độ) xuất hiện của giá trị \( x \)

### 📊 Đặc điểm:
- Dạng **đường cong hình chuông đối xứng** quanh giá trị trung bình \( \mu \)
- Ứng dụng trong:
  - Thống kê và xác suất
  - Machine Learning (Gaussian Naive Bayes, Gaussian Process, v.v.)
  - Xử lý tín hiệu và noise modeling

---

## ⚙️ 2. Gaussian Constant (Hằng số Gaussian)

### 📘 Công thức:
\[
\frac{1}{\sqrt{2\pi}} \approx 0.3989422804
\]

Hoặc trong dạng chuẩn hoá đầy đủ:
\[
\frac{1}{\sigma\sqrt{2\pi}}
\]

### 💡 Vai trò:
- Là **hệ số chuẩn hoá** trong công thức phân phối Gaussian.
- Đảm bảo **tổng xác suất của toàn bộ phân phối bằng 1**.
- Không thay đổi theo \( x \), chỉ là **hằng số toán học**.

---

## 🧠 3. So sánh nhanh

| Thuộc tính | Gaussian Distribution | Gaussian Constant |
|-------------|----------------------|-------------------|
| Loại | Hàm/phân phối xác suất | Hằng số toán học |
| Biểu thức | \( f(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{(x-\mu)^2}{2\sigma^2}} \) | \( \frac{1}{\sqrt{2\pi}} \) |
| Ý nghĩa | Mô tả xác suất của các giá trị quanh trung bình | Chuẩn hoá phân phối để có tổng = 1 |
| Phụ thuộc vào μ, σ | ✅ Có | ❌ Không |
| Dạng biểu đồ | Đường cong hình chuông | Không có (chỉ là giá trị cố định) |

---

## 📈 4. Ví dụ trực quan (Python)

```python
import numpy as np
import matplotlib.pyplot as plt

mu, sigma = 0, 1
x = np.linspace(-4, 4, 200)
f_x = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

plt.plot(x, f_x, label="Gaussian Distribution")
plt.axhline(1 / np.sqrt(2 * np.pi), color='r', linestyle='--', label="Gaussian Constant")
plt.legend()
plt.title("Gaussian Distribution vs Gaussian Constant")
plt.show()
