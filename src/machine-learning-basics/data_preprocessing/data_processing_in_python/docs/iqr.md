# Phân tích và nghiên cứu về IQR (Interquartile Range)

> Tài liệu tóm tắt, dễ hiểu về khái niệm **IQR**, ý nghĩa, cách tính, ứng dụng (phát hiện outlier), ví dụ Python có thể chạy được và lưu ý khi sử dụng.

---

## Mục lục

1. [IQR là gì?](#iqr-l%C3%A0-g%C3%AC)
2. [Cách tính IQR](#c%C3%A1ch-t%C3%ADnh-iqr)
3. [Ứng dụng chính của IQR](#ứng-dụng-ch%C3%ADnh-c%E1%BB%A7a-iqr)
4. [Quy tắc 1.5 \* IQR để phát hiện outlier](#quy-t%E1%BA%BFt-15--iqr-đ%E1%BB%83-ph%C3%A1t-hi%E1%BB%87n-outlier)
5. [Ví dụ Python — step-by-step](#ví-d%E1%BB%A5-python--step-by-step)
6. [Trực quan hóa với boxplot](#tr%E1%BB%B1c-quan-h%C3%B3a-v%E1%BB%9Bi-boxplot)
7. [Ưu điểm, hạn chế và chú ý](#ưu-%C4%91i%E1%BB%83m-h%E1%BA%A1n-ch%E1%BA%A5t-v%C3%A0-ch%C3%BA-%C3%BD)
8. [Tài liệu tham khảo & đọc thêm](#t%C3%A0i-li%E1%BB%87u-tham-kh%E1%BA%A3o--%C4%91%E1%BB%8Fc-th%C3%AAm)

---

## IQR là gì?

**Interquartile Range (IQR)** là khoảng giữa tứ phân vị thứ nhất (Q1, 25%) và tứ phân vị thứ ba (Q3, 75%).

$\text{IQR} = Q3 - Q1$

IQR đo độ phân tán của phần trung tâm (giữa 25% và 75%) của dữ liệu — ít bị ảnh hưởng bởi các giá trị ngoại lai so với phương sai hay độ lệch chuẩn.

---

## Cách tính IQR

1. Sắp xếp dữ liệu theo thứ tự tăng dần.
2. Tìm Q1 (phân vị 25%) và Q3 (phân vị 75%).
3. IQR = Q3 - Q1.

Với bộ dữ liệu rời rạc, có thể dùng nội suy khi phần trăm rơi giữa hai điểm dữ liệu (hầu hết thư viện thống kê xử lý tự động việc này).

---

## Ứng dụng chính của IQR

* **Đo phân tán**: biết được độ dày phần giữa của dữ liệu.
* **Phát hiện outlier**: dùng quy tắc 1.5 \* IQR (hoặc 3 \* IQR cho outlier nghiêm ngặt hơn).
* **So sánh phân bố**: khi dùng boxplot để so sánh nhiều nhóm.

---

## Quy tắc 1.5 \* IQR để phát hiện outlier

* Lower bound = Q1 - 1.5 \* IQR
* Upper bound = Q3 + 1.5 \* IQR

Các điểm nằm ngoài khoảng \[Lower bound, Upper bound] thường được coi là outlier (cần kiểm tra thêm, không tự động xóa).

---

## Ví dụ Python — step-by-step

Dưới đây là ví dụ thực tế — cả bằng `numpy`/`pandas` và trực quan bằng `matplotlib`.

```python
# example_iqr.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Dữ liệu ví dụ
data = np.array([10, 12, 12, 13, 12, 14, 15, 16, 18, 19, 120, 20, 21, 22, 23])

# 1) Tính Q1, Q3, IQR (numpy)
q1 = np.percentile(data, 25)
q3 = np.percentile(data, 75)
iqr = q3 - q1
print(f"Q1={q1}, Q3={q3}, IQR={iqr}")

# 2) Tìm ngưỡng outlier theo quy tắc 1.5 * IQR
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
print(f"Lower bound={lower_bound}, Upper bound={upper_bound}")

# 3) Liệt kê outliers
outliers = data[(data < lower_bound) | (data > upper_bound)]
print(f"Outliers: {outliers}")

# 4) Bằng pandas (thường dùng trong thực tế với DataFrame)
df = pd.DataFrame({"value": data})
Q1 = df['value'].quantile(0.25)
Q3 = df['value'].quantile(0.75)
IQR = Q3 - Q1
print(df.describe())
print(f"Pandas Q1={Q1}, Q3={Q3}, IQR={IQR}")

# 5) Vẽ boxplot để trực quan
plt.figure(figsize=(6,4))
plt.boxplot(data, vert=False)
plt.title('Boxplot ví dụ - minh họa IQR và outliers')
plt.xlabel('Value')
plt.show()
```

**Hướng dẫn chạy:**

* Lưu file `example_iqr.py` rồi chạy `python example_iqr.py`.
* Hoặc chèn đoạn code vào Jupyter notebook cell để xem trực quan.

---

## Trực quan hóa với boxplot

Boxplot hiển thị:

* Một hộp từ Q1 đến Q3 (chiều dài = IQR)
* Một đường ngang trong hộp là median (Q2)
* "Râu" (whiskers) thường là các điểm tới giới hạn không phải outlier (cách xác định whiskers tùy library, phổ biến là 1.5 \* IQR)
* Điểm nằm ngoài whiskers được đánh dấu là outlier

Boxplot rất tiện để so sánh nhanh nhiều nhóm dữ liệu (ví dụ: điểm số theo lớp, theo nhóm điều trị,...).

---

## Ưu điểm, hạn chế và chú ý

**Ưu điểm**:

* Không nhạy cảm với outlier (IQR chỉ dùng phần giữa của dữ liệu).
* Đơn giản, dễ hiểu và triển khai.

**Hạn chế**:

* Mất thông tin phần đuôi (tails) của phân bố.
* Quy tắc 1.5 \* IQR là heuristic — không phù hợp cho mọi ngữ cảnh (ví dụ: phân bố có đuôi dày hoặc dữ liệu mang tính chuỗi thời gian có xu hướng lớn/nhỏ theo thời gian).
* Không thay thế phân tích sâu: nên kiểm tra lý do vì sao giá trị bị coi là outlier (lỗi đo, nhập liệu, hay giá trị thật sự hiếm).

**Chú ý khi áp dụng**:

* Với dữ liệu nhiều nhóm, tính IQR theo từng nhóm riêng (không gộp chung) nếu bạn muốn phát hiện outlier theo ngữ cảnh nhóm.
* Trong xử lý tự động, thay vì xóa outlier ngay lập tức, cân nhắc gán nhãn, điều tra nguồn gốc hoặc thay thế bằng phương pháp robust.

---

## Tài liệu tham khảo & đọc thêm

* Wikipedia — Interquartile Range
* Any thống kê sơ cấp về phân vị và boxplot
