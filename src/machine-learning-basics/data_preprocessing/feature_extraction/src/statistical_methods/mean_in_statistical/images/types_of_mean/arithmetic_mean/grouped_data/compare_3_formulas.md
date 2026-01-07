# 📊 So sánh các công thức **Arithmetic Mean** cho dữ liệu nhóm (Grouped Data)

## 🧠 1. Tổng quan về Arithmetic Mean (Trung bình cộng)

**Arithmetic Mean** là một trong những chỉ số thống kê cơ bản và quan trọng nhất, thể hiện **giá trị trung tâm** của một tập dữ liệu.  
Với dữ liệu **đã được nhóm (grouped data)** — tức là dữ liệu đã chia thành các **lớp (class intervals)** và có **tần số (frequency)** tương ứng — ta không thể tính trung bình cộng đơn giản bằng tổng các giá trị chia cho số phần tử. Thay vào đó, ta phải sử dụng **công thức trung bình cộng cho dữ liệu nhóm**.

Có **3 cách phổ biến** để tính Arithmetic Mean cho dữ liệu nhóm:

1. **Direct Method (Phương pháp trực tiếp)**  
2. **Assumed Mean Method (Phương pháp trung bình giả định)**  
3. **Step-Deviation Method (Phương pháp sai lệch từng bước)**

---

## 📐 2. So sánh 3 công thức Arithmetic Mean (Grouped Data)

| Tiêu chí | Direct Method | Assumed Mean Method | Step-Deviation Method |
|----------|----------------|----------------------|------------------------|
| **Công thức** | \(\bar{x} = \dfrac{\sum f m}{\sum f}\) | \(\bar{x} = A + \dfrac{\sum f d}{\sum f}\) | \(\bar{x} = A + \dfrac{\sum f u}{\sum f} \times h\) |
| **Ký hiệu** | \(f\): tần số, \(m\): trung điểm mỗi lớp | \(A\): giá trị trung bình giả định, \(d = m - A\) | \(u = \dfrac{m - A}{h}\), \(h\): độ rộng lớp |
| **Đặc điểm** | - Dễ hiểu và trực tiếp<br>- Tính toán thủ công dễ khi dữ liệu nhỏ | - Giảm kích thước số khi giá trị \(m\) lớn<br>- Tiện hơn khi dữ liệu lớn | - Tối giản phép tính nhất<br>- Dễ thực hiện khi dữ liệu lớn và khoảng lớp bằng nhau |
| **Ưu điểm** | - Công thức đơn giản<br>- Không cần chọn điểm giả định | - Giảm sai số tính toán<br>- Dễ tính hơn khi giá trị lớn | - Tính toán nhanh nhất<br>- Lý tưởng khi thao tác bằng tay |
| **Nhược điểm** | - Tính toán có thể dài dòng nếu dữ liệu lớn | - Cần chọn trung bình giả định hợp lý | - Chỉ hiệu quả khi khoảng lớp bằng nhau |
| **Khi nên dùng** | Khi số lớp nhỏ, giá trị dễ tính | Khi số lớp lớn, giá trị lớn | Khi số lớp lớn và khoảng cách đều |

---

## 📊 3. Ví dụ minh họa & Use Cases thực tế

### 🧪 Ví dụ dữ liệu

| Khoảng điểm (Class Interval) | Tần số (f) |
|------------------------------|------------|
| 0 - 10                       | 5          |
| 10 - 20                      | 8          |
| 20 - 30                      | 15         |
| 30 - 40                      | 10         |
| 40 - 50                      | 2          |

---

### 3.1 🔎 Direct Method (Phương pháp trực tiếp)

**Bước tính:**

1. Tính trung điểm \(m\) mỗi lớp: (0-10) → 5, (10-20) → 15, (20-30) → 25, ...
2. Tính \(f \times m\)
3. Áp dụng công thức:

\[
\bar{x} = \dfrac{\sum f m}{\sum f}
\]

**Tình huống thực tế:**  
👉 Dùng khi xử lý tập dữ liệu nhỏ trong khảo sát học sinh (~10-20 mẫu), ví dụ điểm kiểm tra của một lớp học.

---

### 3.2 ⚖️ Assumed Mean Method (Trung bình giả định)

**Bước tính:**

1. Chọn giá trị trung bình giả định \(A = 25\)
2. Tính \(d = m - A\)
3. Tính \(f \times d\)
4. Áp dụng công thức:

\[
\bar{x} = A + \dfrac{\sum f d}{\sum f}
\]

**Tình huống thực tế:**  
👉 Dùng khi dữ liệu lớn như khảo sát mức thu nhập của 10.000 người. Việc chọn một trung bình giả định gần với giá trị trung tâm giúp giảm sai số khi tính toán.

---

### 3.3 📉 Step-Deviation Method (Sai lệch từng bước)

**Bước tính:**

1. Chọn \(A = 25\), \(h = 10\) (độ rộng lớp)
2. Tính \(u = \dfrac{m - A}{h}\)
3. Tính \(f \times u\)
4. Áp dụng công thức:

\[
\bar{x} = A + \dfrac{\sum f u}{\sum f} \times h
\]

**Tình huống thực tế:**  
👉 Phương pháp này rất hữu ích trong thống kê quy mô lớn, ví dụ như điều tra dân số, kết quả bầu cử, dữ liệu kinh tế quốc gia… khi cần tính nhanh bằng tay hoặc trên giấy mà không có máy tính.

---

## 🧭 4. Tổng kết

| Tình huống thực tế | Phương pháp khuyến nghị |
|--------------------|--------------------------|
| Dữ liệu nhỏ, số lớp ít | ✅ Direct Method |
| Dữ liệu lớn, giá trị trung bình cao | ✅ Assumed Mean Method |
| Dữ liệu lớn, khoảng lớp đều nhau, cần tính nhanh | ✅ Step-Deviation Method |

> 💡 **Gợi ý:** Trong thực tế, các nhà thống kê hoặc nhà phân tích dữ liệu thường ưu tiên **Assumed Mean** hoặc **Step-Deviation** khi làm việc với dữ liệu quy mô lớn, vì chúng giúp tính toán nhanh và giảm sai số làm tròn.

---

## 📚 Tài liệu tham khảo

- 📘 *Fundamentals of Statistics* – [Book]
- 📊 American Statistical Association – Statistics Education Resources
- 📈 *Introductory Statistics* – [Book]

---

✅ **Tóm lại:** Cả ba phương pháp đều cho kết quả trung bình cộng giống nhau, chỉ khác ở **cách tính và độ thuận tiện**. Việc chọn phương pháp phụ thuộc vào **quy mô dữ liệu, yêu cầu tính toán, và tính chất khoảng lớp.**
