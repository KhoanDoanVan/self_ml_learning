# 🧠 So sánh HOG và CNN trong Image Feature Extraction

## 1. Tổng quan

Trong xử lý ảnh, việc **trích xuất đặc trưng (feature extraction)** là bước quan trọng để biểu diễn hình ảnh dưới dạng số mà mô hình có thể hiểu được.

Hai phương pháp cơ bản và phổ biến nhất là:

- **HOG (Histogram of Oriented Gradients)** — phương pháp **thủ công (handcrafted)**.  
- **CNN (Convolutional Neural Network)** — phương pháp **học tự động (learned)**.

---

## 2. HOG – Trích xuất đặc trưng dựa trên hướng gradient

**Nguyên lý hoạt động:**
1. Chia ảnh thành các ô nhỏ (cell, ví dụ 8×8 pixel).  
2. Tính **gradient** (độ lớn và hướng thay đổi cường độ sáng) cho từng pixel.  
3. Lập **histogram hướng gradient** trong mỗi cell → biểu diễn “cấu trúc cạnh”.  
4. Chuẩn hóa trên nhiều vùng để tạo vector đặc trưng cuối cùng.

**Đặc điểm:**
- Dựa vào **hình dạng và biên cạnh**, không quan tâm đến màu sắc RGB.  
- Không cần huấn luyện, chỉ là toán học xử lý ảnh thuần túy.  
- Phù hợp với bài toán đơn giản: phát hiện người, xe, khuôn mặt,...

---

## 3. CNN – Trích xuất đặc trưng bằng mạng tích chập

**Nguyên lý hoạt động:**
- Gồm nhiều **lớp tích chập (convolutional layers)** tự học ra các bộ lọc (filters).  
- Các lớp đầu học đặc trưng đơn giản (cạnh, góc), lớp sâu hơn học đặc trưng phức tạp (mắt, vật thể).  
- Kết quả cuối là **feature map** đại diện cho nội dung ảnh.  
- Sau đó dùng các lớp fully-connected để phân loại hoặc nhận diện đối tượng.

**Đặc điểm:**
- Tự động học đặc trưng từ dữ liệu (không cần định nghĩa thủ công).  
- Làm việc trực tiếp với ảnh RGB hoặc nhiều kênh.  
- Cần tập huấn luyện lớn và tài nguyên tính toán mạnh.  
- Hiệu năng vượt trội trong hầu hết bài toán thị giác máy tính hiện đại.

---

## 4. So sánh tổng quan

| Thuộc tính | **HOG** | **CNN** |
|-------------|----------|----------|
| Loại | Handcrafted feature | Learned feature |
| Cơ chế | Histogram hướng gradient | Mạng tích chập học filter |
| Dữ liệu đầu vào | Ảnh xám (grayscale) | Ảnh RGB hoặc nhiều kênh |
| Cần huấn luyện? | ❌ Không | ✅ Có |
| Độ phức tạp | Thấp | Cao |
| Hiệu năng | Tốt cho bài toán nhỏ | Vượt trội khi dữ liệu lớn |
| Ứng dụng | Phát hiện người, xe, khuôn mặt | Phân loại, nhận diện, segmentation |

---

## 5. Tóm tắt

> 🔹 **HOG**: Dựa vào **gradient và biên cạnh**, không học từ dữ liệu.  
> 🔹 **CNN**: Dựa vào **mạng tích chập**, tự học đặc trưng tối ưu.  

Cả hai đều là nền tảng của lĩnh vực **Computer Vision**, trong đó **CNN** hiện là chuẩn mực cho hầu hết các hệ thống thị giác nhân tạo hiện đại.
