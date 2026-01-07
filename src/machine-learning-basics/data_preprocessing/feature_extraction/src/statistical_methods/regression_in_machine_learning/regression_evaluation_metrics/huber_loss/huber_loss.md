# Huber Loss trong Machine Learning

## 🔹 Giới thiệu
**Huber loss** (hay **Huber cost function**) là một hàm mất mát lai giữa **Mean Squared Error (MSE)** và **Mean Absolute Error (MAE)**.  
Nó được thiết kế để **giảm ảnh hưởng của outlier (giá trị ngoại lai)** trong bài toán hồi quy.

---

## 🔹 Công thức
Với:
- \( y_i \): giá trị thật
- \( \hat{y}_i \): giá trị dự đoán
- \( \delta \): ngưỡng (hyperparameter)

\[
L_\delta(y_i, \hat{y}_i) =
\begin{cases}
\frac{1}{2}(y_i - \hat{y}_i)^2, & \text{nếu } |y_i - \hat{y}_i| \leq \delta \\
\delta (|y_i - \hat{y}_i| - \frac{1}{2}\delta), & \text{ngược lại}
\end{cases}
\]

---

## 🔹 Ý nghĩa
| Khoảng sai số | Hoạt động như | Mục đích |
|---------------|---------------|-----------|
| \( |error| \leq \delta \) | **MSE** | Trơn, giúp mô hình học chính xác khi lỗi nhỏ |
| \( |error| > \delta \) | **MAE** | Giảm ảnh hưởng của outlier |

---

## 🔹 So sánh

| Tiêu chí | MSE | MAE | Huber |
|-----------|-----|-----|--------|
| Nhạy với outlier | Rất nhạy | Ít nhạy | Trung hòa |
| Gradient | Liên tục | Không liên tục tại 0 | Liên tục |
| Ổn định khi training | Không ổn định nếu có outlier | Ổn định | Ổn định và mượt mà |

---

## 🔹 Ví dụ trong PyTorch

```python
import torch
import torch.nn as nn

loss_fn = nn.HuberLoss(delta=1.0)

y_true = torch.tensor([2.0, 3.0, 4.0])
y_pred = torch.tensor([2.5, 2.0, 4.5])

loss = loss_fn(y_pred, y_true)
print(loss.item())
```

---

## 🔹 Ứng dụng
- Hồi quy khi dữ liệu có outlier.
- Robust regression trong đo lường thực tế.
- Reinforcement Learning (ví dụ: DQN của DeepMind).

---

## 🔹 Đồ thị trực quan
```
Loss
│        /
│       /
│      /
│     /
│    /
│   /
│  /
│ /        ← tuyến tính (MAE-like khi lỗi lớn)
│/____
│    \     ← parabol (MSE-like khi lỗi nhỏ)
│     └────────────── Error
```
