# 🎲 Probability Mass Function (PMF) trong Machine Learning

## 1. Định nghĩa

**PMF (Probability Mass Function)** là hàm xác suất của **biến ngẫu
nhiên rời rạc**, biểu diễn xác suất chính xác mà ( X ) nhận giá trị (
x_i ):

\[ P(X = x_i) = p(x_i) \]

------------------------------------------------------------------------

## 2. Tính chất

-   ( p(x_i) ≥ 0 )
-   Tổng tất cả xác suất bằng 1: ( `\sum`{=tex}\_i p(x_i) = 1 )

------------------------------------------------------------------------

## 3. Ví dụ xúc xắc

Giả sử tung 1 con xúc xắc 6 mặt:

  x   PMF (p(x))   CDF (F(x))
  --- ------------ ------------
  1   0.1667       0.1667
  2   0.1667       0.3333
  3   0.1667       0.5000
  4   0.1667       0.6667
  5   0.1667       0.8333
  6   0.1667       1.0000

------------------------------------------------------------------------

## 4. Ứng dụng trong Machine Learning

-   **Classification:** Softmax output chính là PMF của các class\
    \[ P(y=k\|x) = `\frac{e^{z_k}}{\sum_j e^{z_j}}`{=tex} \]
-   **Loss Function:** Cross-Entropy đo khoảng cách giữa PMF thật và dự
    đoán
-   **HMM / Markov Models:** xác suất rời rạc mô tả trạng thái
-   **Sampling:** dùng PMF để chọn giá trị rời rạc

------------------------------------------------------------------------

## 5. So sánh PMF -- PDF -- CDF

  ---------------------------------------------------------------------------
  Loại      Áp dụng cho       Trả về     Tổng/Tích phân = 1        Ví dụ
  --------- ----------------- ---------- ------------------------- ----------
  **PMF**   Biến rời rạc      Xác suất   Tổng = 1                  Tung xúc
                              tại mỗi                              xắc
                              giá trị                              

  **PDF**   Biến liên tục     Mật độ xác Tích phân = 1             Chiều cao,
                              suất                                 cân nặng

  **CDF**   Cả hai            Xác suất   F(∞)=1                    Tích lũy
                              tích lũy                             của
                              đến x                                PMF/PDF
  ---------------------------------------------------------------------------

------------------------------------------------------------------------

## 6. Ví dụ Python

``` python
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(1, 7)
pmf = np.ones(6) / 6
cdf = np.cumsum(pmf)

plt.bar(x, pmf, label="PMF (Probability of each face)")
plt.plot(x, cdf, marker='o', color='r', label="CDF (Cumulative)")
plt.legend()
plt.show()
```
