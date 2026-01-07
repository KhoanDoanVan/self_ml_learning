# 📈 Cumulative Distribution Function (CDF) trong Machine Learning

## 1. Định nghĩa

**CDF (Cumulative Distribution Function)** -- hàm phân phối tích lũy --
mô tả xác suất rằng một biến ngẫu nhiên ( X ) nhỏ hơn hoặc bằng một giá
trị ( x ):

\[ F(x) = P(X `\leq `{=tex}x) \]

CDF cho biết xác suất tích lũy từ (-∞) đến (x).

------------------------------------------------------------------------

## 2. Đặc điểm

-   Luôn tăng dần từ 0 → 1\
-   ( F(-∞) = 0 ), ( F(∞) = 1 )
-   Với biến liên tục: ( F(x) = `\int`{=tex}\_{-∞}\^{x} f(t) dt ), trong
    đó ( f(t) ) là PDF

------------------------------------------------------------------------

## 3. Ứng dụng trong Machine Learning

-   **Feature scaling:** biến đổi dữ liệu thành xác suất tích lũy
    (quantile transform)
-   **Anomaly detection:** xác định điểm ở "đuôi" phân phối
-   **Evaluation:** ROC/AUC là dạng biểu diễn tích lũy
-   **Modeling:** dùng trong Naive Bayes, GMM, HMM

------------------------------------------------------------------------

## 4. Quan hệ với PDF

-   **PDF (Probability Density Function)**: mật độ xác suất tại một điểm
-   **CDF**: tích phân của PDF\
    \[ F(x) = `\int`{=tex}\_{-∞}\^{x} f(t) dt \]

------------------------------------------------------------------------

## 5. Ví dụ Python

``` python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

x = np.linspace(-4, 4, 200)
pdf = norm.pdf(x, 0, 1)
cdf = norm.cdf(x, 0, 1)

plt.plot(x, pdf, label="PDF (density)")
plt.plot(x, cdf, label="CDF (cumulative)", linestyle="--")
plt.legend()
plt.show()
```
