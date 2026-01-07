
# Variance (Phương sai) trong Machine Learning

## 1. Định nghĩa
Phương sai (variance) là mức độ mà dự đoán của mô hình thay đổi khi ta thay đổi dữ liệu huấn luyện. 
Variance cao khiến mô hình nhạy với nhiễu, dẫn đến overfitting.  
Công thức decomposition (cho squared loss):
```
E[(f̂(x) - y)²] = Bias² + Variance + Noise
```

## 2. Tầm quan trọng
Variance cao => mô hình thiếu ổn định, generalize kém.  
Variance thấp => mô hình ổn định nhưng có thể bias cao.  
Hiểu variance giúp ta cân bằng bias–variance tradeoff.

## 3. Nguyên nhân
- Mô hình quá phức tạp (deep trees, polynomial cao, large NN)
- Ít dữ liệu hoặc dữ liệu nhiễu
- Quá nhiều feature, collinearity
- Quá nhiều randomness trong training

## 4. Cách đo variance
- **Train/Test gap:** chênh lệch lớn giữa train và validation.
- **K-Fold CV:** kiểm tra độ lệch (std) giữa các folds.
- **Bootstrap:** huấn luyện nhiều lần trên các sample khác nhau.
- **Learning curves:** nếu validation error giảm khi tăng data => variance cao.

## 5. Cách giảm variance
- Tăng dữ liệu huấn luyện
- Regularization (L1/L2, dropout, early stopping)
- Bagging / Random Forest / Ensemble Averaging
- Pruning (tree), giảm độ phức tạp mô hình
- Feature selection, PCA, data augmentation

## 6. Ví dụ minh họa
**Polynomial Regression:**
- degree thấp → bias cao, variance thấp
- degree cao → bias thấp, variance cao

**Decision Tree vs Random Forest:**
- Decision Tree: variance cao
- Random Forest: bagging giảm variance

## 7. Ứng dụng thực tế
| Domain | Vấn đề variance | Giải pháp |
|---------|------------------|-----------|
| Tài chính | Dự đoán giá dao động mạnh | Ensemble, regularization |
| Y tế | Model không ổn định giữa các bệnh viện | Cross-hospital CV, ensemble |
| Recommendation | Kết quả gợi ý dao động | Averaging, smoothing |
| Vision | Ảnh khác điều kiện ánh sáng | Augmentation, dropout |
| Monitoring | Nhiều false alarms | Regularization, thresholding |

## 8. Uncertainty estimation liên quan
- **Frequentist:** Bootstrap CI
- **Bayesian:** Posterior predictive variance (BNN, GP)
- **Deep Ensembles / MC-Dropout:** đo spread dự đoán
- **Conformal prediction:** prediction set có độ tin cậy

## 9. Checklist xử lý variance khi deploy
- Kiểm tra train vs val gap
- Chạy k-fold CV, lưu mean ± std
- Thử giảm complexity, thêm regularization
- Tăng hoặc augment dữ liệu
- Ghi log random seeds và kết quả

## 10. Code ví dụ (Bootstrap Variance)
```python
from sklearn.utils import resample
from sklearn.metrics import mean_squared_error
import numpy as np

models_preds = []
for i in range(50):
    Xb, yb = resample(X_train, y_train)
    model = clone(base_model).fit(Xb, yb)
    preds = model.predict(X_val)
    models_preds.append(preds)

preds_arr = np.stack(models_preds, axis=0)
variance = preds_arr.var(axis=0).mean()
print("Mean prediction variance:", variance)
```

## 11. Kết luận
- Variance cao = overfitting, mô hình không ổn định.
- Đo variance bằng CV, bootstrap, ensemble spread.
- Giảm variance bằng regularization, bagging, data augmentation.
- Kiểm soát variance là bước cốt lõi để deploy mô hình ML an toàn.

