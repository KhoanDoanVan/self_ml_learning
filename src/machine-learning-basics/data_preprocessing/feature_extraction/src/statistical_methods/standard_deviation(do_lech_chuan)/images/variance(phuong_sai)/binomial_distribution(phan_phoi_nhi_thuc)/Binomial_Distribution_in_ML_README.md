
# Binomial Distribution in Machine Learning

## 1. Definition
The **Binomial Distribution** is a discrete probability distribution that models the number of successes in `n` independent trials, each with two outcomes:
- Success with probability `p`
- Failure with probability `1 - p`

Formula:
\(
P(X = k) = C(n, k) * p^k * (1 - p)^{n - k}
\)

Where:
- X: number of successes
- n: number of trials
- k: desired number of successes
- p: probability of success
- C(n, k): combinations of choosing k successes from n trials

Example:
> Flipping a fair coin 10 times → X ~ Binomial(n=10, p=0.5)
> P(X=6) = C(10,6)*0.5^6*0.5^4 = 0.205

---

## 2. Relationship to Machine Learning

| Context | Application |
|----------|-------------|
| Binary Classification | Modeling probability of label=1 in logistic regression |
| Bernoulli trials | Binomial = sum of multiple Bernoulli events (click/no-click) |
| MLE Estimation | Logistic Regression log-likelihood based on Binomial |
| Bayesian Inference | Beta prior + Binomial likelihood |
| A/B Testing | Comparing success rates between two groups |
| Feature modeling | Used in Naive Bayes for binary features |

---

## 3. Real-world Use Cases

### 3.1 A/B Testing
Each user either clicks or not → Binomial.
Used to compare conversion rates between two variants.

### 3.2 Logistic Regression
Assumes the probability of y=1 follows a Binomial distribution:
`y_i ~ Binomial(n=1, p_i)`

### 3.3 Spam Detection / Text Classification
Naive Bayes may model each word’s presence/absence via Binomial distribution.

### 3.4 Model Evaluation
Accuracy measured over multiple test batches can be modeled as Binomial.

### 3.5 Medical Diagnosis
Testing disease-positive rates across patients modeled via Binomial.

---

## 4. Simulation Example (Python)

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom

n, p = 10, 0.5
x = np.arange(0, n+1)
pmf = binom.pmf(x, n, p)

plt.bar(x, pmf)
plt.title('Binomial Distribution (n=10, p=0.5)')
plt.xlabel('Number of Successes (k)')
plt.ylabel('Probability')
plt.show()
```

---

## 5. Summary

| Property | Description |
|-----------|-------------|
| Type | Discrete |
| Parameters | n (number of trials), p (probability of success) |
| Mean | E[X] = np |
| Variance | Var[X] = np(1 - p) |
| Applications in ML | Logistic Regression, A/B Testing, Naive Bayes, Model Evaluation |
