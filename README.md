# 🦁 Mathematical Formulas for Machine Learning & Deep Learning Project

This document provides a structured collection of **mathematical formulas**, **derivations**, and **explanations** used throughout this ML/DL project.  
It serves as a **reference for model implementation, optimization, and theoretical understanding**.

---

## 🦄 Table of Formulas Sections

- [Basic Data & Model Symbols](#basic-data--model-symbols)
- [Linear Algebra](#linear-algebra)
- [Calculus Foundations](#calculus-foundations)
- [Calculus Derivative and Transformation](#calculus-derivative-and-transformation)
- [Loss Function](#loss-function)
- [Backpropagation](#backpropagation)
- [Optimization](#optimization)
- [Probability Theory](#probability-theory)
- [Statistics](#statistics)
- [Symbols](#symbols)


# Basic Data & Model Symbols

| Symbol | Meaning |
|--------|---------|
| \( x \) | Input feature |
| \( \mathbf{x} \) | Input vector |
| \( X \) | Input matrix (dataset) |
| \( y \) | Ground truth label |
| \( \hat{y} \) | Predicted output |
| \( w \) | Weight |
| \( \mathbf{w} \) | Weight vector |
| \( W \) | Weight matrix |
| \( b \) | Bias |
| \( z \) | Pre-activation value |
| \( a \) | Activation output |
| \( n \) | Number of samples |
| \( d \) | Number of features |
| \( k \) | Class index |
| \( m \) | Number of neurons |

---

# Linear Algebra

## 1. Scalars

\[
x \in \mathbb{R}
\]

---

## 2. Vectors

### 2.1 Vector Definition
$
\mathbf{x} = 
\begin{bmatrix}
x_1 \\
x_2 \\
\vdots \\
x_n
\end{bmatrix}
\in \mathbb{R}^n
$

---

### 2.2 Vector Addition
\[
\mathbf{a} + \mathbf{b} =
\begin{bmatrix}
a_1 + b_1 \\
a_2 + b_2 \\
\vdots \\
a_n + b_n
\end{bmatrix}
\]

---

### 2.3 Scalar Multiplication
\[
c\mathbf{a} =
\begin{bmatrix}
ca_1 \\
ca_2 \\
\vdots \\
ca_n
\end{bmatrix}
\]

---

## 3. Dot Product (Inner Product)

\[
\mathbf{a} \cdot \mathbf{b} = \sum_{i=1}^{n} a_i b_i
\]

---

## 4. Vector Norms

- **L2 Norm**
\[
\|\mathbf{x}\|_2 = \sqrt{\sum_{i=1}^{n} x_i^2}
\]

- **L1 Norm**
\[
\|\mathbf{x}\|_1 = \sum_{i=1}^{n} |x_i|
\]

- **Infinity Norm**
\[
\|\mathbf{x}\|_{\infty} = \max_i |x_i|
\]

---

## 5. Angle Between Two Vectors

\[
\cos(\theta) =
\frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\|\|\mathbf{b}\|}
\]

---

## 6. Matrices

### 6.1 Matrix Definition

\[
\mathbf{A} =
\begin{bmatrix}
a_{11} & a_{12} & \cdots & a_{1n} \\
a_{21} & a_{22} & \cdots & a_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
a_{m1} & a_{m2} & \cdots & a_{mn}
\end{bmatrix}
\in \mathbb{R}^{m \times n}
\]

---

### 6.2 Matrix Addition

\[
\mathbf{A} + \mathbf{B}
\]

---

### 6.3 Scalar-Matrix Multiplication

\[
c\mathbf{A}
\]

---

### 6.4 Matrix Multiplication

\[
\mathbf{C} = \mathbf{A}\mathbf{B}
\]

\[
c_{ij} = \sum_{k} a_{ik}b_{kj}
\]

---

### 6.5 Hadamard (Element-wise) Product

\[
\mathbf{A} \odot \mathbf{B}
\]

---

## 7. Transpose

\[
\mathbf{A}^T
\]

---

## 8. Identity Matrix

\[
\mathbf{I}_n =
\begin{bmatrix}
1 & 0 & \cdots & 0 \\
0 & 1 & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & 1
\end{bmatrix}
\]

---

## 9. Determinant

### 9.1 2×2 Matrix

\[
\det
\begin{bmatrix}
a & b \\
c & d
\end{bmatrix}
= ad - bc
\]

---

## 10. Inverse Matrix

\[
\mathbf{A}^{-1}
\quad \text{such that} \quad
\mathbf{A}\mathbf{A}^{-1} = \mathbf{I}
\]

### 10.1 2×2 Inverse

\[
\mathbf{A}^{-1}=
\frac{1}{ad-bc}
\begin{bmatrix}
d & -b \\
-c & a
\end{bmatrix}
\]

---

## 11. Trace of a Matrix

\[
\text{tr}(\mathbf{A}) = \sum_i a_{ii}
\]

---

## 12. Rank of a Matrix

\[
\text{rank}(\mathbf{A})
\]

---

## 13. System of Linear Equations

\[
\mathbf{A}\mathbf{x} = \mathbf{b}
\]

Solution (if invertible):

\[
\mathbf{x} = \mathbf{A}^{-1}\mathbf{b}
\]

---

## 14. Linear Independence

Vectors \(\mathbf{v}_1, ..., \mathbf{v}_k\) are **linearly independent** if:

\[
c_1 \mathbf{v}_1 + \cdots + c_k \mathbf{v}_k = \mathbf{0}
\Rightarrow
c_1 = \cdots = c_k = 0
\]

---

## 15. Basis & Dimension

- **Basis**: a set of linearly independent vectors that spans the space
- **Dimension**: the number of vectors in a basis

---

## 16. Eigenvalues & Eigenvectors

\[
\mathbf{A}\mathbf{v} = \lambda \mathbf{v}
\]

---

## 17. Characteristic Equation

\[
\det(\mathbf{A} - \lambda \mathbf{I}) = 0
\]

---

## 18. Diagonal Matrix

\[
\mathbf{D} =
\begin{bmatrix}
\lambda_1 & 0 & 0 \\
0 & \lambda_2 & 0 \\
0 & 0 & \lambda_3
\end{bmatrix}
\]

---

## 19. Orthogonality

\[
\mathbf{u}^T \mathbf{v} = 0
\]

---

## 20. Projection of a Vector

\[
\text{proj}_{\mathbf{u}}(\mathbf{v})=
\frac{\mathbf{u}^T \mathbf{v}}{\mathbf{u}^T \mathbf{u}} \mathbf{u}
\]

---

## 21. Singular Value Decomposition (SVD)

\[
\mathbf{A} = \mathbf{U} \boldsymbol{\Sigma} \mathbf{V}^T
\]

---

## 22. Kronecker Product

\[
\mathbf{A} \otimes \mathbf{B}
\]

---

## 23. Tensor (Generalized Matrix)

\[
\mathcal{T} \in \mathbb{R}^{d_1 \times d_2 \times \cdots \times d_n}
\]

---

# Calculus Foundations

## 1. Limits

### 1.1 Limit Definition

\[
\lim_{x \to a} f(x) = L
\]

---

### 1.2 One-Sided Limits

\[
\lim_{x \to a^-} f(x), \quad \lim_{x \to a^+} f(x)
\]

---

### 1.3 Infinite Limits

\[
\lim_{x \to a} f(x) = \infty
\]

---

## 2. Continuity

A function \( f(x) \) is **continuous at \( x=a \)** if:

\[
\lim_{x \to a} f(x) = f(a)
\]

---

## 3. Derivatives

### 3.1 Derivative Definition

\[
f'(x) =
\lim_{h \to 0}
\frac{f(x+h) - f(x)}{h}
\]

---

### 3.2 Higher-Order Derivatives

\[
f''(x), \quad f^{(n)}(x)
\]

---

## 4. Basic Derivative Rules

| Function | Derivative |
|----------|------------|
| \( c \) | \( 0 \) |
| \( x \) | \( 1 \) |
| \( x^n \) | \( nx^{n-1} \) |
| \( e^x \) | \( e^x \) |
| \( \ln x \) | \( \frac{1}{x} \) |

---

## 5. Differentiation Rules

### 5.1 Sum Rule

\[
(f+g)' = f' + g'
\]

---

### 5.2 Product Rule

\[
(fg)' = f'g + fg'
\]

---

### 5.3 Quotient Rule

\[
\left(\frac{f}{g}\right)' =
\frac{f'g - fg'}{g^2}
\]

---

### 5.4 Chain Rule

\[
\frac{dy}{dx} =
\frac{dy}{du} \cdot \frac{du}{dx}
\]

---

## 6. Implicit Differentiation

If:

\[
F(x, y) = 0
\]

Then:

\[
\frac{dy}{dx} =- \frac{F_x}{F_y}
\]

---

## 7. Partial Derivatives

\[
\frac{\partial f}{\partial x}, \quad
\frac{\partial f}{\partial y}
\]

---

## 8. Gradient

\[
\nabla f =
\left(
\frac{\partial f}{\partial x_1},
\frac{\partial f}{\partial x_2},
\cdots,
\frac{\partial f}{\partial x_n}
\right)
\]

---

## 9. Total Differential

\[
df =
\sum_i \frac{\partial f}{\partial x_i} dx_i
\]

---

## 10. Jacobian Matrix

\[
\mathbf{J} =
\begin{bmatrix}
\frac{\partial f_1}{\partial x_1} & \cdots & \frac{\partial f_1}{\partial x_n} \\
\vdots & \ddots & \vdots \\
\frac{\partial f_m}{\partial x_1} & \cdots & \frac{\partial f_m}{\partial x_n}
\end{bmatrix}
\]

---

## 11. Hessian Matrix

\[
\mathbf{H} =
\begin{bmatrix}
\frac{\partial^2 f}{\partial x_1^2} & \cdots & \frac{\partial^2 f}{\partial x_1 \partial x_n} \\
\vdots & \ddots & \vdots \\
\frac{\partial^2 f}{\partial x_n \partial x_1} & \cdots & \frac{\partial^2 f}{\partial x_n^2}
\end{bmatrix}
\]

---

## 12. Indefinite Integrals

\[
\int f(x)\,dx = F(x) + C
\]

---

## 13. Basic Integration Formulas

| Function | Integral |
|----------|----------|
| \( x^n \) | \( \frac{x^{n+1}}{n+1} + C \) |
| \( e^x \) | \( e^x + C \) |
| \( \frac{1}{x} \) | \( \ln |x| + C \) |
| \( \sin x \) | \( -\cos x + C \) |
| \( \cos x \) | \( \sin x + C \) |

---

## 14. Definite Integrals

\[
\int_a^b f(x)\,dx = F(b) - F(a)
\]

---

## 15. Fundamental Theorem of Calculus

\[
\frac{d}{dx} \left( \int_a^x f(t)\,dt \right) = f(x)
\]

---

## 16. Integration by Substitution

\[
\int f(g(x))g'(x)\,dx = \int f(u)\,du
\]

---

## 17. Integration by Parts

\[
\int u\,dv = uv - \int v\,du
\]

---

## 18. Improper Integrals

\[
\int_a^{\infty} f(x)\,dx
\]

\[
\int_{-\infty}^{\infty} f(x)\,dx
\]

---

## 19. Taylor Series

\[
f(x) =
\sum_{n=0}^{\infty}
\frac{f^{(n)}(a)}{n!}(x-a)^n
\]

---

## 20. Maclaurin Series

\[
f(x) =
\sum_{n=0}^{\infty}
\frac{f^{(n)}(0)}{n!}x^n
\]

---

## 21. Multivariable Integrals

### 21.1 Double Integral

\[
\iint_D f(x,y)\,dx\,dy
\]

---

### 21.2 Triple Integral

\[
\iiint_V f(x,y,z)\,dx\,dy\,dz
\]

---

## 22. Line Integral

\[
\int_C f(x,y)\,ds
\]

---

## 23. Surface Integral

\[
\iint_S f(x,y,z)\,dS
\]

---

## 24. Critical Points

\[
\nabla f = 0
\]

---

## 25. Second Derivative Test (Single Variable)

\[
f'(x_c) = 0
\]

\[
f''(x_c) > 0 \Rightarrow \text{Min}
\]

\[
f''(x_c) < 0 \Rightarrow \text{Max}
\]

---

## 26. Optimization with Constraints (Lagrange Multipliers)

\[
\nabla f = \lambda \nabla g
\]

---

# Calculus Derivative and Transformation

## 1. Fundamental Derivative Definition

\[
\frac{dy}{dx} =
\lim_{h \to 0}
\frac{f(x+h) - f(x)}{h}
\]

---

## 2. First-Order Basic Derivative Rules

\[
\frac{d}{dx}(c) = 0
\]

\[
\frac{d}{dx}(x) = 1
\]

\[
\frac{d}{dx}(x^n) = n x^{n-1}
\]

\[
\frac{d}{dx}(e^x) = e^x
\]

\[
\frac{d}{dx}(\ln x) = \frac{1}{x}
\]

---

## 3. Linear Combination Rule

\[
\frac{d}{dx}(af(x) + bg(x)) =
a f'(x) + b g'(x)
\]

---

## 4. Product Rule

\[
\frac{d}{dx}(f(x)g(x)) =
f'(x)g(x) + f(x)g'(x)
\]

---

## 5. Quotient Rule

\[
\frac{d}{dx}\left(\frac{f(x)}{g(x)}\right) =
\frac{f'(x)g(x)-f(x)g'(x)}{g(x)^2}
\]

---

## 6. Chain Rule (Single Variable – Core Backward Law)

If:

\[
y = f(u), \quad u = g(x)
\]

Then:

\[
\frac{dy}{dx} =
\frac{dy}{du} \cdot \frac{du}{dx}
\]

---

## 7. Multi-Layer Chain Rule (Nested Composition)

If:

\[
y = f(g(h(x)))
\]

Then:

\[
\frac{dy}{dx}=
\frac{dy}{df}
\cdot
\frac{df}{dg}
\cdot
\frac{dg}{dh}
\cdot
\frac{dh}{dx}
\]

---

## 8. Implicit Differentiation

Given:

\[
F(x,y) = 0
\]

Then:

\[
\frac{dy}{dx}=- \frac{F_x}{F_y}
\]

---

## 9. Partial Derivatives

\[
\frac{\partial f}{\partial x_i}
\]

---

## 10. Total Derivative (Backward Accumulation Form)

If:

\[
f = f(x_1, x_2, ..., x_n)
\]

Then:

\[
\frac{df}{dt}=
\sum_{i=1}^{n}
\frac{\partial f}{\partial x_i}
\frac{dx_i}{dt}
\]

---

## 11. Gradient Vector

\[
\nabla f =
\begin{bmatrix}
\frac{\partial f}{\partial x_1} \\
\frac{\partial f}{\partial x_2} \\
\vdots \\
\frac{\partial f}{\partial x_n}
\end{bmatrix}
\]

---

## 12. Jacobian Matrix

For:

\[
\mathbf{y} = \mathbf{f}(\mathbf{x})
\]

\[
\mathbf{J} =
\frac{\partial \mathbf{y}}{\partial \mathbf{x}} =
\begin{bmatrix}
\frac{\partial y_1}{\partial x_1} & \cdots & \frac{\partial y_1}{\partial x_n} \\
\vdots & \ddots & \vdots \\
\frac{\partial y_m}{\partial x_1} & \cdots & \frac{\partial y_m}{\partial x_n}
\end{bmatrix}
\]

---

## 13. Jacobian Chain Rule (Vector Backward Law)

If:

\[
\mathbf{y} = \mathbf{f}(\mathbf{u})
\quad , \quad
\mathbf{u} = \mathbf{g}(\mathbf{x})
\]

Then:

\[
\frac{\partial \mathbf{y}}{\partial \mathbf{x}}=
\frac{\partial \mathbf{y}}{\partial \mathbf{u}}
\cdot
\frac{\partial \mathbf{u}}{\partial \mathbf{x}}
\]

---

## 14. Hessian Matrix (Second Order Backward Curvature)

\[
\mathbf{H} =
\frac{\partial^2 f}{\partial \mathbf{x}^2}
\]

---

## 15. Vector-by-Scalar Backward Derivative

If:

\[
y = \mathbf{a}^T \mathbf{x}
\]

Then:

\[
\frac{\partial y}{\partial \mathbf{x}} = \mathbf{a}
\]

---

## 16. Matrix-by-Vector Backward Rule

If:

\[
\mathbf{y} = \mathbf{A} \mathbf{x}
\]

Then:

\[
\frac{\partial \mathbf{y}}{\partial \mathbf{x}} = \mathbf{A}
\]

---

## 17. Quadratic Form Derivative

\[
f = \mathbf{x}^T \mathbf{A} \mathbf{x}
\]

\[
\frac{\partial f}{\partial \mathbf{x}} =
(\mathbf{A} + \mathbf{A}^T)\mathbf{x}
\]

---

## 18. Element-wise Function Derivative

If:

\[
\mathbf{y} = f(\mathbf{x})
\]

Then:

\[
\frac{\partial \mathbf{y}}{\partial \mathbf{x}} =
\text{diag}(f'(x_1), ..., f'(x_n))
\]

---

## 19. Backward Differential Form

\[
dy =
\frac{\partial y}{\partial x} dx
\]

\[
d\mathbf{y} =
\mathbf{J} \, d\mathbf{x}
\]

---

## 20. Backward Gradient Propagation Identity

If:

\[
z = f(u), \quad u = g(v), \quad v = h(w)
\]

Then:

\[
\frac{dz}{dw} =
\frac{dz}{du}
\cdot
\frac{du}{dv}
\cdot
\frac{dv}{dw}
\]

---

## 21. Second Order Backward Propagation

\[
\frac{d^2 y}{dx^2}=
\frac{d}{dx}
\left(
\frac{dy}{dx}
\right)
\]

---

# Probability Theory


## 1. Basic Concepts

### 1.1 Experiment
An experiment is a process that produces an outcome.

### 1.2 Sample Space
\[
\Omega = \{\omega_1, \omega_2, ..., \omega_n\}
\]

### 1.3 Event
An event \( A \subseteq \Omega \)

---

## 2. Axioms of Probability (Kolmogorov Axioms)

Let \( P(A) \) be the probability of event \( A \):

1. Non-negativity:
\[
P(A) \geq 0
\]

2. Normalization:
\[
P(\Omega) = 1
\]

3. Additivity (Disjoint Events):
\[
A \cap B = \emptyset \Rightarrow P(A \cup B) = P(A) + P(B)
\]

---

## 3. Basic Probability Rules

### 3.1 Complement Rule
\[
P(A^c) = 1 - P(A)
\]

### 3.2 Addition Rule (General)
\[
P(A \cup B) = P(A) + P(B) - P(A \cap B)
\]

### 3.3 Conditional Probability
\[
P(A \mid B) = \frac{P(A \cap B)}{P(B)}, \quad P(B) > 0
\]

### 3.4 Independence
\[
A \perp B \iff P(A \cap B) = P(A)P(B)
\]

---

## 4. Random Variables

A random variable is a mapping:
\[
X: \Omega \rightarrow \mathbb{R}
\]

---

## 5. Discrete Random Variables

### 5.1 Probability Mass Function (PMF)
\[
p_X(x) = P(X = x)
\]

### 5.2 Properties
\[
\sum_x p_X(x) = 1
\]

---

## 6. Continuous Random Variables

### 6.1 Probability Density Function (PDF)
\[
f_X(x) \geq 0
\]

\[
\int_{-\infty}^{\infty} f_X(x) dx = 1
\]

### 6.2 Cumulative Distribution Function (CDF)
\[
F_X(x) = P(X \leq x) = \int_{-\infty}^{x} f_X(t) dt
\]

---

## 7. Expectation (Mean Value)

### 7.1 Discrete Case
\[
\mathbb{E}[X] = \sum_x x p_X(x)
\]

### 7.2 Continuous Case
\[
\mathbb{E}[X] = \int_{-\infty}^{\infty} x f_X(x) dx
\]

### 7.3 Linearity of Expectation
\[
\mathbb{E}[aX + bY + c] = a\mathbb{E}[X] + b\mathbb{E}[Y] + c
\]

---

## 8. Variance and Standard Deviation

### 8.1 Variance Definition
\[
\text{Var}(X) = \mathbb{E}[(X - \mu)^2]
\]

where:
\[
\mu = \mathbb{E}[X]
\]

### 8.2 Alternative Form
\[
\text{Var}(X) = \mathbb{E}[X^2] - (\mathbb{E}[X])^2
\]

### 8.3 Standard Deviation
\[
\sigma_X = \sqrt{\text{Var}(X)}
\]

---

## 9. Joint Distributions

### 9.1 Joint PMF (Discrete)
\[
p_{X,Y}(x,y) = P(X = x, Y = y)
\]

### 9.2 Joint PDF (Continuous)
\[
\int \int f_{X,Y}(x,y) dx dy = 1
\]

---

## 10. Marginal Distributions

### 10.1 Discrete
\[
p_X(x) = \sum_y p_{X,Y}(x,y)
\]

### 10.2 Continuous
\[
f_X(x) = \int_{-\infty}^{\infty} f_{X,Y}(x,y) dy
\]

---

## 11. Conditional Distributions

### 11.1 Conditional PMF
\[
p_{X|Y}(x|y) = \frac{p_{X,Y}(x,y)}{p_Y(y)}
\]

### 11.2 Conditional PDF
\[
f_{X|Y}(x|y) = \frac{f_{X,Y}(x,y)}{f_Y(y)}
\]

---

## 12. Covariance

\[
\text{Cov}(X,Y) = \mathbb{E}[(X - \mu_X)(Y - \mu_Y)]
\]

Alternative form:
\[
\text{Cov}(X,Y) = \mathbb{E}[XY] - \mathbb{E}[X]\mathbb{E}[Y]
\]

---

## 13. Correlation Coefficient

\[
\rho_{X,Y} = \frac{\text{Cov}(X,Y)}{\sigma_X \sigma_Y}
\]

---

## 14. Bayes’ Theorem

\[
P(A|B) = \frac{P(B|A)P(A)}{P(B)}
\]

---

## 15. Law of Total Probability

If \( \{B_1, ..., B_n\} \) is a partition of \( \Omega \):

\[
P(A) = \sum_{i=1}^{n} P(A|B_i)P(B_i)
\]

---

## 16. Common Distributions (Basic List)

### Bernoulli:
\[
P(X=1)=p, \quad P(X=0)=1-p
\]

### Binomial:
\[
P(X=k) = \binom{n}{k} p^k (1-p)^{n-k}
\]

### Uniform (Continuous):
\[
f(x) = \frac{1}{b-a}, \quad a \le x \le b
\]

### Normal (Gaussian):
\[
f(x) = \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}
\]

---

## 17. Limit Theorems (Formula Level)

### Law of Large Numbers (Conceptual Form)
\[
\overline{X}_n \rightarrow \mathbb{E}[X]
\]

### Central Limit Theorem (Standard Form)
\[
\frac{\sum_{i=1}^{n} X_i - n\mu}{\sqrt{n}\sigma} \Rightarrow \mathcal{N}(0,1)
\]

---

# Statistics

## 1. Population and Sample

### 1.1 Population
A population is the complete set of all possible observations.

### 1.2 Sample
\[
x_1, x_2, \dots, x_n
\]

---

## 2. Descriptive Statistics

### 2.1 Sample Mean
\[
\bar{x} = \frac{1}{n} \sum_{i=1}^{n} x_i
\]

### 2.2 Sample Variance (Unbiased)
\[
s^2 = \frac{1}{n-1} \sum_{i=1}^{n} (x_i - \bar{x})^2
\]

### 2.3 Sample Standard Deviation
\[
s = \sqrt{s^2}
\]

---

## 3. Expectation and Moments

### 3.1 r-th Raw Moment
\[
\mu_r' = \mathbb{E}[X^r]
\]

### 3.2 r-th Central Moment
\[
\mu_r = \mathbb{E}[(X - \mu)^r]
\]

---

## 4. Bias of an Estimator

Let \( \hat{\theta} \) be an estimator of \( \theta \):

\[
\text{Bias}(\hat{\theta}) = \mathbb{E}[\hat{\theta}] - \theta
\]

---

## 5. Mean Squared Error (MSE)

\[
\text{MSE}(\hat{\theta}) = \mathbb{E}[(\hat{\theta} - \theta)^2]
\]

Decomposition:
\[
\text{MSE} = \text{Var}(\hat{\theta}) + \text{Bias}(\hat{\theta})^2
\]

---

## 6. Maximum Likelihood Estimation (MLE)

### 6.1 Likelihood Function
\[
L(\theta) = \prod_{i=1}^{n} f(x_i \mid \theta)
\]

### 6.2 Log-Likelihood
\[
\ell(\theta) = \sum_{i=1}^{n} \log f(x_i \mid \theta)
\]

---

## 7. Common Estimators

### 7.1 Gaussian Mean Estimator
\[
\hat{\mu} = \frac{1}{n} \sum_{i=1}^{n} x_i
\]

### 7.2 Gaussian Variance Estimator
\[
\hat{\sigma}^2 = \frac{1}{n} \sum_{i=1}^{n} (x_i - \mu)^2
\]

---

## 8. Confidence Interval (CI)

### 8.1 Mean (Known Variance)
\[
\bar{x} \pm z_{\alpha/2} \frac{\sigma}{\sqrt{n}}
\]

### 8.2 Mean (Unknown Variance)
\[
\bar{x} \pm t_{\alpha/2, n-1} \frac{s}{\sqrt{n}}
\]

---

## 9. Hypothesis Testing

### 9.1 Null and Alternative Hypotheses
\[
H_0, \quad H_1
\]

### 9.2 Z-Test Statistic
\[
z = \frac{\bar{x} - \mu_0}{\sigma / \sqrt{n}}
\]

### 9.3 t-Test Statistic
\[
t = \frac{\bar{x} - \mu_0}{s / \sqrt{n}}
\]

---

## 10. Covariance (Sample Form)

\[
s_{XY} = \frac{1}{n-1} \sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})
\]

---

## 11. Correlation Coefficient (Sample)

\[
r_{XY} = \frac{s_{XY}}{s_X s_Y}
\]

---

## 12. Linear Regression (Simple)

### 12.1 Model
\[
y = \beta_0 + \beta_1 x + \varepsilon
\]

### 12.2 Coefficient Estimates
\[
\beta_1 = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sum (x_i - \bar{x})^2}
\]

\[
\beta_0 = \bar{y} - \beta_1 \bar{x}
\]

---

## 13. Residuals and Errors

### 13.1 Residual
\[
e_i = y_i - \hat{y}_i
\]

### 13.2 Sum of Squared Errors (SSE)
\[
\text{SSE} = \sum_{i=1}^{n} e_i^2
\]

---

## 14. R-Squared

\[
R^2 = 1 - \frac{\text{SSE}}{\text{SST}}
\]

\[
\text{SST} = \sum_{i=1}^{n} (y_i - \bar{y})^2
\]

---

## 15. Chi-Square Statistic

\[
\chi^2 = \sum_{i} \frac{(O_i - E_i)^2}{E_i}
\]

---

## 16. Fisher Information

\[
\mathcal{I}(\theta) = \mathbb{E} \left[ \left( \frac{\partial}{\partial \theta} \log f(X \mid \theta) \right)^2 \right]
\]

---

## 17. Cramér–Rao Lower Bound

\[
\text{Var}(\hat{\theta}) \ge \frac{1}{\mathcal{I}(\theta)}
\]

---
# Optimization

## 1. General Optimization Problem

### 1.1 Unconstrained Optimization
\[
\min_{x \in \mathbb{R}^n} f(x)
\]

### 1.2 Constrained Optimization
\[
\min_{x} f(x)
\]
Subject to:
\[
g_i(x) \le 0, \quad i = 1,\dots,m
\]
\[
h_j(x) = 0, \quad j = 1,\dots,p
\]

---

## 2. First-Order Optimality Condition

For unconstrained optimization:
\[
\nabla f(x^*) = 0
\]

---

## 3. Gradient

For a scalar function \( f(x_1, \dots, x_n) \):

\[
\nabla f =
\begin{bmatrix}
\frac{\partial f}{\partial x_1} \\
\frac{\partial f}{\partial x_2} \\
\vdots \\
\frac{\partial f}{\partial x_n}
\end{bmatrix}
\]

---

## 4. Hessian Matrix

\[
H_f(x) =
\begin{bmatrix}
\frac{\partial^2 f}{\partial x_1^2} & \cdots & \frac{\partial^2 f}{\partial x_1 \partial x_n} \\
\vdots & \ddots & \vdots \\
\frac{\partial^2 f}{\partial x_n \partial x_1} & \cdots & \frac{\partial^2 f}{\partial x_n^2}
\end{bmatrix}
\]

---

## 5. Second-Order Optimality Condition

At critical point \( x^* \):

- If \( H_f(x^*) \succ 0 \) → local minimum  
- If \( H_f(x^*) \prec 0 \) → local maximum  
- If indefinite → saddle point  

---

## 6. Convex Set

A set \( C \) is convex if:
\[
\lambda x + (1 - \lambda)y \in C
\]
for all \( x,y \in C \), \( \lambda \in [0,1] \)

---

## 7. Convex Function

A function \( f \) is convex if:
\[
f(\lambda x + (1 - \lambda)y) \le \lambda f(x) + (1 - \lambda) f(y)
\]

---

## 8. First-Order Convexity Condition

\[
f(y) \ge f(x) + \nabla f(x)^T (y - x)
\]

---

## 9. Second-Order Convexity Condition

\[
H_f(x) \succeq 0
\]

---

## 10. Gradient Descent (GD)

### 10.1 Update Rule
\[
x_{k+1} = x_k - \alpha_k \nabla f(x_k)
\]

where:
- \( \alpha_k \): learning rate

---

## 11. Steepest Descent (Norm-Based)

\[
x_{k+1} = x_k - \alpha_k \frac{\nabla f(x_k)}{\|\nabla f(x_k)\|}
\]

---

## 12. Newton’s Method

### 12.1 Update Rule
\[
x_{k+1} = x_k - H_f(x_k)^{-1} \nabla f(x_k)
\]

---

## 13. Taylor Approximation (Optimization Form)

Second-order approximation:
\[
f(x + d) \approx f(x) + \nabla f(x)^T d + \frac{1}{2} d^T H_f(x) d
\]

---

## 14. Constrained Optimization – Lagrangian

\[
\mathcal{L}(x, \lambda) = f(x) + \sum_{i=1}^{m} \lambda_i g_i(x)
\]

---

## 15. Karush–Kuhn–Tucker (KKT) Conditions

For
\[
\min f(x), \quad g_i(x) \le 0
\]

KKT conditions:

1. Primal feasibility:
\[
g_i(x^*) \le 0
\]

2. Dual feasibility:
\[
\lambda_i \ge 0
\]

3. Complementary slackness:
\[
\lambda_i g_i(x^*) = 0
\]

4. Stationarity:
\[
\nabla f(x^*) + \sum_{i=1}^{m} \lambda_i \nabla g_i(x^*) = 0
\]

---

## 16. Equality Constraint – Lagrange Multiplier

For constraint:
\[
g(x) = 0
\]

\[
\nabla f(x^*) = \lambda \nabla g(x^*)
\]

---

## 17. Dual Problem (General Form)

Primal:
\[
\min_x f(x)
\]

Dual:
\[
\max_\lambda \inf_x \mathcal{L}(x,\lambda)
\]

---

## 18. Saddle Point Condition

\[
\mathcal{L}(x^*, \lambda) \le \mathcal{L}(x^*, \lambda^*) \le \mathcal{L}(x, \lambda^*)
\]

---

## 19. Lipschitz Continuous Gradient

\[
\|\nabla f(x) - \nabla f(y)\| \le L \|x - y\|
\]

---

## 20. Strong Convexity

\[
f(y) \ge f(x) + \nabla f(x)^T (y - x) + \frac{\mu}{2} \|y - x\|^2
\]

---

## 21. Projection Operator

\[
\Pi_C(x) = \arg\min_{y \in C} \|x - y\|
\]

---

## 22. Projected Gradient Descent

\[
x_{k+1} = \Pi_C \left(x_k - \alpha_k \nabla f(x_k)\right)
\]

---

## 23. Subgradient (Non-Smooth Optimization)

\[
f(y) \ge f(x) + g^T (y - x)
\]

with \( g \in \partial f(x) \)

---

## 24. Proximal Operator

\[
\text{prox}_{\lambda f}(x) =
\arg\min_y \left( f(y) + \frac{1}{2\lambda} \|y - x\|^2 \right)
\]

---

# Backpropagation

## 1. Computational Graph (General Form)

Let:
\[
y = f(x)
\]
Where \( f \) is a composition of functions:
\[
y = f_n(f_{n-1}(\dots f_1(x)))
\]

---

## 2. Chain Rule (Scalar Form)

If:
\[
y = f(u), \quad u = g(x)
\]
Then:
\[
\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx}
\]

---

## 3. Multivariable Chain Rule

If:
\[
z = f(x_1, x_2, \dots, x_n)
\]
and each:
\[
x_i = x_i(t)
\]
Then:
\[
\frac{dz}{dt} =
\sum_{i=1}^{n} \frac{\partial z}{\partial x_i} \frac{dx_i}{dt}
\]

---

## 4. Vector-to-Scalar Gradient

If:
\[
y = f(\mathbf{x}), \quad \mathbf{x} \in \mathbb{R}^n
\]
Then:
\[
\nabla_{\mathbf{x}} y =
\begin{bmatrix}
\frac{\partial y}{\partial x_1} \\
\vdots \\
\frac{\partial y}{\partial x_n}
\end{bmatrix}
\]

---

## 5. Vector-to-Vector Jacobian

If:
\[
\mathbf{y} = f(\mathbf{x}), \quad \mathbf{y} \in \mathbb{R}^m
\]
Then:
\[
J =
\frac{\partial \mathbf{y}}{\partial \mathbf{x}} =
\begin{bmatrix}
\frac{\partial y_1}{\partial x_1} & \dots & \frac{\partial y_1}{\partial x_n} \\
\vdots & \ddots & \vdots \\
\frac{\partial y_m}{\partial x_1} & \dots & \frac{\partial y_m}{\partial x_n}
\end{bmatrix}
\]

---

## 6. Chain Rule (Matrix Form)

If:
\[
\mathbf{z} = f(\mathbf{y}), \quad \mathbf{y} = g(\mathbf{x})
\]
Then:
\[
\frac{\partial \mathbf{z}}{\partial \mathbf{x}} =
\frac{\partial \mathbf{z}}{\partial \mathbf{y}} \cdot
\frac{\partial \mathbf{y}}{\partial \mathbf{x}}
\]

---

## 7. Scalar Loss with Vector Input

If:
\[
L = L(\mathbf{y}), \quad \mathbf{y} = f(\mathbf{x})
\]
Then:
\[
\nabla_{\mathbf{x}} L =
J_f(\mathbf{x})^T \nabla_{\mathbf{y}} L
\]

---

## 8. Linear Layer (Matrix Form)

Forward:
\[
\mathbf{z} = W\mathbf{x} + \mathbf{b}
\]

---

## 9. Backward Through Linear Layer

Loss gradient:
\[
\frac{\partial L}{\partial \mathbf{z}} = \delta
\]

Then:

### Gradient w.r.t Weights:
\[
\frac{\partial L}{\partial W} =
\delta \mathbf{x}^T
\]

### Gradient w.r.t Bias:
\[
\frac{\partial L}{\partial \mathbf{b}} =
\delta
\]

### Gradient w.r.t Input:
\[
\frac{\partial L}{\partial \mathbf{x}} =
W^T \delta
\]

---

## 10. Element-wise Nonlinearity

\[
\mathbf{y} = \sigma(\mathbf{z})
\]

Backward:
\[
\frac{\partial L}{\partial \mathbf{z}} =
\frac{\partial L}{\partial \mathbf{y}} \odot \sigma'(\mathbf{z})
\]

---

## 11. Common Activation Derivatives

### Sigmoid:
\[
\sigma(x) = \frac{1}{1 + e^{-x}}
\]
\[
\sigma'(x) = \sigma(x)(1 - \sigma(x))
\]

### Tanh:
\[
\frac{d}{dx} \tanh(x) = 1 - \tanh^2(x)
\]

### ReLU:
\[
\text{ReLU}'(x) =
\begin{cases}
1 & x > 0 \\
0 & x \le 0
\end{cases}
\]

---

## 12. Composition of Layers (Full Backward Rule)

Let:
\[
\mathbf{z}^{(k)} = W^{(k)} \mathbf{a}^{(k-1)} + \mathbf{b}^{(k)}
\]
\[
\mathbf{a}^{(k)} = \sigma(\mathbf{z}^{(k)})
\]

Then backward:
\[
\delta^{(k)} =
(W^{(k+1)})^T \delta^{(k+1)} \odot \sigma'(\mathbf{z}^{(k)})
\]

---

## 13. Matrix Differential Identity

\[
dL = \left(\frac{\partial L}{\partial X}\right)^T dX
\]

---

## 14. Product Rule (Matrix Form)

\[
\frac{\partial}{\partial X}(AXB) = A^T \frac{\partial}{\partial Y} B^T
\]

---

## 15. Hadamard vs Matrix Derivative

Hadamard:
\[
A \odot B
\]

Matrix product:
\[
AB
\]

Backward:
\[
\frac{\partial (A \odot B)}{\partial A} = B
\]

---

## 16. Loss Gradient (General)

If:
\[
L = \ell(\hat{y}, y)
\]
Then:
\[
\frac{\partial L}{\partial W} =
\frac{\partial L}{\partial \hat{y}}
\frac{\partial \hat{y}}{\partial z}
\frac{\partial z}{\partial W}
\]

---

## 17. Hessian (Second-Order Backprop)

\[
H = \frac{\partial^2 L}{\partial \mathbf{x}^2}
\]

---

## 18. Vectorized Backprop Form

\[
\frac{\partial L}{\partial \theta} =
\sum_k \left( \frac{\partial L}{\partial z_k} \frac{\partial z_k}{\partial \theta} \right)
\]

---

## 19. Total Derivative Form

\[
\frac{dL}{d\theta} =
\nabla_\theta L =
\sum_i \frac{\partial L}{\partial x_i}
\frac{\partial x_i}{\partial \theta}
\]

---

## ▶︎ Forward Propagation

## 1. General Function Composition

Let:
\[
y = f(x)
\]

If the function is composed of multiple mappings:
\[
y = f_n(f_{n-1}(\dots f_1(x)))
\]

Each layer is a function:
\[
f_k: \mathbb{R}^{n_{k-1}} \rightarrow \mathbb{R}^{n_k}
\]

---

## 2. Vector Input Representation

Input vector:
\[
\mathbf{x} \in \mathbb{R}^{n}
\]

Output vector:
\[
\mathbf{y} \in \mathbb{R}^{m}
\]

---

## 3. Linear Transformation (Affine Map)

For a single layer:
\[
\mathbf{z} = W\mathbf{x} + \mathbf{b}
\]

Where:
- \( W \in \mathbb{R}^{m \times n} \)
- \( \mathbf{x} \in \mathbb{R}^{n} \)
- \( \mathbf{b} \in \mathbb{R}^{m} \)
- \( \mathbf{z} \in \mathbb{R}^{m} \)

---

## 4. Element-wise Nonlinear Mapping

\[
\mathbf{a} = \sigma(\mathbf{z})
\]

Where \( \sigma \) is applied element-wise:
\[
a_i = \sigma(z_i)
\]

---

## 5. Common Activation Functions

### Sigmoid
\[
\sigma(x) = \frac{1}{1 + e^{-x}}
\]

### Tanh
\[
\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
\]

### ReLU
\[
\text{ReLU}(x) = \max(0, x)
\]

### Leaky ReLU
\[
\text{LReLU}(x) = \max(\alpha x, x)
\]

### Softmax (Vector Function)
\[
\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{m} e^{z_j}}
\]

---

## 6. Single Layer Forward Map

\[
f(\mathbf{x}) = \sigma(W\mathbf{x} + \mathbf{b})
\]

---

## 7. Multi-Layer Forward Propagation

Let:
\[
\mathbf{a}^{(0)} = \mathbf{x}
\]

For each layer \( k = 1,2,\dots,L \):

\[
\mathbf{z}^{(k)} = W^{(k)} \mathbf{a}^{(k-1)} + \mathbf{b}^{(k)}
\]

\[
\mathbf{a}^{(k)} = \sigma^{(k)}(\mathbf{z}^{(k)})
\]

---

## 8. Full Network Mapping

The full forward function is:
\[
\hat{\mathbf{y}} = f^{(L)} \circ f^{(L-1)} \circ \dots \circ f^{(1)} (\mathbf{x})
\]

---

## 9. Batch Input (Matrix Form)

Input batch:
\[
X \in \mathbb{R}^{n \times B}
\]

Then:
\[
Z = WX + \mathbf{b}\mathbf{1}^T
\]

\[
A = \sigma(Z)
\]

---

## 10. Element-wise vs Matrix Operations

- Matrix multiplication:
\[
Z = WX
\]

- Element-wise nonlinearity:
\[
A = \sigma(Z)
\]

- Element-wise scaling:
\[
A = Z \odot M
\]

---

## 11. Output Layer Mapping

Let final layer be:
\[
\hat{\mathbf{y}} = g(\mathbf{z}^{(L)})
\]

Where:
- \( g = \sigma \) (general case)
- \( g = \text{identity} \)
- \( g = \text{softmax} \)

---

## 12. Scalar Output Case

If:
\[
\hat{y} \in \mathbb{R}
\]

Then:
\[
\hat{y} = f(\mathbf{x})
\]

---

## 13. Dimensional Consistency Rule

For each layer:
\[
W^{(k)} \in \mathbb{R}^{n_k \times n_{k-1}}, \quad
\mathbf{a}^{(k-1)} \in \mathbb{R}^{n_{k-1}}
\]

Result:
\[
\mathbf{z}^{(k)}, \mathbf{a}^{(k)} \in \mathbb{R}^{n_k}
\]

---

## 14. Norm of Activations

\[
\|\mathbf{a}\|_2 = \sqrt{\sum_i a_i^2}
\]

---

## 15. Vectorized Forward Mapping

Let:
\[
F(\theta, \mathbf{x}) = \mathbf{a}^{(L)}
\]

Where:
\[
\theta = \{W^{(k)}, \mathbf{b}^{(k)}\}_{k=1}^{L}
\]

---

## 16. Composition as Operator

\[
F = \sigma^{(L)} \circ T^{(L)} \circ \dots \circ \sigma^{(1)} \circ T^{(1)}
\]

Where:
\[
T^{(k)}(\mathbf{a}) = W^{(k)}\mathbf{a} + \mathbf{b}^{(k)}
\]

---

## 17. Forward as Recursive System

\[
\mathbf{a}^{(k)} = \sigma^{(k)}(W^{(k)} \mathbf{a}^{(k-1)} + \mathbf{b}^{(k)})
\]

---

## 18. Forward as Explicit Polynomial (Local Taylor Form)

Local approximation:
\[
\sigma(z) \approx \sigma(z_0) + \sigma'(z_0)(z - z_0)
\]

---

## 19. Deterministic Mapping Property

\[
\mathbf{x} \mapsto \hat{\mathbf{y}} \text{ is a deterministic function}
\]

---

# Loss Function

## 1. General Definition of Loss

Let:
- True target: \( y \)
- Prediction: \( \hat{y} \)

Loss function:
\[
L = \ell(\hat{y}, y)
\]

For a dataset:
\[
\mathcal{L} = \frac{1}{N} \sum_{i=1}^{N} \ell(\hat{y}_i, y_i)
\]

---

## 2. Mean Squared Error (MSE)

### 2.1 Definition
\[
\ell(\hat{y}, y) = (\hat{y} - y)^2
\]

\[
\mathcal{L}_{\text{MSE}} = \frac{1}{N} \sum_{i=1}^{N} (\hat{y}_i - y_i)^2
\]

### 2.2 Derivative
\[
\frac{\partial \ell}{\partial \hat{y}} = 2(\hat{y} - y)
\]

---

## 3. Mean Absolute Error (MAE)

### 3.1 Definition
\[
\ell(\hat{y}, y) = |\hat{y} - y|
\]

### 3.2 Subgradient
\[
\frac{\partial \ell}{\partial \hat{y}} =
\begin{cases}
1 & \hat{y} > y \\
-1 & \hat{y} < y \\
\text{undefined} & \hat{y} = y
\end{cases}
\]

---

## 4. Root Mean Squared Error (RMSE)

\[
\mathcal{L}_{\text{RMSE}} =
\sqrt{\frac{1}{N} \sum_{i=1}^{N} (\hat{y}_i - y_i)^2}
\]

---

## 5. Binary Cross Entropy (Log Loss)

### 5.1 Definition
\[
\ell(\hat{y}, y) =- \left( y \log(\hat{y}) + (1 - y)\log(1 - \hat{y}) \right)
\]

### 5.2 Derivative
\[
\frac{\partial \ell}{\partial \hat{y}} =- \frac{y}{\hat{y}} + \frac{1 - y}{1 - \hat{y}}
\]

---

## 6. Categorical Cross Entropy

For one-hot encoded target:

\[
\ell(\hat{\mathbf{y}}, \mathbf{y}) =- \sum_{k=1}^{C} y_k \log(\hat{y}_k)
\]

Derivative:
\[
\frac{\partial \ell}{\partial \hat{y}_k} =- \frac{y_k}{\hat{y}_k}
\]

---

## 7. Softmax + Cross Entropy (Simplified Gradient)

Softmax:
\[
\hat{y}_k = \frac{e^{z_k}}{\sum_j e^{z_j}}
\]

Cross-entropy:
\[
\ell = - \sum_k y_k \log(\hat{y}_k)
\]

Combined gradient:
\[
\frac{\partial \ell}{\partial z_k} = \hat{y}_k - y_k
\]

---

## 8. Hinge Loss (Margin Loss)

\[
\ell(\hat{y}, y) = \max(0, 1 - y\hat{y})
\]

Subgradient:
\[
\frac{\partial \ell}{\partial \hat{y}} =
\begin{cases}- y & y\hat{y} < 1 \\
0 & y\hat{y} \geq 1
\end{cases}
\]

---

## 9. Huber Loss

\[
\ell(\hat{y}, y) =
\begin{cases}
\frac{1}{2}(\hat{y} - y)^2 & |\hat{y} - y| \le \delta \\
\delta(|\hat{y} - y| - \frac{1}{2}\delta) & |\hat{y} - y| > \delta
\end{cases}
\]

Derivative:
\[
\frac{\partial \ell}{\partial \hat{y}} =
\begin{cases}
\hat{y} - y & |\hat{y} - y| \le \delta \\
\delta \cdot \text{sign}(\hat{y} - y) & |\hat{y} - y| > \delta
\end{cases}
\]

---

## 10. Kullback–Leibler (KL) Divergence

\[
D_{KL}(P \| Q) =
\sum_x P(x) \log \frac{P(x)}{Q(x)}
\]

---

## 11. Negative Log Likelihood (NLL)

\[
\ell = - \log P(y \mid x)
\]

---

## 12. Cosine Similarity Loss

Cosine similarity:
\[
\cos(\theta) = \frac{\mathbf{x}^T \mathbf{y}}{\|\mathbf{x}\| \|\mathbf{y}\|}
\]

Loss:
\[
\ell = 1 - \cos(\theta)
\]

---

## 13. Lp-Norm Loss

\[
\ell = \|\hat{\mathbf{y}} - \mathbf{y}\|_p =
\left( \sum_i |\hat{y}_i - y_i|^p \right)^{1/p}
\]

---

## 14. Exponential Loss

\[
\ell(\hat{y}, y) = e^{-y\hat{y}}
\]

Derivative:
\[
\frac{\partial \ell}{\partial \hat{y}} = -y e^{-y\hat{y}}
\]

---

## 15. Margin-Based Loss (General Form)

\[
\ell = \max(0, m - y\hat{y})
\]

---

## 16. Loss as Expectation

\[
\mathcal{L} = \mathbb{E}_{(x,y)\sim \mathcal{D}} [\ell(f(x), y)]
\]

---

## 17. Regularized Loss (General Form)

\[
\mathcal{L}_{\text{reg}} =
\mathcal{L} + \lambda \Omega(\theta)
\]

---

## 18. Gradient of Total Loss

\[
\frac{\partial \mathcal{L}}{\partial \theta}
= \sum_{i=1}^{N}
\frac{\partial \ell_i}{\partial \hat{y}_i}
\frac{\partial \hat{y}_i}{\partial \theta}
\]

---

## 19. Hessian of Loss

\[
H = \frac{\partial^2 \mathcal{L}}{\partial \theta^2}
\]

---

# Symbols

#### LINEAR ALGEBRA SYMBOLS

| Symbol | Meaning |
|--------|---------|
| \( \mathbf{A} \) | Matrix |
| \( \mathbf{A}^T \) | Transpose |
| \( \mathbf{A}^{-1} \) | Inverse matrix |
| \( \det(\mathbf{A}) \) | Determinant |
| \( \mathbf{I} \) | Identity matrix |
| \( \mathbf{0} \) | Zero matrix |
| \( \mathbf{u} \cdot \mathbf{v} \) | Dot product |
| \( \mathbf{u} \odot \mathbf{v} \) | Hadamard (element-wise) product |
| \( \|\mathbf{x}\| \) | Vector norm |
| \( \lambda \) | Eigenvalue |
| \( \mathbf{v} \) | Eigenvector |
| \( \text{rank}(\mathbf{A}) \) | Matrix rank |

---

#### CALCULUS SYMBOLS

| Symbol | Meaning |
|--------|---------|
| \( \frac{d}{dx} \) | Derivative |
| \( \frac{\partial}{\partial x} \) | Partial derivative |
| \( \nabla \) | Gradient operator |
| \( \nabla f \) | Gradient of function |
| \( \nabla^2 \) | Hessian operator |
| \( \int f(x)\,dx \) | Integral |
| \( \Delta x \) | Small change |
| \( \lim_{x \to a} f(x) \) | Limit |
| \( \frac{df}{dz} \) | Local derivative |
| \( J \) | Jacobian |

---

#### FORWARD PROPAGATION SYMBOLS

| Symbol | Meaning |
|--------|---------|
| \( z^{(l)} \) | Pre-activation at layer \( l \) |
| \( a^{(l)} \) | Activation at layer \( l \) |
| \( W^{(l)} \) | Weight matrix at layer \( l \) |
| \( b^{(l)} \) | Bias vector at layer \( l \) |
| \( f(\cdot) \) | Activation function |
| \( \sigma(z) \) | Sigmoid |
| \( \tanh(z) \) | Hyperbolic tangent |
| \( \text{ReLU}(z) \) | Rectified Linear Unit |
| \( \text{softmax}(\mathbf{z}) \) | Softmax function |

---

#### LOSS FUNCTION SYMBOLS

| Symbol | Meaning |
|--------|---------|
| \( L \) | Loss |
| \( \mathcal{L} \) | Cost function |
| \( L(y, \hat{y}) \) | Sample-level loss |
| \( \frac{1}{n} \sum L \) | Mean loss |
| \( \text{MSE} \) | Mean Squared Error |
| \( \text{MAE} \) | Mean Absolute Error |
| \( \text{BCE} \) | Binary Cross Entropy |
| \( H(p, q) \) | Cross Entropy |
| \( \log \) | Logarithm |
| \( \ln \) | Natural logarithm |

---

#### BACKPROPAGATION SYMBOLS

| Symbol | Meaning |
|--------|---------|
| \( \delta^{(l)} \) | Error term at layer \( l \) |
| \( \frac{\partial L}{\partial W} \) | Gradient w.r.t weight |
| \( \frac{\partial L}{\partial b} \) | Gradient w.r.t bias |
| \( \frac{\partial L}{\partial z} \) | Gradient w.r.t pre-activation |
| \( \frac{\partial L}{\partial a} \) | Gradient w.r.t activation |
| \( \alpha \) | Learning rate |
| \( W := W - \alpha \nabla W \) | Gradient descent update |
| \( \odot \) | Element-wise multiplication |
| \( \text{chain rule} \) | Chain rule |

---

#### OPTIMIZATION SYMBOLS

| Symbol | Meaning |
|--------|---------|
| \( \theta \) | Model parameters |
| \( \min_\theta f(\theta) \) | Optimization objective |
| \( \arg\min \) | Argument minimization |
| \( \eta \) | Step size |
| \( g_t \) | Gradient at step \( t \) |
| \( v_t \) | Momentum term |
| \( \beta \) | Momentum coefficient |

---

#### PROBABILITY THEORY SYMBOLS

| Symbol | Meaning |
|--------|---------|
| \( P(A) \) | Probability of event A |
| \( P(A|B) \) | Conditional probability |
| \( X \sim \mathcal{D} \) | Random variable |
| \( \mathbb{E}[X] \) | Expectation |
| \( \text{Var}(X) \) | Variance |
| \( \sigma^2 \) | Variance |
| \( \mu \) | Mean |
| \( \mathcal{N}(\mu, \sigma^2) \) | Normal distribution |
| \( \text{Bern}(p) \) | Bernoulli distribution |
| \( \sum p(x) = 1 \) | PMF normalization |
| \( \int p(x)\,dx = 1 \) | PDF normalization |

---

#### STATISTICS SYMBOLS

| Symbol | Meaning |
|--------|---------|
| \( \bar{x} \) | Sample mean |
| \( s^2 \) | Sample variance |
| \( \hat{\theta} \) | Estimated parameter |
| \( \text{Cov}(X,Y) \) | Covariance |
| \( \rho \) | Correlation |
| \( \text{MLE} \) | Maximum Likelihood Estimation |
| \( \text{MAP} \) | Maximum A Posteriori |

---

#### INDEXING & NOTATION

| Symbol | Meaning |
|--------|---------|
| \( i \) | Sample index |
| \( j \) | Feature index |
| \( l \) | Layer index |
| \( (i,j) \) | Matrix element |
| \( t \) | Time step / iteration |
| \( ^{(l)} \) | Layer superscript |

---


## Notes

- All formulas are written in LaTeX for rendering in GitHub Markdown.
- This document is intended for **research, debugging, and deployment validation**.

---

