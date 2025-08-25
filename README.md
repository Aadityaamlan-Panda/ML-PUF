# ⚡ PUF Linear Modeling

Elegant, minimal baseline for modeling and decoding XOR‑arbiter PUFs with handcrafted feature maps and linear classifiers (scikit‑learn only). Train a single linear model (weights + bias), map challenges to rich polynomial features, and optionally decode a provided 65‑D linear model into per‑stage delay vectors.

- 🔧 Single entrypoints: my_fit, my_map, my_decode
- 🧠 Linear models only (e.g., LinearSVC, RidgeClassifier)
- 🧩 Custom, high‑variance feature expansions for 8‑bit challenges
- 🚫 No deep learning; no SciPy beyond khatri_rao (unused here)

***

## 🚀 Quick Start

1) Install requirements
- Python 3.8+
- numpy
- scikit-learn
- scipy (only for scipy.linalg.khatri_rao if needed)

2) Train a model
```python
from submit import my_fit, my_map
w, b = my_fit(X_train, y_train)  # X_train: (N, 8), y_train: {0,1}
```

3) Map challenges to features
```python
from submit import my_map
Phi = my_map(X_test)  # returns engineered features
y_pred = (Phi @ w + b > 0).astype(int)
```

4) Decode weights (optional)
```python
from submit import my_decode
p, q, r, s = my_decode(w_full_65d)  # each is (64,)
```

***

## 🧪 Labels and Conventions

- Input challenges X: shape (N, 8), bits in {0,1}
- Labels y: in {0,1}; internally mapped to {-1,+1} for linear SVM
- Returned model: w (vector), b (scalar bias)

***

## 🧰 What’s Inside

- my_map:
  - Bit flip: F = 1 - 2X in {-1,+1}
  - First-order terms: [1, F]
  - Second-order interactions: F_i * F_j for i < j
  - Chain products: ∏ F[i:] to encode cumulative stage effects
  - Outer products of chains: interaction of cumulative prefixes
  - Variance filter: remove near-constant features

- my_fit:
  - Builds features via my_map
  - Trains a linear SVM (LinearSVC) on y ∈ {-1, +1}
  - Returns the learned (w, b)

- my_decode:
  - Attempts to recover four 64‑D delay-like vectors from a 65‑D linear model
  - Produces nonnegative vectors by offsetting with a common shift

***

## 📏 Constraints

- Use only NumPy and scikit‑learn linear models (e.g., LinearSVC, RidgeClassifier, LinearRegression)
- No deep learning frameworks
- No SciPy routines besides khatri_rao (not required by default)
- Submit a single Python file named submit.py inside a ZIP

***

## 🧠 Tips for Better Accuracy

- Tune C and max_iter for LinearSVC (e.g., C in [0.1, 1, 10], max_iter up to 5000)
- Consider feature scaling if swapping to other linear models
- Keep the variance threshold conservative to avoid over-pruning
- Try RidgeClassifier if SVM convergence is slow

***

## ⚠️ Known Caveats

- The provided my_decode is a heuristic and may need refinement per PUF design
- Feature growth can be large; variance filtering mitigates overfitting and runtime
- LinearSVC doesn’t output probabilities; use decision_function if needed

***

## 👤 Credits

- Author: Aaditya Amlan Panda
- Baseline: Handcrafted polynomial/chain features + linear SVM
- License: MIT
