# ⚡ PUF Linear Modeling

Elegant, minimal baseline for modeling and decoding XOR‑arbiter PUFs with handcrafted feature maps and linear classifiers (scikit‑learn only). Train a single linear model (weights + bias), map challenges to rich polynomial features, and optionally decode a provided 65‑D linear model into per‑stage delay vectors.

- 🔧 Single entrypoints: `my_fit`, `my_map`, `my_decode`
- 🧠 Linear models only (e.g., `LinearSVC`, `RidgeClassifier`)
- 🧩 Custom, high‑variance feature expansions for 8‑bit challenges
- 🚫 No deep learning; no SciPy beyond `khatri_rao` (unused here)

---

## 🚀 Quick Start

**Requirements:** Python 3.8+, numpy, scikit-learn, scipy (only for `scipy.linalg.khatri_rao` if needed)

```python
# Train a model
from submit import my_fit, my_map
w, b = my_fit(X_train, y_train)  # X_train: (N, 8), y_train: {0,1}

# Map challenges to features and predict
Phi = my_map(X_test)
y_pred = (Phi @ w + b > 0).astype(int)

# Decode weights (optional)
from submit import my_decode
p, q, r, s = my_decode(w_full_65d)  # each is (64,)
```

---

## 🧪 Labels and Conventions

- Input challenges `X`: shape `(N, 8)`, bits in `{0, 1}`
- Labels `y`: in `{0, 1}`; internally mapped to `{-1, +1}` for linear SVM
- Returned model: `w` (vector), `b` (scalar bias)

---

## 🧰 Feature Map: Mathematical Foundation

The feature map is derived from the dual-PUF arbiter delay analysis. For any stage *i*, the upper and lower delay differences evolve as:

```
Dᵢ = Dᵢ₋₁ + kₚ + cᵢ(kₛ − kₚ − Dᵢ₋₁ + dᵢ₋₁)
dᵢ = dᵢ₋₁ + k_q + cᵢ(Dᵢ₋₁ − dᵢ₋₁ + k_r − k_q)
```

Defining `fᵢ = 1 − 2cᵢ ∈ {−1, +1}` and the sum/difference signals `Aᵢ = Dᵢ + dᵢ`, `Bᵢ = Dᵢ − dᵢ`, the feature map structure becomes:

- **A₇** (linear in challenge bits): feature map = `{f₇, f₆, …, f₀, 1}` — 9 terms
- **B₇** (product chains): feature map = `{f₇f₆…f₀, f₇f₆…f₁, …, f₇f₆, f₇, 1}` — 9 terms

Squaring both gives 45 terms each. Since 4 terms overlap (`1, f₇², f₇f₆, f₇`), the combined feature map for the output `(A₇² − B₇²) / 2` has **85 unique terms** (plus 1 bias term = **D = 86 total dimensions**).

### Kernel Interpretation

The squared terms admit a kernel view:

- `A²` corresponds to a **degree-2 polynomial kernel** `K₁(a, c) = (aᵀc + const)²`
- `B²` corresponds to a **product kernel** `K₂(−2·1, Cᵢ) = ∏(1 − 2cᵢ)` over suffix challenge bits

### `my_map` Implementation

- **Bit flip:** `F = 1 − 2X ∈ {−1, +1}`
- **First-order terms:** `[1, F]`
- **Second-order interactions:** `Fᵢ · Fⱼ` for `i < j`
- **Chain products:** `∏ F[i:]` to encode cumulative stage effects
- **Outer products of chains:** interaction of cumulative prefixes
- **Variance filter:** remove near-constant features

---

## 🔓 Decoding: Recovering Delay Vectors (`my_decode`)

Given a learned 65-D weight vector `w`, the delay parameters are recovered via:

```
αᵢ = (pᵢ − qᵢ + rᵢ − sᵢ) / 2
βᵢ = (pᵢ − qᵢ − rᵢ + sᵢ) / 2

w₀ = α₀
wᵢ = αᵢ + βᵢ₋₁   (for i ≥ 1)
```

This yields the linear system `w = Ax`, where `A` is a sparse matrix with entries in `{±½}`. Since the system is underdetermined (more unknowns than equations), infinitely many solutions exist. The implemented heuristic fixes `p₀ = q₀ = s₀ = 0`, sets `r₀ = 2w₀`, and propagates forward, resolving the boundary conflict at stage 63 via:

```
p₆₃ = w₆₃ + w₆₄ − p₆₂/2   (when p₆₂ ≥ 0)
r₆₃ = w₆₃ − w₆₄ − p₆₂/2
```

Non-negativity is enforced by shifting all vectors by the absolute value of the global minimum.

---

## 📊 Experimental Results

### LinearSVC Hyperparameter Study

| C | tol | Loss | Penalty | Train Time (s) | Accuracy |
|---|-----|------|---------|----------------|----------|
| 1 | 1e-4 | squared hinge | l2 | 2.340 | **1.0000** |
| 1 | 1e-2 | squared hinge | l2 | 0.243 | **1.0000** |
| 1 | 1e-6 | squared hinge | l2 | 1.076 | **1.0000** |
| 1 | 1e-4 | squared hinge | l1 | 5.985 | **1.0000** |
| 1 | 1e-4 | hinge | l2 | 0.471 | 0.9963 |
| 100 | 1e-4 | squared hinge | l2 | 3.651 | 0.9963 |
| 0.01 | 1e-4 | squared hinge | l2 | 0.150 | 0.9569 |

### RidgeClassifier Hyperparameter Study

| C | tol | Penalty | Train Time (s) | Accuracy |
|---|-----|---------|----------------|----------|
| 1 | 1e-4 | l1 | 5.280 | **1.0000** |
| 100 | 1e-4 | l2 | 0.866 | **1.0000** |
| 1 | 1e-2 | l2 | 0.141 | 0.9907 |
| 1 | 1e-4 | l2 | 0.838 | 0.9907 |
| 1 | 1e-6 | l2 | 0.376 | 0.9907 |
| 0.01 | 1e-4 | l2 | 0.112 | 0.8900 |

**Key takeaways:**
- `LinearSVC` with `C=1`, `tol=1e-2`, `squared hinge`, `l2` achieves perfect accuracy in just **0.243s** — the best speed/accuracy trade-off overall.
- `l1` penalty achieves perfect accuracy on both classifiers but at a significant runtime cost (~6s).
- Very small `C` (0.01) noticeably hurts accuracy on both classifiers (~0.89–0.96).
- `RidgeClassifier` with `l2` penalty plateaus at 0.9907 across a wide range of tolerances, suggesting the model is limited by regularization strength rather than convergence.

---

## 📏 Constraints

- Use only NumPy and scikit‑learn linear models (`LinearSVC`, `RidgeClassifier`, `LinearRegression`)
- No deep learning frameworks
- No SciPy routines besides `khatri_rao` (not required by default)
- Submit a single Python file named `submit.py` inside a ZIP

---

## 🧠 Tips for Better Accuracy

- Best fast config: `LinearSVC(C=1, tol=1e-2, loss='squared_hinge', penalty='l2')` — perfect accuracy in 0.24s
- For maximum accuracy with more time budget: use `l1` penalty or increase `max_iter` up to 5000
- Keep variance threshold conservative to avoid over-pruning features
- Avoid `C=0.01` — accuracy drops significantly on this feature space
- `RidgeClassifier` with `C=100` is a strong alternative when SVM convergence is slow

---

## ⚠️ Known Caveats

- `my_decode` is a heuristic and may need refinement per PUF design; the underdetermined system means recovered delay vectors are one valid solution, not the unique ground truth
- Feature growth can be large (85 dimensions before filtering); variance filtering mitigates overfitting and runtime
- `LinearSVC` doesn't output probabilities; use `decision_function` if confidence scores are needed

---

## 👤 Credits

- **Author:** Aadityaamlan Panda
- **Team:** Amit Kumar, Harish Choudhary, Jagdeesh Meena, Shivani Gupta, Sumit Neniwal
- **Course:** CS771 Project
- **Baseline:** Handcrafted polynomial/chain features + linear SVM
- **License:** MIT
