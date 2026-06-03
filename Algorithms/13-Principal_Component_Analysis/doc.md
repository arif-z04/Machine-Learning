
# Principal Component Analysis (PCA) — Complete Documentation

---

## Table of Contents

- [Principal Component Analysis (PCA) — Complete Documentation](#principal-component-analysis-pca--complete-documentation)
  - [Table of Contents](#table-of-contents)
  - [1. Introduction](#1-introduction)
    - [Key Characteristics](#key-characteristics)
    - [When to Use PCA](#when-to-use-pca)
    - [When NOT to Use PCA](#when-not-to-use-pca)
  - [2. Core Idea](#2-core-idea)
    - [2.1 Intuition (Visual Explanation)](#21-intuition-visual-explanation)
    - [2.2 Step-by-Step Conceptual Process](#22-step-by-step-conceptual-process)
    - [2.3 Key Properties](#23-key-properties)
  - [3. Applications (Real Life)](#3-applications-real-life)
    - [Concrete Example Snippets](#concrete-example-snippets)
  - [4. Mathematical View](#4-mathematical-view)
    - [4.1 Problem Statement](#41-problem-statement)
    - [4.2 Data Preprocessing](#42-data-preprocessing)
      - [4.2.1 Mean Centering (Required)](#421-mean-centering-required)
      - [4.2.2 Feature Scaling (Critical when features have different units/scales)](#422-feature-scaling-critical-when-features-have-different-unitsscales)
    - [4.3 Covariance Matrix](#43-covariance-matrix)
    - [4.4 Eigenvalue Decomposition](#44-eigenvalue-decomposition)
    - [4.5 Singular Value Decomposition (SVD) Perspective](#45-singular-value-decomposition-svd-perspective)
    - [4.6 Choosing the Number of Components](#46-choosing-the-number-of-components)
    - [4.7 Reconstruction](#47-reconstruction)
  - [5. Implementation using Python Libraries](#5-implementation-using-python-libraries)
    - [5.1 Required Libraries](#51-required-libraries)
    - [5.2 Standard PCA Workflow](#52-standard-pca-workflow)
    - [5.3 Implementation with Scikit-learn](#53-implementation-with-scikit-learn)
      - [Basic Usage](#basic-usage)
      - [Choosing `n_components` by Variance Threshold](#choosing-n_components-by-variance-threshold)
      - [Accessing Loadings](#accessing-loadings)
    - [5.4 Implementation with NumPy (From Scratch)](#54-implementation-with-numpy-from-scratch)
      - [Using Eigen Decomposition](#using-eigen-decomposition)
      - [Using SVD (More Numerically Stable)](#using-svd-more-numerically-stable)
    - [5.5 Useful Helper Functions and Visualizations](#55-useful-helper-functions-and-visualizations)
      - [Scree Plot](#scree-plot)
      - [2D Projection Plot](#2d-projection-plot)
      - [Biplot (Features + Samples)](#biplot-features--samples)
    - [5.6 Common Pitfalls and Best Practices](#56-common-pitfalls-and-best-practices)
      - [Pitfalls](#pitfalls)
      - [Best Practices](#best-practices)
  - [6. Example Implementations](#6-example-implementations)
    - [Example 1 — Iris Dataset](#example-1--iris-dataset)
      - [Step 1: Load and Explore](#step-1-load-and-explore)
      - [Step 2: Explore Correlation](#step-2-explore-correlation)
      - [Step 3: Standardize](#step-3-standardize)
      - [Step 4: Apply PCA (All Components First)](#step-4-apply-pca-all-components-first)
      - [Step 5: Analyze Explained Variance](#step-5-analyze-explained-variance)
      - [Step 6: 2D Projection](#step-6-2d-projection)
      - [Step 7: Interpret Loadings](#step-7-interpret-loadings)
      - [Step 8: Reconstruction Check](#step-8-reconstruction-check)
    - [Example 2 — Wine Dataset](#example-2--wine-dataset)
      - [Step 1: Load Data](#step-1-load-data)
      - [Step 2: Train/Test Split and Standardize](#step-2-traintest-split-and-standardize)
      - [Step 3: Baseline (No PCA)](#step-3-baseline-no-pca)
      - [Step 4: Find Optimal Number of Components](#step-4-find-optimal-number-of-components)
      - [Step 5: PCA + Classification with Different k](#step-5-pca--classification-with-different-k)
      - [Step 6: Visualize 2D and 3D Projections](#step-6-visualize-2d-and-3d-projections)
      - [Step 7: Interpret Loadings](#step-7-interpret-loadings-1)
    - [Example 3 — Digits Dataset](#example-3--digits-dataset)
      - [Step 1: Load and Explore](#step-1-load-and-explore-1)
      - [Step 2: Visualize Original Images](#step-2-visualize-original-images)
      - [Step 3: Standardize](#step-3-standardize-1)
      - [Step 4: Scree Plot — How Many Components?](#step-4-scree-plot--how-many-components)
      - [Step 5: 2D and 3D Visualization](#step-5-2d-and-3d-visualization)
      - [Step 6: Image Reconstruction (Compression)](#step-6-image-reconstruction-compression)
      - [Step 7: Visualize Principal Components as Images](#step-7-visualize-principal-components-as-images)
      - [Step 8: Compression Ratio Summary](#step-8-compression-ratio-summary)
  - [7. Conclusion and Further Reading](#7-conclusion-and-further-reading)
    - [Key Takeaways](#key-takeaways)
    - [When to Use PCA](#when-to-use-pca-1)
    - [When NOT to Use PCA](#when-not-to-use-pca-1)
    - [Further Reading and Extensions](#further-reading-and-extensions)
    - [References](#references)

---

## 1. Introduction

**Definition:**
Principal Component Analysis (PCA) is an **unsupervised linear dimensionality reduction** technique
that transforms a dataset with potentially correlated features into a new set of **uncorrelated variables**
called *principal components*. These components are ordered such that the first few retain most of the
variation (variance) present in the original dataset.

### Key Characteristics

| Property              | Description                                                                 |
|-----------------------|-----------------------------------------------------------------------------|
| **Type**              | Unsupervised learning (no target labels required for the transformation)    |
| **Goal**              | Reduce dimensionality while preserving maximum variance                     |
| **Output**            | Orthogonal (uncorrelated) principal components                              |
| **Components**        | Linear combinations of original features                                    |
| **Ordering**          | PC1 explains most variance, PC2 second most, etc.                          |
| **Assumptions**       | Linearity, continuous features, large variance ≈ important structure, orthogonal PCs |
| **Preprocessing**     | Typically requires feature scaling (standardization)                        |
| **Limitations**       | Linear method, sensitive to scaling, variance ≠ always importance, hard-to-interpret components |

### When to Use PCA

- High-dimensional data with correlated features
- Need for data visualization (reduce to 2 or 3 dimensions)
- Preprocessing step to reduce noise, multicollinearity, or computational cost
- Feature extraction / compression
- Exploratory data analysis

### When NOT to Use PCA

- Non-linear relationships dominate the data (consider t-SNE, UMAP, Kernel PCA)
- Interpretability of original features in terms of business meaning is paramount
- Categorical features dominate (PCA is designed for continuous numerical data)
- Preservation of distances between all points is critical in ways not captured by variance

---

## 2. Core Idea

At its core, PCA seeks to find a **lower-dimensional subspace** such that when the data
is projected onto it, the **projection error (reconstruction error) is minimized**, which
is equivalent to **maximizing the variance** of the projected data.

### 2.1 Intuition (Visual Explanation)

Imagine a cloud of 2D points that form an elongated ellipse:

- The direction along the **longest axis** of the ellipse captures the most spread
  (variance) of the points → **First Principal Component (PC1)**
- The direction **perpendicular** to it (narrower axis) captures the next most
  variance → **PC2**
- By projecting all points onto the line defined by PC1, we reduce the data
  from 2D to 1D while losing the **least possible variance**

> This generalizes to higher dimensions: we find orthogonal directions
> in order of decreasing variance.

### 2.2 Step-by-Step Conceptual Process

1. **Center the data** — Subtract the mean of each feature so that data is centered at the origin
2. **Find directions of maximum variance** — Identify the unit vector `w₁` such that projecting data onto it gives maximum variance
3. **Find next direction** — Find `w₂` orthogonal to `w₁` with the next highest variance
4. **Repeat** — Continue for all dimensions, obtaining an orthonormal basis
5. **Project** — Transform original data points into coordinates along these new axes (principal components)
6. **Reduce** — Keep only the top `k` components that explain a sufficient amount of variance

### 2.3 Key Properties

- **Orthogonality** — Principal components are mutually orthogonal (uncorrelated)
- **Variance Ordering** — `Var(PC1) ≥ Var(PC2) ≥ ... ≥ Var(PCn)`
- **Total Variance Preservation** — Sum of variances of all PCs equals sum of variances of original features
- **Linear Transformation** — Each PC is a linear combination:

```
PC_i = w_{i1}·x₁ + w_{i2}·x₂ + ... + w_{in}·xₙ
```

> The weight vectors `w_i` are called **loadings** (or eigenvectors) and tell us
> how much each original feature contributes to that component.

---

## 3. Applications (Real Life)

PCA is widely used across industries due to its simplicity and effectiveness.

| Domain                     | Application                   | Purpose                                                 |
|----------------------------|-------------------------------|---------------------------------------------------------|
| **Computer Vision**        | Face Recognition (Eigenfaces) | Reduce image dimensions, extract facial features        |
|                            | Image Compression             | Reduce storage by keeping top components                |
|                            | Object Recognition            | Feature reduction for classifiers                       |
| **Finance & Economics**    | Portfolio Management          | Identify principal factors driving asset returns        |
|                            | Risk Analysis                 | Find common risk factors                                |
|                            | Credit Scoring                | Reduce applicant features, handle multicollinearity     |
| **Bioinformatics**         | Gene Expression Analysis      | Cluster samples, visualize high-dimensional gene data   |
|                            | Population Genetics           | Study population structure via SNP data                 |
| **Marketing**              | Customer Segmentation         | Reduce survey attributes, find underlying traits        |
|                            | Market Research               | Analyze product preferences                             |
|                            | Recommendation Systems        | Preprocess user-item data                               |
| **NLP**                    | Latent Semantic Analysis      | Reduce term-document matrix                             |
|                            | Text Classification           | Dimensionality reduction of TF-IDF features             |
| **Manufacturing**          | Sensor Data Analysis          | Reduce multivariate sensor readings                     |
|                            | Quality Control               | Identify key variables in production                    |
| **Exploratory Analysis**   | Data Visualization            | Project to 2D/3D to understand clusters/outliers        |
|                            | Preprocessing                 | Remove noise, reduce multicollinearity                  |
| **Anomaly Detection**      | Fraud Detection               | Points far from PCA subspace may be anomalies           |

### Concrete Example Snippets

- **Eigenfaces**: 100×100 pixel images (10,000 dims) → 50 components capture most facial variation
- **Finance**: 500 stock prices → 5–10 principal components explain majority of market variance
- **Genomics**: 20,000 genes → 2–3 PCs often separate distinct cell types/populations

---

## 4. Mathematical View

### 4.1 Problem Statement

Given a dataset of `n` samples and `p` features, represented as a matrix **X** of size `n × p`,
we want to find an orthogonal linear transformation **W** (size `p × k`, where `k ≤ p`) such
that the transformed data:

$$
\mathbf{Z} = \mathbf{X} \cdot \mathbf{W}
$$

has dimensions `n × k` and **maximizes the variance** of each column of **Z** (in order),
subject to orthonormality of columns of **W**.

> Equivalently, minimize reconstruction error:
> $$\|\mathbf{X} - \mathbf{Z}\mathbf{W}^T\|^2$$

---

### 4.2 Data Preprocessing

#### 4.2.1 Mean Centering (Required)

We center each feature to have mean 0:

$$
\bar{x}_j = \frac{1}{n}\sum_{i=1}^{n} x_{ij}
$$

$$
\tilde{x}_{ij} = x_{ij} - \bar{x}_j
$$

Matrix form:

$$
\tilde{\mathbf{X}} = \mathbf{X} - \mathbf{1}\bar{\mathbf{x}}^T
$$

> This ensures the first principal component doesn't simply point to the mean.

#### 4.2.2 Feature Scaling (Critical when features have different units/scales)

PCA is **sensitive to the variance** of individual features. Features with larger scales dominate.

- **Standardization (Z-score)** — Most common for PCA:

$$
z_{ij} = \frac{\tilde{x}_{ij}}{\sigma_j}, \quad
\sigma_j = \sqrt{\frac{1}{n-1}\sum_{i=1}^{n} (\tilde{x}_{ij})^2}
$$

Results in each feature having variance 1.

> **Rule of thumb**: Use standardization unless all features are already on the
> same scale and measured in comparable units.

---

### 4.3 Covariance Matrix

For centered data $\tilde{\mathbf{X}}$ (`n × p`), the **sample covariance matrix** is:

$$
\mathbf{C} = \frac{1}{n-1}\tilde{\mathbf{X}}^T \tilde{\mathbf{X}}
$$

- Dimensions: `p × p`
- $C_{jj}$ = variance of feature j
- $C_{jk}$ = covariance between feature j and k
- Symmetric positive semi-definite

If data is **standardized** (Z), this becomes the **correlation matrix**:

$$
\mathbf{R} = \frac{1}{n-1}\mathbf{Z}^T \mathbf{Z}
$$

with 1s on the diagonal.

---

### 4.4 Eigenvalue Decomposition

PCA finds the directions **w** such that the variance of projections is maximized:

$$
\max_{\mathbf{w}: \|\mathbf{w}\|=1} \text{Var}(\tilde{\mathbf{X}}\mathbf{w})
= \max_{\mathbf{w}: \|\mathbf{w}\|=1} \mathbf{w}^T \mathbf{C} \mathbf{w}
$$

Solution is given by the **eigenvectors** of **C**:

$$
\mathbf{C}\mathbf{w}_i = \lambda_i \mathbf{w}_i
$$

Where:
- $\lambda_i$ are **eigenvalues** (real, non-negative) → variance explained by each PC
- $\mathbf{w}_i$ are corresponding **eigenvectors** (orthonormal) → directions

**Properties:**

$$
\lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_p \geq 0
$$

**Total variance:**

$$
\sum_{j=1}^{p} \text{Var}(\text{feature } j) = \text{Tr}(\mathbf{C}) = \sum_{i=1}^{p} \lambda_i
$$

**Variance explained by first k components:**

$$
\frac{\sum_{i=1}^{k} \lambda_i}{\sum_{i=1}^{p} \lambda_i} \times 100\%
$$

---

### 4.5 Singular Value Decomposition (SVD) Perspective

For centered $\tilde{\mathbf{X}}$ (`n × p`), SVD is:

$$
\tilde{\mathbf{X}} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^T
$$

Where:
- **U** (`n × r`): left singular vectors (orthonormal)
- **Σ** (`r × r`): diagonal matrix of singular values $\sigma_1 \geq \cdots \geq \sigma_r > 0$
- **V** (`p × r`): right singular vectors (orthonormal)
- $r = \text{rank}(\tilde{\mathbf{X}}) \leq \min(n, p)$

Then:
- Eigenvectors of C are columns of **V**: $\mathbf{w}_i = \mathbf{v}_i$
- Eigenvalues: $\lambda_i = \frac{\sigma_i^2}{n-1}$
- PC scores: $\mathbf{Z} = \tilde{\mathbf{X}}\mathbf{V} = \mathbf{U}\boldsymbol{\Sigma}$

> **Note**: SVD is the preferred computational method in practice
> (more numerically stable, efficient for sparse data,
> avoids forming the covariance matrix explicitly). This is what
> scikit-learn uses internally.

---

### 4.6 Choosing the Number of Components

| Method                            | Description                                                  | Guideline                 |
|-----------------------------------|--------------------------------------------------------------|---------------------------|
| **Cumulative Explained Variance** | Choose k where cumulative variance ≥ threshold               | 80–95% common             |
| **Kaiser Criterion**              | Keep components with eigenvalue > 1 (correlation matrix)     | Eigenvalue > average      |
| **Scree Plot**                    | Plot eigenvalues vs component number, look for "elbow"       | Subjective, visual        |
| **Per-Component Variance**        | Drop components with very small individual variance          | e.g. < 1%                 |
| **Cross-Validation**              | Minimize reconstruction error on held-out data               | More expensive             |
| **Domain Knowledge**              | Based on interpretability or task requirements               | Context dependent          |

---

### 4.7 Reconstruction

Approximate original data from top k components:

$$
\hat{\mathbf{X}} = \mathbf{Z}_k \mathbf{W}_k^T + \mathbf{1}\bar{\mathbf{x}}^T
$$

**Reconstruction error (MSE):**

$$
\text{MSE} = \frac{1}{np}\|\mathbf{X} - \hat{\mathbf{X}}\|_F^2
= \frac{1}{p}\sum_{i=k+1}^{p} \lambda_i
$$

---

## 5. Implementation using Python Libraries

### 5.1 Required Libraries

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris, load_wine, load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
```

Optional for enhanced plots:

```python
import plotly.express as px
from mpl_toolkits.mplot3d import Axes3D
```

---

### 5.2 Standard PCA Workflow

1. **Load Data** — Get features `X` (and optionally labels `y`)
2. **Explore** — Check shape, missing values, correlations
3. **Preprocess**:
   - Handle missing values (if any)
   - Standardize features: `X_scaled = StandardScaler().fit_transform(X)`
4. **Apply PCA**:
   - Instantiate: `pca = PCA(n_components=k)` or `PCA(n_components=0.95)`
   - Fit: `pca.fit(X_scaled)`
   - Transform: `X_pca = pca.transform(X_scaled)`
5. **Analyze**:
   - Explained variance ratio: `pca.explained_variance_ratio_`
   - Cumulative: `np.cumsum(pca.explained_variance_ratio_)`
   - Components/loadings: `pca.components_`
6. **Visualize** — Scree plot, biplot, projection
7. **Interpret & Use** — Understand components, use for downstream task

---

### 5.3 Implementation with Scikit-learn

#### Basic Usage

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA
pca = PCA(n_components=2)       # Fixed number of components
# pca = PCA(n_components=0.95)  # OR keep 95% variance
X_pca = pca.fit_transform(X_scaled)

# Results
print("Explained variance ratio:", pca.explained_variance_ratio_)
print("Cumulative:", np.cumsum(pca.explained_variance_ratio_))
print("Components shape:", pca.components_.shape)  # (n_components, n_features)
```

#### Choosing `n_components` by Variance Threshold

```python
pca = PCA(n_components=0.95)  # keep 95% variance
X_pca = pca.fit_transform(X_scaled)
print(f"Number of components selected: {pca.n_components_}")
```

#### Accessing Loadings

Loadings indicate the correlation/weight of original features on each component:

```python
loadings = pd.DataFrame(
    pca.components_.T,
    columns=[f'PC{i+1}' for i in range(pca.n_components_)],
    index=feature_names
)
print(loadings)
```

---

### 5.4 Implementation with NumPy (From Scratch)

#### Using Eigen Decomposition

```python
import numpy as np

# 1. Center data
X_meaned = X - np.mean(X, axis=0)

# 2. Covariance matrix
cov_mat = np.cov(X_meaned, rowvar=False)  # (p × p)

# 3. Eigen decomposition
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)

# 4. Sort eigenvalues and eigenvectors (descending)
sorted_idx = np.argsort(eigen_vals)[::-1]
eigen_vals_sorted = eigen_vals[sorted_idx]
eigen_vecs_sorted = eigen_vecs[:, sorted_idx]

# 5. Explained variance
explained_variance = eigen_vals_sorted / np.sum(eigen_vals_sorted)

# 6. Projection to k dimensions
k = 2
W_k = eigen_vecs_sorted[:, :k]       # projection matrix (p × k)
X_pca_np = X_meaned @ W_k            # (n × k)
```

#### Using SVD (More Numerically Stable)

```python
U, S, Vt = np.linalg.svd(X_meaned, full_matrices=False)

eigen_vals_sorted = (S ** 2) / (len(X_meaned) - 1)
W_k = Vt.T[:, :k]
X_pca_svd = X_meaned @ W_k
```

> **Note**: This matches scikit-learn results (signs may vary, which is expected).

---

### 5.5 Useful Helper Functions and Visualizations

#### Scree Plot

```python
def plot_scree(pca, title="Scree Plot"):
    explained = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained)

    plt.figure(figsize=(8, 5))
    plt.bar(range(1, len(explained) + 1), explained, alpha=0.7, label='Individual')
    plt.plot(range(1, len(explained) + 1), cumulative, marker='o',
             color='red', label='Cumulative')
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
```

#### 2D Projection Plot

```python
def plot_2d_pca(X_pca, y, pca, labels=None, title="PCA Projection (2D)"):
    explained = pca.explained_variance_ratio_

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y,
                          cmap='viridis', s=60, alpha=0.8)
    plt.xlabel(f'PC1 ({explained[0]:.1%})')
    plt.ylabel(f'PC2 ({explained[1]:.1%})')
    plt.title(title)
    if labels is not None:
        plt.legend(handles=scatter.legend_elements()[0], labels=labels)
    plt.colorbar(scatter, label='Class')
    plt.grid(True, alpha=0.3)
    plt.show()
```

#### Biplot (Features + Samples)

```python
def plot_biplot(X_pca, loadings, feature_names, scale=1, title="Biplot"):
    plt.figure(figsize=(10, 8))

    # Samples
    plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6, s=50)

    # Feature vectors (arrows)
    for i, feat in enumerate(feature_names):
        plt.arrow(0, 0,
                  loadings[i, 0] * scale,
                  loadings[i, 1] * scale,
                  color='red', alpha=0.8, head_width=0.05)
        plt.text(loadings[i, 0] * scale * 1.15,
                 loadings[i, 1] * scale * 1.15,
                 feat, color='red', fontsize=12)

    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.axhline(0, color='grey', lw=0.5)
    plt.axvline(0, color='grey', lw=0.5)
    plt.show()
```

---

### 5.6 Common Pitfalls and Best Practices

#### Pitfalls

| Pitfall                                    | Solution                                        |
|--------------------------------------------|-------------------------------------------------|
| Not scaling features                       | Always use `StandardScaler()`                   |
| Interpreting PCs as causal                 | PCs are mathematical combinations, not causal   |
| Using PCA on categorical data              | Encode first or use MCA                         |
| Too many components without justification  | Use variance threshold (e.g. 95%)               |
| Ignoring outliers                          | Detect & handle outliers first                  |
| Assuming linearity                         | Try Kernel PCA for non-linear data              |
| Forgetting to inverse-scale reconstruction | Apply `scaler.inverse_transform()`              |
| Misreading loadings signs                  | Signs are arbitrary; magnitude matters          |
| Train/test data leakage                    | Fit PCA on training data only, transform test   |

#### Best Practices

- Always visualize explained variance
- For interpretation, look at top features by **absolute loading** per component
- Combine with domain knowledge
- For large datasets, consider `IncrementalPCA` or randomized SVD
- Compare model performance **with** and **without** PCA
- Document the number of components chosen and the rationale

---

## 6. Example Implementations

---

### Example 1 — Iris Dataset

**Goal**: Visualize 4-dimensional flower measurements in 2D, understand variance structure.

**Dataset**: Iris — 150 samples, 4 features, 3 classes

- Features: sepal length, sepal width, petal length, petal width
- Classes: Setosa, Versicolor, Virginica

---

#### Step 1: Load and Explore

```python
from sklearn.datasets import load_iris
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

print(f"Shape: {X.shape}")          # (150, 4)
print(f"Features: {feature_names}")
print(f"Classes: {target_names}")
```

---

#### Step 2: Explore Correlation

```python
df = pd.DataFrame(X, columns=feature_names)
df['species'] = y

corr = df[feature_names].corr()
plt.figure(figsize=(6, 5))
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, vmin=-1, vmax=1)
plt.title("Feature Correlation (Original)")
plt.show()
```

> **Note**: Petal length and petal width are highly correlated (~0.96) —
> a perfect candidate for PCA compression.

---

#### Step 3: Standardize

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Verify
print("Mean:", X_scaled.mean(axis=0).round(6))  # ~0
print("Std:",  X_scaled.std(axis=0).round(6))    # ~1
```

---

#### Step 4: Apply PCA (All Components First)

```python
pca = PCA(n_components=4)  # keep all to study variance
pca.fit(X_scaled)
X_pca_all = pca.transform(X_scaled)
```

---

#### Step 5: Analyze Explained Variance

```python
explained = pca.explained_variance_ratio_
cumulative = np.cumsum(explained)

print("Explained variance per PC:", explained)
print("Cumulative:", cumulative)

# Scree plot
plt.figure(figsize=(8, 5))
plt.bar(range(1, 5), explained, alpha=0.8, label='Individual')
plt.plot(range(1, 5), cumulative, marker='o', color='red', label='Cumulative')
plt.xlabel('Principal Component')
plt.ylabel('Variance Ratio')
plt.title('Iris — Scree Plot')
plt.xticks(range(1, 5))
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

**Results:**

| PC  | Individual Variance | Cumulative Variance |
|-----|---------------------|---------------------|
| PC1 | ~72.96%             | ~72.96%             |
| PC2 | ~22.85%             | ~95.81%             |
| PC3 | ~3.67%              | ~99.48%             |
| PC4 | ~0.52%              | 100.00%             |

> **Conclusion**: 2 components capture ~96% of variance. Sufficient for visualization.

---

#### Step 6: 2D Projection

```python
pca2 = PCA(n_components=2)
X_pca = pca2.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
for i, name in enumerate(target_names):
    idx = y == i
    plt.scatter(X_pca[idx, 0], X_pca[idx, 1], label=name, s=80, alpha=0.8)
plt.xlabel(f'PC1 ({pca2.explained_variance_ratio_[0]:.2%})')
plt.ylabel(f'PC2 ({pca2.explained_variance_ratio_[1]:.2%})')
plt.title('Iris Dataset — PCA (2 Components)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

> **Observation**: Setosa is clearly separated. Versicolor and Virginica show some overlap in 2D.

---

#### Step 7: Interpret Loadings

```python
loadings = pd.DataFrame(
    pca2.components_.T,
    columns=['PC1', 'PC2'],
    index=feature_names
)
print(loadings)

# Heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(loadings, annot=True, cmap='RdBu_r', center=0)
plt.title('Loadings (Feature Contributions to PCs)')
plt.show()
```

**Interpretation:**

- **PC1** → Dominated by petal length, petal width, sepal length (all positive, large magnitude)
  → Represents **overall flower size**
- **PC2** → Dominated by sepal width (positive) vs others (negative)
  → Contrasts **sepal width** with petal measurements

---

#### Step 8: Reconstruction Check

```python
X_reconstructed = pca2.inverse_transform(X_pca)

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(X_scaled, X_reconstructed)
remaining = 1 - np.sum(pca2.explained_variance_ratio_)

print(f"Reconstruction MSE (2 PCs): {mse:.4f}")
print(f"Remaining variance: {remaining:.4%}")
```

> **Key Notes for Iris:**
> - Excellent for visualization
> - PCA reveals natural separation between species
> - Loadings make biological sense
> - Only 2 out of 4 dimensions needed

---

### Example 2 — Wine Dataset

**Goal**: Reduce 13 chemical features to fewer components for classification. Compare classifier performance.

**Dataset**: Wine recognition — 178 samples, 13 features, 3 cultivar classes

---

#### Step 1: Load Data

```python
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

wine = load_wine()
X = wine.data
y = wine.target
feature_names = wine.feature_names

print(f"Shape: {X.shape}")  # (178, 13)
print(f"Features: {feature_names}")
```

---

#### Step 2: Train/Test Split and Standardize

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # use training scaler
```

> **Important**: Fit scaler on training data only. Transform test data with the same scaler.

---

#### Step 3: Baseline (No PCA)

```python
clf_base = LogisticRegression(max_iter=1000, random_state=42)
clf_base.fit(X_train_scaled, y_train)
y_pred_base = clf_base.predict(X_test_scaled)

print(f"Baseline Accuracy (13 features): {accuracy_score(y_test, y_pred_base):.3f}")
```

---

#### Step 4: Find Optimal Number of Components

```python
pca_full = PCA().fit(X_train_scaled)
explained = pca_full.explained_variance_ratio_
cumulative = np.cumsum(explained)

plt.figure(figsize=(8, 5))
plt.bar(range(1, 14), explained, alpha=0.7, label='Individual')
plt.plot(range(1, 14), cumulative, marker='o', color='red', label='Cumulative')
plt.axhline(0.95, color='green', linestyle='--', label='95% threshold')
plt.axhline(0.90, color='blue', linestyle='--', label='90% threshold')
plt.xlabel('Principal Component')
plt.ylabel('Variance Ratio')
plt.title('Wine — Explained Variance')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Print thresholds
print(f"Components for 90%: {np.argmax(cumulative >= 0.90) + 1}")
print(f"Components for 95%: {np.argmax(cumulative >= 0.95) + 1}")
```

---

#### Step 5: PCA + Classification with Different k

```python
ks = [2, 3, 4, 5, 6, 8, 13]
results = []

for k in ks:
    pca = PCA(n_components=k)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train_pca, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test_pca))
    var = np.sum(pca.explained_variance_ratio_)

    results.append({'k': k, 'Accuracy': acc, 'ExplainedVar': var})
    print(f"k={k:2d}: Accuracy={acc:.3f}, Variance={var:.3f}")

df_results = pd.DataFrame(results)
print("\n", df_results.to_string(index=False))
```

**Typical Results:**

| k  | Accuracy | Explained Variance |
|----|----------|--------------------|
| 2  | ~0.963   | ~0.554             |
| 3  | ~0.981   | ~0.668             |
| 6  | ~1.000   | ~0.875             |
| 13 | ~1.000   | ~1.000             |

> **Conclusion**: Even 2 PCs give ~96% accuracy. Massive dimensionality reduction (13→2) with minimal loss.

---

#### Step 6: Visualize 2D and 3D Projections

```python
# --- 2D ---
pca2 = PCA(n_components=2)
X_pca2 = pca2.fit_transform(X_train_scaled)

plt.figure(figsize=(8, 6))
for i in range(3):
    mask = y_train == i
    plt.scatter(X_pca2[mask, 0], X_pca2[mask, 1],
                label=wine.target_names[i], s=80)
plt.xlabel(f'PC1 ({pca2.explained_variance_ratio_[0]:.1%})')
plt.ylabel(f'PC2 ({pca2.explained_variance_ratio_[1]:.1%})')
plt.title('Wine — 2D PCA')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# --- 3D ---
from mpl_toolkits.mplot3d import Axes3D

pca3 = PCA(n_components=3)
X_pca3 = pca3.fit_transform(X_train_scaled)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
for i in range(3):
    mask = y_train == i
    ax.scatter(X_pca3[mask, 0], X_pca3[mask, 1], X_pca3[mask, 2],
               label=wine.target_names[i], s=60)
ax.set_xlabel(f'PC1 ({pca3.explained_variance_ratio_[0]:.1%})')
ax.set_ylabel(f'PC2 ({pca3.explained_variance_ratio_[1]:.1%})')
ax.set_zlabel(f'PC3 ({pca3.explained_variance_ratio_[2]:.1%})')
ax.set_title('Wine — 3D PCA')
ax.legend()
plt.show()
```

---

#### Step 7: Interpret Loadings

```python
pca6 = PCA(n_components=6).fit(X_train_scaled)
loadings6 = pd.DataFrame(
    pca6.components_.T,
    columns=[f'PC{i+1}' for i in range(6)],
    index=feature_names
)

# Top features per component
for pc in loadings6.columns:
    top = loadings6[pc].abs().sort_values(ascending=False).head(3)
    print(f"\n{pc} — Top features:")
    for feat, val in top.items():
        print(f"  {feat}: {loadings6.loc[feat, pc]:.3f}")
```

> **Key Notes for Wine:**
> - PCA is highly effective as a classification preprocessing step
> - Even 2 PCs provide excellent class separation
> - Loadings reveal which chemical properties differentiate cultivars
> - Always fit PCA on training data only to prevent leakage

---

### Example 3 — Digits Dataset

**Goal**: Image compression, visualization, and reconstruction of handwritten digits.

**Dataset**: Scikit-learn digits — 1,797 samples, 8×8 images, 10 classes (digits 0–9)

Each sample is 64 features (pixel values 0–16).

---

#### Step 1: Load and Explore

```python
from sklearn.datasets import load_digits

digits = load_digits()
X = digits.data       # (1797, 64)
y = digits.target
images = digits.images  # (1797, 8, 8)

print(f"Shape: {X.shape}")
print(f"Classes: {np.unique(y)}")
print(f"Pixel range: [{X.min()}, {X.max()}]")
```

---

#### Step 2: Visualize Original Images

```python
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in enumerate(axes.flat):
    ax.imshow(images[i], cmap='gray')
    ax.set_title(f"Digit: {y[i]}")
    ax.axis('off')
plt.suptitle("Original Digit Samples")
plt.tight_layout()
plt.show()
```

---

#### Step 3: Standardize

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

---

#### Step 4: Scree Plot — How Many Components?

```python
pca_full = PCA().fit(X_scaled)
explained = pca_full.explained_variance_ratio_
cumulative = np.cumsum(explained)

plt.figure(figsize=(10, 5))
plt.plot(range(1, len(cumulative) + 1), cumulative, marker='.', markersize=4)
plt.axhline(0.50, color='r', ls='--', label='50%')
plt.axhline(0.75, color='g', ls='--', label='75%')
plt.axhline(0.90, color='b', ls='--', label='90%')
plt.axhline(0.95, color='m', ls='--', label='95%')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Digits — Cumulative Variance Explained')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Print thresholds
for thresh in [0.50, 0.75, 0.90, 0.95]:
    k = np.argmax(cumulative >= thresh) + 1
    print(f"{thresh*100:.0f}% variance → k = {k} components")
```

**Typical Results:**

| Threshold | Components Needed | Compression Ratio |
|-----------|-------------------|--------------------|
| 50%       | ~7                | 64 → 7 (89% reduction) |
| 75%       | ~21               | 64 → 21 (67% reduction) |
| 90%       | ~37               | 64 → 37 (42% reduction) |
| 95%       | ~54               | 64 → 54 (16% reduction) |

---

#### Step 5: 2D and 3D Visualization

```python
# --- 2D ---
pca2 = PCA(n_components=2)
X_pca2 = pca2.fit_transform(X_scaled)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_pca2[:, 0], X_pca2[:, 1],
                      c=y, cmap='tab10', s=30, alpha=0.7)
plt.colorbar(scatter, ticks=range(10), label='Digit')
plt.xlabel(f'PC1 ({pca2.explained_variance_ratio_[0]:.1%})')
plt.ylabel(f'PC2 ({pca2.explained_variance_ratio_[1]:.1%})')
plt.title('Digits — 2D PCA (colored by digit)')
plt.show()

# --- 3D ---
pca3 = PCA(n_components=3)
X_pca3 = pca3.fit_transform(X_scaled)

fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(X_pca3[:, 0], X_pca3[:, 1], X_pca3[:, 2],
                     c=y, cmap='tab10', s=20, alpha=0.7)
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
plt.colorbar(scatter, ticks=range(10))
plt.title('Digits — 3D PCA')
plt.show()
```

> **Note**: 2D captures only ~12% variance — some clustering visible but heavy overlap.
> For digits, t-SNE or UMAP would give better 2D separation.
> PCA's strength here is compression and denoising.

---

#### Step 6: Image Reconstruction (Compression)

```python
def reconstruct_digits(n_components, n_images=5):
    """Reconstruct digit images from k principal components."""
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    X_recon_scaled = pca.inverse_transform(X_pca)
    X_recon = scaler.inverse_transform(X_recon_scaled)

    fig, axes = plt.subplots(n_images, 2, figsize=(6, n_images * 2))
    for i in range(n_images):
        # Original
        axes[i, 0].imshow(images[i], cmap='gray')
        axes[i, 0].set_title("Original")
        axes[i, 0].axis('off')
        # Reconstructed
        recon_img = X_recon[i].reshape(8, 8)
        axes[i, 1].imshow(recon_img, cmap='gray')
        axes[i, 1].set_title(f"Recon ({n_components} PCs)")
        axes[i, 1].axis('off')

    var = np.sum(pca.explained_variance_ratio_)
    plt.suptitle(f"Reconstruction: {n_components} components ({var:.1%} variance)")
    plt.tight_layout()
    plt.show()

# Compare different compression levels
for k in [5, 10, 21, 37]:
    reconstruct_digits(k)
```

**Observations:**

| Components | Quality                                    |
|------------|--------------------------------------------|
| k=5        | Blurry, basic shape visible                |
| k=10       | Recognizable for most digits               |
| k=21       | Very close to original (~75% variance)     |
| k=37       | Almost indistinguishable (~90% variance)   |

---

#### Step 7: Visualize Principal Components as Images

```python
pca50 = PCA(n_components=50).fit(X_scaled)
components = pca50.components_  # (50, 64)

fig, axes = plt.subplots(5, 5, figsize=(10, 10))
for i, ax in enumerate(axes.flat):
    comp_img = components[i].reshape(8, 8)
    ax.imshow(comp_img, cmap='coolwarm')
    ax.set_title(f"PC{i+1}")
    ax.axis('off')
plt.suptitle("First 25 Principal Components (as 8×8 images)")
plt.tight_layout()
plt.show()
```

> These show the stroke/curve patterns the model uses to reconstruct digits
> — analogous to "eigenfaces" in face recognition.

---

#### Step 8: Compression Ratio Summary

```python
original_size = 64
print(f"{'k':>4} | {'Dims Used':>10} | {'Variance':>10} | {'Compression':>12}")
print("-" * 45)
for k in [5, 10, 21, 37, 54, 64]:
    pca = PCA(n_components=k).fit(X_scaled)
    var = np.sum(pca.explained_variance_ratio_) * 100
    ratio = k / original_size * 100
    print(f"{k:4d} | {ratio:9.1f}% | {var:9.1f}% | {original_size}→{k}")
```

> **Key Notes for Digits:**
> - PCA is excellent for image compression and denoising
> - Visualization in 2D is limited (low variance captured), but structure is visible
> - Components are interpretable as basis patterns (strokes)
> - Can be used as preprocessing for faster classification with minimal accuracy loss
> - For better 2D visualization of digits, pair with t-SNE or UMAP

---

## 7. Conclusion and Further Reading

### Key Takeaways

| Point                   | Summary                                                              |
|-------------------------|----------------------------------------------------------------------|
| **Purpose**             | Reduce dimensionality by finding directions of maximum variance      |
| **Mathematics**         | Eigen decomposition of covariance matrix (or SVD of centered data)   |
| **Preprocessing**       | Standardization is almost always necessary                           |
| **Interpretation**      | Loadings show feature contributions; explained variance guides k     |
| **Strengths**           | Simple, fast, unsupervised, removes correlation, effective for viz   |
| **Weaknesses**          | Linear, variance ≠ always importance, sensitive to scaling/outliers  |

### When to Use PCA

- Data visualization (2–3D)
- High correlation among features
- Need to reduce noise or multicollinearity
- Computational constraints with many features
- Exploratory data analysis

### When NOT to Use PCA

- Non-linear structure is important
- Working primarily with categorical data
- Task requires preserving local distances precisely (use t-SNE/UMAP)
- Interpretability of original features is critical and linear combinations are not acceptable

### Further Reading and Extensions

| Technique              | Purpose                                 | Relation to PCA          |
|------------------------|-----------------------------------------|--------------------------|
| **Kernel PCA**         | Non-linear dimensionality reduction     | Uses kernel trick        |
| **Sparse PCA**         | Sparse loadings for interpretability    | Adds L1 penalty          |
| **Incremental PCA**    | Large datasets (out-of-memory)          | Batch processing         |
| **Randomized PCA**     | Faster for large p                      | Approximate SVD          |
| **Robust PCA**         | Handles outliers/noise                  | Low-rank + sparse        |
| **ICA**                | Find independent sources                | Independence objective   |
| **t-SNE**              | Non-linear visualization                | Preserves local structure|
| **UMAP**               | Faster non-linear visualization         | Local + global structure |
| **Factor Analysis**    | Latent factors with noise model         | Probabilistic framework  |

### References

1. Jolliffe, I. T. (2002). *Principal Component Analysis*. Springer.
2. Pearson, K. (1901). On lines and planes of closest fit to systems of points in space.
3. Hotelling, H. (1933). Analysis of a complex of statistical variables into principal components.
4. Shlens, J. (2014). A tutorial on principal component analysis. [arXiv:1404.1100](https://arxiv.org/abs/1404.1100)
5. Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer.
6. [Scikit-learn PCA Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)

---

> **Final Reminders:**
>
> - Always **scale** your data before PCA
> - Start with a **scree plot** + cumulative variance analysis
> - Don't blindly trust PCA — **visualize and interpret**
> - For classification tasks, **compare** performance with and without PCA
> - Fit PCA on **training data only** to prevent data leakage
> - Document the number of components chosen and the rationale

---
