# K-Means Clustering: Complete Documentation

---

## Table of Contents
1. [Introduction](#introduction)
2. [Core Idea](#core-idea)
3. [Why Do We Need K-Means](#why-do-we-need-k-means)
4. [Applications](#applications)
5. [Mathematical Foundation](#mathematical-foundation)
6. [Working Procedure](#working-procedure)
7. [Implementation using Python](#implementation-using-python)
8. [Datasets and Experiments](#datasets-and-experiments)
9. [Results](#results)
10. [Conclusion](#conclusion)

---

## Introduction

**K-Means Clustering** is one of the most popular and widely used **unsupervised machine learning algorithms**. It is used to partition a dataset into **K distinct, non-overlapping groups (clusters)** based on feature similarity.

Unlike supervised learning, K-Means does **not require labeled data**. Instead, it identifies hidden patterns and natural groupings within the data. The "K" represents the number of clusters we want the algorithm to find, and "Means" refers to the averaging of data points to find the cluster centers (centroids).

> **Type:** Unsupervised Learning
> **Category:** Partitional Clustering
> **Output:** K clusters with centroids

---

## Core Idea

The core idea of K-Means is simple yet powerful:

> **Group similar data points together while keeping dissimilar points apart.**

The algorithm achieves this by:

1. Selecting **K centroids** (cluster centers).
2. Assigning each data point to the **nearest centroid**.
3. Recalculating the centroids based on the mean of assigned points.
4. Repeating until centroids **stabilize** (convergence).

The goal is to **minimize intra-cluster distance** (points within a cluster are close together) and **maximize inter-cluster distance** (clusters are far apart).

```
Data Points  →  Assign to Nearest Centroid  →  Update Centroids  →  Repeat
```

---

## Why Do We Need K-Means

In real-world scenarios, data often comes **without labels**. We need methods to discover structure automatically. K-Means helps us because:

| Need | How K-Means Helps |
|------|-------------------|
| **Pattern Discovery** | Finds natural groupings without supervision |
| **Data Simplification** | Summarizes large datasets into K groups |
| **Scalability** | Efficient and fast even on large datasets |
| **Preprocessing** | Useful for feature engineering and segmentation |
| **Anomaly Detection** | Outliers can be detected as points far from centroids |

### Advantages
- Simple to understand and implement.
- Computationally efficient (linear time complexity).
- Scales well to large datasets.
- Guarantees convergence.

### Limitations
- Requires specifying **K** in advance.
- Sensitive to **initial centroid placement**.
- Assumes **spherical clusters** of similar size.
- Sensitive to **outliers** and **feature scaling**.

---

## Applications

K-Means is used across many domains:

- 🛒 **Customer Segmentation** – Grouping customers by purchasing behavior.
- 🖼️ **Image Compression** – Reducing the number of colors in an image.
- 📄 **Document Clustering** – Organizing text documents by topic.
- 🧬 **Bioinformatics** – Grouping genes with similar expression patterns.
- 🏦 **Fraud / Anomaly Detection** – Identifying unusual transactions.
- 🌍 **Geospatial Analysis** – Identifying location-based clusters.
- 📊 **Market Basket Analysis** – Grouping similar products.

---

## Mathematical Foundation

This section explains **everything** behind K-Means mathematically.

### 1. Notation

- Let the dataset be $X = \{x_1, x_2, \dots, x_n\}$ where each $x_i \in \mathbb{R}^d$ (d-dimensional point).
- We want to partition data into **K clusters**: $C = \{C_1, C_2, \dots, C_K\}$.
- Each cluster $C_k$ has a **centroid** $\mu_k$.

### 2. Distance Metric (Euclidean Distance)

K-Means typically uses **Euclidean distance** to measure similarity between a point $x_i$ and a centroid $\mu_k$:

$$
d(x_i, \mu_k) = \sqrt{\sum_{j=1}^{d} (x_{ij} - \mu_{kj})^2}
$$

For computational efficiency, the **squared Euclidean distance** is used:

$$
d^2(x_i, \mu_k) = \sum_{j=1}^{d} (x_{ij} - \mu_{kj})^2
$$

### 3. Objective Function (Cost Function)

The goal of K-Means is to minimize the **Within-Cluster Sum of Squares (WCSS)**, also called **inertia** or **distortion**:

$$
J = \sum_{k=1}^{K} \sum_{x_i \in C_k} \| x_i - \mu_k \|^2
$$

Where:
- $J$ = total objective (cost) to minimize.
- $\| x_i - \mu_k \|^2$ = squared distance from a point to its cluster centroid.

This measures how **tight** the clusters are. Lower $J$ = better clustering.

### 4. Centroid Update Formula

The centroid of cluster $C_k$ is the **mean** of all points assigned to it:

$$
\mu_k = \frac{1}{|C_k|} \sum_{x_i \in C_k} x_i
$$

Where $|C_k|$ is the number of points in cluster $k$.

### 5. Assignment Step

Each point is assigned to the cluster with the nearest centroid:

$$
C_k = \{ x_i : \| x_i - \mu_k \|^2 \le \| x_i - \mu_j \|^2 \;\; \forall j, 1 \le j \le K \}
$$

### 6. The Optimization (EM-Style)

K-Means is essentially an **Expectation-Maximization (EM)** process with two alternating steps:

- **E-step (Assignment):** Fix centroids, assign points to nearest centroid.
- **M-step (Update):** Fix assignments, recompute centroids as means.

Each iteration **never increases** the cost $J$, guaranteeing convergence to a **local minimum**.

### 7. Choosing K — The Elbow Method

We plot WCSS against different K values. The **"elbow point"** (where the rate of decrease sharply slows) suggests the optimal K.

$$
\text{Optimal K} \approx \text{point of maximum curvature in WCSS vs K plot}
$$

### 8. Silhouette Score

Another way to evaluate clustering quality:

$$
s(i) = \frac{b(i) - a(i)}{\max\{a(i), b(i)\}}
$$

Where:
- $a(i)$ = average distance from point $i$ to other points in its cluster (cohesion).
- $b(i)$ = average distance to points in the nearest neighboring cluster (separation).
- Score ranges from **-1 to +1** (higher is better).

---

## Working Procedure

The step-by-step algorithm:

```
ALGORITHM: K-Means

Step 1: Choose the number of clusters K.

Step 2: Initialize K centroids randomly 
        (or using K-Means++).

Step 3: REPEAT until convergence:
        
        a) ASSIGNMENT STEP:
           For each data point, compute distance to all 
           centroids and assign it to the nearest one.
        
        b) UPDATE STEP:
           Recalculate each centroid as the mean of all 
           points assigned to that cluster.

Step 4: STOP when centroids no longer change 
        (or max iterations reached).
```

### K-Means++ Initialization
To avoid poor random initialization, **K-Means++** chooses initial centroids that are spread out:
1. Choose first centroid randomly.
2. Choose subsequent centroids with probability proportional to squared distance from existing centroids.

---

## Implementation using Python

We will use **Sklearn**, **NumPy**, **Pandas**, and **Matplotlib/Seaborn**.

### Importing Libraries

```python
# Core libraries
import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.datasets import load_iris, load_wine, make_blobs

# Settings
sns.set(style="whitegrid")
%matplotlib inline
```

### Helper Function: Elbow Method

```python
def plot_elbow(data, max_k=10):
    wcss = []
    for k in range(1, max_k + 1):
        km = KMeans(n_clusters=k, init='k-means++', 
                    random_state=42, n_init=10)
        km.fit(data)
        wcss.append(km.inertia_)
    
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, max_k + 1), wcss, marker='o', linestyle='--')
    plt.title('Elbow Method')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('WCSS (Inertia)')
    plt.show()
    return wcss
```

---

## Datasets and Experiments

We use **3 different datasets** with EDA and scaling.

---

### 📊 Dataset 1: Iris Dataset

A classic dataset with 150 samples of iris flowers (4 features).

#### Loading & EDA

```python
# Load dataset
iris = load_iris()
df_iris = pd.DataFrame(iris.data, columns=iris.feature_names)

# Exploratory Data Analysis
print(df_iris.head())
print(df_iris.describe())
print(df_iris.isnull().sum())

# Visualize relationships
sns.pairplot(df_iris)
plt.show()

# Correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df_iris.corr(), annot=True, cmap='coolwarm')
plt.title('Iris Feature Correlation')
plt.show()
```

#### Scaling

```python
scaler = StandardScaler()
iris_scaled = scaler.fit_transform(df_iris)
```

#### Finding Optimal K

```python
plot_elbow(iris_scaled, max_k=10)
```

#### Applying K-Means

```python
# From elbow, optimal K = 3
kmeans_iris = KMeans(n_clusters=3, init='k-means++', 
                     random_state=42, n_init=10)
iris_labels = kmeans_iris.fit_predict(iris_scaled)

df_iris['Cluster'] = iris_labels

# Silhouette Score
score = silhouette_score(iris_scaled, iris_labels)
print(f"Iris Silhouette Score: {score:.3f}")

# Visualization
plt.figure(figsize=(8, 6))
sns.scatterplot(x=df_iris['petal length (cm)'], 
                y=df_iris['petal width (cm)'], 
                hue=df_iris['Cluster'], palette='viridis', s=80)
plt.scatter(kmeans_iris.cluster_centers_[:, 2], 
            kmeans_iris.cluster_centers_[:, 3],
            s=250, c='red', marker='X', label='Centroids')
plt.title('Iris Clusters')
plt.legend()
plt.show()
```

---

### 📊 Dataset 2: Wine Dataset

178 samples of wine with 13 chemical features.

#### Loading & EDA

```python
wine = load_wine()
df_wine = pd.DataFrame(wine.data, columns=wine.feature_names)

print(df_wine.head())
print(df_wine.describe())
print("Missing values:", df_wine.isnull().sum().sum())

# Distribution of a few features
df_wine[['alcohol', 'malic_acid', 'color_intensity']].hist(
    figsize=(12, 4), bins=20)
plt.suptitle('Wine Feature Distributions')
plt.show()
```

#### Scaling (Important - features have different ranges)

```python
scaler = StandardScaler()
wine_scaled = scaler.fit_transform(df_wine)
```

#### Optimal K & Clustering

```python
plot_elbow(wine_scaled, max_k=10)

# Optimal K = 3 (3 wine cultivars)
kmeans_wine = KMeans(n_clusters=3, init='k-means++', 
                     random_state=42, n_init=10)
wine_labels = kmeans_wine.fit_predict(wine_scaled)

df_wine['Cluster'] = wine_labels

score = silhouette_score(wine_scaled, wine_labels)
print(f"Wine Silhouette Score: {score:.3f}")

# Visualize using 2 important features
plt.figure(figsize=(8, 6))
sns.scatterplot(x=df_wine['alcohol'], 
                y=df_wine['color_intensity'],
                hue=df_wine['Cluster'], palette='Set1', s=80)
plt.title('Wine Clusters')
plt.show()
```

---

### 📊 Dataset 3: Synthetic Blobs Dataset

A generated dataset to clearly demonstrate clustering.

#### Creating & EDA

```python
X, y_true = make_blobs(n_samples=500, centers=4, 
                       cluster_std=0.80, random_state=42)
df_blobs = pd.DataFrame(X, columns=['Feature_1', 'Feature_2'])

print(df_blobs.head())
print(df_blobs.describe())

# Visualize raw data
plt.figure(figsize=(8, 6))
plt.scatter(df_blobs['Feature_1'], df_blobs['Feature_2'], s=50)
plt.title('Raw Synthetic Data')
plt.show()
```

#### Scaling

```python
scaler = StandardScaler()
blobs_scaled = scaler.fit_transform(df_blobs)
```

#### Clustering

```python
plot_elbow(blobs_scaled, max_k=10)

# Optimal K = 4
kmeans_blobs = KMeans(n_clusters=4, init='k-means++', 
                      random_state=42, n_init=10)
blobs_labels = kmeans_blobs.fit_predict(blobs_scaled)

df_blobs['Cluster'] = blobs_labels

score = silhouette_score(blobs_scaled, blobs_labels)
print(f"Blobs Silhouette Score: {score:.3f}")

# Visualization
plt.figure(figsize=(8, 6))
sns.scatterplot(x=df_blobs['Feature_1'], 
                y=df_blobs['Feature_2'],
                hue=df_blobs['Cluster'], palette='deep', s=60)
plt.scatter(kmeans_blobs.cluster_centers_[:, 0], 
            kmeans_blobs.cluster_centers_[:, 1],
            s=250, c='black', marker='X', label='Centroids')
plt.title('Synthetic Blobs Clusters')
plt.legend()
plt.show()
```

---

## Results

The following table summarizes typical results obtained from the three datasets:

| Dataset | Samples | Features | Optimal K | Silhouette Score | Notes |
|---------|---------|----------|-----------|------------------|-------|
| **Iris** | 150 | 4 | 3 | ~0.46 | Good separation; some overlap between two species |
| **Wine** | 178 | 13 | 3 | ~0.28 | Scaling critical; clusters match 3 cultivars |
| **Blobs** | 500 | 2 | 4 | ~0.79 | Excellent, well-separated synthetic clusters |

### Key Observations

1. **Scaling matters:** The Wine dataset had features with vastly different ranges. Without `StandardScaler`, large-magnitude features would dominate distance calculations.

2. **Synthetic data performs best:** The blobs dataset achieved the highest silhouette score because clusters were intentionally well-separated and spherical — ideal conditions for K-Means.

3. **Elbow Method effectiveness:** In all three cases, the elbow plot clearly indicated the correct number of clusters matching the ground truth.

4. **Real data is messier:** Iris and Wine had naturally overlapping classes, lowering silhouette scores — a realistic reflection of real-world data.

---

## Conclusion

K-Means Clustering is a **fundamental, intuitive, and efficient** unsupervised learning algorithm for discovering structure in unlabeled data.

### Summary of Learnings
- ✅ K-Means partitions data into **K clusters** by minimizing the **within-cluster sum of squares (WCSS)**.
- ✅ It alternates between **assignment** and **update** steps until convergence.
- ✅ **Feature scaling** is essential because the algorithm relies on distance metrics.
- ✅ The **Elbow Method** and **Silhouette Score** help choose and evaluate K.
- ✅ It works best with **spherical, well-separated, similarly-sized clusters**.

### Best Practices
- Always **scale features** before clustering.
- Use **K-Means++** initialization to avoid poor local minima.
- Run with multiple initializations (`n_init`) for stability.
- Validate results using **silhouette analysis** and domain knowledge.

### Final Thought
While K-Means has limitations (predefined K, sensitivity to outliers, assumption of spherical clusters), its **simplicity, speed, and scalability** make it an indispensable tool in any data scientist's toolkit and an excellent starting point for exploratory cluster analysis.

---

> �‌ **References**
> - Scikit-learn Documentation: https://scikit-learn.org/stable/modules/clustering.html
> - MacQueen, J. (1967). *Some Methods for Classification and Analysis of Multivariate Observations*.
> - Arthur, D. & Vassilvitskii, S. (2007). *k-means++: The Advantages of Careful Seeding*.

---

*Documentation created for educational purposes on K-Means Clustering.*