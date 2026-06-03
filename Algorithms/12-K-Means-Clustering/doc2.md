# K-Means Clustering Documentation

## Table of Contents

1. [Introduction](#introduction)
2. [Core Idea](#core-idea)
3. [Why Do We Need K-Means?](#why-do-we-need-k-means)
4. [Applications](#applications)
5. [Mathematical Foundation](#mathematical-foundation)
6. [Working Procedure](#working-procedure)
7. [Implementation Using Python, Scikit-learn, NumPy, and Pandas](#implementation-using-python-scikit-learn-numpy-and-pandas)
8. [Dataset 1: Iris Dataset](#dataset-1-iris-dataset)
9. [Dataset 2: Wine Dataset](#dataset-2-wine-dataset)
10. [Dataset 3: Digits Dataset](#dataset-3-digits-dataset)
11. [Results](#results)
12. [Conclusion](#conclusion)

---

# Introduction

K-Means Clustering is one of the most popular unsupervised machine learning algorithms used for grouping similar data points into clusters.

Unlike supervised learning, where the model learns from labeled data, K-Means works with unlabeled data. It tries to discover hidden patterns or natural groupings in the dataset.

The main goal of K-Means is to divide a dataset into `K` distinct clusters, where each data point belongs to the cluster with the nearest mean or centroid.

A centroid is the central point of a cluster.

---

# Core Idea

The core idea of K-Means is simple:

> Data points that are close to each other are more similar and should belong to the same cluster.

K-Means tries to find `K` cluster centers such that the total distance between data points and their assigned cluster centers is minimized.

For example, if we have customer data, K-Means can group customers into segments such as:

- Low-spending customers
- Medium-spending customers
- High-spending customers

The algorithm repeatedly performs two main steps:

1. Assign each data point to the nearest centroid.
2. Update each centroid by taking the mean of all points assigned to it.

This process continues until the centroids stop changing significantly or a maximum number of iterations is reached.

---

# Why Do We Need K-Means?

K-Means is useful because many real-world datasets do not come with labels.

For example:

- A company may have customer data but no predefined customer segments.
- A hospital may have patient records but no clear disease subgroups.
- An image may contain millions of pixels that need to be grouped by color.
- A website may want to group users based on browsing behavior.

K-Means helps in:

- Discovering hidden patterns
- Reducing complexity
- Creating meaningful groups
- Data summarization
- Feature engineering
- Market segmentation
- Anomaly detection support

K-Means is especially popular because it is:

- Simple to understand
- Fast for large datasets
- Easy to implement
- Scalable
- Useful as a baseline clustering algorithm

---

# Applications

K-Means Clustering is widely used in many fields.

## 1. Customer Segmentation

Businesses use K-Means to divide customers into groups based on:

- Spending behavior
- Purchase frequency
- Age
- Income
- Product preferences

## 2. Image Compression

Images contain pixels with RGB values. K-Means can reduce the number of colors by grouping similar colors together.

## 3. Document Clustering

Documents can be grouped by topic using word frequency or TF-IDF features.

## 4. Anomaly Detection

After clustering normal data points, points far from cluster centers can be considered suspicious or anomalous.

## 5. Healthcare

Patients can be grouped based on medical measurements, symptoms, or disease progression.

## 6. Recommendation Systems

Users with similar behavior can be grouped to improve recommendations.

## 7. Pattern Recognition

K-Means is used in speech recognition, image recognition, and signal processing.

---

# Mathematical Foundation

## 1. Dataset Representation

Suppose we have a dataset containing `n` data points:

\[
X = \{x_1, x_2, x_3, ..., x_n\}
\]

Each data point has `d` features:

\[
x_i = (x_{i1}, x_{i2}, ..., x_{id})
\]

For example, in the Iris dataset, one sample may be represented as:

\[
x_i = (\text{sepal length}, \text{sepal width}, \text{petal length}, \text{petal width})
\]

---

## 2. Number of Clusters

K-Means requires the number of clusters `K` to be specified before training.

Example:

\[
K = 3
\]

This means the algorithm will divide the dataset into 3 clusters.

---

## 3. Cluster Centroids

Each cluster has a centroid.

If cluster `C_j` contains multiple points, its centroid is the average of all points in that cluster.

\[
\mu_j = \frac{1}{|C_j|} \sum_{x_i \in C_j} x_i
\]

Where:

- \(\mu_j\) = centroid of cluster `j`
- \(C_j\) = set of points assigned to cluster `j`
- \(|C_j|\) = number of points in cluster `j`
- \(x_i\) = data point

---

## 4. Distance Measurement

K-Means usually uses Euclidean distance.

The Euclidean distance between a data point \(x_i\) and a centroid \(\mu_j\) is:

\[
d(x_i, \mu_j) = \sqrt{\sum_{m=1}^{d}(x_{im} - \mu_{jm})^2}
\]

Where:

- \(d\) = number of features
- \(x_{im}\) = value of feature `m` for data point `i`
- \(\mu_{jm}\) = value of feature `m` for centroid `j`

The smaller the distance, the more similar the data point is to the centroid.

---

## 5. Objective Function

K-Means tries to minimize the sum of squared distances between data points and their assigned centroids.

This is known as Within-Cluster Sum of Squares, WCSS, or inertia.

\[
J = \sum_{j=1}^{K} \sum_{x_i \in C_j} ||x_i - \mu_j||^2
\]

Where:

- \(J\) = objective function
- \(K\) = number of clusters
- \(C_j\) = cluster `j`
- \(\mu_j\) = centroid of cluster `j`
- \(||x_i - \mu_j||^2\) = squared Euclidean distance

The goal is:

\[
\text{Minimize } J
\]

A smaller value of \(J\) means data points are closer to their cluster centers.

---

## 6. Assignment Step

Each data point is assigned to the nearest centroid.

\[
C_j = \{x_i : ||x_i - \mu_j||^2 \leq ||x_i - \mu_l||^2, \forall l \}
\]

This means data point \(x_i\) is assigned to cluster `j` if centroid \(\mu_j\) is the closest centroid.

---

## 7. Update Step

After assigning points to clusters, the centroid of each cluster is recalculated.

\[
\mu_j = \frac{1}{|C_j|} \sum_{x_i \in C_j} x_i
\]

This new centroid becomes the mean of all points currently assigned to that cluster.

---

## 8. Convergence

The algorithm repeats the assignment and update steps until convergence.

Convergence occurs when:

- Centroids no longer change significantly
- Cluster assignments stop changing
- Maximum iterations are reached
- The decrease in inertia is very small

K-Means is guaranteed to converge, but it may converge to a local minimum rather than a global minimum.

---

## 9. Importance of Scaling

K-Means is distance-based. Therefore, features with large numeric ranges can dominate the distance calculation.

Example:

| Feature | Range |
|---|---:|
| Age | 18 to 70 |
| Annual Income | 10,000 to 200,000 |

Without scaling, income dominates the distance calculation.

To avoid this, we usually apply standardization:

\[
z = \frac{x - \mu}{\sigma}
\]

Where:

- \(z\) = standardized value
- \(x\) = original value
- \(\mu\) = feature mean
- \(\sigma\) = feature standard deviation

After standardization, each feature has:

\[
\text{mean} = 0
\]

\[
\text{standard deviation} = 1
\]

---

## 10. Choosing the Value of K

Choosing the right value of `K` is important.

Common methods include:

## Elbow Method

The Elbow Method plots inertia against different values of `K`.

As `K` increases, inertia decreases. The best `K` is often where the decrease starts slowing down, forming an elbow shape.

## Silhouette Score

Silhouette Score measures how well-separated clusters are.

For a data point \(i\):

\[
s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}
\]

Where:

- \(a(i)\) = average distance from point `i` to other points in the same cluster
- \(b(i)\) = average distance from point `i` to points in the nearest different cluster

The silhouette score ranges from `-1` to `1`.

| Score | Meaning |
|---:|---|
| Close to 1 | Well-clustered |
| Around 0 | Overlapping clusters |
| Negative | Possibly assigned to wrong cluster |

---

## 11. K-Means++ Initialization

Random initialization can lead to poor clustering results.

K-Means++ improves initialization by choosing initial centroids that are far apart.

In Scikit-learn, K-Means++ is used by default:

```python
KMeans(init="k-means++")
```

This usually improves convergence and clustering quality.

---

## 12. Computational Complexity

The approximate time complexity of K-Means is:

\[
O(n \times K \times d \times i)
\]

Where:

- \(n\) = number of data points
- \(K\) = number of clusters
- \(d\) = number of features
- \(i\) = number of iterations

K-Means is generally efficient, but performance can decrease for very large datasets or high-dimensional data.

---

## 13. Limitations of K-Means

K-Means has some limitations:

- Requires choosing `K` in advance
- Sensitive to outliers
- Sensitive to feature scaling
- Assumes clusters are roughly spherical
- Does not perform well with complex-shaped clusters
- Can converge to local minima
- Poor initialization can affect results

---

# Working Procedure

The K-Means algorithm follows these steps:

1. Select the number of clusters `K`.
2. Initialize `K` centroids randomly or using K-Means++.
3. Calculate the distance between each point and each centroid.
4. Assign each point to the nearest centroid.
5. Recalculate centroids as the mean of assigned points.
6. Repeat steps 3 to 5 until convergence.
7. Return final clusters and centroids.

---

# Implementation Using Python, Scikit-learn, NumPy, and Pandas

The following implementation uses three datasets:

1. Iris Dataset
2. Wine Dataset
3. Digits Dataset

All three datasets are available directly from `sklearn.datasets`.

We will use:

- `NumPy` for numerical operations
- `Pandas` for tabular data handling
- `Scikit-learn` for preprocessing, clustering, datasets, and evaluation

---

## Install Required Libraries

```bash
pip install numpy pandas scikit-learn matplotlib
```

---

## Import Libraries

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris, load_wine, load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.decomposition import PCA
```

---

## Helper Functions

```python
def create_dataframe(dataset):
    """
    Convert sklearn dataset object into a pandas DataFrame.
    """
    X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    y = pd.Series(dataset.target, name="target")
    return X, y


def basic_eda(X, y=None, dataset_name="Dataset"):
    """
    Perform basic exploratory data analysis.
    """
    print("=" * 80)
    print(f"EDA for {dataset_name}")
    print("=" * 80)
    
    print("\nShape:")
    print(X.shape)
    
    print("\nFirst 5 Rows:")
    print(X.head())
    
    print("\nData Types:")
    print(X.dtypes)
    
    print("\nMissing Values:")
    print(X.isnull().sum())
    
    print("\nStatistical Summary:")
    print(X.describe())
    
    if y is not None:
        print("\nTarget Distribution:")
        print(y.value_counts().sort_index())


def scale_data(X):
    """
    Standardize features using StandardScaler.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


def evaluate_kmeans(X_scaled, y_true, k, dataset_name):
    """
    Train K-Means and calculate evaluation metrics.
    """
    model = KMeans(
        n_clusters=k,
        init="k-means++",
        n_init=20,
        max_iter=300,
        random_state=42
    )
    
    clusters = model.fit_predict(X_scaled)
    
    inertia = model.inertia_
    silhouette = silhouette_score(X_scaled, clusters)
    
    if y_true is not None:
        ari = adjusted_rand_score(y_true, clusters)
    else:
        ari = None
    
    print("=" * 80)
    print(f"K-Means Results for {dataset_name}")
    print("=" * 80)
    print(f"Number of clusters: {k}")
    print(f"Inertia: {inertia:.4f}")
    print(f"Silhouette Score: {silhouette:.4f}")
    
    if ari is not None:
        print(f"Adjusted Rand Index: {ari:.4f}")
    
    print("\nCluster Counts:")
    print(pd.Series(clusters).value_counts().sort_index())
    
    if y_true is not None:
        print("\nCluster vs Actual Class Table:")
        print(pd.crosstab(pd.Series(y_true, name="Actual"), pd.Series(clusters, name="Cluster")))
    
    return model, clusters, inertia, silhouette, ari


def plot_elbow_method(X_scaled, max_k=10, dataset_name="Dataset"):
    """
    Plot inertia for multiple values of K.
    """
    inertias = []
    k_values = range(1, max_k + 1)
    
    for k in k_values:
        model = KMeans(
            n_clusters=k,
            init="k-means++",
            n_init=20,
            max_iter=300,
            random_state=42
        )
        model.fit(X_scaled)
        inertias.append(model.inertia_)
    
    plt.figure(figsize=(8, 5))
    plt.plot(k_values, inertias, marker="o")
    plt.title(f"Elbow Method - {dataset_name}")
    plt.xlabel("Number of Clusters K")
    plt.ylabel("Inertia")
    plt.grid(True)
    plt.show()


def plot_silhouette_scores(X_scaled, min_k=2, max_k=10, dataset_name="Dataset"):
    """
    Plot silhouette scores for multiple values of K.
    """
    scores = []
    k_values = range(min_k, max_k + 1)
    
    for k in k_values:
        model = KMeans(
            n_clusters=k,
            init="k-means++",
            n_init=20,
            max_iter=300,
            random_state=42
        )
        clusters = model.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, clusters)
        scores.append(score)
    
    plt.figure(figsize=(8, 5))
    plt.plot(k_values, scores, marker="o", color="green")
    plt.title(f"Silhouette Scores - {dataset_name}")
    plt.xlabel("Number of Clusters K")
    plt.ylabel("Silhouette Score")
    plt.grid(True)
    plt.show()


def plot_clusters_2d(X_scaled, clusters, dataset_name="Dataset"):
    """
    Visualize clusters in 2D using PCA.
    """
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap="viridis", s=40)
    plt.title(f"K-Means Clusters Visualized with PCA - {dataset_name}")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.colorbar(label="Cluster")
    plt.grid(True)
    plt.show()
```

---

# Dataset 1: Iris Dataset

## Dataset Description

The Iris dataset contains measurements of iris flowers.

It has:

- 150 samples
- 4 numerical features
- 3 flower species

Features:

1. Sepal length
2. Sepal width
3. Petal length
4. Petal width

Classes:

1. Setosa
2. Versicolor
3. Virginica

Although the dataset has labels, K-Means will not use them during training. The labels are used only for evaluation.

---

## Load Iris Dataset

```python
iris = load_iris()
X_iris, y_iris = create_dataframe(iris)

basic_eda(X_iris, y_iris, dataset_name="Iris Dataset")
```

---

## Scale Iris Dataset

```python
X_iris_scaled, iris_scaler = scale_data(X_iris)
```

Scaling is applied because K-Means depends on distance calculations.

---

## Choose K for Iris Dataset

Since the Iris dataset has 3 known species, we start with:

```python
k_iris = 3
```

We can also inspect the elbow and silhouette plots.

```python
plot_elbow_method(X_iris_scaled, max_k=10, dataset_name="Iris Dataset")
plot_silhouette_scores(X_iris_scaled, min_k=2, max_k=10, dataset_name="Iris Dataset")
```

---

## Train K-Means on Iris Dataset

```python
iris_model, iris_clusters, iris_inertia, iris_silhouette, iris_ari = evaluate_kmeans(
    X_scaled=X_iris_scaled,
    y_true=y_iris,
    k=3,
    dataset_name="Iris Dataset"
)

plot_clusters_2d(X_iris_scaled, iris_clusters, dataset_name="Iris Dataset")
```

---

# Dataset 2: Wine Dataset

## Dataset Description

The Wine dataset contains chemical analysis results of wines grown in the same region in Italy.

It has:

- 178 samples
- 13 numerical features
- 3 wine classes

Some features include:

- Alcohol
- Malic acid
- Ash
- Magnesium
- Flavanoids
- Color intensity
- Proline

This dataset requires scaling because different features have very different ranges.

For example, `proline` has values much larger than features such as `ash` or `hue`.

---

## Load Wine Dataset

```python
wine = load_wine()
X_wine, y_wine = create_dataframe(wine)

basic_eda(X_wine, y_wine, dataset_name="Wine Dataset")
```

---

## Scale Wine Dataset

```python
X_wine_scaled, wine_scaler = scale_data(X_wine)
```

Scaling is very important for the Wine dataset because feature ranges differ significantly.

---

## Choose K for Wine Dataset

The dataset has 3 known wine classes, so we use:

```python
k_wine = 3
```

We can also inspect elbow and silhouette plots.

```python
plot_elbow_method(X_wine_scaled, max_k=10, dataset_name="Wine Dataset")
plot_silhouette_scores(X_wine_scaled, min_k=2, max_k=10, dataset_name="Wine Dataset")
```

---

## Train K-Means on Wine Dataset

```python
wine_model, wine_clusters, wine_inertia, wine_silhouette, wine_ari = evaluate_kmeans(
    X_scaled=X_wine_scaled,
    y_true=y_wine,
    k=3,
    dataset_name="Wine Dataset"
)

plot_clusters_2d(X_wine_scaled, wine_clusters, dataset_name="Wine Dataset")
```

---

# Dataset 3: Digits Dataset

## Dataset Description

The Digits dataset contains handwritten digit images.

It has:

- 1,797 samples
- 64 numerical features
- 10 digit classes

Each image is an 8 x 8 grayscale image.

The 64 features represent pixel intensity values.

Classes:

```text
0, 1, 2, 3, 4, 5, 6, 7, 8, 9
```

K-Means will attempt to group handwritten digits into 10 clusters.

---

## Load Digits Dataset

```python
digits = load_digits()

X_digits = pd.DataFrame(
    digits.data,
    columns=[f"pixel_{i}" for i in range(digits.data.shape[1])]
)

y_digits = pd.Series(digits.target, name="target")

basic_eda(X_digits, y_digits, dataset_name="Digits Dataset")
```

---

## Display Sample Digit Images

```python
fig, axes = plt.subplots(2, 5, figsize=(10, 5))

for ax, image, label in zip(axes.ravel(), digits.images[:10], digits.target[:10]):
    ax.imshow(image, cmap="gray")
    ax.set_title(f"Digit: {label}")
    ax.axis("off")

plt.tight_layout()
plt.show()
```

---

## Scale Digits Dataset

```python
X_digits_scaled, digits_scaler = scale_data(X_digits)
```

Although pixel values are in a limited range, scaling can still help normalize the feature space.

---

## Choose K for Digits Dataset

The dataset has 10 digit classes, so we use:

```python
k_digits = 10
```

We can also inspect elbow and silhouette plots.

```python
plot_elbow_method(X_digits_scaled, max_k=15, dataset_name="Digits Dataset")
plot_silhouette_scores(X_digits_scaled, min_k=2, max_k=15, dataset_name="Digits Dataset")
```

---

## Train K-Means on Digits Dataset

```python
digits_model, digits_clusters, digits_inertia, digits_silhouette, digits_ari = evaluate_kmeans(
    X_scaled=X_digits_scaled,
    y_true=y_digits,
    k=10,
    dataset_name="Digits Dataset"
)

plot_clusters_2d(X_digits_scaled, digits_clusters, dataset_name="Digits Dataset")
```

---

# Combined Results

The following code creates a combined results table.

```python
results = pd.DataFrame({
    "Dataset": ["Iris", "Wine", "Digits"],
    "Samples": [X_iris.shape[0], X_wine.shape[0], X_digits.shape[0]],
    "Features": [X_iris.shape[1], X_wine.shape[1], X_digits.shape[1]],
    "K": [3, 3, 10],
    "Inertia": [iris_inertia, wine_inertia, digits_inertia],
    "Silhouette Score": [iris_silhouette, wine_silhouette, digits_silhouette],
    "Adjusted Rand Index": [iris_ari, wine_ari, digits_ari]
})

results
```

---

# Results

A typical output may look similar to the table below.

Exact values may vary slightly depending on the installed Scikit-learn version and random initialization settings.

| Dataset | Samples | Features | K | Inertia | Silhouette Score | Adjusted Rand Index |
|---|---:|---:|---:|---:|---:|---:|
| Iris | 150 | 4 | 3 | Around 140 | Around 0.45 | Around 0.60 |
| Wine | 178 | 13 | 3 | Around 1280 | Around 0.28 | Around 0.85 to 0.90 |
| Digits | 1797 | 64 | 10 | Depends on scaling | Around 0.10 to 0.20 | Around 0.45 to 0.65 |

---

## Interpretation of Iris Results

The Iris dataset usually produces fairly good clusters.

This is because one class, Setosa, is clearly separated from the other two species.

However, Versicolor and Virginica overlap more, so K-Means may confuse some samples between those two classes.

Expected observations:

- Good separation for Setosa
- Moderate confusion between Versicolor and Virginica
- Reasonable silhouette score
- Moderate to good Adjusted Rand Index

---

## Interpretation of Wine Results

The Wine dataset often performs well after scaling.

Scaling is very important here because the features have very different ranges.

Expected observations:

- K-Means can discover meaningful wine groups
- Adjusted Rand Index is usually strong
- Silhouette score may be moderate because the data is high-dimensional
- PCA visualization usually shows some separation between groups

---

## Interpretation of Digits Results

The Digits dataset is more difficult for K-Means.

Reasons:

- High-dimensional data
- Handwritten digits can vary greatly
- Different digits may look similar
- Same digit can be written in many different styles
- K-Means assumes spherical clusters, which may not fit image data well

Expected observations:

- Some digits cluster well, such as 0 or 6
- Similar-looking digits may be confused, such as 1 and 7 or 3 and 8
- Silhouette score is usually low
- PCA visualization may show overlapping clusters

---

# Full Combined Script

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris, load_wine, load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.decomposition import PCA


def create_dataframe(dataset):
    X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    y = pd.Series(dataset.target, name="target")
    return X, y


def basic_eda(X, y=None, dataset_name="Dataset"):
    print("=" * 80)
    print(f"EDA for {dataset_name}")
    print("=" * 80)
    
    print("\nShape:")
    print(X.shape)
    
    print("\nFirst 5 Rows:")
    print(X.head())
    
    print("\nData Types:")
    print(X.dtypes)
    
    print("\nMissing Values:")
    print(X.isnull().sum())
    
    print("\nStatistical Summary:")
    print(X.describe())
    
    if y is not None:
        print("\nTarget Distribution:")
        print(y.value_counts().sort_index())


def scale_data(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


def evaluate_kmeans(X_scaled, y_true, k, dataset_name):
    model = KMeans(
        n_clusters=k,
        init="k-means++",
        n_init=20,
        max_iter=300,
        random_state=42
    )
    
    clusters = model.fit_predict(X_scaled)
    
    inertia = model.inertia_
    silhouette = silhouette_score(X_scaled, clusters)
    ari = adjusted_rand_score(y_true, clusters) if y_true is not None else None
    
    print("=" * 80)
    print(f"K-Means Results for {dataset_name}")
    print("=" * 80)
    print(f"Number of clusters: {k}")
    print(f"Inertia: {inertia:.4f}")
    print(f"Silhouette Score: {silhouette:.4f}")
    
    if ari is not None:
        print(f"Adjusted Rand Index: {ari:.4f}")
    
    print("\nCluster Counts:")
    print(pd.Series(clusters).value_counts().sort_index())
    
    if y_true is not None:
        print("\nCluster vs Actual Class Table:")
        print(pd.crosstab(pd.Series(y_true, name="Actual"), pd.Series(clusters, name="Cluster")))
    
    return model, clusters, inertia, silhouette, ari


def plot_elbow_method(X_scaled, max_k=10, dataset_name="Dataset"):
    inertias = []
    k_values = range(1, max_k + 1)
    
    for k in k_values:
        model = KMeans(
            n_clusters=k,
            init="k-means++",
            n_init=20,
            max_iter=300,
            random_state=42
        )
        model.fit(X_scaled)
        inertias.append(model.inertia_)
    
    plt.figure(figsize=(8, 5))
    plt.plot(k_values, inertias, marker="o")
    plt.title(f"Elbow Method - {dataset_name}")
    plt.xlabel("Number of Clusters K")
    plt.ylabel("Inertia")
    plt.grid(True)
    plt.show()


def plot_silhouette_scores(X_scaled, min_k=2, max_k=10, dataset_name="Dataset"):
    scores = []
    k_values = range(min_k, max_k + 1)
    
    for k in k_values:
        model = KMeans(
            n_clusters=k,
            init="k-means++",
            n_init=20,
            max_iter=300,
            random_state=42
        )
        clusters = model.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, clusters)
        scores.append(score)
    
    plt.figure(figsize=(8, 5))
    plt.plot(k_values, scores, marker="o", color="green")
    plt.title(f"Silhouette Scores - {dataset_name}")
    plt.xlabel("Number of Clusters K")
    plt.ylabel("Silhouette Score")
    plt.grid(True)
    plt.show()


def plot_clusters_2d(X_scaled, clusters, dataset_name="Dataset"):
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap="viridis", s=40)
    plt.title(f"K-Means Clusters Visualized with PCA - {dataset_name}")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.colorbar(label="Cluster")
    plt.grid(True)
    plt.show()


# ---------------------------------------------------------
# Iris Dataset
# ---------------------------------------------------------

iris = load_iris()
X_iris, y_iris = create_dataframe(iris)

basic_eda(X_iris, y_iris, dataset_name="Iris Dataset")

X_iris_scaled, iris_scaler = scale_data(X_iris)

plot_elbow_method(X_iris_scaled, max_k=10, dataset_name="Iris Dataset")
plot_silhouette_scores(X_iris_scaled, min_k=2, max_k=10, dataset_name="Iris Dataset")

iris_model, iris_clusters, iris_inertia, iris_silhouette, iris_ari = evaluate_kmeans(
    X_scaled=X_iris_scaled,
    y_true=y_iris,
    k=3,
    dataset_name="Iris Dataset"
)

plot_clusters_2d(X_iris_scaled, iris_clusters, dataset_name="Iris Dataset")


# ---------------------------------------------------------
# Wine Dataset
# ---------------------------------------------------------

wine = load_wine()
X_wine, y_wine = create_dataframe(wine)

basic_eda(X_wine, y_wine, dataset_name="Wine Dataset")

X_wine_scaled, wine_scaler = scale_data(X_wine)

plot_elbow_method(X_wine_scaled, max_k=10, dataset_name="Wine Dataset")
plot_silhouette_scores(X_wine_scaled, min_k=2, max_k=10, dataset_name="Wine Dataset")

wine_model, wine_clusters, wine_inertia, wine_silhouette, wine_ari = evaluate_kmeans(
    X_scaled=X_wine_scaled,
    y_true=y_wine,
    k=3,
    dataset_name="Wine Dataset"
)

plot_clusters_2d(X_wine_scaled, wine_clusters, dataset_name="Wine Dataset")


# ---------------------------------------------------------
# Digits Dataset
# ---------------------------------------------------------

digits = load_digits()

X_digits = pd.DataFrame(
    digits.data,
    columns=[f"pixel_{i}" for i in range(digits.data.shape[1])]
)

y_digits = pd.Series(digits.target, name="target")

basic_eda(X_digits, y_digits, dataset_name="Digits Dataset")

fig, axes = plt.subplots(2, 5, figsize=(10, 5))

for ax, image, label in zip(axes.ravel(), digits.images[:10], digits.target[:10]):
    ax.imshow(image, cmap="gray")
    ax.set_title(f"Digit: {label}")
    ax.axis("off")

plt.tight_layout()
plt.show()

X_digits_scaled, digits_scaler = scale_data(X_digits)

plot_elbow_method(X_digits_scaled, max_k=15, dataset_name="Digits Dataset")
plot_silhouette_scores(X_digits_scaled, min_k=2, max_k=15, dataset_name="Digits Dataset")

digits_model, digits_clusters, digits_inertia, digits_silhouette, digits_ari = evaluate_kmeans(
    X_scaled=X_digits_scaled,
    y_true=y_digits,
    k=10,
    dataset_name="Digits Dataset"
)

plot_clusters_2d(X_digits_scaled, digits_clusters, dataset_name="Digits Dataset")


# ---------------------------------------------------------
# Combined Results
# ---------------------------------------------------------

results = pd.DataFrame({
    "Dataset": ["Iris", "Wine", "Digits"],
    "Samples": [X_iris.shape[0], X_wine.shape[0], X_digits.shape[0]],
    "Features": [X_iris.shape[1], X_wine.shape[1], X_digits.shape[1]],
    "K": [3, 3, 10],
    "Inertia": [iris_inertia, wine_inertia, digits_inertia],
    "Silhouette Score": [iris_silhouette, wine_silhouette, digits_silhouette],
    "Adjusted Rand Index": [iris_ari, wine_ari, digits_ari]
})

print("\nFinal Combined Results:")
print(results)
```

---

# Conclusion

K-Means Clustering is a simple, efficient, and widely used unsupervised learning algorithm.

It groups data points by minimizing the distance between each point and its assigned cluster centroid.

In this documentation, we applied K-Means to three datasets:

1. Iris Dataset
2. Wine Dataset
3. Digits Dataset

Key observations:

- K-Means performs well when clusters are compact and well-separated.
- Feature scaling is very important because K-Means is distance-based.
- The Iris and Wine datasets produce relatively meaningful clusters.
- The Digits dataset is more challenging because image data is high-dimensional and complex.
- The choice of `K` strongly affects clustering quality.
- Elbow Method and Silhouette Score are useful for selecting `K`.
- K-Means is fast and interpretable, but it is sensitive to initialization, outliers, and cluster shape.

K-Means is a strong baseline clustering algorithm and is often the first method tried when exploring unlabeled numerical data.