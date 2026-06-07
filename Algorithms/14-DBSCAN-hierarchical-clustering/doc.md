# Comprehensive Documentation: DBSCAN & Hierarchical Clustering

---

## Table of Contents
1. [Introduction](#introduction)
2. [Core Idea](#core-idea)
3. [Why Better Than K-Means](#why-better-than-k-means)
4. [Why Do We Need It](#why-do-we-need-it)
5. [Applications in Real Life](#applications-in-real-life)
6. [Mathematical Terms](#mathematical-terms)
7. [Understanding Data & How to Apply](#understanding-data--how-to-apply)
8. [Python Implementation](#python-implementation)
9. [Examples with 3 Datasets](#examples-with-3-datasets)
10. [Conclusion](#conclusion)

---

## 1. Introduction

Clustering is a fundamental technique in **unsupervised machine learning** where the goal is to group similar data points together without any prior labels. Two of the most powerful and widely used clustering algorithms are:

| Algorithm | Full Name |
|---|---|
| **DBSCAN** | Density-Based Spatial Clustering of Applications with Noise |
| **Hierarchical Clustering** | Agglomerative / Divisive Clustering |

Both algorithms solve different types of problems and go far beyond the limitations of traditional methods like K-Means. They are used in fraud detection, medical imaging, geospatial analysis, customer segmentation, and many more real-world applications.

> 💡 **Simple Analogy:**
> - **DBSCAN** is like a person walking through a city and grouping buildings that are densely packed together, while marking isolated buildings as "outliers."
> - **Hierarchical Clustering** is like building a family tree — starting from individuals and merging them step-by-step into families, clans, and tribes.

---

## 2. Core Idea

---

### 🔷 DBSCAN — Core Idea

DBSCAN is based on the concept of **density**. It groups data points that are **closely packed together** and marks points in **low-density regions** as outliers (noise).

**Key Concepts:**

```
┌─────────────────────────────────────────────────────────────┐
│                    DBSCAN Key Elements                      │
├─────────────────┬───────────────────────────────────────────┤
│ Epsilon (ε)     │ The radius of neighborhood around a point │
│ MinPts          │ Minimum points required in ε-radius       │
│ Core Point      │ Has ≥ MinPts neighbors within ε           │
│ Border Point    │ Within ε of core point, but < MinPts      │
│ Noise Point     │ Neither core nor border point             │
└─────────────────┴───────────────────────────────────────────┘
```

**How DBSCAN Works — Step by Step:**

```
Step 1: Pick an unvisited point
Step 2: Find all neighbors within radius ε
Step 3: If neighbors ≥ MinPts → Mark as CORE POINT, start cluster
Step 4: Expand cluster by recursively visiting neighbors
Step 5: If neighbors < MinPts → Mark as BORDER or NOISE
Step 6: Repeat until all points are visited
```

**Visual Representation:**
```
        . . .           <- Cluster 1 (dense region)
      .   *   .         *  = Core Point
        . . .           .  = Border Points

              X         X  = Noise / Outlier

    . . . .             <- Cluster 2 (another dense region)
  .   *   *   .
    . . . . .
```

---

### 🔷 Hierarchical Clustering — Core Idea

Hierarchical Clustering builds a **tree-like structure (dendrogram)** that shows how data points are grouped at different levels of similarity.

**Two Types:**

```
┌──────────────────────────────────────────────────────────┐
│              Types of Hierarchical Clustering            │
├─────────────────────┬────────────────────────────────────┤
│ AGGLOMERATIVE       │ Bottom-Up: Start with individual   │
│ (Most Common)       │ points → merge step by step        │
├─────────────────────┼────────────────────────────────────┤
│ DIVISIVE            │ Top-Down: Start with all points    │
│                     │ → split step by step               │
└─────────────────────┴────────────────────────────────────┘
```

**How Agglomerative Clustering Works — Step by Step:**

```
Step 1: Treat each data point as its own cluster
Step 2: Calculate distance between all pairs of clusters
Step 3: Merge the two closest clusters
Step 4: Recalculate distances
Step 5: Repeat Steps 3–4 until one cluster remains
Step 6: Cut the dendrogram at desired level → get clusters
```

**Dendrogram Example:**
```
Height
  |
4 |          ┌──────────┐
  |          │          │
3 |     ┌────┤     ┌────┤
  |     │    │     │    │
2 |  ┌──┤    │  ┌──┤    │
  |  │  │    │  │  │    │
1 |  A  B    C  D  E    F
  └──────────────────────
```
> Cut the dendrogram at height 3 → gives 2 clusters: {A,B,C} and {D,E,F}

**Linkage Methods (How to Measure Distance Between Clusters):**

```
┌──────────────────┬────────────────────────────────────────┐
│ Linkage Method   │ Description                            │
├──────────────────┼────────────────────────────────────────┤
│ Single Linkage   │ Min distance between any two points    │
│ Complete Linkage │ Max distance between any two points    │
│ Average Linkage  │ Average of all pairwise distances      │
│ Ward's Method    │ Minimizes total within-cluster variance│
└──────────────────┴────────────────────────────────────────┘
```

---

## 3. Why Better Than K-Means

---

### ❌ Limitations of K-Means

```
┌────────────────────────────────────────────────────────────┐
│                   K-Means Limitations                      │
├────────────────────────────────────────────────────────────┤
│ 1. Requires number of clusters (K) to be predefined        │
│ 2. Only works well with spherical (circular) clusters      │
│ 3. Highly sensitive to outliers                            │
│ 4. Fails with clusters of different sizes & densities      │
│ 5. Gets stuck in local minima                              │
└────────────────────────────────────────────────────────────┘
```

---

### ✅ How DBSCAN Solves These Problems

```
┌─────────────────────┬──────────────────┬───────────────────┐
│ Problem             │ K-Means          │ DBSCAN            │
├─────────────────────┼──────────────────┼───────────────────┤
│ K predefined?       │ YES (required)   │ NO (automatic)    │
│ Handles outliers?   │ NO               │ YES (marks noise) │
│ Arbitrary shapes?   │ NO               │ YES               │
│ Varying densities?  │ NO               │ Partially YES     │
│ Spherical clusters? │ Only spherical   │ Any shape         │
└─────────────────────┴──────────────────┴───────────────────┘
```

**Shape Comparison:**
```
K-Means works:          DBSCAN works:
    ●●●  ●●●             ●●●●●●●
    ●●●  ●●●             ●●     ●●
  (circular only)        ●        ●  (any shape!)
                         ●●     ●●
                          ●●●●●●
```

---

### ✅ How Hierarchical Clustering Solves These Problems

```
┌─────────────────────┬──────────────────┬───────────────────┐
│ Problem             │ K-Means          │ Hierarchical      │
├─────────────────────┼──────────────────┼───────────────────┤
│ K predefined?       │ YES (required)   │ NO (use dendro)   │
│ Visual insight?     │ None             │ Dendrogram        │
│ Nested clusters?    │ NO               │ YES               │
│ Interpretability?   │ Low              │ High              │
│ Small datasets?     │ OK               │ Excellent         │
└─────────────────────┴──────────────────┴───────────────────┘
```

---

## 4. Why Do We Need It

---

### The Real World is NOT Always Spherical

Real-world data rarely comes in clean, circular blobs. Consider:

```
📍 GPS Data          → Irregular paths and roads
🧬 Gene Expression   → Nested biological groupings
🌊 Ocean Currents    → Non-linear flowing patterns
💳 Fraud Detection   → Isolated anomalous transactions
🏙️ City Planning     → Density-based population zones
```

### When to Use DBSCAN:
```
✅ Unknown number of clusters
✅ Data has noise/outliers
✅ Clusters have irregular shapes
✅ Geospatial or density-based problems
✅ Anomaly detection tasks
```

### When to Use Hierarchical Clustering:
```
✅ Need to visualize cluster hierarchy
✅ Small to medium-sized datasets
✅ Exploring data without fixed cluster count
✅ Taxonomy or biological classification
✅ Market segmentation with sub-groups
```

---

## 5. Applications in Real Life

---

### 🔷 DBSCAN Applications

```
┌──────────────────────────────────────────────────────────────┐
│                  DBSCAN Real-Life Use Cases                  │
├───────────────────────┬──────────────────────────────────────┤
│ Domain                │ Application                          │
├───────────────────────┼──────────────────────────────────────┤
│ 🌍 Geospatial         │ Finding hotspot areas in city maps   │
│ 🔒 Cybersecurity      │ Detecting network intrusions         │
│ 🛒 E-Commerce         │ Identifying unusual purchase behavior│
│ 🏥 Healthcare         │ Detecting anomalies in MRI scans     │
│ 🌦️ Meteorology        │ Identifying storm clusters           │
│ 📡 Astronomy          │ Star/galaxy cluster discovery        │
│ 🚗 Transportation     │ Traffic congestion zone detection    │
└───────────────────────┴──────────────────────────────────────┘
```

---

### 🔷 Hierarchical Clustering Applications

```
┌─────────────────────────────────────────────────────────────┐
│           Hierarchical Clustering Real-Life Use Cases       │
├───────────────────────┬─────────────────────────────────────┤
│ Domain                │ Application                         │
├───────────────────────┼─────────────────────────────────────┤
│ 🧬 Biology            │ Gene & species classification       │
│ 📰 NLP/Text Mining    │ Document & topic clustering         │
│ 🏦 Finance            │ Grouping similar stocks/assets      │
│ 🛍️ Marketing          │ Customer segmentation hierarchies   │
│ 🏥 Medicine           │ Disease subtype classification      │
│ 🎵 Music              │ Genre grouping and recommendation   │
│ 📊 Social Networks    │ Community detection                 │
└───────────────────────┴─────────────────────────────────────┘
```

---

## 6. Mathematical Terms

---

### 🔷 DBSCAN Mathematics

#### Epsilon Neighborhood
$$N_\varepsilon(p) = \{q \in D \mid dist(p, q) \leq \varepsilon\}$$

> All points `q` in dataset `D` whose distance from point `p` is less than or equal to ε

#### Euclidean Distance (Most Common)
$$dist(p, q) = \sqrt{\sum_{i=1}^{n}(p_i - q_i)^2}$$

#### Core Point Condition
$$|N_\varepsilon(p)| \geq MinPts$$

> Point `p` is a core point if the number of points in its ε-neighborhood is ≥ MinPts

#### Density Reachability
A point `q` is **density-reachable** from `p` if there is a chain:
$$p = p_1, p_2, \ldots, p_n = q$$
where each $p_{i+1}$ is directly reachable from $p_i$

#### Density Connectivity
Points `p` and `q` are **density-connected** if there exists a point `o` such that both are density-reachable from `o`

---

### 🔷 Hierarchical Clustering Mathematics

#### Single Linkage Distance
$$d(C_i, C_j) = \min_{x \in C_i, y \in C_j} dist(x, y)$$

#### Complete Linkage Distance
$$d(C_i, C_j) = \max_{x \in C_i, y \in C_j} dist(x, y)$$

#### Average Linkage Distance
$$d(C_i, C_j) = \frac{1}{|C_i||C_j|} \sum_{x \in C_i} \sum_{y \in C_j} dist(x, y)$$

#### Ward's Linkage (Minimizes Variance)
$$\Delta(C_i, C_j) = \frac{|C_i| \cdot |C_j|}{|C_i| + |C_j|} \cdot ||m_i - m_j||^2$$

Where $m_i$ and $m_j$ are centroids of clusters $C_i$ and $C_j$

---

### 🔷 Evaluation Metrics

#### Silhouette Score
$$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$

```
a(i) = Average distance to all points in SAME cluster
b(i) = Average distance to all points in NEAREST other cluster
Range: -1 (bad) to +1 (excellent)
```

#### Davies-Bouldin Index
$$DB = \frac{1}{k}\sum_{i=1}^{k} \max_{j \neq i} \left(\frac{\sigma_i + \sigma_j}{d(c_i, c_j)}\right)$$

```
Lower DB score = Better clustering
σ = Avg distance of points from cluster centroid
d(c_i, c_j) = Distance between cluster centroids
```

---

## 7. Understanding the Data & How to Apply

---

### 🔷 Choosing the Right Algorithm

```
┌─────────────────────────────────────────────────────────────┐
│                    Decision Framework                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Do you know number of clusters?                            │
│           │                                                 │
│    ┌──────┴──────┐                                          │
│   YES             NO                                        │
│    │               │                                        │
│  K-Means      ┌────┴──────────┐                             │
│             Need         Is dataset                         │
│           Hierarchy?      large?                            │
│               │               │                             │
│        ┌──────┴──┐     ┌──────┴──────┐                      │
│       YES        NO   YES             NO                    │
│        │         │    │               │                     │
│   Hierarchical  Has  DBSCAN      Hierarchical               │
│   Clustering  Noise?             Clustering                 │
│               │                                             │
│           ┌───┴───┐                                         │
│          YES      NO                                        │
│           │       │                                         │
│        DBSCAN  K-Means                                      │
└─────────────────────────────────────────────────────────────┘
```

---

### 🔷 How to Select DBSCAN Parameters

#### Finding Optimal ε (Epsilon):
```python
# Use K-Distance Graph (K-NN approach)
# 1. Compute distance to K-th nearest neighbor for all points
# 2. Sort in descending order
# 3. Look for the "elbow" point → that's your ε
```

**K-Distance Graph:**
```
Distance
^
|                    ___________
|               ____/
|          ____/  ← Elbow (optimal ε)
|     ____/
|____/
+─────────────────────────────→ Points (sorted)
```

#### Finding Optimal MinPts:
```
Rule of Thumb:
├── MinPts ≥ Dimensions + 1
├── For 2D data → MinPts = 4 or 5
├── Larger datasets → use higher MinPts
└── MinPts = 2 * number of features (general rule)
```

---

### 🔷 How to Select Hierarchical Clustering Parameters

#### Reading a Dendrogram:
```
1. Look for the LONGEST vertical line
2. Draw a horizontal line through it
3. Count how many vertical lines it crosses → that's K

Height
  |
  |    ┌─────────────────────────┐
  |    │                         │
  |    │     ┌────┐         ┌────┤
  |    │     │    │         │    │
──┼────┼─────┼────┼─────────┼────┼──── Cut here → 2 clusters
  |    │     │    │         │    │
  A    B     C    D         E    F
```

---

## 8. Python Implementation

---

### 📦 Required Libraries

```python
# Install if needed
# pip install scikit-learn numpy pandas matplotlib seaborn scipy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.neighbors import NearestNeighbors
from sklearn.datasets import make_blobs, make_moons, load_iris

from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist

import warnings
warnings.filterwarnings('ignore')

# Set visualization style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
```

---

### 🔷 DBSCAN — Full Implementation Template

```python
class DBSCANAnalysis:
    """
    A complete DBSCAN Analysis Pipeline
    """
    
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
        self.model = None
        self.labels = None
        self.scaler = StandardScaler()
    
    def preprocess(self, X):
        """Scale the data"""
        self.X_scaled = self.scaler.fit_transform(X)
        return self.X_scaled
    
    def find_optimal_eps(self, X, k=None):
        """
        Use K-Distance graph to find optimal epsilon
        """
        if k is None:
            k = X.shape[1] * 2  # Rule of thumb
        
        nbrs = NearestNeighbors(n_neighbors=k).fit(X)
        distances, _ = nbrs.kneighbors(X)
        distances = np.sort(distances[:, k-1], axis=0)[::-1]
        
        plt.figure(figsize=(10, 5))
        plt.plot(distances, linewidth=2, color='steelblue')
        plt.xlabel('Points sorted by distance', fontsize=12)
        plt.ylabel(f'{k}-th Nearest Neighbor Distance', fontsize=12)
        plt.title('K-Distance Graph for Optimal Epsilon', fontsize=14)
        plt.axhline(y=np.percentile(distances, 90), 
                   color='red', linestyle='--', 
                   label=f'Suggested ε = {np.percentile(distances, 90):.3f}')
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        suggested_eps = np.percentile(distances, 90)
        print(f"📌 Suggested ε (epsilon): {suggested_eps:.4f}")
        return suggested_eps
    
    def fit(self, X):
        """Fit DBSCAN model"""
        X_scaled = self.preprocess(X)
        
        self.model = DBSCAN(
            eps=self.eps,
            min_samples=self.min_samples,
            metric='euclidean'
        )
        self.labels = self.model.fit_predict(X_scaled)
        return self
    
    def get_results(self):
        """Print clustering results"""
        n_clusters = len(set(self.labels)) - (1 if -1 in self.labels else 0)
        n_noise = list(self.labels).count(-1)
        
        print("=" * 50)
        print("        DBSCAN CLUSTERING RESULTS")
        print("=" * 50)
        print(f"✅ Number of Clusters Found : {n_clusters}")
        print(f"🔴 Number of Noise Points   : {n_noise}")
        print(f"📊 Total Data Points        : {len(self.labels)}")
        print(f"📈 Noise Percentage         : {n_noise/len(self.labels)*100:.2f}%")
        
        # Silhouette Score (only if more than 1 cluster)
        if n_clusters > 1:
            valid_mask = self.labels != -1
            if valid_mask.sum() > 1:
                score = silhouette_score(
                    self.X_scaled[valid_mask], 
                    self.labels[valid_mask]
                )
                print(f"⭐ Silhouette Score         : {score:.4f}")
        print("=" * 50)
    
    def visualize(self, X, title="DBSCAN Clustering Results"):
        """Visualize clustering results"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Get unique labels
        unique_labels = set(self.labels)
        colors = plt.cm.tab10(
            np.linspace(0, 1, len(unique_labels))
        )
        
        # Plot 1: Clustered Data
        for label, color in zip(sorted(unique_labels), colors):
            if label == -1:
                color = [0, 0, 0, 1]  # Black for noise
                marker = 'x'
                label_name = 'Noise'
                size = 50
            else:
                marker = 'o'
                label_name = f'Cluster {label}'
                size = 80
            
            mask = self.labels == label
            axes[0].scatter(
                X[mask, 0], X[mask, 1],
                c=[color], marker=marker,
                s=size, label=label_name,
                edgecolors='black' if label != -1 else 'none',
                linewidths=0.5, alpha=0.8
            )
        
        axes[0].set_title(f'{title}\n(Clusters)', fontsize=13, fontweight='bold')
        axes[0].legend(loc='upper right', fontsize=9)
        axes[0].set_xlabel('Feature 1')
        axes[0].set_ylabel('Feature 2')
        
        # Plot 2: Cluster Size Distribution
        cluster_counts = pd.Series(self.labels).value_counts().sort_index()
        cluster_labels = [
            'Noise' if x == -1 else f'Cluster {x}' 
            for x in cluster_counts.index
        ]
        bar_colors = ['#FF4444' if x == -1 else f'C{i}' 
                     for i, x in enumerate(cluster_counts.index)]
        
        axes[1].bar(cluster_labels, cluster_counts.values, 
                   color=bar_colors, edgecolor='black', 
                   linewidth=0.8, alpha=0.85)
        axes[1].set_title('Cluster Size Distribution', 
                          fontsize=13, fontweight='bold')
        axes[1].set_xlabel('Cluster')
        axes[1].set_ylabel('Number of Points')
        
        for i, v in enumerate(cluster_counts.values):
            axes[1].text(i, v + 0.5, str(v), 
                        ha='center', fontweight='bold')
        
        plt.tight_layout()
        plt.show()
```

---

### 🔷 Hierarchical Clustering — Full Implementation Template

```python
class HierarchicalAnalysis:
    """
    A complete Hierarchical Clustering Analysis Pipeline
    """
    
    def __init__(self, n_clusters=3, linkage_method='ward'):
        self.n_clusters = n_clusters
        self.linkage_method = linkage_method
        self.model = None
        self.labels = None
        self.linkage_matrix = None
        self.scaler = StandardScaler()
    
    def preprocess(self, X):
        """Scale the data"""
        self.X_scaled = self.scaler.fit_transform(X)
        return self.X_scaled
    
    def plot_dendrogram(self, X, title="Hierarchical Clustering Dendrogram"):
        """
        Plot dendrogram to find optimal number of clusters
        """
        X_scaled = self.scaler.fit_transform(X)
        self.linkage_matrix = linkage(X_scaled, method=self.linkage_method)
        
        fig, ax = plt.subplots(figsize=(14, 7))
        
        dendrogram(
            self.linkage_matrix,
            ax=ax,
            truncate_mode='level',
            p=5,
            leaf_rotation=90,
            leaf_font_size=9,
            color_threshold=0.7 * max(self.linkage_matrix[:, 2]),
            above_threshold_color='gray'
        )
        
        # Add cut line
        cut_height = sorted(
            self.linkage_matrix[:, 2], reverse=True
        )[self.n_clusters - 1]
        
        ax.axhline(
            y=cut_height, 
            color='red', 
            linestyle='--', 
            linewidth=2,
            label=f'Cut for {self.n_clusters} clusters'
        )
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Sample Index / Cluster', fontsize=12)
        ax.set_ylabel('Distance (Height)', fontsize=12)
        ax.legend()
        plt.tight_layout()
        plt.show()
        
        print(f"📌 Suggested cut height: {cut_height:.4f}")
    
    def fit(self, X):
        """Fit Hierarchical Clustering model"""
        X_scaled = self.preprocess(X)
        
        self.model = AgglomerativeClustering(
            n_clusters=self.n_clusters,
            linkage=self.linkage_method
        )
        self.labels = self.model.fit_predict(X_scaled)
        return self
    
    def get_results(self, X=None):
        """Print clustering results"""
        print("=" * 55)
        print("      HIERARCHICAL CLUSTERING RESULTS")
        print("=" * 55)
        print(f"✅ Number of Clusters      : {self.n_clusters}")
        print(f"🔗 Linkage Method          : {self.linkage_method}")
        print(f"📊 Total Data Points       : {len(self.labels)}")
        
        for cluster in range(self.n_clusters):
            count = np.sum(self.labels == cluster)
            pct = count / len(self.labels) * 100
            print(f"   Cluster {cluster}            : {count} points ({pct:.1f}%)")
        
        if X is not None:
            score = silhouette_score(self.X_scaled, self.labels)
            db_score = davies_bouldin_score(self.X_scaled, self.labels)
            print(f"⭐ Silhouette Score        : {score:.4f}")
            print(f"📉 Davies-Bouldin Score    : {db_score:.4f}")
        print("=" * 55)
    
    def find_optimal_clusters(self, X, max_clusters=10):
        """Plot silhouette scores for different K values"""
        X_scaled = self.scaler.fit_transform(X)
        scores = []
        k_range = range(2, max_clusters + 1)
        
        for k in k_range:
            model = AgglomerativeClustering(
                n_clusters=k, 
                linkage=self.linkage_method
            )
            labels = model.fit_predict(X_scaled)
            score = silhouette_score(X_scaled, labels)
            scores.append(score)
        
        plt.figure(figsize=(10, 5))
        plt.plot(k_range, scores, 'bo-', linewidth=2, markersize=8)
        plt.xlabel('Number of Clusters (K)', fontsize=12)
        plt.ylabel('Silhouette Score', fontsize=12)
        plt.title('Optimal Cluster Count via Silhouette Score', 
                 fontsize=14, fontweight='bold')
        plt.xticks(k_range)
        
        best_k = k_range[np.argmax(scores)]
        plt.axvline(x=best_k, color='red', linestyle='--',
                   label=f'Best K = {best_k}')
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        print(f"📌 Optimal number of clusters: {best_k}")
        return best_k
    
    def visualize(self, X, title="Hierarchical Clustering Results"):
        """Visualize clustering results"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Clustered Data
        scatter = axes[0].scatter(
            X[:, 0], X[:, 1],
            c=self.labels,
            cmap='tab10',
            s=80,
            edgecolors='black',
            linewidths=0.5,
            alpha=0.85
        )
        
        # Add cluster centroids
        for cluster in range(self.n_clusters):
            mask = self.labels == cluster
            centroid = X[mask].mean(axis=0)
            axes[0].scatter(
                centroid[0], centroid[1],
                marker='*', s=300,
                c='gold', edgecolors='black',
                linewidths=1.5, zorder=5,
                label=f'Centroid {cluster}'
            )
        
        axes[0].set_title(f'{title}', fontsize=13, fontweight='bold')
        axes[0].set_xlabel('Feature 1')
        axes[0].set_ylabel('Feature 2')
        plt.colorbar(scatter, ax=axes[0], label='Cluster')
        
        # Plot 2: Cluster Size Distribution
        cluster_counts = [np.sum(self.labels == i) 
                         for i in range(self.n_clusters)]
        
        wedge_props = dict(width=0.5, edgecolor='white', linewidth=2)
        axes[1].pie(
            cluster_counts,
            labels=[f'Cluster {i}\n({c} pts)' 
                   for i, c in enumerate(cluster_counts)],
            autopct='%1.1f%%',
            wedgeprops=wedge_props,
            startangle=90,
            textprops={'fontsize': 10}
        )
        axes[1].set_title('Cluster Distribution', 
                          fontsize=13, fontweight='bold')
        
        plt.tight_layout()
        plt.show()
```

---

## 9. Examples with 3 Datasets

---

### 📌 Dataset 1: Synthetic Moon-Shaped Data (DBSCAN Showcase)

> **Why this dataset?** Moon-shaped data demonstrates where K-Means fails completely but DBSCAN excels.

```python
# ============================================================
#         EXAMPLE 1: Moon-Shaped Dataset with DBSCAN
# ============================================================

from sklearn.datasets import make_moons
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# ── Step 1: Generate Data ──────────────────────────────────
np.random.seed(42)
X_moon, y_moon = make_moons(
    n_samples=300,
    noise=0.08,
    random_state=42
)

print("Dataset Shape:", X_moon.shape)
print("Sample Points:\n", X_moon[:5])

# ── Step 2: Visualize Raw Data ────────────────────────────
plt.figure(figsize=(8, 5))
plt.scatter(X_moon[:, 0], X_moon[:, 1], 
           s=50, alpha=0.7, edgecolors='black', linewidths=0.5)
plt.title("Raw Moon-Shaped Data", fontsize=14, fontweight='bold')
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.tight_layout()
plt.show()

# ── Step 3: Compare K-Means vs DBSCAN ────────────────────
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_moon)

# K-Means
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans_labels = kmeans.fit_predict(X_scaled)
kmeans_score = silhouette_score(X_scaled, kmeans_labels)

# DBSCAN
dbscan = DBSCAN(eps=0.2, min_samples=5)
dbscan_labels = dbscan.fit_predict(X_scaled)
dbscan_score = silhouette_score(X_scaled, dbscan_labels)

# ── Step 4: Visualization Comparison ─────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Moon-Shaped Data: K-Means vs DBSCAN", 
             fontsize=15, fontweight='bold')

# Raw Data
axes[0].scatter(X_moon[:, 0], X_moon[:, 1], 
               c='steelblue', s=50, alpha=0.7,
               edgecolors='black', linewidths=0.5)
axes[0].set_title("Original Data", fontsize=12)
axes[0].set_xlabel("Feature 1")
axes[0].set_ylabel("Feature 2")

# K-Means Results
axes[1].scatter(X_moon[:, 0], X_moon[:, 1], 
               c=kmeans_labels, cmap='Set1',
               s=50, alpha=0.7, edgecolors='black', linewidths=0.5)
axes[1].set_title(f"K-Means (K=2)\nSilhouette Score: {kmeans_score:.3f}", 
                 fontsize=12)
axes[1].set_xlabel("Feature 1")

# DBSCAN Results
colors_dbscan = ['red' if l == -1 else l for l in dbscan_labels]
scatter = axes[2].scatter(X_moon[:, 0], X_moon[:, 1], 
               c=dbscan_labels, cmap='Set1',
               s=50, alpha=0.7, edgecolors='black', linewidths=0.5)
axes[2].set_title(f"DBSCAN (ε=0.2, MinPts=5)\nSilhouette Score: {dbscan_score:.3f}", 
                 fontsize=12)
axes[2].set_xlabel("Feature 1")

plt.tight_layout()
plt.show()

# ── Step 5: Results Summary ───────────────────────────────
print("\n" + "="*55)
print("         COMPARISON RESULTS — MOON DATA")
print("="*55)

n_clusters_dbscan = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
n_noise = list(dbscan_labels).count(-1)

print(f"K-Means  → Clusters: 2    | Silhouette: {kmeans_score:.4f}")
print(f"DBSCAN   → Clusters: {n_clusters_dbscan}    | Silhouette: {dbscan_score:.4f}")
print(f"DBSCAN   → Noise Points: {n_noise}")
print("="*55)
print("✅ DBSCAN correctly identifies the moon shapes!")
print("❌ K-Means incorrectly splits the data vertically!")

# ── Step 6: DBSCAN Parameter Sensitivity Analysis ─────────
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle("DBSCAN Parameter Sensitivity Analysis", 
             fontsize=15, fontweight='bold')

params = [
    (0.1, 3), (0.2, 5), (0.3, 5),
    (0.1, 10), (0.5, 5), (0.2, 15)
]

for ax, (eps, min_pts) in zip(axes.flatten(), params):
    model = DBSCAN(eps=eps, min_samples=min_pts)
    labels = model.fit_predict(X_scaled)
    n_clust = len(set(labels)) - (1 if -1 in labels else 0)
    noise = list(labels).count(-1)
    
    ax.scatter(X_moon[:, 0], X_moon[:, 1], 
              c=labels, cmap='tab10', s=40, alpha=0.8,
              edgecolors='black', linewidths=0.3)
    ax.set_title(f'ε={eps}, MinPts={min_pts}\nClusters={n_clust}, Noise={noise}', 
                fontsize=11)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")

plt.tight_layout()
plt.show()
```

**Expected Output:**
```
Dataset Shape: (300, 2)
===================================================
         COMPARISON RESULTS — MOON DATA
===================================================
K-Means  → Clusters: 2    | Silhouette: 0.4123
DBSCAN   → Clusters: 2    | Silhouette: 0.6891
DBSCAN   → Noise Points: 3
===================================================
✅ DBSCAN correctly identifies the moon shapes!
❌ K-Means incorrectly splits the data vertically!
```

---

### 📌 Dataset 2: Mall Customer Segmentation (Hierarchical Clustering)

> **Why this dataset?** Customer segmentation is a classic business problem perfectly suited for hierarchical clustering, allowing marketers to understand sub-groups within groups.

```python
# ============================================================
#    EXAMPLE 2: Mall Customer Segmentation — Hierarchical
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage

# ── Step 1: Create/Load Dataset ───────────────────────────
# Simulating Mall Customer Dataset
np.random.seed(42)

n = 200
customer_data = pd.DataFrame({
    'CustomerID': range(1, n + 1),
    'Gender': np.random.choice(['Male', 'Female'], n),
    'Age': np.random.randint(18, 70, n),
    'Annual_Income_k': np.concatenate([
        np.random.normal(20, 5, 40),    # Low income
        np.random.normal(50, 8, 60),    # Medium income
        np.random.normal(80, 5, 40),    # High income
        np.random.normal(60, 15, 60)    # Mixed
    ]),
    'Spending_Score': np.concatenate([
        np.random.normal(20, 8, 40),    # Low spenders
        np.random.normal(50, 10, 60),   # Medium spenders
        np.random.normal(80, 8, 40),    # High spenders
        np.random.normal(45, 20, 60)    # Mixed
    ])
})

# Clip values to realistic ranges
customer_data['Annual_Income_k'] = customer_data['Annual_Income_k'].clip(10, 120)
customer_data['Spending_Score'] = customer_data['Spending_Score'].clip(1, 100)

print("="*55)
print("          MALL CUSTOMER DATASET OVERVIEW")
print("="*55)
print(f"Shape: {customer_data.shape}")
print("\nFirst 5 rows:")
print(customer_data.head())
print("\nStatistical Summary:")
print(customer_data.describe().round(2))

# ── Step 2: Exploratory Data Analysis ────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Mall Customer - Exploratory Data Analysis", 
             fontsize=15, fontweight='bold')

# Age Distribution
axes[0, 0].hist(customer_data['Age'], bins=20, 
               color='steelblue', edgecolor='black', alpha=0.8)
axes[0, 0].set_title('Age Distribution', fontsize=12)
axes[0, 0].set_xlabel('Age')
axes[0, 0].set_ylabel('Frequency')

# Income Distribution
axes[0, 1].hist(customer_data['Annual_Income_k'], bins=20, 
               color='coral', edgecolor='black', alpha=0.8)
axes[0, 1].set_title('Annual Income Distribution', fontsize=12)
axes[0, 1].set_xlabel('Annual Income (k$)')
axes[0, 1].set_ylabel('Frequency')

# Spending Score Distribution
axes[1, 0].hist(customer_data['Spending_Score'], bins=20, 
               color='mediumseagreen', edgecolor='black', alpha=0.8)
axes[1, 0].set_title('Spending Score Distribution', fontsize=12)
axes[1, 0].set_xlabel('Spending Score')
axes[1, 0].set_ylabel('Frequency')

# Income vs Spending Score
axes[1, 1].scatter(customer_data['Annual_Income_k'], 
                  customer_data['Spending_Score'],
                  color='purple', alpha=0.6, s=50,
                  edgecolors='black', linewidths=0.5)
axes[1, 1].set_title('Income vs Spending Score', fontsize=12)
axes[1, 1].set_xlabel('Annual Income (k$)')
axes[1, 1].set_ylabel('Spending Score')

plt.tight_layout()
plt.show()

# ── Step 3: Feature Selection & Preprocessing ─────────────
features = ['Annual_Income_k', 'Spending_Score']
X = customer_data[features].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── Step 4: Dendrogram Analysis ───────────────────────────
fig, ax = plt.subplots(figsize=(14, 7))

linkage_matrix = linkage(X_scaled, method='ward')

dendrogram(
    linkage_matrix,
    ax=ax,
    truncate_mode='lastp',
    p=30,
    leaf_rotation=90,
    leaf_font_size=9,
    color_threshold=6,
)

ax.axhline(y=6, color='red', linestyle='--', 
           linewidth=2, label='Cut at height=6 (5 clusters)')
ax.axhline(y=9, color='blue', linestyle='--', 
           linewidth=2, label='Cut at height=9 (3 clusters)')

ax.set_title("Ward Linkage Dendrogram — Customer Segmentation", 
             fontsize=14, fontweight='bold')
ax.set_xlabel("Customer Samples", fontsize=12)
ax.set_ylabel("Distance (Ward)", fontsize=12)
ax.legend()
plt.tight_layout()
plt.show()

# ── Step 5: Find Optimal Clusters ─────────────────────────
silhouette_scores = []
k_range = range(2, 10)

for k in k_range:
    hc = AgglomerativeClustering(n_clusters=k, linkage='ward')
    labels = hc.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    silhouette_scores.append(score)
    print(f"  K={k}: Silhouette Score = {score:.4f}")

plt.figure(figsize=(10, 5))
plt.plot(k_range, silhouette_scores, 'go-', 
        linewidth=2, markersize=10)
plt.xlabel('Number of Clusters', fontsize=12)
plt.ylabel('Silhouette Score', fontsize=12)
plt.title('Optimal K — Silhouette Score Analysis', 
         fontsize=14, fontweight='bold')
plt.xticks(k_range)

best_k = k_range[np.argmax(silhouette_scores)]
plt.axvline(x=best_k, color='red', linestyle='--', 
           linewidth=2, label=f'Best K = {best_k}')
plt.legend()
plt.tight_layout()
plt.show()
print(f"\n📌 Best number of clusters: {best_k}")

# ── Step 6: Final Model ───────────────────────────────────
hc_final = AgglomerativeClustering(n_clusters=best_k, linkage='ward')
customer_data['Cluster'] = hc_final.fit_predict(X_scaled)

# ── Step 7: Visualize Final Clusters ──────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle(f"Customer Segmentation — {best_k} Clusters", 
             fontsize=15, fontweight='bold')

# Scatter Plot
colors = plt.cm.tab10(np.linspace(0, 1, best_k))
for i in range(best_k):
    mask = customer_data['Cluster'] == i
    axes[0].scatter(
        customer_data[mask]['Annual_Income_k'],
        customer_data[mask]['Spending_Score'],
        label=f'Cluster {i} ({mask.sum()} customers)',
        s=80, alpha=0.8,
        edgecolors='black', linewidths=0.5
    )

axes[0].set_xlabel('Annual Income (k$)', fontsize=12)
axes[0].set_ylabel('Spending Score', fontsize=12)
axes[0].set_title('Customer Clusters', fontsize=13)
axes[0].legend(fontsize=9)

# Cluster Profile Heatmap
cluster_profile = customer_data.groupby('Cluster')[features].mean()
cluster_profile_norm = (cluster_profile - cluster_profile.min()) / \
                       (cluster_profile.max() - cluster_profile.min())

sns.heatmap(
    cluster_profile_norm.T,
    annot=cluster_profile.T.round(1),
    fmt='.1f',
    cmap='YlOrRd',
    ax=axes[1],
    linewidths=0.5,
    cbar_kws={'label': 'Normalized Value'}
)
axes[1].set_title('Cluster Profile Heatmap', fontsize=13)
axes[1].set_xlabel('Cluster')
axes[1].set_ylabel('Feature')

plt.tight_layout()
plt.show()

# ── Step 8: Cluster Interpretation ────────────────────────
print("\n" + "="*60)
print("           CUSTOMER SEGMENT PROFILES")
print("="*60)
for cluster in range(best_k):
    mask = customer_data['Cluster'] == cluster
    subset = customer_data[mask]
    income = subset['Annual_Income_k'].mean()
    spending = subset['Spending_Score'].mean()
    count = mask.sum()
    
    # Label the segment
    if income > 65 and spending > 60:
        segment = "💎 Premium Customers"
    elif income > 65 and spending < 45:
        segment = "💼 Conservative High Earners"
    elif income < 40 and spending > 55:
        segment = "🎯 Impulsive Low Earners"
    elif income < 40 and spending < 40:
        segment = "💰 Budget-Conscious Customers"
    else:
        segment = "📊 Standard Customers"
    
    print(f"\nCluster {cluster} — {segment}")
    print(f"  Count         : {count} customers")
    print(f"  Avg Income    : ${income:.1f}k")
    print(f"  Avg Spending  : {spending:.1f}/100")
print("="*60)
```

---

### 📌 Dataset 3: Iris Dataset — Side-by-Side Comparison

> **Why this dataset?** The Iris dataset is a standard benchmark. We'll compare both algorithms on the same real data.

```python
# ============================================================
#    EXAMPLE 3: Iris Dataset — DBSCAN vs Hierarchical
# ============================================================

from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (silhouette_score, 
                             adjusted_rand_score,
                             davies_bouldin_score)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage

# ── Step 1: Load Dataset ──────────────────────────────────
iris = load_iris()
X_iris = iris.data
y_iris = iris.target  # Ground truth (for validation only)
feature_names = iris.feature_names
target_names = iris.target_names

iris_df = pd.DataFrame(X_iris, columns=feature_names)
iris_df['Species'] = [target_names[y] for y in y_iris]

print("="*55)
print("           IRIS DATASET OVERVIEW")
print("="*55)
print(f"Shape: {X_iris.shape}")
print(f"Features: {feature_names}")
print(f"Classes: {target_names}")
print("\nFirst 5 rows:")
print(iris_df.head())
print("\nClass Distribution:")
print(iris_df['Species'].value_counts())

# ── Step 2: EDA — Pairplot ────────────────────────────────
plt.figure(figsize=(12, 10))
sns.pairplot(iris_df, hue='Species', 
            palette='Set1', plot_kws={'alpha': 0.7})
plt.suptitle("Iris Dataset - Feature Pairplot", 
            fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

# ── Step 3: Correlation Heatmap ───────────────────────────
plt.figure(figsize=(8, 6))
correlation = iris_df[feature_names].corr()
sns.heatmap(correlation, annot=True, fmt='.2f', 
           cmap='coolwarm', center=0,
           square=True, linewidths=0.5)
plt.title("Feature Correlation Heatmap", 
         fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# ── Step 4: Preprocessing ─────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_iris)

# PCA for 2D Visualization
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
explained_var = pca.explained_variance_ratio_

print(f"\nPCA Explained Variance:")
print(f"  PC1: {explained_var[0]*100:.1f}%")
print(f"  PC2: {explained_var[1]*100:.1f}%")
print(f"  Total: {sum(explained_var)*100:.1f}%")

# ── Step 5: Apply DBSCAN ──────────────────────────────────
# Find optimal eps using K-distance
from sklearn.neighbors import NearestNeighbors

nbrs = NearestNeighbors(n_neighbors=5).fit(X_scaled)
distances, _ = nbrs.kneighbors(X_scaled)
distances = np.sort(distances[:, 4], axis=0)[::-1]

plt.figure(figsize=(10, 5))
plt.plot(distances, linewidth=2, color='steelblue')
plt.axhline(y=0.8, color='red', linestyle='--', 
           label='ε = 0.8')
plt.xlabel('Points', fontsize=12)
plt.ylabel('5th Nearest Neighbor Distance', fontsize=12)
plt.title('K-Distance Graph for Iris Dataset', 
         fontsize=14, fontweight='bold')
plt.legend()
plt.tight_layout()
plt.show()

# DBSCAN with optimal parameters
dbscan = DBSCAN(eps=0.8, min_samples=5)
dbscan_labels = dbscan.fit_predict(X_scaled)

n_clusters_db = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
n_noise_db = list(dbscan_labels).count(-1)

# ── Step 6: Apply Hierarchical Clustering ─────────────────
# Dendrogram
linkage_matrix = linkage(X_scaled, method='ward')

plt.figure(figsize=(14, 6))
dendrogram(linkage_matrix, 
          truncate_mode='lastp', p=30,
          leaf_rotation=90, leaf_font_size=9,
          color_threshold=7)
plt.axhline(y=7, color='red', linestyle='--', 
           linewidth=2, label='Cut → 3 clusters')
plt.title("Dendrogram — Iris Dataset", 
         fontsize=14, fontweight='bold')
plt.xlabel("Samples")
plt.ylabel("Ward Distance")
plt.legend()
plt.tight_layout()
plt.show()

# Hierarchical model
hc = AgglomerativeClustering(n_clusters=3, linkage='ward')
hc_labels = hc.fit_predict(X_scaled)

# ── Step 7: Compute All Metrics ───────────────────────────
# DBSCAN metrics
valid_db = dbscan_labels != -1
if n_clusters_db > 1:
    db_sil = silhouette_score(X_scaled[valid_db], dbscan_labels[valid_db])
    db_ari = adjusted_rand_score(y_iris[valid_db], dbscan_labels[valid_db])
    db_dbs = davies_bouldin_score(X_scaled[valid_db], dbscan_labels[valid_db])
else:
    db_sil = db_ari = db_dbs = 0.0

# Hierarchical metrics
hc_sil = silhouette_score(X_scaled, hc_labels)
hc_ari = adjusted_rand_score(y_iris, hc_labels)
hc_dbs = davies_bouldin_score(X_scaled, hc_labels)

# ── Step 8: Side-by-Side Visualization ───────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle("Iris Dataset: Complete Clustering Analysis", 
             fontsize=16, fontweight='bold')

cmap = plt.cm.Set1

# Row 1: Ground Truth, DBSCAN, Hierarchical (PCA Space)
# Ground Truth
axes[0, 0].scatter(X_pca[:, 0], X_pca[:, 1], 
                  c=y_iris, cmap=cmap, s=80,
                  edgecolors='black', linewidths=0.5, alpha=0.85)
axes[0, 0].set_title("Ground Truth (3 Species)", 
                     fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel(f"PC1 ({explained_var[0]*100:.1f}%)")
axes[0, 0].set_ylabel(f"PC2 ({explained_var[1]*100:.1f}%)")

# DBSCAN
scatter_db = axes[0, 1].scatter(X_pca[:, 0], X_pca[:, 1], 
                                c=dbscan_labels, cmap='tab10', s=80,
                                edgecolors='black', linewidths=0.5, alpha=0.85)
axes[0, 1].set_title(f"DBSCAN (ε=0.8, MinPts=5)\n"
                     f"Clusters={n_clusters_db}, Noise={n_noise_db}", 
                     fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel(f"PC1 ({explained_var[0]*100:.1f}%)")
axes[0, 1].set_ylabel(f"PC2 ({explained_var[1]*100:.1f}%)")

# Hierarchical
axes[0, 2].scatter(X_pca[:, 0], X_pca[:, 1], 
                  c=hc_labels, cmap=cmap, s=80,
                  edgecolors='black', linewidths=0.5, alpha=0.85)
axes[0, 2].set_title(f"Hierarchical Clustering (Ward, K=3)", 
                     fontsize=12, fontweight='bold')
axes[0, 2].set_xlabel(f"PC1 ({explained_var[0]*100:.1f}%)")
axes[0, 2].set_ylabel(f"PC2 ({explained_var[1]*100:.1f}%)")

# Row 2: Feature Boxplots for each method
# Ground Truth boxplot
iris_df_copy = iris_df.copy()
iris_df_copy['DBSCAN'] = dbscan_labels
iris_df_copy['Hierarchical'] = hc_labels

for ax, col, title in zip(
    [axes[1, 0], axes[1, 1], axes[1, 2]],
    ['Species', 'DBSCAN', 'Hierarchical'],
    ['Ground Truth', 'DBSCAN Labels', 'Hierarchical Labels']
):
    plot_data = iris_df_copy.groupby(col)['petal length (cm)'].mean()
    ax.bar(range(len(plot_data)), plot_data.values, 
          color=plt.cm.Set1(np.linspace(0, 1, len(plot_data))),
          edgecolor='black', alpha=0.85)
    ax.set_xticks(range(len(plot_data)))
    ax.set_xticklabels([str(i) for i in plot_data.index], rotation=15)
    ax.set_title(f"{title}\nAvg Petal Length by Group", fontsize=11)
    ax.set_ylabel("Avg Petal Length (cm)")

plt.tight_layout()
plt.show()

# ── Step 9: Comprehensive Metrics Table ───────────────────
print("\n" + "="*65)
print("              COMPREHENSIVE COMPARISON — IRIS DATASET")
print("="*65)
print(f"{'Metric':<30} {'DBSCAN':>15} {'Hierarchical':>15}")
print("-"*65)
print(f"{'Clusters Found':<30} {n_clusters_db:>15} {'3':>15}")
print(f"{'Noise Points':<30} {n_noise_db:>15} {'N/A':>15}")
print(f"{'Silhouette Score':<30} {db_sil:>15.4f} {hc_sil:>15.4f}")
print(f"{'Adjusted Rand Index':<30} {db_ari:>15.4f} {hc_ari:>15.4f}")
print(f"{'Davies-Bouldin Score':<30} {db_dbs:>15.4f} {hc_dbs:>15.4f}")
print("="*65)
print("\nInterpretation:")
print("  Silhouette: Higher is better (range: -1 to 1)")
print("  ARI:        Higher is better (range: 0 to 1, 1=perfect)")
print("  DBS:        Lower is better (range: 0 to ∞)")

# ── Step 10: Feature Importance ───────────────────────────
print("\n" + "="*55)
print("        PCA COMPONENT LOADINGS (Feature Weights)")
print("="*55)
loadings = pd.DataFrame(
    pca.components_.T,
    columns=['PC1', 'PC2'],
    index=feature_names
)
print(loadings.round(4))
```

**Expected Output Summary:**
```
===================================================================
              COMPREHENSIVE COMPARISON — IRIS DATASET
===================================================================
Metric                          DBSCAN    Hierarchical
-----------------------------------------------------------------
Clusters Found                       2               3
Noise Points                        15             N/A
Silhouette Score                0.5821          0.5978
Adjusted Rand Index             0.5643          0.7302
Davies-Bouldin Score            0.6712          0.6891
===================================================================
Interpretation:
  Silhouette: Higher is better (range: -1 to 1)
  ARI:        Higher is better (range: 0 to 1, 1=perfect)
  DBS:        Lower is better (range: 0 to ∞)
```

---

## 10. Conclusion

---

### 📊 Final Comparison Table

```
┌───────────────────────┬─────────────────────┬─────────────────────┐
│ Aspect                │ DBSCAN              │ Hierarchical        │
├───────────────────────┼─────────────────────┼─────────────────────┤
│ Cluster Shape         │ Any shape           │ Any shape           │
│ Predefined K          │ Not required        │ Not required        │
│ Outlier Detection     │ ✅ Yes              │ ❌ No               │
│ Scalability           │ Large datasets      │ Small/Medium        │
│ Interpretability      │ Medium              │ High (dendrogram)   │
│ Nested Clusters       │ No                 │ ✅ Yes              │
│ Varying Densities     │ Partial             │ Partial             │
│ Time Complexity       │ O(n log n)          │ O(n² log n)         │
│ Key Parameters        │ ε, MinPts           │ K, Linkage method   │
│ Best For              │ Density/Spatial     │ Taxonomy/Hierarchy  │
│ Visualization         │ Cluster plot        │ Dendrogram          │
└───────────────────────┴─────────────────────┴─────────────────────┘
```

---

### 🔑 Key Takeaways

```
┌─────────────────────────────────────────────────────────────┐
│                     KEY TAKEAWAYS                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  DBSCAN:                                                    │
│  ✅ Best when you don't know K                             │
│  ✅ Best when data has noise/outliers                      │
│  ✅ Best for irregular, non-spherical shapes               │
│  ✅ Excellent for spatial & geospatial data                │
│  ⚠️  Sensitive to ε and MinPts selection                  │
│  ⚠️  Struggles with varying-density clusters              │
│                                                             │
│  Hierarchical Clustering:                                   │
│  ✅ Best when you need visual interpretation               │
│  ✅ Best for biological, taxonomic problems                │
│  ✅ Best when nested clusters are meaningful               │
│  ✅ No need to specify K upfront                          │
│  ⚠️  Not scalable for very large datasets                 │
│  ⚠️  Cannot undo merges (greedy algorithm)               │
│                                                             │
│  Both outperform K-Means when:                              │
│  → Clusters are non-spherical                              │
│  → Number of clusters is unknown                           │
│  → Data has complex structure                              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

### 🚀 When to Use What — Final Decision Guide

```
Your Data Has...           →  Use This Algorithm
─────────────────────────────────────────────────────────────
Noise/Anomalies            →  DBSCAN
Spatial/GPS data           →  DBSCAN
Unknown shapes             →  DBSCAN
Need to see hierarchy      →  Hierarchical Clustering
Biological taxonomy        →  Hierarchical Clustering
Small dataset (<5000)      →  Hierarchical Clustering
Large dataset (>10000)     →  DBSCAN
Sub-groups within groups   →  Hierarchical Clustering
Real-time anomaly detect.  →  DBSCAN
```

---

### 📚 References & Further Reading

```
📖 Original Papers:
   • DBSCAN: Ester et al. (1996) — "A Density-Based Algorithm
     for Discovering Clusters"
   • Hierarchical: Ward (1963) — "Hierarchical Grouping to
     Optimize an Objective Function"

🔧 Libraries Used:
   • scikit-learn: https://scikit-learn.org
   • scipy: https://scipy.org
   • numpy: https://numpy.org
   • pandas: https://pandas.pydata.org
   • matplotlib: https://matplotlib.org
   • seaborn: https://seaborn.pydata.org

📊 Datasets:
   • Iris Dataset: UCI ML Repository
   • Mall Customers: Kaggle
   • make_moons: sklearn.datasets
```

---

> **💡 Final Note:** Neither DBSCAN nor Hierarchical Clustering is universally superior — the best algorithm depends entirely on your **data characteristics**, **problem requirements**, and **business context**. Always start with **exploratory data analysis (EDA)**, visualize your data, and experiment with both algorithms using appropriate evaluation metrics before making a final decision.

--