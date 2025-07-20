import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, adjusted_rand_score

# Load the preprocessed data
X = np.load("features.npy")  # shape: (num_samples, num_features)
original_texts = np.load("original_texts.npy", allow_pickle=True)
labels = np.load("labels.npy")  # true topic labels

n_clusters = len(np.unique(labels))

# KMeans clustering
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X)
print("\nKMeans:")
print("Cluster labels for first 10 samples:", kmeans_labels[:10])
print(f"Silhouette score: {silhouette_score(X, kmeans_labels):.4f}")
print(f"Adjusted Rand Index: {adjusted_rand_score(labels, kmeans_labels):.4f}")

# Agglomerative Clustering
agglo = AgglomerativeClustering(n_clusters=n_clusters)
agglo_labels = agglo.fit_predict(X)
print("\nAgglomerative Clustering:")
print("Cluster labels for first 10 samples:", agglo_labels[:10])
print(f"Silhouette score: {silhouette_score(X, agglo_labels):.4f}")
print(f"Adjusted Rand Index: {adjusted_rand_score(labels, agglo_labels):.4f}")

# DBSCAN clustering
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(X)
print("\nDBSCAN:")
print("Cluster labels for first 10 samples:", dbscan_labels[:10])
n_clusters_dbscan = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
if n_clusters_dbscan > 1:
    print(f"Silhouette score: {silhouette_score(X, dbscan_labels):.4f}")
    print(f"Adjusted Rand Index: {adjusted_rand_score(labels, dbscan_labels):.4f}")
else:
    print("Silhouette score and ARI not defined for single cluster or noise-only clustering.")