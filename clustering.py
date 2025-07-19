import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

# Load the preprocessed features and original texts
X = np.load("features.npy")
original_texts = np.load("original_texts.npy", allow_pickle=True)

# DBSCAN clustering
dbscan = DBSCAN(eps=0.5, min_samples=5)
cluster_labels = dbscan.fit_predict(X)
print("Cluster labels for first 10 samples:", cluster_labels[:10])

# Only compute silhouette score if there are at least 2 clusters (excluding noise)
n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
if n_clusters > 1:
    score = silhouette_score(X, cluster_labels)
    print(f"Silhouette score: {score:.4f}")
else:
    print("Silhouette score not defined (less than 2 clusters found).")