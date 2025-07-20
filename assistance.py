import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

# Load features and labels for 20 Newsgroups
X = np.load("features.npy")
y = np.load("labels.npy")

# Silhouette score (original space, using topic labels)
sil_score = silhouette_score(X, y)
print(f"Silhouette score (original space): {sil_score:.4f}")

# Reduce dimensionality to 2 components for visualization using t-SNE
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)

print("\nt-SNE result (first 3 samples):")
print(X_tsne[:3])

# Visualization using matplotlib, color by topic label
plt.figure(figsize=(8, 6))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='tab20', alpha=0.7)
plt.title("t-SNE Scatter Plot of 20 Newsgroups Samples (Colored by Topic)")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.grid(True)
plt.colorbar(label="Topic Label")
plt.show()