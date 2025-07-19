import tensorflow as tf
from tensorflow.keras.layers import TextVectorization
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

X = np.load("features.npy")
y = np.load("labels.npy")

sil_score = silhouette_score(X, y)
print(f"Silhouette score (original space): {sil_score:.4f}")

# Reduce dimensionality to 2 components for visualization using t-SNE
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)

print("\nt-SNE result (first 3 samples):")
print(X_tsne[:3])

# Visualization using matplotlib, color by label
plt.figure(figsize=(8, 6))
colors = ['red' if label == 1 else 'blue' for label in y]
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=colors, alpha=0.7)
plt.title("t-SNE Scatter Plot of Text Samples (Red=Positive, Blue=Negative)")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.grid(True)
plt.show()