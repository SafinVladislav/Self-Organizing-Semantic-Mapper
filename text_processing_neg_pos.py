"""This program is supposed to first standartize, tokenize and index text snippets.
Then it should reduce dimensionality of the text data and visualize the results.
The results will be used in another program."""
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

seed = 42
texts = 1000
ngrams = 2
max_tokens = 3000

# Path to supervised data
pos_dir = pathlib.Path(r"C:\Users\edfer\Desktop\New_project\Datasets\aclImdb\train\pos")
neg_dir = pathlib.Path(r"C:\Users\edfer\Desktop\New_project\Datasets\aclImdb\train\neg")

# Number of samples from each class
samples_per_class = texts // 2

# Get .txt file paths for each class
pos_files = list(pos_dir.glob("*.txt"))[:samples_per_class]
neg_files = list(neg_dir.glob("*.txt"))[:samples_per_class]

# Combine and create labels
all_files = pos_files + neg_files
labels = [1] * len(pos_files) + [0] * len(neg_files)  # 1 for pos, 0 for neg

# Shuffle with a fixed seed for reproducibility
tf.random.set_seed(seed)
indices = tf.random.shuffle(tf.range(len(all_files)))
all_files = [all_files[i.numpy()] for i in indices]
labels = [labels[i.numpy()] for i in indices]

file_paths = tf.constant([str(p) for p in all_files])
labels_tf = tf.constant(labels)

# Define the dataset: (text, label) pairs
sup_ds = tf.data.Dataset.from_tensor_slices((file_paths, labels_tf))
sup_ds = sup_ds.map(
    lambda path, label: (tf.io.read_file(path), label),
    num_parallel_calls=tf.data.AUTOTUNE
)
sup_ds = sup_ds.prefetch(tf.data.AUTOTUNE)

# Create the TextVectorization layer
text_vectorization = TextVectorization(
    ngrams=ngrams,
    max_tokens=max_tokens,
    output_mode="tf_idf",
)

# Adapt the vectorizer to your supervised dataset (only text)
text_only_ds = sup_ds.map(lambda text, label: text)
text_vectorization.adapt(text_only_ds)

# Vectorize the dataset
binary_ngram_sup_ds = sup_ds.map(
    lambda text, label: (text_vectorization(text), label),
    num_parallel_calls=tf.data.AUTOTUNE
)

# Collect all vectorized samples and labels into numpy arrays
vectorized_samples = []
sample_labels = []
for vec, label in binary_ngram_sup_ds:
    vectorized_samples.append(vec.numpy())
    sample_labels.append(label.numpy())
X = np.stack(vectorized_samples)
y = np.array(sample_labels)

np.save("features.npy", X)
np.save("labels.npy", y)

original_texts = []
for text, label in sup_ds:
    original_texts.append(text.numpy().decode('utf-8'))
np.save("original_texts.npy", original_texts)

"""# Calculate silhouette score for the original high-dimensional space
from sklearn.metrics import silhouette_score

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
plt.show()"""