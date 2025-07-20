import tensorflow as tf
from tensorflow.keras.layers import TextVectorization
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_20newsgroups
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

seed = 42
num_samples = 1000
ngrams = 2
max_tokens = 3000

# Load 20 Newsgroups dataset
newsgroups = fetch_20newsgroups(subset='all')
text_data = newsgroups.data
labels = newsgroups.target  # Integer topic labels

# Optionally, limit the number of samples for speed
text_data = text_data[:num_samples]
labels = labels[:num_samples]

labels_tf = tf.constant(labels)
text_ds = tf.data.Dataset.from_tensor_slices(text_data)
label_ds = tf.data.Dataset.from_tensor_slices(labels_tf)
sup_ds = tf.data.Dataset.zip((text_ds, label_ds)).prefetch(tf.data.AUTOTUNE)

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
original_texts = []
for vec, label in binary_ngram_sup_ds:
    vectorized_samples.append(vec.numpy())
    sample_labels.append(label.numpy())
for text, label in sup_ds:
    original_texts.append(text.numpy())
X = np.stack(vectorized_samples)
y = np.array(sample_labels)

np.save("features.npy", X)
np.save("labels.npy", y)
np.save("original_texts.npy", original_texts)