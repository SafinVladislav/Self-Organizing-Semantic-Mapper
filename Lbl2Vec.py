from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np
import random
from collections import Counter
from sklearn.metrics.pairwise import euclidean_distances
import json

# 1. Load dataset and shuffle texts
print("Loading and shuffling 20 Newsgroups dataset...")
data = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'), shuffle=True, random_state=42)
documents = data.data
labels = data.target
label_names = data.target_names
n_topics = len(label_names)  # 20 predefined topics

# 2. Keywords
print("\nLoading keywords...")
with open("topics_keywords.json", "r", encoding="utf-8") as f:
    topics_keywords = json.load(f)

# 3. Embed keywords and find centroid for each topic (weighted mean)
print("\nEmbedding keywords and finding centroids...")
model = SentenceTransformer('all-MiniLM-L6-v2')
topic_centroids = []
for idx, keywords in enumerate(topics_keywords):
    embeddings = model.encode(keywords)
    centroid = np.mean(embeddings, axis=0)
    topic_centroids.append(centroid)
    
# 4. Embed all documents
#print("Embedding all documents...")
#doc_embeddings = model.encode(documents, show_progress_bar=True)
#np.save("doc_embeddings_all-MiniLM-L6-v2.npy", doc_embeddings)
print("Loading precomputed document embeddings...")
doc_embeddings = np.load("doc_embeddings_all-MiniLM-L6-v2.npy")

# 5. For each topic, rank documents by cosine similarity to centroid
topic_docs = []
best_docs = int(0.2 * (len(documents) / n_topics))
print(f"Selecting top {best_docs} documents per topic...")
for idx, centroid in enumerate(topic_centroids):
    dists = cosine_similarity([centroid], doc_embeddings)[0]
    ranked_indices = np.argsort(dists)[::-1]
    selected = ranked_indices[:best_docs]
    topic_docs.append(selected)

    # Output most common label among chosen documents
    if len(selected) > 0:
        selected_labels = [labels[doc_idx] for doc_idx in selected]
        most_common_label, count = Counter(selected_labels).most_common(1)[0]

# 6. Find centroid of selected documents for each topic
doc_centroids = []
for idx, doc_indices in enumerate(topic_docs):
    if len(doc_indices) > 0:
        centroid = np.mean(doc_embeddings[doc_indices], axis=0)
    else:
        centroid = topic_centroids[idx]
    doc_centroids.append(centroid)

# 7. Classify remaining documents to closest centroid
print("\nClassifying remaining documents...")
assigned_topics = np.full(len(documents), -1)
for topic_idx, doc_indices in enumerate(topic_docs):
    assigned_topics[doc_indices] = topic_idx

unassigned = np.where(assigned_topics == -1)[0]
if len(unassigned) > 0:
    sims = cosine_similarity(doc_embeddings[unassigned], doc_centroids)
    closest = np.argmax(sims, axis=1)
    for i, doc_idx in enumerate(unassigned):
        assigned_topics[doc_idx] = closest[i]

"""# 8. Print sample classification results
print("\nSample classification results:")
for topic_idx, topic_name in enumerate(label_names):
    #print(f"\nTopic #{topic_idx+1} ({topic_name}):")
    topic_doc_indices = np.where(assigned_topics == topic_idx)[0]
    #sample_indices = random.sample(list(topic_doc_indices), min(1, len(topic_doc_indices)))
    #for doc_idx in sample_indices:
    #    print(f"  - {documents[doc_idx][:500].replace('\n', ' ')}...")"""

# Calculate classification accuracy
correct = np.sum(assigned_topics == labels)
accuracy = correct / len(labels)
print(f"\nClassification accuracy: {accuracy:.4f}")

print("\nProcess complete.")