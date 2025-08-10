import os
import time
import PyPDF2
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords, wordnet
import re
from collections import Counter
# NEW: Import sentence-transformers for advanced keyword generation
from sentence_transformers import SentenceTransformer, util

# --- NLTK Setup ---
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except Exception:
    print("NLTK data not found. Downloading...")
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    print("Download complete.")

class Pipeline:
    """
    A class that encapsulates the logic for processing a document to find
    statistically and semantically relevant chunks of text.
    """

    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """
        Initializes the pipeline by loading the sentence transformer model
        and setting up NLP tools.
        """
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.original_sentences = []
        self.word_frequencies = Counter()
        
        # --- NEW: Semantic Model for Keyword Expansion ---
        print(f"Loading semantic model: '{model_name}'...")
        start_time = time.time()
        self.semantic_model = SentenceTransformer(model_name)
        end_time = time.time()
        print(f"Model loaded in {end_time - start_time:.2f} seconds.")
        
        # --- NEW: Store for document's vocabulary and its embeddings ---
        self.doc_vocabulary = []
        self.doc_vocab_embeddings = None


    def _extract_text_from_pdf(self, file_path):
        """Extracts raw text from a PDF file."""
        print(f"Extracting text from '{os.path.basename(file_path)}'...")
        full_text = ""
        with open(file_path, 'rb') as file:
            try:
                reader = PyPDF2.PdfReader(file)
                num_pages = len(reader.pages)
                print(f"Found {num_pages} pages.")
                for i, page in enumerate(reader.pages):
                    if (i + 1) % 50 == 0:
                        print(f"  - Processing page {i+1}/{num_pages}...")
                    page_text = page.extract_text()
                    if page_text:
                        full_text += page_text + "\n"
            except Exception as e:
                print(f"Error reading PDF file: {e}")
                return ""
        print("Text extraction complete.")
        return full_text

    def _preprocess_text(self, text):
        """Tokenizes, lemmatizes, and calculates word frequencies."""
        print("Preprocessing text...")
        self.original_sentences = sent_tokenize(text)
        
        # Clean and lemmatize the entire document text
        cleaned_text = re.sub(r'[^a-zA-Z\s]', '', text).lower()
        words = word_tokenize(cleaned_text)
        lemmatized_words = [self.lemmatizer.lemmatize(w) for w in words if w not in self.stop_words and len(w) > 1]
        
        self.word_frequencies = Counter(lemmatized_words)
        self.doc_vocabulary = list(self.word_frequencies.keys())
        print(f"Processed {len(lemmatized_words)} words into a vocabulary of {len(self.doc_vocabulary)} unique terms.")

    def process_document(self, file_path):
        """
        Runs the full processing pipeline on a document, including embedding the vocabulary.
        """
        raw_text = self._extract_text_from_pdf(file_path)
        if not raw_text:
            return 0.0

        start_time = time.time()
        self._preprocess_text(raw_text)
        
        # --- NEW: Create embeddings for the document's vocabulary ---
        print("Embedding document vocabulary for semantic keyword expansion...")
        self.doc_vocab_embeddings = self.semantic_model.encode(self.doc_vocabulary, convert_to_tensor=True, show_progress_bar=True)
        
        end_time = time.time()
        processing_time = end_time - start_time
        print(f"Document processing complete in {processing_time:.2f} seconds.")
        return processing_time

    def _lemmatize_query(self, query):
        """Lemmatizes the search query."""
        query_words = word_tokenize(query.lower())
        return [self.lemmatizer.lemmatize(w) for w in query_words if w not in self.stop_words]

    def _expand_query(self, lemmas, use_wordnet=True, use_semantic=True, semantic_similarity_threshold=0.6, semantic_top_k=5):
        """Expands a list of lemmas with synonyms and semantically similar terms."""
        expanded_lemmas = set(lemmas)

        # 1. WordNet Expansion (Synonyms)
        if use_wordnet:
            for lemma in lemmas:
                for syn in wordnet.synsets(lemma):
                    for name in syn.lemma_names():
                        expanded_lemmas.add(name.replace('_', ' '))

        # 2. Semantic Expansion (Similar words from document context)
        if use_semantic and self.doc_vocab_embeddings is not None:
            query_embeddings = self.semantic_model.encode(lemmas, convert_to_tensor=True)
            
            # Find similar words from the document's own vocabulary
            hits = util.semantic_search(query_embeddings, self.doc_vocab_embeddings, top_k=semantic_top_k)
            
            for i in range(len(hits)):
                for hit in hits[i]:
                    if hit['score'] > semantic_similarity_threshold:
                        expanded_lemmas.add(self.doc_vocabulary[hit['corpus_id']])

        return list(expanded_lemmas)

    def search(self, query, top_n=5, chunk_size=3):
        """
        Searches for chunks of text with the highest keyword density.
        
        Args:
            query (str): The user's search query.
            top_n (int): The number of top chunks to return.
            chunk_size (int): The number of sentences to include in each chunk.
        """
        if not self.original_sentences:
            print("Error: You must process a document before searching.")
            return []
            
        # --- Step 1: Expand query with both methods ---
        initial_lemmas = self._lemmatize_query(query)
        print(f"Original keywords: {initial_lemmas}")
        
        expanded_keywords = self._expand_query(initial_lemmas)
        print(f"Expanded keywords ({len(expanded_keywords)}): {expanded_keywords}")

        if not expanded_keywords:
            print("Warning: Could not generate keywords from the query.")
            return []

        # --- Step 2: Use a sliding window to find dense chunks ---
        chunk_scores = []
        for i in range(len(self.original_sentences) - chunk_size + 1):
            # Create a chunk of 'chunk_size' sentences
            chunk_sentences = self.original_sentences[i : i + chunk_size]
            chunk_text = " ".join(chunk_sentences)
            
            # Lemmatize the chunk for comparison
            chunk_words = word_tokenize(chunk_text.lower())
            chunk_lemmas = {self.lemmatizer.lemmatize(w) for w in chunk_words}
            
            # Calculate score based on keyword presence and rarity
            score = 0
            for keyword in expanded_keywords:
                if keyword in chunk_lemmas:
                    # Inverse frequency score: rarer keywords get more points
                    score += 1 / (self.word_frequencies.get(keyword, 1))

            # Normalize by chunk length to get a "density" score
            # (prevents longer chunks from always winning)
            num_words_in_chunk = len(chunk_words)
            if num_words_in_chunk > 0:
                density = score / num_words_in_chunk
                chunk_scores.append({'chunk': chunk_text, 'score': density})

        # --- Step 3: Sort by density and return top N ---
        sorted_chunks = sorted(chunk_scores, key=lambda x: x['score'], reverse=True)
        
        return sorted_chunks[:top_n]