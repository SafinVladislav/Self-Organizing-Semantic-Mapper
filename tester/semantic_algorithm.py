# semantic_algorithm.py

import os
import time
import numpy as np
import PyPDF2
import nltk
import sentence_transformers
import sklearn.metrics.pairwise
import re

# --- NLTK Setup ---
# Ensure the 'punkt' tokenizer is available.
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    print("NLTK 'punkt' tokenizer not found. Downloading...")
    nltk.download('punkt', quiet=True)
    print("Download complete.")

class Pipeline:
    """
    A class that encapsulates the logic for processing a document to find
    semantically relevant chunks of text.
    """

    def __init__(self, model_name='paraphrase-multilingual-MiniLM-L12-v2'):
        """
        Initializes the pipeline by loading the sentence transformer model.
        
        Args:
            model_name (str): The name of the sentence-transformer model to use.
        """
        self.model_name = model_name
        self.model = None
        self.chunks = []
        self.embeddings = None
        self._load_model()

    def _load_model(self):
        """Loads the sentence transformer model into memory."""
        print(f"Loading model: '{self.model_name}'...")
        start_time = time.time()
        self.model = sentence_transformers.SentenceTransformer(self.model_name)
        end_time = time.time()
        print(f"Model loaded in {end_time - start_time:.2f} seconds.")

    def _extract_text_from_pdf(self, file_path):
        """
        Extracts raw text from a PDF file.
        
        Args:
            file_path (str): The path to the PDF document.
            
        Returns:
            str: The concatenated text from all pages of the PDF.
        """
        print(f"Extracting text from '{os.path.basename(file_path)}'...")
        full_text = ""
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            num_pages = len(reader.pages)
            print(f"Found {num_pages} pages.")
            for i, page in enumerate(reader.pages):
                if (i + 1) % 10 == 0:
                    print(f"  - Processing page {i+1}/{num_pages}...")
                page_text = page.extract_text()
                if page_text:
                    full_text += page_text + "\n"
        print("Text extraction complete.")
        return full_text

    def _chunk_text(self, text, chunk_size=256, overlap=50):
        """
        Splits a long text into smaller, overlapping chunks using sentence tokenization.
        
        Args:
            text (str): The input text.
            chunk_size (int): The target maximum number of words per chunk.
            overlap (int): The number of words to overlap between chunks.
            
        Returns:
            list[str]: A list of text chunks.
        """
        print("Chunking text...")
        sentences = nltk.sent_tokenize(text)
        
        chunks = []
        current_chunk_words = []
        for sentence in sentences:
            sentence_words = re.split(r'(\s+)', sentence)
            current_chunk_words.extend(sentence_words)
            
            if len(current_chunk_words) >= chunk_size:
                chunks.append("".join(current_chunk_words).strip())
                # Create overlap
                current_chunk_words = current_chunk_words[-overlap:]

        if current_chunk_words:
            chunks.append("".join(current_chunk_words).strip())
            
        print(f"Created {len(chunks)} chunks.")
        return [c for c in chunks if c] # Filter out any empty chunks

    def process_document(self, file_path):
        """
        Runs the full processing pipeline on a document.
        
        Args:
            file_path (str): The path to the document.
            
        Returns:
            float: The time taken in seconds to create the embeddings.
        """
        # Step 1: Extract text
        raw_text = self._extract_text_from_pdf(file_path)
        if not raw_text:
            print("Error: No text could be extracted from the document.")
            return 0.0

        # Step 2: Chunk text
        self.chunks = self._chunk_text(raw_text)
        if not self.chunks:
            print("Error: No chunks could be generated from the text.")
            return 0.0

        # Step 3: Embed chunks and measure time
        print(f"Embedding {len(self.chunks)} chunks. This may take a while...")
        start_time = time.time()
        self.embeddings = self.model.encode(self.chunks, show_progress_bar=True)
        end_time = time.time()
        
        embedding_time = end_time - start_time
        print(f"Embedding complete.")
        
        return embedding_time

    def search(self, query, top_n=5):
        """
        Searches the processed document for the most relevant chunks.
        
        Args:
            query (str): The user's search query.
            top_n (int): The number of top results to return.
            
        Returns:
            list[dict]: A list of dictionaries, each containing a result 'chunk' and its 'score'.
        """
        if self.embeddings is None or len(self.chunks) == 0:
            print("Error: You must process a document before searching.")
            return []
            
        query_embedding = self.model.encode([query])
        
        # Calculate cosine similarity
        sim_scores = sklearn.metrics.pairwise.cosine_similarity(query_embedding, self.embeddings)[0]
        
        # Get the top N indices
        top_indices = np.argsort(sim_scores)[::-1][:top_n]
        
        results = [
            {"chunk": self.chunks[i], "score": float(sim_scores[i])}
            for i in top_indices
        ]
        
        return results