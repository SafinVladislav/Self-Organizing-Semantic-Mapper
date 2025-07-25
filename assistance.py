import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox
import os
import json
import numpy as np

# It's good practice to wrap third-party imports in a try-except block
# to provide helpful messages if the library isn't installed.
try:
    import PyPDF2
except ImportError:
    messagebox.showerror(
        "Missing Library",
        "The 'PyPDF2' library is not installed.\n"
        "Please install it using: pip install PyPDF2"
    )
    exit() # Exit if essential library is missing

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    messagebox.showerror(
        "Missing Libraries",
        "The 'sentence-transformers' or 'scikit-learn' libraries are not installed.\n"
        "Please install them using: pip install sentence-transformers scikit-learn numpy"
    )
    exit() # Exit if essential libraries are missing

import nltk # Used for robust text chunking
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    # This block handles the case where 'punkt' tokenizer is not downloaded.
    # It attempts to download it, which is crucial for the chunking function.
    messagebox.showinfo(
        "NLTK Download",
        "NLTK 'punkt' tokenizer not found. Downloading it now for text processing. "
        "This is a one-time download."
    )
    nltk.download('punkt')


# --- PDF Extraction and Chunking Functions (Adapted from your first program) ---

def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a PDF file, page by page.
    Includes error handling for file operations.
    """
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                # Check if page.extract_text() returns None or empty string
                extracted_page_text = page.extract_text()
                if extracted_page_text:
                    text += extracted_page_text + "\n" # Add newline between pages
            return text
    except Exception as e:
        # Print error to console and return empty string if extraction fails
        print(f"Error extracting text from PDF: {e}")
        return ""

def chunk_text(text, max_tokens=256, overlap_words=50):
    """
    Splits text into smaller, overlapping snippets (chunks) using NLTK for sentence tokenization.
    This ensures that chunks are semantically more coherent than simple word-based splitting.

    Args:
        text (str): The full text to be chunked.
        max_tokens (int): The maximum number of words allowed in a chunk.
        overlap_words (int): The approximate number of words to overlap between consecutive chunks.

    Returns:
        list: A list of text chunks (strings).
    """
    chunks = []
    # Prefer sentence splitting for better semantic chunks
    sentences = nltk.sent_tokenize(text)
    
    current_chunk_sentences = []
    current_chunk_length = 0

    for sentence in sentences:
        sentence_length = len(sentence.split()) # Approximate word count
        
        # If adding the next sentence exceeds max_tokens and there's content in current_chunk_sentences,
        # finalize the current chunk and start a new one with overlap.
        if current_chunk_length + sentence_length > max_tokens and current_chunk_sentences:
            chunks.append(" ".join(current_chunk_sentences).strip())
            
            # Create overlap: start new chunk with the last 'overlap_words' words.
            # This is done by selecting whole sentences from the end of the previous chunk
            # that roughly sum up to 'overlap_words'.
            overlap_content = []
            overlap_len = 0
            # Iterate backwards from the end of the current chunk's sentences
            for s_idx in range(len(current_chunk_sentences) -1, -1, -1):
                s = current_chunk_sentences[s_idx]
                if overlap_len + len(s.split()) <= overlap_words:
                    overlap_content.insert(0, s) # Add to front to maintain original order
                    overlap_len += len(s.split())
                else:
                    break # Stop if adding the next sentence exceeds overlap_words
            
            current_chunk_sentences = overlap_content
            current_chunk_length = overlap_len

        # Add the current sentence to the chunk
        current_chunk_sentences.append(sentence)
        current_chunk_length += sentence_length
    
    # Add the last chunk if any sentences remain
    if current_chunk_sentences:
        chunks.append(" ".join(current_chunk_sentences).strip())
        
    # Filter out empty or very short snippets to ensure quality chunks
    return [chunk for chunk in chunks if len(chunk) > 20] # Increased minimum length for relevance


def find_best_snippets(query, snippets, snippet_embeddings, model, top_n=5):
    """
    Finds the top N snippets most relevant to the user's query using cosine similarity
    between their embeddings.

    Args:
        query (str): The user's search query.
        snippets (list): A list of text chunks (strings).
        snippet_embeddings (numpy.ndarray): Embeddings for each snippet.
        model (SentenceTransformer): The pre-loaded sentence embedding model.
        top_n (int): The number of top relevant snippets to return.

    Returns:
        list: A list of the top N most relevant snippets.
    """
    if not snippets or snippet_embeddings is None:
        return [] # Return empty if no snippets or embeddings are available

    # Encode the user query into a vector
    query_embedding = model.encode([query])
    
    # Calculate cosine similarity between the query embedding and all snippet embeddings.
    # cosine_similarity returns a 2D array, so we take the first row [0].
    similarities = cosine_similarity(query_embedding, snippet_embeddings)[0]

    # Get the indices of the top N most similar snippets.
    # np.argsort returns indices that would sort an array; [::-1] reverses it for descending order.
    # [:top_n] takes the first N indices (most similar).
    top_n_indices = np.argsort(similarities)[::-1][:top_n]

    # Return the actual snippets corresponding to the top N indices
    return [snippets[i] for i in top_n_indices]

class PDFExtractorApp:
    """
    A desktop application that combines PDF text extraction, intelligent chunking,
    and semantic search to find relevant content based on user queries.
    It leverages SentenceTransformer for semantic understanding.
    """
    def __init__(self, master):
        """
        Initializes the PDFExtractorApp with the main Tkinter window and sets up the GUI elements.

        Args:
            master (tk.Tk): The root Tkinter window.
        """
        self.master = master
        master.title("Semantic PDF Content Extractor")
        master.geometry("1000x700") # Set an initial window size, larger for more content

        # Configure grid layout for responsiveness
        master.grid_rowconfigure(0, weight=0) # PDF selection controls
        master.grid_rowconfigure(1, weight=0) # Request input controls
        master.grid_rowconfigure(2, weight=0) # Action button
        master.grid_rowconfigure(3, weight=1) # Result text area (expands vertically)
        master.grid_columnconfigure(0, weight=1) # Left column expands
        master.grid_columnconfigure(1, weight=1) # Right column expands

        self.pdf_path = "" # Stores the path to the selected PDF file
        self.snippets = [] # Stores extracted text chunks
        self.snippet_embeddings = None # Stores embeddings of the snippets
        self.model = None # Stores the SentenceTransformer model

        # --- Initialize the SentenceTransformer model early ---
        # This can take some time, so it's good to do it once at startup.
        # Inform the user that the model is loading.
        messagebox.showinfo("Loading Model", "Loading SentenceTransformer model (all-MiniLM-L6-v2). This may take a moment...")
        try:
            # Use 'cuda' if a GPU is available, otherwise defaults to CPU.
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            messagebox.showinfo("Model Loaded", "SentenceTransformer model loaded successfully!")
        except Exception as e:
            messagebox.showerror("Model Load Error", f"Failed to load SentenceTransformer model: {e}\n"
                                                      "Please check your internet connection and ensure all necessary libraries are installed.")
            # Disable functionality if model fails to load
            self.extract_button.config(state=tk.DISABLED)
            self.select_pdf_button.config(state=tk.DISABLED)
            return

        # --- PDF Selection Section ---
        # Frame to group PDF selection elements for better organization
        self.select_pdf_frame = tk.Frame(master)
        self.select_pdf_frame.grid(row=0, column=0, columnspan=2, pady=10, padx=10, sticky="ew")
        self.select_pdf_frame.columnconfigure(0, weight=1) # Label column expands horizontally
        self.select_pdf_frame.columnconfigure(1, weight=0) # Button column does not expand

        # Label to display the path of the selected PDF file
        self.pdf_path_label = tk.Label(
            self.select_pdf_frame,
            text="No PDF selected",
            wraplength=700, # Wrap text if the path is too long to fit in one line
            justify="left"
        )
        self.pdf_path_label.grid(row=0, column=0, padx=5, sticky="w")

        # Button to open the file dialog for PDF selection
        self.select_pdf_button = tk.Button(
            self.select_pdf_frame,
            text="Select PDF",
            command=self.select_pdf_file
        )
        self.select_pdf_button.grid(row=0, column=1, padx=5, sticky="e")

        # --- Request Input Section ---
        # Frame to group user request input elements
        self.request_frame = tk.Frame(master)
        self.request_frame.grid(row=1, column=0, columnspan=2, pady=5, padx=10, sticky="ew")
        self.request_frame.columnconfigure(0, weight=0) # Label column
        self.request_frame.columnconfigure(1, weight=1) # Entry column expands

        # Label for the request input field
        self.request_label = tk.Label(self.request_frame, text="Enter your semantic query:")
        self.request_label.grid(row=0, column=0, padx=5, sticky="w")

        # Entry widget for the user to type their request
        self.request_entry = tk.Entry(self.request_frame, width=80)
        self.request_entry.grid(row=0, column=1, padx=5, sticky="ew")
        # Bind the Enter key to the extract_content method for convenience
        self.request_entry.bind("<Return>", lambda event=None: self.extract_content())


        # --- Action Button Section ---
        # Frame to group the action button
        self.action_frame = tk.Frame(master)
        self.action_frame.grid(row=2, column=0, columnspan=2, pady=10, padx=10)

        # Button to trigger the content extraction process
        self.extract_button = tk.Button(
            self.action_frame,
            text="Find Relevant Content",
            command=self.extract_content
        )
        self.extract_button.pack(pady=5)

        # --- Result Display Section ---
        # ScrolledText widget to display the extracted content, with word wrapping
        self.result_text = scrolledtext.ScrolledText(master, wrap=tk.WORD, width=100, height=25, font=("Arial", 10))
        self.result_text.grid(row=3, column=0, columnspan=2, pady=10, padx=10, sticky="nsew")
        self.result_text.insert(tk.END, "Instructions:\n"
                                         "1. Click 'Select PDF' to choose your document.\n"
                                         "2. Once selected, the app will process and embed the PDF's content (this might take a moment).\n"
                                         "3. Enter a query (e.g., 'What are the principles of influence?') in the text box.\n"
                                         "4. Click 'Find Relevant Content' or press Enter to see the top 5 most semantically similar passages.\n")

    def select_pdf_file(self):
        """
        Opens a file dialog to allow the user to select a PDF file.
        Updates the pdf_path_label and triggers the PDF processing (chunking and embedding).
        """
        file_path = filedialog.askopenfilename(
            title="Select PDF File",
            filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")]
        )
        if file_path:
            self.pdf_path = file_path
            self.pdf_path_label.config(text=f"Selected PDF: {os.path.basename(self.pdf_path)}")
            self.result_text.delete(1.0, tk.END) # Clear any previous results
            
            # Initiate PDF processing (extraction, chunking, embedding)
            self.process_pdf()

    def process_pdf(self):
        """
        Handles the extraction, chunking, and embedding of the selected PDF.
        It first checks for pre-computed files to save time.
        """
        if not self.pdf_path:
            return

        # Define file paths for saving/loading snippets and embeddings
        base_name = os.path.splitext(self.pdf_path)[0]
        snippets_filepath = f"{base_name}_snippets.json"
        embeddings_filepath = f"{base_name}_embeddings.npy"

        self.snippets = []
        self.snippet_embeddings = None

        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, "Processing PDF... Please wait. This may take a while for large documents.\n")
        self.master.update_idletasks() # Update GUI to show message

        # Try to load pre-computed snippets and embeddings
        if os.path.exists(snippets_filepath) and os.path.exists(embeddings_filepath):
            try:
                self.result_text.insert(tk.END, f"Loading pre-computed data from {os.path.basename(snippets_filepath)}...\n")
                self.master.update_idletasks()
                with open(snippets_filepath, 'r', encoding='utf-8') as f:
                    self.snippets = json.load(f)
                self.snippet_embeddings = np.load(embeddings_filepath)
                self.result_text.insert(tk.END, f"Loaded {len(self.snippets)} snippets and their embeddings.\n")
                messagebox.showinfo("PDF Processed", f"Loaded pre-computed data for '{os.path.basename(self.pdf_path)}'.")
                return # Exit if successfully loaded
            except Exception as e:
                self.result_text.insert(tk.END, f"Error loading saved data: {e}. Re-processing PDF from scratch.\n")
                self.snippets = [] # Reset to force re-processing
                self.snippet_embeddings = None
        
        # If not loaded, process the PDF from scratch
        try:
            self.result_text.insert(tk.END, f"Extracting text from '{os.path.basename(self.pdf_path)}'...\n")
            self.master.update_idletasks()
            full_text = extract_text_from_pdf(self.pdf_path)
            if not full_text.strip():
                self.result_text.insert(tk.END, "Error: Could not extract text from PDF. Please check the file's content or if it's an image-only PDF.\n")
                messagebox.showerror("Extraction Error", "Could not extract text from PDF. It might be an image-only PDF or corrupted.")
                return

            self.result_text.insert(tk.END, "Chunking text into snippets...\n")
            self.master.update_idletasks()
            self.snippets = chunk_text(full_text, max_tokens=256, overlap_words=50)
            self.result_text.insert(tk.END, f"Created {len(self.snippets)} snippets.\n")

            if not self.snippets:
                self.result_text.insert(tk.END, "Error: No snippets were generated from the PDF text. Text might be too short or chunking parameters too restrictive.\n")
                messagebox.showerror("Chunking Error", "No usable snippets generated. PDF content might be too sparse.")
                return
            
            self.result_text.insert(tk.END, "Embedding snippets (this is CPU/GPU intensive)...\n")
            self.master.update_idletasks()
            # The model automatically handles batching for speed
            self.snippet_embeddings = self.model.encode(self.snippets, show_progress_bar=False, convert_to_numpy=True)
            self.result_text.insert(tk.END, "Embedding complete.\n")

            # Save snippets and their embeddings for future use
            self.result_text.insert(tk.END, "Saving processed data...\n")
            self.master.update_idletasks()
            with open(snippets_filepath, 'w', encoding='utf-8') as f:
                json.dump(self.snippets, f, ensure_ascii=False, indent=4)
            np.save(embeddings_filepath, self.snippet_embeddings)
            self.result_text.insert(tk.END, "Snippets and embeddings saved successfully.\n")
            messagebox.showinfo("PDF Processed", f"'{os.path.basename(self.pdf_path)}' processed and data saved for quick access.")

        except PyPDF2.errors.PdfReadError:
            self.result_text.insert(tk.END, "Error: Invalid PDF file or corrupted. Cannot read this PDF.\n")
            messagebox.showerror("PDF Error", "Invalid or corrupted PDF file.")
        except FileNotFoundError:
            self.result_text.insert(tk.END, f"Error: The file '{self.pdf_path}' was not found. Please make sure the PDF is in the correct directory.\n")
            messagebox.showerror("File Error", "PDF file not found.")
        except Exception as e:
            self.result_text.insert(tk.END, f"An unexpected error occurred during PDF processing: {e}\n")
            messagebox.showerror("Processing Error", f"An unexpected error occurred: {e}")

    def extract_content(self):
        """
        Performs the semantic search based on the user's query and displays the results.
        This method is called when the 'Find Relevant Content' button is clicked or Enter is pressed.
        """
        if not self.pdf_path or self.snippet_embeddings is None or not self.snippets:
            messagebox.showerror("Error", "Please select and process a PDF file first.")
            return

        request = self.request_entry.get().strip()
        if not request:
            messagebox.showerror("Error", "Please enter a semantic query.")
            return

        self.result_text.delete(1.0, tk.END) # Clear previous results
        self.result_text.insert(tk.END, f"Finding top 5 snippets for '{request}'...\n")
        self.master.update_idletasks() # Update GUI immediately

        try:
            # Call the semantic search function
            best_fitting_snippets = find_best_snippets(request, self.snippets, self.snippet_embeddings, self.model, top_n=5)

            if best_fitting_snippets:
                self.result_text.insert(tk.END, "\n--- Top 5 Best Fitting Snippets ---\n")
                for i, snippet in enumerate(best_fitting_snippets):
                    self.result_text.insert(tk.END, f"Snippet {i+1}:\n{snippet}\n---\n")
                messagebox.showinfo("Search Complete", "Relevant snippets found!")
            else:
                self.result_text.insert(tk.END, "No relevant snippets found for your query. Try a different query or check the PDF content.\n")
                messagebox.showinfo("No Results", "No relevant snippets found.")

        except Exception as e:
            self.result_text.insert(tk.END, f"An error occurred during search: {e}\n")
            messagebox.showerror("Search Error", f"An error occurred during semantic search: {e}")

# This block ensures the application runs only when the script is executed directly.
if __name__ == "__main__":
    root = tk.Tk() # Create the main Tkinter window
    app = PDFExtractorApp(root) # Instantiate the application
    root.mainloop() # Start the Tkinter event loop, which keeps the window open