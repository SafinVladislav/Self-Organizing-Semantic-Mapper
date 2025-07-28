import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox, ttk
import os
import json
import numpy as np
import threading
import hashlib # Import hashlib for creating unique identifiers
from tkinter import font # Import the font module

# --- Dependency Checks for Document Formats ---

# PDF Support
try:
    import PyPDF2
except ImportError:
    messagebox.showerror("Missing Library", "PyPDF2 is not installed (for .pdf files).\nPlease use: pip install PyPDF2")
    # We don't exit, allowing the app to run and process other formats.

# DOCX Support
try:
    import docx
except ImportError:
    messagebox.showwarning("Missing Library", "python-docx is not installed.\nProcessing .docx files will not be possible.\nPlease use: pip install python-docx")

# AI & Scientific Computing Support
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    messagebox.showerror("Missing Libraries", "The 'sentence-transformers' or 'scikit-learn' libraries are not installed.\nPlease install them to enable semantic search.\n\nUse: pip install sentence-transformers scikit-learn")
    exit()

# NLTK for Text Chunking
import nltk
try:
    # Check for English punkt tokenizer (already in original code)
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    messagebox.showinfo("NLTK Download", "Downloading 'punkt' tokenizer (English) for text processing. This is a one-time download.")
    nltk.download('punkt')

try:
    # --- NEW: Check and download for Russian punkt tokenizer ---
    nltk.data.find('tokenizers/punkt/russian.pickle')
except nltk.downloader.DownloadError:
    messagebox.showinfo("NLTK Download", "Downloading 'punkt' tokenizer (Russian) for text processing. This is a one-time download.")
    nltk.download('punkt', quiet=True) # Use quiet=True to avoid opening a new window

# --- Text Extraction Functions for Each Format ---

def extract_text_from_pdf(file_path, stop_event=None):
    """Extracts text from a PDF file, returning a list of (page_number, text) tuples."""
    page_data = []
    try:
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page_num, page in enumerate(reader.pages):
                if stop_event and stop_event.is_set(): return []
                if extracted_text := page.extract_text():
                    page_data.append((page_num + 1, extracted_text))
    except Exception as e:
        print(f"Error reading PDF {file_path}: {e}")
        # messagebox.showerror("PDF Error", f"Could not read the PDF file. It might be corrupted or encrypted.\n\nError: {e}")
        # Error messages are now handled by the app class
        raise e # Re-raise to be caught by the app's error handling
    return page_data

def extract_text_from_docx(file_path, stop_event=None):
    """Extracts text from a DOCX file, returning it as a single page."""
    try:
        document = docx.Document(file_path)
        full_text = "\n".join([para.text for para in document.paragraphs])
        if stop_event and stop_event.is_set(): return []
        return [(1, full_text)] if full_text else []
    except Exception as e:
        print(f"Error reading DOCX {file_path}: {e}")
        # messagebox.showerror("DOCX Error", f"Could not read the DOCX file.\n\nError: {e}")
        raise e # Re-raise to be caught by the app's error handling
    return []

def extract_text_from_txt(file_path, stop_event=None):
    """Extracts text from a TXT file, returning it as a single page."""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            full_text = f.read()
            if stop_event and stop_event.is_set(): return []
            return [(1, full_text)] if full_text else []
    except Exception as e:
        print(f"Error reading TXT {file_path}: {e}")
        # messagebox.showerror("TXT Error", f"Could not read the TXT file.\n\nError: {e}")
        raise e # Re-raise to be caught by the app's error handling
    return []

def extract_text_from_file(file_path, stop_event=None, app_instance=None):
    """Dispatcher function to select the correct text extractor based on file extension."""
    _, extension = os.path.splitext(file_path)
    extension = extension.lower()
    
    # Map extensions to their extractor functions
    extractor_map = {
        '.pdf': extract_text_from_pdf,
        '.docx': extract_text_from_docx,
        '.txt': extract_text_from_txt,
    }

    if extractor := extractor_map.get(extension):
        try:
            return extractor(file_path, stop_event)
        except NameError:
             # This happens if a required library (e.g., 'docx') failed to import
            if app_instance:
                app_instance.show_error_message("Missing Dependency", app_instance.get_localized_text("missing_dependency").format(extension))
            else:
                messagebox.showerror("Missing Dependency", f"The library required for '{extension}' files is not installed. Please check the startup warnings.")
            return []
        except Exception as e:
            # Catch specific errors from extractors and pass them to the app instance
            error_key = None
            if extension == '.pdf': error_key = "pdf_error"
            elif extension == '.docx': error_key = "docx_error"
            elif extension == '.txt': error_key = "txt_error"

            if app_instance and error_key:
                app_instance.show_error_message(app_instance.get_localized_text("processing_error_title"), app_instance.get_localized_text(error_key).format(e))
            else:
                messagebox.showerror("File Read Error", f"An error occurred while reading the file: {e}")
            return []
    else:
        if app_instance:
            app_instance.show_error_message(app_instance.get_localized_text("unsupported_format_title"), app_instance.get_localized_text("unsupported_format").format(extension))
        else:
            messagebox.showerror("Unsupported Format", f"File format '{extension}' is not supported.")
        return []

# --- Chunking and Semantic Search Functions ---
def chunk_text(page_data, max_tokens=256, overlap_words=50, stop_event=None):
    """Splits text into smaller, overlapping snippets."""
    chunks_with_pages = []
    current_chunk_sentences_info = []
    current_chunk_length = 0
    first_page_in_chunk = -1
    for page_num, page_text in page_data:
        if stop_event and stop_event.is_set(): return []
        # --- MODIFIED: Use NLTK's Russian tokenizer if available, otherwise default ---
        try:
            sentences = nltk.sent_tokenize(page_text, language='russian')
        except LookupError: # Fallback if Russian tokenizer isn't found for some reason
            sentences = nltk.sent_tokenize(page_text)
        
        for sentence in sentences:
            sentence_length = len(sentence.split())
            if not current_chunk_sentences_info: first_page_in_chunk = page_num
            if current_chunk_length + sentence_length > max_tokens and current_chunk_sentences_info:
                chunks_with_pages.append({'text': " ".join([s_info[0] for s_info in current_chunk_sentences_info]).strip(), 'page': first_page_in_chunk})
                overlap_content_info, overlap_len = [], 0
                for s_info_idx in range(len(current_chunk_sentences_info) - 1, -1, -1):
                    s_text, s_page = current_chunk_sentences_info[s_info_idx]
                    s_len = len(s_text.split())
                    if overlap_len + s_len <= overlap_words:
                        overlap_content_info.insert(0, (s_text, s_page)); overlap_len += s_len
                    else: break
                current_chunk_sentences_info, current_chunk_length = overlap_content_info, overlap_len
                if overlap_content_info: first_page_in_chunk = overlap_content_info[0][1]
                else: first_page_in_chunk = page_num
            current_chunk_sentences_info.append((sentence, page_num)); current_chunk_length += sentence_length
    if current_chunk_sentences_info:
        chunks_with_pages.append({'text': " ".join([s_info[0] for s_info in current_chunk_sentences_info]).strip(), 'page': first_page_in_chunk})
    return [chunk for chunk in chunks_with_pages if len(chunk['text']) > 20]

def find_best_snippets(query, snippets_with_pages, snippet_embeddings, model, top_n=5):
    """Finds the top N snippets most relevant to the user's query."""
    if not snippets_with_pages or snippet_embeddings is None: return []
    query_embedding = model.encode([query])
    sim_scores = cosine_similarity(query_embedding, snippet_embeddings)[0]
    top_n_indices = np.argsort(sim_scores)[::-1][:top_n]
    return [snippets_with_pages[i] for i in top_n_indices]


# --- Main Application Class (Updated) ---
class SemanticDocumentExtractorApp:
    def __init__(self, master):
        self.master = master
        
        # --- Language Configuration ---
        self.languages = {"English": "en", "Русский": "ru"}
        self.current_language_var = tk.StringVar(master)
        self.current_language_var.set("English") # Default language
        self.current_language_var.trace_add("write", self.on_language_change)

        self.translations = {
            "en": {
                "app_title": "Semantic Document Content Extractor",
                "select_doc_label": "Selected Document", # Changed for cleaner display
                "select_doc_button": "Select Document",
                "query_label": "Enter your semantic query:",
                "find_button": "Find Relevant Content",
                "cancel_button": "Cancel Processing",
                "instructions_title": "Instructions:",
                "instructions_step1": "1. Click 'Select Document' to choose a .pdf, .docx, or .txt file.",
                "instructions_step2": "2. Wait for the processing and embedding to complete (this model supports English and Russian).",
                "instructions_step3": "3. Enter a query (in English or Russian) and click 'Find Relevant Content'.",
                "processing_doc": "Processing document...",
                "extracting_text": "Extracting text from '{}'...",
                "extraction_cancelled": "Extraction cancelled.",
                "no_text_extracted": "No text could be extracted.",
                "chunking_text": "Chunking text...",
                "chunking_cancelled": "Chunking cancelled.",
                "no_snippets": "No usable snippets were generated.",
                "snippets_created": "Created {} snippets.",
                "loading_embedding": "Loading and embedding snippets using model: {}...",
                "model_loaded": "Model loaded for embedding.",
                "model_load_error": "Failed to load SentenceTransformer model '{}': {}\n\nPlease check your internet connection and try again.",
                "embedding_cancelled": "Embedding cancelled.",
                "embedding_complete": "Embedding complete.",
                "saving_data": "Saving data to cache...",
                "data_saved": "Data saved.",
                "loading_precomputed": "Loading pre-computed data for {} (using {})...",
                "loaded_cache": "Loaded {} snippets and embeddings from cache.",
                "busy_warning": "Processing is already in progress. Please wait or cancel.",
                "processing_error_title": "Processing Error", # Added title key
                "search_error_title": "Search Error", # Added title key
                "select_doc_error": "Please select and process a document first.",
                "enter_query_error": "Please enter a query.",
                "model_not_loaded_error": "Semantic model not loaded. Please re-process the document.",
                "top_snippets_title": "\n--- Top 5 Best Fitting Snippets ---\n",
                "snippet_info": "\nSnippet {} (Page: {}):\n{}\n---",
                "snippet_info_no_page": "\nSnippet {} (Page: N/A):\n{}\n---",
                "no_snippets_found": "No relevant snippets found for your query.",
                "missing_pypdf2": "PyPDF2 is not installed (for .pdf files).\nPlease use: pip install PyPDF2",
                "missing_python_docx": "python-docx is not installed.\nProcessing .docx files will not be possible.\nPlease use: pip install python-docx",
                "missing_ai_libs": "The 'sentence-transformers' or 'scikit-learn' libraries are not installed.\nPlease install them to enable semantic search.\n\nUse: pip install sentence-transformers scikit-learn",
                "nltk_punkt_en": "Downloading 'punkt' tokenizer (English) for text processing. This is a one-time download.",
                "nltk_punkt_ru": "Downloading 'punkt' tokenizer (Russian) for text processing. This is a one-time download.",
                "pdf_error": "Could not read the PDF file. It might be corrupted or encrypted.\n\nError: {}",
                "docx_error": "Could not read the DOCX file.\n\nError: {}",
                "txt_error": "Could not read the TXT file.\n\nError: {}",
                "unsupported_format_title": "Unsupported Format", # Added title key
                "unsupported_format": "File format '{}' is not supported.",
                "missing_dependency": "The library required for '{}' files is not installed. Please check the startup warnings.",
                "choose_language": "Choosing language (Выбор языка)",
                "supported_docs_filter": "Supported Documents",
                "all_files_filter": "All files",
                "finding_snippets": "Finding snippets for '{}'...",
                "cancellation_requested": "Cancellation requested..."
            },
            "ru": {
                "app_title": "Извлекатель Содержимого Документов",
                "select_doc_label": "Выбранный Документ", # Changed for cleaner display
                "select_doc_button": "Выбрать Документ",
                "query_label": "Введите ваш семантический запрос:",
                "find_button": "Найти Релевантное Содержимое",
                "cancel_button": "Отменить Обработку",
                "instructions_title": "Инструкции:",
                "instructions_step1": "1. Нажмите 'Выбрать Документ', чтобы выбрать файл .pdf, .docx или .txt.",
                "instructions_step2": "2. Дождитесь завершения обработки и встраивания (эта модель поддерживает английский и русский языки).",
                "instructions_step3": "3. Введите запрос (на английском или русском) и нажмите 'Найти Релевантное Содержимое'.",
                "processing_doc": "Обработка документа...",
                "extracting_text": "Извлечение текста из '{}'...",
                "extraction_cancelled": "Извлечение отменено.",
                "no_text_extracted": "Текст не удалось извлечь.",
                "chunking_text": "Разбивка текста на фрагменты...",
                "chunking_cancelled": "Разбивка отменена.",
                "no_snippets": "Не удалось сгенерировать пригодные фрагменты.",
                "snippets_created": "Создано {} фрагментов.",
                "loading_embedding": "Загрузка и встраивание фрагментов с использованием модели: {}...",
                "model_loaded": "Модель загружена для встраивания.",
                "model_load_error": "Не удалось загрузить модель SentenceTransformer '{}': {}\n\nПожалуйста, проверьте ваше интернет-соединение и попробуйте снова.",
                "embedding_cancelled": "Встраивание отменено.",
                "embedding_complete": "Встраивание завершено.",
                "saving_data": "Сохранение данных в кэш...",
                "data_saved": "Данные сохранены.",
                "loading_precomputed": "Загрузка предварительно вычисленных данных для {} (используя {})...",
                "loaded_cache": "Загружено {} фрагментов и встраиваний из кэша.",
                "busy_warning": "Обработка уже выполняется. Пожалуйста, подождите или отмените.",
                "processing_error_title": "Ошибка Обработки", # Added title key
                "search_error_title": "Ошибка Поиска", # Added title key
                "select_doc_error": "Пожалуйста, сначала выберите и обработайте документ.",
                "enter_query_error": "Пожалуйста, введите запрос.",
                "model_not_loaded_error": "Семантическая модель не загружена. Пожалуйста, повторно обработайте документ.",
                "top_snippets_title": "\n--- Топ 5 наиболее подходящих фрагментов ---\n",
                "snippet_info": "\nФрагмент {} (Страница: {}):\n{}\n---",
                "snippet_info_no_page": "\nФрагмент {} (Страница: Н/Д):\n{}\n---",
                "no_snippets_found": "Не найдено релевантных фрагментов для вашего запроса.",
                "missing_pypdf2": "PyPDF2 не установлен (для файлов .pdf).\nПожалуйста, используйте: pip install PyPDF2",
                "missing_python_docx": "python-docx не установлен.\nОбработка файлов .docx будет невозможна.\nПожалуйста, используйте: pip install python-docx",
                "missing_ai_libs": "Библиотеки 'sentence-transformers' или 'scikit-learn' не установлены.\nПожалуйста, установите их, чтобы включить семантический поиск.\n\nИспользуйте: pip install sentence-transformers scikit-learn",
                "nltk_punkt_en": "Загрузка токенизатора 'punkt' (английский) для обработки текста. Это одноразовая загрузка.",
                "nltk_punkt_ru": "Загрузка токенизатора 'punkt' (русский) для обработки текста. Это одноразовая загрузка.",
                "pdf_error": "Не удалось прочитать файл PDF. Возможно, он поврежден или зашифрован.\n\nОшибка: {}",
                "docx_error": "Не удалось прочитать файл DOCX.\n\nОшибка: {}",
                "txt_error": "Не удалось прочитать файл TXT.\n\nОшибка: {}",
                "unsupported_format_title": "Неподдерживаемый Формат", # Added title key
                "unsupported_format": "Формат файла '{}' не поддерживается.",
                "missing_dependency": "Библиотека, необходимая для файлов '{}', не установлена. Пожалуйста, проверьте предупреждения при запуске.",
                "choose_language": "Выбор языка (Choosing language)",
                "supported_docs_filter": "Поддерживаемые Документы",
                "all_files_filter": "Все файлы",
                "finding_snippets": "Поиск фрагментов для '{}'...",
                "cancellation_requested": "Запрос на отмену..."
            }
        }
        self.current_lang_texts = self.translations[self.languages[self.current_language_var.get()]]

        master.geometry("1000x750")

        # Configure grid layout
        master.grid_rowconfigure(4, weight=1); master.grid_columnconfigure(0, weight=1) # Adjusted row for new dropdown

        self.doc_path = ""
        self.snippets = []
        self.snippet_embeddings = None
        # --- MODIFIED: Use a multilingual model for better Russian support ---
        self.model_name = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
        self.model = None # Model will be loaded lazily
        self.data_dir = "processed_data"
        os.makedirs(self.data_dir, exist_ok=True)
        self.processing_thread = None
        self.stop_event = threading.Event()

        # --- GUI Elements ---
        self.setup_ui()
        self.display_initial_instructions() # Display instructions only once at startup
        self.set_ui_state(during_processing=False) # Set initial state

    def get_localized_text(self, key):
        """Retrieves the localized text for a given key."""
        return self.current_lang_texts.get(key, key) # Fallback to key if not found

    def on_language_change(self, *args):
        """Callback function when the language dropdown selection changes."""
        selected_lang_name = self.current_language_var.get()
        lang_code = self.languages.get(selected_lang_name, "en") # Default to English if not found
        self.current_lang_texts = self.translations[lang_code]
        self.update_ui_texts()
        self.master.title(self.get_localized_text("app_title"))

    def update_ui_texts(self):
        """Updates all UI elements with the currently selected language."""
        self.master.title(self.get_localized_text("app_title"))
        # Update doc_path_label based on whether a document is selected
        if self.doc_path:
            self.doc_path_label.config(text=f"{self.get_localized_text('select_doc_label')}: {os.path.basename(self.doc_path)}")
        else:
            self.doc_path_label.config(text=self.get_localized_text("select_doc_label"))
            
        self.select_doc_button.config(text=self.get_localized_text("select_doc_button"))
        self.query_label.config(text=self.get_localized_text("query_label"))
        self.extract_button.config(text=self.get_localized_text("find_button"))
        self.cancel_button.config(text=self.get_localized_text("cancel_button"))
        
        # NOTE: The result_text (log area) is NOT cleared or re-populated with instructions here.
        # New messages will be in the new language, old ones remain as they were.

    def setup_ui(self):
        """Creates and places all GUI widgets."""
        # Define a consistent font size for the application
        # You can adjust this value to your preference
        self.default_font_size = 12
        self.large_font = font.Font(family="Arial", size=self.default_font_size)
        self.bold_font = font.Font(family="Arial", size=self.default_font_size, weight="bold")

        # --- Language Dropdown ---
        lang_frame = tk.Frame(self.master)
        lang_frame.grid(row=0, column=0, columnspan=2, pady=5, padx=10, sticky="ew")
        lang_frame.columnconfigure(1, weight=1)

        tk.Label(lang_frame, text=self.get_localized_text("choose_language"), font=self.large_font).grid(row=0, column=0, padx=5, sticky="w")
        self.language_dropdown = ttk.Combobox(lang_frame, textvariable=self.current_language_var,
                                              values=list(self.languages.keys()), state="readonly", font=self.large_font)
        self.language_dropdown.grid(row=0, column=1, padx=5, sticky="ew")


        # --- Document Selection ---
        select_frame = tk.Frame(self.master)
        select_frame.grid(row=1, column=0, columnspan=2, pady=10, padx=10, sticky="ew")
        select_frame.columnconfigure(0, weight=1)
        
        self.doc_path_label = tk.Label(select_frame, text="", wraplength=700, justify="left", font=self.large_font)
        self.doc_path_label.grid(row=0, column=0, padx=5, sticky="w")
        
        self.select_doc_button = tk.Button(select_frame, text="", command=self.select_document_file, font=self.large_font)
        self.select_doc_button.grid(row=0, column=1, padx=5, sticky="e")

        # --- Query Input ---
        request_frame = tk.Frame(self.master)
        request_frame.grid(row=2, column=0, columnspan=2, pady=5, padx=10, sticky="ew")
        request_frame.columnconfigure(1, weight=1)
        
        self.query_label = tk.Label(request_frame, text="", font=self.large_font)
        self.query_label.grid(row=0, column=0, padx=5, sticky="w")
        self.request_entry = tk.Entry(request_frame, width=80, state=tk.DISABLED, font=self.large_font)
        self.request_entry.grid(row=0, column=1, padx=5, sticky="ew")
        self.request_entry.bind("<Return>", lambda e: self.extract_content())

        # --- Actions and Progress Bar ---
        action_frame = tk.Frame(self.master)
        action_frame.grid(row=3, column=0, columnspan=2, pady=10, padx=10)
        
        buttons_sub_frame = tk.Frame(action_frame)
        buttons_sub_frame.pack(pady=(0, 5))
        
        self.extract_button = tk.Button(buttons_sub_frame, text="", command=self.extract_content, state=tk.DISABLED, font=self.large_font)
        self.extract_button.grid(row=0, column=0, padx=5)
        
        self.cancel_button = tk.Button(buttons_sub_frame, text="", command=self.cancel_processing, state=tk.DISABLED, font=self.large_font)
        self.cancel_button.grid(row=0, column=1, padx=5)
        
        self.progress_bar = ttk.Progressbar(action_frame, orient='horizontal', mode='determinate', length=400)
        self.progress_bar.pack()

        # --- Results Display ---
        self.result_text = scrolledtext.ScrolledText(self.master, wrap=tk.WORD, width=100, height=25, font=("Arial", self.default_font_size + 2)) # Slightly larger for readability
        self.result_text.grid(row=4, column=0, columnspan=2, pady=10, padx=10, sticky="nsew")
        
        # Initialize UI texts with default language
        self.update_ui_texts()

    def display_initial_instructions(self):
        """Displays the initial instructions in the result text area."""
        self.result_text.insert(tk.END, f"{self.get_localized_text('instructions_title')}\n")
        self.result_text.insert(tk.END, f"{self.get_localized_text('instructions_step1')}\n")
        self.result_text.insert(tk.END, f"{self.get_localized_text('instructions_step2')}\n")
        self.result_text.insert(tk.END, f"{self.get_localized_text('instructions_step3')}\n")
        self.result_text.see(tk.END) # Scroll to the end

    def show_error_message(self, title_key, message_key):
        """Helper to show localized error messages."""
        messagebox.showerror(self.get_localized_text(title_key), message_key)

    def show_warning_message(self, title_key, message_key):
        """Helper to show localized warning messages."""
        messagebox.showwarning(self.get_localized_text(title_key), message_key)

    def show_info_message(self, title_key, message_key):
        """Helper to show localized info messages."""
        messagebox.showinfo(self.get_localized_text(title_key), message_key)

    def update_status(self, message):
        self.result_text.insert(tk.END, message + "\n"); self.result_text.see(tk.END); self.master.update_idletasks()

    def update_progress_bar(self, value):
        self.progress_bar['value'] = value; self.master.update_idletasks()

    def set_ui_state(self, during_processing=False):
        # Select document button is always enabled to allow starting new processing
        self.select_doc_button.config(state=tk.NORMAL) 
        
        # Query and extract button enabled only if snippets are loaded and not processing
        can_query = not during_processing and self.snippets and self.snippet_embeddings is not None
        self.extract_button.config(state=tk.NORMAL if can_query else tk.DISABLED)
        self.request_entry.config(state=tk.NORMAL if can_query else tk.DISABLED)
        
        # Cancel button enabled only during processing
        self.cancel_button.config(state=tk.NORMAL if during_processing else tk.DISABLED)

    def select_document_file(self):
        if self.processing_thread and self.processing_thread.is_alive():
            self.show_warning_message("Busy", self.get_localized_text("busy_warning"))
            return
        file_path = filedialog.askopenfilename(
            title=self.get_localized_text("select_doc_button"), # Use localized title
            filetypes=[
                (self.get_localized_text("supported_docs_filter"), "*.pdf *.docx *.txt"), # Add localized filter name
                ("PDF files", "*.pdf"),
                ("Word Documents", "*.docx"),
                ("Text files", "*.txt"),
                (self.get_localized_text("all_files_filter"), "*.*") # Add localized filter name
            ]
        )
        if file_path:
            self.doc_path = file_path
            self.doc_path_label.config(text=f"{self.get_localized_text('select_doc_label')}: {os.path.basename(self.doc_path)}")
            # No clearing of result_text here, only update path label
            self.progress_bar['value'] = 0
            self.stop_event.clear()
            self.set_ui_state(during_processing=True)
            self.processing_thread = threading.Thread(target=self.process_document_threaded)
            self.processing_thread.start()

    def process_document_threaded(self):
        """Processes the selected document in a background thread."""
        try:
            # Generate a unique identifier for the document based on its full path
            unique_doc_id = hashlib.md5(self.doc_path.encode('utf-8')).hexdigest()
            
            self.snippets, self.snippet_embeddings = [], None
            self.master.after(0, self.update_progress_bar, 0)
            self.master.after(0, self.update_status, self.get_localized_text("processing_doc"))

            # Check cache first
            # --- MODIFIED: Include model name in cache check to avoid using old embeddings with new model ---
            model_specific_snippets_filepath = os.path.join(self.data_dir, f"{unique_doc_id}_{self.model_name.replace('/', '_')}_snippets.json")
            model_specific_embeddings_filepath = os.path.join(self.data_dir, f"{unique_doc_id}_{self.model_name.replace('/', '_')}_embeddings.npy")

            if os.path.exists(model_specific_snippets_filepath) and os.path.exists(model_specific_embeddings_filepath):
                self.master.after(0, self.update_status, self.get_localized_text("loading_precomputed").format(os.path.basename(self.doc_path), self.model_name))
                with open(model_specific_snippets_filepath, 'r', encoding='utf-8') as f: self.snippets = json.load(f)
                self.snippet_embeddings = np.load(model_specific_embeddings_filepath)
                self.master.after(0, self.update_status, self.get_localized_text("loaded_cache").format(len(self.snippets)))
                # Ensure model is loaded if not already (e.g., if app just started and cache was hit)
                if self.model is None:
                    # No message box here, just load it silently
                    self.model = SentenceTransformer(self.model_name)
                return

            # If not in cache, process from scratch
            self.master.after(0, self.update_status, self.get_localized_text("extracting_text").format(os.path.basename(self.doc_path)))
            page_data = extract_text_from_file(self.doc_path, self.stop_event, self) # Pass app instance for error handling
            if self.stop_event.is_set(): self.master.after(0, self.update_status, self.get_localized_text("extraction_cancelled")); return
            if not page_data: self.master.after(0, self.update_status, self.get_localized_text("no_text_extracted")); return

            self.master.after(0, self.update_status, self.get_localized_text("chunking_text"))
            self.snippets = chunk_text(page_data, stop_event=self.stop_event)
            if self.stop_event.is_set(): self.master.after(0, self.update_status, self.get_localized_text("chunking_cancelled")); return
            if not self.snippets: self.master.after(0, self.update_status, self.get_localized_text("no_snippets")); return
            self.master.after(0, self.update_status, self.get_localized_text("snippets_created").format(len(self.snippets)))

            self.master.after(0, self.update_status, self.get_localized_text("loading_embedding").format(self.model_name))
            # Lazy load the model here if it hasn't been loaded yet
            if self.model is None:
                try:
                    self.model = SentenceTransformer(self.model_name)
                    self.master.after(0, self.update_status, self.get_localized_text("model_loaded"))
                except Exception as e:
                    self.master.after(0, lambda: self.show_error_message("Model Load Error", self.get_localized_text("model_load_error").format(self.model_name, e)))
                    return # Exit if model fails to load

            snippet_texts = [s['text'] for s in self.snippets]
            encoded_embeddings = []
            for i, text in enumerate(snippet_texts):
                if self.stop_event.is_set(): self.master.after(0, self.update_status, self.get_localized_text("embedding_cancelled")); return
                encoded_embeddings.append(self.model.encode(text))
                self.master.after(0, self.update_progress_bar, (i + 1) * 100 / len(snippet_texts))
            self.snippet_embeddings = np.array(encoded_embeddings)
            self.master.after(0, self.update_status, self.get_localized_text("embedding_complete"))

            self.master.after(0, self.update_status, self.get_localized_text("saving_data"))
            # --- MODIFIED: Save to model-specific cache files ---
            with open(model_specific_snippets_filepath, 'w', encoding='utf-8') as f: json.dump(self.snippets, f, ensure_ascii=False, indent=4)
            np.save(model_specific_embeddings_filepath, self.snippet_embeddings)
            self.master.after(0, self.update_status, self.get_localized_text("data_saved"))
        except Exception as e:
            self.master.after(0, lambda: self.show_error_message(self.get_localized_text("processing_error_title"), self.get_localized_text("processing_error").format(e)))
        finally:
            self.master.after(0, self.set_ui_state, False)
            self.master.after(0, self.update_progress_bar, 0)

    def cancel_processing(self):
        if self.processing_thread and self.processing_thread.is_alive():
            self.stop_event.set()
            self.update_status(self.get_localized_text("cancellation_requested"))
        else:
            self.set_ui_state(False)

    def extract_content(self):
        if self.processing_thread and self.processing_thread.is_alive():
            self.show_warning_message("Busy", self.get_localized_text("busy_warning")); return
        if self.snippet_embeddings is None: # Corrected check
            self.show_error_message("Error", self.get_localized_text("select_doc_error")); return
        if not (request := self.request_entry.get().strip()):
            self.show_error_message("Error", self.get_localized_text("enter_query_error")); return

        self.result_text.delete(1.0, tk.END) # Clear previous search results
        self.update_status(self.get_localized_text("finding_snippets").format(request))
        try:
            # Ensure model is loaded before searching
            if self.model is None:
                self.show_error_message("Error", self.get_localized_text("model_not_loaded_error"))
                return

            best_snippets = find_best_snippets(request, self.snippets, self.snippet_embeddings, self.model, top_n=5)
            if best_snippets:
                self.result_text.insert(tk.END, self.get_localized_text("top_snippets_title"))
                for i, snippet_data in enumerate(best_snippets):
                    page_info = snippet_data['page'] if snippet_data['page'] else "N/A"
                    if snippet_data['page']:
                        self.result_text.insert(tk.END, self.get_localized_text("snippet_info").format(i+1, page_info, snippet_data['text']))
                    else:
                        self.result_text.insert(tk.END, self.get_localized_text("snippet_info_no_page").format(i+1, snippet_data['text']))
            else:
                self.result_text.insert(tk.END, self.get_localized_text("no_snippets_found"))
        except Exception as e:
            self.show_error_message(self.get_localized_text("search_error_title"), self.get_localized_text("search_error").format(e))

if __name__ == "__main__":
    root = tk.Tk()
    app = SemanticDocumentExtractorApp(root)
    root.mainloop()
