import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox, ttk
import os
import json
import numpy as np
import threading
import hashlib
from tkinter import font
import sys
import sentence_transformers
import sklearn
import nltk

# --- Path Resolution for PyInstaller ---
def resolve_path(relative_path):
    """
    Returns the absolute path to a resource, handling PyInstaller's temporary folder.
    This function is now primarily for NLTK data and other bundled resources,
    as the transformer model will be downloaded separately.
    """
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)

# --- Dependency Checks ---
# PDF Support
try:
    import PyPDF2
except ImportError:
    messagebox.showerror("Missing Library", "PyPDF2 is not installed (for .pdf files).\nPlease use: pip install PyPDF2")

# DOCX Support
try:
    import docx
except ImportError:
    messagebox.showwarning("Missing Library", "python-docx is not installed.\nProcessing .docx files will not be possible.\nPlease use: pip install python-docx")

###
nltk_data_path = resolve_path(os.path.join('nltk_data', 'tokenizers', 'punkt'))
nltk.data.path.append(os.path.dirname(nltk_data_path)) # Append the parent directory
nltk.data.find('tokenizers/punkt')

#================================================================================
# Class: SemanticExtractorLogic
# Description: Handles all the backend data processing, independent of the UI.
#================================================================================
class SemanticExtractorLogic:
    """
    Encapsulates the core implementation of document processing and semantic search.
    This class does not contain any GUI code.
    """
    def __init__(self, status_callback=None, progress_callback=None):
        self.model_name = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
        self.model = None

        # New code for a cross-platform solution to find the user data directory
        user_data_path = None
        if sys.platform == "win32":
            # On Windows, use LOCALAPPDATA
            user_data_path = os.path.join(os.getenv('LOCALAPPDATA'), 'SemanticExtractor')
        else: # macOS and Linux
            # On other systems, use a hidden folder in the user's home directory
            user_data_path = os.path.join(os.path.expanduser('~'), '.semantic_extractor')

        self.data_dir = user_data_path
        os.makedirs(self.data_dir, exist_ok=True)

        # The model will be downloaded to this specific directory.
        self.model_local_path = os.path.join(self.data_dir, self.model_name.replace('/', '_'))
        
        # Callbacks to communicate with the UI
        self.status_callback = status_callback
        self.progress_callback = progress_callback

    def _update_status(self, message_key, *args):
        """Invokes the status callback if it's set."""
        if self.status_callback:
            self.status_callback(message_key, *args)

    def _update_progress(self, value):
        """Invokes the progress callback if it's set."""
        if self.progress_callback:
            self.progress_callback(value)

    def _load_model(self):
        """
        Loads the sentence transformer model. If it doesn't exist locally,
        it downloads it to the user's data directory.
        """
        if self.model is None:
            try:
                # Check if the model is already downloaded
                if os.path.exists(self.model_local_path) and os.listdir(self.model_local_path):
                    self._update_status("loading_embedding", self.model_name)
                    self.model = sentence_transformers.SentenceTransformer(self.model_local_path)
                    self._update_status("model_loaded")
                else:
                    self._update_status("downloading_model", self.model_name)
                    # Download the model and save it to the local path
                    self.model = sentence_transformers.SentenceTransformer(self.model_name)
                    self.model.save(self.model_local_path)
                    self._update_status("model_downloaded")
                    self._update_status("model_loaded")

            except Exception as e:
                self._update_status("model_load_error", self.model_name, e)
                raise  # Re-raise the exception to be caught by the calling thread

    # --- Text Extraction Methods ---
    def _extract_text_from_pdf(self, file_path, stop_event):
        page_data = []
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page_num, page in enumerate(reader.pages):
                if stop_event and stop_event.is_set(): return []
                if extracted_text := page.extract_text():
                    page_data.append((page_num + 1, extracted_text))
        return page_data

    def _extract_text_from_docx(self, file_path, stop_event):
        document = docx.Document(file_path)
        full_text = "\n".join([para.text for para in document.paragraphs])
        if stop_event and stop_event.is_set(): return []
        return [(1, full_text)] if full_text else []

    def _extract_text_from_txt(self, file_path, stop_event):
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            full_text = f.read()
            if stop_event and stop_event.is_set(): return []
            return [(1, full_text)] if full_text else []

    def _extract_text_from_file(self, file_path, stop_event):
        """Dispatcher to select the correct text extractor based on file extension."""
        _, extension = os.path.splitext(file_path)
        extension = extension.lower()
        
        extractor_map = {
            '.pdf': self._extract_text_from_pdf,
            '.docx': self._extract_text_from_docx,
            '.txt': self._extract_text_from_txt,
        }

        if extractor := extractor_map.get(extension):
            try:
                return extractor(file_path, stop_event)
            except NameError:
                raise RuntimeError(f"missing_dependency:{extension}")
            except Exception as e:
                error_key = f"{extension[1:]}_error" # .pdf -> pdf_error
                raise RuntimeError(f"{error_key}:{e}")
        else:
            raise ValueError(f"unsupported_format:{extension}")

    # --- Chunking and Semantic Search Methods ---
    def _chunk_text(self, page_data, max_tokens=256, overlap_words=50, stop_event=None):
        chunks_with_pages = []
        current_chunk_sentences_info = []
        current_chunk_length = 0
        first_page_in_chunk = -1

        for page_num, page_text in page_data:
            if stop_event and stop_event.is_set():
                return []

            try:
                sentences = nltk.sent_tokenize(page_text, language='russian')
            except LookupError:
                sentences = nltk.sent_tokenize(page_text)

            for sentence in sentences:
                sentence_words = sentence.split()
                sentence_length = len(sentence_words)

                # Handle the case where a single sentence is longer than max_tokens
                if sentence_length > max_tokens:
                    # Break the long sentence into smaller chunks
                    for i in range(0, sentence_length, max_tokens):
                        sub_sentence_words = sentence_words[i:i + max_tokens]
                        sub_sentence_text = " ".join(sub_sentence_words)
                        
                        # This new sub-chunk needs to be treated like a new sentence.
                        # Append the sub-chunk to the list and reset the current chunk.
                        chunks_with_pages.append({'text': sub_sentence_text.strip(), 'page': page_num})
                        
                    # After breaking down the long sentence, we continue to the next one
                    # without carrying over any content to avoid a cascade of errors.
                    current_chunk_sentences_info = []
                    current_chunk_length = 0
                    first_page_in_chunk = -1
                    continue # Skip the rest of the loop for this sentence

                # Your original logic for building chunks from sentences
                if not current_chunk_sentences_info:
                    first_page_in_chunk = page_num

                if current_chunk_length + sentence_length > max_tokens and current_chunk_sentences_info:
                    chunks_with_pages.append({
                        'text': " ".join([s_info[0] for s_info in current_chunk_sentences_info]).strip(),
                        'page': first_page_in_chunk
                    })

                    overlap_content_info, overlap_len = [], 0
                    for s_info_idx in range(len(current_chunk_sentences_info) - 1, -1, -1):
                        s_text, s_page = current_chunk_sentences_info[s_info_idx]
                        s_len = len(s_text.split())
                        if overlap_len + s_len <= overlap_words:
                            overlap_content_info.insert(0, (s_text, s_page))
                            overlap_len += s_len
                        else:
                            break

                    current_chunk_sentences_info, current_chunk_length = overlap_content_info, overlap_len
                    if overlap_content_info:
                        first_page_in_chunk = overlap_content_info[0][1]
                    else:
                        first_page_in_chunk = page_num

                current_chunk_sentences_info.append((sentence, page_num))
                current_chunk_length += sentence_length

        if current_chunk_sentences_info:
            chunks_with_pages.append({
                'text': " ".join([s_info[0] for s_info in current_chunk_sentences_info]).strip(),
                'page': first_page_in_chunk
            })

        return [chunk for chunk in chunks_with_pages if len(chunk['text']) > 20]

    def find_best_snippets(self, query, snippets, snippet_embeddings, top_n=5):
        """Finds the top N snippets most relevant to the query."""
        if not snippets or snippet_embeddings is None: return []
        self._load_model()
        query_embedding = self.model.encode([query])
        sim_scores = sklearn.metrics.pairwise.cosine_similarity(query_embedding, snippet_embeddings)[0]
        top_n_indices = np.argsort(sim_scores)[::-1][:top_n]
        return [snippets[i] for i in top_n_indices]

    def process_document(self, file_path, stop_event):
        """
        Main processing pipeline for a document. Handles caching, extraction,
        chunking, and embedding.
        """
        unique_doc_id = hashlib.md5(file_path.encode('utf-8')).hexdigest()
        model_id = self.model_name.replace('/', '_')
        snippets_path = os.path.join(self.data_dir, f"{unique_doc_id}_{model_id}_snippets.json")
        embeddings_path = os.path.join(self.data_dir, f"{unique_doc_id}_{model_id}_embeddings.npy")
        
        self._update_progress(0)
        self._update_status("processing_doc")

        # Check cache first
        if os.path.exists(snippets_path) and os.path.exists(embeddings_path):
            self._update_status("loading_precomputed", os.path.basename(file_path), self.model_name)
            with open(snippets_path, 'r', encoding='utf-8') as f:
                snippets = json.load(f)
            embeddings = np.load(embeddings_path)
            self._update_status("loaded_cache", len(snippets))
            return snippets, embeddings

        # Process from scratch
        self._update_status("extracting_text", os.path.basename(file_path))
        page_data = self._extract_text_from_file(file_path, stop_event)
        if stop_event.is_set(): self._update_status("extraction_cancelled"); return None, None
        if not page_data: self._update_status("no_text_extracted"); return None, None

        self._update_status("chunking_text")
        snippets = self._chunk_text(page_data, stop_event=stop_event)
        if stop_event.is_set(): self._update_status("chunking_cancelled"); return None, None
        if not snippets: self._update_status("no_snippets"); return None, None
        self._update_status("snippets_created", len(snippets))
        
        self._load_model() # Ensures model is loaded before embedding
        
        snippet_texts = [s['text'] for s in snippets]
        encoded_embeddings = []
        for i, text in enumerate(snippet_texts):
            if stop_event.is_set(): self._update_status("embedding_cancelled"); return None, None
            encoded_embeddings.append(self.model.encode(text))
            self._update_progress((i + 1) * 100 / len(snippet_texts))

        embeddings = np.array(encoded_embeddings)
        self._update_status("embedding_complete")

        self._update_status("saving_data")
        with open(snippets_path, 'w', encoding='utf-8') as f:
            json.dump(snippets, f, ensure_ascii=False, indent=4)
        np.save(embeddings_path, embeddings)
        self._update_status("data_saved")

        return snippets, embeddings

#================================================================================
# Class: SemanticExtractorUI
# Description: Manages the Tkinter GUI and user interactions.
#================================================================================
class SemanticExtractorUI:
    """
    Manages the entire Tkinter GUI, handling user interactions and delegating
    heavy processing to the SemanticExtractorLogic class.
    """
    def __init__(self, master):
        self.master = master
        self.logic = SemanticExtractorLogic(
            status_callback=self.update_status_from_thread,
            progress_callback=self.update_progress_bar_from_thread
        )

        # --- State Variables ---
        self.doc_path = ""
        self.snippets = []
        self.snippet_embeddings = None
        self.processing_thread = None
        self.stop_event = threading.Event()

        # --- Language and Translations ---
        self.setup_translations()

        # --- UI Setup ---
        self.master.geometry("1000x750")
        self.master.grid_rowconfigure(4, weight=1)
        self.master.grid_columnconfigure(0, weight=1)
        self.setup_ui_widgets()
        self.display_initial_instructions()
        self.set_ui_state(during_processing=False)

    def setup_translations(self):
        self.languages = {"English": "en", "Русский": "ru"}
        self.current_language_var = tk.StringVar(self.master, "English")
        self.current_language_var.trace_add("write", self.on_language_change)
        self.translations = {
             "en": {
                "app_title": "Semantic Document Content Extractor",
                "select_doc_label": "Selected Document", "select_doc_button": "Select Document",
                "query_label": "Enter your query:", "find_button": "Find Relevant Content",
                "cancel_button": "Cancel Processing", "instructions_title": "Instructions:",
                "instructions_step1": "1. Click 'Select Document' to choose a .pdf, .docx, or .txt file.",
                "instructions_step2": "2. Wait for the processing and embedding to complete (this model supports English and Russian).",
                "instructions_step3": "3. Enter a query (in English or Russian) and click 'Find Relevant Content'.",
                "processing_doc": "Processing document...", "extracting_text": "Extracting text from '{}'...",
                "extraction_cancelled": "Extraction cancelled.", "no_text_extracted": "No text could be extracted.",
                "chunking_text": "Chunking text...", "chunking_cancelled": "Chunking cancelled.",
                "no_snippets": "No usable snippets were generated.", "snippets_created": "Created {} snippets.",
                "loading_embedding": "Loading embedding model: {}...", "model_loaded": "Model loaded for embedding.",
                "downloading_model": "Model not found locally. Downloading model: {}...", "model_downloaded": "Model downloaded successfully.",
                "model_load_error": "Failed to load/download SentenceTransformer model '{}': {}\n\nPlease check your internet connection and try again.",
                "embedding_cancelled": "Embedding cancelled.", "embedding_complete": "Embedding complete.",
                "saving_data": "Saving data to cache...", "data_saved": "Data saved.",
                "loading_precomputed": "Loading pre-computed data for {} (using {})...",
                "loaded_cache": "Loaded {} snippets and embeddings from cache.",
                "busy_warning": "Processing is already in progress. Please wait or cancel.",
                "processing_error_title": "Processing Error", "search_error_title": "Search Error",
                "select_doc_error": "Please select and process a document first.", "enter_query_error": "Please enter a query.",
                "model_not_loaded_error": "Semantic model not loaded. Please re-process the document.",
                "top_snippets_title": "\n--- Top 5 Best Fitting Snippets ---\n",
                "snippet_info": "\nSnippet {} (Page: {}):\n{}\n---", "snippet_info_no_page": "\nSnippet {} (Page: N/A):\n{}\n---",
                "no_snippets_found": "No relevant snippets found for your query.",
                "pdf_error": "Could not read the PDF file. It might be corrupted or encrypted.\n\nError: {}",
                "docx_error": "Could not read the DOCX file.\n\nError: {}", "txt_error": "Could not read the TXT file.\n\nError: {}",
                "unsupported_format_title": "Unsupported Format", "unsupported_format": "File format '{}' is not supported.",
                "missing_dependency": "The library required for '{}' files is not installed. Please check the startup warnings.",
                "choose_language": "Choosing language (Выбор языка)", "supported_docs_filter": "Supported Documents",
                "all_files_filter": "All files", "finding_snippets": "Finding snippets for '{}'...",
                "cancellation_requested": "Cancellation requested..."
            },
            "ru": {
                "app_title": "Извлекатель Содержимого Документов",
                "select_doc_label": "Выбранный Документ", "select_doc_button": "Выбрать Документ",
                "query_label": "Введите ваш запрос:", "find_button": "Найти Релевантное Содержимое",
                "cancel_button": "Отменить Обработку", "instructions_title": "Инструкции:",
                "instructions_step1": "1. Нажмите 'Выбрать Документ', чтобы выбрать файл .pdf, .docx или .txt.",
                "instructions_step2": "2. Дождитесь завершения обработки и встраивания (эта модель поддерживает английский и русский языки).",
                "instructions_step3": "3. Введите запрос (на английском или русском) и нажмите 'Найти Релевантное Содержимое'.",
                "processing_doc": "Обработка документа...", "extracting_text": "Извлечение текста из '{}'...",
                "extraction_cancelled": "Извлечение отменено.", "no_text_extracted": "Текст не удалось извлечь.",
                "chunking_text": "Разбивка текста на фрагменты...", "chunking_cancelled": "Разбивка отменена.",
                "no_snippets": "Не удалось сгенерировать пригодные фрагменты.", "snippets_created": "Создано {} фрагментов.",
                "loading_embedding": "Загрузка модели встраивания: {}...", "model_loaded": "Модель загружена для встраивания.",
                "downloading_model": "Модель не найдена локально. Загрузка модели: {}...", "model_downloaded": "Модель успешно загружена.",
                "model_load_error": "Не удалось загрузить/скачать модель SentenceTransformer '{}': {}\n\nПожалуйста, проверьте ваше интернет-соединение и попробуйте снова.",
                "embedding_cancelled": "Встраивание отменено.", "embedding_complete": "Встраивание завершено.",
                "saving_data": "Сохранение данных в кэш...", "data_saved": "Данные сохранены.",
                "loading_precomputed": "Загрузка предварительно вычисленных данных для {} (используя {})...",
                "loaded_cache": "Загружено {} фрагментов и встраиваний из кэша.",
                "busy_warning": "Обработка уже выполняется. Пожалуйста, подождите или отмените.",
                "processing_error_title": "Ошибка Обработки", "search_error_title": "Ошибка Поиска",
                "select_doc_error": "Пожалуйста, сначала выберите и обработайте документ.", "enter_query_error": "Пожалуйста, введите запрос.",
                "model_not_loaded_error": "Семантическая модель не загружена. Пожалуйста, повторно обработайте документ.",
                "top_snippets_title": "\n--- Топ 5 наиболее подходящих фрагментов ---\n",
                "snippet_info": "\nФрагмент {} (Страница: {}):\n{}\n---", "snippet_info_no_page": "\nФрагмент {} (Страница: Н/Д):\n{}\n---",
                "no_snippets_found": "Не найдено релевантных фрагментов для вашего запроса.",
                "pdf_error": "Не удалось прочитать файл PDF. Возможно, он поврежден или зашифрован.\n\nОшибка: {}",
                "docx_error": "Не удалось прочитать файл DOCX.\n\nОшибка: {}", "txt_error": "Не удалось прочитать файл TXT.\n\nОшибка: {}",
                "unsupported_format_title": "Неподдерживаемый Формат", "unsupported_format": "Формат файла '{}' не поддерживается.",
                "missing_dependency": "Библиотека, необходимая для файлов '{}', не установлена. Пожалуйста, проверьте предупреждения при запуске.",
                "choose_language": "Выбор языка (Choosing language)", "supported_docs_filter": "Поддерживаемые Документы",
                "all_files_filter": "Все файлы", "finding_snippets": "Поиск фрагментов для '{}'...",
                "cancellation_requested": "Запрос на отмену..."
            }
        }
        self.current_lang_texts = self.translations["en"]

    def get_localized_text(self, key, *args):
        text = self.current_lang_texts.get(key, key)
        return text.format(*args) if args else text

    def setup_ui_widgets(self):
        # --- Fonts ---
        self.default_font_size = 12
        self.large_font = font.Font(family="Arial", size=self.default_font_size)
        
        # --- Language Dropdown ---
        lang_frame = tk.Frame(self.master); lang_frame.grid(row=0, column=0, pady=5, padx=10, sticky="ew")
        tk.Label(lang_frame, text=self.get_localized_text("choose_language"), font=self.large_font).pack(side=tk.LEFT, padx=5)
        self.language_dropdown = ttk.Combobox(lang_frame, textvariable=self.current_language_var, values=list(self.languages.keys()), state="readonly", font=self.large_font)
        self.language_dropdown.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        # --- Document Selection ---
        select_frame = tk.Frame(self.master); select_frame.grid(row=1, column=0, pady=10, padx=10, sticky="ew"); select_frame.columnconfigure(0, weight=1)
        self.doc_path_label = tk.Label(select_frame, text="", wraplength=700, justify="left", font=self.large_font)
        self.doc_path_label.grid(row=0, column=0, padx=5, sticky="w")
        self.select_doc_button = tk.Button(select_frame, command=self.select_document_file, font=self.large_font)
        self.select_doc_button.grid(row=0, column=1, padx=5, sticky="e")

        # --- Query Input ---
        request_frame = tk.Frame(self.master); request_frame.grid(row=2, column=0, pady=5, padx=10, sticky="ew"); request_frame.columnconfigure(1, weight=1)
        self.query_label = tk.Label(request_frame, font=self.large_font); self.query_label.grid(row=0, column=0, padx=5, sticky="w")
        self.request_entry = tk.Entry(request_frame, width=80, font=self.large_font); self.request_entry.grid(row=0, column=1, padx=5, sticky="ew")
        self.request_entry.bind("<Return>", lambda e: self.find_content())

        # --- Actions and Progress ---
        action_frame = tk.Frame(self.master); action_frame.grid(row=3, column=0, pady=10, padx=10)
        self.extract_button = tk.Button(action_frame, command=self.find_content, font=self.large_font); self.extract_button.grid(row=0, column=0, padx=5)
        self.cancel_button = tk.Button(action_frame, command=self.cancel_processing, font=self.large_font); self.cancel_button.grid(row=0, column=1, padx=5)
        self.progress_bar = ttk.Progressbar(action_frame, orient='horizontal', mode='determinate', length=400); self.progress_bar.grid(row=1, column=0, columnspan=2, pady=5)

        # --- Results Display ---
        self.result_text = scrolledtext.ScrolledText(self.master, wrap=tk.WORD, width=100, height=25, font=("Arial", self.default_font_size + 2))
        self.result_text.grid(row=4, column=0, pady=10, padx=10, sticky="nsew")

        self.update_ui_texts() # Initial text setup

    # --- UI Update and State Management ---
    def on_language_change(self, *args):
        lang_code = self.languages.get(self.current_language_var.get(), "en")
        self.current_lang_texts = self.translations[lang_code]
        self.update_ui_texts()

    def update_ui_texts(self):
        self.master.title(self.get_localized_text("app_title"))
        if self.doc_path:
            self.doc_path_label.config(text=f"{self.get_localized_text('select_doc_label')}: {os.path.basename(self.doc_path)}")
        else:
            self.doc_path_label.config(text=self.get_localized_text("select_doc_label"))
        self.select_doc_button.config(text=self.get_localized_text("select_doc_button"))
        self.query_label.config(text=self.get_localized_text("query_label"))
        self.extract_button.config(text=self.get_localized_text("find_button"))
        self.cancel_button.config(text=self.get_localized_text("cancel_button"))

    def set_ui_state(self, during_processing=False):
        self.select_doc_button.config(state=tk.NORMAL)
        can_query = not during_processing and self.snippets and self.snippet_embeddings is not None
        self.extract_button.config(state=tk.NORMAL if can_query else tk.DISABLED)
        self.request_entry.config(state=tk.NORMAL if can_query else tk.DISABLED)
        self.cancel_button.config(state=tk.NORMAL if during_processing else tk.DISABLED)

    def display_initial_instructions(self):
        self.result_text.insert(tk.END, f"{self.get_localized_text('instructions_title')}\n")
        self.result_text.insert(tk.END, f"{self.get_localized_text('instructions_step1')}\n")
        self.result_text.insert(tk.END, f"{self.get_localized_text('instructions_step2')}\n")
        self.result_text.insert(tk.END, f"{self.get_localized_text('instructions_step3')}\n")
        self.result_text.see(tk.END)

    def show_error_message(self, title_key, message):
        messagebox.showerror(self.get_localized_text(title_key), message)

    # --- Thread Communication ---
    def update_status_from_thread(self, message_key, *args):
        """ Schedule status updates from the worker thread. """
        self.master.after(0, self.update_status, self.get_localized_text(message_key, *args))

    def update_status(self, message):
        self.result_text.insert(tk.END, message + "\n")
        self.result_text.see(tk.END)
        self.master.update_idletasks()
        
    def update_progress_bar_from_thread(self, value):
        self.master.after(0, self.update_progress_bar, value)
        
    def update_progress_bar(self, value):
        self.progress_bar['value'] = value
        self.master.update_idletasks()

    # --- User Actions ---
    def select_document_file(self):
        if self.processing_thread and self.processing_thread.is_alive():
            self.show_error_message("busy_warning", self.get_localized_text("busy_warning"))
            return
            
        file_path = filedialog.askopenfilename(
            title=self.get_localized_text("select_doc_button"),
            filetypes=[
                (self.get_localized_text("supported_docs_filter"), "*.pdf *.docx *.txt"),
                ("PDF files", "*.pdf"), ("Word Documents", "*.docx"), ("Text files", "*.txt"),
                (self.get_localized_text("all_files_filter"), "*.*")
            ]
        )
        if file_path:
            self.doc_path = file_path
            self.update_ui_texts() # Update label with file name
            self.progress_bar['value'] = 0
            self.stop_event.clear()
            self.set_ui_state(during_processing=True)
            
            self.processing_thread = threading.Thread(target=self.process_document_threaded)
            self.processing_thread.start()

    def process_document_threaded(self):
        """Worker thread function to process the document via the logic class."""
        try:
            self.snippets, self.snippet_embeddings = self.logic.process_document(self.doc_path, self.stop_event)
        except Exception as e:
            # Handle specific errors raised by the logic class
            error_str = str(e)
            if ":" in error_str:
                error_key, error_detail = error_str.split(":", 1)
                title_key = "processing_error_title"
                if "unsupported" in error_key: title_key = "unsupported_format_title"
                self.master.after(0, self.show_error_message, title_key, self.get_localized_text(error_key, error_detail))
            else: # Generic error
                self.master.after(0, self.show_error_message, "processing_error_title", str(e))
        finally:
            self.master.after(0, self.set_ui_state, False)
            self.master.after(0, self.update_progress_bar, 0)

    def cancel_processing(self):
        if self.processing_thread and self.processing_thread.is_alive():
            self.stop_event.set()
            self.update_status(self.get_localized_text("cancellation_requested"))

    def find_content(self):
        if self.processing_thread and self.processing_thread.is_alive():
            self.show_error_message("busy_warning", self.get_localized_text("busy_warning")); return
        if self.snippet_embeddings is None:
            self.show_error_message("search_error_title", self.get_localized_text("select_doc_error")); return
        query = self.request_entry.get().strip()
        if not query:
            self.show_error_message("search_error_title", self.get_localized_text("enter_query_error")); return

        self.result_text.delete(1.0, tk.END)
        self.update_status(self.get_localized_text("finding_snippets", query))
        
        try:
            best_snippets = self.logic.find_best_snippets(query, self.snippets, self.snippet_embeddings)
            
            if best_snippets:
                self.result_text.insert(tk.END, self.get_localized_text("top_snippets_title"))
                for i, snippet in enumerate(best_snippets):
                    page_info = snippet.get('page', 'N/A')
                    text_key = "snippet_info" if page_info != 'N/A' else "snippet_info_no_page"
                    self.result_text.insert(tk.END, self.get_localized_text(text_key, i + 1, page_info, snippet['text']))
            else:
                self.result_text.insert(tk.END, self.get_localized_text("no_snippets_found"))
        except Exception as e:
            self.show_error_message("search_error_title", str(e))

if __name__ == "__main__":
    root = tk.Tk()
    app = SemanticExtractorUI(root)
    root.mainloop()