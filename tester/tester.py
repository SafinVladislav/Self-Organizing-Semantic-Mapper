# tester.py

import os
import time
###
from semantic_algorithm import Pipeline

# --- CONFIGURATION ---
# The document you want to test the algorithm on.
# Make sure this file is in the same folder as this script.
DOCUMENT_TO_TEST = "The-art-of-seduction-robert-greene.pdf"

# A sample query to verify the search functionality.
TEST_QUERY = "What are the ethics of seduction?"

def run_test():
    """
    Executes the performance and correctness test.
    """
    print("="*80)
    print("  ALGORITHM TESTER")
    print("="*80)

    # --- Step 1: Check if the document exists ---
    if not os.path.exists(DOCUMENT_TO_TEST):
        print(f"FATAL ERROR: The test document '{DOCUMENT_TO_TEST}' was not found.")
        print("Please place the PDF file in the same directory as this script.")
        # Create a dummy file for demonstration purposes if it doesn't exist
        print("Creating a dummy 'Big_Book.pdf' for demonstration purposes...")
        try:
            from reportlab.pdfgen import canvas
            from reportlab.lib.pagesizes import letter
            c = canvas.Canvas(DOCUMENT_TO_TEST, pagesize=letter)
            c.drawString(100, 750, "This is a test document about the ethics of AI.")
            c.drawString(100, 735, "Artificial intelligence could have a major impact on society.")
            c.showPage()
            c.save()
            print("Dummy PDF created successfully.")
        except ImportError:
            print("\nCould not create dummy PDF. Please install reportlab (`pip install reportlab`)")
            print("Or provide your own 'Big_Book.pdf'. Exiting.")
            return
        except Exception as e:
            print(f"\nAn error occurred while creating the dummy PDF: {e}")
            return


    # --- Step 2: Initialize the algorithm pipeline ---
    print("\n[INITIALIZING PIPELINE]")
    pipeline = Pipeline()
    
    # --- Step 3: Process the document and measure performance ---
    print("\n[PROCESSING DOCUMENT]")
    try:
        embedding_time = pipeline.process_document(DOCUMENT_TO_TEST)
        
        if embedding_time > 0:
            print("\n--- PERFORMANCE RESULT ---")
            print(f"Time to embed document: {embedding_time:.2f} seconds")
            #print(f"Number of text chunks processed: {len(pipeline.chunks)}")
            #if pipeline.embeddings is not None:
            #    print(f"Shape of embedding matrix: {pipeline.embeddings.shape}")
            #print("--------------------------\n")
        else:
            print("\n--- TEST FAILED: Document processing did not complete successfully. ---")
            return

    except FileNotFoundError:
        print(f"ERROR: The file '{DOCUMENT_TO_TEST}' was not found during processing.")
        return
    except Exception as e:
        print(f"\nAn unexpected error occurred during document processing: {e}")
        return

    # --- Step 4: Run a test query to verify correctness ---
    print("[VERIFYING SEARCH FUNCTIONALITY]")
    print(f"Running test query: \"{TEST_QUERY}\"")
    
    start_time = time.time()
    search_results = pipeline.search(TEST_QUERY, top_n=3)
    end_time = time.time()
    
    print(f"Search completed in {end_time - start_time:.4f} seconds.")

    if not search_results:
        print("\n--- VERIFICATION FAILED: Search returned no results. ---")
    else:
        print("\n--- TOP 3 SEARCH RESULTS ---")
        for i, result in enumerate(search_results):
            # Shorten the chunk for cleaner display
            snippet = result['chunk'].replace('\n', ' ').strip()
            snippet = (snippet[:200] + '...') if len(snippet) > 200 else snippet
            
            print(f"\n{i+1}. Score: {result['score']:.4f}")
            print(f"   Chunk: \"{snippet}\"")
        print("----------------------------")
        print("\nVerification complete. Check if the results are relevant to the query.")

    print("\n" + "="*80)
    print("  TEST RUN FINISHED")
    print("="*80)


if __name__ == "__main__":
    run_test()