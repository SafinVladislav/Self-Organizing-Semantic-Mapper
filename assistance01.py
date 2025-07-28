from sentence_transformers import SentenceTransformer
import os

# Define a local path where you want to save the model
model_name = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
script_dir = os.path.dirname(os.path.abspath(__file__))
local_model_path = os.path.join(script_dir, 'models', model_name)


# Download the model to the specified local path
# This will create the 'models' directory and save the model there.
try:
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    model.save(local_model_path)
    print(f"Model saved to: {local_model_path}")
except Exception as e:
    print(f"Error downloading model: {e}")