from sentence_transformers import SentenceTransformer

# This will download the model to the specified path
model_name = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
model_path = r'C:\Users\edfer\Desktop\New_project\SemanticExtractorApp\transformer_model'
SentenceTransformer(model_name).save(model_path)

print(f"Model saved to '{model_path}'")