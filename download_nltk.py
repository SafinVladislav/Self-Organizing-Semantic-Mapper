import nltk
nltk.download('punkt', quiet=True)
print("NLTK 'punkt' tokenizer downloaded.")
print(nltk.data.find('tokenizers/punkt'))

#--add-data "C:\Users\edfer\Desktop\New_project\transformer_model;transformer_model" --add-data "C:\Users\edfer\AppData\Roaming\nltk_data\tokenizers\punkt;nltk_data\tokenizers\punkt"
#pyinstaller --onedir --windowed --add-data "C:\Users\edfer\Desktop\New_project\transformer_model;transformer_model" --add-data "C:\Users\edfer\AppData\Roaming\nltk_data\tokenizers\punkt;nltk_data\tokenizers\punkt" --hidden-import=transformers app.py
#pyinstaller --onedir --windowed --add-data "C:\Users\edfer\Desktop\New_project\transformer_model;transformer_model" --add-data "C:\Users\edfer\AppData\Roaming\nltk_data\tokenizers\punkt;nltk_data\tokenizers\punkt" --additional-hooks-dir=./hooks app.py