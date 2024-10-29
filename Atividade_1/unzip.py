import zipfile
import os

def extract(zip_file_path: str, extract_to_path: str):
    os.makedirs(extract_to_path, exist_ok=True) # extraindo os arquivos utilizando a biblioteca nativa do python zipfile
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to_path)
    print(f'Arquivos extraídos para {extract_to_path}')
    print(f'Número de arquivos extraídos: {len(os.listdir("corpus_extracted"))}')