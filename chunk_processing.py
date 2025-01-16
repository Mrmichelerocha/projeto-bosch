import os
import json
import csv
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Diretórios de entrada e saída
directory_text = 'text'  # Caminho onde os arquivos .txt estão salvos
directory_table = 'table'  # Caminho onde os arquivos de tabelas .csv estão salvos
directory_img = 'img'  # Caminho onde os arquivos de imagens estão salvos
chunk_size = 500  # Tamanho do chunk em palavras
output_file = 'chunked_data.json'  # Arquivo de saída para os chunks

# Função para criar chunks de texto usando LangChain TextSplitter
def chunk_text_with_langchain(text, chunk_size):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, separators=["\n\n", "\n", " "])
    chunks = text_splitter.split_text(text)
    return chunks

# Função para processar arquivos de texto
def process_text_files(directory, chunk_size):
    chunked_texts = []

    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory, filename)
            
            # Ler conteúdo do arquivo
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()

            # Criar chunks usando LangChain TextSplitter
            chunks = chunk_text_with_langchain(content, chunk_size)

            # Identificar a página a partir do nome do arquivo
            page_number = filename.split('_')[1].split('.')[0] if '_' in filename else 'unknown'

            for index, chunk in enumerate(chunks):
                chunked_texts.append(Document(
                    page_content=chunk,
                    metadata={
                        'type': 'text',
                        'filename': filename,
                        'page': page_number,
                        'chunk_index': index + 1
                    }
                ))

    return chunked_texts

# Função para processar arquivos de tabelas
def process_table_files(directory):
    chunked_tables = []

    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            file_path = os.path.join(directory, filename)

            with open(file_path, 'r', encoding='utf-8') as file:
                reader = csv.reader(file)
                headers = next(reader)
                rows = [row for row in reader]

            page_number = filename.split('_')[1] if '_' in filename else 'unknown'

            chunked_tables.append(Document(
                page_content=json.dumps({"headers": headers, "rows": rows}),
                metadata={
                    'type': 'table',
                    'filename': filename,
                    'page': page_number
                }
            ))

    return chunked_tables

# Função para processar arquivos de imagens
def process_image_files(directory):
    chunked_images = []

    for filename in os.listdir(directory):
        if filename.endswith('.png'):
            file_path = os.path.join(directory, filename)

            page_number = filename.split('_')[1] if '_' in filename else 'unknown'
            description = filename.replace('_', ' ').split('.')[0]

            ocr_file_path = file_path.replace('.png', '_ocr.txt')
            ocr_content = ""
            if os.path.exists(ocr_file_path):
                with open(ocr_file_path, 'r', encoding='utf-8') as ocr_file:
                    ocr_content = ocr_file.read()

            chunked_images.append(Document(
                page_content=ocr_content,
                metadata={
                    'type': 'image',
                    'filename': filename,
                    'page': page_number,
                    'description': description,
                    'path': file_path
                }
            ))

    return chunked_images

# Processar os arquivos
chunked_texts = process_text_files(directory_text, chunk_size)
chunked_tables = process_table_files(directory_table)
chunked_images = process_image_files(directory_img)

# Combinar todos os documentos
all_chunks = chunked_texts + chunked_tables + chunked_images

# Salvar os chunks em um arquivo JSON
with open(output_file, 'w', encoding='utf-8') as json_file:
    json.dump([{
        'page_content': doc.page_content,
        'metadata': doc.metadata
    } for doc in all_chunks], json_file, ensure_ascii=False, indent=4)

print(f"Processamento concluído! Chunks salvos em: {output_file}")
