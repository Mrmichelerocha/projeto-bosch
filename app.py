from flask import Flask, request, jsonify
import os
import sys
from subprocess import call
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
from io import BytesIO
import pdfplumber
import fitz
import easyocr
import subprocess

app = Flask(__name__)

IMAGES_DIR = "img"
TEXT_DIR = "text"
TABLES_DIR = "table"

for directory in [IMAGES_DIR, TEXT_DIR, TABLES_DIR]:
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Pasta criada: {directory}")
        else:
            print(f"Pasta já existe: {directory}")
    except Exception as e:
        print(f"Erro ao criar a pasta {directory}: {e}")


reader = easyocr.Reader(['pt', 'en'], gpu=False)

MODEL = SentenceTransformer('all-MiniLM-L6-v2')
INDEX = None
FAISS_INDEX_FILE = 'faiss_index.faiss'
CHUNKED_DATA_FILE = 'chunked_data.json'
LLM_SCRIPT = 'generate_response.py'  # Arquivo para gerar resposta com LLM

@app.route('/')
def home():
    return '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Processamento de PDF</title>
    </head>
    <body>
        <h1>Upload PDF para Processamento</h1>
        <form action="/process" method="POST" enctype="multipart/form-data">
            <input type="file" name="pdf_file" accept="application/pdf" required>
            <button type="submit">Upload PDF</button>
        </form>
        <h2>Executar Processamento</h2>
        <form action="/execute" method="POST">
            <button type="submit">Executar Chunk e Embeddings</button>
        </form>
        <h2>Faça uma Pergunta</h2>
        <form action="/ask" method="POST">
            <input type="text" name="question" placeholder="Digite sua pergunta" required>
            <button type="submit">Perguntar</button>
            <t>Por padrão está um modelo de baixa potência</t>
            <p>⚠️ Ao enviar pode demorara muito. Aguarde ⚠️ </p>
        </form>
    </body>
    </html>
    '''

@app.route('/process', methods=['POST'])
def process_pdf():
    if 'pdf_file' not in request.files:
        return "Erro: Nenhum arquivo enviado", 400

    pdf_file = request.files['pdf_file']
    if not pdf_file or pdf_file.filename == '':
        return "Erro: Nenhum arquivo selecionado", 400

    if not pdf_file.filename.endswith('.pdf'):
        return "Erro: O arquivo enviado não é um PDF", 400

    try:
        dados_pdf = {
            "numero_paginas": 0,
            "imagens_paginas": {},
            "textos_paginas": {},
            "tabelas_paginas": {}
        }

        pdf_bytes = BytesIO(pdf_file.read())

        with pdfplumber.open(pdf_bytes) as pdf:
            dados_pdf["numero_paginas"] = len(pdf.pages)
            for i, pagina in enumerate(pdf.pages):
                tabelas = pagina.extract_tables()
                texto_exclusivo = pagina.extract_text()

                if tabelas:
                    table_path = os.path.join(TABLES_DIR, f"page_{i+1}_table.csv")
                    with open(table_path, "w", encoding="utf-8") as table_file:
                        for tabela in tabelas:
                            for linha in tabela:
                                table_file.write(",".join(map(str, linha)) + "\n")

                    for tabela in tabelas:
                        tabela_texto = [" ".join(map(str, linha)) for linha in tabela]
                        for linha in tabela_texto:
                            texto_exclusivo = texto_exclusivo.replace(linha, "")

                    dados_pdf["tabelas_paginas"][f"pagina_{i+1}"] = table_path

                if texto_exclusivo.strip():
                    text_path = os.path.join(TEXT_DIR, f"page_{i+1}.txt")
                    with open(text_path, "w", encoding="utf-8") as text_file:
                        text_file.write(texto_exclusivo.strip())
                    dados_pdf["textos_paginas"][f"pagina_{i+1}"] = text_path

        pdf_bytes.seek(0) 
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        for i in range(len(doc)):
            page = doc[i]
            imagens = []
            for img_index, img in enumerate(page.get_images(full=True)):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]

                image_path = os.path.join(IMAGES_DIR, f"page_{i+1}_image_{img_index + 1}.{image_ext}")
                with open(image_path, "wb") as image_file:
                    image_file.write(image_bytes)

                ocr_text = reader.readtext(image_path, detail=0)
                ocr_text_path = os.path.join(IMAGES_DIR, f"page_{i+1}_image_{img_index + 1}_ocr.txt")
                with open(ocr_text_path, "w", encoding="utf-8") as ocr_file:
                    ocr_file.write("\n".join(ocr_text))

                imagens.append({"path": image_path, "ocr_text_path": ocr_text_path, "type": image_ext})

            if imagens:
                dados_pdf["imagens_paginas"][f"pagina_{i+1}"] = imagens

        doc.close()

        return jsonify(dados_pdf)

    except Exception as e:
        return f"Erro ao processar o PDF: {e}", 500

@app.route('/execute', methods=['POST'])
def execute_processing():
    try:
        python_executable = sys.executable  
        print("Executando chunk_processing.py...")
        result_chunk = call([python_executable, "chunk_processing.py"])
        if result_chunk != 0:
            return f"Erro ao executar chunk_processing.py. Código de saída: {result_chunk}", 500

        print("Executando generate_embeddings.py...")
        result_embeddings = call([python_executable, "generate_embeddings.py"])
        if result_embeddings != 0:
            return f"Erro ao executar generate_embeddings.py. Código de saída: {result_embeddings}", 500

        # Carregar o índice FAISS
        global INDEX
        if os.path.exists(FAISS_INDEX_FILE):
            INDEX = faiss.read_index(FAISS_INDEX_FILE)

        return "Processamento concluído com sucesso!"
    except Exception as e:
        return f"Erro ao executar o processamento: {e}", 500

@app.route('/ask', methods=['POST'])
def ask_question():
    question = request.form.get('question')
    if not question:
        print("Erro: Nenhuma pergunta enviada.")
        return "Erro: Nenhuma pergunta enviada.", 400

    print(f"Pergunta recebida: {question}")

    try:
        python_executable = sys.executable
        print("Executando script LLM...")

        # Usar Popen para capturar saída em tempo real
        process = subprocess.Popen(
            [python_executable, "-u", LLM_SCRIPT, question],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=1,
            text=True
        )

        output = []  
        for line in iter(process.stdout.readline, ''):
            output.append(line.strip())  
            print(line.strip(), flush=True)  

        process.wait() 
        stderr = process.stderr.read() 

        if process.returncode != 0:
            print(f"Erro ao executar o script LLM: {stderr.strip()}")
            return f"Erro ao gerar a resposta: {stderr.strip()}", 500

        full_output = "\n".join(output)
        print("Script LLM executado com sucesso!")
        return full_output, 200
    except Exception as e:
        print(f"Erro ao processar a pergunta: {e}")
        return f"Erro ao processar a pergunta: {e}", 500

if __name__ == '__main__':
    print("Iniciando o servidor Flask...")
    app.run(debug=True)