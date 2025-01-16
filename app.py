from flask import Flask, request, jsonify
import fitz  # PyMuPDF
import pdfplumber
import os
from io import BytesIO

app = Flask(__name__)

# Diretórios para salvar os resultados extraídos
IMAGES_DIR = "img"
TEXT_DIR = "text"
TABLES_DIR = "table"

for directory in [IMAGES_DIR, TEXT_DIR, TABLES_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)

@app.route('/')
def home():
    return '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Upload PDF</title>
    </head>
    <body>
        <h1>Upload PDF para Processamento</h1>
        <form action="/process" method="POST" enctype="multipart/form-data">
            <input type="file" name="pdf_file" accept="application/pdf" required>
            <button type="submit">Processar PDF</button>
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

        # Salva o PDF em memória usando BytesIO
        pdf_bytes = BytesIO(pdf_file.read())

        # Processa texto e tabelas usando pdfplumber
        with pdfplumber.open(pdf_bytes) as pdf:
            dados_pdf["numero_paginas"] = len(pdf.pages)
            for i, pagina in enumerate(pdf.pages):
                # Extrai e salva texto
                texto = pagina.extract_text()
                if texto:
                    text_path = os.path.join(TEXT_DIR, f"page_{i+1}.txt")
                    with open(text_path, "w", encoding="utf-8") as text_file:
                        text_file.write(texto)
                    dados_pdf["textos_paginas"][f"pagina_{i+1}"] = text_path

                # Extrai e salva tabelas
                tabelas = pagina.extract_tables()
                if tabelas:
                    table_path = os.path.join(TABLES_DIR, f"page_{i+1}_table.csv")
                    with open(table_path, "w", encoding="utf-8") as table_file:
                        for tabela in tabelas:
                            for linha in tabela:
                                table_file.write(",".join(map(str, linha)) + "\n")
                    dados_pdf["tabelas_paginas"][f"pagina_{i+1}"] = table_path

        # Processa imagens usando fitz
        pdf_bytes.seek(0)  # Resetamos o stream para o fitz
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        for i in range(len(doc)):
            page = doc[i]
            imagens = []
            for img_index, img in enumerate(page.get_images(full=True)):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]

                # Define o caminho para salvar a imagem
                image_path = os.path.join(IMAGES_DIR, f"page_{i+1}_image_{img_index + 1}.{image_ext}")
                with open(image_path, "wb") as image_file:
                    image_file.write(image_bytes)

                imagens.append({"path": image_path, "type": image_ext})

            if imagens:
                dados_pdf["imagens_paginas"][f"pagina_{i+1}"] = imagens

        # Fecha o documento após o processamento
        doc.close()

        # Retorna os dados extraídos
        return jsonify(dados_pdf)

    except Exception as e:
        return f"Erro ao processar o PDF: {e}", 500

if __name__ == '__main__':
    app.run(debug=True)
