import pdfplumber
import os

def upload_e_processar_pdf():
    """
    Função para perguntar ao usuário o caminho do arquivo PDF ou fazer upload.
    Processa o arquivo PDF e retorna os dados extraídos.

    Retorno:
        dict: Dados extraídos do PDF contendo texto por página.
    """
    dados_pdf = {
        "numero_paginas": 0,
        "texto_paginas": {},
        "tabelas_paginas": {},
    }

    # Perguntar ao usuário o caminho do arquivo
    caminho_ou_upload = input("Por favor, insira o caminho completo do arquivo PDF que deseja analisar: ")

    # Verifica se o caminho ou arquivo existe
    if not os.path.exists(caminho_ou_upload):
        return f"Erro: Arquivo ou caminho '{caminho_ou_upload}' não encontrado."

    try:
        # Abre o PDF
        with pdfplumber.open(caminho_ou_upload) as pdf:
            dados_pdf["numero_paginas"] = len(pdf.pages)

            for i, pagina in enumerate(pdf.pages):
                # Extrai texto
                texto = pagina.extract_text()
                dados_pdf["texto_paginas"][f"pagina_{i+1}"] = texto

                # Extrai tabelas (se existirem)
                tabelas = pagina.extract_tables()
                if tabelas:
                    dados_pdf["tabelas_paginas"][f"pagina_{i+1}"] = tabelas

        print("PDF processado com sucesso!")
        return dados_pdf

    except Exception as e:
        return f"Erro ao processar o PDF: {e}"

# Executar a função
resultado = upload_e_processar_pdf()

# Exibindo dados processados
if isinstance(resultado, dict):
    print(f"Número de páginas: {resultado['numero_paginas']}")
    for pagina, texto in resultado['texto_paginas'].items():
        print(f"\nTexto da {pagina}:\n{texto[:200]}...\n")  # Mostra os primeiros 200 caracteres
else:
    print(resultado)
