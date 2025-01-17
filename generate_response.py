import sys
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

# Configurações de caminhos
CHUNKED_DATA_FILE = 'chunked_data.json'
FAISS_INDEX_FILE = 'faiss_index.faiss'

# Verificar argumentos de entrada
if len(sys.argv) < 2:
    print("Erro: Por favor, forneça uma pergunta como argumento.\nExemplo: python generate_response.py \"Qual é o objetivo deste projeto?\"")
    sys.exit(1)

# Carregar a pergunta do usuário
question = sys.argv[1]

# Carregar dados chunked e índice FAISS
try:
    with open(CHUNKED_DATA_FILE, 'r', encoding='utf-8') as f:
        chunked_data = json.load(f)
except FileNotFoundError:
    print(f"Erro: Arquivo {CHUNKED_DATA_FILE} não encontrado.")
    sys.exit(1)

try:
    index = faiss.read_index(FAISS_INDEX_FILE)
except Exception as e:
    print(f"Erro ao carregar o índice FAISS: {e}")
    sys.exit(1)

# Inicializar modelo de embeddings
try:
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
except Exception as e:
    print(f"Erro ao carregar o modelo de embeddings: {e}")
    sys.exit(1)

# Gerar embedding para a pergunta
question_embedding = model.encode([question], convert_to_numpy=True)

# Classificação do tipo de pergunta
def classify_question(question):
    keywords_mapping = {
        "text": ["estratégia", "desempenho", "relatório"],
        "table": ["crescimento", "dados", "valores", "tabela", "percentual", "vendas", "produtos", "total"],
        "image": ["características", "design", "função", "imagem"]
    }
    
    for question_type, keywords in keywords_mapping.items():
        if any(word in question.lower() for word in keywords):
            return question_type
    return "text"  # Padrão

question_type = classify_question(question)

# Função para processar perguntas do tipo texto
def process_text_question(question_embedding, chunked_data, model):
    filtered_chunks = [
        (i, chunk) for i, chunk in enumerate(chunked_data)
        if chunk.get('metadata', {}).get('type') == 'text'
    ]

    if not filtered_chunks:
        return "Nenhum dado textual relevante encontrado."

    # Criar índice FAISS local para os chunks filtrados
    filtered_embeddings = [
        model.encode([chunk['page_content']], convert_to_numpy=True)[0]
        for _, chunk in filtered_chunks
    ]
    filtered_index = faiss.IndexFlatL2(len(filtered_embeddings[0]))
    filtered_index.add(np.array(filtered_embeddings))

    # Buscar os chunks mais relevantes
    top_k = 5
    _, indices = filtered_index.search(np.array(question_embedding, dtype=np.float32), top_k)

    relevant_chunks = [
        filtered_chunks[idx][1] for idx in indices[0] if idx < len(filtered_chunks)
    ]
    
    response = "\n".join([
        chunk.get('page_content', 'Conteúdo não encontrado')
        for chunk in relevant_chunks
        if isinstance(chunk, dict)
    ])

    return response or "Nenhum conteúdo relevante encontrado."

# Função para processar perguntas do tipo tabela
import pandas as pd
import os

def process_table_question(question, chunked_data, model):
    # Carregar todas as tabelas da pasta /table
    table_path = './table'
    all_files = [f for f in os.listdir(table_path) if f.endswith('.csv')]

    if not all_files:
        return "Nenhuma tabela encontrada na pasta /table."

    # Concatenar todas as tabelas em um único DataFrame
    dataframes = []
    for file in all_files:
        df = pd.read_csv(os.path.join(table_path, file))
        dataframes.append(df)

    if not dataframes:
        return "As tabelas estão vazias ou não foram carregadas corretamente."

    combined_df = pd.concat(dataframes, ignore_index=True)

    # Processar a pergunta
    if "total de vendas por produto" in question.lower():
        result = combined_df.groupby('Produto')['Venda'].sum()
        return result.to_string()

    elif "ano com maior venda total" in question.lower():
        result = combined_df.groupby('Ano')['Venda'].sum().idxmax()
        return f"O ano com maior venda total foi: {result}"

    elif "vendas por mes" in question.lower():
        result = combined_df.groupby('Mês')['Venda'].sum()
        return result.to_string()

    else:
        filtered_chunks = [
            chunk for chunk in chunked_data
            if chunk.get('metadata', {}).get('type') == 'table'
        ]

        if not filtered_chunks:
            return "Desculpe, não encontrei nenhuma tabela relevante para sua pergunta."

        # Extrair palavras-chave da pergunta
        keywords = [word for word in question.lower().split() if word.isalnum()]

        # Encontrar tabelas contendo palavras-chave
        relevant_tables = [
            chunk.get('page_content', '')
            for chunk in filtered_chunks
            if any(keyword in chunk.get('page_content', '').lower() for keyword in keywords)
        ]

        # Resposta amigável
        if relevant_tables:
            response = "Aqui está o que encontrei relacionado à sua pergunta:\n\n"
            response += "\n\n".join(relevant_tables)
        else:
            response = "Desculpe, não consegui encontrar informações relevantes nas tabelas."

        return response


def process_image_question(question, chunked_data, model):
    # Filtrar chunks relacionados a imagens
    image_chunks = [
        (i, chunk) for i, chunk in enumerate(chunked_data)
        if chunk.get('metadata', {}).get('type') == 'image'
    ]

    if not image_chunks:
        return "Nenhuma imagem relevante encontrada."

    # Procurar a página correspondente ao texto relacionado
    text_chunks = [
        (i, chunk) for i, chunk in enumerate(chunked_data)
        if chunk.get('metadata', {}).get('type') == 'text'
    ]

    # Extrair embeddings para o texto
    text_embeddings = [
        model.encode([chunk['page_content']], convert_to_numpy=True)[0]
        for _, chunk in text_chunks
    ]

    # Criar índice FAISS para encontrar textos relevantes
    text_index = faiss.IndexFlatL2(len(text_embeddings[0]))
    text_index.add(np.array(text_embeddings))

    # Encontrar os textos mais relevantes à pergunta
    question_embedding = model.encode([question], convert_to_numpy=True)[0]
    top_k = 5
    _, indices = text_index.search(np.array([question_embedding], dtype=np.float32), top_k)

    relevant_text_pages = {
        text_chunks[idx][1].get('metadata', {}).get('page') for idx in indices[0] if idx < len(text_chunks)
    }

    # Filtrar textos e imagens associados às páginas relevantes
    relevant_texts = [
        chunk for i, chunk in text_chunks if chunk.get('metadata', {}).get('page') in relevant_text_pages
    ]

    relevant_images = [
        chunk for i, chunk in image_chunks if chunk.get('metadata', {}).get('page') in relevant_text_pages
    ]

    if not relevant_texts and not relevant_images:
        return "Nenhum texto ou imagem correspondente encontrado para as páginas relevantes."

    # Gerar resposta com os textos e URLs das imagens encontradas
    response_texts = "\n".join([
        chunk.get('page_content', 'Conteúdo do texto não encontrado')
        for chunk in relevant_texts
    ])

    response_images = "\n".join([
        chunk.get('metadata', {}).get('path', 'URL da imagem não encontrada')
        for chunk in relevant_images
    ])

    response = "".join([
        f"Textos:\n{response_texts}\n" if response_texts else "",
        f"Imagens:\n{response_images}" if response_images else ""
    ])

    return response



# Processar a pergunta com base no tipo identificado
if question_type == "text":
    response = process_text_question(question_embedding, chunked_data, model)
elif question_type == "table":
    response = process_table_question(question, chunked_data, model)
elif question_type == "image":
    response = process_image_question(question, chunked_data, model)
else:
    response = f"Tipo de pergunta '{question_type}' não suportado no momento."

# Exibir resultado
print(f"Tipo de pergunta detectado: {question_type}")
print("Resposta gerada com base nos dados mais relevantes:\n")
print(response)
