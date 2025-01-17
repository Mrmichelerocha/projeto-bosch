import sys
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from transformers import AutoModelForCausalLM, AutoTokenizer

# Caminhos
CHUNKED_DATA_FILE = 'chunked_data.json'
FAISS_INDEX_FILE = 'faiss_index.faiss'

# Validar argumentos
if len(sys.argv) < 2:
    print("Erro: Por favor, forneça uma pergunta como argumento.\nExemplo: python generate_response.py \"Qual é o objetivo deste projeto?\"")
    sys.exit(1)

# Carregar pergunta
question = sys.argv[1]

# Carregar dados e índice
with open(CHUNKED_DATA_FILE, 'r', encoding='utf-8') as f:
    chunked_data = json.load(f)

index = faiss.read_index(FAISS_INDEX_FILE)

# Carregar modelo de embeddings
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Gerar embedding da pergunta
question_embedding = model.encode([question], convert_to_numpy=True)

# Realizar busca no índice FAISS
top_k = 5  # Alterar conforme necessário
_, indices = index.search(np.array(question_embedding, dtype=np.float32), top_k)

# Recuperar chunks relevantes com validação
relevant_chunks = [
    chunked_data[idx] for idx in indices[0] 
    if 0 <= idx < len(chunked_data)
]

# Validar se os chunks possuem a chave 'page_content'
response = "\n".join([
    chunk.get('page_content', 'Conteúdo não encontrado') 
    for chunk in relevant_chunks 
    if isinstance(chunk, dict)
])

# Depurar dados para verificar estrutura
print("Chunks retornados pela busca FAISS:")
for idx, chunk in enumerate(relevant_chunks):
    print(f"Chunk {idx}: {chunk}")

# Exibir resposta
print("Resposta gerada com base nos dados mais relevantes:\n")
print(response)
