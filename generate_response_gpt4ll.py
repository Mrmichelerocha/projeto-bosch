import sys
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from gpt4all import GPT4All  # Biblioteca para LLM local

# Caminhos
CHUNKED_DATA_FILE = 'chunked_data.json'
FAISS_INDEX_FILE = 'faiss_index.faiss'

# Carregar argumentos
question = sys.argv[1]

# Carregar dados e índice
with open(CHUNKED_DATA_FILE, 'r', encoding='utf-8') as f:
    chunked_data = json.load(f)

index = faiss.read_index(FAISS_INDEX_FILE)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Gerar embedding da pergunta
question_embedding = model.encode([question])

# Buscar os top-N chunks
k = 3  # Número de chunks relevantes

# Garantir que a matriz de embedding seja do tipo correto
question_embedding = np.array(question_embedding, dtype=np.float32)

# Realizar busca no índice
distances, indices = index.search(question_embedding, k)

# Preparar contexto e fontes
relevant_chunks = []
sources = []
for idx in indices[0]:
    chunk = chunked_data[idx]
    relevant_chunks.append(chunk['page_content'])
    sources.append(chunk.get('source', 'Desconhecido'))  # Assume que "source" contém a página ou referência

context = "\n".join(relevant_chunks)

# Carregar o modelo GPT4All
# model = GPT4All("Meta-Llama-3-8B-Instruct.Q4_0.gguf")
model = GPT4All("gpt4all-lora-quantized.bin")

# Gerar resposta
with model.chat_session():
    prompt = f"Contexto:\n{context}\n\nPergunta: {question}\nResposta:"
    response = model.generate(prompt, max_tokens=1024)

# Exibir resposta e fontes
print("Resposta:")
print(response)
print("\nFontes:")
for source in sources:
    print(f"- {source}")

