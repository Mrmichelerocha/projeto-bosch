import sys
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from gpt4all import GPT4All  # Biblioteca para LLM local

# Caminhos
CHUNKED_DATA_FILE = 'chunked_data.json'
FAISS_INDEX_FILE = 'faiss_index.faiss'

question = sys.argv[1]
with open(CHUNKED_DATA_FILE, 'r', encoding='utf-8') as f:
    chunked_data = json.load(f)

index = faiss.read_index(FAISS_INDEX_FILE)
model = SentenceTransformer('all-MiniLM-L6-v2')

question_embedding = model.encode([question])
k = 3  # Número de chunks relevantes
question_embedding = np.array(question_embedding, dtype=np.float32)
distances, indices = index.search(question_embedding, k)

relevant_chunks = []
sources = []
for idx in indices[0]:
    chunk = chunked_data[idx]
    relevant_chunks.append(chunk['page_content'])
    # sources.append(chunk.get('source', 'Desconhecido'))  

context = "\n".join(relevant_chunks)
model = GPT4All("gpt4all-lora-quantized.bin")

with model.chat_session():
    prompt = f"Contexto:\n{context}\n\nPergunta: {question}\nResposta:"
    response = model.generate(prompt, max_tokens=1024)

print("Resposta:")
print(response)
print("\nFontes:")
for source in sources:
    print(f"- {source}")

