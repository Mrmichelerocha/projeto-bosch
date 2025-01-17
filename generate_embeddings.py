import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

input_file = 'chunked_data.json' 
output_embeddings_file = 'embeddings.npy'
output_faiss_index_file = 'faiss_index.faiss'
output_metadata_file = 'embeddings_with_metadata.json' 

print("Carregando chunks do arquivo JSON...")
with open(input_file, 'r', encoding='utf-8') as f:
    chunked_data = json.load(f)

print("Carregando modelo de embeddings...")
model = SentenceTransformer('all-MiniLM-L6-v2')  # modelo

# Gerar embeddings para os chunks
texts = [chunk['page_content'] for chunk in chunked_data]
print(f"Gerando embeddings para {len(texts)} chunks...")
embeddings = model.encode(texts, show_progress_bar=True)

print(f"Salvando embeddings em: {output_embeddings_file}")
np.save(output_embeddings_file, embeddings)

print("Criando índice FAISS...")
dimension = embeddings.shape[1]  
index = faiss.IndexFlatL2(dimension)  

index.add(embeddings)

# Salvar o índice FAISS em arquivo
print(f"Salvando índice FAISS em: {output_faiss_index_file}")
faiss.write_index(index, output_faiss_index_file)

print(f"Salvando embeddings com metadados em: {output_metadata_file}")
with open(output_metadata_file, 'w', encoding='utf-8') as f:
    json.dump([
        {
            'embedding': embedding.tolist(),
            'metadata': chunk['metadata']
        } for embedding, chunk in zip(embeddings, chunked_data)
    ], f, ensure_ascii=False, indent=4)

print("Embeddings e índice FAISS criados com sucesso!")
