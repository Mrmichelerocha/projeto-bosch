import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

# Caminhos dos arquivos
input_file = 'chunked_data.json'  # Arquivo de entrada com os chunks
output_embeddings_file = 'embeddings.npy'  # Arquivo de saída para os embeddings
output_faiss_index_file = 'faiss_index.faiss'  # Arquivo para o índice FAISS
output_metadata_file = 'embeddings_with_metadata.json'  # Embeddings + metadados (opcional)

# Carregar chunks do arquivo JSON
print("Carregando chunks do arquivo JSON...")
with open(input_file, 'r', encoding='utf-8') as f:
    chunked_data = json.load(f)

# Carregar modelo de embeddings
print("Carregando modelo de embeddings...")
model = SentenceTransformer('all-MiniLM-L6-v2')  # Escolha do modelo

# Gerar embeddings para os chunks
texts = [chunk['page_content'] for chunk in chunked_data]
print(f"Gerando embeddings para {len(texts)} chunks...")
embeddings = model.encode(texts, show_progress_bar=True)

# Salvar os embeddings em um arquivo .npy
print(f"Salvando embeddings em: {output_embeddings_file}")
np.save(output_embeddings_file, embeddings)

# Criar índice FAISS
print("Criando índice FAISS...")
dimension = embeddings.shape[1]  # Dimensão dos embeddings
index = faiss.IndexFlatL2(dimension)  # Índice FAISS (distância euclidiana)

# Adicionar os embeddings ao índice
index.add(embeddings)

# Salvar o índice FAISS em arquivo
print(f"Salvando índice FAISS em: {output_faiss_index_file}")
faiss.write_index(index, output_faiss_index_file)

# Opcional: Salvar embeddings junto com metadados
print(f"Salvando embeddings com metadados em: {output_metadata_file}")
with open(output_metadata_file, 'w', encoding='utf-8') as f:
    json.dump([
        {
            'embedding': embedding.tolist(),
            'metadata': chunk['metadata']
        } for embedding, chunk in zip(embeddings, chunked_data)
    ], f, ensure_ascii=False, indent=4)

print("Embeddings e índice FAISS criados com sucesso!")

# =====================
# Validação da Busca
# =====================
# # Teste: Fazer uma consulta de exemplo
# print("\nValidação: Realizando busca de teste no índice FAISS...")

# # Exemplo de consulta
# query = "Quais fatores contribuíram para o aumento das vendas no último trimestre?"
# query_embedding = model.encode([query])  # Gera o embedding da consulta

# # Recuperar os 3 chunks mais similares
# k = 3
# distances, indices = index.search(np.array(query_embedding), k)

# # Exibir os resultados
# print("\nResultados da busca:")
# for i, idx in enumerate(indices[0]):
#     print(f"Rank {i + 1}:")
#     print(f"Chunk: {chunked_data[idx]['page_content']}")
#     print(f"Distância: {distances[0][i]:.4f}")
#     print()
