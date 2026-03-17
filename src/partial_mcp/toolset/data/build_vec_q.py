import json
from sentence_transformers import SentenceTransformer
import numpy as np

# -------------------------------
# Параметры
# -------------------------------
input_json = "user_query.json"   # твой файл с {"tool_id": ..., "user_query": [...]}
output_json = "query_embeddings.json"  # куда сохраняем tool2vec

# -------------------------------
# Загружаем user queries
# -------------------------------
with open(input_json, "r", encoding="utf-8") as f:
    data = json.load(f)

tool_queries = {item["tool_id"]: item["user_query"] for item in data}

# -------------------------------
# Загружаем модель
# -------------------------------
model = SentenceTransformer("BAAI/bge-m3")

# -------------------------------
# Генерируем embeddings и усредняем
# -------------------------------
tool2vec = {}

for tool_id, queries in tool_queries.items():
    query_embeddings = []
    for q in queries:
        prefixed_q = q
        emb = model.encode(prefixed_q, normalize_embeddings=True)
        query_embeddings.append(emb)
    
    tool_vector = np.mean(query_embeddings, axis=0)
    # Конвертируем в список, чтобы сохранить в JSON
    tool2vec[tool_id] = tool_vector.tolist()

# -------------------------------
# Сохраняем в JSON
# -------------------------------
with open(output_json, "w", encoding="utf-8") as f:
    json.dump(tool2vec, f, ensure_ascii=False, indent=2)

print(f"Saved {len(tool2vec)} tool embeddings to '{output_json}'")
