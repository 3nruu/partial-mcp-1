import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
import numpy as np

# ====== пути ======
INPUT_PATH = Path("tool_dense_texts.json")   # id -> text
OUTPUT_PATH = Path("tool_embeddings.npy")    # numpy массив
ID_PATH = Path("tool_ids.json")              # порядок id

# ====== загрузка данных ======
data = json.loads(INPUT_PATH.read_text(encoding="utf-8"))

ids = list(data.keys())
texts = list(data.values())

print(f"Loaded {len(ids)} tools")

# ====== модель ======
model = SentenceTransformer("BAAI/bge-m3")

# ВАЖНО для BGE моделей:
texts = [f"{t}" for t in texts]

# ====== получение эмбеддингов ======
embeddings = model.encode(
    texts,
    batch_size=32,
    convert_to_numpy=True,
    normalize_embeddings=True  # важно для cosine similarity
)

print("Embedding shape:", embeddings.shape)

# ====== сохранение ======
embedding_dict = {
    tool_id: embedding.tolist()
    for tool_id, embedding in zip(ids, embeddings)
}

Path("tool_embeddings.json").write_text(
    json.dumps(embedding_dict, indent=2),
    encoding="utf-8"
)

print("Saved embeddings and id list")
