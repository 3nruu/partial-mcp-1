import json
import numpy as np

# -------------------------------
# Параметры
# -------------------------------
input_json1 = "query_embeddings.json"  
input_json2 = "tool_embeddings.json"  
# -------------------------------
# Загружаем user queries
# -------------------------------
with open(input_json1, "r", encoding="utf-8") as f:
    data_q = json.load(f)

with open(input_json2, "r", encoding="utf-8") as f:
    data_t = json.load(f)

ids = list(data_q.keys())

calc_q = np.array(data_q['calculate'])
calc_t = np.array(data_t['calculate'])

D = []

for i in ids:
    v_q = np.array(data_q[i])
    v_t = np.array(data_t[i])

    D.append(v_q.T@v_t)


print(np.min(D), np.max(D))

