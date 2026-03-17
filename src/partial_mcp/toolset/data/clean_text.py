import json
import re
from pathlib import Path


def clean_description(text: str) -> str:
    if not text:
        return ""

    # Убираем только секцию Raises и всё после неё
    text = re.split(r"\nRaises:", text)[0]

    # Убираем отступы и лишние переносы
    text = re.sub(r"\n\s+", " ", text).strip()
    text = re.sub(r"\s+", " ", text)

    return text


# --- загрузка registry ---
input_path = Path("tools_registry.json")
tools = json.loads(input_path.read_text(encoding="utf-8"))

# --- формируем id -> text ---
embedding_dict = {}

for tool in tools:
    tool_id = tool["id"]
    name = tool["name"]
    description = clean_description(tool.get("description", ""))
    category = tool.get("category", "uncategorized")

    text_for_embedding = f"{name}. {description}. Category: Retail"

    embedding_dict[tool_id] = text_for_embedding


# --- сохранить ---
output_path = Path("tool_dense_texts.json")
output_path.write_text(
    json.dumps(embedding_dict, indent=2, ensure_ascii=False),
    encoding="utf-8"
)

print("Created embedding dictionary for", len(embedding_dict), "tools")