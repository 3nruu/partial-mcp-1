import json
import re
import numpy as np
from pathlib import Path
from typing import Any
import traceback

import openai
from sentence_transformers import SentenceTransformer
from pydantic_ai.toolsets import CombinedToolset, ToolsetTool
from pydantic_ai.tools import RunContext

from .utils import extract_message_history_from_context, extract_query_from_context, AssistantMessage

DECOMPOSE_PROMPT = """
You generate search queries for retrieving API tools.

Goal:
Break the user request into ALL atomic operations needed to complete it, including intermediate data-fetching steps.

Rules:
- 2–4 queries
- Each query should be 3–8 words
- Use verb + object format
- Think step by step: what data do you need BEFORE you can perform the main action?
- Preserve entities from the request: email, username, order id, zipcode, item names, actions

Do NOT skip data retrieval steps that are required before the main action.
Do NOT overly generalize the request.

Output format:
Return ONLY a JSON array of strings.

Examples:

User: "I want to return my last order. My email is john@email.com"
Output:
[
  "find user by email",
  "get user account details",
  "get order details",
  "return order items"
]

Conversation history:
{history}

Latest user message:
{query}

JSON:
"""

VALIDATOR_PROMPT = """You are a tool selection validator.

Your job is to decide which tools are actually required to complete the user's request.

You will receive:
1) The original user message
2) A list of sub-tasks generated from that message
3) A list of candidate tools with descriptions

Rules:
- Keep ONLY tools that are directly useful for solving the user request.
- Use the original user message to understand the real intent and important entities (email, username, zipcode, order id, items, etc).
- If a tool does not clearly help complete one of the sub-tasks or the user request, remove it.
- Prefer tools that match the entities mentioned in the user message.
- Do NOT include tools that rely on information the user did not provide (for example email if only username is given).
- Return only tool names from the candidate list.
- Do NOT explain anything.

Original user message:
{user_query}

Sub-tasks:
{sub_queries}

Candidate tools:
{tools}

Return ONLY a JSON array of tool names.
Output:
"""

TOP_K_PER_SUBQUERY = 2
MIN_TOOLS = 1
MAX_TOOLS_HARD_CAP = 6
SCORE_THRESHOLD = 0.42


READ_ONLY_TOOLS = {
    "find_user_id_by_email",
    "find_user_id_by_name_zip",
    "get_user_details",
    "get_order_details",
    "get_product_details",
    "list_all_product_types",
    "calculate",
}

USER_LOOKUP = {"find_user_id_by_email", "find_user_id_by_name_zip", "get_user_details"}

TOOL_PREREQUISITES: dict[str, set[str]] = {
    "modify_pending_order_items":    USER_LOOKUP | {"get_order_details"},
    "modify_pending_order_address":  USER_LOOKUP | {"get_order_details"},
    "modify_pending_order_payment":  USER_LOOKUP | {"get_order_details"},
    "cancel_pending_order":          USER_LOOKUP | {"get_order_details"},
    "exchange_delivered_order_items": USER_LOOKUP | {"get_order_details"},
    "return_delivered_order_items":  USER_LOOKUP | {"get_order_details"},
    "modify_user_address":           USER_LOOKUP,
    "transfer_to_human_agents":      set(),
}

MUTATING_TOOLS = set(TOOL_PREREQUISITES.keys())

class Toolset(CombinedToolset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.embedding_model = SentenceTransformer("BAAI/bge-m3")

        self.tool2vec: dict[str, np.ndarray] = {}
        self.descvec: dict[str, np.ndarray] = {}

        self._called_tools: set[str] = set()
        self._tool_call_counts: dict[str, int] = {}
        self.MAX_TOOL_CALLS = 50000

        self.llm_client = openai.AsyncOpenAI(
            api_key="sk-ZHM89bb2lzWmkBc-0IL_GQ",
            base_url="http://lnsigo.mipt.ru:4000/v1",
        )
        self.llm_model = "Qwen/Qwen3-30B-A3B"

    async def prepare(self):
        base_path = Path(__file__).parent
        query_file = base_path / "query_embeddings.json"
        desc_file = base_path / "tool_embeddings.json"

        if not query_file.exists():
            raise FileNotFoundError(f"File not found: {query_file}")
        if not desc_file.exists():
            raise FileNotFoundError(f"File not found: {desc_file}")

        with open(query_file, "r", encoding="utf-8") as f:
            self.tool2vec = {k: np.array(v) for k, v in json.load(f).items()}
        with open(desc_file, "r", encoding="utf-8") as f:
            self.descvec = {k: np.array(v) for k, v in json.load(f).items()}

        print(f"Loaded tool2vec: {len(self.tool2vec)}")
        print(f"Loaded descvec:  {len(self.descvec)}")

    # ── LLM ───────────────────────────────────────────────────────────────────

    async def _llm_call(self, prompt: str) -> str | None:
        try:
            response = await self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": "/no_think\n" + prompt}],
                temperature=0,
                max_tokens=2048,
            )
            msg = response.choices[0].message
            content = msg.content or getattr(msg, "reasoning_content", None)
            return content.strip() if content else None
        except Exception as e:
            print(f"LLM call failed: {e}")
            traceback.print_exc()
            return None

    def _parse_json_response(self, content: str):
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

        fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", content)
        if fence_match:
            content = fence_match.group(1).strip()

        bracket_match = re.search(r"(\[.*\]|\{.*\})", content, re.DOTALL)
        if bracket_match:
            content = bracket_match.group(1).strip()

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return None

    # ── Stage 1: QueryDecomposer ───────────────────────────────────────────────

    async def _decompose_query(self, user_query: str, history: list) -> list[str]:
        user_msgs = []
        for msg in history:
            if getattr(msg, "kind", None) != "user":
                continue
            content = getattr(msg, "content", "")
            if isinstance(content, str):
                text = content.strip()
            elif isinstance(content, list):
                text = " ".join(
                    part.get("text", "") if isinstance(part, dict) else getattr(part, "text", "")
                    for part in content
                ).strip()
            else:
                text = str(content).strip()
            if text:
                user_msgs.append(text)

        history_str = "\n".join(user_msgs[-2:]) or "No previous messages."

        prompt = DECOMPOSE_PROMPT.format(history=history_str, query=user_query)
        #print(f"[DECOMPOSE] history_msgs={len(user_msgs)}, prompt_len={len(prompt)}")
        content = await self._llm_call(prompt)
        #print(f"[DECOMPOSE] raw_response={content!r:.300}")

        if content is None:
            return [user_query.strip()]

        parsed = self._parse_json_response(content)
        if isinstance(parsed, list) and parsed:
            valid = [q for q in parsed if isinstance(q, str) and 2 < len(q) < 100]
            if valid:
                return valid

        #print(f"[DECOMPOSE] failed, falling back to raw query")
        return [user_query.strip()]

    # ── Stage 2: Retriever ────────────────────────────────────────────────────

    def _retrieve(
        self,
        sub_queries: list[str],
        available_tools: dict[str, ToolsetTool],
    ) -> dict[str, float]:
        accum: dict[str, float] = {}

        for subquery in sub_queries:
            query_vec = self.embedding_model.encode(subquery, normalize_embeddings=True)

            scores = {}
            for tool_id in self.tool2vec:
                if tool_id not in available_tools:
                    continue
                sim_query = float(np.dot(query_vec, self.tool2vec[tool_id]))
                sim_desc = float(np.dot(query_vec, self.descvec[tool_id]))
                scores[tool_id] = 0.5 * sim_query + 0.5 * sim_desc

            top_k = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:TOP_K_PER_SUBQUERY]
            for tool_id, score in top_k:
                accum[tool_id] = accum.get(tool_id, 0.0) + score

        filtered = {t: s for t, s in accum.items() if s >= SCORE_THRESHOLD}
        return filtered if filtered else accum

    # ── Stage 3: ToolValidator ─────────────────────────────────────────────────

    async def _validate_tools(
        self,
        user_query: str,
        sub_queries: list[str],
        candidates: dict[str, float],
        original_tools: dict[str, ToolsetTool],
    ) -> dict[str, float]:
        if not candidates:
            return candidates

        tool_descriptions = []
        for tool_id in candidates:
            tool = original_tools.get(tool_id)
            description = (
                getattr(tool, "description", "")
                or getattr(getattr(tool, "tool", None), "description", "")
                or ""
            ) if tool else ""
            tool_descriptions.append(f"- {tool_id}: {description}")

        prompt = VALIDATOR_PROMPT.format(
            user_query=json.dumps(user_query),
            sub_queries=json.dumps(sub_queries),
            tools="\n".join(tool_descriptions),
        )

        content = await self._llm_call(prompt)
        if content is None:
            return candidates

        approved = self._parse_json_response(content)
        if not isinstance(approved, list):
            #print(f"[VALIDATOR] parse failed, falling back. raw={content[:200]}")
            return candidates

        validated = {t: s for t, s in candidates.items() if t in approved}
        if not validated:
            #print("[VALIDATOR] approved nothing, falling back")
            return candidates

        return validated

    # ── Stage 4: get_tools ────────────────────────────────────────────────────

    async def get_tools(self, ctx: RunContext) -> dict[str, ToolsetTool]:
        original_tools = await super().get_tools(ctx)

        if not self.tool2vec:
            return original_tools

        user_query = extract_query_from_context(ctx)
        #print(f"[GET_TOOLS] user_query={user_query!r:.100}")

        if not isinstance(user_query, str) or not user_query.strip():
            return original_tools

        history = extract_message_history_from_context(ctx)

        last_answer = next(
                (msg.content for msg in reversed(history) if isinstance(msg, AssistantMessage)),
                None,
            )

        print(f"ASSISTANT: {last_answer}")
        #print(f"HISTORY: {history}")
        history_len = len(history)
        if history_len > 0:
            sample = history[0]
           #print(f"[DEBUG] history sample: type={type(sample)}, attrs={[a for a in dir(sample) if not a.startswith('_')][:8]}")

        if history_len == 0:
            self._tool_call_counts = {}
            self._called_tools = set()

        # Stage 1
        sub_queries = await self._decompose_query(user_query, history)

        # Exclude already-used tools
        available_tools = {
            k: v for k, v in original_tools.items()
            if self._tool_call_counts.get(k, 0) < self.MAX_TOOL_CALLS
        } or original_tools

        #Gate
        def prerequisites_met(tool_id: str) -> bool:
            prereqs = TOOL_PREREQUISITES.get(tool_id) 
            if prereqs is None:
                return True
            if not prereqs:
                return True
            
            user_lookup_y = bool(self._called_tools & USER_LOOKUP)
            order_y = "get_order_details" not in prereqs or "get_order_details" in self._called_tools
            return user_lookup_y and order_y

        available_tools = {
            n: t for n, t in available_tools.items()
            if prerequisites_met(n)
        } or available_tools

        # Stage 2
        candidates = self._retrieve(sub_queries, available_tools)

        # Stage 3
        validated = await self._validate_tools(user_query, sub_queries, candidates, original_tools)

        # Stage 4
        max_tools = min(max(len(sub_queries) * TOP_K_PER_SUBQUERY, MIN_TOOLS), MAX_TOOLS_HARD_CAP)
        top_n = sorted(validated.items(), key=lambda x: x[1], reverse=True)[:max_tools]
        #top_n = list(candidates.items())[:max_tools]

        selected_tools = {
            tool_id: original_tools[tool_id]
            for tool_id, _ in top_n
            if tool_id in original_tools
        }

        print(f"USER QUERY : {user_query}")
        print(f"SUB QUERIES: {sub_queries}")
        print(f"CANDIDATES : {list(candidates)}")
        print(f"VALIDATED  : {list(validated)}")
        print(f"MAX_TOOLS  : {max_tools}")
        print(f"SELECTED   : {list(selected_tools)}")

        return selected_tools if selected_tools else original_tools

    async def call_tool(
        self,
        name: str,
        tool_args: dict[str, Any],
        ctx: RunContext,
        tool: ToolsetTool,
    ) -> Any:
        self._tool_call_counts[name] = self._tool_call_counts.get(name, 0) + 1
        result = await super().call_tool(name, tool_args, ctx, tool)
        self._called_tools.add(name)
        return result

    def visit_and_replace(self, visitor):
        return self