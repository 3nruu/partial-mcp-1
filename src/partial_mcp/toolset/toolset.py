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

# ── Prompts ───────────────────────────────────────────────────────────────────

DECOMPOSE_PROMPT = """You are an intent planner for an e-commerce support agent.

Goal:
Given the conversation history, the latest user message, and the current progress state,
identify 1–3 REMAINING intents that the agent still needs to execute.

Current progress:
{state}

Rules:
- Only output intents for steps that are NOT yet completed.
- If the user is already identified (user_identified = true), do NOT output intents with needs_user_lookup = true.
  Instead, focus on the next steps: fetching order details, performing exchanges, etc.
- If an order is already loaded (order_loaded = true), do NOT output intents with needs_order_details = true.
  Instead, focus on the actual mutation or information retrieval.
- If both are done and a mutating action is needed, output the mutation intent directly.
- Focus on high-level intents, not micro-steps.
- Use the original user request to decide if the intent changes data (mutating) or only reads data.
- Preserve entities from the request: email, username, order id, zipcode, item names, actions.
- The description field is used as a search query for tool retrieval, so make it specific and action-oriented.

Output:
Return ONLY a JSON array of objects with fields:
- name: short snake_case name of intent
- description: 1–2 sentence description (3–8 words, verb + object format, useful as a search query for tool retrieval)
- is_mutating: boolean — true if the action creates, updates, or deletes data; false if it only reads
- needs_user_lookup: boolean — true if executing this intent requires identifying the user first (and user is NOT yet identified)
- needs_order_details: boolean — true if executing this intent requires fetching a specific order (and order is NOT yet loaded)

JSON schema for one intent object:
{{
  "name": "<snake_case>",
  "description": "<short verb+object phrase>",
  "is_mutating": <true|false>,
  "needs_user_lookup": <true|false>,
  "needs_order_details": <true|false>
}}

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
2) A list of intents (high-level goals) derived from that message
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

Intents:
{intents}

Candidate tools:
{tools}

Return ONLY a JSON array of tool names.
Output:
"""

# ── Constants ─────────────────────────────────────────────────────────────────

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

USER_LOOKUP = {"find_user_id_by_email", "find_user_id_by_name_zip"}

TOOL_PREREQUISITES: dict[str, set[str]] = {
    "modify_pending_order_items":     USER_LOOKUP | {"get_order_details"},
    "modify_pending_order_address":   USER_LOOKUP | {"get_order_details"},
    "modify_pending_order_payment":   USER_LOOKUP | {"get_order_details"},
    "cancel_pending_order":           USER_LOOKUP | {"get_order_details"},
    "exchange_delivered_order_items":  USER_LOOKUP | {"get_order_details"},
    "return_delivered_order_items":    USER_LOOKUP | {"get_order_details"},
    "modify_user_address":            USER_LOOKUP,
    "transfer_to_human_agents":       set(),
}

MUTATING_TOOLS = set(TOOL_PREREQUISITES.keys())


# ── Default intent (fallback) ─────────────────────────────────────────────────

def _default_intent(query: str) -> dict:
    return {
        "name": "unknown",
        "description": query.strip(),
        "is_mutating": False,
        "needs_user_lookup": False,
        "needs_order_details": False,
    }


class Toolset(CombinedToolset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.embedding_model = SentenceTransformer("BAAI/bge-m3")

        self.tool2vec: dict[str, np.ndarray] = {}
        self.descvec: dict[str, np.ndarray] = {}

        self._called_tools: set[str] = set()
        self._tool_call_counts: dict[str, int] = {}
        self.MAX_TOOL_CALLS = 50000

        # ── State graph flags ─────────────────────────────────────────────
        self.have_user: bool = False
        self.have_order: bool = False

        self.llm_client = openai.AsyncOpenAI(
            api_key="sk-ZHM89bb2lzWmkBc-0IL_GQ",
            base_url="http://lnsigo.mipt.ru:4000/v1",
        )
        self.llm_model = "Qwen/Qwen3-30B-A3B"

    # ── State helpers ─────────────────────────────────────────────────────────

    def _update_state(self, tool_name: str) -> None:
        """Transition the state graph after a tool call."""
        if tool_name in USER_LOOKUP:
            self.have_user = True
        if tool_name == "get_order_details":
            self.have_order = True

    def _is_ready_for_mutation(self, intent: dict) -> bool:
        """Check if the current state allows mutating actions for this intent."""
        if not intent.get("is_mutating"):
            return False
        if intent.get("needs_user_lookup") and not self.have_user:
            return False
        if intent.get("needs_order_details") and not self.have_order:
            return False
        return True

    def _prerequisites_met(self, tool_id: str, primary_intent: dict) -> bool:
        """
        Graph-based gate: decides whether a tool is allowed given
        the current state + the primary intent.
        """
        # 1) Global ban on mutating tools when context is insufficient
        if tool_id in MUTATING_TOOLS:
            if not self._is_ready_for_mutation(primary_intent):
                return False

        # 2) Cannot call get_user_details without having identified the user
        if tool_id == "get_user_details" and not self.have_user:
            return False

        # 3) Cannot call get_order_details without having identified the user
        if tool_id == "get_order_details" and not self.have_user:
            return False

        # 4) Standard prereqs from the prerequisite map
        prereqs = TOOL_PREREQUISITES.get(tool_id)
        if prereqs:
            if USER_LOOKUP & prereqs and not self.have_user:
                return False
            if "get_order_details" in prereqs and not self.have_order:
                return False

        return True

    # ── Prepare ───────────────────────────────────────────────────────────────

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

    # ── Stage 1: Intent Decomposer ────────────────────────────────────────────

    async def _decompose_query(self, user_query: str, history: list) -> list[dict]:
        """
        Returns a list of intent dicts, each with:
          name, description, is_mutating, needs_user_lookup, needs_order_details

        State-aware: tells the LLM what's already accomplished so it only
        generates remaining steps.
        """
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

        state_str = (
            f"- user_identified: {self.have_user}  "
            f"{'(user account already found, no need to look up again)' if self.have_user else '(user not yet identified)'}\n"
            f"- order_loaded: {self.have_order}  "
            f"{'(order details already fetched)' if self.have_order else '(order not yet fetched — may need get_order_details)'}\n"
            f"- tools_already_called: {sorted(self._called_tools) if self._called_tools else 'none'}"
        )

        prompt = DECOMPOSE_PROMPT.format(
            state=state_str,
            history=history_str,
            query=user_query,
        )
        content = await self._llm_call(prompt)

        if content is None:
            return [_default_intent(user_query)]

        parsed = self._parse_json_response(content)

        # Handle dict-wrapped responses like {"intents": [...]}
        if isinstance(parsed, dict):
            for key in ("intents", "intent", "results", "output"):
                if key in parsed and isinstance(parsed[key], list):
                    parsed = parsed[key]
                    break

        if isinstance(parsed, list) and parsed:
            valid_intents = []
            for item in parsed:
                if isinstance(item, dict) and ("name" in item or "description" in item):
                    valid_intents.append({
                        "name": item.get("name", "unknown"),
                        "description": item.get("description", ""),
                        "is_mutating": bool(item.get("is_mutating", False)),
                        "needs_user_lookup": bool(item.get("needs_user_lookup", False)),
                        "needs_order_details": bool(item.get("needs_order_details", False)),
                    })
            if valid_intents:
                return self._postprocess_intents(valid_intents)

        return [_default_intent(user_query)]

    def _postprocess_intents(self, intents: list[dict]) -> list[dict]:
        """
        Safety net: override intent flags based on actual state.
        If user is already identified, needs_user_lookup must be False, etc.
        This guards against LLM ignoring state context in the prompt.
        """
        for intent in intents:
            if self.have_user:
                intent["needs_user_lookup"] = False
            if self.have_order:
                intent["needs_order_details"] = False
        return intents

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
        intents: list[dict],
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

        intents_str = json.dumps(
            [{"name": i.get("name"), "description": i.get("description")} for i in intents],
            ensure_ascii=False,
        )

        prompt = VALIDATOR_PROMPT.format(
            user_query=json.dumps(user_query),
            intents=intents_str,
            tools="\n".join(tool_descriptions),
        )

        content = await self._llm_call(prompt)
        if content is None:
            return candidates

        approved = self._parse_json_response(content)
        if not isinstance(approved, list):
            return candidates

        validated = {t: s for t, s in candidates.items() if t in approved}
        if not validated:
            return candidates

        return validated

    # ── Prerequisite expansion (graph walk) ─────────────────────────────────

    def _expand_prerequisites(
        self,
        blocked: dict[str, ToolsetTool],
        original: dict[str, ToolsetTool],
    ) -> tuple[dict[str, ToolsetTool], list[str]]:
        """
        For every tool in `blocked` (tools that matched retrieval but can't
        run yet), walk TOOL_PREREQUISITES backwards and collect prerequisite
        tools that haven't been called yet.

        Returns (prerequisite_tools, list_of_injected_names).
        """
        prereq_tools: dict[str, ToolsetTool] = {}
        injected: list[str] = []

        # Seed: all blocked tool ids
        frontier = set(blocked.keys())
        visited: set[str] = set()

        while frontier:
            tool_id = frontier.pop()
            if tool_id in visited:
                continue
            visited.add(tool_id)

            for prereq in TOOL_PREREQUISITES.get(tool_id, set()):
                # Already called in a previous turn → satisfied
                if prereq in self._called_tools:
                    continue
                # Not in the original toolset → can't add
                if prereq not in original:
                    continue
                # Already collected → skip
                if prereq in prereq_tools:
                    continue

                prereq_tools[prereq] = original[prereq]
                injected.append(prereq)

                # The prereq itself may have further prerequisites
                frontier.add(prereq)

        return prereq_tools, injected

    # ── get_tools ─────────────────────────────────────────────────────────────

    async def get_tools(self, ctx: RunContext) -> dict[str, ToolsetTool]:
        original_tools = await super().get_tools(ctx)

        if not self.tool2vec:
            return original_tools

        user_query = extract_query_from_context(ctx)

        if not isinstance(user_query, str) or not user_query.strip():
            return original_tools

        history = extract_message_history_from_context(ctx)

        last_answer = next(
            (msg.content for msg in reversed(history) if isinstance(msg, AssistantMessage)),
            None,
        )

        print(f"ASSISTANT: {last_answer}")

        history_len = len(history)
        if history_len == 0:
            self._tool_call_counts = {}
            self._called_tools = set()
            self.have_user = False
            self.have_order = False

        # ── Stage 1: Intent decomposition ─────────────────────────────────
        intents = await self._decompose_query(user_query, history)
        primary_intent = intents[0] if intents else _default_intent(user_query)

        # Derive sub-queries for embedding retrieval
        sub_queries = [
            intent.get("description") or intent.get("name")
            for intent in intents
        ] or [user_query.strip()]

        # Aggregate intent flags
        any_mutating = any(i.get("is_mutating") for i in intents)

        # ── Call-count filter only (no state gating yet) ──────────────────
        retrieval_pool = {
            k: v for k, v in original_tools.items()
            if self._tool_call_counts.get(k, 0) < self.MAX_TOOL_CALLS
        } or original_tools

        # Rule 3: hide mutating tools from retrieval if no mutating intent
        if not any_mutating:
            retrieval_pool = {
                n: t for n, t in retrieval_pool.items()
                if n not in MUTATING_TOOLS
            } or retrieval_pool

        # ── Stage 2: Embedding retrieval (against full pool) ──────────────
        candidates = self._retrieve(sub_queries, retrieval_pool)

        # ── Stage 3: LLM validation ──────────────────────────────────────
        validated = await self._validate_tools(
            user_query, intents, candidates, original_tools
        )

        # ── Stage 4: Top-N selection ─────────────────────────────────────
        max_tools = min(
            max(len(sub_queries) * TOP_K_PER_SUBQUERY, MIN_TOOLS),
            MAX_TOOLS_HARD_CAP,
        )
        top_n = sorted(validated.items(), key=lambda x: x[1], reverse=True)[:max_tools]

        all_matched = {
            tool_id: original_tools[tool_id]
            for tool_id, _ in top_n
            if tool_id in original_tools
        }

        # ── Stage 5: Split by readiness + expand prerequisites ───────────
        #   allowed  = tools whose prerequisites are already satisfied
        #   blocked  = tools that matched but can't run yet (state not ready)
        allowed: dict[str, ToolsetTool] = {}
        blocked: dict[str, ToolsetTool] = {}

        for tool_id, tool in all_matched.items():
            if self._prerequisites_met(tool_id, primary_intent):
                allowed[tool_id] = tool
            else:
                blocked[tool_id] = tool

        # Walk the graph backwards from blocked tools → inject their prereqs
        prereq_tools, injected = self._expand_prerequisites(blocked, original_tools)

        selected_tools = {**allowed, **prereq_tools}

        print(f"USER QUERY : {user_query}")
        print(f"INTENTS    : {json.dumps(intents, ensure_ascii=False)}")
        print(f"SUB QUERIES: {sub_queries}")
        print(f"STATE      : have_user={self.have_user}, have_order={self.have_order}")
        print(f"MATCHED    : {list(all_matched)}")
        print(f"ALLOWED    : {list(allowed)}")
        print(f"BLOCKED    : {list(blocked)}")
        print(f"INJECTED   : {injected}")
        print(f"SELECTED   : {list(selected_tools)}")

        return selected_tools if selected_tools else original_tools

    # ── call_tool with state transition ───────────────────────────────────────

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
        self._update_state(name)
        return result

    def visit_and_replace(self, visitor):
        return self