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

PLANNER_PROMPT = """
You are a planning agent for an e-commerce assistant.

Your job:
Given the conversation and an existing plan,
produce the UPDATED plan that reflects the user's latest intent.

---

EXISTING PLAN:
{existing_plan}

ALREADY CALLED TOOLS:
{called_tools}

CONVERSATION:
{history}

LATEST USER MESSAGE:
{query}

---

RULES:

1. Always return a FULL plan (remaining steps only).
2. Do NOT include steps for tools that are already in ALREADY CALLED TOOLS.
3. Use abstract, high-level actions (NOT tool names).
4. Keep steps minimal (1–4 steps max).

5. Each step must include:
   - "step": action (verb + object)
   - "evidence": SHORT fragment of user message that justifies this step

6. Evidence must be copied or lightly rewritten from the user message.

7. CRITICAL: Steps must follow correct execution order:

   - FIRST: identify user (if user not yet identified)
   - THEN: fetch required data (e.g. order details)
   - THEN: validate or understand request (if needed)
   - LAST: perform any mutating action (exchange, cancel, update, return)

8. NEVER start with a mutating step (exchange, cancel, update, etc).
9. NEVER perform mutation before user identification and data retrieval.

10. If the user request involves changing something:
    ALWAYS include prerequisite steps (identify user → fetch data → then mutate)

11. Do NOT create unnecessary steps like "determine specifications"
    if the information is already provided in the user message.

12. If plan is still valid → keep it.
13. If intent changed → update plan.

14. CRITICAL: ALREADY CALLED TOOLS lists tools that have already been executed.
    Do NOT generate steps whose sole purpose is to call one of these tools again.
    Their results are already in the conversation history — skip those steps entirely.

---

EXAMPLE:

{{
  "plan": [
    {{
      "step": "identify user",
      "evidence": "my name is john_doe, zipcode 12345"
    }},
    {{
      "step": "exchange items",
      "evidence": "exchange the lamp and bottle"
    }}
  ]
}}

---

OUTPUT FORMAT (JSON ONLY):

{{
  "plan": [
    {{"step": "...", "evidence": "..."}}
  ]
}}
"""


TOP_K_PER_SUBQUERY = 2
MIN_TOOLS = 1
MAX_TOOLS_HARD_CAP = 3
SCORE_THRESHOLD = 0.42

USER_LOOKUP_TOOLS = {"find_user_id_by_name_zip", "find_user_id_by_email"}


class Toolset(CombinedToolset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.embedding_model = SentenceTransformer("BAAI/bge-m3")

        self.tool2vec: dict[str, np.ndarray] = {}
        self.descvec: dict[str, np.ndarray] = {}

        self._called_tools: set[str] = set()
        self._tool_call_counts: dict[str, int] = {}
        self.MAX_TOOL_CALLS = 50000

        self.plan: list[dict] = []
        self.have_user: bool = False
        self._first_msg: str = ""

        self.llm_client = openai.AsyncOpenAI(
            api_key="sk-ZHM89bb2lzWmkBc-0IL_GQ",
            base_url="http://lnsigo.mipt.ru:4000/v1",
        )
        self.llm_model = "Qwen/Qwen3-30B-A3B"

    def _reset_state(self) -> None:
        self._tool_call_counts = {}
        self._called_tools = set()
        self.plan = []
        self.have_user = False

    async def prepare(self):
        base_path = Path(__file__).parent
        query_file = base_path / "query_embeddings_v2.json"
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

    async def _create_plan(self, user_query: str, history: list) -> list[str]:
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

        existing_plan_str = json.dumps([s["step"] for s in self.plan], ensure_ascii=False) if self.plan else "none"
        called_tools_str = json.dumps(sorted(self._called_tools), ensure_ascii=False) if self._called_tools else "[]"

        prompt = PLANNER_PROMPT.format(
            existing_plan=existing_plan_str,
            called_tools=called_tools_str,
            history=history_str,
            query=user_query,
        )

        content = await self._llm_call(prompt)
        if content is None:
            return None

        parsed = self._parse_json_response(content)

        if isinstance(parsed, dict) and "plan" in parsed:
            steps = []
            for item in parsed["plan"]:
                if not isinstance(item, dict):
                    continue

                step = str(item.get("step", "")).strip()
                evidence = str(item.get("evidence", "")).strip()

                if 2 < len(step) < 80:
                    steps.append({
                        "step": step,
                        "evidence": evidence
                    })

            if steps:
                return steps
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

        history_len = len(history)

        # Fix #1: reset state when a new conversation starts
        first_msg = history[0].content if history else ""
        if first_msg != self._first_msg:
            self._reset_state()
            self._first_msg = first_msg

        # If user not yet identified — show only lookup tools
        if not self.have_user:
            lookup_tools = {k: v for k, v in original_tools.items() if k in USER_LOOKUP_TOOLS}
            print(f"-----------------------------------------")
            print(f"HISTORY_LEN: {history_len}  | have_user=False -> showing only lookup tools")
            print(f"USER QUERY : {user_query[:120]!r}")
            print(f"SELECTED   : {list(lookup_tools)}")
            return lookup_tools if lookup_tools else original_tools

        # Always regenerate plan to reflect completed steps
        plan = await self._create_plan(user_query, history)
        if not plan:
            print(f"-----------------------------------------")
            print(f"HISTORY_LEN: {history_len}  | have_user={self.have_user}  | CALLED: {sorted(self._called_tools) or 'none'}")
            print(f"USER QUERY : {user_query[:120]!r}")
            print(f"!! FALLBACK: empty plan -> returning ALL {len(original_tools)} tools")
            return original_tools
        self.plan = plan

        current = plan[0]

        step = current["step"]
        evidence = current.get("evidence", "")
        if len(evidence.split()) < 3:
            evidence = user_query[:100]

        sub_queries = [
            step,
            evidence,
            f"{step}. {evidence}"
        ]

        # Exclude already-used tools
        available_tools = {
            k: v for k, v in original_tools.items()
            if self._tool_call_counts.get(k, 0) < self.MAX_TOOL_CALLS
        } or original_tools


        # Stage 2
        candidates = self._retrieve(sub_queries, available_tools)

        # Stage 4
        max_tools = min(max(len(sub_queries) * TOP_K_PER_SUBQUERY, MIN_TOOLS), MAX_TOOLS_HARD_CAP)
        top_n = sorted(candidates.items(), key=lambda x: x[1], reverse=True)[:max_tools]

        selected_tools = {
            tool_id: original_tools[tool_id]
            for tool_id, _ in top_n
            if tool_id in original_tools
        }

        fallback = not selected_tools

        print(f"-----------------------------------------")
        print(f"HISTORY_LEN: {history_len}  | have_user={self.have_user}  | CALLED: {sorted(self._called_tools) or 'none'}")
        print(f"USER QUERY : {user_query[:120]!r}")
        print(f"ACTIVE STEP: {step!r}  (evidence: {evidence[:60]!r})")
        print(f"FULL PLAN  : {[s['step'] for s in plan]}")
        print(f"SUB QUERIES: {sub_queries}")
        print(f"CANDIDATES : {[(t, round(s, 3)) for t, s in sorted(candidates.items(), key=lambda x: -x[1])]}")
        print(f"MAX_TOOLS  : {max_tools}  | THRESHOLD: {SCORE_THRESHOLD}")
        print(f"SELECTED   : {list(selected_tools)}")
        if fallback:
            print(f"!! FALLBACK : selected_tools empty -> returning ALL {len(original_tools)} tools")

        return selected_tools if not fallback else original_tools

    async def call_tool(
        self,
        name: str,
        tool_args: dict[str, Any],
        ctx: RunContext,
        tool: ToolsetTool,
    ) -> Any:
        self._tool_call_counts[name] = self._tool_call_counts.get(name, 0) + 1
        result = await super().call_tool(name, tool_args, ctx, tool)
        # only reached on success (exception would propagate without updating state)
        result_preview = str(result)[:120]
        print(f"CALL_TOOL  : {name}({tool_args}) -> OK | {result_preview}")
        self._called_tools.add(name)
        if name in USER_LOOKUP_TOOLS:
            self.have_user = True
        return result

    def visit_and_replace(self, visitor):
        return self
