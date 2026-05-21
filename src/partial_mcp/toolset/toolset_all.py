from typing import Any

import logfire
from pydantic_ai.toolsets import CombinedToolset, ToolsetTool
from pydantic_ai.tools import RunContext

from .utils import extract_query_from_context


USER_LOOKUP_TOOLS = {"find_user_id_by_name_zip", "find_user_id_by_email"}


class ToolsetAll(CombinedToolset):

    def __init__(self, *args, policy: str = "", **kwargs):
        super().__init__(*args, **kwargs)
        self.policy = policy
        self._called_tools: set[str] = set()
        self._tool_call_counts: dict[str, int] = {}
        self.have_user: bool = False
        self._first_msg: str = ""

    async def prepare(self) -> None:
        pass

    def _reset_state(self) -> None:
        self._called_tools = set()
        self._tool_call_counts = {}
        self.have_user = False

    async def get_tools(self, ctx: RunContext) -> dict[str, ToolsetTool]:
        original_tools = await super().get_tools(ctx)

        user_query = extract_query_from_context(ctx)
        from .utils import extract_message_history_from_context
        history = extract_message_history_from_context(ctx)

        first_msg = history[0].content if history else ""
        if first_msg != self._first_msg:
            self._reset_state()
            self._first_msg = first_msg

        print(f"-----------------------------------------")
        print(f"[ALL] USER QUERY : {user_query[:120]!r}")
        print(f"[ALL] SELECTED   : ALL ({len(original_tools)} tools)")

        import json as _json
        with logfire.span("Toolset: selected tools", selected=_json.dumps(list(original_tools.keys())), is_fallback=False):
            pass

        return original_tools

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
        if name in USER_LOOKUP_TOOLS:
            self.have_user = True
        print(f"[ALL] CALL_TOOL  : {name}({tool_args}) -> OK | {str(result)[:120]}")
        return result

    def visit_and_replace(self, visitor):
        return self


Toolset = ToolsetAll
