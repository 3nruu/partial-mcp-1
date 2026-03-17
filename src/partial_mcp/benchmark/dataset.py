"""
Dataset
-------
This module provides a pydantic-evals dataset for the benchmark.
It uses `tasks.json` and `split_tasks.json` from `tau2`'s
`data/tau2/domains/retail/tasks.json` and `data/tau2/domains/retail/split_tasks.json` respectively.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json

import logfire
from pydantic import TypeAdapter, BaseModel, JsonValue
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import (
    Contains,
    Evaluator,
    EvaluatorContext,
    EvaluatorOutput,
)

from ..mcp_servers.retail.tools import READ_ONLY_TOOLS
from .tasks import Task, EvaluationCriteria


class ToolCall(BaseModel, frozen=True):
    """Tool call model to assist in calculating the F1 metric."""

    name: str
    arguments: dict[str, JsonValue]

    @classmethod
    def make_json_hashable(cls, obj: JsonValue, value_processor=lambda _: _) -> Any:
        """
        Turn unhashable lists and dicts into hashable frozensets recursively.

        `value_processor` can be used to modify hashable objects.
        """
        if isinstance(obj, list):
            return frozenset(
                (k, cls.make_json_hashable(v, value_processor))
                for k, v in enumerate(obj)
            )
        elif isinstance(obj, dict):
            return frozenset(
                (k, cls.make_json_hashable(v, value_processor)) for k, v in obj.items()
            )
        else:
            return value_processor(obj)

    def __eq__(self, other):
        if isinstance(other, ToolCall):
            return hash(self) == hash(other)
        return NotImplemented

    def __hash__(self):
        """
        Hash function with domain specific changes:
        - For `transfer_to_human_agents` tool arguments are ignored
        - For `find_user_id_by_name_zip` tool arguments are case-insensitive
        """
        if self.name == "transfer_to_human_agents":
            return hash((self.name, ""))
        if self.name == "find_user_id_by_name_zip":
            return hash(
                (
                    self.name,
                    self.make_json_hashable(
                        self.arguments,
                        value_processor=lambda val: val.lower()
                        if isinstance(val, str)
                        else val,
                    ),
                )
            )  # lower case name should be equivalent to capitalized name
        return hash((self.name, self.make_json_hashable(self.arguments)))


@dataclass(repr=False)
class ToolCallMetrics(Evaluator[object, object, object]):
    """Calculate tool call F1, recall, precision metrics based on the spans collected by logfire."""

    tool_calls: list[ToolCall]
    """Expected tool calls."""
    ignore_read_only: bool = False
    """Whether to ignore read only tools when calculating the score."""

# Добавить метрики на сиквенс, 
    def evaluate(
        self,
        ctx: EvaluatorContext[object, object, object],
    ) -> EvaluatorOutput:
        tool_spans = ctx.span_tree.find(
            lambda node: "MCP request: tools/call" in node.name
        )
        actual_tool_calls = set()
        for span in tool_spans:
            request = json.loads(span.attributes["request"])
            if (
                not self.ignore_read_only
                or request["params"]["name"] not in READ_ONLY_TOOLS
            ):
                actual_tool_calls.add(
                    ToolCall(
                        name=request["params"]["name"],
                        arguments=request["params"]["arguments"],
                    )
                )
        expected_tool_calls = set(
            tool_call
            for tool_call in self.tool_calls
            if (not self.ignore_read_only or tool_call.name not in READ_ONLY_TOOLS)
        )

        tp = len(actual_tool_calls & expected_tool_calls)
        fp = len(actual_tool_calls - expected_tool_calls)
        fn = len(expected_tool_calls - actual_tool_calls)

        if fp + fn > 0:
            logfire.info(
                "Tool calls do not match.",
                extras=(actual_tool_calls - expected_tool_calls),
                missing=(expected_tool_calls - actual_tool_calls),
            )

        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0.0
        )
        if self.ignore_read_only:
            prefix = "WriteOnlyTools"
        else:
            prefix = "AllTools"
        return {
            f"{prefix}F1": round(f1, 4),
            f"{prefix}Precision": precision,
            f"{prefix}Recall": recall,
        }


def get_dataset(max_cases: int | None = None):
    """
    Return pydantic-evals dataset.
    `max_cases` can be used to return the first `max_cases` cases.
    """

    def get_evaluators_from_criteria(
        criteria: EvaluationCriteria,
    ) -> tuple[Evaluator, ...]:
        return (
            *(Contains(info) for info in criteria.communicate_info),
            ToolCallMetrics(
                tool_calls=[
                    ToolCall(name=action.name, arguments=action.arguments)
                    for action in criteria.actions
                ],
                ignore_read_only=True,
            ),
            ToolCallMetrics(
                tool_calls=[
                    ToolCall(name=action.name, arguments=action.arguments)
                    for action in criteria.actions
                ],
                ignore_read_only=False,
            ),
        )

    current_dir = Path(__file__).parent
    with open(current_dir / "tasks.json", "r") as f:
        tasks: list[Task] = TypeAdapter(list[Task]).validate_json(f.read())
    with open(current_dir / "split_tasks.json", "r") as f:
        test_split = TypeAdapter(dict[str, list[str]]).validate_json(f.read())["test"]

    cases = [
        Case(
            name=task.id,
            metadata={"description": task.description.purpose},
            inputs=task.user_scenario.instructions,
            evaluators=get_evaluators_from_criteria(task.evaluation_criteria),
        )
        for task in tasks
        if task.id in test_split
    ]

    if max_cases is not None:
        cases = cases[:max_cases]

    dataset = Dataset(cases=cases)
    return dataset
