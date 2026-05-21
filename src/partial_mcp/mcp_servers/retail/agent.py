"""
User Agent
----------
This module provides an evaluatee agent used in `benchmark.py` to simulate a conversation.
The agent uses `policy.md` from `tau2`'s `data/tau2/domains/retail/policy.md`.
"""

from pathlib import Path

from pydantic_ai import Agent, AbstractToolset

from ...toolset.toolset_last import Toolset
from ...model import model


async def get_agent(toolsets: list[AbstractToolset]) -> Agent[None, str]:
    with open(Path(__file__).parent / "policy.md", "r") as f:
        prompt = f.read()

    toolset = Toolset(toolsets)
    await toolset.prepare()

    agent = Agent(
        model=model,
        instructions=prompt,
        toolsets=[toolset],
    )

    return agent
