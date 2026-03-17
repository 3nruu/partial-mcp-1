"""
User Agent
----------
This module provides a user agent used in `benchmark.py` to simulate a conversation.
The user agent uses `user_agent_prompt.md` from `tau2`'s `data/tau2/user_simulator/simulation_guidelines.md`.
"""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel
from pydantic_ai import Agent, RunContext

from ..model import model
from .tasks import UserInstructions


class EndConversation(BaseModel):
    """
    End the conversation with customer service agent.
    Do not use this without a good reason.
    """

    reason: Literal["complete", "stuck", "transfer"]
    """
    Reason behind ending the conversation.
    - "complete" means that the customer service agent has fully satisfied your requests.
    - "stuck" means that you feel like the customer service agent is no longer able to satisfy your requests.
    - "transfer" should be used if the customer service agent has notified you about transferring you to another agent.
    """


def get_user_agent() -> Agent[UserInstructions, str]:
    with open(Path(__file__).parent / "user_agent_prompt.md", "r") as f:
        prompt = f.read()

    user_agent = Agent[UserInstructions, str](
        model=model,
        deps_type=UserInstructions,
        instructions=prompt,
        output_type=[str, EndConversation],
    )

    @user_agent.instructions
    def add_task_instructions(ctx: RunContext[UserInstructions]) -> str:
        return str(ctx.deps)

    return user_agent
