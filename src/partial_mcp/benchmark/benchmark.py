from typing import Literal

from pydantic_ai.toolsets.fastmcp import FastMCPToolset
from pydantic_ai_todo import create_todo_toolset
from pydantic_ai_filesystem_sandbox import FileSystemToolset
from pydantic_ai import UnexpectedModelBehavior
import logfire

from .tasks import UserInstructions
from .dataset import get_dataset
from .user_agent import get_user_agent, EndConversation
from ..mcp_servers.retail.agent import get_agent
from ..mcp_servers.retail.tools import server, retail
from ..mcp_servers.disable_toolcall import DisableToolcallWrapper
from ..mcp_servers.mcp_zero.mcp_zero import get_mcp_zero_toolsets


logfire.configure(
    service_name="benchmark",
    console=False,
    send_to_logfire=False,
    scrubbing=False,
)
logfire.instrument_pydantic_ai()
logfire.instrument_mcp()


async def evaluate(
    toolsets: Literal["relevant-only", "double", "mcp-zero", "half-mcp-zero"],
    max_cases: int | None = None,
    max_turns: int = 10,
):
    """
    Evaluate conversing agents:
    - evaluatee agent who has access to tools and represents a support agent
    - user agent who represents a user with an issue to resolve

    :param toolsets:
        Toolsets to add to the evaluatee agent.
        `relevant-only` adds only tools that are relevant to the dataset.
        `double` adds tools that are completely irrelevant to the dataset.
            This doubles the amount of tools from 15 to 30.
        `mcp-zero` adds 2666 extra tools in addition to the relevant ones.
        `half-mcp-zero` adds 1334 extra tools instead.
    :param max_cases: Limit the number of cases included in the dataset.
    :param max_turns: Maximum number of dialog turns in each conversation.
    """
    user_agent = get_user_agent()
    if toolsets == "relevant-only":
        agent = await get_agent(toolsets=[FastMCPToolset(server)])
    elif toolsets == "double":
        agent = await get_agent(
            toolsets=[
                FastMCPToolset(server),
                DisableToolcallWrapper(create_todo_toolset(enable_subtasks=True)),
                DisableToolcallWrapper(
                    FileSystemToolset.create_default("./data", mode="rw")
                ),
            ]
        )
    elif toolsets == "mcp-zero":
        agent = await get_agent(
            toolsets=[
                FastMCPToolset(server),
                *get_mcp_zero_toolsets(),
            ]
        )
    elif toolsets == "half-mcp-zero":
        agent = await get_agent(
            toolsets=[
                FastMCPToolset(server),
                *(get_mcp_zero_toolsets()[:130]),
            ]
        )

    async def simulate_conversation(instructions: UserInstructions):
        first_message = await user_agent.run(deps=instructions)

        last_message = first_message.output
        user_history = first_message.all_messages()
        agent_history = []
        agent_responses = []
        counter = 0
        try:
            while counter < max_turns and not isinstance(last_message, EndConversation):
                agent_response = await agent.run(
                    user_prompt=last_message,
                    message_history=agent_history,
                )
                agent_responses.append(agent_response.output)
                agent_history = agent_response.all_messages()

                user_agent_response = await user_agent.run(
                    deps=instructions,
                    user_prompt=agent_response.output,
                    message_history=user_history,
                )
                last_message = user_agent_response.output
                user_history = user_agent_response.all_messages()
                counter += 1
        except UnexpectedModelBehavior:
            logfire.exception("Suppressed a case exception")
        finally:
            retail.reset_db()
        return "\n\n---------\n\n".join(agent_responses)

    dataset = get_dataset(max_cases=max_cases)

    report = await dataset.evaluate(
        simulate_conversation,
        max_concurrency=1,  # have to keep at 1 for reset_db to work properly
    )

    report.print()
