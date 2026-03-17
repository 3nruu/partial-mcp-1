# Partial MCP

This repository is for testing ways to reduce LLM context occupied by
MCP tool descriptions.

## Background

This issue was first brought up by Cloudflare:
https://blog.cloudflare.com/code-mode/

Summary of the problem:
1. Tool definitions take up a part of the context window at all times
even if the tools are not used.
2. Tool results are sometimes only needed to be passed to another tool call
  as arguments and are not needed in the context window.
3. It is often clear what tool should be called after another tool call and
  inserting an LLM call inbetween slows down execution and wastes tokens

The proposed solution (called "Code Mode") is to treat tools as functions
available to LLM as part of an SDK. LLM can write code which includes tool calls
(and execute said code) but has no direct access to the tools.

However, implementations of Code Mode vary.

### No discovery code mode

An example implementation is CodeModeToolset from Pydantic:
https://github.com/pydantic/pydantic-ai/blob/3490542f44368d2c935d725b5bfc0f542890401b/pydantic_ai_slim/pydantic_ai/toolsets/code_mode/__init__.py#L89-L128

This implementation doesn't address the first issue of occupying context window
with tool definitions and only implements a solution for the last two issues.

### Discovery code mode

Example implementations (utcp-code-mode and bitfrost code mode):
https://github.com/universal-tool-calling-protocol/code-mode
https://docs.getbifrost.ai/mcp/code-mode#the-four-code-mode-tools

These implement tools for tool discovery.
The second implementation adds a `getToolDocs` tool to further reduce
context consumption of tool descriptions.
However, this approach has a downside of increasing the number of tool calls
(since the model now has to spend time discovering tools) as noted in this
blogpost:
https://block.github.io/goose/blog/2025/12/21/code-mode-doesnt-replace-mcp/#the-value-of-code-mode

## This repository

The primary focus of this repository is the first issue mentioned in the
[Background Section](#Background).

The basis of the benchmark presented in this repository is the [`tau2` benchmark](
https://github.com/sierra-research/tau2-bench
).
It implements a simulation framework for evaluating customer service agent across various domains.

Each domain specifies:
- a policy that the agent must follow
- a set of tools that the agent can use
- a set of tasks to evaluate the agent's performance

From that benchmark we take data pertaining to a single domain (`retail`),
add multiple irrelevant tools on top of the tools available from that domain,
and compare agent's performance.

Irrelevant toolsets added are definition-only.
Whenever agent tries to call them, an exception is raised instead.

Some irrelevant tools come from pydantic toolset packages such as:
- `pydantic-ai-todo`
- `pydantic-ai-filesystem-sandbox`
And others come from [MCP Zero](https://github.com/xfey/MCP-Zero)'s 300+ toolsets.

### Suggested solutions

Below are possible solutions to the problem of presence of irrelevant tools:
1. External tool selector to present the model with relevant tools only
2. Code mode with tool discovery to let the model choose relevant tools.
3. Hybrid approach where code mode doesn't have tool discovery
   (pydantic's approach) and the available tools are chosen by a tool selector.

Your goal is to implement one of these approaches and evaluate it.

### Baseline results

**Todo**: insert baseline results for the benchmark

### Known issues

These are known issues with the current version of the benchmark and will soon
be fixed.

## Repository Usage

### Running benchmark

#### Step 0: Observability

Optionally, you can first set up a Jaeger service for better observability
of benchmark's inner workings. This will let you see what tools get called,
the context of the agent and reasons for low score.

To do so, run:
```bash
docker compose -f services/monitoring/compose.yaml up
```
After running this command, you can access Jaeger's UI via
http://localhost:16686.
This is a very simple version with an in-memory backend
meaning that stopping the process will remove all the collected data from
its memory.

#### Step 1: Env vars

Copy `.env.example` into `.env` and modify it:
- Remove `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` if you're not using Observability.
- Fill out `MODEL_NAME`, `API_KEY`, `BASE_URL`.

#### Step 2: Running Benchmark

You can get the help message with:
```bash
uv run poe --help benchmark
```

To run the benchmark in its original version, execute:
```bash
uv run poe benchmark relevant-only
```

To run the benchmark with irrelevant tools, execute either one of the three:
```bash
uv run poe benchmark double        # 15 extra tools
```
```bash
uv run poe benchmark half-mcp-zero # 1334 extra tools
```
```bash
uv run poe benchmark mcp-zero      # 2666 extra tools
```

In either case, you can limit the number of cases tested if you want
to run a quick check (instead of running all 40 cases):
```bash
uv run poe benchmark double --max-cases 1
```

### Implementing your version of the solution

Place your implementation of the solution in the [toolset.py file](
./src/partial_mcp/toolset/toolset.py
).
There you can redefine the following methods:
- `prepare` to perform once-per-benchmark tool pre-processing
- `get_tools` to change the tools available to the agent
- `call_tool` to change the way tools are called

### Codestyle

First, install pre-commit hooks with:
```bash
uv run pre-commit install
```
This will make it so you cannot make a git commit if it does not satisfy
conditions from [.pre-commit-config.yaml](.pre-commit-config.yaml). Namely:
- Bad formatting
- Ruff's linter fails
- Pyrefly's type checking fails

If you want to run the checks manually, execute the following command
```bash
uv run pre-commit run --all-files
```
