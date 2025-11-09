# Deep Research Agent

## Overview

The Deep Research Agent orchestrates a multi-stage LLM workflow that transforms user queries into comprehensive research reports through three sequential phases: **Scope**, **Research**, and **Write**. The agent begins by clarifying ambiguous queries and generating a structured research brief, then delegates focused research tasks to parallel sub-researchers coordinated by a supervisor agent, and finally synthesizes all findings into a cohesive final report. Each stage uses specialized LLM agents with distinct tools and prompts, enabling the system to handle complex, open-ended research tasks through iterative planning, parallel execution, and intelligent compression of information.

## Agents and Nodes

### Scope Stage
- **`clarify_with_user`**: Analyzes user messages using structured output to determine if the research scope requires clarification. Either requests additional information from the user or proceeds with verification. This is disabled during Deep Research Bench evaluation.
- **`write_research_brief`**: Transforms user messages into a structured research brief using an LLM with `ResearchQuestion` schema, then initializes the supervisor with appropriate system prompts and constraints.

### Research Stage
- **`supervisor`** (Lead Researcher): Plans and coordinates the overall research strategy. Uses `ConductResearch` to delegate tasks to sub-researchers, `think_tool` for strategic reflection, and `ResearchComplete` to signal research completion. Operates within iteration limits.
- **`supervisor_tools`**: Executes supervisor tool calls, managing parallel research delegation through `ConductResearch`, processing strategic reflections, and aggregating raw notes from completed research units. Enforces concurrent research limits and handles token limit errors.
- **`researcher`** (Sub-researcher): Conducts focused research on specific topics assigned by the supervisor. Uses search tools (Tavily, native web search), MCP tools, and `think_tool` for strategic planning between searches.
- **`researcher_tools`**: Executes researcher tool calls in parallel, handles search tool outputs, and determines whether to continue the research loop or proceed to compression based on iteration limits or completion signals.
- **`compress_research`**: Synthesizes individual researcher findings into concise summaries using a compression model. Implements retry logic with message truncation to handle token limit errors while preserving critical information.

### Write Stage
- **`final_report_generation`**: Generates the comprehensive final report by combining the research brief, user context, and all accumulated findings. Implements progressive truncation strategy with retry logic to handle token limit constraints while maximizing report quality.
