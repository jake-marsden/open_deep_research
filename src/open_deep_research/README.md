# Deep Research Agent

## Overview

The Deep Research Agent orchestrates a multi-stage LLM workflow that transforms user queries into comprehensive research reports through three sequential phases: **Scope**, **Research**, and **Write**. The agent begins by clarifying ambiguous queries and generating a structured research brief, then delegates focused research tasks to parallel sub-researchers coordinated by a supervisor agent, with each sub-researcher's output validated by a quality review gate that can request refinements when needed. Finally, it synthesizes all findings into a cohesive final report. Each stage uses specialized LLM agents with distinct tools and prompts, enabling the system to handle complex, open-ended research tasks through iterative planning, parallel execution, quality assurance, and intelligent compression of information.

## Agents and Nodes

### Scope Stage
- **`clarify_with_user`**: Analyzes user messages using structured output to determine if the research scope requires clarification. Either requests additional information from the user or proceeds with verification. Configurable via `allow_clarification`.
- **`write_research_brief`**: Transforms user messages into a structured research brief using the `ResearchQuestion` schema, then initializes the supervisor with appropriate system prompts and constraints.

### Research Stage (Supervisor Subgraph)
- **`supervisor`**: Plans and coordinates overall research strategy using the research model. Uses `ConductResearch` to delegate tasks to sub-researchers, `think_tool` for strategic reflection, and `ResearchComplete` to signal completion. Limited by `max_researcher_iterations`.
- **`supervisor_tools`**: Executes supervisor tool calls, managing parallel research delegation through `ConductResearch` (up to `max_concurrent_research_units`), processing strategic reflections, and aggregating raw notes from completed research units. Handles token limit errors.

### Research Stage (Researcher Subgraph)
- **`researcher`**: Conducts focused research on specific topics assigned by the supervisor using the research model. Uses a search tool (Tavily, Anthropic/OpenAI native web search), MCP tools (Model Context Protocol for external integrations), `ResearchComplete`, and `think_tool` for strategic planning between searches. Adapts its approach when in refinement mode based on reviewer feedback.
- **`researcher_tools`**: Executes researcher tool calls in parallel, handles search tool outputs, and determines whether to continue the research loop or proceed to compression based on `max_react_tool_calls` limit.
- **`compress_research`**: Synthesizes individual researcher findings into concise summaries using the compression model. Implements retry logic with message truncation to handle token limit errors while preserving critical information and citations.
- **`research_reviewer`**: Quality gate that evaluates compressed research output on four dimensions (Comprehensiveness, Insight, Instruction Following, Factuality). Makes binary decisions: accepts research if ALL dimensions pass to proceed to supervisor, or sends research back to researcher with specific feedback and refinement guidance if ANY dimension fails. Limited by `max_research_reviewer_refinements` (default: 2) to prevent infinite loops.

### Write Stage
- **`final_report_generation`**: Generates the comprehensive final report using the final report model by combining the research brief, user context, and all accumulated findings. Implements progressive truncation strategy with retry logic to handle token limit constraints while maximizing report quality.

## Model Specialization

The system uses five specialized models for different tasks:
- **Research Model**: Powers supervisor and researcher agents for planning and conducting research.
- **Summarization Model**: Summarizes webpage content from search results.
- **Compression Model**: Compresses and structures individual researcher findings.
- **Research Reviewer Model**: Evaluates research quality and provides feedback for refinement.
- **Final Report Model**: Synthesizes all findings into the final comprehensive report.
