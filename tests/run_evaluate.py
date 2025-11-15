from dotenv import load_dotenv

load_dotenv(".env")

from langsmith import Client
from evaluators import eval_overall_quality, eval_relevance, eval_structure, eval_correctness, eval_groundedness, eval_completeness
import asyncio
from langgraph.checkpoint.memory import MemorySaver
import uuid

# System selection: Switch between 'open_deep_research' and 'rofc'
SYSTEM_TO_EVALUATE = "open_deep_research"  # Options: "open_deep_research" or "rofc"

if SYSTEM_TO_EVALUATE == "rofc":
    from rofc.deep_researcher import deep_researcher_builder
else:
    from open_deep_research.deep_researcher import deep_researcher_builder

client = Client()

# NOTE: Configure the right dataset and evaluators
dataset_name = "Deep Research Bench"
evaluators = [eval_overall_quality, eval_relevance, eval_structure, eval_correctness, eval_groundedness, eval_completeness]
# NOTE: Configure the right parameters for the experiment, these will be logged in the metadata
max_structured_output_retries = 3
allow_clarification = False
max_concurrent_research_units = 3 # changed from 10 to 3
search_api = "tavily"
max_researcher_iterations = 3     # changed from 6 to 3
max_react_tool_calls = 6          # changed from 10 to 6
summarization_model = "openai:google/gemini-2.5-flash-lite"      # $0.10/M input tokens | $0.40/M output tokens
summarization_model_max_tokens = 8192
research_model = "openai:gpt-4o-mini"                            # $0.15/M input tokens | $0.60/M output tokens
research_model_max_tokens = 10000
compression_model = "openai:google/gemini-2.0-flash-lite-001"    # $0.075/M input tokens | $0.30/M output tokens
compression_model_max_tokens = 10000
final_report_model = "openai:gpt-5-mini"                         # $0.25/M input tokens | $2/M output tokens
final_report_model_max_tokens = 10000

async def target(
    inputs: dict,
):
    graph = deep_researcher_builder.compile(checkpointer=MemorySaver())
    config = {
        "configurable": {
            "thread_id": str(uuid.uuid4()),
        }
    }
    # NOTE: Configure the right dataset and evaluators
    config["configurable"]["max_structured_output_retries"] = max_structured_output_retries
    config["configurable"]["allow_clarification"] = allow_clarification
    config["configurable"]["max_concurrent_research_units"] = max_concurrent_research_units
    config["configurable"]["search_api"] = search_api
    config["configurable"]["max_researcher_iterations"] = max_researcher_iterations
    config["configurable"]["max_react_tool_calls"] = max_react_tool_calls
    config["configurable"]["summarization_model"] = summarization_model
    config["configurable"]["summarization_model_max_tokens"] = summarization_model_max_tokens
    config["configurable"]["research_model"] = research_model
    config["configurable"]["research_model_max_tokens"] = research_model_max_tokens
    config["configurable"]["compression_model"] = compression_model
    config["configurable"]["compression_model_max_tokens"] = compression_model_max_tokens
    config["configurable"]["final_report_model"] = final_report_model
    config["configurable"]["final_report_model_max_tokens"] = final_report_model_max_tokens
    # NOTE: We do not use MCP tools to stay consistent
    final_state = await graph.ainvoke(
        {"messages": [{"role": "user", "content": inputs["messages"][0]["content"]}]},
        config
    )
    return final_state

async def main():
    return await client.aevaluate(
        target,
        data=dataset_name,
        evaluators=evaluators,
        experiment_prefix=f"{SYSTEM_TO_EVALUATE.upper()}_{summarization_model}_{research_model}_{compression_model}_{final_report_model}",
        max_concurrency=1, # Changed from 3 to 1 to avoid rate limits
        metadata={
            "system": SYSTEM_TO_EVALUATE,
            "max_structured_output_retries": max_structured_output_retries,
            "allow_clarification": allow_clarification,
            "max_concurrent_research_units": max_concurrent_research_units,
            "search_api": search_api,
            "max_researcher_iterations": max_researcher_iterations,
            "max_react_tool_calls": max_react_tool_calls,
            "summarization_model": summarization_model,
            "summarization_model_max_tokens": summarization_model_max_tokens,
            "research_model": research_model,
            "research_model_max_tokens": research_model_max_tokens,
            "compression_model": compression_model,
            "compression_model_max_tokens": compression_model_max_tokens,
            "final_report_model": final_report_model,
            "final_report_model_max_tokens": final_report_model_max_tokens,
        }
    )

if __name__ == "__main__":
    results = asyncio.run(main())
    print(results)