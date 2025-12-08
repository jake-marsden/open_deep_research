from dotenv import load_dotenv

load_dotenv(".env")

from langsmith import Client
from evaluators import eval_overall_quality, eval_relevance, eval_structure, eval_correctness, eval_groundedness, eval_completeness
import asyncio
from langgraph.checkpoint.memory import MemorySaver
import uuid
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
research_model = "openai:gpt-4o-mini"                            # $0.15/M input tokens | $0.60/M output tokens (used for supervisor only when adaptive selection is enabled)
research_model_max_tokens = 10000
compression_model = "openai:google/gemini-2.0-flash-lite-001"    # $0.075/M input tokens | $0.30/M output tokens
compression_model_max_tokens = 10000
final_report_model = "openai:gpt-5-mini"                         # $0.25/M input tokens | $2/M output tokens
final_report_model_max_tokens = 10000

# Adaptive Model Selection Configuration
enable_adaptive_selection = True  # Set to False to use research_model for all sub-researchers
adaptive_low_tier_model = "openai:gpt-3.5-turbo"                 # $0.50/M input tokens | $1.50/M output tokens
adaptive_low_tier_max_tokens = 4096
adaptive_low_tier_context_window = 128000
adaptive_mid_tier_model = "openai:gpt-4.1"                       # $2/M input tokens | $8/M output tokens
adaptive_mid_tier_max_tokens = 8192
adaptive_mid_tier_context_window = 128000
adaptive_high_tier_model = "openai:gpt-4o"                       # $2.50/M input tokens | $10/M output tokens
adaptive_high_tier_max_tokens = 16384
adaptive_high_tier_context_window = 200000
adaptive_confidence_threshold_low = 85   # Confidence threshold to use low-tier (below this → mid-tier)
adaptive_confidence_threshold_mid = 70   # Confidence threshold to use mid-tier (below this → high-tier)
adaptive_log_tier_decisions = True       # Log tier selection decisions for analysis

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
    
    # Configure adaptive model selection
    config["configurable"]["adaptive_model_config"] = {
        "enable_adaptive_selection": enable_adaptive_selection,
        "log_tier_decisions": adaptive_log_tier_decisions,
        "low_tier": {
            "model": adaptive_low_tier_model,
            "max_tokens": adaptive_low_tier_max_tokens,
            "context_window": adaptive_low_tier_context_window,
        },
        "mid_tier": {
            "model": adaptive_mid_tier_model,
            "max_tokens": adaptive_mid_tier_max_tokens,
            "context_window": adaptive_mid_tier_context_window,
        },
        "high_tier": {
            "model": adaptive_high_tier_model,
            "max_tokens": adaptive_high_tier_max_tokens,
            "context_window": adaptive_high_tier_context_window,
        },
        "confidence_threshold_low": adaptive_confidence_threshold_low,
        "confidence_threshold_mid": adaptive_confidence_threshold_mid,
    }
    # NOTE: We do not use MCP tools to stay consistent
    final_state = await graph.ainvoke(
        {"messages": [{"role": "user", "content": inputs["messages"][0]["content"]}]},
        config
    )
    return final_state

def is_english_content(text: str) -> bool:
    """Detect if text is primarily English by checking for Chinese characters."""
    if not text:
        return False
    chinese_chars = sum(1 for char in text if '\u4e00' <= char <= '\u9fff')
    return chinese_chars / len(text) < 0.1

async def main():
    # Filter dataset to only include English examples
    examples = list(client.list_examples(dataset_name=dataset_name))
    
    print(f"Total examples in dataset: {len(examples)}")
    
    english_examples = []
    for ex in examples:
        if ex.inputs and 'messages' in ex.inputs:
            messages = ex.inputs['messages']
            if messages and len(messages) > 0:
                content = messages[0].get('content', '')
                if is_english_content(content):
                    english_examples.append(ex)
    
    print(f"English examples detected: {len(english_examples)}")
    
    if len(english_examples) == 0:
        raise ValueError("No English examples found in dataset!")
    
    # Pass the actual examples instead of example_ids (which isn't supported)
    # Build experiment prefix based on adaptive selection status
    if enable_adaptive_selection:
        experiment_prefix = f"OPEN_DEEP_RESEARCH_ADAPTIVE_{adaptive_low_tier_model}_{adaptive_mid_tier_model}_{adaptive_high_tier_model}"
    else:
        experiment_prefix = f"OPEN_DEEP_RESEARCH_EN_ONLY_{summarization_model}_{research_model}_{compression_model}_{final_report_model}"
    
    return await client.aevaluate(
        target,
        data=english_examples,  # Pass filtered examples directly
        evaluators=evaluators,
        experiment_prefix=experiment_prefix,
        max_concurrency=3, # Changed from 3 to 1 to avoid rate limits
        metadata={
            "system": "open_deep_research",
            "language_filter": "en_only",
            "num_examples": len(english_examples),
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
            # Adaptive model selection metadata
            "adaptive_selection_enabled": enable_adaptive_selection,
            "adaptive_low_tier_model": adaptive_low_tier_model,
            "adaptive_low_tier_max_tokens": adaptive_low_tier_max_tokens,
            "adaptive_mid_tier_model": adaptive_mid_tier_model,
            "adaptive_mid_tier_max_tokens": adaptive_mid_tier_max_tokens,
            "adaptive_high_tier_model": adaptive_high_tier_model,
            "adaptive_high_tier_max_tokens": adaptive_high_tier_max_tokens,
            "adaptive_confidence_threshold_low": adaptive_confidence_threshold_low,
            "adaptive_confidence_threshold_mid": adaptive_confidence_threshold_mid,
            "adaptive_log_tier_decisions": adaptive_log_tier_decisions,
        }
    )

if __name__ == "__main__":
    results = asyncio.run(main())
    print(results)