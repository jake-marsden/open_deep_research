"""Configuration management for the Open Deep Research system."""

import os
from enum import Enum
from typing import Any, List, Optional

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field


class SearchAPI(Enum):
    """Enumeration of available search API providers."""
    
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    TAVILY = "tavily"
    NONE = "none"

class MCPConfig(BaseModel):
    """Configuration for Model Context Protocol (MCP) servers."""
    
    url: Optional[str] = Field(
        default=None,
        optional=True,
    )
    """The URL of the MCP server"""
    tools: Optional[List[str]] = Field(
        default=None,
        optional=True,
    )
    """The tools to make available to the LLM"""
    auth_required: Optional[bool] = Field(
        default=False,
        optional=True,
    )
    """Whether the MCP server requires authentication"""


class ModelTierConfig(BaseModel):
    """Configuration for a single model tier in the adaptive model selection system.
    
    Each tier represents a class of models with specific characteristics:
    - Low-tier: Fast, cost-efficient for routine tasks with negligible quality gap
    - Mid-tier: Workhorse tier with highest reliability-to-cost ratio
    - High-tier: Insurance tier for complex tasks where failure is unacceptable
    """
    
    model: str = Field(
        description="Model identifier (e.g., 'openai:gpt-4o-mini', 'anthropic:claude-3-haiku')"
    )
    max_tokens: int = Field(
        description="Maximum output tokens for this tier"
    )
    context_window: int = Field(
        default=128000,
        description="Maximum context window size for this model tier (for token limit enforcement)"
    )
    expected_success_rate: float = Field(
        default=0.90,
        ge=0.0,
        le=1.0,
        description="Expected success rate on tier-appropriate tasks (0.0-1.0), used for reliability-to-cost optimization"
    )
    cost_per_1k_input_tokens: float = Field(
        default=0.0,
        ge=0.0,
        description="Approximate cost per 1K input tokens (for cost tracking and optimization)"
    )
    cost_per_1k_output_tokens: float = Field(
        default=0.0,
        ge=0.0,
        description="Approximate cost per 1K output tokens (for cost tracking and optimization)"
    )
    
    @property
    def reliability_to_cost_ratio(self) -> float:
        """Calculate reliability-to-cost ratio. Higher is better.
        
        Used for mid-tier selection per Model Multiplexing paper's LEC principle.
        """
        avg_cost = (self.cost_per_1k_input_tokens + self.cost_per_1k_output_tokens) / 2
        if avg_cost == 0:
            return float('inf')
        return self.expected_success_rate / avg_cost


class AdaptiveModelConfig(BaseModel):
    """Configuration for adaptive model selection across tiers.
    
    Implements insights from academic papers:
    - MDAgents: Three-tier complexity classification
    - Hybrid LLM: Quality gap-based tier eligibility
    - Early Abstention: Confidence thresholds for safe defaults
    - FrugalGPT: Context window constraints for tier escalation
    - Model Multiplexing: Reliability-to-cost optimization for mid-tier
    """
    
    # Model Tier Definitions
    low_tier: ModelTierConfig = Field(
        default_factory=lambda: ModelTierConfig(
            model="openai:gpt-3.5-turbo", # $0.50/M input tokens | $1.50/M output tokens
            max_tokens=4096,
            context_window=128000,
            expected_success_rate=0.95,
            cost_per_1k_input_tokens=0.00015,
            cost_per_1k_output_tokens=0.0006
        ),
        description="Fast, cost-efficient model for routine tasks where quality gap is negligible (~20% of tasks)"
    )
    
    mid_tier: ModelTierConfig = Field(
        default_factory=lambda: ModelTierConfig(
            model="openai:gpt-4o-mini", # $0.15/M input tokens | $0.60/M output tokens
            max_tokens=8192,
            context_window=128000,
            expected_success_rate=0.92,
            cost_per_1k_input_tokens=0.0025,
            cost_per_1k_output_tokens=0.01
        ),
        description="Workhorse tier with highest reliability-to-cost ratio, handles majority of tasks (~60-70%)"
    )
    
    high_tier: ModelTierConfig = Field(
        default_factory=lambda: ModelTierConfig(
            model="openai:gpt-4o", # $2.50/M input tokens | $10/M output tokens
            max_tokens=16384,
            context_window=200000,
            expected_success_rate=0.98,
            cost_per_1k_input_tokens=0.015,
            cost_per_1k_output_tokens=0.06
        ),
        description="Insurance tier for complex tasks where mid-tier failure risk is unacceptable (~15-25% of tasks)"
    )
    
    # Confidence Thresholds (Early Abstention principle)
    confidence_threshold_low: int = Field(
        default=85,
        ge=0,
        le=100,
        description="Minimum confidence % to use low-tier model (below this → mid-tier)"
    )
    confidence_threshold_mid: int = Field(
        default=70,
        ge=0,
        le=100,
        description="Minimum confidence % to use mid-tier model (below this → high-tier)"
    )
    
    # Context Window Safety Margin
    context_window_safety_margin: float = Field(
        default=0.8,
        ge=0.5,
        le=1.0,
        description="Safety margin for context window (0.8 = use 80% of available context to avoid overflow)"
    )
    
    # Feature Flags
    enable_adaptive_selection: bool = Field(
        default=True,
        description="Enable/disable adaptive selection (when disabled, falls back to research_model)"
    )
    log_tier_decisions: bool = Field(
        default=True,
        description="Log tier selection decisions for analysis and threshold tuning"
    )
    
    def get_tier_config(self, tier: str) -> ModelTierConfig:
        """Get the model configuration for a specific tier."""
        tier_map = {
            "low": self.low_tier,
            "mid": self.mid_tier,
            "high": self.high_tier
        }
        return tier_map.get(tier, self.mid_tier)

class Configuration(BaseModel):
    """Main configuration class for the Deep Research agent."""
    
    # General Configuration
    max_structured_output_retries: int = Field(
        default=3,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 3,
                "min": 1,
                "max": 10,
                "description": "Maximum number of retries for structured output calls from models"
            }
        }
    )
    allow_clarification: bool = Field(
        default=True,
        metadata={
            "x_oap_ui_config": {
                "type": "boolean",
                "default": True,
                "description": "Whether to allow the researcher to ask the user clarifying questions before starting research"
            }
        }
    )
    max_concurrent_research_units: int = Field(
        default=3,
        metadata={
            "x_oap_ui_config": {
                "type": "slider",
                "default": 3,
                "min": 1,
                "max": 20,
                "step": 1,
                "description": "Maximum number of research units to run concurrently. This will allow the researcher to use multiple sub-agents to conduct research. Note: with more concurrency, you may run into rate limits."
            }
        }
    )
    # Research Configuration
    search_api: SearchAPI = Field(
        default=SearchAPI.TAVILY,
        metadata={
            "x_oap_ui_config": {
                "type": "select",
                "default": "tavily",
                "description": "Search API to use for research. NOTE: Make sure your Researcher Model supports the selected search API.",
                "options": [
                    {"label": "Tavily", "value": SearchAPI.TAVILY.value},
                    {"label": "OpenAI Native Web Search", "value": SearchAPI.OPENAI.value},
                    {"label": "Anthropic Native Web Search", "value": SearchAPI.ANTHROPIC.value},
                    {"label": "None", "value": SearchAPI.NONE.value}
                ]
            }
        }
    )
    max_researcher_iterations: int = Field(
        default=3,
        metadata={
            "x_oap_ui_config": {
                "type": "slider",
                "default": 3,
                "min": 1,
                "max": 10,
                "step": 1,
                "description": "Maximum number of research iterations for the Research Supervisor. This is the number of times the Research Supervisor will reflect on the research and ask follow-up questions."
            }
        }
    )
    max_react_tool_calls: int = Field(
        default=6,
        metadata={
            "x_oap_ui_config": {
                "type": "slider",
                "default": 6,
                "min": 1,
                "max": 30,
                "step": 1,
                "description": "Maximum number of tool calling iterations to make in a single researcher step."
            }
        }
    )
    # Model Configuration
    summarization_model: str = Field(
        default="openai:google/gemini-2.5-flash-lite",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "openai:google/gemini-2.5-flash-lite",
                "description": "Model for summarizing research results from Tavily search results"
            }
        }
    )
    summarization_model_max_tokens: int = Field(
        default=8192,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 8192,
                "description": "Maximum output tokens for summarization model"
            }
        }
    )
    max_content_length: int = Field(
        default=50000,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 50000,
                "min": 1000,
                "max": 200000,
                "description": "Maximum character length for webpage content before summarization"
            }
        }
    )
    research_model: str = Field(
        default="openai:gpt-4o-mini",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "openai:gpt-4o-mini",
                "description": "Model for the research supervisor and fallback for sub-researchers when adaptive model selection is disabled. NOTE: Make sure your model supports the selected search API."
            }
        }
    )
    research_model_max_tokens: int = Field(
        default=10000,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 10000,
                "description": "Maximum output tokens for research model"
            }
        }
    )
    compression_model: str = Field(
        default="openai:google/gemini-2.0-flash-lite-001",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "openai:google/gemini-2.0-flash-lite-001",
                "description": "Model for compressing research findings from sub-agents. NOTE: Make sure your Compression Model supports the selected search API."
            }
        }
    )
    compression_model_max_tokens: int = Field(
        default=8192,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 8192,
                "description": "Maximum output tokens for compression model"
            }
        }
    )
    final_report_model: str = Field(
        default="openai:gpt-5-mini",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "openai:gpt-5-mini",
                "description": "Model for writing the final report from all research findings"
            }
        }
    )
    final_report_model_max_tokens: int = Field(
        default=10000,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 10000,
                "description": "Maximum output tokens for final report model"
            }
        }
    )
    # MCP server configuration
    mcp_config: Optional[MCPConfig] = Field(
        default=None,
        optional=True,
        metadata={
            "x_oap_ui_config": {
                "type": "mcp",
                "description": "MCP server configuration"
            }
        }
    )
    mcp_prompt: Optional[str] = Field(
        default=None,
        optional=True,
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "description": "Any additional instructions to pass along to the Agent regarding the MCP tools that are available to it."
            }
        }
    )
    
    # Adaptive Model Selection Configuration
    adaptive_model_config: AdaptiveModelConfig = Field(
        default_factory=AdaptiveModelConfig,
        metadata={
            "x_oap_ui_config": {
                "type": "object",
                "description": "Configuration for adaptive model selection based on task complexity. When enabled, sub-tasks are routed to appropriate model tiers (low/mid/high) based on complexity assessment."
            }
        }
    )


    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """Create a Configuration instance from a RunnableConfig."""
        configurable = config.get("configurable", {}) if config else {}
        field_names = list(cls.model_fields.keys())
        values: dict[str, Any] = {
            field_name: os.environ.get(field_name.upper(), configurable.get(field_name))
            for field_name in field_names
        }
        return cls(**{k: v for k, v in values.items() if v is not None})

    class Config:
        """Pydantic configuration."""
        
        arbitrary_types_allowed = True