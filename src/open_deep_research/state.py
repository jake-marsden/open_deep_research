"""Graph state definitions and data structures for the Deep Research agent."""

import operator
from typing import Annotated, Literal, Optional

from langchain_core.messages import MessageLikeRepresentation
from langgraph.graph import MessagesState
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


###################
# Adaptive Model Selection Schemas
###################

class FeatureExtraction(BaseModel):
    """Task features for complexity classification."""
    
    # Required field
    task_type: Literal["retrieval", "reasoning", "synthesis", "generation"] = Field(
        description="retrieval=fact lookup, reasoning=inference, synthesis=combining sources, generation=creative"
    )
    # Optional fields with sensible defaults
    reasoning_pattern: Literal["single-hop", "multi-hop", "causal", "comparative", "none"] = Field(
        default="multi-hop"
    )
    domain_signal: Literal["general", "specialized", "expert-level"] = Field(
        default="general"
    )
    context_difficulty: Literal["clear", "ambiguous", "contradictory"] = Field(
        default="clear"
    )
    requires_narrative_reasoning: bool = Field(default=False)
    requires_symbolic_execution: bool = Field(default=False)


class ContextEstimation(BaseModel):
    """Token estimation - optional, system can infer from other features."""
    
    estimated_input_tokens: int = Field(default=3000, ge=0)
    estimated_output_tokens: int = Field(default=1000, ge=0)
    retrieval_breadth: Literal["narrow", "moderate", "extensive"] = Field(default="moderate")
    context_accumulation: Literal["minimal", "moderate", "heavy"] = Field(default="moderate")


class ComplexityAssessment(BaseModel):
    """Complexity assessment for adaptive model selection.
    
    The tier is now determined programmatically based on complexity_rank:
    - complexity_rank 1 (simplest) → low tier
    - complexity_rank 2 (moderate) → mid tier  
    - complexity_rank 3 (most complex) → high tier
    """
    
    # Essential fields
    complexity_rank: Literal[1, 2, 3] = Field(
        description="Complexity rank: 1=simplest task, 2=moderate task, 3=most complex task. Tier is assigned automatically based on rank."
    )
    estimated_confidence: int = Field(
        default=75,
        ge=0, 
        le=100,
        description="Confidence (0-100%) that the task will succeed"
    )
    failure_risk: Literal["low", "medium", "high"] = Field(
        default="medium",
        description="Impact if task fails: low=recoverable, high=critical"
    )
    
    # Optional fields with defaults
    features: Optional[FeatureExtraction] = Field(default=None)
    context_estimation: Optional[ContextEstimation] = Field(default=None)
    quality_gap_prediction: Literal["negligible", "moderate", "significant"] = Field(
        default="moderate"
    )
    rationale: str = Field(default="")
    
    def get_tier(self) -> str:
        """Get the model tier based on complexity rank.
        
        Returns:
            'low' for rank 1, 'mid' for rank 2, 'high' for rank 3
        """
        tier_map = {1: "low", 2: "mid", 3: "high"}
        return tier_map.get(self.complexity_rank, "mid")


###################
# Structured Outputs
###################
class ConductResearch(BaseModel):
    """Call this tool to conduct research on a specific topic with complexity assessment."""
    
    research_topic: str = Field(
        description="The topic to research. Should be a single topic, and should be described in high detail (at least a paragraph).",
    )
    complexity_assessment: ComplexityAssessment = Field(
        description="Assessment of task complexity for adaptive model selection. Analyze the research topic to determine the appropriate model tier."
    )

class ResearchComplete(BaseModel):
    """Call this tool to indicate that the research is complete."""

class Summary(BaseModel):
    """Research summary with key findings."""
    
    summary: str
    key_excerpts: str

class ClarifyWithUser(BaseModel):
    """Model for user clarification requests."""
    
    need_clarification: bool = Field(
        description="Whether the user needs to be asked a clarifying question.",
    )
    question: str = Field(
        description="A question to ask the user to clarify the report scope",
    )
    verification: str = Field(
        description="Verify message that we will start research after the user has provided the necessary information.",
    )

class ResearchQuestion(BaseModel):
    """Research question and brief for guiding research."""
    
    research_brief: str = Field(
        description="A research question that will be used to guide the research.",
    )


###################
# State Definitions
###################

def override_reducer(current_value, new_value):
    """Reducer function that allows overriding values in state."""
    if isinstance(new_value, dict) and new_value.get("type") == "override":
        return new_value.get("value", new_value)
    else:
        return operator.add(current_value, new_value)
    
class AgentInputState(MessagesState):
    """InputState is only 'messages'."""

class AgentState(MessagesState):
    """Main agent state containing messages and research data."""
    
    supervisor_messages: Annotated[list[MessageLikeRepresentation], override_reducer]
    research_brief: Optional[str]
    raw_notes: Annotated[list[str], override_reducer] = []
    notes: Annotated[list[str], override_reducer] = []
    final_report: str

class SupervisorState(TypedDict):
    """State for the supervisor that manages research tasks."""
    
    supervisor_messages: Annotated[list[MessageLikeRepresentation], override_reducer]
    research_brief: str
    notes: Annotated[list[str], override_reducer] = []
    research_iterations: int = 0
    raw_notes: Annotated[list[str], override_reducer] = []

class ResearcherState(TypedDict):
    """State for individual researchers conducting research."""
    
    researcher_messages: Annotated[list[MessageLikeRepresentation], operator.add]
    tool_call_iterations: int = 0
    research_topic: str
    compressed_research: str
    raw_notes: Annotated[list[str], override_reducer] = []
    # Adaptive model selection - tier config passed from supervisor
    selected_model: Optional[str] = None
    selected_max_tokens: Optional[int] = None
    selected_tier: Optional[str] = None

class ResearcherOutputState(BaseModel):
    """Output state from individual researchers."""
    
    compressed_research: str
    raw_notes: Annotated[list[str], override_reducer] = []