"""Graph state definitions and data structures for the Deep Research agent."""

import operator
from typing import Annotated, Optional

from langchain_core.messages import MessageLikeRepresentation
from langgraph.graph import MessagesState
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


###################
# Structured Outputs
###################
class ConductResearch(BaseModel):
    """Call this tool to conduct research on a specific topic."""
    research_topic: str = Field(
        description="The topic to research. Should be a single topic, and should be described in high detail (at least a paragraph).",
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

class RefinementDirectives(BaseModel):
    """Structured directives for research refinement."""
    
    missing_subtopics: list[str] = Field(
        description="Specific subtopics or information categories that are missing.",
        default_factory=list
    )
    required_evidence_types: list[str] = Field(
        description="Types of evidence needed (e.g., 'Statistical data', 'Expert quote', 'Contra-arguments').",
        default_factory=list
    )
    action: str = Field(
        description="Recommended action: 'search_specific_queries', 'verify_citations', 'expand_breadth'."
    )

class QualityAssessmentScores(BaseModel):
    """Research review scoring output based on RACE and FACT metrics."""
    
    comprehensiveness_pass: bool = Field(
        description="Does the research cover multiple perspectives and all key sub-questions? (True/False)"
    )
    insight_pass: bool = Field(
        description="Does the research provide causal reasoning and connections, not just facts? (True/False)"
    )
    instruction_following_pass: bool = Field(
        description="Does the research strictly adhere to all constraints and objectives? (True/False)"
    )
    factuality_pass: bool = Field(
        description="Are there sufficient valid, authoritative citations? (True/False)"
    )
    feedback: str = Field(
        description="Detailed feedback explaining which standards were met/failed and why"
    )
    refinement_guidance: RefinementDirectives = Field(
        description="Structured guidance for improvement on failed dimensions"
    )

class QualityAssessment(BaseModel):
    """Complete research review evaluation with calculated decision."""
    
    comprehensiveness_pass: bool
    insight_pass: bool
    instruction_following_pass: bool
    factuality_pass: bool
    all_pass: bool = Field(description="True if all 4 dimensions passed")
    decision: str = Field(description="Calculated decision: 'accept' or 'refine'")
    failed_dimensions: list = Field(description="List of dimension names that failed")
    feedback: str
    refinement_guidance: RefinementDirectives


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
    
    researcher_messages: Annotated[list[MessageLikeRepresentation], override_reducer]
    tool_call_iterations: int = 0
    research_topic: str
    compressed_research: str
    raw_notes: Annotated[list[str], override_reducer] = []
    quality_assessment: Optional[QualityAssessment] = None
    refinement_attempts: int = 0

class ResearcherOutputState(BaseModel):
    """Output state from individual researchers."""
    
    compressed_research: str
    raw_notes: Annotated[list[str], override_reducer] = []
    quality_assessment: Optional[QualityAssessment] = None
    refinement_attempts: int = 0
