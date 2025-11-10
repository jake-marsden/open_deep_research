# Reflective Orchestration for Factual Consistency (ROFC)
## From Task-Execution Cycles to Quality-Assurance Cycles

---

## Problem Statement

The current `open_deep_research` architecture employs **local cycles for task execution** (iterative information gathering) but lacks **global cycles for quality assurance** (cross-stage verification and correction). Once `final_report_generation()` completes, the system terminates without verifying factual consistency, detecting contradictions, or enabling targeted refinement.

This thesis introduces **Reflective Orchestration for Factual Consistency (ROFC)** to address this gap through a verification-correction loop. Additionally, we implement **Adaptive Model Selection (AMS)** as an integrated cost-optimization feature that mitigates the operational overhead introduced by ROFC's additional verification passes.

## Architectural Analysis: Current State

### Main Pipeline (Globally Linear)

```
START → clarify_with_user() → write_research_brief() → research_supervisor → final_report_generation() → END
```

**Key observation:** Strictly unidirectional flow with no backward edges (`deep_researcher.py:714-716`)

### Research Stage (Locally Cyclic)

**Supervisor Loop:** `supervisor()` ↔ `supervisor_tools()`
- **Purpose:** Delegate and coordinate parallel research tasks
- **Exit conditions:** Iteration limit (`max_researcher_iterations`), no tool calls, or `ResearchComplete` signal
- **Scope:** Task orchestration within Research stage only

**Researcher Loop:** `researcher()` ↔ `researcher_tools()`  
- **Purpose:** Iteratively gather information via search tools
- **Exit conditions:** Iteration limit (`max_react_tool_calls`) or `ResearchComplete` signal
- **Scope:** Information retrieval within individual research units

### Critical Gaps

1. **No post-generation verification:** `final_report_generation()` returns immediately without fact-checking
2. **No backward routing:** Once Research stage completes, system cannot return for additional investigation
3. **No quality assessment nodes:** No mechanisms for detecting inconsistencies, contradictions, or unsupported claims
4. **State cleared prematurely:** `cleared_state = {"notes": {"type": "override", "value": []}}` at report generation start

---

## Proposed Architecture: ROFC

### Design Principle

Augment the linear scope→research→write pipeline with a **Verification-Correction Loop** that operates at the global architecture level, distinct from the existing local task-execution loops.

### Extended Main Pipeline

```
START → clarify_with_user() → write_research_brief() → research_supervisor → 
final_report_generation() → verify_factual_consistency() → route_based_on_quality()
                                                               ↓              ↓
                                                              END    ← refine_research()
                                                                            ↓
                                                                    research_supervisor
```

### New Nodes

#### 1. `verify_factual_consistency(state: AgentState, config: RunnableConfig) -> Command`

**Input:**
- `state["final_report"]`: Generated report content
- `state["notes"]`: Compressed research findings from supervisor
- `state["raw_notes"]`: Uncompressed source material

**Responsibilities:**
- Extract structured claims from final report with confidence scores
- Detect factual contradictions between claims
- Identify unsupported assertions (claims without source evidence)
- Calculate consistency metrics

**Output:**
- `state["verification_report"]`: Structured assessment with detected issues
- `state["consistency_score"]`: Quantitative quality metric
- `state["flagged_claims"]`: List of problematic statements requiring attention

**Implementation Strategy:**
- **Rule-based detection:** Numerical contradictions, date conflicts, categorical oppositions
- **LLM-based detection:** Semantic conflicts using structured output (similar to `ClarifyWithUser`, `ResearchQuestion` patterns)
- **Evidence tracing:** Map each claim to supporting sources in raw_notes

#### 2. `route_based_on_quality(state: AgentState, config: RunnableConfig) -> Command[Literal["refine_research", "__end__"]]`

**Responsibilities:**
- Evaluate verification results against quality thresholds
- Decide whether to terminate the research process or continue refinement
- Enforce iteration limits to prevent infinite loops
- Append transparency disclaimers when quality thresholds not met within iteration budget

**Decision Logic:**
```python
quality_threshold_met = state["consistency_score"] >= config.rofc_consistency_threshold
max_iterations_exceeded = state["rofc_iterations"] >= config.max_rofc_iterations
critical_errors_found = any(claim.severity == "critical" for claim in state["flagged_claims"])

if quality_threshold_met and not critical_errors_found:
    return Command(goto=END)
elif max_iterations_exceeded:
    return Command(goto=END, update={"final_report": append_verification_disclaimer(...)})
else:
    return Command(goto="refine_research")
```

**Exit Guarantees:**
- Hard iteration limit prevents infinite loops
- Terminates with verification disclaimer if quality threshold not met within limit

#### 3. `refine_research(state: AgentState, config: RunnableConfig) -> Command[Literal["research_supervisor"]]`

**Responsibilities:**
- Convert `flagged_claims` into targeted research queries
- Preserve existing research findings in state (do NOT clear `notes` or `raw_notes`)
- Inject refinement context into supervisor prompt

**Output:**
```python
return Command(
    goto="research_supervisor",
    update={
        "supervisor_messages": {
            "type": "override",
            "value": [
                SystemMessage(content=lead_researcher_prompt.format(...)),
                HumanMessage(content=refinement_brief)
            ]
        },
        "rofc_iterations": state["rofc_iterations"] + 1
    }
)
```

**Refinement Brief Structure:**
```
Previous research identified the following factual inconsistencies:
1. [Claim A] conflicts with [Claim B] regarding [topic]
   - Source for A: [citation]
   - Source for B: [citation]
   - Resolution needed: Verify which is correct or explain discrepancy

2. [Claim C] lacks supporting evidence
   - Required: Find authoritative sources confirming or refuting this claim

Conduct focused research to resolve these specific issues.
```

### Cost Mitigation: Adaptive Model Selection

ROFC's verification-correction loop introduces additional LLM calls, increasing operational costs. To mitigate this overhead, we implement **Adaptive Model Selection (AMS)**—a runtime optimization that dynamically selects the most cost-effective model for each task while maintaining quality standards.

#### Design Principle

Rather than using fixed models throughout the pipeline, AMS adapts model selection based on:
- **Task complexity:** Simple tasks use cheaper models, complex tasks escalate to premium models
- **Iteration context:** Initial attempts use cost-efficient models, refinement iterations escalate as needed
- **Quality signals:** Verification failures trigger automatic model escalation

#### Integration Strategy

AMS operates transparently within existing node implementations:

```python
# Before each LLM call, select optimal model
async def verify_factual_consistency(state: AgentState, config: RunnableConfig):
    # Determine verification complexity
    complexity_score = estimate_verification_complexity(
        report_length=len(state["final_report"]),
        research_units=len(state["notes"]),
        iteration=state["rofc_iterations"]
    )
    
    # Select model based on complexity and iteration
    if state["rofc_iterations"] == 0 and complexity_score < 0.6:
        verification_model = "openai:gpt-4.1-mini"  # Cost-efficient first pass
    elif state["rofc_iterations"] >= 1:
        verification_model = "anthropic:claude-opus-4"  # Escalate on refinement
    else:
        verification_model = "openai:gpt-4.1"  # Standard quality
```

#### Selection Criteria by Node

1. **`verify_factual_consistency()`**: Escalate on refinement iterations
2. **`refine_research()` → `research_supervisor`**: Use premium model for targeted re-research
3. **`final_report_generation()`**: Maintain consistent model (quality-critical)

#### Cost-Quality Trade-off

```
Iteration 0 (Initial): Fast model for verification → Detect issues
                        ↓ (if issues found)
Iteration 1 (Refine):   Premium model for verification + research → Resolve conflicts
                        ↓ (if still issues)
Iteration 2 (Final):    Premium model throughout → Maximum quality
```

**Expected savings:** 20-30% cost reduction compared to always using premium models, while maintaining comparable quality through strategic escalation.

---

## State Schema Extensions

### New State Fields

```python
class AgentState(MessagesState):
    # Existing fields
    supervisor_messages: Annotated[list[MessageLikeRepresentation], override_reducer]
    research_brief: Optional[str]
    raw_notes: Annotated[list[str], override_reducer] = []
    notes: Annotated[list[str], override_reducer] = []
    final_report: str
    
    # ROFC additions
    verification_report: Optional[VerificationReport] = None
    consistency_score: float = 0.0
    flagged_claims: Annotated[list[FlaggedClaim], operator.add] = []
    rofc_iterations: int = 0
    
    # AMS tracking (optional telemetry)
    model_selections: Annotated[list[ModelSelectionRecord], operator.add] = []
    cumulative_cost: float = 0.0
```

### New Structured Outputs

```python
class FlaggedClaim(BaseModel):
    """Represents a claim requiring attention."""
    claim_text: str
    issue_type: Literal["contradiction", "unsupported", "ambiguous"]
    severity: Literal["critical", "moderate", "minor"]
    conflicting_sources: list[str] = []
    suggested_query: str  # For targeted refinement

class VerificationReport(BaseModel):
    """Structured output from verify_factual_consistency."""
    overall_assessment: str
    consistency_score: float
    flagged_claims: list[FlaggedClaim]
    verified_claims_count: int
    total_claims_count: int

class ModelSelectionRecord(BaseModel):
    """Telemetry for adaptive model selection."""
    node_name: str
    selected_model: str
    complexity_score: float
    iteration: int
    timestamp: float
    estimated_cost: float
```

---

## Configuration Extensions

```python
class Configuration(BaseModel):
    # Existing fields...
    
    # ROFC configuration
    enable_rofc: bool = Field(
        default=False,
        metadata={"description": "Enable Reflective Orchestration for Factual Consistency"}
    )
    
    max_rofc_iterations: int = Field(
        default=2,
        metadata={
            "description": "Maximum verification-correction cycles",
            "min": 1,
            "max": 5
        }
    )
    
    rofc_consistency_threshold: float = Field(
        default=0.85,
        metadata={
            "description": "Minimum consistency score to terminate loop",
            "min": 0.0,
            "max": 1.0
        }
    )
    
    verification_model: str = Field(
        default="openai:gpt-4.1",
        metadata={"description": "Model for factual consistency verification"}
    )
    
    # Adaptive Model Selection (cost mitigation)
    enable_adaptive_model_selection: bool = Field(
        default=False,
        metadata={"description": "Enable dynamic model selection based on task complexity and iteration"}
    )
    
    model_pool: list[str] = Field(
        default=["openai:gpt-4.1-mini", "openai:gpt-4.1", "anthropic:claude-opus-4"],
        metadata={"description": "Available models for adaptive selection (ordered by cost)"}
    )
    
    complexity_escalation_threshold: float = Field(
        default=0.6,
        metadata={
            "description": "Complexity score above which to use premium models",
            "min": 0.0,
            "max": 1.0
        }
    )
```

---

## Integration Points

### Modified Graph Construction

```python
# Current (deep_researcher.py:699-719)
deep_researcher_builder = StateGraph(AgentState, input=AgentInputState, config_schema=Configuration)
deep_researcher_builder.add_node("clarify_with_user", clarify_with_user)
deep_researcher_builder.add_node("write_research_brief", write_research_brief)
deep_researcher_builder.add_node("research_supervisor", supervisor_subgraph)
deep_researcher_builder.add_node("final_report_generation", final_report_generation)

deep_researcher_builder.add_edge(START, "clarify_with_user")
deep_researcher_builder.add_edge("research_supervisor", "final_report_generation")
deep_researcher_builder.add_edge("final_report_generation", END)  # MODIFIED

# Proposed
deep_researcher_builder.add_node("verify_factual_consistency", verify_factual_consistency)
deep_researcher_builder.add_node("refine_research", refine_research)

# Conditional edge based on enable_rofc config
def should_verify(state: AgentState) -> bool:
    config = Configuration.from_runnable_config(...)
    return config.enable_rofc

deep_researcher_builder.add_conditional_edges(
    "final_report_generation",
    lambda state: "verify_factual_consistency" if should_verify(state) else END
)

deep_researcher_builder.add_edge("verify_factual_consistency", "route_based_on_quality")
# route_based_on_quality has internal conditional logic for END vs refine_research
deep_researcher_builder.add_edge("refine_research", "research_supervisor")
```

### State Preservation Strategy

**Critical modification to `final_report_generation()`:**

```python
# Current behavior (line 622)
cleared_state = {"notes": {"type": "override", "value": []}}  # PROBLEM: Destroys evidence

# ROFC-compatible behavior
if config.enable_rofc:
    cleared_state = {}  # Preserve notes and raw_notes for verification
else:
    cleared_state = {"notes": {"type": "override", "value": []}}  # Original behavior
```

---

## Key Design Decisions

### 1. Why Global Cycles vs. Enhanced Local Cycles?

**Alternative rejected:** Add verification within `compress_research()` or `supervisor_tools()`

**Rationale:**
- Compression focuses on individual research unit synthesis, not cross-unit consistency
- Supervisor operates on delegation logic, not final report quality
- Global verification requires access to synthesized final report + all supporting evidence
- Architectural clarity: Separation of concerns between execution (local) and quality assurance (global)

### 2. Why Preserve Existing Loops Unchanged?

- Existing loops are optimized for their specific purposes
- Backward compatibility maintained
- ROFC operates at higher abstraction level
- Can be feature-flagged without affecting baseline behavior

### 3. Why Route Through `research_supervisor` vs. Direct Researcher Access?

- Maintains existing abstraction boundaries
- Leverages supervisor's delegation and coordination logic
- Avoids duplicating research orchestration code
- Enables supervisor to strategically plan refinement (vs. rigid query execution)

---

## Expected Outcomes

### Architectural Benefits

1. **Formal quality assurance:** Explicit verification step with measurable metrics
2. **Targeted refinement:** Only re-research specific gaps rather than full re-execution
3. **Graceful degradation:** System terminates with transparency if quality threshold not met
4. **Backward compatibility:** Existing linear behavior preserved when `enable_rofc=False`

### Limitations & Trade-offs

1. **Latency increase:** Additional verification pass + potential refinement cycles
2. **Cost increase:** Extra LLM calls for verification and potential re-research (mitigated by Adaptive Model Selection)
3. **Complexity increase:** More state management and conditional routing logic
4. **No guarantee of perfection:** Quality threshold is heuristic, not ground truth

---

## Implementation Phases

### Phase 1: Minimal Viable ROFC
- Implement `verify_factual_consistency()` with rule-based detection only
- Add `route_based_on_quality()` with hard iteration limits
- Basic `refine_research()` that regenerates research brief from flagged claims
- Feature flag integration
- **Goal:** Prove cycle execution and termination guarantees

### Phase 2: Enhanced Detection & Cost Optimization
- Add LLM-based semantic conflict detection
- Implement confidence scoring and threshold filtering
- Evidence tracing from claims to raw_notes
- **Implement Adaptive Model Selection (AMS)**:
  - Complexity estimation heuristics
  - Iteration-based escalation logic
  - Cost tracking and telemetry
- **Goal:** Improve detection precision/recall while controlling costs

### Phase 3: Intelligent Refinement
- Targeted query generation from flagged claims
- Preserve vs. augment existing findings
- Refine AMS escalation strategies based on Phase 2 telemetry
- **Goal:** Minimize unnecessary re-research while optimizing cost-quality trade-offs

### Phase 4: Production Readiness
- Comprehensive telemetry and cost tracking
- Performance optimization (caching, batching)
- Edge case handling (timeout recovery, API failures)
- Documentation and configuration best practices
- **Goal:** Deployable system with operational reliability