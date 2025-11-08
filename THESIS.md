# Reflective Orchestration for Factual Consistency in LangGraph Research Agents

The langchain-ai/open_deep_research repository supports implementing **Reflective Orchestration for Factual Consistency (ROFC)** - a cyclic subgraph that detects and resolves factual conflicts during research. The current linear pipeline (Scope→Research→Write) can be enhanced with an internal consistency loop without breaking existing functionality.

**ROFC Flow**: `plan_research` → `execute_retrieval` → `synthesize_claims` → `reflect_on_consistency` → `revise_claims` → **loop until consistent** → `final_report`

---

## Key Technical Findings

### 1. **Current Architecture Analysis**
- **Linear Research Confirmed**: The existing research supervisor (`deep_researcher.py:178-349`) performs single-pass delegation without internal consistency checking
- **Clean Insertion Point**: ROFC can replace the `research_supervisor` node at `deep_researcher.py:710` with zero breaking changes
- **State Management**: The existing `override_reducer` system in `state.py:55-61` supports cyclic state updates required for ROFC

### 2. **Implementation Requirements**

#### New ROFC Nodes (3 core functions)
```python
# A) Extract structured claims with confidence scores
async def synthesize_claims(state, config) -> Command["reflect_on_consistency"]

# B) Detect contradictions using rules + LLM analysis  
async def reflect_on_consistency(state, config) -> Command["revise_claims" | "final_report"]

# C) Resolve conflicts via targeted retrieval or model escalation
async def revise_claims(state, config) -> Command["synthesize_claims" | "final_report"]
```

#### State Schema Extensions
```python
class ROFCState(TypedDict):
    claims: list[Claim] = []                    # Structured factual claims
    evidence_map: dict[str, list[str]] = {}     # Evidence-to-source mapping  
    conflicts: list[Conflict] = []              # Detected contradictions
    rofc_iterations: int = 0                    # Loop counter
    rofc_status: Optional[str] = None           # Terminal state indicator
```

### 3. **Conflict Detection Strategy**
- **Rule-Based**: Fast detection of numerical contradictions, date conflicts, categorical oppositions
- **LLM-Based**: Semantic conflict detection using structured output for nuanced contradictions
- **Confidence Thresholding**: Only flag conflicts above configurable certainty levels (default: 0.3)

### 4. **Resolution Mechanisms**
- **Targeted Retrieval**: Re-query search APIs with conflict-specific terms
- **Model Escalation**: Optional upgrade to higher-capability model (e.g., GPT-4.1 → Claude-Opus-4)
- **Citation Transparency**: Preserve conflicting sources with methodological differences noted

---

## Implementation Plan

### **File Modifications** (~450 LOC total)

#### New Files (~300 LOC)
- `src/open_deep_research/rofc_nodes.py` - Core ROFC node implementations
- `src/open_deep_research/rofc_telemetry.py` - Cost/latency instrumentation  
- `tests/rofc_evaluation.py` - ROFC vs baseline evaluation harness
- `tests/enterprise_conflicts.json` - 20-item conflict detection micro-eval

#### Modified Files (~150 LOC)  
- `state.py` +30 LOC - Add ROFCState, Claim, Conflict models
- `configuration.py` +40 LOC - Add ROFC feature flags and limits
- `deep_researcher.py` +50 LOC - Replace supervisor with ROFC subgraph
- `evaluators.py` +30 LOC - Add consistency-specific evaluation metrics

### **Configuration Toggles**
```python
# Feature flags for gradual rollout
enable_rofc: bool = False                    # Master switch
max_rofc_iterations: int = 3                 # Hard termination limit
rofc_model_escalation: bool = False          # Premium model for conflicts
rofc_confidence_threshold: float = 0.3       # Conflict detection sensitivity
```

### **Evaluation Framework**
- **Existing**: Leverage "Deep Research Bench" dataset and 6-dimensional quality scoring
- **New**: Head-to-head ROFC vs baseline comparison on groundedness and factual accuracy
- **Micro-eval**: 20 planted conflicts with pass/fail criteria (detect + resolve OR report discrepancy)

---

## Risk Analysis & Mitigations

| **Risk** | **Impact** | **Mitigation** |
|----------|------------|----------------|
| Infinite loops | Execution hangs | `max_rofc_iterations=3`, 5-minute timeouts |
| State explosion | Memory issues | Max 50 claims/iteration, garbage collection |
| Citation drift | Lost source URLs | Immutable evidence_map with UUID tracking |
| Tool flakiness | Search failures | Exponential backoff, fallback to existing findings |
| False positives | Unnecessary loops | Confidence thresholding, rule validation |

---

## Success Metrics

### **Quantitative Targets**
- **Groundedness Improvement**: +15% fewer factual errors vs baseline
- **Consistency Score**: >90% claim coherence in final reports  
- **Latency Impact**: <50% increase in total research time
- **Cost Impact**: <30% increase in token usage

### **Qualitative Indicators**
- Conflicting claims properly flagged and addressed
- Source methodology differences transparently reported
- Higher confidence in research conclusions
- Maintained research depth and breadth

---

## Next Steps

### **Phase 1: Core Implementation** (Week 1-2)
1. Implement basic ROFC nodes with rule-based conflict detection
2. Add configuration flags and state schema extensions
3. Create integration test proving loop execution and termination
4. Deploy behind `enable_rofc=false` feature flag

### **Phase 2: Advanced Features** (Week 3-4)  
1. Add LLM-based semantic conflict detection
2. Implement model escalation for complex conflicts
3. Build targeted retrieval for conflict resolution
4. Add comprehensive telemetry and cost tracking

### **Phase 3: Evaluation & Optimization** (Week 5-6)
1. Run ROFC vs baseline evaluation on Deep Research Bench
2. Execute enterprise conflict micro-evaluations  
3. Optimize performance based on real-world usage patterns
4. Document best practices and configuration recommendations