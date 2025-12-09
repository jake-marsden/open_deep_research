"""Adaptive model selection service for the Deep Research agent.

This module implements the tier resolution logic that selects the appropriate
model based on task complexity assessment. It integrates insights from multiple
academic papers on LLM cost optimization:

- MDAgents: Discrete tier classification (low/mid/high)
- LLMRank: Feature-based complexity prediction
- Hybrid LLM: Quality gap for tier eligibility
- Early Abstention: Confidence thresholds for safe defaults
- Mixture of Thought: MoT signals for high-tier escalation
- FrugalGPT: Context estimation for token limit handling
- Model Multiplexing: Reliability-to-cost optimization
"""

import logging
from dataclasses import dataclass
from typing import Optional

from open_deep_research.configuration import AdaptiveModelConfig, ModelTierConfig
from open_deep_research.state import ComplexityAssessment, ContextEstimation, FeatureExtraction
from open_deep_research.tier_logger import get_tier_logger

logger = logging.getLogger(__name__)


@dataclass
class TierDecision:
    """Result of tier resolution with decision metadata."""
    
    tier: str
    model_config: ModelTierConfig
    original_tier: str
    was_overridden: bool
    override_reason: Optional[str]
    
    def to_log_dict(self) -> dict:
        """Convert to dictionary for logging."""
        return {
            "tier": self.tier,
            "model": self.model_config.model,
            "max_tokens": self.model_config.max_tokens,
            "original_tier": self.original_tier,
            "was_overridden": self.was_overridden,
            "override_reason": self.override_reason
        }


class ModelSelector:
    """Service for selecting the appropriate model tier based on task complexity.
    
    Implements a multi-stage tier resolution process:
    1. Accept supervisor's tier recommendation
    2. Apply safety escalation overrides (Early Abstention principle)
    3. Enforce context window constraints (FrugalGPT principle)
    4. Apply cost optimization downgrades (Hybrid LLM quality gap principle)
    
    The goal is to minimize cost while ensuring task success, with a bias
    toward higher tiers when uncertainty exists (since failed tasks cannot
    be re-routed in this stateless architecture).
    """
    
    def __init__(self, config: AdaptiveModelConfig):
        """Initialize the model selector with configuration.
        
        Args:
            config: Adaptive model selection configuration with tier definitions
                   and threshold settings.
        """
        self.config = config
    
    def resolve_tier(self, assessment: ComplexityAssessment) -> TierDecision:
        """Resolve the appropriate model tier based on complexity assessment.
        
        This is the main entry point for tier selection. It applies multiple
        decision rules in sequence to determine the final tier.
        
        Args:
            assessment: ComplexityAssessment from the supervisor containing
                       tier recommendation and optional feature extraction.
        
        Returns:
            TierDecision with the resolved tier and decision metadata.
        """
        original_tier = assessment.tier
        current_tier = assessment.tier
        override_reason = None
        
        # Ensure features exist (use defaults if not provided)
        # Default to "synthesis" as the safest assumption for task_type
        features = assessment.features or FeatureExtraction(task_type="synthesis")
        context_est = assessment.context_estimation or self._infer_context_estimation(features)
        
        # Stage 1: Safety Escalation Overrides (Early Abstention, MoT)
        escalation_result = self._check_safety_escalation(assessment, features, current_tier)
        if escalation_result:
            current_tier, override_reason = escalation_result
        
        # Stage 2: Context Window Constraint Check (FrugalGPT)
        if not override_reason:  # Only if not already escalated
            context_result = self._enforce_context_window_constraint(context_est, current_tier)
            if context_result:
                current_tier, override_reason = context_result
        
        # Stage 3: Cost Optimization Downgrade (Hybrid LLM quality gap)
        # Only apply if we haven't escalated and conditions permit
        if not override_reason and self._can_downgrade(assessment, current_tier):
            downgrade_result = self._apply_cost_optimization(assessment, features, current_tier)
            if downgrade_result:
                current_tier, override_reason = downgrade_result
        
        # Get the final tier configuration
        model_config = self.config.get_tier_config(current_tier)
        
        decision = TierDecision(
            tier=current_tier,
            model_config=model_config,
            original_tier=original_tier,
            was_overridden=(current_tier != original_tier),
            override_reason=override_reason
        )
        
        # Log the decision if enabled
        if self.config.log_tier_decisions:
            self._log_decision(assessment, features, decision)
        
        # Track tier classification in logs.txt
        tier_logger = get_tier_logger()
        tier_logger.log_tier_classification(current_tier)
        
        return decision
    
    def _infer_context_estimation(self, features: "FeatureExtraction") -> "ContextEstimation":
        """Infer context estimation from task features when not provided."""
        from open_deep_research.state import ContextEstimation
        
        # Map task type to token estimates and retrieval breadth
        task_config = {
            "retrieval": {"tokens": (1500, 500), "breadth": "narrow"},
            "reasoning": {"tokens": (3000, 1500), "breadth": "moderate"},
            "synthesis": {"tokens": (5000, 2000), "breadth": "extensive"},
            "generation": {"tokens": (4000, 3000), "breadth": "moderate"}
        }
        
        config = task_config.get(features.task_type, {"tokens": (3000, 1000), "breadth": "moderate"})
        input_tokens, output_tokens = config["tokens"]
        
        return ContextEstimation(
            estimated_input_tokens=input_tokens,
            estimated_output_tokens=output_tokens,
            retrieval_breadth=config["breadth"],
            context_accumulation="moderate"
        )
    
    def _check_safety_escalation(
        self, 
        assessment: ComplexityAssessment,
        features: "FeatureExtraction",
        current_tier: str
    ) -> Optional[tuple[str, str]]:
        """Check if task should be escalated to a higher tier for safety.
        
        Implements the Early Abstention principle: if the supervisor cannot
        confidently classify a task as low complexity, escalate to avoid
        the cost of task failure (which cannot be recovered in stateless arch).
        
        Also implements Mixture of Thought (MoT) signal: tasks requiring both
        narrative reasoning AND symbolic execution should go to high-tier.
        
        Args:
            assessment: The complexity assessment from supervisor.
            features: The extracted features (with defaults applied).
            current_tier: The currently assigned tier.
        
        Returns:
            Tuple of (new_tier, reason) if escalation needed, None otherwise.
        """
        # Check for Mixture of Thought signal (requires high-tier)
        # Per MoT paper: tasks needing both reasoning types benefit from synergy
        mixture_of_thought_required = (
            features.requires_narrative_reasoning and 
            features.requires_symbolic_execution
        )
        if mixture_of_thought_required and current_tier != "high":
            return ("high", "MoT signal: requires both narrative reasoning and symbolic execution")
        
        # Check confidence threshold for high-tier escalation
        if assessment.estimated_confidence < self.config.confidence_threshold_mid:
            if current_tier != "high":
                return ("high", f"Low confidence ({assessment.estimated_confidence}%) below mid threshold ({self.config.confidence_threshold_mid}%)")
        
        # Check for high failure risk
        if assessment.failure_risk == "high" and current_tier != "high":
            return ("high", "High failure risk requires high-tier model")
        
        # Check for contradictory context (ambiguous tasks)
        if features.context_difficulty == "contradictory" and current_tier != "high":
            return ("high", "Contradictory context requires high-tier reasoning")
        
        # Check for expert-level domain with ambiguous context
        if (features.domain_signal == "expert-level" and 
            features.context_difficulty == "ambiguous" and 
            current_tier == "low"):
            return ("mid", "Expert domain with ambiguous context requires at least mid-tier")
        
        # Check confidence threshold for mid-tier escalation (from low)
        if current_tier == "low":
            if assessment.estimated_confidence < self.config.confidence_threshold_low:
                return ("mid", f"Confidence ({assessment.estimated_confidence}%) below low-tier threshold ({self.config.confidence_threshold_low}%)")
        
        return None
    
    def _enforce_context_window_constraint(
        self, 
        context_est: "ContextEstimation", 
        current_tier: str
    ) -> Optional[tuple[str, str]]:
        """Force tier upgrade if context requirements exceed model capacity.
        
        Implements FrugalGPT principle: lower-tier models often have smaller
        context windows. Tasks requiring extensive retrieval should be
        escalated regardless of semantic complexity.
        
        Args:
            context_est: The context estimation (with defaults applied).
            current_tier: The currently assigned tier.
        
        Returns:
            Tuple of (new_tier, reason) if upgrade needed, None otherwise.
        """
        # Calculate estimated total context needs
        base_tokens = context_est.estimated_input_tokens + context_est.estimated_output_tokens
        
        # Apply multiplier for retrieval breadth
        breadth_multipliers = {
            "narrow": 1.0,
            "moderate": 2.0,
            "extensive": 4.0
        }
        breadth_factor = breadth_multipliers.get(context_est.retrieval_breadth, 1.5)
        
        # Apply buffer for context accumulation
        accumulation_buffers = {
            "minimal": 1.0,
            "moderate": 1.5,
            "heavy": 2.5
        }
        accumulation_factor = accumulation_buffers.get(context_est.context_accumulation, 1.5)
        
        estimated_total = base_tokens * breadth_factor * accumulation_factor
        safety_margin = self.config.context_window_safety_margin
        
        # Check against tier context windows
        if current_tier == "low":
            low_limit = self.config.low_tier.context_window * safety_margin
            if estimated_total > low_limit:
                # Check if mid-tier suffices
                mid_limit = self.config.mid_tier.context_window * safety_margin
                if estimated_total > mid_limit:
                    return ("high", f"Estimated context ({int(estimated_total)}) exceeds mid-tier window")
                return ("mid", f"Estimated context ({int(estimated_total)}) exceeds low-tier window ({int(low_limit)})")
        
        elif current_tier == "mid":
            mid_limit = self.config.mid_tier.context_window * safety_margin
            if estimated_total > mid_limit:
                return ("high", f"Estimated context ({int(estimated_total)}) exceeds mid-tier window ({int(mid_limit)})")
        
        return None
    
    def _can_downgrade(
        self, 
        assessment: ComplexityAssessment, 
        current_tier: str
    ) -> bool:
        """Check if downgrade optimization is even possible.
        
        Args:
            assessment: The complexity assessment.
            current_tier: The currently assigned tier.
        
        Returns:
            True if downgrade might be possible, False otherwise.
        """
        # Can't downgrade from low
        if current_tier == "low":
            return False
        
        # Don't downgrade if quality gap is significant
        if assessment.quality_gap_prediction == "significant":
            return False
        
        # Don't downgrade if failure risk is not low
        if assessment.failure_risk != "low":
            return False
        
        return True
    
    def _apply_cost_optimization(
        self, 
        assessment: ComplexityAssessment,
        features: "FeatureExtraction",
        current_tier: str
    ) -> Optional[tuple[str, str]]:
        """Apply cost optimization by downgrading tier if safe.
        
        Implements Hybrid LLM quality gap principle: for tasks where the
        quality gap between tiers is negligible (~20% of queries), we can
        safely use a lower-tier model.
        
        Args:
            assessment: The complexity assessment.
            features: The extracted features (with defaults applied).
            current_tier: The currently assigned tier.
        
        Returns:
            Tuple of (new_tier, reason) if downgrade applied, None otherwise.
        """
        # Conditions for downgrading mid-tier to low-tier
        if current_tier == "mid":
            # Quality gap must be negligible
            if assessment.quality_gap_prediction != "negligible":
                return None
            
            # Must be high confidence
            if assessment.estimated_confidence < 95:
                return None
            
            # Task must be simple retrieval
            is_simple_retrieval = (
                features.task_type == "retrieval" and
                features.reasoning_pattern in {"single-hop", "none"}
            )
            
            if is_simple_retrieval:
                return ("low", "Cost optimization: negligible quality gap for simple retrieval")
        
        # Conditions for downgrading high-tier to mid-tier
        elif current_tier == "high":
            # Quality gap must be negligible
            if assessment.quality_gap_prediction != "negligible":
                return None
            
            # Must be very high confidence
            if assessment.estimated_confidence < 90:
                return None
            
            # Must not require mixture of thought
            if features.requires_narrative_reasoning and features.requires_symbolic_execution:
                return None
            
            # Must not have contradictory context
            if features.context_difficulty == "contradictory":
                return None
            
            # Task should be synthesis with clear context
            is_clear_synthesis = (
                features.task_type == "synthesis" and
                features.context_difficulty == "clear"
            )
            
            if is_clear_synthesis:
                return ("mid", "Cost optimization: negligible quality gap for clear synthesis")
        
        return None
    
    def _is_low_tier_eligible(
        self, 
        assessment: ComplexityAssessment,
        features: "FeatureExtraction"
    ) -> bool:
        """Check if task is eligible for low-tier based on quality gap.
        
        Per Hybrid LLM paper: Only ~20% of queries have negligible quality gap.
        Low-tier should be reserved for tasks where we're confident the
        output quality would be identical to higher tiers.
        
        Args:
            assessment: The complexity assessment.
            features: The extracted features (with defaults applied).
        
        Returns:
            True if low-tier is appropriate, False otherwise.
        """
        # Quality gap must be negligible
        if assessment.quality_gap_prediction != "negligible":
            return False
        
        # Must meet confidence threshold
        if assessment.estimated_confidence < self.config.confidence_threshold_low:
            return False
        
        # Must be low failure risk
        if assessment.failure_risk != "low":
            return False
        
        # Task characteristics for negligible quality gap
        return (
            features.task_type == "retrieval" and
            features.reasoning_pattern in {"single-hop", "none"} and
            features.domain_signal != "expert-level" and
            not features.requires_symbolic_execution
        )
    
    def _log_decision(
        self, 
        assessment: ComplexityAssessment,
        features: "FeatureExtraction",
        decision: TierDecision
    ) -> None:
        """Log the tier decision for analysis and threshold tuning.
        
        Args:
            assessment: The complexity assessment.
            features: The extracted features (with defaults applied).
            decision: The tier decision made.
        """
        if decision.was_overridden:
            logger.info(
                f"Tier override: {decision.original_tier} → {decision.tier} | "
                f"Reason: {decision.override_reason}"
            )
        else:
            logger.debug(
                f"Tier confirmed: {decision.tier} | "
                f"Confidence: {assessment.estimated_confidence}%"
            )


def select_model_for_task(
    assessment: ComplexityAssessment,
    config: AdaptiveModelConfig
) -> TierDecision:
    """Convenience function for selecting model tier.
    
    Args:
        assessment: ComplexityAssessment from the supervisor.
        config: AdaptiveModelConfig with tier definitions.
    
    Returns:
        TierDecision with the resolved tier and model configuration.
    """
    selector = ModelSelector(config)
    return selector.resolve_tier(assessment)

