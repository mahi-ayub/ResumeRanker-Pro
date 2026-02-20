"""Dynamic Weight Calculator — adjusts scoring weights based on role type.

Weights are derived from the JD analysis role classification.
Each role type has a weight profile that amplifies relevant dimensions
and dampens less relevant ones.
"""

from typing import Dict
from loguru import logger

from utils.config import ScoringConfig, ROLE_WEIGHT_PROFILES


class DynamicWeightCalculator:
    """Compute role-aware scoring weights from JD analysis."""

    def __init__(self, base_config: ScoringConfig = None):
        self.base = base_config or ScoringConfig()

    def compute_weights(self, jd_analysis: Dict) -> Dict[str, float]:
        """
        Compute final scoring weights based on the detected role type.

        The base weights are multiplied by role-specific adjustment factors,
        then renormalized so that the main weights sum to 1.0.

        Args:
            jd_analysis: Output of JDAnalyzer.analyze().

        Returns:
            Dict of weight name → final weight value.
        """
        role_type = jd_analysis.get("role_type", "default")
        role_profile = ROLE_WEIGHT_PROFILES.get(role_type, ROLE_WEIGHT_PROFILES["default"])

        # Apply multipliers to base weights
        raw_weights = {
            "skill_match": self.base.skill_match_weight * role_profile.get("skill_match_weight", 1.0),
            "experience": self.base.experience_weight * role_profile.get("experience_weight", 1.0),
            "project_relevance": self.base.project_relevance_weight * role_profile.get("project_relevance_weight", 1.0),
            "education": self.base.education_weight * role_profile.get("education_weight", 1.0),
        }

        # Normalize main weights to sum to ~0.90 (leaving room for bonuses/penalties)
        total = sum(raw_weights.values())
        target_sum = 0.90
        weights = {k: round(v / total * target_sum, 4) for k, v in raw_weights.items()}

        # Add bonus/penalty weights (not normalized)
        weights["certification_bonus"] = self.base.certification_bonus
        weights["missing_skill_penalty"] = self.base.missing_skill_penalty
        weights["max_missing_penalty"] = self.base.max_missing_penalty

        logger.info(
            f"Dynamic weights for role '{role_type}': "
            f"skill={weights['skill_match']:.2f}, exp={weights['experience']:.2f}, "
            f"proj={weights['project_relevance']:.2f}, edu={weights['education']:.2f}"
        )

        return weights

    def explain_weights(self, weights: Dict[str, float], role_type: str) -> str:
        """Generate a human-readable explanation of the weight distribution."""
        explanations = [
            f"**Role detected**: {role_type.replace('_', ' ').title()}",
            "",
            "**Weight Distribution:**",
            f"- Skill Match: {weights.get('skill_match', 0):.0%}",
            f"- Experience: {weights.get('experience', 0):.0%}",
            f"- Project Relevance: {weights.get('project_relevance', 0):.0%}",
            f"- Education: {weights.get('education', 0):.0%}",
            "",
            f"- Certification Bonus: +{weights.get('certification_bonus', 0):.0%} per cert",
            f"- Missing Skill Penalty: -{weights.get('missing_skill_penalty', 0):.0%} per skill",
        ]

        # Role-specific notes
        role_notes = {
            "ml_engineer": "ML roles weight skill match and project relevance higher due to specialized tool requirements.",
            "backend_engineer": "Backend roles emphasize skill match and experience depth.",
            "frontend_engineer": "Frontend roles value skill match and project portfolio.",
            "data_scientist": "Data science roles weight education and research more heavily.",
            "devops_engineer": "DevOps roles emphasize hands-on skill match and experience.",
        }

        note = role_notes.get(role_type, "Weights balanced for general engineering roles.")
        explanations.append(f"\n**Reasoning**: {note}")

        return "\n".join(explanations)
