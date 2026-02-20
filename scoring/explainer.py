"""Score Explainer — generates detailed, human-readable scoring explanations.

Produces structured explanations suitable for display in the Streamlit dashboard.
"""

from typing import Dict, List
from loguru import logger


class ScoreExplainer:
    """Generate detailed explanations for scoring results."""

    def generate_full_explanation(self, score_result) -> Dict:
        """
        Generate a complete explanation package from a ScoreResult.

        Returns:
            Dict with sections: summary, skill_analysis, experience_analysis,
            project_analysis, education_analysis, penalties, recommendations.
        """
        explanation = {
            "summary": self._generate_summary(score_result),
            "skill_analysis": self._explain_skills(score_result),
            "experience_analysis": self._explain_experience(score_result),
            "project_analysis": self._explain_projects(score_result),
            "education_analysis": self._explain_education(score_result),
            "penalties_and_bonuses": self._explain_adjustments(score_result),
            "score_breakdown": score_result.score_reasoning,
        }
        return explanation

    def _generate_summary(self, result) -> str:
        """Generate a one-paragraph summary of the evaluation."""
        score = result.overall_score

        if score >= 80:
            verdict = "Excellent match"
            emoji = "🟢"
        elif score >= 65:
            verdict = "Good match"
            emoji = "🟡"
        elif score >= 50:
            verdict = "Moderate match"
            emoji = "🟠"
        else:
            verdict = "Weak match"
            emoji = "🔴"

        summary = (
            f"{emoji} **{verdict}** — Overall Score: **{score}/100**\n\n"
            f"This resume scores **{result.skill_match_score:.0f}%** on skill alignment, "
            f"**{result.experience_score:.0f}%** on experience match, and "
            f"**{result.project_relevance_score:.0f}%** on project relevance. "
        )

        if result.missing_skills:
            n = len(result.missing_skills)
            summary += f"\n\n⚠️ {n} required skill{'s' if n > 1 else ''} not found on resume."

        if result.strengths:
            summary += f"\n\n✅ **Top strengths**: {'; '.join(result.strengths[:3])}"

        return summary

    def _explain_skills(self, result) -> Dict:
        """Detailed skill matching explanation."""
        strong_matches = [
            m for m in result.matched_skills
            if m.get("similarity", 0) >= 0.7 and not m.get("partial")
        ]
        partial_matches = [
            m for m in result.matched_skills
            if m.get("partial") or 0.35 <= m.get("similarity", 0) < 0.7
        ]

        explanation = {
            "score": result.skill_match_score,
            "strong_matches": [
                f"✅ {m['jd_skill']} ↔ {m['resume_skill']} ({m['similarity']:.0%})"
                for m in strong_matches
            ],
            "partial_matches": [
                f"🟡 {m['jd_skill']} ~ {m['resume_skill']} ({m['similarity']:.0%})"
                for m in partial_matches
            ],
            "missing": [
                f"❌ {m['skill']} (best match: {m.get('best_match', 'N/A')} at {m.get('best_similarity', 0):.0%})"
                for m in result.missing_skills
            ],
            "extra_skills": result.extra_skills[:10],
        }
        return explanation

    def _explain_experience(self, result) -> Dict:
        """Experience analysis explanation."""
        exp = result.experience_analysis
        return {
            "score": result.experience_score,
            "total_years": exp.get("total_years", 0),
            "seniority_level": exp.get("seniority_level", "unknown"),
            "signals": exp.get("seniority_signals", []),
            "quality": exp.get("experience_quality", {}),
        }

    def _explain_projects(self, result) -> Dict:
        """Project relevance explanation."""
        return {
            "score": result.project_relevance_score,
            "projects": [
                f"{'🟢' if p['relevance'] >= 0.6 else '🟡' if p['relevance'] >= 0.4 else '🔴'} "
                f"{p['name']} — {p['relevance']:.0%} relevant"
                for p in result.project_scores
            ],
        }

    def _explain_education(self, result) -> Dict:
        """Education relevance explanation."""
        return {
            "score": result.education_score,
            "note": (
                "Education alignment is measured semantically against the JD. "
                "Relevant degrees and fields of study increase this score."
            ),
        }

    def _explain_adjustments(self, result) -> Dict:
        """Explain bonuses and penalties."""
        items = []
        if result.certification_bonus > 0:
            items.append(f"🎖 Certification bonus: +{result.certification_bonus:.1f} pts")
        if result.missing_skill_penalty > 0:
            items.append(f"⚠️ Missing skills penalty: -{result.missing_skill_penalty:.1f} pts")
        return {"items": items}
