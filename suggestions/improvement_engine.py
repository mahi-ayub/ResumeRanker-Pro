"""Improvement Engine — generates actionable resume improvement suggestions.

Analyzes scoring results and resume content to produce specific,
structured feedback across multiple dimensions:
- Missing skills to add
- Experience bullet improvements
- Quantified achievement suggestions
- Section restructuring
- Impact statement improvements
"""

from typing import List, Dict
from loguru import logger


class ImprovementEngine:
    """Generate actionable resume improvement suggestions from scoring results."""

    def generate_suggestions(self, score_result, resume_data: Dict) -> Dict:
        """
        Generate comprehensive improvement suggestions.

        Args:
            score_result: ScoreResult from the scoring pipeline.
            resume_data: Parsed resume data dict.

        Returns:
            Dict with categorized suggestions.
        """
        suggestions = {
            "critical": self._critical_suggestions(score_result),
            "skill_improvements": self._skill_suggestions(score_result),
            "experience_improvements": self._experience_suggestions(score_result, resume_data),
            "project_improvements": self._project_suggestions(score_result, resume_data),
            "formatting_tips": self._formatting_suggestions(resume_data),
            "impact_improvements": self._impact_suggestions(resume_data),
            "priority_actions": [],
        }

        # Compute top-3 priority actions
        suggestions["priority_actions"] = self._prioritize(suggestions)

        return suggestions

    def _critical_suggestions(self, result) -> List[Dict]:
        """High-priority suggestions for critical gaps."""
        suggestions = []

        # Missing critical skills
        if result.missing_skills:
            missing_names = [m.get("skill", "") for m in result.missing_skills]
            suggestions.append({
                "type": "missing_skills",
                "severity": "high",
                "title": "Add Missing Required Skills",
                "description": (
                    f"The following skills from the JD were not found on your resume: "
                    f"**{', '.join(missing_names[:7])}**. "
                    f"If you have experience with these, add them prominently to your skills section "
                    f"and reference them in your experience bullets."
                ),
                "skills": missing_names,
            })

        # Low overall score
        if result.overall_score < 50:
            suggestions.append({
                "type": "low_score",
                "severity": "high",
                "title": "Major Resume-JD Mismatch",
                "description": (
                    "Your resume scores below 50% match. Consider tailoring your resume "
                    "specifically for this role. Focus on: (1) adding relevant skills to the "
                    "skills section, (2) rewriting experience bullets to highlight relevant work, "
                    "(3) adding projects that demonstrate required competencies."
                ),
            })

        # Experience gap
        exp = result.experience_analysis
        if result.experience_score < 40:
            suggestions.append({
                "type": "experience_gap",
                "severity": "high",
                "title": "Experience Alignment Gap",
                "description": (
                    f"Your experience ({exp.get('total_years', 0)} years, "
                    f"{exp.get('seniority_level', 'unknown')} level) doesn't strongly align "
                    f"with the JD requirements. Reframe existing bullets to emphasize "
                    f"relevant technologies and responsibilities."
                ),
            })

        return suggestions

    def _skill_suggestions(self, result) -> List[Dict]:
        """Suggestions for improving skill section."""
        suggestions = []

        # Partially matched skills — strengthen these
        partial = [m for m in result.matched_skills if m.get("partial")]
        if partial:
            suggestions.append({
                "type": "partial_skills",
                "severity": "medium",
                "title": "Strengthen Partial Skill Matches",
                "description": (
                    f"These JD skills have only partial matches on your resume: "
                    f"{', '.join(m['jd_skill'] for m in partial[:5])}. "
                    f"Use exact terminology from the JD where possible."
                ),
            })

        # Extra skills — potential over-cluttering
        if len(result.extra_skills) > 15:
            suggestions.append({
                "type": "skill_clutter",
                "severity": "low",
                "title": "Streamline Skills Section",
                "description": (
                    f"Your resume lists {len(result.extra_skills)} skills not mentioned in the JD. "
                    f"Consider removing or de-emphasizing irrelevant skills to keep focus "
                    f"on what matters for this role."
                ),
            })

        # Skill grouping
        suggestions.append({
            "type": "skill_organization",
            "severity": "low",
            "title": "Organize Skills by Category",
            "description": (
                "Group skills under categories like: Languages, Frameworks, Databases, "
                "Cloud/DevOps, Tools. This makes scanning easier for recruiters and ATS systems."
            ),
        })

        return suggestions

    def _experience_suggestions(self, result, resume_data: Dict) -> List[Dict]:
        """Suggestions for improving experience section."""
        suggestions = []
        quality = result.experience_analysis.get("experience_quality", {})

        # Quantified achievements
        ratio = quality.get("quantified_ratio", 0)
        if ratio < 0.3:
            suggestions.append({
                "type": "quantify",
                "severity": "high",
                "title": "Add Quantified Achievements",
                "description": (
                    f"Only {ratio:.0%} of your experience bullets include numbers or metrics. "
                    f"Aim for 50%+. Examples:\n"
                    f"- ❌ 'Improved API performance'\n"
                    f"- ✅ 'Improved API response time by 40%, reducing p95 latency from 800ms to 480ms'\n"
                    f"- ❌ 'Built data pipeline'\n"
                    f"- ✅ 'Built data pipeline processing 2M+ events/day with 99.9% uptime'"
                ),
            })

        # Impact statements
        if quality.get("impact_signals", 0) < 3:
            suggestions.append({
                "type": "impact",
                "severity": "medium",
                "title": "Strengthen Impact Statements",
                "description": (
                    "Use the STAR format (Situation-Task-Action-Result) for key bullets. "
                    "Focus on business impact: revenue, cost savings, user growth, "
                    "performance gains, efficiency improvements."
                ),
            })

        # Leadership
        if not quality.get("has_leadership_experience"):
            suggestions.append({
                "type": "leadership",
                "severity": "low",
                "title": "Highlight Any Leadership Experience",
                "description": (
                    "Mention any mentoring, code review ownership, project leadership, "
                    "or cross-team collaboration. Even informal leadership counts."
                ),
            })

        return suggestions

    def _project_suggestions(self, result, resume_data: Dict) -> List[Dict]:
        """Suggestions for improving projects section."""
        suggestions = []

        if result.project_relevance_score < 50:
            suggestions.append({
                "type": "project_relevance",
                "severity": "medium",
                "title": "Add More Relevant Projects",
                "description": (
                    "Your projects don't strongly align with the JD. Consider adding "
                    "1-2 projects that directly showcase the required technologies. "
                    "Even side projects or open-source contributions count."
                ),
            })

        projects = resume_data.get("projects", [])
        for proj in projects:
            if not proj.get("technologies"):
                suggestions.append({
                    "type": "project_tech",
                    "severity": "low",
                    "title": f"Add Tech Stack to '{proj.get('name', 'Project')}'",
                    "description": (
                        "List the specific technologies used in each project. "
                        "This helps ATS systems and reviewers quickly assess relevance."
                    ),
                })
                break  # Only suggest once

        return suggestions

    def _formatting_suggestions(self, resume_data: Dict) -> List[Dict]:
        """Suggestions for resume formatting and structure."""
        suggestions = []

        if not resume_data.get("summary"):
            suggestions.append({
                "type": "add_summary",
                "severity": "medium",
                "title": "Add a Professional Summary",
                "description": (
                    "Add a 2-3 sentence summary at the top tailored to the target role. "
                    "Include years of experience, key specializations, and a notable achievement."
                ),
            })

        if not resume_data.get("certifications"):
            suggestions.append({
                "type": "certifications",
                "severity": "low",
                "title": "Consider Adding Certifications",
                "description": (
                    "Relevant certifications (AWS, GCP, Kubernetes, etc.) add credibility "
                    "and can boost your score. List them in a dedicated section."
                ),
            })

        return suggestions

    def _impact_suggestions(self, resume_data: Dict) -> List[Dict]:
        """Suggestions for improving impact and action verbs."""
        suggestions = []

        weak_verbs = ["worked on", "helped", "assisted", "participated", "was responsible for"]
        strong_verbs = [
            "Architected", "Engineered", "Optimized", "Spearheaded",
            "Automated", "Scaled", "Redesigned", "Implemented",
        ]

        experience = resume_data.get("experience", [])
        has_weak = False
        for entry in experience:
            for bullet in entry.get("bullets", []):
                if any(v in bullet.lower() for v in weak_verbs):
                    has_weak = True
                    break

        if has_weak:
            suggestions.append({
                "type": "action_verbs",
                "severity": "medium",
                "title": "Use Stronger Action Verbs",
                "description": (
                    f"Replace passive verbs ('worked on', 'helped', 'assisted') with "
                    f"strong action verbs: **{', '.join(strong_verbs)}**. "
                    f"Lead each bullet with a powerful verb."
                ),
            })

        return suggestions

    def _prioritize(self, suggestions: Dict) -> List[Dict]:
        """Select top-3 highest-impact actions across all categories."""
        all_items = []
        for category, items in suggestions.items():
            if category == "priority_actions":
                continue
            if isinstance(items, list):
                all_items.extend(items)

        # Sort by severity
        severity_order = {"high": 0, "medium": 1, "low": 2}
        all_items.sort(key=lambda x: severity_order.get(x.get("severity", "low"), 3))

        return all_items[:3]
