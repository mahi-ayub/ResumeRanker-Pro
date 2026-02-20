"""Experience Analyzer — estimates years of experience, seniority, and skill depth.

Uses NLP + rule-based heuristics to analyze:
- Total years of professional experience
- Per-skill experience depth
- Seniority signals (Junior, Mid, Senior, Lead, Principal)
- Professional vs. project-only experience
"""

import re
from typing import List, Dict, Optional, Tuple
from loguru import logger

from utils.config import SENIORITY_LEVELS


class ExperienceAnalyzer:
    """Analyze experience depth and seniority from parsed resume data."""

    def analyze(
        self,
        experience: List[Dict],
        projects: List[Dict],
        skills: List[str],
        raw_text: str = "",
    ) -> Dict:
        """
        Analyze experience from parsed resume data.

        Returns:
            Dict with:
                - total_years: Estimated total professional years
                - seniority_level: Detected seniority (junior/mid/senior/etc.)
                - seniority_signals: Evidence list
                - skill_experience: Per-skill depth estimates
                - experience_quality: Professional vs. project breakdown
        """
        total_years = self._estimate_total_years(experience)
        seniority, signals = self._detect_seniority(experience, raw_text, total_years)
        skill_depth = self._estimate_skill_depth(experience, projects, skills)
        quality = self._assess_experience_quality(experience, projects)

        return {
            "total_years": total_years,
            "seniority_level": seniority,
            "seniority_signals": signals,
            "skill_experience": skill_depth,
            "experience_quality": quality,
        }

    def _estimate_total_years(self, experience: List[Dict]) -> float:
        """Estimate total years from experience entries."""
        total = 0.0
        for entry in experience:
            duration = entry.get("duration_years")
            if duration and isinstance(duration, (int, float)):
                total += duration

        # Fallback: count number of roles × estimated average
        if total == 0 and experience:
            total = len(experience) * 2.0  # Rough heuristic

        return round(total, 1)

    def _detect_seniority(
        self,
        experience: List[Dict],
        raw_text: str,
        total_years: float,
    ) -> Tuple[str, List[str]]:
        """Detect seniority level from experience and text signals."""
        signals = []

        # Signal 1: Years of experience
        years_level = "junior"
        for level, bounds in SENIORITY_LEVELS.items():
            if bounds["min_years"] <= total_years <= bounds["max_years"]:
                years_level = level

        signals.append(f"Estimated {total_years} years → {years_level} by experience")

        # Signal 2: Title keywords in experience entries
        title_keywords = {
            "intern": ["intern", "trainee", "apprentice"],
            "junior": ["junior", "jr", "associate", "entry"],
            "mid": ["mid", "intermediate"],
            "senior": ["senior", "sr", "lead", "principal", "staff", "architect"],
            "lead": ["lead", "team lead", "tech lead", "engineering lead"],
            "manager": ["manager", "engineering manager", "director", "head of", "vp"],
        }

        detected_titles = set()
        for entry in experience:
            title = entry.get("title", "").lower()
            for level, keywords in title_keywords.items():
                if any(kw in title for kw in keywords):
                    detected_titles.add(level)
                    signals.append(f"Title '{entry.get('title', '')}' signals '{level}'")

        # Signal 3: Leadership / architecture keywords in raw text
        leadership_patterns = [
            (r'\b(?:led|managed|mentored|supervised)\s+(?:a\s+)?team', "leadership"),
            (r'\b(?:architected|designed\s+system|system\s+design)', "architecture"),
            (r'\b(?:promoted|advanced)\s+to', "promotion"),
            (r'\b(?:founding|co-founder|startup)', "founder"),
        ]

        for pattern, label in leadership_patterns:
            if re.search(pattern, raw_text, re.IGNORECASE):
                signals.append(f"Text contains '{label}' signals")

        # Determine final seniority: highest detected level
        level_order = ["intern", "junior", "mid", "senior", "staff", "principal", "lead", "manager"]
        final_level = years_level
        for level in reversed(level_order):
            if level in detected_titles:
                final_level = level
                break

        return final_level, signals

    def _estimate_skill_depth(
        self,
        experience: List[Dict],
        projects: List[Dict],
        skills: List[str],
    ) -> List[Dict]:
        """Estimate per-skill depth (professional work vs. project only)."""
        skill_info = {}

        for skill in skills:
            skill_lower = skill.lower()
            info = {
                "skill": skill,
                "professional_mentions": 0,
                "project_mentions": 0,
                "estimated_years": 0.0,
                "context": "unknown",  # professional / project / mentioned_only
            }

            # Check experience entries
            for entry in experience:
                desc = entry.get("description", "").lower()
                bullets = " ".join(entry.get("bullets", [])).lower()
                all_text = f"{desc} {bullets}"

                if skill_lower in all_text:
                    info["professional_mentions"] += 1
                    duration = entry.get("duration_years", 0)
                    if duration:
                        info["estimated_years"] += duration

            # Check projects
            for project in projects:
                proj_text = f"{project.get('description', '')} {' '.join(project.get('technologies', []))}".lower()
                if skill_lower in proj_text:
                    info["project_mentions"] += 1

            # Determine context
            if info["professional_mentions"] > 0:
                info["context"] = "professional"
            elif info["project_mentions"] > 0:
                info["context"] = "project_only"
            else:
                info["context"] = "mentioned_only"

            info["estimated_years"] = round(info["estimated_years"], 1)
            skill_info[skill] = info

        return list(skill_info.values())

    def _assess_experience_quality(
        self,
        experience: List[Dict],
        projects: List[Dict],
    ) -> Dict:
        """Assess overall experience quality and depth."""
        # Quantified achievements
        quantified_count = 0
        total_bullets = 0

        for entry in experience:
            for bullet in entry.get("bullets", []):
                total_bullets += 1
                if re.search(r'\d+[%$]|\$\d|increased|decreased|improved|reduced|saved|generated', bullet, re.IGNORECASE):
                    quantified_count += 1

        # Impact signals
        impact_keywords = [
            "scale", "performance", "optimization", "revenue", "growth",
            "efficiency", "automation", "cost", "users", "traffic",
        ]
        impact_count = 0
        all_text = " ".join(
            " ".join(e.get("bullets", [])) for e in experience
        ).lower()
        for keyword in impact_keywords:
            if keyword in all_text:
                impact_count += 1

        return {
            "total_roles": len(experience),
            "total_projects": len(projects),
            "total_bullets": total_bullets,
            "quantified_achievements": quantified_count,
            "quantified_ratio": round(quantified_count / max(total_bullets, 1), 2),
            "impact_signals": impact_count,
            "has_leadership_experience": bool(
                re.search(r'\b(?:led|managed|mentored|supervised)\b', all_text)
            ),
        }

    def compute_experience_match(
        self,
        resume_analysis: Dict,
        jd_required_years: Optional[float] = None,
        jd_seniority: Optional[str] = None,
    ) -> Dict:
        """
        Compute how well resume experience matches JD requirements.

        Returns:
            Dict with score [0,1] and explanation.
        """
        score = 0.0
        reasons = []

        # Years match
        if jd_required_years is not None:
            actual = resume_analysis.get("total_years", 0)
            if actual >= jd_required_years:
                years_score = 1.0
                reasons.append(f"Experience ({actual}y) meets requirement ({jd_required_years}y)")
            elif actual >= jd_required_years * 0.7:
                years_score = 0.7
                reasons.append(f"Experience ({actual}y) is close to requirement ({jd_required_years}y)")
            else:
                years_score = max(0.2, actual / jd_required_years)
                reasons.append(f"Experience ({actual}y) below requirement ({jd_required_years}y)")
            score += years_score * 0.5
        else:
            score += 0.35  # Neutral if no requirement specified
            reasons.append("No specific years requirement in JD")

        # Seniority match
        if jd_seniority:
            actual_seniority = resume_analysis.get("seniority_level", "mid")
            level_order = ["intern", "junior", "mid", "senior", "staff", "principal", "lead", "manager"]

            actual_idx = level_order.index(actual_seniority) if actual_seniority in level_order else 2
            required_idx = level_order.index(jd_seniority) if jd_seniority in level_order else 2

            if actual_idx >= required_idx:
                seniority_score = 1.0
                reasons.append(f"Seniority ({actual_seniority}) meets JD ({jd_seniority})")
            elif actual_idx >= required_idx - 1:
                seniority_score = 0.7
                reasons.append(f"Seniority ({actual_seniority}) is close to JD ({jd_seniority})")
            else:
                seniority_score = 0.3
                reasons.append(f"Seniority ({actual_seniority}) below JD ({jd_seniority})")
            score += seniority_score * 0.3
        else:
            score += 0.2
            reasons.append("No specific seniority requirement in JD")

        # Quality bonus
        quality = resume_analysis.get("experience_quality", {})
        if quality.get("quantified_ratio", 0) > 0.3:
            score += 0.1
            reasons.append("Strong quantified achievements")
        if quality.get("impact_signals", 0) >= 3:
            score += 0.1
            reasons.append("Multiple impact signals detected")

        return {
            "score": round(min(1.0, score), 3),
            "reasons": reasons,
        }
