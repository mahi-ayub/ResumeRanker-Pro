"""Semantic Matcher — computes fine-grained semantic similarity between resume and JD.

Uses cosine similarity on transformer embeddings at skill, section, and document levels.
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
from loguru import logger

from .embeddings import EmbeddingEngine


class SemanticMatcher:
    """Compute semantic similarity between resume sections and JD requirements."""

    def __init__(self, embedding_engine: Optional[EmbeddingEngine] = None):
        self.engine = embedding_engine or EmbeddingEngine()

    def compute_skill_similarity(
        self,
        resume_skills: List[str],
        jd_skills: List[str],
    ) -> Dict:
        """
        Compute pairwise semantic similarity between resume skills and JD skills.

        Returns:
            Dict with:
                - overall_score (float): Mean best-match similarity [0,1]
                - matched_skills: List of (jd_skill, best_resume_skill, similarity)
                - missing_skills: JD skills with no good resume match
                - extra_skills: Resume skills not matching any JD skill
        """
        if not resume_skills or not jd_skills:
            return {
                "overall_score": 0.0,
                "matched_skills": [],
                "missing_skills": jd_skills.copy(),
                "extra_skills": resume_skills.copy(),
                "similarity_matrix": None,
            }

        # Normalize skills for case-insensitive exact matching
        resume_lower = [s.lower().strip() for s in resume_skills]
        jd_lower = [s.lower().strip() for s in jd_skills]

        # Encode both skill lists
        resume_embeds = self.engine.encode(resume_skills)
        jd_embeds = self.engine.encode(jd_skills)

        # Compute cosine similarity matrix: (n_jd, n_resume)
        sim_matrix = np.dot(jd_embeds, resume_embeds.T)

        matched_skills = []
        missing_skills = []
        matched_resume_indices = set()

        for jd_idx, jd_skill in enumerate(jd_skills):
            # First: check for exact case-insensitive match
            exact_match_idx = None
            for r_idx, r_lower in enumerate(resume_lower):
                if jd_lower[jd_idx] == r_lower:
                    exact_match_idx = r_idx
                    break

            if exact_match_idx is not None:
                matched_skills.append({
                    "jd_skill": jd_skill,
                    "resume_skill": resume_skills[exact_match_idx],
                    "similarity": 1.0,
                })
                matched_resume_indices.add(exact_match_idx)
                continue

            best_resume_idx = int(np.argmax(sim_matrix[jd_idx]))
            best_sim = float(sim_matrix[jd_idx, best_resume_idx])

            if best_sim >= 0.45:  # Match threshold (lowered from 0.50 for better recall)
                matched_skills.append({
                    "jd_skill": jd_skill,
                    "resume_skill": resume_skills[best_resume_idx],
                    "similarity": round(best_sim, 3),
                })
                matched_resume_indices.add(best_resume_idx)
            elif best_sim >= 0.30:  # Partial match (lowered from 0.35)
                matched_skills.append({
                    "jd_skill": jd_skill,
                    "resume_skill": resume_skills[best_resume_idx],
                    "similarity": round(best_sim, 3),
                    "partial": True,
                })
                matched_resume_indices.add(best_resume_idx)
            else:
                missing_skills.append({
                    "skill": jd_skill,
                    "best_match": resume_skills[best_resume_idx],
                    "best_similarity": round(best_sim, 3),
                })

        # Extra skills on resume not matched to any JD skill
        extra_skills = [
            resume_skills[i]
            for i in range(len(resume_skills))
            if i not in matched_resume_indices
        ]

        # Overall score: mean of best similarities for each JD skill
        # Use boosted scoring: exact matches get 1.0, semantic matches get their score
        boosted_sims = []
        for jd_idx in range(len(jd_skills)):
            # Check if this was an exact match
            if jd_lower[jd_idx] in resume_lower:
                boosted_sims.append(1.0)
            else:
                boosted_sims.append(float(np.max(sim_matrix[jd_idx])))

        overall_score = float(np.mean(boosted_sims)) if boosted_sims else 0.0

        return {
            "overall_score": round(overall_score, 3),
            "matched_skills": matched_skills,
            "missing_skills": missing_skills,
            "extra_skills": extra_skills,
            "similarity_matrix": sim_matrix,
        }

    def compute_section_similarity(
        self,
        resume_text: str,
        jd_text: str,
    ) -> float:
        """
        Compute overall semantic similarity between a resume section and JD text.

        Uses document-level embeddings for a holistic similarity score.
        """
        if not resume_text.strip() or not jd_text.strip():
            return 0.0

        resume_embed = self.engine.encode_single(resume_text)
        jd_embed = self.engine.encode_single(jd_text)

        similarity = float(np.dot(resume_embed, jd_embed))
        return round(max(0.0, min(1.0, similarity)), 3)

    def compute_project_relevance(
        self,
        projects: List[Dict],
        jd_text: str,
        jd_skills: List[str] = None,
    ) -> Dict:
        """
        Score each project's relevance to the JD.

        Uses JD skills for targeted matching rather than the full JD text,
        which would dilute similarity with company boilerplate.

        Returns:
            Dict with overall_score and per-project relevance scores.
        """
        if not projects:
            return {"overall_score": 0.0, "project_scores": []}

        # Build a focused JD summary from skills + key requirements
        if jd_skills:
            jd_summary = "Required skills and technologies: " + ", ".join(jd_skills)
        else:
            jd_summary = jd_text

        if not jd_summary.strip():
            return {"overall_score": 0.0, "project_scores": []}

        jd_embed = self.engine.encode_single(jd_summary)

        project_scores = []
        for project in projects:
            # Combine project name + description + technologies
            project_text = f"{project.get('name', '')} {project.get('description', '')} {' '.join(project.get('technologies', []))}"
            if not project_text.strip():
                continue

            proj_embed = self.engine.encode_single(project_text)
            sim = float(np.dot(proj_embed, jd_embed))

            # Boost score: projects inherently won't have 1.0 similarity to a JD
            # Apply sigmoid-like scaling so 0.3 sim → ~0.6, 0.5 sim → ~0.85
            boosted = min(1.0, sim * 1.8 + 0.1) if sim > 0.1 else sim

            project_scores.append({
                "name": project.get("name", "Unknown"),
                "relevance": round(max(0.0, min(1.0, boosted)), 3),
            })

        # Sort by relevance descending
        project_scores.sort(key=lambda x: x["relevance"], reverse=True)

        overall = float(np.mean([p["relevance"] for p in project_scores])) if project_scores else 0.0

        return {
            "overall_score": round(overall, 3),
            "project_scores": project_scores,
        }

    def compute_education_relevance(
        self,
        education: List[Dict],
        jd_text: str,
        jd_skills: List[str] = None,
    ) -> float:
        """Score education alignment with JD.
        
        Uses focused comparison against JD skills/requirements rather than full JD text.
        Also gives a baseline score for having a relevant degree (CS, Engineering, etc.)
        since most tech JDs assume a technical degree.
        """
        if not education:
            return 0.0

        edu_texts = []
        for edu in education:
            parts = [edu.get("degree", ""), edu.get("field", ""), edu.get("institution", "")]
            edu_texts.append(" ".join(p for p in parts if p))

        if not edu_texts:
            return 0.0

        combined_edu = " ".join(edu_texts)

        # Check for relevant STEM degree keywords — baseline score
        stem_keywords = [
            "computer science", "computer engineering", "software", "information technology",
            "electrical engineering", "mathematics", "statistics", "data science",
            "machine learning", "artificial intelligence", "physics", "engineering",
            "b.s.", "m.s.", "b.tech", "m.tech", "bachelor", "master", "ph.d", "mba",
        ]
        edu_lower = combined_edu.lower()
        has_stem = any(kw in edu_lower for kw in stem_keywords)
        baseline = 0.55 if has_stem else 0.2

        # Semantic similarity against focused JD content
        if jd_skills:
            jd_summary = "Required background: " + ", ".join(jd_skills)
        else:
            jd_summary = jd_text

        if not jd_summary.strip():
            return baseline

        semantic_score = self.compute_section_similarity(combined_edu, jd_summary)

        # Blend: max of baseline and boosted semantic score
        # Education inherently has low cosine similarity to skill lists
        boosted_semantic = min(1.0, semantic_score * 2.0 + 0.15) if semantic_score > 0.1 else semantic_score
        final = max(baseline, boosted_semantic)

        return round(final, 3)
