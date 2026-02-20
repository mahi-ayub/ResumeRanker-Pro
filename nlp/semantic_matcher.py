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

        # Encode both skill lists
        resume_embeds = self.engine.encode(resume_skills)
        jd_embeds = self.engine.encode(jd_skills)

        # Compute cosine similarity matrix: (n_jd, n_resume)
        sim_matrix = np.dot(jd_embeds, resume_embeds.T)

        matched_skills = []
        missing_skills = []
        matched_resume_indices = set()

        for jd_idx, jd_skill in enumerate(jd_skills):
            best_resume_idx = int(np.argmax(sim_matrix[jd_idx]))
            best_sim = float(sim_matrix[jd_idx, best_resume_idx])

            if best_sim >= 0.50:  # Match threshold
                matched_skills.append({
                    "jd_skill": jd_skill,
                    "resume_skill": resume_skills[best_resume_idx],
                    "similarity": round(best_sim, 3),
                })
                matched_resume_indices.add(best_resume_idx)
            elif best_sim >= 0.35:  # Partial match
                matched_skills.append({
                    "jd_skill": jd_skill,
                    "resume_skill": resume_skills[best_resume_idx],
                    "similarity": round(best_sim, 3),
                    "partial": True,
                })
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
        best_sims = [float(np.max(sim_matrix[i])) for i in range(len(jd_skills))]
        overall_score = float(np.mean(best_sims)) if best_sims else 0.0

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
    ) -> Dict:
        """
        Score each project's relevance to the JD.

        Returns:
            Dict with overall_score and per-project relevance scores.
        """
        if not projects or not jd_text.strip():
            return {"overall_score": 0.0, "project_scores": []}

        jd_embed = self.engine.encode_single(jd_text)

        project_scores = []
        for project in projects:
            # Combine project name + description + technologies
            project_text = f"{project.get('name', '')} {project.get('description', '')} {' '.join(project.get('technologies', []))}"
            if not project_text.strip():
                continue

            proj_embed = self.engine.encode_single(project_text)
            sim = float(np.dot(proj_embed, jd_embed))
            project_scores.append({
                "name": project.get("name", "Unknown"),
                "relevance": round(max(0.0, min(1.0, sim)), 3),
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
    ) -> float:
        """Score education alignment with JD."""
        if not education or not jd_text.strip():
            return 0.0

        edu_texts = []
        for edu in education:
            parts = [edu.get("degree", ""), edu.get("field", ""), edu.get("institution", "")]
            edu_texts.append(" ".join(p for p in parts if p))

        if not edu_texts:
            return 0.0

        combined_edu = " ".join(edu_texts)
        return self.compute_section_similarity(combined_edu, jd_text)
