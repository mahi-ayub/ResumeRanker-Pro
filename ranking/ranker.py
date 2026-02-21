"""Resume Ranker — ranks multiple resumes against a single JD.

Produces a sorted leaderboard with per-resume score breakdowns.
"""

from typing import List, Dict
from dataclasses import dataclass, field
from loguru import logger

from scoring.scorer import ResumeScorer, ScoreResult


@dataclass
class RankEntry:
    """A single entry in the ranking leaderboard."""
    rank: int = 0
    filename: str = ""
    overall_score: float = 0.0
    skill_match: float = 0.0
    experience: float = 0.0
    projects: float = 0.0
    education: float = 0.0
    missing_skills_count: int = 0
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    full_result: ScoreResult = field(default_factory=ScoreResult)


class ResumeRanker:
    """Rank multiple resumes against a single job description."""

    def __init__(self, scorer: ResumeScorer = None):
        self.scorer = scorer or ResumeScorer()

    def rank_resumes(
        self,
        resumes: List[Dict],  # [{"filename": str, "bytes": bytes}]
        jd_text: str,
    ) -> List[RankEntry]:
        """
        Score and rank multiple resumes against one JD.

        Args:
            resumes: List of dicts with 'filename' and 'bytes' keys.
            jd_text: Job description text.

        Returns:
            Sorted list of RankEntry (highest score first).
        """
        entries = []

        for resume_info in resumes:
            filename = resume_info.get("filename", "unknown")
            file_bytes = resume_info.get("bytes", b"")

            try:
                result = self.scorer.score_resume_from_pdf(
                    file_bytes=file_bytes,
                    jd_text=jd_text,
                    filename=filename,
                )

                entry = RankEntry(
                    filename=filename,
                    overall_score=result.overall_score,
                    skill_match=result.skill_match_score,
                    experience=result.experience_score,
                    projects=result.project_relevance_score,
                    education=result.education_score,
                    missing_skills_count=len(result.missing_skills),
                    strengths=result.strengths[:3],
                    weaknesses=result.weaknesses[:3],
                    full_result=result,
                )
                entries.append(entry)
                logger.info(f"Scored '{filename}': {result.overall_score}/100")

            except Exception as e:
                logger.error(f"Failed to score '{filename}': {e}")
                entries.append(RankEntry(
                    filename=filename,
                    overall_score=0.0,
                    weaknesses=[f"Error: {str(e)}"],
                ))

        # Sort descending by overall score
        entries.sort(key=lambda x: x.overall_score, reverse=True)

        # Assign ranks
        for i, entry in enumerate(entries, 1):
            entry.rank = i

        return entries

    def get_ranking_summary(self, entries: List[RankEntry]) -> Dict:
        """Generate a summary of the ranking results."""
        if not entries:
            return {"message": "No resumes ranked."}

        return {
            "total_resumes": len(entries),
            "top_candidate": {
                "name": entries[0].filename,
                "score": entries[0].overall_score,
            },
            "score_range": {
                "highest": entries[0].overall_score,
                "lowest": entries[-1].overall_score,
                "average": round(sum(e.overall_score for e in entries) / len(entries), 1),
            },
            "leaderboard": [
                {
                    "rank": e.rank,
                    "filename": e.filename,
                    "score": e.overall_score,
                    "skill_match": e.skill_match,
                    "experience": e.experience,
                    "missing_skills": e.missing_skills_count,
                }
                for e in entries
            ],
        }
