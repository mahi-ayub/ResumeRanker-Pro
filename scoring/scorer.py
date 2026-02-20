"""Resume Scorer — the main scoring pipeline.

Orchestrates:
1. Resume parsing
2. JD analysis
3. Semantic matching
4. Experience analysis
5. Dynamic weight computation
6. Final score aggregation

Produces an explainable ScoreResult.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass, field
from loguru import logger

from parsers.pdf_parser import PDFParser
from parsers.section_extractor import SectionExtractor, ResumeData
from parsers.entity_extractor import EntityExtractor
from nlp.embeddings import EmbeddingEngine
from nlp.semantic_matcher import SemanticMatcher
from nlp.experience_analyzer import ExperienceAnalyzer
from nlp.jd_analyzer import JDAnalyzer
from scoring.weights import DynamicWeightCalculator
from utils.helpers import clamp


@dataclass
class ScoreResult:
    """Complete scoring result with breakdown and explanations."""
    overall_score: float = 0.0
    skill_match_score: float = 0.0
    experience_score: float = 0.0
    project_relevance_score: float = 0.0
    education_score: float = 0.0

    # Bonuses and penalties
    certification_bonus: float = 0.0
    missing_skill_penalty: float = 0.0

    # Details
    matched_skills: List[Dict] = field(default_factory=list)
    missing_skills: List[Dict] = field(default_factory=list)
    extra_skills: List[str] = field(default_factory=list)
    project_scores: List[Dict] = field(default_factory=list)

    # Analysis
    experience_analysis: Dict = field(default_factory=dict)
    jd_analysis: Dict = field(default_factory=dict)
    resume_data: Dict = field(default_factory=dict)
    weights_used: Dict = field(default_factory=dict)

    # Explanations
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    score_reasoning: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "overall_score": self.overall_score,
            "skill_match_score": self.skill_match_score,
            "experience_score": self.experience_score,
            "project_relevance_score": self.project_relevance_score,
            "education_score": self.education_score,
            "certification_bonus": self.certification_bonus,
            "missing_skill_penalty": self.missing_skill_penalty,
            "matched_skills": self.matched_skills,
            "missing_skills": self.missing_skills,
            "extra_skills": self.extra_skills,
            "project_scores": self.project_scores,
            "experience_analysis": self.experience_analysis,
            "strengths": self.strengths,
            "weaknesses": self.weaknesses,
            "score_reasoning": self.score_reasoning,
            "weights_used": self.weights_used,
        }


class ResumeScorer:
    """Main scoring pipeline: parse, analyze, match, score, explain."""

    def __init__(self):
        self.pdf_parser = PDFParser()
        self.section_extractor = SectionExtractor()
        self.entity_extractor = EntityExtractor()
        self.embedding_engine = EmbeddingEngine()
        self.semantic_matcher = SemanticMatcher(self.embedding_engine)
        self.experience_analyzer = ExperienceAnalyzer()
        self.jd_analyzer = JDAnalyzer()
        self.weight_calculator = DynamicWeightCalculator()

    def score_resume(
        self,
        resume_text: str,
        jd_text: str,
        resume_data: Optional[ResumeData] = None,
    ) -> ScoreResult:
        """
        Score a resume against a job description.

        Args:
            resume_text: Raw resume text (already extracted from PDF).
            jd_text: Job description text.
            resume_data: Pre-parsed ResumeData (if available, skips parsing).

        Returns:
            ScoreResult with full breakdown.
        """
        result = ScoreResult()

        # 1. Parse resume if not already provided
        if resume_data is None:
            resume_data = self.section_extractor.extract(resume_text)

        # Augment with entity extraction
        additional_skills = self.entity_extractor.extract_skills_from_text(resume_text)
        all_skills = list(set(resume_data.skills + additional_skills))
        resume_data.skills = all_skills

        additional_certs = self.entity_extractor.extract_certifications_from_text(resume_text)
        resume_data.certifications = list(set(resume_data.certifications + additional_certs))

        result.resume_data = resume_data.to_dict()

        # 2. Analyze JD
        jd_analysis = self.jd_analyzer.analyze(jd_text)
        result.jd_analysis = jd_analysis

        # 3. Compute dynamic weights
        weights = self.weight_calculator.compute_weights(jd_analysis)
        result.weights_used = weights

        # 4. Semantic skill matching
        skill_result = self.semantic_matcher.compute_skill_similarity(
            resume_skills=resume_data.skills,
            jd_skills=jd_analysis["all_skills"],
        )
        result.skill_match_score = round(skill_result["overall_score"] * 100, 1)
        result.matched_skills = skill_result["matched_skills"]
        result.missing_skills = skill_result["missing_skills"]
        result.extra_skills = skill_result["extra_skills"]

        # 5. Experience analysis
        exp_analysis = self.experience_analyzer.analyze(
            experience=resume_data.experience,
            projects=resume_data.projects,
            skills=resume_data.skills,
            raw_text=resume_text,
        )
        result.experience_analysis = exp_analysis

        exp_match = self.experience_analyzer.compute_experience_match(
            resume_analysis=exp_analysis,
            jd_required_years=jd_analysis.get("years_required"),
            jd_seniority=jd_analysis.get("seniority"),
        )
        result.experience_score = round(exp_match["score"] * 100, 1)

        # 6. Project relevance
        proj_result = self.semantic_matcher.compute_project_relevance(
            projects=resume_data.projects,
            jd_text=jd_text,
        )
        result.project_relevance_score = round(proj_result["overall_score"] * 100, 1)
        result.project_scores = proj_result["project_scores"]

        # 7. Education relevance
        edu_score = self.semantic_matcher.compute_education_relevance(
            education=resume_data.education,
            jd_text=jd_text,
        )
        result.education_score = round(edu_score * 100, 1)

        # 8. Certification bonus
        cert_bonus = min(
            len(resume_data.certifications) * weights["certification_bonus"] * 100,
            15.0  # Cap at 15 points
        )
        result.certification_bonus = round(cert_bonus, 1)

        # 9. Missing skill penalty
        n_missing = len(result.missing_skills)
        missing_penalty = min(
            n_missing * weights["missing_skill_penalty"] * 100,
            weights["max_missing_penalty"] * 100,
        )
        result.missing_skill_penalty = round(missing_penalty, 1)

        # 10. Weighted aggregation
        overall = (
            result.skill_match_score * weights["skill_match"]
            + result.experience_score * weights["experience"]
            + result.project_relevance_score * weights["project_relevance"]
            + result.education_score * weights["education"]
            + result.certification_bonus
            - result.missing_skill_penalty
        )
        result.overall_score = round(clamp(overall, 0, 100), 1)

        # 11. Generate explanations
        self._generate_explanations(result, weights, jd_analysis)

        logger.info(f"Final Score: {result.overall_score}/100")

        return result

    def _generate_explanations(self, result: ScoreResult, weights: Dict, jd_analysis: Dict):
        """Generate human-readable strengths, weaknesses, and reasoning."""
        # Strengths
        if result.skill_match_score >= 70:
            result.strengths.append(f"Strong skill alignment ({result.skill_match_score:.0f}%)")
        if result.experience_score >= 70:
            result.strengths.append(f"Good experience match ({result.experience_score:.0f}%)")
        if result.project_relevance_score >= 60:
            result.strengths.append(f"Relevant project portfolio ({result.project_relevance_score:.0f}%)")
        if result.certification_bonus > 0:
            result.strengths.append(f"Relevant certifications (+{result.certification_bonus:.0f} pts)")
        if len(result.matched_skills) > 5:
            result.strengths.append(f"{len(result.matched_skills)} skills matched to JD")

        exp_quality = result.experience_analysis.get("experience_quality", {})
        if exp_quality.get("quantified_ratio", 0) > 0.3:
            result.strengths.append("Well-quantified achievements")
        if exp_quality.get("has_leadership_experience"):
            result.strengths.append("Leadership experience detected")

        # Weaknesses
        if result.skill_match_score < 50:
            result.weaknesses.append(f"Low skill alignment ({result.skill_match_score:.0f}%)")
        if result.experience_score < 50:
            result.weaknesses.append(f"Experience gap ({result.experience_score:.0f}%)")
        if len(result.missing_skills) > 3:
            missing_names = [m.get("skill", "") for m in result.missing_skills[:5]]
            result.weaknesses.append(f"Missing critical skills: {', '.join(missing_names)}")
        if result.project_relevance_score < 40:
            result.weaknesses.append("Project portfolio not well aligned to JD")
        if exp_quality.get("quantified_ratio", 0) < 0.15:
            result.weaknesses.append("Few quantified achievements in experience bullets")

        # Score reasoning
        result.score_reasoning.append(
            f"Skill Match ({result.skill_match_score:.0f} × {weights['skill_match']:.0%}) = "
            f"{result.skill_match_score * weights['skill_match']:.1f}"
        )
        result.score_reasoning.append(
            f"Experience ({result.experience_score:.0f} × {weights['experience']:.0%}) = "
            f"{result.experience_score * weights['experience']:.1f}"
        )
        result.score_reasoning.append(
            f"Projects ({result.project_relevance_score:.0f} × {weights['project_relevance']:.0%}) = "
            f"{result.project_relevance_score * weights['project_relevance']:.1f}"
        )
        result.score_reasoning.append(
            f"Education ({result.education_score:.0f} × {weights['education']:.0%}) = "
            f"{result.education_score * weights['education']:.1f}"
        )
        if result.certification_bonus > 0:
            result.score_reasoning.append(f"Certification Bonus: +{result.certification_bonus:.1f}")
        if result.missing_skill_penalty > 0:
            result.score_reasoning.append(f"Missing Skills Penalty: -{result.missing_skill_penalty:.1f}")
        result.score_reasoning.append(f"**Final Score: {result.overall_score}/100**")

    def score_resume_from_pdf(
        self,
        file_bytes: bytes,
        jd_text: str,
        filename: str = "",
    ) -> ScoreResult:
        """Convenience method: parse PDF bytes and score."""
        if filename.lower().endswith(".pdf"):
            text = self.pdf_parser.extract_text(file_bytes=file_bytes)
        else:
            text = self.pdf_parser.extract_from_text_file(file_bytes=file_bytes)

        return self.score_resume(resume_text=text, jd_text=jd_text)
