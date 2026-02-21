"""Resume Scorer — ATS-realistic scoring pipeline.

Orchestrates:
1. Resume parsing & PDF text cleanup
2. JD analysis (required vs preferred skills)
3. Semantic matching (document-level + skill-level)
4. Required-skill match ratio (highest weight)
5. Tool/technology exact matching with controlled boost
6. Experience alignment against JD responsibilities
7. ATS-weighted aggregation (skills dominate, not cosine)
8. Structured JSON output with calibrated scoring

Calibration targets:
  Strong match  = 80–90
  Moderate      = 65–79
  Partial       = 50–64
  Weak          < 50
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
from utils.helpers import clamp, clean_text, deduplicate_list
from utils.config import GENERIC_BLOCKLIST


# ── Critical keywords eligible for exact-match boost ────────────────
CRITICAL_KEYWORDS = {
    # Cloud & DevOps
    "aws", "azure", "gcp", "terraform", "docker", "kubernetes",
    "jenkins", "ansible", "ci/cd", "linux", "prometheus", "grafana",
    # Languages & Frameworks
    "python", "java", "javascript", "typescript", "go", "rust", "c++",
    "react", "angular", "vue", "django", "flask", "fastapi", "spring",
    "node.js", "next.js",
    # Data & ML
    "pytorch", "tensorflow", "scikit-learn", "spark", "kafka", "airflow",
    "sql", "postgresql", "mysql", "mongodb", "redis", "elasticsearch",
    # Data analysis tools
    "power bi", "tableau", "pandas", "numpy", "excel", "matplotlib",
    "seaborn", "looker", "dbt", "snowflake", "bigquery", "redshift",
    "sas", "spss", "stata", "alteryx",
    # Domain-specific
    "incident management", "sla", "on-call", "monitoring",
    "microservices", "rest api", "graphql", "agile", "scrum",
}


@dataclass
class ScoreResult:
    """Complete scoring result with breakdown and explanations."""
    overall_score: float = 0.0
    skill_match_score: float = 0.0
    experience_score: float = 0.0
    project_relevance_score: float = 0.0
    education_score: float = 0.0

    # ATS-specific component scores (0-1 normalized)
    semantic_similarity: float = 0.0
    required_skill_match: float = 0.0
    tool_match: float = 0.0
    experience_relevance: float = 0.0

    # Bonuses and penalties
    certification_bonus: float = 0.0
    missing_skill_penalty: float = 0.0
    keyword_boost: float = 0.0

    # Details
    matched_skills: List[Dict] = field(default_factory=list)
    missing_skills: List[Dict] = field(default_factory=list)
    matched_required_skills: List[str] = field(default_factory=list)
    missing_required_skills: List[str] = field(default_factory=list)
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
            "final_score": self.overall_score,
            "semantic_similarity": self.semantic_similarity,
            "required_skill_match": self.required_skill_match,
            "tool_match": self.tool_match,
            "experience_relevance": self.experience_relevance,
            "matched_required_skills": self.matched_required_skills,
            "missing_required_skills": self.missing_required_skills,
            "boost_applied": self.keyword_boost,
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
    """ATS-realistic scoring pipeline: parse → analyze → match → score → explain."""

    # ── ATS weight distribution (skills dominate) ───────────────────
    W_SEMANTIC   = 0.35   # Full-document cosine similarity
    W_REQUIRED   = 0.45   # Required-skill match ratio (highest)
    W_TOOL       = 0.10   # Exact tool/technology keyword match
    W_EXPERIENCE = 0.10   # Experience alignment
    MAX_KEYWORD_BOOST = 8.0   # Max exact-keyword bonus
    SCORE_CAP = 95.0          # Hard ceiling (avoid unrealistic 100)

    def __init__(self):
        self.pdf_parser = PDFParser()
        self.section_extractor = SectionExtractor()
        self.entity_extractor = EntityExtractor()
        self.embedding_engine = EmbeddingEngine()
        self.semantic_matcher = SemanticMatcher(self.embedding_engine)
        self.experience_analyzer = ExperienceAnalyzer()
        self.jd_analyzer = JDAnalyzer()
        self.weight_calculator = DynamicWeightCalculator()

    # ────────────────────────────────────────────────────────────────
    #  Main entry point
    # ────────────────────────────────────────────────────────────────
    def score_resume(
        self,
        resume_text: str,
        jd_text: str,
        resume_data: Optional[ResumeData] = None,
    ) -> ScoreResult:
        result = ScoreResult()

        # ── 0. Clean resume text (PDF robustness) ──────────────────
        resume_text = clean_text(resume_text)

        # ── 1. Parse resume ────────────────────────────────────────
        if resume_data is None:
            resume_data = self.section_extractor.extract(resume_text)

        # Augment with entity extraction from all sections
        additional_skills = self.entity_extractor.extract_skills_from_text(resume_text)
        all_skills = list(set(resume_data.skills + additional_skills))

        exp_text = " ".join(
            " ".join(e.get("bullets", [])) + " " + e.get("description", "")
            for e in resume_data.experience
        )
        proj_text = " ".join(
            p.get("description", "") + " " + " ".join(p.get("technologies", []))
            for p in resume_data.projects
        )
        from_exp = self.entity_extractor.extract_skills_from_text(exp_text)
        from_proj = self.entity_extractor.extract_skills_from_text(proj_text)
        all_skills = deduplicate_list(list(set(all_skills + from_exp + from_proj)))
        resume_data.skills = all_skills

        additional_certs = self.entity_extractor.extract_certifications_from_text(resume_text)
        resume_data.certifications = list(set(resume_data.certifications + additional_certs))
        result.resume_data = resume_data.to_dict()

        # ── 2. Analyze JD ──────────────────────────────────────────
        jd_analysis = self.jd_analyzer.analyze(jd_text)
        result.jd_analysis = jd_analysis

        required_skills = jd_analysis.get("required_skills", [])
        preferred_skills = jd_analysis.get("preferred_skills", [])
        all_jd_skills = jd_analysis.get("all_skills", [])

        # ── 3. Dynamic weights (for legacy display) ────────────────
        weights = self.weight_calculator.compute_weights(jd_analysis)
        result.weights_used = {
            **weights,
            "ats_semantic": self.W_SEMANTIC,
            "ats_required_skill": self.W_REQUIRED,
            "ats_tool": self.W_TOOL,
            "ats_experience": self.W_EXPERIENCE,
        }

        # ── 4. Semantic skill matching (full list) ─────────────────
        skill_result = self.semantic_matcher.compute_skill_similarity(
            resume_skills=resume_data.skills,
            jd_skills=all_jd_skills,
        )
        result.skill_match_score = round(skill_result["overall_score"] * 100, 1)
        result.matched_skills = skill_result["matched_skills"]
        result.missing_skills = skill_result["missing_skills"]
        result.extra_skills = skill_result["extra_skills"]

        # ── 5. Required-skill match ratio ──────────────────────────
        req_match_ratio, matched_req, missing_req = self._compute_required_skill_match(
            resume_skills=resume_data.skills,
            required_skills=required_skills,
            skill_result=skill_result,
        )
        result.required_skill_match = round(req_match_ratio, 3)
        result.matched_required_skills = matched_req
        result.missing_required_skills = missing_req

        # ── 6. Document-level semantic similarity ──────────────────
        sem_sim = self.semantic_matcher.compute_section_similarity(resume_text, jd_text)
        result.semantic_similarity = round(sem_sim, 3)

        # ── 7. Tool / technology exact match ───────────────────────
        tool_score = self._compute_tool_match(
            resume_skills=resume_data.skills,
            resume_text=resume_text,
            jd_skills=all_jd_skills,
        )
        result.tool_match = round(tool_score, 3)

        # ── 8. Experience analysis & alignment ─────────────────────
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
        # Also factor in responsibility alignment via semantic similarity
        resp_text = jd_analysis.get("raw_text", "")
        resp_sim = self.semantic_matcher.compute_section_similarity(exp_text, resp_text)
        # Blend: 60% structured match + 40% semantic responsibility alignment
        blended_exp = exp_match["score"] * 0.6 + min(1.0, resp_sim * 1.5) * 0.4
        result.experience_relevance = round(blended_exp, 3)
        result.experience_score = round(blended_exp * 100, 1)

        # ── 9. Project relevance (for display) ─────────────────────
        proj_result = self.semantic_matcher.compute_project_relevance(
            projects=resume_data.projects,
            jd_text=jd_text,
            jd_skills=all_jd_skills,
        )
        result.project_relevance_score = round(proj_result["overall_score"] * 100, 1)
        result.project_scores = proj_result["project_scores"]

        # ── 10. Education relevance (for display) ──────────────────
        edu_score = self.semantic_matcher.compute_education_relevance(
            education=resume_data.education,
            jd_text=jd_text,
            jd_skills=all_jd_skills,
        )
        result.education_score = round(edu_score * 100, 1)

        # ── 11. Certification bonus ────────────────────────────────
        cert_bonus = min(
            len(resume_data.certifications) * weights["certification_bonus"] * 100,
            8.0,
        )
        result.certification_bonus = round(cert_bonus, 1)

        # ── 12. Exact keyword boost (controlled, max +8) ──────────
        keyword_boost = self._compute_keyword_boost(
            resume_text=resume_text,
            resume_skills=resume_data.skills,
            jd_skills=all_jd_skills,
        )
        result.keyword_boost = round(keyword_boost, 1)

        # ── 13. Missing-skill penalty (only for required) ─────────
        n_missing_req = len(missing_req)
        n_total_req = max(len(required_skills), 1)
        missing_penalty = min(
            (n_missing_req / n_total_req) * 15.0,  # Proportional, max 15 pts
            15.0,
        )
        result.missing_skill_penalty = round(missing_penalty, 1)

        # ── 14. ATS-weighted aggregation ───────────────────────────
        #
        # Dynamic weight adjustment:
        #   If required_skill_match < 0.60, the resume lacks the core
        #   tools the JD asks for.  Reduce semantic weight from 35% → 25%
        #   and redistribute the freed 10% to required_skill (now 55%).
        #   This prevents a domain-adjacent resume from scoring 70+ purely
        #   on semantic similarity.
        #
        w_sem = self.W_SEMANTIC
        w_req = self.W_REQUIRED
        if result.required_skill_match < 0.60:
            w_sem = 0.25          # reduced from 0.35
            w_req = 0.55          # increased from 0.45

        # Store effective weights for display
        result.weights_used["ats_semantic_effective"] = w_sem
        result.weights_used["ats_required_effective"] = w_req

        # All component scores are 0-1, multiplied by 100
        base_score = (
            w_sem             * result.semantic_similarity
            + w_req           * result.required_skill_match
            + self.W_TOOL     * result.tool_match
            + self.W_EXPERIENCE * result.experience_relevance
        ) * 100.0

        # Required-skill coverage boost: if ≥70% required skills present, boost
        if req_match_ratio >= 0.70:
            coverage_boost = (req_match_ratio - 0.70) / 0.30 * 5.0  # up to +5
            base_score += coverage_boost

        overall = base_score + keyword_boost + cert_bonus - missing_penalty
        result.overall_score = round(clamp(overall, 0, self.SCORE_CAP), 1)

        # ── 15. Generate explanations ──────────────────────────────
        self._generate_explanations(result, weights, jd_analysis)

        logger.info(
            f"Final Score: {result.overall_score}/100 "
            f"[sem={result.semantic_similarity:.2f} req={result.required_skill_match:.2f} "
            f"tool={result.tool_match:.2f} exp={result.experience_relevance:.2f} "
            f"boost={result.keyword_boost:.1f}]"
        )

        return result

    # ────────────────────────────────────────────────────────────────
    #  Required-skill match ratio  (STRICT — tools only, no generics)
    # ────────────────────────────────────────────────────────────────
    def _compute_required_skill_match(
        self,
        resume_skills: List[str],
        required_skills: List[str],
        skill_result: Dict,
    ) -> tuple:
        """
        Compute fraction of required JD skills present in resume.

        RULES:
        - Generic / vague words (data, system, cloud, development …) are
          stripped from the required list BEFORE matching so they cannot
          inflate the ratio.
        - Only exact case-insensitive keyword matches count.  Semantic
          fallback is allowed only at very high similarity (≥ 0.80) to
          prevent domain-adjacent terms from leaking in.
        - If > 40 % of the *technical* required tools are still missing
          after matching, the ratio is hard-capped at 0.55.

        Returns (ratio, matched_list, missing_list).
        """
        if not required_skills:
            # No explicit required section — use all matched skills
            n_matched = len(skill_result.get("matched_skills", []))
            n_total = n_matched + len(skill_result.get("missing_skills", []))
            ratio = n_matched / max(n_total, 1)
            matched = [m.get("jd_skill", "") for m in skill_result.get("matched_skills", [])]
            missing = [m.get("skill", "") for m in skill_result.get("missing_skills", [])]
            return ratio, matched, missing

        # ── 1. Filter out generic words from required skills ───────
        technical_required = [
            s for s in required_skills
            if s.lower().strip() not in GENERIC_BLOCKLIST
        ]
        # Keep the generics for reporting but never let them boost ratio
        generic_dropped = [
            s for s in required_skills
            if s.lower().strip() in GENERIC_BLOCKLIST
        ]
        if generic_dropped:
            logger.debug(
                f"Filtered {len(generic_dropped)} generic terms from required skills: "
                f"{generic_dropped}"
            )

        # If every "required skill" was generic, fall back to the full
        # semantic skill_result so we still return something meaningful.
        if not technical_required:
            n_matched = len(skill_result.get("matched_skills", []))
            n_total = n_matched + len(skill_result.get("missing_skills", []))
            ratio = n_matched / max(n_total, 1)
            matched = [m.get("jd_skill", "") for m in skill_result.get("matched_skills", [])]
            missing = [m.get("skill", "") for m in skill_result.get("missing_skills", [])]
            return ratio, matched, missing

        # ── 2. Strict matching against technical required skills ───
        resume_lower = {s.lower().strip() for s in resume_skills}
        matched: List[str] = []
        missing: List[str] = []

        for req in technical_required:
            req_clean = req.lower().strip()

            # a) Exact case-insensitive match
            if req_clean in resume_lower:
                matched.append(req)
                continue

            # b) Substring: resume skill contains the required keyword
            if any(req_clean in rs for rs in resume_lower):
                matched.append(req)
                continue

            # c) Very-high-confidence semantic fallback only (≥ 0.80)
            #    This prevents "Power BI" from matching "data analysis".
            found_semantic = False
            for m in skill_result.get("matched_skills", []):
                if (m.get("jd_skill", "").lower().strip() == req_clean
                        and m.get("similarity", 0) >= 0.80):
                    matched.append(req)
                    found_semantic = True
                    break

            if not found_semantic:
                missing.append(req)

        n_tech = len(technical_required)
        n_missing = len(missing)
        ratio = len(matched) / max(n_tech, 1)

        # ── 3. Hard cap: if >40% technical tools are missing ───────
        if n_tech > 0 and (n_missing / n_tech) > 0.40:
            ratio = min(ratio, 0.55)

        return ratio, matched, missing

    # ────────────────────────────────────────────────────────────────
    #  Tool / technology exact match  (strict — no generics)
    # ────────────────────────────────────────────────────────────────
    def _compute_tool_match(
        self,
        resume_skills: List[str],
        resume_text: str,
        jd_skills: List[str],
    ) -> float:
        """
        Compute what fraction of JD tools/technologies appear exactly
        in the resume (case-insensitive).  Generic terms are excluded.
        Normalized 0-1.
        """
        if not jd_skills:
            return 0.0

        # Only score concrete tools, not generic words
        tech_jd = [
            s for s in jd_skills
            if s.lower().strip() not in GENERIC_BLOCKLIST
        ]
        if not tech_jd:
            return 0.0

        resume_lower = resume_text.lower()
        resume_skill_set = {s.lower().strip() for s in resume_skills}
        matched = 0

        for skill in tech_jd:
            s = skill.lower().strip()
            if s in resume_skill_set or s in resume_lower:
                matched += 1

        return matched / len(tech_jd)

    # ────────────────────────────────────────────────────────────────
    #  Exact keyword boost (controlled, max +8)
    # ────────────────────────────────────────────────────────────────
    def _compute_keyword_boost(
        self,
        resume_text: str,
        resume_skills: List[str],
        jd_skills: List[str],
    ) -> float:
        """
        Award a controlled bonus when critical JD keywords appear
        exactly in the resume. Only counts keywords that are BOTH
        in the JD and in the critical-keyword set.
        Max +8 points.
        """
        resume_lower = resume_text.lower()
        resume_skill_lower = {s.lower().strip() for s in resume_skills}
        jd_skill_lower = {s.lower().strip() for s in jd_skills}

        # Only consider keywords that the JD actually asks for
        eligible = CRITICAL_KEYWORDS & jd_skill_lower
        if not eligible:
            return 0.0

        hits = 0
        for kw in eligible:
            if kw in resume_skill_lower or kw in resume_lower:
                hits += 1

        # Scale: each hit is worth 1.5 pts, capped at MAX_KEYWORD_BOOST
        boost = min(hits * 1.5, self.MAX_KEYWORD_BOOST)
        return boost

    # ────────────────────────────────────────────────────────────────
    #  Explanations
    # ────────────────────────────────────────────────────────────────
    def _generate_explanations(self, result: ScoreResult, weights: Dict, jd_analysis: Dict):
        """Generate human-readable strengths, weaknesses, and reasoning."""
        # Strengths
        if result.required_skill_match >= 0.70:
            result.strengths.append(
                f"Strong required-skill coverage ({result.required_skill_match:.0%})"
            )
        if result.skill_match_score >= 70:
            result.strengths.append(f"Good overall skill alignment ({result.skill_match_score:.0f}%)")
        if result.experience_relevance >= 0.65:
            result.strengths.append(f"Relevant experience ({result.experience_score:.0f}%)")
        if result.project_relevance_score >= 60:
            result.strengths.append(f"Relevant project portfolio ({result.project_relevance_score:.0f}%)")
        if result.certification_bonus > 0:
            result.strengths.append(f"Relevant certifications (+{result.certification_bonus:.0f} pts)")
        if result.keyword_boost > 0:
            result.strengths.append(f"Critical keyword matches (+{result.keyword_boost:.0f} pts)")
        if len(result.matched_required_skills) > 3:
            result.strengths.append(
                f"{len(result.matched_required_skills)} required skills matched"
            )

        exp_quality = result.experience_analysis.get("experience_quality", {})
        if exp_quality.get("quantified_ratio", 0) > 0.3:
            result.strengths.append("Well-quantified achievements")
        if exp_quality.get("has_leadership_experience"):
            result.strengths.append("Leadership experience detected")

        # Weaknesses
        if result.required_skill_match < 0.50:
            result.weaknesses.append(
                f"Low required-skill coverage ({result.required_skill_match:.0%})"
            )
        if result.skill_match_score < 50:
            result.weaknesses.append(f"Low overall skill alignment ({result.skill_match_score:.0f}%)")
        if result.experience_score < 50:
            result.weaknesses.append(f"Experience gap ({result.experience_score:.0f}%)")
        if result.missing_required_skills:
            names = result.missing_required_skills[:5]
            result.weaknesses.append(f"Missing required skills: {', '.join(names)}")
        if result.project_relevance_score < 40:
            result.weaknesses.append("Project portfolio not well aligned to JD")
        if exp_quality.get("quantified_ratio", 0) < 0.15:
            result.weaknesses.append("Few quantified achievements in experience bullets")

        # Score reasoning (ATS breakdown) — use effective weights
        w_sem_eff = result.weights_used.get("ats_semantic_effective", self.W_SEMANTIC)
        w_req_eff = result.weights_used.get("ats_required_effective", self.W_REQUIRED)

        if w_sem_eff != self.W_SEMANTIC:
            result.score_reasoning.append(
                "⚠️ Semantic weight reduced (25%) — required-skill coverage < 60%"
            )
        result.score_reasoning.append(
            f"Semantic Similarity ({result.semantic_similarity:.2f} × {w_sem_eff:.0%}) = "
            f"{result.semantic_similarity * w_sem_eff * 100:.1f}"
        )
        result.score_reasoning.append(
            f"Required Skill Match ({result.required_skill_match:.2f} × {w_req_eff:.0%}) = "
            f"{result.required_skill_match * w_req_eff * 100:.1f}"
        )
        result.score_reasoning.append(
            f"Tool Match ({result.tool_match:.2f} × {self.W_TOOL:.0%}) = "
            f"{result.tool_match * self.W_TOOL * 100:.1f}"
        )
        result.score_reasoning.append(
            f"Experience Relevance ({result.experience_relevance:.2f} × {self.W_EXPERIENCE:.0%}) = "
            f"{result.experience_relevance * self.W_EXPERIENCE * 100:.1f}"
        )
        if result.keyword_boost > 0:
            result.score_reasoning.append(f"Keyword Boost: +{result.keyword_boost:.1f}")
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
