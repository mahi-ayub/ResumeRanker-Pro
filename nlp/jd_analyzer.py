"""JD Analyzer — parses job descriptions to extract requirements and classify role type.

Extracts:
- Required skills
- Preferred/nice-to-have skills
- Years of experience required
- Seniority level
- Role type (backend, ML, frontend, etc.)
- Mandatory vs. optional requirements
"""

import re
from typing import List, Dict, Optional, Tuple
from loguru import logger

from utils.config import SKILL_CATEGORIES, ROLE_WEIGHT_PROFILES


# Role type classification keywords
ROLE_CLASSIFIERS = {
    "backend_engineer": [
        "backend", "back-end", "server-side", "api", "rest api", "microservices",
        "database", "sql", "distributed systems",
    ],
    "frontend_engineer": [
        "frontend", "front-end", "ui", "ux", "react", "angular", "vue",
        "javascript", "css", "responsive", "web application",
    ],
    "fullstack_engineer": [
        "full-stack", "fullstack", "full stack",
    ],
    "ml_engineer": [
        "machine learning", "ml engineer", "deep learning", "neural network",
        "model training", "model deployment", "mlops", "feature engineering",
        "pytorch", "tensorflow",
    ],
    "data_scientist": [
        "data scientist", "data science", "statistical", "analytics",
        "experimentation", "a/b testing", "hypothesis", "modeling",
    ],
    "data_engineer": [
        "data engineer", "data pipeline", "etl", "data warehouse",
        "spark", "airflow", "kafka", "data infrastructure",
    ],
    "devops_engineer": [
        "devops", "site reliability", "sre", "infrastructure",
        "kubernetes", "docker", "terraform", "ci/cd", "cloud infrastructure",
    ],
}


class JDAnalyzer:
    """Analyze job descriptions to extract structured requirements."""

    def analyze(self, jd_text: str) -> Dict:
        """
        Full analysis of a job description.

        Returns:
            Dict with role_type, required_skills, preferred_skills,
            experience_requirements, seniority_level, and full text.
        """
        role_type, role_confidence = self._classify_role(jd_text)
        required_skills, preferred_skills = self._extract_skills(jd_text)
        years_required = self._extract_years_requirement(jd_text)
        seniority = self._extract_seniority(jd_text, years_required)
        mandatory_keywords = self._extract_mandatory_requirements(jd_text)

        result = {
            "role_type": role_type,
            "role_confidence": role_confidence,
            "required_skills": required_skills,
            "preferred_skills": preferred_skills,
            "all_skills": list(set(required_skills + preferred_skills)),
            "years_required": years_required,
            "seniority": seniority,
            "mandatory_keywords": mandatory_keywords,
            "raw_text": jd_text,
        }

        logger.info(
            f"JD Analysis — Role: {role_type} ({role_confidence:.0%}), "
            f"Required skills: {len(required_skills)}, "
            f"Years: {years_required}, Seniority: {seniority}"
        )

        return result

    def _classify_role(self, jd_text: str) -> Tuple[str, float]:
        """Classify the JD into a role type."""
        jd_lower = jd_text.lower()
        scores = {}

        for role, keywords in ROLE_CLASSIFIERS.items():
            count = sum(1 for kw in keywords if kw in jd_lower)
            scores[role] = count / len(keywords)

        if not scores or max(scores.values()) == 0:
            return "default", 0.0

        best_role = max(scores, key=scores.get)
        return best_role, scores[best_role]

    def _extract_skills(self, jd_text: str) -> Tuple[List[str], List[str]]:
        """Extract required and preferred skills from JD."""
        required = []
        preferred = []

        # Split into required/preferred sections
        jd_lower = jd_text.lower()
        sections = self._split_jd_sections(jd_text)

        required_text = sections.get("requirements", "") + " " + sections.get("required", "")
        preferred_text = sections.get("preferred", "") + " " + sections.get("nice_to_have", "")

        # If no clear sections, use full text as requirements
        if not required_text.strip():
            required_text = jd_text

        # Extract from known skill categories
        all_known_skills = set()
        for category, skills in SKILL_CATEGORIES.items():
            all_known_skills.update(skills)

        # Filter out overly short/ambiguous skill names for substring matching
        # Skills like "r", "c", "go" need word-boundary matching
        short_skills = {s for s in all_known_skills if len(s) <= 2}
        long_skills = all_known_skills - short_skills

        for skill in long_skills:
            if skill in required_text.lower():
                required.append(skill)
            elif skill in preferred_text.lower():
                preferred.append(skill)
            elif skill in jd_lower:
                required.append(skill)  # Default to required

        # For short skills (r, go, c#, etc.), use word-boundary regex to avoid false positives
        import re as _re
        for skill in short_skills:
            # Special handling: "r" alone is very ambiguous — only match "R programming" or "R language" or standalone "R," patterns
            if skill == "r":
                r_pattern = r'\bR\b(?:\s+(?:programming|language|studio)|\s*[,;|/])'
                if _re.search(r_pattern, jd_text):
                    required.append(skill)
                continue

            pattern = r'\b' + _re.escape(skill) + r'\b'
            if _re.search(pattern, required_text, _re.IGNORECASE):
                required.append(skill)
            elif _re.search(pattern, preferred_text, _re.IGNORECASE):
                preferred.append(skill)
            elif _re.search(pattern, jd_text, _re.IGNORECASE):
                required.append(skill)

        # Also extract technology names via patterns
        tech_pattern = re.compile(
            r'\b(?:Python|Java|JavaScript|TypeScript|C\+\+|C#|Go|Rust|Ruby|PHP|Swift|Kotlin|Scala|R|'
            r'React|Angular|Vue|Django|Flask|FastAPI|Spring|Express|Next\.js|'
            r'PyTorch|TensorFlow|Keras|scikit-learn|XGBoost|'
            r'PostgreSQL|MySQL|MongoDB|Redis|Elasticsearch|'
            r'AWS|Azure|GCP|Docker|Kubernetes|Terraform|'
            r'Git|Linux|Node\.js|GraphQL|REST|Spark|Hadoop|Airflow|Kafka)\b',
            re.IGNORECASE,
        )

        for match in tech_pattern.finditer(jd_text):
            skill = match.group().lower()
            if skill not in [s.lower() for s in required] and skill not in [s.lower() for s in preferred]:
                required.append(match.group())

        return list(set(required)), list(set(preferred))

    def _split_jd_sections(self, jd_text: str) -> Dict[str, str]:
        """Split JD into named sections."""
        sections = {}
        patterns = {
            "requirements": r'(?:requirements?|qualifications?|what\s+you.?ll?\s+need|must\s+have)',
            "preferred": r'(?:preferred|nice\s+to\s+have|bonus|good\s+to\s+have|plus)',
            "responsibilities": r'(?:responsibilities|what\s+you.?ll?\s+do|about\s+the\s+role)',
        }

        matches = []
        for section_name, pattern in patterns.items():
            for m in re.finditer(pattern, jd_text, re.IGNORECASE):
                matches.append((m.start(), m.end(), section_name))

        matches.sort(key=lambda x: x[0])

        for i, (start, end, name) in enumerate(matches):
            next_start = matches[i + 1][0] if i + 1 < len(matches) else len(jd_text)
            sections[name] = jd_text[end:next_start]

        return sections

    def _extract_years_requirement(self, jd_text: str) -> Optional[float]:
        """Extract required years of experience from JD."""
        patterns = [
            r'(\d+)\+?\s*(?:years?|yrs?)\s*(?:of)?\s*(?:experience|professional|industry)',
            r'(?:at\s+least|minimum)\s+(\d+)\s*(?:years?|yrs?)',
            r'(\d+)\s*-\s*\d+\s*(?:years?|yrs?)\s*(?:of)?\s*(?:experience|professional)',
            r'(\d+)\+?\s*(?:years?|yrs?)',
        ]

        for pattern in patterns:
            match = re.search(pattern, jd_text, re.IGNORECASE)
            if match:
                return float(match.group(1))

        return None

    def _extract_seniority(self, jd_text: str, years: Optional[float]) -> str:
        """Determine seniority level from JD text and years requirement."""
        jd_lower = jd_text.lower()

        # Direct title mentions
        if any(kw in jd_lower for kw in ["principal", "staff"]):
            return "principal"
        if any(kw in jd_lower for kw in ["senior", "sr."]):
            return "senior"
        if any(kw in jd_lower for kw in ["lead", "team lead", "tech lead"]):
            return "lead"
        if any(kw in jd_lower for kw in ["junior", "jr.", "entry-level", "new grad"]):
            return "junior"
        if any(kw in jd_lower for kw in ["mid-level", "mid level", "intermediate"]):
            return "mid"

        # Infer from years
        if years is not None:
            if years <= 2:
                return "junior"
            elif years <= 5:
                return "mid"
            elif years <= 8:
                return "senior"
            else:
                return "staff"

        return "mid"  # Default assumption

    def _extract_mandatory_requirements(self, jd_text: str) -> List[str]:
        """Extract explicitly mandatory requirements (marked with must/required)."""
        mandatory = []

        patterns = [
            r'(?:must\s+have|required|mandatory|essential)\s*[:\-]?\s*([^\n]+)',
            r'([^\n]+)\s*\(required\)',
            r'([^\n]+)\s*\(must\s+have\)',
        ]

        for pattern in patterns:
            matches = re.findall(pattern, jd_text, re.IGNORECASE)
            mandatory.extend([m.strip() for m in matches if len(m.strip()) > 3])

        return mandatory
