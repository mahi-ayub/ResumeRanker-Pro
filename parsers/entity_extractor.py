"""Entity Extractor — extracts skills, dates, organizations using spaCy NER + patterns.

Augments section-level extraction with fine-grained NLP-based entity recognition.
"""

import re
from typing import List, Dict, Set
from loguru import logger


class EntityExtractor:
    """NLP-based entity extraction from resume text."""

    def __init__(self, spacy_model: str = "en_core_web_sm"):
        self._nlp = None
        self._spacy_model = spacy_model

    @property
    def nlp(self):
        """Lazy-load spaCy model."""
        if self._nlp is None:
            import spacy
            try:
                self._nlp = spacy.load(self._spacy_model)
            except OSError:
                logger.warning(
                    f"spaCy model '{self._spacy_model}' not found. "
                    f"Run: python -m spacy download {self._spacy_model}"
                )
                self._nlp = spacy.blank("en")
        return self._nlp

    def extract_skills_from_text(self, text: str, known_skills: Set[str] = None) -> List[str]:
        """
        Extract skill mentions from free text using NER + pattern matching.

        Args:
            text: Raw text to extract skills from.
            known_skills: Optional set of known skill names for matching.

        Returns:
            List of extracted skill strings.
        """
        skills = []

        # Pattern-based extraction: catch technology names
        tech_patterns = [
            r'\b(?:Python|Java|JavaScript|TypeScript|C\+\+|C#|Go|Rust|Ruby|PHP|Swift|Kotlin|Scala|R|MATLAB)\b',
            r'\b(?:PyTorch|TensorFlow|Keras|Scikit-learn|XGBoost|LightGBM|Hugging\s*Face|JAX)\b',
            r'\b(?:React|Angular|Vue|Django|Flask|FastAPI|Spring|Express|Next\.js|Svelte)\b',
            r'\b(?:PostgreSQL|MySQL|MongoDB|Redis|Elasticsearch|Cassandra|DynamoDB|SQLite)\b',
            r'\b(?:AWS|Azure|GCP|Google\s*Cloud|Docker|Kubernetes|Terraform|Ansible|Jenkins)\b',
            r'\b(?:Git|Linux|Nginx|Apache|RabbitMQ|Kafka|Spark|Hadoop|Airflow|Celery)\b',
            r'\b(?:HTML|CSS|SASS|LESS|GraphQL|REST|gRPC|WebSocket|OAuth|JWT)\b',
            r'\b(?:Pandas|NumPy|Matplotlib|Seaborn|Plotly|SciPy|OpenCV|NLTK|spaCy)\b',
            r'\b(?:CI/CD|DevOps|Agile|Scrum|Kanban|TDD|BDD|Microservices|Serverless)\b',
            r'\b(?:Node\.js|Deno|Bun|Vite|Webpack|Babel|ESLint|Prettier)\b',
        ]

        for pattern in tech_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            skills.extend(matches)

        # If known skills are provided, also do substring matching
        if known_skills:
            text_lower = text.lower()
            for skill in known_skills:
                if skill.lower() in text_lower:
                    skills.append(skill)

        # Deduplicate (case-insensitive)
        seen = set()
        unique = []
        for s in skills:
            key = s.strip().lower()
            if key not in seen:
                seen.add(key)
                unique.append(s.strip())

        return unique

    def extract_organizations(self, text: str) -> List[str]:
        """Extract organization names using spaCy NER."""
        doc = self.nlp(text[:10000])  # Limit for performance
        orgs = []
        for ent in doc.ents:
            if ent.label_ == "ORG":
                orgs.append(ent.text)
        return list(set(orgs))

    def extract_dates(self, text: str) -> List[str]:
        """Extract date expressions from text."""
        doc = self.nlp(text[:10000])
        dates = []
        for ent in doc.ents:
            if ent.label_ == "DATE":
                dates.append(ent.text)
        return dates

    def extract_education_entities(self, text: str) -> List[Dict]:
        """Extract education-related entities using patterns."""
        degrees = []

        degree_pattern = re.compile(
            r'(B\.?S\.?|M\.?S\.?|B\.?A\.?|M\.?A\.?|Ph\.?D\.?|Bachelor|Master|Doctor|MBA|B\.?Tech|M\.?Tech)'
            r'[^.\n]*(?:in|of)\s+([\w\s,&]+?)(?:\n|,|\d{4}|$)',
            re.IGNORECASE,
        )

        for match in degree_pattern.finditer(text):
            degrees.append({
                "degree": match.group(1).strip(),
                "field": match.group(2).strip(),
            })

        return degrees

    def extract_certifications_from_text(self, text: str) -> List[str]:
        """Extract certifications using patterns."""
        cert_patterns = [
            r'(?:AWS|Amazon)\s+(?:Certified|Solutions?\s+Architect|Developer|SysOps)[^\n,]*',
            r'(?:Google|GCP)\s+(?:Certified|Professional|Associate)[^\n,]*',
            r'(?:Microsoft|Azure)\s+(?:Certified|AZ-\d+)[^\n,]*',
            r'(?:Certified\s+)?(?:Kubernetes|CKA|CKAD|CKS)[^\n,]*',
            r'(?:PMP|CISSP|CEH|CompTIA\s+\w+|CCNA|CCNP)[^\n,]*',
            r'(?:TensorFlow|PyTorch|Databricks)\s+(?:Developer|Certified)[^\n,]*',
        ]

        certs = []
        for pattern in cert_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            certs.extend([m.strip() for m in matches])

        return list(set(certs))
