"""Central configuration for the Resume Scanner."""

from dataclasses import dataclass, field
from typing import Dict
import torch


@dataclass
class ModelConfig:
    """Embedding model settings."""
    model_name: str = "all-MiniLM-L6-v2"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    max_seq_length: int = 384
    batch_size: int = 32


@dataclass
class ScoringConfig:
    """Default scoring weights (before role-aware adjustment)."""
    skill_match_weight: float = 0.35
    experience_weight: float = 0.25
    project_relevance_weight: float = 0.20
    education_weight: float = 0.10
    certification_bonus: float = 0.03
    missing_skill_penalty: float = 0.05
    max_missing_penalty: float = 0.20  # Cap total missing-skill penalty

    # Semantic similarity thresholds
    high_similarity_threshold: float = 0.75
    medium_similarity_threshold: float = 0.50
    low_similarity_threshold: float = 0.30


@dataclass
class ParsingConfig:
    """Resume parsing settings."""
    max_pages: int = 10
    min_text_length: int = 50  # Minimum chars to consider a valid parse
    spacy_model: str = "en_core_web_sm"


# Role-type weight overrides
# Each maps a detected role type to weight adjustments (multipliers)
ROLE_WEIGHT_PROFILES: Dict[str, Dict[str, float]] = {
    "backend_engineer": {
        "skill_match_weight": 1.2,
        "experience_weight": 1.1,
        "project_relevance_weight": 1.0,
        "education_weight": 0.7,
    },
    "frontend_engineer": {
        "skill_match_weight": 1.2,
        "experience_weight": 1.0,
        "project_relevance_weight": 1.1,
        "education_weight": 0.7,
    },
    "ml_engineer": {
        "skill_match_weight": 1.3,
        "experience_weight": 1.0,
        "project_relevance_weight": 1.2,
        "education_weight": 1.1,
    },
    "data_scientist": {
        "skill_match_weight": 1.2,
        "experience_weight": 1.0,
        "project_relevance_weight": 1.1,
        "education_weight": 1.2,
    },
    "devops_engineer": {
        "skill_match_weight": 1.3,
        "experience_weight": 1.2,
        "project_relevance_weight": 0.9,
        "education_weight": 0.6,
    },
    "fullstack_engineer": {
        "skill_match_weight": 1.1,
        "experience_weight": 1.1,
        "project_relevance_weight": 1.1,
        "education_weight": 0.7,
    },
    "data_engineer": {
        "skill_match_weight": 1.2,
        "experience_weight": 1.1,
        "project_relevance_weight": 1.0,
        "education_weight": 0.9,
    },
    "default": {
        "skill_match_weight": 1.0,
        "experience_weight": 1.0,
        "project_relevance_weight": 1.0,
        "education_weight": 1.0,
    },
}

# Skill taxonomy for semantic grouping
SKILL_CATEGORIES = {
    "programming_languages": [
        "python", "java", "javascript", "typescript", "c++", "c#", "go",
        "rust", "ruby", "php", "swift", "kotlin", "scala", "r", "matlab",
    ],
    "ml_frameworks": [
        "pytorch", "tensorflow", "keras", "scikit-learn", "xgboost",
        "lightgbm", "hugging face", "transformers", "jax", "caffe",
    ],
    "web_frameworks": [
        "react", "angular", "vue", "django", "flask", "fastapi",
        "spring", "express", "next.js", "nuxt", "svelte",
    ],
    "databases": [
        "postgresql", "mysql", "mongodb", "redis", "elasticsearch",
        "cassandra", "dynamodb", "sqlite", "oracle", "sql server",
    ],
    "cloud_platforms": [
        "aws", "azure", "gcp", "google cloud", "heroku",
        "digitalocean", "vercel", "netlify",
    ],
    "devops_tools": [
        "docker", "kubernetes", "terraform", "ansible", "jenkins",
        "github actions", "gitlab ci", "circleci", "prometheus", "grafana",
    ],
    "data_tools": [
        "spark", "hadoop", "airflow", "kafka", "flink",
        "dbt", "snowflake", "bigquery", "redshift", "databricks",
    ],
    "data_analysis_tools": [
        "pandas", "numpy", "matplotlib", "seaborn", "plotly",
        "power bi", "tableau", "looker", "excel", "google sheets",
        "sas", "spss", "stata", "alteryx", "qlik",
        "sql", "sql server", "scipy", "statsmodels",
    ],
}

# Generic terms that should never count as specific skill matches.
# These are too vague / domain-agnostic to differentiate candidates.
GENERIC_BLOCKLIST: set = {
    "data", "system", "systems", "cloud", "development", "analysis",
    "experience", "design", "management", "engineering", "platform",
    "architecture", "infrastructure", "security", "automation",
    "testing", "deployment", "integration", "communication",
    "leadership", "teamwork", "collaboration", "problem solving",
    "problem-solving", "analytical", "research", "reporting",
    "documentation", "support", "operations", "services", "solutions",
    "technology", "tools", "software", "hardware", "network",
    "networking", "database", "web", "mobile", "frontend", "backend",
    "full-stack", "fullstack", "devops", "machine learning",
    "artificial intelligence", "deep learning", "natural language processing",
}

# Seniority level definitions
SENIORITY_LEVELS = {
    "intern": {"min_years": 0, "max_years": 1},
    "junior": {"min_years": 0, "max_years": 2},
    "mid": {"min_years": 2, "max_years": 5},
    "senior": {"min_years": 5, "max_years": 10},
    "staff": {"min_years": 8, "max_years": 15},
    "principal": {"min_years": 10, "max_years": 25},
    "lead": {"min_years": 5, "max_years": 15},
    "manager": {"min_years": 5, "max_years": 20},
    "director": {"min_years": 10, "max_years": 25},
}
