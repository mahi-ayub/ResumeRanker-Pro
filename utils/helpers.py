"""Shared utility functions."""

from typing import List, Optional
import re
from loguru import logger


def clean_text(text: str) -> str:
    """Normalize whitespace, remove control characters, and clean text."""
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def deduplicate_list(items: List[str]) -> List[str]:
    """Deduplicate a list while preserving order (case-insensitive)."""
    seen = set()
    result = []
    for item in items:
        key = item.strip().lower()
        if key and key not in seen:
            seen.add(key)
            result.append(item.strip())
    return result


def extract_years_from_text(text: str) -> Optional[float]:
    """Extract a numeric year value from text like '3 years', '5+ years'."""
    pattern = r'(\d+\.?\d*)\+?\s*(?:years?|yrs?)'
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        return float(match.group(1))
    return None


def normalize_skill(skill: str) -> str:
    """Normalize a skill name for consistent comparison."""
    skill = skill.strip().lower()
    # Common normalizations
    mappings = {
        "js": "javascript",
        "ts": "typescript",
        "py": "python",
        "node": "node.js",
        "react.js": "react",
        "reactjs": "react",
        "vue.js": "vue",
        "vuejs": "vue",
        "angular.js": "angular",
        "angularjs": "angular",
        "next.js": "nextjs",
        "postgres": "postgresql",
        "mongo": "mongodb",
        "k8s": "kubernetes",
        "tf": "tensorflow",
        "sklearn": "scikit-learn",
        "sci-kit learn": "scikit-learn",
        "aws": "amazon web services",
        "gcp": "google cloud platform",
    }
    return mappings.get(skill, skill)


def chunk_list(lst: list, chunk_size: int) -> list:
    """Split a list into chunks of given size."""
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safe division that returns default on zero denominator."""
    if denominator == 0:
        return default
    return numerator / denominator


def clamp(value: float, min_val: float = 0.0, max_val: float = 100.0) -> float:
    """Clamp a value to a range."""
    return max(min_val, min(max_val, value))
