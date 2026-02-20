"""Bias & Risk Detector — flags potential bias markers and PII in resumes.

Detects:
- Age indicators (graduation year, birth date)
- Gender markers (pronouns, gendered titles)
- Photo presence indicators
- Marital status mentions
- Unnecessary personal details
- Protected class information

This is designed to help hiring teams ensure fair, unbiased evaluation.
"""

import re
from typing import List, Dict
from loguru import logger


class BiasDetector:
    """Detect potential bias markers and PII in resume text."""

    def detect(self, text: str) -> Dict:
        """
        Scan resume text for bias markers and PII.

        Returns:
            Dict with:
                - risk_level: 'low' | 'medium' | 'high'
                - flags: List of detected issues
                - recommendations: List of suggestions
        """
        flags = []

        flags.extend(self._detect_age_markers(text))
        flags.extend(self._detect_gender_markers(text))
        flags.extend(self._detect_photo_markers(text))
        flags.extend(self._detect_marital_status(text))
        flags.extend(self._detect_unnecessary_pii(text))

        # Determine risk level
        high_flags = [f for f in flags if f["severity"] == "high"]
        medium_flags = [f for f in flags if f["severity"] == "medium"]

        if high_flags:
            risk_level = "high"
        elif medium_flags:
            risk_level = "medium"
        else:
            risk_level = "low"

        recommendations = self._generate_recommendations(flags)

        return {
            "risk_level": risk_level,
            "total_flags": len(flags),
            "flags": flags,
            "recommendations": recommendations,
        }

    def _detect_age_markers(self, text: str) -> List[Dict]:
        """Detect age-related bias markers."""
        flags = []

        # Birth date
        birth_pattern = re.compile(
            r'(?:born|birth\s*(?:date)?|d\.?o\.?b\.?|date\s+of\s+birth)\s*:?\s*[\d/\-\.\w]+',
            re.IGNORECASE,
        )
        if birth_pattern.search(text):
            flags.append({
                "type": "age",
                "severity": "high",
                "description": "Birth date or date of birth detected",
                "detail": "Remove date of birth — it enables age discrimination.",
            })

        # Very old graduation years (potential age indicator)
        grad_years = re.findall(r'\b(19[5-8]\d|199\d)\b', text)
        if grad_years:
            flags.append({
                "type": "age",
                "severity": "medium",
                "description": f"Older graduation year(s) detected: {', '.join(grad_years)}",
                "detail": "Consider removing graduation years older than 15 years to avoid age bias.",
            })

        return flags

    def _detect_gender_markers(self, text: str) -> List[Dict]:
        """Detect gender-related bias markers."""
        flags = []

        # Gendered titles
        title_pattern = re.compile(
            r'\b(?:Mr\.|Mrs\.|Ms\.|Miss|Mx\.)\s', re.IGNORECASE
        )
        if title_pattern.search(text):
            flags.append({
                "type": "gender",
                "severity": "medium",
                "description": "Gendered title (Mr./Mrs./Ms.) detected",
                "detail": "Consider removing gendered titles to prevent unconscious bias.",
            })

        # Gendered pronouns in summary/about (unusual in resume)
        pronoun_pattern = re.compile(
            r'\b(?:he\s+is|she\s+is|his\s+experience|her\s+experience)\b',
            re.IGNORECASE,
        )
        if pronoun_pattern.search(text):
            flags.append({
                "type": "gender",
                "severity": "low",
                "description": "Gendered pronouns detected in resume text",
                "detail": "Use neutral language or first person in resumes.",
            })

        return flags

    def _detect_photo_markers(self, text: str) -> List[Dict]:
        """Detect indicators that a photo may be embedded."""
        flags = []

        # Photo keywords
        photo_pattern = re.compile(
            r'\b(?:photograph|photo|headshot|portrait|picture)\b',
            re.IGNORECASE,
        )
        if photo_pattern.search(text):
            flags.append({
                "type": "photo",
                "severity": "high",
                "description": "Photo reference detected",
                "detail": "Remove photos from resumes to prevent appearance-based bias.",
            })

        return flags

    def _detect_marital_status(self, text: str) -> List[Dict]:
        """Detect marital status mentions."""
        flags = []

        marital_pattern = re.compile(
            r'\b(?:married|single|divorced|widowed|marital\s+status|spouse|wife|husband)\b',
            re.IGNORECASE,
        )
        if marital_pattern.search(text):
            flags.append({
                "type": "marital_status",
                "severity": "high",
                "description": "Marital status reference detected",
                "detail": "Remove marital status — it is irrelevant to job qualifications.",
            })

        return flags

    def _detect_unnecessary_pii(self, text: str) -> List[Dict]:
        """Detect unnecessary personal information."""
        flags = []

        # National ID / SSN patterns
        ssn_pattern = re.compile(r'\b\d{3}[-.]?\d{2}[-.]?\d{4}\b')
        if ssn_pattern.search(text):
            flags.append({
                "type": "pii",
                "severity": "high",
                "description": "Possible SSN or national ID number detected",
                "detail": "NEVER include SSN or national ID numbers on a resume.",
            })

        # Physical address (full street address)
        address_pattern = re.compile(
            r'\d+\s+[\w\s]+(?:Street|St|Avenue|Ave|Boulevard|Blvd|Road|Rd|Drive|Dr|Lane|Ln)\b',
            re.IGNORECASE,
        )
        if address_pattern.search(text):
            flags.append({
                "type": "pii",
                "severity": "low",
                "description": "Full street address detected",
                "detail": "Consider using city/state only instead of full address.",
            })

        # Religion
        religion_pattern = re.compile(
            r'\b(?:christian|muslim|jewish|hindu|buddhist|sikh|catholic|protestant|religion|church|mosque|temple)\b',
            re.IGNORECASE,
        )
        if religion_pattern.search(text):
            flags.append({
                "type": "religion",
                "severity": "medium",
                "description": "Religious reference detected",
                "detail": "Consider removing religious references unless directly relevant to the role.",
            })

        # Nationality
        nationality_pattern = re.compile(
            r'\b(?:nationality|citizenship|national\s+of|passport\s+(?:no|number))\b',
            re.IGNORECASE,
        )
        if nationality_pattern.search(text):
            flags.append({
                "type": "nationality",
                "severity": "medium",
                "description": "Nationality or citizenship reference detected",
                "detail": "Only include work authorization status if relevant, not specific nationality.",
            })

        return flags

    def _generate_recommendations(self, flags: List[Dict]) -> List[str]:
        """Generate actionable recommendations based on detected flags."""
        recs = []

        if not flags:
            recs.append("✅ No significant bias markers detected. Resume looks clean.")
            return recs

        flag_types = set(f["type"] for f in flags)

        if "age" in flag_types:
            recs.append(
                "🔴 Remove age-related information (birth date, old graduation years). "
                "Focus on recent 10-15 years of experience."
            )

        if "gender" in flag_types:
            recs.append(
                "🟡 Remove gendered titles and use neutral language throughout."
            )

        if "photo" in flag_types:
            recs.append(
                "🔴 Remove any photo or headshot from the resume."
            )

        if "marital_status" in flag_types:
            recs.append(
                "🔴 Remove marital status — it has no bearing on job qualifications."
            )

        if "pii" in flag_types:
            recs.append(
                "🔴 Remove sensitive PII (SSN, full address). Use city/state only."
            )

        if "religion" in flag_types:
            recs.append(
                "🟡 Consider removing religious references unless directly relevant."
            )

        return recs
