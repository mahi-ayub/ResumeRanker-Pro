"""Section Extractor — splits raw resume text into structured sections.

Uses heuristic header detection + NLP patterns to identify:
- Contact Info, Summary/Objective, Skills, Experience, Projects,
  Education, Certifications, Awards, Publications, etc.
"""

import re
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from loguru import logger


@dataclass
class ResumeData:
    """Structured representation of a parsed resume."""
    raw_text: str = ""
    contact_info: Dict[str, str] = field(default_factory=dict)
    summary: str = ""
    skills: List[str] = field(default_factory=list)
    experience: List[Dict] = field(default_factory=list)
    projects: List[Dict] = field(default_factory=list)
    education: List[Dict] = field(default_factory=list)
    certifications: List[str] = field(default_factory=list)
    awards: List[str] = field(default_factory=list)
    publications: List[str] = field(default_factory=list)
    languages: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "contact_info": self.contact_info,
            "summary": self.summary,
            "skills": self.skills,
            "experience": self.experience,
            "projects": self.projects,
            "education": self.education,
            "certifications": self.certifications,
            "awards": self.awards,
            "publications": self.publications,
            "languages": self.languages,
        }


# Section header patterns — order matters (first match wins)
SECTION_PATTERNS = {
    "contact": re.compile(
        r'^(?:contact\s*(?:info|information|details)?|personal\s*(?:info|information|details)?)\s*:?\s*$',
        re.IGNORECASE | re.MULTILINE,
    ),
    "summary": re.compile(
        r'^(?:summary|professional\s*summary|objective|about\s*me|profile|career\s*summary|overview)\s*:?\s*$',
        re.IGNORECASE | re.MULTILINE,
    ),
    "skills": re.compile(
        r'^(?:skills|technical\s*skills|core\s*(?:competencies|skills)|technologies|tools\s*(?:&|and)\s*technologies|tech\s*stack)\s*:?\s*$',
        re.IGNORECASE | re.MULTILINE,
    ),
    "experience": re.compile(
        r'^(?:experience|work\s*experience|professional\s*experience|employment\s*(?:history)?|work\s*history|career\s*experience)\s*:?\s*$',
        re.IGNORECASE | re.MULTILINE,
    ),
    "projects": re.compile(
        r'^(?:projects|personal\s*projects|key\s*projects|selected\s*projects|portfolio)\s*:?\s*$',
        re.IGNORECASE | re.MULTILINE,
    ),
    "education": re.compile(
        r'^(?:education|academic\s*(?:background|qualifications)|qualifications|degrees?)\s*:?\s*$',
        re.IGNORECASE | re.MULTILINE,
    ),
    "certifications": re.compile(
        r'^(?:certifications?|licenses?\s*(?:&|and)\s*certifications?|professional\s*certifications?|credentials)\s*:?\s*$',
        re.IGNORECASE | re.MULTILINE,
    ),
    "awards": re.compile(
        r'^(?:awards?|honors?\s*(?:&|and)\s*awards?|achievements?|recognitions?)\s*:?\s*$',
        re.IGNORECASE | re.MULTILINE,
    ),
    "publications": re.compile(
        r'^(?:publications?|papers?|research\s*(?:papers?|publications?))\s*:?\s*$',
        re.IGNORECASE | re.MULTILINE,
    ),
    "languages": re.compile(
        r'^(?:languages?|spoken\s*languages?)\s*:?\s*$',
        re.IGNORECASE | re.MULTILINE,
    ),
}


class SectionExtractor:
    """Extract sections from raw resume text using heuristic header detection."""

    def extract(self, text: str) -> ResumeData:
        """
        Parse raw resume text into structured ResumeData.

        Args:
            text: Raw text extracted from a resume PDF/file.

        Returns:
            ResumeData with populated sections.
        """
        resume = ResumeData(raw_text=text)

        # 1. Split text into sections by detected headers
        sections = self._split_into_sections(text)

        # 2. Populate each section
        resume.contact_info = self._extract_contact_info(text)
        resume.summary = sections.get("summary", "").strip()
        resume.skills = self._parse_skills_section(sections.get("skills", ""))
        resume.experience = self._parse_experience_section(sections.get("experience", ""))
        resume.projects = self._parse_projects_section(sections.get("projects", ""))
        resume.education = self._parse_education_section(sections.get("education", ""))
        resume.certifications = self._parse_list_section(sections.get("certifications", ""))
        resume.awards = self._parse_list_section(sections.get("awards", ""))
        resume.publications = self._parse_list_section(sections.get("publications", ""))
        resume.languages = self._parse_list_section(sections.get("languages", ""))

        logger.info(
            f"Extracted sections — Skills: {len(resume.skills)}, "
            f"Experience: {len(resume.experience)}, Projects: {len(resume.projects)}, "
            f"Education: {len(resume.education)}, Certifications: {len(resume.certifications)}"
        )

        return resume

    def _split_into_sections(self, text: str) -> Dict[str, str]:
        """Split text into named sections based on header patterns."""
        # Find all section headers with their positions
        headers = []
        for section_name, pattern in SECTION_PATTERNS.items():
            for match in pattern.finditer(text):
                headers.append((match.start(), match.end(), section_name))

        # Sort by position
        headers.sort(key=lambda x: x[0])

        sections = {}
        for i, (start, end, name) in enumerate(headers):
            # Section content runs from end of header to start of next header
            next_start = headers[i + 1][0] if i + 1 < len(headers) else len(text)
            sections[name] = text[end:next_start].strip()

        # If no sections detected, try to infer from the full text
        if not sections:
            logger.warning("No section headers detected. Attempting full-text extraction.")
            sections["skills"] = text
            sections["experience"] = text

        return sections

    def _extract_contact_info(self, text: str) -> Dict[str, str]:
        """Extract email, phone, LinkedIn, GitHub from text."""
        info = {}

        email_match = re.search(r'[\w.+-]+@[\w-]+\.[\w.-]+', text)
        if email_match:
            info["email"] = email_match.group()

        phone_match = re.search(
            r'(?:\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}', text
        )
        if phone_match:
            info["phone"] = phone_match.group()

        linkedin_match = re.search(r'(?:linkedin\.com/in/|linkedin:\s*)([\w-]+)', text, re.IGNORECASE)
        if linkedin_match:
            info["linkedin"] = linkedin_match.group(1)

        github_match = re.search(r'(?:github\.com/|github:\s*)([\w-]+)', text, re.IGNORECASE)
        if github_match:
            info["github"] = github_match.group(1)

        return info

    def _parse_skills_section(self, text: str) -> List[str]:
        """Parse skills from a skills section."""
        if not text.strip():
            return []

        skills = []

        # Handle comma/pipe/bullet separated skills
        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue

            # Remove leading bullets, dashes, asterisks
            line = re.sub(r'^[\-\*•►▪‣⁃]\s*', '', line)

            # Remove category headers like "Languages:" "Frameworks:"
            if re.match(r'^[\w\s]+:\s*', line):
                line = re.sub(r'^[\w\s]+:\s*', '', line)

            # Split by common delimiters
            parts = re.split(r'[,|;•►▪/]|\s{2,}', line)
            for part in parts:
                skill = part.strip().strip('-').strip('•').strip()
                if skill and len(skill) > 1 and len(skill) < 50:
                    skills.append(skill)

        return list(dict.fromkeys(skills))  # Deduplicate preserving order

    def _parse_experience_section(self, text: str) -> List[Dict]:
        """Parse work experience entries."""
        if not text.strip():
            return []

        entries = []
        # Split by common experience entry patterns (dates or company headers)
        # Pattern: lines that look like "Company Name — Role" or contain date ranges
        blocks = re.split(
            r'\n(?=(?:[A-Z][\w\s,&.]+(?:—|-|–|\|))|(?:\d{4}\s*[-–]\s*(?:\d{4}|present|current)))',
            text,
            flags=re.IGNORECASE,
        )

        for block in blocks:
            block = block.strip()
            if len(block) < 20:
                continue

            entry = self._parse_experience_block(block)
            if entry:
                entries.append(entry)

        return entries

    def _parse_experience_block(self, block: str) -> Optional[Dict]:
        """Parse a single experience entry block."""
        lines = [l.strip() for l in block.split("\n") if l.strip()]
        if not lines:
            return None

        entry = {
            "title": "",
            "company": "",
            "dates": "",
            "duration_years": None,
            "description": "",
            "bullets": [],
        }

        # First line often has title/company
        first_line = lines[0]
        entry["title"] = first_line

        # Extract date range
        date_pattern = r'(\w+\.?\s*\d{4})\s*[-–]\s*(\w+\.?\s*\d{4}|present|current|now)'
        date_match = re.search(date_pattern, block, re.IGNORECASE)
        if date_match:
            entry["dates"] = date_match.group(0)
            entry["duration_years"] = self._estimate_duration(
                date_match.group(1), date_match.group(2)
            )

        # Collect bullet points
        for line in lines[1:]:
            cleaned = re.sub(r'^[\-\*•►▪‣⁃]\s*', '', line).strip()
            if cleaned:
                entry["bullets"].append(cleaned)

        entry["description"] = " ".join(entry["bullets"])
        return entry

    def _estimate_duration(self, start: str, end: str) -> Optional[float]:
        """Estimate duration in years from date strings."""
        from dateutil import parser as date_parser
        from datetime import datetime

        try:
            start_date = date_parser.parse(start, fuzzy=True)
            if re.match(r'present|current|now', end, re.IGNORECASE):
                end_date = datetime.now()
            else:
                end_date = date_parser.parse(end, fuzzy=True)

            diff = (end_date - start_date).days / 365.25
            return round(max(0, diff), 1)
        except Exception:
            return None

    def _parse_projects_section(self, text: str) -> List[Dict]:
        """Parse project entries."""
        if not text.strip():
            return []

        projects = []
        # Split by lines that look like project titles
        blocks = re.split(r'\n(?=[A-Z][\w\s]+(?:[-–—|:]|\n))', text)

        for block in blocks:
            block = block.strip()
            if len(block) < 15:
                continue

            lines = [l.strip() for l in block.split("\n") if l.strip()]
            if not lines:
                continue

            project = {
                "name": lines[0].rstrip(":").rstrip("-").rstrip("–").strip(),
                "description": " ".join(lines[1:]),
                "technologies": [],
            }

            # Extract technologies mentioned in the project
            tech_match = re.search(
                r'(?:tech(?:nologies|stack)?|tools?|built\s+with|using)\s*:\s*(.+)',
                block, re.IGNORECASE
            )
            if tech_match:
                project["technologies"] = [
                    t.strip() for t in re.split(r'[,|;]', tech_match.group(1)) if t.strip()
                ]

            projects.append(project)

        return projects

    def _parse_education_section(self, text: str) -> List[Dict]:
        """Parse education entries."""
        if not text.strip():
            return []

        entries = []
        blocks = re.split(r'\n(?=[A-Z])', text)

        for block in blocks:
            block = block.strip()
            if len(block) < 10:
                continue

            entry = {
                "institution": "",
                "degree": "",
                "field": "",
                "year": "",
                "gpa": "",
            }

            lines = [l.strip() for l in block.split("\n") if l.strip()]
            if lines:
                entry["institution"] = lines[0]

            # Extract degree
            degree_match = re.search(
                r'(B\.?S\.?|M\.?S\.?|B\.?A\.?|M\.?A\.?|Ph\.?D\.?|Bachelor|Master|Doctor|MBA|B\.?Tech|M\.?Tech)'
                r'[\w\s]*(?:in|of)?\s*([\w\s,&]+)?',
                block, re.IGNORECASE
            )
            if degree_match:
                entry["degree"] = degree_match.group(1).strip()
                if degree_match.group(2):
                    entry["field"] = degree_match.group(2).strip()

            # Extract year
            year_match = re.search(r'(20\d{2}|19\d{2})', block)
            if year_match:
                entry["year"] = year_match.group(1)

            # Extract GPA
            gpa_match = re.search(r'(?:GPA|CGPA)[\s:]*(\d+\.?\d*)\s*/?\s*(\d+\.?\d*)?', block, re.IGNORECASE)
            if gpa_match:
                entry["gpa"] = gpa_match.group(1)

            entries.append(entry)

        return entries

    def _parse_list_section(self, text: str) -> List[str]:
        """Parse a generic list section (certifications, awards, etc.)."""
        if not text.strip():
            return []

        items = []
        for line in text.split("\n"):
            line = re.sub(r'^[\-\*•►▪‣⁃\d.)\]]+\s*', '', line).strip()
            if line and len(line) > 3:
                items.append(line)

        return items
