"""Tests for resume parsers."""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from parsers.section_extractor import SectionExtractor, ResumeData
from parsers.entity_extractor import EntityExtractor


# --- Sample resume text for testing ---

SAMPLE_RESUME = """
John Doe
john.doe@email.com | (555) 123-4567
linkedin.com/in/johndoe | github.com/johndoe

Summary
Experienced backend engineer with 5+ years of experience building scalable distributed systems.
Passionate about clean architecture and performance optimization.

Skills
Python, Java, Go, JavaScript
Django, Flask, FastAPI, Spring Boot
PostgreSQL, MongoDB, Redis, Elasticsearch
Docker, Kubernetes, Terraform, AWS
CI/CD, GitHub Actions, Jenkins

Experience
Senior Backend Engineer — TechCorp Inc.
Jan 2021 - Present
- Architected microservices platform handling 10M+ requests/day
- Reduced API latency by 40% through query optimization and caching
- Led team of 5 engineers in rebuilding the payment processing system
- Implemented CI/CD pipeline reducing deployment time from 2 hours to 15 minutes

Backend Developer — StartupXYZ
Jun 2018 - Dec 2020
- Built RESTful APIs serving 50K daily active users
- Designed and implemented real-time notification system using WebSockets
- Migrated monolithic application to microservices architecture
- Improved database query performance by 60% through indexing and optimization

Projects
DistributedCache
Open-source distributed caching library built with Go
Tech stack: Go, gRPC, Raft consensus
Stars: 500+ on GitHub

ResumeAI
AI-powered resume evaluation tool using NLP
Technologies: Python, PyTorch, FastAPI, React

Education
Bachelor of Science in Computer Science
University of Technology — 2018
GPA: 3.8/4.0

Certifications
AWS Certified Solutions Architect – Associate
Certified Kubernetes Administrator (CKA)
"""


class TestSectionExtractor:
    """Test the SectionExtractor."""

    def setup_method(self):
        self.extractor = SectionExtractor()

    def test_extract_returns_resume_data(self):
        result = self.extractor.extract(SAMPLE_RESUME)
        assert isinstance(result, ResumeData)

    def test_extract_skills(self):
        result = self.extractor.extract(SAMPLE_RESUME)
        assert len(result.skills) > 0
        # Check some known skills are extracted
        skills_lower = [s.lower() for s in result.skills]
        assert any("python" in s for s in skills_lower)
        assert any("docker" in s for s in skills_lower)

    def test_extract_experience(self):
        result = self.extractor.extract(SAMPLE_RESUME)
        assert len(result.experience) >= 1
        # Check that experience entries have bullets
        for entry in result.experience:
            if entry.get("bullets"):
                assert len(entry["bullets"]) > 0

    def test_extract_education(self):
        result = self.extractor.extract(SAMPLE_RESUME)
        assert len(result.education) >= 1

    def test_extract_contact_info(self):
        result = self.extractor.extract(SAMPLE_RESUME)
        assert "email" in result.contact_info
        assert "john.doe@email.com" in result.contact_info["email"]

    def test_extract_certifications(self):
        result = self.extractor.extract(SAMPLE_RESUME)
        assert len(result.certifications) >= 1

    def test_extract_projects(self):
        result = self.extractor.extract(SAMPLE_RESUME)
        assert len(result.projects) >= 1

    def test_to_dict(self):
        result = self.extractor.extract(SAMPLE_RESUME)
        d = result.to_dict()
        assert "skills" in d
        assert "experience" in d
        assert "education" in d


class TestEntityExtractor:
    """Test the EntityExtractor."""

    def setup_method(self):
        self.extractor = EntityExtractor()

    def test_extract_skills_from_text(self):
        text = "I have experience with Python, PyTorch, and Docker."
        skills = self.extractor.extract_skills_from_text(text)
        skills_lower = [s.lower() for s in skills]
        assert "python" in skills_lower
        assert "pytorch" in skills_lower
        assert "docker" in skills_lower

    def test_extract_skills_deduplication(self):
        text = "Python and python and PYTHON are all the same."
        skills = self.extractor.extract_skills_from_text(text)
        python_count = sum(1 for s in skills if s.lower() == "python")
        assert python_count == 1

    def test_extract_certifications(self):
        text = "I hold the AWS Certified Solutions Architect credential."
        certs = self.extractor.extract_certifications_from_text(text)
        assert len(certs) >= 1
        assert any("aws" in c.lower() for c in certs)

    def test_extract_skills_empty_text(self):
        skills = self.extractor.extract_skills_from_text("")
        assert skills == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
