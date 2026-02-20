"""Tests for NLP modules."""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nlp.jd_analyzer import JDAnalyzer


SAMPLE_JD = """
Senior Backend Engineer

About the Role:
We are looking for a Senior Backend Engineer to join our team and help build
scalable, high-performance distributed systems.

Requirements:
- 5+ years of professional software engineering experience
- Strong proficiency in Python and/or Go
- Experience with Django, Flask, or FastAPI
- Deep knowledge of PostgreSQL and Redis
- Experience with Docker, Kubernetes, and CI/CD pipelines
- Familiarity with microservices architecture
- Strong understanding of REST API design

Nice to Have:
- Experience with Kafka or RabbitMQ
- Knowledge of Terraform and infrastructure as code
- AWS or GCP cloud certifications
- Experience with GraphQL
- Contributions to open-source projects

Responsibilities:
- Design and implement backend services handling millions of requests
- Optimize database performance and query efficiency
- Mentor junior engineers and conduct code reviews
- Collaborate with frontend and DevOps teams
"""


class TestJDAnalyzer:
    """Test JD analysis."""

    def setup_method(self):
        self.analyzer = JDAnalyzer()

    def test_analyze_returns_dict(self):
        result = self.analyzer.analyze(SAMPLE_JD)
        assert isinstance(result, dict)
        assert "role_type" in result
        assert "required_skills" in result
        assert "preferred_skills" in result

    def test_role_classification(self):
        result = self.analyzer.analyze(SAMPLE_JD)
        # Should detect backend engineer
        assert result["role_type"] in ["backend_engineer", "fullstack_engineer"]

    def test_extract_skills(self):
        result = self.analyzer.analyze(SAMPLE_JD)
        all_skills = [s.lower() for s in result["all_skills"]]
        assert any("python" in s for s in all_skills)
        assert any("docker" in s for s in all_skills)

    def test_extract_years(self):
        result = self.analyzer.analyze(SAMPLE_JD)
        assert result["years_required"] == 5.0

    def test_detect_seniority(self):
        result = self.analyzer.analyze(SAMPLE_JD)
        assert result["seniority"] == "senior"

    def test_empty_jd(self):
        result = self.analyzer.analyze("")
        assert result["role_type"] == "default"
        assert result["years_required"] is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
