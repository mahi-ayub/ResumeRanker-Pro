"""Tests for scoring and weights."""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scoring.weights import DynamicWeightCalculator
from scoring.explainer import ScoreExplainer


class TestDynamicWeights:
    """Test role-aware weight computation."""

    def setup_method(self):
        self.calc = DynamicWeightCalculator()

    def test_default_weights(self):
        jd_analysis = {"role_type": "default"}
        weights = self.calc.compute_weights(jd_analysis)
        assert "skill_match" in weights
        assert "experience" in weights
        assert "project_relevance" in weights
        assert "education" in weights
        # Main weights should sum close to 1.0
        main_sum = (
            weights["skill_match"]
            + weights["experience"]
            + weights["project_relevance"]
            + weights["education"]
        )
        assert abs(main_sum - 1.0) < 0.01

    def test_ml_engineer_weights(self):
        jd_analysis = {"role_type": "ml_engineer"}
        weights = self.calc.compute_weights(jd_analysis)
        # ML role should have higher skill and project weights
        default_weights = self.calc.compute_weights({"role_type": "default"})
        assert weights["skill_match"] >= default_weights["skill_match"] - 0.01

    def test_backend_engineer_weights(self):
        jd_analysis = {"role_type": "backend_engineer"}
        weights = self.calc.compute_weights(jd_analysis)
        assert weights["skill_match"] > 0
        assert weights["experience"] > 0

    def test_unknown_role_defaults(self):
        jd_analysis = {"role_type": "underwater_basket_weaving"}
        weights = self.calc.compute_weights(jd_analysis)
        # Should fall back to default
        assert "skill_match" in weights

    def test_explain_weights(self):
        weights = self.calc.compute_weights({"role_type": "ml_engineer"})
        explanation = self.calc.explain_weights(weights, "ml_engineer")
        assert "ML" in explanation or "ml" in explanation.lower()


class TestScoreExplainer:
    """Test score explanation generation."""

    def setup_method(self):
        self.explainer = ScoreExplainer()

    def test_summary_high_score(self):
        # Create a mock score result
        from dataclasses import dataclass, field
        from typing import List, Dict

        class MockResult:
            overall_score = 85.0
            skill_match_score = 80.0
            experience_score = 75.0
            project_relevance_score = 70.0
            education_score = 60.0
            certification_bonus = 6.0
            missing_skill_penalty = 5.0
            matched_skills = [{"jd_skill": "Python", "resume_skill": "Python", "similarity": 0.95}]
            missing_skills = [{"skill": "Kafka", "best_match": "RabbitMQ", "best_similarity": 0.4}]
            extra_skills = ["Go"]
            project_scores = [{"name": "MyProject", "relevance": 0.8}]
            experience_analysis = {"total_years": 5, "seniority_level": "senior", "experience_quality": {}}
            strengths = ["Strong skills"]
            weaknesses = []
            score_reasoning = ["test"]

        result = MockResult()
        explanation = self.explainer.generate_full_explanation(result)
        assert "summary" in explanation
        assert "Excellent" in explanation["summary"]

    def test_summary_low_score(self):
        class MockResult:
            overall_score = 30.0
            skill_match_score = 25.0
            experience_score = 30.0
            project_relevance_score = 20.0
            education_score = 40.0
            certification_bonus = 0.0
            missing_skill_penalty = 10.0
            matched_skills = []
            missing_skills = [{"skill": "Python"}]
            extra_skills = []
            project_scores = []
            experience_analysis = {"total_years": 1}
            strengths = []
            weaknesses = ["Low match"]
            score_reasoning = ["test"]

        result = MockResult()
        explanation = self.explainer.generate_full_explanation(result)
        assert "Weak" in explanation["summary"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
