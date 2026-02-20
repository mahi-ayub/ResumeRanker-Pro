"""Tests for ranking and bias detection."""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bias_detection.bias_detector import BiasDetector


class TestBiasDetector:
    """Test bias and risk detection."""

    def setup_method(self):
        self.detector = BiasDetector()

    def test_clean_resume(self):
        text = "John Doe, Software Engineer, 5 years of Python experience."
        result = self.detector.detect(text)
        assert result["risk_level"] == "low"

    def test_detect_birth_date(self):
        text = "Date of Birth: 01/15/1990\nSoftware Engineer"
        result = self.detector.detect(text)
        assert any(f["type"] == "age" for f in result["flags"])
        assert result["risk_level"] in ["medium", "high"]

    def test_detect_gender_title(self):
        text = "Mr. John Doe\nSenior Engineer"
        result = self.detector.detect(text)
        assert any(f["type"] == "gender" for f in result["flags"])

    def test_detect_marital_status(self):
        text = "Marital Status: Married\nSoftware Developer"
        result = self.detector.detect(text)
        assert any(f["type"] == "marital_status" for f in result["flags"])
        assert result["risk_level"] == "high"

    def test_detect_photo_reference(self):
        text = "Please see my photograph attached.\nSenior Developer"
        result = self.detector.detect(text)
        assert any(f["type"] == "photo" for f in result["flags"])

    def test_detect_ssn(self):
        text = "SSN: 123-45-6789\nSoftware Engineer"
        result = self.detector.detect(text)
        assert any(f["type"] == "pii" for f in result["flags"])

    def test_recommendations_generated(self):
        text = "Date of Birth: 01/15/1990\nMr. John Doe\nMarried"
        result = self.detector.detect(text)
        assert len(result["recommendations"]) > 0

    def test_empty_text(self):
        result = self.detector.detect("")
        assert result["risk_level"] == "low"
        assert len(result["flags"]) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
