<div align="center">

# ResumeRanker-Pro

**ATS-Calibrated Resume Scoring Engine**

Parse PDF resumes · Extract skills with NLP · Score against any job description

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Tests](https://img.shields.io/badge/Tests-33%20passed-brightgreen?logo=pytest&logoColor=white)](#running-tests)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## What It Does

ResumeRanker-Pro takes a **job description** and one or more **PDF resumes**, then produces a single **ATS-style score (35–88)** with a full breakdown of why the candidate scored that way.

Under the hood it runs four scoring components, applies smooth bonuses/penalties, and compresses the result into the realistic range that actual ATS systems produce — no perfect 100s, no meaningless 0s.

---

## Key Features

| Feature | Description |
|:--------|:------------|
| **Semantic Matching** | Sentence-transformer embeddings (`all-MiniLM-L6-v2`) compare resume ↔ JD at document and skill level |
| **Required-Skill Weighting** | Required JD skills carry the largest weight; generic terms auto-filtered via blocklist |
| **Tool Match** | Case-insensitive exact-match lookup for technologies, frameworks, and tools |
| **Experience Alignment** | Estimates years of experience from resume, compares against JD requirements |
| **Score Compression** | Raw weighted score mapped to **[35, 88]** — mirrors real ATS output ranges |
| **Bias Detection** | Flags PII, gendered language, age markers, and marital status references |
| **Improvement Engine** | Actionable, severity-ranked suggestions to strengthen the resume |
| **Multi-Resume Ranking** | Score and rank multiple candidates against one JD in a single run |
| **Streamlit Dashboard** | Interactive UI with gauge charts, radar plots, and JSON export |

---

## Scoring Formula

### Component Weights

| Component | Symbol | Weight |
|:----------|:------:|-------:|
| Semantic Similarity | `W_SEMANTIC` | **0.28** |
| Required Skill Match | `W_REQUIRED` | **0.45** |
| Tool Match | `W_TOOL` | **0.15** |
| Experience Relevance | `W_EXPERIENCE` | **0.12** |

### Adjustments

| Adjustment | Range | Description |
|:-----------|:-----:|:------------|
| Keyword Boost | 0 to +5 | Bonus for exact JD keyword hits in resume |
| Certification Bonus | 0 to +4 | Bonus for relevant certifications |
| Missing-Skill Penalty | 0 to −8 | Smooth penalty for missing required skills |

### Final Score

```
raw = (W_SEMANTIC × semantic) + (W_REQUIRED × required_match)
    + (W_TOOL × tool_match)  + (W_EXPERIENCE × experience_match)

adjusted = raw × 100 + keyword_boost + cert_bonus − missing_penalty

final = compress(adjusted, floor=35, ceil=88)
```

### Score Bands

| Band | Range | Meaning |
|:-----|:------|:--------|
| 🟢 Strong | 80 – 88 | Excellent JD alignment |
| 🔵 Good | 70 – 79 | Solid match, minor gaps |
| 🟡 Moderate | 55 – 69 | Partial match |
| 🔴 Weak | 35 – 54 | Significant skill gaps |

---

## Project Structure

```
ResumeRanker-Pro/
│
├── app.py                          # Streamlit dashboard entry point
├── requirements.txt                # Python dependencies
│
├── scoring/
│   ├── scorer.py                   # Main ATS scoring pipeline
│   ├── weights.py                  # Role-aware dynamic weight calculator
│   └── explainer.py                # Human-readable score explanations
│
├── nlp/
│   ├── embeddings.py               # Sentence-transformer engine + cache
│   ├── semantic_matcher.py         # Pairwise skill similarity
│   ├── jd_analyzer.py              # JD parsing & role classification
│   └── experience_analyzer.py      # Years & seniority estimation
│
├── parsers/
│   ├── pdf_parser.py               # pdfplumber / PyMuPDF text extraction
│   ├── section_extractor.py        # Resume section segmentation
│   └── entity_extractor.py         # Degree & certification extraction
│
├── ranking/
│   └── ranker.py                   # Multi-resume ranking & leaderboard
│
├── suggestions/
│   └── improvement_engine.py       # Severity-ranked resume suggestions
│
├── bias_detection/
│   └── bias_detector.py            # PII & bias flag detection
│
├── utils/
│   ├── config.py                   # Central config, skill categories, blocklist
│   └── helpers.py                  # Text cleaning, clamping, utilities
│
└── tests/
    ├── test_scoring.py             # Scoring pipeline tests
    ├── test_parsers.py             # PDF & section parser tests
    ├── test_nlp.py                 # NLP module tests
    └── test_ranking.py             # Ranking module tests
```

---

## Installation

```bash
git clone https://github.com/mahi-ayub/ResumeRanker-Pro.git
cd ResumeRanker-Pro

python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

---

## Usage

### Run the Dashboard

```bash
streamlit run app.py
```

Then open **http://localhost:8501** — paste a job description, upload PDF resumes, and get scored results.

### Use Programmatically

```python
from scoring.scorer import ResumeScorer

scorer = ResumeScorer()
result = scorer.score(resume_text="...", jd_text="...")

print(f"ATS Score: {result.overall_score:.1f}")
print(f"Semantic:  {result.semantic_similarity:.2f}")
print(f"Required:  {result.required_skill_match:.2f}")
print(f"Tool:      {result.tool_match:.2f}")
print(f"Exp:       {result.experience_relevance:.2f}")
```

---

## Running Tests

```bash
pytest tests/ -v
```

**33 tests** across scoring, parsing, NLP, and ranking modules.

---

## Tech Stack

| Layer | Technologies |
|:------|:-------------|
| **NLP / ML** | PyTorch, sentence-transformers (`all-MiniLM-L6-v2`), spaCy, scikit-learn |
| **Parsing** | pdfplumber, PyMuPDF, python-dateutil |
| **UI** | Streamlit, Plotly, Pandas |
| **Testing** | pytest, pytest-cov |
| **Runtime** | Python 3.10+, CPU (CUDA optional) |

---

## License

[MIT](LICENSE)

---

<div align="center">

Built by **[Mahi Ayub](https://github.com/mahi-ayub)**

</div>
