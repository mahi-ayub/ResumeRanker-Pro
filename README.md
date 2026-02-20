<div align="center">

# 📄 ResumeRanker Pro

### AI-Powered Resume Evaluation with Semantic Matching & Explainable Scoring

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-33%20Passed-brightgreen?logo=pytest&logoColor=white)](#testing)

*A production-grade system that goes beyond keyword matching — using transformer embeddings, role-aware dynamic weighting, and fully explainable scoring to evaluate resumes against any job description.*

[Getting Started](#-getting-started) · [How It Works](#-how-it-works) · [Architecture](#-architecture) · [Scoring](#-scoring-formula) · [Deploy](#-deployment)

</div>

---

## ✨ Key Features

| Feature | Description |
|:--------|:------------|
| 🧠 **Semantic Skill Matching** | Transformer embeddings (`all-MiniLM-L6-v2`) + cosine similarity — catches synonyms, related skills, and contextual relevance that keyword matching misses |
| 🎯 **Role-Aware Dynamic Weighting** | Automatically detects the role type from the JD (backend, ML, frontend, devops, etc.) and shifts scoring weights to prioritize what matters most |
| 📊 **Explainable Scoring** | Full breakdown with reasoning: skill match %, experience alignment, project relevance, missing skills, strengths, weaknesses, and per-component math |
| 📄 **Resume Parsing** | Extracts structured sections (skills, experience, projects, education, certifications) from PDF and text files using pdfplumber + spaCy NER |
| 💼 **Experience Analysis** | Estimates total years, per-skill depth, seniority level (Junior → Principal), and professional vs. project-only usage via NLP + heuristics |
| 💡 **Improvement Suggestions** | Actionable feedback: add quantified achievements, strengthen partial matches, use stronger action verbs, add missing technologies |
| 🛡️ **Bias & Risk Detection** | Flags age indicators, gender markers, photos, marital status, SSN, and unnecessary PII to support fair hiring |
| 🏆 **Multi-Resume Ranking** | Upload multiple resumes, rank them against one JD, and get a visual leaderboard with per-candidate breakdowns |
| 🖥️ **Interactive Dashboard** | Clean Streamlit UI with Plotly gauge charts, radar charts, skill-match bar charts, and expandable detail panels |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- GPU optional (supports CUDA for faster inference)

### Installation

```bash
# Clone the repository
git clone https://github.com/mahi-ayub/ResumeRanker-Pro.git
cd ResumeRanker-Pro

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Launch

```bash
streamlit run app.py
```

The dashboard opens at `http://localhost:8501`. Upload a resume, paste a JD, and hit **Analyze**.

---

## 🧠 How It Works

```
┌─────────────┐     ┌──────────────┐     ┌──────────────────┐     ┌────────────────┐
│  Resume PDF  │────▶│  PDF Parser   │────▶│ Section Extractor │────▶│ Entity Extractor│
└─────────────┘     └──────────────┘     └──────────────────┘     └───────┬────────┘
                                                                          │
┌─────────────┐     ┌──────────────┐                                      │
│     JD       │────▶│ JD Analyzer   │──── Role Type + Required Skills ────┤
└─────────────┘     └──────────────┘                                      │
                                                                          ▼
                                          ┌──────────────────────────────────┐
                                          │        Scoring Pipeline          │
                                          │                                  │
                                          │  ┌─ Semantic Skill Matching ──┐  │
                                          │  ├─ Experience Analysis ──────┤  │
                                          │  ├─ Project Relevance ────────┤  │
                                          │  ├─ Education Alignment ──────┤  │
                                          │  ├─ Dynamic Weight Engine ────┤  │
                                          │  └─ Score Explainer ──────────┘  │
                                          └───────────────┬──────────────────┘
                                                          │
                                    ┌─────────────────────┼─────────────────────┐
                                    ▼                     ▼                     ▼
                            ┌──────────────┐   ┌──────────────────┐   ┌──────────────┐
                            │ Score: 75/100 │   │ Improvement Tips │   │ Bias Flags   │
                            └──────────────┘   └──────────────────┘   └──────────────┘
```

### Pipeline Steps

1. **Parse** — Extract raw text from PDF (pdfplumber primary, PyMuPDF fallback). Split into structured sections via regex header detection.
2. **Extract** — Identify skills, technologies, dates, certifications, and organizations using pattern matching + spaCy NER.
3. **Analyze JD** — Classify role type, extract required/preferred skills, detect years requirement and seniority level.
4. **Embed** — Encode resume skills and JD skills into 384-dim vectors using `all-MiniLM-L6-v2`.
5. **Match** — Compute pairwise cosine similarity matrix. Identify strong matches (≥0.70), partial matches (≥0.35), and missing skills.
6. **Score** — Apply role-aware dynamic weights, add certification bonuses, subtract missing-skill penalties.
7. **Explain** — Generate human-readable reasoning for every score component.

---

## 📐 Scoring Formula

| Component | Base Weight | Adapts By Role | Description |
|:----------|:----------:|:--------------:|:------------|
| **Skill Match** | 35% | ✅ | Cosine similarity between resume and JD skill embeddings |
| **Experience** | 25% | ✅ | Years of experience + seniority level fit |
| **Projects** | 20% | ✅ | Semantic relevance of project descriptions to JD |
| **Education** | 10% | ✅ | Degree and field alignment |
| **Cert Bonus** | +3%/cert | — | Bonus for each relevant certification (capped at +15) |
| **Missing Penalty** | −5%/skill | — | Deduction per critical missing skill (capped at −20) |

> **Dynamic Weighting**: A Backend Engineer JD shifts skill weight to **39%** and drops education to **7%**. An ML Engineer JD pushes skill match to **41%** and education to **9%**. Weights are derived automatically from JD classification.

```
Final Score = Σ(component_score × dynamic_weight) + cert_bonus − missing_penalty
            → clamped to [0, 100]
```

---

## 🏗️ Architecture

```
ResumeRanker-Pro/
│
├── parsers/                    # Resume ingestion
│   ├── pdf_parser.py           #   PDF → raw text (pdfplumber + PyMuPDF fallback)
│   ├── section_extractor.py    #   Raw text → structured ResumeData
│   └── entity_extractor.py     #   NER + pattern-based skill/cert extraction
│
├── nlp/                        # Core intelligence
│   ├── embeddings.py           #   Sentence-transformer embedding engine with caching
│   ├── semantic_matcher.py     #   Pairwise cosine similarity + match classification
│   ├── experience_analyzer.py  #   Years, seniority, skill depth estimation
│   └── jd_analyzer.py          #   Role classification + requirement extraction
│
├── scoring/                    # Evaluation engine
│   ├── weights.py              #   Role-aware dynamic weight calculator
│   ├── scorer.py               #   Full scoring pipeline orchestrator
│   └── explainer.py            #   Human-readable score explanations
│
├── ranking/                    # Multi-resume comparison
│   └── ranker.py               #   Score, sort, and generate leaderboard
│
├── suggestions/                # Feedback generation
│   └── improvement_engine.py   #   Actionable resume improvement suggestions
│
├── bias_detection/             # Fairness layer
│   └── bias_detector.py        #   PII and bias marker flagging
│
├── utils/                      # Shared infrastructure
│   ├── config.py               #   Scoring config, role profiles, skill taxonomy
│   └── helpers.py              #   Text cleaning, normalization, math utilities
│
├── tests/                      # Test suite (33 tests)
│   ├── test_parsers.py         #   Section + entity extraction tests
│   ├── test_nlp.py             #   JD analysis + role classification tests
│   ├── test_scoring.py         #   Weight computation + explainer tests
│   └── test_ranking.py         #   Bias detection tests
│
├── data/                       # Sample data
│   ├── sample_resumes/         #   Backend + ML engineer sample resumes
│   └── sample_jds/             #   Backend + ML engineer sample JDs
│
├── app.py                      # Streamlit dashboard (613 lines)
├── requirements.txt
└── README.md
```

---

## 🧪 Testing

```bash
# Run all 33 tests
python -m pytest tests/ -v

# Run specific module
python -m pytest tests/test_parsers.py -v
python -m pytest tests/test_scoring.py -v
```

| Module | Tests | Coverage |
|:-------|:-----:|:---------|
| Parsers | 12 | Section extraction, entity extraction, contact parsing, skill deduplication |
| NLP | 6 | JD analysis, role classification, years extraction, seniority detection |
| Scoring | 7 | Dynamic weights, weight normalization, role profiles, score explanations |
| Bias Detection | 8 | Age, gender, photo, marital, SSN, PII, recommendations |

---

## 🚢 Deployment

### Streamlit Cloud (Easiest)
1. Push to GitHub ✅
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect this repo → Set `app.py` as entry point → Deploy

### Hugging Face Spaces
1. Create a new Space (Streamlit SDK)
2. Push this repo's contents
3. Runs automatically

### Docker
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt && python -m spacy download en_core_web_sm
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501"]
```

---

## 🛠️ Tech Stack

| Layer | Technologies |
|:------|:-------------|
| **NLP / ML** | PyTorch, sentence-transformers, spaCy, scikit-learn, Transformers |
| **Parsing** | pdfplumber, PyMuPDF, regex, dateutil |
| **UI** | Streamlit, Plotly, Pandas |
| **Testing** | pytest, pytest-cov |
| **Infra** | Python 3.10+, CUDA (optional), pip |

---

## 📝 License

[MIT](LICENSE) — free for personal and commercial use.

---

<div align="center">

**Built by [Mahi Ayub](https://github.com/mahi-ayub)** · ⭐ Star this repo if you found it useful

</div>
