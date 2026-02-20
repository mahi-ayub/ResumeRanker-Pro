# AI Resume Scanner — Semantic Matching & Explainable Scoring

A production-grade AI-powered resume evaluation system that uses **transformer-based semantic matching**, **role-aware dynamic weighting**, and **explainable scoring** to evaluate resumes against job descriptions.

> Built to showcase strong NLP, ML, and system design skills.

---

## Features

- **PDF & Text Resume Parsing** — Extracts structured sections (skills, experience, education, projects, certifications) from uploaded resumes.
- **Semantic Skill Matching** — Uses `sentence-transformers` embeddings + cosine similarity (not keyword matching).
- **Experience-Level Extraction** — NLP + rule-based heuristics estimate years of experience, seniority level, and skill depth.
- **Role-Aware Dynamic Weighting** — Scoring weights automatically adapt based on the job description (e.g., backend roles weight frameworks higher).
- **Explainable Scoring** — Full breakdown: overall score, skill match, experience alignment, project relevance, missing skills, strengths, weaknesses, and reasoning.
- **Resume Improvement Suggestions** — Actionable, structured feedback (quantified achievements, missing technologies, impact statements).
- **Bias & Risk Detection** — Flags age, gender markers, photos, marital status, and unnecessary personal details.
- **Multi-Resume Ranking** — Rank multiple resumes against a single JD.
- **Streamlit Dashboard** — Clean UI with charts, alerts, and interactive exploration.

---

## Architecture

```
resume_scanner/
├── data/                  # Sample resumes, JDs, and test data
│   ├── sample_resumes/
│   └── sample_jds/
├── parsers/               # Resume parsing (PDF, text, section extraction)
│   ├── __init__.py
│   ├── pdf_parser.py      # PDF-to-text extraction (pdfplumber + PyMuPDF fallback)
│   ├── section_extractor.py  # Splits raw text into structured sections
│   └── entity_extractor.py   # Extracts skills, dates, years, certifications
├── nlp/                   # NLP and ML core
│   ├── __init__.py
│   ├── embeddings.py      # Sentence-transformer embedding engine
│   ├── semantic_matcher.py   # Cosine similarity matching logic
│   ├── experience_analyzer.py # Years-of-experience and seniority estimation
│   └── jd_analyzer.py     # JD parsing and role-type classification
├── scoring/               # Scoring engine
│   ├── __init__.py
│   ├── weights.py         # Role-aware dynamic weight computation
│   ├── scorer.py          # Main scoring pipeline
│   └── explainer.py       # Human-readable score explanations
├── ranking/               # Multi-resume ranking
│   ├── __init__.py
│   └── ranker.py          # Rank resumes against a JD
├── suggestions/           # Resume improvement engine
│   ├── __init__.py
│   └── improvement_engine.py  # Generates actionable feedback
├── bias_detection/        # Bias & risk detection
│   ├── __init__.py
│   └── bias_detector.py   # Flags PII / bias markers
├── utils/                 # Shared utilities
│   ├── __init__.py
│   ├── config.py          # Central configuration
│   └── helpers.py         # Common helper functions
├── tests/                 # Unit and integration tests
│   ├── __init__.py
│   ├── test_parsers.py
│   ├── test_nlp.py
│   ├── test_scoring.py
│   └── test_ranking.py
├── app.py                 # Streamlit dashboard entry point
├── requirements.txt
└── README.md
```

---

## Scoring Formula

| Component                | Base Weight | Description                                    |
|--------------------------|-------------|------------------------------------------------|
| Skill Match (Semantic)   | 0.35        | Cosine similarity of skill embeddings          |
| Experience Alignment     | 0.25        | Years + seniority fit vs JD requirements       |
| Project Relevance        | 0.20        | Semantic similarity of projects to JD          |
| Education Relevance      | 0.10        | Degree/field alignment                         |
| Missing Skills Penalty   | -0.05/skill | Deduction for each critical missing skill      |
| Certification Bonus      | +0.03/cert  | Bonus for relevant certifications              |

> Weights are **dynamically adjusted** based on the detected role type from the JD (e.g., ML Engineer → skill match weight increases, education weight increases).

**Final Score** = Σ(component × dynamic_weight) + bonuses − penalties, clamped to [0, 100].

---

## Quick Start

```bash
# 1. Clone and install
git clone <repo>
cd resume-scanner
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 2. Run the dashboard
streamlit run app.py
```

---

## Deployment

- **Streamlit Cloud**: Push to GitHub → Connect at share.streamlit.io
- **Hugging Face Spaces**: Add `app.py` as the entry point, include `requirements.txt`
- **Docker**: Build with the included architecture (add Dockerfile as needed)

---

## Resume Bullet Point

> *Built an AI Resume Scanner using sentence-transformers, spaCy, and scikit-learn that performs semantic skill matching (cosine similarity on transformer embeddings), role-aware dynamic scoring, and explainable evaluation — processing resumes in <2s with a modular, production-grade architecture deployed via Streamlit.*

---

## Interview Talking Points

1. **Why semantic matching over keyword matching?** — Keyword matching misses synonyms ("React.js" vs "React"), related skills ("PyTorch" ≈ "Deep Learning"), and contextual relevance. Transformer embeddings capture semantic meaning.
2. **How does role-aware weighting work?** — The JD is analyzed to classify the role type (backend, ML, frontend, etc.) and weights are dynamically shifted so that the most relevant dimensions matter most.
3. **Explain the scoring pipeline** — Resume is parsed → sections extracted → skills/experience/projects embedded → compared to JD embeddings → weighted score computed → explanation generated.
4. **How do you handle bias detection?** — Regex + NLP patterns flag PII/bias markers (age, gender, photo, marital status) so hiring teams can ensure fair evaluation.
5. **Scalability** — Modular design allows swapping embedding models, adding new parsers, or plugging in an LLM for suggestions without touching the scoring engine.

---

## License

MIT
