"""
AI Resume Scanner — Streamlit Dashboard
========================================
Production-grade resume evaluation with semantic matching,
role-aware scoring, and explainable results.

Run: streamlit run app.py
"""

import sys
import os
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from parsers.pdf_parser import PDFParser
from parsers.section_extractor import SectionExtractor
from parsers.entity_extractor import EntityExtractor
from nlp.jd_analyzer import JDAnalyzer
from scoring.scorer import ResumeScorer, ScoreResult
from scoring.explainer import ScoreExplainer
from scoring.weights import DynamicWeightCalculator
from ranking.ranker import ResumeRanker
from suggestions.improvement_engine import ImprovementEngine
from bias_detection.bias_detector import BiasDetector

# ----- Page Config -----
st.set_page_config(
    page_title="AI Resume Scanner",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----- Custom CSS -----
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .score-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        margin-bottom: 0.5rem;
    }
    .strength-badge {
        background: #d4edda;
        color: #155724;
        padding: 0.3rem 0.7rem;
        border-radius: 20px;
        font-size: 0.85rem;
        display: inline-block;
        margin: 0.2rem;
    }
    .weakness-badge {
        background: #f8d7da;
        color: #721c24;
        padding: 0.3rem 0.7rem;
        border-radius: 20px;
        font-size: 0.85rem;
        display: inline-block;
        margin: 0.2rem;
    }
</style>
""", unsafe_allow_html=True)


# ----- Cached Resources -----
@st.cache_resource(show_spinner="Loading AI models...")
def load_scorer():
    """Load and cache the scoring pipeline."""
    return ResumeScorer()


@st.cache_resource
def load_explainer():
    return ScoreExplainer()


@st.cache_resource
def load_improvement_engine():
    return ImprovementEngine()


@st.cache_resource
def load_bias_detector():
    return BiasDetector()


# ----- Helper Functions -----

def create_gauge_chart(score: float, title: str) -> go.Figure:
    """Create a gauge chart for a score."""
    if score >= 75:
        color = "#28a745"
    elif score >= 50:
        color = "#ffc107"
    else:
        color = "#dc3545"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={"text": title, "font": {"size": 16}},
        number={"suffix": "/100", "font": {"size": 28}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": color},
            "bgcolor": "white",
            "steps": [
                {"range": [0, 40], "color": "#ffebee"},
                {"range": [40, 65], "color": "#fff8e1"},
                {"range": [65, 100], "color": "#e8f5e9"},
            ],
            "threshold": {
                "line": {"color": "black", "width": 2},
                "thickness": 0.75,
                "value": score,
            },
        },
    ))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
    return fig


def create_radar_chart(result: ScoreResult) -> go.Figure:
    """Create a radar chart showing score breakdown."""
    categories = ["Skills", "Experience", "Projects", "Education"]
    values = [
        result.skill_match_score,
        result.experience_score,
        result.project_relevance_score,
        result.education_score,
    ]
    # Close the polygon
    categories.append(categories[0])
    values.append(values[0])

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill="toself",
        fillcolor="rgba(31, 119, 180, 0.2)",
        line=dict(color="#1f77b4", width=2),
        name="Score",
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=False,
        height=350,
        margin=dict(l=40, r=40, t=40, b=40),
    )
    return fig


def create_skill_match_chart(matched_skills: list, missing_skills: list) -> go.Figure:
    """Create a horizontal bar chart for skill matching."""
    labels = []
    scores = []
    colors = []

    for m in sorted(matched_skills, key=lambda x: x.get("similarity", 0), reverse=True)[:12]:
        labels.append(f"{m['jd_skill']} ↔ {m['resume_skill']}")
        scores.append(m["similarity"] * 100)
        colors.append("#28a745" if m["similarity"] >= 0.7 else "#ffc107")

    for m in missing_skills[:5]:
        labels.append(f"❌ {m['skill']}")
        scores.append(m.get("best_similarity", 0) * 100)
        colors.append("#dc3545")

    fig = go.Figure(go.Bar(
        x=scores,
        y=labels,
        orientation="h",
        marker_color=colors,
        text=[f"{s:.0f}%" for s in scores],
        textposition="auto",
    ))
    fig.update_layout(
        title="Skill Matching Detail",
        xaxis_title="Similarity (%)",
        height=max(300, len(labels) * 35),
        margin=dict(l=20, r=20, t=40, b=20),
        yaxis=dict(autorange="reversed"),
    )
    return fig


def create_ranking_chart(entries: list) -> go.Figure:
    """Create a bar chart for resume ranking."""
    names = [e.filename for e in entries]
    scores = [e.overall_score for e in entries]
    colors = [
        "#28a745" if s >= 70 else "#ffc107" if s >= 50 else "#dc3545"
        for s in scores
    ]

    fig = go.Figure(go.Bar(
        x=names,
        y=scores,
        marker_color=colors,
        text=[f"{s:.0f}" for s in scores],
        textposition="auto",
    ))
    fig.update_layout(
        title="Resume Ranking",
        yaxis_title="Overall Score",
        height=400,
        yaxis=dict(range=[0, 100]),
    )
    return fig


def load_sample_jd(name: str) -> str:
    """Load a sample JD from the data folder."""
    path = Path(__file__).parent / "data" / "sample_jds" / name
    if path.exists():
        return path.read_text(encoding="utf-8")
    return ""


# ----- Main App -----

def main():
    # Header
    st.markdown('<div class="main-header">📄 AI Resume Scanner</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">'
        'Semantic matching • Role-aware scoring • Explainable evaluation'
        '</div>',
        unsafe_allow_html=True,
    )

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        mode = st.radio("Mode", ["Single Resume", "Multi-Resume Ranking"], index=0)

        st.divider()
        st.header("📋 Job Description")

        jd_source = st.radio("JD Source", ["Paste Text", "Sample JD"], index=0)

        if jd_source == "Sample JD":
            sample_choice = st.selectbox("Select Sample JD", [
                "backend_engineer.txt",
                "ml_engineer.txt",
            ])
            jd_text = load_sample_jd(sample_choice)
            st.text_area("Job Description", jd_text, height=200, key="jd_display", disabled=True)
        else:
            jd_text = st.text_area(
                "Paste Job Description",
                height=300,
                placeholder="Paste the full job description here...",
                key="jd_input",
            )

        st.divider()
        st.header("🎚️ Advanced Options")
        show_bias = st.checkbox("Run Bias Detection", value=True)
        show_suggestions = st.checkbox("Generate Improvement Suggestions", value=True)

    # Main content area
    if mode == "Single Resume":
        single_resume_mode(jd_text, show_bias, show_suggestions)
    else:
        multi_resume_mode(jd_text, show_bias)


def single_resume_mode(jd_text: str, show_bias: bool, show_suggestions: bool):
    """Single resume evaluation mode."""

    st.header("📤 Upload Resume")
    uploaded = st.file_uploader(
        "Upload resume (PDF or TXT)",
        type=["pdf", "txt"],
        key="single_upload",
    )

    if uploaded and jd_text.strip():
        if st.button("🔍 Analyze Resume", type="primary", use_container_width=True):
            with st.spinner("🧠 Analyzing resume with AI..."):
                scorer = load_scorer()
                explainer = load_explainer()

                # Score
                result = scorer.score_resume_from_pdf(
                    file_bytes=uploaded.getvalue(),
                    jd_text=jd_text,
                    filename=uploaded.name,
                )

                # Store in session state
                st.session_state["result"] = result
                st.session_state["resume_bytes"] = uploaded.getvalue()
                st.session_state["resume_name"] = uploaded.name

        # Display results if available
        if "result" in st.session_state:
            result = st.session_state["result"]
            display_results(result, jd_text, show_bias, show_suggestions)

    elif not jd_text.strip():
        st.info("👈 Please enter a Job Description in the sidebar.")
    else:
        st.info("👆 Upload a resume to get started.")


def multi_resume_mode(jd_text: str, show_bias: bool):
    """Multi-resume ranking mode."""

    st.header("📤 Upload Multiple Resumes")
    uploaded_files = st.file_uploader(
        "Upload resumes (PDF or TXT)",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        key="multi_upload",
    )

    if uploaded_files and jd_text.strip():
        if st.button("🏆 Rank All Resumes", type="primary", use_container_width=True):
            with st.spinner(f"🧠 Scoring {len(uploaded_files)} resumes..."):
                scorer = load_scorer()
                ranker = ResumeRanker(scorer)

                resumes = [
                    {"filename": f.name, "bytes": f.getvalue()}
                    for f in uploaded_files
                ]

                entries = ranker.rank_resumes(resumes, jd_text)
                summary = ranker.get_ranking_summary(entries)

                st.session_state["ranking_entries"] = entries
                st.session_state["ranking_summary"] = summary

        if "ranking_entries" in st.session_state:
            display_ranking(
                st.session_state["ranking_entries"],
                st.session_state["ranking_summary"],
                jd_text,
                show_bias,
            )

    elif not jd_text.strip():
        st.info("👈 Please enter a Job Description in the sidebar.")
    else:
        st.info("👆 Upload multiple resumes to rank them.")


def display_results(result: ScoreResult, jd_text: str, show_bias: bool, show_suggestions: bool):
    """Display full evaluation results for a single resume."""

    # --- Overall Score ---
    st.divider()
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.plotly_chart(create_gauge_chart(result.overall_score, "Overall Match Score"), use_container_width=True)

    # --- Score Breakdown Cards ---
    st.subheader("📊 ATS Score Breakdown")

    # ATS component scores
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📄 Semantic Similarity", f"{result.semantic_similarity * 100:.0f}%",
                   help="Document-level cosine similarity (35% weight)")
    with col2:
        st.metric("🎯 Required Skill Match", f"{result.required_skill_match * 100:.0f}%",
                   help="Fraction of required JD skills found (45% weight)")
    with col3:
        st.metric("🔧 Tool Match", f"{result.tool_match * 100:.0f}%",
                   help="Exact technology keyword matches (10% weight)")
    with col4:
        st.metric("💼 Experience Relevance", f"{result.experience_relevance * 100:.0f}%",
                   help="Experience alignment with JD (10% weight)")

    # Legacy component scores
    st.caption("Detailed Component Scores")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🎯 Skill Match", f"{result.skill_match_score:.0f}%")
    with col2:
        st.metric("💼 Experience", f"{result.experience_score:.0f}%")
    with col3:
        st.metric("🔧 Projects", f"{result.project_relevance_score:.0f}%")
    with col4:
        st.metric("🎓 Education", f"{result.education_score:.0f}%")

    # Bonuses and penalties
    bonus_cols = st.columns(3)
    with bonus_cols[0]:
        if result.keyword_boost > 0:
            st.metric("⚡ Keyword Boost", f"+{result.keyword_boost:.1f}")
    with bonus_cols[1]:
        if result.certification_bonus > 0:
            st.metric("🎖 Cert Bonus", f"+{result.certification_bonus:.1f}")
    with bonus_cols[2]:
        if result.missing_skill_penalty > 0:
            st.metric("⚠️ Missing Penalty", f"-{result.missing_skill_penalty:.1f}")

    # --- Required Skills Status ---
    if result.matched_required_skills or result.missing_required_skills:
        st.subheader("📋 Required Skills Status")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**✅ Matched Required Skills:**")
            for s in result.matched_required_skills:
                st.markdown(f"- ✅ {s}")
            if not result.matched_required_skills:
                st.caption("None matched")
        with col2:
            st.markdown("**❌ Missing Required Skills:**")
            for s in result.missing_required_skills:
                st.markdown(f"- ❌ {s}")
            if not result.missing_required_skills:
                st.caption("All required skills present! 🎉")

    # --- Radar Chart ---
    st.subheader("🕸️ Score Profile")
    st.plotly_chart(create_radar_chart(result), use_container_width=True)

    # --- Skill Matching Detail ---
    st.subheader("🎯 Skill Matching Detail")
    if result.matched_skills or result.missing_skills:
        st.plotly_chart(
            create_skill_match_chart(result.matched_skills, result.missing_skills),
            use_container_width=True,
        )

    # --- Strengths & Weaknesses ---
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("✅ Strengths")
        for s in result.strengths:
            st.markdown(f'<span class="strength-badge">{s}</span>', unsafe_allow_html=True)
        if not result.strengths:
            st.caption("No significant strengths detected for this JD match.")

    with col2:
        st.subheader("⚠️ Areas for Improvement")
        for w in result.weaknesses:
            st.markdown(f'<span class="weakness-badge">{w}</span>', unsafe_allow_html=True)
        if not result.weaknesses:
            st.caption("No major weaknesses detected.")

    # --- Score Reasoning ---
    with st.expander("📐 Detailed Score Reasoning", expanded=False):
        for reason in result.score_reasoning:
            st.markdown(f"- {reason}")

        # Weight explanation
        st.markdown("---")
        calc = DynamicWeightCalculator()
        weight_explanation = calc.explain_weights(
            result.weights_used,
            result.jd_analysis.get("role_type", "default"),
        )
        st.markdown(weight_explanation)

    # --- Structured JSON Output ---
    with st.expander("📦 Structured ATS Output (JSON)", expanded=False):
        import json
        ats_json = {
            "final_score": result.overall_score,
            "semantic_similarity": round(result.semantic_similarity, 3),
            "required_skill_match": round(result.required_skill_match, 3),
            "tool_match": round(result.tool_match, 3),
            "experience_relevance": round(result.experience_relevance, 3),
            "matched_required_skills": result.matched_required_skills,
            "missing_required_skills": result.missing_required_skills,
            "boost_applied": result.keyword_boost,
        }
        st.json(ats_json)

    # --- Parsed Resume Data ---
    with st.expander("📋 Parsed Resume Sections", expanded=False):
        rd = result.resume_data
        if rd.get("skills"):
            st.markdown("**Skills:**")
            st.write(", ".join(rd["skills"][:30]))
        if rd.get("experience"):
            st.markdown("**Experience:**")
            for exp in rd["experience"]:
                title = exp.get("title", "Unknown")
                dates = exp.get("dates", "")
                st.markdown(f"- **{title}** {dates}")
        if rd.get("education"):
            st.markdown("**Education:**")
            for edu in rd["education"]:
                st.markdown(f"- {edu.get('degree', '')} {edu.get('field', '')} — {edu.get('institution', '')}")
        if rd.get("certifications"):
            st.markdown("**Certifications:**")
            for cert in rd["certifications"]:
                st.markdown(f"- {cert}")

    # --- JD Analysis ---
    with st.expander("📋 JD Analysis", expanded=False):
        jd = result.jd_analysis
        st.markdown(f"**Detected Role**: {jd.get('role_type', 'N/A').replace('_', ' ').title()}")
        st.markdown(f"**Confidence**: {jd.get('role_confidence', 0):.0%}")
        st.markdown(f"**Years Required**: {jd.get('years_required', 'N/A')}")
        st.markdown(f"**Seniority**: {jd.get('seniority', 'N/A').title()}")
        if jd.get("required_skills"):
            st.markdown(f"**Required Skills**: {', '.join(jd['required_skills'][:15])}")
        if jd.get("preferred_skills"):
            st.markdown(f"**Nice-to-Have**: {', '.join(jd['preferred_skills'][:10])}")

    # --- Improvement Suggestions ---
    if show_suggestions:
        st.divider()
        st.subheader("💡 Resume Improvement Suggestions")
        engine = load_improvement_engine()
        suggestions = engine.generate_suggestions(result, result.resume_data)

        # Priority actions
        if suggestions.get("priority_actions"):
            st.markdown("### 🎯 Top Priority Actions")
            for i, action in enumerate(suggestions["priority_actions"], 1):
                severity_icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(action.get("severity"), "⚪")
                st.markdown(f"**{i}. {severity_icon} {action.get('title', '')}**")
                st.markdown(action.get("description", ""))
                st.markdown("")

        # All suggestions in expanders
        for category, items in suggestions.items():
            if category == "priority_actions" or not isinstance(items, list) or not items:
                continue

            with st.expander(f"📝 {category.replace('_', ' ').title()} ({len(items)})"):
                for item in items:
                    if isinstance(item, dict):
                        severity_icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(item.get("severity"), "⚪")
                        st.markdown(f"**{severity_icon} {item.get('title', '')}**")
                        st.markdown(item.get("description", ""))
                        st.markdown("---")

    # --- Bias Detection ---
    if show_bias:
        st.divider()
        st.subheader("🛡️ Bias & Risk Detection")
        detector = load_bias_detector()

        resume_text = ""
        if "resume_bytes" in st.session_state:
            parser = PDFParser()
            name = st.session_state.get("resume_name", "")
            if name.endswith(".pdf"):
                resume_text = parser.extract_text(file_bytes=st.session_state["resume_bytes"])
            else:
                resume_text = st.session_state["resume_bytes"].decode("utf-8", errors="ignore")

        if resume_text:
            bias_result = detector.detect(resume_text)

            risk_colors = {"low": "🟢", "medium": "🟡", "high": "🔴"}
            st.markdown(
                f"**Risk Level**: {risk_colors.get(bias_result['risk_level'], '⚪')} "
                f"{bias_result['risk_level'].upper()} "
                f"({bias_result['total_flags']} flag{'s' if bias_result['total_flags'] != 1 else ''})"
            )

            if bias_result["flags"]:
                for flag in bias_result["flags"]:
                    severity_icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(flag["severity"], "⚪")
                    st.markdown(f"- {severity_icon} **{flag['type'].upper()}**: {flag['description']}")
                    st.caption(f"   ↳ {flag['detail']}")

            if bias_result["recommendations"]:
                st.markdown("**Recommendations:**")
                for rec in bias_result["recommendations"]:
                    st.markdown(f"- {rec}")

            if bias_result["risk_level"] == "low":
                st.success("✅ No significant bias markers detected. Resume looks clean.")


def display_ranking(entries: list, summary: dict, jd_text: str, show_bias: bool):
    """Display multi-resume ranking results."""

    st.divider()
    st.subheader("🏆 Resume Ranking Results")

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Resumes", summary["total_resumes"])
    with col2:
        st.metric("Top Score", f"{summary['score_range']['highest']:.0f}/100")
    with col3:
        st.metric("Average", f"{summary['score_range']['average']:.0f}/100")
    with col4:
        st.metric("Lowest", f"{summary['score_range']['lowest']:.0f}/100")

    # Ranking chart
    st.plotly_chart(create_ranking_chart(entries), use_container_width=True)

    # Leaderboard table
    st.subheader("📋 Leaderboard")
    df = pd.DataFrame([
        {
            "Rank": e.rank,
            "Resume": e.filename,
            "Overall": f"{e.overall_score:.0f}",
            "Skills": f"{e.skill_match:.0f}%",
            "Experience": f"{e.experience:.0f}%",
            "Projects": f"{e.projects:.0f}%",
            "Education": f"{e.education:.0f}%",
            "Missing Skills": e.missing_skills_count,
        }
        for e in entries
    ])
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Individual details
    st.subheader("📄 Individual Results")
    for entry in entries:
        medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(entry.rank, f"#{entry.rank}")
        with st.expander(f"{medal} {entry.filename} — Score: {entry.overall_score:.0f}/100"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Strengths:**")
                for s in entry.strengths:
                    st.markdown(f"- ✅ {s}")
            with col2:
                st.markdown("**Weaknesses:**")
                for w in entry.weaknesses:
                    st.markdown(f"- ⚠️ {w}")

            if entry.full_result and entry.full_result.score_reasoning:
                st.markdown("**Score Breakdown:**")
                for reason in entry.full_result.score_reasoning:
                    st.markdown(f"- {reason}")


# ----- Entry Point -----
if __name__ == "__main__":
    main()
