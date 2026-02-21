"""Quick integration test: verify ATS-realistic scoring."""
import sys, os, json
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scoring.scorer import ResumeScorer

scorer = ResumeScorer()

# --- Load sample data ---
resume_text = open("data/sample_resumes/backend_engineer_resume.txt", encoding="utf-8").read()
jd_text = open("data/sample_jds/backend_engineer.txt", encoding="utf-8").read()

print("="*80)
print("ATS-REALISTIC SCORING TEST")
print("="*80)

result = scorer.score_resume(resume_text=resume_text, jd_text=jd_text)

# Print structured JSON output
ats_output = {
    "final_score": result.overall_score,
    "semantic_similarity": round(result.semantic_similarity, 3),
    "required_skill_match": round(result.required_skill_match, 3),
    "tool_match": round(result.tool_match, 3),
    "experience_relevance": round(result.experience_relevance, 3),
    "matched_required_skills": result.matched_required_skills,
    "missing_required_skills": result.missing_required_skills,
    "boost_applied": result.keyword_boost,
}
print(json.dumps(ats_output, indent=2))

print(f"\nComponent scores:")
print(f"  Skill Match:         {result.skill_match_score}%")
print(f"  Experience:          {result.experience_score}%")
print(f"  Projects:            {result.project_relevance_score}%")
print(f"  Education:           {result.education_score}%")
print(f"  Cert Bonus:          +{result.certification_bonus}")
print(f"  Missing Penalty:     -{result.missing_skill_penalty}")
print(f"  Keyword Boost:       +{result.keyword_boost}")

print(f"\nScore reasoning:")
for r in result.score_reasoning:
    print(f"  {r}")

print(f"\n{'='*80}")
score = result.overall_score
if score >= 80:
    band = "STRONG MATCH (80-90)"
elif score >= 65:
    band = "MODERATE MATCH (65-79)"
elif score >= 50:
    band = "PARTIAL MATCH (50-64)"
else:
    band = "WEAK MATCH (<50)"
print(f"FINAL: {score}/100 => {band}")
print(f"Cap at 95: {'YES' if score <= 95 else 'NO - BUG!'}")
print(f"{'='*80}")
