# ==============================
# Import Modules
# ==============================

from utils.pdf_reader import extract_text_from_pdf
from utils.jd_reader import read_jd
from utils.ats_matcher import extract_skills, match_skills
from utils.prompt_builder import build_prompt
from utils.gemini_service import generate_questions


# ==============================
# Read Resume
# ==============================

resume_path = "data/sample_resume.pdf"

resume_text = extract_text_from_pdf(resume_path)

print("✅ Resume Loaded")


# ==============================
# Read Job Description
# ==============================

jd_path = "data/sample_jd.txt"

jd_text = read_jd(jd_path)

print("✅ Job Description Loaded")


# ==============================
# Extract Skills
# ==============================

resume_skills = extract_skills(resume_text)
jd_skills = extract_skills(jd_text)

print("Resume Skills:", resume_skills)
print("JD Skills:", jd_skills)


# ==============================
# ATS Matching
# ==============================

matched_skills, missing_skills, ats_score = match_skills(
    resume_skills,
    jd_skills
)

print("Matched Skills:", matched_skills)
print("Missing Skills:", missing_skills)
print(f"ATS Score: {ats_score}%")


# ==============================
# Build Prompt
# ==============================

prompt = build_prompt(
    resume_text,
    jd_text,
    matched_skills,
    missing_skills
)


# ==============================
# Generate Questions
# ==============================

questions = generate_questions(prompt)


# ==============================
# Print Output
# ==============================

print("\n")
print("=" * 60)
print("Generated Interview Questions")
print("=" * 60)
print(questions)