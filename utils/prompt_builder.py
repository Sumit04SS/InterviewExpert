def build_prompt(resume_text,
                 jd_text,
                 matched_skills,
                 missing_skills):

    prompt = f"""
You are an Expert Technical Interviewer.

Candidate Resume

{resume_text}

Job Description

{jd_text}

Matched Skills

{matched_skills}

Missing Skills

{missing_skills}

Generate:

10 Technical Questions

5 Project Questions

5 HR Questions

5 Scenario-Based Questions

Mention the reason behind each question.
"""

    return prompt