import re

SKILLS = [
    "Python",
    "Java",
    "C++",
    "SQL",
    "MongoDB",
    "React",
    "Node.js",
    "Flask",
    "Git",
    "Docker",
    "REST API",
    "Machine Learning",
    "Deep Learning",
    "TensorFlow",
    "PyTorch",
    "JavaScript",
    "HTML",
    "CSS"
]


def extract_skills(text):

    found = []

    for skill in SKILLS:
        if re.search(skill, text, re.IGNORECASE):
            found.append(skill)

    return found


def match_skills(resume_skills, jd_skills):

    matched = list(set(resume_skills) & set(jd_skills))

    missing = list(set(jd_skills) - set(resume_skills))

    score = int((len(matched) / len(jd_skills)) * 100) if jd_skills else 0

    return matched, missing, score