"""
Interview Expert
Evidence-Grounded Personalized Interview Question Generator

Purpose:
    Generate exactly 12 personalized interview questions from the
    final stable candidate evaluation JSON and validate that each
    question stays grounded in the available evidence.

Pipeline:
    candidate_evaluation_result.json
            |
            v
    Evidence ledger
            |
            v
    Evidence-grounded prompt
            |
            v
    Groq / openai/gpt-oss-120b
            |
            v
    12 questions
            |
            v
    Deterministic grounding validator
            |
            v
    generated_interview_questions.json

This module does NOT modify the stable matching engine.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv
from groq import Groq


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

EVALUATION_PATH = DATA_DIR / "candidate_evaluation_result.json"
OUTPUT_PATH = DATA_DIR / "generated_interview_questions.json"

load_dotenv(BASE_DIR / ".env")

GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

QUESTION_COUNT = 12

REQUIRED_CATEGORY_COUNTS = {
    "JD_SKILL": 3,
    "PROJECT": 3,
    "EXPERIENCE": 2,
    "SKILL_GAP": 2,
    "BEHAVIORAL": 2,
}

# Fixed difficulty distribution for the 12-question interview.
# 4 EASY + 5 MEDIUM + 3 HARD.
QUESTION_DIFFICULTY_PLAN = {
    1: "EASY",
    2: "MEDIUM",
    3: "HARD",
    4: "EASY",
    5: "MEDIUM",
    6: "HARD",
    7: "MEDIUM",
    8: "HARD",
    9: "EASY",
    10: "MEDIUM",
    11: "EASY",
    12: "MEDIUM",
}

REQUIRED_DIFFICULTY_COUNTS = {
    "EASY": 4,
    "MEDIUM": 5,
    "HARD": 3,
}


ALLOWED_DIFFICULTIES = {"EASY", "MEDIUM", "HARD"}
ALLOWED_GROUNDING = {"DIRECT", "HYPOTHETICAL"}

# ============================================================
# QUESTION OUTPUT SCHEMA
# ============================================================

QUESTION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "candidate": {"type": "string"},
        "job_title": {"type": "string"},
        "company": {"type": "string"},
        "questions": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "id": {"type": "integer"},
                    "question": {"type": "string"},
                    "category": {
                        "type": "string",
                        "enum": [
                            "JD_SKILL",
                            "PROJECT",
                            "EXPERIENCE",
                            "SKILL_GAP",
                            "BEHAVIORAL",
                        ],
                    },
                    "difficulty": {
                        "type": "string",
                        "enum": ["EASY", "MEDIUM", "HARD"],
                    },
                    "source": {"type": "string"},
                    "reason": {"type": "string"},
                    "grounding": {
                        "type": "string",
                        "enum": ["DIRECT", "HYPOTHETICAL"],
                    },
                },
                "required": [
                    "id",
                    "question",
                    "category",
                    "difficulty",
                    "source",
                    "reason",
                    "grounding",
                ],
            },
        },
    },
    "required": ["candidate", "job_title", "company", "questions"],
}


# ============================================================
# GENERAL HELPERS
# ============================================================

def clean_text(value: Any) -> str:
    """Convert a value into clean single-line text."""
    if value is None:
        return ""

    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)

    text = str(value)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def first_nonempty(*values: Any) -> str:
    """Return the first non-empty value."""
    for value in values:
        text = clean_text(value)
        if text:
            return text
    return ""


def normalize_for_match(text: str) -> str:
    """
    Normalize text for conservative lexical grounding checks.

    This is deliberately simple. It is not intended to understand
    semantic equivalence.
    """
    text = clean_text(text).lower()
    text = text.replace("–", "-").replace("—", "-")
    text = re.sub(r"[^a-z0-9+#./& -]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def contains_term(text: str, term: str) -> bool:
    """Case-insensitive normalized containment check."""
    if not text or not term:
        return False

    text_n = normalize_for_match(text)
    term_n = normalize_for_match(term)

    return term_n in text_n


def deduplicate_strings(items: List[str]) -> List[str]:
    """Preserve order while removing duplicate strings."""
    result = []
    seen = set()

    for item in items:
        value = clean_text(item)
        if not value:
            continue

        key = normalize_for_match(value)

        if key not in seen:
            seen.add(key)
            result.append(value)

    return result


# ============================================================
# LOAD EVALUATION
# ============================================================

def load_candidate_evaluation(path: Path = EVALUATION_PATH) -> Dict[str, Any]:
    """Load the final stable matching-engine output."""
    if not path.exists():
        raise FileNotFoundError(
            f"Evaluation file not found:\n{path}\n\n"
            "Run the final stable matching engine first."
        )

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError("Candidate evaluation JSON must contain an object.")

    return data


# ============================================================
# EVIDENCE LEDGER
# ============================================================

def extract_candidate_info(evaluation: Dict[str, Any]) -> Dict[str, str]:
    """Extract candidate/JD metadata from several compatible layouts."""

    candidate_obj = evaluation.get("candidate", {})
    if not isinstance(candidate_obj, dict):
        candidate_obj = {}

    candidate = first_nonempty(
        candidate_obj.get("name"),
        evaluation.get("candidate_name"),
        evaluation.get("name"),
    )

    jd = evaluation.get("job", {})
    if not isinstance(jd, dict):
        jd = evaluation.get("jd", {})
    if not isinstance(jd, dict):
        jd = {}

    job_title = first_nonempty(
        jd.get("job_title"),
        jd.get("title"),
        evaluation.get("job_title"),
    )

    company = first_nonempty(
        jd.get("company"),
        jd.get("company_name"),
        evaluation.get("company"),
    )

    return {
        "candidate": candidate or "Candidate",
        "job_title": job_title or "Software Developer",
        "company": company or "Company",
    }


def extract_required_skills(evaluation: Dict[str, Any]) -> List[str]:
    """Extract JD-required skills."""
    jd = evaluation.get("job", {})
    if not isinstance(jd, dict):
        jd = evaluation.get("jd", {})
    if not isinstance(jd, dict):
        jd = {}

    candidates = [
        jd.get("required_skills"),
        jd.get("skills"),
        evaluation.get("required_skills"),
        evaluation.get("jd_required_skills"),
    ]

    for value in candidates:
        if isinstance(value, list):
            skills = deduplicate_strings([clean_text(x) for x in value])
            if skills:
                return skills

        if isinstance(value, dict):
            skills = deduplicate_strings(list(value.keys()))
            if skills:
                return skills

        if isinstance(value, str):
            skills = deduplicate_strings(
                re.split(r"[,;\n|]", value)
            )
            if skills:
                return skills

    return []


def extract_skill_match_data(
    evaluation: Dict[str, Any],
) -> Tuple[List[str], List[str]]:
    """Extract matched and missing required skills from the stable matcher."""

    skill_section = evaluation.get("skills", {})
    if not isinstance(skill_section, dict):
        skill_section = {}

    matched = skill_section.get("matched_required_skills", [])
    missing = skill_section.get("missing_required_skills", [])

    if not isinstance(matched, list):
        matched = []
    if not isinstance(missing, list):
        missing = []

    return (
        deduplicate_strings([clean_text(x) for x in matched]),
        deduplicate_strings([clean_text(x) for x in missing]),
    )


def extract_projects(evaluation: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract project-local evidence without mixing projects."""

    project_section = evaluation.get("projects", [])

    if isinstance(project_section, dict):
        project_section = (
            project_section.get("items")
            or project_section.get("projects")
            or []
        )

    if not isinstance(project_section, list):
        return []

    projects = []

    for index, project in enumerate(project_section, start=1):
        if not isinstance(project, dict):
            continue

        title = first_nonempty(
            project.get("title"),
            project.get("project_name"),
            project.get("name"),
        )

        if not title:
            continue

        technologies = project.get("technologies", [])
        if isinstance(technologies, str):
            technologies = re.split(r"[,;|]", technologies)

        technologies = deduplicate_strings(
            [clean_text(x) for x in technologies]
        ) if isinstance(technologies, list) else []

        description = first_nonempty(
            project.get("description"),
            project.get("body"),
            project.get("evidence"),
        )

        projects.append({
            "index": index,
            "title": title,
            "technologies": technologies,
            "description": description,
            "score": project.get("score"),
        })

    return projects


def extract_experience(evaluation: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract experience entries with their own local evidence."""

    experience = evaluation.get("experience", {})
    if not isinstance(experience, dict):
        return []

    roles = (
        experience.get("technical_roles")
        or experience.get("roles")
        or []
    )

    if not isinstance(roles, list):
        return []

    result = []

    for index, role in enumerate(roles, start=1):
        if not isinstance(role, dict):
            continue

        role_title = first_nonempty(
            role.get("role_title"),
            role.get("role"),
            role.get("title"),
        )

        if not role_title:
            continue

        company = first_nonempty(
            role.get("company"),
            role.get("organization"),
        )

        date_text = first_nonempty(
            role.get("date_text"),
            role.get("dates"),
            role.get("period"),
        )

        body = first_nonempty(
            role.get("body"),
            role.get("description"),
            role.get("evidence"),
        )

        result.append({
            "index": index,
            "role_title": role_title,
            "company": company,
            "date_text": date_text,
            "body": body,
        })

    return result


def extract_behavioral_evidence(
    evaluation: Dict[str, Any],
) -> List[str]:
    """Extract explicit behavioral evidence from the stable evaluation."""

    evidence = []
    skills = evaluation.get("skills", {})
    if isinstance(skills, dict):
        soft = skills.get("soft_skill_evidence", {})
        if isinstance(soft, dict):
            results = soft.get("results", {})
            if isinstance(results, dict):
                for skill_name, item in results.items():
                    if not isinstance(item, dict):
                        continue
                    for ev in item.get("evidence", []):
                        if isinstance(ev, dict):
                            source = clean_text(ev.get("source"))
                            context = clean_text(ev.get("context"))
                            matched = clean_text(ev.get("matched_text"))
                            text = " | ".join(x for x in [source, matched, context] if x)
                            if text:
                                evidence.append(text)
                        elif isinstance(ev, str):
                            evidence.append(ev)

    # Stable matcher may also expose leadership separately.
    leadership = evaluation.get("leadership")
    if isinstance(leadership, list):
        for item in leadership:
            if isinstance(item, str):
                evidence.append(item)
            elif isinstance(item, dict):
                text = first_nonempty(item.get("evidence"), item.get("role"), item.get("title"), item.get("context"))
                if text:
                    evidence.append(text)

    # Some stable evaluation files store leadership/volunteering only inside
    # resume.sections. Use those explicit entries as behavioral evidence.
    resume = evaluation.get("resume", {})
    if isinstance(resume, dict):
        sections = resume.get("sections", {})
        if isinstance(sections, dict):
            for section_name in [
                "Leadership & Volunteering",
                "Leadership and Volunteering",
                "Leadership",
                "Achievements",
            ]:
                section_items = sections.get(section_name, [])
                if isinstance(section_items, list):
                    for item in section_items:
                        text = clean_text(item)
                        if text:
                            evidence.append(f"{section_name}: {text}")
                elif isinstance(section_items, str) and section_items.strip():
                    evidence.append(f"{section_name}: {section_items.strip()}")

    # Experience-role fallback:
    # Some stable evaluation JSON files contain leadership information only in
    # the resume text, while the structured "soft_skill_evidence" section is
    # empty. In that case, use documented role responsibilities as behavioral
    # evidence. This does NOT invent behavior; it reuses text already extracted
    # from the candidate's experience.
    experience = evaluation.get("experience", {})
    if isinstance(experience, dict):
        roles = experience.get("technical_roles", [])
        if isinstance(roles, list):
            for role in roles:
                if not isinstance(role, dict):
                    continue

                title = clean_text(role.get("role_title"))
                company = clean_text(role.get("company"))
                body = clean_text(
                    first_nonempty(
                        role.get("body"),
                        role.get("description"),
                        role.get("evidence"),
                    )
                )

                if title and re.search(
                    r"\b(team lead|lead|head|coordinator|co-head|vice president|president|"
                    r"manager|managed|leader|leadership|volunteer)\b",
                    title,
                    re.I,
                ):
                    label = "Experience"
                    if company:
                        label += f" | {company}"
                    evidence.append(f"{label}: {title}")
                    if body:
                        evidence.append(f"{label}: {body}")

    # Project/achievement fallback:
    # If no leadership-specific evidence exists, documented achievements or
    # project ownership can still support a behavior question such as
    # "How did you approach...?" without claiming undocumented conflict,
    # teamwork, deadlines, or outcomes.
    if not evidence:
        project_section = evaluation.get("projects", {})
        if isinstance(project_section, dict):
            project_items = project_section.get("projects") or project_section.get("items") or []
        else:
            project_items = project_section

        if isinstance(project_items, list):
            for project in project_items:
                if not isinstance(project, dict):
                    continue
                title = first_nonempty(
                    project.get("title"),
                    project.get("project_name"),
                    project.get("name"),
                )
                body = first_nonempty(
                    project.get("description"),
                    project.get("body"),
                    project.get("evidence"),
                )
                if title and body:
                    evidence.append(f"Project {title}: {body}")

    return deduplicate_strings(evidence)


def build_evidence_ledger(
    evaluation: Dict[str, Any],
) -> Dict[str, Any]:
    """Build a compact evidence ledger for the LLM."""

    info = extract_candidate_info(evaluation)

    required_skills = extract_required_skills(evaluation)
    matched_skills, missing_skills = extract_skill_match_data(evaluation)

    # If the stable matcher did not expose required skills in a nested
    # field, derive them conservatively from matched + missing.
    if not required_skills:
        required_skills = deduplicate_strings(
            matched_skills + missing_skills
        )

    projects = extract_projects(evaluation)
    experience = extract_experience(evaluation)
    behavioral = extract_behavioral_evidence(evaluation)

    ledger = {
        "candidate": info["candidate"],
        "job_title": info["job_title"],
        "company": info["company"],
        "required_skills": required_skills,
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "projects": projects,
        "experience": experience,
        "behavioral_evidence": behavioral,
    }

    return ledger


# ============================================================
# PROMPT BUILDING
# ============================================================

def format_ledger_for_prompt(ledger: Dict[str, Any]) -> str:
    """Create a readable evidence block for the model."""

    lines = []

    lines.append(f"Candidate: {ledger['candidate']}")
    lines.append(f"Job Title: {ledger['job_title']}")
    lines.append(f"Company: {ledger['company']}")

    lines.append("\nREQUIRED JD SKILLS")
    for skill in ledger["required_skills"]:
        status = (
            "MATCHED"
            if skill in ledger["matched_skills"]
            else "MISSING"
        )
        lines.append(f"- {skill} [{status}]")

    lines.append("\nPROJECT EVIDENCE")
    for project in ledger["projects"]:
        lines.append(
            f"- Project {project['index']}: {project['title']}"
        )

        if project["technologies"]:
            lines.append(
                "  Technologies: "
                + ", ".join(project["technologies"])
            )

        if project["description"]:
            lines.append(
                "  Documented evidence: "
                + project["description"]
            )

    lines.append("\nEXPERIENCE EVIDENCE")
    for role in ledger["experience"]:
        label = role["role_title"]

        if role["company"]:
            label += f" | {role['company']}"

        if role["date_text"]:
            label += f" | {role['date_text']}"

        lines.append(f"- Experience {role['index']}: {label}")

        if role["body"]:
            lines.append(f"  Documented evidence: {role['body']}")

    lines.append("\nBEHAVIORAL EVIDENCE")
    if ledger["behavioral_evidence"]:
        for item in ledger["behavioral_evidence"]:
            lines.append(f"- {item}")
    else:
        lines.append("- No explicit behavioral evidence available.")

    return "\n".join(lines)


def build_question_generation_prompt(
    ledger: Dict[str, Any],
) -> str:
    """Build a strict, evidence-grounded prompt with a deterministic slot plan."""

    evidence_block = format_ledger_for_prompt(ledger)

    matched = ledger["matched_skills"]
    missing = ledger["missing_skills"]
    projects = ledger["projects"]
    experience = ledger["experience"]
    behavioral = ledger["behavioral_evidence"]

    # Build explicit slots so the model does not decide the distribution itself.
    if len(matched) < 3:
        raise RuntimeError(
            "At least 3 matched JD skills are required to generate the configured 12-question plan."
    )

    if len(projects) < 3:
        raise RuntimeError(
            "At least 3 projects are required to generate the configured 12-question plan."
    )

    if len(experience) < 2:
        raise RuntimeError(
            "At least 2 experience entries are required to generate the configured 12-question plan."
    )

    if len(missing) < 2:
        raise RuntimeError(
            "At least 2 missing JD skills are required to generate the configured 12-question plan."
    )

    if not behavioral:
        raise RuntimeError(
            "Explicit behavioral evidence is required to generate behavioral questions."
    )

    skill_slots = matched[:3]
    project_slots = projects[:3]
    experience_slots = experience[:2]
    gap_slots = missing[:2]
    behavior_slots = [behavioral[0], behavioral[0]] if len(behavioral) == 1 else behavioral[:2]

    plan_lines = []
    for i, skill in enumerate(skill_slots, 1):
        plan_lines.append(f"{i}. JD_SKILL | source={skill}")
    for i, project in enumerate(project_slots, 4):
        plan_lines.append(f"{i}. PROJECT | source=Project {project['index']} - {project['title']}")
    for i, role in enumerate(experience_slots, 7):
        plan_lines.append(f"{i}. EXPERIENCE | source=Experience {role['index']} - {role['role_title']}")
    for i, skill in enumerate(gap_slots, 9):
        plan_lines.append(f"{i}. SKILL_GAP | source={skill}")
    for i, evidence in enumerate(behavior_slots, 11):
        plan_lines.append(f"{i}. BEHAVIORAL | source evidence: {evidence[:220]}")

    return f"""
You are an expert technical interviewer. Generate personalized interview questions ONLY from the evidence ledger.

========================
EVIDENCE LEDGER
========================
{evidence_block}

========================
FIXED QUESTION PLAN
========================
You MUST produce exactly one question for each of these 12 slots, in this exact order.
Do not add, remove, merge, or reorder slots.

{chr(10).join(plan_lines)}

========================
CRITICAL GROUNDING RULES
========================
1. Use ONLY required JD skills shown in the ledger.
2. JD_SKILL questions MUST use one of the three MATCHED required skills named in the slot. Do NOT use any other technology or resume skill as the primary source. Never substitute AWS, MERN, Kubernetes, security testing, Bedrock, Lambda, React, Node.js, or another non-slot skill.
3. PROJECT questions MUST mention the exact project named in the slot and use only that project's technologies/evidence.
4. EXPERIENCE questions MUST stay inside the exact role named in the slot.
5. SKILL_GAP questions MUST use only the missing skill in the slot and MUST be hypothetical. Never imply previous use.
6. BEHAVIORAL questions MUST be based only on the supplied explicit behavioral evidence. Do not invent conflict, deadlines, stakeholders, failures, team size, or outcomes.
7. Never transfer technologies or responsibilities between projects or roles.
8. The source field must exactly identify the slot source. Do not invent source labels such as "Skill Gap - Containerization" unless that exact skill is the slot.
9. Never invent implementation details. If an implementation detail is not documented, ask what the candidate would do instead.
10. Do not invent numbers, users, architecture, databases, algorithms, tools, frameworks, or achievements.
11. Avoid generic questions when the supplied evidence allows a specific question.
12. Do not repeat the same underlying question.

========================
DIFFICULTY RULES
========================
Use the difficulty exactly as follows:
1 = EASY
2 = MEDIUM
3 = HARD
4 = EASY
5 = MEDIUM
6 = HARD
7 = MEDIUM
8 = HARD
9 = EASY
10 = MEDIUM
11 = EASY
12 = MEDIUM

Difficulty meaning:
- EASY: fundamental concept, definition, purpose, or straightforward explanation.
- MEDIUM: practical application, implementation, or reasoning based on documented evidence.
- HARD: deeper troubleshooting, trade-offs, optimization, architecture, or scenario-based reasoning.
Do not label a question HARD merely because it is long. The wording must actually require deeper reasoning.

========================
OUTPUT
========================
Return exactly 12 JSON questions. Every question must contain:
- id
- question
- category
- difficulty
- source
- reason
- grounding

Use DIRECT only when the question refers to something explicitly documented.
Use HYPOTHETICAL when asking about an undocumented extension/design or a missing skill.

Before returning each question, verify that its source and wording are supported by the ledger.
Return ONLY the JSON object.
""".strip()


# ============================================================
# GROQ GENERATION
# ============================================================

def create_groq_client() -> Groq:
    """Create the Groq client from GROQ_API_KEY."""
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY was not found in the .env file."
        )

    return Groq(api_key=api_key)


def _deterministic_fallback_questions(ledger: Dict[str, Any]) -> Dict[str, Any]:
    """Build guaranteed-grounded questions if Groq output fails validation."""
    questions = []

    for qid, skill in enumerate(ledger["matched_skills"][:3], 1):
        questions.append({
            "id": qid,
            "question": f"For the {skill} skill listed in your resume, explain how you have used {skill} and describe one practical problem you solved with it.",
            "category": "JD_SKILL",
            "difficulty": QUESTION_DIFFICULTY_PLAN[qid],
            "source": skill,
            "reason": f"{skill} is a matched required skill for the target job.",
            "grounding": "DIRECT",
        })

    for qid, project in enumerate(ledger["projects"][:3], 4):
        tech = project["technologies"][0] if project["technologies"] else None
        if tech:
            question = f"In {project['title']}, how did you use {tech}, and what was your responsibility in that project?"
        else:
            question = f"In {project['title']}, what was your main responsibility and how did you contribute to the documented work?"
        questions.append({
            "id": qid,
            "question": question,
            "category": "PROJECT",
            "difficulty": QUESTION_DIFFICULTY_PLAN[qid],
            "source": f"Project {project['index']} - {project['title']}",
            "reason": "The question is grounded in the specific project and its documented evidence.",
            "grounding": "DIRECT",
        })

    for qid, role in enumerate(ledger["experience"][:2], 7):
        role_name = role["role_title"]
        company = role["company"]
        label = f"{role_name} at {company}" if company else role_name
        if role["body"]:
            question = f"In your role as {label}, what was one important responsibility you handled, and how did you approach it?"
        else:
            question = f"In your role as {label}, what were your main responsibilities?"
        questions.append({
            "id": qid,
            "question": question,
            "category": "EXPERIENCE",
            "difficulty": QUESTION_DIFFICULTY_PLAN[qid],
            "source": f"Experience {role['index']} - {role_name}",
            "reason": "The question refers directly to a documented experience role.",
            "grounding": "DIRECT",
        })

    for qid, skill in enumerate(ledger["missing_skills"][:2], 9):
        questions.append({
            "id": qid,
            "question": (
                f"If you were asked to use {skill} in this role, what would you first learn about it "
                f"and how would you apply it to a suitable software development task?"
            ),
            "category": "SKILL_GAP",
            "difficulty": QUESTION_DIFFICULTY_PLAN[qid],
            "source": skill,
            "reason": f"{skill} is a required JD skill that was not matched in the candidate evidence.",
            "grounding": "HYPOTHETICAL",
        })

    behavioral = ledger["behavioral_evidence"]
    behavior_slots = [behavioral[0], behavioral[0]]
    for qid, evidence in enumerate(behavior_slots, 11):
        if qid == 11:
            question = f"Based on the documented evidence '{evidence}', what responsibility did you take on and how did you approach it?"
        else:
            question = f"Based on the documented evidence '{evidence}', what did you learn from taking on that responsibility, and how would you apply that learning in a software team?"
        questions.append({
            "id": qid,
            "question": question,
            "category": "BEHAVIORAL",
            "difficulty": QUESTION_DIFFICULTY_PLAN[qid],
            "source": evidence,
            "reason": "The question is based only on explicit behavioral or leadership evidence in the evaluation.",
            "grounding": "DIRECT",
        })

    return {
        "candidate": ledger["candidate"],
        "job_title": ledger["job_title"],
        "company": ledger["company"],
        "questions": questions,
    }


def generate_personalized_questions(ledger: Dict[str, Any]) -> Dict[str, Any]:
    """Generate questions with Groq, repair invalid grounding, then use a safe fallback."""
    client = create_groq_client()
    prompt = build_question_generation_prompt(ledger)

    def call(user_prompt: str) -> Dict[str, Any]:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            temperature=0.0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You generate evidence-grounded interview questions. "
                        "Complete every numbered slot exactly. Never omit a slot. "
                        "Copy the exact project title into PROJECT questions and the "
                        "exact role title or company into EXPERIENCE questions. "
                        "For SKILL_GAP questions use explicit hypothetical wording."
                    ),
                },
                {"role": "user", "content": user_prompt},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "personalized_interview_questions",
                    "schema": QUESTION_SCHEMA,
                    "strict": True,
                },
            },
        )
        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("Groq returned an empty response.")
        try:
            return json.loads(content)
        except json.JSONDecodeError as exc:
            raise RuntimeError("Groq returned invalid JSON.") from exc

    result = call(prompt)
    passed, errors = validate_generated_questions(result, ledger)
    if passed:
        return result

    repair_prompt = prompt + "\n\nIMPORTANT: The previous output failed validation. Fix EVERY issue below and return the COMPLETE 12-question JSON object.\n"
    repair_prompt += "\n".join(f"- {error}" for error in errors)
    repair_prompt += "\nFor PROJECT slots, the question MUST contain the exact full project title from the slot. For EXPERIENCE slots, the question MUST contain the exact role title or company. For SKILL_GAP slots, begin with wording such as 'If you were asked to use...' or 'How would you...'\n"

    repaired = call(repair_prompt)
    passed, repair_errors = validate_generated_questions(repaired, ledger)
    if passed:
        return repaired

    print("\nGroq output still failed deterministic grounding validation.")
    print("Using deterministic evidence-grounded fallback questions.")
    fallback = _deterministic_fallback_questions(ledger)
    passed, fallback_errors = validate_generated_questions(fallback, ledger)
    if not passed:
        raise RuntimeError("Deterministic fallback failed validation: " + " | ".join(fallback_errors))
    return fallback


# ============================================================
# DETERMINISTIC GROUNDING VALIDATOR
# ============================================================

def validate_basic_structure(
    result: Dict[str, Any],
) -> List[str]:
    """Validate count, categories, IDs, and required fields."""

    errors = []

    if not isinstance(result, dict):
        return ["Output must be a JSON object."]

    questions = result.get("questions")

    if not isinstance(questions, list):
        return ["'questions' must be a list."]

    if len(questions) != QUESTION_COUNT:
        errors.append(
            f"Expected exactly {QUESTION_COUNT} questions; "
            f"received {len(questions)}."
        )

    ids = []

    for index, item in enumerate(questions, start=1):
        if not isinstance(item, dict):
            errors.append(f"Question {index} is not an object.")
            continue

        for field in [
            "id",
            "question",
            "category",
            "difficulty",
            "source",
            "reason",
            "grounding",
        ]:
            if not clean_text(item.get(field)):
                errors.append(
                    f"Question {index} is missing '{field}'."
                )

        item_id = item.get("id")
        if isinstance(item_id, int):
            ids.append(item_id)

        if item.get("difficulty") not in ALLOWED_DIFFICULTIES:
            errors.append(
                f"Question {index} has invalid difficulty."
            )
        expected_difficulty = QUESTION_DIFFICULTY_PLAN.get(item_id)
        if expected_difficulty and item.get("difficulty") != expected_difficulty:
            errors.append(
                f"Question {index}: expected difficulty "
                f"{expected_difficulty}, received {item.get('difficulty')}."
            )

        if item.get("grounding") not in ALLOWED_GROUNDING:
            errors.append(
                f"Question {index} has invalid grounding."
            )

    if sorted(ids) != list(range(1, QUESTION_COUNT + 1)):
        errors.append(
            "Question IDs must be exactly 1 through 12."
        )

    counts = {}

    for item in questions:
        if isinstance(item, dict):
            category = item.get("category")
            counts[category] = counts.get(category, 0) + 1

    for category, expected in REQUIRED_CATEGORY_COUNTS.items():
        actual = counts.get(category, 0)
        if actual != expected:
            errors.append(
                f"{category}: expected {expected}, received {actual}."
            )

    difficulty_counts = {}
    for item in questions:
        if isinstance(item, dict):
            difficulty = item.get("difficulty")
            difficulty_counts[difficulty] = difficulty_counts.get(difficulty, 0) + 1

    for difficulty, expected in REQUIRED_DIFFICULTY_COUNTS.items():
        actual = difficulty_counts.get(difficulty, 0)
        if actual != expected:
            errors.append(
                f"{difficulty}: expected {expected}, received {actual}."
            )

    return errors


def find_project(
    source: str,
    projects: List[Dict[str, Any]],
) -> Dict[str, Any] | None:
    """Find the project referred to by source."""
    for project in projects:
        if contains_term(source, project["title"]):
            return project
    return None


def find_experience(
    source: str,
    experience: List[Dict[str, Any]],
) -> Dict[str, Any] | None:
    """Find the experience role referred to by source."""
    for role in experience:
        if contains_term(source, role["role_title"]):
            return role

        if role["company"] and contains_term(source, role["company"]):
            return role

    return None


def validate_question_grounding(
    question_item: Dict[str, Any],
    ledger: Dict[str, Any],
) -> List[str]:
    """
    Conservative lexical grounding checks.

    This validator intentionally rejects questionable output rather than
    pretending that lexical matching proves semantic correctness.
    """

    errors = []

    question = clean_text(question_item.get("question"))
    source = clean_text(question_item.get("source"))
    category = question_item.get("category")
    grounding = question_item.get("grounding")

    if not question:
        return ["Question text is empty."]

    if not source:
        errors.append("Source is empty.")

    if category == "JD_SKILL":
        matched = ledger["matched_skills"]

        if not any(contains_term(source, skill) for skill in matched):
            errors.append(
                "JD_SKILL source must identify a matched required skill."
            )

        if not any(
            contains_term(question, skill)
            for skill in matched
            if contains_term(source, skill)
        ):
            errors.append(
                "JD_SKILL question should explicitly reference its matched skill."
            )

    elif category == "PROJECT":
        project = find_project(source, ledger["projects"])

        if project is None:
            errors.append(
                "PROJECT source does not identify a known project."
            )
        else:
            if not contains_term(question, project["title"]):
                errors.append(
                    f"PROJECT question must mention project "
                    f"'{project['title']}'."
                )

            if grounding == "DIRECT":
                evidence_terms = [project["title"]] + project["technologies"]

                if not any(
                    contains_term(question, term)
                    for term in evidence_terms
                    if term
                ):
                    errors.append(
                        "DIRECT PROJECT question must reference "
                        "the project's documented evidence."
                    )

    elif category == "EXPERIENCE":
        role = find_experience(source, ledger["experience"])

        if role is None:
            errors.append(
                "EXPERIENCE source does not identify a known role."
            )
        else:
            if not (
                contains_term(question, role["role_title"])
                or (
                    role["company"]
                    and contains_term(question, role["company"])
                )
            ):
                errors.append(
                    "EXPERIENCE question should identify the relevant role "
                    "or company."
                )

    elif category == "SKILL_GAP":
        missing = ledger["missing_skills"]

        if not any(contains_term(source, skill) for skill in missing):
            errors.append(
                "SKILL_GAP source must identify a missing required skill."
            )

        if not any(
            contains_term(question, skill)
            for skill in missing
            if contains_term(source, skill)
        ):
            errors.append(
                "SKILL_GAP question must mention the missing skill."
            )

        hypothetical_markers = [
            "if you",
            "how would you",
            "suppose",
            "would you",
            "what would you",
            "if asked to",
            "if you were",
        ]

        if not any(
            marker in normalize_for_match(question)
            for marker in hypothetical_markers
        ):
            errors.append(
                "SKILL_GAP question must be explicitly hypothetical."
            )

        # Strong protection against falsely framing missing skills as experience.
        experience_markers = [
            "how did you use",
            "how have you used",
            "when you used",
            "describe your experience with",
            "how did you implement",
        ]

        if any(
            marker in normalize_for_match(question)
            for marker in experience_markers
        ):
            errors.append(
                "SKILL_GAP question incorrectly frames the missing skill "
                "as previous candidate experience."
            )

    elif category == "BEHAVIORAL":
        if not ledger["behavioral_evidence"]:
            errors.append(
                "Behavioral question generated without explicit "
                "behavioral evidence."
            )

        if source and not any(
            contains_term(source, evidence)
            or contains_term(evidence, source)
            for evidence in ledger["behavioral_evidence"]
        ):
            errors.append(
                "BEHAVIORAL source does not clearly map to explicit evidence."
            )

    else:
        errors.append(f"Unknown category: {category}")

    return errors


def validate_generated_questions(
    result: Dict[str, Any],
    ledger: Dict[str, Any],
) -> Tuple[bool, List[str]]:
    """
    Run the complete post-generation validation.

    Returns:
        (passed, errors)
    """

    errors = validate_basic_structure(result)

    questions = result.get("questions", [])

    if isinstance(questions, list):
        for index, item in enumerate(questions, start=1):
            if not isinstance(item, dict):
                continue

            question_errors = validate_question_grounding(
                item,
                ledger,
            )

            for error in question_errors:
                errors.append(
                    f"Question {index}: {error}"
                )

    # Basic duplicate-question protection.
    normalized_questions = [
        normalize_for_match(
            item.get("question", "")
        )
        for item in questions
        if isinstance(item, dict)
    ]

    if len(normalized_questions) != len(set(normalized_questions)):
        errors.append(
            "Duplicate question text detected."
        )

    return len(errors) == 0, errors


# ============================================================
# SAVE
# ============================================================

def save_generated_questions(
    result: Dict[str, Any],
    path: Path = OUTPUT_PATH,
) -> None:
    """Save validated questions."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(
            result,
            file,
            indent=2,
            ensure_ascii=False,
        )


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_pipeline(
    evaluation_path: Path = EVALUATION_PATH,
    output_path: Path = OUTPUT_PATH,
) -> Dict[str, Any]:
    """Run the complete question-generation pipeline."""

    print("=" * 70)
    print("INTERVIEW EXPERT - PERSONALIZED QUESTION GENERATION")
    print("=" * 70)

    print(f"\nEvaluation file: {evaluation_path}")
    print(f"Groq model:      {GROQ_MODEL}")
    print(f"Output file:     {output_path}")

    evaluation = load_candidate_evaluation(evaluation_path)

    ledger = build_evidence_ledger(evaluation)

    print(f"\nCandidate:       {ledger['candidate']}")
    print(f"Job:             {ledger['job_title']}")
    print(f"Company:         {ledger['company']}")

    print("\nGenerating questions with Groq...")

    result = generate_personalized_questions(ledger)

    passed, errors = validate_generated_questions(
        result,
        ledger,
    )

    print("\nValidation")
    print("-" * 70)

    if not passed:
        print("FAILED")

        for error in errors:
            print(f"- {error}")

        raise RuntimeError(
            "\nGenerated questions failed grounding validation. "
            "The output was NOT saved."
        )

    print("PASSED")
    print("All 12 questions passed structural and grounding checks.")

    save_generated_questions(
        result,
        output_path,
    )

    print(f"\nSaved successfully:")
    print(output_path)

    print("\nQuestion distribution")
    print("-" * 70)

    counts = {}

    for item in result["questions"]:
        category = item["category"]
        counts[category] = counts.get(category, 0) + 1

    for category in REQUIRED_CATEGORY_COUNTS:
        print(
            f"{category:<12} "
            f"{counts.get(category, 0)}"
        )

    print("\nDifficulty distribution")
    print("-" * 70)

    difficulty_counts = {}

    for item in result["questions"]:
        difficulty = item["difficulty"]
        difficulty_counts[difficulty] = difficulty_counts.get(difficulty, 0) + 1

    for difficulty in ["EASY", "MEDIUM", "HARD"]:
        print(
            f"{difficulty:<12} "
            f"{difficulty_counts.get(difficulty, 0)}"
        )

    print("\nGenerated Interview Questions")
    print("=" * 70)

    for item in result["questions"]:
        print(
            f"\nQ{item['id']}. {item['question']}"
            f"\n   Category: {item['category']}"
            f"\n   Difficulty: {item['difficulty']}"
            f"\n   Source: {item['source']}"
        )

    return result


def main() -> None:
    """CLI entry point."""
    run_pipeline()


if __name__ == "__main__":
    main()
