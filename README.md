# Interview Expert

## AI-Powered Resume–JD Matching and Candidate Evaluation System

Interview Expert analyzes a candidate's resume against a Job Description (JD), evaluates the candidate's skills, experience and projects, and generates personalized interview questions based on verified resume and JD evidence.

## Workflow

Resume PDF + Job Description
        ↓
Text Extraction
        ↓
Resume & JD Analysis
        ↓
Resume–JD Matching
        ↓
Evidence Filtering
        ↓
Prompt Builder
        ↓
Groq LLM
        ↓
Question Validation
        ↓
Personalized Interview Questions

## Key Features

- Resume text extraction
- Job Description analysis
- Required skill matching
- Experience evaluation
- Project extraction
- Evidence-based candidate evaluation
- Personalized interview question generation
- Question grounding and validation
- Deterministic fallback for invalid LLM output

## Technologies

- Python
- PDF processing
- Tesseract OCR
- Groq API
- openai/gpt-oss-120b
- JSON
- Jupyter Notebook
- Git & GitHub

## Question Generation

The system generates 12 personalized interview questions:

- 3 JD Skill questions
- 3 Project questions
- 2 Experience questions
- 2 Skill Gap questions
- 2 Behavioral questions

Difficulty distribution:

- 4 Easy
- 5 Medium
- 3 Hard

## Project Structure

```text
InterviewExpert/
├── app/
│   ├── groq_service.py
│   ├── jd_reader.py
│   └── question_generator.py
├── data/
├── notebooks/
├── utils/
├── app.py
├── .gitignore
└── README.md
