from groq import Groq
from dotenv import load_dotenv

import json
import os


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-120b"
)


if not API_KEY:
    raise ValueError(
        "GROQ_API_KEY is missing. "
        "Please add it to your .env file."
    )


# ============================================================
# GROQ CLIENT
# ============================================================

client = Groq(api_key=API_KEY)


# ============================================================
# GENERATE RESPONSE
# ============================================================

def generate_response(prompt):
    """
    Send a prompt to the Groq model
    and return the generated text.
    """

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.4
    )

    return response.choices[0].message.content


# ============================================================
# TEST GROQ CONNECTION
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("TESTING GROQ")
    print("=" * 70)

    print(f"Model: {MODEL_NAME}")

    test_prompt = """
    Explain the difference between supervised
    and unsupervised machine learning in 3 points.
    """

    try:

        result = generate_response(test_prompt)

        print("\nGroq Response:")
        print("-" * 70)
        print(result)
        print("-" * 70)

        print("\nGroq connection test PASSED.")

    except Exception as error:

        print("\nGroq connection test FAILED.")
        print(error)

    print("=" * 70)