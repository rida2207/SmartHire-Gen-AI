import os
import json
import time
import requests
from pathlib import Path
from dotenv import load_dotenv

from src.generate.prompts import (
    CV_IMPROVEMENT_SYSTEM_PROMPT,
    CV_IMPROVEMENT_USER_PROMPT,
)

load_dotenv(Path(__file__).resolve().parents[2] / ".env")


def get_cv_suggestions(resume_json: str, job_description: str) -> str:
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError("GEMINI_API_KEY is missing from .env")

    formatted_prompt = CV_IMPROVEMENT_USER_PROMPT.format(
        resume_json=resume_json,
        job_description=job_description,
    )

    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": (
                            f"System Instructions:\n{CV_IMPROVEMENT_SYSTEM_PROMPT}"
                            f"\n\nTask:\n{formatted_prompt}"
                        )
                    }
                ]
            }
        ],
        "generationConfig": {"temperature": 0.3},
    }

    configured_model = os.getenv("MODEL_NAME", "gemini-3.6-flash")
    fallback_models = os.getenv(
        "GEMINI_FALLBACK_MODELS", "gemini-3.6-flash"
    ).split(",")
    models = list(dict.fromkeys(
        model.strip() for model in [configured_model, *fallback_models] if model.strip()
    ))

    last_response = None
    for model in models:
        url = (
            "https://generativelanguage.googleapis.com/v1beta/"
            f"models/{model}:generateContent?key={api_key}"
        )

        for attempt in range(3):
            response = requests.post(
                url,
                headers={"Content-Type": "application/json"},
                data=json.dumps(payload),
                timeout=60,
            )
            last_response = response

            if response.status_code == 200:
                return response.json()["candidates"][0]["content"]["parts"][0]["text"]

            if response.status_code not in (429, 500, 502, 503, 504):
                break

            if attempt < 2:
                time.sleep(2 ** attempt)

    if last_response is not None:
        raise Exception(
            f"Gemini API unavailable after trying {', '.join(models)} "
            f"(HTTP {last_response.status_code}): {last_response.text}"
        )

    raise Exception("Gemini API request failed before receiving a response")