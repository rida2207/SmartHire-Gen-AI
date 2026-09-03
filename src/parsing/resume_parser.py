import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from typing import List
import socket
import time


def is_transient_api_error(error: Exception) -> bool:
    """Return whether an API error is likely safe to retry."""
    status_code = getattr(error, "code", None) or getattr(error, "status_code", None)
    if status_code in {408, 429, 500, 502, 503, 504}:
        return True

    error_text = str(error).upper()
    return any(
        marker in error_text
        for marker in (
            "DEADLINE_EXCEEDED",
            "DEADLINE EXCEEDED",
            "TIMED OUT",
            "TIMEOUT",
            "HTTP 408",
            "HTTP 429",
            "HTTP 500",
            "HTTP 502",
            "HTTP 503",
            "HTTP 504",
        )
    )

load_dotenv()

class ResumeProfile(BaseModel):
    name: str = Field(description="Full name of candidate")
    email: str = Field(description="Email address if available, else N/A")
    skills: List[str] = Field(description="List of technical and soft skills")
    experience_years: float = Field(description="Estimated total years of experience")
    education: List[str] = Field(description="Degrees, fields of study, or universities")
    target_role: str = Field(description="Best fitting primary job title for this resume")

def check_api_connectivity(max_retries: int = 3) -> bool:
    """
    Check if the system can reach Google's API endpoints.
    Retry with backoff if connection fails.
    """
    api_hosts = ["api.generativeai.google.com", "generativelanguage.googleapis.com"]
    
    for host in api_hosts:
        for attempt in range(max_retries):
            try:
                socket.gethostbyname(host)
                return True
            except socket.gaierror as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"Connection attempt {attempt + 1} failed, retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"Failed to connect to {host}: {e}")
    
    return False

def parse_resume(resume_text: str, max_retries: int = 3) -> ResumeProfile:
    """
    Parse resume text and extract structured profile information.
    Includes retry logic with exponential backoff for network failures.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    model_name = os.getenv("MODEL_NAME", "gemini-1.5-flash")
    
    # Validate API key is set
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY not found. Please set it in your .env file.\n"
            "Visit https://makersuite.google.com/app/apikey to get your API key."
        )
    
    last_error = None
    
    for attempt in range(max_retries):
        try:
            # Check connectivity before attempting API call
            if attempt == 0 and not check_api_connectivity(max_retries=2):
                raise ConnectionError(
                    "Cannot reach Google API endpoints. Please check your internet connection."
                )
            
            llm = ChatGoogleGenerativeAI(
                model=model_name,
                api_key=api_key,
                temperature=0.0,
                timeout=60,
                max_retries=2
            )
            
            structured_llm = llm.with_structured_output(ResumeProfile)
            prompt = f"Extract structured profile information from the following resume:\n\n{resume_text}"
            
            return structured_llm.invoke(prompt)
            
        except (ConnectionError, socket.gaierror) as e:
            last_error = e
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"Connection error (attempt {attempt + 1}/{max_retries}): {e}")
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                raise ConnectionError(
                    f"Failed to connect to Google API after {max_retries} attempts. "
                    f"Last error: {e}\n"
                    f"Please check:\n"
                    f"1. Your internet connection\n"
                    f"2. Your GEMINI_API_KEY is valid\n"
                    f"3. Firewall/proxy settings are not blocking api.generativeai.google.com"
                ) from e
        except Exception as e:
            if not is_transient_api_error(e):
                raise

            last_error = e
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(
                    f"Transient API error (attempt {attempt + 1}/{max_retries}): {e}"
                )
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                raise ConnectionError(
                    "Resume parsing timed out or the Google API was temporarily "
                    f"unavailable after {max_retries} attempts. Last error: {e}"
                ) from e
    
    # Fallback (shouldn't reach here)
    if last_error:
        raise last_error