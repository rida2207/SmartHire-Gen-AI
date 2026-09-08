import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import re
from typing import Tuple, List, Dict, Set
import pandas as pd
from functools import lru_cache

__all__ = ['search_jobs', 'calculate_job_match', 'extract_skills_and_info', 'test_search', 'load_skills_from_csv']

# Cache for skills loaded from CSV
_SKILLS_CACHE = None

@lru_cache(maxsize=1)
def load_skills_from_csv() -> Set[str]:
    """
    Load all unique skills from the jobs CSV file.
    Extracts skills from the 'Key Skills' column and removes duplicates.
    Uses caching to avoid repeated file reads.
    """
    global _SKILLS_CACHE
    
    if _SKILLS_CACHE is not None:
        return _SKILLS_CACHE
    
    skills_set = set()
    
    # Try multiple possible paths
    csv_paths = [
        "data/jobs/naukri_jobs.csv",
        "../data/jobs/naukri_jobs.csv",
        "../../data/jobs/naukri_jobs.csv",
        os.path.join(os.path.dirname(__file__), "../../data/jobs/naukri_jobs.csv"),
    ]
    
    for csv_path in csv_paths:
        try:
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path, encoding='utf-8')
                if 'Key Skills' in df.columns:
                    for skills_str in df['Key Skills'].dropna():
                        # Split by pipe and comma
                        skills_list = re.split(r'\||\,', str(skills_str))
                        for skill in skills_list:
                            skill_clean = skill.strip().lower()
                            # Filter out very short strings and common non-skills
                            if skill_clean and len(skill_clean) > 2:
                                skills_set.add(skill_clean)
                
                if skills_set:
                    _SKILLS_CACHE = frozenset(skills_set)
                    return frozenset(skills_set)
        except Exception as e:
            print(f"Warning: Could not read CSV from {csv_path}: {e}")
            continue
    
    # If no skills found from CSV, use default set
    print("Using default skills set (CSV not found)")
    default_skills = frozenset({
        'python', 'java', 'c++', 'sql', 'pandas', 'numpy',
        'machine learning', 'deep learning', 'nlp',
        'data structures', 'algorithms', 'system design',
        'version control', 'git', 'docker', 'kubernetes',
        'aws', 'gcp', 'azure', 'api', 'rest api',
        'django', 'fastapi', 'flask', 'react', 'javascript'
    })
    _SKILLS_CACHE = default_skills
    return default_skills

def extract_skills_and_info(text: str, valid_skills: Set[str] = None) -> Dict:
    """Extract skills from text using the provided skill set."""
    if valid_skills is None:
        valid_skills = load_skills_from_csv()
    
    text_lower = text.lower()
    found_skills = []
    
    for skill in valid_skills:
        # Create a pattern that matches whole words only
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, text_lower):
            found_skills.append(skill)
    
    return {'skills': list(set(found_skills))}  # Remove duplicates

def calculate_job_match(candidate_profile: str, job_description: str, search_query: str = "") -> Tuple[int, List[str], List[str], str]:
    """
    Calculate match score between candidate profile and job description.
    Uses skills loaded from the CSV file to ensure consistent skill matching.
    
    Args:
        candidate_profile: Candidate's profile text
        job_description: Job description text
        search_query: Original search query from user (optional)
    
    Returns:
        Tuple of (match_score, matching_skills, skill_gaps, recommendation)
    """
    # Load valid skills from CSV
    valid_skills = load_skills_from_csv()
    
    # Extract skills from candidate profile using CSV skills
    candidate_info = extract_skills_and_info(candidate_profile, valid_skills)
    candidate_skills = set(candidate_info['skills'])
    
    # Extract skills from job description using CSV skills
    job_info = extract_skills_and_info(job_description, valid_skills)
    required_skills = set(job_info['skills'])
    
    # If no candidate skills found, use search query as candidate skills
    if len(candidate_skills) == 0 and search_query:
        query_info = extract_skills_and_info(search_query, valid_skills)
        candidate_skills = set(query_info['skills'])
        
        # If search query has no recognized skills, try to match it directly
        if len(candidate_skills) == 0:
            search_lower = search_query.lower().strip()
            # Check if search query matches any skill in valid_skills
            for skill in valid_skills:
                if skill in search_lower or search_lower in skill:
                    candidate_skills.add(skill)
                    break
    
    # Calculate matching skills (remove duplicates automatically with set)
    matching_skills = sorted(list(candidate_skills & required_skills))
    
    # Calculate skill gaps (remove duplicates automatically with set)
    skill_gaps = sorted(list(required_skills - candidate_skills))
    
    # Calculate match score
    if len(required_skills) == 0:
        # If still no skills detected, show moderate match
        match_score = 50
    else:
        match_score = int((len(matching_skills) / len(required_skills)) * 100)
    
    # Generate recommendation
    if match_score >= 80:
        recommendation = f"Excellent match! You have {match_score}% of required skills. You are well-qualified for this role."
    elif match_score >= 60:
        recommendation = f"Good match ({match_score}%)! You have most required skills. Consider learning: {', '.join(skill_gaps[:3]) if skill_gaps else 'the remaining skills'}"
    elif match_score >= 40:
        recommendation = f"Moderate match ({match_score}%). Focus on learning: {', '.join(skill_gaps[:5]) if skill_gaps else 'more skills'}"
    else:
        recommendation = f"Fair match ({match_score}%). Key areas to focus: {', '.join(skill_gaps[:5]) if skill_gaps else 'Review job requirements'}"
    
    return match_score, matching_skills, skill_gaps, recommendation


def _resolve_vectorstore_dir():
    candidate_paths = [
        os.path.join(os.getcwd(), "vectorstore", "faiss_index"),
        os.path.join(os.getcwd(), "..", "vectorstore", "faiss_index"),
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "vectorstore", "faiss_index"),
        os.path.join(os.path.dirname(__file__), "..", "..", "vectorstore", "faiss_index"),
    ]

    for path in candidate_paths:
        normalized = os.path.abspath(path)
        if os.path.exists(normalized):
            return normalized

    return os.path.abspath(os.path.join(os.getcwd(), "vectorstore", "faiss_index"))


def test_search():
    output_dir = _resolve_vectorstore_dir()

    if not os.path.exists(output_dir):
        print(f"Error: Index directory not found at {output_dir}")
        return

    print("Loading local embeddings and FAISS index...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local(output_dir, embeddings, allow_dangerous_deserialization=True)

    query = "Python developer with machine learning experience"
    print(f"\nSearching for jobs matching: '{query}'\n" + "-"*50)
    results = vectorstore.similarity_search(query, k=3)

    for i, doc in enumerate(results, 1):
        print(f"\n--- Result {i} ---")
        print(doc.page_content)
        print("-" * 50)

def search_jobs(query_text: str, k: int = 3):
    output_dir = _resolve_vectorstore_dir()

    if not os.path.exists(output_dir):
        raise FileNotFoundError(f"FAISS index not found at {output_dir}. Please build the vectorstore first.")

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local(output_dir, embeddings, allow_dangerous_deserialization=True)

    results = vectorstore.similarity_search(query_text, k=k)
    return results

if __name__ == "_main_":
    test_search()