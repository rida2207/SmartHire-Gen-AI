import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from src.search.job_search import search_jobs
import os
from dotenv import load_dotenv
env_path = Path(__file__).resolve().parent.parent / '.env'
load_dotenv(dotenv_path=env_path)
import streamlit as st
st.set_page_config(page_title="SmartHire GenAI", page_icon="🚀", layout="wide")
from pypdf import PdfReader
from src.generate.prompts import (
    JOB_MATCHING_SYSTEM_PROMPT,
    JOB_MATCHING_USER_PROMPT
   )
from src.parsing.resume_parser import parse_resume
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

if "candidate_profile" not in st.session_state:
    st.session_state.candidate_profile = {
        "name": "",
        "email": "",
        "skills": "",
        "experience_years": 0.0,
        "education": "",
        "target_role": "",
        "raw_text": ""
    }
st.title("🚀 SmartHire GenAI - Intelligent Recruitment & Mentor System")
st.markdown("""
Welcome to SmartHire! Use the navigation on the left to search candidate resumes, parse CVs, or get AI-powered career suggestions.
""")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

try:
    from google import genai
except Exception:
    genai = None

client = None
if GEMINI_API_KEY and genai is not None:
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
    except Exception:
        client = None


def format_job_result(result_text: str) -> str:
    lines = []
    for raw_line in str(result_text).splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if ":" in line and not line.startswith("-"):
            label, value = line.split(":", 1)
            lines.append(f"- **{label.strip()}:** {value.strip()}")
        else:
            lines.append(f"- {line.strip()}")
    return "\n".join(lines)


@st.cache_resource
def load_rag_chain():
    vectorstore_path = root_dir / "vectorstore" / "faiss_index"

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return FAISS.load_local(
        str(vectorstore_path),
        embeddings,
        allow_dangerous_deserialization=True
    )

rag_chain = load_rag_chain()           
st.sidebar.header("Navigation")
app_mode = st.sidebar.selectbox(
    "Choose a feature",
    ["Home", "Create Profile", "Semantic Job Search", "Resume Suggestions", "Mentor RAG Chat"]
)

if app_mode == "Home":
    st.subheader("System Status")
    vectorstore_path = root_dir / "vectorstore" / "faiss_index"
    if os.path.exists(vectorstore_path):
        st.success("Vectorstore detected! Ready for queries.")
    else:
        st.warning("Vectorstore not found yet. Please run your notebooks/embedding scripts first.")
    st.info("Select a feature from the sidebar to begin.")
elif app_mode == "Create Profile":
    st.title("👤 Create Candidate Profile")
    st.write("Set up your profile manually or parse it automatically using your resume.")

    input_mode = st.radio(
        "Choose how you want to fill your profile:",
        ("Manually enter the information", "Get information from resume"),
        horizontal=True
    )

    if input_mode == "Get information from resume":
        uploaded_file = st.file_uploader("Upload your resume (PDF)", type=["pdf"])
        
        if uploaded_file is not None:
            from pypdf import PdfReader
            reader = PdfReader(uploaded_file)
            raw_text = "".join([page.extract_text() or "" for page in reader.pages])
            st.session_state.candidate_profile["raw_text"] = raw_text
            
            if st.button("Extract Details with AI"):
                with st.spinner("Parsing resume with your AI model..."):
                    try:
                        from src.parsing.resume_parser import parse_resume
                        profile_obj = parse_resume(raw_text)
                        st.session_state.candidate_profile.update({
                            "name": profile_obj.name,
                            "email": profile_obj.email,
                            "skills": ", ".join(profile_obj.skills) if isinstance(profile_obj.skills, list) else profile_obj.skills,
                            "experience_years": profile_obj.experience_years,
                            "education": ", ".join(profile_obj.education) if isinstance(profile_obj.education, list) else profile_obj.education,
                            "target_role": profile_obj.target_role
                        })
                        st.success("Resume parsed successfully!")
                        st.rerun()
                    except ConnectionError as e:
                        st.error(f"🔌 Connection Error: {e}")
                        st.info(
                            "**Troubleshooting tips:**\n"
                            "1. Check your internet connection\n"
                            "2. Verify your GEMINI_API_KEY is valid (get it from https://makersuite.google.com/app/apikey)\n"
                            "3. Check if firewall/proxy is blocking api.generativeai.google.com\n"
                            "4. Try again in a few moments if the API service is temporarily unavailable"
                        )
                    except ValueError as e:
                        st.error(f"⚙️ Configuration Error: {e}")
                    except Exception as e:
                        st.error(f"Error parsing resume: {e}")
                        with st.expander("Technical Details"):
                            st.code(str(e))
    with st.form("profile_form"):
        name = st.text_input("Full Name", value=st.session_state.candidate_profile["name"])
        email = st.text_input("Email", value=st.session_state.candidate_profile["email"])
        skills = st.text_area("Skills", value=str(st.session_state.candidate_profile["skills"]))
        exp_years = st.number_input("Years of Experience", value=float(st.session_state.candidate_profile["experience_years"]))
        education = st.text_area("Education", value=str(st.session_state.candidate_profile["education"]))
        target_role = st.text_input("Target Role", value=st.session_state.candidate_profile["target_role"])
        
        if st.form_submit_button("Save Profile"):
            current_resume = st.session_state.candidate_profile.get("raw_resume_text", "")
            if not name.strip() or not skills.strip():
                st.error("⚠️ Please upload a valid resume or provide complete details. It cannot be blank.")
            else:
                st.session_state.candidate_profile.update({
                    "name": name, "email": email, "skills": skills,
                    "experience_years": exp_years, "education": education, "target_role": target_role
                })
                st.success("✅ Profile saved globally! Your app is now tied to your data.")
elif app_mode == "Semantic Job Search":
    st.subheader("🔍 Semantic Job Search & Matching")

    search_option = st.radio(
        "Choose Search Method:",
        options=["Search by Keyword/Skill", "Find Jobs Based on My Profile"]
    )

    if search_option == "Search by Keyword/Skill":
        query = st.text_input("Enter job role or key skills you are looking for:")

        if st.button("Search Jobs"):
            if query:
                try:
                    with st.spinner("Searching for matching jobs..."):
                        results = search_jobs(query, k=3)

                    if len(results) == 0:
                        st.warning("No jobs matched your search. Try a broader keyword like 'python', 'data analyst', or 'software engineer'.")
                    else:
                        st.success(f"Found {len(results)} matching job(s)!")

                        for i, doc in enumerate(results[:3], 1):
                            with st.container():
                                st.markdown(f"### Result {i}")
                                st.markdown(format_job_result(doc.page_content))
                                st.markdown("---")
                except Exception as e:
                    st.error(f"Error searching jobs: {e}")
            else:
                st.warning("Please enter a query to search.")

    elif search_option == "Find Jobs Based on My Profile":
        if st.button("Find Jobs Based on My Profile"):
            profile = st.session_state.get("candidate_profile", {})

            if not profile or (not profile.get("skills") and not profile.get("target_role")):
                st.warning("⚠️ Please update your profile first before searching based on your profile!")
            else:
                try:
                    candidate_text = f"""
Skills: {profile.get('skills', '')}
Experience: {profile.get('experience_years', '')}
Education: {profile.get('education', '')}
Target Role: {profile.get('target_role', '')}
"""
                    with st.spinner("Finding jobs related to your profile..."):
                        results = search_jobs(candidate_text, k=3)

                    if len(results) == 0:
                        st.warning("No jobs matched your profile. Add more skills or target role information first.")
                    else:
                        st.success(f"Found {len(results)} matching job(s)!")

                        for i, doc in enumerate(results[:3], 1):
                            with st.container():
                                st.markdown(f"### Result {i}")
                                st.markdown(format_job_result(doc.page_content))
                                st.markdown("---")
                except Exception as e:
                    st.error(f"Error searching jobs: {e}")
             

elif app_mode == "Resume Suggestions":
    st.subheader("💡 Candidate Resume Improvement & Rewriting")
    profile = st.session_state.get("candidate_profile", {})
    raw_resume_text = profile.get("raw_text", "")
    
    if not raw_resume_text:
        st.warning("⚠️ No resume found! Please go to *Create Profile* in the sidebar first to upload your resume.")
    else:
        st.success(f"🔗 Connected to profile: *{profile.get('name', 'Candidate')}*")
        job_description = st.text_area("Paste the Target Job Description:")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Generate Suggestions"):
                if job_description:
                    try:
                        from src.generate.cv_suggestions import get_cv_suggestions

                        suggestions = get_cv_suggestions(
                            resume_json=raw_resume_text,
                            job_description=job_description
                        )

                        st.markdown("### 🎯 Tailored Recommendations")
                        st.write(suggestions)
                    except Exception as e:
                        st.error(f"Error: {e}")
                else:
                    st.warning("Please paste a target job description first.")
                    
        with col2:
            if st.button("✨ Rewrite My Resume for This Job"):
                if job_description:
                    with st.spinner("Rewriting your resume to match the target job..."):
                        try:
                            if not GEMINI_API_KEY:
                                raise ValueError("GEMINI_API_KEY is missing from your .env file.")
                            if client is None:
                                from google import genai
                                client = genai.Client(api_key=GEMINI_API_KEY)
                            
                            prompt = f"""
                            You are an expert resume writer. Take the candidate's current resume text below and rewrite their experience and summary sections so they are heavily tailored to match the target job description. Use strong action verbs and highlight relevant skills.
                            
                            Target Job Description:
                            {job_description}
                            
                            Candidate's Current Resume:
                            {raw_resume_text}
                            """
                            
                            response = client.models.generate_content(
                                model='gemini-3.6-flash',
                                contents=prompt,
                            )
                            
                            st.markdown("### 📝 Your Tailored Resume Version")
                            st.markdown(response.text)
                        except Exception as e:
                            st.error(f"Error rewriting resume: {e}")
                else:
                    st.warning("Please paste a target job description first.")
elif app_mode == "Mentor RAG Chat":
    st.subheader("💬 AI Career Mentor (RAG)")
    if "mentor_messages" not in st.session_state:
        st.session_state.mentor_messages = []

    for message in st.session_state.mentor_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_query := st.chat_input("Ask your career mentor anything..."):
        st.session_state.mentor_messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    from src.mentor.rag_chain import get_mentor_response as ask_mentor
                    
                    response_text = ask_mentor(user_query, history=st.session_state.mentor_messages[:-1])
                    
                    st.markdown(response_text)
                    st.session_state.mentor_messages.append({"role": "assistant", "content": response_text})
                except Exception as e:
                    st.error(f"Error: {e}")