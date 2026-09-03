SmartHire GenAI — Resume Matching & AI Career Mentor
A Generative AI career portal. Upload a resume → get a parsed profile, a ranked list of matching jobs by semantic search, AI-generated CV suggestions, and an AI Career Mentor chatbot that answers grounded in your documents (RAG).

Core scope:
1.Resume parser — LLM structured output → clean JSON profile
2.Semantic job search — embed jobs into FAISS, top-N similarity match
3.AI Career Mentor — RAG over your career notes (LangChain)
4.Strengthen it with: the CV improvement generator, a guardrails layer, an evaluation report, and deployment.

Requirements:
Python 3.10+ (developed on 3.12)
--> git (to clone)
--> A free Gemini API key — https://aistudio.google.com/apikey
--> A free Kaggle account (to download the job dataset)
Getting started:
1. Clone the repo
git clone <REPO_URL> SmartHire-GenAI
cd SmartHire-GenAI

2. Create a virtual environment and install dependencies
python -m venv .venv
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# Windows (cmd)
.venv\Scripts\activate.bat
# macOS / Linux
source .venv/bin/activate
pip install -r requirements.txt

3. Add your API key
copy .env.example .env      # Windows   (macOS/Linux: cp .env.example .env)
Open .env and paste your Gemini key. .env is git-ignored — never commit it.

4. Get the data
The datasets are not committed (see .gitignore). Full details: data/DATASETS.md.
Jobs — download the corpus into data/jobs/:
python download_data.py
Resumes — drop two or three sample PDF/DOCX resumes into data/resumes/.
Career notes — two starter notes are already in data/career_notes/. Add more role guides / skill roadmaps to make the mentor answer more questions.

5. Build in the notebooks (VS Code)
The notebooks run in VS Code (install the Python and Jupyter extensions if prompted). Open a notebook, click Select Kernel → Python Environments, and choose the .venv. Run cells with Shift+Enter.
Run them in order 01 → 03. Each is one part of the project; move working code from a notebook into the matching src/ file, and save the FAISS indexes to vectorstore/ so the app loads them without rebuilding.

01_embeddings_explore.ipynb — resume parser (structured output) + test embeddings
02_build_faiss.ipynb — build the job FAISS index + semantic top-N search
03_rag_prototype.ipynb — the AI Career Mentor RAG chain + guardrails

6. Launch the web app
streamlit run app/streamlit_app.py
Opens the SmartHire GenAI portal in your browser (default http://localhost:8501).

Current status: the plumbing works — src/config.py, src/parsing/loader.py (load + chunk PDF/DOCX/TXT), and download_data.py. The LLM-specific modules (resume_parser, embed, job_search, cv_suggestions, rag_chain, guardrails, evaluate) and the Streamlit UI are stubs — build them via the notebooks first (step 5). Until then the app page will be empty.

Project structure:
smarthire-genai/
├── README.md                       # what it is, setup, how to run
├── requirements.txt                # langchain, langchain-google-genai, faiss-cpu, streamlit, pypdf, ...
├── download_data.py                # fetch the job dataset from Kaggle into data/jobs/
├── .env.example                    # API key placeholder (copy to .env; never commit real keys)
├── .gitignore                      # ignores .env, data/, vectorstore/, venv/
│
├── data/                           # all data lives here (git-ignored)
│   ├── DATASETS.md                 # what to download and where it goes
│   ├── jobs/                       # job dataset CSV (download once)
│   ├── resumes/                    # sample resumes to test the parser (add your own)
│   └── career_notes/               # docs the mentor retrieves from (2 starters provided)
│
├── vectorstore/                    # saved FAISS indexes (git-ignored — rebuilt from data)
│
├── notebooks/                      # prototype here — run in order 01 → 03
│   ├── 01_embeddings_explore.ipynb # resume parser + test embeddings/similarity
│   ├── 02_build_faiss.ipynb        # build the job vector index + semantic search
│   └── 03_rag_prototype.ipynb      # the mentor RAG chain
│
├── src/                            # reusable code — imported by notebooks + app
│   ├── config.py                   # paths, model names, chunk/retrieval params
│   ├── parsing/
│   │   ├── loader.py               # load + chunk PDF/DOCX/TXT   (implemented)
│   │   └── resume_parser.py        # LLM structured output → JSON profile
│   ├── search/
│   │   ├── embed.py                # create embeddings
│   │   └── job_search.py           # FAISS index + top-N query
│   ├── generate/
│   │   ├── prompts.py              # prompt library (all prompts in one place)
│   │   └── cv_suggestions.py       # CV improvement generator
│   ├── mentor/
│   │   └── rag_chain.py            # RAG mentor with LangChain
│   ├── safety/
│   │   └── guardrails.py           # input validation + filters (run before the LLM)
│   └── evaluate.py                 # answer-quality + retrieval checks
│
├── app/
│   └── streamlit_app.py            # the portal UI — build this LAST
│
└── reports/
    ├── answer_quality.md           # evaluation results (template to fill in)
    └── final_report.pdf            # written report (you add this)
What each part is for:
1. data/ — Job CSV in jobs/, sample resumes in resumes/, mentor notes in career_notes/. Git-ignored so data is never committed.
2. vectorstore/ — Saved FAISS indexes. Rebuild from data/, so it is not committed.
3. notebooks/ — Where you prototype. Run 01 → 03; each is one part of the project.
4. src/ — Once code works in a notebook, move it here so the notebooks and the app both import it. config.py holds every path and model name so nothing is hard-coded.
5. app/ — The Streamlit portal. It only wires together pieces that already work in src/, so build it last. Cache the index/chain with @st.cache_resource.
6. reports/ — The evaluation results and the written report.
Notes & constraints:
Keep API keys in .env (local) or Streamlit Secrets (deployed) — never in code.
Watch API usage: cache results while developing and test on small batches first.
The mentor must answer from your notes (RAG), not open-ended — grounding is graded.
Guardrails are required: reject unsafe or off-topic input before every LLM call.