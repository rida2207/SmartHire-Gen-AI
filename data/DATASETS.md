Data — what goes where:
This project has three kinds of data, each in its own folder under data/. None of it is committed to git (see .gitignore) — you add it locally once.

1. data/jobs/ — the job corpus (you download this)
The job postings you embed into FAISS for semantic search. Download once from Kaggle.

Dataset	Kaggle slug (owner/name)	What you get
Jobs on Naukri.com (~22k Indian jobs)	PromptCloudHQ/jobs-on-naukricom	naukri_com-job_sample.csv
Columns you use for the embedded text: jobtitle, skills, jobdescription (combined into one string per job). The file also has company, joblocation_address, experience, payrate, etc. — keep them to show alongside a match, ignore the rest.

Any job dataset with a title + skills + description works. LinkedIn job postings (arshkon/linkedin-job-postings) is a good larger alternative if you want more variety.

How to download:
Option A — one script (easiest):

pip install kagglehub
python download_data.py
The first run asks for your Kaggle credentials (username + API key from Kaggle → Settings → API → Create New Token). It places the CSV in data/jobs/.

Option B — Kaggle website (no code): create a free account at kaggle.com, search "jobs on naukri", Download, unzip, and move the CSV into data/jobs/.

Option C — Kaggle API:
pip install kaggle
# after saving kaggle.json to ~/.kaggle/ (Windows: C:\Users\<you>\.kaggle\)
kaggle datasets download -d PromptCloudHQ/jobs-on-naukricom -p data/jobs --unzip
The filename is already wired up in src/config.py (JOBS_CSV). Only edit that if your downloaded filename differs.

2. data/resumes/ — sample resumes (you add these)
A few PDF or DOCX resumes to test the parser and the job search. Use your own resume, your teammates', or a handful from the Kaggle resume dataset (jillanisofttech/updated-resume-dataset) exported to PDF. Two or three are enough to build with.

3. data/career_notes/ — the mentor's knowledge base (starter notes provided)
The documents the AI Career Mentor retrieves from (RAG). Two starter notes ship with the project:

data/career_notes/
├── data_analyst_roadmap.md
└── resume_writing_tips.md
Add more as markdown or text files — role guides, skill roadmaps, interview tips. The more (relevant) notes you add, the more questions the mentor can answer from grounded sources. The mentor must answer from these notes, not from open-ended generation — that grounding is part of the grade.

Where each dataset is used:
Folder	Feeds	Built in
data/jobs/	Semantic job search (FAISS index)	notebook 02
data/resumes/	Resume parser + profile to search with	notebook 01
data/career_notes/	AI Career Mentor (RAG)