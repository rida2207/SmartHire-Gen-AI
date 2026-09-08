import os
import pandas as pd
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

def build_vector_store():
    print("Loading documents...")
    documents = []

    try:
        df = pd.read_csv("data/jobs/jobs.csv")
        for _, row in df.iterrows():
            content = f"Job Title: {row.get('job_title', '')}\nCompany: {row.get('company', '')}\nDescription: {row.get('description', '')}"
            documents.append(Document(page_content=content))
        print("Jobs loaded successfully!")
    except Exception as e:
        print(f"Could not load jobs CSV: {e}")

    notes_dir = "data/career_notes"
    if os.path.exists(notes_dir):
        for filename in os.listdir(notes_dir):
            if filename.endswith(".txt"):
                filepath = os.path.join(notes_dir, filename)
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        text = f.read()
                    documents.append(Document(page_content=text))
                except Exception as e:
                    print(f"Error reading {filename}: {e}")
        print("Career notes loaded successfully!")

    print(f"Total documents to index: {len(documents)}")

    if not documents:
        print("No documents found to embed! Check your data paths.")
        return

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = text_splitter.split_documents(documents)

    print("Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    print("Building FAISS index...")
    vectorstore = FAISS.from_documents(split_docs, embeddings)

    output_dir = "vectorstore/faiss_index"
    os.makedirs(output_dir, exist_ok=True)
    vectorstore.save_local(output_dir)
    print(f"Success! FAISS index saved locally to '{output_dir}'.")

if __name__ == "__main__":
    build_vector_store()