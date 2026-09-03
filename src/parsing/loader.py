from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
import tempfile
import os

def load_resume_text(uploaded_file) -> str:
    """Extracts raw text from uploaded PDF or DOCX files."""
    try:
        file_name = uploaded_file.name
    except AttributeError:
        file_name = str(uploaded_file)

    suffix = os.path.splitext(file_name)[1]

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        try:
            file_bytes = uploaded_file.getvalue()
        except AttributeError:
            with open(uploaded_file, "rb") as f:
                file_bytes = f.read()
        tmp_file.write(file_bytes)
        tmp_path = tmp_file.name

    try:
        if suffix == ".pdf":
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
            return "\n".join([doc.page_content for doc in docs])
        elif suffix in [".docx", ".doc"]:
            loader = Docx2txtLoader(tmp_path)
            docs = loader.load()
            return "\n".join([doc.page_content for doc in docs])
        else:
            raise ValueError("Unsupported file type")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)