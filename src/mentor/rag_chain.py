import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from src.generate.prompts import MENTOR_SYSTEM_PROMPT, MENTOR_USER_PROMPT
from pathlib import Path

root_dir = Path(__file__).resolve().parents[2]
vectorstore_path = root_dir / "vectorstore" / "faiss_index"

load_dotenv(root_dir / ".env")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

def get_mentor_response(user_query: str, history: list[dict[str, str]] | None = None) -> str:
    api_key = os.getenv("GEMINI_API_KEY")

    if not os.path.exists(vectorstore_path):
        return "Vector store not found. Please build the index first!"

    vectorstore = FAISS.load_local(
        str(vectorstore_path),
        embeddings,
        allow_dangerous_deserialization=True
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    llm = ChatGoogleGenerativeAI(
        model=os.getenv("MODEL_NAME", "gemini-3.6-flash"),
        google_api_key=api_key,
        temperature=0.3,
        timeout=60,
        max_retries=2
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", MENTOR_SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", MENTOR_USER_PROMPT),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    try:
        chat_history = [
            AIMessage(content=message["content"])
            if message["role"] == "assistant"
            else HumanMessage(content=message["content"])
            for message in (history or [])
            if message.get("role") in {"user", "assistant"}
        ]
        response = rag_chain.invoke({"input": user_query, "chat_history": chat_history})
        return response["answer"]
    except Exception as e:
        error_text = str(e)
        if "RESOURCE_EXHAUSTED" in error_text or "429" in error_text:
            return (
                "Gemini API quota exhausted for the configured model. "
                "Please wait for the quota to reset, enable billing, or set "
                "a different MODEL_NAME in your .env file."
            )
        return f"An error occurred: {e}"