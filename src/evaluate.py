import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import VECTORSTORE_DIR
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

def evaluate_retrieval():
    """
    Evaluates the retrieval quality of the FAISS vectorstore by running a test query.
    """
    print("Starting retrieval evaluation...")
    
    if not os.path.exists(VECTORSTORE_DIR):
        print(f"❌ Error: Vectorstore directory not found at {VECTORSTORE_DIR}")
        print("Please build your vectorstore using your notebooks first.")
        return False
        
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.load_local(VECTORSTORE_DIR, embeddings, allow_dangerous_deserialization=True)
        
        test_query = "Data Engineer with Python and SQL experience"
        print(f"Running test query: '{test_query}'")
        
        docs = vectorstore.similarity_search(test_query, k=3)
        
        if len(docs) > 0:
            print(f"Successfully retrieved {len(docs)} relevant documents!")
            for i, doc in enumerate(docs):
                print(f"\n--- Result {i+1} ---")
                print(doc.page_content[:200] + "...")
            print("\n✅ Evaluation passed: Retrieval is working.")
            return True
        else:
            print("⚠️ Warning: Vectorstore loaded, but zero documents were retrieved.")
            return False
            
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        return False

if __name__ == "__main__":
    evaluate_retrieval()