import chromadb
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, AutoModelForQuestionAnswering
from typing import List, Dict, Any
import traceback

# --- Constants ---
VECTOR_STORE_PATH = "vector_store"
COLLECTION_NAME = "complaint_embeddings"
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
GENERATOR_MODEL_NAME = 'distilgpt2'
# We will use the QA model again just to get a confidence score
QA_MODEL_NAME = 'distilbert-base-cased-distilled-squad' 

# --- Component Caching ---
_CLIENT = None
_EMBEDDING_MODEL = None
_GENERATOR_PIPELINE = None
_QA_PIPELINE = None # Adding the QA pipeline back

def initialize_components():
    """
    Initializes and caches all necessary AI models and clients.
    """
    global _CLIENT, _EMBEDDING_MODEL, _GENERATOR_PIPELINE, _QA_PIPELINE
    
    if _CLIENT is None:
        print("Initializing ChromaDB client...")
        _CLIENT = chromadb.PersistentClient(path=VECTOR_STORE_PATH)
    
    if _EMBEDDING_MODEL is None:
        print(f"Loading embedding model '{EMBEDDING_MODEL_NAME}'...")
        _EMBEDDING_MODEL = SentenceTransformer(EMBEDDING_MODEL_NAME)
        print("Embedding model loaded.")
        
    if _GENERATOR_PIPELINE is None:
        print(f"Loading text generation model '{GENERATOR_MODEL_NAME}'...")
        _GENERATOR_PIPELINE = pipeline("text-generation", model=GENERATOR_MODEL_NAME, max_new_tokens=100)
        print("Text generation model loaded.")

    if _QA_PIPELINE is None:
        print(f"Loading QA model for scoring '{QA_MODEL_NAME}'...")
        _QA_PIPELINE = pipeline("question-answering", model=QA_MODEL_NAME)
        print("QA model loaded.")

def retrieve_relevant_chunks(question: str, top_k: int = 5) -> List[str]:
    """
    Retrieves the top-k most relevant text chunks from the vector store.
    """
    collection = _CLIENT.get_collection(name=COLLECTION_NAME)
    question_embedding = _EMBEDDING_MODEL.encode(question).tolist()
    results = collection.query(query_embeddings=[question_embedding], n_results=top_k)
    return results['documents'][0]

def generate_answer_from_context(question: str, context_chunks: List[str]) -> str:
    """
    Generates a new, summary answer based on the question and retrieved context.
    """
    context = "\n\n".join(context_chunks)
    prompt = f"""
You are a financial analyst assistant. Answer the following QUESTION based ONLY on the CONTEXT provided.
Synthesize the information to formulate a concise answer.
If the context is not sufficient, state that you cannot answer based on the provided data.

CONTEXT:
---
{context}
---

QUESTION: {question}

ANSWER:
"""
    generated_text = _GENERATOR_PIPELINE(prompt)[0]['generated_text']
    answer_part = generated_text.split("ANSWER:")[1].strip()
    return answer_part

def get_quality_score(question: str, context_chunks: List[str]) -> float:
    """
    Uses the QA model to get a confidence score for how well the context answers the question.
    """
    context = " ".join(context_chunks)
    result = _QA_PIPELINE(question=question, context=context)
    return result['score']

def answer_question_with_rag(question: str) -> Dict[str, Any]:
    """
    Orchestrates the full RAG pipeline, including getting a quality score.
    """
    context_chunks = retrieve_relevant_chunks(question)
    
    # Generate the main answer using the generator model
    answer = generate_answer_from_context(question, context_chunks)
    
    # Get a quality score using the QA model
    score = get_quality_score(question, context_chunks)
    
    final_response = {
        "answer": answer,
        "score": score,
        "sources": context_chunks
    }
    
    return final_response

