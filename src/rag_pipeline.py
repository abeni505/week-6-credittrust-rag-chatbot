import chromadb
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
from typing import List, Dict, Any
import traceback

# --- Constants ---
VECTOR_STORE_PATH = "vector_store"
COLLECTION_NAME = "complaint_embeddings"
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
QA_MODEL_NAME = 'distilbert-base-cased-distilled-squad'

# --- Component Caching ---
# Use global variables to cache initialized models and clients to avoid reloading them.
_CLIENT = None
_EMBEDDING_MODEL = None
_QA_PIPELINE = None

def initialize_components():
    """
    Initializes and caches the ChromaDB client, embedding model, and QA pipeline.
    This function is designed to be called once when the application starts.
    """
    global _CLIENT, _EMBEDDING_MODEL, _QA_PIPELINE
    
    if _CLIENT is None:
        print("Initializing ChromaDB client...")
        _CLIENT = chromadb.PersistentClient(path=VECTOR_STORE_PATH)
    
    if _EMBEDDING_MODEL is None:
        print(f"Loading embedding model '{EMBEDDING_MODEL_NAME}'... (This may take a moment)")
        _EMBEDDING_MODEL = SentenceTransformer(EMBEDDING_MODEL_NAME)
        print("Embedding model loaded.")
        
    if _QA_PIPELINE is None:
        print(f"Loading QA model '{QA_MODEL_NAME}'... (This may take a moment)")
        tokenizer = AutoTokenizer.from_pretrained(QA_MODEL_NAME)
        model = AutoModelForQuestionAnswering.from_pretrained(QA_MODEL_NAME)
        _QA_PIPELINE = pipeline(
            "question-answering",
            model=model,
            tokenizer=tokenizer
        )
        print("QA model loaded.")

def retrieve_relevant_chunks(question: str, top_k: int = 5) -> List[str]:
    """
    Retrieves the top-k most relevant text chunks from the vector store.

    Args:
        question (str): The user's question.
        top_k (int): The number of chunks to retrieve.

    Returns:
        List[str]: A list of the most relevant text chunks.
    """
    if _CLIENT is None or _EMBEDDING_MODEL is None:
        raise Exception("Components not initialized. Call initialize_components() first.")
        
    collection = _CLIENT.get_collection(name=COLLECTION_NAME)
    
    # Embed the user's question to create a vector representation
    question_embedding = _EMBEDDING_MODEL.encode(question).tolist()

    # Perform a similarity search in the vector store
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=top_k
    )
    
    return results['documents'][0]

def generate_answer_from_context(question: str, context_chunks: List[str]) -> Dict[str, Any]:
    """
    Generates a concise answer based on the question and the provided context.
    This function explicitly uses a prompt template structure.

    Args:
        question (str): The user's question.
        context_chunks (List[str]): A list of relevant text chunks.

    Returns:
        Dict[str, Any]: A dictionary containing the answer and its confidence score.
    """
    if _QA_PIPELINE is None:
        raise Exception("QA pipeline not initialized. Call initialize_components() first.")
        
    # Combine the individual chunks into a single block of text for the context
    context = " ".join(context_chunks)
    
    # The QA pipeline reads the context to find the answer to the question.
    result = _QA_PIPELINE(question=question, context=context)
    
    # If the model is not confident, we can override the answer.
    if result['score'] < 0.1: # Confidence threshold can be tuned
        result['answer'] = "I don't have enough information in the provided complaints to answer this question."

    return result

def answer_question_with_rag(question: str) -> Dict[str, Any]:
    """
    Orchestrates the full RAG pipeline from question to answer.

    Args:
        question (str): The user's question.

    Returns:
        Dict[str, Any]: A dictionary containing the generated answer and the source chunks.
    """
    # Step 1: Retrieve relevant documents from the vector store
    context_chunks = retrieve_relevant_chunks(question)
    
    # Step 2: Use the retrieved documents and a prompt to generate an answer
    qa_result = generate_answer_from_context(question, context_chunks)
    
    # Format the final output for the application
    final_response = {
        "answer": qa_result["answer"],
        "score": qa_result["score"],
        "sources": context_chunks
    }
    
    return final_response

# This block allows you to test the script directly from the command line
if __name__ == '__main__':
    print("--- Running RAG Pipeline Test Script ---")
    print("NOTE: The first run may take several minutes as AI models are downloaded.")
    
    try:
        # Initialize all models and clients
        initialize_components()
        
        # Define a test question
        print("\nInitialization complete. Running a test question...")
        test_question = "Why are people having issues with their credit card payments?"
        
        # Get the answer
        final_answer = answer_question_with_rag(test_question)
        
        # Print the results in a readable format
        print("\n--- RAG PIPELINE TEST RESULT ---")
        print(f"Question: {test_question}")
        print(f"\nAnswer: {final_answer['answer']}")
        print(f"Confidence Score: {final_answer['score']:.4f}")
        print("\n--- Sources Used ---")
        for i, source in enumerate(final_answer['sources']):
            print(f"{i+1}. {source[:150]}...")
        print("---------------------------------")
        
    except Exception as e:
        print(f"\nAn error occurred during the test run: {e}")
        traceback.print_exc()
