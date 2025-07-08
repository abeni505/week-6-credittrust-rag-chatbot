import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
from typing import List, Dict, Any
import os

# --- Constants ---
# Using constants makes the script easier to configure and read.
DATA_PATH = 'data/filtered_complaints.csv'
VECTOR_STORE_PATH = "vector_store"
COLLECTION_NAME = "complaint_embeddings"
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
BATCH_SIZE = 4000  # A safe number well below ChromaDB's max batch size limit.

def load_data(file_path: str) -> pd.DataFrame:
    """
    Loads the cleaned and filtered complaint data from a CSV file.

    Args:
        file_path (str): The path to the input CSV file.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the complaint data.
        
    Raises:
        FileNotFoundError: If the file at file_path does not exist.
    """
    print(f"Loading data from '{file_path}'...")
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Error: The file '{file_path}' was not found. "
            "Please ensure you have run the EDA notebook first to generate it."
        )
    df = pd.read_csv(file_path)
    print("Data loaded successfully.")
    return df

def create_text_chunks(df: pd.DataFrame):
    """
    Splits the complaint narratives into smaller text chunks.

    Args:
        df (pd.DataFrame): The DataFrame containing the complaint narratives.

    Returns:
        tuple: A tuple containing two lists:
               - A list of the text chunks (strings).
               - A list of corresponding metadata dictionaries.
    """
    print("Creating text chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len
    )
    
    chunks = []
    metadata = []
    for index, row in df.iterrows():
        narrative = str(row.get('cleaned_narrative', ''))
        if not narrative:
            continue

        narrative_chunks = text_splitter.split_text(narrative)

        for chunk in narrative_chunks:
            chunks.append(chunk)
            # Store important metadata alongside each chunk
            metadata.append({
                'complaint_id': str(row.get('Complaint ID', '')),
                'product': str(row.get('Product', '')),
                'date_received': str(row.get('Date received', ''))
            })
            
    print(f"Created {len(chunks)} text chunks.")
    return chunks, metadata

def generate_embeddings(chunks: List[str], model_name: str) -> List[List[float]]:
    """
    Generates vector embeddings for a list of text chunks.

    Args:
        chunks (List[str]): The list of text chunks to embed.
        model_name (str): The name of the SentenceTransformer model to use.

    Returns:
        List[List[float]]: A list of embeddings, where each embedding is a list of floats.
    """
    print(f"Loading embedding model '{model_name}'...")
    embedding_model = SentenceTransformer(model_name)
    print("Embedding model loaded.")
    
    print("Generating embeddings for text chunks...")
    embeddings = embedding_model.encode(chunks, show_progress_bar=True)
    print(f"Generated {len(embeddings)} embeddings.")
    return embeddings

def create_and_persist_vector_store(chunks: List[str], embeddings: List[List[float]], metadata: List[Dict[str, Any]]):
    """
    Creates and saves a persistent vector store using ChromaDB.

    Args:
        chunks (List[str]): The list of text documents.
        embeddings (List[List[float]]): The corresponding list of vector embeddings.
        metadata (List[Dict[str, Any]]): The corresponding list of metadata dictionaries.
    """
    print("Creating and persisting the vector store...")
    # Initialize a persistent client, which will save data to disk
    client = chromadb.PersistentClient(path=VECTOR_STORE_PATH)

    # If a collection with the same name already exists, delete it to start fresh
    if COLLECTION_NAME in [c.name for c in client.list_collections()]:
        print(f"Collection '{COLLECTION_NAME}' already exists. Deleting it.")
        client.delete_collection(name=COLLECTION_NAME)

    # Create a new collection
    collection = client.create_collection(name=COLLECTION_NAME)

    # Add the data to the collection in batches to avoid errors
    for i in range(0, len(chunks), BATCH_SIZE):
        end_index = min(i + BATCH_SIZE, len(chunks))
        
        print(f"Adding batch {i // BATCH_SIZE + 1} of {len(chunks) // BATCH_SIZE + 1} to the collection...")
        
        collection.add(
            embeddings=embeddings[i:end_index],
            documents=chunks[i:end_index],
            metadatas=metadata[i:end_index],
            ids=[f"chunk_{j}" for j in range(i, end_index)]
        )

    print(f"\nVector store created successfully at '{VECTOR_STORE_PATH}'.")
    print(f"Collection '{COLLECTION_NAME}' contains {collection.count()} items.")

def main():
    """
    Main function to run the entire data processing and vector store creation pipeline.
    """
    # Step 1: Load the data
    complaints_df = load_data(DATA_PATH)
    
    # Step 2: Create text chunks and metadata
    chunks, metadata = create_text_chunks(complaints_df)
    
    # Step 3: Generate embeddings for the chunks
    embeddings = generate_embeddings(chunks, EMBEDDING_MODEL_NAME)
    
    # Step 4: Create and save the vector store
    create_and_persist_vector_store(chunks, embeddings, metadata)

if __name__ == "__main__":
    main()
