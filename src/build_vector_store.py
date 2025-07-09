import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
import os

# --- 1. Load the Cleaned Data ---
try:
    df = pd.read_csv('data/filtered_complaints.csv')
    print("Cleaned dataset loaded successfully.")
except FileNotFoundError:
    print("Error: 'data/filtered_complaints.csv' not found. Please run the EDA notebook first.")
    exit()

# --- 2. Implement Text Chunking Strategy ---
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
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
        metadata.append({
            'complaint_id': str(row.get('Complaint ID', '')),
            'product': str(row.get('Product', '')),
            'date_received': str(row.get('Date received', ''))
        })

print(f"Created {len(chunks)} text chunks.")


# --- 3. Choose and Use an Embedding Model ---
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
print("Embedding model loaded.")

embeddings = embedding_model.encode(chunks, show_progress_bar=True)
print(f"Generated {len(embeddings)} embeddings.")


# --- 4. Create and Persist the Vector Store (with Batching) ---
vector_store_path = "vector_store"
client = chromadb.PersistentClient(path=vector_store_path)

collection_name = "complaint_embeddings"
if collection_name in [c.name for c in client.list_collections()]:
    client.delete_collection(name=collection_name)

collection = client.create_collection(name=collection_name)

# ** CORRECTED BATCHING LOGIC **
# Define the batch size
batch_size = 4000 # A safe number well below the limit of 5461

# Loop through the data in batches
for i in range(0, len(chunks), batch_size):
    # Find the end of the batch
    end_index = min(i + batch_size, len(chunks))
    
    # Get the batch slices
    chunk_batch = chunks[i:end_index]
    embedding_batch = embeddings[i:end_index]
    metadata_batch = metadata[i:end_index]
    ids_batch = [f"chunk_{j}" for j in range(i, end_index)]
    
    # Add the batch to the collection
    collection.add(
        embeddings=embedding_batch,
        documents=chunk_batch,
        metadatas=metadata_batch,
        ids=ids_batch
    )
    
    print(f"Added batch {i // batch_size + 1} to the collection.")


print(f"\nVector store created successfully at '{vector_store_path}'.")
print(f"Collection '{collection_name}' contains {collection.count()} items.")

