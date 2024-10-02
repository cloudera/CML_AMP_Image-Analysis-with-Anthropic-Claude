import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.storage import StorageContext
from llama_index.core.indices.loading import load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Document

# Define persistent storage directory
PERSIST_DIR = "./embeddings"
DATA_DIR = "./docs"
INDEX_FILE = os.path.join(PERSIST_DIR, "docstore.json")  # File expected for index

embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

def get_metadata(file_path):
    """Function to attach metadata (e.g., filename) to each document."""
    return {"filename": os.path.basename(file_path)}

def create_or_load_index():
    # Check if index file exists in PERSIST_DIR
    if not os.path.exists(PERSIST_DIR) or not os.path.exists(INDEX_FILE):
        print("Index file not found. Creating new index...")
        
        # Load all documents from the specified directory
        reader = SimpleDirectoryReader(input_dir=DATA_DIR, file_metadata=get_metadata)
        documents = reader.load_data()
        
        # Debugging: print document contents and metadata
        for doc in documents:
            print(f"[DEBUG] Loaded document '{doc.metadata['filename']}' with content length: {len(doc.text)} characters")
        
        # Create vector index
        index = VectorStoreIndex.from_documents(documents, show_progress=True, embed_model=embed_model)
        
        # Create PERSIST_DIR if not exists
        if not os.path.exists(PERSIST_DIR):
            os.makedirs(PERSIST_DIR)

        # Persist the index to PERSIST_DIR
        index.storage_context.persist(persist_dir=PERSIST_DIR)
        print(f"[DEBUG] Index persisted to {PERSIST_DIR}")
        
    else:
        print(f"Loading existing index from {PERSIST_DIR}...")
        # Load the index from PERSIST_DIR if it exists
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        # Pass the embed_model explicitly when loading the index
        index = load_index_from_storage(storage_context, embed_model=embed_model)
        print("[DEBUG] Index loaded successfully.")

    return index

def main():
    index = create_or_load_index()
    print("[INFO] Index is ready for use.")

if __name__ == "__main__":
    main()
