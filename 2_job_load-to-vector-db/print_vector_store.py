import os
from llama_index.core.storage import StorageContext
from llama_index.core.indices.loading import load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.settings import Settings

# Define persistent storage directory
PERSIST_DIR = "./embeddings"

# Set up HuggingFace Embedding
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Update global settings
Settings.embed_model = embed_model

# Disable default LLM in llama_index to avoid OpenAI usage
Settings.llm = None

def print_vector_store_contents():
    # Check if the persist directory exists
    if not os.path.exists(PERSIST_DIR):
        print(f"Persist directory '{PERSIST_DIR}' does not exist.")
        return
    
    # Load storage context
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    
    # Load index from storage
    index = load_index_from_storage(storage_context)
    
    # Print each document's content and metadata
    for doc_id, document in index.docstore.docs.items():
        # Fetch metadata and text content
        filename = document.metadata.get('filename', 'Unknown')
        content = document.get_content()
        
        # Debugging: Print content length and check for empty content
        content_length = len(content.strip()) if content else 0
        print(f"Document ID: {doc_id}")
        print(f"Filename: {filename}")
        print(f"Content Length: {content_length} characters")
        
        if content_length == 0:
            print(f"[WARNING] No content found for document '{filename}'.")
        else:
            # Print the first 500 characters for verification
            print(f"Content: {content[:500]}...")
        
        print("=" * 80)

if __name__ == "__main__":
    print_vector_store_contents()
