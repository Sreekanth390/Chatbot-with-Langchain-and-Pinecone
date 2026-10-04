import os
from dotenv import load_dotenv

load_dotenv()


# =====================================================
# PINECONE CONFIGURATION
# =====================================================

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

PINECONE_INDEX_NAME = "ashok-it-knowledge"


# =====================================================
# OLLAMA CONFIGURATION
# =====================================================

OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://localhost:11434"
)

LLM_MODEL = "llama3.2"

EMBEDDING_MODEL = "nomic-embed-text"


# =====================================================
# PINECONE VECTOR DIMENSION
# =====================================================

# nomic-embed-text produces 768-dimensional embeddings.
PINECONE_DIMENSION = 768