from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings

from config import (
    LLM_MODEL,
    EMBEDDING_MODEL,
    OLLAMA_BASE_URL
)


# =====================================================
# OLLAMA LLM
# =====================================================

llm = ChatOllama(
    model=LLM_MODEL,
    base_url=OLLAMA_BASE_URL,
    temperature=0
)


# =====================================================
# OLLAMA EMBEDDINGS
# =====================================================

embeddings = OllamaEmbeddings(
    model=EMBEDDING_MODEL,
    base_url=OLLAMA_BASE_URL
)