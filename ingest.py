from langchain_pinecone import PineconeVectorStore

from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

from pinecone import Pinecone, ServerlessSpec
from llm import embeddings

from config import (
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    PINECONE_DIMENSION,
)


# =====================================================
# 1. INITIALIZE PINECONE
# =====================================================

pc = Pinecone(
    api_key=PINECONE_API_KEY
)


# =====================================================
# 2. CREATE INDEX IF NOT AVAILABLE
# =====================================================

existing_indexes = [
    index["name"]
    for index in pc.list_indexes()
]


if PINECONE_INDEX_NAME not in existing_indexes:

    print(
        f"Index '{PINECONE_INDEX_NAME}' not found."
    )

    print(
        f"Creating index '{PINECONE_INDEX_NAME}'..."
    )

    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=PINECONE_DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

    print("Index created successfully.")

else:

    print(
        f"Index '{PINECONE_INDEX_NAME}' already exists."
    )


# =====================================================
# 3. LOAD Files
# =====================================================

pdf_loader = DirectoryLoader(
    "documents",
    glob="**/*.pdf",
    loader_cls=PyPDFLoader
)

txt_loader = DirectoryLoader(
    "documents",
    glob="**/*.txt",
    loader_cls=TextLoader,
    loader_kwargs={
        "encoding": "utf-8"
    }
)

pdf_documents = pdf_loader.load()
txt_documents = txt_loader.load()

documents = pdf_documents + txt_documents

print(
    "Total pages:",
    len(documents)
)


# =====================================================
# 4. SPLIT DOCUMENT
# =====================================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(
    documents
)

print(
    "Total chunks:",
    len(chunks)
)


# =====================================================
# 5. STORE DOCUMENTS IN PINECONE
# =====================================================

vector_store = PineconeVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings,
    index_name=PINECONE_INDEX_NAME
)


print(
    "Documents successfully stored in Pinecone."
)