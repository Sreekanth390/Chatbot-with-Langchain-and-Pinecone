from langchain_core.output_parsers import StrOutputParser

from llm import llm
from vector_store import retriever
from prompt import RAG_PROMPT


# =====================================================
# CREATE RAG CHAIN
# =====================================================

chain = RAG_PROMPT | llm | StrOutputParser()


# =====================================================
# EXPLAIN TOPIC
# =====================================================

def explain_topic(topic):

    # ---------------------------------------------
    # Validate input
    # ---------------------------------------------

    if not topic or not topic.strip():
        return "Please enter a topic."


    # ---------------------------------------------
    # Retrieve documents from Pinecone
    # ---------------------------------------------

    documents = retriever.invoke(topic)


    # ---------------------------------------------
    # Check if documents were found
    # ---------------------------------------------

    if not documents:
        return (
            "I could not find this information "
            "in the provided knowledge base."
        )


    # ---------------------------------------------
    # Create context
    # ---------------------------------------------

    context = "\n\n".join(
        document.page_content
        for document in documents
    )


    # ---------------------------------------------
    # Send context + question to Ollama
    # ---------------------------------------------

    answer = chain.invoke({
        "topic": topic,
        "context": context
    })


    return answer
