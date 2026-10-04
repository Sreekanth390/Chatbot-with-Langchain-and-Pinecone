import streamlit as st

from rag_service import explain_topic


st.set_page_config(
    page_title="GEN AI Student Assistant",
    page_icon="🤖"
)


st.title("GEN AI Student Assistant")

st.write(
    "Ask a question based on the knowledge base."
)


topic = st.text_input(
    "Enter your question"
)


if st.button("Ask"):

    if not topic.strip():

        st.warning(
            "Please enter a question."
        )

    else:

        with st.spinner("Thinking..."):

            answer = explain_topic(topic)

        st.write(answer)