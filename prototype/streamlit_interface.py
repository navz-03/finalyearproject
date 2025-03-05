import streamlit as st
from datapipeline_rag import *
from dotenv import load_dotenv
from RAG import *
load_dotenv()

LANGSMITH_TRACING = True
LANGSMITH_ENDPOINT="https://api.smith.langchain.com"
LANGSMITH_API_KEY="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"
LANGSMITH_PROJECT="legal rag"


st.title("LEGAL DOC SUMMARIZER")
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

initial_message = """
    Hi there! I'm your legalBot 🤖
"""

# Initialize session state variables
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": initial_message}]

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": initial_message}]



# User-provided prompt
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Hold on, I'm fetching the details for you"):
            # query_text = str(prompt) if isinstance(prompt, str) else " ".join(prompt)
            query_embedding = embeddings.embed_query(prompt)
            response = get_response(query_embedding,prompt)
            placeholder = st.empty()
            full_response = response # Directly use the response
            placeholder.markdown(full_response)
    message = {"role": "assistant", "content": full_response}
    st.button('Clear Chat', on_click=clear_chat_history)
    st.session_state.messages.append(message)   