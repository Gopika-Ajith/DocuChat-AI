import streamlit as st
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from groq import Groq
from dotenv import load_dotenv
import os

# ----------------------------------
# Load Environment
# ----------------------------------

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

# ----------------------------------
# Page Config
# ----------------------------------

st.set_page_config(
    page_title="DocuChat AI",
    page_icon="📚",
    layout="wide"
)

# ----------------------------------
# Session State
# ----------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

# ----------------------------------
# Header
# ----------------------------------

st.title("📚 DocuChat AI")
st.caption("AI-Powered Document Search Assistant")

# ----------------------------------
# Sidebar
# ----------------------------------

with st.sidebar:

    st.header("📄 Document Info")

    if "total_pages" in st.session_state:
        st.write(f"Total Pages: {st.session_state.total_pages}")

    if "pdf_names" in st.session_state:

        st.write("### Files")

        for name in st.session_state.pdf_names:
            st.write(f"• {name}")

    st.divider()

    if st.button("🗑 Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# ----------------------------------
# Upload PDFs
# ----------------------------------

uploaded_files = st.file_uploader(
    "Upload PDF Files",
    type="pdf",
    accept_multiple_files=True
)

# ----------------------------------
# Build Knowledge Base
# ----------------------------------

@st.cache_resource
def create_vector_store(documents):

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = FAISS.from_documents(
        documents,
        embeddings
    )

    return vector_store

# ----------------------------------
# Process PDFs
# ----------------------------------

if uploaded_files:

    st.session_state.pdf_names = [
        file.name for file in uploaded_files
    ]

    documents = []
    total_pages = 0
    extracted_text = ""

    for uploaded_file in uploaded_files:

        pdf_reader = PdfReader(uploaded_file)

        total_pages += len(pdf_reader.pages)

        for page_num, page in enumerate(
            pdf_reader.pages,
            start=1
        ):

            text = page.extract_text()

            if text:

                extracted_text += text + "\n"

                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "page": page_num,
                            "file": uploaded_file.name
                        }
                    )
                )

    st.session_state.total_pages = total_pages

    st.download_button(
        "⬇ Download Extracted Text",
        extracted_text,
        file_name="document_text.txt",
        mime="text/plain"
    )

    with st.expander("📄 View Extracted Text"):
        st.write(extracted_text[:3000])
    
    if st.button("📄 Summarize Document"):

        with st.spinner("Generating summary..."):

            summary_prompt = f"""
            Summarize the following document.

            Keep it concise.
            Use bullet points.
            Highlight important topics.

            Document:
            {extracted_text[:12000]}
            """

            summary_response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "user",
                        "content": summary_prompt
                    }
                ]
            )

            summary = summary_response.choices[0].message.content

            st.subheader("📋 Document Summary")
            st.write(summary)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100
    )

    split_docs = splitter.split_documents(
        documents
    )

    vector_store = create_vector_store(
        split_docs
    )

    # ----------------------------------
    # Welcome Screen
    # ----------------------------------

    if len(st.session_state.messages) == 0:

        st.info(
            "👋 Ask anything about your uploaded documents."
        )

    # ----------------------------------
    # Display Chat
    # ----------------------------------

    for msg in st.session_state.messages:

        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # ----------------------------------
    # Chat Input
    # ----------------------------------

    query = st.chat_input(
        "Ask something about your document..."
    )

    if query:

        st.session_state.messages.append(
            {
                "role": "user",
                "content": query
            }
        )

        with st.chat_message("user"):
            st.write(query)

        with st.spinner("🤖 Thinking..."):

            docs = vector_store.similarity_search(
                query,
                k=3
            )

            context = "\n\n".join(
                [doc.page_content for doc in docs]
            )

            source_page = docs[0].metadata["page"]

            prompt = f"""
You are a document assistant.

Answer ONLY using the information provided.

Context:
{context}

Question:
{query}

Provide a clean and concise answer.
"""

            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

            answer = response.choices[0].message.content

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )

        with st.chat_message("assistant"):

            st.markdown("### 📖 Answer")
            st.write(answer)

            st.caption(
                f"📍 Source Page: {source_page}"
            )