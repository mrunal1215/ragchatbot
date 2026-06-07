import os
from dotenv import load_dotenv
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# -------------------
# Load API Key
# -------------------

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# -------------------
# Streamlit UI
# -------------------

st.set_page_config(
    page_title="Multi PDF RAG Chatbot",
    page_icon="🤖"
)

st.title("🤖 Multi PDF RAG Chatbot")

# -------------------
# Load PDFs
# -------------------

@st.cache_resource
def build_rag():

    pdf_files = [
        "data/travel.pdf",
        "data/movie.pdf",
        "data/yuvi.pdf"
    ]

    documents = []

    for pdf in pdf_files:

        loader = PyPDFLoader(pdf)

        docs = loader.load()

        for doc in docs:
            doc.metadata["source"] = pdf

        documents.extend(docs)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=50
    )

    split_docs = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(
        split_docs,
        embeddings
    )

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        google_api_key=GOOGLE_API_KEY
    )

    prompt = ChatPromptTemplate.from_template(
        """
Answer the question based only on the context.

Context:
{context}

Question:
{question}
"""
    )

    def format_docs(docs):
        return "\n\n".join(
            doc.page_content
            for doc in docs
        )

    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain

# -------------------
# Build Chain Once
# -------------------

rag_chain = build_rag()

# -------------------
# Chat UI
# -------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):
        st.write(msg["content"])

question = st.chat_input(
    "Ask a question from PDFs..."
)

if question:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):
        st.write(question)

    answer = rag_chain.invoke(question)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )

    with st.chat_message("assistant"):
        st.write(answer)