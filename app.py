import os
import streamlit as st
import tempfile
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings import HuggingFaceEmbeddings

# Force HuggingFace embedding to CPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""

st.set_page_config(page_title="Mojo-QA Bot", layout="wide")

# --- SIDEBAR: SETTINGS & NEW CHAT BUTTON ---
st.sidebar.title("Settings")

# New Chat button (placed above model selection)
if st.sidebar.button("➕ New Chat", key="new_chat"):
    st.session_state.chat_history = []
    st.session_state.generated_responses = []
    st.session_state.last_prompt = None
    st.experimental_rerun()

with st.sidebar.expander("💬 Chat History", expanded=True):
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if st.session_state.chat_history:
        for i, (q, a) in enumerate(st.session_state.chat_history, 1):
            st.markdown(f"**{i}. You:** {q}")
            st.markdown(f"**{i}. Bot:** {a}")
        if st.button("Clear Chat History", key="clear_chat"):
            st.session_state.chat_history = []
            st.experimental_rerun()
    else:
        st.write("No chat history yet.")

llm_choice = st.sidebar.selectbox("LLM Model", ["llama3-8b-8192", "llama3-70b-8192"])
temperature = st.sidebar.slider("LLM Temperature", 0.0, 1.0, 0.0, 0.05)
chunk_size = st.sidebar.slider("Chunk Size", 256, 2000, 512, 64)
chunk_overlap = st.sidebar.slider("Chunk Overlap", 0, 512, 64, 16)

# ---- GROQ API KEY (from secrets.toml or env var) ----
if "GROQ_API_KEY" in st.secrets:
    API_KEY = os.getenv("GOOGLE_API_KEY")
else:
    API_KEY = os.getenv("GOOGLE_API_KEY")
os.getenv("GOOGLE_API_KEY")

st.title("🧠 Mojo-QA: RAG Bot using LangChain + GroqCloud")

if "generated_responses" not in st.session_state:
    st.session_state.generated_responses = []
if "last_prompt" not in st.session_state:
    st.session_state.last_prompt = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None

# ---- DOCUMENT INGESTION ----
uploaded_files = st.file_uploader("Upload PDF/TXT files", type=["pdf", "txt"], accept_multiple_files=True)
docs = []
if uploaded_files:
    with st.spinner("Processing files and building knowledge base..."):
        for file in uploaded_files:
            ext = file.name.split('.')[-1].lower()
            with tempfile.NamedTemporaryFile(delete=False, suffix='.'+ext) as tmp:
                tmp.write(file.read())
                tmp_path = tmp.name
            if ext == "pdf":
                with pdfplumber.open(tmp_path) as pdf:
                    text = "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())
            else:
                with open(tmp_path, "r", encoding="utf-8") as f:
                    text = f.read()
            docs.append(text)
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        langchain_docs = splitter.create_documents(docs)
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        db = Chroma.from_documents(langchain_docs, embedding=embeddings, persist_directory="./chroma_db")
        st.session_state.retriever = db.as_retriever()
    st.success("Knowledge base ready! Ask anything below 👇")
else:
    st.session_state.retriever = None

# ---- LLM CHAIN (function) ----
def get_qa_chain(retriever, temp):
    llm = ChatGroq(temperature=temp, model_name=llm_choice)
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
    )

# ---- CHAT UI ----
prompt = st.chat_input("Ask a question about your uploaded documents...")

# Show chat history in main area
for i, (q, a) in enumerate(st.session_state.chat_history):
    st.markdown(f"**You:** {q}")
    st.markdown(f"**Bot:** {a}")

# Generate 3 diverse answers if prompt and retriever are present and not already generated
if prompt and st.session_state.retriever and not st.session_state.generated_responses:
    st.session_state.last_prompt = prompt
    st.session_state.generated_responses = []
    diverse_temperatures = [0.2, 0.5, 0.9]
    with st.spinner("Generating 3 possible answers..."):
        for i, temp in enumerate(diverse_temperatures):
            qa_chain = get_qa_chain(st.session_state.retriever, temp)
            res = qa_chain({
                "question": prompt,
                "chat_history": st.session_state.chat_history,
            })
            answer = res["answer"]
            sources = [doc.page_content for doc in res["source_documents"]]
            st.session_state.generated_responses.append((answer, sources))

# Show response cards for latest question, with "I prefer this" button
if st.session_state.generated_responses:
    st.subheader("Select your preferred answer:")
    cols = st.columns(3)
    for idx, (answer, sources) in enumerate(st.session_state.generated_responses):
        with cols[idx]:
            with st.expander(f"Response {idx+1} (Click to expand)", expanded=False):
                st.markdown(f"**Answer {idx+1}:** {answer}")
                if st.button(f"I prefer this", key=f"choose_{idx}"):
                    st.session_state.chat_history.append((st.session_state.last_prompt, answer))
                    st.session_state.generated_responses = []
                    st.session_state.last_prompt = None
                    st.experimental_rerun()
                with st.expander("Show Sources"):
                    for sidx, src in enumerate(sources):
                        st.write(f"**Source {sidx+1}:**", src)

else:
    if prompt and not st.session_state.retriever:
        st.warning("Please upload documents first!")
