from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import os
import tempfile
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings import HuggingFaceEmbeddings

app = FastAPI()

@app.post("/api/ask")
async def ask_question(files: list[UploadFile] = File(...), question: str = Form(...)):
    docs = []
    for file in files:
        ext = file.filename.split('.')[-1]
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.'+ext)
        tmp.write(await file.read())
        tmp.close()
        if ext == "pdf":
            with pdfplumber.open(tmp.name) as pdf:
                text = "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())
        else:
            with open(tmp.name, "r", encoding="utf-8") as f:
                text = f.read()
        docs.append(text)
    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)
    langchain_docs = splitter.create_documents(docs)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )

    db = Chroma.from_documents(langchain_docs, embedding=embeddings, persist_directory="./chroma_db")
    retriever = db.as_retriever()
    os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
    llm = ChatGroq(temperature=0, model_name="llama3-70b-8192")
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=retriever, return_source_documents=True,
    )
    result = qa_chain({"question": question, "chat_history": []})
    return JSONResponse(content={
        "answer": result["answer"],
        "sources": [doc.page_content for doc in result["source_documents"]]
    })
