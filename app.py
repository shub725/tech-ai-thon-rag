import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import os

# 1. UI Setup
st.set_page_config(page_title="AI Document Q&A", layout="wide")
st.title("📄 AI-Powered Document Q&A System (RAG-lite)")
st.write("Upload a document and ask questions. The AI will only answer based on the provided text.")

# 2. Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("Enter Google Gemini API Key", type="password")
    st.header("📂 Document Upload")
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

# 3. Main Application Logic
if uploaded_file and api_key:
    os.environ["GOOGLE_API_KEY"] = api_key
    
    with st.spinner("Processing Document..."):
        # Step A: Extract Text
        pdf_reader = PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
            
        # Step B: Chunk the Text
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        
        # Step C: Create Embeddings and Vector Store using Google
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_texts(chunks, embeddings)
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        
        # Step D: Engineered Prompt
        prompt_template = """Use the following pieces of context to answer the user's question. 
        If you don't know the answer based on the context, just say that you don't know, don't try to make up an answer.
        
        Context: {context}
        Question: {question}
        
        Answer explicitly based on the text provided:"""
        PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        
        # Step E: QA Chain using Gemini 1.5 Flash
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
        
    st.success("✅ System Ready! Ask your questions below.")
    
    # 4. Chat Interface
    query = st.text_input("Ask a question about your document:")
    if query:
        with st.spinner("Retrieving answer..."):
            result = qa_chain.invoke({"query": query})
            
            st.subheader("🤖 Answer:")
            st.info(result["result"])
            
            # Show the chunks used to generate the answer
            with st.expander("🔍 View Source References (Context Used)"):
                for i, doc in enumerate(result["source_documents"]):
                    st.markdown(f"**Source {i+1}:** {doc.page_content[:400]}...")

elif not api_key:
    st.warning("👈 Please enter your Google Gemini API Key in the sidebar to start.")