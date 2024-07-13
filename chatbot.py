import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from docx import Document
import csv

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_document_text(files):
    text = ""
    for file in files:
        if file.name.lower().endswith('.pdf'):
            pdf_reader = PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
        elif file.name.lower().endswith('.docx'):
            doc = Document(file)
            for para in doc.paragraphs:
                text += para.text + "\n"
        elif file.name.lower().endswith('.csv'):
            file.seek(0)  # Go to the start of the file (necessary if iterating over the file multiple times)
            reader = csv.reader(file.read().decode('utf-8').splitlines())
            for row in reader:
                text += ', '.join(row) + "\n"
        else:
            print(f"Skipping unsupported file format: {file.name}")
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    try:
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    except RuntimeError as e:
        st.error(f"Failed to load FAISS index: {e}")
        return
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.write("Reply: ", response["output_text"])

def main():
    st.set_page_config(page_title="Chat PDF", page_icon="📄", layout="wide")

    st.markdown("""
    <style>
    .big-font {
        font-size:20px !important;
        font-weight: bold;
    }
    .header-style {
        font-size:30px !important;
        font-weight: bold;
        color: #4A4AFF;
    }
    .streamlit-container {
        margin-top: 2rem;
    }
    .stButton>button {
        font-size: 20px;
        border-radius: 20px 20px;
        border: 2px solid #4A4AFF;
        color: #FFFFFF;
        background-color: #4A4AFF;
    }
    .stTextInput>div>div>input {
        font-size: 18px;
    }
    .stFileUploader>div>div>span>button {
        font-size: 16px;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<p class="header-style">📄 Chat with PDF  💁‍♂️</p>', unsafe_allow_html=True)
    col1, col2 = st.columns([2, 3])

    with col1:
        st.markdown('<p class="big-font">Ask a Question:</p>', unsafe_allow_html=True)
        user_question = st.text_input("", placeholder="Type your question here...", help="Type your question and press enter.")
        if user_question:
            user_input(user_question)

    with col2:
        st.markdown('<p class="big-font">Menu:</p>', unsafe_allow_html=True)
        pdf_docs = st.file_uploader("", accept_multiple_files=True, help="Upload PDFs from which you want to fetch answers.", type=['pdf'])
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_document_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done. Ask away!")

if __name__ == "__main__":
    main()
