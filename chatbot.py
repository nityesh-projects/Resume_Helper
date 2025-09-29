import streamlit as st
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

# ------------------------------
# Streamlit UI
# ------------------------------
st.title("📄 Resume Helper Bot")
st.write("Upload your resume and paste a job description. The bot will suggest missing keywords.")

uploaded_file = st.file_uploader("Upload your Resume (PDF)", type=["pdf"])
job_description = st.text_area("Paste the Job Description here")

analyze_button = st.button("Analyze Resume")

# ------------------------------
# Backend Pipeline
# ------------------------------
if analyze_button and uploaded_file is not None and job_description.strip() != "":
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_file_path = tmp_file.name

    # 1. Load PDF
    loader = PyPDFLoader(temp_file_path)
    documents = loader.load()

    # 2. Split text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=70
    )
    docs = text_splitter.split_documents(documents)

    # 3. Embeddings + Vector Store
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name="resume_collection",
        persist_directory="./chroma_db"
    )

    # 4. Similarity Search
    results = vectordb.similarity_search(job_description, k=5)
    resume_content = " ".join([r.page_content for r in results])

    # 5. LLM with Prompt
    llm = ChatOpenAI()
    template = PromptTemplate(
        input_variables=["job_description", "resume_content"],
        template=(
            "Given the job description: {job_description} "
            "and the resume content: {resume_content}, "
            "list the key skills and keywords that are already present in the resume, "
            "and also list the important keywords that are missing and should be added."
        )
    )

    prompt = template.format(
        job_description=job_description,
        resume_content=resume_content
    )

    response = llm.invoke(prompt)

    # ------------------------------
    # Display Results
    # ------------------------------
    st.subheader("🔍 Resume Analysis")
    st.write(response.content)

elif analyze_button:
    st.warning("⚠️ Please upload a PDF and paste a job description before analyzing.")
