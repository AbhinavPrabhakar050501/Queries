# General imports
import os
import requests
import streamlit as st
from streamlit_lottie import st_lottie
from langchain.llms import OpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.agents.agent_toolkits import create_vectorstore_agent, VectorStoreToolkit, VectorStoreInfo
import PyPDF2

# Set OpenAI API key and initialize language model
os.environ['OPENAI_API_KEY'] = "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"
llm = OpenAI(temperature=0.1, verbose=True)
embeddings = OpenAIEmbeddings()

# Set directory for storing PDF files
dir = "C:/Users/abhin/Downloads/Documents/Pwc Project/pdfs"

# Function to store PDF file
def store_pdf_file(file, dir):
    file_path = os.path.join(dir, file.name)
    with open(file_path, 'wb') as fhand:
        fhand.write(file.getbuffer())
    return file_path

#Function to load Lottie animation file from URL
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Load Lottie animation file
lottie_file = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_5rImXbDsO1.json")

# Configure Streamlit
st.set_page_config(page_title="Queries")
st_lottie(lottie_file, height=300, key='coding')

# Display title
st.title("**Queries: Your PDF Analysis Tool**")

if 'uploaded' not in st.session_state:
    st.session_state['uploaded'] = False
    st.session_state['filename'] = None

if 'agent_executor' not in st.session_state:
    st.session_state['agent_executor'] = None
    st.session_state['store'] = None

if not st.session_state['uploaded']:
    st.write("Upload your **:blue[pdf]** files and ask your personal AI assistant any questions about them!")
    input_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)

    if input_files and all([file is not None for file in input_files]):
        with st.spinner("Analyzing documents..."):
            merged_pdf_path = os.path.join(dir, "merged.pdf")
            merged_pdf = PyPDF2.PdfMerger()

            for input_file in input_files:
                if input_file is not None:
                    path = store_pdf_file(input_file, dir)
                    merged_pdf.append(path)

            merged_pdf.write(merged_pdf_path)

            loader = PyPDFLoader(merged_pdf_path)
            pages = loader.load_and_split()
            store = Chroma.from_documents(pages, embeddings, collection_name="analysis")
            vectorstore_info = VectorStoreInfo(name="PDF Analysis", description="Analyzing PDFs", vectorstore=store)
            toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info)
            agent_executor = create_vectorstore_agent(llm=llm, toolkit=toolkit, verbose=True)
            st.session_state['store'] = store
            st.session_state['agent_executor'] = agent_executor
            st.session_state['uploaded'] = True

if st.session_state['uploaded']:
    st.write("Enter your questions about the documents below:")
    prompt = st.text_input("Type your query")

    if prompt:
        agent_executor = st.session_state['agent_executor']
        store = st.session_state['store']
        with st.spinner("Generating response..."):
            response = agent_executor(prompt)
            st.write(response["output"])
            with st.expander("Similarity Search"):
                search = store.similarity_search_with_score(prompt)
                st.write(search[0][0].page_content)
