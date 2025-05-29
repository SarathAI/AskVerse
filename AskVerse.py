import os
import time
import openai
import base64
import config
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

openai.api_key = st.secrets["OPENAI_API_KEY"]

from langchain_community.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import speech_recognition as sr
import tempfile
import uuid

# Ensure directories exist
os.makedirs(config.upload_path, exist_ok=True)
os.makedirs(config.vector_DB_path, exist_ok=True)

@st.cache_resource
def load_chroma_db():
    embeddings = OpenAIEmbeddings()
    return Chroma(persist_directory=config.vector_DB_path, embedding_function=embeddings)

# Mic-based Speech to Text using microphone input
def speech_to_text_from_mic():
    recognizer = sr.Recognizer()
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            with sr.Microphone() as source:
                # Adjust for ambient noise
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
                
                # Create a placeholder for real-time feedback
                status_placeholder = st.empty()
                status_placeholder.info("🎙️ Listening... Speak now!")
                
                # Add timeout to prevent hanging
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=15)
                
                status_placeholder.success("🛑 Processing audio...")
                
                # Attempt to recognize speech
                text = recognizer.recognize_google(audio)
                status_placeholder.empty()
                return text
                
        except sr.WaitTimeoutError:
            status_placeholder.error("No speech detected. Please try again.")
            time.sleep(1)
            retry_count += 1
            continue
        except sr.UnknownValueError:
            status_placeholder.error("Could not understand audio. Please try again.")
            time.sleep(1)
            retry_count += 1
            continue
        except Exception as e:
            status_placeholder.error(f"Error: {str(e)}")
            time.sleep(1)
            retry_count += 1
            continue
    
    return "Error: Maximum retries reached. Please try again."

# Initialize session state for OpenAI model
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-4-32k-0314"

# Custom CSS for professional UI
def set_professional_theme():
    st.markdown("""
        <style>
            /* Global Styles */
            body {
                font-family: 'Inter', sans-serif !important;
                background-color: #f9fafb;
                color: #2d3748;
                margin: 0;
                padding: 0;
            }

            /* Page Frame */
            .page-frame {
                display: flex;
                //flex-direction: column;
                background-color: #1e40af;
                box-shadow: 0 0 20px rgba(0,0,0,0.1);
                max-width: 1400px;
                margin: 0 auto;
                border-radius: 12px;
                overflow: hidden;
            }

            /* Header (Top Navigation Bar) */
            .top-nav {
                background-color: #1e40af; /* Vibrant blue */
                padding: 16px 32px;
                display: flex;
                justify-content: space-between;
                align-items: center;
                border-bottom: 2px solid #1e3a8a;
            }
            .top-nav .brand {
                font-size: 28px;
                font-weight: 700;
                color: #ffffff;
            }

            /* Footer */
            .footer {
                background-color: #1f2937; /* Dark slate */
                color: #e5e7eb;
                padding: 16px 32px;
                text-align: right;
                border-top: 2px solid #374151;
            }
            .footer p {
                margin: 0;
                font-size: 14px;
                font-weight: 400;
            }

            /* Section Styling */
            .section {
                margin-bottom: 40px;
                padding: 24px;
                background-color: #ffffff;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            }

            /* Headings */
            h1, h2, h3 {
                color: #1e40af;
                font-weight: 600;
            }
            h1 {
                font-size: 36px;
                margin-bottom: 20px;
            }
            h2 {
                font-size: 28px;
            }

            /* Buttons */
            .stButton > button {
                background-color: #1e40af;
                color: white;
                border-radius: 8px;
                padding: 12px 24px;
                font-weight: 500;
                border: none;
                transition: background-color 0.3s;
            }
            .stButton > button:hover {
                background-color: #1e3a8a;
            }

            /* File Uploader */
            .stFileUploader {
                border: 2px dashed #d1d5db;
                border-radius: 8px;
                padding: 16px;
                background-color: #ffffff;
            }

            /* Chat Interface */
            .stChatMessage {
                background-color: #ffffff;
                padding: 16px;
                margin-bottom: 12px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            }
            .stChatMessage.user {
                background-color: #dbeafe;
            }

            /* Chat Container */
            .chat-container {
                max-height: 500px;
                overflow-y: auto;
                padding: 16px;
                border: 1px solid #d1d5db;
                border-radius: 8px;
                background-color: #ffffff;
                margin-bottom: 16px;
            }

            /* PDF Viewer */
            iframe {
                border: 1px solid #d1d5db;
                border-radius: 8px;
                width: 100%;
                height: 600px;
            }

            /* Responsive Design */
            @media (max-width: 768px) {
                .page-frame {
                    border-radius: 0;
                }
                .top-nav {
                    flex-direction: column;
                    align-items: flex-start;
                    padding: 12px 16px;
                }
                .footer {
                    padding: 12px 16px;
                    text-align: center;
                }
            }
        </style>
    """, unsafe_allow_html=True)

# Display PDF
def show_pdf(file_path):
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

# Save uploaded PDF
def save_uploaded_pdf(uploaded_file, save_path):
    with open(save_path, mode='wb') as w:
        w.write(uploaded_file.getvalue())
        if save_path.exists():
            st.success(f'File {uploaded_file.name} successfully uploaded and saved!')

# Load PDF
def load_pdf():
    flag = False
    pdf_name = ''
    pdf_file_path = ''

    st.subheader("Upload Your PDF")
    option = st.selectbox('Select PDF Upload Option', ('Select an option', 'Single PDF', 'Multiple PDFs'), index=0)
    st.markdown("""
    </div>
    """, unsafe_allow_html=True)
    if option == 'Single PDF':
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
        if uploaded_file is not None:
            flag = True
            pdf_name = uploaded_file.name
            pdf_file_path = Path(config.upload_path, uploaded_file.name)
            save_uploaded_pdf(uploaded_file, pdf_file_path)
            return pdf_file_path, pdf_name, flag
        else:
            pdf_name = 'No_pdf_found'
            return '/folder/N0_pdf_data', pdf_name, flag

    elif option == 'Multiple PDFs':
        default_value_goes_here = config.pdf_path
        pdf_folder_path = st.text_input("Enter PDFs folder path", default_value_goes_here)
        pdf_name = ''
        flag = True
        return pdf_folder_path, pdf_name, flag

    else:
        pdf_name = 'No_pdf_found'
        return '/folder/N0_pdf_data', pdf_name, flag

# PDF Viewer
def pdf_viewer(pdf_or_folder, status):
    if status:
        st.subheader("Uploaded PDF Preview")
        st.markdown("""
            </div>
            """, unsafe_allow_html=True)
        show_pdf(pdf_or_folder)

# PDF Retriever
def pdf_retriver(pdf_name, status):
    if status:
        pdf_folder_path = os.path.join(config.upload_path, pdf_name)
        loader = PyMuPDFLoader(pdf_folder_path)
        all_pages = loader.load()

        if not os.path.exists(config.vector_DB_path):
            embeddings = OpenAIEmbeddings()
            vectordb = Chroma.from_documents(documents=all_pages, embedding=embeddings, 
                                            persist_directory=config.vector_DB_path)
            vectordb.persist()
            return True
        else:
            return True
    return False

# Chat Application
def chat_app(pdf_name):
    st.subheader("Chat with AskVerse AI")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "is_listening" not in st.session_state:
        st.session_state.is_listening = False
    if "mic_transcribed" not in st.session_state:
        st.session_state.mic_transcribed = ""

    # Clear chat button
    if st.button("🧹 Clear Chat"):
        st.session_state.messages = []
        st.success("Chat history cleared.")

    # Chat container
    with st.container():
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        st.markdown('</div>', unsafe_allow_html=True)

    # Input area with improved layout
    with st.container():
        col1, col2, col3 = st.columns([0.7, 0.15, 0.15])

        with col1:
            # Display transcribed text immediately
            if st.session_state.mic_transcribed:
                st.text_area("Transcribed Text", value=st.session_state.mic_transcribed, height=70, disabled=True)
            prompt = st.chat_input("Ask your question here!") or st.session_state.get("mic_transcribed", "")

        with col2:
            # Toggle microphone button
            mic_label = "🎙️ Stop" if st.session_state.is_listening else "🎙️ Start"
            if st.button(mic_label, help="Toggle microphone"):
                st.session_state.is_listening = not st.session_state.is_listening

        with col3:
            # Cancel button
            if st.button("🛑", help="Cancel voice input", disabled=not st.session_state.is_listening):
                st.session_state.is_listening = False
                st.session_state.mic_transcribed = ""

    # Handle voice input when listening
    if st.session_state.is_listening:
        mic_text = speech_to_text_from_mic()
        if mic_text and not mic_text.startswith("Error"):
            st.session_state.mic_transcribed = mic_text
            st.session_state.is_listening = False
            st.rerun()  # Force rerun to display transcribed text immediately
        elif mic_text.startswith("Error"):
            st.error(mic_text)
            st.session_state.is_listening = False
            st.rerun()

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        pdf_folder_path = os.path.join(config.upload_path, pdf_name)
        loader = PyMuPDFLoader(pdf_folder_path)
        all_pages = loader.load()

        embeddings = OpenAIEmbeddings()
        if os.path.exists(config.vector_DB_path):
            vectordb = load_chroma_db()
        else:
            vectordb = Chroma.from_documents(documents=all_pages, embedding=embeddings,
                                            persist_directory=config.vector_DB_path)
            vectordb.persist()

        llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0)
        qa_chain = RetrievalQA.from_chain_type(
            llm,
            retriever=vectordb.as_retriever()
        )

        result = qa_chain({"query": prompt})
        full_response = result["result"]

        st.session_state.messages.append({"role": "assistant", "content": full_response})
        with st.chat_message("assistant"):
            st.markdown(full_response)
        st.session_state.mic_transcribed = ""  # Clear transcribed text after submission

# Page Functions
def home_page():
    st.title("Welcome to AskVerse AI")
    st.markdown("""
    
    <div class="section">
        AskVerse AI is an advanced platform for interacting with educational PDFs. 
        Upload your documents, preview them, and engage in intelligent conversations 
        powered by cutting-edge AI technology.
        
        ### Why Choose AskVerse AI?
        - **Seamless PDF Interaction**: Upload and manage your educational documents with ease.
        - **Intelligent Conversations**: Ask questions about your PDFs and get precise, context-aware answers.
        - **User-Friendly Interface**: Navigate through our intuitive platform designed for students and educators.
        
        Get started by uploading your PDF in the **Upload PDF** section!
    </div>
    """, unsafe_allow_html=True)

def upload_page():
    pdf_or_folder_, pdf_name_, status_ = load_pdf()
    st.session_state.pdf_or_folder_ = pdf_or_folder_
    st.session_state.pdf_name_ = pdf_name_
    st.session_state.status_ = status_
    
    config.pdf_path_ = pdf_or_folder_
    config.pdf_name_ = pdf_name_
    config.status_ = status_

def preview_page():
    if hasattr(st.session_state, 'pdf_or_folder_') and hasattr(st.session_state, 'status_'):
        pdf_viewer(pdf_or_folder=st.session_state.pdf_or_folder_, status=st.session_state.status_)
    else:
        st.markdown('</div>', unsafe_allow_html=True)
        st.warning("Please upload a PDF first in the Upload PDF section")
        st.markdown('</div>', unsafe_allow_html=True)

def chat_page():
    if hasattr(st.session_state, 'pdf_name_') and hasattr(st.session_state, 'status_'):
        if st.session_state.status_:
            pdf_retriver(st.session_state.pdf_name_, st.session_state.status_)
            chat_app(st.session_state.pdf_name_)
        else:
            st.markdown('</div>', unsafe_allow_html=True)
            st.warning("Please upload a PDF first in the Upload PDF section")
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('</div>', unsafe_allow_html=True)
        st.warning("Please upload a PDF first in the Upload PDF section")
        st.markdown('</div>', unsafe_allow_html=True)

# Wrapper Functions for Navigation
def home_page_wrapper():
    set_professional_theme()
    st.markdown("""
        <div class="page-frame">
            <div class="top-nav">
                <div class="brand">AskVerse AI</div>
            </div>
    """, unsafe_allow_html=True)
    home_page()
    st.markdown("""
            </div>
            <div class="footer">
                © 2025 All rights reserved by <strong>@Sarath</strong>
            </div>
        </div>
    """, unsafe_allow_html=True)

def upload_page_wrapper():
    set_professional_theme()
    st.markdown("""
        <div class="page-frame">
            <div class="top-nav">
                <div class="brand">AskVerse AI</div>
            </div>
    """, unsafe_allow_html=True)
    upload_page()
    st.markdown("""
            </div>
            <div class="footer">
                © 2025 All rights reserved by <strong>@Sarath</strong>
            </div>
        </div>
    """, unsafe_allow_html=True)

def preview_page_wrapper():
    set_professional_theme()
    st.markdown("""
        <div class="page-frame">
            <div class="top-nav">
                <div class="brand">AskVerse AI</div>
            </div>
    """, unsafe_allow_html=True)
    preview_page()
    st.markdown("""
            </div>
            <div class="footer">
                © 2025 All rights reserved by <strong>@Sarath</strong>
            </div>
        </div>
    """, unsafe_allow_html=True)

def chat_page_wrapper():
    set_professional_theme()
    st.markdown("""
        <div class="page-frame">
            <div class="top-nav">
                <div class="brand">AskVerse AI</div>
            </div>
    """, unsafe_allow_html=True)
    chat_page()
    st.markdown("""
            </div>
            <div class="footer">
                © 2025 All rights reserved by <strong>@Sarath</strong>
            </div>
        </div>
    """, unsafe_allow_html=True)

# Define navigation pages
pages = {
    "AskVerse AI": [
        st.Page(home_page_wrapper, title="Home", icon="🏠"),
        st.Page(upload_page_wrapper, title="Upload PDF", icon="📤"),
        st.Page(preview_page_wrapper, title="Preview PDF", icon="📄"),
        st.Page(chat_page_wrapper, title="Chat App", icon="💬"),
    ]
}

# Set page config
st.set_page_config(
    page_title="AskVerse AI",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.AskVerse.com/help',
        'Report a bug': 'https://www.AskVerse.com/bug',
        'About': "# AskVerse AI: Advanced PDF Chat Application"
    }
)

# Run navigation
pg = st.navigation(pages)
pg.run()