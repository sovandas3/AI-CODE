

import os
import streamlit as st
import pdfminer.high_level
from docx import Document
from pptx import Presentation
from bs4 import BeautifulSoup
import pandas as pd
import json
import xml.etree.ElementTree as ET
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import httpx
import tempfile
import whisper
import pyttsx3
from pydub import AudioSegment
import subprocess
from PIL import Image
import pytesseract

# Set Tiktoken cache
os.environ["TIKTOKEN_CACHE_DIR"] = "./token"

# Disable SSL verification (not for production)
client = httpx.Client(verify=False)

# Load Whisper model once
whisper_model = whisper.load_model("base")

# --- Whisper Audio Transcription ---
def extract_text_from_file(uploaded_file):
    file_name = uploaded_file.name.lower()

    # --- Text-based documents ---
    if file_name.endswith(".pdf"):
        uploaded_file.seek(0)
        return pdfminer.high_level.extract_text(uploaded_file)

    elif file_name.endswith(".docx"):
        doc = Document(uploaded_file)
        return "\n".join([para.text for para in doc.paragraphs])

    elif file_name.endswith(".txt"):
        return uploaded_file.read().decode("utf-8", errors="ignore")

    elif file_name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
        return df.to_string()

    elif file_name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)
        return df.to_string()

    elif file_name.endswith(".pptx"):
        prs = Presentation(uploaded_file)
        return "\n".join(shape.text for slide in prs.slides for shape in slide.shapes if hasattr(shape, "text"))

    elif file_name.endswith((".html", ".htm")):
        html_content = uploaded_file.read().decode("utf-8", errors="ignore")
        soup = BeautifulSoup(html_content, "html.parser")
        return soup.get_text(separator="\n")

    elif file_name.endswith(".json"):
        data = json.load(uploaded_file)
        return json.dumps(data, indent=2)

    elif file_name.endswith(".xml"):
        tree = ET.parse(uploaded_file)
        root = tree.getroot()
        return ET.tostring(root, encoding="unicode", method="xml")

    # --- Audio files ---
    elif file_name.endswith((".mp3", ".mp4", ".m4a", ".aac", ".wav")):
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_audio_file:
            tmp_audio_file.write(uploaded_file.read())
            tmp_audio_path = tmp_audio_file.name
        result = whisper_model.transcribe(tmp_audio_path)
        os.unlink(tmp_audio_path)
        return result["text"]

    # --- Image files ---
    elif file_name.endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_img_file:
            tmp_img_file.write(uploaded_file.read())
            tmp_img_path = tmp_img_file.name
        
        # Try OCR first for text extraction
        ocr_text = pytesseract.image_to_string(Image.open(tmp_img_path)).strip()
        if ocr_text:
            return ocr_text
        
        # If OCR fails (empty text), treat as general image → Vision LLM
        chat_model = ChatOpenAI(
            base_url="https://genailab.tcs.in",
            model="azure_ai/genailab-maas-Llama-3.2-90B-Vision-Instruct",  # or another Vision LLM
            api_key="your_api_key",
            http_client=client
        )
        response = chat_model.invoke([
            {"role": "user", "content": [
                {"type": "text", "text": "Please analyze this image and describe it in detail."},
                {"type": "image_url", "image_url": tmp_img_path}
            ]}
        ])
        return response.content

    return "[Unsupported file type]"

# --- Chunk Text ---
def chunk_text(text, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)

# --- Embedding Model ---
embedding_model = OpenAIEmbeddings(
    base_url="https://genailab.tcs.in",
    model="azure/genailab-maas-text-embedding-3-large",
    api_key="sk-nHOz9YiE4mFq1O1XBLvzMg",  # Replace securely
    http_client=client
)

# --- Vector Store Setup ---
CHROMA_DIR = "chroma_store"
vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embedding_model)

# --- Text-to-Speech helper ---
def text_to_speech(text):
    engine = pyttsx3.init()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
        tmp_path = tmp_file.name
    engine.save_to_file(text, tmp_path)
    engine.runAndWait()
    return tmp_path

# --- UI Setup ---
st.set_page_config(page_title="Universal Doc Summarizer", layout="wide")
st.markdown("<h1 style='color:#6c63ff;'>📄 Universal Document Summarizer with Azure AI</h1>", unsafe_allow_html=True)

# --- Sidebar Upload ---
with st.sidebar:
    st.header("📁 Upload & Extract")
    uploaded_file = st.file_uploader(
    "Upload file",
    type=[
        "pdf", "docx", "pptx", "html", "htm", "txt", "csv", "xlsx", "json", "xml",
        "mp3", "wav", "mp4", "m4a", "aac",
        "png", "jpg", "jpeg", "bmp", "tiff"
    ]
)

    doc_text = ""
    if uploaded_file:
        with st.spinner("⏳ Extracting content from file..."):
            doc_text = extract_text_from_file(uploaded_file)
            chunks = chunk_text(doc_text)
        st.success(f"✅ Extracted {len(chunks)} text chunks from *{uploaded_file.name}*")

        st.markdown("#### 📝 Text Preview")
        st.text_area("Preview", doc_text[:2000] + ("..." if len(doc_text) > 2000 else ""), height=300)

        if st.button("💾 Save to Vector DB"):
            vectorstore.add_texts(chunks, metadatas=[{"source": uploaded_file.name}] * len(chunks))
            vectorstore.persist()
            st.success(f"✅ Stored {len(chunks)} chunks into ChromaDB.")

# --- Query + Model Selection ---
st.markdown("---")
st.subheader("🔎 Ask a Question About Your Documents")

col1, col2, col3 = st.columns([3, 2, 1])

with col1:
    query = st.text_input("Type your question")

with col2:
    llm_model_map = {
        "azure/genailab-maas-gpt-35-turbo": "GPT-3.5 Turbo",
        "azure/genailab-maas-gpt-4o": "GPT-4o",
        "azure/genailab-maas-gpt-4o-mini": "GPT-4o Mini",
        "azure_ai/genailab-maas-DeepSeek-R1": "DeepSeek R1",
        "azure_ai/genailab-maas-DeepSeek-V3-0324": "DeepSeek V3",
        "azure_ai/genailab-maas-Llama-3.2-90B-Vision-Instruct": "LLaMA 3.2 90B Vision",
        "azure_ai/genailab-maas-Llama-3.3-70B-Instruct": "LLaMA 3.3 70B Instruct",
        "azure_ai/genailab-maas-Llama-4-Maverick-17B-128E-Instruct-FP8": "LLaMA 4 Maverick 17B",
        "azure_ai/genailab-maas-Phi-3.5-vision-instruct": "Phi 3.5 Vision Instruct",
        "azure_ai/genailab-maas-Phi-4-reasoning": "Phi 4 Reasoning"
    }
    display_names = list(llm_model_map.values())
    internal_model_map = {v: k for k, v in llm_model_map.items()}
    selected_display_name = st.selectbox("🤖 Choose Chat Model", display_names, index=display_names.index("DeepSeek V3"))
    selected_llm = internal_model_map[selected_display_name]

with col3:
    st.markdown("<br>", unsafe_allow_html=True)  # spacer
    run_query = st.button("💬 Get Answer")

# --- Session State ---
if 'last_answer' not in st.session_state:
    st.session_state['last_answer'] = None
if 'audio_file' not in st.session_state:
    st.session_state['audio_file'] = None

# --- Query Execution ---
if run_query:
    if query.strip() == "":
        st.warning("Please enter a question before asking.")
    else:
        chat_model = ChatOpenAI(
            base_url="https://genailab.tcs.in",
            model=selected_llm,
            api_key="sk-nHOz9YiE4mFq1O1XBLvzMg",  # Replace securely
            http_client=client
        )
        with st.spinner("🤔 Thinking..."):
            docs = vectorstore.similarity_search(query, k=4)
            context = "\n\n".join([d.page_content for d in docs])
            prompt = f"Answer the question based on the context:\n{context}\n\nQuestion: {query}"
            answer = chat_model.invoke(prompt)

        st.session_state['last_answer'] = answer.content
        st.session_state['audio_file'] = None  # reset

# --- Answer Display ---
if 'last_answer' in st.session_state and st.session_state['last_answer']:
    st.subheader("💡 Answer")
    st.write(st.session_state['last_answer'])

    if st.button("Read answer aloud"):
        audio_path = text_to_speech(st.session_state['last_answer'])
        st.session_state['audio_file'] = audio_path

if st.session_state['audio_file']:
    audio_bytes = open(st.session_state['audio_file'], 'rb').read()
    st.audio(audio_bytes, format='audio/mp3')
