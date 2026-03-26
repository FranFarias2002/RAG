import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import tempfile
import os

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="RAG Simple", layout="wide", page_icon="📚")

st.title("🤖 Consultor de Documentos (RAG Sin Memoria)")
st.markdown("Cada pregunta se analiza de forma independiente basándose en tus archivos.")

# --- SIDEBAR: CONFIGURACIÓN ---
with st.sidebar:
    st.header("1. Configuración")
    groq_key = st.secrets.get("GROQ_API_KEY", "")
    if not groq_key:
        groq_key = st.text_input("Ingresa tu Groq API Key:", type="password")
    
    st.divider()
    st.header("2. Tus Apuntes")
    uploaded_files = st.file_uploader("Sube uno o más PDFs", accept_multiple_files=True, type="pdf")

# --- FUNCIÓN DE PROCESAMIENTO (Se mantiene igual para eficiencia) ---
@st.cache_resource
def crear_base_conocimiento(files):
    if not files:
        return None
    
    with st.spinner("Indexando documentos..."):
        all_chunks = []
        for file in files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(file.getvalue())
                loader = PyPDFLoader(tmp.name)
                docs = loader.load()
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                chunks = splitter.split_documents(docs)
                all_chunks.extend(chunks)
            os.unlink(tmp.name) 
        
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_db = Chroma.from_documents(documents=all_chunks, embedding=embeddings)
        return vector_db

# --- FLUJO PRINCIPAL ---
if uploaded_files:
    vector_db = crear_base_conocimiento(uploaded_files)
    
    # Entrada del usuario (Sin mostrar historial previo)
    if prompt := st.chat_input("Haz una pregunta sobre el PDF:"):
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            if not groq_key:
                st.error("Error: Falta la API Key.")
            else:
                with st.spinner("Buscando en el PDF..."):
                    # 1. Búsqueda de fragmentos relevantes
                    docs_relacionados = vector_db.similarity_search(prompt, k=4)
                    contexto_pdf = "\n\n".join([f"[Pág {d.metadata.get('page', 0)+1}] {d.page_content}" for d in docs_relacionados])
                    
                    # 2. Modelo e Invocación (Sin variable de historial)
                    llm = ChatGroq(model_name="llama-3.1-8b-instant", groq_api_key=groq_key, temperature=0.1)
                    
                    prompt_final = f"""
                    Eres un asistente técnico. Responde de forma concisa usando solo el CONTEXTO proporcionado.
                    Si la respuesta no está en el contexto, di que no lo sabes.

                    CONTEXTO DEL PDF:
                    {contexto_pdf}
                    
                    PREGUNTA: {prompt}
                    """
                    
                    response = llm.invoke(prompt_final)
                    st.markdown(response.content)
else:
    st.info("👋 Sube tus archivos PDF para comenzar.")