import streamlit as st  # Interfaz web
from langchain_groq import ChatGroq  # Conector para el modelo de lenguaje (LLM) de Groq
from langchain_community.document_loaders import PyPDFLoader  # Cargador de archivos PDF
from langchain_text_splitters import RecursiveCharacterTextSplitter  # Divisor de texto inteligente
from langchain_huggingface import HuggingFaceEmbeddings  # Generador de vectores numéricos
from langchain_chroma import Chroma  # Base de datos vectorial
import tempfile  # Para manejar archivos temporales
import os  # Para operaciones del sistema

# ==========================================================
# --- CONFIGURACIÓN CENTRALIZADA (HIPERPARÁMETROS) ---
# ==========================================================
MODELO_LLM = "llama-3.1-8b-instant"
TEMPERATURA = 0.1
MODELO_EMBEDDINGS = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
TOP_K_DOCS = 4
# ==========================================================

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="RAG Metodología FCyT", layout="wide", page_icon="📚")
st.title("Asistente de Metodología y Normativa")

# --- SIDEBAR: DOCUMENTOS ---
with st.sidebar:
    st.header("Configuración de Acceso")
    api_key_imput = st.text_input("Introducir API Key", type="password")

    st.divider()

    st.header("Documentos de Cátedra")
    uploaded_files = st.file_uploader("Sube los reglamentos o guías (PDF)", accept_multiple_files=True, type="pdf")

# --- FUNCIÓN DE PROCESAMIENTO ---
@st.cache_resource
def crear_base(files):
    if not files:
        return None
    
    with st.spinner("Procesando y analizando documentos..."):
        all_chunks = []
        for file in files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(file.getvalue())
                loader = PyPDFLoader(tmp.name)
                docs = loader.load() 
                
                # Usamos las variables centralizadas para el splitting
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=CHUNK_SIZE, 
                    chunk_overlap=CHUNK_OVERLAP
                )
                chunks = splitter.split_documents(docs)
                all_chunks.extend(chunks)
                
            os.unlink(tmp.name)
        
        # Usamos la variable para el modelo de embeddings
        embeddings = HuggingFaceEmbeddings(model_name=MODELO_EMBEDDINGS)
        
        vector_db = Chroma.from_documents(documents=all_chunks, embedding=embeddings)
        return vector_db

# --- FLUJO PRINCIPAL ---
if uploaded_files:
    if not api_key_imput:
        st.warning("Introduce tu API Key")
    else:
        vector_db = crear_base(uploaded_files)
    
        if prompt := st.chat_input("Ej: ¿Cómo se citan resoluciones internas?"):
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Consultando normativas..."):
                    # Usamos TOP_K_DOCS para la búsqueda
                    docs_relacionados = vector_db.similarity_search(prompt, k=TOP_K_DOCS)
                    
                    contexto_pdf = "\n\n".join([f"[Pág {d.metadata.get('page', 0)+1}] {d.page_content}" for d in docs_relacionados])
                    
                    # Usamos las variables para el LLM
                    llm = ChatGroq(
                        model_name=MODELO_LLM, 
                        groq_api_key=api_key_imput, 
                        temperature=TEMPERATURA
                    )
                    
                    prompt_final = f"""
                    Eres un asistente académico experto en la normativa de la facultad. 
                    Responde de forma concisa y profesional usando solo el CONTEXTO proporcionado.
                    Si la respuesta no está en el contexto, indica amablemente que no dispones de esa información en los reglamentos subidos.

                    CONTEXTO DEL PDF:
                    {contexto_pdf}
                    
                    PREGUNTA DEL ALUMNO: {prompt}
                    """
                    
                    response = llm.invoke(prompt_final)
                    st.markdown(response.content)
else:
    st.info("Para comenzar, subí los reglamentos o apuntes en formato PDF en el panel lateral.")