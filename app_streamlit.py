import streamlit as st  # Interfaz web
from langchain_groq import ChatGroq  # Conector para LLM de Groq
from langchain_community.document_loaders import PyPDFLoader  # Cargador de PDFs
from langchain_text_splitters import RecursiveCharacterTextSplitter  # Divisor de texto
from langchain_huggingface import HuggingFaceEmbeddings  # Generador de embeddings
from langchain_chroma import Chroma  # Base de datos vectorial persistente
import os  # Para manejo de carpetas y rutas

# ==========================================================
# --- CONFIGURACIÓN CENTRALIZADA (HIPERPARÁMETROS) ---
# ==========================================================
MODELO_LLM = "llama-3.1-8b-instant"
TEMPERATURA = 0.1
MODELO_EMBEDDINGS = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
TOP_K_DOCS = 4

# Rutas de carpetas
CARPETA_DATA = "data"          # Donde están PDFs
CARPETA_DB = "db_data"         # Donde Chroma guarda los vectores
# ==========================================================

st.set_page_config(page_title="RAG Metodología FCyT", layout="wide", page_icon="📚")
st.title("Asistente de Metodología y Normativa")

# --- SIDEBAR: CONFIGURACIÓN ---
with st.sidebar:
    st.header("Configuración de Acceso")
    api_key_input = st.text_input("Introducir API Key", type="password")
    
    st.divider()
    st.header("Estado del Sistema")
    # Verificamos si hay archivos en la carpeta data
    if os.path.exists(CARPETA_DATA):
        archivos = [f for f in os.listdir(CARPETA_DATA) if f.endswith('.pdf')]
        if archivos:
            st.success(f"📂 {len(archivos)} documentos listos en /data")
        else:
            st.warning("⚠️ No se encontraron PDFs en la carpeta /data")
    else:
        os.makedirs(CARPETA_DATA)
        st.info("📁 Carpeta /data creada. Coloca tus PDFs allí.")

# --- FUNCIÓN DE PROCESAMIENTO Y PERSISTENCIA ---
@st.cache_resource
def obtener_o_crear_base():
    # Inicializamos el modelo de embeddings
    embeddings = HuggingFaceEmbeddings(model_name=MODELO_EMBEDDINGS)
    
    # CASO 1: La base de datos ya existe en disco
    if os.path.exists(CARPETA_DB) and os.listdir(CARPETA_DB):
        with st.spinner("Cargando base de datos persistente..."):
            vector_db = Chroma(
                persist_directory=CARPETA_DB, 
                embedding_function=embeddings
            )
            return vector_db
    
    # CASO 2: Hay que crear la base de datos desde los PDFs
    else:
        if not os.path.exists(CARPETA_DATA):
            return None
        
        archivos_pdf = [f for f in os.listdir(CARPETA_DATA) if f.endswith('.pdf')]
        if not archivos_pdf:
            return None
            
        with st.spinner("Creando base de datos por primera vez..."):
            all_chunks = []
            for nombre_archivo in archivos_pdf:
                ruta_completa = os.path.join(CARPETA_DATA, nombre_archivo)
                loader = PyPDFLoader(ruta_completa)
                docs = loader.load()
                
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=CHUNK_SIZE, 
                    chunk_overlap=CHUNK_OVERLAP
                )
                chunks = splitter.split_documents(docs)
                all_chunks.extend(chunks)
            
            # Creamos la DB y le decimos dónde persistirse (persist_directory)
            vector_db = Chroma.from_documents(
                documents=all_chunks, 
                embedding=embeddings,
                persist_directory=CARPETA_DB
            )
            return vector_db

# --- FLUJO PRINCIPAL ---
# Intentamos obtener la base (ya sea cargándola o creándola)
vector_db = obtener_o_crear_base()

if vector_db:
    if prompt := st.chat_input("Consulta sobre el reglamento..."):
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            if not api_key_input:
                st.error("⚠️ Por favor, ingresa la API Key en el panel lateral.")
            else:
                with st.spinner("Consultando normativas..."):
                    # Búsqueda semántica
                    docs_relacionados = vector_db.similarity_search(prompt, k=TOP_K_DOCS)
                    
                    # Armamos el contexto con metadata
                    contexto_pdf = "\n\n".join([
                        f"[Documento: {d.metadata.get('source', 'S/N')} - Pág {d.metadata.get('page', 0)+1}] {d.page_content}" 
                        for d in docs_relacionados
                    ])
                    
                    llm = ChatGroq(
                        model_name=MODELO_LLM, 
                        groq_api_key=api_key_input, 
                        temperature=TEMPERATURA
                    )
                    
                    prompt_final = f"""
                    Eres un asistente académico experto en la normativa de la facultad. 
                    Responde de forma concisa usando solo el CONTEXTO proporcionado.
                    Si la respuesta no está, indica que no dispones de esa información.

                    CONTEXTO:
                    {contexto_pdf}
                    
                    PREGUNTA: {prompt}
                    """
                    
                    response = llm.invoke(prompt_final)
                    st.markdown(response.content)
else:
    st.info("Asegúrate de tener archivos PDF en la carpeta 'data' para inicializar el sistema.")