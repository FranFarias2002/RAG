import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import os

# --- CONFIGURACIÓN ---
MODELO_LLM = "llama-3.1-8b-instant"
TEMPERATURA = 0.1
MODELO_EMBEDDINGS = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
TOP_K_DOCS = 4

CARPETA_DATA = "data"
CARPETA_DB = "db_data"

st.set_page_config(page_title="RAG Metodología FCyT", layout="wide", page_icon="📚")
st.title("Asistente de Metodología y Normativa")

# --- LÓGICA DE CARPETAS ---
if not os.path.exists(CARPETA_DATA):
    os.makedirs(CARPETA_DATA)

# --- FUNCIÓN DE PROCESAMIENTO ---
# Quitamos el cache momentáneamente para debuguear si se crea la carpeta
def obtener_o_crear_base():
    embeddings = HuggingFaceEmbeddings(model_name=MODELO_EMBEDDINGS)
    
    # Verificamos si la carpeta existe Y tiene contenido (el archivo de sqlite)
    if os.path.exists(CARPETA_DB) and len(os.listdir(CARPETA_DB)) > 0:
        st.sidebar.info("Cargando DB desde disco...")
        return Chroma(
            persist_directory=CARPETA_DB, 
            embedding_function=embeddings
        )
    
    # Si no existe, la creamos
    archivos_pdf = [f for f in os.listdir(CARPETA_DATA) if f.endswith('.pdf')]
    if not archivos_pdf:
        return None
        
    all_chunks = []
    for nombre_archivo in archivos_pdf:
        loader = PyPDFLoader(os.path.join(CARPETA_DATA, nombre_archivo))
        docs = loader.load()
        chunks = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        ).split_documents(docs)
        all_chunks.extend(chunks)
    
    if all_chunks:
        st.sidebar.warning("Creando DB persistente...")
        # Al crearla así, Chroma DEBE crear la carpeta
        vector_db = Chroma.from_documents(
            documents=all_chunks, 
            embedding=embeddings,
            persist_directory=CARPETA_DB
        )
        # En versiones nuevas, esto asegura que se guarde
        return vector_db
    return None

# --- SIDEBAR ---
with st.sidebar:
    st.header("Configuración")
    api_key_input = st.text_input("Introducir API Key", type="password")
    
    # Botón para forzar la creación si no la ves
    if st.button("Re-indexar documentos"):
        if os.path.exists(CARPETA_DB):
            import shutil
            shutil.rmtree(CARPETA_DB) # Borra la carpeta vieja
        st.rerun()

# --- FLUJO ---
vector_db = obtener_o_crear_base()

if vector_db:
    if prompt := st.chat_input("Consulta sobre el reglamento..."):
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            if not api_key_input:
                st.error("⚠️ Falta API Key")
            else:
                docs_relacionados = vector_db.similarity_search(prompt, k=TOP_K_DOCS)
                contexto_pdf = "\n\n".join([f"[Doc: {d.metadata.get('source')} - Pág {d.metadata.get('page', 0)+1}] {d.page_content}" for d in docs_relacionados])
                
                llm = ChatGroq(model_name=MODELO_LLM, groq_api_key=api_key_input, temperature=TEMPERATURA)
                
                prompt_final = f"Responde usando SOLO este CONTEXTO:\n{contexto_pdf}\n\nPREGUNTA: {prompt}\nSI LA PREGUNTA NO PUEDE SER RESPONDIDA CON EL CONTEXTO DI NO LO SE"
                response = llm.invoke(prompt_final)
                st.markdown(response.content)
else:
    st.info("Subí PDFs a la carpeta /data")