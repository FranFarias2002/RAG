import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import os

# config de hiperparametros
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

# lógica de carpetas
if not os.path.exists(CARPETA_DATA):
    os.makedirs(CARPETA_DATA)

# procesamiento
@st.cache_resource
def obtener_o_crear_base():
    embeddings = HuggingFaceEmbeddings(model_name=MODELO_EMBEDDINGS)
    
    if os.path.exists(CARPETA_DB) and len(os.listdir(CARPETA_DB)) > 0:
        return Chroma(persist_directory=CARPETA_DB, embedding_function=embeddings)
    
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
        vector_db = Chroma.from_documents(
            documents=all_chunks, 
            embedding=embeddings,
            persist_directory=CARPETA_DB
        )
        return vector_db
    return None

# barra lateral
with st.sidebar:
    st.header("Configuración")
    api_key_input = st.text_input("Introducir API Key", type="password")
    if st.button("Re-indexar documentos"):
        if os.path.exists(CARPETA_DB):
            import shutil
            shutil.rmtree(CARPETA_DB)
        st.rerun()

# Flujo principal
vector_db = obtener_o_crear_base()

if vector_db:
    if prompt := st.chat_input("Consulta sobre el reglamento..."):
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            if not api_key_input:
                st.error("⚠️ Falta API Key")
            else:
                # 1.Recuperamos los documentos
                docs_relacionados = vector_db.similarity_search(prompt, k=TOP_K_DOCS)
                
                # 2.Preparamos el contexto para el LLM
                contexto_pdf = "\n\n".join([
                    f"[Archivo: {d.metadata.get('source')} - Pág: {int(d.metadata.get('page', 0))+1}] {d.page_content}" 
                    for d in docs_relacionados
                ])
                
                # 3.Llamada al LLM
                llm = ChatGroq(model_name=MODELO_LLM, groq_api_key=api_key_input, temperature=TEMPERATURA)
                prompt_final = f"""Responde usando SOLO este CONTEXTO:
                {contexto_pdf}
                
                PREGUNTA: {prompt}
                
                SI LA PREGUNTA NO PUEDE SER RESPONDIDA CON EL CONTEXTO DI NO LO SE."""
                
                response = llm.invoke(prompt_final)
                
                # 4.Mostramos la respuesta
                st.markdown(response.content)
                
                # BLOQUE DE FUENTES
                # Usamos un expander para no ensuciar la interfaz, pero que siempre esté disponible
                with st.expander("🔍 Ver fuentes consultadas"):
                    # Extraemos fuentes únicas (Archivo + Página)
                    fuentes = set()
                    for d in docs_relacionados:
                        nombre_archivo = os.path.basename(d.metadata.get('source', 'Desconocido'))
                        pagina = int(d.metadata.get('page', 0)) + 1
                        fuentes.add(f"📄 {nombre_archivo} (Página {pagina})")
                    
                    for f in sorted(fuentes):
                        st.write(f)
else:
    st.info("Coloca tus PDFs en la carpeta /data para comenzar.")