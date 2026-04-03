import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import os

# CONFIGURACIÓN DE HIPERPARÁMETROS 
MODELO_LLM = "llama-3.1-8b-instant"
TEMPERATURA = 0.1
MODELO_EMBEDDINGS = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CHUNK_SIZE = 600
CHUNK_OVERLAP = 120
TOP_K_DOCS = 7

CARPETA_DATA = "data"
CARPETA_DB = "db_data"

st.set_page_config(page_title="RAG Metodología FCyT", layout="wide", page_icon="📚")
st.title("Asistente de Metodología y Normativa")

# Lógica de carpetas inicial
if not os.path.exists(CARPETA_DATA):
    os.makedirs(CARPETA_DATA)

# Función de procesamiento con persistencia
@st.cache_resource
def obtener_o_crear_base():
    embeddings = HuggingFaceEmbeddings(model_name=MODELO_EMBEDDINGS)
    
    # Intenta cargar base existente
    if os.path.exists(CARPETA_DB) and len(os.listdir(CARPETA_DB)) > 0:
        return Chroma(persist_directory=CARPETA_DB, embedding_function=embeddings)
    
    # Si no hay base, procesa los PDFs
    archivos_pdf = [f for f in os.listdir(CARPETA_DATA) if f.endswith('.pdf')]
    if not archivos_pdf:
        return None
        
    all_chunks = []
    for nombre_archivo in archivos_pdf:
        loader = PyPDFLoader(os.path.join(CARPETA_DATA, nombre_archivo))
        docs = loader.load()
        chunks = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, 
            chunk_overlap=CHUNK_OVERLAP
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

# Barra lateral para configuración y mantenimiento
with st.sidebar:
    st.header("Configuración")
    api_key_input = st.text_input("Introducir API Key", type="password")
    
    st.divider()
    if st.button("Re-indexar documentos"):
        if os.path.exists(CARPETA_DB):
            import shutil
            shutil.rmtree(CARPETA_DB)
        st.rerun()

# flujo principal
vector_db = obtener_o_crear_base()

if vector_db:
    if prompt := st.chat_input("Consulta sobre el reglamento o la teoría..."):
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            if not api_key_input:
                st.error("Ingresa la API Key en el panel lateral.")
            else:
                # 1. Recuperación de fragmentos (Retrieval)
                docs_relacionados = vector_db.similarity_search(prompt, k=TOP_K_DOCS)
                
                # 2. Construcción del contexto enriquecido
                contexto_pdf = "\n\n".join([
                    f"[Archivo: {os.path.basename(d.metadata.get('source'))} - Pág: {int(d.metadata.get('page', 0))+1}] {d.page_content}" 
                    for d in docs_relacionados
                ])
                
                # 3. Preparación del LLM y el Prompt Maestro
                llm = ChatGroq(
                    model_name=MODELO_LLM, 
                    groq_api_key=api_key_input, 
                    temperature=TEMPERATURA
                )
                
                # Inyección del prompt solicitado
                prompt_final = f"""
                Eres un asistente técnico LIMITADO de la FCyT. Tu única fuente de verdad es el CONTEXTO proporcionado.

                REGLAS DE CUMPLIMIENTO ESTRICTO:
                1. IDENTIDAD: Si la pregunta menciona una carrera (ej. Gestión de Organizaciones), institución o trámite que NO figura explícitamente en el contexto, responde: "La base de datos actual solo contiene información sobre la Licenciatura en Sistemas (LSI) de la FCyT. No poseo datos sobre [Carrera mencionada]".
                2. SÍNTESIS: Responde de forma DIRECTA. No uses introducciones.
                3. ORIGEN: Si la información proviene de un libro teórico (Marta Marin o Escritura Académica), aclara que son "Sugerencias teóricas" y no "Normativa institucional".
                4. SILENCIO: Si el contexto no contiene la respuesta exacta a la entidad consultada, di "No lo sé".

                CONTEXTO:
                {contexto_pdf}

                PREGUNTA: {prompt}
                """
                # 4. Generación de respuesta
                with st.spinner("Pensando..."):
                    response = llm.invoke(prompt_final)
                    st.markdown(response.content)
                
                # 5. Desplegable de fuentes para validación académica
                with st.expander("Ver fuentes consultadas"):
                    fuentes = set()
                    for d in docs_relacionados:
                        nombre_archivo = os.path.basename(d.metadata.get('source', 'Desconocido'))
                        pagina = int(d.metadata.get('page', 0)) + 1
                        fuentes.add(f"📄 {nombre_archivo} (Página {pagina})")
                    
                    for f in sorted(fuentes):
                        st.write(f)
else:
    st.info("Asegurate de que la carpeta 'data' contenga archivos PDF para inicializar el asistente.")