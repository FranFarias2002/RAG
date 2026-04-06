import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import os

# 1. CONFIGURACIÓN DE HIPERPARÁMETROS 
MODELO_LLM = "llama-3.1-8b-instant"
TEMPERATURA = 0.1 # Baja temperatura para evitar alucinaciones
MODELO_EMBEDDINGS = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CHUNK_SIZE = 600
CHUNK_OVERLAP = 120
TOP_K_DOCS = 7

CARPETA_DATA = "data"
CARPETA_DB = "db_data"

st.set_page_config(page_title="RAG Metodología FCyT", layout="wide", page_icon="📚")
st.title("Asistente de Metodología y Normativa")

# Verificación inicial: Crea la carpeta de origen si no existe
if not os.path.exists(CARPETA_DATA):
    os.makedirs(CARPETA_DATA)

# 2. PROCESAMIENTO DE DOCUMENTOS (ETAPAS 1 A 5 DEL PIPELINE)

@st.cache_resource # Evita reprocesar todo al interactuar con la UI
def obtener_o_crear_base():
    # PIPELINE ETAPA 4: Embeddings
    # Convierte texto en vectores matemáticos usando el modelo multilingüe
    embeddings = HuggingFaceEmbeddings(model_name=MODELO_EMBEDDINGS)
    
    # Intenta cargar base existente para ahorrar cómputo y tiempo
    if os.path.exists(CARPETA_DB) and len(os.listdir(CARPETA_DB)) > 0:
        return Chroma(persist_directory=CARPETA_DB, embedding_function=embeddings)
    
    # Si no hay base, procesa los PDFs de la carpeta data
    archivos_pdf = [f for f in os.listdir(CARPETA_DATA) if f.endswith('.pdf')]
    if not archivos_pdf:
        return None# Si no hay PDFs, no se puede crear la base
        
    all_chunks = []
    for nombre_archivo in archivos_pdf:

        # PIPELINE ETAPA 1 y 2: Carga y Limpieza

        # PyPDFLoader extrae el texto digital ignorando elementos no textuales
        loader = PyPDFLoader(os.path.join(CARPETA_DATA, nombre_archivo))
        docs = loader.load() # Cada página se convierte en un documento separado con metadatos de origen

        # PIPELINE ETAPA 3: Fragmentación (Chunking)

        # Divide el PDF en pedazos procesables respetando el solapamiento
        chunks = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, 
            chunk_overlap=CHUNK_OVERLAP
        ).split_documents(docs)
        all_chunks.extend(chunks)# Acumula todos los fragmentos de todos los PDFs para luego vectorizarlos juntos
    
    if all_chunks:

        # PIPELINE ETAPA 5: Almacenamiento Vectorial
        
        # Crea y guarda la base de datos física en el disco (db_data)
        vector_db = Chroma.from_documents(
            documents=all_chunks, 
            embedding=embeddings,
            persist_directory=CARPETA_DB
        )
        return vector_db
    return None


# 3. INTERFAZ Y MANTENIMIENTO
# Barra lateral para configuración y mantenimiento
with st.sidebar:
    st.header("Configuración")
    api_key_input = st.text_input("Introducir API Key", type="password")
    
    st.divider()
    # Lógica de limpieza: Borra la DB física para forzar un nuevo escaneo de PDFs
    if st.button("Re-indexar documentos"):
        if os.path.exists(CARPETA_DB):
            import shutil
            shutil.rmtree(CARPETA_DB)
        st.rerun()

# flujo principal
# Carga la base de datos (si existe) o la crea (si hay PDFs nuevos)
vector_db = obtener_o_crear_base()

# 4. FLUJO DE CONSULTA (ETAPAS 6 Y 7 DEL PIPELINE)

if vector_db:# Solo muestra el chat si la base de datos está lista (es decir, hay PDFs procesados)
    if prompt := st.chat_input("Consulta sobre el reglamento o la teoría..."):
        with st.chat_message("user"):
            st.markdown(prompt)# Muestra la pregunta del usuario en el formato de chat

        with st.chat_message("assistant"):
            if not api_key_input:
                st.error("Ingresa la API Key en el panel lateral.")
            else:
                # PIPELINE ETAPA 6: Recuperación (Retrieval)
                # Busca los 7 fragmentos más parecidos vectorialmente a la pregunta
                docs_relacionados = vector_db.similarity_search(prompt, k=TOP_K_DOCS)
                
                # PIPELINE ETAPA 7: Construcción del Contexto
                # Construcción del contexto plano con metadatos para el LLM
                contexto_pdf = "\n\n".join([
                    f"[Archivo: {os.path.basename(d.metadata.get('source'))} - Pág: {int(d.metadata.get('page', 0))+1}] {d.page_content}" 
                    for d in docs_relacionados
                ])
                
                # PIPELINE ETAPA 8: Generación de Respuesta
                # Configuración del motor de inferencia (LLM)
                llm = ChatGroq(
                    model_name=MODELO_LLM, 
                    groq_api_key=api_key_input, 
                    temperature=TEMPERATURA
                )
                
                # Definición del Prompt Maestro con reglas de seguridad anti-alucinación
                prompt_final = f"""
                Eres un asistente técnico LIMITADO de la FCyT. Tu única fuente de verdad es el CONTEXTO proporcionado.

                REGLAS DE CUMPLIMIENTO ESTRICTO:
                1. IDENTIDAD: Si la pregunta menciona una carrera, institución o trámite que NO figura explícitamente en el contexto, responde: "La base de datos actual solo contiene información sobre la Licenciatura en Sistemas (LSI) de la FCyT. No poseo datos sobre [Carrera mencionada]".
                2. SÍNTESIS: Responde de forma DIRECTA. No uses introducciones pero da la respuesta completa.
                3. ORIGEN: Si la información proviene de un libro teórico (Marta Marin o Escritura Académica), aclara que son "Sugerencias teóricas" y no "Normativa institucional".
                4. SILENCIO: Si el contexto no contiene la respuesta exacta a la entidad consultada, di "No lo sé".

                CONTEXTO:
                {contexto_pdf}

                PREGUNTA: {prompt}
                """
                # Generación de respuesta
                # Ejecución de la llamada al modelo
                with st.spinner("Pensando..."):
                    response = llm.invoke(prompt_final)# Obtiene la respuesta generada por el LLM basada en el contexto construido
                    st.markdown(response.content)# Muestra la respuesta generada por el LLM en el formato de chat
                
                # PIPELINE ETAPA 9: Validación Académica
                # Desplegable de fuentes consultadas para transparencia y validación por parte del usuario
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