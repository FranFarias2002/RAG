import streamlit as st  # Interfaz web
from langchain_groq import ChatGroq  # Conector para el modelo de lenguaje (LLM) de Groq
from langchain_community.document_loaders import PyPDFLoader  # Cargador de archivos PDF
from langchain_text_splitters import RecursiveCharacterTextSplitter  # Divisor de texto inteligente
from langchain_huggingface import HuggingFaceEmbeddings  # Generador de vectores numéricos para el texto
from langchain_chroma import Chroma  # Base de datos vectorial para almacenar los fragmentos
import tempfile  # Para manejar archivos temporales de forma segura
import os  # Para operaciones del sistema como eliminar archivos

# --- CONFIGURACIÓN DE LA PÁGINA ---
# Seteamos el título de la pestaña, el icono y el ancho de la pantalla
st.set_page_config(page_title="RAG Metodología FCyT", layout="wide", page_icon="📚")

st.title("Asistente de Metodología y Normativa")

# --- SIDEBAR: CONFIGURACIÓN ---
with st.sidebar:
    # Intentamos obtener la API Key desde el archivo secrets de Streamlit
    groq_key = st.secrets.get("GROQ_API_KEY", "")
    
    st.divider()
    st.header("Documentos de Cátedra")
    # Componente para subir archivos (permite varios PDFs a la vez)
    uploaded_files = st.file_uploader("Sube los reglamentos o guías (PDF)", accept_multiple_files=True, type="pdf")

# --- FUNCIÓN DE PROCESAMIENTO (CORE DEL RAG) ---
# @st.cache_resource evita que se procesen los PDFs de nuevo si los archivos no cambian
@st.cache_resource
def crear_base(files):
    if not files:
        return None
    
    with st.spinner("Procesando y analizando documentos..."):
        all_chunks = []
        for file in files:
            # Creamos un archivo temporal porque PyPDFLoader necesita una ruta de archivo en disco
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(file.getvalue())
                loader = PyPDFLoader(tmp.name)
                docs = loader.load() # Extrae el texto del PDF
                
                # Dividimos el texto en trozos (chunks) para que el LLM pueda procesarlos mejor
                # 1000 caracteres por trozo con un solapamiento de 100 para no perder contexto entre cortes
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                chunks = splitter.split_documents(docs)
                all_chunks.extend(chunks) # Sumamos los trozos de este archivo a la lista total
                
            os.unlink(tmp.name) # Borramos el archivo temporal del disco
        
        # Cargamos el modelo que convierte texto en vectores (números que representan significado)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # Creamos la base de datos vectorial en memoria usando los trozos de texto y el modelo de embeddings
        vector_db = Chroma.from_documents(documents=all_chunks, embedding=embeddings)
        return vector_db

# --- FLUJO PRINCIPAL ---
if uploaded_files:
    # Llamamos a la función de procesamiento
    vector_db = crear_base(uploaded_files)
    
    # Campo de entrada de texto para la pregunta del usuario
    if prompt := st.chat_input("Ej: ¿Cómo se citan resoluciones internas?"):
        # Mostramos la pregunta del usuario en la interfaz
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            if not groq_key:
                st.error("Falta configurar la GROQ_API_KEY en los secretos.")
            else:
                with st.spinner("Consultando normativas..."):
                    # 1. Búsqueda de Similitud: Buscamos los 4 fragmentos más parecidos a la pregunta
                    docs_relacionados = vector_db.similarity_search(prompt, k=4)
                    
                    # Unimos los fragmentos en un solo texto indicando el número de página original
                    contexto_pdf = "\n\n".join([f"[Pág {d.metadata.get('page', 0)+1}] {d.page_content}" for d in docs_relacionados])
                    
                    # 2. Configuración del LLM (Llama 3.1) con temperatura baja para evitar alucinaciones
                    llm = ChatGroq(model_name="llama-3.1-8b-instant", groq_api_key=groq_key, temperature=0.1)
                    
                    # Creamos el "System Prompt" que obliga al modelo a usar solo el contexto del PDF
                    prompt_final = f"""
                    Eres un asistente académico experto en la normativa de la facultad. 
                    Responde de forma concisa y profesional usando solo el CONTEXTO proporcionado.
                    Si la respuesta no está en el contexto, indica amablemente que no dispones de esa información en los reglamentos subidos.

                    CONTEXTO DEL PDF:
                    {contexto_pdf}
                    
                    PREGUNTA DEL ALUMNO: {prompt}
                    """
                    
                    # Enviamos todo al modelo y mostramos la respuesta final
                    response = llm.invoke(prompt_final)
                    st.markdown(response.content)
else:
    # Mensaje inicial cuando no hay archivos
    st.info("Para comenzar, subí los reglamentos o apuntes en formato PDF en el panel lateral.")