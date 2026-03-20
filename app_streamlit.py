import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import tempfile
import os

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Estudiante AI - RAG", layout="wide", page_icon="📚")

st.title("🤖 Tu Asistente de Algoritmos (RAG)")
st.markdown("Sube tus apuntes y pregunta lo que necesites. ¡Ideal para preparar finales!")

# --- SIDEBAR: CONFIGURACIÓN Y CARGA ---
with st.sidebar:
    st.header("1. Configuración")
    
    # Intentamos obtener la Key de los Secrets de Streamlit
    # Si no existe en Secrets, permitimos ponerla a mano
    groq_key = st.secrets.get("GROQ_API_KEY", "")
    if not groq_key:
        groq_key = st.text_input("Groq API Key no detectada. Ingrésala aquí:", type="password")
    else:
        st.success("API Key cargada desde Secrets ✅")

    st.divider()
    st.header("2. Tus Apuntes")
    uploaded_files = st.file_uploader("Sube uno o más PDFs", accept_multiple_files=True, type="pdf")
    
    if st.button("Limpiar historial de chat"):
        st.session_state.messages = []
        st.rerun()

# --- INICIALIZAR MEMORIA DE CHAT ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- FUNCIÓN DE PROCESAMIENTO DE PDFS ---
# Usamos @st.cache_resource para que no procese los archivos cada vez que preguntamos
@st.cache_resource
def crear_base_conocimiento(files):
    if not files:
        return None
    
    with st.spinner("Leyendo y procesando documentos..."):
        all_chunks = []
        for file in files:
            # Crear archivo temporal para que PyPDFLoader pueda leerlo
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp.write(file.getvalue())
                loader = PyPDFLoader(tmp.name)
                docs = loader.load()
                
                # Dividir en trozos (Chunks)
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = splitter.split_documents(docs)
                all_chunks.extend(chunks)
            os.unlink(tmp.name) # Borrar temporal
        
        # Crear Embeddings (esto se descarga en la CPU de Streamlit)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # Crear base de datos vectorial en memoria
        vector_db = Chroma.from_documents(documents=all_chunks, embedding=embeddings)
        return vector_db

# --- FLUJO PRINCIPAL ---
if uploaded_files:
    vector_db = crear_base_conocimiento(uploaded_files)
    
    # Mostrar historial de mensajes
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Entrada del usuario
    if prompt := st.chat_input("¿Qué dice el texto sobre...?"):
        # Agregar mensaje del usuario al historial
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Respuesta del asistente
        with st.chat_message("assistant"):
            if not groq_key:
                st.error("Por favor, ingresa tu API Key en la barra lateral.")
            else:
                with st.spinner("Pensando..."):
                    # 1. Buscar fragmentos relevantes
                    docs_relacionados = vector_db.similarity_search(prompt, k=4)
                    contexto = "\n\n".join([f"[Pág {d.metadata.get('page', 'S/P')+1}] {d.page_content}" for d in docs_relacionados])
                    
                    # 2. Configurar el modelo (Llama 3.1)
                    llm = ChatGroq(model_name="llama-3.1-8b-instant", groq_api_key=groq_key, temperature=0)
                    
                    # 3. Crear el prompt final
                    # --- BUSCAR HISTORIAL PARA EL PROMPT ---
                    # Tomamos los últimos 3 mensajes para no saturar al modelo
                    contexto_historial = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages[-3:]])

                    # --- PROMPT FINAL ACTUALIZADO ---
                    prompt_final = f"""
                    Eres un profesor experto ayudando a un alumno con sus apuntes.
                    A continuación tienes el historial reciente de la charla y el contexto del PDF.

                    HISTORIAL RECIENTE:
                    {contexto_historial}

                    CONTEXTO DEL PDF:
                    {contexto}

                    PREGUNTA ACTUAL: {prompt}

                    RESPUESTA (Basándote en el contexto y siguiendo el hilo de la charla):
                    """
                    
                    # 4. Invocar y mostrar
                    response = llm.invoke(prompt_final)
                    respuesta_texto = response.content
                    st.markdown(respuesta_texto)
                    
                    # Guardar en historial
                    st.session_state.messages.append({"role": "assistant", "content": respuesta_texto})
else:
    st.info("👋 ¡Hola! Sube tus PDFs en la barra lateral para empezar a estudiar.")