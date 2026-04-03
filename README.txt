# ASISTENTE RAG - METODOLOGÍA Y NORMATIVA FCyT
Este es un sistema de Recuperación de Información (RAG) diseñado para asistir 
a los alumnos en consultas sobre el Reglamento de Tesina (LSI) y Normas APA.

## REQUISITOS PREVIOS
1. Tener instalado Python 3.11.
2. Contar con una API Key de Groq (https://console.groq.com/), les proporcionamos una de todas formas.

## PASOS PARA LA EJECUCIÓN

1. INSTALACIÓN DE DEPENDENCIAS:
   Abrir una terminal en la carpeta del proyecto y ejecutar:
   pip install -r requirements.txt

2. LANZAMIENTO DE LA APP:
   Ejecutar el siguiente comando en la terminal:
   streamlit run app_streamlit.py

3. USO DEL SISTEMA:
   - Una vez abierta la interfaz en el navegador, ingresar la API Key de Groq en el panel lateral izquierdo.
   - Si es la primera ejecución, el sistema procesará los PDFs y creará la base de datos vectorial automáticamente.
   - Realizar consultas en el chat inferior.

## ESTRUCTURA DEL PROYECTO
- app_streamlit.py: Código principal de la aplicación.
- requirements.txt: Librerías necesarias (Streamlit, LangChain, ChromaDB, etc.).
- /data: Carpeta donde deben residir los documentos fuente (PDF).
- /db_data: Carpeta autogenerada con la base de datos vectorial persistente.

---
Desarrollado por: Caridad Jerónimo, Farías Francisco, García Rivero Brisa G., Taborda Estefanía Valeria
Materia: Inteligencia Artificial - 4to Año