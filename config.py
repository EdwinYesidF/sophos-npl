"""Configuración centralizada de Sophos NLP.

Todos los parámetros ajustables del sistema viven en este archivo, para que
cambiar un modelo o un hiperparámetro no exija tocar el resto del código.
Los secretos (clave de Groq) NO van aquí: se leen de variables de entorno.
"""

# --- Aplicación -------------------------------------------------------------
APP_TITLE = "Sophos NLP"
APP_SUBTITLE = "Your open-source NLP tutor"

# --- LLM (modelo abierto, licencia Apache 2.0, servido por Groq) -------------
LLM_MODEL = "qwen/qwen3.8-27b"
TEMPERATURE = 0.2          # baja: respuestas fieles al contexto, poco creativas
MAX_TOKENS = 600           # longitud máxima de cada respuesta
REWRITE_MAX_TOKENS = 60    # longitud máxima de la consulta de búsqueda reescrita

# --- Embeddings y almacén vectorial -----------------------------------------
# Modelo multilingüe: los documentos están en inglés y los usuarios preguntan
# en español.
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
INDEX_DIR = "faiss_index"  # carpeta del índice FAISS persistente

# --- Ingesta (se usa en build_index.py) --------------------------------------
COURSE_REPO = "https://github.com/huggingface/course.git"
COURSE_CHAPTERS = [1, 2, 6, 7]
CHUNK_SIZE = 500           # el modelo de embeddings procesa ~128 tokens (~500 caracteres)
CHUNK_OVERLAP = 80         # evita cortar una idea en el borde de dos chunks

# --- Recuperación -----------------------------------------------------------
TOP_K = 4                  # fragmentos recuperados por pregunta
MAX_DISTANCE = 1.0         # umbral calibrado: NLP <= 0.79, fuera de tema >= 1.29

# --- Memoria y entrada ------------------------------------------------------
MAX_TURNS = 4              # turnos recientes que conserva la memoria de corto plazo
MAX_INPUT_CHARS = 1000     # longitud máxima de la pregunta del usuario

# --- Registro y perfiles ----------------------------------------------------
LOG_PATH = "logs/conversations.jsonl"
PROFILE_PATH = "profiles.json"
DEFAULT_LEVEL = "principiante"
