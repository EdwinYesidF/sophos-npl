"""Sophos NLP: interfaz de chat en Streamlit.

Ejecutar en local:  streamlit run app.py
"""
import os
import uuid

import streamlit as st

import config

st.set_page_config(page_title=config.APP_TITLE, page_icon="🦉", layout="centered")

# La clave de Groq se toma de los secretos de Streamlit (despliegue) o de la
# variable de entorno GROQ_API_KEY (ejecución local). Nunca se escribe en el código.
try:
    if "GROQ_API_KEY" in st.secrets:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
except Exception:
    pass  # sin archivo de secretos: se usa la variable de entorno

import backend  # noqa: E402  (se importa después de configurar la clave)

EXAMPLES = [
    "¿Qué es la tokenización?",
    "¿Cómo funciona el algoritmo BPE?",
    "¿Qué es el mecanismo de atención?",
    "¿Qué es el reconocimiento de entidades nombradas?",
]


@st.cache_resource(show_spinner="Cargando la base de conocimiento…")
def get_resources():
    """Carga el índice y los modelos una sola vez para todas las sesiones."""
    return backend.load_resources()


# --- Estado de la sesión ------------------------------------------------------
if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex[:8]   # identificador anónimo
    st.session_state.memory = backend.ConversationMemory()
    st.session_state.messages = []                        # historial que se muestra
    st.session_state.level = config.DEFAULT_LEVEL

# --- Barra lateral: orientación al usuario y configuración ---------------------
with st.sidebar:
    st.header("🦉 " + config.APP_TITLE)
    st.caption(config.APP_SUBTITLE)

    st.subheader("¿Qué puedo hacer?")
    st.markdown(
        "- Explicar conceptos de NLP: tokenización, Transformers, atención, tareas clásicas.\n"
        "- Mostrar ejemplos cortos de código del curso de Hugging Face.\n"
        "- Recordar el hilo de la conversación e indicar mis fuentes."
    )
    st.subheader("Limitaciones")
    st.markdown(
        "- Solo respondo sobre NLP y solo con lo que está en mi base de conocimiento "
        "(capítulos 1, 2, 6 y 7 del curso de Hugging Face).\n"
        "- No resuelvo tareas ni exámenes, y no reemplazo a tu tutor."
    )
    st.warning(
        "Soy un modelo de lenguaje y puedo equivocarme o inventar información. "
        "Verifica las respuestas importantes en las fuentes citadas.",
        icon="⚠️",
    )

    level = st.selectbox(
        "Tu nivel",
        list(backend.LEVELS),
        index=list(backend.LEVELS).index(st.session_state.level),
        help="Adapta la profundidad de las explicaciones.",
    )
    if level != st.session_state.level:
        st.session_state.level = level
    backend.update_profile(st.session_state.session_id, level=level)

    if st.button("🗑️ Nueva conversación", use_container_width=True):
        st.session_state.memory.clear()
        st.session_state.messages = []
        st.rerun()

    st.caption(
        "No escribas datos personales. Las conversaciones se registran de forma "
        "anónima para mejorar el sistema."
    )
    st.caption(
        "Base de conocimiento: [Hugging Face Course](https://huggingface.co/course) "
        "(licencia Apache 2.0)."
    )

# --- Encabezado ----------------------------------------------------------------
st.title("🦉 " + config.APP_TITLE)
st.caption(config.APP_SUBTITLE + " · Tutor de Procesamiento del Lenguaje Natural")


def show_sources(sources) -> None:
    """Muestra las fuentes recuperadas con la misma numeración de las citas."""
    if sources:
        with st.expander("Fuentes consultadas"):
            for s in sources:
                st.markdown(f"**[{s['n']}]** [{s['source']}]({s['url']})")


# --- Preguntas de ejemplo (solo al inicio de la conversación) -------------------
if not st.session_state.messages:
    st.markdown("**Prueba con una de estas preguntas:**")
    cols = st.columns(2)
    for i, example in enumerate(EXAMPLES):
        if cols[i % 2].button(example, use_container_width=True):
            st.session_state.pending = example
            st.rerun()

# --- Historial ------------------------------------------------------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="🦉" if message["role"] == "assistant" else None):
        st.markdown(message["content"])
        show_sources(message.get("sources"))

# --- Nueva pregunta ---------------------------------------------------------------
question = st.chat_input("Escribe tu pregunta sobre NLP…")
question = question or st.session_state.pop("pending", None)

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant", avatar="🦉"):
        with st.spinner("Pensando…"):
            try:
                result = backend.answer_question(
                    get_resources(), question,
                    st.session_state.memory, st.session_state.session_id,
                )
                backend.log_interaction(st.session_state.session_id, question, result)
            except Exception as error:  # límite de uso, red, clave ausente, etc.
                result = {
                    "answer": "Lo siento, ocurrió un problema al generar la respuesta. "
                              "Espera un momento y vuelve a intentarlo.",
                    "sources": [],
                }
                st.caption(f"Detalle técnico: {type(error).__name__}")
        st.markdown(result["answer"])
        show_sources(result["sources"])

    st.session_state.messages.append(
        {"role": "assistant", "content": result["answer"], "sources": result["sources"]}
    )
