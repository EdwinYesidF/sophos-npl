# Sophos NLP

**Your open-source NLP tutor.** Sophos NLP is a conversational tutor for Natural Language Processing. It answers questions with retrieval-augmented generation (RAG) over open course material, remembers the thread of the conversation, cites its sources, and says so when it does not know.

**Live demo:** https://sophos-nlp-tutor.streamlit.app

This project is the practical component of Phase 4 (*Advanced LLMs and chatbot models*) of the Natural Language Processing course, Universidad Nacional Abierta y a Distancia (UNAD). It uses only free and open-source tools and models.

## Features

- **Grounded answers (RAG):** responses are built only from retrieved fragments of the knowledge base, with numbered source citations.
- **Short-term memory:** a sliding window over the last 4 turns supports follow-up questions.
- **Query rewriting:** each question is rewritten as a standalone English search query, which improves retrieval for follow-ups and for questions asked in Spanish.
- **Edge-case handling:** out-of-scope questions are declined, ambiguous questions get a clarification request, and questions without relevant context are escalated to the official documentation or the course tutor.
- **Lightweight personalization:** explanations adapt to the user's level (beginner, intermediate, advanced).
- **Privacy-aware logging:** interactions are logged as JSONL with a random session ID; e-mail addresses and long numbers are masked.

## How it works

**Ingestion (once):** course files → cleaning → 500-character chunks → multilingual embeddings → persistent FAISS index.

**Conversation (every question):**

1. The user's question is cleaned and validated.
2. The LLM rewrites it as a standalone English search query, using the conversation history.
3. The FAISS index returns the 4 most similar chunks; chunks above a calibrated distance threshold (1.0) are discarded.
4. The prompt is assembled with clearly separated parts: system rules, personalization, history, retrieved context and the question.
5. The LLM generates the answer; the turn is stored in memory and in the log.

## Technology stack

| Component | Choice | License |
|---|---|---|
| Language | Python 3.12 | PSF |
| LLM | Qwen 3.8 27B (`qwen/qwen3.8-27b`), served by Groq's free tier | Apache 2.0 |
| Embeddings | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | Apache 2.0 |
| Vector store | FAISS | MIT |
| Orchestration | LangChain (core, community, Groq and Hugging Face integrations) | MIT |
| Frontend | Streamlit | Apache 2.0 |
| Hosting | Streamlit Community Cloud | Free service |
| Knowledge base | Hugging Face Course, chapters 1, 2, 6 and 7 | Apache 2.0 |

## Project structure

```
.
├── app.py                    # Streamlit chat interface
├── backend.py                # Preprocessing, prompts, memory, RAG pipeline, profiles, logging
├── config.py                 # All tunable parameters in one place
├── build_index.py            # Rebuilds the FAISS index from the course material
├── requirements.txt          # Dependencies
├── logo.png                  # Application icon
├── faiss_index/              # Persistent vector index (index.faiss, index.pkl)
└── sophos_nlp_backend.ipynb  # Commented notebook with the full backend development
```

## Run locally

Requirements: Python 3.12, git and a free Groq API key (https://console.groq.com).

```bash
# 1. Clone the repository
git clone https://github.com/EdwinYesidF/sophos-npl.git
cd sophos-npl

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install the dependencies
pip install -r requirements.txt

# 4. Provide the Groq API key as an environment variable (never write it in the code)
export GROQ_API_KEY="your_key_here"   # Windows (PowerShell): $env:GROQ_API_KEY="your_key_here"

# 5. Start the app
streamlit run app.py
```

The app opens at http://localhost:8501. The first question takes longer because the embedding model is downloaded.

### Rebuild the knowledge base (optional)

The repository already includes the index. To regenerate it from the course material:

```bash
python build_index.py
```

## Deploy on Streamlit Community Cloud

1. Fork this repository (or push it to your own public GitHub repository).
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **Create app** and choose to deploy from a repository.
4. Fill in the form: your repository, branch `main`, main file path `app.py`, and an app URL of your choice.
5. Open **Advanced settings**, select **Python 3.12**, and add the key under **Secrets**:

   ```toml
   GROQ_API_KEY = "your_key_here"
   ```

6. Click **Deploy**. The first build takes a few minutes.
7. After later changes to several files, restart the app from **Manage app → Reboot app** so every module is reloaded.

## Configuration

All parameters live in `config.py`.

| Parameter | Default | Purpose |
|---|---|---|
| `LLM_MODEL` | `qwen/qwen3.8-27b` | Open LLM served by Groq |
| `TEMPERATURE` | `0.2` | Low value keeps answers faithful to the context |
| `MAX_TOKENS` | `600` | Maximum length of each answer |
| `EMBEDDING_MODEL` | `paraphrase-multilingual-MiniLM-L12-v2` | Multilingual sentence embeddings |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `500` / `80` | Chunking, sized to the embedding model's input limit |
| `TOP_K` | `4` | Chunks retrieved per question |
| `MAX_DISTANCE` | `1.0` | Relevance threshold, calibrated on in-domain and off-topic questions |
| `MAX_TURNS` | `4` | Turns kept in short-term memory |
| `MAX_INPUT_CHARS` | `1000` | Maximum length of a user question |

## Limitations and responsible use

- Sophos NLP only answers NLP questions covered by its knowledge base. It may still make mistakes or produce inaccurate statements: verify important answers in the cited sources.
- Code examples can be incomplete, because chunking separates a snippet from the definitions it depends on.
- The free hosting tier stores logs and profiles on an ephemeral disk, and Groq's free tier applies rate limits.
- Do not enter personal data. The app does not ask for it and masks e-mails and long numbers before logging.
- `langchain-community` is being sunset; a future version should migrate the FAISS integration to its standalone package.

## Attribution

- Knowledge base: [Hugging Face Course](https://huggingface.co/course), © Hugging Face, licensed under [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0). The material was cleaned, filtered (end-of-chapter quizzes removed) and split into chunks.
- LLM: Qwen 3.8 27B by Alibaba Cloud (Apache 2.0), served by Groq.
- Embeddings: Sentence-Transformers (Apache 2.0).

## License

This project is released under the MIT License. See `LICENSE`.

## Author

Edwin Yesid Fonseca Ahumada · Master's in Data Science, UNAD · 2026
