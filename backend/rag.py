import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from groq import Groq, APIError, AuthenticationError, RateLimitError

from vector_store import create_embeddings, store_embeddings, search_similar_chunks

# ── Environment ───────────────────────────────────────────────────────────────
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise EnvironmentError(
        "GROQ_API_KEY is not set. Add it to your .env file:\n"
        "  GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxx"
    )

client = Groq(api_key=GROQ_API_KEY)

# Free-tier Groq models in priority order (first available is used)
GROQ_MODELS = [
    "llama-3.3-70b-versatile",   # production, best quality
    "llama-3.1-8b-instant",      # production, fastest
    "llama4-scout-17b-16e-instruct",  # production, multimodal
]


# ── PDF ingestion ─────────────────────────────────────────────────────────────
def load_pdf(file_path: str) -> list[str]:
    """Load a PDF, split into chunks, embed, and store. Returns chunk list."""
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    text_chunks = [chunk.page_content for chunk in chunks if chunk.page_content.strip()]

    if not text_chunks:
        raise ValueError("No text could be extracted from the PDF. Is it scanned/image-based?")

    embeddings = create_embeddings(text_chunks)
    store_embeddings(text_chunks, embeddings)

    return text_chunks


# ── Question answering ────────────────────────────────────────────────────────
def ask_question(question: str) -> str:
    """Retrieve relevant context and answer via Groq with model fallback."""
    retrieved_chunks = search_similar_chunks(question)

    if not retrieved_chunks:
        return (
            "No relevant content found in the knowledge base. "
            "Please upload and process a PDF first."
        )

    context = "\n\n".join(retrieved_chunks)

    prompt = f"""You are a helpful assistant. Use ONLY the context below to answer \
the user's question. If the answer is not in the context, say so clearly.

Context:
{context}

User Question:
{question}

Answer:"""

    last_error = None

    for model in GROQ_MODELS:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024,
                temperature=0.2,
            )
            return response.choices[0].message.content

        except AuthenticationError:
            # Key is wrong — no point trying other models
            raise RuntimeError(
                "Groq authentication failed. Check that GROQ_API_KEY in your "
                ".env file is correct (starts with 'gsk_')."
            )

        except RateLimitError:
            last_error = f"Rate limit hit on model '{model}'"
            continue

        except APIError as e:
            last_error = f"API error on model '{model}': {e}"
            continue

        except Exception as e:
            last_error = str(e)
            continue

    raise RuntimeError(
        f"All Groq models failed. Last error: {last_error}\n\n"
        "Possible fixes:\n"
        "  1. Verify your GROQ_API_KEY at https://console.groq.com/keys\n"
        "  2. Check your internet connection\n"
        "  3. Check Groq status at https://status.groq.com"
    )