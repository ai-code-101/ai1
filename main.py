from pathlib import Path
from datetime import datetime
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from ollama import chat
import chromadb
from sentence_transformers import SentenceTransformer
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="ES Gaming AI Assistant")
templates = Jinja2Templates(directory="templates")

KNOWLEDGE_FILE = "esgaming_catalogue.txt"
MODEL = "qwen2.5:3b "
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 300
CHUNK_OVERLAP = 80
TOP_K = 5
DB_PATH = "./chroma_db"
COLLECTION_NAME = "esgaming"

history = []

def chunk_document(file_path: str) -> list[str]:
    try:
        text = Path(file_path).read_text(encoding="utf-8")
        chunks = []
        step = CHUNK_SIZE - CHUNK_OVERLAP
        for i in range(0, len(text), step):
            chunk = text[i:i + CHUNK_SIZE].strip()
            if chunk:
                chunks.append(chunk)
        logger.info(f"✅ {len(chunks)} chunks created")
        return chunks
    except FileNotFoundError:
        logger.error(f"❌ File not found: {file_path}")
        return []

def build_vector_db(chunks: list[str], client, embedder):
    logger.info("🧮 Building vector DB from scratch...")
    embeddings = embedder.encode(chunks, show_progress_bar=True)

    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(COLLECTION_NAME)
    collection.add(
        documents=chunks,
        embeddings=embeddings.tolist(),
        ids=[f"chunk_{i}" for i in range(len(chunks))]
    )
    logger.info(f"✅ {len(chunks)} chunks stored in vector DB")
    return collection

def load_or_build_db(chunks: list[str], client, embedder):
    try:
        # Try to load existing collection
        collection = client.get_collection(COLLECTION_NAME)
        existing_count = collection.count()

        if existing_count == len(chunks) and existing_count > 0:
            logger.info(f"✅ Loaded existing vector DB — {existing_count} chunks (no rebuild needed)")
            return collection
        else:
            logger.info("♻️  Chunk count mismatch — rebuilding vector DB...")
            return build_vector_db(chunks, client, embedder)

    except Exception:
        logger.info("🆕 No existing DB found — building fresh...")
        return build_vector_db(chunks, client, embedder)

def retrieve(question: str, collection, embedder) -> str:
    q_embedding = embedder.encode([question])
    results = collection.query(
        query_embeddings=q_embedding.tolist(),
        n_results=TOP_K
    )
    chunks = results["documents"][0] if results["documents"] else []
    return "\n\n".join(chunks)

# ─────────────────────────────────────────────
#  INITIALIZE — only downloads embedding model once
# ─────────────────────────────────────────────
print("\n🚀 Starting ES Gaming AI Assistant...")
print("─" * 50)

chunks = chunk_document(KNOWLEDGE_FILE)

logger.info("📦 Loading embedding model (cached after first run)...")
embedder = SentenceTransformer(EMBEDDING_MODEL)

client = chromadb.PersistentClient(path=DB_PATH)
collection = load_or_build_db(chunks, client, embedder)

if collection:
    print(f"✅ RAG Pipeline ready — {collection.count()} chunks indexed")
else:
    print("⚠️  RAG Pipeline failed")

print("─" * 50 + "\n")

def now() -> str:
    return datetime.now().strftime("%I:%M %p")

def get_answer(question: str) -> str:
    context = ""

    if collection and embedder:
        context = retrieve(question, collection, embedder)
        logger.info(f"📋 Context retrieved: {len(context)} chars")

    system_prompt = """You are a smart and friendly AI assistant for ES Gaming, a gaming shop in Nairobi, Kenya.

Your job:
- Help customers find the right gaming products and prices from the catalog provided
- Answer clearly with product names and prices when available
- If the catalog does not have the exact answer, use your own knowledge to reason, suggest alternatives, or give helpful gaming advice
- Never just say I don't know — always try to help, reason, or suggest something useful
- If a customer needs something specific not in the catalog, recommend they call +254 703 539 102
- Be conversational, friendly, and knowledgeable about gaming

You can think beyond the catalog — give opinions, comparisons, recommendations, and gaming advice freely."""

    user_prompt = f"""ES Gaming Product Catalog:
─────────────────────────────────────
{context if context else "No catalog context retrieved for this query."}
─────────────────────────────────────

Customer question: {question}

Answer helpfully. Use catalog info first, then your own gaming knowledge to fill any gaps."""

    try:
        response = chat(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            options={
                "num_predict": 700,
                "temperature": 0.5,
                "top_k": 50,
                "top_p": 0.95,
                "repeat_penalty": 1.1
            }
        )

        print(f"DEBUG RESPONSE: {response}")
        print(f"DEBUG TYPE: {type(response)}")
        try:
            answer = response.message.content
            print(f"DEBUG ANSWER: {answer}")
        except AttributeError:
            answer = response.get("message", {}).get("content", "")
            print(f"DEBUG ANSWER2: {answer}")
            
        if not answer or not answer.strip():
            return "I'm not sure about that — please call ES Gaming at +254 703 539 102 for the latest info."

        return answer.strip()

    except Exception as e:
        logger.error(f"❌ Ollama error: {e}")
        return "I'm having trouble right now. Please contact ES Gaming at +254 703 539 102."

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"history": history}
    )

@app.post("/", response_class=HTMLResponse)
async def ask_question(request: Request, question: str = Form(...)):
    history.append({"role": "user", "content": question, "time": now()})
    answer = get_answer(question)
    history.append({"role": "bot", "content": answer, "time": now()})
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"history": history}
    )

@app.get("/clear", response_class=HTMLResponse)
async def clear_chat(request: Request):
    history.clear()
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"history": history}
    )

@app.get("/stats")
async def stats():
    return {
        "model": MODEL,
        "embedding_model": EMBEDDING_MODEL,
        "chunks": len(chunks),
        "top_k": TOP_K,
        "rag_active": collection is not None,
        "chat_history": len(history)
    }

@app.get("/test")
async def test():
    question = "What gaming keyboards do you have?"
    return {
        "question": question,
        "answer": get_answer(question)
    }