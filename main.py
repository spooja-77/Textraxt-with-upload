import itertools
import os
from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI, File, HTTPException, Header, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from backend.config import CHAT_MODEL, get_groq_client, BASE_DIR
from backend.database import (
    delete_document,
    get_all_documents_text,
    get_relevant_context,
    init_db,
    list_documents,
    save_document,
    search_documents,
)
from backend.extractor import extract_file_content

app = FastAPI(title="DocBot Backend API", version="1.0.0")

# CORS middleware for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    """Initialize database tables on app startup."""
    init_db()


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    api_key: Optional[str] = None


SYSTEM_TEMPLATE = """You are a precise document assistant. Answer the user's questions using ONLY the document contents below.

Rules:
- Quote values (dates, IDs, amounts, locations, names) exactly as they appear in the document.
- If the answer is not present in any document, say so plainly — never guess or invent values.
- When multiple documents are loaded, mention which document the answer came from.
- Keep answers short and direct.

DOCUMENTS:
{docs}
"""


@app.get("/api/health")
def health_check():
    docs = list_documents()
    return {
        "status": "online",
        "total_documents": len(docs),
        "docs": [d["filename"] for d in docs],
    }


@app.get("/api/documents")
def get_documents():
    """Retrieve list of all indexed documents in the database."""
    return {"documents": list_documents()}


@app.delete("/api/documents/{doc_id}")
def remove_document(doc_id: int):
    """Delete a document by ID."""
    success = delete_document(doc_id)
    if not success:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"message": "Document deleted successfully", "id": doc_id}


@app.post("/api/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    x_groq_api_key: Optional[str] = Header(None),
):
    """Upload a file, extract its text, and save to SQLite DB."""
    data = await file.read()
    client = get_groq_client(x_groq_api_key)
    
    text, kind = extract_file_content(client, data, file.filename, file.content_type)
    file_size_kb = len(data) / 1024.0
    
    doc = save_document(
        filename=file.filename,
        file_type=kind,
        file_size_kb=file_size_kb,
        extracted_text=text,
    )
    return {"message": "Document uploaded and indexed successfully", "document": doc}


@app.post("/api/seed")
def seed_documents(x_groq_api_key: Optional[str] = Header(None)):
    """Auto-seed documents from /Users/bot_rane/unstructured_documents/fwddocs."""
    seed_folder = Path("/Users/bot_rane/unstructured_documents/fwddocs")
    if not seed_folder.exists():
        # Fallback to local fwddocs if available
        seed_folder = BASE_DIR / "fwddocs"
        
    if not seed_folder.exists():
        raise HTTPException(status_code=404, detail=f"Seed folder not found at {seed_folder}")
        
    client = get_groq_client(x_groq_api_key)
    seeded = []
    
    for item in seed_folder.iterdir():
        if item.is_file() and not item.name.startswith("."):
            with open(item, "rb") as f:
                content = f.read()
            text, kind = extract_file_content(client, content, item.name)
            file_size_kb = len(content) / 1024.0
            doc = save_document(
                filename=item.name,
                file_type=kind,
                file_size_kb=file_size_kb,
                extracted_text=text,
            )
            seeded.append(doc["filename"])
            
    return {"message": f"Seeded {len(seeded)} documents successfully", "seeded_files": seeded}


@app.post("/api/chat")
def chat_stream(req: ChatRequest):
    """Stream response from Groq LLM grounded on SQLite document contents."""
    client = get_groq_client(req.api_key)
    if client is None:
        raise HTTPException(status_code=400, detail="Groq API Key is required.")
        
    last_user_msg = next((m.content for m in reversed(req.messages) if m.role == "user"), "")
    docs_block = get_relevant_context(last_user_msg)
    if not docs_block:
        docs_block = "(No matching documents loaded in database yet. Please upload or seed documents first.)"
        
    system_msg = {"role": "system", "content": SYSTEM_TEMPLATE.format(docs=docs_block)}
    formatted_messages = [system_msg] + [{"role": m.role, "content": m.content} for m in req.messages]
    
    def generate():
        try:
            stream = client.chat.completions.create(
                model=CHAT_MODEL,
                messages=formatted_messages,
                temperature=0.2,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        except Exception as e:
            yield f"\n\n[Error generating response: {str(e)}]"
            
    return StreamingResponse(generate(), media_type="text/plain")
