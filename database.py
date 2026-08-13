import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from backend.config import DB_PATH


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Initialize SQLite database tables and FTS5 full-text search index."""
    with get_connection() as conn:
        cursor = conn.cursor()
        
        # Core document storage table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT UNIQUE NOT NULL,
                file_type TEXT NOT NULL,
                file_size_kb REAL NOT NULL,
                extracted_text TEXT NOT NULL,
                char_count INTEGER NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        
        # SQLite FTS5 Virtual Table for full-text search
        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
                doc_id UNINDEXED,
                filename,
                extracted_text
            )
        """)
        
        conn.commit()


def save_document(filename: str, file_type: str, file_size_kb: float, extracted_text: str) -> Dict[str, Any]:
    """Save or update a document record in SQLite and update its FTS index."""
    created_at = datetime.now(timezone.utc).isoformat()
    char_count = len(extracted_text)
    
    with get_connection() as conn:
        cursor = conn.cursor()
        
        # Delete existing entry if present
        cursor.execute("SELECT id FROM documents WHERE filename = ?", (filename,))
        existing = cursor.fetchone()
        if existing:
            doc_id = existing["id"]
            cursor.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
            cursor.execute("DELETE FROM documents_fts WHERE doc_id = ?", (doc_id,))
        
        cursor.execute("""
            INSERT INTO documents (filename, file_type, file_size_kb, extracted_text, char_count, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (filename, file_type, file_size_kb, extracted_text, char_count, created_at))
        
        doc_id = cursor.lastrowid
        
        # Populate FTS5 index
        cursor.execute("""
            INSERT INTO documents_fts (doc_id, filename, extracted_text)
            VALUES (?, ?, ?)
        """, (doc_id, filename, extracted_text))
        
        conn.commit()
        
        return {
            "id": doc_id,
            "filename": filename,
            "file_type": file_type,
            "file_size_kb": file_size_kb,
            "char_count": char_count,
            "created_at": created_at,
        }


def list_documents() -> List[Dict[str, Any]]:
    """Return all stored documents without full extracted_text to keep list lightweight."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, filename, file_type, file_size_kb, char_count, created_at
            FROM documents
            ORDER BY id DESC
        """)
        rows = cursor.fetchall()
        return [dict(row) for row in rows]


def get_document_by_id(doc_id: int) -> Optional[Dict[str, Any]]:
    """Fetch full document details including extracted_text by ID."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM documents WHERE id = ?", (doc_id,))
        row = cursor.fetchone()
        return dict(row) if row else None


def delete_document(doc_id: int) -> bool:
    """Delete document by ID from both main table and FTS5 table."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        cursor.execute("DELETE FROM documents_fts WHERE doc_id = ?", (doc_id,))
        conn.commit()
        return cursor.rowcount > 0


def get_all_documents_text() -> str:
    """Retrieve combined text of all documents in the database."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT filename, extracted_text FROM documents ORDER BY id ASC")
        rows = cursor.fetchall()
        if not rows:
            return ""
        
        blocks = [f"=== Document: {row['filename']} ===\n{row['extracted_text']}" for row in rows]
        return "\n\n".join(blocks)


def clean_search_term(text: str) -> str:
    """Remove special characters and punctuation for safe SQLite searching."""
    import re
    cleaned = re.sub(r"[^\w\s]", " ", text)
    return " ".join(cleaned.split())


def search_documents(query: str, limit: int = 4) -> List[Dict[str, Any]]:
    """Search for relevant documents using filename matching first, then FTS5."""
    with get_connection() as conn:
        cursor = conn.cursor()
        
        cleaned_q = clean_search_term(query)
        query_lower = cleaned_q.lower()
        
        # Detect "all" queries (e.g. "all bills", "all documents", "every file")
        all_indicators = {"all", "every", "each", "entire", "complete", "total"}
        wants_all = any(word in query_lower.split() for word in all_indicators)
        
        # If user wants ALL documents, return everything
        if wants_all:
            cursor.execute("SELECT id as doc_id, filename, extracted_text FROM documents ORDER BY id ASC")
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        
        # Skip generic stopwords
        stopwords = {"what", "when", "where", "which", "who", "whom", "this", "that",
                      "there", "their", "from", "with", "about", "the", "how", "does",
                      "are", "was", "were", "been", "being", "have", "has", "had", "can"}
        keywords = [w for w in cleaned_q.split() if len(w) > 2 and w.lower() not in stopwords]
        
        # Generate keyword variations (handle plurals: bills→bill, invoices→invoice)
        expanded_keywords = set()
        for kw in keywords:
            expanded_keywords.add(kw)
            kw_lower = kw.lower()
            if kw_lower.endswith("s") and len(kw_lower) > 3:
                expanded_keywords.add(kw[:-1])  # bills → bill
            if kw_lower.endswith("es") and len(kw_lower) > 4:
                expanded_keywords.add(kw[:-2])  # invoices → invoic
        
        results_by_id = {}
        
        # Priority 1: Match keywords against document filenames (e.g. BillFor6)
        for kw in expanded_keywords:
            cursor.execute("SELECT id as doc_id, filename, extracted_text FROM documents WHERE filename LIKE ?", (f"%{kw}%",))
            for row in cursor.fetchall():
                d = dict(row)
                results_by_id[d["doc_id"]] = d
                
        # Priority 2: FTS5 Full-text search across content
        if len(results_by_id) < limit and keywords:
            fts_query = " OR ".join(keywords)
            try:
                cursor.execute("""
                    SELECT doc_id, filename, extracted_text
                    FROM documents_fts
                    WHERE documents_fts MATCH ?
                    LIMIT ?
                """, (fts_query, limit))
                for row in cursor.fetchall():
                    d = dict(row)
                    results_by_id[d["doc_id"]] = d
            except sqlite3.OperationalError:
                pass
                
        # Priority 3: Fallback to all documents if no matches found
        if not results_by_id:
            cursor.execute("SELECT id as doc_id, filename, extracted_text FROM documents LIMIT ?", (limit,))
            for row in cursor.fetchall():
                d = dict(row)
                results_by_id[d["doc_id"]] = d
                
        return list(results_by_id.values())[:limit]


def get_relevant_context(query: str, max_chars: int = 20000) -> str:
    """Retrieve relevant context for a user query capped to stay within Groq API token limits."""
    results = search_documents(query, limit=10)
    if not results:
        return ""
        
    blocks = []
    current_chars = 0
    cleaned_q = clean_search_term(query)
    query_keywords = [w.lower() for w in cleaned_q.split() if len(w) > 2]
    
    for r in results:
        filename = r.get("filename", "Document")
        text = r.get("extracted_text", "")
        
        # If document is large (>6,000 chars), extract matching sections
        if len(text) > 6000 and query_keywords:
            lines = text.split("\n")
            matching_paragraphs = []
            for i, line in enumerate(lines):
                if any(kw in line.lower() for kw in query_keywords):
                    start_idx = max(0, i - 2)
                    end_idx = min(len(lines), i + 3)
                    matching_paragraphs.append("\n".join(lines[start_idx:end_idx]))
            
            if matching_paragraphs:
                extracted = "\n---\n".join(matching_paragraphs[:10])
            else:
                extracted = text[:5000]
            block = f"=== Document: {filename} ===\n{extracted}"
        else:
            block = f"=== Document: {filename} ===\n{text[:5000]}"
            
        if current_chars + len(block) > max_chars:
            break
        blocks.append(block)
        current_chars += len(block)
        
    return "\n\n".join(blocks)
