import sqlite3
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from app.config import settings

class SQLiteVectorStore:
    """
    Persistent SQLite-based storage for processed documents and text chunks.
    Includes custom numpy cosine-similarity search and word-overlap BM25 keyword fallback.
    """
    
    def __init__(self, db_path: Path = settings.DATABASE_PATH):
        self.db_path = db_path
        self._init_db()
        
    def _get_connection(self):
        # Enable foreign key support
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("PRAGMA foreign_keys = ON")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        """Initializes database tables."""
        with self._get_connection() as conn:
            # Table for parent documents
            conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    file_type TEXT NOT NULL,
                    file_size INTEGER,
                    char_count INTEGER,
                    word_count INTEGER,
                    upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Table for document chunks
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    metadata TEXT,
                    embedding BLOB,
                    FOREIGN KEY (document_id) REFERENCES documents (id) ON DELETE CASCADE
                )
            """)
            conn.commit()

    def add_document(self, doc_id: str, filename: str, file_type: str, file_size: int, 
                     char_count: int, word_count: int) -> None:
        """Saves main document record."""
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO documents (id, filename, file_type, file_size, char_count, word_count)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (doc_id, filename, file_type, file_size, char_count, word_count)
            )
            conn.commit()

    def add_chunks(self, document_id: str, chunks: List[Dict[str, Any]], 
                   embeddings: Optional[List[List[float]]] = None) -> None:
        """
        Saves document chunks with optional pre-computed float embeddings.
        """
        with self._get_connection() as conn:
            for idx, chunk in enumerate(chunks):
                text = chunk["text"]
                meta_json = json.dumps(chunk["metadata"])
                
                # Check if embedding exists
                embedding_blob = None
                if embeddings and idx < len(embeddings) and embeddings[idx] is not None:
                    arr = np.array(embeddings[idx], dtype=np.float32)
                    embedding_blob = arr.tobytes()
                    
                conn.execute(
                    """
                    INSERT INTO chunks (document_id, chunk_index, text, metadata, embedding)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (document_id, idx, text, meta_json, embedding_blob)
                )
            conn.commit()

    def delete_document(self, doc_id: str) -> bool:
        """Deletes a document and automatically cascades to delete all its chunks."""
        with self._get_connection() as conn:
            cursor = conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
            conn.commit()
            return cursor.rowcount > 0

    def list_documents(self) -> List[Dict[str, Any]]:
        """Lists all uploaded documents with metadata details."""
        with self._get_connection() as conn:
            rows = conn.execute("""
                SELECT d.*, COUNT(c.id) as chunk_count 
                FROM documents d 
                LEFT JOIN chunks c ON d.id = c.document_id 
                GROUP BY d.id
                ORDER BY d.upload_time DESC
            """).fetchall()
            return [dict(r) for r in rows]

    def clear_all(self) -> None:
        """Purges the entire database."""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM documents")
            conn.commit()

    def _keyword_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """
        Fallback keyword retrieval engine using standard token frequency scoring (similar to BM25/TF-IDF).
        Highly optimized to execute directly on the text contents.
        """
        query_words = [w.lower() for w in query.split() if len(w) > 2]
        if not query_words:
            # fallback to returning any words if query is short
            query_words = [w.lower() for w in query.split() if w]
            
        with self._get_connection() as conn:
            rows = conn.execute("SELECT c.id, c.document_id, c.chunk_index, c.text, c.metadata, d.filename FROM chunks c JOIN documents d ON c.document_id = d.id").fetchall()
            
        scored_chunks = []
        for row in rows:
            text = row["text"].lower()
            score = 0
            for word in query_words:
                # Add score based on occurrences of keyword
                count = text.count(word)
                if count > 0:
                    # Give higher weight to exact word matches (wrapped by boundaries)
                    # and minor weight to substring matches
                    score += count * 2.0
                    if f" {word} " in f" {text} ":
                        score += 5.0
            
            if score > 0:
                scored_chunks.append((score, row))
                
        # Sort by score descending
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        
        # Format results
        results = []
        for score, row in scored_chunks[:top_k]:
            results.append({
                "document_id": row["document_id"],
                "filename": row["filename"],
                "chunk_index": row["chunk_index"],
                "text": row["text"],
                "metadata": json.loads(row["metadata"] or "{}"),
                "score": float(score)
            })
            
        return results

    def similarity_search(self, query_embedding: List[float], top_k: int) -> List[Dict[str, Any]]:
        """
        Executes a vector cosine similarity search on all stored chunk embeddings.
        """
        q_arr = np.array(query_embedding, dtype=np.float32)
        q_norm = np.linalg.norm(q_arr)
        
        if q_norm == 0:
            return []
            
        with self._get_connection() as conn:
            # Query only chunks that have precomputed embeddings
            rows = conn.execute("""
                SELECT c.id, c.document_id, c.chunk_index, c.text, c.metadata, c.embedding, d.filename 
                FROM chunks c 
                JOIN documents d ON c.document_id = d.id 
                WHERE c.embedding IS NOT NULL
            """).fetchall()
            
        scored_chunks = []
        for row in rows:
            emb_bytes = row["embedding"]
            c_arr = np.frombuffer(emb_bytes, dtype=np.float32)
            
            # If dimensions match, compute cosine similarity
            if c_arr.shape == q_arr.shape:
                c_norm = np.linalg.norm(c_arr)
                if c_norm > 0:
                    similarity = float(np.dot(q_arr, c_arr) / (q_norm * c_norm))
                    scored_chunks.append((similarity, row))
                    
        # Sort by similarity descending
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        
        results = []
        for similarity, row in scored_chunks[:top_k]:
            results.append({
                "document_id": row["document_id"],
                "filename": row["filename"],
                "chunk_index": row["chunk_index"],
                "text": row["text"],
                "metadata": json.loads(row["metadata"] or "{}"),
                "score": similarity
            })
            
        return results

    def retrieve(self, query: str, query_embedding: Optional[List[float]] = None, 
                 top_k: int = 4) -> List[Dict[str, Any]]:
        """
        Unified retrieve method. Uses vector search if query_embedding is supplied,
        otherwise falls back to keyword-based search.
        """
        if query_embedding is not None:
            try:
                results = self.similarity_search(query_embedding, top_k)
                # If no vector match results, fall back to keyword search
                if not results:
                    return self._keyword_search(query, top_k)
                return results
            except Exception:
                # If vector similarity search fails, fall back
                return self._keyword_search(query, top_k)
        else:
            return self._keyword_search(query, top_k)
