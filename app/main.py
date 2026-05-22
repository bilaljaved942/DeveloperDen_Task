import os
import uuid
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.core.parser import DocumentParser
from app.core.splitter import RecursiveTextSplitter
from app.core.vector_store import SQLiteVectorStore
from app.core.llm import EmbeddingGenerator, LLMClient

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RAGApp")

app = FastAPI(title=settings.PROJECT_NAME, version="1.0.0")

# Enable CORS for API development flexibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize vector store
db = SQLiteVectorStore()

# Request schemas
class Message(BaseModel):
    role: str = Field(..., description="Role: 'user' or 'assistant'")
    content: str = Field(..., description="Message text content")

class ChatRequest(BaseModel):
    query: str
    history: List[Message] = []
    
    # LLM Settings
    provider: str = "groq"
    apiKey: str = ""
    model: str = ""
    temperature: float = 0.3
    
    # RAG Settings
    embeddingProvider: Optional[str] = "huggingface"
    embeddingApiKey: Optional[str] = ""
    topK: int = 4

# API Endpoints

@app.post("/api/upload")
async def upload_file(
    file: UploadFile = File(...),
    chunkSize: int = Form(1000),
    chunkOverlap: int = Form(200),
    embeddingProvider: str = Form("huggingface"),
    embeddingApiKey: str = Form("")
):
    """
    Receives document file, parses text, splits into chunks, 
    generates embeddings (if configured), and indexes into SQLite.
    """
    filename = file.filename
    ext = Path(filename).suffix.lower()
    
    if ext not in [".pdf", ".docx", ".txt"]:
        raise HTTPException(status_code=400, detail=f"Unsupported file format: {ext}")
        
    doc_id = str(uuid.uuid4())
    temp_path = settings.UPLOAD_DIR / f"{doc_id}{ext}"
    
    try:
        # 1. Save uploaded file to local disk
        with open(temp_path, "wb") as f:
            content = await file.read()
            f.write(content)
            
        logger.info(f"Saved file {filename} as {temp_path.name}")
        
        # 2. Extract text and metadata
        text, metadata = DocumentParser.parse(temp_path)
        logger.info(f"Successfully parsed {filename} ({len(text)} chars)")
        
        # 3. Chunk the text
        splitter = RecursiveTextSplitter(chunk_size=chunkSize, chunk_overlap=chunkOverlap)
        chunks = splitter.split_document(text, metadata)
        logger.info(f"Created {len(chunks)} chunks from {filename}")
        
        # 4. Generate Embeddings (if requested and possible)
        embeddings = None
        if embeddingProvider and embeddingProvider != "none":
            try:
                chunk_texts = [c["text"] for c in chunks]
                embeddings = await EmbeddingGenerator.get_embeddings(
                    texts=chunk_texts,
                    provider=embeddingProvider,
                    api_key=embeddingApiKey
                )
                logger.info(f"Generated embeddings for {len(chunks)} chunks using {embeddingProvider}")
            except Exception as emb_err:
                logger.warning(f"Embedding generation failed, falling back to keyword indexing: {str(emb_err)}")
                embeddings = None
                
        # 5. Store in vector/text database
        db.add_document(
            doc_id=doc_id,
            filename=filename,
            file_type=metadata["file_type"],
            file_size=metadata["file_size"],
            char_count=metadata["char_count"],
            word_count=metadata["word_count"]
        )
        
        db.add_chunks(document_id=doc_id, chunks=chunks, embeddings=embeddings)
        logger.info(f"Successfully indexed document {filename}")
        
        return {
            "success": True,
            "document_id": doc_id,
            "filename": filename,
            "chunks_count": len(chunks),
            "embedded": embeddings is not None,
            "metadata": {
                "file_size": metadata["file_size"],
                "char_count": metadata["char_count"],
                "word_count": metadata["word_count"]
            }
        }
        
    except Exception as e:
        # Clean up temp file on failure
        if temp_path.exists():
            os.remove(temp_path)
        logger.error(f"Error uploading file {filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")


@app.get("/api/documents")
async def get_documents():
    """Lists all uploaded and indexed documents."""
    try:
        docs = db.list_documents()
        return docs
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/documents/{doc_id}")
async def delete_document(doc_id: str):
    """Deletes document and associated chunks from database and disk."""
    try:
        # Check upload folder to delete physical file
        deleted = db.delete_document(doc_id)
        
        # Remove any file in upload directory matching doc_id
        for path in settings.UPLOAD_DIR.glob(f"{doc_id}.*"):
            try:
                os.remove(path)
            except Exception:
                pass
                
        if not deleted:
            raise HTTPException(status_code=404, detail="Document not found")
            
        return {"success": True, "message": "Document deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat")
async def chat_query(request: ChatRequest):
    """
    RAG chat endpoint. Retrieves context and streams answer using Server-Sent Events (SSE).
    """
    # 1. Input Validation
    if not request.apiKey and request.provider in ["groq", "together", "openai", "gemini"]:
        # Allow gemini/groq etc to try default environment variable if apiKey is blank
        has_env_key = False
        p_upper = request.provider.upper()
        if p_upper == "GROQ" and settings.GROQ_API_KEY:
            has_env_key = True
        elif p_upper == "TOGETHER" and settings.TOGETHER_API_KEY:
            has_env_key = True
        elif p_upper == "OPENAI" and settings.OPENAI_API_KEY:
            has_env_key = True
        elif p_upper == "GEMINI" and settings.GEMINI_API_KEY:
            has_env_key = True
            
        if not has_env_key:
            raise HTTPException(status_code=400, detail=f"API Key is required for provider '{request.provider}'.")
            
    # Resolve correct API key
    api_key = request.apiKey
    if not api_key:
        if request.provider == "groq":
            api_key = settings.GROQ_API_KEY
        elif request.provider == "together":
            api_key = settings.TOGETHER_API_KEY
        elif request.provider == "openai":
            api_key = settings.OPENAI_API_KEY
        elif request.provider == "gemini":
            api_key = settings.GEMINI_API_KEY

    # 2. Retrieve Context Chunks
    query_embedding = None
    if request.embeddingProvider and request.embeddingProvider != "none":
        try:
            emb_key = request.embeddingApiKey
            if not emb_key:
                if request.embeddingProvider == "openai":
                    emb_key = settings.OPENAI_API_KEY
                elif request.embeddingProvider == "together":
                    emb_key = settings.TOGETHER_API_KEY
                    
            embs = await EmbeddingGenerator.get_embeddings(
                texts=[request.query],
                provider=request.embeddingProvider,
                api_key=emb_key
            )
            if embs and embs[0] is not None:
                query_embedding = embs[0]
        except Exception as e:
            logger.warning(f"Could not embed chat query: {str(e)}. Falling back to text retrieval.")

    # Fetch context
    context_chunks = db.retrieve(
        query=request.query,
        query_embedding=query_embedding,
        top_k=request.topK
    )
    
    # 3. Create SSE Stream Generator
    async def sse_generator() -> AsyncGenerator[str, None]:
        # Send retrieve sources as first message
        sources_payload = {
            "type": "sources",
            "sources": [
                {
                    "document_id": chunk["document_id"],
                    "filename": chunk["filename"],
                    "chunk_index": chunk["chunk_index"],
                    "score": chunk.get("score", 0.0)
                }
                for chunk in context_chunks
            ]
        }
        yield f"data: {json.dumps(sources_payload)}\n\n"
        
        # Build clean history format for generator
        formatted_history = []
        for msg in request.history:
            formatted_history.append({
                "role": msg.role,
                "content": msg.content
            })
            
        # Stream response chunks from LLM Client
        try:
            async for text_chunk in LLMClient.stream_chat(
                query=request.query,
                context_chunks=context_chunks,
                history=formatted_history,
                provider=request.provider,
                api_key=api_key,
                model=request.model,
                temperature=request.temperature
            ):
                content_payload = {"type": "text", "content": text_chunk}
                yield f"data: {json.dumps(content_payload)}\n\n"
        except Exception as stream_err:
            logger.error(f"Stream error: {str(stream_err)}")
            error_payload = {"type": "error", "message": f"LLM stream error: {str(stream_err)}"}
            yield f"data: {json.dumps(error_payload)}\n\n"
            
        # Close connection
        done_payload = {"type": "done"}
        yield f"data: {json.dumps(done_payload)}\n\n"

    return StreamingResponse(sse_generator(), media_type="text/event-stream")


@app.post("/api/clear")
async def clear_database():
    """Clears database and deletes all uploaded files."""
    try:
        db.clear_all()
        # Delete upload file directory contents
        for path in settings.UPLOAD_DIR.iterdir():
            if path.is_file():
                try:
                    os.remove(path)
                except Exception:
                    pass
        return {"success": True, "message": "Vector store and upload archives cleared successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Setup Static Files serving
# Mount static files at "/static" and return index.html for root "/"

# Create static directory if it does not exist
BASE_DIR = Path(__file__).resolve().parent.parent
static_dir = BASE_DIR / "app" / "static"
static_dir.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

@app.get("/")
async def serve_home():
    """Serves the main single-page UI."""
    index_path = static_dir / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    else:
        return {"message": "Premium RAG Chatbot is running! Place your index.html inside the static folder."}
from typing import AsyncGenerator
