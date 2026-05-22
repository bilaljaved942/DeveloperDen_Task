import sys
import logging
from pathlib import Path
from app.core.parser import DocumentParser
from app.core.splitter import RecursiveTextSplitter
from app.core.vector_store import SQLiteVectorStore

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("RAGTest")

def run_tests():
    logger.info("=== Starting RAG Chatbot Integration Tests ===")
    
    # 1. Check workspace PDF availability
    pdf_path = Path("RAG Task.pdf")
    if not pdf_path.exists():
        logger.error("RAG Task.pdf not found in the workspace directory. Cannot proceed with parsing tests.")
        sys.exit(1)
        
    logger.info(f"1. Found test document: {pdf_path.name}")
    
    # 2. Test DocumentParser on PDF
    logger.info("2. Parsing PDF using DocumentParser...")
    try:
        text, metadata = DocumentParser.parse(pdf_path)
        logger.info("✓ PDF successfully parsed!")
        logger.info(f"   Metadata: {metadata}")
        logger.info(f"   Extracted text character count: {len(text)}")
        logger.info(f"   Extracted text word count: {metadata.get('word_count', 0)}")
        
        # Quick validation
        assert len(text) > 0, "Parsed text is empty!"
        assert metadata["file_type"] == "pdf", "Invalid parsed file type metadata!"
    except Exception as e:
        logger.error(f"✗ Parser test failed: {str(e)}")
        sys.exit(1)
        
    # 3. Test RecursiveTextSplitter
    logger.info("3. Chunking document using RecursiveTextSplitter...")
    try:
        chunk_size = 500
        chunk_overlap = 100
        splitter = RecursiveTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = splitter.split_document(text, metadata)
        
        logger.info("✓ Document successfully chunked!")
        logger.info(f"   Generated {len(chunks)} text chunks.")
        
        # Validate chunks structure
        assert len(chunks) > 0, "No chunks were generated!"
        for i, chunk in enumerate(chunks[:2]):
            logger.info(f"   Chunk #{i} length: {len(chunk['text'])} chars")
            assert "text" in chunk, f"Chunk #{i} is missing 'text'!"
            assert "metadata" in chunk, f"Chunk #{i} is missing 'metadata'!"
    except Exception as e:
        logger.error(f"✗ Splitter test failed: {str(e)}")
        sys.exit(1)
        
    # 4. Test SQLiteVectorStore Database Operations
    logger.info("4. Testing SQLiteVectorStore database indexing and persistence...")
    try:
        # Use a temporary test database
        test_db_path = Path("test_rag_store.db")
        if test_db_path.exists():
            test_db_path.unlink()
            
        db = SQLiteVectorStore(db_path=test_db_path)
        logger.info("✓ Test database initialized.")
        
        # Save main doc
        doc_id = "test-doc-uuid-1234"
        db.add_document(
            doc_id=doc_id,
            filename=pdf_path.name,
            file_type=metadata["file_type"],
            file_size=metadata["file_size"],
            char_count=metadata["char_count"],
            word_count=metadata["word_count"]
        )
        logger.info("✓ Main document record indexed.")
        
        # Save chunks
        db.add_chunks(document_id=doc_id, chunks=chunks)
        logger.info("✓ Document chunks indexed successfully.")
        
        # List files
        docs_in_db = db.list_documents()
        logger.info(f"✓ Retrieved document list: {docs_in_db}")
        assert len(docs_in_db) == 1, "Document count in DB is incorrect!"
        assert docs_in_db[0]["id"] == doc_id, "Document UUID in DB mismatch!"
    except Exception as e:
        logger.error(f"✗ Database persistence test failed: {str(e)}")
        sys.exit(1)
        
    # 5. Test Keyword Retrieval Fallback
    logger.info("5. Testing keyword fallback retrieval on index...")
    try:
        # Query for terms known to be in RAG Task.pdf
        query = "FastAPI backend"
        logger.info(f"   Querying: '{query}'")
        retrieved_chunks = db.retrieve(query=query, top_k=2)
        
        logger.info(f"✓ Retrieval returned {len(retrieved_chunks)} relevant matches.")
        
        for i, match in enumerate(retrieved_chunks):
            logger.info(f"   Match #{i} (Score: {match['score']}):")
            logger.info(f"      Source: {match['filename']} (chunk #{match['chunk_index']})")
            logger.info(f"      Text: {match['text'][:120]}...")
            
        assert len(retrieved_chunks) > 0, "No chunks retrieved for high-frequency terms!"
    except Exception as e:
        logger.error(f"✗ Retrieval search test failed: {str(e)}")
        sys.exit(1)
        
    # Clean up test database
    if test_db_path.exists():
        test_db_path.unlink()
        logger.info("✓ Cleaned up test database files.")
        
    logger.info("=== All Integration Tests Completed Successfully! ===")

if __name__ == "__main__":
    run_tests()
