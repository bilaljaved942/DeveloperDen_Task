# RAGGpt | Premium Glassmorphic RAG Chatbot

A high-performance, modular **Retrieval-Augmented Generation (RAG) Chatbot** featuring a stunning light-pastel and electric-blue glassmorphic UI and a robust FastAPI backend. Designed to be highly flexible, modular, and extremely lightweight—running out of the box with zero complex external dependencies or heavy deep-learning packages (no PyTorch required!).

---

## 🌟 Key Features

### 🎨 Premium UI & UX Theme (Vanilla HTML5 / CSS3 / ES6)
- **Vibrant Light Pastel Aesthetic:** A high-fidelity, Figma-inspired lilac-to-sky-blue flowing backdrop gradient on the body.
- **Glassmorphic Floating Panels:** Sidebar and settings drawers designed using clean semi-transparent panels (`backdrop-filter: blur(20px)` and white glowing borders).
- **Vibrant Royal/Electric Blue Accents:** High-contrast focus outlines, buttons, and user chat bubbles using premium electric-blue gradients.
- **Responsive Layout:** Adapts dynamically to desktop, tablet, and mobile browsers (featuring a slide-in sidebar and full-screen settings drawer on small screens).
- **SSE Chat Streaming:** Real-time Server-Sent Events with fluid typing indicators and typewriter effects.
- **Collapsible Sources Drawer:** Displays matching chunk filenames and similarity percentage scores directly below every streamed answer.

### ⚙️ Production-Grade FastAPI Backend
- **Multi-Format Document Parsing:** Native page-by-page PDF extraction (`pypdf`), DOCX paragraph & table parsing (`python-docx`), and clean TXT reader.
- **Strict File Format Validation:** Dynamic multi-stage verification restricting uploads **exclusively** to `.pdf`, `.docx`, and `.txt` files.
  - *Client-side:* Filters the file selection picker dynamically in the browser.
  - *Server-side:* Validates extensions during upload, rejecting other formats with an HTTP `400 Bad Request` error.
- **Semantic Text Splitter:** Custom `RecursiveTextSplitter` splitting long documents by double newlines, single newlines, and spaces to respect sentence boundaries.
- **Lightweight Hybrid Vector DB:** Persistent SQLite database storing chunks and embeddings.
  - **Vector Cosine Similarity Search** using fast `numpy` calculations.
  - **BM25 Keyword Fallback Search** counting phrase frequencies, allowing the app to run completely offline without neural embedding keys!
- **Flexible LLM Provider Factory:** High-speed streaming APIs supporting **Groq**, **Together AI**, **OpenAI**, and **Gemini** (utilizing async `httpx` SSE streams).
- **Batch Embedding Engine:** Connects to OpenAI, Together AI, or Hugging Face serverless embedding inference API.
- **Chat History & Context Maintenance:** Maintains full conversation memory (`chatHistory`) in the frontend state, packaging past exchanges back to the API for rich, follow-up conversation capabilities (e.g. *"Summarize what you just said in that first paragraph"*).

---

## 🛠️ Architecture Overview

The system is structured as a modular and decoupled Python package:

```
developerden_task/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI routing, SSE endpoint, static mounts
│   ├── config.py               # Env/dotenv variables configuration
│   ├── core/
│   │   ├── __init__.py
│   │   ├── parser.py           # Multi-format doc text parser (PDF, DOCX, TXT)
│   │   ├── splitter.py         # Recursive semantic text chunker
│   │   ├── vector_store.py     # SQLite db manager, numpy vector similarity & keyword search
│   │   └── llm.py              # Embedding batcher and multi-provider streaming clients
│   └── static/
│       ├── index.html          # Responsive glassmorphic landing & layout structure
│       ├── styles.css          # Curated UI variables, neon borders, and animations
│       └── app.js              # SSE streaming stream reader and dynamic controls
├── docs/                       # Automatically generated dummy test files folder
│   ├── dummy.txt               # Sample plain text document
│   ├── dummy.docx              # Sample Microsoft Word document
│   └── dummy.pdf               # Sample PDF document
├── RAG Task.pdf                # Original task description document
├── test_rag.py                 # Automated RAG pipeline integration test
├── requirements.txt            # Python dependencies list
└── README.md                   # Setup and usage guide (this file)
```

---

## 🚀 Setup & Execution Guide

### Prerequisites
- Python 3.10+ (Tested successfully on Python 3.14)
- Pip package manager

### 1. Install Dependencies
Clone or open the project folder in your terminal and create a virtual environment:

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

### 2. Configure Environment Variables (Optional)
You can pre-configure LLM API keys by creating a `.env` file in the root folder, or input them dynamically directly through the **RAG Settings** panel in the frontend UI!

Create a `.env` file:
```env
OPENAI_API_KEY=your_openai_api_key_here
GROQ_API_KEY=your_groq_api_key_here
TOGETHER_API_KEY=your_together_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
```

### 3. Run the Server
Launch the FastAPI development server using Uvicorn:

```bash
python -m uvicorn app.main:app --reload --port 8000
```

Once running, navigate your web browser to:
👉 **[http://127.0.0.1:8000](http://127.0.0.1:8000)**

---

## 🧪 Running Integration Tests

To verify that the text parsing, semantic chunk splitting, database indexing, and keyword retrieval flows are working flawlessly, run the automated integration test script:

```bash
python test_rag.py
```

### Expected Output
The test script automatically parses `RAG Task.pdf`, splits it into 5 overlapping chunks, indexes it in SQLite, and runs a search:
```text
INFO: === Starting RAG Chatbot Integration Tests ===
INFO: 1. Found test document: RAG Task.pdf
INFO: 2. Parsing PDF using DocumentParser...
INFO: ✓ PDF successfully parsed!
INFO:    Extracted text character count: 1629
3. Chunking document using RecursiveTextSplitter...
INFO: ✓ Document successfully chunked! (Generated 5 text chunks)
4. Testing SQLiteVectorStore database indexing...
INFO: ✓ Test database initialized.
INFO: ✓ Main document record indexed.
INFO: ✓ Document chunks indexed successfully.
5. Testing keyword fallback retrieval on index...
INFO:    Querying: 'FastAPI backend'
INFO: ✓ Retrieval returned 2 relevant matches.
INFO:    Match #0 (Score: 16.0): Source: RAG Task.pdf (chunk #1)...
INFO: === All Integration Tests Completed Successfully! ===
```

---

## 💡 How to Use the RAG App
1. **Upload Content:** Drag and drop your PDFs, DOCX, or text files into the left sidebar. Feel free to use the pre-generated sample documents in the **`docs/`** directory (`dummy.pdf`, `dummy.docx`, and `dummy.txt`) to test indexing!
2. **Review Knowledge Base:** Processed files will appear in the **Indexed Knowledge Base** list with chunk and size details. Click the Trash icon to remove any document.
3. **Configure Settings:** Click the blue **RAG Settings** button on the top right. Here you can:
   - Select your LLM Provider (Groq, Together AI, OpenAI, Gemini).
   - Enter your API Key.
   - Choose your specific model.
   - Adjust sliding RAG parameters (Chunk Size, Overlap, Retrieved top-K, Temperature).
4. **Chat & Stream:** Type a question in the main chat prompt (or click a Floating Suggestion card). Watch the response stream in real-time. Collapsed **Retrieved Context Sources** will display at the bottom of the assistant bubble, indicating which specific document chunks contributed to the answer.
5. **Follow-up Dialog:** Continue asking follow-up questions naturally—RAGGpt automatically tracks conversation memory to provide contextual replies.
