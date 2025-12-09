"""
Universal Search API - Complete Single File
All-in-one AI-powered search and document analysis service
"""
import os
import io
import json
import textwrap
import re
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
from PIL import Image
import pytesseract
import pdfplumber
from bs4 import BeautifulSoup
import fitz  # PyMuPDF
import base64

# LangChain + Gemini LLM
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent, Tool

# Tavily search
from langchain_community.tools.tavily_search import TavilySearchResults

# ============================================================================
# FASTAPI APP SETUP
# ============================================================================

app = FastAPI(
    title="Universal Search API",
    description="AI-powered search and document analysis service",
    version="2.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# CONFIGURATION & INITIALIZATION
# ============================================================================

# Load API Keys from secrets.json
try:
    with open("secrets.json", "r") as f:
        secrets = json.load(f)
    GEMINI_API_KEY = secrets["gemini_api_key"]
    TAVILY_API_KEY = secrets["tavily_api_key"]
except FileNotFoundError:
    print("❌ secrets.json not found! Please create it with your API keys.")
    exit(1)
except KeyError as e:
    print(f"❌ Missing key in secrets.json: {e}")
    exit(1)

# Configuration
CONFIG = {
    "assistant_name": "Chatbot",
    "version": "2.0",
    "port": 8003
}

# Directories
CHAT_DIR = "chat_sessions"
UPLOAD_DIR = "uploads"
GLOBAL_MEMORY_PATH = os.path.join(CHAT_DIR, "global_memory.json")

# Create directories
os.makedirs(CHAT_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)
if not os.path.exists(GLOBAL_MEMORY_PATH):
    with open(GLOBAL_MEMORY_PATH, "w", encoding="utf-8") as f:
        json.dump([], f)

# Initialize LLM and tools
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp",
    google_api_key=GEMINI_API_KEY,
)

tavily_search = TavilySearchResults(
    tavily_api_key=TAVILY_API_KEY,
    max_results=5,
    include_answer=True,
    include_raw_content=False,
)

tools = [
    Tool(
        name="tavily_search",
        func=tavily_search.run,
        description="Search the web for real-time information",
    )
]

agent = initialize_agent(
    tools,
    llm,
    agent="zero-shot-react-description",
    verbose=False,
    handle_parsing_errors=True,
)

# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class UniversalRequest(BaseModel):
    action: str  # "chat", "upload", "web_search", "analyze_url", "get_history", "list_chats", "delete_chat"
    message: Optional[str] = None
    document_base64: Optional[str] = None
    filename: Optional[str] = None
    url: Optional[str] = None
    chat_id: Optional[str] = None
    auto_analyze: bool = True

class UniversalResponse(BaseModel):
    success: bool
    action: str
    response: Optional[str] = None
    chat_id: Optional[str] = None
    document_info: Optional[Dict[str, Any]] = None
    chats: Optional[List[str]] = None
    history: Optional[List[Dict[str, str]]] = None
    last_file: Optional[str] = None
    message: str

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def empty_chat_data():
    return {
        "chat_history": [],
        "doc_text": "",
        "last_file": None,
        "last_file_meta": {},
    }

def new_chat_id() -> str:
    return f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

def chat_path(chat_id: str) -> str:
    return os.path.join(CHAT_DIR, chat_id)

def load_chat(chat_id: str):
    path = chat_path(chat_id)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return empty_chat_data()

def save_chat(chat_id: str, data: dict):
    with open(chat_path(chat_id), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def list_chats():
    return sorted(
        f for f in os.listdir(CHAT_DIR)
        if f.endswith(".json") and f != "global_memory.json"
    )

def update_global_memory(question: str, answer: str):
    mem = []
    if os.path.exists(GLOBAL_MEMORY_PATH):
        with open(GLOBAL_MEMORY_PATH, encoding="utf-8") as f:
            mem = json.load(f)
    mem.append({
        "question": question,
        "answer": answer,
        "time": datetime.now(timezone.utc).isoformat(),
    })
    with open(GLOBAL_MEMORY_PATH, "w", encoding="utf-8") as f:
        json.dump(mem, f, indent=2, ensure_ascii=False)

def safe_filename(filename: str) -> str:
    return filename.replace(" ", "_").replace("/", "_").replace("\\", "_")

# ============================================================================
# DOCUMENT PROCESSING FUNCTIONS
# ============================================================================

def ocr_pil_image(pil_img: Image.Image) -> str:
    try:
        return pytesseract.image_to_string(pil_img)
    except Exception as e:
        print(f"OCR Error: {e}")
        return ""

def extract_text_from_pdf_bytes(file_bytes: bytes):
    text_chunks = []
    pages = 0

    # Try text extraction first
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            pages = len(pdf.pages)
            for page in pdf.pages:
                t = page.extract_text() or ""
                text_chunks.append(t)
    except Exception as e:
        print(f"PDF text extraction error: {e}")
        text_chunks = []
        pages = 0

    # If no text found, use OCR
    if not any(t.strip() for t in text_chunks):
        try:
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            pages = doc.page_count
            text_chunks = []
            for i in range(pages):
                page = doc[i]
                pix = page.get_pixmap(dpi=200)
                pil_img = Image.open(io.BytesIO(pix.tobytes("png")))
                t = ocr_pil_image(pil_img)
                text_chunks.append(t or "")
            doc.close()
        except Exception as e:
            print(f"PDF OCR error: {e}")
            text_chunks = []
            pages = 0

    text = "\n\n".join(t.strip() for t in text_chunks if t and t.strip())
    words = len(text.split()) if text else 0
    return text, pages, words

def extract_text_from_image_bytes(file_bytes: bytes):
    try:
        img = Image.open(io.BytesIO(file_bytes))
        txt = ocr_pil_image(img)
        txt = " ".join(txt.split())
        return txt, 1, len(txt.split()) if txt else 0
    except Exception as e:
        print(f"Image processing error: {e}")
        return "", 0, 0

def base64_to_bytes(base64_string: str) -> bytes:
    try:
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        return base64.b64decode(base64_string)
    except Exception as e:
        raise ValueError(f"Invalid base64 format: {str(e)}")

# ============================================================================
# AI PROCESSING FUNCTIONS
# ============================================================================

def analyze_text_with_llm(text: str, filename=None, page_count=None, word_count=None):
    meta = []
    if filename:
        meta.append(f"Filename: {filename}")
    if page_count is not None:
        meta.append(f"Pages: {page_count}")
    if word_count is not None:
        meta.append(f"Words: {word_count}")
    meta_line = " | ".join(meta)

    prompt = (
        "You are Chatbot, an AI document analyst. Analyze the following document and provide:\n"
        "1. A concise summary\n"
        "2. Key points and topics\n"
        "3. Notable observations\n"
        "4. Any actionable insights\n\n"
        f"{'[' + meta_line + ']' if meta_line else ''}\n\n"
        f"Document Content:\n{textwrap.shorten(text, 10000)}"
    )

    try:
        res = llm.invoke(prompt)
        return res.content
    except Exception as e:
        return f"Analysis error: {e}"

def build_doc_aware_prompt(user_message: str, chat_data: dict) -> str:
    # Get recent chat history
    ctx_lines = []
    for h in chat_data.get("chat_history", [])[-6:]:
        ctx_lines.append(f"User: {h['question']}")
        ctx_lines.append(f"Chatbot: {h['answer']}")
    context = "\n".join(ctx_lines)

    doc_text = chat_data.get("doc_text") or ""
    doc_meta = chat_data.get("last_file_meta", {})
    doc_name = chat_data.get("last_file")

    if doc_text.strip():
        doc_snippet = textwrap.shorten(doc_text, 8000)
        doc_info = f"Document: {doc_name} | Pages: {doc_meta.get('pages')} | Words: {doc_meta.get('words')}"

        prompt = f"""You are Chatbot, an intelligent AI assistant with access to a document.

DOCUMENT CONTEXT:
{doc_info}
--- DOCUMENT CONTENT ---
{doc_snippet}
--- END DOCUMENT ---

CHAT HISTORY:
{context}

USER: {user_message}

CHATBOT: """
    else:
        prompt = f"""You are Chatbot, an intelligent AI assistant.

CHAT HISTORY:
{context}

USER: {user_message}

CHATBOT: """

    return prompt

def answer_user_message(user_message: str, chat_data: dict) -> str:
    low = user_message.lower()

    # Check if web search is needed
    web_indicators = [
        "latest", "current", "today", "now", "recent", "2024", "2025",
        "news", "weather", "stock", "price", "score", "live", "update"
    ]
    needs_web = any(indicator in low for indicator in web_indicators)

    if needs_web:
        try:
            search_results = tavily_search.run(user_message)
            prompt = f"""You are Chatbot. Based on this web search, provide a comprehensive answer:

WEB SEARCH RESULTS:
{search_results}

USER QUESTION: {user_message}

Provide a clear, informative response based on the search results."""

            res = llm.invoke(prompt)
            return res.content
        except Exception as e:
            return f"Web search error: {e}"

    # Regular chat with document context
    prompt = build_doc_aware_prompt(user_message, chat_data)
    try:
        res = llm.invoke(prompt)
        return res.content
    except Exception as e:
        return f"Chat error: {e}"

# ============================================================================
# ACTION HANDLERS
# ============================================================================

async def handle_chat(request: UniversalRequest) -> UniversalResponse:
    if not request.message or not request.message.strip():
        return UniversalResponse(
            success=False,
            action="chat",
            message="Message cannot be empty"
        )
    
    chat_id = request.chat_id or new_chat_id()
    chat_data = load_chat(chat_id)
    
    try:
        user_message = request.message.strip()
        low = user_message.lower()
        
        # Handle document analysis commands
        if (any(kw in low for kw in ["analyze", "summarize", "summary"]) and 
            any(kw in low for kw in ["document", "file", "pdf", "image", "doc"])):
            if chat_data.get("doc_text", "").strip():
                meta = chat_data.get("last_file_meta", {})
                filename = chat_data.get("last_file")
                response = analyze_text_with_llm(
                    chat_data["doc_text"],
                    filename=filename,
                    page_count=meta.get("pages"),
                    word_count=meta.get("words"),
                )
            else:
                response = "No document found in this chat. Please upload a document first using the 'upload' action."
        else:
            response = answer_user_message(user_message, chat_data)
        
        # Save to history
        chat_data["chat_history"].append({"question": user_message, "answer": response})
        save_chat(chat_id, chat_data)
        update_global_memory(user_message, response)
        
        return UniversalResponse(
            success=True,
            action="chat",
            response=response,
            chat_id=chat_id,
            message="Success"
        )
        
    except Exception as e:
        return UniversalResponse(
            success=False,
            action="chat",
            message=f"Chat error: {str(e)}"
        )

async def handle_upload(request: UniversalRequest) -> UniversalResponse:
    if not request.document_base64 or not request.filename:
        return UniversalResponse(
            success=False,
            action="upload",
            message="Both document_base64 and filename are required"
        )
    
    chat_id = request.chat_id or new_chat_id()
    chat_data = load_chat(chat_id)
    
    try:
        file_content = base64_to_bytes(request.document_base64)
        filename = safe_filename(request.filename)
        ext = os.path.splitext(filename)[1].lower()
        
        # Save file
        saved_path = os.path.join(UPLOAD_DIR, filename)
        with open(saved_path, "wb") as f:
            f.write(file_content)
        
        # Extract text based on file type
        if ext == ".txt":
            text = file_content.decode("utf-8", errors="ignore")
            pages, words = 1, len(text.split())
        elif ext == ".pdf":
            text, pages, words = extract_text_from_pdf_bytes(file_content)
        elif ext in [".png", ".jpg", ".jpeg", ".gif", ".bmp"]:
            text, pages, words = extract_text_from_image_bytes(file_content)
        else:
            return UniversalResponse(
                success=False,
                action="upload",
                message=f"Unsupported file type: {ext}. Supported: .pdf, .txt, .png, .jpg, .jpeg"
            )
        
        if not text.strip():
            return UniversalResponse(
                success=False,
                action="upload",
                message="No readable text found in the uploaded file"
            )
        
        # Update chat data
        chat_data["doc_text"] = text
        chat_data["last_file"] = filename
        chat_data["last_file_meta"] = {
            "pages": pages,
            "words": words,
            "uploaded_at": datetime.now(timezone.utc).isoformat(),
        }
        
        # Generate analysis if requested
        if request.auto_analyze:
            analysis = analyze_text_with_llm(text, filename=filename, page_count=pages, word_count=words)
            chat_data["chat_history"].append({
                "question": f"Uploaded: {filename}",
                "answer": analysis
            })
        else:
            analysis = f"File '{filename}' uploaded successfully. Use 'analyze document' to get analysis."
            chat_data["chat_history"].append({
                "question": f"Uploaded: {filename}",
                "answer": analysis
            })
        
        save_chat(chat_id, chat_data)
        
        return UniversalResponse(
            success=True,
            action="upload",
            response=analysis,
            chat_id=chat_id,
            document_info={
                "filename": filename,
                "pages": pages,
                "words": words,
                "type": ext
            },
            message="Document uploaded successfully"
        )
        
    except Exception as e:
        return UniversalResponse(
            success=False,
            action="upload",
            message=f"Upload error: {str(e)}"
        )

async def handle_web_search(request: UniversalRequest) -> UniversalResponse:
    if not request.message or not request.message.strip():
        return UniversalResponse(
            success=False,
            action="web_search",
            message="Search query (message) cannot be empty"
        )
    
    chat_id = request.chat_id or new_chat_id()
    chat_data = load_chat(chat_id)
    
    try:
        query = request.message.strip()
        search_results = tavily_search.run(query)
        
        prompt = f"""You are Chatbot. Based on this web search, provide a comprehensive and informative answer:

SEARCH QUERY: {query}

SEARCH RESULTS:
{search_results}

Provide a well-structured response with key information, sources when relevant, and actionable insights."""
        
        response = llm.invoke(prompt).content
        
        # Save to history
        chat_data["chat_history"].append({
            "question": f"Web Search: {query}",
            "answer": response
        })
        save_chat(chat_id, chat_data)
        update_global_memory(f"Web search: {query}", response)
        
        return UniversalResponse(
            success=True,
            action="web_search",
            response=response,
            chat_id=chat_id,
            message="Web search completed successfully"
        )
        
    except Exception as e:
        return UniversalResponse(
            success=False,
            action="web_search",
            message=f"Web search error: {str(e)}"
        )

async def handle_url_analysis(request: UniversalRequest) -> UniversalResponse:
    if not request.url or not request.url.strip():
        return UniversalResponse(
            success=False,
            action="analyze_url",
            message="URL cannot be empty"
        )
    
    chat_id = request.chat_id or new_chat_id()
    chat_data = load_chat(chat_id)
    
    try:
        url = request.url.strip()
        
        # Fetch webpage content
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        # Parse content
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get text content
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        # Analyze with AI
        prompt = f"""You are Chatbot. Analyze this web page content and provide:

1. Main topic/purpose of the page
2. Key information and insights
3. Summary of important points
4. Any notable findings

URL: {url}
Content: {textwrap.shorten(text, 8000)}

Provide a comprehensive analysis:"""
        
        analysis = llm.invoke(prompt).content
        
        # Save to history
        chat_data["chat_history"].append({
            "question": f"URL Analysis: {url}",
            "answer": analysis
        })
        save_chat(chat_id, chat_data)
        update_global_memory(f"URL analysis: {url}", analysis)
        
        return UniversalResponse(
            success=True,
            action="analyze_url",
            response=analysis,
            chat_id=chat_id,
            message="URL analyzed successfully"
        )
        
    except requests.RequestException as e:
        return UniversalResponse(
            success=False,
            action="analyze_url",
            message=f"Failed to fetch URL: {str(e)}"
        )
    except Exception as e:
        return UniversalResponse(
            success=False,
            action="analyze_url",
            message=f"URL analysis error: {str(e)}"
        )

async def handle_get_history(request: UniversalRequest) -> UniversalResponse:
    if not request.chat_id:
        return UniversalResponse(
            success=False,
            action="get_history",
            message="chat_id is required to get history"
        )
    
    try:
        chat_data = load_chat(request.chat_id)
        return UniversalResponse(
            success=True,
            action="get_history",
            chat_id=request.chat_id,
            history=chat_data.get("chat_history", []),
            last_file=chat_data.get("last_file"),
            message="Chat history retrieved successfully"
        )
    except Exception as e:
        return UniversalResponse(
            success=False,
            action="get_history",
            message=f"Error retrieving history: {str(e)}"
        )

async def handle_list_chats(request: UniversalRequest) -> UniversalResponse:
    try:
        chats = list_chats()
        return UniversalResponse(
            success=True,
            action="list_chats",
            chats=chats,
            message=f"Found {len(chats)} chat sessions"
        )
    except Exception as e:
        return UniversalResponse(
            success=False,
            action="list_chats",
            message=f"Error listing chats: {str(e)}"
        )

async def handle_delete_chat(request: UniversalRequest) -> UniversalResponse:
    if not request.chat_id:
        return UniversalResponse(
            success=False,
            action="delete_chat",
            message="chat_id is required to delete chat"
        )
    
    try:
        chat_file_path = chat_path(request.chat_id)
        if os.path.exists(chat_file_path):
            os.remove(chat_file_path)
            return UniversalResponse(
                success=True,
                action="delete_chat",
                message=f"Chat '{request.chat_id}' deleted successfully"
            )
        else:
            return UniversalResponse(
                success=False,
                action="delete_chat",
                message=f"Chat '{request.chat_id}' not found"
            )
    except Exception as e:
        return UniversalResponse(
            success=False,
            action="delete_chat",
            message=f"Error deleting chat: {str(e)}"
        )

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint - API information"""
    return {
        "service": "Universal Search API",
        "version": CONFIG["version"],
        "assistant": CONFIG["assistant_name"],
        "status": "active",
        "endpoints": {
            "main": "POST /universal - Single endpoint for all actions",
            "health": "GET /health - Health check"
        },
        "available_actions": [
            "chat - AI conversation",
            "upload - Document upload & analysis",
            "web_search - Real-time web search", 
            "analyze_url - URL content analysis",
            "get_history - Retrieve chat history",
            "list_chats - List all chat sessions",
            "delete_chat - Delete chat session"
        ],
        "documentation": "See /docs for interactive API documentation"
    }

@app.post("/universal", response_model=UniversalResponse)
async def universal_endpoint(request: UniversalRequest):
    """
    Universal endpoint for all search and AI functionality
    
    Actions:
    - chat: AI conversation with document context
    - upload: Upload and analyze documents (PDF, images, text)
    - web_search: Real-time web search
    - analyze_url: Analyze web page content
    - get_history: Get chat session history
    - list_chats: List all chat sessions
    - delete_chat: Delete a chat session
    """
    try:
        action = request.action.lower().strip()
        
        if action == "chat":
            return await handle_chat(request)
        elif action == "upload":
            return await handle_upload(request)
        elif action == "web_search":
            return await handle_web_search(request)
        elif action == "analyze_url":
            return await handle_url_analysis(request)
        elif action == "get_history":
            return await handle_get_history(request)
        elif action == "list_chats":
            return await handle_list_chats(request)
        elif action == "delete_chat":
            return await handle_delete_chat(request)
        else:
            available_actions = ["chat", "upload", "web_search", "analyze_url", "get_history", "list_chats", "delete_chat"]
            return UniversalResponse(
                success=False,
                action=action,
                message=f"Unknown action '{action}'. Available actions: {', '.join(available_actions)}"
            )
            
    except Exception as e:
        return UniversalResponse(
            success=False,
            action=getattr(request, 'action', 'unknown'),
            message=f"Server error: {str(e)}"
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Universal Search API",
        "version": CONFIG["version"],
        "assistant": CONFIG["assistant_name"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "endpoints_active": True
    }

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    print("=" * 60)
    print("UNIVERSAL SEARCH API - STARTING")
    print("=" * 60)
    print(f"Assistant: {CONFIG['assistant_name']}")
    print(f"Server: http://localhost:{CONFIG['port']}")
    print(f"API Docs: http://localhost:{CONFIG['port']}/docs")
    print(f"Main Endpoint: POST /universal")
    print()
    print("Available Actions:")
    print("  chat - AI conversation")
    print("  upload - Document analysis")
    print("  web_search - Real-time search")
    print("  analyze_url - URL analysis")
    print("  get_history - Chat history")
    print("  list_chats - List chats")
    print("  delete_chat - Delete chat")
    print("=" * 60)
    
    uvicorn.run(app, host="127.0.0.1", port=CONFIG["port"])
