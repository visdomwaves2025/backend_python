# Universal Search API - Frontend Integration Guide

## 🚀 Quick Start

**Single File**: `universal_search_complete.py`  
**Single Endpoint**: `POST http://localhost:8003/universal`  
**Port**: 8003

## 📋 Setup Instructions

### 1. Install Dependencies
```bash
pip install -r requirements_US.txt
```

### 2. Update API Keys
Edit `universal_search_complete.py` lines 48-49:
```python
GEMINI_API_KEY = "your_actual_gemini_api_key"
TAVILY_API_KEY = "your_actual_tavily_api_key"
```

### 3. Run Server
```bash
python universal_search_complete.py
```

## 🎯 Frontend API Calls

### Base Configuration
```javascript
const API_BASE_URL = 'http://localhost:8003/universal';
const headers = {
    'Content-Type': 'application/json'
};
```

## 📞 API Call Examples

### 1. 💬 Chat with AI
```javascript
// Basic chat
const chatWithAI = async (message, chatId = null) => {
    const response = await fetch(API_BASE_URL, {
        method: 'POST',
        headers: headers,
        body: JSON.stringify({
            action: 'chat',
            message: message,
            chat_id: chatId
        })
    });
    return await response.json();
};

// Usage
const result = await chatWithAI("What is artificial intelligence?");
console.log(result.response); // AI's answer
console.log(result.chat_id);  // Save this for conversation continuity
```

### 2. 📄 Upload Document
```javascript
// Upload and analyze document
const uploadDocument = async (file, chatId = null, autoAnalyze = true) => {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = async (e) => {
            const base64 = e.target.result; // Already includes data:type;base64,
            
            const response = await fetch(API_BASE_URL, {
                method: 'POST',
                headers: headers,
                body: JSON.stringify({
                    action: 'upload',
                    document_base64: base64,
                    filename: file.name,
                    chat_id: chatId,
                    auto_analyze: autoAnalyze
                })
            });
            
            const result = await response.json();
            resolve(result);
        };
        reader.onerror = reject;
        reader.readAsDataURL(file);
    });
};

// Usage with file input
const fileInput = document.getElementById('fileInput');
const file = fileInput.files[0];
const result = await uploadDocument(file);
console.log(result.response); // Analysis result
console.log(result.document_info); // File details
```

### 3. 🔍 Web Search
```javascript
// Real-time web search
const webSearch = async (query, chatId = null) => {
    const response = await fetch(API_BASE_URL, {
        method: 'POST',
        headers: headers,
        body: JSON.stringify({
            action: 'web_search',
            message: query,
            chat_id: chatId
        })
    });
    return await response.json();
};

// Usage
const result = await webSearch("latest AI news 2024");
console.log(result.response); // Search results and analysis
```

### 4. 🌐 Analyze URL
```javascript
// Analyze web page content
const analyzeURL = async (url, chatId = null) => {
    const response = await fetch(API_BASE_URL, {
        method: 'POST',
        headers: headers,
        body: JSON.stringify({
            action: 'analyze_url',
            url: url,
            chat_id: chatId
        })
    });
    return await response.json();
};

// Usage
const result = await analyzeURL("https://example.com/article");
console.log(result.response); // URL content analysis
```

### 5. 📚 Get Chat History
```javascript
// Get conversation history
const getChatHistory = async (chatId) => {
    const response = await fetch(API_BASE_URL, {
        method: 'POST',
        headers: headers,
        body: JSON.stringify({
            action: 'get_history',
            chat_id: chatId
        })
    });
    return await response.json();
};

// Usage
const result = await getChatHistory("chat_20241210_132700.json");
console.log(result.history); // Array of {question, answer} objects
console.log(result.last_file); // Last uploaded file name
```

### 6. 📋 List All Chats
```javascript
// Get list of all chat sessions
const listChats = async () => {
    const response = await fetch(API_BASE_URL, {
        method: 'POST',
        headers: headers,
        body: JSON.stringify({
            action: 'list_chats'
        })
    });
    return await response.json();
};

// Usage
const result = await listChats();
console.log(result.chats); // Array of chat session IDs
```

### 7. 🗑️ Delete Chat
```javascript
// Delete a chat session
const deleteChat = async (chatId) => {
    const response = await fetch(API_BASE_URL, {
        method: 'POST',
        headers: headers,
        body: JSON.stringify({
            action: 'delete_chat',
            chat_id: chatId
        })
    });
    return await response.json();
};

// Usage
const result = await deleteChat("chat_20241210_132700.json");
console.log(result.message); // Deletion confirmation
```

## 🔧 Complete Frontend Integration Example

```html
<!DOCTYPE html>
<html>
<head>
    <title>Universal Search Frontend</title>
</head>
<body>
    <div id="app">
        <!-- Chat Interface -->
        <div id="chatContainer">
            <div id="messages"></div>
            <input type="text" id="messageInput" placeholder="Ask me anything...">
            <button onclick="sendMessage()">Send</button>
        </div>

        <!-- File Upload -->
        <div id="uploadContainer">
            <input type="file" id="fileInput" accept=".pdf,.txt,.png,.jpg,.jpeg">
            <button onclick="uploadFile()">Upload & Analyze</button>
        </div>

        <!-- URL Analysis -->
        <div id="urlContainer">
            <input type="url" id="urlInput" placeholder="Enter URL to analyze...">
            <button onclick="analyzeURL()">Analyze URL</button>
        </div>

        <!-- Web Search -->
        <div id="searchContainer">
            <input type="text" id="searchInput" placeholder="Search the web...">
            <button onclick="webSearch()">Search</button>
        </div>
    </div>

    <script>
        const API_BASE_URL = 'http://localhost:8003/universal';
        let currentChatId = null;

        // Universal API call function
        async function callAPI(action, data = {}) {
            try {
                const response = await fetch(API_BASE_URL, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        action: action,
                        chat_id: currentChatId,
                        ...data
                    })
                });
                
                const result = await response.json();
                
                if (result.success && result.chat_id) {
                    currentChatId = result.chat_id;
                }
                
                return result;
            } catch (error) {
                console.error('API Error:', error);
                return { success: false, message: error.message };
            }
        }

        // Display message in chat
        function displayMessage(question, answer, isError = false) {
            const messagesDiv = document.getElementById('messages');
            const messageDiv = document.createElement('div');
            messageDiv.innerHTML = `
                <div class="question"><strong>You:</strong> ${question}</div>
                <div class="answer ${isError ? 'error' : ''}"><strong>Chatbot:</strong> ${answer}</div>
            `;
            messagesDiv.appendChild(messageDiv);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
        }

        // Send chat message
        async function sendMessage() {
            const input = document.getElementById('messageInput');
            const message = input.value.trim();
            if (!message) return;

            input.value = '';
            displayMessage(message, 'Thinking...');

            const result = await callAPI('chat', { message: message });
            
            // Update the last message with the response
            const messages = document.getElementById('messages');
            const lastAnswer = messages.lastElementChild.querySelector('.answer');
            lastAnswer.textContent = `Chatbot: ${result.success ? result.response : result.message}`;
            lastAnswer.className = `answer ${result.success ? '' : 'error'}`;
        }

        // Upload and analyze file
        async function uploadFile() {
            const fileInput = document.getElementById('fileInput');
            const file = fileInput.files[0];
            if (!file) return;

            displayMessage(`Uploading: ${file.name}`, 'Processing file...');

            const reader = new FileReader();
            reader.onload = async (e) => {
                const result = await callAPI('upload', {
                    document_base64: e.target.result,
                    filename: file.name,
                    auto_analyze: true
                });

                const messages = document.getElementById('messages');
                const lastAnswer = messages.lastElementChild.querySelector('.answer');
                lastAnswer.textContent = `Chatbot: ${result.success ? result.response : result.message}`;
                lastAnswer.className = `answer ${result.success ? '' : 'error'}`;
            };
            reader.readAsDataURL(file);
        }

        // Analyze URL
        async function analyzeURL() {
            const urlInput = document.getElementById('urlInput');
            const url = urlInput.value.trim();
            if (!url) return;

            urlInput.value = '';
            displayMessage(`Analyzing: ${url}`, 'Fetching and analyzing content...');

            const result = await callAPI('analyze_url', { url: url });

            const messages = document.getElementById('messages');
            const lastAnswer = messages.lastElementChild.querySelector('.answer');
            lastAnswer.textContent = `Chatbot: ${result.success ? result.response : result.message}`;
            lastAnswer.className = `answer ${result.success ? '' : 'error'}`;
        }

        // Web search
        async function webSearch() {
            const searchInput = document.getElementById('searchInput');
            const query = searchInput.value.trim();
            if (!query) return;

            searchInput.value = '';
            displayMessage(`Searching: ${query}`, 'Searching the web...');

            const result = await callAPI('web_search', { message: query });

            const messages = document.getElementById('messages');
            const lastAnswer = messages.lastElementChild.querySelector('.answer');
            lastAnswer.textContent = `Chatbot: ${result.success ? result.response : result.message}`;
            lastAnswer.className = `answer ${result.success ? '' : 'error'}`;
        }

        // Enter key support for inputs
        document.getElementById('messageInput').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') sendMessage();
        });

        document.getElementById('searchInput').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') webSearch();
        });

        document.getElementById('urlInput').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') analyzeURL();
        });
    </script>

    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        #chatContainer { margin-bottom: 20px; }
        #messages { height: 400px; overflow-y: auto; border: 1px solid #ccc; padding: 10px; margin-bottom: 10px; }
        .question { color: #0066cc; margin-bottom: 5px; }
        .answer { color: #333; margin-bottom: 15px; }
        .error { color: #cc0000; }
        input[type="text"], input[type="url"] { width: 300px; padding: 5px; }
        input[type="file"] { margin-bottom: 10px; }
        button { padding: 5px 10px; margin-left: 5px; }
        div { margin-bottom: 10px; }
    </style>
</body>
</html>
```

## 📊 Response Format

All API calls return this consistent format:

```json
{
    "success": true/false,
    "action": "action_name",
    "response": "AI response text",
    "chat_id": "chat_session_id",
    "document_info": { "filename": "...", "pages": 5, "words": 1500 },
    "chats": ["chat1.json", "chat2.json"],
    "history": [{"question": "...", "answer": "..."}],
    "last_file": "filename.pdf",
    "message": "Status message"
}
```

## 🔍 Error Handling

```javascript
const result = await callAPI('chat', { message: 'Hello' });

if (result.success) {
    console.log('Success:', result.response);
    // Update UI with response
} else {
    console.error('Error:', result.message);
    // Show error to user
}
```

## 🎯 Key Points for Frontend Integration

1. **Single Endpoint**: All functionality through `/universal`
2. **Action-Based**: Specify action in request body
3. **Chat Continuity**: Save and reuse `chat_id`
4. **File Upload**: Convert to base64 before sending
5. **Error Handling**: Check `success` field in response
6. **Consistent Format**: Same response structure for all actions

## 🚀 Production Considerations

1. **CORS**: Already configured for all origins
2. **File Size**: Consider adding file size limits
3. **Rate Limiting**: Add rate limiting for production
4. **Authentication**: Add API key authentication if needed
5. **HTTPS**: Use HTTPS in production
6. **Error Logging**: Implement proper error logging

This guide provides everything needed to integrate the Universal Search API into any frontend application!
