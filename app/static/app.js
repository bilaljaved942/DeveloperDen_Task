/* ==========================================
   Antigravity Premium RAG Chatbot Javascript
   ========================================== */

// Default Configurations
const PROVIDER_MODELS = {
    groq: [
        { id: "llama-3.3-70b-versatile", name: "Llama 3.3 70B (Default)" },
        { id: "mixtral-8x7b-32768", name: "Mixtral 8x7B" },
        { id: "gemma2-9b-it", name: "Gemma 2 9B" }
    ],
    together: [
        { id: "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo", name: "Llama 3.1 8B (Default)" },
        { id: "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo", name: "Llama 3.1 70B" },
        { id: "mistralai/Mixtral-8x7B-Instruct-v0.1", name: "Mixtral 8x7B" }
    ],
    openai: [
        { id: "gpt-4o-mini", name: "GPT-4o Mini (Default)" },
        { id: "gpt-4o", name: "GPT-4o" },
        { id: "gpt-3.5-turbo", name: "GPT-3.5 Turbo" }
    ],
    gemini: [
        { id: "gemini-1.5-flash", name: "Gemini 1.5 Flash (Default)" },
        { id: "gemini-1.5-pro", name: "Gemini 1.5 Pro" }
    ]
};

// Application State
let state = {
    chatHistory: [],
    documents: [],
    // Settings configuration with local storage load/fallback defaults
    config: {
        provider: localStorage.getItem("rag_provider") || "groq",
        apiKey: localStorage.getItem("rag_apiKey") || "",
        model: localStorage.getItem("rag_model") || "llama-3.3-70b-versatile",
        embeddingProvider: localStorage.getItem("rag_embProvider") || "huggingface",
        embeddingApiKey: localStorage.getItem("rag_embApiKey") || "",
        chunkSize: parseInt(localStorage.getItem("rag_chunkSize")) || 1000,
        chunkOverlap: parseInt(localStorage.getItem("rag_chunkOverlap")) || 200,
        topK: parseInt(localStorage.getItem("rag_topK")) || 4,
        temperature: parseFloat(localStorage.getItem("rag_temperature")) || 0.3
    }
};

// Document Elements cache
const els = {
    sidebar: document.getElementById("sidebar"),
    sidebarToggleBtn: document.getElementById("sidebarToggleBtn"),
    sidebarCloseBtn: document.getElementById("sidebarCloseBtn"),
    uploadZone: document.getElementById("uploadZone"),
    fileInput: document.getElementById("fileInput"),
    browseBtn: document.getElementById("browseBtn"),
    uploadProgress: document.getElementById("uploadProgress"),
    progressFileName: document.getElementById("progressFileName"),
    progressBarFill: document.getElementById("progressBarFill"),
    progressPercent: document.getElementById("progressPercent"),
    stageRead: document.getElementById("stageRead"),
    stageChunk: document.getElementById("stageChunk"),
    stageEmbed: document.getElementById("stageEmbed"),
    documentList: document.getElementById("documentList"),
    documentBadge: document.getElementById("documentBadge"),
    clearStoreBtn: document.getElementById("clearStoreBtn"),
    settingsToggleBtn: document.getElementById("settingsToggleBtn"),
    settingsDrawer: document.getElementById("settingsDrawer"),
    settingsCloseBtn: document.getElementById("settingsCloseBtn"),
    settingsSaveBtn: document.getElementById("settingsSaveBtn"),
    overlay: document.getElementById("overlay"),
    
    // Config items
    providerSelect: document.getElementById("providerSelect"),
    apiKeyInput: document.getElementById("apiKeyInput"),
    modelSelect: document.getElementById("modelSelect"),
    eyeToggle: document.getElementById("eyeToggle"),
    embeddingProviderSelect: document.getElementById("embeddingProviderSelect"),
    embeddingKeyGroup: document.getElementById("embeddingKeyGroup"),
    embeddingApiKeyInput: document.getElementById("embeddingApiKeyInput"),
    chunkSizeRange: document.getElementById("chunkSizeRange"),
    chunkSizeVal: document.getElementById("chunkSizeVal"),
    chunkOverlapRange: document.getElementById("chunkOverlapRange"),
    chunkOverlapVal: document.getElementById("chunkOverlapVal"),
    topKRange: document.getElementById("topKRange"),
    topKVal: document.getElementById("topKVal"),
    tempRange: document.getElementById("tempRange"),
    tempVal: document.getElementById("tempVal"),
    
    // Chat items
    chatMessages: document.getElementById("chatMessages"),
    welcomeScreen: document.getElementById("welcomeScreen"),
    chatForm: document.getElementById("chatForm"),
    chatInput: document.getElementById("chatInput"),
    sendBtn: document.getElementById("sendBtn"),
    clearChatBtn: document.getElementById("clearChatBtn"),
    typingIndicator: document.getElementById("typingIndicator"),
    statusMessage: document.getElementById("statusMessage"),
    activeDocsCount: document.getElementById("activeDocsCount")
};

// Initial setup
document.addEventListener("DOMContentLoaded", () => {
    initUIValues();
    setupEventHandlers();
    fetchDocuments();
    configureMarked();
});

// Configure Marked Markdown compiler
function configureMarked() {
    marked.setOptions({
        breaks: true,
        gfm: true,
        headerIds: false,
        mangle: false
    });
}

// Sync Form UI inputs with current State Config
function initUIValues() {
    els.providerSelect.value = state.config.provider;
    els.apiKeyInput.value = state.config.apiKey;
    els.embeddingProviderSelect.value = state.config.embeddingProvider;
    els.embeddingApiKeyInput.value = state.config.embeddingApiKey;
    
    els.chunkSizeRange.value = state.config.chunkSize;
    els.chunkSizeVal.textContent = `${state.config.chunkSize} chars`;
    els.chunkOverlapRange.value = state.config.chunkOverlap;
    els.chunkOverlapVal.textContent = `${state.config.chunkOverlap} chars`;
    els.topKRange.value = state.config.topK;
    els.topKVal.textContent = `${state.config.topK} chunks`;
    els.tempRange.value = state.config.temperature;
    els.tempVal.textContent = state.config.temperature;
    
    toggleEmbeddingKeyField();
    updateModelOptions(state.config.provider, state.config.model);
}

// Populate model select dynamically
function updateModelOptions(provider, selectedModel) {
    els.modelSelect.innerHTML = "";
    const models = PROVIDER_MODELS[provider] || [];
    models.forEach(model => {
        const opt = document.createElement("option");
        opt.value = model.id;
        opt.textContent = model.name;
        els.modelSelect.appendChild(opt);
    });
    
    if (selectedModel && models.some(m => m.id === selectedModel)) {
        els.modelSelect.value = selectedModel;
    } else if (models.length > 0) {
        els.modelSelect.value = models[0].id;
    }
}

function toggleEmbeddingKeyField() {
    const prov = els.embeddingProviderSelect.value;
    if (prov === "openai" || prov === "together") {
        els.embeddingKeyGroup.style.display = "block";
    } else {
        els.embeddingKeyGroup.style.display = "none";
    }
}

// Save options from drawer into State & LocalStorage
function saveSettings() {
    state.config.provider = els.providerSelect.value;
    state.config.apiKey = els.apiKeyInput.value;
    state.config.model = els.modelSelect.value;
    state.config.embeddingProvider = els.embeddingProviderSelect.value;
    state.config.embeddingApiKey = els.embeddingApiKeyInput.value;
    state.config.chunkSize = parseInt(els.chunkSizeRange.value);
    state.config.chunkOverlap = parseInt(els.chunkOverlapRange.value);
    state.config.topK = parseInt(els.topKRange.value);
    state.config.temperature = parseFloat(els.tempRange.value);
    
    localStorage.setItem("rag_provider", state.config.provider);
    localStorage.setItem("rag_apiKey", state.config.apiKey);
    localStorage.setItem("rag_model", state.config.model);
    localStorage.setItem("rag_embProvider", state.config.embeddingProvider);
    localStorage.setItem("rag_embApiKey", state.config.embeddingApiKey);
    localStorage.setItem("rag_chunkSize", state.config.chunkSize);
    localStorage.setItem("rag_chunkOverlap", state.config.chunkOverlap);
    localStorage.setItem("rag_topK", state.config.topK);
    localStorage.setItem("rag_temperature", state.config.temperature);
    
    showNotification("Configurations successfully applied!", "success");
    closeSettingsDrawer();
}

// Setup standard event handlers
function setupEventHandlers() {
    // Sidebar drawer toggles
    els.sidebarToggleBtn.addEventListener("click", () => els.sidebar.classList.add("open"));
    els.sidebarCloseBtn.addEventListener("click", () => els.sidebar.classList.remove("open"));
    
    // Settings panel toggles
    els.settingsToggleBtn.addEventListener("click", openSettingsDrawer);
    els.settingsCloseBtn.addEventListener("click", closeSettingsDrawer);
    els.settingsSaveBtn.addEventListener("click", saveSettings);
    els.overlay.addEventListener("click", closeSettingsDrawer);
    
    // Eye key password mask switcher
    els.eyeToggle.addEventListener("click", () => {
        const type = els.apiKeyInput.getAttribute("type") === "password" ? "text" : "password";
        els.apiKeyInput.setAttribute("type", type);
    });
    
    // Slider values updates
    els.chunkSizeRange.addEventListener("input", (e) => els.chunkSizeVal.textContent = `${e.target.value} chars`);
    els.chunkOverlapRange.addEventListener("input", (e) => els.chunkOverlapVal.textContent = `${e.target.value} chars`);
    els.topKRange.addEventListener("input", (e) => els.topKVal.textContent = `${e.target.value} chunks`);
    els.tempRange.addEventListener("input", (e) => els.tempVal.textContent = e.target.value);
    
    // Provider list updates models selection
    els.providerSelect.addEventListener("change", (e) => {
        updateModelOptions(e.target.value);
    });
    els.embeddingProviderSelect.addEventListener("change", toggleEmbeddingKeyField);
    
    // File inputs & Drag triggers
    els.browseBtn.addEventListener("click", () => els.fileInput.click());
    els.fileInput.addEventListener("change", (e) => handleFileSelect(e.target.files));
    
    els.uploadZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        els.uploadZone.classList.add("dragover");
    });
    els.uploadZone.addEventListener("dragleave", () => els.uploadZone.classList.remove("dragover"));
    els.uploadZone.addEventListener("drop", (e) => {
        e.preventDefault();
        els.uploadZone.classList.remove("dragover");
        handleFileSelect(e.dataTransfer.files);
    });
    
    // Chat clear & trigger suggestion cards
    els.clearChatBtn.addEventListener("click", clearChatLogs);
    els.clearStoreBtn.addEventListener("click", clearSystemStore);
    
    // Suggestion prompt card click triggers
    document.querySelectorAll(".suggestion-card").forEach(card => {
        card.addEventListener("click", () => {
            const prompt = card.getAttribute("data-prompt");
            els.chatInput.value = prompt;
            els.chatInput.focus();
            autoResizeTextarea();
        });
    });
    
    // Auto-resize chat input text box
    els.chatInput.addEventListener("input", autoResizeTextarea);
    els.chatInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            els.chatForm.requestSubmit();
        }
    });
    
    // Chat submit query
    els.chatForm.addEventListener("submit", (e) => {
        e.preventDefault();
        submitQuery();
    });
}

function openSettingsDrawer() {
    els.settingsDrawer.classList.add("open");
    els.overlay.classList.add("active");
}
function closeSettingsDrawer() {
    els.settingsDrawer.classList.remove("open");
    els.overlay.classList.remove("active");
}

function autoResizeTextarea() {
    els.chatInput.style.height = "auto";
    els.chatInput.style.height = (els.chatInput.scrollHeight - 4) + "px";
}

// Format byte file sizes cleanly
function formatBytes(bytes) {
    if (bytes === 0) return "0 Bytes";
    const k = 1024;
    const sizes = ["Bytes", "KB", "MB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + " " + sizes[i];
}

// UI notification helper
function showNotification(text, type = "info") {
    els.statusMessage.textContent = text;
    els.statusMessage.className = "status-text";
    
    const dot = els.statusMessage.previousElementSibling;
    dot.className = "status-dot";
    if (type === "success") {
        dot.classList.add("online");
    } else if (type === "working") {
        dot.classList.add("working");
    } else {
        dot.classList.add("online");
    }
}

// Fetch documents from FastAPI
async function fetchDocuments() {
    try {
        const response = await fetch("/api/documents");
        if (!response.ok) throw new Error("Failed to load knowledge base");
        
        state.documents = await response.json();
        renderDocuments();
    } catch (err) {
        showNotification(`Error: ${err.message}`, "error");
    }
}

// Render document items in sidebar list
function renderDocuments() {
    els.documentBadge.textContent = `${state.documents.length} docs`;
    
    const badgeText = state.documents.length === 1 ? "1 document loaded" : `${state.documents.length} documents loaded`;
    els.activeDocsCount.querySelector("span").textContent = state.documents.length > 0 ? badgeText : "No context files";
    if (state.documents.length > 0) {
        els.activeDocsCount.classList.add("active");
    } else {
        els.activeDocsCount.classList.remove("active");
    }
    
    els.documentList.innerHTML = "";
    if (state.documents.length === 0) {
        els.documentList.innerHTML = `
            <div class="empty-state">
                <p>No documents uploaded yet.</p>
            </div>`;
        return;
    }
    
    state.documents.forEach(doc => {
        const item = document.createElement("div");
        item.className = "document-item";
        
        const ext = doc.file_type.toUpperCase();
        
        // Define clean icons based on file type
        let iconSvg = `<svg viewBox="0 0 24 24" width="16" height="16" stroke="currentColor" stroke-width="2" fill="none"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline></svg>`;
        if (ext === "PDF") {
            iconSvg = `<svg viewBox="0 0 24 24" width="16" height="16" stroke="#ef4444" stroke-width="2" fill="none"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line><polyline points="10 9 9 9 8 9"></polyline></svg>`;
        } else if (ext === "DOCX") {
            iconSvg = `<svg viewBox="0 0 24 24" width="16" height="16" stroke="#3b82f6" stroke-width="2" fill="none"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="12" y1="17" x2="8" y2="17"></line></svg>`;
        }
        
        item.innerHTML = `
            <div class="doc-info">
                <div class="doc-icon">${iconSvg}</div>
                <div class="doc-meta">
                    <div class="doc-name" title="${doc.filename}">${doc.filename}</div>
                    <div class="doc-details">
                        <span>${formatBytes(doc.file_size)}</span>
                        <span>•</span>
                        <span>${doc.chunk_count} chunks</span>
                    </div>
                </div>
            </div>
            <button class="icon-btn delete-doc-btn" data-id="${doc.id}" aria-label="Delete document">
                <svg viewBox="0 0 24 24" width="14" height="14" stroke="currentColor" stroke-width="2" fill="none"><polyline points="3 6 5 6 21 6"></polyline><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path></svg>
            </button>
        `;
        
        // Delete button listener
        item.querySelector(".delete-doc-btn").addEventListener("click", async (e) => {
            e.stopPropagation();
            const id = e.currentTarget.getAttribute("data-id");
            await deleteDocument(id);
        });
        
        els.documentList.appendChild(item);
    });
}

// Trigger parsing and upload of files
async function handleFileSelect(files) {
    if (!files || files.length === 0) return;
    
    showNotification("Processing upload...", "working");
    els.uploadProgress.style.display = "block";
    
    for (let i = 0; i < files.length; i++) {
        const file = files[i];
        
        // Progress UI init
        els.progressFileName.textContent = file.name;
        updateProgressUI(10, "stageRead");
        
        const formData = new FormData();
        formData.append("file", file);
        formData.append("chunkSize", state.config.chunkSize);
        formData.append("chunkOverlap", state.config.chunkOverlap);
        formData.append("embeddingProvider", state.config.embeddingProvider);
        formData.append("embeddingApiKey", state.config.embeddingApiKey || state.config.apiKey);
        
        try {
            updateProgressUI(40, "stageChunk");
            
            // Artificial delay to let user see beautiful animation steps
            await new Promise(r => setTimeout(r, 600));
            updateProgressUI(70, "stageEmbed");
            
            const response = await fetch("/api/upload", {
                method: "POST",
                body: formData
            });
            
            if (!response.ok) {
                const errData = await response.json();
                throw new Error(errData.detail || "Upload error");
            }
            
            updateProgressUI(100, "finished");
            await new Promise(r => setTimeout(r, 400));
            
            const result = await response.json();
            showNotification(`Successfully indexed ${file.name}!`, "success");
            
        } catch (err) {
            showNotification(`Upload failed: ${err.message}`, "error");
            console.error(err);
            break;
        }
    }
    
    // Cleanup progress frame & refresh
    setTimeout(() => {
        els.uploadProgress.style.display = "none";
        resetProgressStages();
    }, 1500);
    
    fetchDocuments();
}

function updateProgressUI(pct, activeStage) {
    els.progressBarFill.style.width = `${pct}%`;
    els.progressPercent.textContent = `${pct}%`;
    
    document.querySelectorAll(".progress-stages .stage").forEach(s => s.classList.remove("active", "done"));
    
    if (activeStage === "stageRead") {
        els.stageRead.classList.add("active");
    } else if (activeStage === "stageChunk") {
        els.stageRead.classList.add("done");
        els.stageChunk.classList.add("active");
    } else if (activeStage === "stageEmbed") {
        els.stageRead.classList.add("done");
        els.stageChunk.classList.add("done");
        els.stageEmbed.classList.add("active");
    } else if (activeStage === "finished") {
        els.stageRead.classList.add("done");
        els.stageChunk.classList.add("done");
        els.stageEmbed.classList.add("done");
    }
}

function resetProgressStages() {
    els.progressBarFill.style.width = "0%";
    els.progressPercent.textContent = "0%";
    document.querySelectorAll(".progress-stages .stage").forEach(s => s.classList.remove("active", "done"));
}

// Delete single document
async function deleteDocument(id) {
    showNotification("Deleting document...", "working");
    try {
        const response = await fetch(`/api/documents/${id}`, {
            method: "DELETE"
        });
        if (!response.ok) throw new Error("Delete failed");
        
        showNotification("Document removed successfully.", "success");
        fetchDocuments();
    } catch (err) {
        showNotification(`Error: ${err.message}`, "error");
    }
}

// Purge vector DB and uploads
async function clearSystemStore() {
    if (!confirm("Are you sure you want to completely purge the document store database? This will clear all parsed content, index embeddings, and local files.")) return;
    
    showNotification("Purging database...", "working");
    try {
        const response = await fetch("/api/clear", { method: "POST" });
        if (!response.ok) throw new Error("Failed to clear database");
        
        showNotification("Database wiped successfully.", "success");
        fetchDocuments();
        clearChatLogs();
    } catch (err) {
        showNotification(`Error: ${err.message}`, "error");
    }
}

// Clear UI Chat logs
function clearChatLogs() {
    state.chatHistory = [];
    els.chatMessages.innerHTML = "";
    els.welcomeScreen.style.display = "block";
    els.chatMessages.appendChild(els.welcomeScreen);
    showNotification("Conversation history cleared.", "success");
}

// Send chat query & read server streaming events
async function submitQuery() {
    const text = els.chatInput.value.trim();
    if (!text) return;
    
    // Close welcome layout
    if (els.welcomeScreen.parentNode) {
        els.welcomeScreen.style.display = "none";
    }
    
    // 1. Render User Message
    appendMessage("user", text);
    
    // Prepare input field resets
    els.chatInput.value = "";
    autoResizeTextarea();
    setFormDisabledState(true);
    
    // Show typing active progress
    showNotification("Retrieving documents and thinking...", "working");
    els.typingIndicator.style.display = "flex";
    els.chatMessages.scrollTop = els.chatMessages.scrollHeight;
    
    // Create new Bot Message Bubble in UI
    const botMsgEl = appendMessage("assistant", "");
    const botTextEl = botMsgEl.querySelector(".message-text-content");
    const sourcesContainerEl = botMsgEl.querySelector(".message-sources-wrapper");
    
    let streamTextBuffer = "";
    
    try {
        // Post request with stream reader
        const chatReq = {
            query: text,
            history: state.chatHistory,
            provider: state.config.provider,
            apiKey: state.config.apiKey,
            model: state.config.model,
            temperature: state.config.temperature,
            embeddingProvider: state.config.embeddingProvider,
            embeddingApiKey: state.config.embeddingApiKey || state.config.apiKey,
            topK: state.config.topK
        };
        
        const response = await fetch("/api/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(chatReq)
        });
        
        if (!response.ok) {
            const errData = await response.json();
            throw new Error(errData.detail || "Server error");
        }
        
        // Hide typing indicator once the first stream chunk arrives
        els.typingIndicator.style.display = "none";
        
        // Stream parser
        const reader = response.body.getReader();
        const decoder = new TextDecoder("utf-8");
        let buffer = "";
        
        while (true) {
            const { value, done } = await reader.read();
            if (done) break;
            
            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split("\n\n");
            
            // Keep the last partial line in the buffer
            buffer = lines.pop();
            
            for (const line of lines) {
                if (!line.trim() || !line.startsWith("data: ")) continue;
                
                try {
                    const jsonStr = line.substring(6).trim();
                    const payload = JSON.parse(jsonStr);
                    
                    if (payload.type === "sources") {
                        // Render document source contexts
                        renderSourceReferences(sourcesContainerEl, payload.sources);
                    } else if (payload.type === "text") {
                        streamTextBuffer += payload.content;
                        // Render parsed markdown inside the chatbubble
                        botTextEl.innerHTML = marked.parse(streamTextBuffer);
                        els.chatMessages.scrollTop = els.chatMessages.scrollHeight;
                    } else if (payload.type === "error") {
                        botTextEl.innerHTML += `<div class="error-alert" style="color: var(--accent-rose); font-weight:500; margin-top: 8px;">[Error: ${payload.message}]</div>`;
                    }
                } catch (e) {
                    // Ignore malformed JSON lines
                }
            }
        }
        
        // SSE connection completed
        showNotification("Response completed.", "success");
        
        // Save to active chatbot context history
        state.chatHistory.push({ role: "user", content: text });
        state.chatHistory.push({ role: "assistant", content: streamTextBuffer });
        
    } catch (err) {
        showNotification(`Query failed: ${err.message}`, "error");
        els.typingIndicator.style.display = "none";
        botTextEl.innerHTML = `<span style="color: var(--accent-rose);">Failed to load response: ${err.message}</span>`;
    } finally {
        setFormDisabledState(false);
        els.chatMessages.scrollTop = els.chatMessages.scrollHeight;
    }
}

// UI input lock toggle
function setFormDisabledState(disabled) {
    els.chatInput.disabled = disabled;
    els.sendBtn.disabled = disabled;
    if (disabled) {
        els.sendBtn.style.opacity = 0.5;
    } else {
        els.sendBtn.style.opacity = 1;
        els.chatInput.focus();
    }
}

// Append Chat Message elements
function appendMessage(role, text) {
    const msg = document.createElement("div");
    msg.className = `chat-message ${role}`;
    
    if (role === "user") {
        msg.innerHTML = `
            <div class="message-bubble">
                <div class="message-text-content">${text}</div>
            </div>
        `;
    } else {
        msg.innerHTML = `
            <div class="message-bubble">
                <div class="message-text-content">${marked.parse(text || "...")}</div>
            </div>
            <div class="message-sources-wrapper" style="display: none; width: 85%;"></div>
        `;
    }
    
    els.chatMessages.appendChild(msg);
    els.chatMessages.scrollTop = els.chatMessages.scrollHeight;
    return msg;
}

// Render Context Sources drawer below bot responses
function renderSourceReferences(containerEl, sources) {
    if (!sources || sources.length === 0) {
        containerEl.style.display = "none";
        return;
    }
    
    containerEl.style.display = "block";
    
    const details = document.createElement("details");
    details.className = "message-sources";
    
    // Group duplicates or list cleanly
    const uniqueDocs = {};
    sources.forEach(src => {
        if (!uniqueDocs[src.filename]) {
            uniqueDocs[src.filename] = [];
        }
        uniqueDocs[src.filename].push(src.chunk_index);
    });
    
    const sourcesSummaryText = Object.entries(uniqueDocs)
        .map(([name, chunks]) => `${name} (Chunks: ${chunks.join(",")})`)
        .join(", ");
        
    const summary = document.createElement("summary");
    summary.className = "sources-title";
    summary.innerHTML = `
        <svg viewBox="0 0 24 24" width="12" height="12" stroke="currentColor" stroke-width="2.5" fill="none"><polyline points="9 18 15 12 9 6"></polyline></svg>
        <span>Retrieved Context Sources: ${Object.keys(uniqueDocs).length} files</span>
    `;
    
    summary.addEventListener("click", () => {
        summary.classList.toggle("open");
    });
    
    const list = document.createElement("div");
    list.className = "sources-list";
    
    sources.forEach(src => {
        const item = document.createElement("div");
        item.className = "source-tag";
        
        // Convert score metric to percentage or round off
        const scorePct = src.score > 0 ? `Similarity: ${(src.score * 100).toFixed(0)}%` : "Keyword Match";
        
        item.innerHTML = `
            <span class="source-tag-left">${src.filename} (chunk #${src.chunk_index})</span>
            <span class="source-tag-right">${scorePct}</span>
        `;
        list.appendChild(item);
    });
    
    details.appendChild(summary);
    details.appendChild(list);
    
    containerEl.innerHTML = "";
    containerEl.appendChild(details);
}
