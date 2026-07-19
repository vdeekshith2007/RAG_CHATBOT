/**
 * RAG Chatbot — Frontend Logic
 * Handles: file upload, query submission, chat rendering, UI state
 */

// ─── State ───────────────────────────────────────────────────
const state = {
  documentCount: 0,
  loadedDocs: [],       // { name, chunks, type }
  isQuerying: false,
  isUploading: false,
};

// ─── DOM References ──────────────────────────────────────────
const $ = (id) => document.getElementById(id);

const uploadArea = $("upload-area");
const fileInput = $("file-input");
const uploadStatus = $("upload-status");
const progressBar = $("progress-bar");
const progressLabel = $("progress-label");
const docsSection = $("docs-section");
const docList = $("doc-list");
const resetBtn = $("reset-btn");
const messagesContainer = $("messages-container");
const messagesList = $("messages-list");
const welcomeScreen = $("welcome-screen");
const queryInput = $("query-input");
const sendBtn = $("send-btn");
const statusDot = $("status-dot");
const statusText = $("status-text");
const docCountBadge = $("doc-count-badge");
const docCountText = $("doc-count-text");
const newChatBtn = $("new-chat-btn");
const sidebarToggle = $("sidebar-toggle");
const sidebar = $("sidebar");
const toast = $("toast");

// ─── Toast ───────────────────────────────────────────────────
let toastTimer = null;
function showToast(message, type = "info") {
  toast.textContent = message;
  toast.className = `toast show ${type}`;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => {
    toast.classList.remove("show");
  }, 3500);
}

// ─── Status ──────────────────────────────────────────────────
function setStatus(status, text) {
  statusDot.className = `status-dot ${status}`;
  statusText.textContent = text;
}

async function checkStatus() {
  try {
    const res = await fetch("/api/status");
    const data = await res.json();
    if (data.has_documents) {
      setStatus("ready", `Ready · ${data.documents_loaded} chunks`);
    } else {
      setStatus("ready", "Ready · No documents");
    }
    updateDocCountBadge(data.documents_loaded);
  } catch {
    setStatus("error", "Server offline");
  }
}

// ─── Doc Count Badge ─────────────────────────────────────────
function updateDocCountBadge(count) {
  state.documentCount = count;
  if (count > 0) {
    docCountBadge.style.display = "flex";
    docCountText.textContent = `${count} chunk${count !== 1 ? "s" : ""}`;
  } else {
    docCountBadge.style.display = "none";
  }
}

// ─── Sidebar Toggle ──────────────────────────────────────────
sidebarToggle.addEventListener("click", () => {
  sidebar.classList.toggle("collapsed");
});

// ─── New Chat ────────────────────────────────────────────────
newChatBtn.addEventListener("click", async () => {
  const confirmReset = confirm("Start a new chat? This will clear the current document and conversation.");
  if (!confirmReset) return;

  newChatBtn.disabled = true;

  try {
    const res = await fetch("/api/reset", { method: "DELETE" });
    const data = await res.json();

    if (!res.ok) {
      throw new Error(data.error || "Reset failed");
    }

    state.loadedDocs = [];
    state.documentCount = 0;
    messagesList.innerHTML = "";
    docList.innerHTML = "";
    docsSection.style.display = "none";
    updateDocCountBadge(0);
    welcomeScreen.style.display = "flex";
    queryInput.value = "";
    queryInput.disabled = false;
    queryInput.style.height = "auto";
    sendBtn.disabled = false;
    uploadStatus.style.display = "none";
    progressBar.style.width = "0%";
    progressLabel.textContent = "Processing...";
    setStatus("ready", "Ready · No documents");
    showToast("New chat started.", "success");
  } catch (err) {
    showToast(`New chat failed: ${err.message}`, "error");
  } finally {
    newChatBtn.disabled = false;
    queryInput.focus();
  }
});

// ─── File Upload ─────────────────────────────────────────────
// Drag & Drop
uploadArea.addEventListener("dragover", (e) => {
  e.preventDefault();
  uploadArea.classList.add("dragover");
});
uploadArea.addEventListener("dragleave", () => {
  uploadArea.classList.remove("dragover");
});
uploadArea.addEventListener("drop", (e) => {
  e.preventDefault();
  uploadArea.classList.remove("dragover");
  const file = e.dataTransfer.files[0];
  if (file) handleFileUpload(file);
});

fileInput.addEventListener("change", () => {
  if (fileInput.files[0]) handleFileUpload(fileInput.files[0]);
});

async function handleFileUpload(file) {
  if (state.isUploading) return;

  const ext = file.name.includes(".") ? "." + file.name.split(".").pop().toLowerCase() : "";

  state.isUploading = true;
  uploadStatus.style.display = "flex";
  progressBar.style.width = "0%";
  progressLabel.textContent = `Processing "${file.name}"...`;
  setStatus("loading", "Uploading & embedding...");

  // Animate progress
  let fakeProgress = 0;
  const fakeTimer = setInterval(() => {
    fakeProgress = Math.min(fakeProgress + Math.random() * 8, 85);
    progressBar.style.width = fakeProgress + "%";
  }, 200);

  try {
    const formData = new FormData();
    formData.append("file", file);

    const res = await fetch("/api/upload", {
      method: "POST",
      body: formData,
    });
    const data = await res.json();

    clearInterval(fakeTimer);

    if (res.ok && data.success) {
      progressBar.style.width = "100%";
      progressLabel.textContent = `✓ Loaded ${data.chunks_loaded} chunk(s)`;

      // Add to doc list
      addDocToSidebar(file.name, data.chunks_loaded, ext.replace(".", "") || "file");
      updateDocCountBadge(data.total_documents);
      setStatus("ready", `Ready · ${data.total_documents} chunks`);
      showToast(`"${file.name}" loaded with ${data.chunks_loaded} chunk(s)!`, "success");

      setTimeout(() => {
        uploadStatus.style.display = "none";
        progressBar.style.width = "0%";
      }, 2000);
    } else {
      throw new Error(data.error || "Upload failed");
    }
  } catch (err) {
    clearInterval(fakeTimer);
    progressBar.style.width = "0%";
    progressLabel.textContent = `✗ Error: ${err.message}`;
    setStatus("error", "Upload failed");
    showToast(`Upload failed: ${err.message}`, "error");
    setTimeout(() => { uploadStatus.style.display = "none"; }, 3000);
  } finally {
    state.isUploading = false;
    fileInput.value = "";
  }
}

function getDocTypeLabel(ext) {
  const labels = {
    pdf: "PDF", txt: "TXT", doc: "DOC", docx: "DOC",
    xlsx: "XLS", xls: "XLS", csv: "CSV",
  };
  return labels[ext] || ext.toUpperCase();
}

function addDocToSidebar(filename, chunks, ext) {
  const docRecord = { name: filename, chunks, type: ext };
  state.loadedDocs.push(docRecord);
  docsSection.style.display = "flex";

  const item = document.createElement("div");
  item.className = "doc-item";

  const typeClass = `doc-type-${ext}`;
  item.innerHTML = `
    <div class="doc-item-icon ${typeClass}">${getDocTypeLabel(ext)}</div>
    <div class="doc-item-info">
      <div class="doc-item-name" title="${filename}">${filename}</div>
      <div class="doc-item-chunks">${chunks} chunk${chunks !== 1 ? "s" : ""}</div>
    </div>
    <svg class="doc-check" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" width="14" height="14">
      <polyline points="20 6 9 17 4 12"/>
    </svg>
  `;
  docList.appendChild(item);
}

// ─── Reset ───────────────────────────────────────────────────
resetBtn.addEventListener("click", async () => {
  if (!confirm("Clear all loaded documents? This cannot be undone.")) return;

  try {
    const res = await fetch("/api/reset", { method: "DELETE" });
    const data = await res.json();
    if (res.ok) {
      state.loadedDocs = [];
      docList.innerHTML = "";
      docsSection.style.display = "none";
      updateDocCountBadge(0);
      setStatus("ready", "Ready · No documents");
      showToast("All documents cleared.", "success");
    } else {
      throw new Error(data.error);
    }
  } catch (err) {
    showToast(`Reset failed: ${err.message}`, "error");
  }
});

// ─── Query Input ─────────────────────────────────────────────
// Auto-resize textarea
queryInput.addEventListener("input", () => {
  queryInput.style.height = "auto";
  queryInput.style.height = Math.min(queryInput.scrollHeight, 160) + "px";
});

// Send on Enter (not Shift+Enter)
queryInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    handleQuery();
  }
});

sendBtn.addEventListener("click", handleQuery);

// Example chips
document.querySelectorAll(".example-chip").forEach((chip) => {
  chip.addEventListener("click", () => {
    queryInput.value = chip.dataset.query;
    queryInput.dispatchEvent(new Event("input"));
    queryInput.focus();
  });
});

// ─── Query Handler ───────────────────────────────────────────
async function handleQuery() {
  const query = queryInput.value.trim();
  if (!query || state.isQuerying) return;

  if (state.documentCount === 0) {
    showToast("Please upload a document first!", "error");
    return;
  }

  state.isQuerying = true;
  sendBtn.disabled = true;
  queryInput.disabled = true;

  // Hide welcome screen
  welcomeScreen.style.display = "none";

  // Append user message
  appendUserMessage(query);
  queryInput.value = "";
  queryInput.style.height = "auto";

  // Show typing indicator
  const typingId = appendTypingIndicator();

  try {
    const res = await fetch("/api/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query,
        top_k: 5,
      }),
    });
    const data = await res.json();

    removeTypingIndicator(typingId);

    if (res.ok) {
      appendBotMessage(data.answer, data.sources, data.confidence);
    } else {
      appendErrorMessage(data.error || "An error occurred");
    }
  } catch (err) {
    removeTypingIndicator(typingId);
    appendErrorMessage(`Network error: ${err.message}`);
  } finally {
    state.isQuerying = false;
    sendBtn.disabled = false;
    queryInput.disabled = false;
    queryInput.focus();
  }
}

// ─── Message Rendering ───────────────────────────────────────
function scrollToBottom() {
  setTimeout(() => {
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
  }, 50);
}

function appendUserMessage(text) {
  const div = document.createElement("div");
  div.className = "message user";
  div.innerHTML = `
    <div class="msg-avatar">U</div>
    <div class="msg-bubble">
      <div class="msg-content">${escapeHtml(text)}</div>
    </div>
  `;
  messagesList.appendChild(div);
  scrollToBottom();
}

function appendBotMessage(answer, sources, confidence) {
  const div = document.createElement("div");
  div.className = "message bot";

  div.innerHTML = `
    <div class="msg-avatar">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="16" height="16">
        <path d="M12 2L2 7l10 5 10-5-10-5z"/>
        <path d="M2 17l10 5 10-5"/>
        <path d="M2 12l10 5 10-5"/>
      </svg>
    </div>
    <div class="msg-bubble">
      <div class="msg-content">${formatAnswer(answer)}</div>
    </div>
  `;

  messagesList.appendChild(div);
  scrollToBottom();
}

function appendErrorMessage(text) {
  const div = document.createElement("div");
  div.className = "message bot";
  div.innerHTML = `
    <div class="msg-avatar">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="16" height="16">
        <circle cx="12" cy="12" r="10"/>
        <line x1="15" y1="9" x2="9" y2="15"/>
        <line x1="9" y1="9" x2="15" y2="15"/>
      </svg>
    </div>
    <div class="msg-bubble">
      <div class="msg-content msg-error">⚠ ${escapeHtml(text)}</div>
    </div>
  `;
  messagesList.appendChild(div);
  scrollToBottom();
}

let typingCounter = 0;
function appendTypingIndicator() {
  const id = "typing-" + (++typingCounter);
  const div = document.createElement("div");
  div.className = "message bot typing-indicator";
  div.id = id;
  div.innerHTML = `
    <div class="msg-avatar">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="16" height="16">
        <path d="M12 2L2 7l10 5 10-5-10-5z"/>
        <path d="M2 17l10 5 10-5"/>
        <path d="M2 12l10 5 10-5"/>
      </svg>
    </div>
    <div class="msg-bubble">
      <div class="msg-content">
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
      </div>
    </div>
  `;
  messagesList.appendChild(div);
  scrollToBottom();
  return id;
}

function removeTypingIndicator(id) {
  const el = document.getElementById(id);
  if (el) el.remove();
}

// ─── Helpers ─────────────────────────────────────────────────
function escapeHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function formatAnswer(text) {
  // Basic markdown-like formatting
  return escapeHtml(text)
    .replace(/\n\n/g, "</p><p>")
    .replace(/\n/g, "<br>")
    .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
    .replace(/\*(.*?)\*/g, "<em>$1</em>")
    .replace(/`(.*?)`/g, "<code>$1</code>");
}

function getBasename(path) {
  return path.replace(/\\/g, "/").split("/").pop();
}

function getConfidenceClass(score) {
  if (score >= 0.3) return "confidence-high";
  if (score >= 0.0) return "confidence-medium";
  return "confidence-low";
}

function getConfidenceLabel(score) {
  if (score >= 0.3) return "High";
  if (score >= 0.0) return "Medium";
  return "Low";
}

// ─── Init ────────────────────────────────────────────────────
async function init() {
  setStatus("loading", "Connecting...");
  await checkStatus();
  // Poll status every 30s
  setInterval(checkStatus, 30000);
}

init();
