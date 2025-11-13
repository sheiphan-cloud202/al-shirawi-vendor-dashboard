const $ = (sel) => document.querySelector(sel);

// Global session ID to track the current workflow session
let currentSessionId = null;

// Log version for debugging
console.log('📦 App.js loaded - Version 2.0 (Session ID fix applied)');

// Create a new session
async function createSession() {
  try {
    console.log('🔄 Creating new session...');
    const res = await fetch("/create-session", { method: "POST" });
    
    if (!res.ok) {
      throw new Error(`HTTP ${res.status}: ${res.statusText}`);
    }
    
    const data = await res.json();
    
    if (data.session_id) {
      currentSessionId = data.session_id;
      updateSessionDisplay();
      console.log(`✅ Session created successfully: ${currentSessionId}`);
      console.log(`📁 Output directory: ${data.output_directory}`);
      
      // Show success notification
      showNotification(`Session created: ${currentSessionId}`, 'success');
      return true;
    } else {
      throw new Error('No session_id in response');
    }
  } catch (err) {
    console.error('❌ Failed to create session:', err);
    showNotification('Failed to create session', 'error');
    
    // Update UI to show error
    const sessionBadge = $("#session-badge");
    if (sessionBadge) {
      sessionBadge.style.display = "flex";
      sessionBadge.innerHTML = `
        <span class="session-icon">❌</span>
        <span class="session-text">Failed to create session</span>
        <button class="session-new-btn" onclick="createSession()" title="Retry">🔄 Retry</button>
      `;
    }
    return false;
  }
}

// Update session display in UI
function updateSessionDisplay() {
  const sessionDisplay = $("#current-session-display");
  const sessionBadge = $("#session-badge");
  
  if (currentSessionId) {
    if (sessionDisplay) {
      sessionDisplay.textContent = currentSessionId;
      sessionDisplay.style.display = "inline";
    }
    if (sessionBadge) {
      sessionBadge.style.display = "flex";
      sessionBadge.innerHTML = `
        <span class="session-icon">🔗</span>
        <span class="session-text">Session: <strong>${currentSessionId}</strong></span>
        <button class="session-new-btn" onclick="createSession()" title="Create new session">🔄</button>
      `;
    }
  } else {
    if (sessionDisplay) {
      sessionDisplay.textContent = "No active session";
      sessionDisplay.style.display = "inline";
    }
    if (sessionBadge) {
      sessionBadge.style.display = "flex";
      sessionBadge.innerHTML = `
        <span class="session-icon">⚠️</span>
        <span class="session-text">No active session</span>
        <button class="session-new-btn" onclick="createSession()" title="Create new session">➕ Create Session</button>
      `;
    }
  }
}

// Show notification
function showNotification(message, type = 'info') {
  const notification = document.createElement('div');
  notification.className = `notification notification-${type}`;
  notification.textContent = message;
  notification.style.cssText = `
    position: fixed;
    top: 20px;
    right: 20px;
    background: ${type === 'success' ? '#10b981' : type === 'error' ? '#ef4444' : '#3b82f6'};
    color: white;
    padding: 16px 24px;
    border-radius: 8px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    z-index: 10000;
    animation: slideIn 0.3s ease;
  `;
  
  document.body.appendChild(notification);
  
  setTimeout(() => {
    notification.style.animation = 'slideOut 0.3s ease';
    setTimeout(() => notification.remove(), 300);
  }, 3000);
}

async function uploadEnquiry(e) {
  e.preventDefault();
  
  // Ensure we have a session
  if (!currentSessionId) {
    const out = $("#enquiry-result");
    out.style.display = "block";
    out.textContent = "⚠️ No active session! Creating one now...";
    await createSession();
    if (!currentSessionId) {
      out.textContent = "❌ Failed to create session. Please try again.";
      return;
    }
  }
  
  const fileInput = $("#enquiry-file");
  const out = $("#enquiry-result");
  out.style.display = "block";
  out.textContent = "Uploading...";
  const fd = new FormData();
  if (!fileInput.files || fileInput.files.length === 0) {
    out.textContent = "Please choose an Excel file.";
    return;
  }
  fd.append("file", fileInput.files[0]);
  
  // Only append session_id if it's a valid string
  if (currentSessionId && typeof currentSessionId === 'string' && currentSessionId !== 'null' && currentSessionId !== 'undefined') {
    fd.append("session_id", currentSessionId);
    console.log(`📤 Uploading enquiry to session: ${currentSessionId}`);
  } else {
    console.error('⚠️ No valid session_id available for enquiry upload!');
    out.textContent = "❌ Invalid session. Please refresh the page and try again.";
    return;
  }
  
  try {
    const res = await fetch("/upload/enquiry", { method: "POST", body: fd });
    const data = await res.json();
    
    out.textContent = JSON.stringify(data, null, 2);
    
    if (res.ok) {
      showNotification('BOQ uploaded successfully!', 'success');
      out.textContent += `\n\n✅ Uploaded to session: ${currentSessionId}`;
    }
  } catch (err) {
    out.textContent = String(err);
    showNotification('Upload failed', 'error');
  }
}

async function uploadVendor(e) {
  e.preventDefault();
  
  // Ensure we have a session
  if (!currentSessionId) {
    const out = $("#vendor-result");
    out.style.display = "block";
    out.textContent = "⚠️ No active session! Creating one now...";
    await createSession();
    if (!currentSessionId) {
      out.textContent = "❌ Failed to create session. Please try again.";
      return;
    }
  }
  
  const responseNum = $("#vendor-response").value;
  const filesInput = $("#vendor-files");
  const out = $("#vendor-result");
  out.style.display = "block";
  out.textContent = "Uploading...";
  const fd = new FormData();
  fd.append("response_number", responseNum);
  
  // Only append session_id if it's a valid string
  if (currentSessionId && typeof currentSessionId === 'string' && currentSessionId !== 'null' && currentSessionId !== 'undefined') {
    fd.append("session_id", currentSessionId);
    console.log(`📤 Uploading vendor ${responseNum} to session: ${currentSessionId}`);
  } else {
    console.error('⚠️ No valid session_id available for vendor upload!');
    out.textContent = "❌ Invalid session. Please refresh the page and try again.";
    return;
  }
  
  const files = filesInput.files;
  if (!files || files.length === 0) {
    out.textContent = "Please select one or more PDF files.";
    return;
  }
  for (const f of files) fd.append("files", f);
  
  // Debug: Log FormData contents
  console.log('📋 FormData being sent:');
  for (let pair of fd.entries()) {
    if (pair[1] instanceof File) {
      console.log(`  ${pair[0]}: [File: ${pair[1].name}]`);
    } else {
      console.log(`  ${pair[0]}: ${pair[1]}`);
    }
  }
  
  try {
    const res = await fetch("/upload/vendor", { method: "POST", body: fd });
    const data = await res.json();
    out.textContent = JSON.stringify(data, null, 2);
    
    if (res.ok) {
      showNotification(`Vendor ${responseNum} uploaded successfully!`, 'success');
      out.textContent += `\n\n✅ Uploaded to session: ${currentSessionId}`;
    }
  } catch (err) {
    out.textContent = String(err);
    showNotification('Upload failed', 'error');
  }
}

async function runWorkflow(e) {
  e.preventDefault();
  
  // Ensure we have a session
  if (!currentSessionId) {
    const out = $("#run-result");
    out.style.display = "block";
    out.textContent = "❌ No active session! Please upload files first.";
    showNotification('No active session. Please upload files first.', 'error');
    return;
  }
  
  const skipTextract = $("#skip-textract").checked;
  const skipBoq = $("#skip-boq").checked;
  const out = $("#run-result");
  out.style.display = "block";
  out.textContent = `Running workflow for session: ${currentSessionId}\n(this can take several minutes)...`;
  
  const fd = new FormData();
  if (skipTextract) fd.append("skip_textract", "true");
  if (skipBoq) fd.append("skip_boq", "true");
  
  // Only append session_id if it's a valid string
  if (currentSessionId && typeof currentSessionId === 'string' && currentSessionId !== 'null' && currentSessionId !== 'undefined') {
    fd.append("session_id", currentSessionId);
    console.log(`🚀 Running workflow for session: ${currentSessionId}`);
  } else {
    console.error('⚠️ No valid session_id available for workflow!');
    out.textContent = "❌ Invalid session. Please refresh the page and try again.";
    return;
  }
  
  try {
    const res = await fetch("/run-workflow", { method: "POST", body: fd });
    const data = await res.json();
    out.textContent = JSON.stringify(data, null, 2);
    
    // Show success message with links
    if (res.ok && data.code === 0) {
      showNotification('Workflow completed successfully!', 'success');
      out.textContent += `\n\n✅ Workflow completed for session: ${currentSessionId}`;
      out.textContent += `\n📁 Output directory: ${data.output_directory}`;
      
      const successDiv = $("#workflow-success");
      if (successDiv) {
        successDiv.style.display = "block";
      }
    } else {
      showNotification('Workflow failed', 'error');
    }
  } catch (err) {
    out.textContent = String(err);
    showNotification('Workflow error', 'error');
  }
}

// Make createSession available globally for onclick handlers
window.createSession = createSession;

window.addEventListener("DOMContentLoaded", async () => {
  // Add CSS animations for notifications
  const style = document.createElement('style');
  style.textContent = `
    @keyframes slideIn {
      from {
        transform: translateX(400px);
        opacity: 0;
      }
      to {
        transform: translateX(0);
        opacity: 1;
      }
    }
    @keyframes slideOut {
      from {
        transform: translateX(0);
        opacity: 1;
      }
      to {
        transform: translateX(400px);
        opacity: 0;
      }
    }
    .session-badge {
      display: flex;
      align-items: center;
      gap: 8px;
      padding: 8px 16px;
      background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
      color: white;
      border-radius: 8px;
      font-size: 14px;
      box-shadow: 0 2px 4px rgba(0,0,0,0.1);
      margin-bottom: 16px;
    }
    .session-new-btn {
      background: rgba(255,255,255,0.2);
      border: none;
      color: white;
      padding: 4px 12px;
      border-radius: 4px;
      cursor: pointer;
      font-size: 12px;
      transition: all 0.2s;
    }
    .session-new-btn:hover {
      background: rgba(255,255,255,0.3);
    }
  `;
  document.head.appendChild(style);
  
  // Create session automatically on page load - AWAIT to ensure it completes
  await createSession();
  
  $("#form-enquiry").addEventListener("submit", uploadEnquiry);
  $("#form-vendor").addEventListener("submit", uploadVendor);
  $("#form-run").addEventListener("submit", runWorkflow);
  
  // File display handlers
  const enquiryFile = $("#enquiry-file");
  const enquiryDisplay = $("#enquiry-display");
  if (enquiryFile && enquiryDisplay) {
    enquiryFile.addEventListener("change", (e) => {
      if (e.target.files && e.target.files.length > 0) {
        enquiryDisplay.classList.add("has-file");
        enquiryDisplay.innerHTML = `
          <div class="file-icon">📊</div>
          <div class="file-text">Selected file:</div>
          <div class="file-name">${e.target.files[0].name}</div>
        `;
      }
    });
  }
  
  const vendorFiles = $("#vendor-files");
  const vendorDisplay = $("#vendor-display");
  if (vendorFiles && vendorDisplay) {
    vendorFiles.addEventListener("change", (e) => {
      if (e.target.files && e.target.files.length > 0) {
        vendorDisplay.classList.add("has-file");
        const fileNames = Array.from(e.target.files).map(f => f.name).join(", ");
        vendorDisplay.innerHTML = `
          <div class="file-icon">📄</div>
          <div class="file-text">Selected ${e.target.files.length} file(s):</div>
          <div class="file-name">${fileNames}</div>
        `;
      }
    });
  }
});


