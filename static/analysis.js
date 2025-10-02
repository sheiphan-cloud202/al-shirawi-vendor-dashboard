const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

let allVendors = [];
let selectedVendors = new Set();
let vendorDetails = {};

// Load vendors on page load
window.addEventListener("DOMContentLoaded", async () => {
  await loadVendors();
});

async function loadVendors() {
  try {
    const res = await fetch("/api/vendors");
    const data = await res.json();
    
    $("#loading").style.display = "none";
    
    if (!data.vendors || data.vendors.length === 0) {
      $("#no-data").style.display = "block";
      return;
    }
    
    allVendors = data.vendors;
    renderVendorList();
    $("#vendor-section").style.display = "block";
    
    // Load details for all vendors in parallel
    await Promise.all(allVendors.map(v => loadVendorDetails(v.id)));
    renderVendorCards();
    
  } catch (err) {
    $("#loading").textContent = `Error loading vendors: ${err.message}`;
  }
}

function renderVendorList() {
  const container = $("#vendor-list");
  container.innerHTML = "";
  
  allVendors.forEach(vendor => {
    const card = document.createElement("div");
    card.className = "vendor-card";
    card.innerHTML = `
      <input type="checkbox" id="vendor-${vendor.id}" value="${vendor.id}" />
      <label for="vendor-${vendor.id}">
        <strong>${vendor.name}</strong>
        <span class="vendor-id">Response #${vendor.id}</span>
      </label>
    `;
    
    const checkbox = card.querySelector("input");
    checkbox.addEventListener("change", (e) => {
      if (e.target.checked) {
        selectedVendors.add(vendor.id);
      } else {
        selectedVendors.delete(vendor.id);
      }
      updateAnalyzeButton();
    });
    
    container.appendChild(card);
  });
}

function updateAnalyzeButton() {
  const btn = $("#analyze-btn");
  btn.disabled = selectedVendors.size === 0;
  btn.textContent = selectedVendors.size === 0
    ? "Select at least one vendor"
    : `Analyze ${selectedVendors.size} Selected Vendor${selectedVendors.size > 1 ? "s" : ""}`;
}

async function loadVendorDetails(vendorId) {
  try {
    const res = await fetch(`/api/vendors/${vendorId}/comparison`);
    const data = await res.json();
    vendorDetails[vendorId] = data;
  } catch (err) {
    console.error(`Failed to load details for vendor ${vendorId}:`, err);
  }
}

function renderVendorCards() {
  const container = $("#vendor-details");
  container.innerHTML = "";
  
  allVendors.forEach(vendor => {
    const details = vendorDetails[vendor.id];
    if (!details) return;
    
    const summary = details.summary;
    const card = document.createElement("div");
    card.className = "detail-card";
    card.innerHTML = `
      <h3>${vendor.name}</h3>
      <div class="metrics">
        <div class="metric">
          <span class="metric-label">Match Rate</span>
          <span class="metric-value">${summary.match_rate}%</span>
        </div>
        <div class="metric">
          <span class="metric-label">Total Price</span>
          <span class="metric-value">$${summary.total_price.toLocaleString()}</span>
        </div>
        <div class="metric">
          <span class="metric-label">Items Matched</span>
          <span class="metric-value">${summary.matched_items}/${summary.total_items}</span>
        </div>
      </div>
      <div class="match-breakdown">
        <span class="badge excellent">✅ ${summary.excellent_matches} Excellent</span>
        <span class="badge good">✓ ${summary.good_matches} Good</span>
      </div>
      ${renderIssues(summary.issues)}
      <button class="btn-small" onclick="viewDetails('${vendor.id}')">View Full Details</button>
    `;
    
    container.appendChild(card);
  });
  
  $("#details-section").style.display = "block";
}

function renderIssues(issues) {
  if (!issues || Object.keys(issues).length === 0) {
    return '<p class="no-issues">✅ No major issues detected</p>';
  }
  
  let html = '<div class="issues"><strong>Issues:</strong><ul>';
  for (const [type, count] of Object.entries(issues)) {
    html += `<li>${type}: ${count}</li>`;
  }
  html += '</ul></div>';
  return html;
}

function viewDetails(vendorId) {
  const details = vendorDetails[vendorId];
  if (!details) return;
  
  // Create modal or navigate to detail view
  alert(`Full details for vendor ${vendorId}\n\nItems: ${details.items.length}\n\nSee browser console for raw data.`);
  console.log("Vendor Details:", details);
}

// Analyze button handler
$("#analyze-btn").addEventListener("click", async () => {
  if (selectedVendors.size === 0) return;
  
  $("#recommendations-section").style.display = "block";
  $("#ai-loading").style.display = "block";
  $("#ai-results").innerHTML = "";
  
  // Scroll to results
  $("#recommendations-section").scrollIntoView({ behavior: "smooth" });
  
  try {
    const formData = new FormData();
    selectedVendors.forEach(id => formData.append("vendor_ids", id));
    
    const res = await fetch("/api/analyze", {
      method: "POST",
      body: formData,
    });
    
    if (!res.ok) {
      throw new Error(`API error: ${res.status}`);
    }
    
    const data = await res.json();
    
    $("#ai-loading").style.display = "none";
    renderAnalysisResults(data);
    
  } catch (err) {
    $("#ai-loading").style.display = "none";
    $("#ai-results").innerHTML = `<div class="error">Error: ${err.message}</div>`;
  }
});

function renderAnalysisResults(data) {
  const container = $("#ai-results");
  
  // Render vendor comparison table
  let html = '<div class="comparison-table">';
  html += '<h3>Vendor Comparison Summary</h3>';
  html += '<table><thead><tr>';
  html += '<th>Vendor</th><th>Match Rate</th><th>Total Price</th><th>Issues</th>';
  html += '</tr></thead><tbody>';
  
  data.vendors.forEach(v => {
    const issueCount = Object.values(v.issues).reduce((a, b) => a + b, 0);
    html += `<tr>
      <td><strong>${v.vendor_name}</strong></td>
      <td>${v.match_rate}%</td>
      <td>$${v.total_price.toLocaleString()}</td>
      <td>${issueCount > 0 ? `⚠️ ${issueCount}` : '✅ None'}</td>
    </tr>`;
  });
  
  html += '</tbody></table></div>';
  
  // Render AI recommendation
  html += '<div class="ai-recommendation">';
  html += '<h3>🤖 AI Analysis</h3>';
  html += `<div class="recommendation-text">${formatRecommendation(data.ai_recommendation)}</div>`;
  html += `<p class="model-info"><small>Model: ${data.model_used}</small></p>`;
  html += '</div>';
  
  container.innerHTML = html;
}

function formatRecommendation(text) {
  // Convert markdown-style formatting to HTML
  return text
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/\n\n/g, '</p><p>')
    .replace(/\n(\d+\.)/g, '<br/>$1')
    .replace(/^(.+)$/, '<p>$1</p>');
}

