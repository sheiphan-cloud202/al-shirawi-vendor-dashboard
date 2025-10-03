// Comprehensive Dashboard JavaScript
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

// Global State
let allStatistics = null;
let selectedVendorIds = new Set();
let charts = {};
let currentVendorItems = [];

// Initialize dashboard
window.addEventListener("DOMContentLoaded", async () => {
  await loadDashboard();
});

// Main load function
async function loadDashboard() {
  try {
    // Load comprehensive statistics
    const res = await fetch("/api/statistics");
    const data = await res.json();
    
    $("#loading").style.display = "none";
    
    if (!data.vendors || data.vendors.length === 0) {
      $("#no-data").style.display = "flex";
      return;
    }
    
    allStatistics = data;
    renderDashboard();
    $("#dashboard").style.display = "block";
    
  } catch (err) {
    $("#loading").innerHTML = `<div class="error">Error loading data: ${err.message}</div>`;
  }
}

// Render all dashboard components
function renderDashboard() {
  renderSummaryStats();
  renderCharts();
  renderVendorTable();
}

// Executive Summary Stats
function renderSummaryStats() {
  const stats = allStatistics;
  
  $("#stat-vendors").textContent = stats.total_vendors;
  $("#stat-match-rate").textContent = `${stats.overall_match_rate}%`;
  $("#stat-items").textContent = stats.total_boq_items.toLocaleString();
  
  const priceMin = stats.price_range.min.toLocaleString();
  const priceMax = stats.price_range.max.toLocaleString();
  $("#stat-price-range").textContent = `$${priceMin} - $${priceMax}`;
}

// Render all charts
function renderCharts() {
  renderQualityChart();
  renderMatchRateChart();
  renderPriceChart();
  renderIssuesChart();
}

// Quality Distribution Pie Chart
function renderQualityChart() {
  const ctx = $("#qualityChart");
  const dist = allStatistics.quality_distribution;
  
  if (charts.quality) charts.quality.destroy();
  
  charts.quality = new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: ['✅ Excellent', '✓ Good', '⚠ Fair', '❌ Missing'],
      datasets: [{
        data: [dist.excellent, dist.good, dist.fair, dist.missing],
        backgroundColor: [
          'rgba(34, 197, 94, 0.8)',
          'rgba(59, 130, 246, 0.8)',
          'rgba(251, 146, 60, 0.8)',
          'rgba(239, 68, 68, 0.8)'
        ],
        borderWidth: 2,
        borderColor: '#fff'
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: true,
      plugins: {
        legend: {
          position: 'bottom',
          labels: {
            padding: 15,
            font: { size: 12 }
          }
        },
        tooltip: {
          callbacks: {
            label: function(context) {
              const total = context.dataset.data.reduce((a, b) => a + b, 0);
              const percent = ((context.parsed / total) * 100).toFixed(1);
              return `${context.label}: ${context.parsed} (${percent}%)`;
            }
          }
        }
      }
    }
  });
}

// Vendor Match Rates Bar Chart
function renderMatchRateChart() {
  const ctx = $("#matchRateChart");
  const vendors = allStatistics.vendors;
  
  if (charts.matchRate) charts.matchRate.destroy();
  
  charts.matchRate = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: vendors.map(v => v.name),
      datasets: [{
        label: 'Match Rate (%)',
        data: vendors.map(v => v.match_rate),
        backgroundColor: vendors.map(v => {
          if (v.match_rate >= 80) return 'rgba(34, 197, 94, 0.8)';
          if (v.match_rate >= 60) return 'rgba(59, 130, 246, 0.8)';
          if (v.match_rate >= 40) return 'rgba(251, 146, 60, 0.8)';
          return 'rgba(239, 68, 68, 0.8)';
        }),
        borderRadius: 6,
        borderWidth: 0
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: true,
      scales: {
        y: {
          beginAtZero: true,
          max: 100,
          ticks: { callback: (val) => val + '%' }
        },
        x: {
          ticks: { maxRotation: 45, minRotation: 45 }
        }
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (ctx) => `Match Rate: ${ctx.parsed.y}%`
          }
        }
      }
    }
  });
}

// Price Comparison Bar Chart
function renderPriceChart() {
  const ctx = $("#priceChart");
  const vendors = allStatistics.vendors.filter(v => v.total_price > 0);
  
  if (charts.price) charts.price.destroy();
  
  charts.price = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: vendors.map(v => v.name),
      datasets: [{
        label: 'Total Price ($)',
        data: vendors.map(v => v.total_price),
        backgroundColor: 'rgba(139, 92, 246, 0.8)',
        borderRadius: 6,
        borderWidth: 0
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: true,
      scales: {
        y: {
          beginAtZero: true,
          ticks: {
            callback: (val) => '$' + val.toLocaleString()
          }
        },
        x: {
          ticks: { maxRotation: 45, minRotation: 45 }
        }
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (ctx) => `Total: $${ctx.parsed.y.toLocaleString()}`
          }
        }
      }
    }
  });
}

// Issues Breakdown Horizontal Bar Chart
function renderIssuesChart() {
  const ctx = $("#issuesChart");
  const issues = allStatistics.common_issues;
  
  if (charts.issues) charts.issues.destroy();
  
  const sortedIssues = Object.entries(issues)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 8);  // Top 8 issues
  
  charts.issues = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: sortedIssues.map(([type, _]) => type),
      datasets: [{
        label: 'Count',
        data: sortedIssues.map(([_, count]) => count),
        backgroundColor: 'rgba(251, 146, 60, 0.8)',
        borderRadius: 6,
        borderWidth: 0
      }]
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: true,
      scales: {
        x: { beginAtZero: true },
      },
      plugins: {
        legend: { display: false },
      }
    }
  });
}

// Render Vendor Comparison Table
function renderVendorTable() {
  const tbody = $("#vendor-table-body");
  tbody.innerHTML = "";
  
  const vendors = allStatistics.vendors.sort((a, b) => b.match_rate - a.match_rate);
  
  vendors.forEach((vendor, index) => {
    const row = document.createElement("tr");
    row.className = index % 2 === 0 ? "row-even" : "row-odd";
    
    const qualityScore = calculateQualityScore(vendor.quality);
    const totalIssues = Object.values(vendor.quality).reduce((a, b) => a + b, 0) - 
                        vendor.quality.excellent - vendor.quality.good;
    
    row.innerHTML = `
      <td>
        <input type="checkbox" class="vendor-checkbox" value="${vendor.id}" />
      </td>
      <td class="vendor-name">
        <strong>${vendor.name}</strong>
        <span class="vendor-id-badge">#${vendor.id}</span>
      </td>
      <td>
        <div class="progress-bar">
          <div class="progress-fill" style="width: ${vendor.match_rate}%"></div>
          <span class="progress-text">${vendor.match_rate}%</span>
        </div>
      </td>
      <td>
        <span class="quality-badge ${getQualityClass(qualityScore)}">
          ${qualityScore}/100
        </span>
      </td>
      <td class="price-cell">$${vendor.total_price.toLocaleString()}</td>
      <td><span class="badge badge-excellent">${vendor.quality.excellent}</span></td>
      <td><span class="badge badge-good">${vendor.quality.good}</span></td>
      <td><span class="badge badge-fair">${vendor.quality.fair}</span></td>
      <td><span class="badge badge-missing">${vendor.quality.missing}</span></td>
      <td>${totalIssues > 0 ? `<span class="badge badge-warning">${totalIssues}</span>` : '-'}</td>
      <td>
        <button class="btn-action" onclick="viewVendorItems('${vendor.id}', '${vendor.name}')">
          View Items
        </button>
      </td>
    `;
    
    tbody.appendChild(row);
  });
  
  // Add event listeners to checkboxes
  $$(".vendor-checkbox").forEach(cb => {
    cb.addEventListener("change", handleCheckboxChange);
  });
  
  // Select all checkbox
  $("#checkbox-all").addEventListener("change", (e) => {
    const checked = e.target.checked;
    $$(".vendor-checkbox").forEach(cb => {
      cb.checked = checked;
      if (checked) {
        selectedVendorIds.add(cb.value);
      } else {
        selectedVendorIds.delete(cb.value);
      }
    });
    updateAnalyzeButton();
  });
}

function calculateQualityScore(quality) {
  const total = quality.excellent + quality.good + quality.fair + quality.missing;
  if (total === 0) return 0;
  
  const score = (
    (quality.excellent * 100) + 
    (quality.good * 75) + 
    (quality.fair * 40) + 
    (quality.missing * 0)
  ) / total;
  
  return Math.round(score);
}

function getQualityClass(score) {
  if (score >= 80) return 'excellent';
  if (score >= 60) return 'good';
  if (score >= 40) return 'fair';
  return 'poor';
}

function handleCheckboxChange(e) {
  const vendorId = e.target.value;
  if (e.target.checked) {
    selectedVendorIds.add(vendorId);
  } else {
    selectedVendorIds.delete(vendorId);
  }
  updateAnalyzeButton();
}

function updateAnalyzeButton() {
  const btn = $("#analyze-selected-btn");
  const count = selectedVendorIds.size;
  
  btn.disabled = count === 0;
  btn.innerHTML = count === 0 
    ? '<span class="btn-icon">🤖</span> Select Vendors to Analyze'
    : `<span class="btn-icon">🤖</span> Analyze ${count} Vendor${count > 1 ? 's' : ''}`;
}

// Select All Button
$("#select-all-btn").addEventListener("click", () => {
  $("#checkbox-all").checked = true;
  $("#checkbox-all").dispatchEvent(new Event('change'));
});

// Analyze Selected Button
$("#analyze-selected-btn").addEventListener("click", async () => {
  if (selectedVendorIds.size === 0) return;
  
  $("#ai-section").style.display = "block";
  $("#ai-loading").style.display = "block";
  $("#ai-results").style.display = "none";
  
  // Scroll to AI section
  $("#ai-section").scrollIntoView({ behavior: 'smooth', block: 'start' });
  
  try {
    const formData = new FormData();
    selectedVendorIds.forEach(id => formData.append("vendor_ids", id));
    
    const res = await fetch("/api/analyze", {
      method: "POST",
      body: formData
    });
    
    if (!res.ok) {
      throw new Error(`API error: ${res.status} ${res.statusText}`);
    }
    
    const data = await res.json();
    
    $("#ai-loading").style.display = "none";
    $("#ai-results").style.display = "block";
    renderAIAnalysis(data);
    
  } catch (err) {
    $("#ai-loading").style.display = "none";
    $("#ai-results").style.display = "block";
    $("#ai-results").innerHTML = `
      <div class="error-box">
        <h3>⚠️ Analysis Error</h3>
        <p>${err.message}</p>
        <small>Please check your AWS credentials and Bedrock configuration.</small>
      </div>
    `;
  }
});

// Close AI Section
$("#close-ai-btn").addEventListener("click", () => {
  $("#ai-section").style.display = "none";
});

// Render AI Analysis Results
function renderAIAnalysis(data) {
  const container = $("#ai-results");
  
  let html = `
    <div class="ai-header">
      <div class="ai-badge">
        <span class="ai-icon">🤖</span>
        <span>Powered by ${data.model_used}</span>
      </div>
    </div>
  `;
  
  // Vendor metrics comparison table
  html += `
    <div class="metrics-comparison">
      <h3>Vendor Metrics Comparison</h3>
      <div class="comparison-table-wrapper">
        <table class="comparison-table">
          <thead>
            <tr>
              <th>Vendor</th>
              <th>Match Rate</th>
              <th>Quality</th>
              <th>Total Price</th>
              <th>Confidence</th>
              <th>Issues</th>
            </tr>
          </thead>
          <tbody>
  `;
  
  data.vendors.forEach(v => {
    const qualityScore = calculateQualityScore(v.quality);
    const totalIssues = Object.values(v.issues).reduce((a, b) => a + b, 0);
    
    html += `
      <tr>
        <td><strong>${v.vendor_name}</strong></td>
        <td>
          <div class="mini-progress">
            <div class="mini-progress-fill" style="width: ${v.match_rate}%"></div>
          </div>
          ${v.match_rate}%
        </td>
        <td>
          <span class="quality-badge ${getQualityClass(qualityScore)}">${qualityScore}</span>
        </td>
        <td class="price-cell">$${v.total_price.toLocaleString()}</td>
        <td>${(v.avg_confidence * 100).toFixed(0)}%</td>
        <td>${totalIssues > 0 ? `<span class="badge badge-warning">${totalIssues}</span>` : '✅'}</td>
      </tr>
    `;
  });
  
  html += `
          </tbody>
        </table>
      </div>
    </div>
  `;
  
  // AI Recommendation Text
  html += `
    <div class="ai-recommendation-box">
      <h3>📋 Detailed Analysis & Recommendations</h3>
      <div class="recommendation-content">
        ${formatAIRecommendation(data.ai_recommendation)}
      </div>
    </div>
  `;
  
  container.innerHTML = html;
}

function formatAIRecommendation(text) {
  // Format markdown-style text to HTML
  let formatted = text
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.+?)\*/g, '<em>$1</em>')
    .replace(/^### (.+)$/gm, '<h4>$1</h4>')
    .replace(/^## (.+)$/gm, '<h3>$1</h3>')
    .replace(/^# (.+)$/gm, '<h2>$1</h2>')
    .replace(/^\d+\.\s+(.+)$/gm, '<li>$1</li>')
    .replace(/^-\s+(.+)$/gm, '<li>$1</li>')
    .replace(/\n\n/g, '</p><p>')
    .replace(/(<li>.*<\/li>\s*)+/g, '<ul>$&</ul>');
  
  return `<p>${formatted}</p>`;
}

// View Vendor Items Modal
async function viewVendorItems(vendorId, vendorName) {
  const modal = $("#item-modal");
  $("#modal-vendor-name").textContent = vendorName;
  modal.style.display = "flex";
  
  try {
    const res = await fetch(`/api/vendors/${vendorId}/comparison`);
    const data = await res.json();
    
    currentVendorItems = data.items;
    renderItemList(currentVendorItems);
    
    // Setup filters
    $("#item-status-filter").addEventListener("change", filterItems);
    $("#item-search").addEventListener("input", filterItems);
    
  } catch (err) {
    $("#item-list").innerHTML = `<div class="error">Error loading items: ${err.message}</div>`;
  }
}

function renderItemList(items) {
  const container = $("#item-list");
  
  if (items.length === 0) {
    container.innerHTML = '<p class="no-results">No items match the filter criteria.</p>';
    return;
  }
  
  let html = '<div class="items-grid">';
  
  items.forEach(item => {
    const statusClass = getStatusClass(item["Match Status"]);
    const hasIssues = item.Issues && item.Issues !== "None";
    
    html += `
      <div class="item-card ${statusClass}">
        <div class="item-header">
          <span class="item-sr-no">Sr.No: ${item["BOQ Sr.No"]}</span>
          <span class="item-status">${item["Match Status"]}</span>
        </div>
        
        <div class="item-section">
          <h4>BOQ Requirement</h4>
          <p class="item-desc">${item["BOQ Description"]}</p>
          <div class="item-meta">
            <span>Qty: ${item["BOQ Qty"]} ${item["BOQ UOM"]}</span>
            <span>Type: ${item["BOQ Item Type"]}</span>
          </div>
          ${item["BOQ Dimensions"] ? `<div class="item-dims">📏 ${item["BOQ Dimensions"]}</div>` : ''}
        </div>
        
        <div class="item-section vendor-section">
          <h4>Vendor Quote</h4>
          ${item["Vendor Description"] === "NOT QUOTED" ? 
            '<p class="not-quoted">❌ Not Quoted</p>' :
            `
            <p class="item-desc">${item["Vendor Description"]}</p>
            <div class="item-meta">
              ${item["Vendor Qty"] ? `<span>Qty: ${item["Vendor Qty"]} ${item["Vendor UOM"] || ''}</span>` : ''}
              ${item["Vendor Unit Price"] ? `<span>Unit: $${parseFloat(item["Vendor Unit Price"]).toLocaleString()}</span>` : ''}
              ${item["Vendor Total Price"] ? `<span class="price-highlight">Total: $${parseFloat(item["Vendor Total Price"]).toLocaleString()}</span>` : ''}
            </div>
            ${item["Vendor Brand"] ? `<div class="item-brand">🏷️ ${item["Vendor Brand"]}</div>` : ''}
            `
          }
        </div>
        
        ${hasIssues ? `
          <div class="item-issues">
            <strong>⚠️ Issues:</strong> ${item.Issues}
          </div>
        ` : ''}
        
        <div class="item-footer">
          <div class="confidence">
            Confidence: ${parseFloat(item["Match Confidence"]).toFixed(2)}
          </div>
          <button class="btn-analyze-item" onclick='analyzeItem("${vendorId}", "${item["BOQ Sr.No"]}")'>
            🤖 AI Analysis
          </button>
        </div>
      </div>
    `;
  });
  
  html += '</div>';
  container.innerHTML = html;
}

function getStatusClass(status) {
  if (status.includes("EXCELLENT")) return "status-excellent";
  if (status.includes("GOOD")) return "status-good";
  if (status.includes("FAIR")) return "status-fair";
  return "status-missing";
}

function filterItems() {
  const statusFilter = $("#item-status-filter").value.toLowerCase();
  const searchTerm = $("#item-search").value.toLowerCase();
  
  let filtered = currentVendorItems;
  
  // Apply status filter
  if (statusFilter) {
    const statusMap = {
      "excellent": "EXCELLENT",
      "good": "GOOD",
      "fair": "FAIR",
      "missing": "MISSING"
    };
    const targetStatus = statusMap[statusFilter];
    filtered = filtered.filter(item => item["Match Status"].includes(targetStatus));
  }
  
  // Apply search filter
  if (searchTerm) {
    filtered = filtered.filter(item => 
      item["BOQ Sr.No"].toLowerCase().includes(searchTerm) ||
      item["BOQ Description"].toLowerCase().includes(searchTerm) ||
      (item["Vendor Description"] && item["Vendor Description"].toLowerCase().includes(searchTerm))
    );
  }
  
  renderItemList(filtered);
}

function closeItemModal() {
  $("#item-modal").style.display = "none";
  currentVendorItems = [];
}

// Analyze Single Item with AI
async function analyzeItem(vendorId, boqSrNo) {
  const modal = $("#item-detail-modal");
  modal.style.display = "flex";
  
  $("#item-detail-loading").style.display = "block";
  $("#item-detail-content").style.display = "none";
  
  try {
    const formData = new FormData();
    formData.append("vendor_id", vendorId);
    formData.append("boq_sr_no", boqSrNo);
    
    const res = await fetch("/api/analyze-item", {
      method: "POST",
      body: formData
    });
    
    const data = await res.json();
    
    $("#item-detail-loading").style.display = "none";
    $("#item-detail-content").style.display = "block";
    
    renderItemAnalysis(data);
    
  } catch (err) {
    $("#item-detail-loading").style.display = "none";
    $("#item-detail-content").style.display = "block";
    $("#item-detail-content").innerHTML = `<div class="error">Error: ${err.message}</div>`;
  }
}

function renderItemAnalysis(data) {
  const item = data.item;
  const analysis = data.analysis;
  
  const html = `
    <div class="item-analysis">
      <div class="analysis-item-info">
        <h4>Item: ${item["BOQ Sr.No"]}</h4>
        <p><strong>BOQ:</strong> ${item["BOQ Description"]}</p>
        <p><strong>Vendor:</strong> ${item["Vendor Description"]}</p>
      </div>
      
      <div class="analysis-result">
        <h4>🤖 AI Analysis</h4>
        <div class="analysis-text">
          ${formatAIRecommendation(analysis)}
        </div>
      </div>
      
      <div class="analysis-meta">
        <small>Model: ${data.model_used}</small>
      </div>
    </div>
  `;
  
  $("#item-detail-content").innerHTML = html;
}

function closeItemDetailModal() {
  $("#item-detail-modal").style.display = "none";
}

// Close modals on outside click
window.addEventListener("click", (e) => {
  if (e.target.classList.contains("modal")) {
    e.target.style.display = "none";
  }
});

// Keyboard shortcuts
window.addEventListener("keydown", (e) => {
  if (e.key === "Escape") {
    closeItemModal();
    closeItemDetailModal();
  }
});

// Quick Access Functions
function scrollToVendorTable() {
  const table = document.getElementById("vendor-table");
  if (table) {
    table.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }
}

function exportStatistics() {
  if (!allStatistics) {
    alert("No statistics available to export");
    return;
  }
  
  // Create downloadable JSON file
  const dataStr = JSON.stringify(allStatistics, null, 2);
  const dataBlob = new Blob([dataStr], { type: 'application/json' });
  const url = URL.createObjectURL(dataBlob);
  const link = document.createElement('a');
  link.href = url;
  link.download = `vendor-statistics-${new Date().toISOString().split('T')[0]}.json`;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}


