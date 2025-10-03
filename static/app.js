const $ = (sel) => document.querySelector(sel);

async function uploadEnquiry(e) {
  e.preventDefault();
  const fileInput = $("#enquiry-file");
  const out = $("#enquiry-result");
  out.textContent = "Uploading...";
  const fd = new FormData();
  if (!fileInput.files || fileInput.files.length === 0) {
    out.textContent = "Please choose an Excel file.";
    return;
  }
  fd.append("file", fileInput.files[0]);
  try {
    const res = await fetch("/upload/enquiry", { method: "POST", body: fd });
    const data = await res.json();
    out.textContent = JSON.stringify(data, null, 2);
  } catch (err) {
    out.textContent = String(err);
  }
}

async function uploadVendor(e) {
  e.preventDefault();
  const responseNum = $("#vendor-response").value;
  const filesInput = $("#vendor-files");
  const out = $("#vendor-result");
  out.textContent = "Uploading...";
  const fd = new FormData();
  fd.append("response_number", responseNum);
  const files = filesInput.files;
  if (!files || files.length === 0) {
    out.textContent = "Please select one or more PDF files.";
    return;
  }
  for (const f of files) fd.append("files", f);
  try {
    const res = await fetch("/upload/vendor", { method: "POST", body: fd });
    const data = await res.json();
    out.textContent = JSON.stringify(data, null, 2);
  } catch (err) {
    out.textContent = String(err);
  }
}

async function runWorkflow(e) {
  e.preventDefault();
  const skipTextract = $("#skip-textract").checked;
  const skipBoq = $("#skip-boq").checked;
  const out = $("#run-result");
  out.textContent = "Running... (this can take several minutes)";
  const fd = new FormData();
  if (skipTextract) fd.append("skip_textract", "true");
  if (skipBoq) fd.append("skip_boq", "true");
  try {
    const res = await fetch("/run-workflow", { method: "POST", body: fd });
    const data = await res.json();
    out.textContent = JSON.stringify(data, null, 2);
    
    // Show success message with links
    if (res.ok && data.code === 0) {
      const successDiv = $("#workflow-success");
      if (successDiv) {
        successDiv.style.display = "block";
      }
    }
  } catch (err) {
    out.textContent = String(err);
  }
}

window.addEventListener("DOMContentLoaded", () => {
  $("#form-enquiry").addEventListener("submit", uploadEnquiry);
  $("#form-vendor").addEventListener("submit", uploadVendor);
  $("#form-run").addEventListener("submit", runWorkflow);
});


