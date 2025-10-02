from pathlib import Path
from typing import List
import csv
import json
import os
import re

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from complete_workflow import CompleteWorkflowOrchestrator

# Bedrock client for AI recommendations
try:
    import boto3
    BEDROCK_AVAILABLE = True
except ImportError:
    BEDROCK_AVAILABLE = False


BASE_DIR = Path("/Users/sheiphanjoseph/Desktop/Developer/al_shirawi_orc_poc")
DATA_DIR = BASE_DIR / "data"
OUT_DIR = BASE_DIR / "out"
COMPARISON_DIR = OUT_DIR / "vendor_comparisons"

ENQUIRY_ATTACHMENT_DIR = DATA_DIR / "Enquiry Attachment"

app = FastAPI(title="Al Shirawi ORC POC API", version="0.1.0")

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

# Static UI
STATIC_DIR = BASE_DIR / "static"
ensure_dir(STATIC_DIR)
app.mount("/ui", StaticFiles(directory=str(STATIC_DIR), html=True), name="ui")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/upload/enquiry")
async def upload_enquiry_excel(file: UploadFile = File(...)) -> JSONResponse:
    # Only accept Excel files
    allowed_suffixes = {".xlsx", ".xlsm", ".xls"}
    suffix = Path(file.filename).suffix.lower()
    if suffix not in allowed_suffixes:
        raise HTTPException(status_code=400, detail="Only Excel files are allowed (.xlsx/.xlsm/.xls)")

    ensure_dir(ENQUIRY_ATTACHMENT_DIR)

    dest_path = ENQUIRY_ATTACHMENT_DIR / Path(file.filename).name
    contents = await file.read()
    dest_path.write_bytes(contents)

    return JSONResponse({
        "message": "Enquiry Excel uploaded",
        "filename": file.filename,
        "saved_to": str(dest_path),
    })


@app.post("/upload/vendor")
async def upload_vendor_pdfs(
    response_number: int = Form(..., description="Response number N to map folder 'Response N Attachment'"),
    files: List[UploadFile] = File(..., description="One or more vendor PDF files"),
) -> JSONResponse:
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    # Create folder like: data/Response N Attachment/
    folder_name = f"Response {response_number} Attachment"
    dest_dir = DATA_DIR / folder_name
    ensure_dir(dest_dir)

    saved: List[str] = []
    for f in files:
        suffix = Path(f.filename).suffix.lower()
        if suffix != ".pdf":
            raise HTTPException(status_code=400, detail=f"Only PDF files are allowed: {f.filename}")
        dest_path = dest_dir / Path(f.filename).name
        contents = await f.read()
        dest_path.write_bytes(contents)
        saved.append(str(dest_path))

    return JSONResponse({
        "message": "Vendor PDFs uploaded",
        "response_number": response_number,
        "count": len(saved),
        "saved_to": saved,
    })


@app.post("/run-workflow")
def run_workflow(
    skip_textract: bool = Form(False),
    skip_boq: bool = Form(False),
) -> JSONResponse:
    try:
        orchestrator = CompleteWorkflowOrchestrator()
        result = orchestrator.run(skip_textract=skip_textract, skip_boq=skip_boq)
        if result != 0:
            raise HTTPException(status_code=500, detail=f"Workflow failed with code {result}")
        return JSONResponse({"message": "Workflow completed successfully", "code": result})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Redirect root to UI
@app.get("/")
def root() -> RedirectResponse:
    return RedirectResponse(url="/ui/")


# ========================================================================
# Vendor Comparison & Analysis APIs
# ========================================================================

@app.get("/api/vendors")
def list_vendors() -> JSONResponse:
    """List all available vendor comparisons"""
    if not COMPARISON_DIR.exists():
        return JSONResponse({"vendors": [], "message": "No comparisons found. Run workflow first."})
    
    vendors = []
    for csv_file in COMPARISON_DIR.glob("*_comparison.csv"):
        # Extract vendor name from filename like "3_-_IKKT_comparison.csv"
        match = re.match(r"(\d+)_-_(.+)_comparison\.csv", csv_file.name)
        if match:
            response_num = match.group(1)
            vendor_name = match.group(2).replace("_", " ")
            vendors.append({
                "id": response_num,
                "name": vendor_name,
                "filename": csv_file.name,
            })
    
    return JSONResponse({"vendors": sorted(vendors, key=lambda v: int(v["id"]))})


@app.get("/api/vendors/{vendor_id}/comparison")
def get_vendor_comparison(vendor_id: str) -> JSONResponse:
    """Get detailed comparison data for a specific vendor"""
    # Find the comparison CSV
    matches = list(COMPARISON_DIR.glob(f"{vendor_id}_-_*_comparison.csv"))
    
    if not matches:
        raise HTTPException(status_code=404, detail=f"Vendor {vendor_id} not found")
    
    csv_file = matches[0]
    
    try:
        with open(csv_file, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        # Calculate summary statistics
        total_items = len(rows)
        matched_items = sum(1 for r in rows if r.get("Vendor Description"))
        excellent_matches = sum(1 for r in rows if "✅ EXCELLENT" in r.get("Match Status", ""))
        good_matches = sum(1 for r in rows if "✓ GOOD" in r.get("Match Status", ""))
        
        # Calculate total price
        total_price = 0.0
        for r in rows:
            try:
                price_str = r.get("Vendor Total Price", "").strip()
                if price_str:
                    total_price += float(price_str)
            except (ValueError, AttributeError):
                pass
        
        # Extract issues summary
        issues_summary = {}
        for r in rows:
            issues = r.get("Issues", "")
            if issues and issues != "None":
                issue_types = issues.split(";")
                for issue in issue_types:
                    issue = issue.strip()
                    if ":" in issue:
                        issue_type = issue.split(":")[0].strip()
                        issues_summary[issue_type] = issues_summary.get(issue_type, 0) + 1
        
        return JSONResponse({
            "vendor_id": vendor_id,
            "filename": csv_file.name,
            "summary": {
                "total_items": total_items,
                "matched_items": matched_items,
                "excellent_matches": excellent_matches,
                "good_matches": good_matches,
                "match_rate": round(matched_items / total_items * 100, 1) if total_items > 0 else 0,
                "total_price": round(total_price, 2),
                "issues": issues_summary,
            },
            "items": rows,
        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading comparison: {str(e)}")


@app.post("/api/analyze")
def analyze_vendors(vendor_ids: List[str] = Form(...)) -> JSONResponse:
    """AI-powered analysis and recommendation of vendors using AWS Bedrock"""
    
    if not BEDROCK_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="AWS Bedrock not available. Install boto3 and configure AWS credentials."
        )
    
    # Collect comparison data for all vendors
    vendors_data = []
    for vendor_id in vendor_ids:
        matches = list(COMPARISON_DIR.glob(f"{vendor_id}_-_*_comparison.csv"))
        if not matches:
            continue
        
        csv_file = matches[0]
        vendor_name = re.match(r"\d+_-_(.+)_comparison\.csv", csv_file.name).group(1).replace("_", " ")
        
        with open(csv_file, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        # Calculate metrics
        total_items = len(rows)
        matched = sum(1 for r in rows if r.get("Vendor Description"))
        total_price = sum(float(r.get("Vendor Total Price", 0) or 0) for r in rows if r.get("Vendor Total Price", "").strip())
        
        # Count issues
        qty_issues = sum(1 for r in rows if "Qty variance" in r.get("Issues", ""))
        uom_issues = sum(1 for r in rows if "UOM mismatch" in r.get("Issues", ""))
        type_issues = sum(1 for r in rows if "Type mismatch" in r.get("Issues", ""))
        
        vendors_data.append({
            "vendor_id": vendor_id,
            "vendor_name": vendor_name,
            "total_items": total_items,
            "matched_items": matched,
            "match_rate": round(matched / total_items * 100, 1) if total_items > 0 else 0,
            "total_price": round(total_price, 2),
            "issues": {
                "qty_variance": qty_issues,
                "uom_mismatch": uom_issues,
                "type_mismatch": type_issues,
            }
        })
    
    # Generate AI recommendation using Bedrock
    try:
        profile = os.getenv("AWS_PROFILE", "thinktank")
        region = os.getenv("AWS_REGION", "us-east-1")
        model_id = os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-5-sonnet-20240620-v1:0")
        
        session = boto3.Session(profile_name=profile, region_name=region)
        bedrock = session.client("bedrock-runtime")
        
        prompt = f"""You are an expert procurement analyst. Analyze the following vendor quotes and provide a recommendation on which vendor offers the best value.

Vendor Comparison Data:
{json.dumps(vendors_data, indent=2)}

Please provide:
1. A ranking of vendors from best to worst
2. Key strengths and weaknesses of each vendor
3. Your recommended vendor with clear justification
4. Any red flags or concerns to be aware of
5. Price competitiveness analysis

Be concise but thorough. Focus on practical business value."""

        payload = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 2000,
            "temperature": 0.3,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }
        
        response = bedrock.invoke_model(
            modelId=model_id,
            body=json.dumps(payload)
        )
        
        result = json.loads(response["body"].read())
        ai_recommendation = result["content"][0]["text"]
        
        return JSONResponse({
            "vendors": vendors_data,
            "ai_recommendation": ai_recommendation,
            "model_used": model_id,
        })
    
    except Exception as e:
        # Fallback to heuristic recommendation
        if not vendors_data:
            raise HTTPException(status_code=400, detail="No valid vendor data found")
        
        # Simple heuristic: best match rate, then lowest price
        best_vendor = max(vendors_data, key=lambda v: (v["match_rate"], -v["total_price"]))
        
        return JSONResponse({
            "vendors": vendors_data,
            "ai_recommendation": f"[Heuristic] Based on match rate and price, **{best_vendor['vendor_name']}** appears to be the best option with {best_vendor['match_rate']}% match rate and total price of ${best_vendor['total_price']:,.2f}.\n\nNote: AI analysis unavailable ({str(e)}). This is a simple heuristic recommendation.",
            "model_used": "heuristic",
            "error": str(e),
        })


