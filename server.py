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


# Redirect root to Home page
@app.get("/")
def root() -> RedirectResponse:
    return RedirectResponse(url="/ui/home.html")


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


@app.get("/api/vendors/{vendor_id}/items")
def get_vendor_items(vendor_id: str, status: str = None, limit: int = None) -> JSONResponse:
    """Get detailed item-level data for a vendor with optional filtering"""
    matches = list(COMPARISON_DIR.glob(f"{vendor_id}_-_*_comparison.csv"))
    
    if not matches:
        raise HTTPException(status_code=404, detail=f"Vendor {vendor_id} not found")
    
    csv_file = matches[0]
    
    try:
        with open(csv_file, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f)
            items = list(reader)
        
        # Filter by status if provided
        if status:
            status_map = {
                "excellent": "✅ EXCELLENT",
                "good": "✓ GOOD",
                "fair": "⚠ FAIR",
                "missing": "❌ MISSING/POOR"
            }
            filter_status = status_map.get(status.lower())
            if filter_status:
                items = [item for item in items if filter_status in item.get("Match Status", "")]
        
        # Limit results if specified
        if limit and limit > 0:
            items = items[:limit]
        
        return JSONResponse({"vendor_id": vendor_id, "items": items, "total": len(items)})
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading items: {str(e)}")


@app.get("/api/statistics")
def get_overall_statistics() -> JSONResponse:
    """Get comprehensive statistics across all vendors"""
    if not COMPARISON_DIR.exists():
        return JSONResponse({"message": "No data available"})
    
    all_stats = {
        "total_vendors": 0,
        "total_boq_items": 0,
        "overall_match_rate": 0.0,
        "price_range": {"min": float('inf'), "max": 0.0, "avg": 0.0},
        "quality_distribution": {
            "excellent": 0,
            "good": 0,
            "fair": 0,
            "missing": 0
        },
        "common_issues": {},
        "vendors": []
    }
    
    total_prices = []
    
    for csv_file in COMPARISON_DIR.glob("*_comparison.csv"):
        match = re.match(r"(\d+)_-_(.+)_comparison\.csv", csv_file.name)
        if not match:
            continue
        
        vendor_id = match.group(1)
        vendor_name = match.group(2).replace("_", " ")
        
        with open(csv_file, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        total_items = len(rows)
        matched = sum(1 for r in rows if r.get("Vendor Description") and r.get("Vendor Description") != "NOT QUOTED")
        
        # Quality distribution
        excellent = sum(1 for r in rows if "✅ EXCELLENT" in r.get("Match Status", ""))
        good = sum(1 for r in rows if "✓ GOOD" in r.get("Match Status", ""))
        fair = sum(1 for r in rows if "⚠ FAIR" in r.get("Match Status", ""))
        missing = sum(1 for r in rows if "❌ MISSING/POOR" in r.get("Match Status", ""))
        
        all_stats["quality_distribution"]["excellent"] += excellent
        all_stats["quality_distribution"]["good"] += good
        all_stats["quality_distribution"]["fair"] += fair
        all_stats["quality_distribution"]["missing"] += missing
        
        # Price analysis
        total_price = 0.0
        for r in rows:
            try:
                price_str = r.get("Vendor Total Price", "").strip()
                if price_str:
                    price = float(price_str)
                    total_price += price
            except (ValueError, AttributeError):
                pass
        
        if total_price > 0:
            total_prices.append(total_price)
            all_stats["price_range"]["min"] = min(all_stats["price_range"]["min"], total_price)
            all_stats["price_range"]["max"] = max(all_stats["price_range"]["max"], total_price)
        
        # Issues
        for r in rows:
            issues = r.get("Issues", "")
            if issues and issues != "None":
                for issue in issues.split(";"):
                    issue_type = issue.split(":")[0].strip() if ":" in issue else issue.strip()
                    if issue_type:
                        all_stats["common_issues"][issue_type] = all_stats["common_issues"].get(issue_type, 0) + 1
        
        all_stats["vendors"].append({
            "id": vendor_id,
            "name": vendor_name,
            "total_items": total_items,
            "matched_items": matched,
            "match_rate": round(matched / total_items * 100, 1) if total_items > 0 else 0,
            "total_price": round(total_price, 2),
            "quality": {
                "excellent": excellent,
                "good": good,
                "fair": fair,
                "missing": missing
            }
        })
        
        all_stats["total_vendors"] += 1
        all_stats["total_boq_items"] = total_items  # Assuming all vendors quote same BOQ
    
    # Calculate overall metrics
    if all_stats["vendors"]:
        all_stats["overall_match_rate"] = round(
            sum(v["match_rate"] for v in all_stats["vendors"]) / len(all_stats["vendors"]), 1
        )
    
    if total_prices:
        all_stats["price_range"]["avg"] = round(sum(total_prices) / len(total_prices), 2)
    
    if all_stats["price_range"]["min"] == float('inf'):
        all_stats["price_range"]["min"] = 0.0
    
    return JSONResponse(all_stats)


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
        matched = sum(1 for r in rows if r.get("Vendor Description") and r.get("Vendor Description") != "NOT QUOTED")
        
        # Quality breakdown
        excellent = sum(1 for r in rows if "✅ EXCELLENT" in r.get("Match Status", ""))
        good = sum(1 for r in rows if "✓ GOOD" in r.get("Match Status", ""))
        fair = sum(1 for r in rows if "⚠ FAIR" in r.get("Match Status", ""))
        missing = total_items - (excellent + good + fair)
        
        total_price = sum(float(r.get("Vendor Total Price", 0) or 0) for r in rows if r.get("Vendor Total Price", "").strip())
        
        # Count issues
        qty_issues = sum(1 for r in rows if "Qty variance" in r.get("Issues", ""))
        uom_issues = sum(1 for r in rows if "UOM mismatch" in r.get("Issues", ""))
        type_issues = sum(1 for r in rows if "Type mismatch" in r.get("Issues", ""))
        
        # Calculate average confidence
        confidences = []
        for r in rows:
            try:
                conf = float(r.get("Match Confidence", "0"))
                confidences.append(conf)
            except (ValueError, TypeError):
                pass
        avg_confidence = round(sum(confidences) / len(confidences), 3) if confidences else 0.0
        
        vendors_data.append({
            "vendor_id": vendor_id,
            "vendor_name": vendor_name,
            "total_items": total_items,
            "matched_items": matched,
            "match_rate": round(matched / total_items * 100, 1) if total_items > 0 else 0,
            "total_price": round(total_price, 2),
            "avg_confidence": avg_confidence,
            "quality": {
                "excellent": excellent,
                "good": good,
                "fair": fair,
                "missing": missing
            },
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
        
        prompt = f"""You are an expert procurement analyst with deep experience in vendor evaluation and RFQ analysis. 

Analyze the following vendor quotation data and provide a comprehensive, data-driven recommendation:

Vendor Comparison Data:
{json.dumps(vendors_data, indent=2)}

Provide a detailed analysis with:

1. **Executive Summary** - One paragraph with the recommended vendor and key rationale

2. **Vendor Rankings** - Rank all vendors from best to worst with scores out of 100

3. **Detailed Vendor Analysis** - For each vendor:
   - Key strengths (2-3 points)
   - Key weaknesses (2-3 points)
   - Quality assessment (based on match rates and confidence)
   - Pricing position (competitive/average/high)

4. **Risk Assessment** - Identify top 3 risks across all vendors with mitigation strategies

5. **Price Competitiveness** - Comparative price analysis with recommendations

6. **Final Recommendation** - Specific, actionable recommendation with justification

Be thorough, specific, and data-driven. Use the actual numbers provided."""

        payload = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 4000,
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
        
        # Enhanced heuristic: consider multiple factors
        def score_vendor(v):
            match_score = v["match_rate"] * 0.4
            quality_score = (v["quality"]["excellent"] * 3 + v["quality"]["good"] * 2 + v["quality"]["fair"]) * 0.3
            # Normalize price (lower is better)
            max_price = max(vd["total_price"] for vd in vendors_data if vd["total_price"] > 0)
            price_score = (1 - (v["total_price"] / max_price if max_price > 0 else 0)) * 30
            return match_score + quality_score + price_score
        
        vendors_data.sort(key=score_vendor, reverse=True)
        best_vendor = vendors_data[0]
        
        recommendation = f"""**[Heuristic Analysis]**

Based on automated scoring considering match rate, quality, and pricing:

**Recommended Vendor: {best_vendor['vendor_name']}**

**Key Metrics:**
- Match Rate: {best_vendor['match_rate']}%
- Total Price: ${best_vendor['total_price']:,.2f}
- Quality: {best_vendor['quality']['excellent']} Excellent, {best_vendor['quality']['good']} Good matches

**Vendor Rankings:**
"""
        for i, v in enumerate(vendors_data, 1):
            recommendation += f"\n{i}. **{v['vendor_name']}** - {v['match_rate']}% match, ${v['total_price']:,.2f}"
        
        recommendation += f"\n\n**Note:** AI analysis unavailable. This is a heuristic recommendation based on weighted scoring.\nError: {str(e)}"
        
        return JSONResponse({
            "vendors": vendors_data,
            "ai_recommendation": recommendation,
            "model_used": "heuristic",
            "error": str(e),
        })


@app.post("/api/analyze-item")
def analyze_single_item(
    vendor_id: str = Form(...),
    boq_sr_no: str = Form(...),
) -> JSONResponse:
    """Get AI-powered analysis for a specific BOQ item and vendor match"""
    
    if not BEDROCK_AVAILABLE:
        raise HTTPException(status_code=503, detail="Bedrock not available")
    
    # Find the vendor CSV
    matches = list(COMPARISON_DIR.glob(f"{vendor_id}_-_*_comparison.csv"))
    if not matches:
        raise HTTPException(status_code=404, detail="Vendor not found")
    
    csv_file = matches[0]
    
    # Find the specific item
    with open(csv_file, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        item_data = None
        for row in reader:
            if row.get("BOQ Sr.No", "").strip() == boq_sr_no.strip():
                item_data = row
                break
    
    if not item_data:
        raise HTTPException(status_code=404, detail="Item not found")
    
    # Use Bedrock to analyze this specific match
    try:
        profile = os.getenv("AWS_PROFILE", "thinktank")
        region = os.getenv("AWS_REGION", "us-east-1")
        model_id = os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-5-sonnet-20240620-v1:0")
        
        session = boto3.Session(profile_name=profile, region_name=region)
        bedrock = session.client("bedrock-runtime")
        
        prompt = f"""Analyze this specific BOQ item and vendor quotation match:

**BOQ Requirement:**
- Sr.No: {item_data.get('BOQ Sr.No')}
- Description: {item_data.get('BOQ Description')}
- Quantity: {item_data.get('BOQ Qty')} {item_data.get('BOQ UOM')}
- Item Type: {item_data.get('BOQ Item Type')}
- Dimensions: {item_data.get('BOQ Dimensions')}
- Material: {item_data.get('BOQ Material')}

**Vendor Quote:**
- Description: {item_data.get('Vendor Description')}
- Quantity: {item_data.get('Vendor Qty')} {item_data.get('Vendor UOM')}
- Unit Price: {item_data.get('Vendor Unit Price')}
- Total Price: {item_data.get('Vendor Total Price')}
- Brand: {item_data.get('Vendor Brand')}

**Match Status:** {item_data.get('Match Status')}
**Match Confidence:** {item_data.get('Match Confidence')}
**Issues:** {item_data.get('Issues')}
**LLM Reasoning:** {item_data.get('LLM Reasoning')}

Provide:
1. Is this a correct match? (Yes/No/Uncertain)
2. Key observations about the match quality
3. Any discrepancies or concerns
4. Pricing assessment (fair/high/low if applicable)
5. Recommendation (Accept/Reject/Review)

Be specific and concise."""

        payload = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1000,
            "temperature": 0.2,
            "messages": [{"role": "user", "content": prompt}]
        }
        
        response = bedrock.invoke_model(modelId=model_id, body=json.dumps(payload))
        result = json.loads(response["body"].read())
        analysis = result["content"][0]["text"]
        
        return JSONResponse({
            "item": item_data,
            "analysis": analysis,
            "model_used": model_id
        })
    
    except Exception as e:
        return JSONResponse({
            "item": item_data,
            "analysis": f"AI analysis unavailable: {str(e)}",
            "model_used": "none",
            "error": str(e)
        })


