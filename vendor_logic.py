"""
Business logic and utility functions for vendor analysis and workflow.
"""
from pathlib import Path
import csv
import datetime
import json
import os
import re
import shutil
import uuid
from typing import List, Dict, Any, Optional

from complete_workflow import CompleteWorkflowOrchestrator

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
STATIC_DIR = BASE_DIR / "static"
UPLOADED_FILES_TRACKER = OUT_DIR / "uploaded_files_tracker.json"

def generate_session_id() -> str:
    """Generate a unique session ID"""
    return str(uuid.uuid4())[:8]  # Use first 8 characters for readability

def get_session_out_dir(session_id: str) -> Path:
    """Get the output directory for a specific session"""
    return OUT_DIR / "sessions" / session_id

def get_session_tracker_path(session_id: Optional[str]) -> Path:
    """Get the tracker file path for a specific session"""
    if session_id:
        return get_session_out_dir(session_id) / "uploaded_files_tracker.json"
    return UPLOADED_FILES_TRACKER

def get_session_upload_dir(session_id: str) -> Path:
    """Get the uploads directory for a specific session"""
    return get_session_out_dir(session_id) / "uploads"

def get_session_enquiry_dir(session_id: str) -> Path:
    """Get the enquiry upload directory for a specific session"""
    return get_session_upload_dir(session_id) / "enquiry"

def get_session_vendor_dir(session_id: str, response_number: int) -> Path:
    """Get the vendor upload directory for a specific response in a session"""
    return get_session_upload_dir(session_id) / "vendors" / f"Response_{response_number}"

def get_session_data_dir(session_id: str) -> Path:
    """Get the session-specific data directory (replaces global DATA_DIR)"""
    return get_session_upload_dir(session_id)

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _load_uploaded_files_tracker(session_id: Optional[str] = None) -> Dict[str, Any]:
    """Load the tracker of uploaded files (response_number -> list of filenames, enquiry_excel -> filename)"""
    tracker_path = get_session_tracker_path(session_id)
    
    if tracker_path.exists():
        try:
            return json.loads(tracker_path.read_text())
        except Exception:
            return {}
    return {}

def _save_uploaded_files_tracker(tracker: Dict[str, Any], session_id: Optional[str] = None) -> None:
    """Save the tracker of uploaded files"""
    tracker_path = get_session_tracker_path(session_id)
    ensure_dir(tracker_path.parent)
    tracker_path.write_text(json.dumps(tracker, indent=2))

def save_enquiry_excel(file_name: str, contents: bytes, session_id: Optional[str] = None) -> str:
    """
    Save enquiry Excel file to session-specific or global directory.
    
    Args:
        file_name: Original filename
        contents: File contents as bytes
        session_id: Session ID (if None, uses legacy global directory)
    
    Returns:
        String path to saved file
    """
    if session_id:
        # Session-isolated storage
        dest_dir = get_session_enquiry_dir(session_id)
        ensure_dir(dest_dir)
        dest_path = dest_dir / Path(file_name).name
        print(f"  💾 Saving to session directory: {dest_path}")
    else:
        # Legacy mode - global storage
        ensure_dir(ENQUIRY_ATTACHMENT_DIR)
        dest_path = ENQUIRY_ATTACHMENT_DIR / Path(file_name).name
        print(f"  💾 Saving to global directory (legacy): {dest_path}")
    
    # Write file
    dest_path.write_bytes(contents)
    
    # Update tracker with enhanced metadata
    tracker = _load_uploaded_files_tracker(session_id)
    
    # Initialize tracker structure if needed
    if "session_id" not in tracker and session_id:
        tracker["session_id"] = session_id
        tracker["created_at"] = datetime.datetime.now().isoformat()
    
    tracker["enquiry_excel"] = {
        "filename": file_name,
        "upload_path": str(dest_path.relative_to(get_session_out_dir(session_id))) if session_id else str(dest_path),
        "uploaded_at": datetime.datetime.now().isoformat(),
        "size_bytes": len(contents)
    }
    tracker["last_updated"] = datetime.datetime.now().isoformat()
    
    _save_uploaded_files_tracker(tracker, session_id)
    
    return str(dest_path)

def get_uploaded_enquiry_excel(session_id: Optional[str] = None) -> Optional[str]:
    """
    Get the filename of the uploaded enquiry Excel file.
    
    Args:
        session_id: Session ID (if None, uses legacy tracker)
    
    Returns:
        Filename or None if not uploaded
    """
    tracker = _load_uploaded_files_tracker(session_id)
    enquiry_data = tracker.get("enquiry_excel")
    
    if isinstance(enquiry_data, dict):
        return enquiry_data.get("filename")
    elif isinstance(enquiry_data, str):
        # Legacy format (backward compatibility)
        return enquiry_data
    
    return None

def save_vendor_pdfs(response_number: int, files: List[Dict[str, Any]], 
                    session_id: Optional[str] = None) -> List[str]:
    """
    Save vendor PDF files to session-specific or global directory.
    
    Args:
        response_number: Response number (e.g., 11 for "Response 11")
        files: List of dicts with 'filename' and 'contents' keys
        session_id: Session ID (if None, uses legacy global directory)
    
    Returns:
        List of saved file paths
    """
    if session_id:
        # Session-isolated storage
        dest_dir = get_session_vendor_dir(session_id, response_number)
        ensure_dir(dest_dir)
        print(f"  💾 Saving to session vendor directory: {dest_dir}")
    else:
        # Legacy mode - global storage
        folder_name = f"Response {response_number} Attachment"
        dest_dir = DATA_DIR / folder_name
        ensure_dir(dest_dir)
        print(f"  💾 Saving to global directory (legacy): {dest_dir}")
    
    saved = []
    tracker = _load_uploaded_files_tracker(session_id)
    
    # Initialize tracker structure if needed
    if "session_id" not in tracker and session_id:
        tracker["session_id"] = session_id
        tracker["created_at"] = datetime.datetime.now().isoformat()
    
    if "vendors" not in tracker:
        tracker["vendors"] = {}
    
    # Initialize this response number if not exists
    response_key = str(response_number)
    if response_key not in tracker["vendors"]:
        tracker["vendors"][response_key] = {
            "response_number": response_number,
            "files": []
        }
    
    # Save files and track metadata
    for f in files:
        filename = f['filename']
        suffix = Path(filename).suffix.lower()
        
        if suffix != ".pdf":
            raise ValueError(f"Only PDF files are allowed: {filename}")
        
        dest_path = dest_dir / Path(filename).name
        dest_path.write_bytes(f['contents'])
        saved.append(str(dest_path))
        
        # Track file metadata
        file_info = {
            "filename": filename,
            "upload_path": str(dest_path.relative_to(get_session_out_dir(session_id))) if session_id else str(dest_path),
            "uploaded_at": datetime.datetime.now().isoformat(),
            "size_bytes": len(f['contents'])
        }
        
        # Avoid duplicates
        existing_files = [f["filename"] for f in tracker["vendors"][response_key]["files"]]
        if filename not in existing_files:
            tracker["vendors"][response_key]["files"].append(file_info)
    
    tracker["last_updated"] = datetime.datetime.now().isoformat()
    _save_uploaded_files_tracker(tracker, session_id)
    
    return saved

def get_uploaded_pdfs(session_id: Optional[str] = None) -> Dict[int, List[str]]:
    """
    Get all uploaded PDFs organized by response number.
    
    Args:
        session_id: Session ID (if None, uses legacy tracker)
    
    Returns:
        Dict mapping response_number to list of filenames
    """
    tracker = _load_uploaded_files_tracker(session_id)
    result = {}
    
    # Handle new format (nested vendors structure)
    vendors = tracker.get("vendors", {})
    if isinstance(vendors, dict):
        for response_key, response_data in vendors.items():
            try:
                response_num = int(response_key)
                if isinstance(response_data, dict):
                    files = response_data.get("files", [])
                    result[response_num] = [f["filename"] if isinstance(f, dict) else f for f in files]
            except (ValueError, TypeError):
                continue
    
    # Handle legacy format (flat structure) - for backward compatibility
    for key, value in tracker.items():
        if key.isdigit() and isinstance(value, list):
            try:
                response_num = int(key)
                if response_num not in result:  # Don't overwrite new format data
                    result[response_num] = value
            except ValueError:
                continue
    
    return result

def run_workflow(skip_textract: bool = False, skip_boq: bool = False, session_id: Optional[str] = None) -> int:
    orchestrator = CompleteWorkflowOrchestrator(session_id=session_id)
    return orchestrator.run(skip_textract=skip_textract, skip_boq=skip_boq)

def list_vendors(session_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """List vendors for a specific session or default comparison directory"""
    if session_id:
        comparison_dir = get_session_out_dir(session_id) / "vendor_comparisons"
    else:
        comparison_dir = COMPARISON_DIR
    
    if not comparison_dir.exists():
        return []
    vendors = []
    for csv_file in comparison_dir.glob("*_comparison.csv"):
        match = re.match(r"(\d+)_-_(.+)_comparison\.csv", csv_file.name)
        if match:
            response_num = match.group(1)
            vendor_name = match.group(2).replace("_", " ")
            vendors.append({
                "id": response_num,
                "name": vendor_name,
                "filename": csv_file.name,
            })
    return sorted(vendors, key=lambda v: int(v["id"]))

def get_vendor_comparison(vendor_id: str) -> Dict[str, Any]:
    matches = list(COMPARISON_DIR.glob(f"{vendor_id}_-_*_comparison.csv"))
    if not matches:
        raise FileNotFoundError(f"Vendor {vendor_id} not found")
    csv_file = matches[0]
    with open(csv_file, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    total_items = len(rows)
    matched_items = sum(1 for r in rows if r.get("Vendor Description"))
    excellent_matches = sum(1 for r in rows if "✅ EXCELLENT" in r.get("Match Status", ""))
    good_matches = sum(1 for r in rows if "✓ GOOD" in r.get("Match Status", ""))
    total_price = 0.0
    for r in rows:
        try:
            price_str = r.get("Vendor Total Price", "").strip()
            if price_str:
                total_price += float(price_str)
        except (ValueError, AttributeError):
            pass
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
    return {
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
    }

def get_vendor_items(vendor_id: str, status: str = None, limit: int = None) -> Dict[str, Any]:
    matches = list(COMPARISON_DIR.glob(f"{vendor_id}_-_*_comparison.csv"))
    if not matches:
        raise FileNotFoundError(f"Vendor {vendor_id} not found")
    csv_file = matches[0]
    with open(csv_file, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        items = list(reader)
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
    if limit and limit > 0:
        items = items[:limit]
    return {"vendor_id": vendor_id, "items": items, "total": len(items)}

def get_overall_statistics() -> Dict[str, Any]:
    if not COMPARISON_DIR.exists():
        return {"message": "No data available"}
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
        excellent = sum(1 for r in rows if "✅ EXCELLENT" in r.get("Match Status", ""))
        good = sum(1 for r in rows if "✓ GOOD" in r.get("Match Status", ""))
        fair = sum(1 for r in rows if "⚠ FAIR" in r.get("Match Status", ""))
        missing = sum(1 for r in rows if "❌ MISSING/POOR" in r.get("Match Status", ""))
        all_stats["quality_distribution"]["excellent"] += excellent
        all_stats["quality_distribution"]["good"] += good
        all_stats["quality_distribution"]["fair"] += fair
        all_stats["quality_distribution"]["missing"] += missing
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
        all_stats["total_boq_items"] = total_items
    if all_stats["vendors"]:
        all_stats["overall_match_rate"] = round(
            sum(v["match_rate"] for v in all_stats["vendors"]) / len(all_stats["vendors"]), 1
        )
    if total_prices:
        all_stats["price_range"]["avg"] = round(sum(total_prices) / len(total_prices), 2)
    if all_stats["price_range"]["min"] == float('inf'):
        all_stats["price_range"]["min"] = 0.0
    return all_stats

def analyze_vendors(vendor_ids: List[str]) -> Dict[str, Any]:
    vendors_data = []
    for vendor_id in vendor_ids:
        matches = list(COMPARISON_DIR.glob(f"{vendor_id}_-_*_comparison.csv"))
        if not matches:
            continue
        csv_file = matches[0]
        m = re.match(r"\d+_-_(.+)_comparison\.csv", csv_file.name)
        if m:
            vendor_name = m.group(1).replace("_", " ")
        else:
            # fallback to filename without suffix
            vendor_name = Path(csv_file.name).stem.replace("_", " ")
        with open(csv_file, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        total_items = len(rows)
        matched = sum(1 for r in rows if r.get("Vendor Description") and r.get("Vendor Description") != "NOT QUOTED")
        excellent = sum(1 for r in rows if "✅ EXCELLENT" in r.get("Match Status", ""))
        good = sum(1 for r in rows if "✓ GOOD" in r.get("Match Status", ""))
        fair = sum(1 for r in rows if "⚠ FAIR" in r.get("Match Status", ""))
        missing = total_items - (excellent + good + fair)
        total_price = sum(float(r.get("Vendor Total Price", 0) or 0) for r in rows if r.get("Vendor Total Price", "").strip())
        qty_issues = sum(1 for r in rows if "Qty variance" in r.get("Issues", ""))
        uom_issues = sum(1 for r in rows if "UOM mismatch" in r.get("Issues", ""))
        type_issues = sum(1 for r in rows if "Type mismatch" in r.get("Issues", ""))
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
    # ... Bedrock logic omitted for brevity ...
    # Try Bedrock if available, otherwise fall back to heuristic recommendation
    ai_recommendation = None
    model_used = None
    error = None
    try:
        if BEDROCK_AVAILABLE:
            profile = os.getenv("AWS_PROFILE", "thinktank")
            region = os.getenv("AWS_REGION", "us-east-1")
            model_id = os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-5-sonnet-20240620-v1:0")

            session = boto3.Session(profile_name=profile, region_name=region)
            bedrock = session.client("bedrock-runtime")

            prompt = (
                "You are an expert procurement analyst with deep experience in vendor evaluation and RFQ analysis.\n\n"
                "Analyze the following vendor quotation data and provide a concise recommendation and ranking.\n\n"
                f"Vendor Comparison Data:\n{json.dumps(vendors_data, indent=2)}\n\n"
                "Provide: 1) Executive summary (one paragraph), 2) Rankings, 3) Short rationale for each vendor, 4) Final recommendation."
            )

            payload = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 2000,
                "temperature": 0.3,
                "messages": [{"role": "user", "content": prompt}],
            }

            response = bedrock.invoke_model(modelId=model_id, body=json.dumps(payload))
            result = json.loads(response["body"].read())
            # Defensive: navigate result structure safely
            ai_text = None
            try:
                content = result.get("content") if isinstance(result, dict) else None
                if content and isinstance(content, list) and len(content) > 0:
                    first = content[0]
                    ai_text = first.get("text") if isinstance(first, dict) else None
                if not ai_text:
                    # Some bedrock responses put text under 'response' keys - fallback to raw body
                    ai_text = str(result)
            except Exception:
                ai_text = str(result)

            ai_recommendation = ai_text
            model_used = model_id
        else:
            raise RuntimeError("Bedrock unavailable")
    except Exception as e:
        # Heuristic fallback (deterministic, safe)
        error = str(e)
        if not vendors_data:
            ai_recommendation = "No vendor data available to analyze."
            model_used = "none"
        else:
            # Scoring: match_rate (40%), quality (30%), price (30%)
            def score_vendor(v):
                match_score = (v.get("match_rate", 0) or 0) * 0.4
                qual = v.get("quality", {})
                quality_score = ((qual.get("excellent", 0) * 3) + (qual.get("good", 0) * 2) + qual.get("fair", 0)) * 0.3
                prices = [vd.get("total_price", 0) for vd in vendors_data if vd.get("total_price", 0) > 0]
                max_price = max(prices) if prices else 0.0
                price_score = (1 - (v.get("total_price", 0) / max_price if max_price > 0 else 0)) * 30
                return match_score + quality_score + price_score

            vendors_sorted = sorted(vendors_data, key=score_vendor, reverse=True)
            best = vendors_sorted[0]
            recommendation = []
            recommendation.append("[Heuristic Analysis] AI unavailable or failed; using deterministic scoring.")
            recommendation.append(f"Recommended Vendor: {best.get('vendor_name')} (Match Rate: {best.get('match_rate')}%, Total Price: ${best.get('total_price')})")
            recommendation.append("Rankings:")
            for i, v in enumerate(vendors_sorted, 1):
                recommendation.append(f"{i}. {v.get('vendor_name')} - {v.get('match_rate')}% match - ${v.get('total_price')}")
            recommendation.append(f"Error: {error}")
            ai_recommendation = "\n".join(recommendation)
            model_used = "heuristic"

    return {
        "vendors": vendors_data,
        "ai_recommendation": ai_recommendation,
        "model_used": model_used,
        "error": error,
    }

def analyze_single_item(vendor_id: str, boq_sr_no: str) -> Dict[str, Any]:
    item_data = None
    matches = list(COMPARISON_DIR.glob(f"{vendor_id}_-_*_comparison.csv"))
    if not matches:
        raise FileNotFoundError("Vendor not found")
    matches = list(COMPARISON_DIR.glob(f"{vendor_id}_-_*_comparison.csv"))
    if not matches:
        raise FileNotFoundError("Vendor not found")
    csv_file = matches[0]
    with open(csv_file, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("BOQ Sr.No", "").strip() == boq_sr_no.strip():
                item_data = row
                break
    if not item_data:
        raise FileNotFoundError("Item not found")
    analysis = None
    model_used = None
    error = None
    try:
        if BEDROCK_AVAILABLE:
            profile = os.getenv("AWS_PROFILE", "thinktank")
            region = os.getenv("AWS_REGION", "us-east-1")
            model_id = os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-5-sonnet-20240620-v1:0")
            session = boto3.Session(profile_name=profile, region_name=region)
            bedrock = session.client("bedrock-runtime")

            prompt = f"""Analyze this specific BOQ item and vendor quotation match:\n\nBOQ: {json.dumps({
                'sr_no': item_data.get('BOQ Sr.No'),
                'description': item_data.get('BOQ Description'),
                'qty': item_data.get('BOQ Qty'),
                'uom': item_data.get('BOQ UOM')
            }, indent=2)}\n\nVendor Quote: {json.dumps({
                'description': item_data.get('Vendor Description'),
                'qty': item_data.get('Vendor Qty'),
                'uom': item_data.get('Vendor UOM'),
                'unit_price': item_data.get('Vendor Unit Price'),
                'total_price': item_data.get('Vendor Total Price')
            }, indent=2)}\n\nMatch Status: {item_data.get('Match Status')}\nMatch Confidence: {item_data.get('Match Confidence')}\nIssues: {item_data.get('Issues')}\n\nProvide: 1) Is this a correct match? 2) Key observations. 3) Discrepancies. 4) Pricing assessment. 5) Recommendation. Be concise."""

            payload = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 800,
                "temperature": 0.2,
                "messages": [{"role": "user", "content": prompt}],
            }
            response = bedrock.invoke_model(modelId=model_id, body=json.dumps(payload))
            result = json.loads(response["body"].read())
            ai_text = None
            try:
                content = result.get("content") if isinstance(result, dict) else None
                if content and isinstance(content, list) and len(content) > 0:
                    first = content[0]
                    ai_text = first.get("text") if isinstance(first, dict) else None
                if not ai_text:
                    ai_text = str(result)
            except Exception:
                ai_text = str(result)

            analysis = ai_text
            model_used = model_id
        else:
            raise RuntimeError("Bedrock unavailable")
    except Exception as e:
        error = str(e)
        # Simple heuristic analysis fallback
        observations = []
        ms = item_data.get('Match Status', '')
        conf = item_data.get('Match Confidence', '')
        vendor_qty = item_data.get('Vendor Qty')
        boq_qty = item_data.get('BOQ Qty')
        if ms:
            observations.append(f"Match Status: {ms}")
        if conf:
            observations.append(f"Match Confidence: {conf}")
        if vendor_qty and boq_qty and vendor_qty != boq_qty:
            observations.append(f"Quantity mismatch: BOQ {boq_qty} vs Vendor {vendor_qty}")
        issues = item_data.get('Issues')
        if issues:
            observations.append(f"Issues: {issues}")
        observations.append("AI analysis unavailable; this is a heuristic summary.")
        analysis = "\n".join(observations)
        model_used = "heuristic"

    return {"item": item_data, "analysis": analysis, "model_used": model_used, "error": error}


# ============================================================================
# Session Management Functions
# ============================================================================

def get_session_info(session_id: str) -> Dict[str, Any]:
    """
    Get detailed information about a session.
    
    Args:
        session_id: Session ID
    
    Returns:
        Dict with session metadata
    """
    session_dir = get_session_out_dir(session_id)
    
    if not session_dir.exists():
        raise FileNotFoundError(f"Session {session_id} not found")
    
    tracker_path = get_session_tracker_path(session_id)
    
    if not tracker_path.exists():
        return {
            "session_id": session_id,
            "exists": True,
            "has_tracker": False,
            "path": str(session_dir)
        }
    
    tracker = _load_uploaded_files_tracker(session_id)
    
    # Calculate statistics
    enquiry_file = tracker.get("enquiry_excel")
    vendors = tracker.get("vendors", {})
    vendor_count = len(vendors)
    total_vendor_files = sum(len(v.get("files", [])) for v in vendors.values() if isinstance(v, dict))
    
    # Extract enquiry filename (handle both dict and string formats)
    if isinstance(enquiry_file, dict):
        enquiry_filename = enquiry_file.get("filename")
    elif isinstance(enquiry_file, str):
        enquiry_filename = enquiry_file
    else:
        enquiry_filename = None
    
    # Check for outputs
    outputs_exist = {
        "inquiry_csv": (session_dir / "inquiry_csv" / "FINAL.csv").exists(),
        "textract": (session_dir / "textract").exists() and any((session_dir / "textract").iterdir()) if (session_dir / "textract").exists() else False,
        "comparisons": (session_dir / "vendor_comparisons").exists() and any((session_dir / "vendor_comparisons").iterdir()) if (session_dir / "vendor_comparisons").exists() else False
    }
    
    return {
        "session_id": session_id,
        "path": str(session_dir),
        "exists": True,
        "has_tracker": True,
        "created_at": tracker.get("created_at"),
        "last_updated": tracker.get("last_updated"),
        "status": tracker.get("status", "unknown"),
        "enquiry_excel": enquiry_filename,
        "vendor_count": vendor_count,
        "total_vendor_files": total_vendor_files,
        "outputs": outputs_exist,
        "workflow_runs": tracker.get("workflow_runs", [])
    }


def list_all_sessions() -> List[Dict[str, Any]]:
    """
    List all available sessions.
    
    Returns:
        List of session info dicts
    """
    sessions_dir = OUT_DIR / "sessions"
    
    if not sessions_dir.exists():
        return []
    
    sessions = []
    for session_dir in sessions_dir.iterdir():
        if session_dir.is_dir():
            try:
                session_info = get_session_info(session_dir.name)
                sessions.append(session_info)
            except Exception as e:
                print(f"Warning: Could not load session {session_dir.name}: {e}")
                continue
    
    # Sort by creation time (newest first), handle None values
    sessions.sort(key=lambda x: x.get("created_at") or "", reverse=True)
    return sessions


def delete_session(session_id: str) -> bool:
    """
    Delete a session and all its data.
    
    Args:
        session_id: Session ID to delete
    
    Returns:
        True if deleted successfully
    
    Raises:
        FileNotFoundError: If session doesn't exist
    """
    session_dir = get_session_out_dir(session_id)
    
    if not session_dir.exists():
        raise FileNotFoundError(f"Session {session_id} not found")
    
    # Safety check - ensure we're deleting from sessions directory
    if "sessions" not in str(session_dir):
        raise ValueError(f"Invalid session directory: {session_dir}")
    
    shutil.rmtree(session_dir)
    print(f"🗑️  Deleted session: {session_id}")
    return True


def archive_session(session_id: str, archive_dir: Optional[Path] = None) -> Path:
    """
    Archive a session to a zip file.
    
    Args:
        session_id: Session ID to archive
        archive_dir: Directory to save archive (default: OUT_DIR / "archives")
    
    Returns:
        Path to created archive file
    """
    session_dir = get_session_out_dir(session_id)
    
    if not session_dir.exists():
        raise FileNotFoundError(f"Session {session_id} not found")
    
    if archive_dir is None:
        archive_dir = OUT_DIR / "archives"
    
    ensure_dir(archive_dir)
    
    # Create archive filename with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_name = f"session_{session_id}_{timestamp}"
    archive_path = archive_dir / archive_name
    
    # Create zip archive
    shutil.make_archive(str(archive_path), 'zip', session_dir)
    
    final_path = archive_path.with_suffix('.zip')
    print(f"📦 Archived session {session_id} to: {final_path}")
    
    return final_path
