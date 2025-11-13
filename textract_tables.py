import json
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import boto3

from index_responses import build_index

# Import fallback PDF processing
try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False


@dataclass
class TableCell:
    row: int
    col: int
    text: str


@dataclass
class Table:
    page: int
    cells: List[TableCell]


def _load_uploaded_files_tracker(tracker_path: Path) -> Dict[str, List[str]]:
    """
    Load the tracker of uploaded files (response_number -> list of filenames).
    
    Handles both NEW nested format and OLD flat format:
    - NEW: {"vendors": {"21": {"files": [{"filename": "x.pdf", ...}]}}}
    - OLD: {"21": ["file1.pdf", "file2.pdf"]}
    
    Returns: {"21": ["file1.pdf", "file2.pdf"], ...}
    """
    if not tracker_path.exists():
        return {}
    
    try:
        tracker_data = json.loads(tracker_path.read_text())
        result = {}
        
        # Handle NEW nested format (from vendor_logic.py)
        vendors = tracker_data.get("vendors", {})
        if isinstance(vendors, dict):
            for response_key, response_data in vendors.items():
                if isinstance(response_data, dict):
                    files = response_data.get("files", [])
                    # Extract filenames from file info dicts
                    filenames = []
                    for f in files:
                        if isinstance(f, dict):
                            filenames.append(f.get("filename"))
                        elif isinstance(f, str):
                            filenames.append(f)
                    if filenames:
                        result[response_key] = filenames
        
        # Handle OLD flat format (backward compatibility)
        # Only add if not already in result from new format
        for key, value in tracker_data.items():
            if key != "vendors" and key != "enquiry_excel" and key != "last_updated":
                if isinstance(value, list) and key not in result:
                    result[key] = value
                elif isinstance(value, dict) and "files" in value and key not in result:
                    # Handle intermediate format
                    files = value.get("files", [])
                    filenames = [f if isinstance(f, str) else f.get("filename") for f in files]
                    result[key] = filenames
        
        # Debug logging
        if result:
            print(f"  📋 Loaded tracker from {tracker_path.name}: Found {len(result)} response(s) with files")
            for response_key, filenames in result.items():
                print(f"     Response {response_key}: {len(filenames)} file(s)")
        else:
            print(f"  ⚠️  Tracker {tracker_path.name} is empty or has no vendor files")
        
        return result
    except Exception as e:
        print(f"⚠️  Error loading tracker {tracker_path}: {e}")
        return {}


def list_vendor_pdfs(data_dir: Path, tracker_path: Optional[Path] = None) -> List[Tuple[str, Path]]:
    """
    List vendor PDFs, optionally filtered to only uploaded files.
    
    Args:
        data_dir: Data directory containing Response folders
        tracker_path: Optional path to uploaded files tracker JSON.
                     If provided, only returns PDFs that were uploaded via API.
    """
    pdfs: List[Tuple[str, Path]] = []
    
    # Load tracker if provided
    uploaded_files = {}
    if tracker_path:
        uploaded_files = _load_uploaded_files_tracker(tracker_path)
    
    # Check if this is a session-based directory structure (Response_N folders)
    # Session structure: uploads/vendors/Response_21/file.pdf
    # Legacy structure: Response 3 - IKKT Attachment/file.pdf (with Response 3 - IKKT.txt)
    is_session_structure = False
    if data_dir.exists():
        # Check for Response_N pattern (session structure)
        session_folders = [d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("Response_")]
        if session_folders:
            is_session_structure = True
    
    if is_session_structure:
        # Session-based structure: directly enumerate Response_N folders
        print(f"  📂 Using session directory structure: {data_dir}")
        for response_dir in data_dir.iterdir():
            if not response_dir.is_dir():
                continue
            
            # Extract response number from folder name (e.g., "Response_21" -> "21")
            if not response_dir.name.startswith("Response_"):
                continue
            
            try:
                response_num = response_dir.name.split("_")[1]
            except IndexError:
                continue
            
            # If tracker is provided, only process tracked files
            if tracker_path:
                tracked_filenames = uploaded_files.get(response_num, [])
                if not tracked_filenames:
                    print(f"     ⚠️  No tracked files for Response {response_num}, skipping")
                    continue
                
                # Only include PDFs that match tracked filenames
                for filename in tracked_filenames:
                    pdf_path = response_dir / filename
                    if pdf_path.exists() and pdf_path.is_file() and pdf_path.suffix.lower() == ".pdf":
                        key = f"Response {response_num} - Uploaded"
                        pdfs.append((key, pdf_path))
                        print(f"     ✓ Found: {filename}")
                    else:
                        print(f"     ⚠️  Tracked file not found: {filename} at {pdf_path}")
            else:
                # No tracker - return all PDFs in this response folder
                for pdf_file in response_dir.iterdir():
                    if pdf_file.is_file() and pdf_file.suffix.lower() == ".pdf":
                        key = f"Response {response_num} - Uploaded"
                        pdfs.append((key, pdf_file))
    else:
        # Legacy structure: use build_index
        print("  📂 Using legacy directory structure with build_index")
        index = build_index(data_dir)
        
        for key, info in index.items():
            if info.attachments_dir:
                base = Path(info.attachments_dir)
                
                # Extract response number from key (e.g., "Response 3 - IKKT" -> "3")
                response_num = None
                if info.response_number and info.response_number > 0:
                    response_num = str(info.response_number)
                
                # If tracker is provided, only process tracked files
                if tracker_path and response_num:
                    tracked_filenames = uploaded_files.get(response_num, [])
                    if not tracked_filenames:
                        # No tracked files for this response number, skip
                        continue
                    
                    # Only include PDFs that match tracked filenames
                    for filename in tracked_filenames:
                        pdf_path = base / filename
                        if pdf_path.exists() and pdf_path.is_file() and pdf_path.suffix.lower() == ".pdf":
                            pdfs.append((key, pdf_path))
                else:
                    # No tracker provided - return all PDFs (backward compatibility)
                    for p in base.iterdir():
                        if p.is_file() and p.suffix.lower() == ".pdf":
                            pdfs.append((key, p))
    
    return pdfs


def start_document_analysis_local(textract, file_path: Path, feature_types: List[str]) -> Dict:
    # Keep this helper for images; PDFs need async/S3. It will likely fail on PDFs.
    with open(file_path, "rb") as f:
        bytes_data = f.read()
    return textract.analyze_document(Document={"Bytes": bytes_data}, FeatureTypes=feature_types)


def extract_tables_from_textract(resp: Dict) -> List[Table]:
    blocks = resp.get("Blocks", [])
    # Map block ids
    by_id: Dict[str, Dict] = {b["Id"]: b for b in blocks if "Id" in b}
    tables: List[Table] = []
    # Find tables
    for b in blocks:
        if b.get("BlockType") == "TABLE":
            page = b.get("Page", 1)
            # Collect cell blocks under this table
            cell_ids: List[str] = []
            for rel in b.get("Relationships", []) or []:
                if rel.get("Type") == "CHILD":
                    cell_ids.extend(rel.get("Ids", []))
            cells: List[TableCell] = []
            for cid in cell_ids:
                cb = by_id.get(cid)
                if not cb or cb.get("BlockType") != "CELL":
                    continue
                row = cb.get("RowIndex", 0)
                col = cb.get("ColumnIndex", 0)
                # Extract text from child WORD blocks
                text_parts: List[str] = []
                for rel in cb.get("Relationships", []) or []:
                    if rel.get("Type") == "CHILD":
                        for wid in rel.get("Ids", []):
                            wb = by_id.get(wid)
                            if wb and wb.get("BlockType") in ("WORD", "SELECTION_ELEMENT"):
                                if wb.get("BlockType") == "WORD":
                                    text_parts.append(wb.get("Text", ""))
                                elif wb.get("SelectionStatus") == "SELECTED":
                                    text_parts.append("[X]")
                cells.append(TableCell(row=row, col=col, text=" ".join(t for t in text_parts if t)))
            tables.append(Table(page=page, cells=cells))
    return tables


def extract_tables_local_pdfplumber(pdf_path: Path) -> List[Table]:
    """
    Fallback: Extract tables from PDF using pdfplumber (no AWS required).
    This is used when TEXTRACT_S3_BUCKET is not set.
    """
    if not PDFPLUMBER_AVAILABLE:
        raise ImportError("pdfplumber not installed. Run: uv pip install pdfplumber")
    
    tables: List[Table] = []
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                # Extract tables from this page
                page_tables = page.extract_tables()
                
                if not page_tables:
                    continue
                
                for table_data in page_tables:
                    if not table_data or len(table_data) < 2:  # Need at least header + 1 row
                        continue
                    
                    cells: List[TableCell] = []
                    
                    # Convert table_data (List[List[str]]) to TableCell format
                    for row_idx, row in enumerate(table_data, start=1):
                        if not row:
                            continue
                        
                        for col_idx, cell_text in enumerate(row, start=1):
                            if cell_text is None:
                                cell_text = ""
                            
                            # Clean up text
                            text = str(cell_text).strip()
                            
                            cells.append(TableCell(
                                row=row_idx,
                                col=col_idx,
                                text=text
                            ))
                    
                    if cells:
                        tables.append(Table(page=page_num, cells=cells))
        
        return tables
    
    except Exception as e:
        print(f"      ⚠️  pdfplumber extraction error: {e}")
        return []


def _session_clients():
    profile = os.getenv("AWS_PROFILE", "thinktank")
    region = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION", "us-east-1")
    session = boto3.Session(profile_name=profile, region_name=region)
    return session.client("textract"), session.client("s3")


def analyze_pdf_tables_async(file_path: Path, bucket: str, prefix: str = "") -> List[Table]:
    textract, s3 = _session_clients()
    # Upload
    key = "/".join([p for p in [prefix.strip("/"), f"vendor_pdfs/{file_path.name}"] if p])
    extra_args = None
    acl = os.getenv("S3_UPLOAD_ACL")
    if acl:
        extra_args = {"ACL": acl}
    s3.upload_file(str(file_path), bucket, key, ExtraArgs=extra_args)
    # Start async analysis
    start = textract.start_document_analysis(
        DocumentLocation={"S3Object": {"Bucket": bucket, "Name": key}},
        FeatureTypes=["TABLES"],
    )
    job_id = start["JobId"]
    # Poll
    while True:
        desc = textract.get_document_analysis(JobId=job_id)
        status = desc.get("JobStatus")
        if status in ("SUCCEEDED", "FAILED", "PARTIAL_SUCCESS"):
            break
        time.sleep(2)
    if status != "SUCCEEDED":
        raise RuntimeError(f"Textract job failed: {status}")
    # Fetch all pages
    blocks: List[Dict] = []
    next_token: Optional[str] = None
    while True:
        if next_token:
            page = textract.get_document_analysis(JobId=job_id, NextToken=next_token)
        else:
            page = desc
        blocks.extend(page.get("Blocks", []))
        next_token = page.get("NextToken")
        if not next_token:
            break
    return extract_tables_from_textract({"Blocks": blocks})


def save_tables_json(out_dir: Path, vendor_key: str, pdf_path: Path, tables: List[Table]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_name = f"{vendor_key.replace(' ', '_').replace('/', '_')}__{pdf_path.stem}.tables.json"
    out_path = out_dir / out_name
    data = [asdict(t) for t in tables]
    out_path.write_text(json.dumps(data, indent=2))
    return out_path


def main() -> int:
    data_dir = Path("/Users/sheiphanjoseph/Desktop/Developer/al_shirawi_orc_poc/data")
    out_dir = Path("/Users/sheiphanjoseph/Desktop/Developer/al_shirawi_orc_poc/out/textract")
    pdfs = list_vendor_pdfs(data_dir)
    if not pdfs:
        print("No PDFs found.")
        return 0
    print(f"Found {len(pdfs)} PDFs.")
    bucket = os.getenv("TEXTRACT_S3_BUCKET")
    prefix = os.getenv("TEXTRACT_S3_PREFIX", "")
    if not bucket:
        print("TEXTRACT_S3_BUCKET not set; skipping AWS calls. Set env var to enable async PDF analysis.")
        for vendor_key, pdf_path in pdfs:
            print(f"Would analyze (set TEXTRACT_S3_BUCKET first): {vendor_key} -> {pdf_path}")
        return 0
    for vendor_key, pdf_path in pdfs:
        print(f"Analyzing (async): {vendor_key} -> {pdf_path}")
        tables = analyze_pdf_tables_async(pdf_path, bucket=bucket, prefix=prefix)
        out_path = save_tables_json(out_dir, vendor_key, pdf_path, tables)
        print(f"Saved tables: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


