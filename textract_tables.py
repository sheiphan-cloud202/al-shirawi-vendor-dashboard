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


def list_vendor_pdfs(data_dir: Path) -> List[Tuple[str, Path]]:
    index = build_index(data_dir)
    pdfs: List[Tuple[str, Path]] = []
    for key, info in index.items():
        if info.attachments_dir:
            base = Path(info.attachments_dir)
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


