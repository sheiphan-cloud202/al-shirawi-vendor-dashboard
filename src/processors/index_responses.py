"""Vendor Response Metadata Extraction & Indexing"""
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


EMAIL_REGEX = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")


@dataclass
class VendorResponse:
    response_number: int
    vendor_name: str
    response_file: str
    sender_email: Optional[str]
    all_emails_found: List[str]
    attachments_dir: Optional[str]
    attachments: List[str]


def extract_sender_email(text: str) -> Tuple[Optional[str], List[str]]:
    """
    Extract the sender email. Prefer email on a line starting with 'From:'.
    Fallback: first non-Al-Shirawi email found anywhere in the text.
    Returns (sender_email, all_emails_found_unique)
    """
    sender_email: Optional[str] = None
    lines = [ln.strip() for ln in text.splitlines()]
    for ln in lines:
        if ln.lower().startswith("from:"):
            match = EMAIL_REGEX.search(ln)
            if match:
                sender_email = match.group(0)
                break
    all_emails = EMAIL_REGEX.findall(text)
    # Deduplicate while preserving order
    seen: set = set()
    unique_emails: List[str] = []
    for e in all_emails:
        if e not in seen:
            seen.add(e)
            unique_emails.append(e)

    if sender_email is None:
        # Fallback to first non-Al-Shirawi address
        for e in unique_emails:
            if "alshirawi" not in e.lower():
                sender_email = e
                break

    return sender_email, unique_emails


def find_response_files(data_dir: Path) -> List[Path]:
    # Support: "Response N - Vendor.txt", allow multiple spaces/dashes/case-insensitive
    # glob is case-sensitive; we'll filter with regex afterward.
    candidates = list(data_dir.glob("Response*"))
    rx = re.compile(r"^Response\s*(\d+)\s*[-–—]\s*(.+)\.txt$", re.IGNORECASE)
    out: List[Path] = []
    for p in candidates:
        if rx.match(p.name):
            out.append(p)
    return sorted(out)


def parse_response_filename(path: Path) -> Tuple[Optional[int], str]:
    # Expected: "Response <num> - <Vendor>.txt" with flexible dashes/spaces
    name = path.name
    m = re.match(r"^Response\s*(\d+)\s*[-–—]\s*(.+)\.txt$", name, flags=re.IGNORECASE)
    if not m:
        return None, path.stem
    return int(m.group(1)), m.group(2).strip()


def find_attachments_folder(data_dir: Path, response_number: int) -> Optional[Path]:
    # Match both Attachment and Attachement typos, allow case/spacing
    # Search dirs starting with "Response <num>"
    candidates = [p for p in data_dir.iterdir() if p.is_dir() and p.name.lower().startswith(f"response {response_number} ")]
    if not candidates:
        return None
    # Prefer names that contain 'attachment' (any spelling) and start with exact pattern
    def score(p: Path) -> int:
        n = p.name.lower()
        s = 0
        if n.startswith(f"response {response_number} "):
            s += 2
        if "attachment" in n or "attachement" in n:
            s += 2
        if n.endswith("attachment") or n.endswith("attachement"):
            s += 1
        return s
    candidates.sort(key=score, reverse=True)
    return candidates[0]


def build_index(data_dir: Path) -> Dict[str, VendorResponse]:
    index: Dict[str, VendorResponse] = {}
    for resp_file in find_response_files(data_dir):
        resp_num, vendor_name = parse_response_filename(resp_file)
        try:
            text = resp_file.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            text = resp_file.read_text(errors="ignore")
        sender_email, all_emails = extract_sender_email(text)
        attachments_dir: Optional[Path] = None
        attachments: List[str] = []
        if resp_num is not None:
            attachments_dir = find_attachments_folder(data_dir, resp_num)
            if attachments_dir and attachments_dir.exists() and attachments_dir.is_dir():
                attachments = sorted([p.name for p in attachments_dir.iterdir() if p.is_file()])
        key = f"Response {resp_num if resp_num is not None else '?'} - {vendor_name}"
        index[key] = VendorResponse(
            response_number=resp_num if resp_num is not None else -1,
            vendor_name=vendor_name,
            response_file=str(resp_file),
            sender_email=sender_email,
            all_emails_found=all_emails,
            attachments_dir=str(attachments_dir) if attachments_dir else None,
            attachments=attachments,
        )
    return index


def main() -> int:
    from src.utils.constants import DATA_DIR
    
    if not DATA_DIR.exists():
        print(f"Data directory not found: {DATA_DIR}")
        return 1
    index = build_index(DATA_DIR)
    # Print pretty JSON for human readability
    out = {k: asdict(v) for k, v in index.items()}
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

