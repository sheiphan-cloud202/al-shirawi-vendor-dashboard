import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


def normalize_column_name(name: str) -> str:
    """
    Normalize a column header to a consistent snake_case key for matching.
    """
    if name is None:
        return ""
    lowered = str(name).strip().lower()
    # Replace separators with spaces, collapse multiple spaces
    for ch in ["\n", "\t", "/", "-", "|", ":", ";", ",", "(", ")", "[", "]"]:
        lowered = lowered.replace(ch, " ")
    lowered = " ".join(lowered.split())
    # Common aliases mapping
    aliases = {
        "sr no": "sr_no",
        "s no": "sr_no",
        "item no": "item_no",
        "item#": "item_no",
        "item": "item",
        "description": "description",
        "desc": "description",
        "specification": "specification",
        "spec": "specification",
        "uom": "uom",
        "unit": "uom",
        "unit of measure": "uom",
        "qty": "qty",
        "quantity": "qty",
        "size": "size",
        "width": "width",
        "height": "height",
        "thickness": "thickness",
        "material": "material",
        "remark": "remarks",
        "remarks": "remarks",
        "make": "make",
        "brand": "make",
        "model": "model",
        "type": "type",
        "rate": "rate",
        "unit rate": "rate",
        "price": "rate",
        "amount": "amount",
        "total": "amount",
        "boq ref": "boq_ref",
        "boq reference": "boq_ref",
        "code": "code",
        "part no": "part_no",
        "part number": "part_no",
        "manufacturer": "make",
    }
    if lowered in aliases:
        return aliases[lowered]
    return lowered.replace(" ", "_")


def find_header_row(df_sample: pd.DataFrame, max_rows_to_scan: int = 50) -> Optional[int]:
    """
    Heuristic to find the row index that likely contains headers:
    - Prefer the first row that yields at least 3 non-empty string-like values
    - Avoid rows that look like totals or notes
    """
    keywords = {
        "item", "description", "desc", "uom", "unit", "qty", "quantity",
        "rate", "amount", "total", "remarks", "remark", "spec", "specification",
        "size", "width", "height", "thickness", "material", "make", "brand",
        "code", "part", "boq", "ref"
    }
    best_idx: Optional[int] = None
    best_score = -1
    limit = min(max_rows_to_scan, len(df_sample))
    for i in range(limit):
        row = df_sample.iloc[i]
        cells = [str(v).strip().lower() for v in row.tolist() if pd.notna(v) and str(v).strip() not in ("", "nan")]
        if not cells:
            continue
        # Score based on presence of header keywords and textiness
        token_score = 0
        for c in cells:
            parts = c.replace("/", " ").replace("-", " ").split()
            if any(p in keywords for p in parts):
                token_score += 2
            if any(ch.isalpha() for ch in c):
                token_score += 1
        # Penalize rows that look like titles or forms
        if len(cells) <= 2:
            token_score -= 2
        if any(c.startswith(prefix) for prefix in ("total", "grand total", "note", "remarks")):
            token_score -= 3
        if token_score > best_score:
            best_score = token_score
            best_idx = i
    return best_idx


def analyze_boq_excel(xlsx_path: Path) -> Dict[str, Dict[str, object]]:
    """
    Read the BOQ Excel and infer headers and normalized columns for each sheet.
    Returns a dict keyed by sheet name with:
      - header_row: int | None
      - raw_columns: List[str]
      - normalized_columns: List[str]
      - sample_rows: List[Dict[str, object]] (first few rows)
    """
    xl = pd.ExcelFile(xlsx_path)
    report: Dict[str, Dict[str, object]] = {}
    for sheet in xl.sheet_names:
        df_raw = xl.parse(sheet_name=sheet, header=None)
        header_idx = find_header_row(df_raw)
        if header_idx is None:
            # Fallback: use first row as header
            header_idx = 0
        # Build DataFrame with inferred header
        header_values = df_raw.iloc[header_idx].tolist()
        columns = [str(c) if pd.notna(c) else "" for c in header_values]
        df = df_raw.iloc[header_idx + 1 :].copy()
        df.columns = columns
        # Drop fully empty columns
        df = df.dropna(axis=1, how="all")
        raw_columns = [c for c in df.columns]
        normalized_columns = [normalize_column_name(c) for c in raw_columns]
        # Provide a small sample with only top 5 non-empty rows
        df_sample = df.dropna(how="all").head(5)
        sample_rows = df_sample.to_dict(orient="records")
        report[sheet] = {
            "header_row": header_idx,
            "raw_columns": raw_columns,
            "normalized_columns": normalized_columns,
            "sample_rows": sample_rows,
        }
    return report


def main(argv: List[str]) -> int:
    if len(argv) >= 2:
        xlsx = Path(argv[1])
    else:
        xlsx = Path("/Users/sheiphanjoseph/Desktop/Developer/al_shirawi_orc_poc/data/Enquiry Attachment/J1684 Cable Tray & Trunking BOQ.xlsx")
    if not xlsx.exists():
        print(f"BOQ Excel not found at: {xlsx}")
        return 1
    report = analyze_boq_excel(xlsx)
    # Pretty print concise summary
    print("Sheets found:")
    for sheet, info in report.items():
        print(f"- {sheet} (header at row {info['header_row']})")
        print("  Raw columns:")
        print("   ", ", ".join(info["raw_columns"]))
        print("  Normalized columns:")
        print("   ", ", ".join(info["normalized_columns"]))
        if info["sample_rows"]:
            print("  Sample row:")
            print("   ", info["sample_rows"][0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))


