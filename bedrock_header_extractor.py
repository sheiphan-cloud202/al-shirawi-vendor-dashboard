import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import boto3
import pandas as pd

try:
    from textract_tables import analyze_boq_excel, normalize_column_name  # type: ignore
except Exception:
    # Fallback minimal implementations to avoid hard dependency during header extraction
    def normalize_column_name(name: str) -> str:
        if name is None:
            return ""
        lowered = str(name).strip().lower()
        for ch in ["\n", "\t", "/", "-", "|", ":", ";", ",", "(", ")", "[", "]"]:
            lowered = lowered.replace(ch, " ")
        lowered = " ".join(lowered.split())
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

    def _find_header_row(df_sample: pd.DataFrame, max_rows_to_scan: int = 50) -> Optional[int]:
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
            token_score = 0
            for c in cells:
                parts = c.replace("/", " ").replace("-", " ").split()
                if any(p in keywords for p in parts):
                    token_score += 2
                if any(ch.isalpha() for ch in c):
                    token_score += 1
            if len(cells) <= 2:
                token_score -= 2
            if any(c.startswith(prefix) for prefix in ("total", "grand total", "note", "remarks")):
                token_score -= 3
            if token_score > best_score:
                best_score = token_score
                best_idx = i
        return best_idx

    def analyze_boq_excel(xlsx_path: Path) -> Dict[str, Dict[str, object]]:
        xl = pd.ExcelFile(xlsx_path)
        report: Dict[str, Dict[str, object]] = {}
        for sheet in xl.sheet_names:
            df_raw = xl.parse(sheet_name=sheet, header=None)
            header_idx = _find_header_row(df_raw)
            if header_idx is None:
                header_idx = 0
            header_values = df_raw.iloc[header_idx].tolist()
            columns = [str(c) if pd.notna(c) else "" for c in header_values]
            df = df_raw.iloc[header_idx + 1 :].copy()
            df.columns = columns
            df = df.dropna(axis=1, how="all")
            raw_columns = [c for c in df.columns]
            normalized_columns = [normalize_column_name(c) for c in raw_columns]
            df_sample = df.dropna(how="all").head(5)
            sample_rows = df_sample.to_dict(orient="records")
            report[sheet] = {
                "header_row": header_idx,
                "raw_columns": raw_columns,
                "normalized_columns": normalized_columns,
                "sample_rows": sample_rows,
            }
        return report


@dataclass
class SheetHeaderDecision:
    sheet_name: str
    header_row_index: int
    raw_columns: List[str]
    normalized_columns: List[str]


def find_inquiry_excel(default_root: Optional[Path] = None) -> Optional[Path]:
    base = default_root or Path("/Users/sheiphanjoseph/Desktop/Developer/al_shirawi_orc_poc/data/Enquiry Attachment")
    if not base.exists():
        return None
    candidates = sorted([p for p in base.iterdir() if p.is_file() and p.suffix.lower() in (".xlsx", ".xlsm", ".xls")])
    return candidates[0] if candidates else None


def load_sheet_preview(xlsx_path: Path, max_rows: int = 50, max_cols: int = 30, truncate: int = 80) -> Dict[str, List[List[str]]]:
    xl = pd.ExcelFile(xlsx_path)
    previews: Dict[str, List[List[str]]] = {}
    for sheet in xl.sheet_names:
        df = xl.parse(sheet_name=sheet, header=None, nrows=max_rows)
        df = df.iloc[:, :max_cols]
        grid: List[List[str]] = []
        for _, row in df.iterrows():
            vals: List[str] = []
            for v in row.tolist():
                if pd.isna(v):
                    vals.append("")
                else:
                    s = str(v)
                    if len(s) > truncate:
                        s = s[:truncate] + "…"
                    vals.append(s)
            grid.append(vals)
        previews[sheet] = grid
    return previews


def _bedrock_client() -> Optional[object]:
    try:
        profile = os.getenv("AWS_PROFILE")
        region = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")
        session = boto3.Session(profile_name=profile, region_name=region)
        return session.client("bedrock-runtime")
    except Exception:
        return None


def _build_prompt_for_sheet(sheet_name: str, grid: List[List[str]]) -> str: 
    lines: List[str] = []
    lines.append("You are a data extraction assistant. Given a preview of an Excel sheet, identify the table header row.")
    lines.append("Return strictly JSON matching this schema: {\"header_row_index\": <int>, \"raw_columns\": [<str>...] } with 0-based row index within the preview provided.")
    lines.append("Use the first row that best represents column headers for the main BOQ table. Avoid title pages and totals.")
    lines.append("")
    lines.append(f"Sheet: {sheet_name}")
    lines.append("Preview (rows shown with row_index: [cell1 | cell2 | ...]):")
    for i, row in enumerate(grid):
        joined = " | ".join((c or "").replace("\n", " ") for c in row)
        lines.append(f"{i:02d}: [ {joined} ]")
    return "\n".join(lines)


def _invoke_bedrock_annotate_headers(client, model_id: str, prompt: str, max_tokens: int = 1024, temperature: float = 0.0) -> Optional[Dict]:
    try:
        if model_id.startswith("anthropic."):
            body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
            }
        else:
            body = {
                "inputText": prompt,
                "textGenerationConfig": {
                    "maxTokenCount": max_tokens,
                    "temperature": temperature,
                    "topP": 0.9,
                },
            }

        resp = client.invoke_model(modelId=model_id, body=json.dumps(body))
        payload = resp.get("body")
        if hasattr(payload, "read"):
            data = json.loads(payload.read())
        else:
            data = json.loads(payload)

        # Parse output for Anthropic or Titan
        if model_id.startswith("anthropic."):
            content = data.get("content", [])
            text = "".join([p.get("text", "") for p in content if p.get("type") == "text"]) or data.get("output_text") or ""
        else:
            text = data.get("results", [{}])[0].get("outputText", "")

        # Extract JSON from text (either raw or fenced)
        json_str = text.strip()
        if "```" in json_str:
            # Try to pull the first fenced block
            parts = json_str.split("```")
            if len(parts) >= 2:
                candidate = parts[1]
                # Remove possible language tag on first line
                lines = candidate.splitlines()
                if lines and ":" not in lines[0] and lines[0].strip().isalpha():
                    json_str = "\n".join(lines[1:])
                else:
                    json_str = candidate
        return json.loads(json_str)
    except Exception:
        return None


def decide_headers_with_bedrock(xlsx_path: Path) -> List[SheetHeaderDecision]:
    client = None if os.getenv("BEDROCK_DISABLE") == "1" else _bedrock_client()
    model_id = os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-5-sonnet-20240620-v1:0")

    decisions: List[SheetHeaderDecision] = []
    previews = load_sheet_preview(xlsx_path)
    for sheet, grid in previews.items():
        header_row_index: Optional[int] = None
        raw_columns: List[str] = []
        if client is not None:
            prompt = _build_prompt_for_sheet(sheet, grid)
            result = _invoke_bedrock_annotate_headers(client, model_id=model_id, prompt=prompt)
            if isinstance(result, dict) and isinstance(result.get("raw_columns"), list) and isinstance(result.get("header_row_index"), int):
                header_row_index = max(0, int(result["header_row_index"]))
                raw_columns = [str(c) for c in result["raw_columns"]]

        # Fallback or completion via heuristic
        if header_row_index is None or not raw_columns:
            heuristic = analyze_boq_excel(xlsx_path)
            info = heuristic.get(sheet)
            if info:
                header_row_index = int(info["header_row"]) if isinstance(info.get("header_row"), int) else 0
                raw_columns = [str(c) for c in info.get("raw_columns", [])]

        normalized_columns = [normalize_column_name(c) for c in raw_columns]
        decisions.append(
            SheetHeaderDecision(
                sheet_name=sheet,
                header_row_index=header_row_index if header_row_index is not None else 0,
                raw_columns=raw_columns,
                normalized_columns=normalized_columns,
            )
        )
    return decisions


def export_clean_sheet_csvs(xlsx_path: Path, decisions: List[SheetHeaderDecision], out_dir: Path) -> List[Path]:
    xl = pd.ExcelFile(xlsx_path)
    written: List[Path] = []
    for d in decisions:
        df_raw = xl.parse(sheet_name=d.sheet_name, header=None)
        header_values = df_raw.iloc[d.header_row_index].tolist()
        columns = [str(c) if pd.notna(c) else "" for c in header_values]
        df = df_raw.iloc[d.header_row_index + 1 :].copy()
        df.columns = columns
        df = df.dropna(axis=1, how="all")
        df = df.dropna(how="all")
        safe_sheet = d.sheet_name.replace("/", "_").replace("\\", "_").replace(" ", "_")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{safe_sheet}.csv"
        df.to_csv(out_path, index=False)
        written.append(out_path)
    return written


def write_headers_summary(decisions: List[SheetHeaderDecision], out_json: Path) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    data = [
        {
            "sheet": d.sheet_name,
            "header_row_index": d.header_row_index,
            "raw_columns": d.raw_columns,
            "normalized_columns": d.normalized_columns,
        }
        for d in decisions
    ]
    out_json.write_text(json.dumps(data, indent=2))


def main(argv: Optional[List[str]] = None) -> int:
    xlsx: Optional[Path]
    if argv and len(argv) >= 2:
        xlsx = Path(argv[1])
    else:
        xlsx = find_inquiry_excel()
    if not xlsx or not xlsx.exists():
        print("Inquiry Excel not found. Place the file under data/Enquiry Attachment/.")
        return 1
    print(f"Analyzing Excel: {xlsx}")
    decisions = decide_headers_with_bedrock(xlsx)
    out_dir = Path("/Users/sheiphanjoseph/Desktop/Developer/al_shirawi_orc_poc/out/inquiry_csv")
    written = export_clean_sheet_csvs(xlsx, decisions, out_dir)
    summary_path = Path("/Users/sheiphanjoseph/Desktop/Developer/al_shirawi_orc_poc/out/inquiry_headers.json")
    write_headers_summary(decisions, summary_path)
    print(f"Wrote {len(written)} sheet CSVs to {out_dir}")
    print(f"Wrote headers summary: {summary_path}")
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(main(sys.argv))


