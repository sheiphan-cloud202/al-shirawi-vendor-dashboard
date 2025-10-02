import json
import csv
from pathlib import Path
from typing import Dict, List


def load_tables(json_path: Path) -> List[Dict]:
    data = json.loads(json_path.read_text())
    return data


def tables_to_csv_rows(tables: List[Dict]) -> List[List[str]]:
    rows: List[List[str]] = []
    for table in tables:
        cells = table.get("cells", [])
        # Build sparse grid
        max_row = 0
        max_col = 0
        for c in cells:
            max_row = max(max_row, int(c.get("row", 0)))
            max_col = max(max_col, int(c.get("col", 0)))
        grid: List[List[str]] = [["" for _ in range(max_col)] for _ in range(max_row)]
        for c in cells:
            r = int(c.get("row", 0)) - 1
            k = int(c.get("col", 0)) - 1
            if r >= 0 and k >= 0:
                text = c.get("text", "") or ""
                grid[r][k] = text
        # Append a blank line between tables
        if rows:
            rows.append([])
        rows.extend(grid)
    return rows


def write_csv(rows: List[List[str]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for r in rows:
            w.writerow(r)


def main() -> int:
    in_dir = Path("/Users/sheiphanjoseph/Desktop/Developer/al_shirawi_orc_poc/out/textract")
    out_dir = Path("/Users/sheiphanjoseph/Desktop/Developer/al_shirawi_orc_poc/out/textract_csv")
    json_files = sorted(in_dir.glob("*.tables.json"))
    if not json_files:
        print("No Textract table JSON files found.")
        return 0
    for j in json_files:
        tables = load_tables(j)
        rows = tables_to_csv_rows(tables)
        out_csv = out_dir / (j.stem + ".csv")
        write_csv(rows, out_csv)
        print(f"Wrote {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



