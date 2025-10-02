#!/usr/bin/env python3
"""Test pandas header parameter behavior"""

import pandas as pd
import csv
from pathlib import Path

csv_path = Path("/Users/sheiphanjoseph/Desktop/Developer/al_shirawi_orc_poc/out/textract_csv/Response_5_-_Bonn_Group__142 ASEMCO.tables.csv")

print("=" * 80)
print("Comparing csv.reader line numbers vs pandas header numbers")
print("=" * 80)

# Find header using csv.reader
with open(csv_path, 'r', encoding='utf-8') as f:
    csv_reader = csv.reader(f)
    for line_num, row in enumerate(csv_reader):
        if line_num >= 12 and line_num <= 15:
            print(f"csv.reader line {line_num}: {row[:3]}...")
            
        cells = [c.strip().lower() for c in row if c.strip()]
        has_desc = any('description' in c or 'desc' in c for c in cells)
        price_indicators = sum([
            any('price' in c for c in cells),
            any('qty' in c or 'quantity' in c for c in cells),
            any('unit' == c or 'uom' in c for c in cells)
        ])
        avg_cell_length = sum(len(c) for c in cells) / len(cells) if cells else 100
        cells_short = avg_cell_length < 30
        
        if len(cells) >= 3 and has_desc and price_indicators >= 2 and cells_short:
            csv_header_line = line_num
            print(f"\n✓ csv.reader found header at line_num={csv_header_line}")
            break

# Test different pandas header values
print("\n" + "=" * 80)
print("Testing pandas with different header values:")
print("=" * 80)

for h in [12, 13, 14]:
    print(f"\nTrying header={h}:")
    df = pd.read_csv(csv_path, header=h, on_bad_lines='skip', encoding='utf-8', engine='python')
    print(f"  Columns: {df.columns.tolist()[:3]}...")
    print(f"  First row values: {df.iloc[0].tolist()[:3] if len(df) > 0 else 'N/A'}...")


