#!/usr/bin/env python3
"""Test the fixed header detection"""

import csv
from pathlib import Path

csv_path = Path("/Users/sheiphanjoseph/Desktop/Developer/al_shirawi_orc_poc/out/textract_csv/Response_5_-_Bonn_Group__142 ASEMCO.tables.csv")

print("Testing fixed header detection...")
print("=" * 80)

header_row = None

with open(csv_path, 'r', encoding='utf-8') as f:
    csv_reader = csv.reader(f)
    for line_num, row in enumerate(csv_reader):
        if line_num > 50:
            break
        
        # Clean and lowercase cells
        cells = [c.strip().lower() for c in row if c.strip()]
        
        # Check if this row looks like a header
        has_desc = any('desc' in c or 'item' in c for c in cells)
        has_qty_or_price = any(('qty' in c or 'quantity' in c or 'price' in c or 'unit' in c) for c in cells)
        
        print(f"Line {line_num}: {len(cells)} cells, desc={has_desc}, qty/price={has_qty_or_price}")
        if len(cells) > 0:
            print(f"  -> {cells[:5]}...")
        
        # Must have at least 3 non-empty cells and match criteria
        if len(cells) >= 3 and has_desc and has_qty_or_price:
            header_row = line_num
            print(f"\n✓✓✓ HEADER FOUND at line {line_num}!")
            print(f"    Full header: {cells}")
            break

if header_row is None:
    print("\n❌ NO HEADER FOUND")
else:
    print(f"\n✓ Will use header row: {header_row}")
    
    # Test reading with this header
    import pandas as pd
    df = pd.read_csv(csv_path, header=header_row, on_bad_lines='skip', encoding='utf-8', engine='python')
    print(f"\n✓ DataFrame shape: {df.shape}")
    print(f"✓ Columns: {df.columns.tolist()}")
    print(f"\n✓ First 3 rows:")
    print(df.head(3))



