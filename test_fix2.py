#!/usr/bin/env python3
"""Test the stricter header detection"""

import csv
from pathlib import Path

csv_path = Path("/Users/sheiphanjoseph/Desktop/Developer/al_shirawi_orc_poc/out/textract_csv/Response_5_-_Bonn_Group__142 ASEMCO.tables.csv")

print("Testing STRICTER header detection...")
print("=" * 80)

header_row = None

with open(csv_path, 'r', encoding='utf-8') as f:
    csv_reader = csv.reader(f)
    for line_num, row in enumerate(csv_reader):
        if line_num > 20:
            break
        
        # Clean and lowercase cells
        cells = [c.strip().lower() for c in row if c.strip()]
        
        # Check if this row looks like a header (more strict criteria)
        # 1. Must have description column
        has_desc = any('description' in c or 'desc' in c for c in cells)
        
        # 2. Must have at least 2 of: qty, unit, price
        price_indicators = sum([
            any('price' in c for c in cells),
            any('qty' in c or 'quantity' in c for c in cells),
            any('unit' == c or 'uom' in c for c in cells)
        ])
        
        # 3. Cells should be short (headers are typically 1-4 words, not long sentences)
        avg_cell_length = sum(len(c) for c in cells) / len(cells) if cells else 100
        cells_short = avg_cell_length < 30  # Reject lines with very long text
        
        print(f"Line {line_num}: cells={len(cells)}, desc={has_desc}, indicators={price_indicators}, short={cells_short}, avg_len={avg_cell_length:.1f}")
        if len(cells) > 0 and len(cells) <= 7:
            print(f"  -> {cells}")
        
        # Must meet all criteria
        if len(cells) >= 3 and has_desc and price_indicators >= 2 and cells_short:
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



