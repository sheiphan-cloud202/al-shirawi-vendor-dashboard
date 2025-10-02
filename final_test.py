#!/usr/bin/env python3
"""Final test - verify header detection and pandas reading work correctly"""

import pandas as pd
import csv
from pathlib import Path

csv_path = Path("/Users/sheiphanjoseph/Desktop/Developer/al_shirawi_orc_poc/out/textract_csv/Response_5_-_Bonn_Group__142 ASEMCO.tables.csv")

print("=" * 80)
print("FINAL VERIFICATION TEST")
print("=" * 80)

# Step 1: Find header using our algorithm
print("\n[Step 1] Finding header with csv.reader...")
header_row = None

with open(csv_path, 'r', encoding='utf-8') as f:
    csv_reader = csv.reader(f)
    for line_num, row in enumerate(csv_reader):
        if line_num > 50:
            break
        
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
            header_row = line_num
            print(f"✓ Found header at line {line_num} (0-indexed)")
            print(f"  Header cells: {cells}")
            break

if header_row is None:
    print("❌ ERROR: No header found!")
    exit(1)

# Step 2: Read with pandas using detected header
print(f"\n[Step 2] Reading CSV with pandas.read_csv(header={header_row})...")

# Try WITHOUT on_bad_lines to avoid skipping behavior
try:
    df = pd.read_csv(csv_path, header=header_row, encoding='utf-8', engine='python')
    print(f"✓ Success!")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {df.columns.tolist()}")
    
    # Verify columns are correct
    expected_cols = ['SL.NO', 'BRAND', 'DESCRIPTION', 'UNIT', 'QTY', 'PRICE', 'TOTAL PRICE']
    if df.columns.tolist() == expected_cols:
        print(f"  ✓✓✓ COLUMNS ARE CORRECT!")
    else:
        print(f"  ❌ Columns don't match expected: {expected_cols}")
    
    # Step 3: Check data rows
    print(f"\n[Step 3] Checking data rows...")
    print(f"  First row:")
    print(f"    SL.NO: {df.iloc[0]['SL.NO']}")
    print(f"    DESCRIPTION: {df.iloc[0]['DESCRIPTION'][:50]}...")
    print(f"    QTY: {df.iloc[0]['QTY']}")
    print(f"    PRICE: {df.iloc[0]['PRICE']}")
    
    # Check if first row is the expected data
    if '900 MM CABLE TRAY' in str(df.iloc[0]['DESCRIPTION']):
        print(f"  ✓✓✓ FIRST DATA ROW IS CORRECT!")
    else:
        print(f"  ❌ First row data doesn't match expected")
        
except Exception as e:
    print(f"❌ Error: {e}")
    
print("\n" + "=" * 80)
print("TEST COMPLETE")
print("=" * 80)


