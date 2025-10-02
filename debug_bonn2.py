#!/usr/bin/env python3
"""Debug script - testing different pandas read approaches"""

import pandas as pd
from pathlib import Path

csv_path = Path("/Users/sheiphanjoseph/Desktop/Developer/al_shirawi_orc_poc/out/textract_csv/Response_5_-_Bonn_Group__142 ASEMCO.tables.csv")

print("=" * 80)
print("TESTING DIFFERENT PANDAS READ STRATEGIES")
print("=" * 80)

# Test 1: Read with header=None (what the code does now)
print("\n[Test 1] pd.read_csv with header=None, nrows=20")
print("-" * 80)
df1 = pd.read_csv(csv_path, nrows=20, header=None, on_bad_lines='skip', encoding='utf-8', engine='python')
print(f"Shape: {df1.shape}")
print(f"Columns: {df1.columns.tolist()}")
print(f"\nFirst 5 rows:")
print(df1.head())

# Test 2: Read entire file with header=None  
print("\n\n[Test 2] pd.read_csv with header=None (full file)")
print("-" * 80)
df2 = pd.read_csv(csv_path, header=None, on_bad_lines='skip', encoding='utf-8', engine='python')
print(f"Shape: {df2.shape}")
print(f"Columns: {df2.columns.tolist()}")

# Test 3: Manually inspect for header row
print("\n\n[Test 3] Manual header detection from header=None dataframe")
print("-" * 80)
for idx, row in df1.iterrows():
    # Get all cells in this row
    all_cells = row.tolist()
    cells = [str(cell).lower().strip() for cell in all_cells if pd.notna(cell) and str(cell).strip()]
    
    print(f"Row {idx} ({len(all_cells)} total, {len(cells)} non-empty):")
    print(f"  Cells: {cells}")
    
    # Check header criteria
    has_desc = any('desc' in c or 'item' in c for c in cells)
    has_qty_or_price = any(('qty' in c or 'quantity' in c or 'price' in c or 'unit' in c) for c in cells)
    
    if len(cells) >= 3 and has_desc and has_qty_or_price:
        print(f"  ✓ HEADER MATCH! desc={has_desc}, qty/price={has_qty_or_price}")
        print(f"  This is row index {idx}")
        break
else:
    print("\n❌ No header found")

# Test 4: Try reading with detected header row  
print("\n\n[Test 4] Reading with header=13 (if detected)")
print("-" * 80)
df4 = pd.read_csv(csv_path, header=13, on_bad_lines='skip', encoding='utf-8', engine='python')
print(f"Shape: {df4.shape}")
print(f"Columns: {df4.columns.tolist()}")
print(f"\nFirst 3 data rows:")
print(df4.head(3))


