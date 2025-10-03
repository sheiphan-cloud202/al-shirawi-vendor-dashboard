#!/usr/bin/env python3
"""Debug script to understand Bonn Group CSV parsing issue"""

import pandas as pd
from pathlib import Path

csv_path = Path("/Users/sheiphanjoseph/Desktop/Developer/al_shirawi_orc_poc/out/textract_csv/Response_5_-_Bonn_Group__142 ASEMCO.tables.csv")

print("=" * 80)
print("DEBUGGING BONN GROUP CSV PARSING")
print("=" * 80)

# Step 1: Read first 20 rows to detect header
print("\n[Step 1] Preview first 20 rows (raw):")
print("-" * 80)
df_preview = pd.read_csv(csv_path, nrows=20, header=None, on_bad_lines='skip', encoding='utf-8', engine='python')

for idx, row in df_preview.iterrows():
    cells = [str(cell).lower().strip() for cell in row.tolist() if pd.notna(cell) and str(cell).strip()]
    print(f"Row {idx}: {cells[:7]}...")  # Show first 7 cells
    
    # Check header detection criteria
    has_desc = any('desc' in c or 'item' in c for c in cells)
    has_qty_or_price = any(('qty' in c or 'quantity' in c or 'price' in c or 'unit' in c) for c in cells)
    
    if len(cells) >= 3 and has_desc and has_qty_or_price:
        print(f"  ✓ HEADER DETECTED! has_desc={has_desc}, has_qty_or_price={has_qty_or_price}")
        header_row = idx
        break
else:
    header_row = None
    print("\n❌ NO HEADER FOUND IN FIRST 20 ROWS")

# Step 2: Read with detected header
print(f"\n[Step 2] Reading CSV with header_row={header_row}")
print("-" * 80)

if header_row is not None:
    df = pd.read_csv(csv_path, header=header_row, on_bad_lines='skip', encoding='utf-8', engine='python')
else:
    df = pd.read_csv(csv_path, on_bad_lines='skip', encoding='utf-8', engine='python')

print(f"DataFrame shape: {df.shape}")
print(f"Column names: {df.columns.tolist()}")

# Step 3: Check for DESCRIPTION column
print(f"\n[Step 3] Looking for DESCRIPTION column:")
print("-" * 80)

desc_col = None
for col in df.columns:
    col_lower = str(col).lower().strip()
    if 'desc' in col_lower or 'item' in col_lower:
        desc_col = col
        print(f"✓ Found description column: '{col}'")
        break

if not desc_col:
    print("❌ NO DESCRIPTION COLUMN FOUND!")
    print("Available columns:", df.columns.tolist())

# Step 4: Process rows looking for descriptions
print(f"\n[Step 4] Processing rows to extract vendor line items:")
print("-" * 80)

vendor_count = 0
for idx, row in df.iterrows():
    row_data = row.to_dict()
    
    # Try to find description
    description = ""
    for key, value in row_data.items():
        key_lower = str(key).lower().strip()
        if 'description' in key_lower or 'desc' in key_lower or 'item' in key_lower:
            if value and str(value).strip() and len(str(value).strip()) >= 5:
                description = str(value).strip()
                break
    
    if description:
        vendor_count += 1
        if vendor_count <= 5:  # Show first 5
            print(f"Row {idx}: {description[:60]}...")

print(f"\n✓ Found {vendor_count} vendor items with valid descriptions")

# Step 5: Show sample row data structure
print(f"\n[Step 5] Sample row data structure (row 15):")
print("-" * 80)
if len(df) > 1:
    sample_row = df.iloc[1].to_dict()
    for key, val in sample_row.items():
        print(f"  '{key}': '{val}'")



