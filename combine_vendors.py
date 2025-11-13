"""
Script to combine all vendor comparison CSVs into a single comparison file.
Creates a format similar to the image: BOQ items with multiple vendor columns.
"""
import csv
import re
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict

BASE_DIR = Path("/Users/sheiphanjoseph/Desktop/Developer/al_shirawi_orc_poc")
COMPARISON_DIR = BASE_DIR / "out" / "vendor_comparisons"
OUTPUT_FILE = BASE_DIR / "out" / "all_vendors_comparison.csv"


def get_vendor_name_from_filename(filename: str) -> str:
    """Extract vendor name from filename like '3_-_IKKT_comparison.csv'"""
    match = re.match(r"\d+_-_(.+)_comparison\.csv", filename)
    if match:
        return match.group(1).replace("_", " ")
    return filename.replace("_comparison.csv", "").replace("_", " ")


def load_all_vendor_data() -> tuple[Dict[str, Dict], List[str]]:
    """Load all vendor comparison files and organize by BOQ Sr.No"""
    vendor_data = defaultdict(dict)  # {boq_sr_no: {vendor_name: row_data}}
    vendor_names = []
    
    # Get all comparison files
    comparison_files = sorted(COMPARISON_DIR.glob("*_comparison.csv"))
    
    for csv_file in comparison_files:
        vendor_name = get_vendor_name_from_filename(csv_file.name)
        if vendor_name not in vendor_names:
            vendor_names.append(vendor_name)
        
        with open(csv_file, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f)
            for row in reader:
                boq_sr_no = row.get("BOQ Sr.No", "").strip()
                if boq_sr_no:
                    vendor_data[boq_sr_no][vendor_name] = row
    
    return vendor_data, vendor_names


def get_boq_unit_price(boq_qty: str, boq_uom: str, vendor_data: Dict) -> str:
    """Calculate or extract BOQ unit price if available"""
    # Try to get from first vendor's data if available
    # Or calculate from vendor prices if we have them
    for vendor_name, row in vendor_data.items():
        vendor_unit_price = row.get("Vendor Unit Price", "").strip()
        if vendor_unit_price:
            try:
                return str(float(vendor_unit_price))
            except (ValueError, TypeError):
                pass
    return ""


def create_combined_csv(vendor_data: Dict[str, Dict], vendor_names: List[str]) -> None:
    """Create the combined CSV file with all vendors"""
    
    # Sort BOQ items by Sr.No (handle numeric sorting)
    sorted_boq_items = sorted(
        vendor_data.keys(),
        key=lambda x: (int(x) if x.isdigit() else float('inf'), x)
    )
    
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        
        # Create header row 1: BOQ columns + vendor headers (vendor name spans 3 columns)
        header_row1 = [
            "SI NO",
            "BOQ Items",
            "Unit",
            "Qty",
            "Unit Price(AED)"
        ]
        # Add vendor headers - vendor name in first column, empty for next 2 (simulating merged cells)
        for vendor_name in vendor_names:
            header_row1.append(vendor_name)
            header_row1.append("")  # Empty for merged cell effect
            header_row1.append("")  # Empty for merged cell effect
        
        writer.writerow(header_row1)
        
        # Create header row 2: sub-headers for vendor columns
        header_row2 = ["", "", "", "", ""]  # Empty for BOQ columns
        for vendor_name in vendor_names:
            header_row2.extend(["Unit", "Qty", "Unit Price(AED)"])
        
        writer.writerow(header_row2)
        
        # Write data rows
        for boq_sr_no in sorted_boq_items:
            vendors_for_item = vendor_data[boq_sr_no]
            
            # Get BOQ data from first available vendor (all should have same BOQ data)
            first_vendor_data = next(iter(vendors_for_item.values())) if vendors_for_item else {}
            
            boq_description = first_vendor_data.get("BOQ Description", "").strip()
            boq_qty = first_vendor_data.get("BOQ Qty", "").strip()
            boq_uom = first_vendor_data.get("BOQ UOM", "").strip()
            boq_unit_price = get_boq_unit_price(boq_qty, boq_uom, vendors_for_item)
            
            # Start building the row
            row = [
                boq_sr_no,
                boq_description,
                boq_uom,
                boq_qty,
                boq_unit_price
            ]
            
            # Add vendor data for each vendor
            for vendor_name in vendor_names:
                if vendor_name in vendors_for_item:
                    vendor_row = vendors_for_item[vendor_name]
                    row.extend([
                        vendor_row.get("Vendor UOM", "").strip(),
                        vendor_row.get("Vendor Qty", "").strip(),
                        vendor_row.get("Vendor Unit Price", "").strip()
                    ])
                else:
                    # Vendor didn't quote this item
                    row.extend(["", "", ""])
            
            writer.writerow(row)
    
    print(f"✅ Created combined vendor comparison file: {OUTPUT_FILE}")
    print(f"   Total BOQ items: {len(sorted_boq_items)}")
    print(f"   Total vendors: {len(vendor_names)}")
    print(f"   Vendors: {', '.join(vendor_names)}")


def main():
    """Main function"""
    if not COMPARISON_DIR.exists():
        print(f"❌ Comparison directory not found: {COMPARISON_DIR}")
        return
    
    print("Loading vendor comparison files...")
    vendor_data, vendor_names = load_all_vendor_data()
    
    if not vendor_data:
        print("❌ No vendor data found!")
        return
    
    print(f"Found {len(vendor_names)} vendors: {', '.join(vendor_names)}")
    print(f"Found {len(vendor_data)} unique BOQ items")
    
    print("\nCreating combined CSV...")
    create_combined_csv(vendor_data, vendor_names)
    
    print("\n✅ Done!")


if __name__ == "__main__":
    main()

