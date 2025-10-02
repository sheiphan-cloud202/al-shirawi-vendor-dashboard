"""
Complete End-to-End Workflow for BOQ vs Vendor Comparison

This orchestrates all existing modules:
1. bedrock_header_extractor - Analyzes inquiry Excel and extracts BOQ headers
2. textract_tables - OCR vendor PDFs using AWS Textract
3. textract_to_csv - Converts Textract JSON to CSV
4. workflow_enhanced - LLM-powered comparison and matching

Usage:
    python complete_workflow.py
"""

import sys
from pathlib import Path
from typing import List
import shutil
import pandas as pd

# Import existing modules
from bedrock_header_extractor import (
    find_inquiry_excel,
    decide_headers_with_bedrock,
    export_clean_sheet_csvs,
    write_headers_summary
)

from textract_tables import (
    list_vendor_pdfs,
    analyze_pdf_tables_async,
    extract_tables_local_pdfplumber,  # Fallback for local PDF extraction
    save_tables_json
)

from textract_to_csv import (
    load_tables,
    tables_to_csv_rows,
    write_csv as write_textract_csv
)

from workflow_enhanced import EnhancedWorkflowOrchestrator


class CompleteWorkflowOrchestrator:
    """Orchestrates the complete end-to-end workflow"""
    
    def __init__(self):
        self.base_dir = Path("/Users/sheiphanjoseph/Desktop/Developer/al_shirawi_orc_poc")
        self.data_dir = self.base_dir / "data"
        self.out_dir = self.base_dir / "out"
        self.inquiry_csv_dir = self.out_dir / "inquiry_csv"
        self.textract_dir = self.out_dir / "textract"
        self.textract_csv_dir = self.out_dir / "textract_csv"
        self.comparison_dir = self.out_dir / "vendor_comparisons"
        
        # Create output directories
        for d in [self.out_dir, self.inquiry_csv_dir, self.textract_dir, 
                  self.textract_csv_dir, self.comparison_dir]:
            d.mkdir(parents=True, exist_ok=True)
    
    def _combine_boq_csvs(self) -> None:
        """Combines all individual BOQ sheet CSVs into a single FINAL.csv"""
        
        print("   - Combining BOQ sheets into FINAL.csv...")
        
        # Find all generated CSVs
        sheet_csvs = sorted(self.inquiry_csv_dir.glob("*.csv"))
        
        # Filter out the final csv if it already exists
        sheet_csvs = [p for p in sheet_csvs if p.name != "FINAL.csv"]

        if not sheet_csvs:
            print("   ⚠️ No individual BOQ sheet CSVs found to combine.")
            return

        # Read and concatenate
        all_dfs = []
        for csv_path in sheet_csvs:
            try:
                df = pd.read_csv(csv_path)
                all_dfs.append(df)
            except Exception as e:
                print(f"   ❌ Could not read {csv_path.name}: {e}")
                continue
        
        if not all_dfs:
            print("   ❌ Failed to read any BOQ sheet CSVs.")
            return

        combined_df = pd.concat(all_dfs, ignore_index=True)
        
        # Write to FINAL.csv
        final_csv_path = self.inquiry_csv_dir / "FINAL.csv"
        combined_df.to_csv(final_csv_path, index=False)
        
        print(f"   ✓ Wrote {len(combined_df)} rows to {final_csv_path.name}")

    def _clear_outputs(self):
        """Deletes all generated output files for a fresh run"""
        print("🔥 --force specified, clearing all previous outputs...")
        
        dirs_to_clear = [
            self.inquiry_csv_dir,
            self.textract_dir,
            self.textract_csv_dir,
            self.comparison_dir
        ]
        
        for d in dirs_to_clear:
            if d.exists():
                print(f"   - Deleting contents of {d.relative_to(self.base_dir)}")
                shutil.rmtree(d)
                d.mkdir(parents=True, exist_ok=True)
        
        # Also delete summary files in root of out/
        for f in self.out_dir.glob("*"):
            if f.is_file():
                f.unlink()
        
        print("✓ Outputs cleared.")

    def run(self, skip_textract: bool = False, skip_boq: bool = False, 
              force: bool = False, use_bedrock: bool = False) -> int:
        """
        Execute complete workflow
        
        Args:
            skip_textract: Skip Textract OCR if CSVs already exist
            skip_boq: Skip BOQ extraction if CSV already exists
            force: Force a fresh run by deleting all previous outputs
            use_bedrock: Force enable Bedrock LLM for analysis
        """
        
        if force:
            self._clear_outputs()

        print("=" * 80)
        print("COMPLETE END-TO-END BOQ COMPARISON WORKFLOW")
        print("=" * 80)
        
        # ========================================================================
        # STEP 1: Extract and Analyze Inquiry BOQ (Excel → CSV)
        # ========================================================================
        
        print("\n" + "=" * 80)
        print("STEP 1: Analyzing Inquiry BOQ Excel")
        print("=" * 80)
        
        inquiry_excel = find_inquiry_excel(self.data_dir / "Enquiry Attachment")
        
        if not inquiry_excel or not inquiry_excel.exists():
            print("❌ ERROR: No inquiry Excel file found in data/Enquiry Attachment/")
            print("   Place your BOQ Excel file there first.")
            return 1
        
        print(f"📄 Found inquiry Excel: {inquiry_excel.name}")
        
        # Check if already processed
        boq_csv = self.inquiry_csv_dir / "FINAL.csv"
        if skip_boq and boq_csv.exists() and not force:
            print(f"✓ BOQ CSV already exists, skipping extraction")
        else:
            print("🧠 Using Bedrock LLM to analyze headers and extract tables...")
            
            # Use Bedrock to understand headers
            decisions = decide_headers_with_bedrock(inquiry_excel)
            
            # Export clean CSVs
            written = export_clean_sheet_csvs(inquiry_excel, decisions, self.inquiry_csv_dir)
            print(f"✓ Extracted {len(written)} BOQ sheet(s) to CSV")
            
            # Combine into a single FINAL.csv for downstream use
            self._combine_boq_csvs()

            # Write headers summary
            headers_json = self.out_dir / "inquiry_headers.json"
            write_headers_summary(decisions, headers_json)
            print(f"✓ Saved header analysis: {headers_json.name}")
        
        # ========================================================================
        # STEP 2: OCR Vendor PDF Responses (PDF → Textract JSON)
        # ========================================================================
        
        print("\n" + "=" * 80)
        print("STEP 2: OCR Vendor PDF Quotations using AWS Textract")
        print("=" * 80)
        
        vendor_pdfs = list_vendor_pdfs(self.data_dir)
        
        if not vendor_pdfs:
            print("⚠️  WARNING: No vendor PDF files found")
            print("   Place vendor PDFs in data/Response N Attachment/ folders")
        else:
            print(f"📄 Found {len(vendor_pdfs)} vendor PDF file(s)")
            
            # Check if Textract should be run
            existing_json = list(self.textract_dir.glob("*.tables.json"))
            if skip_textract and existing_json and not force:
                print(f"✓ Found {len(existing_json)} existing Textract JSON files, skipping OCR")
            else:
                import os
                # Default to provided S3 bucket/prefix if env not set
                bucket = os.getenv("TEXTRACT_S3_BUCKET", "textract-s3-bucket-al")
                prefix = os.getenv("TEXTRACT_S3_PREFIX", "textract-s3-prefix")
                
                if not bucket:
                    print("⚠️  TEXTRACT_S3_BUCKET not set")
                    print("   Falling back to local PDF extraction using pdfplumber...")
                    print()
                    
                    # Use fallback local PDF extraction
                    for vendor_key, pdf_path in vendor_pdfs:
                        print(f"\n  Processing: {vendor_key}")
                        print(f"    📄 PDF: {pdf_path.name}")
                        print(f"    🔍 Running local PDF extraction (pdfplumber)...")
                        
                        try:
                            tables = extract_tables_local_pdfplumber(pdf_path)
                            out_path = save_tables_json(self.textract_dir, vendor_key, pdf_path, tables)
                            print(f"    ✓ Extracted {len(tables)} table(s) → {out_path.name}")
                        except Exception as e:
                            print(f"    ❌ ERROR: {e}")
                            import traceback
                            traceback.print_exc()
                            continue
                else:
                    print(f"☁️  Using AWS Textract with S3 bucket: {bucket}")
                    
                    for vendor_key, pdf_path in vendor_pdfs:
                        print(f"\n  Processing: {vendor_key}")
                        print(f"    📄 PDF: {pdf_path.name}")
                        print("    🔍 Running AWS Textract (async)...")
                        
                        try:
                            tables = analyze_pdf_tables_async(pdf_path, bucket=bucket, prefix=prefix)
                            out_path = save_tables_json(self.textract_dir, vendor_key, pdf_path, tables)
                            print(f"    ✓ Extracted {len(tables)} table(s) → {out_path.name}")
                        except Exception as e:
                            print(f"    ❌ ERROR: {e}")
                            continue
        
        # ========================================================================
        # STEP 3: Convert Textract JSON to CSV
        # ============ ============================================================
        
        print("\n" + "=" * 80)
        print("STEP 3: Converting Textract JSON to CSV Format")
        print("=" * 80)
        
        textract_json_files = sorted(self.textract_dir.glob("*.tables.json"))
        existing_vendor_csvs = list(self.textract_csv_dir.glob("*.csv"))
        
        if not textract_json_files and not existing_vendor_csvs:
            print("⚠️  WARNING: No Textract JSON files found")
            print("   Run Step 2 (Textract OCR) first")
        elif existing_vendor_csvs and not textract_json_files:
            print(f"✓ Found {len(existing_vendor_csvs)} existing vendor CSV file(s)")
            print("  Skipping Textract JSON conversion (CSVs already exist)")
        else:
            print(f"📄 Found {len(textract_json_files)} Textract JSON file(s)")
            
            for json_file in textract_json_files:
                print(f"\n  Converting: {json_file.name}")
                
                try:
                    tables = load_tables(json_file)
                    rows = tables_to_csv_rows(tables)
                    
                    csv_name = json_file.stem + ".csv"
                    out_csv = self.textract_csv_dir / csv_name
                    write_textract_csv(rows, out_csv)
                    
                    print(f"    ✓ Wrote {len(rows)} rows → {csv_name}")
                except Exception as e:
                    print(f"    ❌ ERROR: {e}")
                    continue
        
        # ========================================================================
        # STEP 4: LLM-Powered BOQ vs Vendor Comparison
        # ========================================================================
        
        print("\n" + "=" * 80)
        print("STEP 4: Multi-Agent LLM Comparison (BOQ vs Vendor Quotes)")
        print("=" * 80)
        
        # Check prerequisites
        if not boq_csv.exists():
            print("❌ ERROR: BOQ CSV not found. Run Step 1 first.")
            return 1
        
        vendor_csvs = list(self.textract_csv_dir.glob("*.csv"))
        if not vendor_csvs:
            print("❌ ERROR: No vendor CSV files found.")
            print("   Either:")
            print("   1. Run Steps 2-3 (Textract OCR + CSV conversion), OR")
            print("   2. Place vendor CSV files manually in out/textract_csv/")
            return 1
        
        print(f"✓ BOQ CSV ready: {boq_csv.name}")
        print(f"✓ Vendor CSVs ready: {len(vendor_csvs)} file(s)")
        print()
        
        # Run enhanced workflow
        try:
            orchestrator = EnhancedWorkflowOrchestrator(use_bedrock=use_bedrock)
            result = orchestrator.run()
            
            if result == 0:
                print("\n" + "=" * 80)
                print("✅ COMPLETE WORKFLOW FINISHED SUCCESSFULLY!")
                print("=" * 80)
                
                # Print summary of outputs
                print("\n📂 Output Files:")
                print(f"   BOQ CSV: {boq_csv}")
                print(f"   Vendor CSVs: {self.textract_csv_dir}")
                print(f"   Comparisons: {self.comparison_dir}")
                
                comparison_files = list(self.comparison_dir.glob("*_comparison.csv"))
                if comparison_files:
                    print(f"\n📊 Generated {len(comparison_files)} comparison CSV(s):")
                    for cf in comparison_files:
                        print(f"      • {cf.name}")
                
                print("\n💡 Next Steps:")
                print("   1. Review comparison CSVs in Excel/Google Sheets")
                print("   2. Check vendor_summary.json for quality scores")
                print("   3. Identify best vendor and flag issues")
                
                return 0
            else:
                print("\n❌ Workflow completed with errors")
                return result
                
        except Exception as e:
            print(f"\n❌ ERROR in Step 4: {e}")
            import traceback
            traceback.print_exc()
            return 1


def main(argv: List[str]) -> int:
    """Main entry point"""
    
    # Parse simple command-line flags
    skip_textract = "--skip-textract" in argv or "--skip-ocr" in argv
    skip_boq = "--skip-boq" in argv
    force = "--force" in argv
    use_bedrock = "--use-bedrock" in argv
    
    if "--help" in argv or "-h" in argv:
        print("""
Complete BOQ vs Vendor Comparison Workflow

This script runs the full end-to-end pipeline:
  1. Extract BOQ from inquiry Excel (using Bedrock LLM)
  2. OCR vendor PDFs (using AWS Textract)
  3. Convert Textract output to CSV
  4. Compare BOQ vs vendor quotes (using multi-agent LLM)

Usage:
  python complete_workflow.py [options]

Options:
  --skip-textract    Skip Textract OCR if CSV files already exist
  --skip-boq         Skip BOQ extraction if CSV already exists
  --force            Force a fresh run by deleting all previous outputs
  --use-bedrock      Force enable Bedrock LLM for analysis steps
  --help, -h         Show this help message

Environment Variables:
  AWS_PROFILE                  AWS credentials profile
  AWS_REGION                   AWS region (default: us-east-1)
  TEXTRACT_S3_BUCKET          S3 bucket for Textract (required for Step 2)
  TEXTRACT_S3_PREFIX          S3 prefix for temp files (optional)
  BEDROCK_MODEL_ID            Bedrock model (default: Claude 3.5 Sonnet)
  BEDROCK_DISABLE             Set to "1" to use heuristics only

Examples:
  # Run full workflow
  python complete_workflow.py

  # Run full workflow, forcing regeneration of all files
  python complete_workflow.py --force

  # Run full workflow with Bedrock LLMs enabled
  python complete_workflow.py --use-bedrock

  # Skip OCR if already done
  python complete_workflow.py --skip-textract

  # Run only comparison step
  python complete_workflow.py --skip-textract --skip-boq
""")
        return 0
    
    orchestrator = CompleteWorkflowOrchestrator()
    return orchestrator.run(
        skip_textract=skip_textract, 
        skip_boq=skip_boq,
        force=force,
        use_bedrock=use_bedrock
    )


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

