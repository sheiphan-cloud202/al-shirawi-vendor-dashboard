
# Al Shirawi ORC POC - AI-Powered Vendor Response Comparison

Complete end-to-end multi-agent AI workflow using AWS Bedrock for automated BOQ (Bill of Quantities) vs vendor response comparison and validation.

## 🚀 Quick Start

```bash
# Set environment variables
export AWS_PROFILE=your-profile
export AWS_REGION=us-east-1
export TEXTRACT_S3_BUCKET=your-bucket

# Run complete workflow (all steps)
uv run complete_workflow.py

# Or skip already-completed steps
uv run complete_workflow.py --skip-textract --skip-boq
```

### Start the API server

```bash
# Install deps
uv sync

# Start FastAPI with uvicorn
uv run uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```

### Open the UI

- Navigate to: `http://localhost:8000/` (auto-redirects to `/ui/`)
- **Upload Page**: Upload the inquiry Excel and vendor PDFs, then click Run Workflow.
- **Analysis Page**: `/ui/analysis.html` - AI-powered vendor comparison and recommendations

### Analysis Features

The Analysis Dashboard provides:

1. **Vendor Selection**: Select one or more vendors to compare
2. **Detailed Metrics**: View match rates, total prices, and quality scores per vendor
3. **AI Recommendations**: Get Claude 3.5 Sonnet-powered analysis on which vendor offers the best value
4. **Issue Detection**: Automatically identifies quantity variances, UOM mismatches, and type conflicts

The AI analyzes:
- Match quality and completion rates
- Price competitiveness
- Issue severity and frequency
- Overall vendor reliability

API endpoints:

**Upload & Workflow:**
- `GET /health` – health check
- `POST /upload/enquiry` – multipart form-data with file field `file` (Excel)
- `POST /upload/vendor` – multipart form-data with fields `response_number` (int) and `files` (multiple PDFs)
- `POST /run-workflow` – form fields `skip_textract` and `skip_boq` (optional booleans)

**Analysis:**
- `GET /api/vendors` – list all vendor comparisons
- `GET /api/vendors/{vendor_id}/comparison` – detailed comparison data for a vendor
- `POST /api/analyze` – AI-powered analysis of selected vendors (form-data with `vendor_ids[]`)

Uploaded files are saved under `data/` in the expected structure:

- Enquiry Excel → `data/Enquiry Attachment/<filename>`
- Vendor PDFs → `data/Response N Attachment/<filename>`

## Overview

This system processes vendor quotations against inquiry requirements using a sophisticated multi-agent LLM architecture:

1. **Inquiry Analysis** - Extracts and normalizes BOQ requirements from Excel
2. **OCR Processing** - Extracts tables from vendor PDF responses using AWS Textract
3. **Multi-Agent Workflow** - Uses 4 specialized AI agents to compare, validate, and aggregate data

## Architecture

### Agent-Based Workflow

#### Agent 1: Header Normalization Agent
- Maps vendor CSV headers to standard BOQ schema
- Uses LLM to handle variations in terminology
- Provides confidence scores for each mapping

#### Agent 2: Line Item Extraction Agent
- Extracts structured data from vendor CSVs
- Validates extraction quality using LLM sampling
- Handles malformed CSV data gracefully

#### Agent 3: Requirement Matching Agent
- Fuzzy matches vendor items to BOQ requirements
- Uses LLM for intelligent semantic matching
- Pre-filters candidates using string similarity

#### Agent 4: Validation & QA Agent
- Validates all comparisons and generates quality scores
- Identifies mismatches (quantity, UOM, missing items)
- Creates comprehensive comparison reports

## Requirements

- Python 3.11+
- AWS Account with credentials configured
- Access to:
  - AWS Bedrock (Claude 3.5 Sonnet)
  - AWS Textract (for PDF table extraction)
  - S3 bucket (for Textract async operations)

### Environment Variables

```bash
# Required
export AWS_PROFILE=your-profile
export AWS_REGION=us-east-1

# For Textract PDF processing
export TEXTRACT_S3_BUCKET=your-bucket
export TEXTRACT_S3_PREFIX=textract-temp/

# Optional
export BEDROCK_MODEL_ID=anthropic.claude-3-5-sonnet-20240620-v1:0
export BEDROCK_DISABLE=1  # To disable LLM and use heuristics only
```

## Installation

```bash
# Install dependencies
uv sync

# Or with pip
pip install -e .
```

## Usage

### Option 1: Complete Integrated Workflow (Recommended)

Run all steps with a single command:

```bash
uv run complete_workflow.py
```

This orchestrates all modules:
1. ✅ **BOQ Extraction** - Analyzes Excel with Bedrock LLM
2. ✅ **PDF OCR** - Extracts vendor tables with Textract
3. ✅ **CSV Conversion** - Converts Textract JSON to CSV
4. ✅ **LLM Comparison** - Multi-agent matching and validation

**Outputs:**
- `out/inquiry_csv/FINAL.csv` - Clean BOQ requirements
- `out/textract_csv/*.csv` - Vendor quote CSVs
- `out/vendor_comparisons/*_comparison.csv` - Per-vendor comparisons (4 files)

### Option 2: Manual Step-by-Step

#### Step 1: Extract BOQ Headers

```bash
uv run bedrock_header_extractor.py
```

#### Step 2: OCR Vendor PDFs

```bash
uv run textract_tables.py
```

#### Step 3: Convert to CSV

```bash
uv run textract_to_csv.py
```

#### Step 4: Run Comparison

```bash
uv run workflow_enhanced.py
```

## Output Files

### vendor_comparison.csv

Columns:
- `BOQ Sr No, BOQ Description, BOQ Qty, BOQ UOM` - Original requirements
- `Vendor, Vendor Description, Vendor Qty, Vendor UOM` - Vendor response
- `Unit Price, Total Price, Brand` - Pricing information
- `Match Confidence, Match Reasoning` - AI matching details
- `Validation Status, Validation Notes` - Quality validation

### vendor_summary.json

Contains:
- Per-vendor quality scores and match rates
- Total quoted values
- Missing items and mismatches
- Overall comparison analytics

## Key Features

### LLM-Powered Validation

- **Header Mapping**: Uses Claude to intelligently map vendor headers to standard schema
- **Extraction Validation**: Samples line items for LLM quality checks
- **Fuzzy Matching**: Semantic matching of descriptions (e.g., "900MM CABLE TRAY HDG" matches "CABLE TRAY 900X50MM, HDG")
- **Quality Scoring**: Multi-factor quality assessment with confidence metrics

### Robust Error Handling

- Graceful CSV parsing with malformed data handling
- Fallback heuristics when LLM is unavailable
- Multiple encoding support (UTF-8, Latin-1)
- Skip bad rows automatically

### Scalability

- Processes multiple vendors in parallel
- Efficient LLM sampling (validates every 10th item)
- Candidate pre-filtering reduces LLM calls
- Configurable confidence thresholds

## Project Structure

```
data/
  ├── Enquiry Attachment/       # BOQ Excel files
  │   └── J1684 Cable Tray & Trunking BOQ.xlsx
  ├── Enquiry Email.txt         # Original inquiry email
  ├── Response N - Vendor.txt   # Vendor email responses
  └── Response N Attachment/    # Vendor PDF quotations

out/
  ├── inquiry_csv/              # Extracted BOQ CSVs
  ├── inquiry_headers.json      # BOQ header mappings
  ├── textract/                 # Raw Textract JSON
  ├── textract_csv/             # Vendor response CSVs
  ├── vendor_comparison.csv     # Final comparison output
  └── vendor_summary.json       # Analytics summary

Scripts:
  - complete_workflow.py         # 🎯 MAIN: Complete integrated pipeline
  - bedrock_header_extractor.py  # Step 1: BOQ analysis
  - textract_tables.py           # Step 2: PDF OCR  
  - textract_to_csv.py           # Step 3: Format conversion
  - workflow_enhanced.py         # Step 4: Enhanced multi-agent workflow
  - workflow.py                  # Legacy: Basic workflow
  - index_responses.py           # Helper: Index vendor files
  - main.py                      # Helper: BOQ analysis functions
```

## Advanced Configuration

### Customize LLM Behavior

Edit `workflow.py` to adjust:

- Temperature (currently 0.0 for deterministic output)
- Max tokens per request
- Confidence thresholds
- Sampling frequency for validation

### Add Custom Validation Rules

Extend `ValidationAgent._validate_match()` with domain-specific rules:

```python
# Example: Check material compatibility
if "HDG" in boq_req.description and "HDG" not in vendor_item.description:
    return "MATERIAL_MISMATCH", "Vendor may not be HDG (Hot Dip Galvanized)"
```

## Limitations & Future Work

Current limitations:
- Textract CSV quality depends on PDF structure
- Some vendors have inconsistent table formats (manual review recommended)
- LLM costs scale with data volume

Potential improvements:
- Add multi-currency conversion
- Implement cost optimization analysis
- Build interactive dashboard for results
- Add support for image-based BOQs
- Integrate with ERP systems

## License

Internal POC - Not for distribution


