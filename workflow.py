"""
Multi-Agent Bedrock Workflow for BOQ vs Vendor Response Comparison

This workflow uses AWS Bedrock LLMs at multiple steps to:
1. Normalize and map vendor CSV headers to BOQ schema
2. Extract and validate line items from vendor responses
3. Match vendor items to BOQ requirements with fuzzy matching
4. Validate extractions and generate comparison reports

Architecture:
- Agent 1: Header Normalization Agent - Maps vendor headers to standard schema
- Agent 2: Line Item Extraction Agent - Extracts structured data from vendor CSVs
- Agent 3: Requirement Matching Agent - Matches vendor items to BOQ requirements
- Agent 4: Validation & QA Agent - Validates all extractions and matches
"""

import csv
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import boto3
import pandas as pd



# ============================================================================
# Data Models
# ============================================================================

@dataclass
class HeaderMapping:
    """Mapping of vendor CSV headers to standard BOQ schema"""
    vendor_name: str
    raw_headers: List[str]
    mapped_headers: Dict[str, str]  # vendor_col -> standard_col
    confidence: float
    llm_reasoning: str


@dataclass
class LineItem:
    """Extracted line item from vendor response"""
    vendor_name: str
    row_index: int
    sr_no: Optional[str]
    description: str
    qty: Optional[float]
    uom: Optional[str]
    unit_price: Optional[float]
    total_price: Optional[float]
    brand: Optional[str]
    remarks: Optional[str]
    raw_data: Dict[str, Any]
    extraction_confidence: float
    llm_validation: str


@dataclass
class BOQRequirement:
    """BOQ requirement item"""
    sr_no: str
    description: str
    qty: float
    uom: str
    remarks: Optional[str]


@dataclass
class MatchedItem:
    """Vendor item matched to BOQ requirement"""
    boq_sr_no: str
    boq_description: str
    boq_qty: float
    boq_uom: str
    vendor_name: str
    vendor_description: str
    vendor_qty: Optional[float]
    vendor_uom: Optional[str]
    vendor_unit_price: Optional[float]
    vendor_total_price: Optional[float]
    vendor_brand: Optional[str]
    match_confidence: float
    match_reasoning: str
    validation_status: str
    validation_notes: str


# ============================================================================
# Bedrock LLM Client
# ============================================================================

class BedrockClient:
    """Wrapper for AWS Bedrock invocations with retry and error handling"""
    
    def __init__(self):
        profile = os.getenv("AWS_PROFILE")
        region = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION", "us-east-1")
        session = boto3.Session(profile_name=profile, region_name=region)
        self.client = session.client("bedrock-runtime")
        self.model_id = os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-5-sonnet-20240620-v1:0")
        self.use_bedrock = os.getenv("BEDROCK_DISABLE") != "1"
    
    def invoke(self, prompt: str, max_tokens: int = 4096, temperature: float = 0.0, 
               system_prompt: Optional[str] = None) -> str:
        """Invoke Bedrock model with prompt"""
        if not self.use_bedrock:
            return "{}"  # Return empty JSON for fallback
        
        try:
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
            
            if self.model_id.startswith("anthropic."):
                body = {
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "messages": messages,
                }
                if system_prompt:
                    body["system"] = system_prompt
            else:
                # Fallback for other models
                body = {
                    "inputText": prompt,
                    "textGenerationConfig": {
                        "maxTokenCount": max_tokens,
                        "temperature": temperature,
                    },
                }
            
            response = self.client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(body)
            )
            
            payload = response.get("body")
            if hasattr(payload, "read"):
                data = json.loads(payload.read())
            else:
                data = json.loads(payload)
            
            # Extract text from response
            if self.model_id.startswith("anthropic."):
                content = data.get("content", [])
                text = "".join([p.get("text", "") for p in content if p.get("type") == "text"])
            else:
                text = data.get("results", [{}])[0].get("outputText", "")
            
            return text.strip()
        
        except Exception as e:
            print(f"Bedrock invocation error: {e}")
            return "{}"
    
    def extract_json(self, text: str) -> Optional[Dict]:
        """Extract JSON from LLM response (handles code fences)"""
        try:
            # Try direct parse
            return json.loads(text)
        except Exception:
            pass
        
        # Try to extract from code fence
        if "```" in text:
            parts = text.split("```")
            for part in parts:
                # Skip language tags
                lines = part.strip().splitlines()
                if lines and lines[0].strip().lower() in ("json", ""):
                    candidate = "\n".join(lines[1:] if lines[0].strip() else lines)
                    try:
                        return json.loads(candidate)
                    except Exception:
                        continue
        
        return None


# ============================================================================
# Agent 1: Header Normalization Agent
# ============================================================================

class HeaderNormalizationAgent:
    """Uses LLM to map vendor CSV headers to standard BOQ schema"""
    
    def __init__(self, bedrock: BedrockClient):
        self.bedrock = bedrock
        self.standard_schema = {
            "sr_no": "Serial number or item number",
            "description": "Item description or specification",
            "qty": "Quantity",
            "uom": "Unit of measure (e.g., Mtr, Nos, Lth)",
            "unit_price": "Unit price or rate",
            "total_price": "Total price or amount",
            "brand": "Brand or manufacturer",
            "remarks": "Remarks or notes",
        }
    
    def normalize_headers(self, vendor_name: str, raw_headers: List[str]) -> HeaderMapping:
        """Use LLM to map vendor headers to standard schema"""
        
        system_prompt = """You are a data mapping expert. Your job is to map vendor CSV column headers 
to a standard BOQ schema. Be flexible with variations and synonyms."""
        
        prompt = f"""Given these vendor CSV headers, map them to the standard BOQ schema.

Vendor: {vendor_name}

Vendor Headers:
{json.dumps(raw_headers, indent=2)}

Standard BOQ Schema:
{json.dumps(self.standard_schema, indent=2)}

Return ONLY a JSON object with this structure:
{{
  "mapped_headers": {{
    "vendor_column_name": "standard_schema_key",
    ...
  }},
  "confidence": 0.95,
  "reasoning": "Brief explanation of mapping choices"
}}

Important:
- Map each vendor header to the most appropriate standard schema key
- Use null if no good mapping exists
- Consider synonyms: S.I/SL.NO -> sr_no, U/Price -> unit_price, Qty/Quantity -> qty
- Confidence should be 0.0-1.0
"""
        
        response = self.bedrock.invoke(prompt, system_prompt=system_prompt, max_tokens=2048)
        result = self.bedrock.extract_json(response)
        
        if not result:
            # Fallback: basic heuristic mapping
            result = self._fallback_mapping(raw_headers)
        
        return HeaderMapping(
            vendor_name=vendor_name,
            raw_headers=raw_headers,
            mapped_headers=result.get("mapped_headers", {}),
            confidence=float(result.get("confidence", 0.5)),
            llm_reasoning=result.get("reasoning", "Fallback heuristic mapping")
        )
    
    def _fallback_mapping(self, headers: List[str]) -> Dict:
        """Heuristic fallback when LLM is unavailable"""
        mapping = {}
        for h in headers:
            h_lower = h.lower().strip()
            if any(k in h_lower for k in ["sr", "s.i", "sl.no", "item no", "item#"]):
                mapping[h] = "sr_no"
            elif any(k in h_lower for k in ["desc", "specification", "item"]):
                mapping[h] = "description"
            elif "qty" in h_lower or "quantity" in h_lower:
                mapping[h] = "qty"
            elif "uom" in h_lower or "unit" in h_lower:
                mapping[h] = "uom"
            elif "unit price" in h_lower or "u/price" in h_lower or "rate" in h_lower:
                mapping[h] = "unit_price"
            elif "total" in h_lower or "amount" in h_lower:
                mapping[h] = "total_price"
            elif "brand" in h_lower or "make" in h_lower:
                mapping[h] = "brand"
            elif "remark" in h_lower or "note" in h_lower:
                mapping[h] = "remarks"
        
        return {
            "mapped_headers": mapping,
            "confidence": 0.6,
            "reasoning": "Heuristic pattern matching"
        }


# ============================================================================
# Agent 2: Line Item Extraction Agent
# ============================================================================

class LineItemExtractionAgent:
    """Extracts and validates line items from vendor CSVs using header mappings"""
    
    def __init__(self, bedrock: BedrockClient):
        self.bedrock = bedrock
    
    def extract_line_items(self, vendor_name: str, csv_path: Path, 
                          header_mapping: HeaderMapping) -> List[LineItem]:
        """Extract line items from vendor CSV using the header mapping"""
        
        # Read CSV with error handling for inconsistent rows
        try:
            df = pd.read_csv(csv_path, on_bad_lines='skip', encoding='utf-8', engine='python')
        except Exception:
            # Fallback: try different encoding
            try:
                df = pd.read_csv(csv_path, on_bad_lines='skip', encoding='latin-1', engine='python')
            except Exception as e:
                print(f"      ERROR reading CSV: {e}")
                return []
        line_items: List[LineItem] = []
        
        # Build reverse mapping
        reverse_map = {std: vendor for vendor, std in header_mapping.mapped_headers.items()}
        
        for idx, row in df.iterrows():
            raw_data = row.to_dict()
            
            # Extract fields using mapping
            sr_no = self._extract_field(row, reverse_map.get("sr_no"))
            description = self._extract_field(row, reverse_map.get("description"))
            qty = self._parse_number(self._extract_field(row, reverse_map.get("qty")))
            uom = self._extract_field(row, reverse_map.get("uom"))
            unit_price = self._parse_number(self._extract_field(row, reverse_map.get("unit_price")))
            total_price = self._parse_number(self._extract_field(row, reverse_map.get("total_price")))
            brand = self._extract_field(row, reverse_map.get("brand"))
            remarks = self._extract_field(row, reverse_map.get("remarks"))
            
            # Skip header rows or empty rows
            if not description or self._is_header_row(description):
                continue
            
            # LLM validation for complex items (sample every 10th for efficiency)
            validation = "Auto-extracted"
            confidence = 0.85
            if idx % 10 == 0 and self.bedrock.use_bedrock:
                confidence, validation = self._validate_extraction(
                    description, qty, uom, unit_price, raw_data
                )
            
            line_items.append(LineItem(
                vendor_name=vendor_name,
                row_index=int(idx),
                sr_no=sr_no,
                description=description or "",
                qty=qty,
                uom=uom,
                unit_price=unit_price,
                total_price=total_price,
                brand=brand,
                remarks=remarks,
                raw_data=raw_data,
                extraction_confidence=confidence,
                llm_validation=validation
            ))
        
        return line_items
    
    def _extract_field(self, row: pd.Series, column_name: Optional[str]) -> Optional[str]:
        """Extract field value from row"""
        if not column_name or column_name not in row.index:
            return None
        val = row[column_name]
        if pd.isna(val):
            return None
        return str(val).strip()
    
    def _parse_number(self, value: Optional[str]) -> Optional[float]:
        """Parse numeric value from string"""
        if not value:
            return None
        # Remove commas and parse
        clean = re.sub(r"[,\s]", "", str(value))
        try:
                return float(clean)
        except Exception:
            return None
    
    def _is_header_row(self, text: str) -> bool:
        """Check if row looks like a header or section title"""
        text_lower = text.lower()
        return any(k in text_lower for k in [
            "cable tray", "description", "item", "sr.no", "accessories",
            "total", "sub-total", "validity", "payment"
        ]) and len(text.split()) < 5
    
    def _validate_extraction(self, description: str, qty: Optional[float], 
                            uom: Optional[str], price: Optional[float],
                            raw_data: Dict) -> Tuple[float, str]:
        """Use LLM to validate extraction quality"""
        
        prompt = f"""Validate this extracted line item data quality.

Description: {description}
Quantity: {qty}
UOM: {uom}
Unit Price: {price}

Raw CSV row: {json.dumps(raw_data, indent=2)}

Return JSON:
{{
  "confidence": 0.95,
  "validation": "Good extraction" or "Issue description"
}}
"""
        
        response = self.bedrock.invoke(prompt, max_tokens=512, temperature=0.0)
        result = self.bedrock.extract_json(response)
        
        if result:
            return float(result.get("confidence", 0.8)), result.get("validation", "Validated")
        return 0.8, "Validated"


# ============================================================================
# Agent 3: Requirement Matching Agent
# ============================================================================

class RequirementMatchingAgent:
    """Matches vendor line items to BOQ requirements using LLM fuzzy matching"""
    
    def __init__(self, bedrock: BedrockClient):
        self.bedrock = bedrock
    
    def load_boq_requirements(self, boq_csv_path: Path) -> List[BOQRequirement]:
        """Load BOQ requirements from CSV"""
        df = pd.read_csv(boq_csv_path)
        requirements = []
        
        for idx, row in df.iterrows():
            sr_no = str(row.get("Sr. No", "")).strip()
            description = str(row.get("Description", "")).strip()
            qty_str = str(row.get("TOTAL Qty in Length / Nos", "")).strip()
            uom = str(row.get("UOM", "")).strip()
            remarks = str(row.get("REMARKS", "")).strip() if pd.notna(row.get("REMARKS")) else None
            
            # Skip headers and section titles
            if not description or not sr_no or sr_no.lower() in ("sr. no", "cable"):
                continue
            
            # Parse quantity
            try:
                qty = float(qty_str.replace(",", ""))
            except Exception:
                continue
            
            requirements.append(BOQRequirement(
                sr_no=sr_no,
                description=description,
                qty=qty,
                uom=uom,
                remarks=remarks
            ))
        
        return requirements
    
    def match_items(self, boq_requirements: List[BOQRequirement], 
                    vendor_items: List[LineItem]) -> List[MatchedItem]:
        """Match vendor items to BOQ requirements using LLM"""
        
        matched_items: List[MatchedItem] = []
        
        for boq_req in boq_requirements:
            # Find best matching vendor item using LLM
            best_match = self._find_best_match(boq_req, vendor_items)
            
            if best_match:
                vendor_item, confidence, reasoning = best_match
                
                # Validate match
                validation_status, validation_notes = self._validate_match(
                    boq_req, vendor_item, confidence
                )
                
                matched_items.append(MatchedItem(
                    boq_sr_no=boq_req.sr_no,
                    boq_description=boq_req.description,
                    boq_qty=boq_req.qty,
                    boq_uom=boq_req.uom,
                    vendor_name=vendor_item.vendor_name,
                    vendor_description=vendor_item.description,
                    vendor_qty=vendor_item.qty,
                    vendor_uom=vendor_item.uom,
                    vendor_unit_price=vendor_item.unit_price,
                    vendor_total_price=vendor_item.total_price,
                    vendor_brand=vendor_item.brand,
                    match_confidence=confidence,
                    match_reasoning=reasoning,
                    validation_status=validation_status,
                    validation_notes=validation_notes
                ))
            else:
                # No match found
                matched_items.append(MatchedItem(
                    boq_sr_no=boq_req.sr_no,
                    boq_description=boq_req.description,
                    boq_qty=boq_req.qty,
                    boq_uom=boq_req.uom,
                    vendor_name="N/A",
                    vendor_description="",
                    vendor_qty=None,
                    vendor_uom=None,
                    vendor_unit_price=None,
                    vendor_total_price=None,
                    vendor_brand=None,
                    match_confidence=0.0,
                    match_reasoning="No match found",
                    validation_status="MISSING",
                    validation_notes="Item not quoted by vendor"
                ))
        
        return matched_items
    
    def _find_best_match(self, boq_req: BOQRequirement, 
                        vendor_items: List[LineItem]) -> Optional[Tuple[LineItem, float, str]]:
        """Use LLM to find best matching vendor item for BOQ requirement"""
        
        if not vendor_items:
            return None
        
        # Heuristic pre-filter: get top candidates by string similarity
        candidates = self._get_candidate_items(boq_req, vendor_items, top_k=5)
        
        if not candidates:
            return None
        
        # Use LLM for final matching decision
        if self.bedrock.use_bedrock and len(candidates) > 1:
            return self._llm_match(boq_req, candidates)
        else:
            # Fallback: return first candidate with basic similarity
            return candidates[0], 0.7, "Heuristic match"
    
    def _get_candidate_items(self, boq_req: BOQRequirement, 
                            vendor_items: List[LineItem], top_k: int = 5) -> List[LineItem]:
        """Get candidate items using basic string similarity"""
        scored_items = []
        
        boq_desc_lower = boq_req.description.lower()
        boq_tokens = set(re.findall(r'\w+', boq_desc_lower))
        
        for item in vendor_items:
            vendor_desc_lower = item.description.lower()
            vendor_tokens = set(re.findall(r'\w+', vendor_desc_lower))
            
            # Jaccard similarity
            intersection = len(boq_tokens & vendor_tokens)
            union = len(boq_tokens | vendor_tokens)
            similarity = intersection / union if union > 0 else 0.0
            
            # Boost if key dimensions match (e.g., "900 MM", "2.0 MM")
            dimensions = re.findall(r'\d+\.?\d*\s*mm', boq_desc_lower)
            if dimensions:
                dim_match = sum(1 for d in dimensions if d in vendor_desc_lower)
                similarity += 0.2 * (dim_match / len(dimensions))
            
            if similarity > 0.3:  # Threshold
                scored_items.append((item, similarity))
        
        # Sort by similarity and return top-k
        scored_items.sort(key=lambda x: x[1], reverse=True)
        return [item for item, _ in scored_items[:top_k]]
    
    def _llm_match(self, boq_req: BOQRequirement, 
                   candidates: List[LineItem]) -> Tuple[LineItem, float, str]:
        """Use LLM to select best match from candidates"""
        
        candidates_text = "\n".join([
            f"Candidate {i+1}: {item.description} (Qty: {item.qty}, UOM: {item.uom})"
            for i, item in enumerate(candidates)
        ])
        
        prompt = f"""Match the BOQ requirement to the best vendor candidate.

BOQ Requirement:
Description: {boq_req.description}
Quantity: {boq_req.qty}
UOM: {boq_req.uom}

Vendor Candidates:
{candidates_text}

Return JSON:
{{
  "best_match_index": 0,
  "confidence": 0.95,
  "reasoning": "Brief explanation"
}}

Select the candidate that best matches the BOQ item specification.
"""
        
        response = self.bedrock.invoke(prompt, max_tokens=512, temperature=0.0)
        result = self.bedrock.extract_json(response)
        
        if result:
            idx = int(result.get("best_match_index", 0))
            confidence = float(result.get("confidence", 0.8))
            reasoning = result.get("reasoning", "LLM match")
            
            if 0 <= idx < len(candidates):
                return candidates[idx], confidence, reasoning
        
        # Fallback to first candidate
        return candidates[0], 0.7, "Fallback to first candidate"
    
    def _validate_match(self, boq_req: BOQRequirement, vendor_item: LineItem, 
                       confidence: float) -> Tuple[str, str]:
        """Validate the match quality"""
        
        # Basic validation rules
        if confidence < 0.5:
            return "UNCERTAIN", "Low confidence match"
        
        # Check quantity mismatch
        if vendor_item.qty and abs(vendor_item.qty - boq_req.qty) / boq_req.qty > 0.1:
            return "QTY_MISMATCH", f"Vendor qty {vendor_item.qty} vs BOQ qty {boq_req.qty}"
        
        # Check UOM compatibility
        if vendor_item.uom and boq_req.uom.lower() not in vendor_item.uom.lower():
            return "UOM_MISMATCH", f"Vendor UOM '{vendor_item.uom}' vs BOQ UOM '{boq_req.uom}'"
        
        if confidence >= 0.8:
            return "GOOD", "High confidence match"
        else:
            return "FAIR", "Acceptable match"


# ============================================================================
# Agent 4: Validation & QA Agent
# ============================================================================

class ValidationAgent:
    """Final validation and quality assurance using LLM"""
    
    def __init__(self, bedrock: BedrockClient):
        self.bedrock = bedrock
    
    def validate_comparison(self, matched_items: List[MatchedItem], 
                           vendor_name: str) -> Dict[str, Any]:
        """Generate validation report for vendor comparison"""
        
        total_items = len(matched_items)
        matched = sum(1 for m in matched_items if m.match_confidence > 0.5)
        missing = sum(1 for m in matched_items if m.validation_status == "MISSING")
        mismatches = sum(1 for m in matched_items if "MISMATCH" in m.validation_status)
        
        # Calculate total quoted value
        total_value = sum(
            m.vendor_total_price for m in matched_items 
            if m.vendor_total_price is not None
        )
        
        report = {
            "vendor_name": vendor_name,
            "total_boq_items": total_items,
            "matched_items": matched,
            "missing_items": missing,
            "mismatched_items": mismatches,
            "match_rate": matched / total_items if total_items > 0 else 0.0,
            "total_quoted_value": total_value,
            "quality_score": self._calculate_quality_score(matched_items),
            "issues": self._identify_issues(matched_items)
        }
        
        return report
    
    def _calculate_quality_score(self, matched_items: List[MatchedItem]) -> float:
        """Calculate overall quality score"""
        if not matched_items:
            return 0.0
        
        scores = []
        for item in matched_items:
            # Factors: match confidence, validation status
            score = item.match_confidence
            
            if item.validation_status == "GOOD":
                score *= 1.0
            elif item.validation_status == "FAIR":
                score *= 0.8
            elif "MISMATCH" in item.validation_status:
                score *= 0.5
            elif item.validation_status == "MISSING":
                score = 0.0
            
            scores.append(score)
        
        return sum(scores) / len(scores)
    
    def _identify_issues(self, matched_items: List[MatchedItem]) -> List[str]:
        """Identify major issues in the comparison"""
        issues = []
        
        for item in matched_items:
            if item.validation_status == "MISSING":
                issues.append(f"Missing item: {item.boq_sr_no} - {item.boq_description}")
            elif "QTY_MISMATCH" in item.validation_status:
                issues.append(f"Quantity mismatch: {item.boq_sr_no} - {item.validation_notes}")
            elif "UOM_MISMATCH" in item.validation_status:
                issues.append(f"UOM mismatch: {item.boq_sr_no} - {item.validation_notes}")
            elif item.match_confidence < 0.5:
                issues.append(f"Low confidence match: {item.boq_sr_no} - {item.boq_description}")
        
        return issues[:20]  # Limit to top 20 issues


# ============================================================================
# Workflow Orchestrator
# ============================================================================

class WorkflowOrchestrator:
    """Main workflow orchestrator that runs all agents in sequence"""
    
    def __init__(self):
        self.bedrock = BedrockClient()
        self.header_agent = HeaderNormalizationAgent(self.bedrock)
        self.extraction_agent = LineItemExtractionAgent(self.bedrock)
        self.matching_agent = RequirementMatchingAgent(self.bedrock)
        self.validation_agent = ValidationAgent(self.bedrock)
        
        self.data_dir = Path("/Users/sheiphanjoseph/Desktop/Developer/al_shirawi_orc_poc/data")
        self.out_dir = Path("/Users/sheiphanjoseph/Desktop/Developer/al_shirawi_orc_poc/out")
        self.textract_csv_dir = self.out_dir / "textract_csv"
        self.inquiry_csv_dir = self.out_dir / "inquiry_csv"
    
    def run(self) -> int:
        """Execute the full workflow"""
        
        print("=" * 80)
        print("BOQ vs Vendor Response Comparison Workflow")
        print("=" * 80)
        
        # Step 1: Load BOQ requirements
        print("\n[Step 1] Loading BOQ requirements...")
        boq_csv = self.inquiry_csv_dir / "FINAL.csv"
        if not boq_csv.exists():
            print(f"ERROR: BOQ CSV not found at {boq_csv}")
            return 1
        
        boq_requirements = self.matching_agent.load_boq_requirements(boq_csv)
        print(f"  Loaded {len(boq_requirements)} BOQ requirements")
        
        # Step 2: Process each vendor response
        print("\n[Step 2] Processing vendor responses...")
        vendor_csvs = list(self.textract_csv_dir.glob("*.csv"))
        print(f"  Found {len(vendor_csvs)} vendor CSV files")
        
        all_comparisons = []
        all_reports = []
        
        for csv_path in vendor_csvs:
            vendor_name = self._extract_vendor_name(csv_path.name)
            print(f"\n  Processing: {vendor_name}")
            
            # Step 2a: Normalize headers
            print("    [Agent 1] Normalizing headers...")
            raw_headers = self._read_csv_headers(csv_path)
            header_mapping = self.header_agent.normalize_headers(vendor_name, raw_headers)
            print(f"      Confidence: {header_mapping.confidence:.2f}")
            print(f"      Mapped {len(header_mapping.mapped_headers)} columns")
            
            # Step 2b: Extract line items
            print("    [Agent 2] Extracting line items...")
            line_items = self.extraction_agent.extract_line_items(
                vendor_name, csv_path, header_mapping
            )
            print(f"      Extracted {len(line_items)} line items")
            
            # Step 2c: Match to BOQ requirements
            print("    [Agent 3] Matching to BOQ requirements...")
            matched_items = self.matching_agent.match_items(boq_requirements, line_items)
            print(f"      Matched {len(matched_items)} items")
            
            # Step 2d: Validate comparison
            print("    [Agent 4] Validating comparison...")
            report = self.validation_agent.validate_comparison(matched_items, vendor_name)
            print(f"      Quality Score: {report['quality_score']:.2f}")
            print(f"      Match Rate: {report['match_rate']:.2%}")
            print(f"      Total Value: AED {report['total_quoted_value']:,.2f}")
            
            all_comparisons.extend(matched_items)
            all_reports.append(report)
        
        # Step 3: Generate aggregated outputs
        print("\n[Step 3] Generating aggregated outputs...")
        self._write_comparison_csv(all_comparisons)
        self._write_summary_report(all_reports)
        
        print("\n" + "=" * 80)
        print("Workflow completed successfully!")
        print("=" * 80)
        
        return 0
    
    def _extract_vendor_name(self, filename: str) -> str:
        """Extract vendor name from CSV filename"""
        # Format: Response_N_-_VendorName__filename.tables.csv
        parts = filename.replace(".tables.csv", "").split("__")
        if len(parts) >= 1:
            vendor_part = parts[0].replace("Response_", "").replace("_-_", " - ")
            return vendor_part
        return filename.replace(".csv", "")
    
    def _read_csv_headers(self, csv_path: Path) -> List[str]:
        """Read CSV headers"""
        try:
            df = pd.read_csv(csv_path, nrows=0, on_bad_lines='skip', engine='python')
            return df.columns.tolist()
        except Exception:
            # Fallback: read first line manually
            with open(csv_path, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
                return first_line.split(',')
    
    def _write_comparison_csv(self, matched_items: List[MatchedItem]) -> None:
        """Write aggregated comparison CSV"""
        out_path = self.out_dir / "vendor_comparison.csv"
        
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow([
                "BOQ Sr No", "BOQ Description", "BOQ Qty", "BOQ UOM",
                "Vendor", "Vendor Description", "Vendor Qty", "Vendor UOM",
                "Unit Price", "Total Price", "Brand",
                "Match Confidence", "Match Reasoning",
                "Validation Status", "Validation Notes"
            ])
            
            # Data rows
            for item in matched_items:
                writer.writerow([
                    item.boq_sr_no,
                    item.boq_description,
                    item.boq_qty,
                    item.boq_uom,
                    item.vendor_name,
                    item.vendor_description,
                    item.vendor_qty or "",
                    item.vendor_uom or "",
                    item.vendor_unit_price or "",
                    item.vendor_total_price or "",
                    item.vendor_brand or "",
                    f"{item.match_confidence:.2f}",
                    item.match_reasoning,
                    item.validation_status,
                    item.validation_notes
                ])
        
        print(f"  Written: {out_path}")
    
    def _write_summary_report(self, reports: List[Dict]) -> None:
        """Write summary report JSON"""
        out_path = self.out_dir / "vendor_summary.json"
        
        summary = {
            "generated_at": pd.Timestamp.now().isoformat(),
            "vendor_reports": reports,
            "overall": {
                "total_vendors": len(reports),
                "average_quality_score": sum(r["quality_score"] for r in reports) / len(reports) if reports else 0.0,
                "average_match_rate": sum(r["match_rate"] for r in reports) / len(reports) if reports else 0.0,
            }
        }
        
        out_path.write_text(json.dumps(summary, indent=2))
        print(f"  Written: {out_path}")


# ============================================================================
# Main Entry Point
# ============================================================================

def main() -> int:
    """Main entry point"""
    orchestrator = WorkflowOrchestrator()
    return orchestrator.run()


if __name__ == "__main__":
    raise SystemExit(main())
