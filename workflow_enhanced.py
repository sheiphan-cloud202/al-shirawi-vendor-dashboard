"""
Enhanced Multi-Agent Workflow: Per-Vendor BOQ Comparison

This workflow creates ONE CSV per vendor with:
- All BOQ requirements (left columns)
- Vendor's specific quotes (right columns)
- LLM-powered deep understanding and alignment

Each line is analyzed by LLM to ensure proper matching.
"""

import csv
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any

import boto3
import pandas as pd


# ============================================================================
# Bedrock LLM Client (Enhanced)
# ============================================================================

class EnhancedBedrockClient:
    """Enhanced Bedrock client with detailed prompt engineering"""
    
    def __init__(self):
        # Default to provided AWS CLI profile if not set
        profile = os.getenv("AWS_PROFILE", "thinktank")
        region = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION", "us-east-1")
        session = boto3.Session(profile_name=profile, region_name=region)
        self.client = session.client("bedrock-runtime")
        self.model_id = os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-5-sonnet-20240620-v1:0")
        self.use_bedrock = os.getenv("BEDROCK_DISABLE") != "1"
    
    def invoke(self, prompt: str, system_prompt: Optional[str] = None, 
               max_tokens: int = 4096, temperature: float = 0.0) -> str:
        """Invoke Bedrock model"""
        if not self.use_bedrock:
            return "{}"
        
        try:
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
            
            body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": messages,
            }
            
            if system_prompt:
                body["system"] = system_prompt
            
            response = self.client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(body)
            )
            
            payload = response.get("body")
            if hasattr(payload, "read"):
                data = json.loads(payload.read())
            else:
                data = json.loads(payload)
            
            content = data.get("content", [])
            text = "".join([p.get("text", "") for p in content if p.get("type") == "text"])
            return text.strip()
        
        except Exception as e:
            print(f"      ⚠️  Bedrock error: {e}")
            return "{}"
    
    def extract_json(self, text: str) -> Optional[Dict]:
        """Extract JSON from LLM response"""
        try:
            return json.loads(text)
        except Exception:
            pass
        
        # Try to extract from code fence
        if "```" in text:
            parts = text.split("```")
            for part in parts:
                lines = part.strip().splitlines()
                if lines and lines[0].strip().lower() in ("json", ""):
                    candidate = "\n".join(lines[1:] if lines[0].strip() else lines)
                    try:
                        return json.loads(candidate)
                    except Exception:
                        continue
        
        return None


# ============================================================================
# Data Models
# ============================================================================

@dataclass
class BOQLineUnderstanding:
    """LLM's deep understanding of a BOQ line item"""
    sr_no: str
    raw_description: str
    
    # LLM extracted semantics
    item_type: str  # e.g., "cable tray", "bend", "tee"
    dimensions: Dict[str, str]  # e.g., {"width": "900mm", "thickness": "2.0mm"}
    material: str  # e.g., "HDG" (Hot Dip Galvanized)
    quantity: float
    uom: str
    key_specs: List[str]  # Other important specs
    search_keywords: List[str]  # Keywords for matching
    
    reasoning: str  # LLM's reasoning about the item


@dataclass
class VendorLineUnderstanding:
    """LLM's deep understanding of a vendor quote line"""
    row_index: int
    raw_description: str
    
    # LLM extracted semantics
    item_type: str
    dimensions: Dict[str, str]
    material: str
    quantity: Optional[float]
    uom: Optional[str]
    unit_price: Optional[float]
    total_price: Optional[float]
    brand: Optional[str]
    key_specs: List[str]
    search_keywords: List[str]
    
    reasoning: str


@dataclass
class AlignedMatch:
    """BOQ line aligned with vendor quote"""
    boq_line: BOQLineUnderstanding
    vendor_line: Optional[VendorLineUnderstanding]
    
    match_confidence: float
    match_reasoning: str
    
    # Comparison insights
    price_available: bool
    qty_matches: bool
    uom_matches: bool
    spec_matches: bool
    
    issues: List[str]  # Any discrepancies found


# ============================================================================
# Agent 1: BOQ Understanding Agent
# ============================================================================

class BOQUnderstandingAgent:
    """Uses LLM to deeply understand each BOQ line item"""
    
    def __init__(self, bedrock: EnhancedBedrockClient):
        self.bedrock = bedrock
    
    def understand_boq_line(self, sr_no: str, description: str, qty: float, uom: str) -> BOQLineUnderstanding:
        """Use LLM to extract semantic understanding from BOQ line"""
        
        system_prompt = """You are an expert in construction materials and BOQ (Bill of Quantities) analysis.
                        Your job is to deeply understand BOQ line items and extract structured semantic information."""
                                
        prompt = f"""Analyze this BOQ line item and extract detailed semantic information.

                    BOQ Line:
                    Sr. No: {sr_no}
                    Description: {description}
                    Quantity: {qty}
                    UOM: {uom}

                    Extract and return ONLY valid JSON:
                    {{
                    "item_type": "cable tray | bend | tee | cover | accessory | ...",
                    "dimensions": {{
                        "width": "900mm or null",
                        "height": "50mm or null", 
                        "thickness": "2.0mm or null",
                        "length": "3m or null"
                    }},
                    "material": "HDG | GI | SS304 | Aluminum | ...",
                    "quantity": {qty},
                    "uom": "{uom}",
                    "key_specs": ["spec1", "spec2", ...],
                    "search_keywords": ["keyword1", "keyword2", ...],
                    "reasoning": "Brief explanation of what this item is"
                    }}

                    Important:
                    - Extract ALL dimensions mentioned (width, height, thickness, length)
                    - Identify material type (HDG = Hot Dip Galvanized)
                    - List key specifications beyond dimensions
                    - Generate keywords for fuzzy matching
                    - Be precise with units (mm, m, inch, etc.)
                """
        
        response = self.bedrock.invoke(prompt, system_prompt=system_prompt, max_tokens=1024)
        result = self.bedrock.extract_json(response)
        
        if result:
            try:
                parsed_qty = float(result.get("quantity") or qty)
            except (ValueError, TypeError):
                parsed_qty = qty
            
            return BOQLineUnderstanding(
                sr_no=sr_no,
                raw_description=description,
                item_type=result.get("item_type") or "unknown",
                dimensions=result.get("dimensions") or {},
                material=result.get("material") or "",
                quantity=parsed_qty,
                uom=result.get("uom") or uom,
                key_specs=result.get("key_specs") or [],
                search_keywords=result.get("search_keywords") or [],
                reasoning=result.get("reasoning") or ""
            )
        else:
            # Fallback heuristic
            return self._fallback_understanding(sr_no, description, qty, uom)
    
    def _fallback_understanding(self, sr_no: str, description: str, qty: float, uom: str) -> BOQLineUnderstanding:
        """Heuristic fallback when LLM unavailable"""
        desc_lower = description.lower()
        
        # Extract dimensions
        dimensions = {}
        dim_matches = re.findall(r'(\d+\.?\d*)\s*(?:x\s*(\d+\.?\d*))?\s*mm', desc_lower)
        if dim_matches and dim_matches[0][0]:
            dimensions["width"] = f"{dim_matches[0][0]}mm"
            if dim_matches[0][1]:
                dimensions["height"] = f"{dim_matches[0][1]}mm"
        
        thickness_match = re.search(r'(\d+\.?\d*)\s*mm\s+th', desc_lower)
        if thickness_match:
            dimensions["thickness"] = f"{thickness_match.group(1)}mm"
        
        # Identify item type
        item_type = "unknown"
        if "cable tray" in desc_lower and "cover" not in desc_lower:
            item_type = "cable_tray"
        elif "cover" in desc_lower:
            item_type = "cable_tray_cover"
        elif "bend" in desc_lower:
            item_type = "bend"
        elif "tee" in desc_lower:
            item_type = "tee"
        
        # Material
        material = ""
        if "hdg" in desc_lower:
            material = "HDG"
        elif " gi " in desc_lower or desc_lower.startswith("gi "):
            material = "GI"
        
        # Keywords
        keywords = [w for w in re.findall(r'\w+', desc_lower) if len(w) > 2]
        
        return BOQLineUnderstanding(
            sr_no=sr_no,
            raw_description=description,
            item_type=item_type,
            dimensions=dimensions,
            material=material,
            quantity=qty,
            uom=uom,
            key_specs=[],
            search_keywords=keywords[:10],
            reasoning="Heuristic extraction"
        )


# ============================================================================
# Agent 2: Vendor Quote Understanding Agent
# ============================================================================

class VendorQuoteUnderstandingAgent:
    """Uses LLM to deeply understand each vendor quote line"""
    
    def __init__(self, bedrock: EnhancedBedrockClient):
        self.bedrock = bedrock
    
    def understand_vendor_line(self, row_index: int, row_data: Dict[str, Any]) -> Optional[VendorLineUnderstanding]:
        """Use LLM to extract semantic understanding from vendor quote line"""
        
        # Skip if no meaningful description
        # Try case-insensitive column matching
        description = ""
        for key, value in row_data.items():
            key_lower = str(key).lower().strip()
            if 'description' in key_lower or 'desc' in key_lower or 'item' in key_lower:
                if value and str(value).strip() and len(str(value).strip()) >= 5:
                    description = str(value).strip()
                    break
        
        if not description:
            return None
        
        system_prompt = """You are an expert in construction materials and vendor quotations.
                            Extract detailed semantic information from vendor quote lines."""
        
        prompt = f"""Analyze this vendor quotation line and extract structured information.

                    Vendor Quote Line:
                    {json.dumps(row_data, indent=2)}

                    Extract and return ONLY valid JSON:
                    {{
                    "item_type": "cable tray | bend | tee | cover | accessory | ...",
                    "dimensions": {{
                        "width": "900mm or null",
                        "height": "50mm or null",
                        "thickness": "2.0mm or null",
                        "length": "3m or null"
                    }},
                    "material": "HDG | GI | SS304 | ...",
                    "quantity": numeric_value_or_null,
                    "uom": "Mtr | Lth | NOS | ...",
                    "unit_price": numeric_value_or_null,
                    "total_price": numeric_value_or_null,
                    "brand": "brand_name or null",
                    "key_specs": ["spec1", "spec2", ...],
                    "search_keywords": ["keyword1", "keyword2", ...],
                    "reasoning": "What this item is"
                    }}

                    Important:
                    - Extract ALL numeric values (qty, prices) from any column
                    - Identify dimensions from description
                    - Material type identification
                    - Parse UOM variations (Mtr/Lth/NOS/Nos/etc.)
                """
        
        response = self.bedrock.invoke(prompt, system_prompt=system_prompt, max_tokens=1024)
        result = self.bedrock.extract_json(response)
        
        if result:
            return VendorLineUnderstanding(
                row_index=row_index,
                raw_description=str(description),
                item_type=result.get("item_type") or "unknown",
                dimensions=result.get("dimensions") or {},
                material=result.get("material") or "",
                quantity=self._parse_float(result.get("quantity")),
                uom=result.get("uom"),
                unit_price=self._parse_float(result.get("unit_price")),
                total_price=self._parse_float(result.get("total_price")),
                brand=result.get("brand"),
                key_specs=result.get("key_specs") or [],
                search_keywords=result.get("search_keywords") or [],
                reasoning=result.get("reasoning") or ""
            )
        else:
            # Fallback heuristic
            return self._fallback_understanding(row_index, description, row_data)
    
    def _parse_float(self, value: Any) -> Optional[float]:
        """Parse float from various formats"""
        if value is None:
            return None
        try:
            if isinstance(value, (int, float)):
                return float(value)
            clean = str(value).replace(",", "").strip()
            return float(clean) if clean else None
        except Exception:
            return None
    
    def _fallback_understanding(self, row_index: int, description: str, row_data: Dict) -> VendorLineUnderstanding:
        """Heuristic fallback"""
        desc_lower = str(description).lower()
        
        # Extract dimensions
        dimensions = {}
        dim_matches = re.findall(r'(\d+\.?\d*)\s*(?:x\s*(\d+\.?\d*))?\s*mm', desc_lower)
        if dim_matches and dim_matches[0][0]:
            dimensions["width"] = f"{dim_matches[0][0]}mm"
            if dim_matches[0][1]:
                dimensions["height"] = f"{dim_matches[0][1]}mm"
        
        # Item type
        item_type = "unknown"
        if "cable tray" in desc_lower and "cover" not in desc_lower:
            item_type = "cable_tray"
        elif "cover" in desc_lower:
            item_type = "cable_tray_cover"
        elif "bend" in desc_lower:
            item_type = "bend"
        
        # Material
        material = ""
        if "hdg" in desc_lower:
            material = "HDG"
        
        # Try to extract prices from row data
        unit_price = None
        total_price = None
        quantity = None
        
        for key, val in row_data.items():
            key_lower = str(key).lower()
            if "unit" in key_lower and "price" in key_lower:
                unit_price = self._parse_float(val)
            elif "total" in key_lower and "price" in key_lower:
                total_price = self._parse_float(val)
            elif "qty" in key_lower or "quantity" in key_lower:
                quantity = self._parse_float(val)
        
        keywords = [w for w in re.findall(r'\w+', desc_lower) if len(w) > 2]
        
        return VendorLineUnderstanding(
            row_index=row_index,
            raw_description=description,
            item_type=item_type,
            dimensions=dimensions,
            material=material,
            quantity=quantity,
            uom=None,
            unit_price=unit_price,
            total_price=total_price,
            brand=None,
            key_specs=[],
            search_keywords=keywords[:10],
            reasoning="Heuristic extraction"
        )


# ============================================================================
# Agent 3: Intelligent Alignment Agent
# ============================================================================

class IntelligentAlignmentAgent:
    """Uses LLM to intelligently align BOQ items to vendor quotes"""
    
    def __init__(self, bedrock: EnhancedBedrockClient):
        self.bedrock = bedrock
    
    def find_best_match(self, boq_line: BOQLineUnderstanding, 
                       vendor_lines: List[VendorLineUnderstanding]) -> AlignedMatch:
        """Find best matching vendor line for BOQ item"""
        
        if not vendor_lines:
            return self._create_missing_match(boq_line)
        
        # Pre-filter candidates using heuristics
        candidates = self._prefilter_candidates(boq_line, vendor_lines, top_k=5)
        
        if not candidates:
            return self._create_missing_match(boq_line)
        
        # Use LLM for final alignment decision
        if self.bedrock.use_bedrock and len(candidates) > 1:
            best_match = self._llm_alignment(boq_line, candidates)
        else:
            best_match = candidates[0]  # Take top candidate
        
        # Validate the match
        return self._create_aligned_match(boq_line, best_match)
    
    def _prefilter_candidates(self, boq_line: BOQLineUnderstanding,
                             vendor_lines: List[VendorLineUnderstanding],
                             top_k: int = 5) -> List[VendorLineUnderstanding]:
        """Pre-filter using heuristic scoring"""
        
        scored = []
        
        for vline in vendor_lines:
            score = 0.0
            
            # Item type match
            if boq_line.item_type == vline.item_type:
                score += 3.0
            
            # Dimension matches (most important for cable trays)
            for dim_key in ["width", "height", "thickness"]:
                boq_dim = (boq_line.dimensions.get(dim_key, "") or "").lower()
                vendor_dim = (vline.dimensions.get(dim_key, "") or "").lower()
                if boq_dim and vendor_dim:
                    # Extract numeric values for comparison
                    boq_num = re.search(r'(\d+)', boq_dim)
                    vendor_num = re.search(r'(\d+)', vendor_dim)
                    if boq_num and vendor_num:
                        if boq_num.group(1) == vendor_num.group(1):
                            # Exact dimension match - VERY important
                            score += 5.0
                        elif dim_key == "width":  # Width is critical
                            # Penalize width mismatch heavily
                            score -= 3.0
            
            # Material match
            if boq_line.material and vline.material:
                if boq_line.material.upper() == vline.material.upper():
                    score += 1.5
            
            # Keyword overlap
            boq_kw = set(boq_line.search_keywords)
            vendor_kw = set(vline.search_keywords)
            overlap = len(boq_kw & vendor_kw)
            score += 0.1 * overlap
            
            if score > 3.0:  # Minimum threshold (raised for better quality)
                scored.append((vline, score))
        
        # Sort by score and return top-k
        scored.sort(key=lambda x: x[1], reverse=True)
        return [vline for vline, _ in scored[:top_k]]
    
    def _llm_alignment(self, boq_line: BOQLineUnderstanding,
                      candidates: List[VendorLineUnderstanding]) -> VendorLineUnderstanding:
        """Use LLM to select best match from candidates"""
        
        system_prompt = """You are an expert in matching BOQ requirements to vendor quotations.
                            Select the vendor line that best matches the BOQ requirement."""
                                    
        candidates_text = "\n".join([
                            f"""Candidate {i+1}:
                                Description: {c.raw_description}
                                Type: {c.item_type}
                                Dimensions: {c.dimensions}
                                Material: {c.material}
                                Qty: {c.quantity} {c.uom}
                                Price: {c.unit_price} (unit), {c.total_price} (total)"""
                                            for i, c in enumerate(candidates)
                            ])
        
        prompt = f"""Match the BOQ requirement to the best vendor candidate.

                    BOQ Requirement:
                    Sr.No: {boq_line.sr_no}
                    Description: {boq_line.raw_description}
                    Type: {boq_line.item_type}
                    Dimensions: {boq_line.dimensions}
                    Material: {boq_line.material}
                    Qty: {boq_line.quantity} {boq_line.uom}

                    Vendor Candidates:
                    {candidates_text}

                    Return ONLY JSON:
                    {{
                    "best_candidate_index": 0,
                    "confidence": 0.95,
                    "reasoning": "Why this is the best match"
                    }}

                    Select based on:
                    1. Item type match
                    2. Dimension match (most important)
                    3. Material match
                    4. Specification similarity
                """
        
        response = self.bedrock.invoke(prompt, system_prompt=system_prompt, max_tokens=512)
        result = self.bedrock.extract_json(response)
        
        if result and result.get("best_candidate_index") is not None:
            try:
                idx = int(result.get("best_candidate_index", 0))
                if 0 <= idx < len(candidates):
                    return candidates[idx]
            except (ValueError, TypeError):
                pass
        
        return candidates[0]  # Fallback to first
    
    def _create_aligned_match(self, boq_line: BOQLineUnderstanding,
                             vendor_line: VendorLineUnderstanding) -> AlignedMatch:
        """Create aligned match with validation"""
        
        issues = []
        
        # Check quantity match
        qty_matches = False
        if vendor_line.quantity:
            variance = abs(vendor_line.quantity - boq_line.quantity) / boq_line.quantity
            qty_matches = variance < 0.1
            if not qty_matches:
                issues.append(f"Qty variance: BOQ={boq_line.quantity}, Vendor={vendor_line.quantity}")
        
        # Check UOM
        uom_matches = False
        if vendor_line.uom:
            uom_matches = boq_line.uom.lower() in vendor_line.uom.lower() or \
                         vendor_line.uom.lower() in boq_line.uom.lower()
            if not uom_matches:
                issues.append(f"UOM mismatch: BOQ={boq_line.uom}, Vendor={vendor_line.uom}")
        
        # Check specs
        spec_matches = boq_line.item_type == vendor_line.item_type
        if not spec_matches:
            issues.append(f"Type mismatch: BOQ={boq_line.item_type}, Vendor={vendor_line.item_type}")
        
        # Calculate confidence
        confidence = 0.5
        if boq_line.item_type == vendor_line.item_type:
            confidence += 0.2
        if qty_matches:
            confidence += 0.15
        if uom_matches:
            confidence += 0.1
        if not issues:
            confidence += 0.05
        
        return AlignedMatch(
            boq_line=boq_line,
            vendor_line=vendor_line,
            match_confidence=min(confidence, 1.0),
            match_reasoning=vendor_line.reasoning,
            price_available=vendor_line.unit_price is not None,
            qty_matches=qty_matches,
            uom_matches=uom_matches,
            spec_matches=spec_matches,
            issues=issues
        )
    
    def _create_missing_match(self, boq_line: BOQLineUnderstanding) -> AlignedMatch:
        """Create match for missing vendor quote"""
        return AlignedMatch(
            boq_line=boq_line,
            vendor_line=None,
            match_confidence=0.0,
            match_reasoning="No vendor quote found",
            price_available=False,
            qty_matches=False,
            uom_matches=False,
            spec_matches=False,
            issues=["Item not quoted by vendor"]
        )


# ============================================================================
# Main Workflow Orchestrator
# ============================================================================

class EnhancedWorkflowOrchestrator:
    """Enhanced workflow that creates per-vendor comparison CSVs"""
    
    def __init__(self, use_bedrock: bool = False):
        self.bedrock = EnhancedBedrockClient()
        self.boq_agent = BOQUnderstandingAgent(self.bedrock)
        self.vendor_agent = VendorQuoteUnderstandingAgent(self.bedrock)
        self.alignment_agent = IntelligentAlignmentAgent(self.bedrock)
        
        self.data_dir = Path("/Users/sheiphanjoseph/Desktop/Developer/al_shirawi_orc_poc/data")
        self.out_dir = Path("/Users/sheiphanjoseph/Desktop/Developer/al_shirawi_orc_poc/out")
        self.inquiry_csv = self.out_dir / "inquiry_csv" / "FINAL.csv"
        self.textract_csv_dir = self.out_dir / "textract_csv"
        self.comparison_dir = self.out_dir / "vendor_comparisons"
    
    def run(self) -> int:
        """Execute enhanced workflow"""
        
        print("=" * 80)
        print("Enhanced Per-Vendor BOQ Comparison Workflow")
        print("=" * 80)
        
        # Step 1: Load and understand BOQ requirements
        print("\n[Step 1] 🧠 Understanding BOQ Requirements with LLM...")
        boq_lines = self._load_and_understand_boq()
        print(f"  ✓ Understood {len(boq_lines)} BOQ line items")
        
        # Step 2: Process each vendor
        print("\n[Step 2] 🏭 Processing Vendor Quotations...")
        vendor_csvs = list(self.textract_csv_dir.glob("*.csv"))
        print(f"  Found {len(vendor_csvs)} vendor CSV files\n")
        
        self.comparison_dir.mkdir(parents=True, exist_ok=True)
        
        for csv_path in vendor_csvs:
            vendor_name = self._extract_vendor_name(csv_path.name)
            print(f"  Processing: {vendor_name}")
            
            # Step 2a: Understand vendor lines
            print("    🧠 Understanding vendor quotes with LLM...")
            vendor_lines = self._load_and_understand_vendor(csv_path)
            print(f"      ✓ Understood {len(vendor_lines)} vendor line items")
            
            # Step 2b: Align BOQ to vendor
            print("    🔗 Aligning BOQ to vendor quotes...")
            alignments = []
            for boq_line in boq_lines:
                match = self.alignment_agent.find_best_match(boq_line, vendor_lines)
                alignments.append(match)
            
            matched = sum(1 for a in alignments if a.match_confidence > 0.5)
            print(f"      ✓ Aligned {matched}/{len(alignments)} items (confidence > 0.5)")
            
            # Step 2c: Generate per-vendor CSV
            print("    📄 Generating comparison CSV...")
            output_path = self._write_vendor_comparison_csv(vendor_name, alignments)
            print(f"      ✓ Written: {output_path.name}\n")
        
        print("=" * 80)
        print("✅ Workflow completed successfully!")
        print(f"📂 Output directory: {self.comparison_dir}")
        print("=" * 80)
        
        return 0
    
    def _load_and_understand_boq(self) -> List[BOQLineUnderstanding]:
        """Load BOQ and use LLM to understand each line"""
        
        if not self.inquiry_csv.exists():
            print(f"  ❌ Error: BOQ CSV not found at {self.inquiry_csv}")
            return []
        
        df = pd.read_csv(self.inquiry_csv)
        boq_lines = []
        
        for idx, row in df.iterrows():
            sr_no = str(row.get("Sr. No", "")).strip()
            description = str(row.get("Description", "")).strip()
            qty_str = str(row.get("TOTAL Qty in Length / Nos", "")).strip()
            uom = str(row.get("UOM", "")).strip()
            
            # Skip headers and invalid rows
            if not description or not sr_no or sr_no.lower() in ("sr. no", "sr.no", "cable"):
                continue
            
            # Skip section headers (no quantity)
            if not qty_str or pd.isna(qty_str) or str(qty_str).strip() in ("", "nan"):
                continue
            
            # Parse quantity
            try:
                qty = float(str(qty_str).replace(",", ""))
            except Exception:
                continue
            
            # Use LLM to understand this line (only every 3rd for efficiency)
            if idx % 3 == 0:
                print(f"    🤖 LLM analyzing: {sr_no} - {description[:50]}...")
            
            understanding = self.boq_agent.understand_boq_line(sr_no, description, qty, uom)
            boq_lines.append(understanding)
        
        return boq_lines
    
    def _load_and_understand_vendor(self, csv_path: Path) -> List[VendorLineUnderstanding]:
        """Load vendor CSV and use LLM to understand each line"""
        
        try:
            # Read raw lines to find header row (more reliable than pandas with inconsistent columns)
            import csv
            header_row = None
            
            with open(csv_path, 'r', encoding='utf-8') as f:
                csv_reader = csv.reader(f)
                for line_num, row in enumerate(csv_reader):
                    if line_num > 50:  # Don't search beyond first 50 lines
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
                    
                    # Must meet all criteria
                    if len(cells) >= 3 and has_desc and price_indicators >= 2 and cells_short:
                        header_row = line_num
                        print(f"      ✓ Found header at line {line_num}: {cells[:5]}...")
                        break
            
            # Read with detected header or default
            if header_row is not None:
                # Use skiprows instead of header for more reliable parsing with metadata rows
                df = pd.read_csv(csv_path, skiprows=header_row, encoding='utf-8', engine='python')
            else:
                df = pd.read_csv(csv_path, encoding='utf-8', engine='python')
        except Exception:
            try:
                df = pd.read_csv(csv_path, on_bad_lines='skip', encoding='latin-1', engine='python')
            except Exception as e:
                print(f"      ❌ Error reading CSV: {e}")
                return []
        
        vendor_lines = []
        
        for idx, row in df.iterrows():
            row_data = row.to_dict()
            
            # Skip rows that are all NaN
            if all(pd.isna(v) or str(v).strip() == '' for v in row_data.values()):
                continue
            
            # Use LLM to understand (sample every 5th)
            if idx % 5 == 0:
                # Case-insensitive description lookup for debug print
                desc = ""
                for key, value in row_data.items():
                    if 'desc' in str(key).lower() or 'item' in str(key).lower():
                        desc = str(value)[:40]
                        break
                print(f"      🤖 LLM analyzing row {idx}: {desc}...")
            
            understanding = self.vendor_agent.understand_vendor_line(idx, row_data)
            if understanding:
                vendor_lines.append(understanding)
        
        return vendor_lines
    
    def _extract_vendor_name(self, filename: str) -> str:
        """Extract clean vendor name from filename"""
        # Response_3_-_IKKT__AL SHIRAWI ELECTRICAL - MEP.tables.csv
        parts = filename.replace(".tables.csv", "").replace(".csv", "").split("__")
        if len(parts) >= 1:
            vendor_part = parts[0].replace("Response_", "").replace("_-_", " - ").replace("_", " ")
            return vendor_part
        return filename
    
    def _write_vendor_comparison_csv(self, vendor_name: str, alignments: List[AlignedMatch]) -> Path:
        """Write per-vendor comparison CSV"""
        
        safe_name = vendor_name.replace("/", "_").replace("\\", "_").replace(" ", "_")
        output_path = self.comparison_dir / f"{safe_name}_comparison.csv"
        
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow([
                "BOQ Sr.No",
                "BOQ Description",
                "BOQ Qty",
                "BOQ UOM",
                "BOQ Item Type",
                "BOQ Dimensions",
                "BOQ Material",
                "",  # Separator
                "Vendor Description",
                "Vendor Qty",
                "Vendor UOM",
                "Vendor Unit Price",
                "Vendor Total Price",
                "Vendor Brand",
                "",  # Separator
                "Match Confidence",
                "Match Status",
                "Issues",
                "LLM Reasoning"
            ])
            
            # Data rows
            for alignment in alignments:
                boq = alignment.boq_line
                vendor = alignment.vendor_line
                
                # BOQ columns
                boq_dims = ", ".join(f"{k}:{v}" for k, v in boq.dimensions.items() if v)
                
                # Vendor columns
                if vendor:
                    vendor_desc = vendor.raw_description
                    vendor_qty = vendor.quantity or ""
                    vendor_uom = vendor.uom or ""
                    vendor_unit_price = vendor.unit_price or ""
                    vendor_total_price = vendor.total_price or ""
                    vendor_brand = vendor.brand or ""
                else:
                    vendor_desc = "NOT QUOTED"
                    vendor_qty = vendor_uom = vendor_unit_price = vendor_total_price = vendor_brand = ""
                
                # Status
                if alignment.match_confidence >= 0.8:
                    status = "✅ EXCELLENT"
                elif alignment.match_confidence >= 0.5:
                    status = "✓ GOOD"
                elif alignment.match_confidence >= 0.3:
                    status = "⚠ FAIR"
                else:
                    status = "❌ MISSING/POOR"
                
                issues_str = "; ".join(alignment.issues) if alignment.issues else "None"
                
                writer.writerow([
                    boq.sr_no,
                    boq.raw_description,
                    boq.quantity,
                    boq.uom,
                    boq.item_type,
                    boq_dims,
                    boq.material,
                    "",  # Separator
                    vendor_desc,
                    vendor_qty,
                    vendor_uom,
                    vendor_unit_price,
                    vendor_total_price,
                    vendor_brand,
                    "",  # Separator
                    f"{alignment.match_confidence:.2f}",
                    status,
                    issues_str,
                    alignment.match_reasoning
                ])
        
        return output_path


# ============================================================================
# Main Entry Point
# ============================================================================

def main() -> int:
    """Main entry point"""
    orchestrator = EnhancedWorkflowOrchestrator()
    return orchestrator.run()


if __name__ == "__main__":
    raise SystemExit(main())
