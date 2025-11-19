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
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import boto3
import pandas as pd


# ============================================================================
# Bedrock LLM Client (Enhanced)
# ============================================================================

class EnhancedBedrockClient:
    """Enhanced Bedrock client with detailed prompt engineering and batch processing"""
    
    def __init__(self, max_parallel_calls: int = 5):
        # Default to provided AWS CLI profile if not set
        profile = os.getenv("AWS_PROFILE", "thinktank")
        region = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION", "us-east-1")
        session = boto3.Session(profile_name=profile, region_name=region)
        self.client = session.client("bedrock-runtime")
        self.model_id = os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-5-sonnet-20240620-v1:0")
        self.use_bedrock = os.getenv("BEDROCK_DISABLE") != "1"
        self.max_parallel_calls = max_parallel_calls
        # Rate limiting
        self.request_delay = 0.1  # 100ms delay between batches to avoid throttling
    
    def invoke_structured(self, prompt: str, json_schema: Dict, 
                         tool_name: str = "extract_data",
                         tool_description: str = "Extract structured data from the input",
                         max_tokens: int = 4096, 
                         temperature: float = 0.0,
                         max_retries: int = 3) -> Optional[Dict]:
        """
        Invoke Bedrock with structured output using Tool Use (Converse API) with retry logic.
        This provides native JSON schema support for reliable structured responses.
        Reference: https://aws.amazon.com/blogs/machine-learning/structured-data-response-with-amazon-bedrock-prompt-engineering-and-tool-use/
        
        Args:
            prompt: The user prompt/question
            json_schema: JSON schema defining the expected output structure
            tool_name: Name of the tool to use
            tool_description: Description of what the tool does
            max_tokens: Maximum tokens for response
            temperature: Sampling temperature (0.0 for deterministic)
            max_retries: Maximum number of retries on throttling (default: 3)
        
        Returns:
            Parsed JSON dict matching the schema, or None on error
        """
        if not self.use_bedrock:
            return None
        
        # Define tool with JSON schema
        tool_config = {
            "tools": [
                {
                    "toolSpec": {
                        "name": tool_name,
                        "description": tool_description,
                        "inputSchema": {
                            "json": json_schema
                        }
                    }
                }
            ]
        }
        
        # Prepare message
        messages = [
            {
                "role": "user",
                "content": [{"text": prompt}]
            }
        ]
        
        # Retry loop with exponential backoff
        for attempt in range(max_retries + 1):
            try:
                # Call Converse API with tool use
                response = self.client.converse(
                    modelId=self.model_id,
                    messages=messages,
                    toolConfig=tool_config,
                    inferenceConfig={
                        "maxTokens": max_tokens,
                        "temperature": temperature
                    }
                )
                
                # Extract tool use result
                output_message = response.get("output", {}).get("message", {})
                content = output_message.get("content", [])
                
                # Find tool use in response
                for item in content:
                    if "toolUse" in item:
                        tool_use = item["toolUse"]
                        # Return the structured input data (which is our output)
                        return tool_use.get("input", {})
                
                # Fallback: try to extract text response
                for item in content:
                    if "text" in item:
                        return self.extract_json(item["text"])
                
                return None
                
            except Exception as e:
                error_msg = str(e)
                # Check if it's a throttling error
                if "ThrottlingException" in error_msg or "Too many requests" in error_msg:
                    if attempt < max_retries:
                        # Exponential backoff: 1s, 2s, 4s, 8s...
                        wait_time = 2 ** attempt
                        print(f"      ⚠️  Throttled, waiting {wait_time}s before retry {attempt + 1}/{max_retries}...")
                        time.sleep(wait_time)
                        continue
                    else:
                        print("      ❌ Max retries reached for throttling error")
                        return None
                else:
                    # Non-throttling error, don't retry
                    print(f"      ⚠️  Bedrock structured invoke error: {e}")
                    return None
        
        return None
    
    def invoke(self, prompt: str, system_prompt: Optional[str] = None, 
               max_tokens: int = 4096, temperature: float = 0.0) -> str:
        """Invoke Bedrock model (legacy text-based method)"""
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
        
        # Try to extract JSON embedded in text (find first { to last })
        start_idx = text.find('{')
        if start_idx != -1:
            # Find the matching closing brace
            end_idx = text.rfind('}')
            if end_idx != -1 and end_idx > start_idx:
                candidate = text[start_idx:end_idx + 1]
                try:
                    return json.loads(candidate)
                except Exception:
                    pass
        
        return None
    
    def invoke_structured_batch(self, 
                               requests: List[Tuple[str, Dict, str, str]], 
                               max_tokens: int = 4096, 
                               temperature: float = 0.0,
                               progress_label: str = "Processing") -> List[Optional[Dict]]:
        """
        Invoke Bedrock with structured output in parallel for multiple requests with progress tracking.
        
        Args:
            requests: List of (prompt, json_schema, tool_name, tool_description) tuples
            max_tokens: Maximum tokens for each response
            temperature: Sampling temperature
            progress_label: Label for progress messages
        
        Returns:
            List of parsed JSON dicts matching schemas, in same order as requests
        """
        if not self.use_bedrock or not requests:
            return [None] * len(requests)
        
        results = [None] * len(requests)
        completed_count = 0
        total_count = len(requests)
        lock = threading.Lock()
        
        def process_single(index: int, prompt: str, json_schema: Dict, 
                          tool_name: str, tool_description: str) -> Tuple[int, Optional[Dict]]:
            """Process a single request and return (index, result)"""
            nonlocal completed_count
            try:
                result = self.invoke_structured(
                    prompt=prompt,
                    json_schema=json_schema,
                    tool_name=tool_name,
                    tool_description=tool_description,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                
                # Update progress
                with lock:
                    completed_count += 1
                    if completed_count % 5 == 0 or completed_count == total_count:
                        print(f"      📊 {progress_label}: {completed_count}/{total_count} completed")
                
                return (index, result)
            except Exception as e:
                with lock:
                    completed_count += 1
                print(f"      ⚠️  Batch item {index} error: {e}")
                return (index, None)
        
        # Process in parallel using ThreadPoolExecutor with rate limiting
        with ThreadPoolExecutor(max_workers=self.max_parallel_calls) as executor:
            futures = []
            for idx, (prompt, schema, tool_name, tool_desc) in enumerate(requests):
                future = executor.submit(process_single, idx, prompt, schema, tool_name, tool_desc)
                futures.append(future)
                # Small delay to avoid burst throttling
                time.sleep(self.request_delay)
            
            # Collect results
            for future in as_completed(futures):
                try:
                    idx, result = future.result()
                    results[idx] = result
                except Exception as e:
                    print(f"      ⚠️  Batch processing error: {e}")
        
        return results
    
    def invoke_batch(self, 
                    requests: List[Tuple[str, Optional[str]]], 
                    max_tokens: int = 4096, 
                    temperature: float = 0.0) -> List[str]:
        """
        Invoke Bedrock in parallel for multiple text-based requests.
        
        Args:
            requests: List of (prompt, system_prompt) tuples
            max_tokens: Maximum tokens for each response
            temperature: Sampling temperature
        
        Returns:
            List of text responses, in same order as requests
        """
        if not self.use_bedrock or not requests:
            return ["{}"] * len(requests)
        
        results = ["{}"] * len(requests)
        
        def process_single(index: int, prompt: str, system_prompt: Optional[str]) -> Tuple[int, str]:
            """Process a single request and return (index, result)"""
            try:
                result = self.invoke(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                return (index, result)
            except Exception as e:
                print(f"      ⚠️  Batch item {index} error: {e}")
                return (index, "{}")
        
        # Process in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=self.max_parallel_calls) as executor:
            futures = []
            for idx, (prompt, system_prompt) in enumerate(requests):
                future = executor.submit(process_single, idx, prompt, system_prompt)
                futures.append(future)
            
            # Collect results
            for future in as_completed(futures):
                try:
                    idx, result = future.result()
                    results[idx] = result
                except Exception as e:
                    print(f"      ⚠️  Batch processing error: {e}")
        
        return results


# ============================================================================
# Data Models
# ============================================================================

@dataclass
class BOQLineUnderstanding:
    """LLM's deep understanding of a BOQ line item"""
    row_index: int  # DataFrame row index
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
    """Uses LLM to deeply understand each BOQ line item with batch processing support"""
    
    def __init__(self, bedrock: EnhancedBedrockClient):
        self.bedrock = bedrock
        # Define JSON schema for BOQ extraction (shared across all requests)
        self.json_schema = {
            "type": "object",
            "properties": {
                "item_type": {
                    "type": "string",
                    "description": "Type of item: cable tray, bend, tee, cover, accessory, etc."
                },
                "dimensions": {
                    "type": "object",
                    "properties": {
                        "width": {"type": ["string", "null"], "description": "Width with unit (e.g., 900mm)"},
                        "height": {"type": ["string", "null"], "description": "Height with unit (e.g., 50mm)"},
                        "thickness": {"type": ["string", "null"], "description": "Thickness with unit (e.g., 2.0mm)"},
                        "length": {"type": ["string", "null"], "description": "Length with unit (e.g., 3m)"}
                    }
                },
                "material": {
                    "type": "string",
                    "description": "Material type: HDG (Hot Dip Galvanized), GI, SS304, Aluminum, etc."
                },
                "quantity": {
                    "type": "number",
                    "description": "Numeric quantity value"
                },
                "uom": {
                    "type": "string",
                    "description": "Unit of measure: Mtr, Lth, NOS, etc."
                },
                "key_specs": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Key specifications beyond dimensions"
                },
                "search_keywords": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Keywords for fuzzy matching"
                },
                "reasoning": {
                    "type": "string",
                    "description": "Brief explanation of what this item is"
                }
            },
            "required": ["item_type", "dimensions", "material", "quantity", "uom", "search_keywords", "reasoning"]
        }
    
    def understand_boq_line(self, row_index: int, sr_no: str, description: str, qty: float, uom: str) -> BOQLineUnderstanding:
        """Use LLM to extract semantic understanding from BOQ line"""
                                
        prompt = f"""Analyze this BOQ line item and extract detailed semantic information.

BOQ Line:
Sr. No: {sr_no}
Description: {description}
Quantity: {qty}
UOM: {uom}

Important:
- Extract ALL dimensions mentioned (width, height, thickness, length) with units
- Identify material type (HDG = Hot Dip Galvanized, GI, SS304, etc.)
- List key specifications beyond dimensions
- Generate comprehensive keywords for fuzzy matching
- Be precise with units (mm, m, inch, etc.)
"""
        
        result = self.bedrock.invoke_structured(
            prompt=prompt,
            json_schema=self.json_schema,
            tool_name="extract_boq_data",
            tool_description="Extract structured information from BOQ line item",
            max_tokens=1024,
            temperature=0.0
        )
        
        if result:
            try:
                parsed_qty = float(result.get("quantity") or qty)
            except (ValueError, TypeError):
                parsed_qty = qty
            
            return BOQLineUnderstanding(
                row_index=row_index,
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
            return self._fallback_understanding(row_index, sr_no, description, qty, uom)
    
    def _fallback_understanding(self, row_index: int, sr_no: str, description: str, qty: float, uom: str) -> BOQLineUnderstanding:
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
            row_index=row_index,
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
    
    def understand_boq_lines_batch(self, 
                                   boq_data: List[Tuple[int, str, str, float, str]]) -> List[BOQLineUnderstanding]:
        """
        Batch process multiple BOQ lines in parallel using LLM.
        
        Args:
            boq_data: List of (row_index, sr_no, description, qty, uom) tuples
        
        Returns:
            List of BOQLineUnderstanding objects in same order as input
        """
        if not boq_data:
            return []
        
        # Build batch requests
        requests = []
        for row_index, sr_no, description, qty, uom in boq_data:
            prompt = f"""Analyze this BOQ line item and extract detailed semantic information.

BOQ Line:
Sr. No: {sr_no}
Description: {description}
Quantity: {qty}
UOM: {uom}

Important:
- Extract ALL dimensions mentioned (width, height, thickness, length) with units
- Identify material type (HDG = Hot Dip Galvanized, GI, SS304, etc.)
- List key specifications beyond dimensions
- Generate comprehensive keywords for fuzzy matching
- Be precise with units (mm, m, inch, etc.)
"""
            requests.append((prompt, self.json_schema, "extract_boq_data", 
                           "Extract structured information from BOQ line item"))
        
        # Process in parallel with progress tracking
        results = self.bedrock.invoke_structured_batch(
            requests, 
            max_tokens=1024, 
            temperature=0.0,
            progress_label="BOQ lines"
        )
        
        # Convert results to BOQLineUnderstanding objects
        understandings = []
        for i, ((row_index, sr_no, description, qty, uom), result) in enumerate(zip(boq_data, results)):
            if result:
                try:
                    parsed_qty = float(result.get("quantity") or qty)
                except (ValueError, TypeError):
                    parsed_qty = qty
                
                understanding = BOQLineUnderstanding(
                    row_index=row_index,
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
                # Use fallback for failed items
                understanding = self._fallback_understanding(row_index, sr_no, description, qty, uom)
            
            understandings.append(understanding)
        
        return understandings


# ============================================================================
# Agent 2: Vendor Quote Understanding Agent
# ============================================================================

class VendorQuoteUnderstandingAgent:
    """Uses LLM to deeply understand each vendor quote line with batch processing support"""
    
    def __init__(self, bedrock: EnhancedBedrockClient):
        self.bedrock = bedrock
        # Define JSON schema for vendor extraction (shared across all requests)
        self.json_schema = {
            "type": "object",
            "properties": {
                "item_type": {
                    "type": "string",
                    "description": "Type of item: cable tray, bend, tee, cover, accessory, etc."
                },
                "dimensions": {
                    "type": "object",
                    "properties": {
                        "width": {"type": ["string", "null"], "description": "Width with unit (e.g., 900mm)"},
                        "height": {"type": ["string", "null"], "description": "Height with unit (e.g., 50mm)"},
                        "thickness": {"type": ["string", "null"], "description": "Thickness with unit (e.g., 2.0mm)"},
                        "length": {"type": ["string", "null"], "description": "Length with unit (e.g., 3m)"}
                    }
                },
                "material": {
                    "type": "string",
                    "description": "Material type: HDG, GI, SS304, etc."
                },
                "quantity": {
                    "type": ["number", "null"],
                    "description": "Numeric quantity value"
                },
                "uom": {
                    "type": ["string", "null"],
                    "description": "Unit of measure: Mtr, Lth, NOS, etc."
                },
                "unit_price": {
                    "type": ["number", "null"],
                    "description": "Unit price as numeric value"
                },
                "total_price": {
                    "type": ["number", "null"],
                    "description": "Total price as numeric value"
                },
                "brand": {
                    "type": ["string", "null"],
                    "description": "Brand or manufacturer name"
                },
                "key_specs": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Key specifications as list of strings"
                },
                "search_keywords": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Keywords for searching/matching"
                },
                "reasoning": {
                    "type": "string",
                    "description": "Brief explanation of what this item is"
                }
            },
            "required": ["item_type", "dimensions", "material", "search_keywords", "reasoning"]
        }
    
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
        
        prompt = f"""Analyze this vendor quotation line and extract structured information.

Vendor Quote Line:
{json.dumps(row_data, indent=2)}

Important:
- Extract ALL numeric values (qty, prices) from any column
- Identify dimensions from description with units
- Identify material type (HDG, GI, SS304, etc.)
- Parse UOM variations (Mtr/Lth/NOS/Nos/etc.)
- Provide comprehensive search keywords for matching
"""
        
        result = self.bedrock.invoke_structured(
            prompt=prompt,
            json_schema=self.json_schema,
            tool_name="extract_vendor_quote_data",
            tool_description="Extract structured information from vendor quotation line",
            max_tokens=1024,
            temperature=0.0
        )
        
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
    
    def understand_vendor_lines_batch(self, 
                                     vendor_data: List[Tuple[int, Dict[str, Any]]]) -> List[Optional[VendorLineUnderstanding]]:
        """
        Batch process multiple vendor lines in parallel using LLM.
        
        Args:
            vendor_data: List of (row_index, row_data) tuples
        
        Returns:
            List of VendorLineUnderstanding objects (or None) in same order as input
        """
        if not vendor_data:
            return []
        
        # Filter out rows without meaningful descriptions and build requests
        valid_indices = []
        requests = []
        descriptions = []
        
        for idx, (row_index, row_data) in enumerate(vendor_data):
            # Try case-insensitive column matching
            description = ""
            for key, value in row_data.items():
                key_lower = str(key).lower().strip()
                if 'description' in key_lower or 'desc' in key_lower or 'item' in key_lower:
                    if value and str(value).strip() and len(str(value).strip()) >= 5:
                        description = str(value).strip()
                        break
            
            if description:
                valid_indices.append(idx)
                descriptions.append(description)
                
                prompt = f"""Analyze this vendor quotation line and extract structured information.

Vendor Quote Line:
{json.dumps(row_data, indent=2)}

Important:
- Extract ALL numeric values (qty, prices) from any column
- Identify dimensions from description with units
- Identify material type (HDG, GI, SS304, etc.)
- Parse UOM variations (Mtr/Lth/NOS/Nos/etc.)
- Provide comprehensive search keywords for matching
"""
                requests.append((prompt, self.json_schema, "extract_vendor_quote_data", 
                               "Extract structured information from vendor quotation line"))
        
        # Process valid requests in parallel with progress tracking
        if requests:
            results = self.bedrock.invoke_structured_batch(
                requests, 
                max_tokens=1024, 
                temperature=0.0,
                progress_label="Vendor lines"
            )
        else:
            results = []
        
        # Build final result list with None for invalid rows
        understandings: List[Optional[VendorLineUnderstanding]] = [None] * len(vendor_data)
        
        for valid_idx, result_idx in enumerate(valid_indices):
            row_index, row_data = vendor_data[result_idx]
            result = results[valid_idx] if valid_idx < len(results) else None
            description = descriptions[valid_idx]
            
            if result:
                understanding = VendorLineUnderstanding(
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
                # Use fallback for failed items
                understanding = self._fallback_understanding(row_index, description, row_data)
            
            understandings[result_idx] = understanding
        
        return understandings


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
                             top_k: int = 5,
                             window_size: int = 5) -> List[VendorLineUnderstanding]:
        """Pre-filter using heuristic scoring with sliding window based on row_index"""
        
        # Apply sliding window filter based on BOQ row index
        # For row 0: compare with vendor rows 0-4
        # For row 1: compare with vendor rows 1-5
        # For row 2: compare with vendor rows 2-6, etc.
        try:
            boq_row_idx = boq_line.row_index
            start_index = boq_row_idx
            end_index = boq_row_idx + window_size
            
            # Filter vendor lines within the window based on row_index
            windowed_vendor_lines = [
                vline for vline in vendor_lines 
                if start_index <= vline.row_index < end_index
            ]
            
            # If no items in window, fall back to all vendor lines
            if not windowed_vendor_lines:
                windowed_vendor_lines = vendor_lines
            
        except (ValueError, AttributeError):
            # If row_index is missing, use all vendor lines
            windowed_vendor_lines = vendor_lines
        
        scored = []
        
        for vline in windowed_vendor_lines:
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
                            f"""Candidate {i}:
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
    """Enhanced workflow that creates per-vendor comparison CSVs with parallel batch processing"""
    
    def __init__(self, use_bedrock: bool = False, max_workers: Optional[int] = None, 
                 max_parallel_bedrock_calls: int = 5, session_id: Optional[str] = None):
        """
        Initialize the workflow orchestrator with parallel processing capabilities.
        
        Args:
            use_bedrock: Whether to use Bedrock (overridden by BEDROCK_DISABLE env var)
            max_workers: Number of parallel workers for vendor processing (None = auto-detect)
            max_parallel_bedrock_calls: Number of parallel Bedrock API calls per batch (default: 5)
                Set to 5 by default; adjust based on your AWS Bedrock throttling limits.
            session_id: Optional session ID for session-specific output directories
        """
        from src.utils.constants import DATA_DIR, OUT_DIR
        
        self.bedrock = EnhancedBedrockClient(max_parallel_calls=max_parallel_bedrock_calls)
        self.boq_agent = BOQUnderstandingAgent(self.bedrock)
        self.vendor_agent = VendorQuoteUnderstandingAgent(self.bedrock)
        self.alignment_agent = IntelligentAlignmentAgent(self.bedrock)
        
        self.data_dir = DATA_DIR
        self.session_id = session_id
        
        # Use session-specific output directory if session_id is provided
        if session_id:
            from src.core import vendor_logic
            self.out_dir = vendor_logic.get_session_out_dir(session_id)
        else:
            self.out_dir = OUT_DIR
        
        self.inquiry_csv = self.out_dir / "inquiry_csv" / "FINAL.csv"
        self.textract_csv_dir = self.out_dir / "textract_csv"
        self.comparison_dir = self.out_dir / "vendor_comparisons"
        self.cache_dir = self.out_dir / "vendor_cache"
        
        # Thread-safe print lock
        self.print_lock = threading.Lock()
        # Number of parallel workers (None = auto-detect based on CPU count)
        self.max_workers = max_workers
    
    def run(self) -> int:
        """Execute enhanced workflow"""
        
        print("=" * 80)
        print("Enhanced Per-Vendor BOQ Comparison Workflow")
        print("=" * 80)
        
        # Step 1: Load and understand BOQ requirements (with caching)
        print("\n[Step 1] 🧠 Understanding BOQ Requirements with LLM...")
        print(f"  📄 BOQ CSV: {self.inquiry_csv.name}")
        print(f"  📍 Path: {self.inquiry_csv}")
        
        # Try to load from cache first
        boq_lines = self._load_boq_lines_cache()
        
        if boq_lines is None:
            # Cache miss or invalid - process with LLM
            print("  🤖 Processing BOQ with LLM (cache miss)...")
            boq_lines = self._load_and_understand_boq()
            print(f"  ✓ Understood {len(boq_lines)} BOQ line items")
            
            # Save to cache for next time
            self._save_boq_lines_cache(boq_lines)
        else:
            print(f"  ✓ Loaded {len(boq_lines)} BOQ line items from cache")
        
        # Step 2: Process each vendor
        print("\n[Step 2] 🏭 Processing Vendor Quotations...")
        vendor_csvs = list(self.textract_csv_dir.glob("*.csv"))
        print(f"  Found {len(vendor_csvs)} vendor CSV files")
        print("  Files to process:")
        for csv_file in sorted(vendor_csvs):
            print(f"    • {csv_file.name}")
        print()
        
        self.comparison_dir.mkdir(parents=True, exist_ok=True)
        
        # Process vendors in parallel using multithreading
        self._thread_safe_print(f"\n🚀 Processing {len(vendor_csvs)} vendor(s) in parallel...")
        
        # Process vendors in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all vendor processing tasks
            future_to_vendor = {
                executor.submit(self._process_single_vendor, csv_path, boq_lines): csv_path
                for csv_path in vendor_csvs
            }
            
            # Collect results as they complete
            completed = 0
            for future in as_completed(future_to_vendor):
                csv_path = future_to_vendor[future]
                try:
                    future.result()  # Wait for completion and check for exceptions
                    completed += 1
                    vendor_name = self._extract_vendor_name(csv_path.name)
                    self._thread_safe_print(f"  ✓ [{completed}/{len(vendor_csvs)}] Completed: {vendor_name}")
                except Exception as e:
                    vendor_name = self._extract_vendor_name(csv_path.name)
                    self._thread_safe_print(f"  ❌ Error processing {vendor_name}: {e}")
                    import traceback
                    self._thread_safe_print(traceback.format_exc())
        
        # Step 3: Generate consolidated all_comparison table
        print("\n[Step 3] 📊 Generating consolidated comparison table...")
        comparison_csv_path = self._generate_all_comparison_table(boq_lines)
        
        # Step 4: Upload comparison CSV to S3 (if session_id is provided)
        if self.session_id:
            print("\n[Step 4] ☁️  Uploading comparison CSV to S3...")
            try:
                from src.services.s3_service import upload_comparison_to_s3
                s3_info = upload_comparison_to_s3(self.session_id, comparison_csv_path)
                if s3_info:
                    print(f"  ✓ Uploaded to S3: {s3_info['s3_uri']}")
                    print("  📥 Download URL generated (expires in 24 hours)")
                    # Store S3 info in session directory for later retrieval
                    s3_info_file = self.comparison_dir / "s3_upload_info.json"
                    import json
                    with open(s3_info_file, 'w') as f:
                        json.dump(s3_info, f, indent=2)
                    print(f"  ✓ S3 info saved to: {s3_info_file.name}")
                else:
                    print("  ⚠️  S3 upload failed or S3 not configured")
            except Exception as e:
                print(f"  ⚠️  S3 upload error (continuing anyway): {e}")
        
        print("=" * 80)
        print("✅ Workflow completed successfully!")
        print(f"📂 Output directory: {self.comparison_dir}")
        print("=" * 80)
        
        return 0
    
    def _load_and_understand_boq(self) -> List[BOQLineUnderstanding]:
        """Load BOQ and use LLM to understand each line with batch processing"""
        
        if not self.inquiry_csv.exists():
            print(f"  ❌ Error: BOQ CSV not found at {self.inquiry_csv}")
            return []
        
        df = pd.read_csv(self.inquiry_csv)
        
        # Collect all valid BOQ lines
        boq_data = []
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
            
            boq_data.append((idx, sr_no, description, qty, uom))
        
        print(f"    🤖 Processing {len(boq_data)} BOQ lines in parallel batches...")
        
        # Use batch processing for all BOQ lines
        boq_lines = self.boq_agent.understand_boq_lines_batch(boq_data)
        
        return boq_lines
    
    def _load_and_understand_vendor(self, csv_path: Path) -> List[VendorLineUnderstanding]:
        """Load vendor CSV and use LLM to understand each line with batch processing"""
        
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
        
        # Collect all valid vendor lines for batch processing
        vendor_data = []
        for idx, row in df.iterrows():
            row_data = row.to_dict()
            
            # Skip rows that are all NaN
            if all(pd.isna(v) or str(v).strip() == '' for v in row_data.values()):
                continue
            
            vendor_data.append((idx, row_data))
        
        print(f"      🤖 Processing {len(vendor_data)} vendor lines in parallel batches...")
        
        # Use batch processing for all vendor lines
        all_understandings = self.vendor_agent.understand_vendor_lines_batch(vendor_data)
        
        # Filter out None values
        vendor_lines = [u for u in all_understandings if u is not None]
        
        return vendor_lines
    
    def _extract_vendor_name(self, filename: str) -> str:
        """Extract clean vendor name from filename"""
        # Response_3_-_IKKT__AL SHIRAWI ELECTRICAL - MEP.tables.csv
        parts = filename.replace(".tables.csv", "").replace(".csv", "").split("__")
        if len(parts) >= 1:
            vendor_part = parts[0].replace("Response_", "").replace("_-_", " - ").replace("_", " ")
            return vendor_part
        return filename
    
    def _get_cache_path(self, csv_path: Path) -> Path:
        """Get cache file path for a vendor CSV"""
        vendor_name = self._extract_vendor_name(csv_path.name)
        safe_name = vendor_name.replace("/", "_").replace("\\", "_").replace(" ", "_")
        return self.cache_dir / f"{safe_name}_vendor_lines.json"
    
    def _save_vendor_lines_cache(self, csv_path: Path, vendor_lines: List[VendorLineUnderstanding]) -> None:
        """Save vendor lines to cache"""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = self._get_cache_path(csv_path)
        
        # Convert dataclasses to dictionaries for JSON serialization
        cache_data = []
        for line in vendor_lines:
            line_dict = {
                'row_index': line.row_index,
                'raw_description': line.raw_description,
                'item_type': line.item_type,
                'dimensions': line.dimensions,
                'material': line.material,
                'quantity': line.quantity,
                'uom': line.uom,
                'unit_price': line.unit_price,
                'total_price': line.total_price,
                'brand': line.brand,
                'key_specs': line.key_specs,
                'search_keywords': line.search_keywords,
                'reasoning': line.reasoning
            }
            cache_data.append(line_dict)
        
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, indent=2, ensure_ascii=False)
        
        print(f"      💾 Cached vendor lines: {cache_path.name}")
    
    def _load_vendor_lines_cache(self, csv_path: Path) -> Optional[List[VendorLineUnderstanding]]:
        """Load vendor lines from cache if available and valid (non-threaded version)"""
        cache_path = self._get_cache_path(csv_path)
        
        if not cache_path.exists():
            return None
        
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            # Convert dictionaries back to dataclasses
            vendor_lines = []
            for line_dict in cache_data:
                line = VendorLineUnderstanding(
                    row_index=line_dict['row_index'],
                    raw_description=line_dict['raw_description'],
                    item_type=line_dict['item_type'],
                    dimensions=line_dict['dimensions'],
                    material=line_dict['material'],
                    quantity=line_dict['quantity'],
                    uom=line_dict['uom'],
                    unit_price=line_dict['unit_price'],
                    total_price=line_dict['total_price'],
                    brand=line_dict['brand'],
                    key_specs=line_dict['key_specs'],
                    search_keywords=line_dict['search_keywords'],
                    reasoning=line_dict['reasoning']
                )
                vendor_lines.append(line)
            
            print(f"      📂 Loaded {len(vendor_lines)} vendor lines from cache")
            return vendor_lines
            
        except Exception as e:
            print(f"      ⚠️  Cache load error: {e}, regenerating...")
            return None
    
    def _load_vendor_lines_cache_threaded(self, csv_path: Path) -> Optional[List[VendorLineUnderstanding]]:
        """Load vendor lines from cache if available and valid (thread-safe version)"""
        cache_path = self._get_cache_path(csv_path)
        
        if not cache_path.exists():
            return None
        
        try:
            # Use lock for file reading to prevent race conditions
            with self.print_lock:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
            
            # Convert dictionaries back to dataclasses
            vendor_lines = []
            for line_dict in cache_data:
                line = VendorLineUnderstanding(
                    row_index=line_dict['row_index'],
                    raw_description=line_dict['raw_description'],
                    item_type=line_dict['item_type'],
                    dimensions=line_dict['dimensions'],
                    material=line_dict['material'],
                    quantity=line_dict['quantity'],
                    uom=line_dict['uom'],
                    unit_price=line_dict['unit_price'],
                    total_price=line_dict['total_price'],
                    brand=line_dict['brand'],
                    key_specs=line_dict['key_specs'],
                    search_keywords=line_dict['search_keywords'],
                    reasoning=line_dict['reasoning']
                )
                vendor_lines.append(line)
            
            return vendor_lines
            
        except Exception as e:
            self._thread_safe_print(f"      ⚠️  Cache load error: {e}, regenerating...")
            return None
    
    def clear_vendor_cache(self) -> None:
        """Clear all cached data (BOQ and vendor lines)"""
        if self.cache_dir.exists():
            import shutil
            shutil.rmtree(self.cache_dir)
            print("🗑️  Cleared all cache (BOQ and vendor lines)")
        else:
            print("ℹ️  No cache to clear")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about cached data"""
        cache_info = {
            'cache_dir': str(self.cache_dir),
            'cache_exists': self.cache_dir.exists(),
            'boq_cache': {
                'exists': False,
                'path': str(self._get_boq_cache_path()),
                'size_bytes': 0,
                'modified_time': 0
            },
            'vendor_caches': []
        }
        
        if self.cache_dir.exists():
            # Check BOQ cache
            boq_cache_path = self._get_boq_cache_path()
            if boq_cache_path.exists():
                try:
                    stat = boq_cache_path.stat()
                    cache_info['boq_cache'] = {
                        'exists': True,
                        'path': str(boq_cache_path),
                        'size_bytes': stat.st_size,
                        'modified_time': stat.st_mtime
                    }
                except Exception:
                    pass
            
            # Check vendor caches
            for cache_file in self.cache_dir.glob("*_vendor_lines.json"):
                try:
                    stat = cache_file.stat()
                    cache_info['vendor_caches'].append({
                        'filename': cache_file.name,
                        'size_bytes': stat.st_size,
                        'modified_time': stat.st_mtime
                    })
                except Exception:
                    pass
        
        return cache_info
    
    def _get_boq_cache_path(self) -> Path:
        """Get cache file path for BOQ lines"""
        return self.cache_dir / "boq_lines.json"
    
    def _save_boq_lines_cache(self, boq_lines: List[BOQLineUnderstanding]) -> None:
        """Save BOQ lines to cache"""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = self._get_boq_cache_path()
        
        # Convert dataclasses to dictionaries for JSON serialization
        cache_data = []
        for line in boq_lines:
            line_dict = {
                'row_index': line.row_index,
                'sr_no': line.sr_no,
                'raw_description': line.raw_description,
                'item_type': line.item_type,
                'dimensions': line.dimensions,
                'material': line.material,
                'quantity': line.quantity,
                'uom': line.uom,
                'key_specs': line.key_specs,
                'search_keywords': line.search_keywords,
                'reasoning': line.reasoning
            }
            cache_data.append(line_dict)
        
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, indent=2, ensure_ascii=False)
        
        print(f"  💾 Cached BOQ lines: {cache_path.name}")
    
    def _load_boq_lines_cache(self) -> Optional[List[BOQLineUnderstanding]]:
        """Load BOQ lines from cache if available and valid"""
        cache_path = self._get_boq_cache_path()
        
        if not cache_path.exists():
            return None
        
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            # Convert dictionaries back to dataclasses
            boq_lines = []
            for idx, line_dict in enumerate(cache_data):
                # Handle backward compatibility - old cache files may not have row_index
                row_index = line_dict.get('row_index', idx)
                
                line = BOQLineUnderstanding(
                    row_index=row_index,
                    sr_no=line_dict['sr_no'],
                    raw_description=line_dict['raw_description'],
                    item_type=line_dict['item_type'],
                    dimensions=line_dict['dimensions'],
                    material=line_dict['material'],
                    quantity=line_dict['quantity'],
                    uom=line_dict['uom'],
                    key_specs=line_dict['key_specs'],
                    search_keywords=line_dict['search_keywords'],
                    reasoning=line_dict['reasoning']
                )
                boq_lines.append(line)
            
            print(f"  📂 Loaded {len(boq_lines)} BOQ lines from cache")
            return boq_lines
            
        except Exception as e:
            print(f"  ⚠️  BOQ cache load error: {e}, regenerating...")
            return None
    
    def _thread_safe_print(self, *args, **kwargs):
        """Thread-safe print function"""
        with self.print_lock:
            print(*args, **kwargs)
    
    def _process_single_vendor(self, csv_path: Path, boq_lines: List[BOQLineUnderstanding]) -> Dict[str, Any]:
        """
        Process a single vendor CSV file. This method is designed to be called in parallel.
        Each thread creates its own Bedrock client and agents for thread safety.
        
        Args:
            csv_path: Path to the vendor CSV file
            boq_lines: List of BOQ line understandings (shared across all vendors)
        
        Returns:
            Dictionary with processing results
        """
        vendor_name = self._extract_vendor_name(csv_path.name)
        
        # Create thread-local Bedrock client and agents for thread safety
        # Each thread needs its own client instance to avoid conflicts
        # Use the same max_parallel_calls configuration as the main orchestrator
        thread_bedrock = EnhancedBedrockClient(max_parallel_calls=self.bedrock.max_parallel_calls)
        thread_vendor_agent = VendorQuoteUnderstandingAgent(thread_bedrock)
        thread_alignment_agent = IntelligentAlignmentAgent(thread_bedrock)
        
        self._thread_safe_print(f"  🏭 Processing: {vendor_name}")
        self._thread_safe_print(f"    📄 CSV file: {csv_path.name}")
        self._thread_safe_print(f"    📍 Path: {csv_path}")
        
        # Step 2a: Understand vendor lines (with caching)
        self._thread_safe_print("    🧠 Understanding vendor quotes with LLM...")
        
        # Try to load from cache first (thread-safe)
        vendor_lines = self._load_vendor_lines_cache_threaded(csv_path)
        
        if vendor_lines is None:
            # Cache miss or invalid - process with LLM 
            self._thread_safe_print("    🤖 Processing with LLM (cache miss)...")
            vendor_lines = self._load_and_understand_vendor_threaded(csv_path, thread_vendor_agent)
            self._thread_safe_print(f"      ✓ Understood {len(vendor_lines)} vendor line items")
            
            # Save to cache for next time (with lock to prevent race conditions)
            with self.print_lock:
                self._save_vendor_lines_cache(csv_path, vendor_lines)
        else:
            self._thread_safe_print(f"      ✓ Loaded {len(vendor_lines)} vendor line items from cache")
        
        # Step 2b: Align BOQ to vendor
        self._thread_safe_print("    🔗 Aligning BOQ to vendor quotes...")
        alignments = []
        for boq_line in boq_lines:
            match = thread_alignment_agent.find_best_match(boq_line, vendor_lines)
            alignments.append(match)
        
        matched = sum(1 for a in alignments if a.match_confidence > 0.5)
        self._thread_safe_print(f"      ✓ Aligned {matched}/{len(alignments)} items (confidence > 0.5)")
        
        # Step 2c: Generate per-vendor CSV
        self._thread_safe_print("    📄 Generating comparison CSV...")
        output_path = self._write_vendor_comparison_csv(vendor_name, alignments)
        self._thread_safe_print(f"      ✓ Written: {output_path.name}")
        
        return {
            "vendor_name": vendor_name,
            "csv_path": str(csv_path),
            "output_path": str(output_path),
            "vendor_lines_count": len(vendor_lines),
            "matched_items": matched,
            "total_items": len(alignments)
        }
    
    def _load_and_understand_vendor_threaded(self, csv_path: Path, vendor_agent: VendorQuoteUnderstandingAgent) -> List[VendorLineUnderstanding]:
        """
        Thread-safe version of _load_and_understand_vendor that uses a provided vendor agent.
        This allows each thread to use its own Bedrock client.
        """
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
                        self._thread_safe_print(f"      ✓ Found header at line {line_num}: {cells[:5]}...")
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
                self._thread_safe_print(f"      ❌ Error reading CSV: {e}")
                return []
        
        # Collect all valid vendor lines for batch processing
        vendor_data = []
        for idx, row in df.iterrows():
            row_data = row.to_dict()
            
            # Skip rows that are all NaN
            if all(pd.isna(v) or str(v).strip() == '' for v in row_data.values()):
                continue
            
            vendor_data.append((idx, row_data))
        
        self._thread_safe_print(f"      🤖 Processing {len(vendor_data)} vendor lines in parallel batches...")
        
        # Use batch processing for all vendor lines
        all_understandings = vendor_agent.understand_vendor_lines_batch(vendor_data)
        
        # Filter out None values
        vendor_lines = [u for u in all_understandings if u is not None]
        
        return vendor_lines
    
    def _generate_all_comparison_table(self, boq_lines: List[BOQLineUnderstanding]) -> Path:
        """
        Generate a consolidated comparison table with all vendors side-by-side.
        Format: BOQ Data | Vendor 1 Data | Vendor 2 Data | Vendor 3 Data | ...
        
        Args:
            boq_lines: List of BOQ line understandings
        
        Returns:
            Path to the generated all_comparison.csv file
        """
        
        # Load all individual comparison CSVs to extract vendor data
        comparison_files = sorted(self.comparison_dir.glob("*_comparison.csv"))
        
        if not comparison_files:
            print("  ⚠️  No individual comparison files found, skipping all_comparison generation")
            return None
        
        print(f"  📋 Found {len(comparison_files)} comparison file(s)")
        
        # Dictionary to store all vendor data keyed by composite key (Sr.No + Description)
        # This handles cases where the same Sr.No appears in different sections (e.g., Cable Ladder vs Cable Tray)
        all_data = {}
        vendor_names = []
        
        # Read each comparison CSV and extract vendor data
        for comparison_file in comparison_files:
            vendor_name = self._extract_vendor_name(comparison_file.name).replace("_comparison", "")
            vendor_names.append(vendor_name)
            print(f"    📄 Processing: {vendor_name}")
            
            try:
                df = pd.read_csv(comparison_file)
                
                for idx, row in df.iterrows():
                    boq_sr_no = str(row.get("BOQ Sr.No", "")).strip()
                    boq_description = str(row.get("BOQ Description", "")).strip()
                    
                    # Skip rows without BOQ Sr.No
                    if not boq_sr_no or boq_sr_no.lower() in ("boq sr.no", ""):
                        continue
                    
                    # Use composite key: Sr.No + Description to handle duplicate Sr.Nos across sections
                    # This ensures unique identification of BOQ items (e.g., Cable Ladder item 1 vs Cable Tray item 1)
                    composite_key = f"{boq_sr_no}|||{boq_description}"
                    
                    # Initialize entry for this BOQ item if not exists
                    if composite_key not in all_data:
                        all_data[composite_key] = {
                            "boq_sr_no": boq_sr_no,
                            "boq_description": boq_description,
                            "boq_qty": row.get("BOQ Qty", ""),
                            "boq_uom": row.get("BOQ UOM", ""),
                            "boq_item_type": row.get("BOQ Item Type", ""),
                            "boq_dimensions": row.get("BOQ Dimensions", ""),
                            "boq_material": row.get("BOQ Material", ""),
                            "vendors": {}
                        }
                    
                    # Store vendor-specific data
                    all_data[composite_key]["vendors"][vendor_name] = {
                        "description": row.get("Vendor Description", ""),
                        "qty": row.get("Vendor Qty", ""),
                        "uom": row.get("Vendor UOM", ""),
                        "unit_price": row.get("Vendor Unit Price", ""),
                        "total_price": row.get("Vendor Total Price", ""),
                        "brand": row.get("Vendor Brand", ""),
                        "match_confidence": row.get("Match Confidence", ""),
                        "match_status": row.get("Match Status", ""),
                        "issues": row.get("Issues", ""),
                        "reasoning": row.get("LLM Reasoning", "")
                    }
            except Exception as e:
                print(f"      ⚠️  Error reading {comparison_file.name}: {e}")
                continue
        
        # Generate the consolidated CSV
        output_path = self.comparison_dir / "all_comparison.csv"
        
        print(f"  ✍️  Writing consolidated table to: {output_path.name}")
        
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            # Build dynamic header with all vendors
            header = [
                "BOQ Sr.No",
                "BOQ Description",
                "BOQ Qty",
                "BOQ UOM",
                "BOQ Item Type",
                "BOQ Dimensions",
                "BOQ Material"
            ]
            
            # Add vendor columns for each vendor
            for vendor_name in vendor_names:
                header.extend([
                    f"{vendor_name} - Description",
                    f"{vendor_name} - Qty",
                    f"{vendor_name} - UOM",
                    f"{vendor_name} - Unit Price",
                    f"{vendor_name} - Total Price",
                    f"{vendor_name} - Brand",
                    f"{vendor_name} - Match Confidence",
                    f"{vendor_name} - Match Status",
                    f"{vendor_name} - Issues",
                    f"{vendor_name} - Reasoning"
                ])
            
            writer = csv.writer(f)
            writer.writerow(header)
            
            # Write data rows sorted by BOQ Sr.No
            # Extract Sr.No from composite key for sorting
            for composite_key in sorted(all_data.keys(), key=lambda x: self._sort_key(x.split("|||")[0] if "|||" in x else x)):
                item = all_data[composite_key]
                row = [
                    item["boq_sr_no"],
                    item["boq_description"],
                    item["boq_qty"],
                    item["boq_uom"],
                    item["boq_item_type"],
                    item["boq_dimensions"],
                    item["boq_material"]
                ]
                
                # Add vendor data for each vendor
                for vendor_name in vendor_names:
                    vendor_data = item["vendors"].get(vendor_name, {})
                    row.extend([
                        vendor_data.get("description", ""),
                        vendor_data.get("qty", ""),
                        vendor_data.get("uom", ""),
                        vendor_data.get("unit_price", ""),
                        vendor_data.get("total_price", ""),
                        vendor_data.get("brand", ""),
                        vendor_data.get("match_confidence", ""),
                        vendor_data.get("match_status", ""),
                        vendor_data.get("issues", ""),
                        vendor_data.get("reasoning", "")
                    ])
                
                writer.writerow(row)
        
        print(f"  ✓ Generated all_comparison.csv with {len(all_data)} BOQ items and {len(vendor_names)} vendor(s)")
        return output_path
    
    def _sort_key(self, sr_no: str):
        """Helper to sort BOQ items by numeric serial number"""
        try:
            # Extract leading number from sr_no
            match = re.match(r'(\d+)', str(sr_no).strip())
            if match:
                return int(match.group(1))
        except (ValueError, AttributeError):
            pass
        return float('inf')
    
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
    # Use parallel processing with optimized settings:
    # - max_workers=None: Auto-detect CPU count for vendor parallelization
    # - max_parallel_bedrock_calls=5: Process 5 Bedrock API calls simultaneously per batch
    #   (adjust this if you experience throttling or have higher rate limits)
    orchestrator = EnhancedWorkflowOrchestrator(
        max_workers=None, 
        max_parallel_bedrock_calls=5
    )
    return orchestrator.run()


if __name__ == "__main__":
    raise SystemExit(main())
