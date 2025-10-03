#!/usr/bin/env python3
"""
Test script demonstrating AWS Bedrock Structured Output using Tool Use API
Reference: https://aws.amazon.com/blogs/machine-learning/structured-data-response-with-amazon-bedrock-prompt-engineering-and-tool-use/
"""

import json
import os
from workflow_enhanced import EnhancedBedrockClient

def test_structured_output():
    """Test the new structured output functionality"""
    
    # Initialize Bedrock client
    bedrock = EnhancedBedrockClient()
    
    # Sample vendor quote line data
    sample_data = {
        "Description": "CABLE TRAY LADDER TYPE HDG 900 MM WIDE 2.0 MM THICK 3 METER LENGTH",
        "Make": "METAR",
        "Qty": 119.00,
        "Unit": "Lth",
        "Unit Rate": 271.92,
        "Total Amount": 32358.48
    }
    
    # Define JSON schema
    json_schema = {
        "type": "object",
        "properties": {
            "item_type": {
                "type": "string",
                "description": "Type of item: cable tray, bend, tee, cover, accessory, etc."
            },
            "dimensions": {
                "type": "object",
                "properties": {
                    "width": {"type": ["string", "null"], "description": "Width with unit"},
                    "height": {"type": ["string", "null"], "description": "Height with unit"},
                    "thickness": {"type": ["string", "null"], "description": "Thickness with unit"},
                    "length": {"type": ["string", "null"], "description": "Length with unit"}
                }
            },
            "material": {
                "type": "string",
                "description": "Material type: HDG, GI, SS304, etc."
            },
            "quantity": {"type": ["number", "null"]},
            "uom": {"type": ["string", "null"]},
            "unit_price": {"type": ["number", "null"]},
            "total_price": {"type": ["number", "null"]},
            "brand": {"type": ["string", "null"]},
            "key_specs": {
                "type": "array",
                "items": {"type": "string"}
            },
            "search_keywords": {
                "type": "array",
                "items": {"type": "string"}
            },
            "reasoning": {"type": "string"}
        },
        "required": ["item_type", "dimensions", "material", "search_keywords", "reasoning"]
    }
    
    # Create prompt
    prompt = f"""Analyze this vendor quotation line and extract structured information.

Vendor Quote Line:
{json.dumps(sample_data, indent=2)}

Important:
- Extract ALL numeric values (qty, prices) from any column
- Identify dimensions from description with units
- Identify material type (HDG, GI, SS304, etc.)
- Parse UOM variations
- Provide comprehensive search keywords
"""
    
    print("=" * 80)
    print("Testing AWS Bedrock Structured Output with Tool Use API")
    print("=" * 80)
    print("\nInput Data:")
    print(json.dumps(sample_data, indent=2))
    print("\n" + "=" * 80)
    
    # Call structured output
    result = bedrock.invoke_structured(
        prompt=prompt,
        json_schema=json_schema,
        tool_name="extract_vendor_quote_data",
        tool_description="Extract structured information from vendor quotation line",
        max_tokens=1024,
        temperature=0.0
    )
    
    print("\nStructured Output Result:")
    print("=" * 80)
    if result:
        print(json.dumps(result, indent=2))
        print("\n✅ Successfully extracted structured data!")
        print(f"\n📊 Item Type: {result.get('item_type')}")
        print(f"📏 Dimensions: {result.get('dimensions')}")
        print(f"🔧 Material: {result.get('material')}")
        print(f"💰 Unit Price: {result.get('unit_price')}")
        print(f"🏷️  Brand: {result.get('brand')}")
    else:
        print("❌ Failed to extract structured data")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    test_structured_output()

