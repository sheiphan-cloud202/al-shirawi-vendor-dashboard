#!/usr/bin/env python3
"""
Test script to demonstrate vendor lines caching functionality
"""

from workflow_enhanced import EnhancedWorkflowOrchestrator

def test_cache_functionality():
    """Test the caching mechanism"""
    
    print("=" * 60)
    print("Testing BOQ and Vendor Lines Caching")
    print("=" * 60)
    
    # Initialize workflow
    workflow = EnhancedWorkflowOrchestrator()
    
    # Check if we have vendor CSV files
    vendor_csvs = list(workflow.textract_csv_dir.glob("*.csv"))
    
    if not vendor_csvs:
        print("❌ No vendor CSV files found in textract_csv directory")
        return
    
    print(f"Found {len(vendor_csvs)} vendor CSV files")
    
    # Test BOQ caching
    print("\n" + "="*40)
    print("Testing BOQ Caching")
    print("="*40)
    
    # Test BOQ cache loading (should be None on first run)
    print("\n1. Testing BOQ cache load (first run)...")
    cached_boq = workflow._load_boq_lines_cache()
    print(f"   Result: {'Cache hit' if cached_boq else 'Cache miss'}")
    
    # Test BOQ processing and caching
    print("\n2. Processing BOQ lines and caching...")
    boq_lines = workflow._load_and_understand_boq()
    print(f"   Processed {len(boq_lines)} BOQ lines")
    
    # Save to cache
    workflow._save_boq_lines_cache(boq_lines)
    
    # Test BOQ cache loading again (should hit now)
    print("\n3. Testing BOQ cache load (second run)...")
    cached_boq = workflow._load_boq_lines_cache()
    print(f"   Result: {'Cache hit' if cached_boq else 'Cache miss'}")
    if cached_boq:
        print(f"   Loaded {len(cached_boq)} BOQ lines from cache")
    
    # Test vendor caching
    print("\n" + "="*40)
    print("Testing Vendor Caching")
    print("="*40)
    
    if vendor_csvs:
        # Test with the first vendor CSV
        csv_path = vendor_csvs[0]
        vendor_name = workflow._extract_vendor_name(csv_path.name)
        
        print(f"\nTesting with: {vendor_name}")
        print(f"CSV Path: {csv_path}")
        
        # Test cache path generation
        cache_path = workflow._get_cache_path(csv_path)
        print(f"Cache Path: {cache_path}")
        
        # Test cache loading (should be None on first run)
        print("\n1. Testing vendor cache load (first run)...")
        cached_lines = workflow._load_vendor_lines_cache(csv_path)
        print(f"   Result: {'Cache hit' if cached_lines else 'Cache miss'}")
        
        # Test processing and caching
        print("\n2. Processing vendor lines and caching...")
        vendor_lines = workflow._load_and_understand_vendor(csv_path)
        print(f"   Processed {len(vendor_lines)} lines")
        
        # Save to cache
        workflow._save_vendor_lines_cache(csv_path, vendor_lines)
        
        # Test cache loading again (should hit now)
        print("\n3. Testing vendor cache load (second run)...")
        cached_lines = workflow._load_vendor_lines_cache(csv_path)
        print(f"   Result: {'Cache hit' if cached_lines else 'Cache miss'}")
        if cached_lines:
            print(f"   Loaded {len(cached_lines)} lines from cache")
    
    # Test cache info
    print("\n" + "="*40)
    print("Cache Information")
    print("="*40)
    cache_info = workflow.get_cache_info()
    print(f"Cache Directory: {cache_info['cache_dir']}")
    print(f"BOQ Cache Exists: {cache_info['boq_cache']['exists']}")
    print(f"Vendor Caches: {len(cache_info['vendor_caches'])} files")
    
    print("\n✅ Cache functionality test completed!")

if __name__ == "__main__":
    test_cache_functionality()
