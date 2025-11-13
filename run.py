#!/usr/bin/env python3
"""
Application entry point for Al Shirawi ORC POC

Usage:
    python run.py          # Start the API server
"""
import sys
import uvicorn
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent))

from src.api.server import app

if __name__ == "__main__":
    print("🚀 Starting Al Shirawi ORC POC API Server...")
    print("📖 API Documentation: http://localhost:8000/docs")
    print("🏠 Web UI: http://localhost:8000/ui/index.html")
    print()
    
    uvicorn.run(
        "src.api.server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

