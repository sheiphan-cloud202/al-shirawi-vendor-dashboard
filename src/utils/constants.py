"""Shared constants and configuration"""
from pathlib import Path
import os

# Base directories
BASE_DIR = Path(__file__).parent.parent.parent  # Project root
DATA_DIR = BASE_DIR / "data"
OUT_DIR = BASE_DIR / "out"
STATIC_DIR = BASE_DIR / "static"

# Output subdirectories
COMPARISON_DIR = OUT_DIR / "vendor_comparisons"
TEXTRACT_DIR = OUT_DIR / "textract"
TEXTRACT_CSV_DIR = OUT_DIR / "textract_csv"
INQUIRY_CSV_DIR = OUT_DIR / "inquiry_csv"
SESSIONS_DIR = OUT_DIR / "sessions"
VENDOR_CACHE_DIR = OUT_DIR / "vendor_cache"

# Input directories
ENQUIRY_ATTACHMENT_DIR = DATA_DIR / "Enquiry Attachment"

# Tracker files
UPLOADED_FILES_TRACKER = OUT_DIR / "uploaded_files_tracker.json"

# AWS Configuration
AWS_PROFILE = os.getenv("AWS_PROFILE", "thinktank")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-5-sonnet-20240620-v1:0")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "al-shirawi-orc-poc")

