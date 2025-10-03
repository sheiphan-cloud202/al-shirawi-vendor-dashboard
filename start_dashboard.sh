#!/bin/bash

# Al Shirawi ORC Dashboard - Quick Start Script
# This script starts the FastAPI server with the advanced dashboard

echo "🚀 Starting Al Shirawi ORC Advanced Dashboard..."
echo ""
echo "📊 Dashboard Features:"
echo "  - Executive Summary with Statistics"
echo "  - Interactive Charts (Chart.js)"
echo "  - AI-Powered Vendor Analysis"
echo "  - Item-Level Drill-Down"
echo "  - Comprehensive Filtering & Search"
echo ""
echo "🌐 Available URLs:"
echo "  - Upload Page:        http://localhost:8000/ui/"
echo "  - Basic Analysis:     http://localhost:8000/ui/analysis.html"
echo "  - Advanced Dashboard: http://localhost:8000/ui/dashboard.html"
echo "  - API Docs:           http://localhost:8000/docs"
echo "  - Health Check:       http://localhost:8000/health"
echo ""
echo "⚙️  Configuration:"
echo "  - AWS Profile: ${AWS_PROFILE:-thinktank}"
echo "  - AWS Region:  ${AWS_REGION:-us-east-1}"
echo "  - Model:       ${BEDROCK_MODEL_ID:-anthropic.claude-3-5-sonnet-20240620-v1:0}"
echo ""
echo "📝 Logs will appear below..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Set default environment variables if not set
export AWS_PROFILE="${AWS_PROFILE:-thinktank}"
export AWS_REGION="${AWS_REGION:-us-east-1}"
export BEDROCK_MODEL_ID="${BEDROCK_MODEL_ID:-anthropic.claude-3-5-sonnet-20240620-v1:0}"

# Start the server
cd "$(dirname "$0")"
uvicorn server:app --host 0.0.0.0 --port 8000 --reload


