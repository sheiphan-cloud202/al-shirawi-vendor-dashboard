# Vendor Analysis Dashboard - Feature Documentation

## Overview

A comprehensive AI-powered vendor comparison and analysis system built on top of the existing BOQ comparison workflow. This feature allows users to visually compare multiple vendors and receive intelligent recommendations powered by AWS Bedrock (Claude 3.5 Sonnet).

## Architecture

### Backend (server.py)

Three new API endpoints were added:

1. **GET `/api/vendors`**
   - Lists all available vendor comparison CSV files
   - Extracts vendor names and IDs from filenames
   - Returns structured vendor list for UI consumption

2. **GET `/api/vendors/{vendor_id}/comparison`**
   - Loads detailed comparison CSV for a specific vendor
   - Calculates comprehensive metrics:
     - Total items / matched items
     - Match quality breakdown (excellent/good)
     - Total quoted price
     - Issue frequency analysis (qty variance, UOM mismatch, type mismatch)
   - Returns both summary statistics and full line-item details

3. **POST `/api/analyze`**
   - Accepts multiple vendor IDs for comparison
   - Aggregates metrics across all selected vendors
   - Invokes AWS Bedrock (Claude 3.5 Sonnet) for AI analysis
   - Generates intelligent recommendations considering:
     - Match rates and quality
     - Price competitiveness
     - Issue severity
     - Overall value assessment
   - Falls back to heuristic ranking if Bedrock unavailable

### Frontend

#### analysis.html
- Main dashboard page for vendor comparison
- Navigation between Upload and Analysis pages
- Responsive card-based layout
- Loading states and error handling

#### analysis.js
- Fetches vendor list on page load
- Parallel loading of all vendor details for performance
- Interactive vendor selection with checkboxes
- Real-time "Analyze" button state management
- Calls AI analysis API and renders results
- Formats AI recommendations with markdown-like styling

#### analysis.css
- Modern, accessible design system
- Dark/light mode support via CSS variables
- Responsive grid layouts for vendor cards
- Color-coded match quality badges
- Issue highlighting with visual warnings
- Clean table design for comparison summaries

## User Flow

1. **Navigate to Analysis Dashboard** (`/ui/analysis.html`)
   - System automatically loads all vendor comparison files
   - Displays vendor cards with key metrics:
     - Match rate percentage
     - Total quoted price
     - Items matched vs total
     - Quality breakdown (excellent/good matches)
     - Issue summary

2. **Select Vendors to Compare**
   - Check boxes for 1+ vendors
   - "Analyze" button activates when selections made
   - Shows count: "Analyze 3 Selected Vendors"

3. **Get AI Recommendations**
   - Click "Analyze" button
   - System calls AWS Bedrock API
   - AI analyzes:
     - Vendor ranking (best to worst)
     - Strengths and weaknesses
     - Recommended vendor with justification
     - Red flags and concerns
     - Price competitiveness
   - Results display in formatted, readable layout

## Key Features

### Smart Metrics Calculation
- **Match Rate**: Percentage of BOQ items successfully matched to vendor quotes
- **Price Analysis**: Automatic summation of vendor total prices
- **Issue Detection**: Categorizes and counts:
  - Quantity variances
  - Unit of Measurement mismatches
  - Item type conflicts

### AI-Powered Insights
- Uses Claude 3.5 Sonnet (configurable via `BEDROCK_MODEL_ID`)
- Provides business-focused recommendations
- Considers multiple factors beyond just price
- Explains reasoning transparently
- Fallback to heuristic if AI unavailable

### Professional UI/UX
- Clean, modern interface
- Responsive design (mobile-friendly)
- Loading states during async operations
- Error handling with user-friendly messages
- Smooth scrolling to results
- Color-coded quality indicators

## Technical Highlights

### Performance Optimizations
- Parallel API calls for vendor details
- Efficient CSV parsing with Python's csv.DictReader
- Client-side caching of vendor data
- Minimal re-renders in UI

### Error Handling
- Graceful fallback when Bedrock unavailable
- Handles missing/malformed CSV files
- User-friendly error messages
- Console logging for debugging

### Extensibility
- Easy to add new metrics/calculations
- Modular API endpoint design
- Reusable UI components
- CSS variable system for theming

## Configuration

### Environment Variables
```bash
# Required for AI recommendations
export AWS_PROFILE=your-profile
export AWS_REGION=us-east-1

# Optional
export BEDROCK_MODEL_ID=anthropic.claude-3-5-sonnet-20240620-v1:0
```

### File Structure
```
out/vendor_comparisons/
├── 3_-_IKKT_comparison.csv
├── 4_-_ElectroMechProjects_comparison.csv
├── 5_-_Bonn_Group_comparison.csv
└── 6_-_nbiuae_comparison.csv
```

## Future Enhancements

Potential improvements:
1. **Export functionality** - Download analysis report as PDF/Excel
2. **Historical tracking** - Compare quotes over time
3. **Custom weighting** - Let users prioritize price vs quality
4. **Vendor profiles** - Track vendor performance history
5. **Notification system** - Alert on significant price changes
6. **Multi-currency support** - Handle international vendors
7. **Advanced filtering** - Filter by item category, price range, etc.
8. **Collaboration features** - Share analysis with team members

## Dependencies

- **FastAPI**: Web framework for API endpoints
- **boto3**: AWS SDK for Bedrock integration
- **Python csv module**: CSV parsing
- **Vanilla JS**: No frontend framework dependencies
- **Modern CSS**: Grid, flexbox, CSS variables

## Testing

To test the analysis feature:

1. Ensure vendor comparison CSVs exist in `out/vendor_comparisons/`
2. Start server: `uv run uvicorn server:app --reload`
3. Navigate to `http://localhost:8000/ui/analysis.html`
4. Select vendors and click "Analyze"
5. Review AI recommendations

## API Examples

### List Vendors
```bash
curl http://localhost:8000/api/vendors
```

### Get Vendor Details
```bash
curl http://localhost:8000/api/vendors/3/comparison
```

### Analyze Vendors
```bash
curl -X POST http://localhost:8000/api/analyze \
  -F "vendor_ids=3" \
  -F "vendor_ids=4" \
  -F "vendor_ids=5"
```

## Summary

This feature transforms raw comparison CSV data into actionable business intelligence, enabling procurement teams to make data-driven decisions quickly and confidently. The AI recommendations provide expert-level analysis at scale, while the intuitive UI makes complex comparisons accessible to all stakeholders.

