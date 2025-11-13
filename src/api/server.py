
"""
Al Shirawi ORC (Optical Character Recognition) POC API Server

This FastAPI server provides a comprehensive API for managing vendor quotation analysis
and comparison workflows. It handles file uploads, workflow orchestration, and provides
analytics for vendor comparison data.

Key Features:
- File upload management for enquiry Excel files and vendor PDF documents
- Workflow orchestration for document processing and analysis
- Vendor comparison and analytics APIs
- AI-powered vendor analysis using AWS Bedrock (when available)
- Static UI serving for web-based interface

The server integrates with vendor_logic module for business logic and file operations.
"""

from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from src.core import vendor_logic
from src.utils.constants import STATIC_DIR

# Initialize FastAPI application
app = FastAPI(title="Al Shirawi ORC POC API", version="0.1.0")

# Mount static files for UI
app.mount("/ui", StaticFiles(directory=str(STATIC_DIR), html=True), name="ui")



@app.get("/health")
def health() -> dict:
    """
    Health check endpoint to verify API server status.
    
    Returns:
        dict: Simple status response indicating the API is operational
    """
    return {"status": "ok"}


@app.post("/create-session")
def create_session() -> JSONResponse:
    """
    Create a new session for isolated workflow runs.
    
    Returns:
        JSONResponse: Session ID for the new session
    """
    session_id = vendor_logic.generate_session_id()
    return JSONResponse({
        "message": "Session created",
        "session_id": session_id,
        "output_directory": str(vendor_logic.get_session_out_dir(session_id))
    })



@app.post("/upload/enquiry")
async def upload_enquiry_excel(
    file: UploadFile = File(...),
    session_id: Optional[str] = Form(None, description="Session ID (optional - auto-generated if not provided)")
) -> JSONResponse:
    """
    Upload enquiry Excel file containing BOQ (Bill of Quantities) data to a specific session.
    
    **Recommended Workflow:**
    1. POST /create-session → get session_id
    2. POST /upload/enquiry with session_id
    3. POST /upload/vendor with same session_id
    4. POST /run-workflow with same session_id
    
    If no session_id is provided, one will be auto-generated and returned.
    Use the returned session_id for subsequent uploads and workflow execution.
    
    Args:
        file (UploadFile): Excel file containing BOQ data
        session_id (str, optional): Session ID from /create-session. Auto-generated if not provided.
        
    Returns:
        JSONResponse: Success response with file details and save location
        
    Raises:
        HTTPException: 400 if file format is not supported
        HTTPException: 500 if file processing fails
    """
    allowed_suffixes = {".xlsx", ".xlsm", ".xls"}
    suffix = file.filename.split(".")[-1].lower()
    if f".{suffix}" not in allowed_suffixes:
        raise HTTPException(status_code=400, detail="Only Excel files are allowed (.xlsx/.xlsm/.xls)")
    
    # Auto-generate session_id if not provided
    if not session_id:
        session_id = vendor_logic.generate_session_id()
        print(f"🆔 Auto-generated session ID for enquiry upload: {session_id}")
    
    contents = await file.read()
    try:
        saved_path = vendor_logic.save_enquiry_excel(file.filename, contents, session_id=session_id)
        # Get tracked enquiry Excel filename
        tracked_filename = vendor_logic.get_uploaded_enquiry_excel(session_id=session_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return JSONResponse({
        "message": f"Enquiry Excel uploaded to session {session_id}",
        "filename": file.filename,
        "saved_to": saved_path,
        "tracked_filename": tracked_filename,
        "session_id": session_id,
        "note": "IMPORTANT: Use this session_id when uploading vendor PDFs and running the workflow!"
    })



@app.post("/upload/vendor")
async def upload_vendor_pdfs(
    response_number: int = Form(..., description="Response number N to map folder 'Response N Attachment'"),
    files: List[UploadFile] = File(..., description="One or more vendor PDF files"),
    session_id: Optional[str] = Form(None, description="Session ID (optional - use same as enquiry upload)")
) -> JSONResponse:
    """
    Upload vendor PDF quotation files for a specific response number to a session.
    
    **IMPORTANT**: Use the SAME session_id that was returned when uploading the enquiry Excel.
    All files in a session must use the same session_id for proper tracking.
    
    If no session_id is provided, files will be uploaded to the default location.
    
    Args:
        response_number (int): Response number to map to 'Response N Attachment' folder
        files (List[UploadFile]): One or more PDF files containing vendor quotations
        session_id (str, optional): Session ID from enquiry upload response
        
    Returns:
        JSONResponse: Success response with upload details and file locations
        
    Raises:
        HTTPException: 400 if no files provided or invalid file format
        HTTPException: 500 if file processing fails
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    if not session_id:
        print(f"⚠️  Warning: No session_id provided for vendor upload. Files uploaded without session tracking.")
    else:
        print(f"✅ Received vendor upload for Response {response_number} with session_id: {session_id}")
    
    file_objs = []
    for f in files:
        contents = await f.read()
        file_objs.append({"filename": f.filename, "contents": contents})
    try:
        saved = vendor_logic.save_vendor_pdfs(response_number, file_objs, session_id=session_id)
        # Get tracked filenames for this response
        uploaded_pdfs = vendor_logic.get_uploaded_pdfs(session_id=session_id)
        tracked_filenames = uploaded_pdfs.get(response_number, [])
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    response_msg = f"Vendor PDFs uploaded to session {session_id}" if session_id else "Vendor PDFs uploaded"
    note_msg = "IMPORTANT: Use this session_id when running the workflow!" if session_id else "Warning: No session_id - files not tracked in any session"
    
    return JSONResponse({
        "message": response_msg,
        "response_number": response_number,
        "count": len(saved),
        "saved_to": saved,
        "tracked_filenames": tracked_filenames,
        "session_id": session_id,
        "note": note_msg
    })



@app.post("/run-workflow")
def run_workflow(
    skip_textract: bool = Form(False),
    skip_boq: bool = Form(False),
    session_id: Optional[str] = Form(None, description="Session ID to process (optional)")
) -> JSONResponse:
    """
    Execute the complete vendor analysis workflow.
    
    **Recommended Workflow:**
    1. POST /create-session → get session_id
    2. POST /upload/enquiry with session_id → upload BOQ Excel
    3. POST /upload/vendor with session_id → upload vendor PDFs (repeat for each vendor)
    4. POST /run-workflow with session_id → process all files in session
    
    The workflow:
    - Extracts data from uploaded PDFs using AWS Textract (unless skipped)
    - Processes BOQ data from Excel files (unless skipped)
    - Performs vendor comparison and matching with LLM (in parallel)
    - Generates analysis reports and comparison data
    
    If session_id is provided: ONLY processes files from that specific session
    If session_id is NOT provided: Processes files from default location (legacy mode)
    
    Args:
        skip_textract (bool): Skip PDF text extraction step (default: False)
        skip_boq (bool): Skip BOQ processing step (default: False)
        session_id (str, optional): Session ID to process. Use session_id from upload responses.
        
    Returns:
        JSONResponse: Success response with workflow completion status and session info
        
    Raises:
        HTTPException: 400 if session has no uploaded files
        HTTPException: 500 if workflow execution fails
    """
    try:
        # Validate session has uploaded files (if session_id provided)
        if session_id:
            tracker_path = vendor_logic.get_session_tracker_path(session_id)
            if not tracker_path.exists():
                raise HTTPException(
                    status_code=400, 
                    detail=f"Session '{session_id}' has no uploaded files. Upload files first using /upload/enquiry and /upload/vendor with this session_id."
                )
        
        result = vendor_logic.run_workflow(skip_textract=skip_textract, skip_boq=skip_boq, session_id=session_id)
        if result != 0:
            raise HTTPException(status_code=500, detail=f"Workflow failed with code {result}")
        
        output_dir = str(vendor_logic.get_session_out_dir(session_id)) if session_id else "out"
        note = "All outputs are isolated in the session-specific directory" if session_id else "Files processed from default location"
        
        return JSONResponse({
            "message": "Workflow completed successfully",
            "code": result,
            "session_id": session_id,
            "output_directory": output_dir,
            "note": note
        })
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



# Redirect root to Home page
@app.get("/")
def root() -> RedirectResponse:
    """
    Redirect root URL to the home page UI.
    
    Returns:
        RedirectResponse: Redirects to /ui/home.html
    """
    return RedirectResponse(url="/ui/home.html")


# ========================================================================
# Session Management APIs
# ========================================================================


@app.get("/api/sessions")
def list_sessions() -> JSONResponse:
    """
    List all available sessions.
    
    Returns all sessions with their metadata, sorted by creation time (newest first).
    Each session includes information about uploaded files, outputs, and workflow runs.
    
    Returns:
        JSONResponse: List of sessions with metadata including:
        - session_id: Unique session identifier
        - created_at: Session creation timestamp
        - enquiry_excel: Name of uploaded BOQ file
        - vendor_count: Number of vendors uploaded
        - outputs: Status of generated outputs
    """
    try:
        sessions = vendor_logic.list_all_sessions()
        return JSONResponse({
            "sessions": sessions,
            "total": len(sessions)
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing sessions: {str(e)}")


@app.get("/api/sessions/{session_id}")
def get_session(session_id: str) -> JSONResponse:
    """
    Get detailed information about a specific session.
    
    Returns comprehensive session details including all uploaded files,
    workflow status, and generated outputs.
    
    Args:
        session_id (str): Session ID
        
    Returns:
        JSONResponse: Detailed session information including:
        - session_id: Unique identifier
        - path: File system path to session directory
        - created_at: Creation timestamp
        - last_updated: Last modification timestamp
        - enquiry_excel: Uploaded BOQ filename
        - vendor_count: Number of vendors
        - outputs: Status of generated outputs (inquiry_csv, textract, comparisons)
        - workflow_runs: History of workflow executions
        
    Raises:
        HTTPException: 404 if session not found
        HTTPException: 500 if error retrieving session info
    """
    try:
        session_info = vendor_logic.get_session_info(session_id)
        return JSONResponse(session_info)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting session info: {str(e)}")


@app.delete("/api/sessions/{session_id}")
def delete_session(session_id: str) -> JSONResponse:
    """
    Delete a session and all its data.
    
    This permanently removes the session directory including all uploaded files,
    outputs, and metadata. This action cannot be undone.
    
    **Warning**: Use with caution. Consider archiving important sessions before deletion.
    
    Args:
        session_id (str): Session ID to delete
        
    Returns:
        JSONResponse: Deletion confirmation with session_id
        
    Raises:
        HTTPException: 404 if session not found
        HTTPException: 500 if deletion fails
    """
    try:
        vendor_logic.delete_session(session_id)
        return JSONResponse({
            "message": f"Session {session_id} deleted successfully",
            "session_id": session_id
        })
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting session: {str(e)}")


# ========================================================================
# Vendor Comparison & Analysis APIs
# ========================================================================


@app.get("/api/vendors")
def list_vendors(session_id: Optional[str] = None) -> JSONResponse:
    """
    Get list of all available vendors with their comparison data.
    
    Returns a list of vendors that have been processed and have comparison
    data available. Each vendor includes ID, name, and filename information.
    
    Args:
        session_id (str, optional): Session ID to filter vendors. If provided, only
                                   vendors from that session are returned.
    
    Returns:
        JSONResponse: List of vendors with optional message if no data found
    """
    vendors = vendor_logic.list_vendors(session_id=session_id)
    message = None
    if not vendors:
        msg = "No comparisons found. Run workflow first."
        if session_id:
            msg += f" (Session: {session_id})"
        message = msg
    return JSONResponse({
        "vendors": vendors,
        "message": message,
        "session_id": session_id
    })



@app.get("/api/vendors/{vendor_id}/comparison")
def get_vendor_comparison(vendor_id: str) -> JSONResponse:
    """
    Get detailed comparison data for a specific vendor.
    
    Returns comprehensive comparison data including:
    - Summary statistics (total items, matched items, match rates)
    - Quality distribution (excellent, good, fair, missing matches)
    - Total pricing information
    - Issues summary
    - Detailed item-by-item comparison data
    
    Args:
        vendor_id (str): Unique identifier for the vendor
        
    Returns:
        JSONResponse: Detailed comparison data for the vendor
        
    Raises:
        HTTPException: 404 if vendor not found
        HTTPException: 500 if data reading fails
    """
    try:
        data = vendor_logic.get_vendor_comparison(vendor_id)
        return JSONResponse(data)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Vendor {vendor_id} not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading comparison: {str(e)}")



@app.get("/api/vendors/{vendor_id}/items")
def get_vendor_items(vendor_id: str, status: str = None, limit: int = None) -> JSONResponse:
    """
    Get filtered list of items for a specific vendor.
    
    Returns item data for a vendor with optional filtering by match status
    and limiting the number of results returned.
    
    Args:
        vendor_id (str): Unique identifier for the vendor
        status (str, optional): Filter by match status ('excellent', 'good', 'fair', 'missing')
        limit (int, optional): Maximum number of items to return
        
    Returns:
        JSONResponse: Filtered list of vendor items with metadata
        
    Raises:
        HTTPException: 404 if vendor not found
        HTTPException: 500 if data reading fails
    """
    try:
        data = vendor_logic.get_vendor_items(vendor_id, status, limit)
        return JSONResponse(data)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Vendor {vendor_id} not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading items: {str(e)}")



@app.get("/api/statistics")
def get_overall_statistics() -> JSONResponse:
    """
    Get overall statistics across all vendors and comparisons.
    
    Returns comprehensive statistics including:
    - Total vendors and BOQ items processed
    - Overall match rates and quality distribution
    - Price range analysis (min, max, average)
    - Common issues across all vendors
    - Individual vendor performance summaries
    
    Returns:
        JSONResponse: Comprehensive statistics data
    """
    stats = vendor_logic.get_overall_statistics()
    return JSONResponse(stats)



@app.post("/api/analyze")
def analyze_vendors(vendor_ids: List[str] = Form(...)) -> JSONResponse:
    """
    Perform AI-powered analysis and comparison of multiple vendors.
    
    This endpoint uses AWS Bedrock (when available) to perform intelligent
    analysis of vendor quotations, providing recommendations and rankings.
    Falls back to heuristic analysis if AI services are unavailable.
    
    Args:
        vendor_ids (List[str]): List of vendor IDs to analyze and compare
        
    Returns:
        JSONResponse: Analysis results including:
        - Vendor performance data
        - AI recommendations and rankings
        - Model used for analysis
        - Error information if applicable
        
    Raises:
        HTTPException: 503 if AI services are unavailable
        HTTPException: 500 if analysis fails
    """
    try:
        result = vendor_logic.analyze_vendors(vendor_ids)
        return JSONResponse(result)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/api/analyze-item")
def analyze_single_item(
    vendor_id: str = Form(...),
    boq_sr_no: str = Form(...),
) -> JSONResponse:
    """
    Perform AI-powered analysis of a single BOQ item and its vendor match.
    
    This endpoint analyzes a specific BOQ item and its corresponding vendor
    quotation match, providing detailed insights about the match quality,
    discrepancies, and recommendations.
    
    Args:
        vendor_id (str): Unique identifier for the vendor
        boq_sr_no (str): BOQ serial number of the item to analyze
        
    Returns:
        JSONResponse: Analysis results including:
        - Item data and match details
        - AI analysis of the match quality
        - Model used for analysis
        - Error information if applicable
        
    Raises:
        HTTPException: 503 if AI services are unavailable
        HTTPException: 404 if vendor or item not found
    """
    try:
        result = vendor_logic.analyze_single_item(vendor_id, boq_sr_no)
        return JSONResponse(result)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        return JSONResponse({
            "item": None,
            "analysis": f"AI analysis unavailable: {str(e)}",
            "model_used": "none",
            "error": str(e)
        })


