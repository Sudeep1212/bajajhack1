from fastapi import FastAPI, HTTPException, Depends, Header
from pydantic import BaseModel
from typing import List
import logging
import time
import sys
import os
import traceback

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the bulletproof system
from optimized_system_v2 import process_questions

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Bajaj Finserv Health HackRx 6.0 - Bulletproof LLM-Powered Query Retrieval System",
    description="An intelligent system for processing insurance documents and answering queries with bulletproof error handling",
    version="3.0.0"
)

# Pydantic models for request/response
class QueryRequest(BaseModel):
    documents: str
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]

# Authentication
async def verify_token(authorization: str = Header(None)):
    """Verify the Bearer token."""
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header required")
    
    expected_token = "0fce51ab380da7e61785e46ae2ba8cee5037bae3ff8d86c68b1a4a1cefe03556"
    if authorization != f"Bearer {expected_token}":
        raise HTTPException(status_code=401, detail="Invalid token")
    
    return True

@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "message": "Bajaj Finserv Health HackRx 6.0 - Bulletproof LLM-Powered Query Retrieval System",
        "status": "healthy",
        "version": "3.0.0",
        "optimization": "Bulletproof error handling with multi-API rotation",
        "features": [
            "Memory management and cleanup",
            "Multi-API key rotation",
            "Timeout handling",
            "Graceful degradation",
            "Complete error recovery"
        ]
    }

@app.post("/api/v1/hackrx/run", response_model=QueryResponse)
async def run_hackrx_system(
    request: QueryRequest,
    token_verified: bool = Depends(verify_token)
):
    """
    Run the Bulletproof HackRx 6.0 LLM-Powered Query Retrieval System.
    
    This endpoint processes insurance documents and answers questions using:
    - PDF document processing with error handling
    - Multi-API key rotation for reliability
    - Memory management and cleanup
    - Timeout handling and graceful degradation
    - Complete error recovery mechanisms
    - NLP-based keyword extraction and chunking
    - Cosine similarity for relevant text selection
    - Batch processing to minimize API calls
    - LLM-powered semantic understanding
    - Structured response generation
    """
    start_time = time.time()
    logger.info("=== Starting Bulletproof HackRx 6.0 System ===")
    
    try:
        # Validate input
        if not request.documents:
            raise HTTPException(status_code=400, detail="Documents URL is required")
        
        if not request.questions or len(request.questions) == 0:
            raise HTTPException(status_code=400, detail="At least one question is required")
        
        logger.info(f"Processing {len(request.questions)} questions with bulletproof error handling")
        
        # Process questions using the bulletproof system
        result = process_questions(request.documents, request.questions)
        
        # Calculate success rate
        good_answers = sum(1 for answer in result['answers'] 
                          if "No relevant information" not in answer and len(answer) > 50 and "Error" not in answer)
        success_rate = (good_answers / len(result['answers'])) * 100 if result['answers'] else 0
        
        logger.info(f"Success Rate: {success_rate:.1f}%")
        logger.info(f"API Calls Reduced: {result.get('api_calls_reduced', 'N/A')}")
        logger.info(f"Chunks Created: {result.get('chunks_created', 'N/A')}")
        logger.info(f"Memory Usage: {result.get('memory_usage', 'N/A')}")
        
        if 'error' in result:
            logger.warning(f"System used fallback due to: {result['error']}")
        
        total_time = time.time() - start_time
        logger.info(f"=== Completed in {total_time:.2f}s ===")
        
        # Return only the answers array as per the exact format requirement
        return QueryResponse(answers=result['answers'])
        
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Return fallback responses instead of crashing
        fallback_answers = [
            "System temporarily unavailable. Please try again.",
            "Processing error occurred. Please retry.",
            "Service experiencing issues. Please try later.",
            "Error in processing. Please contact support.",
            "System error. Please try again."
        ]
        
        # Extend fallback answers to match question count
        while len(fallback_answers) < len(request.questions):
            fallback_answers.extend(fallback_answers)
        
        return QueryResponse(answers=fallback_answers[:len(request.questions)])

@app.post("/hackrx/run", response_model=QueryResponse)
async def legacy_endpoint(
    request: QueryRequest,
    token_verified: bool = Depends(verify_token)
):
    """
    Legacy endpoint for backward compatibility.
    Exact same format as original HackRx 6.0 system.
    """
    return await run_hackrx_system(request, token_verified)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 