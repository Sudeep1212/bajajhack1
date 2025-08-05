from fastapi import FastAPI, HTTPException, Depends, Header
from pydantic import BaseModel
from typing import List
import logging
import time
import sys
import os

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the optimized system
from optimized_system import process_questions

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Bajaj Finserv Health HackRx 6.0 - Optimized LLM-Powered Query Retrieval System",
    description="An intelligent system for processing insurance documents and answering queries with NLP optimization",
    version="2.0.0"
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
        "message": "Bajaj Finserv Health HackRx 6.0 - Optimized LLM-Powered Query Retrieval System",
        "status": "healthy",
        "version": "2.0.0",
        "optimization": "NLP preprocessing with batch processing"
    }

@app.post("/api/v1/hackrx/run", response_model=QueryResponse)
async def run_hackrx_system(
    request: QueryRequest,
    token_verified: bool = Depends(verify_token)
):
    """
    Run the Optimized HackRx 6.0 LLM-Powered Query Retrieval System.
    
    This endpoint processes insurance documents and answers questions using:
    - PDF document processing
    - NLP-based keyword extraction and chunking
    - Cosine similarity for relevant text selection
    - Batch processing to minimize API calls
    - LLM-powered semantic understanding
    - Structured response generation
    """
    start_time = time.time()
    logger.info("=== Starting Optimized HackRx 6.0 System ===")
    
    try:
        # Validate input
        if not request.documents:
            raise HTTPException(status_code=400, detail="Documents URL is required")
        
        if not request.questions or len(request.questions) == 0:
            raise HTTPException(status_code=400, detail="At least one question is required")
        
        logger.info(f"Processing {len(request.questions)} questions with NLP optimization")
        
        # Process questions using the optimized system
        result = process_questions(request.documents, request.questions)
        
        # Calculate success rate
        good_answers = sum(1 for answer in result['answers'] 
                          if "No relevant information" not in answer and len(answer) > 50)
        success_rate = (good_answers / len(result['answers'])) * 100 if result['answers'] else 0
        
        logger.info(f"Success Rate: {success_rate:.1f}%")
        logger.info(f"API Calls Reduced: {result.get('api_calls_reduced', 'N/A')}")
        logger.info(f"Chunks Created: {result.get('chunks_created', 'N/A')}")
        
        total_time = time.time() - start_time
        logger.info(f"=== Completed in {total_time:.2f}s ===")
        
        # Return only the answers array as per the exact format requirement
        return QueryResponse(answers=result['answers'])
        
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

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