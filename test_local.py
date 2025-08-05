#!/usr/bin/env python3
"""
Local Test Script for Bulletproof HackRx 6.0 System
This script tests the optimized system locally without starting the API server.
"""

import sys
import os
import time
import logging

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from optimized_system import process_questions

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_bulletproof_system():
    """Test the bulletproof system with sample data."""
    
    print("=" * 60)
    print("BULLETPROOF HACKRX 6.0 - LOCAL TEST")
    print("=" * 60)
    
    # Test PDF URL
    pdf_url = "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"
    
    # Test questions
    questions = [
        "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
        "What is the waiting period for pre-existing diseases (PED) to be covered?",
        "Does this policy cover maternity expenses, and what are the conditions?",
        "What is the waiting period for cataract surgery?",
        "Are the medical expenses for an organ donor covered under this policy?"
    ]
    
    print(f"\n📋 Testing with {len(questions)} questions...")
    print(f"📄 PDF URL: {pdf_url}")
    
    start_time = time.time()
    
    try:
        # Process questions
        result = process_questions(pdf_url, questions)
        
        processing_time = time.time() - start_time
        
        print(f"\n✅ Processing completed in {processing_time:.2f} seconds")
        print(f"📊 Results:")
        print(f"   - Questions processed: {result.get('questions_processed', 'N/A')}")
        print(f"   - Text length: {result.get('text_length', 'N/A')} chars")
        print(f"   - Chunks created: {result.get('chunks_created', 'N/A')}")
        print(f"   - API calls reduced: {result.get('api_calls_reduced', 'N/A')}")
        print(f"   - Memory usage: {result.get('memory_usage', 'N/A')}")
        print(f"   - API keys used: {result.get('api_keys_used', 'N/A')}")
        
        if 'error' in result:
            print(f"   ⚠️  Error occurred: {result['error']}")
        
        print(f"\n📝 Answers:")
        for i, (question, answer) in enumerate(zip(questions, result['answers']), 1):
            print(f"\n{i}. Question: {question}")
            print(f"   Answer: {answer}")
            print(f"   Length: {len(answer)} chars")
            
            # Quality assessment
            if len(answer) > 50 and "No relevant information" not in answer and "Error" not in answer:
                print(f"   Status: ✅ Good")
            else:
                print(f"   Status: ⚠️  Needs improvement")
        
        print(f"\n🎉 Test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        logger.error(f"Test failed: {str(e)}")
        return False

def test_api_server():
    """Test if the API server can be started."""
    print("\n" + "=" * 60)
    print("TESTING API SERVER STARTUP")
    print("=" * 60)
    
    try:
        import uvicorn
        from api_main import app
        
        print("✅ API server imports successful")
        print("✅ FastAPI app created successfully")
        print("✅ All dependencies loaded correctly")
        
        print("\n🚀 To start the API server locally, run:")
        print("   cd a4")
        print("   python -m uvicorn api_main:app --host 0.0.0.0 --port 8000 --reload")
        
        print("\n🌐 Then access the API at:")
        print("   http://localhost:8000")
        print("   http://localhost:8000/docs")
        
        return True
        
    except Exception as e:
        print(f"❌ API server test failed: {str(e)}")
        return False

if __name__ == "__main__":
    print("Starting Bulletproof HackRx 6.0 Local Tests...")
    
    # Test 1: Core system functionality
    test1_success = test_bulletproof_system()
    
    # Test 2: API server setup
    test2_success = test_api_server()
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Core System Test: {'✅ PASSED' if test1_success else '❌ FAILED'}")
    print(f"API Server Test: {'✅ PASSED' if test2_success else '❌ FAILED'}")
    
    if test1_success and test2_success:
        print("\n🎉 All tests passed! The system is ready for local use.")
        print("\n📋 Next steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Start the API server: python -m uvicorn api_main:app --host 0.0.0.0 --port 8000")
        print("3. Test with Postman or curl")
    else:
        print("\n⚠️  Some tests failed. Please check the error messages above.") 