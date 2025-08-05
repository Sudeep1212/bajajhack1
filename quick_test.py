#!/usr/bin/env python3
"""
Quick Test Script for Bulletproof HackRx 6.0 System
Tests the API compatibility fixes.
"""

import sys
import os
import time

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_api_compatibility():
    """Test if the API calls work with the fixed configuration."""
    
    print("=" * 60)
    print("TESTING API COMPATIBILITY FIXES")
    print("=" * 60)
    
    try:
        import google.generativeai as genai
        from optimized_system import OptimizedHackRxSystem
        
        print("✅ Google Generative AI imported successfully")
        
        # Test model initialization
        system = OptimizedHackRxSystem()
        print("✅ System initialized successfully")
        
        # Test a simple API call
        test_prompt = "Hello, this is a test. Please respond with 'Test successful'."
        
        print("🔄 Testing API call...")
        response = system._safe_api_call(
            system.model.generate_content,
            test_prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.1,
                max_output_tokens=100,
                top_p=0.8
            )
        )
        
        print(f"✅ API call successful!")
        print(f"📝 Response: {response.text}")
        
        return True
        
    except Exception as e:
        print(f"❌ API compatibility test failed: {str(e)}")
        return False

def test_memory_optimization():
    """Test memory optimization features."""
    
    print("\n" + "=" * 60)
    print("TESTING MEMORY OPTIMIZATION")
    print("=" * 60)
    
    try:
        from optimized_system import OptimizedHackRxSystem
        import psutil
        
        system = OptimizedHackRxSystem()
        
        # Test memory threshold
        initial_memory = psutil.virtual_memory().used / 1024 / 1024
        print(f"📊 Initial memory usage: {initial_memory:.1f}MB")
        
        # Test memory check
        memory_cleaned = system._check_memory()
        print(f"🧹 Memory cleanup triggered: {memory_cleaned}")
        
        final_memory = psutil.virtual_memory().used / 1024 / 1024
        print(f"📊 Final memory usage: {final_memory:.1f}MB")
        
        return True
        
    except Exception as e:
        print(f"❌ Memory optimization test failed: {str(e)}")
        return False

if __name__ == "__main__":
    print("Starting Quick Tests...")
    
    # Test 1: API Compatibility
    api_test = test_api_compatibility()
    
    # Test 2: Memory Optimization
    memory_test = test_memory_optimization()
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"API Compatibility: {'✅ PASSED' if api_test else '❌ FAILED'}")
    print(f"Memory Optimization: {'✅ PASSED' if memory_test else '❌ FAILED'}")
    
    if api_test and memory_test:
        print("\n🎉 All tests passed! The system is ready to run.")
        print("\n📋 Next steps:")
        print("1. Run: python test_local.py")
        print("2. Start server: python -m uvicorn api_main:app --host 0.0.0.0 --port 8000")
    else:
        print("\n⚠️  Some tests failed. Please check the error messages above.") 