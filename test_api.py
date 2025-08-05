import requests
import json
import time

def test_optimized_hackrx_api():
    """Test the optimized HackRx API with the exact same request format."""
    
    # API endpoint
    url = "http://localhost:8000/hackrx/run"
    
    # Headers with authentication
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": "Bearer 0fce51ab380da7e61785e46ae2ba8cee5037bae3ff8d86c68b1a4a1cefe03556"
    }
    
    # Request payload (exact same as original)
    payload = {
        "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
        "questions": [
            "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
            "What is the waiting period for pre-existing diseases (PED) to be covered?",
            "Does this policy cover maternity expenses, and what are the conditions?",
            "What is the waiting period for cataract surgery?",
            "Are the medical expenses for an organ donor covered under this policy?",
            "What is the No Claim Discount (NCD) offered in this policy?",
            "Is there a benefit for preventive health check-ups?",
            "How does the policy define a 'Hospital'?",
            "What is the extent of coverage for AYUSH treatments?",
            "Are there any sub-limits on room rent and ICU charges for Plan A?"
        ]
    }
    
    print("🚀 Testing Optimized HackRx 6.0 API")
    print("=" * 60)
    print(f"Endpoint: {url}")
    print(f"Questions: {len(payload['questions'])}")
    print()
    
    try:
        # Make the request
        start_time = time.time()
        response = requests.post(url, headers=headers, json=payload, timeout=120)
        end_time = time.time()
        
        print(f"⏱️  Response Time: {end_time - start_time:.2f} seconds")
        print(f"📊 Status Code: {response.status_code}")
        print()
        
        if response.status_code == 200:
            result = response.json()
            
            print("✅ SUCCESS - Optimized System Response:")
            print("=" * 60)
            
            # Display answers
            for i, answer in enumerate(result['answers'], 1):
                print(f"\n{i}. Answer: {answer}")
                print(f"   Length: {len(answer)} chars")
                
                # Quality check
                is_good = len(answer) > 50 and "No relevant information" not in answer
                has_specifics = any(word in answer.lower() for word in ['days', 'months', 'years', 'percent', '%', 'clause', 'section', 'policy', 'coverage', 'limit'])
                status = "✅ Excellent" if is_good and has_specifics else "✅ Good" if is_good else "❌ Needs improvement"
                print(f"   Status: {status}")
            
            print(f"\n📈 Performance Summary:")
            print(f"   Total Answers: {len(result['answers'])}")
            print(f"   Average Answer Length: {sum(len(a) for a in result['answers']) / len(result['answers']):.0f} chars")
            print(f"   Good Answers: {sum(1 for a in result['answers'] if len(a) > 50 and 'No relevant information' not in a)}/{len(result['answers'])}")
            
            # Verify format matches exactly
            expected_keys = ['answers']
            actual_keys = list(result.keys())
            
            if actual_keys == expected_keys:
                print("✅ Response format matches exactly!")
            else:
                print(f"❌ Response format mismatch. Expected: {expected_keys}, Got: {actual_keys}")
            
            return True
            
        else:
            print(f"❌ ERROR - Status Code: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ ERROR - Could not connect to server. Make sure the server is running on localhost:8000")
        return False
    except requests.exceptions.Timeout:
        print("❌ ERROR - Request timed out")
        return False
    except Exception as e:
        print(f"❌ ERROR - {str(e)}")
        return False

def test_health_endpoint():
    """Test the health check endpoint."""
    try:
        response = requests.get("http://localhost:8000/")
        if response.status_code == 200:
            data = response.json()
            print("🏥 Health Check:")
            print(f"   Message: {data.get('message', 'N/A')}")
            print(f"   Status: {data.get('status', 'N/A')}")
            print(f"   Version: {data.get('version', 'N/A')}")
            print(f"   Optimization: {data.get('optimization', 'N/A')}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {str(e)}")
        return False

if __name__ == "__main__":
    print("🧪 Testing Optimized HackRx 6.0 System")
    print("=" * 60)
    
    # Test health endpoint first
    print("\n1. Testing Health Endpoint...")
    health_ok = test_health_endpoint()
    
    if health_ok:
        print("\n2. Testing Main API Endpoint...")
        api_ok = test_optimized_hackrx_api()
        
        if api_ok:
            print("\n🎉 ALL TESTS PASSED! Optimized system is working correctly.")
        else:
            print("\n❌ API test failed.")
    else:
        print("\n❌ Health check failed. Please start the server first.")
        print("   Run: python api_main.py") 