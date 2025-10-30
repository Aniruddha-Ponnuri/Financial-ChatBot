"""
Quick test script to verify RL implementation
Run this after starting the Flask server
"""
import requests
import json
import time

BASE_URL = "http://localhost:5000"

def test_health():
    """Test health endpoint"""
    print("Testing /health endpoint...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

def test_ask_question(use_rl=True):
    """Test ask endpoint with RL"""
    print(f"Testing /ask endpoint (use_rl={use_rl})...")
    data = {
        "question": "What are the benefits of diversification in investment?",
        "history": "",
        "use_rl": use_rl
    }
    start_time = time.time()
    response = requests.post(f"{BASE_URL}/ask", json=data)
    elapsed = time.time() - start_time
    
    print(f"Status: {response.status_code}")
    print(f"Time: {elapsed:.2f}s")
    
    if response.status_code == 200:
        result = response.json()
        print(f"RL Used: {result.get('rl_used', False)}")
        print(f"Answer: {result['answer'][:200]}...")
        return result['answer']
    else:
        print(f"Error: {response.text}")
    print()
    return None

def test_submit_feedback(question, answer, rating):
    """Test feedback submission"""
    print(f"Testing /feedback endpoint (rating={rating})...")
    data = {
        "question": question,
        "answer": answer,
        "rating": rating,
        "session_id": "test-session-001"
    }
    response = requests.post(f"{BASE_URL}/feedback", json=data)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

def test_feedback_stats():
    """Test feedback stats"""
    print("Testing /feedback/stats endpoint...")
    response = requests.get(f"{BASE_URL}/feedback/stats")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

def test_generate_candidates():
    """Test candidate generation"""
    print("Testing /generate_candidates endpoint...")
    data = {
        "question": "Should I invest in mutual funds?",
        "n": 3
    }
    start_time = time.time()
    response = requests.post(f"{BASE_URL}/generate_candidates", json=data)
    elapsed = time.time() - start_time
    
    print(f"Status: {response.status_code}")
    print(f"Time: {elapsed:.2f}s")
    
    if response.status_code == 200:
        result = response.json()
        print(f"Generated {result['count']} candidates")
        if result.get('scores'):
            print(f"Scores: {result['scores']}")
        for i, candidate in enumerate(result['candidates'], 1):
            print(f"\nCandidate {i}: {candidate[:150]}...")
    else:
        print(f"Error: {response.text}")
    print()

if __name__ == "__main__":
    print("=" * 80)
    print("RL Implementation Test Suite")
    print("=" * 80)
    print("\nMake sure the Flask server is running: python app.py\n")
    
    try:
        # Test 1: Health check
        test_health()
        
        # Test 2: Ask without RL
        print("Test: Standard question (no RL)")
        answer1 = test_ask_question(use_rl=False)
        
        # Test 3: Ask with RL
        print("Test: RL-enabled question")
        answer2 = test_ask_question(use_rl=True)
        
        # Test 4: Submit positive feedback
        if answer2:
            test_submit_feedback(
                "What are the benefits of diversification in investment?",
                answer2,
                1  # Positive rating
            )
        
        # Test 5: Submit negative feedback
        test_submit_feedback(
            "Test question",
            "Test answer",
            0  # Negative rating
        )
        
        # Test 6: Get feedback stats
        test_feedback_stats()
        
        # Test 7: Generate candidates
        test_generate_candidates()
        
        print("=" * 80)
        print("All tests completed!")
        print("=" * 80)
        
    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Could not connect to server.")
        print("Make sure the Flask server is running:")
        print("  cd Backend")
        print("  python app.py")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
