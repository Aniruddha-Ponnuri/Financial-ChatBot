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
        "use_rl": use_rl,
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
        return result["answer"]
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
        "session_id": "test-session-001",
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
    data = {"question": "Should I invest in mutual funds?", "n": 3}
    start_time = time.time()
    response = requests.post(f"{BASE_URL}/generate_candidates", json=data)
    elapsed = time.time() - start_time

    print(f"Status: {response.status_code}")
    print(f"Time: {elapsed:.2f}s")

    if response.status_code == 200:
        result = response.json()
        print(f"Generated {result['count']} candidates")
        if result.get("scores"):
            print(f"Scores: {result['scores']}")
        for i, candidate in enumerate(result["candidates"], 1):
            print(f"\nCandidate {i}: {candidate[:150]}...")
    else:
        print(f"Error: {response.text}")
    print()


def test_stock_queries():
    """Test various stock-related queries"""

    test_cases = [
        {
            "name": "Single stock query - Apple",
            "question": "What's the current price of Apple stock?",
            "expected_symbols": ["AAPL"],
        },
        {
            "name": "Multiple stocks - Tech giants",
            "question": "Compare Microsoft and Google stocks",
            "expected_symbols": ["MSFT", "GOOGL"],
        },
        {
            "name": "Stock by ticker symbol",
            "question": "Tell me about TSLA",
            "expected_symbols": ["TSLA"],
        },
        {
            "name": "Stock analysis request",
            "question": "Should I invest in NVDA? What's the analysis?",
            "expected_symbols": ["NVDA"],
        },
        {
            "name": "Company name to ticker",
            "question": "How is Amazon doing in the market?",
            "expected_symbols": ["AMZN"],
        },
        {
            "name": "Non-stock query",
            "question": "What are mutual funds?",
            "expected_symbols": [],
        },
    ]

    print("=" * 80)
    print("STOCK INTEGRATION TEST SUITE")
    print("=" * 80)
    print()

    for i, test in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test['name']}")
        print("-" * 80)
        print(f"Question: {test['question']}")

        try:
            response = requests.post(
                f"{BASE_URL}/ask",
                json={
                    "question": test["question"],
                    "history": "",
                    "use_rl": False,  # Disable RL for faster testing
                },
                timeout=30,
            )

            if response.status_code == 200:
                data = response.json()

                # Check if stock symbols were detected
                detected_symbols = data.get("stock_symbols", [])
                answer = data.get("answer", "")

                print(f"✓ Status: Success")
                print(f"Detected Symbols: {detected_symbols}")
                print(f"Expected Symbols: {test['expected_symbols']}")

                # Check if symbols match expectations
                if set(detected_symbols) == set(test["expected_symbols"]):
                    print("✓ Symbol detection: PASSED")
                else:
                    print("✗ Symbol detection: MISMATCH")

                # Show a snippet of the answer
                answer_snippet = answer[:200].replace("<br>", " ").replace("\n", " ")
                print(f"\nAnswer preview: {answer_snippet}...")

                # Check if answer contains real-time data indicators
                if detected_symbols:
                    has_price = any(
                        keyword in answer.lower() for keyword in ["price", "$", "current", "market"]
                    )
                    print(f"Contains price info: {'✓ Yes' if has_price else '✗ No'}")

            else:
                print(f"✗ Status: Failed ({response.status_code})")
                print(f"Error: {response.text}")

        except requests.exceptions.Timeout:
            print("✗ Request timed out")
        except Exception as e:
            print(f"✗ Error: {str(e)}")

        print()

    print("=" * 80)
    print("TEST SUITE COMPLETED")
    print("=" * 80)


def test_stock_data_fetcher():
    """Test the stock data fetcher directly"""
    print("\n" + "=" * 80)
    print("TESTING STOCK DATA FETCHER MODULE")
    print("=" * 80)

    try:
        import sys
        import os

        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

        from utils.stock_data import StockDataFetcher

        fetcher = StockDataFetcher()

        # Test single stock
        print("\nTest 1: Fetching Apple (AAPL) stock data")
        print("-" * 80)
        stock_info = fetcher.get_stock_info("AAPL")

        if stock_info:
            print("✓ Successfully fetched stock data")
            print(f"Company: {stock_info['name']}")
            print(f"Symbol: {stock_info['symbol']}")
            print(f"Current Price: ${stock_info['current_price']}")
            print(f"Market Cap: {stock_info['market_cap']}")
            print(f"Sector: {stock_info['sector']}")
        else:
            print("✗ Failed to fetch stock data")

        # Test formatted context
        print("\nTest 2: Formatted stock context")
        print("-" * 80)
        context = fetcher.format_stock_context("MSFT", include_historical=True)
        print(context[:500] + "...")

        # Test invalid symbol
        print("\nTest 3: Invalid symbol handling")
        print("-" * 80)
        invalid_stock = fetcher.get_stock_info("INVALID123")
        if invalid_stock is None:
            print("✓ Correctly handled invalid symbol")
        else:
            print("✗ Should return None for invalid symbol")

        print("\n" + "=" * 80)
        print("STOCK DATA FETCHER TESTS COMPLETED")
        print("=" * 80)

    except ImportError as e:
        print(f"✗ Import error: {e}")
        print("Make sure yfinance is installed: pip install yfinance")
    except Exception as e:
        print(f"✗ Error: {e}")


if __name__ == "__main__":
    print("=" * 80)
    print("\n\nNow testing API integration...")
    print("RL & Stock Price Implementation Test Suite")
    print("=" * 80)
    print("\nMake sure the Flask server is running: python app.py\n")
    input("Press Enter to continue...")

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
                1,  # Positive rating
            )

        # Test 5: Submit negative feedback
        test_submit_feedback(
            "Test question",
            "Test answer",
            0,  # Negative rating
        )

        # Test 6: Get feedback stats
        test_feedback_stats()

        # Test 7: Generate candidates
        test_generate_candidates()

        # First test the stock data fetcher module
        test_stock_data_fetcher()
        # Then test the API integration
        test_stock_queries()

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
