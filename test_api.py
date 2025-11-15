import requests
import json

BASE_URL = BASE_URL = "http://localhost:8001"

def test_basic_connection():
    """Test if API is running"""
    print("1. Testing basic connection...")
    try:
        response = requests.get(f"{BASE_URL}/")
        print(f"✓ API is running: {response.json()}")
        print()
        return True
    except requests.exceptions.ConnectionError:
        print("✗ Cannot connect to API. Make sure it's running with: uvicorn main:app --reload --port 8001")
        return False

def test_chat():
    """Test the main chat endpoint"""
    print("2. Testing chat endpoint...")
    
    payload = {
        "messages": [
            {
                "role": "user",
                "content": "What should I eat for a healthy breakfast?"
            }
        ]
    }
    
    try:
        response = requests.post(f"{BASE_URL}/chat", json=payload)
        
        if response.status_code == 200:
            data = response.json()
            print("✓ Chat is working!")
            print(f"\nBot Response:\n{data['response']}\n")
            return True
        else:
            print(f"✗ Error {response.status_code}: {response.text}")
            return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_conversation():
    """Test multi-turn conversation"""
    print("3. Testing multi-turn conversation...")
    
    payload = {
        "messages": [
            {"role": "user", "content": "I want to lose weight"},
            {"role": "assistant", "content": "I can help you with that! What are your current eating habits?"},
            {"role": "user", "content": "I eat a lot of fast food and rarely exercise"}
        ],
        "user_profile": {
            "age": 30,
            "goal": "weight loss"
        }
    }
    
    try:
        response = requests.post(f"{BASE_URL}/chat", json=payload)
        
        if response.status_code == 200:
            data = response.json()
            print("✓ Conversation context is working!")
            print(f"\nBot Response:\n{data['response']}\n")
            return True
        else:
            print(f"✗ Error {response.status_code}: {response.text}")
            return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_meal_analysis():
    """Test meal analysis endpoint"""
    print("4. Testing meal analysis...")
    
    try:
        response = requests.post(
            f"{BASE_URL}/analyze-meal",
            params={"meal_description": "Grilled chicken with brown rice and broccoli"}
        )
        
        if response.status_code == 200:
            data = response.json()
            print("✓ Meal analysis is working!")
            print(f"\nAnalysis:\n{data['analysis']}\n")
            return True
        else:
            print(f"✗ Error {response.status_code}: {response.text}")
            return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_meal_suggestion():
    """Test meal suggestion endpoint"""
    print("5. Testing meal suggestion...")
    
    try:
        response = requests.post(
            f"{BASE_URL}/suggest-meal",
            params={
                "dietary_preferences": "vegetarian",
                "calorie_target": 500,
                "meal_type": "lunch"
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            print("✓ Meal suggestions are working!")
            print(f"\nSuggestion:\n{data['suggestion']}\n")
            return True
        else:
            print(f"✗ Error {response.status_code}: {response.text}")
            return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

if __name__ == "__main__":
    print("="*60)
    print("TESTING NUTRITION CHATBOT API")
    print("="*60)
    print()
    
    # Run all tests
    if test_basic_connection():
        test_chat()
        test_conversation()
        test_meal_analysis()
        test_meal_suggestion()
    
    print("="*60)
    print("Testing complete!")
    print("="*60)