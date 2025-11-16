from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.environ.get("ANTHROPIC_API_KEY")

if api_key:
    print(f"✓ API Key found: {api_key[:20]}...") 
    print(f"✓ Key length: {len(api_key)} characters")
else:
    print("✗ API Key NOT found!")
    print("Check your .env file")