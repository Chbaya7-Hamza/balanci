from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import anthropic
import os

app = FastAPI(title="Nutrition Chatbot API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Anthropic client
client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))


# Models
class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[Message]
    user_profile: Optional[dict] = None


class ChatResponse(BaseModel):
    response: str
    conversation_id: Optional[str] = None


# Nutrition chatbot system prompt
NUTRITION_SYSTEM_PROMPT = """You are a knowledgeable and supportive nutrition assistant. 
Your role is to:
- Provide evidence-based nutritional advice
- Help users understand macro and micronutrients
- Suggest healthy meal ideas based on dietary preferences
- Answer questions about food choices, calories, and nutrition
- Offer guidance on dietary goals (weight loss, muscle gain, general health)

Always:
- Ask clarifying questions when needed
- Consider dietary restrictions and allergies
- Provide balanced, realistic advice
- Encourage consulting healthcare professionals for medical conditions
- Be supportive and non-judgmental

Never:
- Diagnose medical conditions
- Prescribe specific treatments
- Make extreme dietary recommendations"""


@app.get("/")
async def root():
    return {"message": "Nutrition Chatbot API", "status": "running"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint for nutrition conversations
    """
    try:
        # Format messages for Claude
        formatted_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in request.messages
        ]

        # Add user profile context if provided
        system_prompt = NUTRITION_SYSTEM_PROMPT
        if request.user_profile:
            profile_info = f"\n\nUser Profile:\n"
            for key, value in request.user_profile.items():
                profile_info += f"- {key}: {value}\n"
            system_prompt += profile_info

        # Call Claude API
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            system=system_prompt,
            messages=formatted_messages
        )

        return ChatResponse(
            response=response.content[0].text,
            conversation_id=response.id
        )

    except anthropic.APIError as e:
        raise HTTPException(status_code=500, detail=f"API error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.post("/analyze-meal")
async def analyze_meal(meal_description: str):
    """
    Analyze a meal description for nutritional content
    """
    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            system="You are a nutrition expert. Analyze meals and provide estimated nutritional breakdown including calories, protein, carbs, fats, and key vitamins/minerals.",
            messages=[{
                "role": "user",
                "content": f"Please analyze this meal: {meal_description}"
            }]
        )

        return {"analysis": response.content[0].text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.post("/suggest-meal")
async def suggest_meal(
        dietary_preferences: Optional[str] = None,
        calorie_target: Optional[int] = None,
        meal_type: Optional[str] = None
):
    """
    Suggest a meal based on preferences and goals
    """
    try:
        prompt = "Suggest a healthy meal"
        if meal_type:
            prompt += f" for {meal_type}"
        if dietary_preferences:
            prompt += f" that is {dietary_preferences}"
        if calorie_target:
            prompt += f" with approximately {calorie_target} calories"

        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            system=NUTRITION_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}]
        )

        return {"suggestion": response.content[0].text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


# Run with: uvicorn main:app --reload
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
