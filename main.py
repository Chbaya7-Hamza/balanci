from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv
import os

# Kaggle model imports
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import torch
import numpy as np

load_dotenv()

app = FastAPI(title="Nutrition Chatbot API - Kaggle Model")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================
# EXACT SAME SETUP AS YOUR KAGGLE NOTEBOOK
# =========================================

MODEL_NAME = "google/gemma-2-2b-it"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Your exact nutrition database
NUTRITION_DB = {
    "chicken breast": 165, "chicken thigh": 209, "turkey breast": 189, "ground turkey": 203,
    "beef sirloin": 250, "ground beef": 250, "pork chop": 242, "pork tenderloin": 143,
    "salmon fillet": 208, "tuna steak": 132, "tilapia": 128, "cod": 82,
    "eggs": 155, "egg whites": 52, "tofu": 144, "tempeh": 193,
    "white rice": 130, "brown rice": 111, "quinoa": 120, "pasta": 131,
    "whole wheat pasta": 124, "oats": 389, "oatmeal": 389, "bread": 265,
    "whole wheat bread": 247, "sweet potato": 86, "potato": 77, "white potato": 77,
    "broccoli": 34, "spinach": 23, "kale": 35, "carrots": 41,
    "tomatoes": 18, "cucumber": 15, "bell pepper": 20, "red pepper": 20,
    "green beans": 31, "asparagus": 20, "cauliflower": 25,
    "banana": 89, "apple": 52, "orange": 47, "berries": 57,
    "strawberries": 32, "blueberries": 57, "avocado": 160, "mango": 60,
    "olive oil": 884, "coconut oil": 862, "butter": 717, "peanut butter": 588,
    "almond butter": 614, "almonds": 579, "walnuts": 654, "cashews": 553,
    "milk": 42, "almond milk": 17, "greek yogurt": 59, "yogurt": 59,
    "cheese": 402, "cottage cheese": 98, "cheddar cheese": 403,
    "chickpeas": 164, "lentils": 116, "black beans": 132, "kidney beans": 127,
}

# Your workout plans
WORKOUT_PLANS = {
    "beginner": """Beginner Full Body (3x/week)
- Warm-up: 5 min cardio
- Squats: 3x10
- Push-ups: 3x8
- Dumbbell rows: 3x10
- Plank: 3x20-30s
- Walking lunges: 2x10/leg
- Cool-down: 5 min stretch""",

    "intermediate": """Intermediate Split (4x/week)
Upper Body: Bench press 4x8, Pull-ups 4x8, Shoulder press 3x10, Bicep curls 3x12
Lower Body: Squats 4x8, Romanian deadlifts 3x10, Leg press 3x12, Calf raises 4x15""",

    "advanced": """Advanced Push/Pull/Legs (6x/week)
Push Day: Barbell bench 5x5, Incline press 4x8, Overhead press 4x6, Lateral raises 3x12
Pull Day: Deadlifts 5x5, Pull-ups 4xmax, Barbell rows 4x8, Face pulls 3x15
Leg Day: Back squats 5x5, Front squats 3x8, Leg press 4x10, Leg curls 3x12"""
}

# Global variables for models
embedding_model = None
food_items = None
food_embeddings = None
chatbot = None
tokenizer = None
llm_model = None

print("=" * 60)
print("LOADING MODELS - This will take 5-10 minutes first time...")
print("=" * 60)

# Load embedding model for cosine similarity
print("\n1. Loading embedding model...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
print("✓ Embedding model loaded!")

# Precompute embeddings
print("\n2. Computing food embeddings...")
food_items = list(NUTRITION_DB.keys())
food_embeddings = embedding_model.encode(food_items)
print(f"✓ Embeddings computed for {len(food_items)} foods!")

# Load main LLM
print(f"\n3. Loading Gemma model: {MODEL_NAME}")
print("   This is the big one - please wait...")

try:
    HF_TOKEN = os.environ.get("HUGGINGFACE_API_KEY")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
    llm_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        token=HF_TOKEN,
    )

    pipe = pipeline(
        "text-generation",
        model=llm_model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id,
        return_full_text=False,
    )

    llm = HuggingFacePipeline(pipeline=pipe)

    # Create prompt template
    prompt = PromptTemplate(
        input_variables=["chat_history", "question", "nutrition_context"],
        template="""<bos><start_of_turn>user
You are a professional nutritionist and certified personal trainer with expertise in meal planning and fitness coaching.

Context: {nutrition_context}
Previous conversation: {chat_history}
Question: {question}
<end_of_turn>
<start_of_turn>model
""")

    # Create chain
    chatbot = (
            {
                "question": lambda x: x["question"],
                "chat_history": lambda x: x["chat_history"],
                "nutrition_context": lambda x: x["nutrition_context"]
            }
            | prompt
            | llm
            | StrOutputParser()
    )

    print("✓ Gemma model loaded and chain created!")

except Exception as e:
    print(f"⚠ Error loading Gemma model: {e}")
    print("⚠ API will run with limited functionality (calculations only)")
    chatbot = None

print("\n" + "=" * 60)
print("MODEL LOADING COMPLETE!")
print("=" * 60 + "\n")


# =========================================
# YOUR EXACT KAGGLE FUNCTIONS
# =========================================

def find_similar_food(query_food, threshold=0.6):
    """Use cosine similarity to find the most similar food"""
    query_embedding = embedding_model.encode([query_food.lower()])
    similarities = cosine_similarity(query_embedding, food_embeddings)[0]
    best_idx = np.argmax(similarities)
    best_score = similarities[best_idx]

    if best_score >= threshold:
        matched_food = food_items[best_idx]
        calories = NUTRITION_DB[matched_food]
        return matched_food, best_score, calories

    return None, 0, 0


def calculate_recipe_calories_with_similarity(ingredients_text):
    """Enhanced calorie calculation using cosine similarity"""
    total_calories = 0
    breakdown = []
    not_found = []

    ingredients = [i.strip() for i in ingredients_text.lower().split(',')]

    for ingredient in ingredients:
        parts = ingredient.split()
        if len(parts) < 2:
            continue

        amount_str = parts[0].replace('g', '').replace('kg', '')
        try:
            amount = float(amount_str)
            if 'kg' in parts[0]:
                amount *= 1000
        except:
            not_found.append(ingredient)
            continue

        ingredient_name = ' '.join(parts[1:])

        # Try exact match first
        found = False
        for db_item, cal_per_100g in NUTRITION_DB.items():
            if db_item in ingredient_name or ingredient_name in db_item:
                calories = (amount / 100) * cal_per_100g
                total_calories += calories
                breakdown.append(f"- {amount}g {db_item}: {calories:.0f} cal (exact match)")
                found = True
                break

        # Use cosine similarity if no exact match
        if not found:
            matched_food, similarity, cal_per_100g = find_similar_food(ingredient_name)

            if matched_food:
                calories = (amount / 100) * cal_per_100g
                total_calories += calories
                breakdown.append(f"- {amount}g {matched_food}: {calories:.0f} cal (similarity: {similarity:.2f})")
            else:
                not_found.append(ingredient)

    return total_calories, breakdown, not_found


def find_similar_recipes_by_ingredients(target_recipe, top_k=5):
    """Find recipes with similar ingredients using cosine similarity"""
    recipe_database = {
        "High Protein Meal": "200g chicken breast, 100g brown rice, 100g broccoli",
        "Balanced Lunch": "150g salmon fillet, 80g quinoa, 150g spinach, 50g avocado",
        "Low-Carb Dinner": "200g turkey breast, 200g cauliflower, 50g olive oil",
        "Vegetarian Bowl": "150g tofu, 100g chickpeas, 200g kale, 30g almonds",
        "Post-Workout": "200g beef sirloin, 150g sweet potato, 100g asparagus",
        "Breakfast Power": "100g oats, 50g blueberries, 30g walnuts, 200g greek yogurt",
        "Mediterranean": "150g salmon fillet, 100g white rice, 100g tomatoes, 20g olive oil",
        "Vegan Protein": "150g tempeh, 150g lentils, 100g spinach, 50g avocado",
    }

    target_embedding = embedding_model.encode([target_recipe])
    recipe_names = list(recipe_database.keys())
    recipe_texts = list(recipe_database.values())
    recipe_embeddings = embedding_model.encode(recipe_texts)

    similarities = cosine_similarity(target_embedding, recipe_embeddings)[0]
    top_indices = np.argsort(similarities)[::-1][:top_k]

    suggestions = []
    for idx in top_indices:
        name = recipe_names[idx]
        ingredients = recipe_texts[idx]
        score = similarities[idx]
        cals, _, _ = calculate_recipe_calories_with_similarity(ingredients)
        suggestions.append(f"- {name} (similarity: {score:.2f}): {ingredients} (~{cals:.0f} cal)")

    return suggestions


def suggest_similar_recipes_by_calories(target_calories, tolerance=100):
    """Traditional calorie-based recipe suggestions"""
    recipes = {
        "High Protein (500 cal)": "200g chicken breast, 100g brown rice, 100g broccoli",
        "Balanced Lunch (450 cal)": "150g salmon fillet, 80g quinoa, 150g spinach",
        "Low-Carb (400 cal)": "200g turkey breast, 200g cauliflower, 50g avocado",
        "Vegetarian (380 cal)": "150g tofu, 100g chickpeas, 200g kale",
        "Post-Workout (550 cal)": "200g beef sirloin, 150g sweet potato, 100g asparagus",
    }

    suggestions = []
    for name, ingredients in recipes.items():
        cals, _, _ = calculate_recipe_calories_with_similarity(ingredients)
        if abs(cals - target_calories) <= tolerance:
            suggestions.append(f"- {name}: {ingredients} (~{cals:.0f} cal)")

    return suggestions if suggestions else ["- No matches in calorie range"]


def format_chat_history(history):
    """Format chat history for the model"""
    if not history:
        return "New conversation"

    formatted = []
    for user_msg, bot_msg in history[-3:]:
        formatted.append(f"User: {user_msg}")
        formatted.append(f"Assistant: {bot_msg}")

    return "\n".join(formatted)


def process_message(message, chat_history=""):
    """Main processing function - same logic as Kaggle"""
    nutrition_context = ""

    # Calculate calories
    if "calculate" in message.lower() and "calorie" in message.lower():
        if ":" in message:
            ingredients = message.split(":", 1)[1].strip()
            total_cals, breakdown, not_found = calculate_recipe_calories_with_similarity(ingredients)

            context_parts = ["CALORIE CALCULATION (with AI matching):"]

            if breakdown:
                context_parts.append("\nIngredients breakdown:")
                context_parts.extend(breakdown)
                context_parts.append(f"\nTotal Calories: {total_cals:.0f} cal")

            if not_found:
                context_parts.append(f"\nNot found in database: {', '.join(not_found)}")

            context_parts.append("\n--- Similar recipes by CALORIES ---")
            context_parts.extend(suggest_similar_recipes_by_calories(total_cals))

            context_parts.append("\n--- Similar recipes by INGREDIENTS (AI) ---")
            context_parts.extend(find_similar_recipes_by_ingredients(ingredients))

            nutrition_context = "\n".join(context_parts)

    # Find similar recipes
    elif "similar" in message.lower() and "recipe" in message.lower():
        if ":" in message:
            recipe = message.split(":", 1)[1].strip()
            suggestions = find_similar_recipes_by_ingredients(recipe, top_k=5)
            nutrition_context = f"Similar recipes to '{recipe}':\n\n" + "\n".join(suggestions)

    # Workout plans
    elif "workout" in message.lower():
        level = "beginner"
        if "intermediate" in message.lower():
            level = "intermediate"
        elif "advanced" in message.lower():
            level = "advanced"

        nutrition_context = f"\nWORKOUT PLAN:\n\n{WORKOUT_PLANS[level]}"

    # If chatbot is loaded, use it for response
    if chatbot is not None:
        try:
            response = chatbot.invoke({
                "question": message,
                "chat_history": chat_history,
                "nutrition_context": nutrition_context
            })
            return response
        except Exception as e:
            return f"{nutrition_context}\n\nError generating AI response: {str(e)}"
    else:
        # Fallback if model not loaded
        return nutrition_context if nutrition_context else "AI model not loaded. Please provide calculation or workout request."


# =========================================
# API MODELS
# =========================================

class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[Message]
    user_profile: Optional[dict] = None


class ChatResponse(BaseModel):
    response: str
    conversation_id: Optional[str] = None


class CalorieRequest(BaseModel):
    ingredients: str


class CalorieResponse(BaseModel):
    total_calories: float
    breakdown: List[str]
    not_found: List[str]
    similar_recipes_by_calories: List[str]
    similar_recipes_by_ingredients: List[str]


# =========================================
# API ENDPOINTS
# =========================================

@app.get("/")
async def root():
    return {
        "message": "Nutrition Chatbot API - Kaggle Model with Cosine Similarity",
        "status": "running",
        "model": MODEL_NAME,
        "embedding_model": "all-MiniLM-L6-v2",
        "features": [
            "AI-powered calorie calculation",
            "Fuzzy ingredient matching (cosine similarity)",
            "Similar recipe recommendations",
            "Workout plans",
            "Conversational AI (Gemma-2-2b-it)"
        ],
        "ai_loaded": chatbot is not None
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Main chat endpoint - exactly like your Kaggle chatbot"""
    try:
        # Get conversation history
        chat_history = format_chat_history(
            [(msg.content, "") for msg in request.messages if msg.role == "user"]
        )

        # Get last user message
        user_message = ""
        for msg in request.messages:
            if msg.role == "user":
                user_message = msg.content

        # Process message
        response_text = process_message(user_message, chat_history)

        return ChatResponse(response=response_text, conversation_id=None)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.post("/calculate-calories", response_model=CalorieResponse)
async def calculate_calories(request: CalorieRequest):
    """Calculate calories with AI matching - your Kaggle feature!"""
    try:
        total_cals, breakdown, not_found = calculate_recipe_calories_with_similarity(request.ingredients)

        similar_by_cals = suggest_similar_recipes_by_calories(total_cals)
        similar_by_ingredients = find_similar_recipes_by_ingredients(request.ingredients)

        return CalorieResponse(
            total_calories=total_cals,
            breakdown=breakdown,
            not_found=not_found,
            similar_recipes_by_calories=similar_by_cals,
            similar_recipes_by_ingredients=similar_by_ingredients
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.post("/analyze-meal")
async def analyze_meal(meal_description: str):
    """Analyze meal using cosine similarity"""
    try:
        total_cals, breakdown, not_found = calculate_recipe_calories_with_similarity(meal_description)

        analysis = f"Total Calories: {total_cals:.0f} cal\n\n"
        analysis += "Breakdown:\n" + "\n".join(breakdown)

        if not_found:
            analysis += f"\n\nNot found: {', '.join(not_found)}"

        return {"analysis": analysis}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.post("/suggest-meal")
async def suggest_meal(
        dietary_preferences: Optional[str] = None,
        calorie_target: Optional[int] = None,
        meal_type: Optional[str] = None
):
    """Suggest meals using your Kaggle logic"""
    try:
        if calorie_target:
            suggestions = suggest_similar_recipes_by_calories(calorie_target)
            return {"suggestion": "\n".join(suggestions)}
        else:
            return {"suggestion": "Please provide a calorie target for meal suggestions."}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.get("/workout-plan")
async def get_workout_plan(level: str = "beginner"):
    """Get workout plan - your Kaggle feature!"""
    try:
        level = level.lower()
        if level not in WORKOUT_PLANS:
            level = "beginner"

        return {"plan": WORKOUT_PLANS[level], "level": level}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)


@app.on_event("startup")
async def startup_event():
    print("\n=== Available Endpoints ===")
    for route in app.routes:
        if hasattr(route, 'methods'):
            print(f"{list(route.methods)[0]:6} {route.path}")
    print("===========================\n")
