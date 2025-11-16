from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

model_name = "distilbert-base-uncased-finetuned-sst-2-english"  # Replace with your desired model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
# =========================================
# NUTRITION CHATBOT WITH COSINE SIMILARITY
# Enhanced ingredient matching and recipe recommendations
# =========================================

# ============= CELL 1: INSTALL =============
print("Installing dependencies with embeddings support...")
import sys
import subprocess

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                       "langchain", "langchain-community", "langchain-huggingface",
                       "transformers", "gradio", "huggingface-hub",
                       "sentence-transformers", "scikit-learn", "numpy"])

print("Installation complete!")

# ============= CELL 2: VERIFY =============
print("Verifying packages...")

import langchain
import transformers
import gradio
import torch
import numpy
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

print("langchain:", langchain.__version__)
print("transformers:", transformers.__version__)
print("gradio:", gradio.__version__)
print("torch:", torch.__version__)
print("numpy:", numpy.__version__)
print("sentence-transformers: OK")
print("scikit-learn: OK")
print("\nAll packages loaded successfully!")

# ============= CELL 3: LOGIN =============
from huggingface_hub import login
import os

print("Logging in to Hugging Face...")

try:
    from kaggle_secrets import UserSecretsClient

    user_secrets = UserSecretsClient()
    HF_TOKEN = user_secrets.get_secret("HF_TOKEN")
    login(token=HF_TOKEN)
    print("Login successful!")
except Exception as e:
    print("Error:", e)
    from getpass import getpass

    HF_TOKEN = getpass("Enter your HF token: ")
    login(token=HF_TOKEN)
    print("Login successful!")

# ============= CELL 4: SETUP =============
import os
import torch

os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda/lib64:' + os.environ.get('LD_LIBRARY_PATH', '')

print("GPU Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU Name:", torch.cuda.get_device_name(0))

# ============= CELL 5: COSINE SIMILARITY SETUP =============
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

print("Loading embedding model for cosine similarity...")
# Use a lightweight model for faster computation
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
print("Embedding model loaded!")

# ============= CELL 6: MAIN CODE WITH COSINE SIMILARITY =============
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableMap

MODEL_NAME = "google/gemma-2-2b-it"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Enhanced Nutrition Database with categories
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

# Precompute embeddings for all foods in database
print("Computing embeddings for food database...")
food_items = list(NUTRITION_DB.keys())
food_embeddings = embedding_model.encode(food_items)
print(f"Embeddings computed for {len(food_items)} food items!")

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


def find_similar_food(query_food, threshold=0.6):
    """
    Use cosine similarity to find the most similar food in database
    Returns: (matched_food, similarity_score, calories)
    """
    # Encode the query
    query_embedding = embedding_model.encode([query_food.lower()])

    # Calculate cosine similarity with all foods
    similarities = cosine_similarity(query_embedding, food_embeddings)[0]

    # Find best match
    best_idx = np.argmax(similarities)
    best_score = similarities[best_idx]

    if best_score >= threshold:
        matched_food = food_items[best_idx]
        calories = NUTRITION_DB[matched_food]
        return matched_food, best_score, calories

    return None, 0, 0


def calculate_recipe_calories_with_similarity(ingredients_text):
    """
    Enhanced calorie calculation using cosine similarity for fuzzy matching
    """
    total_calories = 0
    breakdown = []
    not_found = []

    ingredients = [i.strip() for i in ingredients_text.lower().split(',')]

    for ingredient in ingredients:
        parts = ingredient.split()
        if len(parts) < 2:
            continue

        # Extract amount
        amount_str = parts[0].replace('g', '').replace('kg', '')
        try:
            amount = float(amount_str)
            if 'kg' in parts[0]:
                amount *= 1000
        except:
            not_found.append(ingredient)
            continue

        # Get ingredient name
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

        # If no exact match, use cosine similarity
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
    """
    Find recipes with similar ingredients using cosine similarity
    """
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

    # Encode target recipe
    target_embedding = embedding_model.encode([target_recipe])

    # Encode all recipes
    recipe_names = list(recipe_database.keys())
    recipe_texts = list(recipe_database.values())
    recipe_embeddings = embedding_model.encode(recipe_texts)

    # Calculate similarities
    similarities = cosine_similarity(target_embedding, recipe_embeddings)[0]

    # Get top k most similar
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
    """
    Traditional calorie-based recipe suggestions
    """
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


def setup_llm(model_name=MODEL_NAME):
    print(f"Loading model: {model_name}")
    print("This may take 5-10 minutes...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        token=True,
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id,
        return_full_text=False,
    )

    print("Model loaded successfully!")
    return HuggingFacePipeline(pipeline=pipe)


def create_nutrition_prompt():
    return PromptTemplate(
        input_variables=["chat_history", "question", "nutrition_context"],
        template="""<bos><start_of_turn>user
You are a professional nutritionist and certified personal trainer with expertise in meal planning and fitness coaching.

Context: {nutrition_context}
Previous conversation: {chat_history}
Question: {question}
<end_of_turn>
<start_of_turn>model
""")


def format_chat_history(history):
    if not history:
        return "New conversation"

    formatted = []
    for user_msg, bot_msg in history[-3:]:
        formatted.append(f"User: {user_msg}")
        formatted.append(f"Assistant: {bot_msg}")

    return "\n".join(formatted)


def create_chatbot():
    print("Initializing chatbot...")

    llm = setup_llm()
    prompt = create_nutrition_prompt()

    chain = (
            RunnableMap({
                "question": lambda x: x["question"],
                "chat_history": lambda x: x["chat_history"],
                "nutrition_context": lambda x: x["nutrition_context"]
            })
            | prompt
            | llm
            | StrOutputParser()
    )

    print("Chatbot ready!")
    return chain


def process_message(message, history):
    nutrition_context = ""

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

            # Add both calorie-based and ingredient-based suggestions
            context_parts.append("\n--- Similar recipes by CALORIES ---")
            context_parts.extend(suggest_similar_recipes_by_calories(total_cals))

            context_parts.append("\n--- Similar recipes by INGREDIENTS (AI) ---")
            context_parts.extend(find_similar_recipes_by_ingredients(ingredients))

            nutrition_context = "\n".join(context_parts)

    elif "similar" in message.lower() and "recipe" in message.lower():
        if ":" in message:
            recipe = message.split(":", 1)[1].strip()
            suggestions = find_similar_recipes_by_ingredients(recipe, top_k=5)
            nutrition_context = f"Similar recipes to '{recipe}':\n\n" + "\n".join(suggestions)

    elif "workout" in message.lower():
        level = "beginner"
        if "intermediate" in message.lower():
            level = "intermediate"
        elif "advanced" in message.lower():
            level = "advanced"

        nutrition_context = f"\nWORKOUT PLAN:\n\n{WORKOUT_PLANS[level]}"

    chat_history = format_chat_history(history)

    response = chatbot.invoke({
        "question": message,
        "chat_history": chat_history,
        "nutrition_context": nutrition_context
    })

    return response


print("Creating chatbot with cosine similarity...")
chatbot = create_chatbot()
print("Done!")

# ============= CELL 7: LAUNCH =============
import gradio as gr

print("Creating web interface...")

demo = gr.ChatInterface(
    fn=process_message,
    title="AI Nutrition Coach (with Cosine Similarity)",
    description="""
    Enhanced with AI-powered ingredient matching using cosine similarity!

    Features:
    - Fuzzy ingredient matching (e.g., 'chiken' will match 'chicken')
    - Similar recipe recommendations based on ingredients
    - Calorie-based meal suggestions
    - Workout plans
    """,
    examples=[
        "Calculate calories: 200g chicken breast, 150g brown rice, 100g broccoli",
        "Calculate calories: 180g chiken, 120g rize, 80g brocoli",
        "Find similar recipe: 200g salmon, 100g quinoa, 50g avocado",
        "Give me a beginner workout plan",
        "What should I eat to build muscle?",
        "Calculate calories: 150g turkey, 100g sweet potato, 100g spinach",
    ],
)

print("Launching chatbot...")
print("Wait for the public URL to appear...")

demo.launch(share=True, debug=True)

# ============= TESTING CELL (Optional) =============
# Run this cell to test cosine similarity separately

print("\n" + "=" * 60)
print("TESTING COSINE SIMILARITY")
print("=" * 60)

# Test 1: Fuzzy ingredient matching
print("\nTest 1: Fuzzy Matching")
test_ingredients = ["chiken breast", "rize", "brocoli", "samon"]
for ingredient in test_ingredients:
    matched, score, cals = find_similar_food(ingredient)
    print(f"'{ingredient}' -> '{matched}' (score: {score:.2f}, {cals} cal/100g)")

# Test 2: Recipe similarity
print("\nTest 2: Recipe Similarity")
test_recipe = "200g chicken, 150g rice, 100g vegetables"
similar = find_similar_recipes_by_ingredients(test_recipe, top_k=3)
print(f"\nSimilar to: {test_recipe}")
for recipe in similar:
    print(recipe)

# Test 3: Calorie calculation with fuzzy matching
print("\nTest 3: Calorie Calculation")
test_input = "200g chiken breast, 150g browne rize, 100g spinach"
total, breakdown, not_found = calculate_recipe_calories_with_similarity(test_input)
print(f"Input: {test_input}")
for item in breakdown:
    print(item)
print(f"Total: {total:.0f} calories")

print("=" * 60)

# ============= CELL 8: CHATBOT RESPONSE EVALUATION =============
print("\n" + "=" * 60)
print("EVALUATING CHATBOT RESPONSES WITH COSINE SIMILARITY")
print("=" * 60)

from sentence_transformers import util

# Reuse the embedding model already loaded
eval_model = embedding_model

# Define test examples (you can add more later)
evaluation_pairs = [
    {
        "user_input": "What should I eat to build muscle?",
        "reference_response": "You should eat foods rich in protein such as chicken, fish, eggs, and beans, along with complex carbs and healthy fats."
    },
    {
        "user_input": "Give me a beginner workout plan",
        "reference_response": "A beginner workout plan includes full-body exercises 3 times a week, such as squats, push-ups, and planks."
    },
    {
        "user_input": "How many calories are in 200g chicken breast?",
        "reference_response": "200 grams of chicken breast contain around 330 calories."
    },
    {
        "user_input": "Find similar recipe: 150g tofu, 100g chickpeas, 200g kale",
        "reference_response": "Here are recipes with similar ingredients such as vegetarian or vegan protein bowls."
    }
]


def evaluate_chatbot_response_similarity(chatbot, examples):
    """
    Evaluate chatbot responses by comparing to reference answers using cosine similarity.
    """
    scores = []
    for ex in examples:
        user_msg = ex["user_input"]
        reference = ex["reference_response"]

        # Generate chatbot reply
        bot_reply = chatbot.invoke({
            "question": user_msg,
            "chat_history": "",
            "nutrition_context": ""
        })

        # Encode both responses
        emb_ref = eval_model.encode(reference, convert_to_tensor=True)
        emb_bot = eval_model.encode(bot_reply, convert_to_tensor=True)

        # Compute cosine similarity
        score = util.cos_sim(emb_ref, emb_bot).item()
        scores.append(score)

        print("\n-------------------------------------")
        print(f"User Input: {user_msg}")
        print(f"Chatbot Response: {bot_reply}")
        print(f"Reference Response: {reference}")
        print(f"Cosine Similarity: {score:.3f}")

    avg_score = sum(scores) / len(scores)
    print("\n=====================================")
    print(f"Average Chatbot Cosine Similarity: {avg_score:.3f}")
    print("=====================================")
    return avg_score


# Run the evaluation
average_similarity = evaluate_chatbot_response_similarity(chatbot, evaluation_pairs)
print(f"\nFinal Evaluation Complete! Average similarity score: {average_similarity:.3f}")
print("=" * 60)
