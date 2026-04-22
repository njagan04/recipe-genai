import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def generate_response(state: dict):
    recipes = state.get("filtered_recipes", [])
    ingredients = state.get("ingredients", [])

    # ---------------------------
    # Format recipes cleanly
    # ---------------------------
    recipe_text = ""
    for r in recipes:
        recipe_text += f"""
                            Recipe: {r['title']}
                            Available: {r['available_ingredients']}
                            Missing: {r['missing_ingredients']}
                            Steps:
                            {chr(10).join(r['steps'])}
                            -------------------
                            """

    # ---------------------------
    # Strict Prompt (cleaned)
    # ---------------------------
    prompt = f"""
                User ingredients: {ingredients}

                You are given candidate recipes.

                STRICT RULES:
                - Select ONLY ONE best recipe
                - Return ONLY ONE recipe
                - DO NOT list multiple recipes
                - DO NOT repeat instructions
                - DO NOT explain anything extra
                - DO NOT output analysis
                - DO NOT invent ingredients
                - ONLY use given data

                Recipes:
                {recipe_text}

                OUTPUT FORMAT:

                Recipe: <name>

                Missing ingredients:
                - item1
                - item2
                OR
                No missing ingredients

                Steps:
                1. ...
                2. ...
            """

    # ---------------------------
    # LLM Call
    # ---------------------------
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You are a strict cooking assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2   # lower = more controlled output
    )

    output = response.choices[0].message.content.strip()

    # ---------------------------
    # Output Guardrail
    # ---------------------------
    if "Recipe:" not in output or "Steps:" not in output:
        output = "Could not generate proper response. Try again."

    return {
        "final_output": output
    }