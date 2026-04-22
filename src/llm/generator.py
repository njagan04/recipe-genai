import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def generate_response(state: dict):
    recipes = state["filtered_recipes"]
    ingredients = state["ingredients"]

    # Format recipes cleanly
    recipe_text = ""
    for r in recipes:
        recipe_text += f"""
                        Recipe: {r['title']}
                        Available ingredients: {r['available_ingredients']}
                        Missing ingredients: {r['missing_ingredients']}
                        Steps: {r['steps']}
                        -------------------
                        """

    prompt = f"""
                User has these ingredients: {ingredients}

                Here are some candidate recipes:
                {recipe_text}

                Tasks:
                1. Choose the BEST recipe based on ingredient availability
                2. Clearly list missing ingredients
                3. Suggest substitutions if possible
                4. Provide clean step-by-step instructions
                5. Keep answer concise and structured
                """

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You are a smart cooking assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )

    return {
        **state,
        "final_output": response.choices[0].message.content
    }