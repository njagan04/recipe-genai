# -------------------------------
# Node 1 — Input Processing
# -------------------------------
def process_input(state: dict):
    user_input = state["user_input"]

    ingredients = [
        i.strip().lower()
        for i in user_input.split(",")
        if i.strip()
    ]

    return {
        "ingredients": ingredients
    }


# -------------------------------
# Node 2 — Retrieval
# -------------------------------
from src.retrieval.search import search


def retrieve_recipes(state: dict):
    query = " ".join(state["ingredients"])

    results = search(query, top_k=10)

    return {
        "retrieved_recipes": results
    }


# -------------------------------
# Node 3 — Filtering + Ranking
# -------------------------------
def filter_recipes(state: dict):
    user_ingredients = set(state["ingredients"])

    scored = []

    for recipe in state["retrieved_recipes"]:
        recipe_ingredients = set(recipe["ingredients"])

        overlap = len(user_ingredients & recipe_ingredients)

        # weighted score (important improvement)
        score = overlap / len(recipe_ingredients)

        if overlap > 0:
            scored.append((score, recipe))

    # sort by best match
    scored.sort(key=lambda x: x[0], reverse=True)

    filtered = [r for score, r in scored[:5]]

    return {
        "filtered_recipes": filtered
    }


# -------------------------------
# Node 4 — Missing Ingredient Detection
# -------------------------------
def detect_missing_ingredients(state: dict):
    user_ingredients = set(state["ingredients"])

    enriched_recipes = []

    for recipe in state["filtered_recipes"]:
        recipe_ingredients = set(recipe["ingredients"])

        missing = [m for m in recipe_ingredients if m not in user_ingredients]
        available = list(recipe_ingredients & user_ingredients)

        enriched_recipes.append({
            "title": recipe["title"],
            "ingredients": recipe["ingredients"],
            "steps": recipe["steps"],
            "missing_ingredients": missing,
            "available_ingredients": available
        })

    return {
        "filtered_recipes": enriched_recipes
    }