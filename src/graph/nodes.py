# Collections of steps

def process_input(state: dict):
    user_input = state["user_input"]

    ingredients = [
        i.strip().lower()
        for i in user_input.split(",")
        if i.strip()
    ]

    return {
        **state,
        "ingredients": ingredients
    }


from src.retrieval.search import search


def retrieve_recipes(state: dict):
    query = " ".join(state["ingredients"])

    results = search(query, top_k=10)

    return {
        **state,
        "retrieved_recipes": results
    }

def filter_recipes(state: dict):
    user_ingredients = set(state["ingredients"])

    scored = []

    for recipe in state["retrieved_recipes"]:
        recipe_ingredients = set(recipe["ingredients"])

        overlap = len(user_ingredients & recipe_ingredients)

        # NEW: weighted score
        score = overlap / len(recipe_ingredients)

        if overlap > 0:  # avoid useless recipes
            scored.append((score, recipe))

    # sort by score (descending)
    scored.sort(key=lambda x: x[0], reverse=True)

    # take top 5
    filtered = [r for score, r in scored[:5]]

    return {
        **state,
        "filtered_recipes": filtered
    }

def detect_missing_ingredients(state: dict):
    user_ingredients = set(state["ingredients"])

    enriched_recipes = []

    for recipe in state["filtered_recipes"]:
        recipe_ingredients = set(recipe["ingredients"])

        missing = list(recipe_ingredients - user_ingredients)
        available = list(recipe_ingredients & user_ingredients)

        enriched_recipes.append({
            "title": recipe["title"],
            "ingredients": recipe["ingredients"],
            "steps": recipe["steps"],
            "missing_ingredients": missing,
            "available_ingredients": available
        })

    return {
        **state,
        "filtered_recipes": enriched_recipes
    }