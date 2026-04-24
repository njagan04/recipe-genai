def _format_recipe_response(recipe: dict) -> str:
    missing = recipe.get("missing_ingredients", [])
    available = recipe.get("available_ingredients", [])
    steps = recipe.get("steps", [])

    if missing:
        missing_text = "\n".join(f"- {item}" for item in missing)
    else:
        missing_text = "No missing ingredients"

    if available:
        available_text = ", ".join(available)
    else:
        available_text = "None"

    if steps:
        steps_text = "\n".join(f"{idx + 1}. {step}" for idx, step in enumerate(steps))
    else:
        steps_text = "1. No steps available"

    return (
        f"Recipe: {recipe.get('title', 'Unknown recipe')}\n\n"
        f"Available ingredients: {available_text}\n\n"
        f"Missing ingredients:\n{missing_text}\n\n"
        f"Steps:\n{steps_text}"
    )


def generate_response(state: dict):
    recipes = state.get("filtered_recipes", [])

    if not recipes:
        return {
            "final_output": "Could not find any matching recipes."
        }

    return {
        "final_output": _format_recipe_response(recipes[0])
    }