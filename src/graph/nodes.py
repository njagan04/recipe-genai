# Node 1 — Input Processing

import json
import re

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.config import HF_CACHE_DIR


tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small", cache_dir=HF_CACHE_DIR)
ingredient_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small", cache_dir=HF_CACHE_DIR)


MULTI_WORD_INGREDIENTS = {
    "green onion",
    "spring onion",
    "red onion",
    "soy sauce",
    "fish sauce",
    "oyster sauce",
    "olive oil",
    "sesame oil",
    "coconut oil",
    "cream cheese",
    "sour cream",
    "heavy cream",
    "brown sugar",
    "powdered sugar",
    "black pepper",
    "white pepper",
    "garlic powder",
    "onion powder",
    "baking powder",
    "baking soda",
    "lemon juice",
    "lime juice",
    "coconut milk",
}


INGREDIENT_SYNONYMS = {
    "scallion": "green onion",
    "spring onion": "green onion",
    "shallot": "onion",
    "capsicum": "bell pepper",
    "coriander": "cilantro",
    "curd": "yogurt",
    "garbanzo": "chickpea",
    "chilli": "chili",
    "mince": "ground meat",
    "maida": "all purpose flour",
    "atta": "wheat flour",
}


def ingredient_extractor(prompt: str):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = ingredient_model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def _normalize_ingredient(text: str):
    text = re.sub(r"[^a-z0-9\s]", " ", str(text).lower())
    text = " ".join(text.split())

    if not text:
        return ""

    quantity_tokens = {
        "a", "an", "and", "about", "approx", "approximate", "around",
        "cup", "cups", "tablespoon", "tablespoons", "tbsp", "teaspoon",
        "teaspoons", "tsp", "gram", "grams", "g", "kg", "ml", "l",
        "liter", "liters", "pinch", "pinches", "dash", "dashes",
        "slice", "slices", "piece", "pieces", "package", "packages",
        "can", "cans", "bottle", "bottles", "lb", "lbs", "oz", "ounces",
    }

    tokens = []
    for word in text.split():
        if re.fullmatch(r"\d+(?:/\d+)?(?:\.\d+)?", word):
            continue
        if word in quantity_tokens:
            continue
        tokens.append(word)

    text = " ".join(tokens)
    if not text:
        return ""

    normalized_words = []
    for word in text.split():
        if len(word) > 4 and word.endswith("ies"):
            word = word[:-3] + "y"
        elif len(word) > 3 and word.endswith("es") and not word.endswith(("ses", "xes", "zes", "ches", "shes")):
            word = word[:-2]
        elif len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
            word = word[:-1]

        normalized_words.append(word)

    normalized = " ".join(normalized_words)
    normalized = INGREDIENT_SYNONYMS.get(normalized, normalized)
    return normalized


def _parse_ingredient_list(content: str):
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)

    match = re.search(r"\[[\s\S]*\]", content)
    if match:
        content = match.group(0)

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        content = re.sub(r"^(ingredients?|ingredient list)\s*:\s*", "", content, flags=re.IGNORECASE)
        parsed = [part.strip() for part in re.split(r"[,\n;]+", content) if part.strip()]

    if not isinstance(parsed, list):
        parsed = [str(parsed)]

    ingredients = []
    for item in parsed:
        normalized = _normalize_ingredient(item)
        if normalized:
            ingredients.append(normalized)

    ingredients = list(dict.fromkeys(ingredients))

    return ingredients


def _extract_candidates_from_text(user_input: str):
    filler_tokens = {
        "i", "have", "some", "ingredient", "ingredients", "with", "what", "can", "make",
        "cook", "using", "like", "that", "this", "for", "to", "do", "a", "an", "the",
    }

    chunks = re.split(r"\s*(?:,|;|/|&|\band\b|\bwith\b|\bplus\b)\s*", user_input.lower())
    candidates = []

    for chunk in chunks:
        chunk = re.sub(r"[^a-z0-9\s]", " ", chunk)
        words = [w for w in chunk.split() if w not in filler_tokens]
        if not words:
            continue

        candidate = _normalize_ingredient(" ".join(words))
        if not candidate:
            continue

        candidates.append(candidate)

    return list(dict.fromkeys(candidates))


def process_input(state: dict):
    user_input = state["user_input"]

    prompt = f"""
Extract the ingredient names from this cooking request.

Return ONLY a JSON array of strings.
Rules:
- Include only ingredient names, not quantities
- Remove filler words like "have", "some", "ingredients", "what", "can", "I", "do"
- If the user says "egg and onion", return ["egg", "onion"]
- Normalize to simple ingredient names

User request: {user_input}
"""

    response = ingredient_extractor(prompt)
    ingredients = _parse_ingredient_list(response)
    text_candidates = _extract_candidates_from_text(user_input)

    # Keep LLM extraction as primary, but merge obvious candidates from the user text.
    ingredients = list(dict.fromkeys(ingredients + text_candidates))

    return {
        "ingredients": ingredients
    }


# Node 2 — Retrieval

from src.retrieval.search import search


def retrieve_recipes(state: dict):
    # Pass the raw list of ingredients so the search function preserves multi-word phrases
    results = search(state["ingredients"], top_k=50)

    return {
        "retrieved_recipes": results
    }


# Node 3 — Filtering + Ranking

def filter_recipes(state: dict):
    user_ingredients = {
        _normalize_ingredient(ingredient)
        for ingredient in state["ingredients"]
        if _normalize_ingredient(ingredient)
    }

    scored = []
    max_missing_allowed = max(2, len(user_ingredients))

    for position, recipe in enumerate(state["retrieved_recipes"]):
        recipe_ingredients = [
            _normalize_ingredient(ingredient)
            for ingredient in recipe["ingredients"]
        ]
        recipe_ingredients = [ingredient for ingredient in recipe_ingredients if ingredient]
        recipe_ingredient_set = set(recipe_ingredients)

        overlap_items = user_ingredients & recipe_ingredient_set
        overlap = len(overlap_items)

        if overlap == 0:
            continue

        coverage_score = overlap / max(len(user_ingredients), 1)
        precision_score = overlap / max(len(recipe_ingredient_set), 1)

        missing_count = max(len(recipe_ingredient_set) - overlap, 0)
        if missing_count > max_missing_allowed:
            continue

        # Prefer recipes that can mostly be made with available ingredients.
        simplicity_score = 1 / (1 + missing_count)

        complete_match_bonus = 0.2 if user_ingredients.issubset(recipe_ingredient_set) else 0
        score = (
            (coverage_score * 0.55)
            + (precision_score * 0.35)
            + (simplicity_score * 0.10)
            + complete_match_bonus
        )

        scored.append((score, missing_count, position, recipe))

    # sort by best match
    scored.sort(key=lambda item: (-item[0], item[1], item[2]))

    filtered = []
    for score, _, _, recipe in scored[:5]:
        recipe_ingredients = recipe["ingredients"]
        available = [
            ingredient
            for ingredient in recipe_ingredients
            if _normalize_ingredient(ingredient) in user_ingredients
        ]
        missing = [
            ingredient
            for ingredient in recipe_ingredients
            if _normalize_ingredient(ingredient) not in user_ingredients
        ]

        filtered.append({
            "title": recipe["title"],
            "ingredients": recipe["ingredients"],
            "steps": recipe["steps"],
            "missing_ingredients": missing,
            "available_ingredients": available,
            "match_score": score,
        })

    return {
        "filtered_recipes": filtered
    }

# Node 4 — LLM Reranking

from src.llm.generator import rerank_recipes_with_llm

def rerank_recipes(state: dict):
    filtered = state.get("filtered_recipes", [])
    if len(filtered) <= 1:
        return {"filtered_recipes": filtered}
        
    user_ingredients = state.get("ingredients", [])
    reranked = rerank_recipes_with_llm(user_ingredients, filtered)
    
    return {"filtered_recipes": reranked}

