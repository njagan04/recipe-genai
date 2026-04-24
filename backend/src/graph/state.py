#data passed between steps

from typing import List, Dict, TypedDict

class GraphState(TypedDict, total=False):
    user_input: str
    ingredients: List[str]
    retrieved_recipes: List[Dict]
    filtered_recipes: List[Dict]
    final_output: str