# how steps are connected

from langgraph.graph import StateGraph
from src.graph.nodes import process_input, retrieve_recipes, filter_recipes, detect_missing_ingredients
from src.llm.generator import generate_response


def build_graph():
    graph = StateGraph(dict)

    graph.add_node("input", process_input)
    graph.add_node("retrieve", retrieve_recipes)
    graph.add_node("filter", filter_recipes)
    graph.add_node("generate", generate_response)
    graph.add_node("missing", detect_missing_ingredients)

    graph.set_entry_point("input")

    graph.add_edge("input", "retrieve")
    graph.add_edge("retrieve", "filter")
    graph.add_edge("filter", "generate")
    graph.add_edge("filter", "missing")
    graph.add_edge("missing", "generate")

    return graph.compile()