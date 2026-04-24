# how steps are connected

from langgraph.graph import StateGraph, END
from src.graph.nodes import process_input, retrieve_recipes, filter_recipes, rerank_recipes
from src.graph.state import GraphState


def build_graph():

    graph = StateGraph(GraphState)

    graph.add_node("input", process_input)
    graph.add_node("retrieve", retrieve_recipes)
    graph.add_node("filter", filter_recipes)
    graph.add_node("rerank", rerank_recipes)

    graph.set_entry_point("input")

    graph.add_edge("input", "retrieve")
    graph.add_edge("retrieve", "filter")
    graph.add_edge("filter", "rerank")
    graph.add_edge("rerank", END)

    return graph.compile()