from src.graph.graph import build_graph

app = build_graph()


if __name__ == "__main__":
    user_input = input("Enter ingredients: ")

    result = app.invoke({
        "user_input": user_input
    })

    print("\n=== RESULT ===\n")
    print(result["final_output"])