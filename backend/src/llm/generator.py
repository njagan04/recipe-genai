import os
import json
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

def get_chat_response(messages, recipe_context):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return "Error: GROQ_API_KEY not found in environment."

    model_name = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
    llm = ChatGroq(
        api_key=api_key,
        model=model_name,
        temperature=0.3,
        max_tokens=1024
    )

    # Format recipe context
    title = recipe_context.get("title", "Unknown")
    ingredients = ", ".join(recipe_context.get("ingredients", []))
    steps = "\n".join([f"{i+1}. {s}" for i, s in enumerate(recipe_context.get("steps", []))])

    system_prompt = f"""You are a helpful AI Chef assistant.
The user has selected the following recipe:
Title: {title}
Ingredients: {ingredients}
Steps:
{steps}

Your job is to answer the user's questions strictly based on this recipe. 
Do not suggest completely different recipes unless explicitly asked. Be encouraging, concise, and helpful."""

    langchain_messages = [SystemMessage(content=system_prompt)]
    
    for msg in messages:
        if msg["role"] == "user":
            langchain_messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            langchain_messages.append(AIMessage(content=msg["content"]))

    response = llm.invoke(langchain_messages)
    return response.content

def rerank_recipes_with_llm(user_ingredients: list, recipes: list):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or len(recipes) <= 1:
        return recipes
        
    try:
        model_name = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
        llm = ChatGroq(
            api_key=api_key,
            model=model_name,
            temperature=0.0
        )
        
        recipes_context = ""
        for i, r in enumerate(recipes):
            recipes_context += f"Index {i}:\nTitle: {r['title']}\nIngredients: {', '.join(r['ingredients'])}\n\n"
            
        sys_msg = SystemMessage(content="You are an expert culinary AI. Your task is to rank recipes based on how well they utilize the user's available ingredients from a culinary perspective.")
        
        user_msg = HumanMessage(content=f"""
User's available ingredients: {', '.join(user_ingredients)}

Here are the top matches found mathematically:
{recipes_context}

Rank these recipes from best to worst based on culinary logic. The best recipe should creatively and cohesively use the user's provided ingredients (especially strong savory or sweet ingredients) without clashing flavor profiles.

Return ONLY a JSON array of the indices, ordered from best to worst match (e.g., [2, 0, 1, 4, 3]). Do not include any explanation or markdown formatting. Just the JSON array.
""")
        
        resp = llm.invoke([sys_msg, user_msg])
        content = resp.content.strip()
        
        # Cleanup markdown formatting if the model still includes it
        if content.startswith("```json"):
            content = content[7:-3].strip()
        elif content.startswith("```"):
            content = content[3:-3].strip()
            
        indices = json.loads(content)
        
        # Reorder based on valid indices returned
        reranked = [recipes[i] for i in indices if i < len(recipes)]
        
        # Append any recipes that were missing from the LLM's response
        for i in range(len(recipes)):
            if i not in indices:
                reranked.append(recipes[i])
                
        return reranked
    except Exception as e:
        print(f"Reranking error: {e}")
        return recipes