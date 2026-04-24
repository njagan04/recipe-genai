import streamlit as st
import os
import warnings
from dotenv import load_dotenv

# Suppress HuggingFace/SentenceTransformer terminal logs
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

load_dotenv()

from src.graph.graph import build_graph
from src.llm.generator import get_chat_response

st.set_page_config(page_title="Recipe GenAI", layout="centered", initial_sidebar_state="collapsed")

def inject_custom_css():
    st.markdown("""
    <style>
        /* Sleek modern design - NO effects */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap');
        
        html, body, [class*="css"]  {
            font-family: 'Inter', sans-serif;
            background-color: #ffffff;
            color: #333333;
        }
        
        /* Hide default Streamlit elements */
        [data-testid="stSidebar"] { display: none !important; }
        [data-testid="collapsedControl"] { display: none !important; }
        header { visibility: hidden; }
        footer { visibility: hidden; }
        
        .block-container {
            padding-top: 2rem;
            max-width: 800px;
        }

        /* Clean borders for search results, no shadows/hover effects */
        div[data-testid="stVerticalBlock"] > div > div[data-testid="stVerticalBlock"] {
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 24px;
            margin-bottom: 15px;
            background-color: transparent;
        }
        
        /* Dark mode handling */
        @media (prefers-color-scheme: dark) {
            html, body, [class*="css"] {
                background-color: #121212;
                color: #e0e0e0;
            }
            div[data-testid="stVerticalBlock"] > div > div[data-testid="stVerticalBlock"] {
                border: 1px solid #333333;
            }
        }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def get_graph():
    return build_graph()

def main():
    inject_custom_css()
    
    st.markdown("<h1 style='text-align: center; font-weight: 600;'>Recipe AI</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #888;'>Discover what you can cook with what you have.</p>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    app = get_graph()

    if "selected_recipe" not in st.session_state:
        st.session_state.selected_recipe = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "search_results" not in st.session_state:
        st.session_state.search_results = []

    # VIEW 1: Search View
    if not st.session_state.selected_recipe:
        user_input = st.text_area("What ingredients do you have?", placeholder="e.g., chicken, rice, garlic, onions")
        
        if st.button("Generate Recipes", type="primary", use_container_width=True):
            if user_input.strip():
                with st.spinner("Analyzing ingredients & Reranking..."):
                    try:
                        result = app.invoke({"user_input": user_input})
                        st.session_state.search_results = result.get("filtered_recipes", [])
                        st.session_state.chat_history = []
                    except Exception as e:
                        st.error(f"Search error: {e}")
            else:
                st.warning("Please enter some ingredients.")

        if st.session_state.search_results:
            st.markdown("<br>### Top Matches", unsafe_allow_html=True)
            for idx, recipe in enumerate(st.session_state.search_results):
                with st.container():
                    st.subheader(recipe['title'])
                    
                    available = recipe.get('available_ingredients', [])
                    missing = recipe.get('missing_ingredients', [])
                    
                    if available:
                        st.markdown("**You have:** " + ", ".join([f"`{ing}`" for ing in available]))
                    if missing:
                        st.markdown("**You need:** " + ", ".join([f"`{ing}`" for ing in missing]))
                    
                    if st.button("Cook This", key=f"select_{idx}", type="primary"):
                        st.session_state.selected_recipe = recipe
                        st.session_state.chat_history = []
                        st.rerun()

    # VIEW 2: Chat & Details View
    else:
        recipe = st.session_state.selected_recipe
        
        if st.button("← Back to Search"):
            st.session_state.selected_recipe = None
            st.session_state.chat_history = []
            st.rerun()
            
        st.markdown(f"<h2>{recipe['title']}</h2>", unsafe_allow_html=True)
        
        st.markdown("### Ingredients")
        ingredients_list = "\n".join([f"- {ing}" for ing in recipe.get("ingredients", [])])
        st.markdown(ingredients_list)
        
        st.markdown("### Instructions")
        steps_list = "\n\n".join([f"**{i+1}.** {step}" for i, step in enumerate(recipe.get("steps", []))])
        st.markdown(steps_list)
            
        st.markdown("---")
        st.markdown("### Ask the Chef")
        
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
        
        if prompt := st.chat_input("Ask a question about this recipe..."):
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.chat_message("assistant"):
                with st.spinner("..."):
                    try:
                        response = get_chat_response(
                            st.session_state.chat_history, 
                            st.session_state.selected_recipe
                        )
                        st.markdown(response)
                        st.session_state.chat_history.append({"role": "assistant", "content": response})
                    except Exception as e:
                        st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
