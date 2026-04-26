# Recipe GenAI

An AI-powered recipe recommendation system. Users enter ingredients they have available and receive ranked recipe suggestions. Each recipe can be explored further through a conversational AI Chef interface.

---

## Table of Contents

- [Overview](#overview)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Architecture Overview](#architecture-overview)
- [Data Pipeline](#data-pipeline)
- [Recipe Data Schema](#recipe-data-schema)
- [Setup and Installation](#setup-and-installation)
- [Running the Project](#running-the-project)
- [API Reference](#api-reference)
- [Usage Instructions](#usage-instructions)
- [Environment Variables](#environment-variables)
- [Production Deployment](#production-deployment)
- [Performance Notes](#performance-notes)
- [Known Limitations](#known-limitations)
- [Troubleshooting](#troubleshooting)

---

## Overview

Recipe GenAI takes a free-text ingredient list from the user and runs it through a multi-stage AI pipeline to find the most relevant recipes from a pre-indexed dataset. Once a recipe is selected, users can ask follow-up questions to an AI Chef assistant powered by the Groq LLM API.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | React + Vite |
| Backend | FastAPI (Python) |
| AI Pipeline | LangGraph |
| Vector Search | FAISS (IVF index) |
| Embeddings | SentenceTransformers (all-mpnet-base-v2 / all-MiniLM-L6-v2) |
| Ingredient Extraction | Flan-T5 Small (runs locally, no API cost) |
| Chat and Reranking | Groq API (Llama 3.1 8B Instant) |
| Data Cleaning | Jupyter Notebook |

---

## Project Structure

```
recipe-genai/
├── .env                             API keys and runtime config
├── main.py                          Standalone CLI runner for pipeline testing
│
├── backend/
│   ├── api/
│   │   └── main.py                  FastAPI application entry point
│   ├── src/
│   │   ├── config.py                Centralised file path configuration
│   │   ├── graph/
│   │   │   ├── graph.py             Builds and compiles the LangGraph pipeline
│   │   │   ├── nodes.py             Logic for each pipeline node
│   │   │   └── state.py             TypedDict schema for pipeline state
│   │   ├── llm/
│   │   │   └── generator.py         Groq-based chat responses and LLM reranking
│   │   ├── retrieval/
│   │   │   ├── search.py            Hybrid FAISS search (semantic + lexical)
│   │   │   └── build_faiss.py       One-time script to build the FAISS index
│   │   └── data_pipeline/
│   │       └── clean_data.ipynb     Notebook to clean raw recipe data
│   ├── data/
│   │   └── processed/
│   │       └── recipes_cleaned.json Cleaned recipe dataset (input to FAISS builder)
│   ├── faiss_index/
│   │   ├── recipes_ivf.index        Trained FAISS IVF vector index
│   │   └── metadata.json            Recipe records aligned to FAISS index positions
│   ├── model_cache/                 Auto-created cache directory for HuggingFace models
│   └── requirements.txt
│
└── frontend/
    ├── index.html
    ├── vite.config.js
    └── src/
        ├── App.jsx                  Main React component (search + chat UI)
        ├── App.css                  Component styles
        ├── index.css                Global styles
        └── main.jsx                 React DOM entry point
```

---

## Architecture Overview

The backend exposes two endpoints. The `/api/search` endpoint runs the full LangGraph pipeline. The `/api/chat` endpoint handles conversational follow-up on a selected recipe.

### LangGraph Pipeline (4 nodes)

```
User Input (raw text)
        |
        v
  Node 1: Input Processing
  - Runs Flan-T5 Small locally to extract ingredient names
  - Falls back to text-based candidate extraction
  - Normalises and deduplicates the ingredient list
        |
        v
  Node 2: Retrieval
  - Embeds the ingredient list using SentenceTransformers
  - Searches FAISS index (candidate pool up to 10,000)
  - Applies hybrid scoring: 65% lexical + 35% semantic
  - Returns top 50 candidates
        |
        v
  Node 3: Filtering
  - Scores each candidate by ingredient overlap, coverage, and precision
  - Discards recipes with too many missing ingredients
  - Returns top 5 matches with available/missing ingredient breakdown
        |
        v
  Node 4: Reranking
  - Sends top 5 recipes to Groq LLM
  - LLM reorders based on culinary logic and flavour compatibility
  - Skipped if only 1 recipe remains after filtering
        |
        v
  Final Results (up to 5 ranked recipes)
```

### Pipeline State Schema

Each node reads from and writes to a shared `GraphState` dictionary:

| Field | Type | Description |
|---|---|---|
| `user_input` | str | Raw text entered by the user |
| `ingredients` | List[str] | Extracted and normalised ingredient names |
| `retrieved_recipes` | List[Dict] | Top 50 FAISS candidates |
| `filtered_recipes` | List[Dict] | Final ranked recipes returned to the API |
| `final_output` | str | Used only by the CLI runner |

### Filter Scoring Formula

Each recipe is scored using a weighted combination:

```
score = (coverage * 0.55) + (precision * 0.35) + (simplicity * 0.10) + complete_match_bonus

coverage  = overlapping ingredients / total user ingredients
precision = overlapping ingredients / total recipe ingredients
simplicity = 1 / (1 + missing_count)
complete_match_bonus = 0.2 if all user ingredients are in the recipe
```

### FAISS Search Scoring Formula

```
final_score = (semantic_rank_score * 0.35) + (lexical_score * 0.65)

semantic_rank_score = 1 / (1 + rank_position)
lexical_score = (ingredient_overlap * 1.0) + (title_word_overlap * 1.25)
```

---

## Data Pipeline

Raw recipe data must be cleaned and indexed before the backend can run. This is a one-time setup step.

### Step 1: Clean the raw data

Open and run the notebook:

```
backend/src/data_pipeline/clean_data.ipynb
```

This produces `backend/data/processed/recipes_cleaned.json`.

### Step 2: Build the FAISS index

Run from inside the `backend/` directory with the virtual environment active:

```bash
python -m src.retrieval.build_faiss
```

What this script does:
1. Loads `recipes_cleaned.json`
2. Generates embeddings using `all-mpnet-base-v2` (768-dimensional vectors)
3. Trains a FAISS IVF index with 256 cluster centroids
4. Saves the index to `faiss_index/recipes_ivf.index`
5. Saves corresponding recipe metadata to `faiss_index/metadata.json`

To limit the number of recipes indexed (useful for development):

```bash
MAX_RECIPES=10000 python -m src.retrieval.build_faiss
```

The default is 120,000 recipes.

### Embedding Model Auto-Detection

At search time, the embedding model is automatically chosen based on the index dimension:

| Index Dimension | Model Used |
|---|---|
| 768 | all-mpnet-base-v2 (higher accuracy) |
| 384 | all-MiniLM-L6-v2 (faster, smaller) |

The build script always uses `all-mpnet-base-v2`, so the search model will match automatically.

---

## Recipe Data Schema

The `recipes_cleaned.json` file must be a JSON array where each object follows this structure:

```json
{
  "title": "Garlic Chicken Stir Fry",
  "ingredients": [
    "chicken breast",
    "garlic",
    "soy sauce",
    "olive oil",
    "onion"
  ],
  "steps": [
    "Heat olive oil in a pan over medium heat.",
    "Add garlic and onion, cook until softened.",
    "Add chicken and cook through.",
    "Add soy sauce and stir to combine."
  ]
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `title` | string | Yes | Name of the recipe |
| `ingredients` | array of strings | Yes | List of ingredient names (plain text, no quantities) |
| `steps` | array of strings | Yes | Ordered cooking instructions |

---

## Setup and Installation

### Prerequisites

- Python 3.9 or higher
- Node.js 18 or higher
- A Groq API key (free tier available at https://console.groq.com)

### 1. Clone the repository

```bash
git clone <repository-url>
cd recipe-genai
```

### 2. Backend setup

```bash
cd backend

# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (Mac/Linux)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Create the environment file

Create a `.env` file inside the `backend/` directory:

```
LLM_PROVIDER=groq
GROQ_API_KEY=your_groq_api_key_here
LLM_MODEL=llama-3.1-8b-instant
```

### 4. Run the data pipeline (first time only)

```bash
# Clean raw data (run the notebook manually in Jupyter or VS Code)
# Then build the FAISS index:
python -m src.retrieval.build_faiss
```

### 5. Frontend setup

```bash
cd ../frontend
npm install
```

---

## Running the Project

### Backend

```bash
cd backend
.venv\Scripts\activate    # Windows
uvicorn api.main:app --reload
```

The API runs at: `http://localhost:8000`

Note: Run uvicorn from inside the `backend/` directory. Running it from the project root or from inside `backend/api/` will cause import errors because `src.*` imports are resolved relative to `backend/`.

### Frontend

```bash
cd frontend
npm run dev
```

The UI runs at: `http://localhost:5173`

### CLI Runner (for pipeline testing only)

```bash
cd recipe-genai
python main.py
```

This runs the LangGraph pipeline in the terminal without the API or frontend. Useful for testing the pipeline in isolation.

---

## API Reference

All endpoints are served from `http://localhost:8000`.

### GET /

Health check endpoint.

Response:
```json
{ "status": "Backend is running" }
```

---

### POST /api/search

Runs the full LangGraph pipeline and returns ranked recipe matches.

Request body:
```json
{
  "user_input": "chicken, garlic, onion"
}
```

Response:
```json
{
  "recipes": [
    {
      "title": "Garlic Chicken Stir Fry",
      "ingredients": ["chicken", "garlic", "onion", "soy sauce"],
      "steps": ["Heat oil...", "Add garlic..."],
      "available_ingredients": ["chicken", "garlic", "onion"],
      "missing_ingredients": ["soy sauce"],
      "match_score": 0.87
    }
  ]
}
```

Error responses:
- `400` — user_input is empty
- `500` — pipeline execution failed (detail field contains the error message)

---

### POST /api/chat

Sends a message to the AI Chef in the context of a selected recipe.

Request body:
```json
{
  "messages": [
    { "role": "user", "content": "Can I use butter instead of oil?" },
    { "role": "assistant", "content": "Yes, butter works well here..." },
    { "role": "user", "content": "How much butter should I use?" }
  ],
  "recipe": {
    "title": "Garlic Chicken Stir Fry",
    "ingredients": ["chicken", "garlic", "onion", "soy sauce"],
    "steps": ["Heat oil...", "Add garlic..."]
  }
}
```

The full conversation history must be sent on each request. The backend is stateless.

Response:
```json
{
  "response": "For this recipe, around 1 tablespoon of butter should be sufficient..."
}
```

---

## Usage Instructions

1. Open the app at `http://localhost:5173`
2. Type your available ingredients in the search box, separated by commas
   - Example: `eggs, tomato, cheddar cheese, onion`
3. Click **Generate Recipes**
4. The app displays up to 5 recipe cards:
   - Ingredients shown in green are ones you already have
   - Ingredients shown in red are ones you still need to buy
5. Click **Cook This** on any card to open the full recipe with step-by-step instructions
6. Use the **Ask the Chef** chat input at the bottom to ask questions about the recipe
   - Press Enter or click Send to submit
   - The AI Chef answers only in the context of the selected recipe

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `LLM_PROVIDER` | No | groq | LLM provider identifier (currently only groq is supported) |
| `GROQ_API_KEY` | Yes | None | API key from https://console.groq.com |
| `LLM_MODEL` | No | llama-3.1-8b-instant | Groq model name to use for chat and reranking |
| `MAX_RECIPES` | No | 120000 | Maximum recipes to index when running build_faiss.py |
| `TOKENIZERS_PARALLELISM` | Auto-set | false | Set to false automatically to suppress HuggingFace warnings |

---

## Production Deployment

### Backend

For production, replace uvicorn's development server with gunicorn and multiple workers:

```bash
pip install gunicorn
gunicorn api.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

Recommended settings:
- Set `-w` (workers) to `2 * CPU cores + 1`
- Run behind a reverse proxy such as nginx
- Store `.env` values as proper environment variables on the host, not in a file

### Frontend

Build the static bundle:

```bash
cd frontend
npm run build
```

The output is placed in `frontend/dist/`. Serve this directory with nginx, or deploy to a static hosting platform such as Vercel, Netlify, or Cloudflare Pages.

Update the API base URL in `App.jsx` to point to your production API server before building:

```js
// Change this in App.jsx before building for production
const res = await fetch('https://your-api-domain.com/api/search', { ... })
```

---

## Performance Notes

- **Cold start is slow**: When the backend starts, it loads Flan-T5 Small and the SentenceTransformer model into memory. This takes 20-60 seconds depending on hardware. Subsequent requests are fast.
- **FAISS search is fast**: Searching 120,000 recipes takes under 100ms using the IVF index with nprobe=32.
- **Groq reranking adds latency**: The LLM reranking step makes a network call to Groq and typically adds 1-2 seconds. It is skipped automatically when only one recipe passes the filter.
- **Model cache**: HuggingFace models are cached in `backend/model_cache/` after the first download. Do not delete this folder between runs.

---

## Known Limitations

- Ingredient extraction using Flan-T5 can produce unexpected output for unusual or highly specific ingredient names. A text-based fallback extractor runs alongside it to compensate.
- The system is optimised for English ingredient names. Non-English inputs are not currently handled.
- The FAISS index is static. Adding new recipes requires rebuilding the entire index by re-running `build_faiss.py`.
- The chat endpoint is stateless. The full conversation history must be sent by the frontend on every request.
- LLM reranking may occasionally fail to return a valid JSON index array. In that case the original filter-ranked order is preserved without raising an error.
- CORS is currently set to allow all origins (`*`). This should be restricted to your frontend domain in production.

---

## Troubleshooting

**Error: Could not import module "main"**

You are running `uvicorn main:app` from the wrong directory or with the wrong module path. Run this from inside `backend/`:

```bash
uvicorn api.main:app --reload
```

**Error: No module named 'fastapi'**

Dependencies are not installed in the active environment. Activate the virtual environment first:

```bash
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

**Error: No such file or directory: faiss_index/recipes_ivf.index**

The FAISS index has not been built yet. Run the build script:

```bash
python -m src.retrieval.build_faiss
```

**Error: GROQ_API_KEY not found**

The `.env` file is either missing, in the wrong location, or the variable is not set. Place `.env` inside the `backend/` directory and confirm it contains `GROQ_API_KEY=your_key`.

**Frontend shows "Error fetching recipes. Is the FastAPI backend running?"**

The backend is not running or is running on a different port. Start the backend with `uvicorn api.main:app --reload` and confirm it is accessible at `http://localhost:8000`.

**Model download is very slow on first run**

Flan-T5 and SentenceTransformer models are downloaded from HuggingFace on first use and cached in `backend/model_cache/`. This is a one-time download. Subsequent starts will load from cache.
