import { useState } from 'react'

function App() {
  const [ingredients, setIngredients] = useState('')
  const [searchResults, setSearchResults] = useState([])
  const [selectedRecipe, setSelectedRecipe] = useState(null)
  const [chatHistory, setChatHistory] = useState([])
  const [chatInput, setChatInput] = useState('')
  const [loading, setLoading] = useState(false)

  const handleSearch = async () => {
    if (!ingredients.trim()) return
    setLoading(true)
    try {
      const res = await fetch('http://localhost:8000/api/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_input: ingredients })
      })
      const data = await res.json()
      setSearchResults(data.recipes || [])
    } catch (err) {
      console.error(err)
      alert("Error fetching recipes. Is the FastAPI backend running?")
    }
    setLoading(false)
  }

  const handleChat = async () => {
    if (!chatInput.trim()) return
    const newMsg = { role: 'user', content: chatInput }
    const updatedHistory = [...chatHistory, newMsg]
    setChatHistory(updatedHistory)
    setChatInput('')
    setLoading(true)
    
    try {
      const res = await fetch('http://localhost:8000/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ messages: updatedHistory, recipe: selectedRecipe })
      })
      const data = await res.json()
      setChatHistory([...updatedHistory, { role: 'assistant', content: data.response }])
    } catch (err) {
      console.error(err)
      alert("Error sending message. Is the FastAPI backend running?")
    }
    setLoading(false)
  }

  if (selectedRecipe) {
    return (
      <div className="app-container">
        <button className="btn btn-secondary" onClick={() => setSelectedRecipe(null)}>
          ← Back to Search
        </button>
        
        <div className="recipe-detail">
          <h1 className="recipe-title">{selectedRecipe.title}</h1>
          
          <h3>Ingredients</h3>
          <ul>
            {selectedRecipe.ingredients.map((ing, i) => (
              <li key={i}>{ing}</li>
            ))}
          </ul>

          <h3>Instructions</h3>
          {selectedRecipe.steps.map((step, i) => (
            <p key={i}><strong>{i + 1}.</strong> {step}</p>
          ))}
        </div>

        <div className="chat-container">
          <h3>Ask the Chef</h3>
          {chatHistory.map((msg, i) => (
            <div key={i} className={`chat-message ${msg.role}`}>
              <strong>{msg.role === 'user' ? 'You: ' : 'Chef: '}</strong>
              {msg.content}
            </div>
          ))}
          
          <div className="chat-input">
            <input 
              value={chatInput} 
              onChange={e => setChatInput(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && handleChat()}
              placeholder="Ask a question about this recipe..." 
              disabled={loading}
            />
            <button className="btn" onClick={handleChat} disabled={loading}>
              {loading ? '...' : 'Send'}
            </button>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="app-container">
      <div className="header">
        <h1>Recipe <span>AI</span></h1>
        <p>Discover what you can cook with what you have.</p>
      </div>

      <div className="search-box">
        <textarea 
          placeholder="e.g., chicken, rice, garlic, onions" 
          value={ingredients}
          onChange={(e) => setIngredients(e.target.value)}
        />
        <button className="btn" onClick={handleSearch} disabled={loading}>
          {loading ? 'Analyzing...' : 'Generate Recipes'}
        </button>
      </div>

      {searchResults.length > 0 && (
        <div>
          <h3 style={{marginBottom: '24px'}}>Top Matches</h3>
          {searchResults.map((recipe, idx) => (
            <div key={idx} className="recipe-card">
              <h2 className="recipe-title">{recipe.title}</h2>
              
              <div className="ingredient-section">
                <span className="ingredient-label">You have:</span>
                <div className="badge-container">
                  {recipe.available_ingredients?.map((ing, i) => (
                    <span key={i} className="ingredient-badge">{ing}</span>
                  ))}
                </div>
              </div>

              {recipe.missing_ingredients?.length > 0 && (
                <div className="ingredient-section">
                  <span className="ingredient-label">You need:</span>
                  <div className="badge-container">
                    {recipe.missing_ingredients.map((ing, i) => (
                      <span key={i} className="ingredient-badge missing">{ing}</span>
                    ))}
                  </div>
                </div>
              )}
              
              <div className="cook-btn-wrapper">
                <button 
                  className="btn" 
                  onClick={() => { setSelectedRecipe(recipe); setChatHistory([]); }}
                >
                  Cook This Recipe
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

export default App
