import { useState } from 'react'
import ReactMarkdown from 'react-markdown'

const capitalize = (text) => text ? text.charAt(0).toUpperCase() + text.slice(1) : ''

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
        <button className="btn-back" onClick={() => setSelectedRecipe(null)}>
          <svg fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M10 19l-7-7m0 0l7-7m-7 7h18"></path></svg>
          Back to Recipes
        </button>
        
        <div className="recipe-hero">
          <h1 className="recipe-title-large">{selectedRecipe.title}</h1>
        </div>
        
        <div className="recipe-content-grid">
          <div className="ingredients-panel">
            <h3>Ingredients</h3>
            <ul className="modern-list">
              {selectedRecipe.ingredients.map((ing, i) => (
                <li key={i}>{capitalize(ing)}</li>
              ))}
            </ul>
          </div>

          <div className="instructions-panel">
            <h3>Instructions</h3>
            <div className="instructions-list">
              {selectedRecipe.steps.map((step, i) => (
                <div key={i} className="instruction-step">
                  <span className="step-number">{i + 1}</span>
                  <p>{capitalize(step)}</p>
                </div>
              ))}
            </div>
          </div>
        </div>

        <div className="chat-container">
          <div className="chat-header">
            <h3>Chef's Assistant</h3>
            <p>Ask anything about this recipe</p>
          </div>
          
          <div className="chat-box">
            {chatHistory.map((msg, i) => (
              <div key={i} className={`chat-message ${msg.role}`}>
                <div className="message-sender">{msg.role === 'user' ? 'You' : 'Chef'}</div>
                <div className="message-content">
                  {msg.role === 'user' ? (
                    <p>{msg.content}</p>
                  ) : (
                    <ReactMarkdown>{msg.content}</ReactMarkdown>
                  )}
                </div>
              </div>
            ))}
          </div>
          
          <div className="chat-input-wrapper">
            <input 
              value={chatInput} 
              onChange={e => setChatInput(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && handleChat()}
              placeholder="Ask a question about this recipe..." 
              disabled={loading}
            />
            <button onClick={handleChat} disabled={loading}>
              {loading ? '...' : 'Send'}
            </button>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="landing-page">
      <nav className="navbar">
        <div className="logo">Recipe<span>AI</span></div>
      </nav>

      <div className="hero-section">
        <h1 className="hero-title">Discover what you can cook.</h1>
        <p className="hero-subtitle">Enter the ingredients you have, and let our AI chef find the perfect recipe for you.</p>
        
        <div className="search-widget">
          <textarea 
            placeholder="e.g., chicken breast, garlic, olive oil, onions..." 
            value={ingredients}
            onChange={(e) => setIngredients(e.target.value)}
          />
          <button className="btn-primary-large" onClick={handleSearch} disabled={loading}>
            {loading ? 'Analyzing Ingredients...' : 'Generate Recipes'}
          </button>
        </div>
      </div>

      {searchResults.length > 0 && (
        <div className="results-section">
          <h2 className="section-title">Your Top Matches</h2>
          <div className="recipes-grid">
            {searchResults.map((recipe, idx) => (
              <div key={idx} className="recipe-card">
                <div className="recipe-card-content">
                  <h3 className="card-title">{recipe.title}</h3>
                  
                  <div className="badge-group">
                    <span className="badge-label">Available:</span>
                    <div className="badges">
                      {recipe.available_ingredients?.map((ing, i) => (
                        <span key={i} className="badge available">{capitalize(ing)}</span>
                      ))}
                    </div>
                  </div>

                  {recipe.missing_ingredients?.length > 0 && (
                    <div className="badge-group">
                      <span className="badge-label">Missing:</span>
                      <div className="badges">
                        {recipe.missing_ingredients.map((ing, i) => (
                          <span key={i} className="badge missing">{capitalize(ing)}</span>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
                
                <button 
                  className="btn-cook" 
                  onClick={() => { setSelectedRecipe(recipe); setChatHistory([]); }}
                >
                  View Full Recipe
                </button>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

export default App
