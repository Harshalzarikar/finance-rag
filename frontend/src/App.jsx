import { useState, useRef, useEffect } from 'react';
import './index.css';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

function App() {
  const [messages, setMessages] = useState([
    {
      id: 1,
      role: 'bot',
      content: 'Welcome to the Quantitative Finance AI. Ask me any question based on the ArXiv mathematical physics and quantitative finance research database.',
      sources: []
    }
  ]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessage = { id: Date.now(), role: 'user', content: input, sources: [] };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      // In production (Vercel), this URL should be updated to point to the Hugging Face Space URL
      const response = await fetch(`${API_URL}/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query: input }),
      });

      if (!response.ok) throw new Error('Network response was not ok');
      
      const data = await response.json();
      
      setMessages(prev => [...prev, {
        id: Date.now() + 1,
        role: 'bot',
        content: data.answer,
        sources: data.sources || []
      }]);
    } catch (error) {
      console.error('Error fetching chat response:', error);
      setMessages(prev => [...prev, {
        id: Date.now() + 1,
        role: 'bot',
        content: 'Sorry, I encountered an error communicating with the backend. Please ensure the FastAPI server is running.',
        sources: []
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="app-container">
      <div className="header">
        <h1>QuantRAG AI</h1>
        <p>Enterprise Mathematical Finance & Research Intelligence</p>
      </div>

      <div className="chat-container glass-panel">
        <div className="message-list">
          {messages.map((msg) => (
            <div key={msg.id} className={`message ${msg.role}`}>
              <div className="message-content">
                {msg.content}
              </div>
              
              {msg.sources && msg.sources.length > 0 && (
                <SourceDropdown sources={msg.sources} />
              )}
            </div>
          ))}
          
          {isLoading && (
            <div className="message bot">
              <div className="typing-indicator">
                <div className="dot"></div>
                <div className="dot"></div>
                <div className="dot"></div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        <div className="input-area">
          <form className="input-form" onSubmit={handleSubmit}>
            <input
              type="text"
              className="chat-input"
              placeholder="Ask about stochastic volatility, Heston models, options pricing..."
              value={input}
              onChange={(e) => setInput(e.target.value)}
              disabled={isLoading}
            />
            <button type="submit" className="send-button" disabled={!input.trim() || isLoading}>
              Send
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}

function SourceDropdown({ sources }) {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <div className="sources-container">
      <button 
        className="source-toggle" 
        onClick={() => setIsOpen(!isOpen)}
        type="button"
      >
        {isOpen ? '▼' : '▶'} View {sources.length} Citations
      </button>
      
      {isOpen && (
        <div className="source-cards">
          {sources.map((source, index) => (
            <div key={index} className="source-card">
              <span className="source-title">{source.source}</span>
              <span className="source-text">{source.content}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default App;
