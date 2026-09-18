import { useEffect, useRef, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import './index.css';

// A relative path works in both environments: Vite proxies /api to the backend
// during development, and Caddy proxies it to the api container in production.
const API_URL = import.meta.env.VITE_API_URL || '/api';
const API_KEY = import.meta.env.VITE_API_KEY || '';

const WELCOME = {
  id: 1,
  role: 'bot',
  content:
    'Welcome to the Quantitative Finance AI. Ask me any question based on the indexed arXiv quantitative finance research corpus.',
  sources: [],
  cached: false,
};

class ApiError extends Error {
  constructor(message, status) {
    super(message);
    this.status = status;
  }
}

function describeError(error) {
  if (error instanceof ApiError) {
    if (error.status === 401) return 'Unauthorized. Check that the X-API-Key is configured correctly.';
    if (error.status === 429) return 'Rate limit exceeded. Please wait a moment and try again.';
    if (error.status === 503) return 'The service is not ready yet. Ingestion may still be running.';
    return `The backend returned an error (HTTP ${error.status}).`;
  }
  return 'Could not reach the backend. Make sure the API server is running.';
}

async function streamAnswer(query, onEvent, signal) {
  const response = await fetch(`${API_URL}/chat/stream`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...(API_KEY ? { 'X-API-Key': API_KEY } : {}),
    },
    body: JSON.stringify({ query }),
    signal,
  });

  if (!response.ok) {
    let detail = '';
    try {
      const body = await response.json();
      detail = body?.detail ?? '';
    } catch {
      detail = '';
    }
    throw new ApiError(detail || `HTTP ${response.status}`, response.status);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    const frames = buffer.split('\n\n');
    buffer = frames.pop() ?? '';
    for (const frame of frames) {
      const line = frame.trim();
      if (!line.startsWith('data:')) continue;
      try {
        onEvent(JSON.parse(line.slice(5).trim()));
      } catch {
        // Ignore malformed frames rather than aborting the stream.
      }
    }
  }
}

function App() {
  const [messages, setMessages] = useState([WELCOME]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSubmit = async (event) => {
    event.preventDefault();
    const query = input.trim();
    if (!query || isLoading) return;

    const userMessage = { id: Date.now(), role: 'user', content: query, sources: [] };
    const botId = Date.now() + 1;

    setMessages((prev) => [
      ...prev,
      userMessage,
      { id: botId, role: 'bot', content: '', sources: [], cached: false },
    ]);
    setInput('');
    setIsLoading(true);

    const patch = (updater) =>
      setMessages((prev) => prev.map((message) => (message.id === botId ? updater(message) : message)));

    try {
      await streamAnswer(query, (event) => {
        if (event.type === 'sources') {
          patch((message) => ({ ...message, sources: event.sources ?? [], cached: Boolean(event.cached) }));
        } else if (event.type === 'token') {
          patch((message) => ({ ...message, content: message.content + event.value }));
        } else if (event.type === 'error') {
          patch((message) => ({ ...message, content: event.detail ?? 'The answer could not be generated.' }));
        }
      });
    } catch (error) {
      console.error('Streaming request failed:', error);
      patch((message) => ({ ...message, content: describeError(error) }));
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="app-container">
      <div className="header">
        <h1>QuantRAG AI</h1>
        <p>Enterprise Mathematical Finance &amp; Research Intelligence</p>
      </div>

      <div className="chat-container glass-panel">
        <div className="message-list">
          {messages.map((message) => (
            <div key={message.id} className={`message ${message.role}`}>
              <div className="message-content">
                {message.role === 'bot' ? <ReactMarkdown>{message.content}</ReactMarkdown> : message.content}
              </div>

              {message.cached && <span className="cached-badge">cached</span>}

              {message.sources?.length > 0 && <SourceDropdown sources={message.sources} />}
            </div>
          ))}

          {isLoading && messages[messages.length - 1]?.content === '' && (
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
              onChange={(event) => setInput(event.target.value)}
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
      <button className="source-toggle" onClick={() => setIsOpen(!isOpen)} type="button">
        {isOpen ? '▼' : '▶'} View {sources.length} Citations
      </button>

      {isOpen && (
        <div className="source-cards">
          {sources.map((source, index) => (
            <div key={`${source.source}-${source.page ?? index}`} className="source-card">
              <span className="source-title">
                {source.source}
                {source.page ? ` · p.${source.page}` : ''}
                {typeof source.score === 'number' ? ` · ${source.score.toFixed(2)}` : ''}
              </span>
              <span className="source-text">{source.snippet}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default App;
