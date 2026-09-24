import { useEffect, useRef, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import './index.css';

// A relative path works in both environments: Vite proxies /api to the backend
// during development, and Caddy proxies it to the api container in production.
const API_URL = import.meta.env.VITE_API_URL || '/api';
const API_KEY = import.meta.env.VITE_API_KEY || '';

const SUGGESTIONS = [
  'What is the minimum probability of lifetime ruin?',
  'Explain the Black-Scholes model and its limitations.',
  'How does stochastic volatility affect options pricing?',
];

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

/* ── SVG Icons ── */

function SendIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <line x1="22" y1="2" x2="11" y2="13" />
      <polygon points="22 2 15 22 11 13 2 9 22 2" />
    </svg>
  );
}

function DocIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
      <polyline points="14 2 14 8 20 8" />
      <line x1="16" y1="13" x2="8" y2="13" />
      <line x1="16" y1="17" x2="8" y2="17" />
      <polyline points="10 9 9 9 8 9" />
    </svg>
  );
}

function SparkleIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
      <path d="M12 2L14.09 8.26L20 9.27L15.55 13.97L16.91 20L12 16.9L7.09 20L8.45 13.97L4 9.27L9.91 8.26L12 2Z" />
    </svg>
  );
}

function ShieldCheckIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
      <path d="M9 12l2 2 4-4"/>
    </svg>
  );
}

function ChevronIcon({ open }) {
  return (
    <svg
      width="14"
      height="14"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2.5"
      strokeLinecap="round"
      strokeLinejoin="round"
      style={{
        transition: 'transform 200ms cubic-bezier(0.22, 1, 0.36, 1)',
        transform: open ? 'rotate(90deg)' : 'rotate(0deg)',
      }}
    >
      <polyline points="9 18 15 12 9 6" />
    </svg>
  );
}


function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const askQuestion = async (query) => {
    if (!query.trim() || isLoading) return;

    const userMessage = { id: Date.now(), role: 'user', content: query, sources: [] };
    const botId = Date.now() + 1;

    setMessages((prev) => [
      ...prev,
      userMessage,
      { id: botId, role: 'bot', content: '', sources: [], cached: false, confidenceScore: null, faithfulnessPassed: null },
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
        } else if (event.type === 'done') {
          patch((message) => ({
            ...message,
            confidenceScore: event.confidence_score !== undefined ? event.confidence_score : message.confidenceScore,
            faithfulnessPassed: event.faithfulness_passed !== undefined ? event.faithfulness_passed : message.faithfulnessPassed
          }));
        }
      });
    } catch (error) {
      console.error('Streaming request failed:', error);
      patch((message) => ({ ...message, content: describeError(error) }));
    } finally {
      setIsLoading(false);
    }
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    askQuestion(input);
  };

  const isEmpty = messages.length === 0;

  return (
    <div className="app-container">
      {/* ── Header ── */}
      <header className="header">
        <div className="brand-lockup">
          <span className="eyebrow">Research desk</span>
          <h1>QuantRAG <span>AI</span></h1>
        </div>
        <div className="header-meta">
          <span className="status-dot" aria-hidden="true" />
          <span>Indexed finance corpus</span>
        </div>
      </header>

      {/* ── Intro section ── */}
      <div className="workspace-intro">
        <div>
          <p className="section-kicker">Evidence-led answers</p>
          <h2>Ask the literature a sharper question.</h2>
        </div>
        <p className="intro-note">
          Explore quantitative finance research with page-level provenance and citations.
        </p>
      </div>

      {/* ── Chat Panel ── */}
      <div className="chat-container">
        <div className="message-list">
          {isEmpty && (
            <div className="empty-state">
              <h2>How can I help you today?</h2>
              <p>
                Ask any question about the indexed arXiv quantitative finance research corpus.
                Answers are grounded in retrieved passages with full citations.
              </p>
              <div className="empty-label">
                <SparkleIcon /> Suggested questions
              </div>
              <div className="suggestions">
                {SUGGESTIONS.map((s) => (
                  <button
                    key={s}
                    className="suggestion"
                    onClick={() => askQuestion(s)}
                    type="button"
                  >
                    {s}
                  </button>
                ))}
              </div>
            </div>
          )}

          {messages.map((message) => (
            <div key={message.id} className={`message ${message.role}`}>
              <div className="message-content">
                {message.role === 'bot' ? <ReactMarkdown>{message.content}</ReactMarkdown> : message.content}
              </div>

              {message.role === 'bot' && message.content && (
                <div className="telemetry-badges">
                  {message.cached && <span className="telemetry-badge cached">⚡ Cached</span>}
                  
                  {message.confidenceScore !== null && (
                    <span className={`telemetry-badge confidence ${message.confidenceScore >= 0.8 ? 'high' : message.confidenceScore >= 0.5 ? 'medium' : 'low'}`}>
                      Confidence: {message.confidenceScore >= 0.8 ? 'High' : message.confidenceScore >= 0.5 ? 'Medium' : 'Low'}
                    </span>
                  )}
                  
                  {message.faithfulnessPassed === true && (
                    <span className="telemetry-badge verified">
                      <ShieldCheckIcon /> Verified
                    </span>
                  )}

                  {message.faithfulnessPassed === false && (
                    <span className="telemetry-badge unverified">
                      ⚠️ Guard Blocked
                    </span>
                  )}
                </div>
              )}

              {message.sources?.length > 0 && <SourceDropdown sources={message.sources} />}
            </div>
          ))}

          {isLoading && messages[messages.length - 1]?.content === '' && (
            <div className="message bot">
              <div className="typing-indicator">
                <div className="dot" />
                <div className="dot" />
                <div className="dot" />
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* ── Composer ── */}
        <div className="input-area">
          <form className="input-form" onSubmit={handleSubmit}>
            <input
              type="text"
              id="chat-input"
              className="chat-input"
              placeholder="Ask about stochastic volatility, Heston models, options pricing..."
              value={input}
              onChange={(event) => setInput(event.target.value)}
              disabled={isLoading}
              autoComplete="off"
            />
            <button type="submit" id="send-button" className="send-button" disabled={!input.trim() || isLoading}>
              <SendIcon />
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
        aria-expanded={isOpen}
      >
        <ChevronIcon open={isOpen} />
        <DocIcon />
        {isOpen ? 'Hide' : 'View'} {sources.length} {sources.length === 1 ? 'source' : 'sources'}
      </button>

      {isOpen && (
        <div className="source-cards">
          {sources.map((source, index) => (
            <div key={`${source.source}-${source.page ?? index}`} className="source-card">
              <span className="source-index">{String(index + 1).padStart(2, '0')}</span>
              <span className="source-body">
                <span className="source-title">
                  {source.source}
                  <span className="source-meta">{source.page ? `Page ${source.page}` : 'Page unavailable'}</span>
                  {typeof source.score === 'number' && (
                    <span className="source-score">Match {source.score.toFixed(2)}</span>
                  )}
                </span>
                <span className="source-text">{source.snippet}</span>
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default App;
