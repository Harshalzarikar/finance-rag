import { useEffect, useRef, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import { describeError, streamAnswer } from '../lib/api.js';
import { SendIcon, ShieldCheckIcon, SparkleIcon, UploadIcon } from './Icons.jsx';
import { SourceDropdown } from './SourceDropdown.jsx';

const SUGGESTIONS = [
  'What is implied volatility?',
  'Explain the Black-Scholes model and its limitations.',
  'How does stochastic volatility affect options pricing?',
];

export function ChatPanel({
  accessToken,
  session,
  messages,
  onMessagesChange,
  onNavigateUpload,
}) {
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  const persist = (next) => {
    onMessagesChange(next);
  };

  const askQuestion = async (query) => {
    if (!query.trim() || isLoading) return;

    const userMessage = { id: Date.now(), role: 'user', content: query, sources: [] };
    const botId = Date.now() + 1;
    const botPlaceholder = {
      id: botId,
      role: 'bot',
      content: '',
      sources: [],
      cached: false,
      confidenceScore: null,
      faithfulnessPassed: null,
    };
    let working = [...messages, userMessage, botPlaceholder];
    persist(working);
    setInput('');
    setIsLoading(true);

    const applyPatch = (updater) => {
      working = working.map((message) => (message.id === botId ? updater(message) : message));
      persist(working);
    };

    const historyPayload = messages.map((m) => ({
      role: m.role === 'bot' ? 'assistant' : 'user',
      content: m.content,
    }));

    try {
      await streamAnswer(query, historyPayload, accessToken, (event) => {
        if (event.type === 'sources') {
          applyPatch((message) => ({
            ...message,
            sources: event.sources ?? [],
            cached: Boolean(event.cached),
          }));
        } else if (event.type === 'token') {
          applyPatch((message) => ({ ...message, content: message.content + event.value }));
        } else if (event.type === 'error') {
          applyPatch((message) => ({
            ...message,
            content: event.detail ?? 'The answer could not be generated.',
          }));
        } else if (event.type === 'done') {
          applyPatch((message) => ({
            ...message,
            confidenceScore:
              event.confidence_score !== undefined ? event.confidence_score : message.confidenceScore,
            faithfulnessPassed:
              event.faithfulness_passed !== undefined ? event.faithfulness_passed : message.faithfulnessPassed,
          }));
        }
      });
    } catch (error) {
      console.error('Streaming request failed:', error);
      applyPatch((message) => ({ ...message, content: describeError(error) }));
    } finally {
      setIsLoading(false);
    }
  };

  const handleSubmit = (event) => {
    event.preventDefault();
    askQuestion(input);
  };

  const isEmpty = messages.length === 0;

  return (
    <div className="chat-container">
      <div className="message-list">
        {isEmpty && (
          <div className="empty-state">
            <h2>Welcome, {session?.tenant?.name || session?.user?.full_name || 'Analyst'}</h2>
            <p>
              Ask questions over your indexed research corpus. Answers include citations and a faithfulness check.
            </p>
            <div className="empty-label">
              <SparkleIcon /> Suggested questions
            </div>
            <div className="suggestions">
              {SUGGESTIONS.map((s) => (
                <button key={s} className="suggestion" onClick={() => askQuestion(s)} type="button">
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
                {message.cached && <span className="telemetry-badge cached">Cached</span>}
                {message.confidenceScore !== null && (
                  <span
                    className={`telemetry-badge confidence ${
                      message.confidenceScore >= 0.8 ? 'high' : message.confidenceScore >= 0.5 ? 'medium' : 'low'
                    }`}
                  >
                    Confidence:{' '}
                    {message.confidenceScore >= 0.8 ? 'High' : message.confidenceScore >= 0.5 ? 'Medium' : 'Low'}
                  </span>
                )}
                {message.faithfulnessPassed === true && (
                  <span className="telemetry-badge verified">
                    <ShieldCheckIcon /> Verified
                  </span>
                )}
                {message.faithfulnessPassed === false && (
                  <span className="telemetry-badge unverified">Guard blocked</span>
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

      <div className="input-area">
        <form className="input-form" onSubmit={handleSubmit}>
          <input
            type="text"
            className="chat-input"
            placeholder="Ask about volatility, hedging, risk models…"
            value={input}
            onChange={(event) => setInput(event.target.value)}
            disabled={isLoading}
            autoComplete="off"
          />
          <button
            type="button"
            onClick={onNavigateUpload}
            className="icon-button"
            title="Upload document"
          >
            <UploadIcon />
          </button>
          <button type="submit" className="send-button" disabled={!input.trim() || isLoading}>
            <SendIcon />
            Send
          </button>
        </form>
      </div>
    </div>
  );
}
