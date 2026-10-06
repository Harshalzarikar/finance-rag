import { useEffect, useRef, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import './index.css';

const API_URL = import.meta.env.VITE_API_URL || '/api';

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

function authHeaders(accessToken) {
  return {
    Authorization: `Bearer ${accessToken}`,
  };
}

function describeError(error) {
  if (error instanceof ApiError) {
    if (error.status === 401) return 'Invalid email or password, or your session expired. Sign in again.';
    if (error.status === 429) return 'Rate limit exceeded. Please wait a moment and try again.';
    if (error.status === 503) return 'The service is not ready yet. Ingestion may still be running.';
    return `The backend returned an error (HTTP ${error.status}).`;
  }
  return 'Could not reach the backend. Make sure the API server is running.';
}

async function streamAnswer(query, chat_history, accessToken, onEvent, signal) {
  const response = await fetch(`${API_URL}/chat/stream`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...authHeaders(accessToken),
    },
    body: JSON.stringify({ query, chat_history }),
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
        // Ignore malformed frames
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
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"
      style={{ transition: 'transform 200ms ease', transform: open ? 'rotate(90deg)' : 'rotate(0deg)' }}>
      <polyline points="9 18 15 12 9 6" />
    </svg>
  );
}

function UploadIcon() {
  return (
    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
      <polyline points="17 8 12 3 7 8" />
      <line x1="12" y1="3" x2="12" y2="15" />
    </svg>
  );
}

function LoginScreen({ onLogin, bootstrapping }) {
  const [mode, setMode] = useState('signin');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [fullName, setFullName] = useState('');
  const [tenantId, setTenantId] = useState('');
  const [organizationName, setOrganizationName] = useState('');
  const [isNewOrg, setIsNewOrg] = useState(false);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const completeAuth = (data) => {
    localStorage.setItem('access_token', data.access_token);
    onLogin(data.access_token, data);
  };

  const handleSignIn = async (e) => {
    e.preventDefault();
    setError('');
    setLoading(true);
    try {
      const res = await fetch(`${API_URL}/auth/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email: email.trim(), password }),
      });
      if (!res.ok) {
        let detail = 'Invalid email or password.';
        try {
          const body = await res.json();
          if (body?.detail) detail = typeof body.detail === 'string' ? body.detail : detail;
        } catch {
          // ignore
        }
        throw new Error(detail);
      }
      completeAuth(await res.json());
    } catch (err) {
      setError(err.message || 'Sign in failed.');
      localStorage.removeItem('access_token');
    } finally {
      setLoading(false);
    }
  };

  const handleSignUp = async (e) => {
    e.preventDefault();
    setError('');
    setLoading(true);
    try {
      const payload = {
        email: email.trim(),
        password,
        full_name: fullName.trim(),
        tenant_id: tenantId.trim().toLowerCase(),
        organization_name: isNewOrg ? organizationName.trim() : null,
      };
      const res = await fetch(`${API_URL}/auth/signup`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      if (!res.ok) {
        let detail = 'Could not create your account.';
        try {
          const body = await res.json();
          if (body?.detail) detail = typeof body.detail === 'string' ? body.detail : detail;
        } catch {
          // ignore
        }
        throw new Error(detail);
      }
      completeAuth(await res.json());
    } catch (err) {
      setError(err.message || 'Sign up failed.');
      localStorage.removeItem('access_token');
    } finally {
      setLoading(false);
    }
  };

  if (bootstrapping) {
    return (
      <div className="auth-shell auth-shell--loading">
        <div className="auth-loading">
          <div className="auth-loading-spinner" aria-hidden="true" />
          <p>Restoring your session…</p>
        </div>
      </div>
    );
  }

  return (
    <div className="auth-shell">
      <aside className="auth-brand-panel" aria-hidden="true">
        <div className="auth-brand-inner">
          <span className="eyebrow">Enterprise RAG</span>
          <h1 className="auth-headline">
            Policy Intelligence
            <span className="auth-headline-accent">Research Desk</span>
          </h1>
          <p className="auth-lead">
            Ask questions over your agency&apos;s documents. Every answer is grounded in retrieved
            passages with citations you can audit.
          </p>
          <ul className="auth-trust-list">
            <li>Tenant-isolated document store</li>
            <li>Hybrid search + reranking</li>
            <li>Faithfulness checks on answers</li>
          </ul>
        </div>
      </aside>

      <main className="auth-form-panel">
        <div className="auth-form-card">
          <div className="auth-tabs" role="tablist" aria-label="Authentication">
            <button
              type="button"
              role="tab"
              className={`auth-tab ${mode === 'signin' ? 'auth-tab--active' : ''}`}
              aria-selected={mode === 'signin'}
              onClick={() => { setMode('signin'); setError(''); }}
            >
              Sign in
            </button>
            <button
              type="button"
              role="tab"
              className={`auth-tab ${mode === 'signup' ? 'auth-tab--active' : ''}`}
              aria-selected={mode === 'signup'}
              onClick={() => { setMode('signup'); setError(''); }}
            >
              Create account
            </button>
          </div>

          {mode === 'signin' ? (
            <>
              <div className="auth-form-header">
                <h2>Welcome back</h2>
                <p>Sign in to your agency workspace.</p>
              </div>
              <form onSubmit={handleSignIn} className="auth-form">
                <label className="form-field">
                  <span className="form-label">Work email</span>
                  <input
                    type="email"
                    name="email"
                    placeholder="you@agency.com"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    disabled={loading}
                    className="chat-input auth-input"
                    autoComplete="username"
                    required
                  />
                </label>
                <label className="form-field">
                  <span className="form-label">Password</span>
                  <input
                    type="password"
                    name="password"
                    placeholder="Enter your password"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    disabled={loading}
                    className="chat-input auth-input"
                    autoComplete="current-password"
                    required
                  />
                </label>
                {error && <div className="auth-error" role="alert">{error}</div>}
                <button type="submit" className="btn-primary auth-submit" disabled={loading || !email.trim() || !password}>
                  {loading ? 'Signing in…' : 'Sign in to workspace'}
                </button>
              </form>
            </>
          ) : (
            <>
              <div className="auth-form-header">
                <h2>Create your account</h2>
                <p>Join your agency or register a new organization workspace.</p>
              </div>
              <form onSubmit={handleSignUp} className="auth-form">
                <label className="form-field">
                  <span className="form-label">Full name</span>
                  <input
                    type="text"
                    name="fullName"
                    placeholder="Jane Analyst"
                    value={fullName}
                    onChange={(e) => setFullName(e.target.value)}
                    disabled={loading}
                    className="chat-input auth-input"
                    autoComplete="name"
                  />
                </label>
                <label className="form-field">
                  <span className="form-label">Work email</span>
                  <input
                    type="email"
                    name="email"
                    placeholder="you@agency.com"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    disabled={loading}
                    className="chat-input auth-input"
                    autoComplete="email"
                    required
                  />
                </label>
                <label className="form-field">
                  <span className="form-label">Password</span>
                  <input
                    type="password"
                    name="password"
                    placeholder="At least 8 characters"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    disabled={loading}
                    className="chat-input auth-input"
                    autoComplete="new-password"
                    minLength={8}
                    required
                  />
                </label>
                <label className="form-field">
                  <span className="form-label">Organization ID</span>
                  <input
                    type="text"
                    name="tenantId"
                    placeholder="acme-insurance"
                    value={tenantId}
                    onChange={(e) => setTenantId(e.target.value.toLowerCase())}
                    disabled={loading}
                    className="chat-input auth-input"
                    pattern="^[a-z0-9][a-z0-9-]*[a-z0-9]$"
                    title="Lowercase letters, numbers, and hyphens only"
                    required
                  />
                  <span className="form-hint">Use your company slug, or pick one for a new agency.</span>
                </label>
                <label className="auth-checkbox">
                  <input
                    type="checkbox"
                    checked={isNewOrg}
                    onChange={(e) => setIsNewOrg(e.target.checked)}
                    disabled={loading}
                  />
                  <span>I am registering a new organization</span>
                </label>
                {isNewOrg && (
                  <label className="form-field">
                    <span className="form-label">Organization name</span>
                    <input
                      type="text"
                      name="organizationName"
                      placeholder="Acme Insurance Co."
                      value={organizationName}
                      onChange={(e) => setOrganizationName(e.target.value)}
                      disabled={loading}
                      className="chat-input auth-input"
                      required={isNewOrg}
                    />
                  </label>
                )}
                {error && <div className="auth-error" role="alert">{error}</div>}
                <button
                  type="submit"
                  className="btn-primary auth-submit"
                  disabled={
                    loading
                    || !email.trim()
                    || password.length < 8
                    || !tenantId.trim()
                    || (isNewOrg && !organizationName.trim())
                  }
                >
                  {loading ? 'Creating account…' : 'Create account & sign in'}
                </button>
              </form>
            </>
          )}
        </div>
      </main>
    </div>
  );
}

/* ── Upload Dashboard ── */
function UploadDashboard({ accessToken }) {
  const [file, setFile] = useState(null);
  const [status, setStatus] = useState('');
  const [jobId, setJobId] = useState(null);

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
    }
  };

  const handleUpload = async () => {
    if (!file) return;
    setStatus('Uploading...');
    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch(`${API_URL}/documents/upload`, {
        method: 'POST',
        headers: authHeaders(accessToken),
        body: formData,
      });
      if (!res.ok) throw new Error('Upload failed');
      const data = await res.json();
      setJobId(data.job_id);
      setStatus(`Job queued: ${data.job_id}. Processing...`);
      pollStatus(data.job_id);
    } catch (err) {
      setStatus('Error: ' + err.message);
    }
  };

  const pollStatus = async (id) => {
    const interval = setInterval(async () => {
      try {
        const res = await fetch(`${API_URL}/documents/jobs/${id}`, {
          headers: authHeaders(accessToken),
        });
        const data = await res.json();
        if (data.status === 'completed') {
          setStatus(`Success! Embedded ${data.pages} pages into ${data.chunks} chunks.`);
          clearInterval(interval);
        } else if (data.status === 'failed') {
          setStatus('Failed to process document.');
          clearInterval(interval);
        } else if (data.status === 'not_found') {
           // Celery might not have picked it up immediately
        } else {
          setStatus(`Processing... (${data.status})`);
        }
      } catch (err) {
        clearInterval(interval);
        setStatus('Error checking status.');
      }
    }, 2000);
  };

  return (
    <div className="upload-container" style={{ padding: '2rem', maxWidth: '800px', margin: '0 auto' }}>
      <div className="workspace-intro">
        <h2>Document Ingestion Hub</h2>
        <p className="intro-note">Upload policy documents to your isolated tenant database.</p>
      </div>
      <div className="upload-box" style={{ border: '2px dashed var(--rule-gold)', padding: '3rem', borderRadius: 'var(--radius)', textAlign: 'center', backgroundColor: 'var(--surface)' }}>
        <UploadIcon />
        <h3 style={{ margin: '1rem 0' }}>Upload a PDF</h3>
        <input type="file" accept="application/pdf" onChange={handleFileChange} style={{ marginBottom: '1rem' }} />
        <br />
        <button onClick={handleUpload} className="send-button" disabled={!file || status.includes('Processing')}>
          Upload & Process
        </button>
        {status && <p style={{ marginTop: '1rem', fontWeight: 500, color: 'var(--ink)' }}>{status}</p>}
      </div>
    </div>
  );
}


/* ── Chat Panel ── */
function ChatPanel({ accessToken, session, onNavigateUpload }) {
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

    const historyPayload = messages.map(m => ({
      role: m.role === 'bot' ? 'assistant' : 'user',
      content: m.content
    }));

    try {
      await streamAnswer(query, historyPayload, accessToken, (event) => {
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
    <div className="chat-container">
      <div className="message-list">
        {isEmpty && (
          <div className="empty-state">
            <h2>Welcome, {session?.tenant?.name || session?.user?.name || 'Agency'}</h2>
            <p>
              Ask any question about your uploaded policy documents.
              Answers are securely grounded in your private multi-tenant database.
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
            placeholder="Ask about minimum probability of lifetime ruin..."
            value={input}
            onChange={(event) => setInput(event.target.value)}
            disabled={isLoading}
            autoComplete="off"
          />
          <button type="button" onClick={onNavigateUpload} className="suggestion" style={{ padding: '0.5rem 1rem', height: '100%' }} title="Upload Document">
            <UploadIcon />
          </button>
          <button type="submit" id="send-button" className="send-button" disabled={!input.trim() || isLoading}>
            <SendIcon />
            Send
          </button>
        </form>
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

export default function App() {
  const [accessToken, setAccessToken] = useState('');
  const [session, setSession] = useState(null);
  const [view, setView] = useState('chat');
  const [bootstrapping, setBootstrapping] = useState(true);

  useEffect(() => {
    const stored = localStorage.getItem('access_token');
    if (!stored) {
      setBootstrapping(false);
      return;
    }

    fetch(`${API_URL}/tenants/me`, { headers: authHeaders(stored) })
      .then(async (res) => {
        if (!res.ok) {
          localStorage.removeItem('access_token');
          return null;
        }
        return res.json();
      })
      .then((data) => {
        if (data) {
          setAccessToken(stored);
          setSession(data);
        }
      })
      .catch(() => localStorage.removeItem('access_token'))
      .finally(() => setBootstrapping(false));
  }, []);

  const handleLogin = (token, loginPayload) => {
    setAccessToken(token);
    setSession({ tenant: loginPayload.tenant, user: loginPayload.user });
  };

  const handleLogout = () => {
    setAccessToken('');
    setSession(null);
    localStorage.removeItem('access_token');
  };

  if (!accessToken) {
    return <LoginScreen onLogin={handleLogin} bootstrapping={bootstrapping} />;
  }

  return (
    <div className="app-container">
      <header className="header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div className="brand-lockup">
          <span className="eyebrow">Agency Workspace</span>
          <h1>Policy Intelligence <span>SaaS</span></h1>
        </div>
        <div style={{ display: 'flex', gap: '1rem', alignItems: 'center' }}>
          <button 
            className="suggestion" 
            style={{ background: view === 'chat' ? 'var(--gold-100)' : 'transparent', border: '1px solid var(--rule-gold)' }} 
            onClick={() => setView('chat')}
          >
            Chat
          </button>
          <button 
            className="suggestion" 
            style={{ background: view === 'upload' ? 'var(--gold-100)' : 'transparent', border: '1px solid var(--rule-gold)' }} 
            onClick={() => setView('upload')}
          >
            Upload
          </button>
          <div className="header-meta" style={{ cursor: 'pointer' }} onClick={handleLogout} title="Log Out">
            <span className="status-dot" aria-hidden="true" style={{ backgroundColor: 'var(--success)' }} />
            <span>{session?.user?.email || session?.tenant?.name} ({session?.tenant?.plan || 'pro'})</span>
          </div>
        </div>
      </header>

      {view === 'chat' ? (
        <ChatPanel accessToken={accessToken} session={session} onNavigateUpload={() => setView('upload')} />
      ) : (
        <UploadDashboard accessToken={accessToken} />
      )}
    </div>
  );
}
