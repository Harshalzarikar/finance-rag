import { useState } from 'react';
import { API_URL } from '../lib/api.js';

export function LoginScreen({ onLogin, bootstrapping }) {
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
          <span className="eyebrow">Quantitative finance RAG</span>
          <h1 className="auth-headline">
            Research
            <span className="auth-headline-accent">Intelligence</span>
          </h1>
          <p className="auth-lead">
            Grounded answers over your indexed papers with hybrid search, Cohere reranking, and citation-backed responses.
          </p>
          <ul className="auth-trust-list">
            <li>Tenant-isolated corpus</li>
            <li>Chat history on this device</li>
            <li>Faithfulness verification</li>
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
              onClick={() => {
                setMode('signin');
                setError('');
              }}
            >
              Sign in
            </button>
            <button
              type="button"
              role="tab"
              className={`auth-tab ${mode === 'signup' ? 'auth-tab--active' : ''}`}
              aria-selected={mode === 'signup'}
              onClick={() => {
                setMode('signup');
                setError('');
              }}
            >
              Create account
            </button>
          </div>

          {mode === 'signin' ? (
            <>
              <div className="auth-form-header">
                <h2>Welcome back</h2>
                <p>Sign in to your workspace (use tenant <code>default</code> if you indexed locally).</p>
              </div>
              <form onSubmit={handleSignIn} className="auth-form">
                <label className="form-field">
                  <span className="form-label">Work email</span>
                  <input
                    type="email"
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
                  {loading ? 'Signing in…' : 'Sign in'}
                </button>
              </form>
            </>
          ) : (
            <>
              <div className="auth-form-header">
                <h2>Create account</h2>
                <p>Join an existing org or register a new one.</p>
              </div>
              <form onSubmit={handleSignUp} className="auth-form">
                <label className="form-field">
                  <span className="form-label">Full name</span>
                  <input
                    type="text"
                    value={fullName}
                    onChange={(e) => setFullName(e.target.value)}
                    disabled={loading}
                    className="chat-input auth-input"
                  />
                </label>
                <label className="form-field">
                  <span className="form-label">Work email</span>
                  <input
                    type="email"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    disabled={loading}
                    className="chat-input auth-input"
                    required
                  />
                </label>
                <label className="form-field">
                  <span className="form-label">Password</span>
                  <input
                    type="password"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    disabled={loading}
                    className="chat-input auth-input"
                    minLength={8}
                    required
                  />
                </label>
                <label className="form-field">
                  <span className="form-label">Organization ID</span>
                  <input
                    type="text"
                    value={tenantId}
                    onChange={(e) => setTenantId(e.target.value.toLowerCase())}
                    disabled={loading}
                    className="chat-input auth-input"
                    pattern="^[a-z0-9][a-z0-9-]*[a-z0-9]$"
                    required
                  />
                </label>
                <label className="auth-checkbox">
                  <input type="checkbox" checked={isNewOrg} onChange={(e) => setIsNewOrg(e.target.checked)} disabled={loading} />
                  <span>New organization</span>
                </label>
                {isNewOrg && (
                  <label className="form-field">
                    <span className="form-label">Organization name</span>
                    <input
                      type="text"
                      value={organizationName}
                      onChange={(e) => setOrganizationName(e.target.value)}
                      disabled={loading}
                      className="chat-input auth-input"
                      required={isNewOrg}
                    />
                  </label>
                )}
                {error && <div className="auth-error" role="alert">{error}</div>}
                <button type="submit" className="btn-primary auth-submit" disabled={loading}>
                  {loading ? 'Creating…' : 'Create & sign in'}
                </button>
              </form>
            </>
          )}
        </div>
      </main>
    </div>
  );
}
