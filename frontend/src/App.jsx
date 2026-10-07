import { useCallback, useEffect, useMemo, useState } from 'react';
import './index.css';
import { ChatPanel } from './components/ChatPanel.jsx';
import { LoginScreen } from './components/LoginScreen.jsx';
import { Sidebar } from './components/Sidebar.jsx';
import { UploadDashboard } from './components/UploadDashboard.jsx';
import { API_URL, authHeaders } from './lib/api.js';
import {
  createConversation,
  loadConversations,
  saveConversations,
  titleFromMessages,
} from './lib/conversations.js';

export default function App() {
  const [accessToken, setAccessToken] = useState('');
  const [session, setSession] = useState(null);
  const [view, setView] = useState('chat');
  const [bootstrapping, setBootstrapping] = useState(true);
  const [conversations, setConversations] = useState([]);
  const [activeConversationId, setActiveConversationId] = useState(null);

  const tenantId = session?.tenant?.id || 'default';

  const activeConversation = useMemo(
    () => conversations.find((c) => c.id === activeConversationId) ?? null,
    [conversations, activeConversationId],
  );

  const initConversations = useCallback((tid) => {
    const loaded = loadConversations(tid);
    if (loaded.length === 0) {
      const fresh = createConversation();
      saveConversations(tid, [fresh]);
      setConversations([fresh]);
      setActiveConversationId(fresh.id);
      return;
    }
    setConversations(loaded);
    setActiveConversationId(loaded[0].id);
  }, []);

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
          initConversations(data.tenant?.id || 'default');
        }
      })
      .catch(() => localStorage.removeItem('access_token'))
      .finally(() => setBootstrapping(false));
  }, [initConversations]);

  const handleLogin = (token, loginPayload) => {
    setAccessToken(token);
    const sess = { tenant: loginPayload.tenant, user: loginPayload.user };
    setSession(sess);
    initConversations(loginPayload.tenant?.id || 'default');
  };

  const handleLogout = () => {
    setAccessToken('');
    setSession(null);
    setConversations([]);
    setActiveConversationId(null);
    localStorage.removeItem('access_token');
  };

  const updateMessages = (messages) => {
    setConversations((prev) => {
      const next = prev.map((c) =>
        c.id === activeConversationId
          ? {
              ...c,
              messages,
              updatedAt: Date.now(),
              title: titleFromMessages(messages),
            }
          : c,
      );
      saveConversations(tenantId, next);
      return next;
    });
  };

  const handleNewConversation = () => {
    const fresh = createConversation();
    setConversations((prev) => {
      const next = [fresh, ...prev];
      saveConversations(tenantId, next);
      return next;
    });
    setActiveConversationId(fresh.id);
    setView('chat');
  };

  const handleDeleteConversation = (id) => {
    setConversations((prev) => {
      let next = prev.filter((c) => c.id !== id);
      if (next.length === 0) {
        const fresh = createConversation();
        next = [fresh];
      }
      saveConversations(tenantId, next);
      if (activeConversationId === id) {
        setActiveConversationId(next[0].id);
      }
      return next;
    });
  };

  if (!accessToken) {
    return <LoginScreen onLogin={handleLogin} bootstrapping={bootstrapping} />;
  }

  return (
    <div className="app-root">
      <header className="header app-header">
        <div className="brand-lockup">
          <span className="eyebrow">Agency workspace</span>
          <h1>Quant Research <span>RAG</span></h1>
        </div>
        <button type="button" className="header-logout" onClick={handleLogout} title="Sign out">
          <span className="status-dot" aria-hidden="true" />
          {session?.user?.email || session?.tenant?.name}
          <span className="header-plan">{session?.tenant?.plan || 'pro'}</span>
        </button>
      </header>

      <div className="app-layout">
        <Sidebar
          view={view}
          onChangeView={setView}
          conversations={conversations}
          activeConversationId={activeConversationId}
          onSelectConversation={setActiveConversationId}
          onNewConversation={handleNewConversation}
          onDeleteConversation={handleDeleteConversation}
          tenantName={session?.tenant?.name}
        />

        <main className="main-panel">
          {view === 'chat' ? (
            <ChatPanel
              key={activeConversationId}
              accessToken={accessToken}
              session={session}
              messages={activeConversation?.messages ?? []}
              onMessagesChange={updateMessages}
              onNavigateUpload={() => setView('upload')}
            />
          ) : (
            <UploadDashboard accessToken={accessToken} tenantId={tenantId} />
          )}
        </main>
      </div>
    </div>
  );
}
