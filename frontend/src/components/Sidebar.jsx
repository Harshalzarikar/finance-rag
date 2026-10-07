import { formatConversationDate } from '../lib/conversations.js';
import { ChatIcon, PlusIcon, TrashIcon, UploadIcon } from './Icons.jsx';

export function Sidebar({
  view,
  onChangeView,
  conversations,
  activeConversationId,
  onSelectConversation,
  onNewConversation,
  onDeleteConversation,
  tenantName,
}) {
  const sorted = [...conversations].sort((a, b) => (b.updatedAt || 0) - (a.updatedAt || 0));

  return (
    <aside className="sidebar" aria-label="Workspace navigation">
      <div className="sidebar-brand">
        <span className="eyebrow">Research desk</span>
        <p className="sidebar-tenant">{tenantName || 'Workspace'}</p>
      </div>

      <button type="button" className="sidebar-new-chat" onClick={onNewConversation}>
        <PlusIcon />
        New conversation
      </button>

      <nav className="sidebar-nav">
        <button
          type="button"
          className={`sidebar-nav-item ${view === 'chat' ? 'sidebar-nav-item--active' : ''}`}
          onClick={() => onChangeView('chat')}
        >
          <ChatIcon />
          Chat
        </button>
        <button
          type="button"
          className={`sidebar-nav-item ${view === 'upload' ? 'sidebar-nav-item--active' : ''}`}
          onClick={() => onChangeView('upload')}
        >
          <UploadIcon />
          Documents
        </button>
      </nav>

      <div className="sidebar-section">
        <h2 className="sidebar-section-title">Previous chats</h2>
        {sorted.length === 0 ? (
          <p className="sidebar-empty">No conversations yet. Start a new chat.</p>
        ) : (
          <ul className="conversation-list">
            {sorted.map((conv) => (
              <li key={conv.id}>
                <button
                  type="button"
                  className={`conversation-item ${conv.id === activeConversationId ? 'conversation-item--active' : ''}`}
                  onClick={() => {
                    onSelectConversation(conv.id);
                    onChangeView('chat');
                  }}
                >
                  <span className="conversation-item-title">{conv.title || 'New conversation'}</span>
                  <span className="conversation-item-meta">
                    {formatConversationDate(conv.updatedAt)}
                    {conv.messages?.length ? ` · ${Math.floor(conv.messages.length / 2) || 1} Q` : ''}
                  </span>
                </button>
                <button
                  type="button"
                  className="conversation-delete"
                  title="Delete conversation"
                  aria-label={`Delete ${conv.title}`}
                  onClick={(e) => {
                    e.stopPropagation();
                    onDeleteConversation(conv.id);
                  }}
                >
                  <TrashIcon />
                </button>
              </li>
            ))}
          </ul>
        )}
      </div>
    </aside>
  );
}
