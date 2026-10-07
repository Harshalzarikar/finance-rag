const STORAGE_VERSION = 1;

export function conversationsKey(tenantId) {
  return `rag_conversations_v${STORAGE_VERSION}_${tenantId || 'default'}`;
}

export function loadConversations(tenantId) {
  try {
    const raw = localStorage.getItem(conversationsKey(tenantId));
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

export function saveConversations(tenantId, conversations) {
  localStorage.setItem(conversationsKey(tenantId), JSON.stringify(conversations));
}

export function createConversation() {
  const now = Date.now();
  return {
    id: `c_${now}_${Math.random().toString(36).slice(2, 9)}`,
    title: 'New conversation',
    createdAt: now,
    updatedAt: now,
    messages: [],
  };
}

export function titleFromMessages(messages) {
  const first = messages.find((m) => m.role === 'user' && m.content?.trim());
  if (!first) return 'New conversation';
  const text = first.content.trim();
  return text.length > 52 ? `${text.slice(0, 52)}…` : text;
}

export function formatConversationDate(ts) {
  const d = new Date(ts);
  const now = new Date();
  const sameDay =
    d.getFullYear() === now.getFullYear() &&
    d.getMonth() === now.getMonth() &&
    d.getDate() === now.getDate();
  if (sameDay) {
    return d.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' });
  }
  return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
}
