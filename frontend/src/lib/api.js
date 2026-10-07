export const API_URL = import.meta.env.VITE_API_URL || '/api';

export class ApiError extends Error {
  constructor(message, status) {
    super(message);
    this.status = status;
  }
}

export function authHeaders(accessToken) {
  return {
    Authorization: `Bearer ${accessToken}`,
  };
}

export function describeError(error) {
  if (error instanceof ApiError) {
    if (error.status === 401) return 'Invalid email or password, or your session expired. Sign in again.';
    if (error.status === 429) return 'Rate limit exceeded. Please wait a moment and try again.';
    if (error.status === 503) return 'The service is not ready yet. Ingestion may still be running.';
    return `The backend returned an error (HTTP ${error.status}).`;
  }
  return 'Could not reach the backend. Make sure the API server is running.';
}

export async function streamAnswer(query, chat_history, accessToken, onEvent, signal) {
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
