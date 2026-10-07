const STORAGE_VERSION = 1;

function storageKey(tenantId) {
  return `rag_uploads_v${STORAGE_VERSION}_${tenantId || 'default'}`;
}

export function loadUploadHistory(tenantId) {
  try {
    const raw = localStorage.getItem(storageKey(tenantId));
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

export function saveUploadHistory(tenantId, records) {
  localStorage.setItem(storageKey(tenantId), JSON.stringify(records.slice(0, 50)));
}

export function addUploadRecord(tenantId, record) {
  const list = loadUploadHistory(tenantId);
  const entry = {
    id: record.id || `u_${Date.now()}`,
    fileName: record.fileName,
    jobId: record.jobId ?? null,
    status: record.status || 'queued',
    pages: record.pages ?? null,
    chunks: record.chunks ?? null,
    error: record.error ?? null,
    startedAt: record.startedAt || Date.now(),
    updatedAt: Date.now(),
  };
  const next = [entry, ...list.filter((r) => r.id !== entry.id)];
  saveUploadHistory(tenantId, next);
  return entry;
}

export function patchUploadRecord(tenantId, jobId, patch) {
  const list = loadUploadHistory(tenantId);
  const next = list.map((r) =>
    r.jobId === jobId ? { ...r, ...patch, updatedAt: Date.now() } : r,
  );
  saveUploadHistory(tenantId, next);
  return next;
}
