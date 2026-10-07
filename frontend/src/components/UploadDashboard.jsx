import { useCallback, useEffect, useState } from 'react';
import { API_URL, authHeaders } from '../lib/api.js';
import { addUploadRecord, loadUploadHistory, patchUploadRecord } from '../lib/uploadHistory.js';
import { UploadIcon } from './Icons.jsx';

function statusLabel(status) {
  if (status === 'completed') return 'Completed';
  if (status === 'failed') return 'Failed';
  if (status === 'processing' || status === 'started') return 'Processing';
  return 'Queued';
}

export function UploadDashboard({ accessToken, tenantId }) {
  const [file, setFile] = useState(null);
  const [status, setStatus] = useState('');
  const [history, setHistory] = useState([]);

  const refreshHistory = useCallback(() => {
    setHistory(loadUploadHistory(tenantId));
  }, [tenantId]);

  useEffect(() => {
    refreshHistory();
  }, [refreshHistory]);

  const pollStatus = (jobId) => {
    const interval = setInterval(async () => {
      try {
        const res = await fetch(`${API_URL}/documents/jobs/${jobId}`, {
          headers: authHeaders(accessToken),
        });
        const data = await res.json();
        if (data.status === 'PENDING' || data.status === 'RETRY') {
          setStatus('Queued — waiting for the ingestion worker (can take a minute to start).');
        } else if (data.status === 'SUCCESS' && data.result) {
          if (data.result.status === 'empty' || (data.result.chunks === 0 && data.result.pages === 0)) {
            setStatus('Ingestion finished but no content was indexed. Re-upload the PDF (shared storage fix is required).');
            patchUploadRecord(tenantId, jobId, { status: 'failed', error: 'empty document' });
            refreshHistory();
            clearInterval(interval);
            return;
          }
          const pages = data.result.pages;
          const chunks = data.result.chunks;
          setStatus(`Indexed: ${pages ?? '?'} pages → ${chunks ?? '?'} chunks. You can chat now.`);
          patchUploadRecord(tenantId, jobId, {
            status: 'completed',
            pages,
            chunks,
          });
          refreshHistory();
          clearInterval(interval);
        } else if (data.status === 'FAILURE') {
          setStatus(data.error || 'Ingestion failed. Check worker logs.');
          patchUploadRecord(tenantId, jobId, { status: 'failed', error: data.error });
          refreshHistory();
          clearInterval(interval);
        } else {
          setStatus(`Processing… (${data.status})`);
          patchUploadRecord(tenantId, jobId, { status: data.status || 'processing' });
        }
      } catch {
        clearInterval(interval);
        setStatus('Error checking job status.');
      }
    }, 2000);
  };

  const handleUpload = async () => {
    if (!file) return;
    setStatus('Uploading…');
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
      addUploadRecord(tenantId, {
        fileName: file.name,
        jobId: data.job_id,
        status: 'queued',
      });
      refreshHistory();
      setStatus(`Job queued: ${data.job_id}`);
      pollStatus(data.job_id);
    } catch (err) {
      setStatus(`Error: ${err.message}`);
    }
  };

  return (
    <div className="upload-page">
      <div className="workspace-intro">
        <div>
          <p className="section-kicker">Knowledge base</p>
          <h2>Document ingestion</h2>
        </div>
        <p className="intro-note">
          Upload PDFs to your tenant index. Processing runs in the background when the worker is enabled.
        </p>
      </div>

      <div className="upload-box">
        <UploadIcon />
        <h3>Upload a PDF</h3>
        <p className="upload-hint">Files are scoped to your organization workspace.</p>
        <input type="file" accept="application/pdf" onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
        <button type="button" onClick={handleUpload} className="send-button" disabled={!file || status.includes('Processing')}>
          Upload &amp; process
        </button>
        {status && <p className="upload-status">{status}</p>}
      </div>

      <section className="history-section" aria-labelledby="upload-history-title">
        <h3 id="upload-history-title">Upload history</h3>
        <p className="history-note">Recent uploads on this browser (last 50). Server corpus is shared for your tenant.</p>
        {history.length === 0 ? (
          <p className="sidebar-empty">No uploads recorded yet.</p>
        ) : (
          <div className="history-table-wrap">
            <table className="history-table">
              <thead>
                <tr>
                  <th>File</th>
                  <th>Status</th>
                  <th>Pages</th>
                  <th>Chunks</th>
                  <th>When</th>
                </tr>
              </thead>
              <tbody>
                {history.map((row) => (
                  <tr key={row.id}>
                    <td className="history-file">{row.fileName}</td>
                    <td>
                      <span className={`history-pill history-pill--${row.status === 'completed' ? 'ok' : row.status === 'failed' ? 'bad' : 'pending'}`}>
                        {statusLabel(row.status)}
                      </span>
                    </td>
                    <td>{row.pages ?? '—'}</td>
                    <td>{row.chunks ?? '—'}</td>
                    <td className="history-when">{new Date(row.startedAt).toLocaleString()}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  );
}
