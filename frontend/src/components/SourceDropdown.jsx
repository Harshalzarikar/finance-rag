import { useState } from 'react';
import { ChevronIcon, DocIcon } from './Icons.jsx';

export function SourceDropdown({ sources }) {
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
