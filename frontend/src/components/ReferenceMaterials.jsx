import React, { useState, useEffect } from 'react';
import { api } from '../api';

export default function ReferenceMaterials() {
  const [documents, setDocuments] = useState([]);
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadStage, setUploadStage] = useState(''); // 'uploading', 'extracting_and_indexing', 'complete'
  const [error, setError] = useState(null);
  const [successMsg, setSuccessMsg] = useState(null);

  // Form states
  const [file, setFile] = useState(null);
  const [title, setTitle] = useState('');
  const [docType, setDocType] = useState('textbook');

  // Deletion modal / confirmation state
  const [docToDelete, setDocToDelete] = useState(null);
  const [deleting, setDeleting] = useState(false);

  const fetchDocuments = async () => {
    try {
      setLoading(true);
      const res = await api.getDocuments();
      setDocuments(res.documents || []);
      setError(null);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchDocuments();
  }, []);

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      const selected = e.target.files[0];
      setFile(selected);
      if (!title) {
        // Auto-suggest title from filename without extension
        const cleanName = selected.name.replace(/\.[^/.]+$/, '').replace(/[_.-]+/g, ' ');
        setTitle(cleanName);
      }
    }
  };

  const handleUpload = async (e) => {
    e.preventDefault();
    if (!file) {
      setError('Please select a file to upload.');
      return;
    }
    if (!title.trim()) {
      setError('Please provide a descriptive document title.');
      return;
    }

    try {
      setUploading(true);
      setError(null);
      setSuccessMsg(null);
      setUploadStage('Uploading document to server...');

      // Trigger upload & indexing pipeline
      setUploadStage('Extracting text, scrubbing PII, chunking & generating embeddings...');
      const result = await api.uploadDocument(file, title.trim(), docType);

      setUploadStage('Indexing complete.');
      setSuccessMsg(`Document "${result.title}" indexed successfully (${result.total_chunks} chunks ready for retrieval).`);

      // Reset form
      setFile(null);
      setTitle('');
      setDocType('textbook');
      if (document.getElementById('doc-file-input')) {
        document.getElementById('doc-file-input').value = '';
      }

      // Refresh list
      await fetchDocuments();
    } catch (err) {
      setError(err.message || 'Failed to upload and index document.');
    } finally {
      setUploading(false);
      setUploadStage('');
    }
  };

  const confirmDelete = async () => {
    if (!docToDelete) return;
    try {
      setDeleting(true);
      await api.deleteDocument(docToDelete.id);
      setSuccessMsg(`Document "${docToDelete.title}" and its indexed vector chunks have been deleted.`);
      setDocToDelete(null);
      await fetchDocuments();
    } catch (err) {
      setError(err.message || 'Failed to delete document.');
    } finally {
      setDeleting(false);
    }
  };

  const formatFileSize = (bytes) => {
    if (!bytes) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return `${(bytes / Math.pow(k, i)).toFixed(1)} ${sizes[i]}`;
  };

  const formatDate = (isoStr) => {
    if (!isoStr) return '—';
    try {
      return new Date(isoStr).toLocaleString();
    } catch {
      return isoStr;
    }
  };

  return (
    <div>
      {/* Upload Form Card */}
      <div className="card">
        <h2 className="card-title">Upload Reference Material</h2>
        <p className="card-description">
          Add authoritative textbooks, lecture notes, or syllabi to the institutional knowledge base.
        </p>

        {error && <div className="alert alert-danger">{error}</div>}
        {successMsg && <div className="alert alert-success">{successMsg}</div>}

        <form onSubmit={handleUpload}>
          <div className="grid-2">
            <div className="form-group">
              <label className="form-label" htmlFor="doc-file-input">
                Select File (PDF, DOCX, TXT) *
              </label>
              <input
                id="doc-file-input"
                type="file"
                className="form-input"
                accept=".pdf,.docx,.txt,.md"
                onChange={handleFileChange}
                disabled={uploading}
              />
              {file && (
                <div style={{ marginTop: '0.25rem', fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                  Selected: {file.name} ({formatFileSize(file.size)})
                </div>
              )}
            </div>

            <div className="form-group">
              <label className="form-label" htmlFor="doc-type-select">
                Document Type *
              </label>
              <select
                id="doc-type-select"
                className="form-select"
                value={docType}
                onChange={(e) => setDocType(e.target.value)}
                disabled={uploading}
              >
                <option value="textbook">Textbook</option>
                <option value="lecture_notes">Lecture Notes</option>
                <option value="reference_book">Reference Book</option>
                <option value="research_paper">Research Paper</option>
                <option value="syllabus">Syllabus</option>
                <option value="other">Other Institutional Material</option>
              </select>
            </div>
          </div>

          <div className="form-group">
            <label className="form-label" htmlFor="doc-title-input">
              Document Title *
            </label>
            <input
              id="doc-title-input"
              type="text"
              className="form-input"
              placeholder="e.g. Speech and Language Processing (Jurafsky & Martin)"
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              disabled={uploading}
            />
          </div>

          <div className="flex-between">
            <button
              type="submit"
              className="btn btn-primary"
              disabled={uploading || !file || !title.trim()}
            >
              {uploading ? (
                <>
                  <span className="spinner" />
                  <span>Processing & Indexing...</span>
                </>
              ) : (
                'Upload & Index Material'
              )}
            </button>

            {uploadStage && (
              <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                {uploadStage}
              </span>
            )}
          </div>
        </form>
      </div>

      {/* Document Knowledge Base Table */}
      <div className="card">
        <div className="flex-between" style={{ marginBottom: '1rem' }}>
          <div>
            <h2 className="card-title">Institutional Documents ({documents.length})</h2>
            <p className="card-description" style={{ marginBottom: 0 }}>
              Documents currently indexed in ChromaDB and BM25 for grounding evaluations.
            </p>
          </div>
          <button
            className="btn btn-secondary btn-sm"
            onClick={fetchDocuments}
            disabled={loading}
          >
            {loading ? 'Refreshing...' : 'Refresh'}
          </button>
        </div>

        {documents.length === 0 ? (
          <div className="empty-state">
            <div className="empty-state-icon">📚</div>
            <h3 className="empty-state-title">No reference documents indexed yet</h3>
            <p className="empty-state-text">
              The institutional knowledge base is empty. Upload a reference document above to make evidence available during grading.
            </p>
          </div>
        ) : (
          <div className="table-container">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Title</th>
                  <th>Type</th>
                  <th>Pages</th>
                  <th>Indexed Chunks</th>
                  <th>Status</th>
                  <th>Date Added</th>
                  <th style={{ textAlign: 'right' }}>Actions</th>
                </tr>
              </thead>
              <tbody>
                {documents.map((doc) => (
                  <tr key={doc.id}>
                    <td>
                      <strong>{doc.title}</strong>
                      <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                        {doc.filename} ({formatFileSize(doc.file_size)})
                      </div>
                    </td>
                    <td>
                      <span className="badge badge-neutral">
                        {doc.doc_type.replace('_', ' ')}
                      </span>
                    </td>
                    <td>{doc.total_pages !== null ? doc.total_pages : 'N/A (text)'}</td>
                    <td>{doc.total_chunks}</td>
                    <td>
                      <span className={`badge badge-${doc.status}`}>
                        {doc.status}
                      </span>
                      {doc.error_message && (
                        <div style={{ fontSize: '0.75rem', color: 'var(--danger)', marginTop: '0.25rem' }}>
                          {doc.error_message}
                        </div>
                      )}
                    </td>
                    <td>{formatDate(doc.created_at)}</td>
                    <td style={{ textAlign: 'right' }}>
                      <button
                        className="btn btn-danger btn-sm"
                        onClick={() => setDocToDelete(doc)}
                      >
                        Delete
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Delete Confirmation Modal */}
      {docToDelete && (
        <div style={{
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          backgroundColor: 'rgba(0, 0, 0, 0.5)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          zIndex: 100,
        }}>
          <div style={{
            background: '#ffffff',
            borderRadius: '8px',
            padding: '1.75rem',
            maxWidth: '480px',
            width: '90%',
            boxShadow: 'var(--shadow-md)'
          }}>
            <h3 style={{ marginBottom: '0.75rem', color: 'var(--danger)' }}>
              Confirm Document Deletion
            </h3>
            <p style={{ fontSize: '0.9rem', color: 'var(--text-secondary)', marginBottom: '1.25rem' }}>
              Are you sure you want to delete <strong>"{docToDelete.title}"</strong>?
              This will permanently remove the source file, all its {docToDelete.total_chunks} indexed chunks from ChromaDB, and its BM25 lexical entries.
            </p>
            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '0.75rem' }}>
              <button
                className="btn btn-secondary"
                onClick={() => setDocToDelete(null)}
                disabled={deleting}
              >
                Cancel
              </button>
              <button
                className="btn btn-danger"
                onClick={confirmDelete}
                disabled={deleting}
              >
                {deleting ? 'Deleting...' : 'Delete Permanently'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
