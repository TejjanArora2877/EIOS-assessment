import React, { useState, useEffect, useRef } from 'react';
import { api } from '../api';

export default function GradeSubmissions({
  selectedAssessmentId,
  onSelectAssessment,
  onGradingSuccess
}) {
  const [assessments, setAssessments] = useState([]);
  const [loadingAssessments, setLoadingAssessments] = useState(false);
  const [inputMode, setInputMode] = useState('text'); // 'text' | 'file' | 'batch'

  // Single Text mode
  const [studentText, setStudentText] = useState('');

  // Single File mode
  const [studentFile, setStudentFile] = useState(null);

  // Batch ZIP mode
  const [batchZipFile, setBatchZipFile] = useState(null);
  const [batchJob, setBatchJob] = useState(null);
  const [batchPolling, setBatchPolling] = useState(false);
  const [selectedBatchItem, setSelectedBatchItem] = useState(null);
  const pollTimerRef = useRef(null);

  // Common submission state
  const [grading, setGrading] = useState(false);
  const [gradingStage, setGradingStage] = useState('');
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchAssessments = async () => {
      try {
        setLoadingAssessments(true);
        const res = await api.getAssessments();
        setAssessments(res.assessments || []);
        if (!selectedAssessmentId && res.assessments && res.assessments.length > 0) {
          onSelectAssessment(res.assessments[0].id);
        }
      } catch (err) {
        setError(err.message);
      } finally {
        setLoadingAssessments(false);
      }
    };
    fetchAssessments();
  }, [selectedAssessmentId, onSelectAssessment]);

  // Polling effect for batch grading job
  useEffect(() => {
    if (batchPolling && batchJob && batchJob.batch_id && batchJob.status === 'processing') {
      pollTimerRef.current = setInterval(async () => {
        try {
          const updated = await api.getBatchStatus(batchJob.batch_id);
          setBatchJob(updated);
          if (updated.status !== 'processing') {
            setBatchPolling(false);
            if (pollTimerRef.current) clearInterval(pollTimerRef.current);
          }
        } catch (pollErr) {
          console.error('Batch status poll error:', pollErr);
        }
      }, 2000);
    } else {
      if (pollTimerRef.current) clearInterval(pollTimerRef.current);
    }

    return () => {
      if (pollTimerRef.current) clearInterval(pollTimerRef.current);
    };
  }, [batchPolling, batchJob]);

  const selectedAssessment = assessments.find((a) => a.id === selectedAssessmentId);

  const handleSingleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setStudentFile(e.target.files[0]);
    }
  };

  const handleZipFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setBatchZipFile(e.target.files[0]);
    }
  };

  const handleSingleGrade = async (e) => {
    e.preventDefault();
    setError(null);

    if (!selectedAssessmentId) {
      setError('Validation Error: Please select an active assessment before submitting.');
      return;
    }

    if (inputMode === 'text') {
      if (!studentText || !studentText.trim()) {
        setError('Validation Error: Student answer text cannot be empty or whitespace only.');
        return;
      }
    } else if (inputMode === 'file') {
      if (!studentFile) {
        setError('Validation Error: Please select a student answer document (PDF, DOCX, TXT) to upload.');
        return;
      }
    }

    try {
      setGrading(true);
      setGradingStage('Analyzing semantics, checking contradictions, and retrieving institutional evidence...');

      let result;
      if (inputMode === 'text') {
        result = await api.gradePastedText(selectedAssessmentId, studentText.trim());
      } else {
        setGradingStage('Extracting document text and routing to grading engine...');
        result = await api.gradeUploadedFile(selectedAssessmentId, studentFile);
      }

      setGradingStage('Generating constructive feedback from local LLM...');
      onGradingSuccess(result);
    } catch (err) {
      setError(err.message || 'Evaluation failed.');
    } finally {
      setGrading(false);
      setGradingStage('');
    }
  };

  const handleBatchSubmit = async (e) => {
    e.preventDefault();
    setError(null);

    if (!selectedAssessmentId) {
      setError('Validation Error: Please select an active assessment before submitting.');
      return;
    }

    if (!batchZipFile) {
      setError('Validation Error: Please select a ZIP archive containing student answers (.pdf, .docx, .txt).');
      return;
    }

    if (!batchZipFile.name.toLowerCase().endsWith('.zip')) {
      setError('Validation Error: The selected file must be a .zip archive.');
      return;
    }

    try {
      setGrading(true);
      setGradingStage('Uploading archive and validating documents...');
      const initialJob = await api.gradeBatchZip(selectedAssessmentId, batchZipFile);
      setBatchJob(initialJob);
      setBatchPolling(true);
    } catch (err) {
      setError(err.message || 'Failed to start batch grading.');
    } finally {
      setGrading(false);
      setGradingStage('');
    }
  };

  // Helper calculation for batch progress
  const totalCount = batchJob ? batchJob.total_files : 0;
  const processedCount = batchJob ? (batchJob.completed + batchJob.failed + batchJob.skipped) : 0;
  const progressPercent = totalCount > 0 ? Math.min(100, Math.round((processedCount / totalCount) * 100)) : 0;

  return (
    <div>
      <div className="card">
        <h2 className="card-title">Grade Student Submissions</h2>
        <p className="card-description">
          Evaluate student answers using the deterministic EIOS RAG pipeline. Select an assessment and choose single-answer or batch ZIP mode.
        </p>

        {error && <div className="alert alert-danger" style={{ marginBottom: '1.25rem' }}>{error}</div>}

        {/* Assessment Selector */}
        <div className="form-group">
          <label className="form-label" htmlFor="select-assessment">
            Select Active Assessment *
          </label>
          {assessments.length === 0 ? (
            <div className="alert alert-warning" style={{ marginTop: '0.5rem' }}>
              No assessments found in the system. Please create an assessment in Section 2 first.
            </div>
          ) : (
            <select
              id="select-assessment"
              className="form-select"
              value={selectedAssessmentId}
              onChange={(e) => onSelectAssessment(e.target.value)}
              disabled={grading || batchPolling}
            >
              {assessments.map((a) => (
                <option key={a.id} value={a.id}>
                  {a.title} ({a.total_marks} Marks, {a.criteria.length} Criteria)
                </option>
              ))}
            </select>
          )}
        </div>

        {selectedAssessment && (
          <div style={{
            backgroundColor: 'var(--bg-main)',
            border: '1px solid var(--border)',
            borderRadius: '6px',
            padding: '1rem',
            marginBottom: '1.5rem',
            fontSize: '0.85rem'
          }}>
            <div style={{ marginBottom: '0.5rem' }}>
              <strong>Question Prompt:</strong> {selectedAssessment.question}
            </div>
            <div style={{ marginBottom: '0.5rem' }}>
              <strong>Model Answer:</strong> {selectedAssessment.model_answer}
            </div>
            <div>
              <strong>Rubric Criteria ({selectedAssessment.total_marks} marks total):</strong>
              <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap', marginTop: '0.35rem' }}>
                {selectedAssessment.criteria.map((c) => (
                  <span key={c.id} className="badge badge-neutral">
                    {c.name}: {c.max_marks} marks
                  </span>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Input Mode Selector Tabs */}
        <div style={{ display: 'flex', gap: '0.75rem', marginBottom: '1.5rem', flexWrap: 'wrap' }}>
          <button
            type="button"
            className={`btn ${inputMode === 'text' ? 'btn-primary' : 'btn-secondary'}`}
            onClick={() => setInputMode('text')}
            disabled={grading || batchPolling}
          >
            ✏️ Single: Paste Text
          </button>
          <button
            type="button"
            className={`btn ${inputMode === 'file' ? 'btn-primary' : 'btn-secondary'}`}
            onClick={() => setInputMode('file')}
            disabled={grading || batchPolling}
          >
            📄 Single: Upload Document
          </button>
          <button
            type="button"
            className={`btn ${inputMode === 'batch' ? 'btn-primary' : 'btn-secondary'}`}
            onClick={() => setInputMode('batch')}
            disabled={grading || batchPolling}
          >
            📦 Batch: Upload ZIP Archive
          </button>
        </div>

        {/* Single Submission Form */}
        {inputMode !== 'batch' && (
          <form onSubmit={handleSingleGrade}>
            {inputMode === 'text' ? (
              <div className="form-group">
                <label className="form-label" htmlFor="student-answer-text">
                  Student Answer Text *
                </label>
                <textarea
                  id="student-answer-text"
                  className="form-textarea"
                  rows="8"
                  placeholder="Paste the student's free-text response here. (Empty submissions are rejected with a validation error)..."
                  value={studentText}
                  onChange={(e) => setStudentText(e.target.value)}
                  disabled={grading}
                />
              </div>
            ) : (
              <div className="form-group">
                <label className="form-label" htmlFor="student-file-input">
                  Upload Student Document (PDF, DOCX, TXT) *
                </label>
                <input
                  id="student-file-input"
                  type="file"
                  className="form-input"
                  accept=".pdf,.docx,.txt"
                  onChange={handleSingleFileChange}
                  disabled={grading}
                />
                {studentFile && (
                  <div style={{ marginTop: '0.35rem', fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                    Selected: {studentFile.name} ({(studentFile.size / 1024).toFixed(1)} KB)
                  </div>
                )}
              </div>
            )}

            <div className="flex-between" style={{ marginTop: '1.5rem' }}>
              <button
                type="submit"
                className="btn btn-primary"
                disabled={grading || !selectedAssessmentId}
                style={{ minWidth: '180px' }}
              >
                {grading ? (
                  <>
                    <span className="spinner" />
                    <span>Grading Submission...</span>
                  </>
                ) : (
                  '🚀 Evaluate & Grade Answer'
                )}
              </button>

              {gradingStage && (
                <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                  {gradingStage}
                </span>
              )}
            </div>
          </form>
        )}

        {/* Batch ZIP Submission Mode */}
        {inputMode === 'batch' && (
          <div>
            {!batchJob && (
              <form onSubmit={handleBatchSubmit}>
                <div className="form-group">
                  <label className="form-label" htmlFor="batch-zip-input">
                    Select ZIP Archive of Student Submissions (.zip) *
                  </label>
                  <input
                    id="batch-zip-input"
                    type="file"
                    className="form-input"
                    accept=".zip"
                    onChange={handleZipFileChange}
                    disabled={grading}
                  />
                  <div style={{ marginTop: '0.4rem', fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                    Archive safety bounds: Max 50 files, 50MB total uncompressed. Supported extensions: <code>.pdf</code>, <code>.docx</code>, <code>.txt</code>. Hidden metadata files (e.g. <code>__MACOSX</code>) are filtered automatically.
                  </div>
                  {batchZipFile && (
                    <div style={{ marginTop: '0.35rem', fontSize: '0.8rem', color: 'var(--primary)', fontWeight: 500 }}>
                      Selected archive: {batchZipFile.name} ({(batchZipFile.size / 1024).toFixed(1)} KB)
                    </div>
                  )}
                </div>

                <div className="flex-between" style={{ marginTop: '1.5rem' }}>
                  <button
                    type="submit"
                    className="btn btn-primary"
                    disabled={grading || !batchZipFile || !selectedAssessmentId}
                    style={{ minWidth: '200px' }}
                  >
                    {grading ? (
                      <>
                        <span className="spinner" />
                        <span>Starting Batch Job...</span>
                      </>
                    ) : (
                      '⚡ Start ZIP Batch Grading'
                    )}
                  </button>
                  {gradingStage && (
                    <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                      {gradingStage}
                    </span>
                  )}
                </div>
              </form>
            )}

            {/* Batch Job Live Monitor */}
            {batchJob && (
              <div style={{ marginTop: '1rem' }}>
                <div style={{
                  backgroundColor: 'var(--bg-main)',
                  border: '1px solid var(--border)',
                  borderRadius: '8px',
                  padding: '1.25rem',
                  marginBottom: '1.5rem'
                }}>
                  <div className="flex-between" style={{ marginBottom: '1rem' }}>
                    <div>
                      <h3 style={{ fontSize: '1.1rem', fontWeight: 600, color: 'var(--text-primary)' }}>
                        Batch Evaluation Monitor
                      </h3>
                      <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                        Assessment: <strong>{batchJob.assessment_title}</strong> (ID: {batchJob.batch_id.slice(0, 8)})
                      </div>
                    </div>
                    <div>
                      {batchJob.status === 'processing' && (
                        <span className="badge badge-primary" style={{ backgroundColor: '#2563eb', color: '#fff', padding: '0.4rem 0.8rem' }}>
                          <span className="spinner" style={{ width: '12px', height: '12px', marginRight: '6px', borderWidth: '2px' }} />
                          Processing ({processedCount}/{totalCount})
                        </span>
                      )}
                      {batchJob.status === 'completed' && (
                        <span className="badge badge-success" style={{ backgroundColor: '#16a34a', color: '#fff', padding: '0.4rem 0.8rem' }}>
                          ✓ Completed ({batchJob.completed} Graded)
                        </span>
                      )}
                      {batchJob.status === 'failed' && (
                        <span className="badge badge-danger" style={{ backgroundColor: '#dc2626', color: '#fff', padding: '0.4rem 0.8rem' }}>
                          ✕ Failed
                        </span>
                      )}
                    </div>
                  </div>

                  {/* Progress bar */}
                  <div style={{
                    width: '100%',
                    height: '10px',
                    backgroundColor: 'var(--border)',
                    borderRadius: '9999px',
                    overflow: 'hidden',
                    marginBottom: '1rem'
                  }}>
                    <div style={{
                      width: `${progressPercent}%`,
                      height: '100%',
                      backgroundColor: batchJob.status === 'failed' ? 'var(--danger)' : 'var(--primary)',
                      transition: 'width 0.4s ease'
                    }} />
                  </div>

                  {/* Summary Metric Counters */}
                  <div style={{
                    display: 'grid',
                    gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))',
                    gap: '0.75rem',
                    marginBottom: '1.25rem'
                  }}>
                    <div style={{ padding: '0.75rem', backgroundColor: '#fff', border: '1px solid var(--border)', borderRadius: '6px', textAlign: 'center' }}>
                      <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', textTransform: 'uppercase' }}>Total Files</div>
                      <div style={{ fontSize: '1.3rem', fontWeight: 700 }}>{batchJob.total_files}</div>
                    </div>
                    <div style={{ padding: '0.75rem', backgroundColor: '#fff', border: '1px solid var(--border)', borderRadius: '6px', textAlign: 'center' }}>
                      <div style={{ fontSize: '0.75rem', color: 'var(--success)', textTransform: 'uppercase' }}>Completed</div>
                      <div style={{ fontSize: '1.3rem', fontWeight: 700, color: 'var(--success)' }}>{batchJob.completed}</div>
                    </div>
                    <div style={{ padding: '0.75rem', backgroundColor: '#fff', border: '1px solid var(--border)', borderRadius: '6px', textAlign: 'center' }}>
                      <div style={{ fontSize: '0.75rem', color: 'var(--warning)', textTransform: 'uppercase' }}>Skipped (Empty)</div>
                      <div style={{ fontSize: '1.3rem', fontWeight: 700, color: 'var(--warning)' }}>{batchJob.skipped}</div>
                    </div>
                    <div style={{ padding: '0.75rem', backgroundColor: '#fff', border: '1px solid var(--border)', borderRadius: '6px', textAlign: 'center' }}>
                      <div style={{ fontSize: '0.75rem', color: 'var(--danger)', textTransform: 'uppercase' }}>Failed</div>
                      <div style={{ fontSize: '1.3rem', fontWeight: 700, color: 'var(--danger)' }}>{batchJob.failed}</div>
                    </div>
                  </div>

                  {/* Action buttons */}
                  <div style={{ display: 'flex', gap: '0.75rem', flexWrap: 'wrap' }}>
                    <a
                      href={api.getBatchCsvUrl(batchJob.batch_id)}
                      download={`batch_${batchJob.batch_id}_results.csv`}
                      className="btn btn-secondary btn-sm"
                      style={{ textDecoration: 'none' }}
                    >
                      📥 Download CSV Report
                    </a>
                    {batchJob.status !== 'processing' && (
                      <button
                        type="button"
                        className="btn btn-secondary btn-sm"
                        onClick={() => {
                          setBatchJob(null);
                          setBatchZipFile(null);
                        }}
                      >
                        🔄 Upload Another Batch
                      </button>
                    )}
                  </div>
                </div>

                {/* Per-File Batch Results Table */}
                <h3 style={{ fontSize: '1rem', fontWeight: 600, marginBottom: '0.75rem' }}>
                  Extracted Submissions ({batchJob.results.length} of {batchJob.total_files})
                </h3>

                {batchJob.results.length === 0 ? (
                  <div style={{ padding: '2rem', textAlign: 'center', color: 'var(--text-secondary)', border: '1px dashed var(--border)', borderRadius: '6px' }}>
                    {batchJob.status === 'processing' ? 'Processing files in sequential isolation...' : 'No files processed yet.'}
                  </div>
                ) : (
                  <div style={{ overflowX: 'auto', border: '1px solid var(--border)', borderRadius: '6px' }}>
                    <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.875rem' }}>
                      <thead>
                        <tr style={{ backgroundColor: 'var(--bg-main)', borderBottom: '1px solid var(--border)', textAlign: 'left' }}>
                          <th style={{ padding: '0.75rem' }}>Submission File</th>
                          <th style={{ padding: '0.75rem' }}>Status</th>
                          <th style={{ padding: '0.75rem' }}>Score</th>
                          <th style={{ padding: '0.75rem' }}>Percentage</th>
                          <th style={{ padding: '0.75rem' }}>Flags</th>
                          <th style={{ padding: '0.75rem', textAlign: 'right' }}>Details</th>
                        </tr>
                      </thead>
                      <tbody>
                        {batchJob.results.map((item, idx) => (
                          <tr key={idx} style={{ borderBottom: '1px solid var(--border)' }}>
                            <td style={{ padding: '0.75rem', fontWeight: 500 }}>
                              {item.filename}
                              {item.student_text_preview && (
                                <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginTop: '2px' }}>
                                  "{item.student_text_preview}..."
                                </div>
                              )}
                            </td>
                            <td style={{ padding: '0.75rem' }}>
                              {item.status === 'completed' && (
                                <span className="badge badge-success" style={{ backgroundColor: 'var(--success-bg)', color: 'var(--success)' }}>
                                  Graded
                                </span>
                              )}
                              {item.status === 'skipped' && (
                                <span className="badge badge-warning" style={{ backgroundColor: 'var(--warning-bg)', color: 'var(--warning)' }}>
                                  Skipped
                                </span>
                              )}
                              {item.status === 'failed' && (
                                <span className="badge badge-danger" style={{ backgroundColor: 'var(--danger-bg)', color: 'var(--danger)' }}>
                                  Failed
                                </span>
                              )}
                            </td>
                            <td style={{ padding: '0.75rem' }}>
                              {item.total_score != null ? `${item.total_score.toFixed(2)} / ${item.max_score.toFixed(2)}` : '-'}
                            </td>
                            <td style={{ padding: '0.75rem' }}>
                              {item.percentage != null ? `${item.percentage.toFixed(1)}%` : '-'}
                            </td>
                            <td style={{ padding: '0.75rem' }}>
                              {item.requires_teacher_review && (
                                <span className="badge badge-warning" style={{ backgroundColor: 'var(--warning-bg)', color: 'var(--warning)' }}>
                                  ⚠️ Needs Review
                                </span>
                              )}
                              {item.error_message && (
                                <span style={{ fontSize: '0.75rem', color: 'var(--danger)' }}>
                                  {item.error_message}
                                </span>
                              )}
                            </td>
                            <td style={{ padding: '0.75rem', textAlign: 'right' }}>
                              {item.status === 'completed' && (
                                <button
                                  type="button"
                                  className="btn btn-secondary btn-sm"
                                  onClick={() => setSelectedBatchItem(item)}
                                >
                                  🔍 View Breakdown
                                </button>
                              )}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </div>
            )}
          </div>
        )}
      </div>

      {/* Modal Dialog for Viewing Batch Item Breakdown */}
      {selectedBatchItem && (
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
          padding: '1.5rem'
        }}>
          <div style={{
            backgroundColor: '#fff',
            borderRadius: '8px',
            maxWidth: '800px',
            width: '100%',
            maxHeight: '90vh',
            overflowY: 'auto',
            padding: '1.5rem',
            boxShadow: 'var(--shadow-md)'
          }}>
            <div className="flex-between" style={{ marginBottom: '1rem', borderBottom: '1px solid var(--border)', paddingBottom: '0.75rem' }}>
              <div>
                <h3 style={{ fontSize: '1.2rem', fontWeight: 600 }}>
                  Submission Details: {selectedBatchItem.filename}
                </h3>
                <div style={{ fontSize: '0.9rem', color: 'var(--text-secondary)' }}>
                  Total Score: <strong>{selectedBatchItem.total_score?.toFixed(2)} / {selectedBatchItem.max_score?.toFixed(2)}</strong> ({selectedBatchItem.percentage?.toFixed(1)}%)
                </div>
              </div>
              <button
                type="button"
                className="btn btn-secondary btn-sm"
                onClick={() => setSelectedBatchItem(null)}
              >
                ✕ Close
              </button>
            </div>

            {selectedBatchItem.requires_teacher_review && (
              <div className="alert alert-warning" style={{ marginBottom: '1rem' }}>
                ⚠️ Flagged for teacher review: Contradictions or ungrounded assertions were flagged.
              </div>
            )}

            <h4 style={{ fontSize: '0.95rem', fontWeight: 600, marginBottom: '0.5rem' }}>
              Rubric Criterion Breakdown
            </h4>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
              {selectedBatchItem.criterion_breakdown && selectedBatchItem.criterion_breakdown.map((crit, idx) => (
                <div key={idx} style={{
                  border: '1px solid var(--border)',
                  borderRadius: '6px',
                  padding: '0.85rem',
                  backgroundColor: 'var(--bg-main)'
                }}>
                  <div className="flex-between" style={{ marginBottom: '0.35rem' }}>
                    <strong style={{ fontSize: '0.9rem' }}>{crit.criterion_name}</strong>
                    <span className="badge badge-neutral">
                      {crit.awarded_marks.toFixed(2)} / {crit.max_marks.toFixed(2)} marks
                    </span>
                  </div>
                  <div style={{ fontSize: '0.85rem', color: 'var(--text-primary)', marginTop: '0.35rem' }}>
                    <strong>Feedback:</strong> {crit.feedback}
                  </div>
                  {crit.contradictions_detected && crit.contradictions_detected.length > 0 && (
                    <div style={{ fontSize: '0.8rem', color: 'var(--danger)', marginTop: '0.25rem' }}>
                      ⚠️ <strong>Contradiction:</strong> {crit.contradictions_detected.join('; ')}
                    </div>
                  )}
                  {crit.missing_concepts && crit.missing_concepts.length > 0 && (
                    <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginTop: '0.25rem' }}>
                      <em>Missing:</em> {crit.missing_concepts.join('; ')}
                    </div>
                  )}
                </div>
              ))}
            </div>

            <div style={{ textAlign: 'right', marginTop: '1.25rem' }}>
              <button
                type="button"
                className="btn btn-primary btn-sm"
                onClick={() => setSelectedBatchItem(null)}
              >
                Done
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
