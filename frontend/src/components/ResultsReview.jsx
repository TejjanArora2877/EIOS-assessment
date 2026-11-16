import React, { useState, useEffect } from 'react';
import { api } from '../api';

function getScoreColor(pct) {
  if (pct >= 80) return 'var(--success)';
  if (pct >= 50) return 'var(--warning)';
  return 'var(--danger)';
}

function CriterionCard({ crit, getScoreColor }) {
  const critPct = crit.max_marks > 0 ? (crit.awarded_marks / crit.max_marks) * 100 : 0;
  const isFullMarks = crit.awarded_marks >= crit.max_marks;
  const isZeroMarks = crit.awarded_marks === 0;

  return (
    <div className="criterion-result-card">
      <div>
        {/* Criterion Header Row */}
        <div className="flex-between" style={{ alignItems: 'flex-start', flexWrap: 'wrap', gap: '0.75rem', marginBottom: '0.75rem' }}>
          <div style={{ flex: '1 1 240px', minWidth: 0 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', flexWrap: 'wrap' }}>
              <h4 style={{ fontSize: '1.05rem', fontWeight: 700, color: 'var(--text-primary)', wordBreak: 'break-word' }}>
                {crit.criterion_name}
              </h4>
              {isFullMarks && (
                <span className="badge badge-success" style={{ fontSize: '0.7rem' }}>
                  ✓ Full Marks
                </span>
              )}
              {isZeroMarks && (
                <span className="badge badge-danger" style={{ fontSize: '0.7rem' }}>
                  Zero Marks
                </span>
              )}
            </div>
            <p style={{ fontSize: '0.825rem', color: 'var(--text-secondary)', marginTop: '0.2rem', lineHeight: 1.4 }}>
              {crit.reason}
            </p>
          </div>

          <div style={{ textAlign: 'right', whiteSpace: 'nowrap' }}>
            <span style={{
              fontSize: '1.25rem',
              fontWeight: 800,
              color: getScoreColor(critPct)
            }}>
              {crit.awarded_marks.toFixed(2)} / {crit.max_marks.toFixed(2)}
            </span>
            <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginLeft: '0.35rem' }}>
              ({critPct.toFixed(0)}%)
            </span>
          </div>
        </div>

        {/* Contradictions Alert */}
        {crit.contradictions_detected && crit.contradictions_detected.length > 0 && (
          <div className="alert alert-danger" style={{ padding: '0.6rem 0.85rem', marginBottom: '0.65rem', fontSize: '0.825rem' }}>
            <strong>Contradictions Detected:</strong>
            <ul style={{ paddingLeft: '1.15rem', marginTop: '0.25rem' }}>
              {crit.contradictions_detected.map((cd, cIdx) => (
                <li key={cIdx}>{cd}</li>
              ))}
            </ul>
          </div>
        )}

        {/* Missing Concepts Alert */}
        {crit.missing_concepts && crit.missing_concepts.length > 0 && crit.awarded_marks < crit.max_marks && (
          <div style={{
            padding: '0.6rem 0.85rem',
            borderRadius: '6px',
            backgroundColor: 'var(--warning-bg)',
            border: '1px solid rgba(180, 83, 9, 0.25)',
            fontSize: '0.825rem',
            color: 'var(--warning)',
            marginBottom: '0.65rem'
          }}>
            <strong>Missing or Incomplete Concepts:</strong>
            <ul style={{ paddingLeft: '1.15rem', marginTop: '0.25rem' }}>
              {crit.missing_concepts.map((mc, mIdx) => (
                <li key={mIdx}>{mc}</li>
              ))}
            </ul>
          </div>
        )}
      </div>

      {/* Constructive LLM Feedback Panel */}
      <div className="feedback-panel">
        <strong style={{ display: 'block', marginBottom: '0.25rem', fontSize: '0.825rem', color: 'var(--primary)' }}>
          Constructive Feedback:
        </strong>
        {crit.feedback ? (
          <p className="results-text-full" style={{ margin: 0, fontSize: '0.9rem', color: 'var(--text-primary)' }}>
            {crit.feedback}
          </p>
        ) : (
          <p className="results-text-full" style={{ margin: 0, fontSize: '0.85rem', fontStyle: 'italic', color: 'var(--text-muted)' }}>
            Feedback skipped (score-only mode).
          </p>
        )}
      </div>
    </div>
  );
}

export default function ResultsReview({ result, onNewEvaluation }) {
  const [activeResult, setActiveResult] = useState(result);
  const [records, setRecords] = useState([]);
  const [assessments, setAssessments] = useState([]);
  const [selectedAssessmentFilter, setSelectedAssessmentFilter] = useState('');
  const [loadingRecords, setLoadingRecords] = useState(false);
  const [recordsError, setRecordsError] = useState(null);

  // Synchronize activeResult when parent provides a new result
  useEffect(() => {
    setActiveResult(result);
  }, [result]);

  const loadAssessments = async () => {
    try {
      const data = await api.getAssessments();
      setAssessments(data.assessments || []);
    } catch (err) {
      console.error('Failed to load assessments for filter:', err);
    }
  };

  const loadRecords = async (assessmentId = selectedAssessmentFilter) => {
    setLoadingRecords(true);
    setRecordsError(null);
    try {
      const data = await api.fetchGradingRecords(assessmentId);
      setRecords(data || []);
    } catch (err) {
      console.error('Failed to fetch grading records:', err);
      setRecordsError(err.message || 'Failed to load historical grading records.');
    } finally {
      setLoadingRecords(false);
    }
  };

  useEffect(() => {
    loadAssessments();
  }, []);

  useEffect(() => {
    loadRecords(selectedAssessmentFilter);
  }, [selectedAssessmentFilter]);

  const renderPreviousResultsSection = () => (
    <div className="card" style={{ padding: '1.25rem 1.5rem', marginBottom: '1.25rem' }}>
      <div className="flex-between" style={{ alignItems: 'center', flexWrap: 'wrap', gap: '0.75rem', marginBottom: '1rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
          <h3 className="card-title" style={{ fontSize: '1.2rem', margin: 0 }}>
            Previous Assessment Results
          </h3>
          <span className="badge badge-neutral" style={{ fontSize: '0.75rem' }}>
            {records.length} {records.length === 1 ? 'Record' : 'Records'}
          </span>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', flexWrap: 'wrap' }}>
          <select
            className="form-control"
            style={{ width: 'auto', minWidth: '220px', padding: '0.35rem 0.65rem', fontSize: '0.85rem' }}
            value={selectedAssessmentFilter}
            onChange={(e) => setSelectedAssessmentFilter(e.target.value)}
          >
            <option value="">All Stored Assessments</option>
            {assessments.map((a) => (
              <option key={a.id} value={a.id}>
                {a.title}
              </option>
            ))}
          </select>
          <button
            className="btn btn-secondary btn-sm"
            onClick={() => loadRecords(selectedAssessmentFilter)}
            disabled={loadingRecords}
            title="Refresh previous records"
          >
            {loadingRecords ? 'Refreshing...' : '🔄 Refresh'}
          </button>
          <button
            className="btn btn-danger btn-sm"
            onClick={async () => {
              const targetDesc = selectedAssessmentFilter
                ? 'all grading records for the selected assessment'
                : 'ALL stored grading records';
              const confirmed = window.confirm(
                `Are you sure you want to permanently delete ${targetDesc}? This action cannot be undone.`
              );
              if (!confirmed) return;
              try {
                await api.deleteGradingRecords(selectedAssessmentFilter);
                if (activeResult && (!selectedAssessmentFilter || activeResult.assessment_id === selectedAssessmentFilter)) {
                  setActiveResult(null);
                }
                await loadRecords(selectedAssessmentFilter);
              } catch (err) {
                console.error('Failed to delete grading records:', err);
                setRecordsError(err.message || 'Failed to delete grading records.');
              }
            }}
            disabled={loadingRecords || records.length === 0}
            title={selectedAssessmentFilter ? "Delete all records for selected assessment" : "Delete all stored records"}
          >
            {selectedAssessmentFilter ? '🗑️ Delete Filtered' : '🗑️ Clear All'}
          </button>
        </div>
      </div>

      {recordsError && (
        <div className="alert alert-danger" style={{ marginBottom: '1rem', fontSize: '0.85rem' }}>
          {recordsError}
        </div>
      )}

      {loadingRecords ? (
        <div style={{ textAlign: 'center', padding: '1.5rem', color: 'var(--text-muted)' }}>
          Loading historical grading records...
        </div>
      ) : records.length === 0 ? (
        <div className="empty-state" style={{ padding: '1.5rem' }}>
          <p className="empty-state-text" style={{ margin: 0 }}>
            No historical grading records found in SQLite for this filter.
          </p>
        </div>
      ) : (
        <div style={{ overflowX: 'auto' }}>
          <table className="table" style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.875rem' }}>
            <thead>
              <tr style={{ borderBottom: '2px solid var(--border)', textAlign: 'left', color: 'var(--text-secondary)' }}>
                <th style={{ padding: '0.65rem 0.75rem' }}>Submission / File</th>
                <th style={{ padding: '0.65rem 0.75rem' }}>Assessment</th>
                <th style={{ padding: '0.65rem 0.75rem' }}>Score</th>
                <th style={{ padding: '0.65rem 0.75rem' }}>Status</th>
                <th style={{ padding: '0.65rem 0.75rem' }}>Date / Time</th>
                <th style={{ padding: '0.65rem 0.75rem', textAlign: 'right' }}>Action</th>
              </tr>
            </thead>
            <tbody>
              {records.map((rec) => {
                const recPct = rec.max_score > 0 ? (rec.total_score / rec.max_score) * 100 : 0;
                const isSelected = activeResult?.id === rec.id;

                return (
                  <tr
                    key={rec.id}
                    style={{
                      borderBottom: '1px solid var(--border)',
                      backgroundColor: isSelected ? 'rgba(79, 70, 229, 0.05)' : 'transparent',
                      transition: 'background-color 0.15s ease'
                    }}
                  >
                    <td style={{ padding: '0.65rem 0.75rem', fontWeight: 600 }}>
                      📄 {rec.submission_filename || (rec.submission_type === 'text' ? 'Pasted Text' : 'File Upload')}
                    </td>
                    <td style={{ padding: '0.65rem 0.75rem', color: 'var(--text-secondary)' }}>
                      {rec.assessment_title}
                    </td>
                    <td style={{ padding: '0.65rem 0.75rem', fontWeight: 700, color: getScoreColor(recPct) }}>
                      {rec.total_score.toFixed(2)} / {rec.max_score.toFixed(2)} ({recPct.toFixed(1)}%)
                    </td>
                    <td style={{ padding: '0.65rem 0.75rem' }}>
                      {rec.requires_teacher_review ? (
                        <span className="badge badge-warning" style={{ fontSize: '0.75rem' }}>⚠️ Needs Review</span>
                      ) : (
                        <span className="badge badge-success" style={{ fontSize: '0.75rem' }}>✓ Verified</span>
                      )}
                    </td>
                    <td style={{ padding: '0.65rem 0.75rem', fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                      {rec.created_at ? new Date(rec.created_at).toLocaleString() : 'N/A'}
                    </td>
                    <td style={{ padding: '0.65rem 0.75rem', textAlign: 'right' }}>
                      <button
                        className="btn btn-secondary btn-sm"
                        style={{
                          backgroundColor: isSelected ? 'var(--primary)' : undefined,
                          color: isSelected ? '#ffffff' : undefined,
                          padding: '0.25rem 0.65rem',
                          fontSize: '0.8rem'
                        }}
                        onClick={() => {
                          setActiveResult(rec);
                          window.scrollTo({ top: 0, behavior: 'smooth' });
                        }}
                      >
                        {isSelected ? 'Viewing' : 'View Breakdown'}
                      </button>
                      <button
                        className="btn btn-danger btn-sm"
                        style={{
                          padding: '0.25rem 0.65rem',
                          fontSize: '0.8rem',
                          marginLeft: '0.4rem'
                        }}
                        onClick={async (e) => {
                          e.stopPropagation();
                          const filenameOrType = rec.submission_filename || (rec.submission_type === 'text' ? 'Pasted Text' : 'File');
                          const confirmed = window.confirm(
                            `Are you sure you want to delete the grading record for "${filenameOrType}"?`
                          );
                          if (!confirmed) return;
                          try {
                            await api.deleteGradingRecord(rec.id);
                            if (activeResult?.id === rec.id) {
                              setActiveResult(null);
                            }
                            await loadRecords(selectedAssessmentFilter);
                          } catch (err) {
                            console.error('Failed to delete record:', err);
                            setRecordsError(err.message || 'Failed to delete record.');
                          }
                        }}
                        title="Delete this record"
                      >
                        Delete
                      </button>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );

  // If there is no active result selected, render historical records or empty state
  if (!activeResult) {
    return (
      <div className="results-container">
        {records.length > 0 ? (
          <div>
            <div className="card" style={{ padding: '1.25rem 1.5rem', marginBottom: '1.25rem' }}>
              <div className="flex-between" style={{ alignItems: 'center' }}>
                <div>
                  <h2 className="card-title" style={{ fontSize: '1.4rem', margin: 0 }}>Results &amp; Review</h2>
                  <p style={{ fontSize: '0.875rem', color: 'var(--text-secondary)', marginTop: '0.25rem', marginBottom: 0 }}>
                    Select a previous evaluation below to view its complete breakdown, or grade a new answer in Section 3.
                  </p>
                </div>
                <button className="btn btn-primary btn-sm" onClick={onNewEvaluation}>
                  Grade New Submission
                </button>
              </div>
            </div>
            {renderPreviousResultsSection()}
          </div>
        ) : (
          <div className="card">
            <h2 className="card-title">Results &amp; Review</h2>
            <div className="empty-state">
              <div className="empty-state-icon">📊</div>
              <h3 className="empty-state-title">No grading results to display</h3>
              <p className="empty-state-text">
                Submit a student response in Section 3 to view the objective criterion breakdown, citations, and LLM feedback.
              </p>
            </div>
            {renderPreviousResultsSection()}
          </div>
        )}
      </div>
    );
  }

  const {
    assessment_title,
    total_score,
    max_score,
    percentage,
    criterion_breakdown,
    overall_feedback,
    requires_teacher_review,
    review_reasons,
    citations,
    provenance,
    question_results,
    submission_type,
    submission_filename,
    created_at
  } = activeResult;

  const isLowScoreOrIrrelevant = total_score === 0 || percentage < 30;
  const hasMultipleQuestions = question_results && question_results.length > 1;

  const displaySubType = submission_type || provenance?.submission_type || 'text';
  const displayFilename = submission_filename || provenance?.submission_filename;

  return (
    <div className="results-container">
      {/* Top Banner & Assessment Summary Card */}
      <div className="card" style={{ padding: '1.25rem 1.5rem', marginBottom: '1.25rem' }}>
        <div className="flex-between" style={{ alignItems: 'flex-start', flexWrap: 'wrap', gap: '1.25rem' }}>
          <div style={{ flex: '1 1 500px', minWidth: 0 }}>
            <span className="badge badge-neutral" style={{ marginBottom: '0.35rem', display: 'inline-block' }}>
              Official Evaluation Result
            </span>
            <h2 className="card-title" style={{ fontSize: '1.4rem', fontWeight: 700, color: 'var(--text-primary)', wordBreak: 'break-word' }}>
              {assessment_title}
            </h2>
            <div style={{ fontSize: '0.825rem', color: 'var(--text-secondary)', marginTop: '0.25rem' }}>
              Evaluated via EIOS RAG Engine • Submission Type: <strong>{displaySubType}</strong>
              {displayFilename ? ` (${displayFilename})` : ''}
              {hasMultipleQuestions && ` • ${question_results.length} Questions`}
              {created_at && ` • Evaluated: ${new Date(created_at).toLocaleString()}`}
            </div>
          </div>

          <div style={{ textAlign: 'right', minWidth: '180px' }}>
            <div style={{
              fontSize: '2.2rem',
              fontWeight: 800,
              color: getScoreColor(percentage),
              lineHeight: 1.1
            }}>
              {total_score.toFixed(2)} <span style={{ fontSize: '1.1rem', color: 'var(--text-muted)', fontWeight: 500 }}>/ {max_score.toFixed(2)}</span>
            </div>
            <div style={{ fontSize: '0.9rem', fontWeight: 600, color: 'var(--text-secondary)', marginTop: '0.2rem' }}>
              {percentage.toFixed(1)}% Total Score
            </div>
            <button
              className="btn btn-secondary btn-sm"
              style={{ marginTop: '0.65rem' }}
              onClick={onNewEvaluation}
            >
              Grade Another Answer
            </button>
          </div>
        </div>

        {/* Teacher Review Alert if flagged */}
        {requires_teacher_review && (
          <div className="alert alert-warning" style={{ marginTop: '1rem', marginBottom: 0, padding: '0.75rem 1rem' }}>
            <div>
              <strong>⚠️ Teacher Review Recommended</strong>
              <div style={{ fontSize: '0.825rem', marginTop: '0.2rem' }}>
                {review_reasons && review_reasons.length > 0 ? (
                  <ul style={{ paddingLeft: '1.25rem', margin: '0.2rem 0 0 0' }}>
                    {review_reasons.map((r, i) => (
                      <li key={i}>{r}</li>
                    ))}
                  </ul>
                ) : (
                  'Institutional reference evidence is insufficient or contradictory statements were flagged.'
                )}
              </div>
            </div>
          </div>
        )}

        {/* Overall Summary Feedback */}
        <div className="results-summary-box">
          <strong style={{ display: 'block', marginBottom: '0.25rem', fontSize: '0.85rem', color: 'var(--text-primary)' }}>
            Overall Assessment Summary:
          </strong>
          {overall_feedback ? (
            <p className="results-text-full" style={{ fontSize: '0.9rem', color: 'var(--text-primary)', margin: 0 }}>
              {overall_feedback}
            </p>
          ) : (
            <p className="results-text-full" style={{ fontSize: '0.85rem', fontStyle: 'italic', color: 'var(--text-muted)', margin: 0 }}>
              Feedback skipped (score-only mode).
            </p>
          )}
        </div>
      </div>

      {/* Multi-Question Breakdown Cards */}
      {hasMultipleQuestions ? (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem', marginBottom: '1.25rem' }}>
          {question_results.map((qRes, qIdx) => {
            const qPct = qRes.max_marks > 0 ? (qRes.awarded_marks / qRes.max_marks) * 100 : 0;

            return (
              <div key={qRes.question_id || qIdx} className="card" style={{ padding: '1.25rem 1.5rem', marginBottom: 0 }}>
                <div className="flex-between" style={{ alignItems: 'flex-start', flexWrap: 'wrap', gap: '0.75rem', marginBottom: '0.75rem' }}>
                  <div style={{ flex: '1 1 500px', minWidth: 0 }}>
                    <span className="badge badge-neutral" style={{ marginBottom: '0.25rem', display: 'inline-block' }}>
                      Question {qRes.question_order}
                    </span>
                    <h3 style={{ fontSize: '1.15rem', fontWeight: 700, color: 'var(--text-primary)', margin: 0 }}>
                      {qRes.question_text}
                    </h3>
                  </div>
                  <div style={{ textAlign: 'right' }}>
                    <span style={{ fontSize: '1.4rem', fontWeight: 800, color: getScoreColor(qPct) }}>
                      {qRes.awarded_marks.toFixed(2)} / {qRes.max_marks.toFixed(2)}
                    </span>
                    <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                      ({qPct.toFixed(1)}%)
                    </div>
                  </div>
                </div>

                {/* Segmented Answer Preview */}
                {qRes.student_answer_slice ? (
                  <details style={{ marginBottom: '0.75rem', backgroundColor: 'var(--bg-main)', borderRadius: '6px', padding: '0.5rem 0.75rem' }}>
                    <summary style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-secondary)', cursor: 'pointer' }}>
                      📄 View Segmented Student Answer ({qRes.student_answer_slice.split(/\s+/).length} words)
                    </summary>
                    <p style={{ fontSize: '0.85rem', color: 'var(--text-primary)', whiteSpace: 'pre-wrap', marginTop: '0.5rem', margin: '0.5rem 0 0 0', lineHeight: 1.5 }}>
                      {qRes.student_answer_slice}
                    </p>
                  </details>
                ) : (
                  <div style={{ fontSize: '0.825rem', color: 'var(--danger)', fontStyle: 'italic', marginBottom: '0.75rem' }}>
                    ⚠️ Question was skipped (no answer detected in submission).
                  </div>
                )}

                {/* Question Feedback */}
                {qRes.question_feedback && (
                  <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '0.75rem', padding: '0.5rem 0.75rem', backgroundColor: '#fdfdfd', borderLeft: '3px solid var(--primary)', borderRadius: '4px' }}>
                    <strong>Question Feedback:</strong> {qRes.question_feedback}
                  </div>
                )}

                {/* Criteria Breakdown Grid */}
                <div className="criteria-results-grid">
                  {(qRes.criterion_breakdown || []).map((crit, cIdx) => (
                    <CriterionCard key={cIdx} crit={crit} getScoreColor={getScoreColor} />
                  ))}
                </div>
              </div>
            );
          })}
        </div>
      ) : (
        /* Single-Question Flat Breakdown */
        <div className="card" style={{ padding: '1.25rem 1.5rem', marginBottom: '1.25rem' }}>
          <div className="flex-between" style={{ marginBottom: '1rem', flexWrap: 'wrap', gap: '0.5rem' }}>
            <div>
              <h3 className="card-title" style={{ fontSize: '1.2rem' }}>
                Criterion-Wise Evaluation Breakdown ({criterion_breakdown ? criterion_breakdown.length : 0})
              </h3>
              <p className="card-description" style={{ marginBottom: 0, fontSize: '0.825rem' }}>
                Transparent, objective assessment per rubric requirement.
              </p>
            </div>
          </div>

          <div className="criteria-results-grid">
            {(criterion_breakdown || []).map((crit, idx) => (
              <CriterionCard key={idx} crit={crit} getScoreColor={getScoreColor} />
            ))}
          </div>
        </div>
      )}

      {/* Institutional Reference Materials */}
      <div className="card" style={{ padding: '1.25rem 1.5rem', marginBottom: '1.25rem' }}>
        <h3 className="card-title" style={{ fontSize: '1.2rem', marginBottom: '0.3rem' }}>
          Retrieved Institutional Reference Materials ({citations ? citations.length : 0})
        </h3>
        <p className="card-description" style={{ marginBottom: isLowScoreOrIrrelevant ? '0.65rem' : '1rem', fontSize: '0.825rem' }}>
          Passages retrieved from institutional textbooks to ground the assessment criteria and model answer. They represent official reference material, not proof that the student's submission is correct.
        </p>

        {isLowScoreOrIrrelevant && (
          <div style={{
            padding: '0.6rem 0.85rem',
            backgroundColor: 'var(--bg-main)',
            border: '1px solid var(--border)',
            borderRadius: '6px',
            fontSize: '0.825rem',
            color: 'var(--text-secondary)',
            marginBottom: '1rem'
          }}>
            ℹ️ <strong>Note:</strong> Reference materials ground the grading rubric for this assessment. Their presence does not validate or endorse incorrect or unrelated student statements.
          </div>
        )}

        {!citations || citations.length === 0 ? (
          <div className="empty-state" style={{ padding: '1.5rem' }}>
            <p className="empty-state-text">
              No direct institutional evidence citations retrieved for this assessment.
            </p>
          </div>
        ) : (
          <div className="citations-grid">
            {citations.map((c, i) => (
              <div
                key={i}
                style={{
                  border: '1px solid var(--border)',
                  borderLeft: '4px solid var(--secondary)',
                  borderRadius: '6px',
                  padding: '0.875rem 1rem',
                  backgroundColor: '#ffffff',
                  boxShadow: 'var(--shadow-sm)',
                  display: 'flex',
                  flexDirection: 'column',
                  justifyContent: 'space-between',
                  minWidth: 0
                }}
              >
                <div>
                  <div className="flex-between" style={{ marginBottom: '0.4rem', flexWrap: 'wrap', gap: '0.35rem' }}>
                    <span style={{ fontWeight: 600, fontSize: '0.85rem', color: 'var(--secondary)' }}>
                      📖 {c.document_title}
                    </span>
                    <span className="badge badge-neutral" style={{ fontSize: '0.725rem' }}>
                      {c.page_number !== null && c.page_number !== undefined ? `Page ${c.page_number}` : 'Page N/A'}
                    </span>
                  </div>
                  <p className="results-text-full" style={{
                    fontSize: '0.85rem',
                    color: 'var(--text-secondary)',
                    fontStyle: 'italic',
                    lineHeight: 1.45,
                    marginBottom: '0.65rem'
                  }}>
                    "{c.snippet}"
                  </p>
                </div>
                <div className="flex-between" style={{
                  fontSize: '0.75rem',
                  color: 'var(--text-muted)',
                  borderTop: '1px solid var(--border)',
                  paddingTop: '0.4rem',
                  marginTop: 'auto'
                }}>
                  <span>Relevance Score: {c.similarity_score}</span>
                  <span style={{ color: 'var(--text-secondary)', fontWeight: 500 }}>
                    Reference Context for Rubric
                  </span>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Provenance and Audit Trail (Collapsible) */}
      <details className="card" style={{ padding: '0.85rem 1.25rem', marginBottom: '1.25rem', cursor: 'pointer' }}>
        <summary style={{ fontWeight: 600, color: 'var(--text-secondary)', fontSize: '0.85rem' }}>
          🔍 Audit &amp; Provenance Details (Click to view SHA256 &amp; Models)
        </summary>
        <div style={{ marginTop: '0.75rem', display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '0.6rem', fontSize: '0.8rem', color: 'var(--text-muted)' }}>
          <div><strong>SHA256 Hash:</strong> {provenance?.text_sha256 ? `${provenance.text_sha256.substring(0, 16)}...` : 'N/A'}</div>
          <div><strong>Feedback Mode:</strong> {provenance?.feedback_mode || 'N/A'}</div>
          <div><strong>Ollama Calls:</strong> {provenance?.ollama_calls ?? 0}</div>
          <div><strong>Embedding Model:</strong> {provenance?.embedding_model || 'N/A'}</div>
          <div><strong>Reranker:</strong> {provenance?.reranker_model || 'N/A'}</div>
          <div><strong>NLI Classifier:</strong> {provenance?.nli_model || 'N/A'}</div>
          {provenance?.stage_durations && (
            <div style={{ gridColumn: '1 / -1', marginTop: '0.4rem', borderTop: '1px solid var(--border)', paddingTop: '0.4rem' }}>
              <strong>Stage Durations:</strong> Scoring: {provenance.stage_durations.numerical_scoring_ms}ms | Feedback: {provenance.stage_durations.feedback_generation_ms}ms | Total: {provenance.stage_durations.total_grading_ms}ms
            </div>
          )}
        </div>
      </details>

      {/* Previous Assessment Results (Persistent History) */}
      {renderPreviousResultsSection()}
    </div>
  );
}
