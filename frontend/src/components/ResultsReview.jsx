import React from 'react';

export default function ResultsReview({ result, onNewEvaluation }) {
  if (!result) {
    return (
      <div className="card">
        <h2 className="card-title">Results & Review</h2>
        <div className="empty-state">
          <div className="empty-state-icon">📊</div>
          <h3 className="empty-state-title">No grading results to display</h3>
          <p className="empty-state-text">
            Submit a student response in Section 3 to view the objective criterion breakdown, citations, and LLM feedback.
          </p>
        </div>
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
    provenance
  } = result;

  const getScoreColor = (pct) => {
    if (pct >= 80) return 'var(--success)';
    if (pct >= 50) return 'var(--warning)';
    return 'var(--danger)';
  };

  const isLowScoreOrIrrelevant = total_score === 0 || percentage < 30;

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
              Evaluated via EIOS RAG Engine • Submission Type: <strong>{provenance?.submission_type || 'text'}</strong>
              {provenance?.submission_filename ? ` (${provenance.submission_filename})` : ''}
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
          <p className="results-text-full" style={{ fontSize: '0.9rem', color: 'var(--text-primary)', margin: 0 }}>
            {overall_feedback}
          </p>
        </div>
      </div>

      {/* Criterion Breakdown Section */}
      <div className="card" style={{ padding: '1.25rem 1.5rem', marginBottom: '1.25rem' }}>
        <div className="flex-between" style={{ marginBottom: '1rem', flexWrap: 'wrap', gap: '0.5rem' }}>
          <div>
            <h3 className="card-title" style={{ fontSize: '1.2rem' }}>
              Criterion-Wise Evaluation Breakdown ({criterion_breakdown.length})
            </h3>
            <p className="card-description" style={{ marginBottom: 0, fontSize: '0.825rem' }}>
              Transparent, objective assessment per rubric requirement.
            </p>
          </div>
        </div>

        <div className="criteria-results-grid">
          {criterion_breakdown.map((crit, idx) => {
            const critPct = crit.max_marks > 0 ? (crit.awarded_marks / crit.max_marks) * 100 : 0;
            const isFullMarks = crit.awarded_marks >= crit.max_marks;
            const isZeroMarks = crit.awarded_marks === 0;

            return (
              <div key={idx} className="criterion-result-card">
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
                  <p className="results-text-full" style={{ margin: 0, fontSize: '0.9rem', color: 'var(--text-primary)' }}>
                    {crit.feedback}
                  </p>
                </div>
              </div>
            );
          })}
        </div>
      </div>

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
          <div><strong>Embedding Model:</strong> {provenance?.embedding_model || 'N/A'}</div>
          <div><strong>Reranker:</strong> {provenance?.reranker_model || 'N/A'}</div>
          <div><strong>NLI Classifier:</strong> {provenance?.nli_model || 'N/A'}</div>
        </div>
      </details>
    </div>
  );
}
