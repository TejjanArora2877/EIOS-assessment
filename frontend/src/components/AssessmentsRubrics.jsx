import React, { useState, useEffect } from 'react';
import { api } from '../api';

export default function AssessmentsRubrics({ selectedAssessmentId, onSelectAssessment }) {
  const [assessments, setAssessments] = useState([]);
  const [documents, setDocuments] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [successMsg, setSuccessMsg] = useState(null);

  // Form mode: null (view list) | 'create' | 'edit'
  const [formMode, setFormMode] = useState(null);
  const [editingId, setEditingId] = useState(null);

  // Form fields
  const [title, setTitle] = useState('');
  const [totalMarks, setTotalMarks] = useState(10);
  const [docScope, setDocScope] = useState('ALL');
  const [questions, setQuestions] = useState([
    {
      question_order: 1,
      question_text: '',
      model_answer: '',
      criteria: [
        { name: '', description: '', max_marks: 5 },
        { name: '', description: '', max_marks: 5 },
      ]
    }
  ]);

  // Deletion modal
  const [assessmentToDelete, setAssessmentToDelete] = useState(null);
  const [deleting, setDeleting] = useState(false);

  const fetchData = async () => {
    try {
      setLoading(true);
      const [aRes, dRes] = await Promise.all([
        api.getAssessments(),
        api.getDocuments()
      ]);
      setAssessments(aRes.assessments || []);
      setDocuments(dRes.documents || []);
      setError(null);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, []);

  const openCreateForm = () => {
    setTitle('');
    setTotalMarks(10);
    setDocScope('ALL');
    setQuestions([
      {
        question_order: 1,
        question_text: '',
        model_answer: '',
        criteria: [
          { name: '', description: '', max_marks: 5 },
          { name: '', description: '', max_marks: 5 },
        ]
      }
    ]);
    setEditingId(null);
    setFormMode('create');
    setError(null);
    setSuccessMsg(null);
  };

  const openEditForm = (assessment) => {
    setTitle(assessment.title);
    setTotalMarks(assessment.total_marks);
    setDocScope(assessment.doc_scope || 'ALL');

    if (assessment.questions && assessment.questions.length > 0) {
      setQuestions(assessment.questions.map((q, idx) => ({
        id: q.id,
        question_order: q.question_order || idx + 1,
        question_text: q.question_text || '',
        model_answer: q.model_answer || '',
        criteria: (q.criteria && q.criteria.length > 0 ? q.criteria : []).map(c => ({
          name: c.name,
          description: c.description,
          max_marks: c.max_marks
        }))
      })));
    } else {
      setQuestions([
        {
          question_order: 1,
          question_text: assessment.question || '',
          model_answer: assessment.model_answer || '',
          criteria: (assessment.criteria || []).map(c => ({
            name: c.name,
            description: c.description,
            max_marks: c.max_marks
          }))
        }
      ]);
    }

    setEditingId(assessment.id);
    setFormMode('edit');
    setError(null);
    setSuccessMsg(null);
  };

  const handleAddQuestion = () => {
    setQuestions([
      ...questions,
      {
        question_order: questions.length + 1,
        question_text: '',
        model_answer: '',
        criteria: [
          { name: '', description: '', max_marks: 5 }
        ]
      }
    ]);
  };

  const handleRemoveQuestion = (qIndex) => {
    if (questions.length <= 1) {
      setError('An assessment must have at least one question.');
      return;
    }
    const updated = questions
      .filter((_, i) => i !== qIndex)
      .map((q, idx) => ({ ...q, question_order: idx + 1 }));
    setQuestions(updated);
  };

  const handleQuestionChange = (qIndex, field, value) => {
    const updated = [...questions];
    updated[qIndex][field] = value;
    setQuestions(updated);
  };

  const handleAddCriterion = (qIndex) => {
    const updated = [...questions];
    updated[qIndex].criteria.push({ name: '', description: '', max_marks: 5 });
    setQuestions(updated);
  };

  const handleRemoveCriterion = (qIndex, critIndex) => {
    const updated = [...questions];
    if (updated[qIndex].criteria.length <= 1) {
      setError(`Question #${qIndex + 1} must have at least one rubric criterion.`);
      return;
    }
    updated[qIndex].criteria = updated[qIndex].criteria.filter((_, i) => i !== critIndex);
    setQuestions(updated);
  };

  const handleCriterionChange = (qIndex, critIndex, field, value) => {
    const updated = [...questions];
    if (field === 'max_marks') {
      updated[qIndex].criteria[critIndex][field] = parseFloat(value) || 0;
    } else {
      updated[qIndex].criteria[critIndex][field] = value;
    }
    setQuestions(updated);
  };

  // Overall criteria sum across all questions
  const totalCriteriaSum = questions.reduce(
    (sum, q) => sum + q.criteria.reduce((s, c) => s + (parseFloat(c.max_marks) || 0), 0),
    0
  );
  const marksMatch = Math.abs(totalCriteriaSum - parseFloat(totalMarks || 0)) < 0.01;

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);
    setSuccessMsg(null);

    // Validation
    if (!title.trim()) {
      setError('Assessment title is required.');
      return;
    }
    if (questions.length === 0) {
      setError('At least one question is required.');
      return;
    }

    for (let qIdx = 0; qIdx < questions.length; qIdx++) {
      const q = questions[qIdx];
      if (!q.question_text.trim()) {
        setError(`Question #${qIdx + 1}: Question text / prompt is required.`);
        return;
      }
      if (!q.model_answer.trim()) {
        setError(`Question #${qIdx + 1}: Authoritative model answer is required.`);
        return;
      }
      if (!q.criteria || q.criteria.length === 0) {
        setError(`Question #${qIdx + 1}: At least one rubric criterion is required.`);
        return;
      }
      for (let cIdx = 0; cIdx < q.criteria.length; cIdx++) {
        const c = q.criteria[cIdx];
        if (!c.name.trim()) {
          setError(`Question #${qIdx + 1}, Criterion #${cIdx + 1}: Name is required.`);
          return;
        }
        if (!c.description.trim()) {
          setError(`Question #${qIdx + 1}, Criterion #${cIdx + 1}: Description is required.`);
          return;
        }
        if (c.max_marks <= 0) {
          setError(`Question #${qIdx + 1}, Criterion #${cIdx + 1}: Marks must be greater than 0.`);
          return;
        }
      }
    }

    if (!marksMatch) {
      setError(
        `Sum of criterion marks (${totalCriteriaSum.toFixed(2)}) must equal total maximum marks (${parseFloat(totalMarks).toFixed(2)}).`
      );
      return;
    }

    const payload = {
      title: title.trim(),
      question: questions[0]?.question_text?.trim() || '',
      model_answer: questions[0]?.model_answer?.trim() || '',
      total_marks: parseFloat(totalMarks),
      doc_scope: docScope,
      criteria: questions.flatMap(q => q.criteria.map(c => ({
        name: c.name.trim(),
        description: c.description.trim(),
        max_marks: parseFloat(c.max_marks)
      }))),
      questions: questions.map((q, idx) => ({
        question_order: idx + 1,
        question_text: q.question_text.trim(),
        model_answer: q.model_answer.trim(),
        max_marks: q.criteria.reduce((sum, c) => sum + (parseFloat(c.max_marks) || 0), 0),
        criteria: q.criteria.map(c => ({
          name: c.name.trim(),
          description: c.description.trim(),
          max_marks: parseFloat(c.max_marks)
        }))
      }))
    };

    try {
      setLoading(true);
      if (formMode === 'create') {
        const created = await api.createAssessment(payload);
        setSuccessMsg(`Assessment "${created.title}" saved successfully with ${questions.length} question(s).`);
        onSelectAssessment(created.id);
      } else {
        const updated = await api.updateAssessment(editingId, payload);
        setSuccessMsg(`Assessment "${updated.title}" updated successfully.`);
      }
      setFormMode(null);
      await fetchData();
    } catch (err) {
      setError(err.message || 'Failed to save assessment.');
    } finally {
      setLoading(false);
    }
  };

  const confirmDelete = async () => {
    if (!assessmentToDelete) return;
    try {
      setDeleting(true);
      await api.deleteAssessment(assessmentToDelete.id);
      setSuccessMsg(`Assessment "${assessmentToDelete.title}" deleted.`);
      if (selectedAssessmentId === assessmentToDelete.id) {
        onSelectAssessment('');
      }
      setAssessmentToDelete(null);
      await fetchData();
    } catch (err) {
      setError(err.message || 'Failed to delete assessment.');
    } finally {
      setDeleting(false);
    }
  };

  return (
    <div>
      {/* Top action bar */}
      <div className="card">
        <div className="flex-between">
          <div>
            <h2 className="card-title">Assessments & Rubrics</h2>
            <p className="card-description" style={{ marginBottom: 0 }}>
              Define single or multi-question assessments with authoritative model answers and explicit rubric criteria.
            </p>
          </div>
          {formMode === null ? (
            <button className="btn btn-primary" onClick={openCreateForm}>
              + Create Assessment
            </button>
          ) : (
            <button className="btn btn-secondary" onClick={() => setFormMode(null)}>
              Cancel Form
            </button>
          )}
        </div>
      </div>

      {error && <div className="alert alert-danger">{error}</div>}
      {successMsg && <div className="alert alert-success">{successMsg}</div>}

      {/* Create / Edit Form */}
      {formMode !== null && (
        <div className="card">
          <h3 className="card-title" style={{ marginBottom: '1.25rem' }}>
            {formMode === 'create' ? 'Define New Assessment & Rubric' : 'Edit Assessment & Rubric'}
          </h3>

          <form onSubmit={handleSubmit}>
            <div className="form-group">
              <label className="form-label" htmlFor="assessment-title">
                Assessment Title *
              </label>
              <input
                id="assessment-title"
                type="text"
                className="form-input"
                placeholder="e.g. Operating Systems Final Exam / Security Concepts Assessment"
                value={title}
                onChange={(e) => setTitle(e.target.value)}
                required
              />
            </div>

            <div className="grid-2">
              <div className="form-group">
                <label className="form-label" htmlFor="assessment-total-marks">
                  Total Maximum Marks *
                </label>
                <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
                  <input
                    id="assessment-total-marks"
                    type="number"
                    step="0.5"
                    min="0.5"
                    className="form-input"
                    value={totalMarks}
                    onChange={(e) => setTotalMarks(e.target.value)}
                    required
                  />
                  <button
                    type="button"
                    className="btn btn-secondary btn-sm"
                    title="Auto-set total marks from criteria sum"
                    onClick={() => setTotalMarks(totalCriteriaSum)}
                    style={{ whiteSpace: 'nowrap' }}
                  >
                    Auto-Sum ({totalCriteriaSum.toFixed(1)})
                  </button>
                </div>
              </div>

              <div className="form-group">
                <label className="form-label" htmlFor="assessment-doc-scope">
                  Reference Evidence Scope
                </label>
                <select
                  id="assessment-doc-scope"
                  className="form-select"
                  value={docScope}
                  onChange={(e) => setDocScope(e.target.value)}
                >
                  <option value="ALL">All Indexed Knowledge Base Documents</option>
                  {documents.map((d) => (
                    <option key={d.id} value={d.id}>
                      Only: {d.title} ({d.filename})
                    </option>
                  ))}
                </select>
              </div>
            </div>

            {/* Questions Container */}
            <div style={{ marginTop: '1.5rem', marginBottom: '1.5rem' }}>
              <div className="flex-between" style={{ marginBottom: '1rem' }}>
                <div>
                  <h4 style={{ fontWeight: 700, color: 'var(--text-primary)', fontSize: '1.1rem' }}>
                    Assessment Questions ({questions.length})
                  </h4>
                  <div style={{ fontSize: '0.85rem', color: marksMatch ? 'var(--success)' : 'var(--danger)', marginTop: '0.2rem' }}>
                    Criteria Sum: <strong>{totalCriteriaSum.toFixed(2)}</strong> / {parseFloat(totalMarks || 0).toFixed(2)} total marks
                    {marksMatch ? ' ✓ (Balanced)' : ' ⚠️ (Sum must equal total marks)'}
                  </div>
                </div>
                <button
                  type="button"
                  className="btn btn-secondary btn-sm"
                  onClick={handleAddQuestion}
                >
                  + Add Question
                </button>
              </div>

              {questions.map((q, qIdx) => {
                const qSubtotal = q.criteria.reduce((s, c) => s + (parseFloat(c.max_marks) || 0), 0);

                return (
                  <div
                    key={qIdx}
                    style={{
                      border: '2px solid var(--border)',
                      borderRadius: '8px',
                      padding: '1.25rem',
                      marginBottom: '1.25rem',
                      backgroundColor: '#fbfbfb'
                    }}
                  >
                    <div className="flex-between" style={{ marginBottom: '0.75rem', borderBottom: '1px solid var(--border)', paddingBottom: '0.5rem' }}>
                      <span style={{ fontWeight: 700, fontSize: '0.95rem', color: 'var(--primary)' }}>
                        Question {qIdx + 1}
                        <span className="badge badge-neutral" style={{ marginLeft: '0.5rem', fontWeight: 500 }}>
                          Subtotal: {qSubtotal.toFixed(2)} Marks
                        </span>
                      </span>
                      {questions.length > 1 && (
                        <button
                          type="button"
                          className="btn btn-sm"
                          style={{ color: 'var(--danger)', background: 'none', border: 'none', cursor: 'pointer', fontWeight: 600 }}
                          onClick={() => handleRemoveQuestion(qIdx)}
                        >
                          Remove Question
                        </button>
                      )}
                    </div>

                    <div className="form-group">
                      <label className="form-label" style={{ fontSize: '0.85rem' }}>
                        Question / Assignment Prompt *
                      </label>
                      <textarea
                        className="form-textarea"
                        rows="2"
                        placeholder="State the exact question given to students (e.g. Distinguish passive vs active attacks)..."
                        value={q.question_text}
                        onChange={(e) => handleQuestionChange(qIdx, 'question_text', e.target.value)}
                        required
                      />
                    </div>

                    <div className="form-group">
                      <label className="form-label" style={{ fontSize: '0.85rem' }}>
                        Authoritative Model Answer (Expected Concepts) *
                      </label>
                      <textarea
                        className="form-textarea"
                        rows="4"
                        placeholder="Provide the complete teacher's model answer containing all core concepts for this question..."
                        value={q.model_answer}
                        onChange={(e) => handleQuestionChange(qIdx, 'model_answer', e.target.value)}
                        required
                      />
                    </div>

                    {/* Criteria Sub-section for this question */}
                    <div style={{ marginTop: '1rem', borderTop: '1px dashed var(--border)', paddingTop: '0.75rem' }}>
                      <div className="flex-between" style={{ marginBottom: '0.5rem' }}>
                        <span style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-secondary)' }}>
                          Rubric Criteria for Question {qIdx + 1} ({q.criteria.length})
                        </span>
                        <button
                          type="button"
                          className="btn btn-secondary btn-sm"
                          style={{ fontSize: '0.75rem', padding: '0.2rem 0.5rem' }}
                          onClick={() => handleAddCriterion(qIdx)}
                        >
                          + Add Criterion
                        </button>
                      </div>

                      {q.criteria.map((crit, cIdx) => (
                        <div
                          key={cIdx}
                          style={{
                            border: '1px solid var(--border)',
                            borderRadius: '6px',
                            padding: '0.75rem',
                            marginBottom: '0.5rem',
                            backgroundColor: '#ffffff'
                          }}
                        >
                          <div className="flex-between" style={{ marginBottom: '0.35rem' }}>
                            <span style={{ fontWeight: 600, fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                              Criterion #{cIdx + 1}
                            </span>
                            {q.criteria.length > 1 && (
                              <button
                                type="button"
                                className="btn btn-sm"
                                style={{ color: 'var(--danger)', background: 'none', border: 'none', cursor: 'pointer', fontSize: '0.75rem' }}
                                onClick={() => handleRemoveCriterion(qIdx, cIdx)}
                              >
                                Remove
                              </button>
                            )}
                          </div>

                          <div className="grid-2">
                            <div className="form-group" style={{ marginBottom: '0.35rem' }}>
                              <label className="form-label" style={{ fontSize: '0.75rem' }}>Criterion Name *</label>
                              <input
                                type="text"
                                className="form-input"
                                placeholder="e.g. Definition of Cybersecurity"
                                value={crit.name}
                                onChange={(e) => handleCriterionChange(qIdx, cIdx, 'name', e.target.value)}
                                required
                              />
                            </div>
                            <div className="form-group" style={{ marginBottom: '0.35rem' }}>
                              <label className="form-label" style={{ fontSize: '0.75rem' }}>Max Marks *</label>
                              <input
                                type="number"
                                step="0.5"
                                min="0.5"
                                className="form-input"
                                value={crit.max_marks}
                                onChange={(e) => handleCriterionChange(qIdx, cIdx, 'max_marks', e.target.value)}
                                required
                              />
                            </div>
                          </div>

                          <div className="form-group" style={{ marginBottom: 0 }}>
                            <label className="form-label" style={{ fontSize: '0.75rem' }}>
                              Description (Required Propositions & Concepts) *
                            </label>
                            <textarea
                              className="form-textarea"
                              rows="2"
                              placeholder="Describe the exact technical facts the student must demonstrate..."
                              value={crit.description}
                              onChange={(e) => handleCriterionChange(qIdx, cIdx, 'description', e.target.value)}
                              required
                            />
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                );
              })}
            </div>

            <div style={{ display: 'flex', gap: '0.75rem', justifyContent: 'flex-end', marginTop: '1.5rem' }}>
              <button
                type="button"
                className="btn btn-secondary"
                onClick={() => setFormMode(null)}
              >
                Cancel
              </button>
              <button
                type="submit"
                className="btn btn-primary"
                disabled={loading || !marksMatch}
              >
                {loading ? 'Saving...' : formMode === 'create' ? 'Save Assessment' : 'Update Assessment'}
              </button>
            </div>
          </form>
        </div>
      )}

      {/* List of Existing Assessments */}
      <div className="card">
        <h3 className="card-title" style={{ marginBottom: '1.25rem' }}>
          Saved Assessments ({assessments.length})
        </h3>

        {assessments.length === 0 ? (
          <div className="empty-state">
            <div className="empty-state-icon">📝</div>
            <h3 className="empty-state-title">No assessments created yet</h3>
            <p className="empty-state-text">
              Create your first assessment with a model answer and explicit rubric to evaluate student responses.
            </p>
          </div>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
            {assessments.map((a) => {
              const isSelected = selectedAssessmentId === a.id;
              const hasMultipleQuestions = a.questions && a.questions.length > 1;

              return (
                <div
                  key={a.id}
                  style={{
                    border: `2px solid ${isSelected ? 'var(--primary)' : 'var(--border)'}`,
                    borderRadius: '8px',
                    padding: '1.25rem',
                    backgroundColor: isSelected ? 'var(--primary-light)' : '#ffffff',
                    transition: 'border-color 0.2s ease',
                  }}
                >
                  <div className="flex-between" style={{ marginBottom: '0.5rem' }}>
                    <div>
                      <h4 style={{ fontSize: '1.1rem', fontWeight: 600, color: 'var(--text-primary)' }}>
                        {a.title}
                      </h4>
                      <div style={{ display: 'flex', gap: '0.4rem', marginTop: '0.25rem', flexWrap: 'wrap' }}>
                        <span className="badge badge-neutral">
                          Total: {a.total_marks} Marks
                        </span>
                        {hasMultipleQuestions ? (
                          <span className="badge badge-neutral">
                            {a.questions.length} Questions ({a.criteria?.length || 0} Total Criteria)
                          </span>
                        ) : (
                          <span className="badge badge-neutral">
                            1 Question ({a.criteria?.length || 0} Criteria)
                          </span>
                        )}
                      </div>
                    </div>
                    <div style={{ display: 'flex', gap: '0.5rem' }}>
                      <button
                        className={`btn btn-sm ${isSelected ? 'btn-primary' : 'btn-secondary'}`}
                        onClick={() => onSelectAssessment(a.id)}
                      >
                        {isSelected ? '✓ Selected for Grading' : 'Select for Grading'}
                      </button>
                      <button
                        className="btn btn-secondary btn-sm"
                        onClick={() => openEditForm(a)}
                      >
                        Edit
                      </button>
                      <button
                        className="btn btn-danger btn-sm"
                        onClick={() => setAssessmentToDelete(a)}
                      >
                        Delete
                      </button>
                    </div>
                  </div>

                  {hasMultipleQuestions ? (
                    <div style={{ marginTop: '0.75rem' }}>
                      <div style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: '0.35rem' }}>
                        Questions & Rubric Breakdown:
                      </div>
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                        {a.questions.map((q, qIdx) => (
                          <div
                            key={q.id || qIdx}
                            style={{
                              backgroundColor: isSelected ? '#ffffff' : 'var(--bg-main)',
                              padding: '0.65rem 0.85rem',
                              borderRadius: '6px',
                              fontSize: '0.8rem'
                            }}
                          >
                            <div className="flex-between" style={{ marginBottom: '0.25rem' }}>
                              <strong style={{ color: 'var(--text-primary)' }}>
                                Q{q.question_order || qIdx + 1}: {q.question_text}
                              </strong>
                              <span style={{ fontWeight: 600, color: 'var(--text-muted)' }}>
                                {q.max_marks} marks
                              </span>
                            </div>
                            <ul style={{ paddingLeft: '1.25rem', margin: 0, color: 'var(--text-secondary)' }}>
                              {(q.criteria || []).map((c) => (
                                <li key={c.id || c.name}>
                                  <strong>{c.name}</strong> ({c.max_marks} marks): {c.description}
                                </li>
                              ))}
                            </ul>
                          </div>
                        ))}
                      </div>
                    </div>
                  ) : (
                    <>
                      <p style={{ fontSize: '0.875rem', color: 'var(--text-secondary)', marginBottom: '0.75rem' }}>
                        <strong>Question:</strong> {a.question}
                      </p>

                      <div style={{
                        backgroundColor: isSelected ? '#ffffff' : 'var(--bg-main)',
                        padding: '0.75rem',
                        borderRadius: '6px',
                        fontSize: '0.8rem',
                      }}>
                        <strong style={{ display: 'block', marginBottom: '0.25rem', color: 'var(--text-primary)' }}>
                          Rubric Breakdown:
                        </strong>
                        <ul style={{ paddingLeft: '1.25rem', color: 'var(--text-secondary)' }}>
                          {(a.criteria || []).map((c) => (
                            <li key={c.id || c.name}>
                              <strong>{c.name}</strong> ({c.max_marks} marks): {c.description}
                            </li>
                          ))}
                        </ul>
                      </div>
                    </>
                  )}
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* Delete Confirmation Modal */}
      {assessmentToDelete && (
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
              Confirm Assessment Deletion
            </h3>
            <p style={{ fontSize: '0.9rem', color: 'var(--text-secondary)', marginBottom: '1.25rem' }}>
              Are you sure you want to delete <strong>"{assessmentToDelete.title}"</strong>?
              This will permanently remove the questions, model answers, and rubric criteria.
            </p>
            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '0.75rem' }}>
              <button
                className="btn btn-secondary"
                onClick={() => setAssessmentToDelete(null)}
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
