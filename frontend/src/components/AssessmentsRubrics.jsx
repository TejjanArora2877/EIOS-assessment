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
  const [question, setQuestion] = useState('');
  const [modelAnswer, setModelAnswer] = useState('');
  const [totalMarks, setTotalMarks] = useState(10);
  const [docScope, setDocScope] = useState('ALL');
  const [criteria, setCriteria] = useState([
    { name: '', description: '', max_marks: 5 },
    { name: '', description: '', max_marks: 5 },
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
    setQuestion('');
    setModelAnswer('');
    setTotalMarks(10);
    setDocScope('ALL');
    setCriteria([
      { name: '', description: '', max_marks: 5 },
      { name: '', description: '', max_marks: 5 },
    ]);
    setEditingId(null);
    setFormMode('create');
    setError(null);
    setSuccessMsg(null);
  };

  const openEditForm = (assessment) => {
    setTitle(assessment.title);
    setQuestion(assessment.question);
    setModelAnswer(assessment.model_answer);
    setTotalMarks(assessment.total_marks);
    setDocScope(assessment.doc_scope);
    setCriteria(assessment.criteria.map(c => ({
      name: c.name,
      description: c.description,
      max_marks: c.max_marks
    })));
    setEditingId(assessment.id);
    setFormMode('edit');
    setError(null);
    setSuccessMsg(null);
  };

  const handleAddCriterion = () => {
    setCriteria([...criteria, { name: '', description: '', max_marks: 1 }]);
  };

  const handleRemoveCriterion = (index) => {
    if (criteria.length <= 1) {
      setError('An assessment must have at least one rubric criterion.');
      return;
    }
    const updated = criteria.filter((_, i) => i !== index);
    setCriteria(updated);
  };

  const handleCriterionChange = (index, field, value) => {
    const updated = [...criteria];
    if (field === 'max_marks') {
      updated[index][field] = parseFloat(value) || 0;
    } else {
      updated[index][field] = value;
    }
    setCriteria(updated);
  };

  const criteriaSum = criteria.reduce((sum, c) => sum + (parseFloat(c.max_marks) || 0), 0);
  const marksMatch = Math.abs(criteriaSum - parseFloat(totalMarks || 0)) < 0.01;

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);
    setSuccessMsg(null);

    // Validation
    if (!title.trim()) {
      setError('Assessment title is required.');
      return;
    }
    if (!question.trim()) {
      setError('Question / assignment prompt is required.');
      return;
    }
    if (!modelAnswer.trim()) {
      setError('Authoritative model answer is required.');
      return;
    }
    if (criteria.length === 0) {
      setError('At least one rubric criterion is required.');
      return;
    }
    for (let i = 0; i < criteria.length; i++) {
      if (!criteria[i].name.trim()) {
        setError(`Criterion #${i + 1} requires a name.`);
        return;
      }
      if (!criteria[i].description.trim()) {
        setError(`Criterion #${i + 1} requires a description.`);
        return;
      }
      if (criteria[i].max_marks <= 0) {
        setError(`Criterion #${i + 1} marks must be greater than 0.`);
        return;
      }
    }
    if (!marksMatch) {
      setError(`Sum of criterion marks (${criteriaSum.toFixed(2)}) must equal total marks (${parseFloat(totalMarks).toFixed(2)}).`);
      return;
    }

    const payload = {
      title: title.trim(),
      question: question.trim(),
      model_answer: modelAnswer.trim(),
      total_marks: parseFloat(totalMarks),
      doc_scope: docScope,
      criteria: criteria.map(c => ({
        name: c.name.trim(),
        description: c.description.trim(),
        max_marks: parseFloat(c.max_marks)
      }))
    };

    try {
      setLoading(true);
      if (formMode === 'create') {
        const created = await api.createAssessment(payload);
        setSuccessMsg(`Assessment "${created.title}" saved successfully.`);
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
              Define questions, model answers, and explicit teacher-defined criteria with maximum marks.
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
                placeholder="e.g. Memory-Based Learning Concept Exam"
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

            <div className="form-group">
              <label className="form-label" htmlFor="assessment-question">
                Question / Assignment Prompt *
              </label>
              <textarea
                id="assessment-question"
                className="form-textarea"
                rows="3"
                placeholder="State the exact question given to students..."
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                required
              />
            </div>

            <div className="form-group">
              <label className="form-label" htmlFor="assessment-model-answer">
                Authoritative Model Answer (Expected Concepts) *
              </label>
              <textarea
                id="assessment-model-answer"
                className="form-textarea"
                rows="5"
                placeholder="Provide the complete teacher's model answer containing all core concepts and technical details..."
                value={modelAnswer}
                onChange={(e) => setModelAnswer(e.target.value)}
                required
              />
            </div>

            {/* Rubric Criteria Section */}
            <div style={{ marginTop: '1.5rem', marginBottom: '1.5rem' }}>
              <div className="flex-between" style={{ marginBottom: '0.75rem' }}>
                <div>
                  <h4 style={{ fontWeight: 600, color: 'var(--text-primary)' }}>
                    Rubric Criteria ({criteria.length})
                  </h4>
                  <div style={{ fontSize: '0.85rem', color: marksMatch ? 'var(--success)' : 'var(--danger)' }}>
                    Current Criteria Sum: <strong>{criteriaSum.toFixed(2)}</strong> / {parseFloat(totalMarks || 0).toFixed(2)} marks
                    {marksMatch ? ' ✓ (Balanced)' : ' ⚠️ (Sum must equal total marks)'}
                  </div>
                </div>
                <button
                  type="button"
                  className="btn btn-secondary btn-sm"
                  onClick={handleAddCriterion}
                >
                  + Add Criterion
                </button>
              </div>

              {criteria.map((crit, idx) => (
                <div
                  key={idx}
                  style={{
                    border: '1px solid var(--border)',
                    borderRadius: '6px',
                    padding: '1rem',
                    marginBottom: '0.75rem',
                    backgroundColor: 'var(--bg-main)'
                  }}
                >
                  <div className="flex-between" style={{ marginBottom: '0.5rem' }}>
                    <span style={{ fontWeight: 600, fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                      Criterion #{idx + 1}
                    </span>
                    {criteria.length > 1 && (
                      <button
                        type="button"
                        className="btn btn-sm"
                        style={{ color: 'var(--danger)', background: 'none', border: 'none', cursor: 'pointer' }}
                        onClick={() => handleRemoveCriterion(idx)}
                      >
                        Remove
                      </button>
                    )}
                  </div>

                  <div className="grid-2">
                    <div className="form-group" style={{ marginBottom: '0.5rem' }}>
                      <label className="form-label" style={{ fontSize: '0.8rem' }}>Criterion Name *</label>
                      <input
                        type="text"
                        className="form-input"
                        placeholder="e.g. Definition of MBL"
                        value={crit.name}
                        onChange={(e) => handleCriterionChange(idx, 'name', e.target.value)}
                        required
                      />
                    </div>
                    <div className="form-group" style={{ marginBottom: '0.5rem' }}>
                      <label className="form-label" style={{ fontSize: '0.8rem' }}>Max Marks *</label>
                      <input
                        type="number"
                        step="0.5"
                        min="0.5"
                        className="form-input"
                        value={crit.max_marks}
                        onChange={(e) => handleCriterionChange(idx, 'max_marks', e.target.value)}
                        required
                      />
                    </div>
                  </div>

                  <div className="form-group" style={{ marginBottom: 0 }}>
                    <label className="form-label" style={{ fontSize: '0.8rem' }}>Concept Description *</label>
                    <input
                      type="text"
                      className="form-input"
                      placeholder="Expected concepts, keywords, or properties to evaluate..."
                      value={crit.description}
                      onChange={(e) => handleCriterionChange(idx, 'description', e.target.value)}
                      required
                    />
                  </div>
                </div>
              ))}
            </div>

            <div style={{ display: 'flex', gap: '1rem', justifyContent: 'flex-end' }}>
              <button
                type="button"
                className="btn btn-secondary"
                onClick={() => setFormMode(null)}
                disabled={loading}
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

      {/* Saved Assessments List */}
      <div className="card">
        <h3 className="card-title" style={{ marginBottom: '1rem' }}>
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
                      <span className="badge badge-neutral" style={{ marginTop: '0.25rem' }}>
                        Total: {a.total_marks} Marks ({a.criteria.length} Criteria)
                      </span>
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
                      {a.criteria.map((c) => (
                        <li key={c.id}>
                          <strong>{c.name}</strong> ({c.max_marks} marks): {c.description}
                        </li>
                      ))}
                    </ul>
                  </div>
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
              This will permanently remove the question, model answer, and its rubric criteria.
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
