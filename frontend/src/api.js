const API_BASE = 'http://127.0.0.1:8000';

async function handleResponse(response) {
  if (!response.ok) {
    let errorDetail = `HTTP Error ${response.status}: ${response.statusText}`;
    try {
      const data = await response.json();
      if (data.detail) {
        errorDetail = typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail);
      } else if (data.error) {
        errorDetail = data.error;
      }
    } catch {
      // Use fallback errorDetail
    }
    throw new Error(errorDetail);
  }
  return response.json();
}

export const api = {
  // System Health
  async getHealth() {
    const res = await fetch(`${API_BASE}/health`);
    return handleResponse(res);
  },

  // Section 1: Reference Materials
  async getDocuments() {
    const res = await fetch(`${API_BASE}/documents`);
    return handleResponse(res);
  },

  async uploadDocument(file, title, docType) {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('title', title);
    formData.append('doc_type', docType);

    const res = await fetch(`${API_BASE}/documents/upload`, {
      method: 'POST',
      body: formData,
    });
    return handleResponse(res);
  },

  async getDocumentStatus(docId) {
    const res = await fetch(`${API_BASE}/documents/${docId}/status`);
    return handleResponse(res);
  },

  async deleteDocument(docId) {
    const res = await fetch(`${API_BASE}/documents/${docId}`, {
      method: 'DELETE',
    });
    return handleResponse(res);
  },

  // Section 2: Assessments and Rubrics
  async getAssessments() {
    const res = await fetch(`${API_BASE}/assessments`);
    return handleResponse(res);
  },

  async getAssessment(id) {
    const res = await fetch(`${API_BASE}/assessments/${id}`);
    return handleResponse(res);
  },

  async createAssessment(assessmentData) {
    const res = await fetch(`${API_BASE}/assessments`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(assessmentData),
    });
    return handleResponse(res);
  },

  async updateAssessment(id, assessmentData) {
    const res = await fetch(`${API_BASE}/assessments/${id}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(assessmentData),
    });
    return handleResponse(res);
  },

  async deleteAssessment(id) {
    const res = await fetch(`${API_BASE}/assessments/${id}`, {
      method: 'DELETE',
    });
    return handleResponse(res);
  },

  // Section 3: Grade Student Answers
  async gradePastedText(assessmentId, studentText) {
    const res = await fetch(`${API_BASE}/grading/grade-text`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        assessment_id: assessmentId,
        student_text: studentText,
      }),
    });
    return handleResponse(res);
  },

  async gradeUploadedFile(assessmentId, file) {
    const formData = new FormData();
    formData.append('assessment_id', assessmentId);
    formData.append('file', file);

    const res = await fetch(`${API_BASE}/grading/grade-file`, {
      method: 'POST',
      body: formData,
    });
    return handleResponse(res);
  },

  async gradeBatchZip(assessmentId, zipFile) {
    const formData = new FormData();
    formData.append('assessment_id', assessmentId);
    formData.append('file', zipFile);

    const res = await fetch(`${API_BASE}/grading/batch-zip`, {
      method: 'POST',
      body: formData,
    });
    return handleResponse(res);
  },

  async getBatchStatus(batchId) {
    const res = await fetch(`${API_BASE}/grading/batch/${batchId}`);
    return handleResponse(res);
  },

  getBatchCsvUrl(batchId) {
    return `${API_BASE}/grading/batch/${batchId}/csv`;
  },
};
