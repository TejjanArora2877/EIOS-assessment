import React, { useState, useEffect } from 'react';
import Navbar from './components/Navbar';
import ReferenceMaterials from './components/ReferenceMaterials';
import AssessmentsRubrics from './components/AssessmentsRubrics';
import GradeSubmissions from './components/GradeSubmissions';
import ResultsReview from './components/ResultsReview';
import { api } from './api';

export default function App() {
  const [activeSection, setActiveSection] = useState('materials');
  const [healthStatus, setHealthStatus] = useState(null);
  const [selectedAssessmentId, setSelectedAssessmentId] = useState('');
  const [gradingResult, setGradingResult] = useState(null);

  // Poll system health
  useEffect(() => {
    const fetchHealth = async () => {
      try {
        const data = await api.getHealth();
        setHealthStatus(data);
      } catch (err) {
        console.error('Health check failed:', err);
        setHealthStatus({ status: 'offline', ollama_ready: false });
      }
    };

    fetchHealth();
    const interval = setInterval(fetchHealth, 10000);
    return () => clearInterval(interval);
  }, []);

  const handleGradingSuccess = (result) => {
    setGradingResult(result);
    setActiveSection('results');
  };

  return (
    <div className="app-container">
      <Navbar
        activeSection={activeSection}
        onSelectSection={setActiveSection}
        healthStatus={healthStatus}
      />

      <main className="main-content">
        {activeSection === 'materials' && <ReferenceMaterials />}
        {activeSection === 'assessments' && (
          <AssessmentsRubrics
            selectedAssessmentId={selectedAssessmentId}
            onSelectAssessment={setSelectedAssessmentId}
          />
        )}
        {activeSection === 'grade' && (
          <GradeSubmissions
            selectedAssessmentId={selectedAssessmentId}
            onSelectAssessment={setSelectedAssessmentId}
            onGradingSuccess={handleGradingSuccess}
          />
        )}
        {activeSection === 'results' && (
          <ResultsReview
            result={gradingResult}
            onNewEvaluation={() => {
              setGradingResult(null);
              setActiveSection('grade');
            }}
          />
        )}
      </main>
    </div>
  );
}
