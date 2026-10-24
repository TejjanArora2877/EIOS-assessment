import React from 'react';

export default function Navbar({ activeSection, onSelectSection, healthStatus }) {
  const sections = [
    { id: 'materials', label: '1. Reference Materials', icon: '📚' },
    { id: 'assessments', label: '2. Assessments & Rubrics', icon: '📝' },
    { id: 'grade', label: '3. Grade Student Answers', icon: '✍️' },
    { id: 'results', label: '4. Results & Review', icon: '📊' },
  ];

  const isHealthy = healthStatus && healthStatus.status === 'healthy';

  return (
    <>
      <header className="app-header">
        <div className="brand">
          <span className="brand-icon">🎓</span>
          <div>
            <h1 className="brand-title">EIOS Grading System</h1>
            <p className="brand-subtitle">Explainable RAG Assessment & Grading</p>
          </div>
        </div>

        <div className="system-status-pill">
          <span className={`status-dot ${isHealthy ? 'online' : 'offline'}`} />
          <span>
            {healthStatus ? (
              isHealthy ? (
                <>Backend Online (Ollama {healthStatus.ollama_ready ? 'Ready' : 'Not Ready'})</>
              ) : (
                'System Degraded'
              )
            ) : (
              'Connecting...'
            )}
          </span>
        </div>
      </header>

      <nav className="nav-tabs">
        {sections.map((sec) => (
          <button
            key={sec.id}
            className={`nav-tab-btn ${activeSection === sec.id ? 'active' : ''}`}
            onClick={() => onSelectSection(sec.id)}
          >
            <span>{sec.icon}</span>
            <span>{sec.label}</span>
          </button>
        ))}
      </nav>
    </>
  );
}
