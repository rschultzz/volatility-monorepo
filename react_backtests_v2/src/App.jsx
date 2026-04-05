export default function App() {
  return (
    <div className="page">
      <div className="hero-card">
        <div className="eyebrow">Surface Dynamics</div>
        <h1>Backtests</h1>
        <p className="lead">
          React shell is live. The old Dash Backtests tab is removed from navigation,
          and this tab is now mounted from a separate React app.
        </p>

        <div className="status-grid">
          <div className="status-card">
            <div className="status-label">Frontend</div>
            <div className="status-value">React / Vite</div>
          </div>
          <div className="status-card">
            <div className="status-label">Mount path</div>
            <div className="status-value">/backtests-v2-preview/</div>
          </div>
          <div className="status-card">
            <div className="status-label">Step</div>
            <div className="status-value">Shell only</div>
          </div>
        </div>

        <div className="notes-card">
          <h2>Next build step</h2>
          <p>
            Add the first real backtesting workspace: a left control rail, a center results grid,
            and a lightweight strategy definition panel.
          </p>
        </div>
      </div>
    </div>
  );
}
