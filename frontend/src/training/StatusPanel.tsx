import { type TrainingStatus } from "../api";

interface StatusPanelProps {
  status: TrainingStatus;
}

export default function StatusPanel({ status }: StatusPanelProps) {
  const isTraining = status.is_training;
  const lastUpdated = status.last_updated
    ? new Date(status.last_updated).toLocaleTimeString()
    : "—";

  return (
    <div
      style={{
        background: "#0f172a",
        border: "1px solid #1e293b",
        borderRadius: "10px",
        padding: "1rem 1.25rem",
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: "0.75rem", marginBottom: "0.75rem" }}>
        {/* Spinning/status indicator */}
        <div
          style={{
            width: 10,
            height: 10,
            borderRadius: "50%",
            background: isTraining ? "#22c55e" : "#64748b",
            boxShadow: isTraining ? "0 0 8px rgba(34,197,94,0.6)" : "none",
            animation: isTraining ? "pulse 1.2s ease-in-out infinite" : "none",
          }}
        />
        <span style={{ color: "#f1f5f9", fontWeight: 700, fontSize: "0.9rem" }}>
          {isTraining ? "Training in Progress" : "Idle"}
        </span>
        {isTraining && (
          <span style={{ color: "#22c55e", fontSize: "0.7rem", fontWeight: 600, marginLeft: "auto" }}>
            ● LIVE
          </span>
        )}
      </div>

      {/* Phase */}
      <div style={rowStyle}>
        <span style={labelStyle}>Phase</span>
        <span style={valueStyle}>
          {status.current_phase || (isTraining ? "Active" : "—")}
        </span>
      </div>

      {/* Progress bar */}
      {status.iteration != null && status.total_iterations != null && (
        <div style={{ marginBottom: "0.6rem" }}>
          <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "0.25rem" }}>
            <span style={labelStyle}>Progress</span>
            <span style={{ ...valueStyle, fontSize: "0.75rem" }}>
              {status.iteration} / {status.total_iterations}
            </span>
          </div>
          <div
            style={{
              width: "100%",
              height: 6,
              background: "#1e293b",
              borderRadius: 3,
              overflow: "hidden",
            }}
          >
            <div
              style={{
                height: "100%",
                width: `${Math.min(100, (status.iteration / status.total_iterations) * 100)}%`,
                background: "linear-gradient(90deg, #3b82f6, #8b5cf6)",
                borderRadius: 3,
                transition: "width 0.5s ease",
              }}
            />
          </div>
        </div>
      )}

      {/* Loss (only relevant during AlphaZero training) */}
      {status.avg_loss != null && (
        <div style={rowStyle}>
          <span style={labelStyle}>Avg Loss</span>
          <span style={{ ...valueStyle, fontFamily: "monospace" }}>
            {status.avg_loss.toFixed(4)}
          </span>
        </div>
      )}

      {/* Win rate (if available in status) */}
      {status.win_rate != null && (
        <div style={rowStyle}>
          <span style={labelStyle}>Win Rate</span>
          <span style={valueStyle}>
            {(status.win_rate * 100).toFixed(1)}%
          </span>
        </div>
      )}

      {/* Last updated */}
      <div
        style={{
          ...rowStyle,
          borderTop: "1px solid #1e293b",
          paddingTop: "0.5rem",
          marginTop: "0.25rem",
        }}
      >
        <span style={{ ...labelStyle, fontSize: "0.7rem" }}>Updated</span>
        <span style={{ ...valueStyle, fontSize: "0.75rem", color: "#64748b" }}>
          {lastUpdated}
        </span>
      </div>

      {/* CSS keyframes injected via style tag */}
      <style>{`
        @keyframes pulse {
          0%, 100% { opacity: 1; transform: scale(1); }
          50% { opacity: 0.5; transform: scale(0.85); }
        }
      `}</style>
    </div>
  );
}

const rowStyle: React.CSSProperties = {
  display: "flex",
  justifyContent: "space-between",
  alignItems: "center",
  marginBottom: "0.4rem",
};

const labelStyle: React.CSSProperties = {
  color: "#94a3b8",
  fontSize: "0.8rem",
  fontWeight: 500,
};

const valueStyle: React.CSSProperties = {
  color: "#f1f5f9",
  fontSize: "0.85rem",
  fontWeight: 600,
};
