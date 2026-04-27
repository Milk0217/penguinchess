import { useEffect, useState, useRef, useCallback } from "react";
import { api, type TrainingStatus, type TrainingMetrics } from "../api";
import StatusPanel from "./StatusPanel";
import EloChart from "./EloChart";
import WinRateChart from "./WinRateChart";

const POLL_INTERVAL_MS = 5000;

export default function TrainingDashboard() {
  const [status, setStatus] = useState<TrainingStatus>({ is_training: false });
  const [metrics, setMetrics] = useState<TrainingMetrics | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const pollingRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const fetchStatus = useCallback(async () => {
    try {
      const s = await api.getTrainingStatus();
      setStatus(s);
      setError(null);
    } catch (e: any) {
      setError(e.message || "Failed to fetch training status");
    }
  }, []);

  const fetchMetrics = useCallback(async () => {
    try {
      const m = await api.getTrainingMetrics();
      setMetrics(m);
    } catch {
      // metrics fetch failure is non-fatal
    }
  }, []);

  // Initial fetch
  useEffect(() => {
    const init = async () => {
      setLoading(true);
      await Promise.all([fetchStatus(), fetchMetrics()]);
      setLoading(false);
    };
    init();
  }, [fetchStatus, fetchMetrics]);

  // Polling every 5s
  useEffect(() => {
    pollingRef.current = setInterval(fetchStatus, POLL_INTERVAL_MS);
    return () => {
      if (pollingRef.current) clearInterval(pollingRef.current);
    };
  }, [fetchStatus]);

  // Re-fetch metrics when training status changes to "not training"
  // (training just finished → new metrics may be available)
  const prevTraining = useRef(status.is_training);
  useEffect(() => {
    if (prevTraining.current && !status.is_training) {
      fetchMetrics();
    }
    prevTraining.current = status.is_training;
  }, [status.is_training, fetchMetrics]);

  if (loading) {
    return (
      <div style={containerBase}>
        <div style={{ textAlign: "center", color: "#64748b", padding: "2rem" }}>
          Loading training data...
        </div>
      </div>
    );
  }

  return (
    <div style={containerBase}>
      {/* Header */}
      <div style={{ display: "flex", alignItems: "center", gap: "0.5rem", marginBottom: "0.75rem" }}>
        <span style={{ fontSize: "1.1rem" }}>🧠</span>
        <h2
          style={{
            color: "#f1f5f9",
            fontSize: "1rem",
            fontWeight: 700,
            margin: 0,
          }}
        >
          Training Dashboard
        </h2>
        {status.is_training && (
          <span
            style={{
              fontSize: "0.65rem",
              background: "rgba(34,197,94,0.15)",
              color: "#22c55e",
              padding: "0.15rem 0.5rem",
              borderRadius: "10px",
              fontWeight: 700,
              animation: "pulse 1.2s ease-in-out infinite",
            }}
          >
            LIVE
          </span>
        )}
      </div>

      {error && (
        <div
          style={{
            color: "#f87171",
            fontSize: "0.8rem",
            marginBottom: "0.75rem",
            padding: "0.5rem 0.75rem",
            background: "rgba(248,113,113,0.08)",
            border: "1px solid rgba(248,113,113,0.2)",
            borderRadius: "6px",
          }}
        >
          ⚠ {error}
        </div>
      )}

      {/* Grid: Status panel + Charts */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "260px 1fr",
          gap: "0.75rem",
          alignItems: "start",
        }}
      >
        {/* Left: Status panel */}
        <StatusPanel status={status} />

        {/* Right: Charts stacked vertically */}
        <div style={{ display: "flex", flexDirection: "column", gap: "0.75rem" }}>
          <EloChart metrics={metrics ?? emptyMetrics} />
          <WinRateChart metrics={metrics ?? emptyMetrics} />
        </div>
      </div>
    </div>
  );
}

const containerBase: React.CSSProperties = {
  width: "100%",
  maxWidth: "760px",
  margin: "0 auto",
  padding: "1rem 0",
};

const emptyMetrics: TrainingMetrics = {
  ppo: { generations: [], elos: [], win_rates: [] },
  alphazero: { generations: [], elos: [], win_rates: [] },
  models: [],
};
