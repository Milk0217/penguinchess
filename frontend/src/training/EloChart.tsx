import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";
import type { TrainingMetrics } from "../api";

interface EloChartProps {
  metrics: TrainingMetrics;
}

export default function EloChart({ metrics }: EloChartProps) {
  const { ppo, alphazero } = metrics;

  // Merge PPO and AlphaZero data points into a single series by generation index
  const allGens = new Set<number>();
  for (const g of ppo.generations) allGens.add(g);
  for (const g of alphazero.generations) allGens.add(g);

  const sortedGens = Array.from(allGens).sort((a, b) => a - b);

  const ppoMap = new Map<number, number | null>();
  ppo.generations.forEach((g, i) => ppoMap.set(g, ppo.elos[i]));

  const azMap = new Map<number, number | null>();
  alphazero.generations.forEach((g, i) => azMap.set(g, alphazero.elos[i]));

  const chartData = sortedGens.map((gen) => ({
    generation: gen,
    "PPO Elo": ppoMap.get(gen) ?? undefined,
    "AlphaZero Elo": azMap.get(gen) ?? undefined,
  }));

  // Compute sensible Y-axis domain
  const allElos = [
    ...ppo.elos.filter((e): e is number => e !== null),
    ...alphazero.elos.filter((e): e is number => e !== null),
  ];
  const minElo = allElos.length > 0 ? Math.min(...allElos) : 1000;
  const maxElo = allElos.length > 0 ? Math.max(...allElos) : 1400;
  const padding = Math.max(20, (maxElo - minElo) * 0.15);
  const yMin = Math.max(800, Math.floor((minElo - padding) / 50) * 50);
  const yMax = Math.ceil((maxElo + padding) / 50) * 50;

  if (chartData.length === 0) {
    return (
      <div style={emptyStyle}>
        <span style={{ fontSize: "1.2rem" }}>📊</span>
        <span>No ELO data yet — train a model to see the trend.</span>
      </div>
    );
  }

  return (
    <div style={containerStyle}>
      <h3 style={titleStyle}>ELO Trend</h3>
      <ResponsiveContainer width="100%" height={220}>
        <LineChart data={chartData} margin={{ top: 8, right: 12, left: 0, bottom: 4 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
          <XAxis
            dataKey="generation"
            stroke="#64748b"
            tick={{ fill: "#94a3b8", fontSize: 11 }}
            tickLine={{ stroke: "#334155" }}
            label={{ value: "Generation", position: "insideBottom", offset: -4, fill: "#64748b", fontSize: 11 }}
          />
          <YAxis
            domain={[yMin, yMax]}
            stroke="#64748b"
            tick={{ fill: "#94a3b8", fontSize: 11 }}
            tickLine={{ stroke: "#334155" }}
            tickFormatter={(v: number) => v.toFixed(0)}
          />
          <Tooltip
            contentStyle={{
              background: "#0f172a",
              border: "1px solid #1e293b",
              borderRadius: "6px",
              fontSize: "0.8rem",
              color: "#f1f5f9",
            }}
            formatter={(value: any) => {
              const num = typeof value === 'number' ? value : 0;
              return `${num.toFixed(0)} ELO`;
            }}
            labelFormatter={(label) => `Gen ${label}`}
          />
          <Legend
            wrapperStyle={{ fontSize: "0.75rem", color: "#94a3b8" }}
          />
          {ppo.elos.some((e) => e !== null) && (
            <Line
              type="monotone"
              dataKey="PPO Elo"
              stroke="#3b82f6"
              strokeWidth={2}
              dot={{ r: 3, fill: "#3b82f6", strokeWidth: 0 }}
              activeDot={{ r: 5, fill: "#3b82f6", strokeWidth: 0 }}
              connectNulls
            />
          )}
          {alphazero.elos.some((e) => e !== null) && (
            <Line
              type="monotone"
              dataKey="AlphaZero Elo"
              stroke="#a855f7"
              strokeWidth={2}
              dot={{ r: 3, fill: "#a855f7", strokeWidth: 0 }}
              activeDot={{ r: 5, fill: "#a855f7", strokeWidth: 0 }}
              connectNulls
            />
          )}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

const containerStyle: React.CSSProperties = {
  background: "#0f172a",
  border: "1px solid #1e293b",
  borderRadius: "10px",
  padding: "1rem 1rem 0.5rem",
};

const titleStyle: React.CSSProperties = {
  color: "#f1f5f9",
  fontSize: "0.85rem",
  fontWeight: 700,
  margin: "0 0 0.5rem",
};

const emptyStyle: React.CSSProperties = {
  background: "#0f172a",
  border: "1px solid #1e293b",
  borderRadius: "10px",
  padding: "2rem",
  display: "flex",
  flexDirection: "column",
  alignItems: "center",
  gap: "0.5rem",
  color: "#64748b",
  fontSize: "0.85rem",
};
