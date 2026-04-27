import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
  ReferenceLine,
} from "recharts";
import type { TrainingMetrics } from "../api";

interface WinRateChartProps {
  metrics: TrainingMetrics;
}

export default function WinRateChart({ metrics }: WinRateChartProps) {
  const { ppo, alphazero } = metrics;

  // Merge generations
  const allGens = new Set<number>();
  for (const g of ppo.generations) allGens.add(g);
  for (const g of alphazero.generations) allGens.add(g);

  const sortedGens = Array.from(allGens).sort((a, b) => a - b);

  const ppoMap = new Map<number, number | null>();
  ppo.generations.forEach((g, i) => ppoMap.set(g, ppo.win_rates[i]));

  const azMap = new Map<number, number | null>();
  alphazero.generations.forEach((g, i) => azMap.set(g, alphazero.win_rates[i]));

  const chartData = sortedGens.map((gen) => ({
    generation: gen,
    "PPO Win Rate": ppoMap.get(gen) ?? undefined,
    "AlphaZero Win Rate": azMap.get(gen) ?? undefined,
  }));

  if (chartData.length === 0) {
    return (
      <div style={emptyStyle}>
        <span style={{ fontSize: "1.2rem" }}>📈</span>
        <span>No win rate data yet — evaluate a model to populate.</span>
      </div>
    );
  }

  return (
    <div style={containerStyle}>
      <h3 style={titleStyle}>Win Rate vs Random</h3>
      <ResponsiveContainer width="100%" height={220}>
        <AreaChart data={chartData} margin={{ top: 8, right: 12, left: 0, bottom: 4 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
          <XAxis
            dataKey="generation"
            stroke="#64748b"
            tick={{ fill: "#94a3b8", fontSize: 11 }}
            tickLine={{ stroke: "#334155" }}
            label={{ value: "Generation", position: "insideBottom", offset: -4, fill: "#64748b", fontSize: 11 }}
          />
          <YAxis
            domain={[0, 1]}
            stroke="#64748b"
            tick={{ fill: "#94a3b8", fontSize: 11 }}
            tickLine={{ stroke: "#334155" }}
            tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`}
          />
          <ReferenceLine
            y={0.5}
            stroke="#f59e0b"
            strokeDasharray="4 4"
            strokeWidth={1.5}
            label={{
              value: "Random",
              position: "insideTopRight",
              fill: "#f59e0b",
              fontSize: 10,
            }}
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
              return `${(num * 100).toFixed(1)}%`;
            }}
            labelFormatter={(label) => `Gen ${label}`}
          />
          <Legend
            wrapperStyle={{ fontSize: "0.75rem", color: "#94a3b8" }}
          />
          {ppo.win_rates.some((w) => w !== null) && (
            <Area
              type="monotone"
              dataKey="PPO Win Rate"
              stroke="#3b82f6"
              strokeWidth={2}
              fill="#3b82f6"
              fillOpacity={0.08}
              dot={{ r: 3, fill: "#3b82f6", strokeWidth: 0 }}
              activeDot={{ r: 5, fill: "#3b82f6", strokeWidth: 0 }}
              connectNulls
            />
          )}
          {alphazero.win_rates.some((w) => w !== null) && (
            <Area
              type="monotone"
              dataKey="AlphaZero Win Rate"
              stroke="#a855f7"
              strokeWidth={2}
              fill="#a855f7"
              fillOpacity={0.08}
              dot={{ r: 3, fill: "#a855f7", strokeWidth: 0 }}
              activeDot={{ r: 5, fill: "#a855f7", strokeWidth: 0 }}
              connectNulls
            />
          )}
        </AreaChart>
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
