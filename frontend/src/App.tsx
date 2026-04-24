import { useState, useEffect, useCallback, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Shield,
  Activity,
  Search,
  BarChart3,
  Brain,
  Database,
  Bell,
  Upload,
  Download,
  Play,
  RefreshCw,
  AlertTriangle,
  CheckCircle,
  XCircle,
  ChevronRight,
  Menu,
  X,
  Eye,
  EyeOff,
  Zap,
  TrendingUp,
  Users,
  FileText,
  Settings,
} from "lucide-react";
import {
  AreaChart,
  Area,
  BarChart,
  Bar,
  LineChart,
  Line,
  PieChart,
  Pie,
  Cell,
  RadarChart,
  Radar,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ComposedChart,
  Scatter,
} from "recharts";
import * as api from "./api";
import type {
  DashboardStats,
  AllModelStats,
  PredictionResponse,
  Transaction,
  Alert,
} from "./api";

const COLORS = {
  teal: "#00d4aa",
  tealDark: "#00b893",
  blue: "#4fc3f7",
  danger: "#ef4444",
  warning: "#f59e0b",
  success: "#10b981",
  purple: "#a78bfa",
  pink: "#f472b6",
  navy900: "#0a1628",
  navy700: "#111d35",
  navy600: "#152642",
};

const NAV_ITEMS = [
  { id: "dashboard", label: "Dashboard", icon: Activity },
  { id: "detection", label: "Detection", icon: Shield },
  { id: "explorer", label: "Transactions", icon: Search },
  { id: "models", label: "Model Comparison", icon: BarChart3 },
  { id: "learning", label: "Adaptive Learning", icon: Brain },
  { id: "datasets", label: "Datasets", icon: Database },
  { id: "alerts", label: "Alerts", icon: Bell },
];

// ─── Sidebar ─────────────────────────────────────────────
function Sidebar({
  active,
  onNav,
  collapsed,
  onToggle,
  alertCount,
}: {
  active: string;
  onNav: (id: string) => void;
  collapsed: boolean;
  onToggle: () => void;
  alertCount: number;
}) {
  return (
    <aside
      className={`fixed left-0 top-0 h-full z-30 transition-all duration-300 ${collapsed ? "w-[72px]" : "w-60"}`}
      style={{
        background: "linear-gradient(180deg, #0d1b2a 0%, #0a1628 100%)",
        borderRight: "1px solid rgba(0,212,170,0.08)",
      }}
    >
      <div className="flex items-center gap-3 px-4 h-16 border-b border-white/5">
        <div
          className="w-9 h-9 rounded-xl flex items-center justify-center"
          style={{
            background: "linear-gradient(135deg, #00d4aa 0%, #00b893 100%)",
          }}
        >
          <Shield size={18} className="text-navy-900" />
        </div>
        {!collapsed && (
          <span className="font-bold text-base tracking-tight">
            FraudShield<span className="text-teal-400"> AI</span>
          </span>
        )}
        <button
          onClick={onToggle}
          className="ml-auto text-gray-500 hover:text-teal-400 transition-colors"
        >
          {collapsed ? <Menu size={18} /> : <X size={18} />}
        </button>
      </div>
      <nav className="mt-4 px-2 space-y-1">
        {NAV_ITEMS.map((item) => (
          <button
            key={item.id}
            onClick={() => onNav(item.id)}
            className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm font-medium transition-all duration-200
              ${
                active === item.id
                  ? "bg-teal-400/10 text-teal-400"
                  : "text-gray-400 hover:text-gray-200 hover:bg-white/5"
              }`}
          >
            <item.icon size={18} />
            {!collapsed && <span>{item.label}</span>}
            {item.id === "alerts" && alertCount > 0 && (
              <span className="ml-auto bg-red-500 text-white text-xs w-5 h-5 rounded-full flex items-center justify-center">
                {alertCount}
              </span>
            )}
          </button>
        ))}
      </nav>
    </aside>
  );
}

// ─── Stat Card ───────────────────────────────────────────
function StatCard({
  title,
  value,
  sub,
  icon: Icon,
  color,
  trend,
}: {
  title: string;
  value: string | number;
  sub?: string;
  icon: any;
  color: string;
  trend?: string;
}) {
  return (
    <motion.div
      className="stat-card"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
    >
      <div className="flex items-start justify-between">
        <div>
          <p className="text-xs text-gray-400 uppercase tracking-wider mb-1">
            {title}
          </p>
          <p className="text-2xl font-bold" style={{ color }}>
            {value}
          </p>
          {sub && <p className="text-xs text-gray-500 mt-1">{sub}</p>}
          {trend && (
            <p className="text-xs text-teal-400 mt-1 flex items-center gap-1">
              <TrendingUp size={12} />
              {trend}
            </p>
          )}
        </div>
        <div
          className="w-10 h-10 rounded-xl flex items-center justify-center"
          style={{ background: `${color}15` }}
        >
          <Icon size={20} style={{ color }} />
        </div>
      </div>
    </motion.div>
  );
}

// ─── Dashboard Page ──────────────────────────────────────
function DashboardPage({
  stats,
  modelStats,
  onTrain,
  training,
  professorMode,
}: {
  stats: DashboardStats | null;
  modelStats: AllModelStats | null;
  onTrain: () => void;
  training: boolean;
  professorMode: boolean;
}) {
  if (!stats)
    return (
      <div className="grid grid-cols-4 gap-4">
        {[1, 2, 3, 4].map((i) => (
          <div key={i} className="skeleton h-28" />
        ))}
      </div>
    );

  const modelData = modelStats
    ? [
        {
          name: "Isolation Forest",
          precision:
            +(modelStats.isolation_forest?.precision * 100).toFixed(1) || 0,
          recall: +(modelStats.isolation_forest?.recall * 100).toFixed(1) || 0,
          f1: +(modelStats.isolation_forest?.f1 * 100).toFixed(1) || 0,
          auc: +(modelStats.isolation_forest?.auc_roc * 100).toFixed(1) || 0,
        },
        {
          name: "Autoencoder",
          precision: +(modelStats.autoencoder?.precision * 100).toFixed(1) || 0,
          recall: +(modelStats.autoencoder?.recall * 100).toFixed(1) || 0,
          f1: +(modelStats.autoencoder?.f1 * 100).toFixed(1) || 0,
          auc: +(modelStats.autoencoder?.auc_roc * 100).toFixed(1) || 0,
        },
        {
          name: "XGBoost",
          precision: +(modelStats.xgboost?.precision * 100).toFixed(1) || 0,
          recall: +(modelStats.xgboost?.recall * 100).toFixed(1) || 0,
          f1: +(modelStats.xgboost?.f1 * 100).toFixed(1) || 0,
          auc: +(modelStats.xgboost?.auc_roc * 100).toFixed(1) || 0,
        },
      ]
    : [];

  return (
    <div className="space-y-6 animate-fade-in">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">Dashboard</h1>
          <p className="text-gray-400 text-sm mt-1">
            Real-time fraud monitoring overview
          </p>
        </div>
        <div className="flex gap-3">
          <button
            className="btn-primary flex items-center gap-2"
            onClick={onTrain}
            disabled={training}
          >
            {training ? (
              <RefreshCw size={16} className="animate-spin" />
            ) : (
              <Play size={16} />
            )}
            {training
              ? "Training..."
              : stats.models_trained
                ? "Retrain Models"
                : "Train Models"}
          </button>
        </div>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          title="Total Predictions"
          value={stats.total_predictions.toLocaleString()}
          icon={Activity}
          color={COLORS.teal}
        />
        <StatCard
          title="Fraud Detected"
          value={stats.total_fraud.toLocaleString()}
          icon={AlertTriangle}
          color={COLORS.danger}
          sub={`${stats.fraud_rate.toFixed(2)}% fraud rate`}
        />
        <StatCard
          title="Avg Confidence"
          value={`${(stats.avg_fraud_confidence * 100).toFixed(1)}%`}
          icon={Zap}
          color={COLORS.blue}
        />
        <StatCard
          title="Pending Alerts"
          value={stats.pending_alerts}
          icon={Bell}
          color={COLORS.warning}
        />
      </div>
      {modelStats?.models_trained && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="glass-card p-6">
            <h3 className="text-sm font-semibold text-gray-300 mb-4">
              Model Performance Comparison
            </h3>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={modelData} barGap={8}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e3150" />
                <XAxis
                  dataKey="name"
                  tick={{ fill: "#94a3b8", fontSize: 11 }}
                />
                <YAxis
                  tick={{ fill: "#94a3b8", fontSize: 11 }}
                  domain={[0, 100]}
                />
                <Tooltip
                  contentStyle={{
                    background: "#111d35",
                    border: "1px solid #2a3f5f",
                    borderRadius: 12,
                    color: "#e0e6ed",
                  }}
                />
                <Bar
                  dataKey="precision"
                  fill={COLORS.teal}
                  radius={[4, 4, 0, 0]}
                  name="Precision %"
                />
                <Bar
                  dataKey="recall"
                  fill={COLORS.blue}
                  radius={[4, 4, 0, 0]}
                  name="Recall %"
                />
                <Bar
                  dataKey="f1"
                  fill={COLORS.warning}
                  radius={[4, 4, 0, 0]}
                  name="F1 %"
                />
                {!professorMode && (
                  <Bar
                    dataKey="auc"
                    fill={COLORS.purple}
                    radius={[4, 4, 0, 0]}
                    name="AUC-ROC %"
                  />
                )}
                <Legend />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div className="glass-card p-6">
            <h3 className="text-sm font-semibold text-gray-300 mb-4">
              Model Status
            </h3>
            <div className="space-y-4">
              {["Isolation Forest", "Autoencoder", "XGBoost"].map((name, i) => {
                const key = name
                  .toLowerCase()
                  .replace(" ", "_") as keyof AllModelStats;
                const m = modelStats[key] as any;
                if (!m) return null;
                return (
                  <div
                    key={name}
                    className="flex items-center justify-between p-3 rounded-xl bg-white/[0.03] border border-white/5"
                  >
                    <div className="flex items-center gap-3">
                      <div className="w-2.5 h-2.5 rounded-full bg-green-400 pulse-ring relative" />
                      <span className="font-medium text-sm">{name}</span>
                    </div>
                    <div className="flex items-center gap-4 text-xs text-gray-400">
                      <span>
                        F1:{" "}
                        <b className="text-teal-400">
                          {(m.f1 * 100).toFixed(1)}%
                        </b>
                      </span>
                      <span>
                        AUC:{" "}
                        <b className="text-blue-400">
                          {(m.auc_roc * 100).toFixed(1)}%
                        </b>
                      </span>
                      <span className="badge-legit">Ready</span>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// ─── Detection Page ──────────────────────────────────────
function DetectionPage({
  onPredict,
  predResult,
  loading,
  professorMode,
}: {
  onPredict: (f: File) => void;
  predResult: PredictionResponse | null;
  loading: boolean;
  professorMode: boolean;
}) {
  const fileRef = useRef<HTMLInputElement>(null);
  const [dragOver, setDragOver] = useState(false);

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    if (e.dataTransfer.files[0]) onPredict(e.dataTransfer.files[0]);
  };

  return (
    <div className="space-y-6 animate-fade-in">
      <h1 className="text-2xl font-bold">Fraud Detection</h1>
      <div
        className={`glass-card p-8 border-2 border-dashed transition-colors duration-300 text-center cursor-pointer
        ${dragOver ? "border-teal-400 bg-teal-400/5" : "border-white/10"}`}
        onDragOver={(e) => {
          e.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={handleDrop}
        onClick={() => fileRef.current?.click()}
      >
        <input
          ref={fileRef}
          type="file"
          accept=".csv"
          className="hidden"
          onChange={(e) => e.target.files?.[0] && onPredict(e.target.files[0])}
        />
        <Upload size={40} className="mx-auto mb-4 text-teal-400" />
        <p className="text-lg font-semibold">
          Drop CSV file here or click to upload
        </p>
        <p className="text-sm text-gray-400 mt-2">
          Supports creditcard.csv format (V1-V28, Amount, Class)
        </p>
        {loading && (
          <div className="mt-4">
            <RefreshCw
              size={24}
              className="mx-auto animate-spin text-teal-400"
            />
            <p className="text-sm text-teal-400 mt-2">Running predictions...</p>
          </div>
        )}
      </div>
      {predResult && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="space-y-6"
        >
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <StatCard
              title="Total Analyzed"
              value={predResult.summary.total_transactions.toLocaleString()}
              icon={FileText}
              color={COLORS.teal}
            />
            <StatCard
              title="Fraud Flagged"
              value={predResult.summary.flagged_fraud}
              icon={AlertTriangle}
              color={COLORS.danger}
            />
            <StatCard
              title="Avg Score"
              value={predResult.summary.avg_ensemble_score.toFixed(4)}
              icon={Activity}
              color={COLORS.blue}
            />
            <StatCard
              title="Drift Detected"
              value={predResult.summary.drift_detected ? "Yes" : "No"}
              icon={TrendingUp}
              color={
                predResult.summary.drift_detected
                  ? COLORS.warning
                  : COLORS.success
              }
            />
          </div>
          {predResult.summary.metrics &&
            Object.keys(predResult.summary.metrics).length > 0 &&
            !professorMode && (
              <div className="glass-card p-6">
                <h3 className="text-sm font-semibold text-gray-300 mb-4">
                  Model Metrics on this Dataset
                </h3>
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="text-gray-400 border-b border-white/5">
                        <th className="text-left py-2 px-3">Model</th>
                        <th className="text-right py-2 px-3">Precision</th>
                        <th className="text-right py-2 px-3">Recall</th>
                        <th className="text-right py-2 px-3">F1</th>
                        <th className="text-right py-2 px-3">AUC-ROC</th>
                      </tr>
                    </thead>
                    <tbody>
                      {Object.entries(predResult.summary.metrics).map(
                        ([name, m]) => (
                          <tr
                            key={name}
                            className="table-row border-b border-white/5"
                          >
                            <td className="py-2 px-3 font-medium capitalize">
                              {name.replace(/_/g, " ")}
                            </td>
                            <td className="py-2 px-3 text-right">
                              {(m.precision * 100).toFixed(1)}%
                            </td>
                            <td className="py-2 px-3 text-right">
                              {(m.recall * 100).toFixed(1)}%
                            </td>
                            <td className="py-2 px-3 text-right text-teal-400 font-bold">
                              {(m.f1 * 100).toFixed(1)}%
                            </td>
                            <td className="py-2 px-3 text-right">
                              {(m.auc_roc * 100).toFixed(1)}%
                            </td>
                          </tr>
                        ),
                      )}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          {/* SHAP Feature Importance */}
          {predResult.shap_global &&
            Object.keys(predResult.shap_global).length > 0 && (
              <div className="glass-card p-6">
                <h3 className="text-sm font-semibold text-gray-300 mb-4">
                  {professorMode
                    ? "Key Risk Factors"
                    : "SHAP Feature Importance"}
                </h3>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart
                    data={Object.entries(predResult.shap_global)
                      .slice(0, 12)
                      .map(([k, v]) => ({
                        name: k,
                        importance: +v.toFixed(4),
                      }))}
                    layout="vertical"
                  >
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e3150" />
                    <XAxis
                      type="number"
                      tick={{ fill: "#94a3b8", fontSize: 11 }}
                    />
                    <YAxis
                      dataKey="name"
                      type="category"
                      width={60}
                      tick={{ fill: "#94a3b8", fontSize: 11 }}
                    />
                    <Tooltip
                      contentStyle={{
                        background: "#111d35",
                        border: "1px solid #2a3f5f",
                        borderRadius: 12,
                        color: "#e0e6ed",
                      }}
                    />
                    <Bar
                      dataKey="importance"
                      fill={COLORS.teal}
                      radius={[0, 4, 4, 0]}
                    />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            )}
          {/* Top flagged transactions */}
          <div className="glass-card p-6">
            <h3 className="text-sm font-semibold text-gray-300 mb-4">
              Top Flagged Transactions
            </h3>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-gray-400 border-b border-white/5">
                    <th className="text-left py-2 px-3">ID</th>
                    <th className="text-right py-2 px-3">Ensemble</th>
                    <th className="text-right py-2 px-3">IF</th>
                    <th className="text-right py-2 px-3">AE</th>
                    <th className="text-right py-2 px-3">XGB</th>
                    <th className="text-center py-2 px-3">Verdict</th>
                    {!professorMode && (
                      <th className="text-left py-2 px-3">Explanation</th>
                    )}
                  </tr>
                </thead>
                <tbody>
                  {[...predResult.transactions]
                    .sort((a, b) => b.ensemble_score - a.ensemble_score)
                    .slice(0, 15)
                    .map((t) => (
                      <tr
                        key={t.transaction_id}
                        className="table-row border-b border-white/5"
                      >
                        <td className="py-2 px-3 font-mono text-xs">
                          {t.transaction_id}
                        </td>
                        <td
                          className="py-2 px-3 text-right font-bold"
                          style={{
                            color:
                              t.ensemble_score > 0.7
                                ? COLORS.danger
                                : t.ensemble_score > 0.3
                                  ? COLORS.warning
                                  : COLORS.success,
                          }}
                        >
                          {t.ensemble_score.toFixed(4)}
                        </td>
                        <td className="py-2 px-3 text-right text-gray-400">
                          {t.isolation_forest_score.toFixed(3)}
                        </td>
                        <td className="py-2 px-3 text-right text-gray-400">
                          {t.autoencoder_score.toFixed(3)}
                        </td>
                        <td className="py-2 px-3 text-right text-gray-400">
                          {t.xgboost_score.toFixed(3)}
                        </td>
                        <td className="py-2 px-3 text-center">
                          {t.is_fraud ? (
                            <span className="badge-fraud">FRAUD</span>
                          ) : (
                            <span className="badge-legit">OK</span>
                          )}
                        </td>
                        {!professorMode && (
                          <td className="py-2 px-3 text-xs text-gray-400 max-w-xs truncate">
                            {t.explanation?.explanation || "—"}
                          </td>
                        )}
                      </tr>
                    ))}
                </tbody>
              </table>
            </div>
          </div>
        </motion.div>
      )}
    </div>
  );
}

// ─── Transaction Explorer ────────────────────────────────
function ExplorerPage() {
  const [txns, setTxns] = useState<Transaction[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [search, setSearch] = useState("");
  const [debouncedSearch, setDebouncedSearch] = useState("");
  const [fraudOnly, setFraudOnly] = useState(false);
  const [loading, setLoading] = useState(false);
  const [feedbackMap, setFeedbackMap] = useState<Record<string, number>>({});
  const [feedbackSaving, setFeedbackSaving] = useState<string | null>(null);
  const [feedbackToast, setFeedbackToast] = useState("");

  // Debounce search input
  useEffect(() => {
    const timer = setTimeout(() => setDebouncedSearch(search), 300);
    return () => clearTimeout(timer);
  }, [search]);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const res = await api.fetchTransactions({
        page,
        per_page: 30,
        fraud_only: fraudOnly,
        search: debouncedSearch,
      });
      setTxns(res.items || []);
      setTotal(res.total || 0);
      // Populate feedbackMap from existing labels
      const existing: Record<string, number> = {};
      for (const t of res.items || []) {
        if (t.feedback_label !== null && t.feedback_label !== undefined) {
          existing[t.transaction_id] = t.feedback_label;
        }
      }
      setFeedbackMap((prev) => ({ ...prev, ...existing }));
    } catch {
      /* ignore */
    }
    setLoading(false);
  }, [page, debouncedSearch, fraudOnly]);

  useEffect(() => {
    load();
  }, [load]);

  const handleFeedback = async (transactionId: string, label: number) => {
    setFeedbackSaving(transactionId);
    try {
      await api.submitFeedback([transactionId], [label]);
      setFeedbackMap((prev) => ({ ...prev, [transactionId]: label }));
      const labelName = label === 1 ? "Fraud" : "Legit";
      setFeedbackToast(`Labeled ${transactionId} as ${labelName}`);
      setTimeout(() => setFeedbackToast(""), 2500);
    } catch {
      setFeedbackToast("Failed to submit feedback");
      setTimeout(() => setFeedbackToast(""), 3000);
    }
    setFeedbackSaving(null);
  };

  const feedbackCount = Object.keys(feedbackMap).length;

  return (
    <div className="space-y-6 animate-fade-in">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <h1 className="text-2xl font-bold">Transaction Explorer</h1>
          {feedbackCount > 0 && (
            <span className="px-2.5 py-1 rounded-lg text-xs font-medium bg-teal-400/15 text-teal-400 border border-teal-400/20">
              {feedbackCount} labeled
            </span>
          )}
        </div>
        <button
          className="btn-secondary flex items-center gap-2"
          onClick={() => api.exportCSV()}
        >
          <Download size={16} /> Export CSV
        </button>
      </div>
      {feedbackToast && (
        <motion.div
          className="glass-card p-3 text-sm text-teal-400 flex items-center gap-2"
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0 }}
        >
          <CheckCircle size={14} /> {feedbackToast}
        </motion.div>
      )}
      <div className="flex gap-3">
        <div className="relative flex-1">
          <Search
            size={16}
            className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-500"
          />
          <input
            className="input-field pl-10"
            placeholder="Search transaction ID..."
            value={search}
            onChange={(e) => {
              setSearch(e.target.value);
              setPage(1);
            }}
          />
        </div>
        <button
          className={`btn-secondary ${fraudOnly ? "!bg-red-500/20 !text-red-400 !border-red-500/30" : ""}`}
          onClick={() => {
            setFraudOnly(!fraudOnly);
            setPage(1);
          }}
        >
          {fraudOnly ? <XCircle size={16} /> : <AlertTriangle size={16} />}
          <span className="ml-2">{fraudOnly ? "Show All" : "Fraud Only"}</span>
        </button>
      </div>
      <div className="glass-card overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-gray-400 text-xs uppercase border-b border-white/5">
                <th className="text-left py-3 px-4">Transaction ID</th>
                <th className="text-left py-3 px-4">Timestamp</th>
                <th className="text-right py-3 px-4">Ensemble Score</th>
                <th className="text-center py-3 px-4">Status</th>
                <th className="text-center py-3 px-4">Feedback</th>
              </tr>
            </thead>
            <tbody>
              {loading ? (
                [1, 2, 3, 4, 5].map((i) => (
                  <tr key={i}>
                    <td colSpan={5} className="py-3 px-4">
                      <div className="skeleton h-6" />
                    </td>
                  </tr>
                ))
              ) : txns.length === 0 ? (
                <tr>
                  <td colSpan={5} className="py-12 text-center text-gray-500">
                    No transactions found. Run predictions first.
                  </td>
                </tr>
              ) : (
                txns.map((t) => {
                  const currentLabel = feedbackMap[t.transaction_id] ?? t.feedback_label;
                  const isSaving = feedbackSaving === t.transaction_id;
                  return (
                    <tr
                      key={t.transaction_id}
                      className="table-row border-b border-white/5"
                    >
                      <td className="py-3 px-4 font-mono text-xs">
                        {t.transaction_id}
                      </td>
                      <td className="py-3 px-4 text-gray-400 text-xs">
                        {t.timestamp}
                      </td>
                      <td
                        className="py-3 px-4 text-right font-bold"
                        style={{
                          color:
                            t.ensemble_score > 0.7
                              ? COLORS.danger
                              : t.ensemble_score > 0.3
                                ? COLORS.warning
                                : COLORS.success,
                        }}
                      >
                        {t.ensemble_score.toFixed(4)}
                      </td>
                      <td className="py-3 px-4 text-center">
                        {t.is_fraud ? (
                          <span className="badge-fraud">FRAUD</span>
                        ) : (
                          <span className="badge-legit">LEGIT</span>
                        )}
                      </td>
                      <td className="py-3 px-4 text-center">
                        {currentLabel !== null && currentLabel !== undefined ? (
                          <span
                            className={`inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-medium ${
                              currentLabel === 1
                                ? "bg-red-500/15 text-red-400 border border-red-500/20"
                                : "bg-green-500/15 text-green-400 border border-green-500/20"
                            }`}
                          >
                            <CheckCircle size={10} />
                            {currentLabel === 1 ? "Fraud" : "Legit"}
                          </span>
                        ) : isSaving ? (
                          <RefreshCw size={12} className="animate-spin mx-auto text-gray-400" />
                        ) : (
                          <div className="flex items-center justify-center gap-1">
                            <button
                              className="px-2 py-0.5 rounded text-[10px] font-semibold bg-red-500/10 text-red-400 border border-red-500/20 hover:bg-red-500/25 transition-colors"
                              onClick={() => handleFeedback(t.transaction_id, 1)}
                              title="Label as Fraud"
                            >
                              Fraud
                            </button>
                            <button
                              className="px-2 py-0.5 rounded text-[10px] font-semibold bg-green-500/10 text-green-400 border border-green-500/20 hover:bg-green-500/25 transition-colors"
                              onClick={() => handleFeedback(t.transaction_id, 0)}
                              title="Label as Legit"
                            >
                              Legit
                            </button>
                          </div>
                        )}
                      </td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </div>
        {total > 30 && (
          <div className="flex items-center justify-between px-4 py-3 border-t border-white/5">
            <span className="text-xs text-gray-400">
              {total.toLocaleString()} transactions
            </span>
            <div className="flex gap-2">
              <button
                className="btn-secondary !px-3 !py-1 text-xs"
                disabled={page <= 1}
                onClick={() => setPage((p) => p - 1)}
              >
                Previous
              </button>
              <span className="text-xs text-gray-400 py-1">Page {page}</span>
              <button
                className="btn-secondary !px-3 !py-1 text-xs"
                disabled={page * 30 >= total}
                onClick={() => setPage((p) => p + 1)}
              >
                Next
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

// ─── Model Comparison ────────────────────────────────────
function ModelComparisonPage({
  modelStats,
}: {
  modelStats: AllModelStats | null;
}) {
  if (!modelStats?.models_trained)
    return (
      <div className="text-center py-20 text-gray-400">
        <BarChart3 size={48} className="mx-auto mb-4 opacity-30" />
        <p className="text-lg">No model data available. Train models first.</p>
      </div>
    );

  const models = ["isolation_forest", "autoencoder", "xgboost"] as const;
  const names = ["Isolation Forest", "Autoencoder", "XGBoost"];
  const colors = [COLORS.teal, COLORS.blue, COLORS.warning];

  const radarData = ["precision", "recall", "f1", "auc_roc", "accuracy"].map(
    (metric) => {
      const item: any = {
        metric:
          metric === "auc_roc"
            ? "AUC-ROC"
            : metric.charAt(0).toUpperCase() + metric.slice(1),
      };
      models.forEach((m, i) => {
        const ms = modelStats[m] as any;
        item[names[i]] = ms ? +(ms[metric] * 100).toFixed(1) : 0;
      });
      return item;
    },
  );

  return (
    <div className="space-y-6 animate-fade-in">
      <h1 className="text-2xl font-bold">Model Comparison</h1>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {models.map((m, i) => {
          const ms = modelStats[m] as any;
          if (!ms) return null;
          return (
            <motion.div
              key={m}
              className="glass-card p-6"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.1 }}
            >
              <div className="flex items-center gap-3 mb-4">
                <div
                  className="w-3 h-3 rounded-full"
                  style={{ background: colors[i] }}
                />
                <h3 className="font-semibold">{names[i]}</h3>
                <span className="ml-auto badge-legit text-xs">
                  v{modelStats.versions?.[m] || 1}
                </span>
              </div>
              <div className="space-y-3">
                {(["precision", "recall", "f1", "auc_roc"] as const).map(
                  (metric) => (
                    <div key={metric}>
                      <div className="flex justify-between text-xs mb-1">
                        <span className="text-gray-400 capitalize">
                          {metric === "auc_roc" ? "AUC-ROC" : metric}
                        </span>
                        <span
                          className="font-bold"
                          style={{ color: colors[i] }}
                        >
                          {(ms[metric] * 100).toFixed(1)}%
                        </span>
                      </div>
                      <div className="h-1.5 bg-white/5 rounded-full overflow-hidden">
                        <motion.div
                          className="h-full rounded-full"
                          style={{ background: colors[i] }}
                          initial={{ width: 0 }}
                          animate={{ width: `${ms[metric] * 100}%` }}
                          transition={{ duration: 1, delay: i * 0.1 }}
                        />
                      </div>
                    </div>
                  ),
                )}
              </div>
            </motion.div>
          );
        })}
      </div>
      <div className="glass-card p-6">
        <h3 className="text-sm font-semibold text-gray-300 mb-4">
          Radar Comparison
        </h3>
        <ResponsiveContainer width="100%" height={350}>
          <RadarChart data={radarData}>
            <PolarGrid stroke="#1e3150" />
            <PolarAngleAxis
              dataKey="metric"
              tick={{ fill: "#94a3b8", fontSize: 11 }}
            />
            <PolarRadiusAxis
              angle={30}
              domain={[0, 100]}
              tick={{ fill: "#64748b", fontSize: 10 }}
            />
            {names.map((n, i) => (
              <Radar
                key={n}
                name={n}
                dataKey={n}
                stroke={colors[i]}
                fill={colors[i]}
                fillOpacity={0.15}
                strokeWidth={2}
              />
            ))}
            <Legend />
            <Tooltip
              contentStyle={{
                background: "#111d35",
                border: "1px solid #2a3f5f",
                borderRadius: 12,
                color: "#e0e6ed",
              }}
            />
          </RadarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

// ─── Adaptive Learning ───────────────────────────────────
function LearningPage({ modelStats }: { modelStats: AllModelStats | null }) {
  const [history, setHistory] = useState<any[]>([]);
  const [retraining, setRetraining] = useState(false);
  const [message, setMessage] = useState("");

  useEffect(() => {
    let isMounted = true;
    api
      .fetchTrainingHistory()
      .then((data) => {
        if (isMounted) setHistory(data);
      })
      .catch((err) => {
        console.error("Failed to fetch training history:", err);
      });
    return () => {
      isMounted = false;
    };
  }, []);

  const handleRetrain = async () => {
    setRetraining(true);
    setMessage("");
    try {
      const res = await api.retrain();
      setMessage(`Retrained with ${res.feedback_samples} feedback samples ✅`);
      api.fetchTrainingHistory().then(setHistory);
    } catch (e: any) {
      setMessage(e.message || "Retrain failed");
    }
    setRetraining(false);
  };

  return (
    <div className="space-y-6 animate-fade-in">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">Adaptive Learning</h1>
          <p className="text-gray-400 text-sm mt-1">
            Models improve over time with your feedback
          </p>
        </div>
        <button
          className="btn-primary flex items-center gap-2"
          onClick={handleRetrain}
          disabled={retraining}
        >
          {retraining ? (
            <RefreshCw size={16} className="animate-spin" />
          ) : (
            <Brain size={16} />
          )}
          {retraining ? "Retraining..." : "Retrain with Feedback"}
        </button>
      </div>
      {message && (
        <div className="glass-card p-4 text-sm text-teal-400">{message}</div>
      )}
      {history.length > 0 ? (
        <div className="space-y-4">
          <div className="glass-card p-6">
            <h3 className="text-sm font-semibold text-gray-300 mb-4">
              Learning Progress
            </h3>
            <ResponsiveContainer width="100%" height={250}>
              <LineChart
                data={history.map((h, i) => ({
                  cycle: i + 1,
                  improvement: +(h.improvement * 100).toFixed(2),
                  samples: h.num_feedback_samples,
                }))}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#1e3150" />
                <XAxis
                  dataKey="cycle"
                  tick={{ fill: "#94a3b8", fontSize: 11 }}
                  label={{
                    value: "Retraining Cycle",
                    fill: "#64748b",
                    position: "bottom",
                  }}
                />
                <YAxis tick={{ fill: "#94a3b8", fontSize: 11 }} />
                <Tooltip
                  contentStyle={{
                    background: "#111d35",
                    border: "1px solid #2a3f5f",
                    borderRadius: 12,
                    color: "#e0e6ed",
                  }}
                />
                <Line
                  type="monotone"
                  dataKey="improvement"
                  stroke={COLORS.teal}
                  strokeWidth={2}
                  dot={{ fill: COLORS.teal }}
                  name="F1 Improvement %"
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
          <div className="glass-card p-6">
            <h3 className="text-sm font-semibold text-gray-300 mb-4">
              Training History
            </h3>
            <div className="space-y-3">
              {history.slice(0, 10).map((h, i) => (
                <div
                  key={i}
                  className="flex items-center justify-between p-3 rounded-xl bg-white/[0.03] border border-white/5"
                >
                  <div>
                    <span className="font-medium text-sm">{h.model_name}</span>
                    <span className="text-xs text-gray-500 ml-3">
                      {h.timestamp}
                    </span>
                  </div>
                  <div className="flex items-center gap-4 text-xs">
                    <span className="text-gray-400">
                      {h.num_feedback_samples} samples
                    </span>
                    <span
                      className={
                        h.improvement >= 0 ? "text-green-400" : "text-red-400"
                      }
                    >
                      {h.improvement >= 0 ? "+" : ""}
                      {(h.improvement * 100).toFixed(2)}% F1
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      ) : (
        <div className="glass-card p-12 text-center text-gray-400">
          <Brain size={48} className="mx-auto mb-4 opacity-30" />
          <p>
            No training history yet. Submit feedback on transactions, then click
            "Retrain with Feedback" to see adaptive learning in action.
          </p>
        </div>
      )}
    </div>
  );
}

// ─── Datasets Page ───────────────────────────────────────
function DatasetsPage() {
  const [datasets, setDatasets] = useState<any[]>([]);
  const [uploading, setUploading] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    let isMounted = true;
    api
      .fetchDatasets()
      .then((data) => {
        if (isMounted) setDatasets(data);
      })
      .catch((err) => {
        console.error("Failed to fetch datasets:", err);
      });
    return () => {
      isMounted = false;
    };
  }, []);

  const handleUpload = async (file: File) => {
    setUploading(true);
    try {
      await api.uploadDataset(file);
      const ds = await api.fetchDatasets();
      setDatasets(ds);
    } catch (err) {
      console.error("Failed to upload dataset:", err);
    }
    setUploading(false);
  };

  return (
    <div className="space-y-6 animate-fade-in">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold">Dataset Manager</h1>
        <div>
          <input
            ref={fileRef}
            type="file"
            accept=".csv"
            className="hidden"
            onChange={(e) =>
              e.target.files?.[0] && handleUpload(e.target.files[0])
            }
          />
          <button
            className="btn-primary flex items-center gap-2"
            onClick={() => fileRef.current?.click()}
            disabled={uploading}
          >
            <Upload size={16} /> {uploading ? "Uploading..." : "Upload Dataset"}
          </button>
        </div>
      </div>
      {datasets.length > 0 ? (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {datasets.map((ds) => (
            <div
              key={ds.id}
              className={`glass-card p-6 ${ds.is_active ? "border-teal-400/30" : ""}`}
            >
              <div className="flex items-center justify-between mb-3">
                <h3 className="font-semibold">{ds.name}</h3>
                {ds.is_active ? (
                  <span className="badge-legit">Active</span>
                ) : (
                  <span className="text-xs text-gray-500">Inactive</span>
                )}
              </div>
              <div className="grid grid-cols-3 gap-3 text-sm">
                <div>
                  <p className="text-gray-500 text-xs">Rows</p>
                  <p className="font-bold">{ds.num_rows?.toLocaleString()}</p>
                </div>
                <div>
                  <p className="text-gray-500 text-xs">Features</p>
                  <p className="font-bold">{ds.num_features}</p>
                </div>
                <div>
                  <p className="text-gray-500 text-xs">Fraud Rate</p>
                  <p className="font-bold text-red-400">
                    {(ds.fraud_ratio * 100).toFixed(3)}%
                  </p>
                </div>
              </div>
              <p className="text-xs text-gray-500 mt-3">
                Uploaded: {ds.upload_time}
              </p>
            </div>
          ))}
        </div>
      ) : (
        <div className="glass-card p-12 text-center text-gray-400">
          <Database size={48} className="mx-auto mb-4 opacity-30" />
          <p>No datasets uploaded yet. Upload a CSV to get started.</p>
        </div>
      )}
    </div>
  );
}

// ─── Alerts Page ─────────────────────────────────────────
function AlertsPage() {
  const [alerts, setAlerts] = useState<Alert[]>([]);
  useEffect(() => {
    let isMounted = true;
    api
      .fetchAlerts()
      .then((data) => {
        if (isMounted) setAlerts(data);
      })
      .catch((err) => {
        console.error("Failed to fetch alerts:", err);
      });
    return () => {
      isMounted = false;
    };
  }, []);

  const handleAck = async (id: number) => {
    await api.acknowledgeAlert(id);
    setAlerts((prev) =>
      prev.map((a) => (a.id === id ? { ...a, acknowledged: 1 } : a)),
    );
  };

  return (
    <div className="space-y-6 animate-fade-in">
      <h1 className="text-2xl font-bold">Fraud Alerts</h1>
      {alerts.length > 0 ? (
        <div className="space-y-3">
          {alerts.map((a) => (
            <motion.div
              key={a.id}
              className={`glass-card p-4 flex items-center justify-between ${a.acknowledged ? "opacity-50" : ""}`}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: a.acknowledged ? 0.5 : 1, x: 0 }}
            >
              <div className="flex items-center gap-4">
                <div
                  className={`w-10 h-10 rounded-xl flex items-center justify-center ${a.acknowledged ? "bg-gray-600/20" : "bg-red-500/20"}`}
                >
                  {a.acknowledged ? (
                    <CheckCircle size={20} className="text-gray-400" />
                  ) : (
                    <AlertTriangle size={20} className="text-red-400" />
                  )}
                </div>
                <div>
                  <p className="font-medium text-sm">{a.transaction_id}</p>
                  <p className="text-xs text-gray-400">{a.timestamp}</p>
                </div>
              </div>
              <div className="flex items-center gap-4">
                <span className="font-bold text-red-400">
                  {(a.fraud_score * 100).toFixed(1)}%
                </span>
                {!a.acknowledged && (
                  <button
                    className="btn-secondary !px-3 !py-1 text-xs"
                    onClick={() => handleAck(a.id)}
                  >
                    Acknowledge
                  </button>
                )}
              </div>
            </motion.div>
          ))}
        </div>
      ) : (
        <div className="glass-card p-12 text-center text-gray-400">
          <Bell size={48} className="mx-auto mb-4 opacity-30" />
          <p>
            No alerts yet. Alerts are generated when high-risk transactions are
            detected.
          </p>
        </div>
      )}
    </div>
  );
}

// ─── Main App ────────────────────────────────────────────
export default function App() {
  const [page, setPage] = useState("dashboard");
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [professorMode, setProfessorMode] = useState(false);
  const [stats, setStats] = useState<DashboardStats | null>(null);
  const [modelStats, setModelStats] = useState<AllModelStats | null>(null);
  const [predResult, setPredResult] = useState<PredictionResponse | null>(null);
  const [training, setTraining] = useState(false);
  const [predicting, setPredicting] = useState(false);
  const [alertCount, setAlertCount] = useState(0);
  const [toast, setToast] = useState("");

  const refresh = useCallback(async () => {
    try {
      const d = await api.fetchDashboard();
      setStats(d);
      setAlertCount(d.pending_alerts);
      if (d.models_trained) {
        const m = await api.fetchModelStats();
        setModelStats(m);
      }
    } catch {
      /* ignore */
    }
  }, []);

  useEffect(() => {
    refresh();
    const t = setInterval(refresh, 15000);
    return () => clearInterval(t);
  }, [refresh]);

  const handleTrain = async () => {
    setTraining(true);
    setToast("Training models... This may take a few minutes.");
    try {
      await api.trainModels();
      setToast("Models trained successfully! ✅");
      await refresh();
    } catch (e: any) {
      setToast(`Training failed: ${e.message}`);
    }
    setTraining(false);
    setTimeout(() => setToast(""), 5000);
  };

  const handlePredict = async (file: File) => {
    setPredicting(true);
    try {
      const res = await api.uploadAndPredict(file);
      setPredResult(res);
      await refresh();
      setToast(
        `Analyzed ${res.summary.total_transactions.toLocaleString()} transactions. ${res.summary.flagged_fraud} flagged as fraud.`,
      );
    } catch (e: any) {
      setToast(`Prediction failed: ${e.message}`);
    }
    setPredicting(false);
    setTimeout(() => setToast(""), 5000);
  };

  const renderPage = () => {
    switch (page) {
      case "dashboard":
        return (
          <DashboardPage
            stats={stats}
            modelStats={modelStats}
            onTrain={handleTrain}
            training={training}
            professorMode={professorMode}
          />
        );
      case "detection":
        return (
          <DetectionPage
            onPredict={handlePredict}
            predResult={predResult}
            loading={predicting}
            professorMode={professorMode}
          />
        );
      case "explorer":
        return <ExplorerPage />;
      case "models":
        return <ModelComparisonPage modelStats={modelStats} />;
      case "learning":
        return <LearningPage modelStats={modelStats} />;
      case "datasets":
        return <DatasetsPage />;
      case "alerts":
        return <AlertsPage />;
      default:
        return null;
    }
  };

  return (
    <div className="flex min-h-screen">
      <Sidebar
        active={page}
        onNav={setPage}
        collapsed={sidebarCollapsed}
        onToggle={() => setSidebarCollapsed(!sidebarCollapsed)}
        alertCount={alertCount}
      />
      <main
        className={`flex-1 transition-all duration-300 ${sidebarCollapsed ? "ml-[72px]" : "ml-60"}`}
      >
        {/* Top bar */}
        <header
          className="sticky top-0 z-20 h-14 flex items-center justify-between px-6 border-b border-white/5"
          style={{
            background: "rgba(10,22,40,0.85)",
            backdropFilter: "blur(12px)",
          }}
        >
          <div className="flex items-center gap-2 text-sm text-gray-400">
            <span className="text-gray-200 font-medium capitalize">{page}</span>
            <ChevronRight size={14} />
            <span>Overview</span>
          </div>
          <div className="flex items-center gap-4">
            <button
              className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${professorMode ? "bg-teal-400/20 text-teal-400" : "bg-white/5 text-gray-400 hover:text-gray-200"}`}
              onClick={() => setProfessorMode(!professorMode)}
            >
              {professorMode ? <Eye size={14} /> : <EyeOff size={14} />}
              Professor Mode
            </button>
            <div className="flex items-center gap-2">
              <div
                className={`w-2 h-2 rounded-full ${stats?.models_trained ? "bg-green-400" : "bg-yellow-400"}`}
              />
              <span className="text-xs text-gray-400">
                {stats?.models_trained ? "Models Ready" : "Not Trained"}
              </span>
            </div>
          </div>
        </header>
        {/* Content */}
        <div className="p-6">{renderPage()}</div>
      </main>
      {/* Toast */}
      <AnimatePresence>
        {toast && (
          <motion.div
            className="fixed bottom-6 right-6 z-50 glass-card px-5 py-3 text-sm max-w-md"
            style={{ borderColor: "rgba(0,212,170,0.3)" }}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 20 }}
          >
            {toast}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
