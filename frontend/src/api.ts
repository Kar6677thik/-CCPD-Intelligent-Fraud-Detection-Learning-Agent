const API = '';

export interface DashboardStats {
  total_predictions: number;
  total_fraud: number;
  fraud_rate: number;
  avg_fraud_confidence: number;
  pending_alerts: number;
  models_trained: boolean;
  training_in_progress: boolean;
  active_models: ModelVersion[];
}

export interface ModelVersion {
  id: number;
  model_name: string;
  version: number;
  timestamp: string;
  metrics: ModelMetrics;
  training_samples: number;
  is_active: number;
}

export interface ModelMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  auc_roc: number;
  avg_precision: number;
  confusion_matrix: number[][];
}

export interface Transaction {
  transaction_id: string;
  timestamp: string;
  features: Record<string, number>;
  isolation_forest_score: number;
  autoencoder_score: number;
  xgboost_score: number;
  ensemble_score: number;
  is_fraud: number;
  confidence: number;
  explanation: {
    top_features?: Record<string, number>;
    explanation?: string;
    base_value?: number;
  };
  feedback_label: number | null;
}

export interface PredictionResponse {
  summary: {
    batch_id: string;
    total_transactions: number;
    flagged_fraud: number;
    avg_ensemble_score: number;
    max_ensemble_score: number;
    model_scores: Record<string, { flagged: number; avg_score: number }>;
    drift_detected: boolean;
    drift_score: number;
    uncertain_transaction_count: number;
    output_folder: string | null;
    metrics: Record<string, ModelMetrics>;
  };
  transactions: Transaction[];
  shap_global: Record<string, number>;
  drift: {
    overall_drift: boolean;
    drift_score: number;
    num_drifted_features: number;
    total_features: number;
    features: Record<string, { ks_statistic: number; p_value: number; is_drifted: boolean }>;
  };
  uncertain_indices: number[];
}

export interface AllModelStats {
  isolation_forest: ModelMetrics;
  autoencoder: ModelMetrics;
  xgboost: ModelMetrics;
  models_trained: boolean;
  versions: Record<string, number>;
}

export interface TrainingResult {
  status: string;
  results: Record<string, ModelMetrics>;
  output_folder: string;
  generated_files: string[];
}

export interface Alert {
  id: number;
  timestamp: string;
  transaction_id: string;
  fraud_score: number;
  alert_type: string;
  acknowledged: number;
}

// ─── API Functions ──────────────────────────────────────────

export async function fetchDashboard(): Promise<DashboardStats> {
  const res = await fetch(`${API}/api/dashboard`);
  if (!res.ok) throw new Error('Failed to fetch dashboard');
  return res.json();
}

export async function fetchModelStats(): Promise<AllModelStats> {
  const res = await fetch(`${API}/api/model-stats`);
  if (!res.ok) throw new Error('Failed to fetch model stats');
  return res.json();
}

export async function trainModels(): Promise<TrainingResult> {
  const res = await fetch(`${API}/api/train`, { method: 'POST' });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Training failed' }));
    throw new Error(err.detail || 'Training failed');
  }
  return res.json();
}

export async function uploadAndPredict(file: File): Promise<PredictionResponse> {
  const form = new FormData();
  form.append('file', file);
  const res = await fetch(`${API}/api/predict`, { method: 'POST', body: form });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Prediction failed' }));
    throw new Error(err.detail || 'Prediction failed');
  }
  return res.json();
}

export async function uploadDataset(file: File) {
  const form = new FormData();
  form.append('file', file);
  const res = await fetch(`${API}/api/upload-dataset`, { method: 'POST', body: form });
  if (!res.ok) throw new Error('Upload failed');
  return res.json();
}

export async function fetchTransactions(params: {
  page?: number; per_page?: number; fraud_only?: boolean; min_score?: number; search?: string;
}) {
  const qs = new URLSearchParams();
  if (params.page) qs.set('page', String(params.page));
  if (params.per_page) qs.set('per_page', String(params.per_page));
  if (params.fraud_only) qs.set('fraud_only', 'true');
  if (params.min_score) qs.set('min_score', String(params.min_score));
  if (params.search) qs.set('search', params.search);
  const res = await fetch(`${API}/api/transactions?${qs}`);
  if (!res.ok) throw new Error('Failed to fetch transactions');
  return res.json();
}

export async function fetchFeatureImportance() {
  const res = await fetch(`${API}/api/feature-importance`);
  if (!res.ok) throw new Error('Failed to fetch feature importance');
  return res.json();
}

export async function submitFeedback(transaction_ids: string[], labels: number[]) {
  const res = await fetch(`${API}/api/feedback`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ transaction_ids, labels }),
  });
  if (!res.ok) throw new Error('Feedback submission failed');
  return res.json();
}

export async function retrain() {
  const res = await fetch(`${API}/api/retrain`, { method: 'POST' });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Retrain failed' }));
    throw new Error(err.detail || 'Retrain failed');
  }
  return res.json();
}

export async function fetchTrainingHistory() {
  const res = await fetch(`${API}/api/training-history`);
  if (!res.ok) throw new Error('Failed to fetch training history');
  return res.json();
}

export async function fetchAlerts(acknowledged?: boolean): Promise<Alert[]> {
  const qs = acknowledged !== undefined ? `?acknowledged=${acknowledged}` : '';
  const res = await fetch(`${API}/api/alerts${qs}`);
  if (!res.ok) throw new Error('Failed to fetch alerts');
  return res.json();
}

export async function acknowledgeAlert(id: number) {
  const res = await fetch(`${API}/api/alerts/${id}/acknowledge`, { method: 'POST' });
  if (!res.ok) throw new Error('Failed to acknowledge alert');
  return res.json();
}

export async function generateSamples(num: number = 100, fraudRatio: number = 0.05) {
  const res = await fetch(`${API}/api/generate-samples`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ num_transactions: num, fraud_ratio: fraudRatio }),
  });
  if (!res.ok) throw new Error('Failed to generate samples');
  return res.json();
}

export async function fetchDatasets() {
  const res = await fetch(`${API}/api/datasets`);
  if (!res.ok) throw new Error('Failed to fetch datasets');
  return res.json();
}

export async function fetchModelVersions(modelName?: string) {
  const qs = modelName ? `?model_name=${modelName}` : '';
  const res = await fetch(`${API}/api/model-versions${qs}`);
  if (!res.ok) throw new Error('Failed to fetch model versions');
  return res.json();
}

export async function exportCSV() {
  const res = await fetch(`${API}/api/export/csv`);
  if (!res.ok) throw new Error('Export failed');
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'predictions_export.csv';
  a.click();
  URL.revokeObjectURL(url);
}

export async function fetchHealth() {
  const res = await fetch(`${API}/api/health`);
  return res.json();
}

export function createTrainingWebSocket(
  onMessage: (data: { model: string; message: string; progress: number }) => void
): WebSocket {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const ws = new WebSocket(`${protocol}//${window.location.host}/ws/training`);
  ws.onmessage = (ev) => {
    try {
      onMessage(JSON.parse(ev.data));
    } catch { /* ignore */ }
  };
  return ws;
}
