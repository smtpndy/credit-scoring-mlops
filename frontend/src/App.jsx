
import { useEffect, useState } from "react";

import {
  Activity,
  ArrowUpRight,
  BadgeCheck,
  ChartNoAxesCombined,
  ChevronRight,
  CircleAlert,
  CreditCard,
  Fingerprint,
  Gauge,
  RefreshCw,
  ShieldAlert,
  ShieldCheck,
  Sparkles,
  WalletCards,
  BellRing,
  Database,
  AlertTriangle,
} from "lucide-react";

import {
  Area,
  AreaChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
  BarChart,
  Bar,
} from "recharts";

import { apiGet, apiPost } from "./api";


/* --------------------------------------------------
   ILLUSTRATIVE CHART DATA
-------------------------------------------------- */

const creditTrend = [
  { month: "Jan", applications: 320, defaults: 42 },
  { month: "Feb", applications: 390, defaults: 47 },
  { month: "Mar", applications: 360, defaults: 39 },
  { month: "Apr", applications: 470, defaults: 55 },
  { month: "May", applications: 510, defaults: 61 },
  { month: "Jun", applications: 560, defaults: 58 },
];

const fraudTrend = [
  { day: "Mon", transactions: 820, flagged: 24 },
  { day: "Tue", transactions: 940, flagged: 31 },
  { day: "Wed", transactions: 880, flagged: 19 },
  { day: "Thu", transactions: 1100, flagged: 42 },
  { day: "Fri", transactions: 1250, flagged: 54 },
  { day: "Sat", transactions: 1020, flagged: 37 },
  { day: "Sun", transactions: 980, flagged: 29 },
];


/* --------------------------------------------------
   REUSABLE COMPONENTS
-------------------------------------------------- */

function Metric({
  label,
  value,
  change,
  icon: Icon,
  tone = "blue",
}) {
  return (
    <div className="metric-card">
      <div className={`metric-icon ${tone}`}>
        <Icon size={19} />
      </div>

      <div className="metric-label">{label}</div>
      <div className="metric-value">{value}</div>

      <div className="metric-foot">
        <span className="positive">{change}</span>
      </div>
    </div>
  );
}


function Panel({ title, subtitle, action, children }) {
  return (
    <section className="panel">
      <div className="panel-head">
        <div>
          <h3>{title}</h3>
          {subtitle && <p>{subtitle}</p>}
        </div>

        {action}
      </div>

      {children}
    </section>
  );
}


/* --------------------------------------------------
   GLOBAL SHAP FEATURE IMPORTANCE
-------------------------------------------------- */

function ShapImportance({ title, data, loading, error }) {
  const chartData = [...(data || [])]
    .filter(
      (item) =>
        item.feature != null &&
        Number.isFinite(Number(item.importance))
    )
    .sort(
      (a, b) =>
        Number(b.importance) - Number(a.importance)
    )
    .slice(0, 10)
    .reverse();

  return (
    <Panel
      title={title}
      subtitle="Top 10 features by mean absolute SHAP value"
    >
      {loading ? (
        <div className="muted">
          Loading SHAP feature importance...
        </div>
      ) : error ? (
        <div className="error-text">{error}</div>
      ) : chartData.length > 0 ? (
        <>
          <div style={{ width: "100%", height: 370 }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={chartData}
                layout="vertical"
                margin={{
                  top: 5,
                  right: 25,
                  left: 10,
                  bottom: 5,
                }}
              >
                <CartesianGrid
                  strokeDasharray="3 3"
                  horizontal={false}
                  stroke="#263247"
                />

                <XAxis
                  type="number"
                  axisLine={false}
                  tickLine={false}
                  tick={{ fill: "#8491a8", fontSize: 11 }}
                />

                <YAxis
                  type="category"
                  dataKey="feature"
                  width={210}
                  axisLine={false}
                  tickLine={false}
                  tick={{ fill: "#8491a8", fontSize: 10 }}
                />

                <Tooltip
                  formatter={(value) => [
                    Number(value).toFixed(6),
                    "SHAP importance",
                  ]}
                  contentStyle={{
                    background: "#151f30",
                    border: "1px solid #2b3850",
                    borderRadius: 10,
                    color: "#ffffff",
                  }}
                  labelStyle={{ color: "#ffffff" }}
                />

                <Bar
                  dataKey="importance"
                  name="Importance"
                  fill="#7894ff"
                  radius={[0, 5, 5, 0]}
                />
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div className="chart-legend">
            <span>
              <i className="dot blue-dot" />
              Mean absolute SHAP value
            </span>
          </div>
        </>
      ) : (
        <div className="muted">
          No SHAP feature importance data available.
        </div>
      )}
    </Panel>
  );
}


/* --------------------------------------------------
   PREDICTION RESULT
-------------------------------------------------- */

function Result({ result, type }) {
  if (!result) return null;

  const isCredit = type === "credit";

  const probability = isCredit
    ? result.default_probability
    : result.fraud_probability;

  const high = isCredit
    ? result.risk_band === "High Risk"
    : probability >= 0.5;

  return (
    <div className={`result-card ${high ? "risk" : "safe"}`}>
      <div className="result-top">
        <span>
          {isCredit
            ? "DEFAULT RISK ESTIMATE"
            : "FRAUD RISK ESTIMATE"}
        </span>

        <span className="result-pill">
          {isCredit
            ? result.risk_band
            : high
              ? "Flagged"
              : "Normal"}
        </span>
      </div>

      <div className="probability">
        {(probability * 100).toFixed(1)}
        <small>%</small>
      </div>

      <div className="prob-label">
        Model-estimated probability
      </div>

      <div className="result-message">
        {isCredit ? result.message : result.decision}
      </div>

      <div className="disclaimer">
        Educational decision-support output. Not for real-world
        lending or fraud decisions.
      </div>
    </div>
  );
}


/* --------------------------------------------------
   INDIVIDUAL SHAP EXPLANATION
-------------------------------------------------- */

function IndividualExplanation({ result, type }) {
  const features = result?.explanation?.top_features;

  if (!Array.isArray(features) || features.length === 0) {
    return null;
  }

  const maxValue = Math.max(
    ...features.map((item) => Math.abs(Number(item.shap_value) || 0)),
    0.000001
  );

  const positiveClass = type === "credit" ? "bad credit" : "fraud";

  return (
    <Panel
      title="Why did the model make this prediction?"
      subtitle={`Individual SHAP explanation — top ${features.length} contributing features`}
    >
      <div className="shap-explanation">
        <div className="shap-intro">
          <Sparkles size={17} />

          <span>
            These features influenced this specific prediction.
            Positive SHAP values push the model output towards {positiveClass};
            negative SHAP values push it away from {positiveClass}.
          </span>
        </div>

        {features.map((item, index) => {
          const value = Number(item.shap_value) || 0;
          const width = (Math.abs(value) / maxValue) * 100;

          const direction =
            value > 0
              ? "Increases risk"
              : value < 0
                ? "Reduces risk"
                : "No material effect";

          const directionClass =
            value > 0
              ? "positive-shap"
              : value < 0
                ? "negative-shap"
                : "";

          const barClass =
            value > 0
              ? "positive-bar"
              : "negative-bar";

          return (
            <div className="shap-feature" key={item.feature}>
              <div className="shap-feature-heading">
                <div className="shap-feature-name">
                  <span className="shap-rank">
                    {index + 1}
                  </span>

                  <strong>{item.feature}</strong>
                </div>

                <span className={`shap-direction ${directionClass}`}>
                  {direction}
                </span>
              </div>

              <div className="shap-bar-track">
                <div
                  className={`shap-bar ${barClass}`}
                  style={{ width: `${width}%` }}
                />
              </div>

              <div className="shap-value">
                SHAP contribution: {value > 0 ? "+" : ""}
                {value.toFixed(6)}
              </div>
            </div>
          );
        })}

        <div className="shap-disclaimer">
          SHAP values represent contributions to the model output.
          They are not percentages of probability and do not prove
          causation. Feature values shown by the backend may be
          transformed model inputs.
        </div>
      </div>
    </Panel>
  );
}


/* --------------------------------------------------
   CREDIT DASHBOARD
-------------------------------------------------- */

function CreditDashboard() {
  const [form, setForm] = useState({
    status: "... < 100 DM",
    duration: 6,
    credit_history: "critical account/other credits existing",
    purpose: "domestic appliances",
    amount: 1169,
    savings: "unknown/no savings account",
    employment_duration: "... >= 7 years",
    installment_rate: 4,
    personal_status_sex: "male : single",
    other_debtors: "none",
    present_residence: 4,
    property: "real estate",
    age: 67,
    other_installment_plans: "none",
    housing: "own",
    number_credits: 2,
    job: "skilled employee/official",
    people_liable: 1,
    telephone: "yes",
    foreign_worker: "yes",
  });

  const [result, setResult] = useState(null);
  const [metrics, setMetrics] = useState(null);

  const [shapData, setShapData] = useState([]);
  const [shapLoading, setShapLoading] = useState(true);
  const [shapError, setShapError] = useState("");

  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    apiGet("/api/credit/metrics")
      .then(setMetrics)
      .catch((err) => console.error("Credit metrics:", err));
  }, []);

  useEffect(() => {
    apiGet("/api/credit/shap")
      .then((data) => {
        const values = Array.isArray(data)
          ? data
          : data.features || [];

        setShapData(values);
        setShapError("");
      })
      .catch((err) => {
        console.error("Credit SHAP:", err);
        setShapError("Unable to load credit SHAP data.");
      })
      .finally(() => setShapLoading(false));
  }, []);

  const numericFields = new Set([
    "duration",
    "amount",
    "installment_rate",
    "present_residence",
    "age",
    "number_credits",
    "people_liable",
  ]);

  const update = (key, value) => {
    setForm((previous) => ({
      ...previous,
      [key]: numericFields.has(key) ? Number(value) : value,
    }));
  };

  async function predict(e) {
    e.preventDefault();

    setBusy(true);
    setError("");
    setResult(null);

    try {
      const response = await apiPost(
        "/api/credit/explain",
        form
      );

      setResult(response);
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  const fields = [
    { key: "status", label: "Checking account status", type: "text" },
    { key: "duration", label: "Duration (months)", type: "number" },
    { key: "credit_history", label: "Credit history", type: "text" },
    { key: "purpose", label: "Loan purpose", type: "text" },
    { key: "amount", label: "Credit amount", type: "number" },
    { key: "savings", label: "Savings account/bonds", type: "text" },
    { key: "employment_duration", label: "Employment duration", type: "text" },
    { key: "installment_rate", label: "Installment rate (1–4)", type: "number" },
    { key: "personal_status_sex", label: "Personal status and sex", type: "text" },
    { key: "other_debtors", label: "Other debtors/guarantors", type: "text" },
    { key: "present_residence", label: "Present residence (1–4)", type: "number" },
    { key: "property", label: "Property", type: "text" },
    { key: "age", label: "Age (years)", type: "number" },
    { key: "other_installment_plans", label: "Other installment plans", type: "text" },
    { key: "housing", label: "Housing", type: "text" },
    { key: "number_credits", label: "Number of existing credits", type: "number" },
    { key: "job", label: "Job", type: "text" },
    { key: "people_liable", label: "People liable for maintenance", type: "number" },
    { key: "telephone", label: "Telephone", type: "text" },
    { key: "foreign_worker", label: "Foreign worker", type: "text" },
  ];

  return (
    <main className="page">
      <div className="page-heading">
        <div>
          <div className="eyebrow">
            RISK INTELLIGENCE / LENDING
          </div>

          <h1>Credit Scoring</h1>

          <p>
            Assess applicant credit risk using the German Credit
            dataset model.
          </p>
        </div>

        <div className="live-status">
          <span className="pulse" />
          CREDIT MODEL
        </div>
      </div>

      <div className="metric-grid">
        <Metric
          label="Dataset records"
          value={metrics?.rows ?? "1,000"}
          change="GermanCredit.csv"
          icon={WalletCards}
        />

        <Metric
          label="Input features"
          value="20"
          change="German Credit features"
          icon={BadgeCheck}
          tone="green"
        />

        <Metric
          label="Target"
          value="Bad credit"
          change="Risk prediction"
          icon={CircleAlert}
          tone="orange"
        />

        <Metric
          label="Model ROC-AUC"
          value={
            metrics?.roc_auc != null
              ? metrics.roc_auc.toFixed(3)
              : "—"
          }
          change="Evaluation metric"
          icon={Gauge}
          tone="purple"
        />
      </div>

      <div className="content-grid">
        <Panel
          title="Applicant risk assessment"
          subtitle="Enter the 20 features required by the German Credit model"
        >
          <form className="prediction-form" onSubmit={predict}>
            <div className="form-grid">
              {fields.map((field) => (
                <label key={field.key}>
                  {field.label}

                  <input
                    type={field.type}
                    step={field.type === "number" ? "any" : undefined}
                    value={form[field.key]}
                    onChange={(e) =>
                      update(field.key, e.target.value)
                    }
                    required
                  />
                </label>
              ))}
            </div>

            <button className="primary-btn" disabled={busy}>
              {busy ? "Scoring applicant..." : "Calculate credit risk"}
              <ArrowUpRight size={17} />
            </button>

            {error && (
              <div className="error-text">{error}</div>
            )}
          </form>

          <Result result={result} type="credit" />

          <IndividualExplanation
            result={result}
            type="credit"
          />
        </Panel>

        <div className="stack">
          <Panel
            title="Portfolio performance"
            subtitle="Illustrative monthly application volume"
          >
            <div className="chart-wrap">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={creditTrend}>
                  <defs>
                    <linearGradient
                      id="creditFill"
                      x1="0"
                      y1="0"
                      x2="0"
                      y2="1"
                    >
                      <stop
                        offset="0%"
                        stopColor="#6687f5"
                        stopOpacity={0.28}
                      />
                      <stop
                        offset="100%"
                        stopColor="#6687f5"
                        stopOpacity={0}
                      />
                    </linearGradient>
                  </defs>

                  <CartesianGrid
                    strokeDasharray="3 3"
                    vertical={false}
                    stroke="#263247"
                  />

                  <XAxis
                    dataKey="month"
                    axisLine={false}
                    tickLine={false}
                    tick={{ fill: "#8491a8", fontSize: 11 }}
                  />

                  <YAxis
                    axisLine={false}
                    tickLine={false}
                    tick={{ fill: "#8491a8", fontSize: 11 }}
                  />

                  <Tooltip
                    contentStyle={{
                      background: "#151f30",
                      border: "1px solid #2b3850",
                      borderRadius: 10,
                    }}
                  />

                  <Area
                    type="monotone"
                    dataKey="applications"
                    stroke="#7894ff"
                    strokeWidth={2.5}
                    fill="url(#creditFill)"
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>

            <div className="chart-legend">
              <span>
                <i className="dot blue-dot" />
                Applications
              </span>

              <span className="muted">
                Illustrative chart
              </span>
            </div>
          </Panel>

          <Panel
            title="Model evaluation"
            subtitle="Latest training run metrics"
          >
            <div className="evaluation-list">
              {["accuracy", "precision", "recall", "roc_auc"].map((key) => (
                <div className="eval-row" key={key}>
                  <span>
                    {key.replace("_", " ").toUpperCase()}
                  </span>

                  <strong>
                    {metrics?.[key] != null
                      ? `${(metrics[key] * 100).toFixed(1)}%`
                      : "Train model first"}
                  </strong>
                </div>
              ))}
            </div>
          </Panel>

          <ShapImportance
            title="Credit feature importance (SHAP)"
            data={shapData}
            loading={shapLoading}
            error={shapError}
          />
        </div>
      </div>

      <div className="notice">
        <ShieldCheck size={17} />

        Model trained using GermanCredit.csv. This is an
        educational decision-support project, not a real-world
        lending decision system.
      </div>
    </main>
  );
}


/* --------------------------------------------------
   FRAUD DETECTION DASHBOARD
-------------------------------------------------- */

function FraudDashboard() {
  const initialForm = {
    ...Object.fromEntries(
      Array.from(
        { length: 28 },
        (_, index) => [`V${index + 1}`, 0]
      )
    ),
    Amount: 100,
  };

  const [form, setForm] = useState(initialForm);
  const [result, setResult] = useState(null);
  const [metrics, setMetrics] = useState(null);

  const [shapData, setShapData] = useState([]);
  const [shapLoading, setShapLoading] = useState(true);
  const [shapError, setShapError] = useState("");

  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    apiGet("/api/fraud/metrics")
      .then(setMetrics)
      .catch((err) => console.error("Fraud metrics:", err));
  }, []);

  useEffect(() => {
    apiGet("/api/fraud/shap")
      .then((data) => {
        const values = Array.isArray(data)
          ? data
          : data.features || [];

        setShapData(values);
        setShapError("");
      })
      .catch((err) => {
        console.error("Fraud SHAP:", err);
        setShapError("Unable to load fraud SHAP data.");
      })
      .finally(() => setShapLoading(false));
  }, []);

  const update = (key, value) => {
    setForm((previous) => ({
      ...previous,
      [key]: Number(value),
    }));
  };

  async function predict(e) {
    e.preventDefault();

    setBusy(true);
    setError("");
    setResult(null);

    try {
      const response = await apiPost(
        "/api/fraud/explain",
        form
      );

      setResult(response);
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  const fields = [
    ...Array.from(
      { length: 28 },
      (_, index) => ({
        key: `V${index + 1}`,
        label: `Feature V${index + 1}`,
      })
    ),
    {
      key: "Amount",
      label: "Transaction Amount",
    },
  ];

  return (
    <main className="page">
      <div className="page-heading">
        <div>
          <div className="eyebrow">
            RISK INTELLIGENCE / TRANSACTIONS
          </div>

          <h1>Fraud Detection</h1>

          <p>
            Detect fraudulent transactions using the Kaggle
            Credit Card Fraud Detection 2023 dataset.
          </p>
        </div>

        <div className="live-status">
          <span className="pulse" />
          FRAUD MODEL
        </div>
      </div>

      <div className="metric-grid">
        <Metric
          label="Dataset records"
          value={
            metrics?.rows != null
              ? Number(metrics.rows).toLocaleString()
              : "—"
          }
          change="creditcard_2023.csv"
          icon={Activity}
        />

        <Metric
          label="Input features"
          value="29"
          change="V1–V28 + Amount"
          icon={Fingerprint}
          tone="green"
        />

        <Metric
          label="Target column"
          value="Class"
          change="0 = Normal, 1 = Fraud"
          icon={ShieldAlert}
          tone="orange"
        />

        <Metric
          label="Model ROC-AUC"
          value={
            metrics?.roc_auc != null
              ? metrics.roc_auc.toFixed(3)
              : "—"
          }
          change="Evaluation metric"
          icon={Gauge}
          tone="purple"
        />
      </div>

      <div className="content-grid">
        <Panel
          title="Transaction risk analysis"
          subtitle="Enter all 29 features required by the trained fraud model"
        >
          <form className="prediction-form" onSubmit={predict}>
            <div className="form-grid">
              {fields.map((field) => (
                <label key={field.key}>
                  {field.label}

                  <input
                    type="number"
                    step="any"
                    value={form[field.key]}
                    onChange={(e) =>
                      update(field.key, e.target.value)
                    }
                    required
                  />
                </label>
              ))}
            </div>

            <button className="primary-btn" disabled={busy}>
              {busy
                ? "Analyzing transaction..."
                : "Analyze transaction"}

              <ArrowUpRight size={17} />
            </button>

            {error && (
              <div className="error-text">{error}</div>
            )}
          </form>

          <Result result={result} type="fraud" />

          <IndividualExplanation
            result={result}
            type="fraud"
          />
        </Panel>

        <div className="stack">
          <Panel
            title="Transaction monitoring"
            subtitle="Illustrative daily transaction volume"
          >
            <div className="chart-wrap">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={fraudTrend}>
                  <CartesianGrid
                    strokeDasharray="3 3"
                    vertical={false}
                    stroke="#263247"
                  />

                  <XAxis
                    dataKey="day"
                    axisLine={false}
                    tickLine={false}
                    tick={{ fill: "#8491a8", fontSize: 11 }}
                  />

                  <YAxis
                    axisLine={false}
                    tickLine={false}
                    tick={{ fill: "#8491a8", fontSize: 11 }}
                  />

                  <Tooltip
                    contentStyle={{
                      background: "#151f30",
                      border: "1px solid #2b3850",
                      borderRadius: 10,
                    }}
                  />

                  <Bar
                    dataKey="transactions"
                    fill="#6687f5"
                    radius={[5, 5, 0, 0]}
                  />

                  <Bar
                    dataKey="flagged"
                    fill="#f29b62"
                    radius={[5, 5, 0, 0]}
                  />
                </BarChart>
              </ResponsiveContainer>
            </div>

            <div className="chart-legend">
              <span>
                <i className="dot blue-dot" />
                Transactions
              </span>

              <span>
                <i className="dot orange-dot" />
                Flagged
              </span>
            </div>
          </Panel>

          <Panel
            title="Model evaluation"
            subtitle="Latest training run metrics"
          >
            <div className="evaluation-list">
              {["accuracy", "precision", "recall", "roc_auc"].map((key) => (
                <div className="eval-row" key={key}>
                  <span>
                    {key.replace("_", " ").toUpperCase()}
                  </span>

                  <strong>
                    {metrics?.[key] != null
                      ? `${(metrics[key] * 100).toFixed(1)}%`
                      : "Train model first"}
                  </strong>
                </div>
              ))}
            </div>
          </Panel>

          <ShapImportance
            title="Fraud feature importance (SHAP)"
            data={shapData}
            loading={shapLoading}
            error={shapError}
          />
        </div>
      </div>

      <div className="notice">
        <ShieldCheck size={17} />

        Model trained using creditcard_2023.csv. V1–V28 are
        anonymized numerical features. This is an educational
        decision-support project.
      </div>
    </main>
  );
}


/* --------------------------------------------------
   MONITORING DASHBOARD
-------------------------------------------------- */

function MonitoringDashboard() {
  const [summary, setSummary] = useState(null);
  const [trends, setTrends] = useState([]);
  const [history, setHistory] = useState([]);

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  async function loadMonitoring() {
    setLoading(true);
    setError("");

    try {
      const [summaryData, trendData, historyData] =
        await Promise.all([
          apiGet("/api/monitoring/summary"),
          apiGet("/api/monitoring/trends"),
          apiGet("/api/history"),
        ]);

      setSummary(summaryData);
      setTrends(trendData.trends || []);
      setHistory(historyData || []);
    } catch (err) {
      setError(err.message || "Unable to load monitoring data.");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    loadMonitoring();
  }, []);

  const creditRisk = summary?.credit_risk_distribution || {};
  const fraudDistribution = summary?.fraud_distribution || {};
  const predictionCounts = summary?.prediction_counts || {};
  const alerts = summary?.alerts || {};

  const riskChartData = [
    { name: "High Risk", count: creditRisk.high_risk || 0 },
    { name: "Medium Risk", count: creditRisk.medium_risk || 0 },
    { name: "Low Risk", count: creditRisk.low_risk || 0 },
  ];

  const fraudChartData = [
    { name: "Flagged", count: fraudDistribution.flagged || 0 },
    { name: "Normal", count: fraudDistribution.normal || 0 },
  ];

  const formatDate = (timestamp) => {
    if (!timestamp) return "—";

    return new Date(timestamp).toLocaleString();
  };

  const getDecisionClass = (item) => {
    if (item.domain === "credit") {
      if (item.decision === "High Risk") return "risk";
      if (item.decision === "Medium Risk") return "warning";
      return "safe";
    }

    return item.decision === "Flagged" ? "risk" : "safe";
  };

  return (
    <main className="page">
      <div className="page-heading">
        <div>
          <div className="eyebrow">
            RISK INTELLIGENCE / MODEL MONITORING
          </div>

          <h1>Model Monitoring</h1>

          <p>
            Monitor credit predictions, fraud detection activity,
            risk distribution and recent model outputs.
          </p>
        </div>

        <button
          className="primary-btn"
          onClick={loadMonitoring}
          disabled={loading}
        >
          <RefreshCw size={16} />
          {loading ? "Refreshing..." : "Refresh data"}
        </button>
      </div>

      {error && (
        <div className="notice">
          <AlertTriangle size={17} />
          {error}
        </div>
      )}

      {loading && !summary ? (
        <Panel title="Loading monitoring data">
          <div className="muted">
            Fetching live prediction statistics from FastAPI...
          </div>
        </Panel>
      ) : summary ? (
        <>
          <div className="metric-grid">
            <Metric
              label="Total predictions"
              value={summary.total_predictions ?? 0}
              change="Recorded predictions"
              icon={Activity}
            />

            <Metric
              label="Credit predictions"
              value={predictionCounts.credit ?? 0}
              change="German Credit model"
              icon={CreditCard}
              tone="green"
            />

            <Metric
              label="Fraud predictions"
              value={predictionCounts.fraud ?? 0}
              change="Fraud detection model"
              icon={Fingerprint}
              tone="purple"
            />

            <Metric
              label="Total alerts"
              value={alerts.total_alerts ?? 0}
              change="High risk + flagged fraud"
              icon={BellRing}
              tone="orange"
            />
          </div>

          <div className="content-grid">
            <Panel
              title="Credit risk distribution"
              subtitle="Actual recorded credit predictions"
            >
              <div className="evaluation-list">
                <div className="eval-row">
                  <span>HIGH RISK</span>
                  <strong>{creditRisk.high_risk ?? 0}</strong>
                </div>

                <div className="eval-row">
                  <span>MEDIUM RISK</span>
                  <strong>{creditRisk.medium_risk ?? 0}</strong>
                </div>

                <div className="eval-row">
                  <span>LOW RISK</span>
                  <strong>{creditRisk.low_risk ?? 0}</strong>
                </div>
              </div>

              <div className="chart-wrap">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={riskChartData}>
                    <CartesianGrid
                      strokeDasharray="3 3"
                      vertical={false}
                      stroke="#263247"
                    />

                    <XAxis
                      dataKey="name"
                      axisLine={false}
                      tickLine={false}
                      tick={{ fill: "#8491a8", fontSize: 10 }}
                    />

                    <YAxis
                      allowDecimals={false}
                      axisLine={false}
                      tickLine={false}
                      tick={{ fill: "#8491a8", fontSize: 11 }}
                    />

                    <Tooltip
                      contentStyle={{
                        background: "#151f30",
                        border: "1px solid #2b3850",
                        borderRadius: 10,
                      }}
                    />

                    <Bar
                      dataKey="count"
                      name="Predictions"
                      fill="#7894ff"
                      radius={[5, 5, 0, 0]}
                    />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </Panel>

            <Panel
              title="Fraud detection distribution"
              subtitle="Actual recorded fraud predictions"
            >
              <div className="evaluation-list">
                <div className="eval-row">
                  <span>FLAGGED TRANSACTIONS</span>
                  <strong>{fraudDistribution.flagged ?? 0}</strong>
                </div>

                <div className="eval-row">
                  <span>NORMAL TRANSACTIONS</span>
                  <strong>{fraudDistribution.normal ?? 0}</strong>
                </div>
              </div>

              <div className="chart-wrap">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={fraudChartData}>
                    <CartesianGrid
                      strokeDasharray="3 3"
                      vertical={false}
                      stroke="#263247"
                    />

                    <XAxis
                      dataKey="name"
                      axisLine={false}
                      tickLine={false}
                      tick={{ fill: "#8491a8", fontSize: 11 }}
                    />

                    <YAxis
                      allowDecimals={false}
                      axisLine={false}
                      tickLine={false}
                      tick={{ fill: "#8491a8", fontSize: 11 }}
                    />

                    <Tooltip
                      contentStyle={{
                        background: "#151f30",
                        border: "1px solid #2b3850",
                        borderRadius: 10,
                      }}
                    />

                    <Bar
                      dataKey="count"
                      name="Transactions"
                      fill="#f29b62"
                      radius={[5, 5, 0, 0]}
                    />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </Panel>
          </div>

          <Panel
            title="Prediction activity"
            subtitle="Daily recorded prediction volume"
          >
            {trends.length > 0 ? (
              <>
                <div className="chart-wrap">
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={trends}>
                      <defs>
                        <linearGradient
                          id="monitoringFill"
                          x1="0"
                          y1="0"
                          x2="0"
                          y2="1"
                        >
                          <stop
                            offset="0%"
                            stopColor="#7894ff"
                            stopOpacity={0.3}
                          />

                          <stop
                            offset="100%"
                            stopColor="#7894ff"
                            stopOpacity={0}
                          />
                        </linearGradient>
                      </defs>

                      <CartesianGrid
                        strokeDasharray="3 3"
                        vertical={false}
                        stroke="#263247"
                      />

                      <XAxis
                        dataKey="date"
                        axisLine={false}
                        tickLine={false}
                        tick={{ fill: "#8491a8", fontSize: 10 }}
                      />

                      <YAxis
                        allowDecimals={false}
                        axisLine={false}
                        tickLine={false}
                        tick={{ fill: "#8491a8", fontSize: 11 }}
                      />

                      <Tooltip
                        contentStyle={{
                          background: "#151f30",
                          border: "1px solid #2b3850",
                          borderRadius: 10,
                        }}
                      />

                      <Area
                        type="monotone"
                        dataKey="total"
                        name="Total predictions"
                        stroke="#7894ff"
                        strokeWidth={2.5}
                        fill="url(#monitoringFill)"
                      />

                      <Area
                        type="monotone"
                        dataKey="credit"
                        name="Credit"
                        stroke="#60d5a9"
                        fill="none"
                      />

                      <Area
                        type="monotone"
                        dataKey="fraud"
                        name="Fraud"
                        stroke="#f29b62"
                        fill="none"
                      />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>

                <div className="chart-legend">
                  <span>
                    <i className="dot blue-dot" />
                    Total
                  </span>

                  <span>
                    <i className="dot green-dot" />
                    Credit
                  </span>

                  <span>
                    <i className="dot orange-dot" />
                    Fraud
                  </span>
                </div>
              </>
            ) : (
              <div className="muted">
                No daily prediction activity recorded yet.
                Make predictions to populate this chart.
              </div>
            )}
          </Panel>

          <div className="content-grid">
            <Panel
              title="Average prediction probability"
              subtitle="Mean probability across recorded predictions"
            >
              <div className="probability">
                {(
                  (summary.average_probability || 0) * 100
                ).toFixed(2)}
                <small>%</small>
              </div>

              <div className="prob-label">
                Average model output probability
              </div>

              <div className="shap-disclaimer">
                This is an aggregate of credit and fraud probabilities.
                The two model outputs represent different tasks and
                should not be compared as equivalent risk measures.
              </div>
            </Panel>

            <Panel
              title="Risk alerts"
              subtitle="Recorded high-risk and flagged predictions"
            >
              <div className="evaluation-list">
                <div className="eval-row">
                  <span>HIGH-RISK CREDIT</span>
                  <strong>{alerts.high_risk_credit ?? 0}</strong>
                </div>

                <div className="eval-row">
                  <span>FLAGGED FRAUD</span>
                  <strong>{alerts.fraud_flagged ?? 0}</strong>
                </div>

                <div className="eval-row">
                  <span>TOTAL ALERTS</span>
                  <strong>{alerts.total_alerts ?? 0}</strong>
                </div>
              </div>
            </Panel>
          </div>

          <Panel
            title="Recent prediction history"
            subtitle="Latest recorded model predictions"
            action={
              <span className="top-tag">
                <Database size={14} />
                LIVE API DATA
              </span>
            }
          >
            {history.length > 0 ? (
              <div className="evaluation-list">
                {history.slice(0, 15).map((item) => (
                  <div
                    className="eval-row"
                    key={item.id}
                  >
                    <div>
                      <strong>
                        {item.domain === "credit"
                          ? "Credit Scoring"
                          : "Fraud Detection"}
                      </strong>

                      <div className="muted">
                        {formatDate(item.timestamp)}
                      </div>
                    </div>

                    <div style={{ textAlign: "right" }}>
                      <strong>
                        {(Number(item.probability) * 100).toFixed(2)}%
                      </strong>

                      <div>
                        <span
                          className={`result-pill ${getDecisionClass(item)}`}
                        >
                          {item.decision}
                        </span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="muted">
                No prediction history available.
              </div>
            )}
          </Panel>
        </>
      ) : (
        <Panel title="Monitoring unavailable">
          <div className="muted">
            Could not retrieve monitoring information from the backend.
          </div>
        </Panel>
      )}
    </main>
  );
}


/* --------------------------------------------------
   MAIN APP
-------------------------------------------------- */

export default function App() {
  const [active, setActive] = useState("credit");
  const [apiStatus, setApiStatus] = useState("checking");

  useEffect(() => {
    apiGet("/health")
      .then(() => setApiStatus("online"))
      .catch(() => setApiStatus("offline"));
  }, []);

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">
            <ChartNoAxesCombined size={21} />
          </div>

          <div>
            <strong>
              Cred<span>Shield</span>
            </strong>

            <small>ML PLATFORM</small>
          </div>
        </div>

        <div className="side-label">
          WORKSPACE
        </div>

        <button
          className={`nav-item ${active === "credit" ? "active" : ""}`}
          onClick={() => setActive("credit")}
        >
          <CreditCard size={18} />
          Credit Scoring
          <ChevronRight size={15} />
        </button>

        <button
          className={`nav-item ${active === "fraud" ? "active" : ""}`}
          onClick={() => setActive("fraud")}
        >
          <Fingerprint size={18} />
          Fraud Detection
          <ChevronRight size={15} />
        </button>

        <button
          className={`nav-item ${active === "monitoring" ? "active" : ""}`}
          onClick={() => setActive("monitoring")}
        >
          <Activity size={18} />
          Monitoring
          <ChevronRight size={15} />
        </button>

        <div className="sidebar-bottom">
          <div className="environment">
            <span className={`status-dot ${apiStatus}`} />

            <div>
              <strong>API {apiStatus}</strong>
              <small>Local environment</small>
            </div>

            <RefreshCw size={14} />
          </div>

          <div className="user-chip">
            <div className="avatar">ML</div>

            <div>
              <strong>MLOps Workspace</strong>
              <small>Developer view</small>
            </div>
          </div>
        </div>
      </aside>

      <div className="main-area">
        <header className="topbar">
          <div className="breadcrumb">
            Workspace
            <span>/</span>

            <strong>
              {active === "credit"
                ? "Credit Scoring"
                : active === "fraud"
                  ? "Fraud Detection"
                  : "Model Monitoring"}
            </strong>
          </div>

          <div className="topbar-right">
            <span className="top-tag">
              <Sparkles size={14} />
              MODEL-POWERED
            </span>

            <div className="top-avatar">ML</div>
          </div>
        </header>

        {active === "credit" ? (
          <CreditDashboard />
        ) : active === "fraud" ? (
          <FraudDashboard />
        ) : (
          <MonitoringDashboard />
        )}
      </div>
    </div>
  );
}
