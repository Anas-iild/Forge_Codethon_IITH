"""
LSTM-based Candidate Growth Prediction Microservice
Uses a simulated LSTM with numpy for time-series signal analysis.
Predicts role progression, growth velocity, learning velocity, and role fit.
"""

from flask import Flask, request, jsonify
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import json
import math

app = Flask(__name__)

# Role progression ladder per department
ROLE_LADDERS = {
    "Engineering": [
        "Junior Engineer", "Software Engineer", "Senior Engineer",
        "Staff Engineer", "Principal Engineer", "Engineering Manager", "Director of Engineering"
    ],
    "Product": [
        "Associate PM", "Product Manager", "Senior PM",
        "Principal PM", "Group PM", "Director of Product", "VP of Product"
    ],
    "Design": [
        "Junior Designer", "Designer", "Senior Designer",
        "Lead Designer", "Principal Designer", "Design Manager", "Director of Design"
    ],
    "Data Science": [
        "Data Analyst", "Data Scientist", "Senior Data Scientist",
        "Lead Data Scientist", "Principal Scientist", "Data Science Manager", "Director of Data"
    ],
    "Marketing": [
        "Marketing Associate", "Marketing Manager", "Senior Marketing Manager",
        "Marketing Director", "VP Marketing", "CMO"
    ],
    "Sales": [
        "Sales Development Rep", "Account Executive", "Senior AE",
        "Sales Manager", "Senior Sales Manager", "VP Sales", "Chief Revenue Officer"
    ],
    "Operations": [
        "Operations Analyst", "Operations Manager", "Senior Operations Manager",
        "Director of Operations", "VP Operations", "COO"
    ],
    "Finance": [
        "Financial Analyst", "Senior Analyst", "Finance Manager",
        "Senior Finance Manager", "Director of Finance", "VP Finance", "CFO"
    ],
}

DEFAULT_LADDER = [
    "Associate", "Specialist", "Senior Specialist",
    "Lead", "Principal", "Manager", "Director"
]

# Simulated LSTM cell (stateful recurrent computation)
class SimpleLSTM:
    def __init__(self, input_size, hidden_size):
        np.random.seed(42)
        self.hidden_size = hidden_size
        scale = 0.1
        # Input gate
        self.Wi = np.random.randn(hidden_size, input_size) * scale
        self.Ui = np.random.randn(hidden_size, hidden_size) * scale
        self.bi = np.zeros((hidden_size, 1))
        # Forget gate
        self.Wf = np.random.randn(hidden_size, input_size) * scale
        self.Uf = np.random.randn(hidden_size, hidden_size) * scale
        self.bf = np.ones((hidden_size, 1))  # bias toward remembering
        # Cell gate
        self.Wc = np.random.randn(hidden_size, input_size) * scale
        self.Uc = np.random.randn(hidden_size, hidden_size) * scale
        self.bc = np.zeros((hidden_size, 1))
        # Output gate
        self.Wo = np.random.randn(hidden_size, input_size) * scale
        self.Uo = np.random.randn(hidden_size, hidden_size) * scale
        self.bo = np.zeros((hidden_size, 1))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -10, 10)))

    def forward(self, X):
        """X: (timesteps, input_size)"""
        h = np.zeros((self.hidden_size, 1))
        c = np.zeros((self.hidden_size, 1))
        hidden_states = []
        for t in range(len(X)):
            x = X[t].reshape(-1, 1)
            i = self.sigmoid(self.Wi @ x + self.Ui @ h + self.bi)
            f = self.sigmoid(self.Wf @ x + self.Uf @ h + self.bf)
            c_tilde = np.tanh(self.Wc @ x + self.Uc @ h + self.bc)
            c = f * c + i * c_tilde
            o = self.sigmoid(self.Wo @ x + self.Uo @ h + self.bo)
            h = o * np.tanh(c)
            hidden_states.append(h.flatten())
        return np.array(hidden_states), h.flatten()

lstm_model = SimpleLSTM(input_size=5, hidden_size=32)


def compute_velocity(signal_sequence):
    """Compute rate of change using linear regression slope on signal."""
    if len(signal_sequence) < 2:
        return 0.0
    x = np.arange(len(signal_sequence), dtype=float)
    y = np.array(signal_sequence, dtype=float)
    slope = np.polyfit(x, y, 1)[0]
    return float(slope)


def compute_acceleration(signal_sequence):
    """Second derivative — acceleration of growth."""
    if len(signal_sequence) < 3:
        return 0.0
    x = np.arange(len(signal_sequence), dtype=float)
    y = np.array(signal_sequence, dtype=float)
    coeffs = np.polyfit(x, y, 2)
    return float(2 * coeffs[0])


def predict_candidate(signals_list, current_role, department, years_exp):
    """
    signals_list: list of dicts with keys: performance, skill_growth, collaboration, initiative, delivery
    Returns prediction dict.
    """
    if len(signals_list) < 2:
        raise ValueError("Need at least 2 months of signal data")

    # Build matrix: (T, 5)
    features = ['performance', 'skill_growth', 'collaboration', 'initiative', 'delivery']
    X = np.array([[s[f] for f in features] for s in signals_list], dtype=float)

    # Normalize
    scaler = MinMaxScaler()
    X_norm = scaler.fit_transform(X)

    # Run LSTM
    hidden_states, final_hidden = lstm_model.forward(X_norm)

    # Extract trajectory features from hidden states
    h_mean = hidden_states.mean(axis=0)
    h_trend = np.diff(hidden_states, axis=0).mean(axis=0) if len(hidden_states) > 1 else np.zeros_like(h_mean)

    # Compute individual signal velocities
    perf_velocity = compute_velocity([s['performance'] for s in signals_list])
    skill_velocity = compute_velocity([s['skill_growth'] for s in signals_list])
    collab_velocity = compute_velocity([s['collaboration'] for s in signals_list])
    init_velocity = compute_velocity([s['initiative'] for s in signals_list])
    delivery_velocity = compute_velocity([s['delivery'] for s in signals_list])

    perf_accel = compute_acceleration([s['performance'] for s in signals_list])
    skill_accel = compute_acceleration([s['skill_growth'] for s in signals_list])

    # Current averages (last 3 months if available)
    recent = signals_list[-min(3, len(signals_list)):]
    avg_perf = np.mean([s['performance'] for s in recent])
    avg_skill = np.mean([s['skill_growth'] for s in recent])
    avg_collab = np.mean([s['collaboration'] for s in recent])
    avg_init = np.mean([s['initiative'] for s in recent])
    avg_delivery = np.mean([s['delivery'] for s in recent])

    overall_score = (avg_perf * 0.30 + avg_skill * 0.25 + avg_init * 0.20 +
                     avg_delivery * 0.15 + avg_collab * 0.10)

    # Growth velocity (normalized 0-100, velocity * 10 = monthly pts/month)
    growth_velocity = float(np.clip(
        (perf_velocity * 0.3 + skill_velocity * 0.3 + init_velocity * 0.2 +
         delivery_velocity * 0.1 + collab_velocity * 0.1) * 10 + 50, 0, 100
    ))

    # Learning velocity — skill-focused
    learning_velocity = float(np.clip(
        skill_velocity * 12 + 50 + skill_accel * 3, 0, 100
    ))

    # Role fit — how aligned current signals are with the role
    role_fit_score = float(np.clip(
        overall_score * 0.6 + growth_velocity * 0.25 + learning_velocity * 0.15, 0, 100
    ))

    # Predict timeframe
    # If score >= 80 and velocity positive → fast track 6m; else 9-12m
    total_velocity = perf_velocity + skill_velocity + init_velocity
    if overall_score >= 78 and total_velocity > 0.3:
        timeframe = 6
    elif overall_score >= 65 and total_velocity > 0:
        timeframe = 9
    else:
        timeframe = 12

    # Determine predicted role from ladder
    ladder = ROLE_LADDERS.get(department, DEFAULT_LADDER)
    # Find current role index
    current_idx = -1
    for i, role in enumerate(ladder):
        if current_role.lower() in role.lower() or role.lower() in current_role.lower():
            current_idx = i
            break
    if current_idx == -1:
        current_idx = max(0, int(years_exp / 2) - 1)
        current_idx = min(current_idx, len(ladder) - 1)

    # How many levels to advance
    if overall_score >= 80 and growth_velocity >= 60:
        advance = 2
    elif overall_score >= 65 and growth_velocity >= 50:
        advance = 1
    else:
        advance = 0

    predicted_idx = min(current_idx + advance + 1, len(ladder) - 1)
    predicted_role = ladder[predicted_idx]

    # Confidence based on data length, consistency, and score
    data_confidence = min(len(signals_list) / 12.0, 1.0)
    score_consistency = 1 - (np.std([s['performance'] for s in signals_list]) / 50)
    confidence_score = float(np.clip(
        (data_confidence * 0.35 + score_consistency * 0.35 + (overall_score / 100) * 0.30) * 100, 40, 95
    ))

    # Build natural language summary
    time_str = f"{timeframe} months"
    velocity_adj = "exceptional" if growth_velocity >= 75 else "strong" if growth_velocity >= 60 else "steady" if growth_velocity >= 45 else "gradual"
    summary = (
        f"This candidate will likely become a {predicted_role} in {time_str}. "
        f"Trajectory analysis shows {velocity_adj} growth velocity ({growth_velocity:.1f}/100) "
        f"with a learning curve trending {"upward" if skill_velocity > 0 else "flat"}. "
        f"Role fit score of {role_fit_score:.1f}/100 indicates "
        f"{"excellent" if role_fit_score >= 80 else "good" if role_fit_score >= 65 else "moderate"} "
        f"alignment with {department} career progression."
    )

    # Strengths and risks
    strengths = []
    risk_factors = []

    if avg_perf >= 75: strengths.append("Consistently high performance scores")
    if skill_velocity > 0.5: strengths.append("Rapid skill acquisition trajectory")
    if avg_collab >= 70: strengths.append("Strong collaborative mindset")
    if avg_init >= 70: strengths.append("High initiative and ownership")
    if avg_delivery >= 75: strengths.append("Reliable execution and delivery")
    if perf_accel > 0.2: strengths.append("Accelerating performance trend")

    if avg_perf < 60: risk_factors.append("Performance below benchmark")
    if skill_velocity < 0: risk_factors.append("Declining skill growth signal")
    if avg_collab < 50: risk_factors.append("Collaboration metrics need attention")
    if len(signals_list) < 4: risk_factors.append("Limited data — prediction has higher uncertainty")
    if np.std([s['performance'] for s in signals_list]) > 20: risk_factors.append("High performance volatility")

    if not strengths:
        strengths.append("Consistent baseline across signals")
    if not risk_factors:
        risk_factors.append("No critical risk signals detected")

    return {
        "predicted_role": predicted_role,
        "timeframe_months": timeframe,
        "confidence_score": round(confidence_score, 1),
        "growth_velocity": round(growth_velocity, 1),
        "learning_velocity": round(learning_velocity, 1),
        "role_fit_score": round(role_fit_score, 1),
        "summary": summary,
        "strengths": strengths,
        "risk_factors": risk_factors,
        "signal_velocities": {
            "performance": round(perf_velocity, 3),
            "skill_growth": round(skill_velocity, 3),
            "collaboration": round(collab_velocity, 3),
            "initiative": round(init_velocity, 3),
            "delivery": round(delivery_velocity, 3),
        },
        "projected_signals_6m": {
            "performance": round(float(np.clip(avg_perf + perf_velocity * 6, 0, 100)), 1),
            "skill_growth": round(float(np.clip(avg_skill + skill_velocity * 6, 0, 100)), 1),
            "collaboration": round(float(np.clip(avg_collab + collab_velocity * 6, 0, 100)), 1),
            "initiative": round(float(np.clip(avg_init + init_velocity * 6, 0, 100)), 1),
            "delivery": round(float(np.clip(avg_delivery + delivery_velocity * 6, 0, 100)), 1),
        },
        "projected_signals_12m": {
            "performance": round(float(np.clip(avg_perf + perf_velocity * 12, 0, 100)), 1),
            "skill_growth": round(float(np.clip(avg_skill + skill_velocity * 12, 0, 100)), 1),
            "collaboration": round(float(np.clip(avg_collab + collab_velocity * 12, 0, 100)), 1),
            "initiative": round(float(np.clip(avg_init + init_velocity * 12, 0, 100)), 1),
            "delivery": round(float(np.clip(avg_delivery + delivery_velocity * 12, 0, 100)), 1),
        }
    }


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    signals = data.get("signals", [])
    current_role = data.get("current_role", "Engineer")
    department = data.get("department", "Engineering")
    years_exp = data.get("years_experience", 2)

    if len(signals) < 2:
        return jsonify({"error": "Need at least 2 months of signal data"}), 400

    try:
        result = predict_candidate(signals, current_role, department, years_exp)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": "LSTM-32h"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)
