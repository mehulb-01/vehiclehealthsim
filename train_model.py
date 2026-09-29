import numpy as np, json
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

rng = np.random.default_rng(42)
N = 12000

# ---- sensors: (name, safe_low, safe_high, danger_low, danger_high, unit) ----
SENSORS = [
    ("engine_temp",      80, 105,  60, 125, "°C"),
    ("rpm",              700, 3200, 500, 6000, "rpm"),
    ("oil_pressure",     25, 65,   10, 80,  "psi"),
    ("vibration",        0.2, 2.5, 0,  7.0, "mm/s"),
    ("battery_voltage",  12.4, 14.4, 10.5, 15.5, "V"),
    ("coolant_temp",     80, 100,  60, 120, "°C"),
    ("fuel_consumption", 6, 10,    4,  16,  "L/100km"),
    ("vehicle_speed",    0, 120,   0,  180, "km/h"),
    ("operating_hours",  0, 4000,  0,  9000, "hrs"),
]
NAMES = [s[0] for s in SENSORS]

def risk(v, lo, hi, dlo, dhi):
    """0 inside safe band, ->1 approaching/through danger bounds."""
    r = np.zeros_like(v, dtype=float)
    below = v < lo
    above = v > hi
    r[below] = np.clip((lo - v[below]) / max(lo - dlo, 1e-6), 0, 1.4)
    r[above] = np.clip((v[above] - hi) / max(dhi - hi, 1e-6), 0, 1.4)
    return r

# simulate raw sensor values: mostly healthy, with a degrading fraction
raw = {}
frac_degraded = 0.35
degraded = rng.random(N) < frac_degraded
severity = np.where(degraded, rng.beta(2, 2, N), rng.beta(0.6, 6, N))  # 0..1 wear level

for name, lo, hi, dlo, dhi, unit in SENSORS:
    center = (lo + hi) / 2
    span = (hi - lo) / 2
    base = rng.normal(center, span * 0.35, N)
    drift_dir = 1 if name in ("engine_temp","rpm","vibration","coolant_temp","fuel_consumption","operating_hours") else -1
    if name == "battery_voltage":
        drift = -severity * (span * 1.6) * (rng.random(N) < 0.7) + severity * (span*1.6)*0.3*(rng.random(N)>=0.7)*1  # mostly drops, sometimes overcharges
    elif name == "oil_pressure":
        drift = -severity * span * 2.2
    elif name == "vehicle_speed":
        drift = rng.normal(0, span*0.2, N)  # not a wear indicator
    elif name == "operating_hours":
        drift = severity * (dhi - hi) * rng.random(N)
    else:
        drift = drift_dir * severity * span * 2.0
    val = base + drift + rng.normal(0, span*0.05, N)
    raw[name] = val

X_risk = np.stack([risk(raw[n], lo, hi, dlo, dhi) for n, lo, hi, dlo, dhi, u in SENSORS], axis=1)
# vehicle_speed excluded from the failure-risk score (context only, not a wear signal)
speed_idx = NAMES.index("vehicle_speed")
score_idx = [i for i in range(len(NAMES)) if i != speed_idx]

domain_weight = np.array([1.3, 0.8, 1.5, 1.6, 1.1, 1.2, 0.9, 0.0, 0.7])  # vehicle_speed weight=0
true_logit = -3.4 + (X_risk * domain_weight).sum(axis=1) + rng.normal(0, 0.4, N)
prob_true = 1 / (1 + np.exp(-true_logit))
y = (rng.random(N) < prob_true).astype(int)

X = X_risk[:, score_idx]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

logreg = LogisticRegression(max_iter=2000).fit(Xtr, ytr)
rf = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42).fit(Xtr, ytr)

def report(name, model):
    p = model.predict(Xte)
    pr = model.predict_proba(Xte)[:, 1]
    return dict(model=name, accuracy=round(accuracy_score(yte,p),4),
                precision=round(precision_score(yte,p),4), recall=round(recall_score(yte,p),4),
                f1=round(f1_score(yte,p),4), roc_auc=round(roc_auc_score(yte,pr),4))

results = [report("LogisticRegression", logreg), report("RandomForest", rf)]
for r in results: print(r)

coef = logreg.coef_[0]
intercept = logreg.intercept_[0]
out = {
    "feature_order": [NAMES[i] for i in score_idx],
    "sensors": {n: {"safe_low": lo, "safe_high": hi, "danger_low": dlo, "danger_high": dhi, "unit": u}
                for n, lo, hi, dlo, dhi, u in SENSORS},
    "weights": {NAMES[score_idx[i]]: round(float(coef[i]), 4) for i in range(len(score_idx))},
    "intercept": round(float(intercept), 4),
    "metrics": results,
}
json.dump(out, open("model.json", "w"), indent=2)
print(json.dumps(out, indent=2))
