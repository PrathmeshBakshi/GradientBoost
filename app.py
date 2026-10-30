"""
PayScope - Employee Monthly Salary Predictor
Flask app for a Gradient Boosting model (GradientBoost.pkl). Vercel-ready.
"""
import json
import os
import pickle
import warnings

import numpy as np
from flask import Flask, jsonify, request, Response

warnings.filterwarnings("ignore")  # hides sklearn "feature names" warnings

# --------------------------------------------------------------------------
# Load model
# --------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "GradientBoost.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

FEATURES = list(model.feature_names_in_)
FEATURE_INDEX = {name: i for i, name in enumerate(FEATURES)}

# The model was trained with pd.get_dummies(..., drop_first=True), so ONE
# category per column is "missing" (all zeros). That is the baseline below.
# If a label is wrong, check it with: df['Department'].unique() in your notebook
# (the baseline is the alphabetically-first value that is NOT in the dummy list).
BASELINES = {
    "Department": "Finance",
    "Education_Level": "Bachelor",
    "Gender": "Female",
    "City": "Bangalore",
}

LABELS = {
    "Bachelor": "Bachelor's degree",
    "Master": "Master's degree",
    "PhD": "PhD",
    "High School": "High School",
}

AGE_MIN, AGE_MAX = 18, 65
EXP_MAX = 45
MIN_WORKING_AGE = 16  # experience can't exceed age - 16


def _options(group):
    prefix = group + "_"
    found = [f[len(prefix):] for f in FEATURES if f.startswith(prefix)]
    return [BASELINES[group]] + found


OPTIONS = {g: _options(g) for g in BASELINES}

app = Flask(__name__)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def inr(n):
    """Format a number in Indian grouping: 1234567 -> ₹12,34,567"""
    n = int(round(n))
    s = str(abs(n))
    if len(s) > 3:
        head, tail = s[:-3], s[-3:]
        parts = []
        while len(head) > 2:
            parts.insert(0, head[-2:])
            head = head[:-2]
        if head:
            parts.insert(0, head)
        s = ",".join(parts + [tail])
    return ("-" if n < 0 else "") + "₹" + s


def signed_inr(n):
    return ("+" if n >= 0 else "−") + inr(abs(n))


def parse_input(data):
    errors = {}
    out = {}

    try:
        age = int(float(data.get("age")))
        if not AGE_MIN <= age <= AGE_MAX:
            raise ValueError
        out["age"] = age
    except (TypeError, ValueError):
        errors["age"] = f"Age must be between {AGE_MIN} and {AGE_MAX}."

    try:
        exp = float(data.get("experience"))
        if not 0 <= exp <= EXP_MAX:
            raise ValueError
        out["experience"] = round(exp, 1)
    except (TypeError, ValueError):
        errors["experience"] = f"Experience must be between 0 and {EXP_MAX} years."

    if "age" in out and "experience" in out:
        if out["experience"] > out["age"] - MIN_WORKING_AGE:
            errors["experience"] = (
                f"Experience can't be more than {out['age'] - MIN_WORKING_AGE} years "
                f"for someone aged {out['age']}."
            )

    keymap = {
        "department": "Department",
        "education": "Education_Level",
        "gender": "Gender",
        "city": "City",
    }
    for field, group in keymap.items():
        val = data.get(field)
        if val not in OPTIONS[group]:
            errors[field] = f"Please choose a valid {field}."
        else:
            out[group] = val

    return out, errors


def to_vector(p):
    v = np.zeros(len(FEATURES))
    v[FEATURE_INDEX["Experience_Years"]] = p["experience"]
    v[FEATURE_INDEX["Age"]] = p["age"]
    for group, baseline in BASELINES.items():
        if p[group] != baseline:
            v[FEATURE_INDEX[f"{group}_{p[group]}"]] = 1
    return v


def predict_many(profiles):
    X = np.vstack([to_vector(p) for p in profiles])
    return np.clip(model.predict(X), 0, None)


def career_level(exp):
    if exp < 1:
        return "Fresher"
    if exp < 3:
        return "Junior"
    if exp < 8:
        return "Mid-level"
    if exp < 15:
        return "Senior"
    return "Veteran"


def compare(p, group):
    profiles = [{**p, group: opt} for opt in OPTIONS[group]]
    preds = predict_many(profiles)
    rows = [
        {
            "label": LABELS.get(opt, opt),
            "value": float(v),
            "fmt": inr(v),
            "current": opt == p[group],
        }
        for opt, v in zip(OPTIONS[group], preds)
    ]
    return sorted(rows, key=lambda r: r["value"], reverse=True)


def build_response(p):
    monthly = float(predict_many([p])[0])
    annual = monthly * 12
    low, high = monthly * 0.9, monthly * 1.1

    comp = {
        "city": compare(p, "City"),
        "department": compare(p, "Department"),
        "education": compare(p, "Education_Level"),
    }

    # 5-year outlook (age and experience both grow)
    outlook = []
    for n in range(0, 6):
        if p["age"] + n > 70:
            break
        q = {**p, "age": p["age"] + n, "experience": round(p["experience"] + n, 1)}
        outlook.append({"year": n, "age": q["age"], "monthly": float(predict_many([q])[0])})
    for o in outlook:
        o["fmt"] = inr(o["monthly"])

    level = career_level(p["experience"])
    edu = LABELS.get(p["Education_Level"], p["Education_Level"])
    exp_txt = (
        f"{p['experience']:g} year{'s' if p['experience'] != 1 else ''} of experience"
        if p["experience"] > 0
        else "no prior work experience"
    )
    summary = (
        f"For a {p['age']}-year-old {p['Gender'].lower()} professional with {exp_txt}, "
        f"working in {p['Department']} in {p['City']} with a {edu}, "
        f"the model estimates a monthly salary of about {inr(monthly)} "
        f"(roughly {annual / 100000:.1f} lakh per year). "
        f"A realistic range is {inr(low)} to {inr(high)} per month."
    )

    # Human-readable insights, all computed from the model itself
    insights = []
    top_city = comp["city"][0]
    if not top_city["current"]:
        insights.append(
            f"Same profile in {top_city['label']} is predicted at "
            f"{signed_inr(top_city['value'] - monthly)} per month versus {p['City']}."
        )
    else:
        insights.append(f"{p['City']} is already the highest-paying city for this profile.")

    top_dep = comp["department"][0]
    if not top_dep["current"]:
        insights.append(
            f"{top_dep['label']} is the best-paying department for this profile "
            f"({signed_inr(top_dep['value'] - monthly)} per month vs {p['Department']})."
        )
    else:
        insights.append(f"{p['Department']} is already the best-paying department for this profile.")

    if len(outlook) > 1:
        diff = outlook[-1]["monthly"] - monthly
        direction = "grow" if diff >= 0 else "dip"
        insights.append(
            f"In {outlook[-1]['year']} years the model expects your salary to {direction} "
            f"by {inr(abs(diff))} per month (to about {outlook[-1]['fmt']})."
        )

    return {
        "monthly": monthly,
        "monthly_fmt": inr(monthly),
        "annual_fmt": inr(annual),
        "annual_lakh": f"{annual / 100000:.1f}",
        "low_fmt": inr(low),
        "high_fmt": inr(high),
        "level": level,
        "summary": summary,
        "insights": insights,
        "compare": comp,
        "outlook": outlook,
        "profile": {
            "age": p["age"],
            "experience": p["experience"],
            "department": p["Department"],
            "education": edu,
            "gender": p["Gender"],
            "city": p["City"],
        },
    }


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------
@app.route("/")
def index():
    config = {
        "options": OPTIONS,
        "labels": LABELS,
        "ageMin": AGE_MIN,
        "ageMax": AGE_MAX,
        "expMax": EXP_MAX,
        "minWorkingAge": MIN_WORKING_AGE,
    }
    html = PAGE.replace("__CONFIG__", json.dumps(config))
    return Response(html, mimetype="text/html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    parsed, errors = parse_input(data)
    if errors:
        return jsonify({"ok": False, "errors": errors}), 400
    try:
        return jsonify({"ok": True, **build_response(parsed)})
    except Exception as exc:  # pragma: no cover
        return jsonify({"ok": False, "errors": {"server": str(exc)}}), 500


@app.route("/health")
def health():
    return jsonify({"status": "ok", "features": len(FEATURES)})


# --------------------------------------------------------------------------
# Front-end (single file so it deploys cleanly on Vercel)
# --------------------------------------------------------------------------
PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PayScope - Salary Predictor</title>
<meta name="description" content="Predict monthly salary with a Gradient Boosting machine learning model.">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<script>
  (function () {
    var t = null;
    try { t = localStorage.getItem("theme"); } catch (e) {}
    if (!t) t = window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
    document.documentElement.setAttribute("data-theme", t);
  })();
</script>
<style>
  :root {
    --bg: #f4f6f8; --bg2: #e9eef2;
    --card: #ffffff; --card2: #f7f9fa;
    --text: #0f1b24; --muted: #5b6b78; --line: #dde4ea;
    --accent: #0b8f7a; --accent2: #0a6f5f; --accent-soft: #dcf4ee; --on-accent: #ffffff;
    --warn: #c2410c; --warn-soft: #fff1e8;
    --shadow: 0 1px 2px rgba(15,27,36,.06), 0 8px 30px rgba(15,27,36,.06);
    --radius: 18px;
  }
  [data-theme="dark"] {
    --bg: #0b1217; --bg2: #101a21;
    --card: #131e26; --card2: #182630;
    --text: #e8eff4; --muted: #92a4b1; --line: #25343f;
    --accent: #2dd4b0; --accent2: #5eead4; --accent-soft: #11332e; --on-accent: #04251f;
    --warn: #fb923c; --warn-soft: #2f1d12;
    --shadow: 0 1px 2px rgba(0,0,0,.4), 0 10px 40px rgba(0,0,0,.35);
  }
  * { box-sizing: border-box; }
  html { color-scheme: light; }
  [data-theme="dark"] { color-scheme: dark; }
  body {
    margin: 0; font-family: Inter, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
    background: var(--bg); color: var(--text); line-height: 1.5; -webkit-font-smoothing: antialiased;
    transition: background .25s, color .25s;
  }
  body::before {
    content: ""; position: fixed; inset: 0; z-index: -1; pointer-events: none;
    background:
      radial-gradient(900px 400px at 85% -10%, color-mix(in srgb, var(--accent) 16%, transparent), transparent 70%),
      radial-gradient(700px 400px at -10% 10%, color-mix(in srgb, var(--accent) 9%, transparent), transparent 70%);
  }
  .wrap { max-width: 1120px; margin: 0 auto; padding: 0 16px 56px; }

  header { display: flex; align-items: center; justify-content: space-between; padding: 20px 0; }
  .brand { display: flex; align-items: center; gap: 10px; font-weight: 800; font-size: 20px; letter-spacing: -.02em; }
  .logo { width: 34px; height: 34px; border-radius: 10px; background: var(--accent); color: var(--on-accent);
          display: grid; place-items: center; }
  .icon-btn { border: 1px solid var(--line); background: var(--card); color: var(--text); width: 42px; height: 42px;
              border-radius: 12px; cursor: pointer; display: grid; place-items: center; transition: .2s; }
  .icon-btn:hover { border-color: var(--accent); color: var(--accent); }
  .icon-btn svg { width: 20px; height: 20px; }
  .sun { display: none; } [data-theme="dark"] .sun { display: block; } [data-theme="dark"] .moon { display: none; }

  .hero { padding: 14px 0 26px; max-width: 720px; }
  .hero h1 { margin: 0 0 8px; font-size: clamp(28px, 5vw, 44px); line-height: 1.1; letter-spacing: -.03em; font-weight: 800; }
  .hero h1 span { color: var(--accent); }
  .hero p { margin: 0; color: var(--muted); font-size: 17px; }

  .grid { display: grid; grid-template-columns: minmax(0, 5fr) minmax(0, 6fr); gap: 20px; align-items: start; }
  @media (max-width: 900px) { .grid { grid-template-columns: 1fr; } }

  .card { background: var(--card); border: 1px solid var(--line); border-radius: var(--radius); padding: 22px; box-shadow: var(--shadow); }
  .card h2 { margin: 0 0 4px; font-size: 18px; letter-spacing: -.01em; }
  .sub { margin: 0 0 18px; color: var(--muted); font-size: 14px; }

  .presets { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 18px; }
  .chip { border: 1px solid var(--line); background: var(--card2); color: var(--text); border-radius: 999px;
          padding: 6px 12px; font-size: 13px; font-weight: 500; cursor: pointer; transition: .15s; font-family: inherit; }
  .chip:hover { border-color: var(--accent); color: var(--accent); }

  .field { margin-bottom: 18px; }
  .field label, .field .lbl { display: flex; justify-content: space-between; align-items: baseline; font-size: 13px;
                              font-weight: 600; margin-bottom: 8px; }
  .field .hint { font-weight: 400; color: var(--muted); font-size: 12px; }
  .row { display: grid; grid-template-columns: 1fr 84px; gap: 12px; align-items: center; }
  input[type=range] { width: 100%; accent-color: var(--accent); height: 28px; cursor: pointer; }
  input[type=number], select {
    width: 100%; padding: 11px 12px; border-radius: 12px; border: 1px solid var(--line); background: var(--card2);
    color: var(--text); font: inherit; font-size: 15px; outline: none; transition: .15s;
  }
  input[type=number]:focus, select:focus { border-color: var(--accent); box-shadow: 0 0 0 3px var(--accent-soft); }
  .two { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
  @media (max-width: 480px) { .two { grid-template-columns: 1fr; } }

  .seg { display: grid; grid-template-columns: 1fr 1fr; background: var(--card2); border: 1px solid var(--line);
         border-radius: 12px; padding: 4px; gap: 4px; }
  .seg input { position: absolute; opacity: 0; pointer-events: none; }
  .seg label { margin: 0; justify-content: center; padding: 9px; border-radius: 9px; cursor: pointer; font-weight: 600;
               color: var(--muted); transition: .15s; text-align: center; display: block; }
  .seg input:checked + label { background: var(--accent); color: var(--on-accent); }
  .seg input:focus-visible + label { outline: 2px solid var(--accent); outline-offset: 2px; }

  .err { color: var(--warn); font-size: 12.5px; margin-top: 6px; min-height: 0; }
  .actions { display: flex; gap: 10px; margin-top: 8px; }
  .btn { border: 0; border-radius: 12px; padding: 13px 18px; font: inherit; font-weight: 700; font-size: 15px; cursor: pointer; transition: .15s; }
  .btn-primary { flex: 1; background: var(--accent); color: var(--on-accent); display: flex; align-items: center; justify-content: center; gap: 8px; }
  .btn-primary:hover { background: var(--accent2); transform: translateY(-1px); }
  .btn-primary:disabled { opacity: .7; cursor: progress; transform: none; }
  .btn-ghost { background: transparent; border: 1px solid var(--line); color: var(--text); }
  .btn-ghost:hover { border-color: var(--accent); color: var(--accent); }
  .spin { width: 16px; height: 16px; border: 2px solid currentColor; border-right-color: transparent; border-radius: 50%; animation: sp .7s linear infinite; display: none; }
  .loading .spin { display: inline-block; }
  @keyframes sp { to { transform: rotate(360deg); } }

  /* Result */
  .empty { text-align: center; padding: 56px 12px; color: var(--muted); }
  .empty .ring { width: 64px; height: 64px; margin: 0 auto 14px; border-radius: 20px; background: var(--accent-soft); color: var(--accent); display: grid; place-items: center; }
  .empty h3 { margin: 0 0 6px; color: var(--text); }
  #result { display: none; animation: rise .45s ease; }
  @keyframes rise { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: none; } }
  .badge { display: inline-block; background: var(--accent-soft); color: var(--accent); border-radius: 999px; padding: 4px 11px; font-size: 12.5px; font-weight: 700; }
  .big { font-size: clamp(38px, 7vw, 58px); font-weight: 800; letter-spacing: -.035em; line-height: 1.05; margin: 10px 0 2px; color: var(--accent); }
  .big small { font-size: 16px; color: var(--muted); font-weight: 600; letter-spacing: 0; }
  .annual { color: var(--muted); font-size: 15px; }
  .annual b { color: var(--text); }

  .range { margin: 20px 0 6px; }
  .track { height: 10px; border-radius: 99px; background: var(--bg2); position: relative; overflow: hidden; }
  .track i { position: absolute; left: 12%; right: 12%; top: 0; bottom: 0; background: linear-gradient(90deg, var(--accent-soft), var(--accent), var(--accent-soft)); border-radius: 99px; }
  .range-lbl { display: flex; justify-content: space-between; font-size: 12.5px; color: var(--muted); margin-top: 6px; }

  .summary { background: var(--card2); border: 1px solid var(--line); border-radius: 14px; padding: 14px 16px; margin: 18px 0; font-size: 15px; }
  .facts { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 4px; }
  .fact { font-size: 12.5px; background: var(--card2); border: 1px solid var(--line); padding: 5px 10px; border-radius: 9px; color: var(--muted); }
  .fact b { color: var(--text); font-weight: 600; }

  h4 { margin: 24px 0 10px; font-size: 14px; letter-spacing: .01em; }
  .insights { list-style: none; padding: 0; margin: 0; display: grid; gap: 8px; }
  .insights li { padding: 10px 12px 10px 38px; position: relative; background: var(--card2); border: 1px solid var(--line); border-radius: 12px; font-size: 14px; }
  .insights li::before { content: "i"; position: absolute; left: 11px; top: 10px; width: 18px; height: 18px; border-radius: 50%; background: var(--accent); color: var(--on-accent); font-size: 11px; font-weight: 800; display: grid; place-items: center; }

  .tabs { display: inline-flex; background: var(--card2); border: 1px solid var(--line); border-radius: 11px; padding: 3px; gap: 3px; margin-bottom: 12px; }
  .tab { border: 0; background: transparent; color: var(--muted); font: inherit; font-size: 13px; font-weight: 600; padding: 6px 12px; border-radius: 8px; cursor: pointer; }
  .tab[aria-selected="true"] { background: var(--accent); color: var(--on-accent); }
  .bars { display: grid; gap: 9px; }
  .bar { display: grid; grid-template-columns: 112px 1fr 84px; gap: 10px; align-items: center; font-size: 13px; }
  @media (max-width: 480px) { .bar { grid-template-columns: 84px 1fr 74px; font-size: 12px; } }
  .bar .name { color: var(--muted); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .bar .name.cur { color: var(--text); font-weight: 700; }
  .bar .fill { height: 10px; border-radius: 99px; background: var(--line); width: 0; transition: width .7s cubic-bezier(.2,.8,.2,1); }
  .bar .fill.cur { background: var(--accent); }
  .bar .val { text-align: right; font-variant-numeric: tabular-nums; font-weight: 600; }
  .holder { background: var(--bg2); border-radius: 99px; }

  svg.outlook { width: 100%; height: auto; display: block; }
  .outlook .axis { fill: var(--muted); font-size: 10px; font-family: inherit; }
  .outlook .val { fill: var(--text); font-size: 10px; font-weight: 600; font-family: inherit; }
  .outlook .line { fill: none; stroke: var(--accent); stroke-width: 2.5; stroke-linecap: round; stroke-linejoin: round; }
  .outlook .area { fill: var(--accent-soft); opacity: .8; }
  .outlook .dot { fill: var(--card); stroke: var(--accent); stroke-width: 2.5; }

  .res-actions { display: flex; gap: 10px; margin-top: 22px; flex-wrap: wrap; }
  .res-actions .btn { padding: 10px 14px; font-size: 14px; }

  .history { margin-top: 20px; }
  .history h3 { margin: 0 0 10px; font-size: 15px; display: flex; justify-content: space-between; align-items: center; }
  .history button.link { background: none; border: 0; color: var(--muted); cursor: pointer; font: inherit; font-size: 12.5px; }
  .history button.link:hover { color: var(--accent); }
  .hlist { display: grid; gap: 8px; }
  .hitem { text-align: left; display: flex; justify-content: space-between; gap: 10px; background: var(--card); border: 1px solid var(--line); color: var(--text);
           padding: 10px 14px; border-radius: 12px; cursor: pointer; font: inherit; font-size: 13.5px; transition: .15s; }
  .hitem:hover { border-color: var(--accent); }
  .hitem span { color: var(--muted); }
  .hitem b { color: var(--accent); white-space: nowrap; }

  .note { margin-top: 26px; font-size: 12.5px; color: var(--muted); text-align: center; }
  .toast { position: fixed; left: 50%; bottom: 24px; transform: translate(-50%, 30px); background: var(--text); color: var(--bg); padding: 10px 16px;
           border-radius: 12px; font-size: 14px; font-weight: 500; opacity: 0; pointer-events: none; transition: .3s; z-index: 10; }
  .toast.show { opacity: 1; transform: translate(-50%, 0); }
  @media (prefers-reduced-motion: reduce) { * { animation: none !important; transition: none !important; } }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div class="brand">
      <div class="logo"><svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="M3 17l5-5 4 4 8-9"/><path d="M15 7h5v5"/></svg></div>
      PayScope
    </div>
    <button class="icon-btn" id="themeBtn" aria-label="Toggle light and dark theme" title="Toggle theme">
      <svg class="moon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12.8A9 9 0 1111.2 3a7 7 0 009.8 9.8z"/></svg>
      <svg class="sun" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg>
    </button>
  </header>

  <section class="hero">
    <h1>Know what a role <span>should pay</span>.</h1>
    <p>Enter a few details and our Gradient Boosting model estimates the monthly salary, then shows how city, department and growth change it.</p>
  </section>

  <main class="grid">
    <!-- FORM -->
    <form class="card" id="form" novalidate>
      <h2>Employee profile</h2>
      <p class="sub">Try a quick example or fill in your own details.</p>
      <div class="presets" id="presets"></div>

      <div class="field">
        <label for="age">Age <span class="hint" id="ageHint"></span></label>
        <div class="row"><input type="range" id="ageR"><input type="number" id="age"></div>
        <div class="err" id="e-age"></div>
      </div>

      <div class="field">
        <label for="experience">Experience (years) <span class="hint" id="expHint"></span></label>
        <div class="row"><input type="range" id="expR" step="0.5"><input type="number" id="experience" step="0.5"></div>
        <div class="err" id="e-experience"></div>
      </div>

      <div class="field">
        <div class="lbl">Gender</div>
        <div class="seg" id="gender" role="radiogroup" aria-label="Gender"></div>
      </div>

      <div class="two">
        <div class="field"><label for="department">Department</label><select id="department"></select></div>
        <div class="field"><label for="education">Education</label><select id="education"></select></div>
      </div>
      <div class="field"><label for="city">City</label><select id="city"></select></div>

      <div class="actions">
        <button class="btn btn-primary" id="go" type="submit"><span class="spin"></span><span id="goTxt">Predict salary</span></button>
        <button class="btn btn-ghost" id="reset" type="button">Reset</button>
      </div>
    </form>

    <!-- RESULT -->
    <div>
      <section class="card" aria-live="polite">
        <div class="empty" id="empty">
          <div class="ring"><svg width="30" height="30" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2v20M17 6H9.5a3.5 3.5 0 000 7h5a3.5 3.5 0 010 7H6"/></svg></div>
          <h3>Your estimate will appear here</h3>
          <p>Fill in the profile and press <b>Predict salary</b>.</p>
        </div>

        <div id="result">
          <span class="badge" id="level"></span>
          <div class="big"><span id="monthly">₹0</span> <small>/ month</small></div>
          <div class="annual">About <b id="annual"></b> per year (<b id="lakh"></b> lakh)</div>

          <div class="range">
            <div class="track"><i></i></div>
            <div class="range-lbl"><span id="low"></span><span>likely range (±10%)</span><span id="high"></span></div>
          </div>

          <div class="summary" id="summary"></div>
          <div class="facts" id="facts"></div>

          <h4>Key takeaways</h4>
          <ul class="insights" id="insights"></ul>

          <h4>What if you changed one thing?</h4>
          <div class="tabs" role="tablist">
            <button class="tab" role="tab" data-k="city" aria-selected="true" type="button">City</button>
            <button class="tab" role="tab" data-k="department" aria-selected="false" type="button">Department</button>
            <button class="tab" role="tab" data-k="education" aria-selected="false" type="button">Education</button>
          </div>
          <div class="bars" id="bars"></div>

          <h4>Next 5 years (age and experience grow together)</h4>
          <svg class="outlook" id="outlook" viewBox="0 0 340 140" role="img" aria-label="Five year salary outlook"></svg>

          <div class="res-actions">
            <button class="btn btn-ghost" id="copy" type="button">Copy summary</button>
            <button class="btn btn-ghost" id="print" type="button">Print / Save PDF</button>
          </div>
        </div>
      </section>

      <section class="history" id="historyBox" style="display:none">
        <h3>Recent predictions <button class="link" id="clearHist" type="button">Clear</button></h3>
        <div class="hlist" id="hist"></div>
      </section>
    </div>
  </main>

  <p class="note">Estimates come from a machine-learning model trained on a sample dataset. They are a guide, not a salary offer.</p>
</div>
<div class="toast" id="toast" role="status"></div>

<script>
const CFG = __CONFIG__;
const $ = (s) => document.querySelector(s);
const inrFmt = new Intl.NumberFormat("en-IN");
const label = (v) => CFG.labels[v] || v;
let lastData = null, activeTab = "city";

/* ---------- theme ---------- */
$("#themeBtn").addEventListener("click", () => {
  const next = document.documentElement.getAttribute("data-theme") === "dark" ? "light" : "dark";
  document.documentElement.setAttribute("data-theme", next);
  try { localStorage.setItem("theme", next); } catch (e) {}
});

/* ---------- toast ---------- */
let tt;
function toast(msg) {
  const t = $("#toast"); t.textContent = msg; t.classList.add("show");
  clearTimeout(tt); tt = setTimeout(() => t.classList.remove("show"), 2200);
}

/* ---------- build form ---------- */
function fillSelect(id, group) {
  $(id).innerHTML = CFG.options[group].map(o => `<option value="${o}">${label(o)}</option>`).join("");
}
fillSelect("#department", "Department");
fillSelect("#education", "Education_Level");
fillSelect("#city", "City");
$("#gender").innerHTML = CFG.options.Gender.map((g, i) =>
  `<input type="radio" name="gender" id="g${i}" value="${g}" ${i === 0 ? "checked" : ""}><label for="g${i}">${g}</label>`).join("");

const ageR = $("#ageR"), age = $("#age"), expR = $("#expR"), exp = $("#experience");
ageR.min = age.min = CFG.ageMin; ageR.max = age.max = CFG.ageMax;
expR.min = exp.min = 0; expR.max = exp.max = CFG.expMax;

function syncLimits() {
  const maxExp = Math.max(0, Math.min(CFG.expMax, (+age.value || CFG.ageMin) - CFG.minWorkingAge));
  expR.max = exp.max = maxExp;
  if (+exp.value > maxExp) { exp.value = expR.value = maxExp; }
  $("#expHint").textContent = "max " + maxExp + " for this age";
}
function bindPair(range, num, after) {
  range.addEventListener("input", () => { num.value = range.value; after && after(); });
  num.addEventListener("input", () => { if (num.value !== "") range.value = num.value; after && after(); });
}
bindPair(ageR, age, syncLimits);
bindPair(expR, exp);

const PRESETS = [
  { name: "Fresher in IT", age: 23, experience: 0.5, department: "IT", education: "Bachelor", gender: "Male", city: "Bangalore" },
  { name: "Mid-career HR", age: 32, experience: 7, department: "HR", education: "Master", gender: "Female", city: "Mumbai" },
  { name: "Senior Ops lead", age: 41, experience: 15, department: "Operations", education: "Bachelor", gender: "Male", city: "Delhi" },
];
function setForm(v) {
  age.value = ageR.value = v.age; syncLimits();
  exp.value = expR.value = v.experience;
  $("#department").value = v.department; $("#education").value = v.education; $("#city").value = v.city;
  const g = document.querySelector(`input[name=gender][value="${v.gender}"]`); if (g) g.checked = true;
  Object.keys(CFG.options).length && document.querySelectorAll(".err").forEach(e => e.textContent = "");
}
function validOption(group, v) { return CFG.options[group].includes(v) ? v : CFG.options[group][0]; }
$("#presets").innerHTML = PRESETS.map((p, i) => `<button type="button" class="chip" data-i="${i}">${p.name}</button>`).join("");
$("#presets").addEventListener("click", (e) => {
  const b = e.target.closest(".chip"); if (!b) return;
  const p = { ...PRESETS[+b.dataset.i] };
  p.department = validOption("Department", p.department); p.education = validOption("Education_Level", p.education);
  p.gender = validOption("Gender", p.gender); p.city = validOption("City", p.city);
  setForm(p); submit();
});
const DEFAULT = { age: 28, experience: 4, department: CFG.options.Department[0], education: CFG.options.Education_Level[0], gender: CFG.options.Gender[0], city: CFG.options.City[0] };
setForm(DEFAULT);
$("#reset").addEventListener("click", () => {
  setForm(DEFAULT); $("#result").style.display = "none"; $("#empty").style.display = "block"; lastData = null;
});

const getForm = () => ({
  age: age.value, experience: exp.value, department: $("#department").value, education: $("#education").value,
  gender: document.querySelector("input[name=gender]:checked").value, city: $("#city").value,
});

/* ---------- submit ---------- */
$("#form").addEventListener("submit", (e) => { e.preventDefault(); submit(); });

async function submit() {
  document.querySelectorAll(".err").forEach(e => e.textContent = "");
  const btn = $("#go"); btn.disabled = true; btn.classList.add("loading"); $("#goTxt").textContent = "Predicting…";
  try {
    const res = await fetch("/predict", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(getForm()) });
    const data = await res.json();
    if (!data.ok) {
      Object.entries(data.errors || {}).forEach(([k, m]) => { const el = $("#e-" + k); if (el) el.textContent = m; });
      toast(Object.values(data.errors || { x: "Something went wrong" })[0]);
      return;
    }
    lastData = data; render(data); saveHistory(data);
    if (window.innerWidth < 900) $("#result").scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (err) {
    toast("Could not reach the server. Please try again.");
  } finally {
    btn.disabled = false; btn.classList.remove("loading"); $("#goTxt").textContent = "Predict salary";
  }
}

/* ---------- render ---------- */
function countUp(el, to) {
  const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (reduce) { el.textContent = "₹" + inrFmt.format(Math.round(to)); return; }
  const start = performance.now(), dur = 800;
  (function tick(now) {
    const p = Math.min(1, (now - start) / dur), eased = 1 - Math.pow(1 - p, 3);
    el.textContent = "₹" + inrFmt.format(Math.round(to * eased));
    if (p < 1) requestAnimationFrame(tick);
  })(start);
}

function render(d) {
  $("#empty").style.display = "none";
  const r = $("#result"); r.style.display = "none"; void r.offsetWidth; r.style.display = "block";
  $("#level").textContent = d.level + " professional";
  countUp($("#monthly"), d.monthly);
  $("#annual").textContent = d.annual_fmt; $("#lakh").textContent = d.annual_lakh;
  $("#low").textContent = d.low_fmt; $("#high").textContent = d.high_fmt;
  $("#summary").textContent = d.summary;
  const p = d.profile;
  $("#facts").innerHTML = [
    ["Age", p.age], ["Experience", p.experience + " yrs"], ["Dept", p.department],
    ["Education", p.education], ["City", p.city], ["Gender", p.gender],
  ].map(([k, v]) => `<span class="fact">${k}: <b>${v}</b></span>`).join("");
  $("#insights").innerHTML = d.insights.map(t => `<li>${t}</li>`).join("");
  renderBars(); renderOutlook(d.outlook);
}

function renderBars() {
  if (!lastData) return;
  const rows = lastData.compare[activeTab], max = Math.max(...rows.map(r => r.value));
  $("#bars").innerHTML = rows.map(r => `
    <div class="bar"><div class="name ${r.current ? "cur" : ""}" title="${r.label}">${r.label}${r.current ? " ✓" : ""}</div>
      <div class="holder"><div class="fill ${r.current ? "cur" : ""}" data-w="${(r.value / max * 100).toFixed(1)}"></div></div>
      <div class="val">${r.fmt}</div></div>`).join("");
  requestAnimationFrame(() => requestAnimationFrame(() =>
    document.querySelectorAll(".fill").forEach(f => f.style.width = f.dataset.w + "%")));
}
document.querySelectorAll(".tab").forEach(t => t.addEventListener("click", () => {
  activeTab = t.dataset.k;
  document.querySelectorAll(".tab").forEach(x => x.setAttribute("aria-selected", x === t));
  renderBars();
}));

function renderOutlook(pts) {
  const W = 340, H = 140, padX = 28, top = 22, bottom = 26;
  const vals = pts.map(p => p.monthly), min = Math.min(...vals), max = Math.max(...vals);
  const span = max - min || 1;
  const x = (i) => padX + i * ((W - padX * 2) / Math.max(1, pts.length - 1));
  const y = (v) => top + (1 - (v - min) / span) * (H - top - bottom);
  const line = pts.map((p, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(p.monthly).toFixed(1)}`).join(" ");
  const area = `${line} L${x(pts.length - 1)},${H - bottom} L${x(0)},${H - bottom} Z`;
  $("#outlook").innerHTML = `
    <path class="area" d="${area}"/><path class="line" d="${line}"/>
    ${pts.map((p, i) => `
      <circle class="dot" cx="${x(i)}" cy="${y(p.monthly)}" r="4.5"/>
      <text class="val" x="${x(i)}" y="${y(p.monthly) - 10}" text-anchor="middle">${(p.monthly / 1000).toFixed(0)}k</text>
      <text class="axis" x="${x(i)}" y="${H - 8}" text-anchor="middle">${i === 0 ? "Now" : "+" + p.year + "y"}</text>`).join("")}`;
}

/* ---------- copy / print ---------- */
$("#copy").addEventListener("click", async () => {
  if (!lastData) return;
  const text = `${lastData.summary}`;
  try { await navigator.clipboard.writeText(text); toast("Summary copied"); }
  catch (e) {
    const ta = document.createElement("textarea"); ta.value = text; document.body.appendChild(ta); ta.select();
    try { document.execCommand("copy"); toast("Summary copied"); } catch (_) { toast("Copy not supported"); }
    ta.remove();
  }
});
$("#print").addEventListener("click", () => window.print());

/* ---------- history ---------- */
const HKEY = "salary_history_v1";
function loadHist() { try { return JSON.parse(localStorage.getItem(HKEY)) || []; } catch (e) { return []; } }
function saveHist(h) { try { localStorage.setItem(HKEY, JSON.stringify(h)); } catch (e) {} }
function saveHistory(d) {
  const form = getForm();
  const h = loadHist().filter(x => JSON.stringify(x.form) !== JSON.stringify(form));
  h.unshift({ form, monthly: d.monthly_fmt, title: `${d.profile.department} · ${d.profile.city}`, sub: `${d.profile.age} yrs, ${d.profile.experience} yrs exp` });
  saveHist(h.slice(0, 5)); renderHist();
}
function renderHist() {
  const h = loadHist(), box = $("#historyBox");
  box.style.display = h.length ? "block" : "none";
  $("#hist").innerHTML = h.map((x, i) => `<button type="button" class="hitem" data-i="${i}"><div>${x.title}<br><span>${x.sub}</span></div><b>${x.monthly}</b></button>`).join("");
}
$("#hist").addEventListener("click", (e) => {
  const b = e.target.closest(".hitem"); if (!b) return;
  const f = loadHist()[+b.dataset.i].form;
  setForm({ age: f.age, experience: f.experience, department: validOption("Department", f.department),
            education: validOption("Education_Level", f.education), gender: validOption("Gender", f.gender), city: validOption("City", f.city) });
  submit();
});
$("#clearHist").addEventListener("click", () => { saveHist([]); renderHist(); });
renderHist();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    app.run(debug=True)