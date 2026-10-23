import os
import io
import json
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pickle

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "crop_model.pkl"
DATASET_CANDIDATES = [
    BASE_DIR / "Crop_recommendation.csv",
    BASE_DIR / "Crop_recommendation.csv.csv",
    BASE_DIR / "crop_recommendation.csv",
    BASE_DIR / "dataset.csv",
]
DB_PATH = BASE_DIR / "prediction_history.db"
FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
DISPLAY_NAMES = {
    "N": "Nitrogen (N)", "P": "Phosphorus (P)", "K": "Potassium (K)",
    "temperature": "Temperature (°C)", "humidity": "Humidity (%)",
    "ph": "Soil pH", "rainfall": "Rainfall (mm)"
}

st.set_page_config(page_title="Smart Crop AI", page_icon="🌱", layout="wide")

st.markdown("""
<style>
.stApp {background: radial-gradient(circle at 15% 0%, #112b30 0%, #080b0d 42%, #070809 100%); color:#f5f7fa;}
.block-container {max-width: 1400px; padding-top: 2rem; padding-bottom: 3rem;}
[data-testid="stMetric"] {background:linear-gradient(145deg,#151d20,#101416);border:1px solid #26383d;padding:18px;border-radius:16px;}
div[data-testid="stForm"] {background:#101719;border:1px solid #26383d;padding:20px;border-radius:18px;}
.stButton button, .stDownloadButton button {border-radius:10px;font-weight:700;border:1px solid #35bdf2;}
h1,h2,h3 {letter-spacing:-0.02em;}
.small-muted {color:#aab7bb;font-size:0.9rem;}
</style>
""", unsafe_allow_html=True)

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            crop TEXT NOT NULL,
            inputs_json TEXT NOT NULL,
            probabilities_json TEXT
        )""")
init_db()

@st.cache_resource
def load_model(path_string):
    with open(path_string, "rb") as f:
        return pickle.load(f)

def find_dataset():
    for path in DATASET_CANDIDATES:
        if path.exists():
            try:
                df = pd.read_csv(path)
                if "label" in df.columns:
                    return df, path
            except Exception:
                pass
    return None, None

def get_history(limit=500):
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query(
            "SELECT id, created_at, crop, inputs_json, probabilities_json FROM predictions ORDER BY id DESC LIMIT ?",
            conn, params=(limit,)
        )

def save_prediction(crop, inputs, probs):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO predictions(created_at,crop,inputs_json,probabilities_json) VALUES(?,?,?,?)",
            (datetime.now().isoformat(timespec="seconds"), str(crop),
             json.dumps(inputs), json.dumps(probs) if probs else None)
        )

def make_pdf_report(crop, inputs, probs):
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib.styles import getSampleStyleSheet
    except ImportError:
        return None
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = [Paragraph("Smart Crop Recommendation Report", styles["Title"]),
             Spacer(1, 10),
             Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles["Normal"]),
             Spacer(1, 10),
             Paragraph(f"<b>Recommended crop:</b> {crop}", styles["Heading2"]),
             Spacer(1, 8)]
    rows = [["Input", "Value"]] + [[DISPLAY_NAMES.get(k, k), f"{v:g}"] for k, v in inputs.items()]
    table = Table(rows, colWidths=[250, 150])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#0F6674")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
        ("PADDING", (0,0), (-1,-1), 7),
    ]))
    story.append(table)
    if probs:
        story += [Spacer(1, 12), Paragraph("Model output scores (not a guarantee of field success)", styles["Heading3"])]
        p_rows = [["Crop", "Model score (%)"]] + [[k, f"{v*100:.2f}%"] for k,v in sorted(probs.items(), key=lambda x:x[1], reverse=True)[:10]]
        p_table = Table(p_rows, colWidths=[250, 150])
        p_table.setStyle(TableStyle([("GRID",(0,0),(-1,-1),0.5,colors.grey),("PADDING",(0,0),(-1,-1),6)]))
        story.append(p_table)
    story.append(Spacer(1, 10))
    story.append(Paragraph("This is a machine-learning suggestion, not agronomic or financial advice. Confirm soil test and local growing conditions before planting.", styles["Italic"]))
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

st.caption("INTELLIGENT AGRICULTURE  /  MACHINE LEARNING")
st.title("🌱 Smart Crop Recommendation")
st.write("Explore crop suggestions from soil nutrients and climate conditions.")
st.markdown("---")

if not MODEL_PATH.exists():
    st.error(f"Model file not found: {MODEL_PATH.name}")
    st.info("Place crop_model.pkl beside app.py. If it has not been generated, run your existing train_model.py first.")
    st.stop()

try:
    model = load_model(str(MODEL_PATH))
except Exception as exc:
    st.error(f"Could not load crop_model.pkl: {exc}")
    st.stop()

with st.sidebar:
    st.header("About this app")
    st.write("Uses your existing trained model. Predictions depend on the quality and coverage of the training dataset.")
    st.caption("Weather lookup is optional. Enter measured values manually for best control.")
    weather_enabled = st.checkbox("Enable optional live weather lookup", value=False)
    weather_city = st.text_input("City / place name", placeholder="e.g. Chennai")
    if weather_enabled and st.button("Get weather"):
        try:
            import requests
            geo = requests.get("https://geocoding-api.open-meteo.com/v1/search",
                               params={"name": weather_city, "count": 1, "language": "en", "format": "json"},
                               timeout=8)
            geo.raise_for_status()
            results = geo.json().get("results", [])
            if not results:
                st.warning("Place not found. Try a nearby city or enter values manually.")
            else:
                loc = results[0]
                wx = requests.get("https://api.open-meteo.com/v1/forecast",
                                  params={"latitude": loc["latitude"], "longitude": loc["longitude"],
                                          "current": "temperature_2m,relative_humidity_2m,precipitation",
                                          "daily": "precipitation_sum", "timezone": "auto"},
                                  timeout=8)
                wx.raise_for_status()
                current = wx.json().get("current", {})
                st.session_state["weather_suggested"] = {
                    "temperature": float(current.get("temperature_2m", 25)),
                    "humidity": float(current.get("relative_humidity_2m", 60)),
                    "rainfall": float(current.get("precipitation", 0)),
                }
                st.success(f"Weather loaded for {loc.get('name', weather_city)}. Review the suggested values in the form.")
                st.json(st.session_state["weather_suggested"])
        except Exception as exc:
            st.warning(f"Weather service unavailable: {exc}. Manual entry remains available.")

wx = st.session_state.get("weather_suggested", {})
with st.container(border=True):
    st.subheader("🌾 Soil and climate inputs")
    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            n = st.number_input("Nitrogen (N)", min_value=0.0, max_value=300.0, value=90.0, step=1.0)
            p = st.number_input("Phosphorus (P)", min_value=0.0, max_value=300.0, value=42.0, step=1.0)
            k = st.number_input("Potassium (K)", min_value=0.0, max_value=300.0, value=43.0, step=1.0)
        with c2:
            temperature = st.number_input("Temperature (°C)", min_value=-10.0, max_value=60.0, value=float(wx.get("temperature", 25.0)), step=0.5)
            humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=float(wx.get("humidity", 70.0)), step=1.0)
        with c3:
            ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, value=6.5, step=0.1)
            rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=5000.0, value=float(wx.get("rainfall", 100.0)), step=5.0)
        submitted = st.form_submit_button("✨ Recommend crop", use_container_width=True)

if submitted:
    input_values = {"N": n, "P": p, "K": k, "temperature": temperature,
                    "humidity": humidity, "ph": ph, "rainfall": rainfall}
    input_df = pd.DataFrame([[input_values[f] for f in FEATURES]], columns=FEATURES)
    try:
        pred = model.predict(input_df)[0]
        probs = None
        if hasattr(model, "predict_proba"):
            raw_probs = model.predict_proba(input_df)[0]
            classes = getattr(model, "classes_", None)
            if classes is None and hasattr(model, "named_steps"):
                final_est = list(model.named_steps.values())[-1]
                classes = getattr(final_est, "classes_", None)
            if classes is not None and len(classes) == len(raw_probs):
                probs = {str(c): float(p) for c, p in zip(classes, raw_probs)}
        st.session_state["latest_prediction"] = {
            "crop": str(pred), "inputs": input_values, "probs": probs,
            "created_at": datetime.now().isoformat(timespec="seconds")
        }
        save_prediction(str(pred), input_values, probs)
    except Exception as exc:
        st.error(f"Prediction failed: {exc}. Check that the model was trained with the expected seven features.")

latest = st.session_state.get("latest_prediction")
if latest:
    st.markdown("---")
    st.subheader("🌿 Recommendation")
    m1, m2, m3 = st.columns(3)
    m1.metric("Recommended crop", latest["crop"])
    m2.metric("Inputs used", "7 features")
    m3.metric("Generated at", datetime.fromisoformat(latest["created_at"]).strftime("%H:%M:%S"))
    if latest["probs"]:
        best = latest["probs"].get(latest["crop"])
        if best is not None:
            st.progress(min(max(best, 0.0), 1.0), text=f"Model score for recommended crop: {best*100:.1f}% (not a guarantee)")
        chart_df = pd.DataFrame([{"Crop": k, "Model score": v} for k,v in latest["probs"].items()])
        chart_df = chart_df.sort_values("Model score", ascending=False).head(10)
        fig = px.bar(chart_df, x="Model score", y="Crop", orientation="h",
                     title="Top model scores", text=chart_df["Model score"].map(lambda x: f"{x:.1%}"))
        fig.update_layout(template="plotly_dark", yaxis={"categoryorder":"total ascending"}, xaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("These are model output scores, not calibrated probabilities unless the model has been calibrated and validated.")
    else:
        st.info("This model does not expose predict_proba; the app will show the predicted crop without inventing a confidence score.")

    with st.expander("🔍 Explain this prediction"):
        st.write("The prediction was produced from these seven input features:")
        st.dataframe(pd.DataFrame([latest["inputs"]]).rename(columns=DISPLAY_NAMES), use_container_width=True)
        st.write("Feature-level explanations are model-dependent. The chart below uses native feature importance only when the trained estimator exposes it.")
        estimator = model
        if hasattr(model, "named_steps"):
            estimator = list(model.named_steps.values())[-1]
        importances = getattr(estimator, "feature_importances_", None)
        if importances is not None and len(importances) == len(FEATURES):
            imp = pd.DataFrame({"Feature": [DISPLAY_NAMES[f] for f in FEATURES], "Importance": importances}).sort_values("Importance")
            fig_imp = px.bar(imp, x="Importance", y="Feature", orientation="h", title="Global feature importance")
            fig_imp.update_layout(template="plotly_dark")
            st.plotly_chart(fig_imp, use_container_width=True)
            st.caption("Global feature importance describes overall model behavior, not a guaranteed causal explanation for this individual prediction.")
        elif hasattr(estimator, "coef_"):
            st.write("This estimator exposes coefficients, but class-specific interpretation requires inspecting the fitted model and its preprocessing.")
        else:
            st.info("This model does not expose built-in feature importance. For local SHAP explanations, install and configure SHAP for the exact estimator; unsupported models should not be given fabricated explanations.")

    report_pdf = make_pdf_report(latest["crop"], latest["inputs"], latest["probs"])
    csv_bytes = pd.DataFrame([{"timestamp": latest["created_at"], "recommended_crop": latest["crop"], **latest["inputs"]}]).to_csv(index=False).encode("utf-8")
    d1, d2 = st.columns(2)
    d1.download_button("⬇️ Download CSV report", data=csv_bytes, file_name="crop_recommendation.csv", mime="text/csv", use_container_width=True)
    if report_pdf:
        d2.download_button("⬇️ Download PDF report", data=report_pdf, file_name="crop_recommendation.pdf", mime="application/pdf", use_container_width=True)
    else:
        d2.warning("PDF export needs reportlab. Install it using: pip install reportlab")

st.markdown("---")
tab_history, tab_eval, tab_notes = st.tabs(["🕘 Prediction history", "📈 Model evaluation", "ℹ️ Notes"])

with tab_history:
    hist = get_history()
    if hist.empty:
        st.info("Your predictions will appear here after the first recommendation.")
    else:
        view = hist.copy()
        view["inputs_json"] = view["inputs_json"].apply(lambda x: json.loads(x))
        expanded_inputs = pd.json_normalize(view["inputs_json"])
        display = pd.concat([view[["id","created_at","crop"]].reset_index(drop=True), expanded_inputs.reset_index(drop=True)], axis=1)
        st.dataframe(display, use_container_width=True, hide_index=True)
        st.download_button("Export history as CSV", data=display.to_csv(index=False).encode("utf-8"),
                           file_name="crop_prediction_history.csv", mime="text/csv")
        if st.button("Clear prediction history"):
            with sqlite3.connect(DB_PATH) as conn:
                conn.execute("DELETE FROM predictions")
            st.success("History cleared. Refresh the page to update the table.")
    st.caption("History is stored locally in prediction_history.db on this computer.")

with tab_eval:
    df, dataset_path = find_dataset()
    if df is None:
        st.info("Evaluation requires the original labeled dataset CSV beside app.py. Supported default names: Crop_recommendation.csv, crop_recommendation.csv, or dataset.csv.")
        st.write("No accuracy number is shown until the labeled dataset is found and a held-out test split can be evaluated.")
    else:
        missing = [f for f in FEATURES + ["label"] if f not in df.columns]
        if missing:
            st.warning(f"Dataset {dataset_path.name} is missing columns: {', '.join(missing)}")
        else:
            st.write(f"Dataset loaded: `{dataset_path.name}` — {len(df):,} rows.")
            try:
                from sklearn.model_selection import train_test_split
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
                X = df[FEATURES]
                y = df["label"]
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
                eval_model = model
                # Only evaluate directly if the existing fitted model can predict on the held-out data.
                y_pred = eval_model.predict(X_test)
                a, b, c, d = st.columns(4)
                a.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.1%}")
                b.metric("Macro precision", f"{precision_score(y_test, y_pred, average='macro', zero_division=0):.1%}")
                c.metric("Macro recall", f"{recall_score(y_test, y_pred, average='macro', zero_division=0):.1%}")
                d.metric("Macro F1", f"{f1_score(y_test, y_pred, average='macro', zero_division=0):.1%}")
                st.warning("These scores compare the already-trained model against a random held-out split from the supplied CSV. If that same CSV was used to train the model, this is not an independent test and may overstate generalization. For a valid evaluation, retrain using only the training split.")
                labels = sorted(pd.Series(y).astype(str).unique())
                cm = confusion_matrix(y_test, y_pred, labels=labels)
                fig_cm = px.imshow(cm, x=labels, y=labels, labels=dict(x="Predicted", y="Actual", color="Count"),
                                   title="Confusion matrix", aspect="auto", color_continuous_scale="Blues")
                fig_cm.update_layout(template="plotly_dark")
                st.plotly_chart(fig_cm, use_container_width=True)
                with st.expander("Classification report"):
                    st.code(classification_report(y_test, y_pred, zero_division=0))
            except Exception as exc:
                st.error(f"Could not evaluate this model on the dataset: {exc}")
                st.info("If your model expects different feature names or preprocessing, update train_model.py to train and evaluate using the same pipeline.")

with tab_notes:
    st.markdown("""
- **Crop suggestion:** A model output, not a guarantee of yield or profitability.
- **Weather:** Uses Open-Meteo when enabled; current precipitation is not the same as seasonal rainfall. Review all values before predicting.
- **XAI:** Native feature importance is shown only when the estimator exposes it. It is not a causal explanation.
- **Privacy:** Prediction history stays in a local SQLite database. Optional weather lookup sends the entered place name to Open-Meteo.
- **Evaluation:** Use a proper train/test split before reporting accuracy in your project documentation.
""")

st.markdown("<p class='small-muted' style='text-align:center'>🌿 Smart Agriculture AI · Verify recommendations with local soil testing and agricultural guidance.</p>", unsafe_allow_html=True)
