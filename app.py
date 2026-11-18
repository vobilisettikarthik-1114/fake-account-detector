"""
app.py  -  Streamlit FRONTEND: Fake vs Genuine social-media account detector.

This app does NOT load the model. It asks the FastAPI backend (hosted on Render) for
  GET  /metadata  -> the form fields, allowed categories and training ranges
  POST /predict   -> the prediction and the probabilities
The backend address is read from the secret / environment variable  API_URL.

Run locally:   streamlit run app.py
"""

import math
import os
import time

import requests
import streamlit as st

st.set_page_config(
    page_title="Fake vs Genuine Account Detector",
    page_icon="🕵️",
    layout="wide",
)


def get_api_url() -> str:
    """API_URL from Streamlit secrets, else the environment, else a local backend."""
    url = None
    try:
        url = st.secrets["API_URL"]
    except Exception:
        url = None
    url = url or os.environ.get("API_URL") or "http://127.0.0.1:8000"
    return str(url).strip().rstrip("/")


API_URL = get_api_url()

# How the 32 input columns are grouped on screen (any new column goes to "Other").
SECTIONS = {
    "👤 Profile & account": [
        "account_age_days", "profile_completion_pct", "profile_photo_present", "bio_length",
        "username_digit_ratio", "username_length", "account_type", "verification_status",
        "profile_visibility", "signup_method", "bio_language", "email_domain_type",
        "phone_verified",
    ],
    "📣 Network & content": [
        "followers_count", "following_count", "posts_count", "avg_posts_per_week",
        "avg_likes_per_post", "avg_comments_per_post", "avg_shares_per_post",
        "posting_pattern", "content_similarity_score",
    ],
    "🔐 Login & messaging behaviour": [
        "login_days_last_30d", "unique_ip_count_30d", "device_count_30d",
        "failed_login_count_30d", "messages_sent_30d", "messages_received_30d",
        "friend_request_acceptance_rate", "user_reports_90d", "primary_device",
        "region_group",
    ],
}
BINARY_LABELS = {0.0: "No", 1.0: "Yes"}


# ----------------------------------------------------------------------------
# Talking to the backend
# ----------------------------------------------------------------------------
@st.cache_data(
    ttl=3600,
    show_spinner="Connecting to the API... (a free Render server sleeps when idle and can "
                 "take up to a minute to wake up)",
)
def fetch_metadata(api_url: str) -> dict:
    """Get the form description from the backend. Errors are raised, so they are not cached."""
    last_error = "unknown error"
    for attempt in range(3):
        try:
            response = requests.get(f"{api_url}/metadata", timeout=60)
            if response.status_code == 200:
                return response.json()
            last_error = f"the API answered {response.status_code}: {response.text[:300]}"
        except requests.RequestException as error:
            last_error = str(error)
        if attempt < 2:
            time.sleep(5)
    raise RuntimeError(last_error)


def get_metadata() -> dict:
    try:
        return fetch_metadata(API_URL)
    except Exception as error:
        st.error(f"Could not reach the backend at **{API_URL}**.")
        st.caption(f"Details: {error}")
        st.info(
            "Check that the Render service is running (open `" + API_URL + "/health` in a new tab) "
            "and that `API_URL` is set correctly in the Streamlit secrets."
        )
        if st.button("🔄 Try again"):
            st.rerun()
        st.stop()


def call_predict(payload: dict):
    """POST the account to the backend. Returns (result_dict, None) or (None, error_text)."""
    try:
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=90)
    except requests.Timeout:
        return None, "The API took too long to answer. If it was asleep, wait a moment and click Predict again."
    except requests.RequestException as error:
        return None, f"Could not reach the API: {error}"

    if response.status_code == 200:
        return response.json(), None
    try:
        detail = response.json().get("detail", response.text)
    except ValueError:
        detail = response.text
    if isinstance(detail, list):  # validation errors: list of {loc, msg, ...}
        lines = [f"{'.'.join(str(p) for p in item.get('loc', [])[1:])}: {item.get('msg')}" for item in detail]
        return None, "The API rejected the input:\n\n" + "\n".join(f"- {line}" for line in lines)
    return None, f"The API returned {response.status_code}: {detail}"


# ----------------------------------------------------------------------------
# Helpers for the input form
# ----------------------------------------------------------------------------
def widget_key(name: str) -> str:
    return f"input_{name}"


def pretty(name: str) -> str:
    return name.replace("_", " ").capitalize()


def numeric_bounds(feature):
    """Lowest value allowed in the box: 0 if the training data is never negative."""
    low = feature["min"]
    if feature["is_integer"]:
        return 0 if low >= 0 else int(math.floor(low))
    return 0.0 if low >= 0 else float(math.floor(low))


def default_value(feature):
    if feature["kind"] == "numeric":
        if feature["is_integer"]:
            return int(round(feature["median"]))
        return float(feature["median"])
    if feature["kind"] == "binary":
        return BINARY_LABELS[float(feature["default"])]
    return str(feature["default"])


def render_input(feature):
    """Draw one input widget and return its current value."""
    key, label = widget_key(feature["name"]), pretty(feature["name"])
    if feature["kind"] == "numeric":
        help_text = f"Range seen in training data: {feature['min']:g} to {feature['max']:g}"
        if feature["is_integer"]:
            return st.number_input(label, min_value=numeric_bounds(feature), step=1,
                                   key=key, help=help_text)
        return st.number_input(label, min_value=numeric_bounds(feature), step=0.01,
                               format="%.2f", key=key, help=help_text)
    if feature["kind"] == "binary":
        return st.selectbox(label, list(BINARY_LABELS.values()), key=key)
    return st.selectbox(label, feature["options"], key=key,
                        help="Choose one of the categories seen in training")


def collect_problems(features, values):
    """Return a list of blocking errors (empty or impossible values)."""
    errors = []
    for feature in features:
        name, value = feature["name"], values[feature["name"]]
        if value is None or (isinstance(value, float) and math.isnan(value)):
            errors.append(f"{pretty(name)} is empty.")
        elif feature["kind"] == "numeric" and value < numeric_bounds(feature):
            errors.append(f"{pretty(name)} cannot be below {numeric_bounds(feature)}.")
    return errors


def build_payload(features, values):
    """The JSON body for POST /predict: one entry per column, with plain JSON types."""
    payload = {}
    for feature in features:
        value = values[feature["name"]]
        if feature["kind"] == "numeric":
            payload[feature["name"]] = int(value) if feature["is_integer"] else float(value)
        elif feature["kind"] == "binary":
            payload[feature["name"]] = 1 if value == "Yes" else 0
        else:
            payload[feature["name"]] = str(value)
    return payload


def probability_bar(label, probability, color):
    """A coloured progress bar with the label on the left and the percentage on the right."""
    return (
        '<div style="margin:14px 0 6px 0">'
        '<div style="display:flex;justify-content:space-between;font-weight:600;font-size:1.05rem">'
        f"<span>{label}</span><span>{probability:.1%}</span></div>"
        '<div style="background:rgba(128,128,128,0.25);border-radius:10px;height:20px;overflow:hidden">'
        f'<div style="width:{probability * 100:.1f}%;background:{color};height:100%;border-radius:10px"></div>'
        "</div></div>"
    )


# ----------------------------------------------------------------------------
# Page
# ----------------------------------------------------------------------------
meta = get_metadata()
features = meta["features"]
classes = list(meta["classes"])
positive_label = meta["positive_label"]

# default values for every widget (set once, so the widgets never need a `value=` argument)
for feature in features:
    st.session_state.setdefault(widget_key(feature["name"]), default_value(feature))

with st.sidebar:
    st.subheader("About the model")
    st.caption("Tuned Logistic Regression, served by a FastAPI backend.")
    if meta.get("metrics"):
        st.caption("Held-out test results (Fake = positive class):")
        for name, value in meta["metrics"].items():
            st.write(f"**{name.replace('_', ' ').title()}**: {value:.3f}")
    st.caption(f"API: {API_URL}")

# Header ----------------------------------------------------------------------
st.title("🕵️ Fake vs Genuine Account Detector")
st.write(
    "Enter an account's profile, activity and login details. "
    "This application estimates whether the account is **Fake** or **Genuine**."
)

# Input form ------------------------------------------------------------------
by_name = {f["name"]: f for f in features}
placed = set()
sections = {}
for title, names in SECTIONS.items():
    present = [n for n in names if n in by_name]
    if present:
        sections[title] = present
        placed.update(present)
leftover = [f["name"] for f in features if f["name"] not in placed]
if leftover:
    sections["Other"] = leftover

values = {}
with st.form("prediction_form"):
    tabs = st.tabs(list(sections.keys()))
    for tab, names in zip(tabs, sections.values()):
        with tab:
            grid = st.columns(3)
            for index, name in enumerate(names):
                with grid[index % 3]:
                    values[name] = render_input(by_name[name])
    submitted = st.form_submit_button("🔍 Predict", type="primary")

# Prediction --------------------------------------------------------------------
if submitted:
    errors = collect_problems(features, values)
    if errors:
        for message in errors:
            st.error(message)
        st.stop()

    with st.spinner("Asking the API..."):
        result, error = call_predict(build_payload(features, values))
    if error:
        st.error(error)
        st.stop()

    prediction = result["prediction"]
    probability_by_class = result["probabilities"]
    other_label = next(c for c in classes if c != positive_label)

    st.divider()
    if prediction == positive_label:
        st.error(f"## 🚩 This account is predicted to be {prediction.upper()}")
    else:
        st.success(f"## ✅ This account is predicted to be {prediction.upper()}")

    st.markdown(
        probability_bar(positive_label, probability_by_class[positive_label], "#e5484d")
        + probability_bar(other_label, probability_by_class[other_label], "#2eb872"),
        unsafe_allow_html=True,
    )
