"""
api.py  -  FastAPI backend: Fake vs Genuine social-media account detector.

Backend for Render. Folder: backend/  (api.py, train.py, model.pkl, requirements.txt).
The file MUST be called api.py (Render start command: uvicorn api:app ...).

model.pkl is loaded as it is. It was saved with scikit-learn 1.7.2, so requirements.txt pins that
exact version. Only if model.pkl is unusable AND fake_account_detection.csv is next to this file,
the model is rebuilt with train.py (not recommended on Render's free plan: slow, little memory).

Run locally:  uvicorn api:app --reload
Docs:         http://127.0.0.1:8000/docs   (interactive page where you can try /predict)
"""

import sys
import warnings
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict

import joblib
import pandas as pd
import sklearn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))          # so that `from train import train_model` works
MODEL_PATH = BASE_DIR / "model.pkl"
DATA_PATH = BASE_DIR / "fake_account_detection.csv"

REQUIRED_KEYS = {"pipeline", "feature_order", "features", "classes", "positive_label"}
STATE = {"bundle": None, "error": None}


# ----------------------------------------------------------------------------
# Loading the saved model (same logic as app.py)
# ----------------------------------------------------------------------------
def default_row(bundle):
    """A one-row DataFrame of typical values, used only to check that the model works."""
    row = {}
    for feature in bundle["features"]:
        if feature["kind"] == "numeric":
            row[feature["name"]] = float(feature["median"])
        elif feature["kind"] == "binary":
            row[feature["name"]] = float(feature["default"])
        else:
            row[feature["name"]] = str(feature["default"])
    return pd.DataFrame([row])[bundle["feature_order"]]


def bundle_works(bundle):
    """True if the bundle has everything needed AND can predict on this computer."""
    if not isinstance(bundle, dict) or not REQUIRED_KEYS.issubset(bundle):
        return False
    try:
        bundle["pipeline"].predict_proba(default_row(bundle))
        return True
    except Exception:
        return False


def load_bundle():
    """Load model.pkl; rebuild it with train.py if it does not match this computer."""
    bundle = None
    if MODEL_PATH.exists():
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                bundle = joblib.load(MODEL_PATH)
        except Exception:
            bundle = None

    if bundle_works(bundle):
        if bundle.get("sklearn_version") != sklearn.__version__:
            print(f"NOTE: model.pkl was saved with scikit-learn {bundle.get('sklearn_version')}, "
                  f"installed is {sklearn.__version__}. It still predicts, so it is used as it is.")
        return bundle

    if DATA_PATH.exists():  # model.pkl unusable here: rebuild once with the installed scikit-learn
        print("model.pkl is not usable here. Rebuilding it from the CSV (takes a minute)...")
        from train import train_model
        return train_model(DATA_PATH, MODEL_PATH, verbose=False)

    raise FileNotFoundError(
        "model.pkl cannot be used with the installed scikit-learn "
        f"({sklearn.__version__}) and {DATA_PATH.name} is not available to rebuild it. "
        "Pin scikit-learn to the version stored in model.pkl (see requirements.txt)."
    )


@asynccontextmanager
async def lifespan(app):
    try:
        bundle = load_bundle()
        if set(Account.model_fields) != set(bundle["feature_order"]):
            raise ValueError("The columns in model.pkl do not match the fields in api.py.")
        STATE["bundle"], STATE["error"] = bundle, None
        print("Model ready.")
    except Exception as error:
        STATE["bundle"], STATE["error"] = None, str(error)
        print("WARNING: the model could not be loaded:", error)
    yield


app = FastAPI(title="Fake vs Genuine Account Detector API", lifespan=lifespan)


def get_bundle():
    if STATE["bundle"] is None:
        raise HTTPException(status_code=503, detail=STATE["error"] or "The model is not loaded.")
    return STATE["bundle"]


# ----------------------------------------------------------------------------
# Input / output data structures (the 32 feature columns of the dataset)
# ----------------------------------------------------------------------------
class Account(BaseModel):
    model_config = ConfigDict(extra="forbid")   # reject misspelled or unknown fields

    followers_count: float = Field(..., ge=0)
    following_count: float = Field(..., ge=0)
    posts_count: float = Field(..., ge=0)
    account_age_days: float = Field(..., ge=0)
    avg_posts_per_week: float = Field(..., ge=0)
    avg_likes_per_post: float = Field(..., ge=0)
    avg_comments_per_post: float = Field(..., ge=0)
    avg_shares_per_post: float = Field(..., ge=0)
    profile_completion_pct: float = Field(..., ge=0)
    profile_photo_present: int = Field(..., ge=0, le=1)      # 0 = No, 1 = Yes
    bio_length: float = Field(..., ge=0)
    username_digit_ratio: float = Field(..., ge=0)
    username_length: float = Field(..., ge=0)
    login_days_last_30d: float = Field(..., ge=0)
    unique_ip_count_30d: float = Field(..., ge=0)
    device_count_30d: float = Field(..., ge=0)
    failed_login_count_30d: float = Field(..., ge=0)
    messages_sent_30d: float = Field(..., ge=0)
    messages_received_30d: float = Field(..., ge=0)
    friend_request_acceptance_rate: float = Field(..., ge=0)
    user_reports_90d: float = Field(..., ge=0)
    content_similarity_score: float = Field(..., ge=0)
    account_type: str
    verification_status: str
    profile_visibility: str
    signup_method: str
    primary_device: str
    region_group: str
    posting_pattern: str
    bio_language: str
    email_domain_type: str
    phone_verified: str


class Prediction(BaseModel):
    prediction: str                      # "Fake" or "Genuine"
    confidence: float                    # probability of the predicted class
    probabilities: Dict[str, float]      # probability of every class


# ----------------------------------------------------------------------------
# API endpoints
# ----------------------------------------------------------------------------
@app.get("/")
def home():
    return {"message": "Fake vs Genuine Account Detector API is running. Open /docs to try it."}


@app.get("/health")
def health():
    return {"status": "ok" if STATE["bundle"] is not None else "model not loaded",
            "detail": STATE["error"],
            "sklearn": sklearn.__version__}


@app.get("/metadata")
def metadata():
    """Columns, allowed categories, training ranges and the held-out test results."""
    bundle = get_bundle()
    return {
        "features": bundle["features"],
        "classes": list(bundle["classes"]),
        "positive_label": bundle["positive_label"],
        "metrics": bundle.get("metrics"),
        "best_params": bundle.get("best_params"),
        "n_test": bundle.get("n_test"),
    }


@app.post("/predict", response_model=Prediction)
def predict(account: Account):
    bundle = get_bundle()
    values = account.model_dump()

    # Category values must be ones the model saw in training (same lists as the app's dropdowns).
    problems = []
    for feature in bundle["features"]:
        if feature["kind"] == "category" and values[feature["name"]] not in feature["options"]:
            problems.append({
                "loc": ["body", feature["name"]],
                "msg": f"must be one of {feature['options']}",
                "type": "value_error",
            })
    if problems:
        raise HTTPException(status_code=422, detail=problems)

    # One row with exactly the training columns, in the training order (same as app.py).
    row = {}
    for feature in bundle["features"]:
        value = values[feature["name"]]
        if feature["kind"] in ("numeric", "binary"):
            row[feature["name"]] = float(value)
        else:
            row[feature["name"]] = str(value)
    model_input = pd.DataFrame([row])[bundle["feature_order"]]

    try:
        pipeline = bundle["pipeline"]
        label = str(pipeline.predict(model_input)[0])
        probabilities = pipeline.predict_proba(model_input)[0]
        classes = [str(c) for c in pipeline.classes_]
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {error}")

    by_class = {c: float(p) for c, p in zip(classes, probabilities)}
    return Prediction(prediction=label, confidence=by_class[label], probabilities=by_class)
