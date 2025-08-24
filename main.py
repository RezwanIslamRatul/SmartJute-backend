import io
import os
import time
import pickle
import logging
from typing import List, Literal

import numpy as np
from PIL import Image

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  
import tensorflow as tf
from tensorflow.keras.models import load_model

# ---------------------------
# Config
# ---------------------------
INPUT_SIZE = (122, 122)  # trained on 122x122x3
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATHS = {
    "micro": os.path.join(MODEL_DIR, "ResNet152v2_micro.keras"),
    "phone": os.path.join(MODEL_DIR, "nasnetmobile_phone.keras"),
}

LABEL_PATHS = {
    "micro": os.path.join(MODEL_DIR, "label_micro.pkl"),
    "phone": os.path.join(MODEL_DIR, "label_phone.pkl"),
}

FALLBACK_CLASSES = ["Bangladeshi_White", "Kenaf", "Mesta", "Tossa"]

# ---------------------------
# App & CORS
# ---------------------------
app = FastAPI(title="SmartJute API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for demo, open to all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# Logging
# ---------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("smartjute")

# ---------------------------
# Utils
# ---------------------------
def load_label_binarizer(path: str) -> List[str]:
    if os.path.exists(path):
        with open(path, "rb") as f:
            lb = pickle.load(f)
        if hasattr(lb, "classes_"):
            return list(lb.classes_)
    logger.warning(f"Label file not found at {path}, using fallback classes")
    return FALLBACK_CLASSES

def preprocess_image_bytes(file_bytes: bytes) -> np.ndarray:
    try:
        img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")
    img = img.resize(INPUT_SIZE)
    arr = np.asarray(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

def softmax_topk(logits: np.ndarray, labels: List[str], k: int = 3):
    probs = logits.squeeze()
    if probs.ndim != 1:
        probs = probs.reshape(-1)
    topk_idx = np.argsort(probs)[::-1][:k]
    return [{"label": labels[i], "confidence": float(probs[i])} for i in topk_idx]

# ---------------------------
# Load models & labels
# ---------------------------
MODELS = {}
LABELS = {}

def load_everything():
    t0 = time.time()

    for key, path in MODEL_PATHS.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing model file: {path}")
        logger.info(f"Loading {key} model...")
        MODELS[key] = load_model(path, compile=False)

    for key, path in LABEL_PATHS.items():
        LABELS[key] = load_label_binarizer(path)
        logger.info(f"Loaded labels for {key}: {LABELS[key]}")

    dummy = np.zeros((1, INPUT_SIZE[0], INPUT_SIZE[1], 3), dtype="float32")
    for key, mdl in MODELS.items():
        _ = mdl.predict(dummy, verbose=0)

    logger.info(f"Models loaded in {time.time() - t0:.2f}s.")

load_everything()

# ---------------------------
# Schemas
# ---------------------------
ModelName = Literal["micro", "phone"]

class PredictResponse(BaseModel):
    model: ModelName
    top1_label: str
    top1_confidence: float
    topk: List[dict]
    latency_ms: float

# ---------------------------
# Routes
# ---------------------------
@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": list(MODELS.keys())}

@app.post("/predict/{model_name}", response_model=PredictResponse)
async def predict(model_name: ModelName, file: UploadFile = File(...)):
    if model_name not in MODELS:
        raise HTTPException(status_code=404, detail=f"Unknown model name: {model_name}")

    file_bytes = await file.read()
    img_arr = preprocess_image_bytes(file_bytes)

    model = MODELS[model_name]
    labels = LABELS[model_name]

    # ---------------------------
    # Validate label vs. model output
    # ---------------------------
    preds = model.predict(img_arr, verbose=0)
    n_outputs = preds.shape[-1]
    n_labels = len(labels)

    if n_outputs != n_labels:
        raise HTTPException(
            status_code=500,
            detail=(
                f"Model/label mismatch for '{model_name}': "
                f"model outputs {n_outputs} classes but labels file has {n_labels} classes. "
                f"Please check {LABEL_PATHS[model_name]} or retrain with consistent classes."
            )
        )

    t0 = time.time()
    preds = model.predict(img_arr, verbose=0)
    latency = (time.time() - t0) * 1000.0


    k = min(3, n_outputs)
    topk = softmax_topk(preds, labels, k=k)

    return PredictResponse(
        model=model_name,
        top1_label=topk[0]["label"],
        top1_confidence=topk[0]["confidence"],
        topk=topk,
        latency_ms=latency,
    )
