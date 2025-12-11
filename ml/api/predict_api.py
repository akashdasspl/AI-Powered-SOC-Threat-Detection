from fastapi import FastAPI
import uvicorn
import torch
from torch import nn
import joblib
import numpy as np
import json
from pathlib import Path

app = FastAPI()

# -------------------------------------------
# Load model, scaler, and feature metadata
# -------------------------------------------

project_root = Path(__file__).resolve().parents[2]
model_dir = project_root / "ml" / "models"

model_path = model_dir / "unsw_ids_mlp.pt"
scaler_path = model_dir / "unsw_scaler.pkl"
features_path = model_dir / "unsw_features_metadata.txt"

# Load features
with open(features_path, "r") as f:
    feature_cols = [line.strip() for line in f.readlines()]

# Define model architecture (must match training)
class IDSModel(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# Load scaler and model
scaler = joblib.load(scaler_path)
model = IDSModel(len(feature_cols))
model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()

# -------------------------------------------
# Prediction endpoint
# -------------------------------------------
@app.post("/predict")
def predict(payload: dict):
    """
    Input example:
    {
        "sport": 80,
        "dsport": 443,
        "proto": 6,
        "dur": 1.2,
        "sbytes": 200,
        "dbytes": 350,
        "sttl": 64,
        "dttl": 63,
        "ct_state_ttl": 5,
        ...
    }
    """

    # Build feature vector in correct order
    x_list = []
    for col in feature_cols:
        x_list.append(float(payload.get(col, 0)))  # default 0 if missing

    # Convert to array
    x = np.array(x_list).reshape(1, -1)

    # Scale
    x_scaled = scaler.transform(x)
    x_tensor = torch.tensor(x_scaled, dtype=torch.float32)

    # Predict
    with torch.no_grad():
        prob = float(model(x_tensor).item())

    label = 1 if prob > 0.5 else 0

    return {
        "label": label,
        "probability": prob
    }


# Run server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
