import os
import io
from typing import List

import numpy as np
import onnxruntime as ort
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="german-traffic-sign-recognition")

# Configuration
MODEL_PATH = "model_gtsr.onnx"
IMAGE_SIZE = 224

def preprocess_image(image_array: np.ndarray) -> np.ndarray:
    """Preprocess image array for PyTorch-style models (ImageNet normalization).
    
    Args:
        image_array: numpy array of shape (H, W, C) with values in [0, 255]
    
    Returns:
        Preprocessed array of shape (1, C, H, W) ready for model inference
    """
    # Convert PIL Image or numpy array to PIL Image for resizing
    if isinstance(image_array, np.ndarray):
        # Ensure it's uint8
        if image_array.dtype != np.uint8:
            image_array = (image_array * 255).astype(np.uint8)
        # Convert to PIL Image
        image = Image.fromarray(image_array)
    else:
        image = image_array
    
    # Resize to target size
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)
    
    # Convert back to numpy array
    X = np.array(image, dtype=np.float32)  # Shape: (H, W, C)
    
    # Normalize to [0, 1]
    X = X / 255.0
    
    # ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    # Normalize: (X - mean) / std
    X = (X - mean) / std
    
    # Convert from (H, W, C) to (1, C, H, W) for batch processing
    X = X.transpose(2, 0, 1)  # (C, H, W)
    X = np.expand_dims(X, axis=0)  # (1, C, H, W)
    
    return X.astype(np.float32)


# Load ONNX model
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"ONNX model not found at {MODEL_PATH}. Please train the model first.")

session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

id2label = [
    'Speed limit (20km/h)', 'Speed limit (30km/h)', 'Speed limit (50km/h)',
    'Speed limit (60km/h)', 'Speed limit (70km/h)', 'Speed limit (80km/h)',
    'Speed limit (80km/h)', 'Speed limit (100km/h)', 'Speed limit (120km/h)',
    'No passing', 'No passing for vehicles over 3.5 metric tons', 'Right-of-way at intersection',
    'Priority road', 'Yield', 'Stop', 'No vehicles',
    'No vehicles > 3.5 tons', 'No entry', 'General caution',
    'Dangerous curve left', 'Dangerous curve right', 'Double curve',
    'Bumpy road', 'Slippery road', 'Road narrows on the right',
    'Road work', 'Traffic signals', 'Pedestrians', 'Children crossing',
    'Bicycles crossing', 'Beware of ice/snow', 'Wild animals crossing',
    'End speed + ending of all restrictions', 'Turn right ahead',
    'Turn left ahead', 'Ahead only', 'Go straight or right',
    'Go straight or left', 'Keep right', 'Keep left',
    'Roundabout mandatory', 'End of no passing', 'End no passing > 3.5t'
]


class PredictionItem(BaseModel):
    """Single prediction item with label id, name, and probability."""
    label: int
    label_name: str
    proba: float


class PredictResponse(BaseModel):
    """Response containing top-3 predictions."""
    top_3: List[PredictionItem]


def predict_from_array(image_array: np.ndarray) -> List[PredictionItem]:
    """Run inference on preprocessed image array and return top-3 predictions."""

    X = preprocess_image(image_array)
    result = session.run([output_name], {input_name: X})
    logits = result[0][0]
    
    # Apply softmax to get probabilities
    exp_logits = np.exp(logits - np.max(logits))  # Numerical stability
    probs = exp_logits / exp_logits.sum()
    
    # Get top-3 predictions
    top_3_indices = np.argsort(probs)[-3:][::-1]
    
    predictions = []
    for idx in top_3_indices:
        predictions.append(
            PredictionItem(
                label=int(idx),
                label_name=id2label[idx] if idx < len(id2label) else f"Class {idx}",
                proba=float(probs[idx])
            )
        )
    
    return predictions


@app.get("/")
def root():
    return {"message": "German Traffic Sign Recognition Service"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictResponse)
async def predict_endpoint(file: UploadFile = File(...)):
    """Predict traffic sign class from uploaded image file."""
    # Read image file
    contents = await file.read()
    
    # Convert to numpy array (PIL Image -> numpy)
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    image_array = np.array(image)
    
    # Get predictions
    predictions = predict_from_array(image_array)
    
    return PredictResponse(top_3=predictions)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)