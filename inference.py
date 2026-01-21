import os
import io
from typing import List

import numpy as np
import onnxruntime as ort
from PIL import Image


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

model = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
input_name = model.get_inputs()[0].name
output_name = model.get_outputs()[0].name

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



def predict_from_array(image_array) :
    """Run inference on preprocessed image array and return top-3 predictions."""
    X = preprocess_image(image_array)
    result = model.run([output_name], {input_name: X})
    logits = result[0][0]
    
    # Apply softmax to get probabilities
    exp_logits = np.exp(logits - np.max(logits))  # Numerical stability
    probs = exp_logits / exp_logits.sum()
    
    # Get top-3 predictions
    top_3_indices = np.argsort(probs)[-3:][::-1]
        
    predictions = []
    for idx in top_3_indices:
        predictions.append([int(idx),
            id2label[idx] if idx < len(id2label) else f"Class {idx}",
            float(probs[idx])])
    
    return predictions


if __name__ == '__main__':
    
    for i in range(1,4):
        file_name = f'images/sign{i}.jpg' 
        image = Image.open(file_name).convert("RGB")
        image_array = np.array(image)
        
        # Get predictions
        predictions = predict_from_array(image_array)
        print(file_name, '\n', predictions)
        print()
        print()


    