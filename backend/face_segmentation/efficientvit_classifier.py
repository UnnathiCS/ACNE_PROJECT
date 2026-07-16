import os
import sys
from typing import Tuple, Optional

# Ensure backend root is importable when loaded via face_segmentation package.
_BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

import cv2
import numpy as np
import torch
from torchvision import transforms

# Lightweight wrapper for loading an EfficientViT classifier and running
# inference on lesion crops. Keeps behaviour robust: if model or weights
# are missing, functions will return None to allow graceful fallback.

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'weights', 'efficientvit_acne_classifier.pth')
MODEL_NAME = 'efficientvit_b0'
CLASS_NAMES = ['blackhead', 'nodulocystic', 'papule', 'pustule', 'whitehead']

_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

_model = None
_device = 'cuda' if torch.cuda.is_available() else 'cpu'


def _load_model():
    global _model
    if _model is not None:
        return _model
    try:
        from models.efficientvit_acne import build_efficientvit_acne_classifier
    except Exception:
        return None
    try:
        model = build_efficientvit_acne_classifier(
            num_classes=len(CLASS_NAMES),
            model_name=MODEL_NAME,
            pretrained=False,
        )
        if os.path.exists(MODEL_PATH):
            state = torch.load(MODEL_PATH, map_location=_device)
            model.load_state_dict(state)
        else:
            return None
        model.to(_device)
        model.eval()
        _model = model
        return _model
    except Exception:
        return None


def classify_lesion_crop(image_crop: np.ndarray) -> Optional[Tuple[str, float]]:
    """Classify a single lesion crop (BGR numpy image). Returns (label, confidence)
    or None on failure.
    """
    try:
        model = _load_model()
        if model is None:
            return None
        img = cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB)
        x = _transform(img).unsqueeze(0).to(_device)
        with torch.no_grad():
            outputs = model(x)
            # Support (logits, feature_maps) tuples for Grad-CAM-compatible models.
            logits = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            idx = int(probs.argmax())
            return CLASS_NAMES[idx], float(probs[idx])
    except Exception:
        return None
