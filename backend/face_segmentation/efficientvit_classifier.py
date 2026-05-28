import os
from typing import Tuple, Optional

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
        import timm
    except Exception:
        return None
    try:
        model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=len(CLASS_NAMES))
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
            logits = model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            idx = int(probs.argmax())
            return CLASS_NAMES[idx], float(probs[idx])
    except Exception:
        return None
