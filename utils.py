import os
import json
from datetime import datetime
import numpy as np
import cv2
from typing import Any

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
EXPORT_DIR = os.path.join(BASE_DIR, 'exports')
THUMBS_DIR = os.path.join(BASE_DIR, 'thumbs')

def ensure_dirs():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(EXPORT_DIR, exist_ok=True)
    os.makedirs(THUMBS_DIR, exist_ok=True)

def load_json(path, default: Any):
    if not os.path.exists(path):
        return default
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def now_iso():
    return datetime.now().isoformat(timespec='seconds')

def today_str():
    return datetime.now().strftime('%Y-%m-%d')

def encoding_to_list(enc: np.ndarray):
    return enc.tolist()

def list_to_encoding(lst):
    return np.array(lst)

def save_thumbnail(emp_id: str, img_bgr):
    """
    img_bgr: OpenCV BGR image (cropped face)
    Saves to thumbs/<emp_id>.jpg (overwrites if exists)
    """
    ensure_dirs()
    path = os.path.join(THUMBS_DIR, f'{emp_id}.jpg')
    # ensure we have some content
    if img_bgr is None or img_bgr.size == 0:
        return None
    # optional: resize thumbnail to fixed height
    h = 200
    h0, w0 = img_bgr.shape[:2]
    scale = h / h0 if h0 > 0 else 1.0
    thumb = cv2.resize(img_bgr, (int(w0*scale), h)) if scale != 1.0 else img_bgr
    cv2.imwrite(path, thumb)
    return path
