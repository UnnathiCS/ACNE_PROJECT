"""
Cloud Inference Engine - V7 Multi-Scale Parallel Version
Handles Roboflow API communication with automated resolution scaling
and concurrent request handling for minimum latency.
"""
import logging
import os
import tempfile
import cv2
import time
import json
import concurrent.futures
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from roboflow import Roboflow
from usage_tracker import log_api_call

logger = logging.getLogger(__name__)

# Per-scale confidence thresholds:
# Scaled images use confidence=10 (permissive) because downscaling
# reduces lesion contrast and the ensemble NMS + SAG gating
# eliminate false positives downstream.
# Native resolution uses confidence=35 (stricter) because full-size
# images have sufficient detail for reliable single-pass detection.
SCALED_CONFIDENCE = 10
NATIVE_CONFIDENCE = 35

# Timeout in seconds for each Roboflow API call
API_CALL_TIMEOUT = 60

# JPEG compression quality for API uploads (0-100).
# Default 85 balances file size vs detail — keeps uploads well under
# Roboflow's ~1 MB limit even at 1280px.
JPEG_QUALITY = int(os.getenv('JPEG_QUALITY', 85))

class CloudInferenceEngine:
    def __init__(self, api_key: str):
        self.rf = Roboflow(api_key=api_key)
        self.max_api_dim = int(os.getenv('MAX_API_DIM', 2048))
        # Model B needs a lower resolution cap to avoid 413 errors.
        # At 2048px, large images produce JPEGs >1 MB which Roboflow
        # rejects.  1280px keeps all tested images under 650 KB.
        self.max_model_b_dim = int(os.getenv('MAX_MODEL_B_DIM', 1280))
        # When False (default), only Model A @ 1280px is sent — saving
        # 33 % of API calls.  The 640px pass is a subset of the 1280px
        # pass and adds negligible detection lift.
        self.enable_dual_scale_a = os.getenv(
            'ENABLE_DUAL_SCALE_A', 'false',
        ).lower() in ('true', '1', 'yes')

    def fetch_multi_scale_consensus(
        self,
        image: np.ndarray,
        model_a_id: str,
        model_b_id: str,
    ) -> Dict[str, Any]:
        """Execute the multi-scale consensus strategy in parallel.

        When ``enable_dual_scale_a`` is *True* (legacy "Triple-Look"),
        three API calls are made: Model A @ 640px, Model A @ 1280px,
        and Model B @ ``max_model_b_dim``.

        When *False* (default), the 640px pass is skipped — reducing
        API usage by 33 % with negligible detection loss.

        Returns:
            Dict with keys ``preds_a_640``, ``preds_a_1280``, ``preds_b``.
            ``preds_a_640`` is always present but may be an empty list
            when dual-scale is disabled.

            Also includes ``_timing`` dict with per-task latency in ms
            and ``_file_sizes`` dict with JPEG upload sizes in bytes.
        """
        H, W = image.shape[:2]

        # Build the task list — optionally include the 640px pass.
        tasks = []
        if self.enable_dual_scale_a:
            tasks.append((model_a_id, 640))
        tasks.append((model_a_id, 1280))
        tasks.append((model_b_id, self.max_model_b_dim))

        results: Dict[str, List[Dict]] = {}
        timing: Dict[str, float] = {}
        file_sizes: Dict[str, int] = {}

        wall_start = time.perf_counter()

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(tasks)) as executor:
            future_to_task = {
                executor.submit(self._fetch_single_scale, image, m_id, dim): f'{m_id}_{dim}'
                for m_id, dim in tasks
            }

            for future in concurrent.futures.as_completed(future_to_task):
                task_name = future_to_task[future]
                try:
                    preds, elapsed_ms, upload_bytes = future.result(timeout=API_CALL_TIMEOUT)
                    results[task_name] = preds
                    timing[task_name] = round(elapsed_ms, 1)
                    file_sizes[task_name] = upload_bytes
                except Exception as e:
                    logger.error('[Cloud Engine] %s: %s', task_name, e)
                    # Extract model_id from task_name (format: model_id_dim)
                    parts = task_name.rsplit('_', 1)
                    failed_model_id = parts[0] if len(parts) > 1 else task_name
                    log_api_call(failed_model_id, 'error', error=str(e))
                    results[task_name] = []

        wall_ms = round((time.perf_counter() - wall_start) * 1000, 1)
        timing['total_wall_ms'] = wall_ms

        # Map internal task keys to stable output keys
        preds_key_a_640 = f'{model_a_id}_640'
        preds_key_a_1280 = f'{model_a_id}_1280'
        preds_key_b = f'{model_b_id}_{self.max_model_b_dim}'

        return {
            'preds_a_640': results.get(preds_key_a_640, []),
            'preds_a_1280': results.get(preds_key_a_1280, []),
            'preds_b': results.get(preds_key_b, []),
            '_timing': {
                'model_a_640_ms': timing.get(preds_key_a_640),
                'model_a_1280_ms': timing.get(preds_key_a_1280),
                'model_b_ms': timing.get(preds_key_b),
                'total_wall_ms': wall_ms,
            },
            '_file_sizes': {
                'model_a_640_bytes': file_sizes.get(preds_key_a_640),
                'model_a_1280_bytes': file_sizes.get(preds_key_a_1280),
                'model_b_bytes': file_sizes.get(preds_key_b),
            },
        }

    def _fetch_single_scale(self, image: np.ndarray, model_id: str, target_dim: int):
        """Internal helper for single API call with scaling.

        Returns:
            Tuple of (predictions list, elapsed_ms, upload_bytes).
        """
        parts = model_id.split('/')
        ws = parts[0] if len(parts) == 3 else 'runner-e0dmy'
        proj = parts[1] if len(parts) == 3 else parts[0]
        ver = parts[2] if len(parts) == 3 else parts[1]

        H, W = image.shape[:2]
        model = self.rf.workspace(ws).project(proj).version(int(ver)).model
        jpeg_params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]

        # Scaling logic
        if max(H, W) > target_dim:
            scale = target_dim / max(H, W)
            temp = cv2.resize(image, (int(W * scale), int(H * scale)))
            tmp = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
            temp_path = tmp.name
            tmp.close()
            cv2.imwrite(temp_path, temp, jpeg_params)

            try:
                upload_bytes = os.path.getsize(temp_path)
                t0 = time.perf_counter()
                res = model.predict(temp_path, confidence=SCALED_CONFIDENCE).json()
                elapsed_ms = (time.perf_counter() - t0) * 1000
                preds = res.get('predictions', [])
                # Re-scale coordinates back to original image size
                for p in preds:
                    p['x'] /= scale; p['y'] /= scale
                    p['width'] /= scale; p['height'] /= scale
                log_api_call(model_id, 'success')
                return preds, elapsed_ms, upload_bytes
            finally:
                os.remove(temp_path)
        else:
            # Native resolution
            tmp = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
            temp_path = tmp.name
            tmp.close()
            cv2.imwrite(temp_path, image, jpeg_params)
            try:
                upload_bytes = os.path.getsize(temp_path)
                t0 = time.perf_counter()
                res = model.predict(temp_path, confidence=NATIVE_CONFIDENCE).json()
                elapsed_ms = (time.perf_counter() - t0) * 1000
                log_api_call(model_id, 'success')
                return res.get('predictions', []), elapsed_ms, upload_bytes
            finally:
                os.remove(temp_path)
