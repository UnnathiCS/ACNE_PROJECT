import asyncio
import base64
import json
import logging
import os
import re
import sqlite3
import threading
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Set, Tuple

import cv2
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel, Field
from reportlab.lib import colors as rl_colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.platypus import Table, TableStyle
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from usage_tracker import get_usage_summary

logger = logging.getLogger(__name__)

load_dotenv()

APP_VERSION = '2.0.0'
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / 'uploads'
OUTPUT_DIR = BASE_DIR / 'outputs'
REPORT_DIR = OUTPUT_DIR / 'reports'
DB_PATH = BASE_DIR / 'sessions.db'
MAX_UPLOAD_BYTES = int(os.getenv('MAX_UPLOAD_BYTES', str(10 * 1024 * 1024)))
DEFAULT_RETENTION_HOURS = int(os.getenv('DEFAULT_RETENTION_HOURS', '72'))
MAX_RETENTION_HOURS = int(os.getenv('MAX_RETENTION_HOURS', '720'))
MODEL_A_ID = os.getenv('MODEL_A_ID', 'runner-e0dmy/acne-ijcab/2')
MODEL_B_ID = os.getenv('MODEL_B_ID', 'acne-project-2auvb/acne-detection-v2/1')
ROBOFLOW_API_KEY: Optional[str] = os.getenv('ROBOFLOW_API_KEY')

if not ROBOFLOW_API_KEY:
    raise RuntimeError('ROBOFLOW_API_KEY environment variable is required')

assert ROBOFLOW_API_KEY is not None
API_KEY = ROBOFLOW_API_KEY

CloudInferenceEngine = None
EnsembleLesionMapper = None
FaceSegmentationPipeline = None
draw_region_masks = None
draw_lesion_boxes = None


def ensure_runtime_imports() -> None:
    global CloudInferenceEngine, EnsembleLesionMapper, FaceSegmentationPipeline, draw_region_masks, draw_lesion_boxes
    if CloudInferenceEngine is None:
        from cloud_inference import CloudInferenceEngine as runtime_cloud_engine
        CloudInferenceEngine = runtime_cloud_engine
    if EnsembleLesionMapper is None:
        from face_segmentation.ensemble_mapper import EnsembleLesionMapper as runtime_mapper
        EnsembleLesionMapper = runtime_mapper
    if FaceSegmentationPipeline is None:
        from face_segmentation.pipeline import FaceSegmentationPipeline as runtime_pipeline
        FaceSegmentationPipeline = runtime_pipeline
    if draw_region_masks is None:
        from face_segmentation.utils.visualization import draw_region_masks as runtime_draw_region_masks
        draw_region_masks = runtime_draw_region_masks
    if draw_lesion_boxes is None:
        from face_segmentation.utils.visualization import draw_lesion_boxes as runtime_draw_lesion_boxes
        draw_lesion_boxes = runtime_draw_lesion_boxes


class AnalysisStartRequest(BaseModel):
    session_id: Optional[str] = None
    profile_id: Optional[str] = None
    privacy_mode: bool = False
    retention_hours: int = Field(
        default=DEFAULT_RETENTION_HOURS,
        ge=1,
        le=MAX_RETENTION_HOURS,
    )


class ExportRequest(BaseModel):
    include_pdf_data: bool = True
    preset: Literal['clinical', 'compact', 'presentation'] = 'clinical'
    previous_session_id: Optional[str] = None


class NotesRequest(BaseModel):
    note: str = Field(default='', max_length=5000)


# ---------------------------------------------------------------------------
# Response models (D-4)
# ---------------------------------------------------------------------------


class RootResponse(BaseModel):
    app: str
    status: str
    version: str
    docs: str
    health: str


class HealthResponse(BaseModel):
    status: str
    version: str
    roboflow_api_key_configured: bool
    pipeline_initialized: bool
    cloud_engine_initialized: bool


class VersionResponse(BaseModel):
    app: str
    version: str
    model_a_id: str
    model_b_id: str
    max_upload_bytes: int


class PrivacyResponse(BaseModel):
    privacy_mode_supported: bool
    default_retention_hours: int
    max_retention_hours: int
    purge_endpoint: str
    stored_fields: List[str]


class PurgeResponse(BaseModel):
    purged: bool
    session_id: str


class StatusPayload(BaseModel):
    """Inline status object returned by multiple endpoints."""
    model_config = {'extra': 'allow'}

    session_id: str
    stage: str
    detail: str
    progress: int
    updated_at: str
    completed: Optional[bool] = None
    failed: Optional[bool] = None


class AnalysisStartResponse(BaseModel):
    session_id: str
    profile_id: str
    privacy_mode: bool
    retention_hours: int
    status: StatusPayload


class StatusLatestWrapper(BaseModel):
    """Wrapper for /status/latest which may return a real status or idle stub."""
    model_config = {'extra': 'allow'}

    stage: str
    detail: str
    progress: int
    session_id: Optional[str] = None
    updated_at: Optional[str] = None
    completed: Optional[bool] = None
    failed: Optional[bool] = None


class StatusLatestResponse(BaseModel):
    status: StatusLatestWrapper


class SessionSummaryItem(BaseModel):
    """Lightweight session row used in /history items."""
    model_config = {'extra': 'allow'}

    session_id: str
    profile_id: Optional[str] = None
    timestamp: str
    severity: Optional[str] = None
    gags_score: Optional[int] = None
    lesion_count: Optional[int] = None
    symmetry_delta: Optional[float] = None
    privacy_mode: bool
    retention_hours: int
    status: Optional[Dict[str, Any]] = None


class HistoryResponse(BaseModel):
    items: List[SessionSummaryItem]
    next_cursor: Optional[str] = None


class ProfileItem(BaseModel):
    profile_id: str
    sessions: int
    latest_timestamp: Optional[str] = None
    latest_severity: Optional[str] = None


class ProfilesResponse(BaseModel):
    items: List[ProfileItem]


class SessionDetailResponse(BaseModel):
    """Full session detail returned by GET /session/{id}."""
    model_config = {'extra': 'allow'}

    session_id: str
    profile_id: Optional[str] = None
    timestamp: str
    severity: Optional[str] = None
    gags_score: Optional[int] = None
    lesion_count: Optional[int] = None
    symmetry_delta: Optional[float] = None
    results: Optional[Dict[str, Any]] = None
    note: Optional[str] = None
    diagnostic_image_path: Optional[str] = None
    original_image_path: Optional[str] = None
    diagnostic_image: Optional[str] = None
    original_image: Optional[str] = None
    privacy_mode: bool
    retention_hours: int
    status: Optional[Dict[str, Any]] = None


class NotesResponse(BaseModel):
    session_id: str
    note: str


class CompareDetailModel(BaseModel):
    """Shape of the compare payload when comparison data is available."""
    model_config = {'extra': 'allow'}

    previous_session_id: str
    current_session_id: str
    previous_timestamp: str
    current_timestamp: str
    severity_change: Dict[str, str]
    lesion_delta: int
    gags_delta: int
    symmetry_delta_change: float
    regions: Dict[str, Dict[str, Any]]
    comparison_mode: Optional[str] = None


class CompareResponse(BaseModel):
    current_session_id: str
    compare: Optional[CompareDetailModel] = None


class ReportDetail(BaseModel):
    model_config = {'extra': 'allow'}

    clinical_analysis: Dict[str, Any] = Field(default_factory=dict)
    consensus_summary: Dict[str, Any] = Field(default_factory=dict)
    compare: Optional[Dict[str, Any]] = None
    pdf_path: str
    pdf_data_uri: Optional[str] = None


class ReportResponse(BaseModel):
    session_id: str
    report: ReportDetail


class ExportResponse(BaseModel):
    session_id: str
    pdf_path: str
    preset: str
    pdf_data_uri: Optional[str] = None


class AnalyzeResultResponse(BaseModel):
    """Response for POST /analyze — large payload with full analysis results."""
    model_config = {'extra': 'allow'}

    session_id: str
    status: StatusPayload
    severity: Optional[str] = None
    gags_score: Optional[int] = None
    lesion_count: Optional[int] = None
    symmetry_delta: Optional[float] = None
    results: Optional[Dict[str, Any]] = None
    compare: Optional[Dict[str, Any]] = None
    original_image: str
    diagnostic_image: str


model_init_lock = threading.Lock()


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def utcnow_iso() -> str:
    return utcnow().isoformat()


def parse_ts(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def ensure_dirs() -> None:
    for path in (UPLOAD_DIR, OUTPUT_DIR, REPORT_DIR):
        path.mkdir(parents=True, exist_ok=True)


def validate_session_id(session_id: str) -> str:
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,128}', session_id):
        raise HTTPException(status_code=400, detail='Invalid session_id format')
    return session_id


def validate_profile_id(profile_id: str) -> str:
    """Validate profile_id: alphanumeric, dashes, underscores, dots; max 64 chars."""
    if not re.fullmatch(r'[A-Za-z0-9_.\-]{1,64}', profile_id):
        raise HTTPException(status_code=400, detail='Invalid profile_id format')
    return profile_id


def validate_cursor(cursor: str) -> str:
    """Validate pagination cursor: ISO-8601 timestamp format, max 64 chars."""
    if len(cursor) > 64:
        raise HTTPException(status_code=400, detail='Invalid cursor format')
    if not re.fullmatch(r'[\d\-T:+.Z ]{1,64}', cursor):
        raise HTTPException(status_code=400, detail='Invalid cursor format')
    return cursor


def normalize_retention(hours: int) -> int:
    return max(1, min(int(hours), MAX_RETENTION_HOURS))


def safe_unlink(path: Optional[str]) -> None:
    if not path:
        return
    try:
        file_path = Path(path).resolve()
        # Path containment: only delete files under managed directories
        managed_roots = (UPLOAD_DIR.resolve(), OUTPUT_DIR.resolve(), REPORT_DIR.resolve())
        if not any(file_path.is_relative_to(r) for r in managed_roots):
            return
        if file_path.exists() and file_path.is_file():
            file_path.unlink()
    except OSError:
        pass


def absolute_managed_path(path: Path) -> str:
    return str(path.resolve())


def bytes_to_data_uri(payload: bytes, mime: str) -> str:
    encoded = base64.b64encode(payload).decode('ascii')
    return f'data:{mime};base64,{encoded}'


def file_to_data_uri(path: Optional[str], mime: str) -> Optional[str]:
    if not path:
        return None
    file_path = Path(path)
    if not file_path.is_absolute():
        file_path = (BASE_DIR / file_path).resolve()
    managed_roots = (UPLOAD_DIR.resolve(), OUTPUT_DIR.resolve(), REPORT_DIR.resolve())
    if not any(file_path.is_relative_to(r) for r in managed_roots):
        return None
    if not file_path.exists():
        return None
    return bytes_to_data_uri(file_path.read_bytes(), mime)


def image_to_jpeg_bytes(image: np.ndarray) -> bytes:
    ok, buf = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        raise HTTPException(status_code=500, detail='Failed to encode image output')
    return buf.tobytes()


def save_image(path: Path, image: np.ndarray) -> str:
    if not cv2.imwrite(str(path), image):
        raise HTTPException(status_code=500, detail=f'Failed to save image: {path.name}')
    return str(path)


def validate_upload(upload: UploadFile, payload: bytes) -> None:
    if upload.content_type not in {'image/jpeg', 'image/png', 'image/webp'}:
        raise HTTPException(
            status_code=415,
            detail='Only JPEG, PNG, and WEBP images are supported',
        )
    if not payload:
        raise HTTPException(status_code=400, detail='Uploaded file is empty')
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f'File exceeds {MAX_UPLOAD_BYTES} byte limit',
        )
    # Detect actual content type from magic bytes
    MAGIC_BYTES = {b'\xff\xd8\xff': 'image/jpeg', b'\x89PNG': 'image/png'}
    detected_type: Optional[str] = None
    for sig, mime in MAGIC_BYTES.items():
        if payload.startswith(sig):
            detected_type = mime
            break
    # WEBP: RIFF container with WEBP identifier at bytes 8-12
    if payload[:4] == b'RIFF' and len(payload) >= 12 and payload[8:12] == b'WEBP':
        detected_type = 'image/webp'
    if detected_type is None:
        raise HTTPException(status_code=400, detail='File content does not match a supported image format')
    # Cross-validate: declared Content-Type must match detected type
    if upload.content_type != detected_type:
        raise HTTPException(status_code=400, detail='Declared Content-Type does not match file content')


def decode_image(payload: bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail='Failed to decode image')
    return image


def consensus_summary(assignments: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    # Filter out internal metadata keys (e.g. _pipeline_metrics) that are
    # not region->lesion-list mappings.
    region_assignments = {
        k: v for k, v in assignments.items()
        if not k.startswith('_') and isinstance(v, list)
    }
    region_counts = {
        region: len(items)
        for region, items in region_assignments.items()
        if region != 'unassigned' and items
    }
    confidence_values = [
        item.get('confidence', 0.0)
        for items in region_assignments.values()
        for item in items
    ]
    total = sum(region_counts.values())
    top_regions = sorted(region_counts.items(), key=lambda item: item[1], reverse=True)[:3]
    summary = 'No verified lesions detected'
    if total:
        summary = f"{total} verified lesions concentrated in {', '.join(name for name, _ in top_regions)}"

    # Build flat lesion list with region tag for the frontend
    lesions: List[Dict[str, Any]] = []
    for region, items in region_assignments.items():
        if region == 'unassigned':
            continue
        for item in items:
            lesions.append({**item, 'region': region})

    # Acne type distribution across all verified lesions
    type_counts: Dict[str, int] = {}
    for lesion in lesions:
        cls = lesion.get('class_name', 'acne')
        type_counts[cls] = type_counts.get(cls, 0) + 1

    return {
        'verified_lesions': total,
        'average_confidence': round(sum(confidence_values) / len(confidence_values), 3)
        if confidence_values
        else 0.0,
        'top_regions': [{'region': name, 'count': count} for name, count in top_regions],
        'region_counts': region_counts,
        'unassigned_count': len(region_assignments.get('unassigned', [])),
        'summary': summary,
        'lesions': lesions,
        'type_counts': type_counts,
    }


def _safe_json_loads(raw: Optional[str]) -> Optional[Any]:
    """Parse a JSON string, returning None on failure instead of crashing."""
    if not raw:
        return None
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None


def row_to_session(row: sqlite3.Row, status: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        'session_id': row['session_id'],
        'profile_id': row['profile_id'] if 'profile_id' in row.keys() else None,
        'timestamp': row['timestamp'],
        'severity': row['severity'],
        'gags_score': row['gags_score'],
        'lesion_count': row['lesion_count'],
        'symmetry_delta': row['symmetry_delta'],
        'results': _safe_json_loads(row['results_json']),
        'note': row['note'] if 'note' in row.keys() else '',
        'diagnostic_image_path': row['diagnostic_image_path'],
        'original_image_path': row['original_image_path'],
        'diagnostic_image': file_to_data_uri(row['diagnostic_image_path'], 'image/jpeg'),
        'original_image': file_to_data_uri(row['original_image_path'], 'image/jpeg'),
        'privacy_mode': bool(row['privacy_mode']),
        'retention_hours': row['retention_hours'],
        'status': status,
    }


def compare_payload(
    previous: Optional[Dict[str, Any]],
    current: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    if not previous or not previous.get('results') or not current.get('results'):
        return None
    prev = previous['results'].get('clinical_analysis', {})
    curr = current['results'].get('clinical_analysis', {})
    prev_regions = prev.get('regions', {})
    curr_regions = curr.get('regions', {})
    regions = {}
    for region in sorted(set(prev_regions) | set(curr_regions)):
        prev_region = prev_regions.get(region, {})
        curr_region = curr_regions.get(region, {})
        regions[region] = {
            'previous_count': prev_region.get('count', 0),
            'current_count': curr_region.get('count', 0),
            'count_delta': curr_region.get('count', 0) - prev_region.get('count', 0),
            'previous_lpi': prev_region.get('lpi', 0),
            'current_lpi': curr_region.get('lpi', 0),
            'lpi_delta': round(curr_region.get('lpi', 0) - prev_region.get('lpi', 0), 2),
        }
    return {
        'previous_session_id': previous['session_id'],
        'current_session_id': current['session_id'],
        'previous_timestamp': previous['timestamp'],
        'current_timestamp': current['timestamp'],
        'severity_change': {
            'from': prev.get('clinical_severity', 'Unknown'),
            'to': curr.get('clinical_severity', 'Unknown'),
        },
        'lesion_delta': curr.get('total_lesions', 0) - prev.get('total_lesions', 0),
        'gags_delta': curr.get('gags_total_score', 0) - prev.get('gags_total_score', 0),
        'symmetry_delta_change': round(
            curr.get('symmetry_delta', 0) - prev.get('symmetry_delta', 0),
            2,
        ),
        'regions': regions,
    }


def annotate_compare_payload(
    compare: Optional[Dict[str, Any]],
    explicit_previous: bool,
) -> Optional[Dict[str, Any]]:
    if not compare:
        return None
    compare['comparison_mode'] = 'selected_baseline' if explicit_previous else 'previous_archived_session'
    return compare


def get_delta_status(value: float, better_when_lower: bool = True) -> str:
    if value == 0:
        return 'stable'
    improved = value < 0 if better_when_lower else value > 0
    return 'improved' if improved else 'worsened'


def format_delta_number(value: Any) -> str:
    numeric = float(value)
    if numeric.is_integer():
        return str(int(numeric))
    return str(round(numeric, 2))


def format_signed_delta(value: Any, suffix: str = '') -> str:
    numeric = float(value)
    prefix = '+' if numeric > 0 else ''
    return f'{prefix}{format_delta_number(numeric)}{suffix}'


def describe_delta(value: float, label: str, better_when_lower: bool = True) -> str:
    status = get_delta_status(value, better_when_lower)
    if status == 'stable':
        return f'{label} remained stable'
    magnitude = format_delta_number(abs(float(value)))
    return f'{label} {status} by {magnitude}'


def compare_target_label(compare: Dict[str, Any]) -> str:
    if compare.get('comparison_mode') == 'selected_baseline':
        return 'the selected baseline session'
    return 'the previous archived session'


def comparison_summary(compare: Dict[str, Any]) -> str:
    return (
        f"{describe_delta(compare.get('lesion_delta', 0), 'Lesion burden')} and "
        f"{describe_delta(compare.get('gags_delta', 0), 'GAGS')} versus "
        f"{compare_target_label(compare)}."
    )


def top_region_changes(compare: Dict[str, Any], limit: int = 3) -> List[Any]:
    region_items = list((compare.get('regions') or {}).items())
    ranked = sorted(
        region_items,
        key=lambda item: abs(item[1].get('count_delta', 0)),
        reverse=True,
    )
    return [item for item in ranked if item[1].get('count_delta', 0) != 0][:limit]


class BridgeStore:
    def __init__(self, db_path: Path):
        self.lock = threading.Lock()
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.conn.execute('PRAGMA journal_mode=WAL')
        self.conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self) -> None:
        with self.lock:
            self.conn.execute(
                '''
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    profile_id TEXT,
                    timestamp TEXT NOT NULL,
                    severity TEXT,
                    gags_score INTEGER,
                    lesion_count INTEGER,
                    symmetry_delta REAL,
                    results_json TEXT,
                    note TEXT,
                    diagnostic_image_path TEXT,
                    original_image_path TEXT,
                    privacy_mode INTEGER NOT NULL DEFAULT 0,
                    retention_hours INTEGER NOT NULL DEFAULT 72
                )
                '''
            )
            try:
                self.conn.execute('ALTER TABLE sessions ADD COLUMN profile_id TEXT')
            except sqlite3.OperationalError:
                pass
            try:
                self.conn.execute('ALTER TABLE sessions ADD COLUMN retention_hours INTEGER NOT NULL DEFAULT 72')
            except sqlite3.OperationalError:
                pass
            try:
                self.conn.execute('ALTER TABLE sessions ADD COLUMN privacy_mode INTEGER NOT NULL DEFAULT 0')
            except sqlite3.OperationalError:
                pass
            try:
                self.conn.execute('ALTER TABLE sessions ADD COLUMN original_image_path TEXT')
            except sqlite3.OperationalError:
                pass
            try:
                self.conn.execute('ALTER TABLE sessions ADD COLUMN note TEXT')
            except sqlite3.OperationalError:
                pass
            self.conn.execute(
                '''
                CREATE TABLE IF NOT EXISTS statuses (
                    session_id TEXT PRIMARY KEY,
                    updated_at TEXT NOT NULL,
                    status_json TEXT NOT NULL
                )
                '''
            )
            self.conn.execute('CREATE INDEX IF NOT EXISTS idx_sessions_timestamp ON sessions (timestamp)')
            self.conn.execute('CREATE INDEX IF NOT EXISTS idx_sessions_profile_id ON sessions (profile_id)')
            self.conn.execute('CREATE INDEX IF NOT EXISTS idx_statuses_updated_at ON statuses (updated_at)')
            self.conn.commit()

    def close(self) -> None:
        with self.lock:
            self.conn.close()

    def upsert_session(self, payload: Dict[str, Any]) -> None:
        with self.lock:
            self.conn.execute(
                '''
                INSERT INTO sessions (
                    session_id, profile_id, timestamp, severity, gags_score, lesion_count,
                    symmetry_delta, results_json, note, diagnostic_image_path,
                    original_image_path, privacy_mode, retention_hours
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    profile_id=excluded.profile_id,
                    timestamp=excluded.timestamp,
                    severity=excluded.severity,
                    gags_score=excluded.gags_score,
                    lesion_count=excluded.lesion_count,
                    symmetry_delta=excluded.symmetry_delta,
                    results_json=excluded.results_json,
                    note=excluded.note,
                    diagnostic_image_path=excluded.diagnostic_image_path,
                    original_image_path=excluded.original_image_path,
                    privacy_mode=excluded.privacy_mode,
                    retention_hours=excluded.retention_hours
                ''',
                (
                    payload['session_id'],
                    payload.get('profile_id'),
                    payload['timestamp'],
                    payload.get('severity'),
                    payload.get('gags_score'),
                    payload.get('lesion_count'),
                    payload.get('symmetry_delta'),
                    payload.get('results_json'),
                    payload.get('note', ''),
                    payload.get('diagnostic_image_path'),
                    payload.get('original_image_path'),
                    1 if payload.get('privacy_mode') else 0,
                    payload.get('retention_hours', DEFAULT_RETENTION_HOURS),
                ),
            )
            self.conn.commit()

    def set_status(
        self,
        session_id: str,
        stage: str,
        detail: str,
        progress: int,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        status = {
            'session_id': session_id,
            'stage': stage,
            'detail': detail,
            'progress': int(max(0, min(progress, 100))),
            'updated_at': utcnow_iso(),
        }
        if extra:
            status.update(extra)
        with self.lock:
            self.conn.execute(
                '''
                INSERT INTO statuses (session_id, updated_at, status_json)
                VALUES (?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    updated_at=excluded.updated_at,
                    status_json=excluded.status_json
                ''',
                (session_id, status['updated_at'], json.dumps(status)),
            )
            self.conn.commit()
        return status

    def get_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        with self.lock:
            row = self.conn.execute(
                'SELECT status_json FROM statuses WHERE session_id = ?',
                (session_id,),
            ).fetchone()
        return _safe_json_loads(row['status_json']) if row else None

    def latest_status(self) -> Optional[Dict[str, Any]]:
        with self.lock:
            row = self.conn.execute(
                'SELECT status_json FROM statuses ORDER BY updated_at DESC LIMIT 1'
            ).fetchone()
        return _safe_json_loads(row['status_json']) if row else None

    def get_session_row(self, session_id: str) -> Optional[sqlite3.Row]:
        with self.lock:
            return self.conn.execute(
                'SELECT * FROM sessions WHERE session_id = ?',
                (session_id,),
            ).fetchone()

    def session_payload(self, session_id: str) -> Optional[Dict[str, Any]]:
        row = self.get_session_row(session_id)
        if not row:
            return None
        return row_to_session(row, self.get_status(session_id))

    def previous_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        current = self.get_session_row(session_id)
        if not current:
            return None
        with self.lock:
            row = self.conn.execute(
                '''
                SELECT * FROM sessions
                WHERE timestamp < ?
                  AND session_id != ?
                  AND results_json IS NOT NULL
                ORDER BY timestamp DESC
                LIMIT 1
                ''',
                (current['timestamp'], session_id),
            ).fetchone()
        if not row:
            return None
        return row_to_session(row, self.get_status(row['session_id']))

    def history(self, limit: int = 25, profile_id: Optional[str] = None, cursor: Optional[str] = None) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        """Fetch session history ordered by timestamp DESC.

        Args:
            limit: Max rows to return (clamped 1-200). One extra row is fetched
                internally to determine whether a next page exists.
            profile_id: Optional filter to a single profile.
            cursor: ISO-8601 timestamp cursor; only rows with timestamp
                strictly less than this value are returned.

        Returns:
            A list of session summary dicts. The caller (endpoint) wraps this
            in an envelope that includes ``next_cursor``.
        """
        clamped_limit = max(1, min(limit, 200))
        # Fetch one extra to detect whether a next page exists
        fetch_limit = clamped_limit + 1
        conditions: List[str] = []
        params: List[object] = []

        if profile_id:
            conditions.append('profile_id = ?')
            params.append(profile_id)
        if cursor:
            conditions.append('timestamp < ?')
            params.append(cursor)

        where_clause = (' WHERE ' + ' AND '.join(conditions)) if conditions else ''
        query = (
            'SELECT session_id, profile_id, timestamp, severity, gags_score, '
            'lesion_count, symmetry_delta, privacy_mode, retention_hours '
            f'FROM sessions{where_clause} ORDER BY timestamp DESC LIMIT ?'
        )
        params.append(fetch_limit)

        with self.lock:
            rows = self.conn.execute(query, params).fetchall()
        if not rows:
            return [], None
        # Determine whether there is a next page
        has_next = len(rows) > clamped_limit
        page_rows = rows[:clamped_limit]
        session_ids = [row['session_id'] for row in page_rows]
        placeholders = ','.join('?' for _ in session_ids)
        with self.lock:
            status_rows = self.conn.execute(
                f'SELECT session_id, status_json FROM statuses WHERE session_id IN ({placeholders})',
                session_ids,
            ).fetchall()
        status_map: Dict[str, Dict[str, Any]] = {
            r['session_id']: json.loads(r['status_json']) for r in status_rows
        }
        items = [
            {
                'session_id': row['session_id'],
                'profile_id': row['profile_id'],
                'timestamp': row['timestamp'],
                'severity': row['severity'],
                'gags_score': row['gags_score'],
                'lesion_count': row['lesion_count'],
                'symmetry_delta': row['symmetry_delta'],
                'privacy_mode': bool(row['privacy_mode']),
                'retention_hours': row['retention_hours'],
                'status': status_map.get(row['session_id']),
            }
            for row in page_rows
        ]
        next_cursor = page_rows[-1]['timestamp'] if has_next else None
        return items, next_cursor

    def list_sessions(self, limit: int = 1000) -> List[Dict[str, Any]]:
        """Return lightweight session dicts with parsed results for metrics.

        Unlike ``history()`` this includes ``results_json`` (parsed) but
        skips image data-URI generation for performance.

        Args:
            limit: Maximum number of sessions to return.

        Returns:
            List of dicts with session_id, timestamp, and parsed results.
        """
        with self.lock:
            rows = self.conn.execute(
                'SELECT session_id, timestamp, results_json '
                'FROM sessions ORDER BY timestamp DESC LIMIT ?',
                (limit,),
            ).fetchall()
        items: List[Dict[str, Any]] = []
        for row in rows:
            results = None
            if row['results_json']:
                try:
                    results = json.loads(row['results_json'])
                except (json.JSONDecodeError, TypeError):
                    pass
            items.append({
                'session_id': row['session_id'],
                'timestamp': row['timestamp'],
                'results': results,
            })
        return items

    def purge(self, session_id: str) -> bool:
        row = self.get_session_row(session_id)
        if not row:
            return False
        safe_unlink(row['diagnostic_image_path'])
        safe_unlink(row['original_image_path'])
        # Purge all preset variants of the report PDF
        for report_file in REPORT_DIR.glob(f'{session_id}_*_report.pdf'):
            safe_unlink(str(report_file))
        with self.lock:
            self.conn.execute('DELETE FROM statuses WHERE session_id = ?', (session_id,))
            self.conn.execute('DELETE FROM sessions WHERE session_id = ?', (session_id,))
            self.conn.commit()
        return True

    def cleanup_expired(self) -> Dict[str, int]:
        now = utcnow()
        with self.lock:
            rows = self.conn.execute(
                '''
                SELECT session_id, timestamp, retention_hours, diagnostic_image_path, original_image_path
                FROM sessions
                '''
            ).fetchall()
        purged_sessions = 0
        live_paths: Set[str] = set()
        for row in rows:
            try:
                expires_at = parse_ts(row['timestamp']) + timedelta(
                    hours=int(row['retention_hours'] or DEFAULT_RETENTION_HOURS)
                )
            except Exception:
                expires_at = now - timedelta(seconds=1)
            if expires_at <= now:
                if self.purge(row['session_id']):
                    purged_sessions += 1
                continue
            for key in ('diagnostic_image_path', 'original_image_path'):
                if row[key]:
                    live_paths.add(str(Path(row[key]).resolve()))
            # Track all preset variants of the report PDF
            for report_file in REPORT_DIR.glob(f"{row['session_id']}_*_report.pdf"):
                live_paths.add(str(report_file.resolve()))
        with self.lock:
            self.conn.execute('DELETE FROM statuses WHERE session_id NOT IN (SELECT session_id FROM sessions)')
            self.conn.commit()
        purged_files = self._cleanup_files(live_paths)
        return {'purged_sessions': purged_sessions, 'purged_files': purged_files}

    def _cleanup_files(self, live_paths: Set[str]) -> int:
        cutoff = utcnow() - timedelta(hours=DEFAULT_RETENTION_HOURS)
        removed = 0
        for root in (UPLOAD_DIR, OUTPUT_DIR, REPORT_DIR):
            if not root.exists():
                continue
            for path in root.rglob('*'):
                if not path.is_file():
                    continue
                resolved = str(path.resolve())
                if resolved in live_paths:
                    continue
                try:
                    modified = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
                except OSError:
                    continue
                managed = path.name.endswith('_original.jpg') or path.name.endswith('_diagnostic.jpg')
                managed = managed or path.name.endswith('_report.pdf')
                if managed and modified <= cutoff:
                    safe_unlink(str(path))
                    removed += 1
        return removed


def session_stub(session_id: str, privacy_mode: bool, retention_hours: int) -> Dict[str, Any]:
    return {
        'session_id': session_id,
        'profile_id': None,
        'timestamp': utcnow_iso(),
        'severity': None,
        'gags_score': None,
        'lesion_count': None,
        'symmetry_delta': None,
        'results_json': None,
        'note': '',
        'diagnostic_image_path': None,
        'original_image_path': None,
        'privacy_mode': privacy_mode,
        'retention_hours': normalize_retention(retention_hours),
    }


def require_session(session_id: str) -> Dict[str, Any]:
    session = get_store().session_payload(validate_session_id(session_id))
    if not session:
        raise HTTPException(status_code=404, detail='Session not found')
    return session


def summarize_stream_provenance(cloud_results: Dict[str, Any]) -> Dict[str, Any]:
    stream_keys = {
        'model_a_640': 'preds_a_640',
        'model_a_1280': 'preds_a_1280',
        'model_b_native': 'preds_b',
    }
    streams: Dict[str, int] = {}
    stream_classes: Dict[str, Dict[str, int]] = {}
    for display_name, data_key in stream_keys.items():
        preds = cloud_results.get(data_key, [])
        streams[display_name] = len(preds)
        class_dist: Dict[str, int] = {}
        for p in preds:
            cls = str(p.get('class', 'acne'))
            class_dist[cls] = class_dist.get(cls, 0) + 1
        if class_dist:
            stream_classes[display_name] = class_dist
    strongest_stream = max(streams, key=lambda key: streams[key]) if streams else None
    return {
        'streams': streams,
        'strongest_stream': strongest_stream,
        'stream_total': sum(streams.values()),
        'stream_classes': stream_classes,
    }


def write_pdf_report(session: Dict[str, Any], compare: Optional[Dict[str, Any]], preset: str = 'clinical') -> Path:
    """Generate a professional clinical PDF report for an analysis session.

    Args:
        session: Full session dict with results, severity, gags_score, etc.
        compare: Optional comparison payload against a previous session.
        preset: Report style — 'clinical' (full), 'compact' (one-page), or 'presentation'.

    Returns:
        Path to the generated PDF file.
    """
    report_path = REPORT_DIR / f"{session['session_id']}_{preset}_report.pdf"
    results = session.get('results') or {}
    clinical = results.get('clinical_analysis', {})
    consensus = results.get('consensus_summary', {})
    regions = clinical.get('regions', {})
    type_counts = consensus.get('type_counts', {})
    pipeline_metrics = results.get('pipeline_metrics') or results.get('lesions', {}).get('_pipeline_metrics', {})
    timing_ms = results.get('timing_ms', {})

    page_width, page_height = letter
    left_margin = 54
    right_margin = 54
    content_width = page_width - left_margin - right_margin
    page_num = [1]  # mutable counter for closures

    # --- Color palette (dark text on white) ---
    CLR_TITLE = (0.1, 0.1, 0.15)
    CLR_HEADING = (0.12, 0.35, 0.55)
    CLR_BODY = (0.15, 0.15, 0.2)
    CLR_MUTED = (0.4, 0.4, 0.45)
    CLR_ACCENT = (0.0, 0.55, 0.7)
    CLR_RULE = (0.75, 0.8, 0.85)

    pdf = canvas.Canvas(str(report_path), pagesize=letter)
    pdf.setTitle(f'Acne Diagnostic Report - {session["session_id"]}')
    pdf.setAuthor('ClearSkin Analytics')

    dominant_regions = sorted(
        regions.items(),
        key=lambda item: (item[1].get('gags_score', 0), item[1].get('count', 0)),
        reverse=True,
    )[:3]
    region_changes = top_region_changes(compare) if compare else []

    # --- Helper functions ---
    def draw_footer() -> None:
        pdf.setFont('Helvetica', 7)
        pdf.setFillColorRGB(*CLR_MUTED)
        pdf.drawString(left_margin, 36, f'ClearSkin Analytics  |  Session {session["session_id"][:16]}...')
        pdf.drawRightString(page_width - right_margin, 36, f'Page {page_num[0]}')
        pdf.setStrokeColorRGB(*CLR_RULE)
        pdf.setLineWidth(0.5)
        pdf.line(left_margin, 48, page_width - right_margin, 48)

    def new_page() -> float:
        draw_footer()
        pdf.showPage()
        page_num[0] += 1
        return page_height - 48

    def ensure_space(y: float, needed: float = 80) -> float:
        if y >= needed + 56:
            return y
        return new_page()

    def line(
        y: float,
        text: str,
        font: str = 'Helvetica',
        size: int = 10,
        color: tuple = CLR_BODY,
    ) -> float:
        pdf.setFillColorRGB(*color)
        pdf.setFont(font, size)
        pdf.drawString(left_margin, y, text)
        return y - (size + 4)

    def wrap_text(text: str, font: str = 'Helvetica', size: int = 10, max_w: float = 0) -> List[str]:
        w = max_w or content_width
        words = str(text).split()
        if not words:
            return ['']
        lines: List[str] = []
        current = words[0]
        for word in words[1:]:
            candidate = f'{current} {word}'
            if pdf.stringWidth(candidate, font, size) <= w:
                current = candidate
            else:
                lines.append(current)
                current = word
        lines.append(current)
        return lines

    def block(
        y: float,
        text: str,
        font: str = 'Helvetica',
        size: int = 10,
        color: tuple = CLR_BODY,
    ) -> float:
        for chunk in wrap_text(text, font, size):
            y = ensure_space(y)
            y = line(y, chunk, font, size, color)
        return y - 2

    def section(y: float, label: str, size: int = 12) -> float:
        y = ensure_space(y, 100)
        y -= 6
        pdf.setStrokeColorRGB(*CLR_RULE)
        pdf.setLineWidth(0.5)
        pdf.line(left_margin, y + size + 8, left_margin + content_width, y + size + 8)
        y = line(y, label, 'Helvetica-Bold', size, CLR_HEADING)
        return y - 2

    def bullet(y: float, text: str, size: int = 10) -> float:
        y = ensure_space(y)
        pdf.setFillColorRGB(*CLR_ACCENT)
        pdf.setFont('Helvetica', size)
        pdf.drawString(left_margin + 4, y, '\u2022')
        for i, chunk in enumerate(wrap_text(text, 'Helvetica', size, content_width - 16)):
            if i > 0:
                y = ensure_space(y)
            pdf.setFillColorRGB(*CLR_BODY)
            pdf.setFont('Helvetica', size)
            pdf.drawString(left_margin + 16, y, chunk)
            y -= (size + 4)
        return y - 1

    def metric(y: float, label: str, value: str) -> float:
        y = ensure_space(y)
        pdf.setFont('Helvetica-Bold', 10)
        pdf.setFillColorRGB(*CLR_BODY)
        pdf.drawString(left_margin, y, f'{label}:')
        label_w = pdf.stringWidth(f'{label}:', 'Helvetica-Bold', 10)
        pdf.setFont('Helvetica', 10)
        pdf.drawString(left_margin + label_w + 6, y, value)
        return y - 14

    def severity_color(sev: str) -> tuple:
        sev_lower = (sev or '').lower()
        if 'very severe' in sev_lower or 'cystic' in sev_lower:
            return (0.7, 0.1, 0.1)
        if 'severe' in sev_lower:
            return (0.8, 0.3, 0.0)
        if 'moderate' in sev_lower:
            return (0.75, 0.55, 0.0)
        if 'mild' in sev_lower:
            return (0.0, 0.6, 0.3)
        return CLR_BODY

    def draw_table(y: float, headers: List[str], rows: List[List[str]],
                   col_widths: Optional[List[float]] = None) -> float:
        """Draw a simple table and return the new y position."""
        data = [headers] + rows
        if col_widths is None:
            col_widths = [content_width / len(headers)] * len(headers)
        table_height = len(data) * 18
        y = ensure_space(y, table_height + 20)
        t = Table(data, colWidths=col_widths, rowHeights=[18] * len(data))
        t.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('TEXTCOLOR', (0, 0), (-1, 0), rl_colors.HexColor('#1E5A8C')),
            ('TEXTCOLOR', (0, 1), (-1, -1), rl_colors.HexColor('#262630')),
            ('BACKGROUND', (0, 0), (-1, 0), rl_colors.HexColor('#E8EEF4')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [rl_colors.white, rl_colors.HexColor('#F5F7FA')]),
            ('GRID', (0, 0), (-1, -1), 0.5, rl_colors.HexColor('#C0C8D0')),
            ('ALIGN', (1, 0), (-1, -1), 'RIGHT'),
            ('TOPPADDING', (0, 0), (-1, -1), 2),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
            ('LEFTPADDING', (0, 0), (-1, -1), 6),
            ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ]))
        tw, th = t.wrap(content_width, page_height)
        t.drawOn(pdf, left_margin, y - th)
        return y - th - 8

    def draw_diagnostic_image(y: float) -> float:
        """Embed the diagnostic image if available."""
        diag_path = session.get('diagnostic_image_path')
        if not diag_path:
            return y
        img_file = Path(diag_path)
        if not img_file.exists():
            return y
        y = ensure_space(y, 260)
        try:
            max_img_w = content_width * 0.75
            max_img_h = 220
            img = cv2.imread(str(img_file))
            if img is None:
                return y
            ih, iw = img.shape[:2]
            scale = min(max_img_w / iw, max_img_h / ih, 1.0)
            draw_w = iw * scale
            draw_h = ih * scale
            # Center the image
            x_offset = left_margin + (content_width - draw_w) / 2
            pdf.drawImage(str(img_file), x_offset, y - draw_h,
                          width=draw_w, height=draw_h, preserveAspectRatio=True)
            y -= draw_h + 6
            pdf.setFont('Helvetica', 8)
            pdf.setFillColorRGB(*CLR_MUTED)
            caption = 'Diagnostic overlay with detected lesion bounding boxes'
            cw = pdf.stringWidth(caption, 'Helvetica', 8)
            pdf.drawString(left_margin + (content_width - cw) / 2, y, caption)
            y -= 14
        except Exception:
            pass
        return y

    # ================================================================
    # TITLE BLOCK (all presets)
    # ================================================================
    y = page_height - 48
    pdf.setFillColorRGB(*CLR_TITLE)
    pdf.setFont('Helvetica-Bold', 20)

    title_map = {
        'clinical': 'Acne Severity Analysis Report',
        'compact': 'Acne Summary Report',
        'presentation': 'Case Presentation Report',
    }
    subtitle_map = {
        'clinical': 'Full diagnostic worksheet with regional GAGS scoring and lesion classification.',
        'compact': 'One-page follow-up snapshot for quick clinical reference.',
        'presentation': 'Narrative summary for case conferences and review meetings.',
    }
    pdf.drawString(left_margin, y, title_map.get(preset, 'Acne Severity Analysis Report'))
    y -= 26
    pdf.setFont('Helvetica', 10)
    pdf.setFillColorRGB(*CLR_MUTED)
    pdf.drawString(left_margin, y, subtitle_map.get(preset, ''))
    y -= 18

    # Metadata line
    ts_display = session.get('timestamp', 'N/A')
    if len(ts_display) > 19:
        ts_display = ts_display[:19].replace('T', ' ')
    y = metric(y, 'Session', session['session_id'])
    y = metric(y, 'Profile', session.get('profile_id') or 'default-profile')
    y = metric(y, 'Date', ts_display)
    y -= 4

    # ================================================================
    # SEVERITY BADGE
    # ================================================================
    sev = str(session.get('severity') or 'Unknown')
    gags = session.get('gags_score') or 0
    lesion_count = session.get('lesion_count') or 0
    sym_delta = session.get('symmetry_delta') or 0

    y = ensure_space(y, 50)
    # Draw severity highlight box
    box_h = 36
    pdf.setStrokeColorRGB(*CLR_RULE)
    pdf.setLineWidth(1)
    sev_clr = severity_color(sev)
    pdf.setFillColorRGB(sev_clr[0], sev_clr[1], sev_clr[2])
    pdf.roundRect(left_margin, y - box_h + 6, content_width, box_h, 4, stroke=1, fill=1)
    pdf.setFont('Helvetica-Bold', 14)
    pdf.setFillColorRGB(*sev_clr)
    pdf.drawString(left_margin + 10, y - 12, sev.upper())
    pdf.setFont('Helvetica', 10)
    pdf.setFillColorRGB(*CLR_BODY)
    stats_text = f'GAGS Score: {gags}   |   Lesions: {lesion_count}   |   Symmetry Delta: {sym_delta}%'
    pdf.drawString(left_margin + 10, y - 26, stats_text)
    y -= box_h + 10

    # ================================================================
    # COMPACT PRESET — one-page summary
    # ================================================================
    if preset == 'compact':
        y = section(y, 'Consensus Summary')
        y = block(y, consensus.get('summary', 'N/A'))

        if dominant_regions:
            y = section(y, 'Top Burden Regions')
            for region, values in dominant_regions[:3]:
                y = bullet(y, f'{region}: {values.get("count", 0)} lesions, GAGS {values.get("gags_score", 0)}')

        if compare:
            y = section(y, 'Longitudinal Snapshot')
            y = block(y, comparison_summary(compare))
            y = bullet(y, f'Lesions {format_signed_delta(compare.get("lesion_delta", 0))} ({get_delta_status(float(compare.get("lesion_delta", 0)))})')
            y = bullet(y, f'GAGS {format_signed_delta(compare.get("gags_delta", 0))} ({get_delta_status(float(compare.get("gags_delta", 0)))})')

        if session.get('note'):
            y = section(y, 'Session Note')
            y = block(y, str(session['note']))

        draw_footer()
        pdf.save()
        return report_path

    # ================================================================
    # CLINICAL / PRESENTATION — full multi-page report
    # ================================================================

    # --- Diagnostic Image ---
    y = draw_diagnostic_image(y)

    # --- Consensus ---
    y = section(y, 'Consensus Summary')
    y = block(y, consensus.get('summary', 'N/A'))

    if session.get('note'):
        y = section(y, 'Session Note')
        y = block(y, str(session['note']))

    # --- Regional GAGS Table ---
    if regions:
        y = section(y, 'Regional GAGS Breakdown')
        headers = ['Region', 'Lesions', 'LPI', 'Area (px)', 'GAGS Score']
        rows = []
        for rname, rvals in regions.items():
            rows.append([
                rname.replace('_', ' ').title(),
                str(rvals.get('count', 0)),
                str(round(rvals.get('lpi', 0), 1)),
                str(rvals.get('area_px', 0)),
                str(rvals.get('gags_score', 0)),
            ])
        # Sort by GAGS score descending
        rows.sort(key=lambda r: int(r[4]), reverse=True)
        # Add total row
        total_lesions = sum(int(r[1]) for r in rows)
        total_gags = sum(int(r[4]) for r in rows)
        rows.append(['TOTAL', str(total_lesions), '', '', str(total_gags)])
        col_widths = [content_width * 0.28, content_width * 0.15, content_width * 0.17,
                      content_width * 0.22, content_width * 0.18]
        y = draw_table(y, headers, rows, col_widths)

    # --- Lesion Type Distribution ---
    if type_counts:
        y = section(y, 'Lesion Type Distribution')
        type_headers = ['Type', 'Count', 'Percentage']
        total_typed = sum(type_counts.values()) or 1
        type_rows = []
        for tname, tcount in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
            pct = f'{(tcount / total_typed) * 100:.1f}%'
            type_rows.append([tname.title(), str(tcount), pct])
        type_rows.append(['Total', str(sum(type_counts.values())), '100%'])
        type_col_widths = [content_width * 0.40, content_width * 0.30, content_width * 0.30]
        y = draw_table(y, type_headers, type_rows, type_col_widths)

    # --- Temporal Comparison ---
    if compare:
        y = section(y, 'Temporal Comparison')
        y = metric(y, 'Compared against', compare_target_label(compare))
        y = metric(y, 'Previous session', str(compare.get('previous_session_id', 'N/A')))
        y = block(y, comparison_summary(compare))
        sev_change = compare.get('severity_change', {})
        y = metric(y, 'Severity change', f'{sev_change.get("from", "?")} -> {sev_change.get("to", "?")}')
        ld = float(compare.get('lesion_delta', 0))
        gd = float(compare.get('gags_delta', 0))
        sd = float(compare.get('symmetry_delta_change', 0))
        y = metric(y, 'Lesion delta', f'{format_signed_delta(ld)} ({get_delta_status(ld)})')
        y = metric(y, 'GAGS delta', f'{format_signed_delta(gd)} ({get_delta_status(gd)})')
        y = metric(y, 'Symmetry delta change', f'{format_signed_delta(sd, "%")} ({get_delta_status(sd)})')
        if region_changes:
            y -= 4
            for region, values in region_changes:
                y = bullet(
                    y,
                    f'{region}: {get_delta_status(values.get("count_delta", 0))} '
                    f'({format_signed_delta(values.get("count_delta", 0))} lesions, '
                    f'LPI {format_signed_delta(values.get("lpi_delta", 0))})',
                )

    # --- Pipeline Metadata (clinical only) ---
    if preset == 'clinical' and (pipeline_metrics or timing_ms):
        y = section(y, 'Pipeline Metadata')
        y = metric(y, 'Model A', MODEL_A_ID)
        y = metric(y, 'Model B', MODEL_B_ID)
        if pipeline_metrics:
            y = metric(y, 'Raw detections', str(pipeline_metrics.get('raw_detections', 'N/A')))
            y = metric(y, 'Post-NMS', str(pipeline_metrics.get('post_nms', 'N/A')))
            y = metric(y, 'Post-gating', str(pipeline_metrics.get('post_gating', 'N/A')))
            tc = pipeline_metrics.get('type_coverage', {})
            if tc:
                y = metric(y, 'Type coverage', f'direct={tc.get("direct", 0)}, proximity={tc.get("proximity", 0)}, none={tc.get("none", 0)}')
        if timing_ms:
            total_ms = timing_ms.get('total', 0)
            y = metric(y, 'Segmentation time', f'{total_ms:.0f}ms')
        cloud_timing = results.get('cloud_timing', {})
        if cloud_timing:
            total_cloud = sum(v for v in cloud_timing.values() if isinstance(v, (int, float)))
            y = metric(y, 'Cloud inference time', f'{total_cloud:.0f}ms')

    # --- Presentation extras ---
    if preset == 'presentation':
        y = section(y, 'Clinical Headline', 14)
        y = block(
            y,
            f'This case is currently graded {sev} with a GAGS score of '
            f'{gags} across {lesion_count} verified lesions.',
            'Helvetica', 11,
        )
        y = section(y, 'Key Observations', 13)
        y = bullet(y, f'Consensus: {consensus.get("summary", "N/A")}')
        if dominant_regions:
            y = bullet(
                y,
                'Top burden regions: ' + ', '.join(
                    f'{r} (GAGS {v.get("gags_score", 0)})' for r, v in dominant_regions
                ),
            )
        if compare:
            y = section(y, 'Longitudinal Story', 13)
            y = block(y, comparison_summary(compare), 'Helvetica', 11)
            sev_change = compare.get('severity_change', {})
            y = bullet(y, f'Severity shifted from {sev_change.get("from", "?")} to {sev_change.get("to", "?")}')
            for region, values in region_changes[:2]:
                y = bullet(
                    y,
                    f'{region}: {get_delta_status(values.get("count_delta", 0))} with '
                    f'{format_signed_delta(values.get("count_delta", 0))} lesion change',
                )
        if session.get('note'):
            y = section(y, 'Presenter Note', 13)
            y = block(y, str(session['note']))

    # --- Disclaimer ---
    y = ensure_space(y, 50)
    y -= 10
    pdf.setStrokeColorRGB(*CLR_RULE)
    pdf.setLineWidth(0.5)
    pdf.line(left_margin, y + 4, left_margin + content_width, y + 4)
    y = block(
        y,
        'DISCLAIMER: This report is generated by an automated AI analysis pipeline and is intended '
        'for informational purposes only. It does not constitute medical advice. Consult a qualified '
        'dermatologist for clinical decisions.',
        'Helvetica', 8, CLR_MUTED,
    )

    draw_footer()
    pdf.save()
    return report_path


async def build_app_state() -> Dict[str, Any]:
    ensure_dirs()
    store = BridgeStore(DB_PATH)
    startup_cleanup = store.cleanup_expired()
    return {
        'store': store,
        'pipeline': None,
        'cloud_engine': None,
        'startup_cleanup': startup_cleanup,
    }


async def _periodic_cleanup(app: FastAPI) -> None:
    while True:
        try:
            await asyncio.sleep(300)
            app.state.resources['store'].cleanup_expired()
        except asyncio.CancelledError:
            break
        except Exception:
            logger.exception('Periodic cleanup failed')


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.resources = await build_app_state()
    cleanup_task = asyncio.create_task(_periodic_cleanup(app))
    try:
        yield
    finally:
        cleanup_task.cancel()
        app.state.resources['store'].close()


app = FastAPI(title='Acne V7 API Bridge', version=APP_VERSION, lifespan=lifespan)

limiter = Limiter(key_func=get_remote_address, default_limits=['60/minute'])
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv('CORS_ALLOWED_ORIGINS', 'http://localhost:3000,http://localhost:5173').split(',')
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=['GET', 'POST', 'DELETE', 'OPTIONS'],
    allow_headers=['Content-Type', 'Accept'],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)


@app.middleware('http')
async def add_security_headers(request: Request, call_next) -> Response:
    response = await call_next(request)
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
    response.headers['Content-Security-Policy'] = "default-src 'none'; frame-ancestors 'none'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'"
    response.headers['Permissions-Policy'] = 'camera=(), microphone=(), geolocation=()'
    response.headers['Cache-Control'] = 'no-store'
    # Add HSTS only for non-localhost origins (production deployments)
    host = request.headers.get('host', '')
    if host and not host.startswith('localhost') and not host.startswith('127.0.0.1'):
        response.headers['Strict-Transport-Security'] = 'max-age=63072000; includeSubDomains'
    return response


def get_store() -> BridgeStore:
    return app.state.resources['store']


@app.get('/', response_model=RootResponse)
async def root() -> Dict[str, Any]:
    return {
        'app': 'Acne V7 API Bridge',
        'status': 'ok',
        'version': APP_VERSION,
        'docs': '/docs',
        'health': '/health',
    }


def get_pipeline() -> Any:
    pipeline = app.state.resources['pipeline']
    if pipeline is None:
        with model_init_lock:
            pipeline = app.state.resources['pipeline']
            if pipeline is None:
                ensure_runtime_imports()
                runtime_pipeline = FaceSegmentationPipeline
                logger.info('Initializing FaceSegmentationPipeline...')
                pipeline = runtime_pipeline(
                    bisenet_weights='weights/79999_iter.pth',
                    smooth_edges=True,
                )
                app.state.resources['pipeline'] = pipeline
                logger.info('FaceSegmentationPipeline ready.')
    return pipeline


def get_cloud_engine() -> Any:
    cloud_engine = app.state.resources['cloud_engine']
    if cloud_engine is None:
        with model_init_lock:
            cloud_engine = app.state.resources['cloud_engine']
            if cloud_engine is None:
                ensure_runtime_imports()
                runtime_cloud_engine = CloudInferenceEngine
                logger.info('Initializing CloudInferenceEngine...')
                cloud_engine = runtime_cloud_engine(api_key=API_KEY)
                app.state.resources['cloud_engine'] = cloud_engine
                logger.info('CloudInferenceEngine ready.')
    return cloud_engine


@app.get('/health', response_model=HealthResponse)
async def health() -> Dict[str, Any]:
    return {
        'status': 'ok',
        'version': APP_VERSION,
        'roboflow_api_key_configured': True,
        'pipeline_initialized': app.state.resources['pipeline'] is not None,
        'cloud_engine_initialized': app.state.resources['cloud_engine'] is not None,
    }


@app.get('/version', response_model=VersionResponse)
async def version() -> Dict[str, Any]:
    return {
        'app': 'Acne V7 API Bridge',
        'version': APP_VERSION,
        'model_a_id': MODEL_A_ID,
        'model_b_id': MODEL_B_ID,
        'max_upload_bytes': MAX_UPLOAD_BYTES,
    }


@app.get('/privacy', response_model=PrivacyResponse)
async def privacy() -> Dict[str, Any]:
    return {
        'privacy_mode_supported': True,
        'default_retention_hours': DEFAULT_RETENTION_HOURS,
        'max_retention_hours': MAX_RETENTION_HOURS,
        'purge_endpoint': '/privacy/purge/{session_id}',
        'stored_fields': [
            'session_id',
            'timestamp',
            'severity',
            'gags_score',
            'lesion_count',
            'symmetry_delta',
            'results_json',
            'diagnostic_image_path',
            'original_image_path',
            'privacy_mode',
            'retention_hours',
        ],
    }


@app.delete('/privacy/purge/{session_id}', response_model=PurgeResponse)
async def purge_session(session_id: str) -> Dict[str, Any]:
    session_id = validate_session_id(session_id)
    if not get_store().purge(session_id):
        raise HTTPException(status_code=404, detail='Session not found')
    return {'purged': True, 'session_id': session_id}


@app.post('/analysis/start', response_model=AnalysisStartResponse)
@limiter.limit('10/minute')
async def analysis_start(request: Request, body: AnalysisStartRequest) -> Dict[str, Any]:
    store = get_store()
    session_id = validate_session_id(body.session_id or uuid.uuid4().hex)
    if store.get_session_row(session_id):
        raise HTTPException(status_code=409, detail='Session already exists')
    stub = session_stub(session_id, body.privacy_mode, body.retention_hours)
    stub['profile_id'] = validate_profile_id(body.profile_id) if body.profile_id else 'default-profile'
    store.upsert_session(stub)
    status = store.set_status(session_id, 'queued', 'Analysis session created', 0)
    return {
        'session_id': session_id,
        'profile_id': stub['profile_id'],
        'privacy_mode': body.privacy_mode,
        'retention_hours': normalize_retention(body.retention_hours),
        'status': status,
    }


@app.get('/status/stream/{session_id}')
async def session_status_stream(session_id: str) -> StreamingResponse:
    session_id = validate_session_id(session_id)

    async def event_generator():
        previous = None
        iterations = 0
        while iterations < 500:
            iterations += 1
            current = get_store().get_status(session_id)
            if current and current != previous:
                previous = current
                yield f"data: {json.dumps(current)}\n\n"
                if current.get('completed') or current.get('failed'):
                    break
            else:
                yield ": keepalive\n\n"
            await asyncio.sleep(0.6)

    return StreamingResponse(event_generator(), media_type='text/event-stream')


@app.get('/status/latest', response_model=StatusLatestResponse)
async def status_latest() -> Dict[str, Any]:
    status = get_store().latest_status()
    return {'status': status or {'stage': 'idle', 'detail': 'No analyses yet', 'progress': 0}}


@app.get('/status/{session_id}', response_model=StatusPayload)
async def session_status(session_id: str) -> Dict[str, Any]:
    status = get_store().get_status(validate_session_id(session_id))
    if not status:
        raise HTTPException(status_code=404, detail='Status not found')
    return status


@app.get('/history', response_model=HistoryResponse)
async def history(limit: int = 25, profile_id: Optional[str] = None, cursor: Optional[str] = None) -> Dict[str, Any]:
    if profile_id:
        validate_profile_id(profile_id)
    if cursor:
        validate_cursor(cursor)
    items, next_cursor = get_store().history(limit, profile_id=profile_id, cursor=cursor)
    result: Dict[str, Any] = {'items': items}
    if next_cursor:
        result['next_cursor'] = next_cursor
    return result


@app.get('/profiles', response_model=ProfilesResponse)
async def profiles() -> Dict[str, Any]:
    store = get_store()
    with store.lock:
        rows = store.conn.execute(
            '''
            SELECT
                profile_id,
                COUNT(*) AS sessions,
                MAX(timestamp) AS latest_timestamp,
                (SELECT severity FROM sessions s2 WHERE s2.profile_id = s1.profile_id ORDER BY s2.timestamp DESC LIMIT 1) AS latest_severity
            FROM sessions s1
            GROUP BY profile_id
            '''
        ).fetchall()
    items = [
        {
            'profile_id': row['profile_id'] or 'default-profile',
            'sessions': row['sessions'],
            'latest_timestamp': row['latest_timestamp'],
            'latest_severity': row['latest_severity'],
        }
        for row in rows
    ]
    return {'items': items}


@app.get('/session/{session_id}', response_model=SessionDetailResponse)
async def session_detail(session_id: str) -> Dict[str, Any]:
    return require_session(session_id)


@app.get('/session/{session_id}/image/{kind}')
async def session_image(session_id: str, kind: Literal['diagnostic', 'original']) -> FileResponse:
    """Serve a session image as a file response instead of a data URI."""
    session_id = validate_session_id(session_id)
    row = get_store().get_session_row(session_id)
    if not row:
        raise HTTPException(status_code=404, detail='Session not found')
    path_key = f'{kind}_image_path'
    raw_path = row[path_key]
    if not raw_path:
        raise HTTPException(status_code=404, detail=f'No {kind} image for this session')
    file_path = Path(raw_path)
    if not file_path.is_absolute():
        file_path = (BASE_DIR / file_path).resolve()
    managed_roots = (UPLOAD_DIR.resolve(), OUTPUT_DIR.resolve(), REPORT_DIR.resolve())
    if not any(file_path.is_relative_to(r) for r in managed_roots):
        raise HTTPException(status_code=403, detail='Access denied')
    if not file_path.exists():
        raise HTTPException(status_code=404, detail='Image file not found on disk')
    return FileResponse(file_path, media_type='image/jpeg')


@app.post('/session/{session_id}/notes', response_model=NotesResponse)
async def update_session_notes(session_id: str, request: NotesRequest) -> Dict[str, Any]:
    session_id = validate_session_id(session_id)
    session = require_session(session_id)
    store = get_store()
    with store.lock:
        store.conn.execute(
            'UPDATE sessions SET note = ? WHERE session_id = ?',
            (request.note, session_id),
        )
        store.conn.commit()
    return {'session_id': session_id, 'note': request.note}


@app.get('/compare/{current_session_id}', response_model=CompareResponse)
async def compare(current_session_id: str, previous_session_id: Optional[str] = None) -> Dict[str, Any]:
    current = require_session(current_session_id)
    if previous_session_id and previous_session_id == current['session_id']:
        previous = None
    else:
        previous = require_session(previous_session_id) if previous_session_id else get_store().previous_session(current['session_id'])
    return {
        'current_session_id': current['session_id'],
        'compare': annotate_compare_payload(compare_payload(previous, current), bool(previous_session_id)),
    }


@app.get('/report/{session_id}', response_model=ReportResponse)
@limiter.limit('15/minute')
async def report(request: Request, session_id: str, previous_session_id: Optional[str] = None) -> Dict[str, Any]:
    session = require_session(session_id)
    if not session.get('results'):
        raise HTTPException(status_code=409, detail='Analysis not completed for this session')
    if previous_session_id and previous_session_id == session['session_id']:
        previous = None
    else:
        previous = require_session(previous_session_id) if previous_session_id else get_store().previous_session(session['session_id'])
    compare_data = annotate_compare_payload(compare_payload(previous, session), bool(previous_session_id))
    loop = asyncio.get_event_loop()
    pdf_path = await loop.run_in_executor(None, write_pdf_report, session, compare_data, 'clinical')
    return {
        'session_id': session['session_id'],
        'report': {
            'clinical_analysis': session['results'].get('clinical_analysis', {}),
            'consensus_summary': session['results'].get('consensus_summary', {}),
            'compare': compare_data,
            'pdf_path': str(pdf_path),
            'pdf_data_uri': file_to_data_uri(str(pdf_path), 'application/pdf'),
        },
    }


@app.post('/export/{session_id}', response_model=ExportResponse)
@limiter.limit('10/minute')
async def export(request: Request, session_id: str, body: ExportRequest) -> Dict[str, Any]:
    session = require_session(session_id)
    if not session.get('results'):
        raise HTTPException(status_code=409, detail='Analysis not completed for this session')
    if body.previous_session_id and body.previous_session_id == session['session_id']:
        previous = None
    else:
        previous = require_session(body.previous_session_id) if body.previous_session_id else get_store().previous_session(session['session_id'])
    compare_data = annotate_compare_payload(compare_payload(previous, session), bool(body.previous_session_id))
    loop = asyncio.get_event_loop()
    pdf_path = await loop.run_in_executor(None, write_pdf_report, session, compare_data, body.preset)
    payload: Dict[str, Any] = {
        'session_id': session['session_id'],
        'pdf_path': str(pdf_path),
        'preset': body.preset,
    }
    if body.include_pdf_data:
        payload['pdf_data_uri'] = file_to_data_uri(str(pdf_path), 'application/pdf')
    return payload


@app.get('/metrics')
@limiter.limit('30/minute')
async def metrics(request: Request) -> Dict[str, Any]:
    """Aggregated system metrics: API usage, pipeline timing, detection stats.

    Returns:
        Dict with api_usage, session_stats, and timing_averages.
    """
    store = get_store()

    # --- API usage from usage_tracker ---
    api_usage = get_usage_summary()

    # --- Session statistics from sessions.db ---
    sessions = store.list_sessions(limit=1000)
    session_count = len(sessions)
    timing_samples = []
    cloud_timing_samples = []
    pipeline_metrics_samples = []
    detection_counts = []
    gags_scores = []

    for sess in sessions:
        results = sess.get('results')
        if not results:
            continue
        # Local pipeline timing
        timing_ms = results.get('timing_ms', {})
        if timing_ms:
            timing_samples.append(timing_ms)
        # Cloud timing (Phase 5 instrumented)
        cloud_timing = results.get('cloud_timing', {})
        if cloud_timing:
            cloud_timing_samples.append(cloud_timing)
        # Pipeline metrics (Phase 5 instrumented)
        pm = results.get('pipeline_metrics', {})
        if pm:
            pipeline_metrics_samples.append(pm)
        # Clinical analysis
        ca = results.get('clinical_analysis', {})
        total_lesions = ca.get('total_lesions')
        if total_lesions is not None:
            detection_counts.append(total_lesions)
        gags = ca.get('gags_score')
        if gags is not None:
            gags_scores.append(gags)

    # Compute timing averages
    def _avg_timing(samples: list, keys: list) -> Dict[str, Optional[float]]:
        result = {}
        for k in keys:
            vals = [s[k] for s in samples if k in s and s[k] is not None]
            result[f'{k}_mean'] = round(sum(vals) / len(vals), 1) if vals else None
        return result

    local_timing_avg = _avg_timing(timing_samples, [
        'bisenet', 'landmarks', 'geometry', 'combine', 'total',
    ])
    cloud_timing_avg = _avg_timing(cloud_timing_samples, [
        'model_a_1280_ms', 'model_b_ms', 'total_wall_ms',
    ])

    # Aggregate pipeline metrics
    agg_pipeline = None
    if pipeline_metrics_samples:
        total_raw = sum(pm.get('raw_detections', 0) for pm in pipeline_metrics_samples)
        total_nms = sum(pm.get('post_nms', 0) for pm in pipeline_metrics_samples)
        total_gated = sum(pm.get('post_gating', 0) for pm in pipeline_metrics_samples)
        total_prox = sum(pm.get('proximity_propagated', 0) for pm in pipeline_metrics_samples)
        agg_coverage = {}
        for pm in pipeline_metrics_samples:
            for k, v in pm.get('type_coverage', {}).items():
                agg_coverage[k] = agg_coverage.get(k, 0) + v
        agg_pipeline = {
            'sample_count': len(pipeline_metrics_samples),
            'total_raw_detections': total_raw,
            'total_post_nms': total_nms,
            'total_post_gating': total_gated,
            'total_proximity_propagated': total_prox,
            'nms_reduction_pct': round((1 - total_nms / total_raw) * 100, 1) if total_raw > 0 else None,
            'gating_reduction_pct': round((1 - total_gated / total_nms) * 100, 1) if total_nms > 0 else None,
            'type_coverage_aggregate': agg_coverage,
        }

    return {
        'api_usage': api_usage,
        'session_stats': {
            'total_sessions': session_count,
            'sessions_with_results': len(timing_samples),
            'detection_counts': {
                'mean': round(sum(detection_counts) / len(detection_counts), 1) if detection_counts else None,
                'min': min(detection_counts) if detection_counts else None,
                'max': max(detection_counts) if detection_counts else None,
            },
            'gags_scores': {
                'mean': round(sum(gags_scores) / len(gags_scores), 1) if gags_scores else None,
                'min': min(gags_scores) if gags_scores else None,
                'max': max(gags_scores) if gags_scores else None,
            },
        },
        'timing': {
            'local_pipeline': local_timing_avg,
            'cloud_inference': cloud_timing_avg,
            'sample_count': len(timing_samples),
        },
        'pipeline_metrics': agg_pipeline,
    }


@app.post('/analyze', response_model=AnalyzeResultResponse)
@limiter.limit('10/minute')
async def analyze(
    request: Request,
    file: UploadFile = File(...),
    session_id: Optional[str] = Form(None),
    profile_id: Optional[str] = Form(None),
    privacy_mode: bool = Form(False),
    retention_hours: int = Form(DEFAULT_RETENTION_HOURS),
) -> Dict[str, Any]:
    store = get_store()
    # Chunked read with early size cutoff to prevent DoS
    chunks: list[bytes] = []
    total_read = 0
    while True:
        chunk = await file.read(64 * 1024)  # 64KB chunks
        if not chunk:
            break
        total_read += len(chunk)
        if total_read > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f'File exceeds {MAX_UPLOAD_BYTES} byte limit',
            )
        chunks.append(chunk)
    payload = b''.join(chunks)
    validate_upload(file, payload)
    image = decode_image(payload)
    h, w = image.shape[:2]
    if h > 10000 or w > 10000:
        raise HTTPException(status_code=400, detail='Image dimensions exceed 10000px limit')

    session_id = validate_session_id(session_id or uuid.uuid4().hex)
    retention_hours = normalize_retention(retention_hours)
    existing = store.get_session_row(session_id)
    if existing:
        privacy_mode = bool(existing['privacy_mode'])
        retention_hours = int(existing['retention_hours'])
        profile_id = existing['profile_id'] if 'profile_id' in existing.keys() else profile_id
    else:
        stub = session_stub(session_id, privacy_mode, retention_hours)
        stub['profile_id'] = validate_profile_id(profile_id) if profile_id else 'default-profile'
        store.upsert_session(stub)
        profile_id = stub['profile_id']

    original_path = (UPLOAD_DIR / f'{session_id}_original.jpg').resolve()
    diagnostic_path = (OUTPUT_DIR / f'{session_id}_diagnostic.jpg').resolve()
    original_jpeg = image_to_jpeg_bytes(image)
    save_image(original_path, image)

    def _run_analysis_sync() -> Dict[str, Any]:
        """Run the blocking ML pipeline off the async event loop."""
        store.set_status(session_id, 'received', 'Upload accepted', 5)
        store.set_status(session_id, 'segmenting', 'Running face segmentation', 20)
        segmentation = get_pipeline().segment(image, return_intermediates=True)

        store.set_status(session_id, 'cloud_inference', 'Fetching ensemble detections', 45)
        cloud_results = get_cloud_engine().fetch_multi_scale_consensus(
            image,
            MODEL_A_ID,
            MODEL_B_ID,
        )

        store.set_status(session_id, 'mapping', 'Assigning lesions to regions', 70)
        img_height, img_width = image.shape[:2]
        mapper = EnsembleLesionMapper(segmentation['masks'])
        assignments = mapper.ensemble_map_multi_scale(
            cloud_results['preds_a_640'],
            cloud_results['preds_a_1280'],
            cloud_results['preds_b'],
            (img_height, img_width),
            image=image,
        )
        clinical_report = mapper.get_clinical_report(assignments)

        store.set_status(session_id, 'rendering', 'Rendering diagnostic overlay', 82)
        ensure_runtime_imports()
        
        # Filter assignments to ensure only list-of-dicts are passed to visualization
        clean_assignments = {
            k: v for k, v in assignments.items() 
            if isinstance(v, list) and not k.startswith('_')
        }
        
        # Optionally refine lesion types using EfficientViT classifier.
        # This is non-blocking: failures fall back to original detector labels.
        try:
            from face_segmentation.efficientvit_classifier import classify_lesion_crop
            refined_assignments = {}
            for region, dets in clean_assignments.items():
                refined_list = []
                for d in dets:
                    # keep original detector class
                    detected_class = d.get('class_name')
                    refined = None
                    # Attempt to crop the lesion region and classify
                    try:
                        x1, y1, x2, y2 = d.get('bbox', [0, 0, 0, 0])
                        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
                        crop = image[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
                        if crop.size != 0:
                            out = classify_lesion_crop(crop)
                            if out:
                                refined_label, conf = out
                                # Map detector cyst/nodule -> nodulocystic when classifier returns nodulocystic
                                refined = {
                                    'refined_class': refined_label,
                                    'refined_confidence': conf,
                                }
                    except Exception:
                        refined = None
                    entry = dict(d)
                    if refined is not None:
                        entry['refined'] = refined
                    refined_list.append(entry)
                refined_assignments[region] = refined_list
            # replace clean_assignments for visualization with the refined version
            viz_assignments = refined_assignments
        except Exception:
            viz_assignments = clean_assignments

        overlay = draw_lesion_boxes(
            image,
            lesions=viz_assignments,
            clinical_report=clinical_report,
        )
        diagnostic_jpeg = image_to_jpeg_bytes(overlay)
        save_image(diagnostic_path, overlay)

        timestamp = utcnow_iso()
        analysis_results = dict(segmentation.get('metadata', {}))
        analysis_results['timestamp'] = timestamp
        analysis_results['session_id'] = session_id
        analysis_results['privacy_mode'] = privacy_mode
        analysis_results['retention_hours'] = retention_hours
        analysis_results['clinical_analysis'] = clinical_report
        analysis_results['lesions'] = assignments
        analysis_results['consensus_summary'] = consensus_summary(assignments)
        analysis_results['cloud_results'] = cloud_results
        analysis_results['source_stream_provenance'] = summarize_stream_provenance(cloud_results)

        # Phase 5 instrumentation: capture cloud timing and pipeline metrics
        cloud_timing = cloud_results.get('_timing')
        if cloud_timing:
            analysis_results['cloud_timing'] = cloud_timing
        cloud_file_sizes = cloud_results.get('_file_sizes')
        if cloud_file_sizes:
            analysis_results['cloud_file_sizes'] = cloud_file_sizes
        pipeline_metrics = assignments.get('_pipeline_metrics')
        if pipeline_metrics:
            analysis_results['pipeline_metrics'] = pipeline_metrics

        managed_original = absolute_managed_path(original_path)
        managed_diagnostic = absolute_managed_path(diagnostic_path)
        final_original: Optional[str] = managed_original
        final_diagnostic: Optional[str] = managed_diagnostic
        if privacy_mode:
            safe_unlink(managed_original)
            safe_unlink(managed_diagnostic)
            final_original = None
            final_diagnostic = None

        session_record = {
            'session_id': session_id,
            'profile_id': profile_id or 'default-profile',
            'timestamp': timestamp,
            'severity': clinical_report.get('clinical_severity', 'Unknown'),
            'gags_score': int(clinical_report.get('gags_total_score', 0)),
            'lesion_count': int(clinical_report.get('total_lesions', 0)),
            'symmetry_delta': float(clinical_report.get('symmetry_delta', 0.0)),
            'results_json': json.dumps(analysis_results),
            'diagnostic_image_path': final_diagnostic,
            'original_image_path': final_original,
            'privacy_mode': privacy_mode,
            'retention_hours': retention_hours,
        }
        store.set_status(session_id, 'saving', 'Persisting analysis outputs', 92)
        store.upsert_session(session_record)

        return {
            'diagnostic_jpeg': diagnostic_jpeg,
        }

    try:
        loop = asyncio.get_event_loop()
        sync_result = await loop.run_in_executor(None, _run_analysis_sync)

        session_data = store.session_payload(session_id)
        if not session_data:
            raise HTTPException(status_code=500, detail='Failed to reload saved session')
        previous = store.previous_session(session_id)
        compare_data = compare_payload(previous, session_data)
        final_status = store.set_status(
            session_id,
            'completed',
            'Analysis complete',
            100,
            {'completed': True},
        )

        return {
            'session_id': session_id,
            'status': final_status,
            'severity': session_data['severity'],
            'gags_score': session_data['gags_score'],
            'lesion_count': session_data['lesion_count'],
            'symmetry_delta': session_data['symmetry_delta'],
            'results': session_data['results'],
            'compare': compare_data,
            'original_image': '' if privacy_mode else bytes_to_data_uri(original_jpeg, 'image/jpeg'),
            'diagnostic_image': '' if privacy_mode else bytes_to_data_uri(sync_result['diagnostic_jpeg'], 'image/jpeg'),
        }
    except HTTPException:
        store.set_status(session_id, 'failed', 'Analysis failed', 100, {'failed': True})
        raise
    except Exception as exc:
        logger.error('Analysis failed for session %s: %s', session_id, exc, exc_info=True)
        store.set_status(session_id, 'failed', 'Internal analysis error', 100, {'failed': True})
        raise HTTPException(status_code=500, detail='Analysis failed') from exc
