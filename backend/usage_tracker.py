"""Roboflow API usage tracker backed by SQLite.

Replaces the previous JSON file-based tracker for thread safety
and reduced I/O overhead on each call.

Phase 5 additions:
- ``latency_ms`` and ``error`` columns for richer diagnostics
- ``log_api_call`` now accepts optional latency and error message
- ``get_usage_summary()`` returns aggregated stats for the /metrics endpoint
"""
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional

_DB_PATH = Path(__file__).resolve().parent / 'usage.db'
_lock = threading.Lock()
_conn: sqlite3.Connection | None = None


def _get_conn() -> sqlite3.Connection:
    """Return a module-level SQLite connection, creating the table on first use."""
    global _conn
    if _conn is None:
        with _lock:
            if _conn is None:
                _conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
                _conn.execute('PRAGMA journal_mode=WAL')
                _conn.execute('''
                    CREATE TABLE IF NOT EXISTS api_calls (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        model TEXT NOT NULL,
                        status TEXT NOT NULL DEFAULT 'success',
                        latency_ms REAL,
                        error TEXT
                    )
                ''')
                # Migrate existing tables that lack the new columns.
                # SQLite silently ignores duplicate ALTER TABLE ADD COLUMN.
                for col, typ in [('latency_ms', 'REAL'), ('error', 'TEXT')]:
                    try:
                        _conn.execute(f'ALTER TABLE api_calls ADD COLUMN {col} {typ}')
                    except sqlite3.OperationalError:
                        pass  # column already exists
                _conn.commit()
    return _conn


def log_api_call(
    model_id: str,
    status: str = 'success',
    latency_ms: Optional[float] = None,
    error: Optional[str] = None,
) -> None:
    """Log a Roboflow API call with optional latency and error detail.

    Args:
        model_id: The Roboflow model identifier (e.g. 'runner-e0dmy/acne-ijcab/2').
        status: 'success' or 'error'.
        latency_ms: Wall-clock time for the API call in milliseconds.
        error: Error message if the call failed.
    """
    conn = _get_conn()
    with _lock:
        conn.execute(
            'INSERT INTO api_calls (timestamp, model, status, latency_ms, error) '
            'VALUES (?, ?, ?, ?, ?)',
            (datetime.now(timezone.utc).isoformat(), model_id, status, latency_ms, error),
        )
        conn.commit()


def get_usage_stats() -> int:
    """Returns total API calls recorded."""
    conn = _get_conn()
    with _lock:
        row = conn.execute('SELECT COUNT(*) FROM api_calls').fetchone()
        return row[0] if row else 0


def get_usage_summary() -> Dict[str, Any]:
    """Return aggregated usage statistics for the /metrics endpoint.

    Returns:
        Dict with total_calls, calls_by_model, calls_by_status,
        latency_stats (mean, min, max, p50, p95), error_rate,
        and recent_errors list.
    """
    conn = _get_conn()
    with _lock:
        total = conn.execute('SELECT COUNT(*) FROM api_calls').fetchone()[0]

        # Breakdown by model
        model_rows = conn.execute(
            'SELECT model, COUNT(*) FROM api_calls GROUP BY model'
        ).fetchall()
        calls_by_model = {r[0]: r[1] for r in model_rows}

        # Breakdown by status
        status_rows = conn.execute(
            'SELECT status, COUNT(*) FROM api_calls GROUP BY status'
        ).fetchall()
        calls_by_status = {r[0]: r[1] for r in status_rows}

        # Latency stats (only for rows that have latency_ms)
        latency_rows = conn.execute(
            'SELECT latency_ms FROM api_calls WHERE latency_ms IS NOT NULL '
            'ORDER BY latency_ms'
        ).fetchall()

        latency_stats = None
        if latency_rows:
            vals = [r[0] for r in latency_rows]
            n = len(vals)
            latency_stats = {
                'count': n,
                'mean_ms': round(sum(vals) / n, 1),
                'min_ms': round(vals[0], 1),
                'max_ms': round(vals[-1], 1),
                'p50_ms': round(vals[n // 2], 1),
                'p95_ms': round(vals[int(n * 0.95)], 1) if n >= 2 else round(vals[-1], 1),
            }

        # Error rate
        error_count = calls_by_status.get('error', 0)
        error_rate = round(error_count / total, 4) if total > 0 else 0.0

        # Recent errors (last 10)
        error_rows = conn.execute(
            'SELECT timestamp, model, error FROM api_calls '
            'WHERE status = ? ORDER BY id DESC LIMIT 10',
            ('error',),
        ).fetchall()
        recent_errors = [
            {'timestamp': r[0], 'model': r[1], 'error': r[2]}
            for r in error_rows
        ]

    return {
        'total_calls': total,
        'calls_by_model': calls_by_model,
        'calls_by_status': calls_by_status,
        'latency_stats': latency_stats,
        'error_rate': error_rate,
        'recent_errors': recent_errors,
    }
