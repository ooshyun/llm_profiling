import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Union

from schemas.profile_schema import ProfileRecord

_TEXT_COLS = [
    "experiment_id",
    "timestamp",
    "device",
    "framework",
    "model",
    "architecture",
    "phase",
    "sublayer",
    "op_type",
    "quantization",
]

_INT_COLS = [
    "mem_bytes",
    "mem_peak_bytes",
    "flops",
    "bandwidth_bytes_sec",
    "input_tokens",
    "output_token_idx",
    "batch_size",
    "block_idx",
]

_REAL_COLS = [
    "latency_us",
    "power_mw",
    "cache_l1_hit_ratio",
    "cache_l2_hit_ratio",
    "thermal_c",
    "gpu_sm_util",
    "gpu_tensor_util",
]

_ALL_COLS = _TEXT_COLS + _INT_COLS + _REAL_COLS

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS profile_records (
    experiment_id        TEXT,
    timestamp            TEXT,
    device               TEXT,
    framework            TEXT,
    model                TEXT,
    architecture         TEXT,
    phase                TEXT,
    sublayer             TEXT,
    op_type              TEXT,
    quantization         TEXT,
    mem_bytes            INTEGER,
    mem_peak_bytes       INTEGER,
    flops                INTEGER,
    bandwidth_bytes_sec  INTEGER,
    input_tokens         INTEGER,
    output_token_idx     INTEGER,
    batch_size           INTEGER,
    block_idx            INTEGER,
    latency_us           REAL,
    power_mw             REAL,
    cache_l1_hit_ratio   REAL,
    cache_l2_hit_ratio   REAL,
    thermal_c            REAL,
    gpu_sm_util          REAL,
    gpu_tensor_util      REAL
)
"""

_CREATE_IDX_1 = """
CREATE INDEX IF NOT EXISTS idx_exp_device_model_block
ON profile_records (experiment_id, device, model, block_idx)
"""

_CREATE_IDX_2 = """
CREATE INDEX IF NOT EXISTS idx_arch_block_sublayer
ON profile_records (architecture, block_idx, sublayer)
"""

_INSERT_SQL = """
INSERT INTO profile_records ({cols}) VALUES ({placeholders})
""".format(
    cols=", ".join(_ALL_COLS),
    placeholders=", ".join("?" for _ in _ALL_COLS),
)


def _record_to_row(record: ProfileRecord) -> tuple:
    ts = record.timestamp.isoformat() if isinstance(record.timestamp, datetime) else record.timestamp
    return (
        record.experiment_id,
        ts,
        record.device,
        record.framework,
        record.model,
        record.architecture,
        record.phase,
        record.sublayer,
        record.op_type,
        record.quantization,
        record.mem_bytes,
        record.mem_peak_bytes,
        record.flops,
        record.bandwidth_bytes_sec,
        record.input_tokens,
        record.output_token_idx,
        record.batch_size,
        record.block_idx,
        record.latency_us,
        record.power_mw,
        record.cache_l1_hit_ratio,
        record.cache_l2_hit_ratio,
        record.thermal_c,
        record.gpu_sm_util,
        record.gpu_tensor_util,
    )


def _row_to_record(row: sqlite3.Row) -> ProfileRecord:
    d = dict(row)
    d["timestamp"] = datetime.fromisoformat(d["timestamp"])
    return ProfileRecord(**d)


class ProfileDB:
    def __init__(self, db_path: Union[Path, str]) -> None:
        self._conn = sqlite3.connect(str(db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(_CREATE_TABLE)
        self._conn.execute(_CREATE_IDX_1)
        self._conn.execute(_CREATE_IDX_2)
        self._conn.commit()

    def insert(self, record: ProfileRecord) -> None:
        self._conn.execute(_INSERT_SQL, _record_to_row(record))
        self._conn.commit()

    def insert_batch(self, records: List[ProfileRecord]) -> None:
        self._conn.executemany(_INSERT_SQL, [_record_to_row(r) for r in records])
        self._conn.commit()

    def query(
        self,
        experiment_id: str,
        device: Optional[str] = None,
        model: Optional[str] = None,
    ) -> List[ProfileRecord]:
        sql = "SELECT * FROM profile_records WHERE experiment_id = ?"
        params: list = [experiment_id]
        if device is not None:
            sql += " AND device = ?"
            params.append(device)
        if model is not None:
            sql += " AND model = ?"
            params.append(model)
        sql += " ORDER BY block_idx"
        cursor = self._conn.execute(sql, params)
        return [_row_to_record(row) for row in cursor.fetchall()]

    def count(self, experiment_id: str) -> int:
        cursor = self._conn.execute(
            "SELECT COUNT(*) FROM profile_records WHERE experiment_id = ?",
            (experiment_id,),
        )
        return cursor.fetchone()[0]

    def close(self) -> None:
        self._conn.close()
