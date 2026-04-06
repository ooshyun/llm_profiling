from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Union

from schemas.profile_schema import ProfileRecord


class LocalStore:
    """Append-only JSONL file store for ProfileRecord objects."""

    def __init__(
        self,
        output_dir: Union[Path, str],
        filename: Optional[str] = None,
    ) -> None:
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

        if filename is None:
            ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%S")
            filename = f"profile_{ts}.jsonl"

        self._filepath = self._output_dir / filename

    @property
    def filepath(self) -> Path:
        return self._filepath

    def write(self, record: ProfileRecord) -> None:
        """Append one record as a JSON line."""
        with self._filepath.open("a", encoding="utf-8") as fh:
            fh.write(record.model_dump_json())
            fh.write("\n")

    def read_all(self) -> List[ProfileRecord]:
        """Read all records from the JSONL file."""
        if not self._filepath.exists():
            return []

        records: List[ProfileRecord] = []
        with self._filepath.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(ProfileRecord.model_validate_json(line))
        return records
