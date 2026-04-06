from pathlib import Path
from typing import List, Union

import pandas as pd

from schemas.profile_schema import ProfileRecord


class ParquetStore:
    def __init__(self, filepath: Union[Path, str]) -> None:
        self._filepath = Path(filepath)

    @property
    def filepath(self) -> Path:
        return self._filepath

    def _records_to_df(self, records: List[ProfileRecord]) -> pd.DataFrame:
        return pd.DataFrame([r.model_dump() for r in records])

    def write(self, records: List[ProfileRecord]) -> None:
        df = self._records_to_df(records)
        df.to_parquet(self._filepath, index=False)

    def append(self, records: List[ProfileRecord]) -> None:
        if self._filepath.exists():
            existing = pd.read_parquet(self._filepath)
            new_df = self._records_to_df(records)
            combined = pd.concat([existing, new_df], ignore_index=True)
            combined.to_parquet(self._filepath, index=False)
        else:
            self.write(records)

    def read(self) -> pd.DataFrame:
        return pd.read_parquet(self._filepath)
