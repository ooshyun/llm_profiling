from typing import List

import pandas as pd

from schemas.profile_schema import ProfileRecord


def _records_to_df(records: List[ProfileRecord]) -> pd.DataFrame:
    return pd.DataFrame([r.model_dump() for r in records])


class DeviceComparator:
    """Compare a metric across devices, broken down by block and sublayer."""

    def compare(self, records: List[ProfileRecord], metric: str) -> pd.DataFrame:
        """Group by [device, block_idx, sublayer] and compute mean of metric.

        Returns a DataFrame with columns: device, block_idx, sublayer, <metric>.
        """
        df = _records_to_df(records)
        result = (
            df.groupby(["device", "block_idx", "sublayer"])[metric]
            .mean()
            .reset_index()
        )
        return result


class ArchitectureComparator:
    """Compare a metric across model architectures."""

    def compare(self, records: List[ProfileRecord], metric: str) -> pd.DataFrame:
        """Group by [architecture, block_idx, sublayer] and compute mean of metric.

        Returns a DataFrame with columns: architecture, block_idx, sublayer, <metric>.
        """
        df = _records_to_df(records)
        result = (
            df.groupby(["architecture", "block_idx", "sublayer"])[metric]
            .mean()
            .reset_index()
        )
        return result


class ScalingComparator:
    """Compare how a metric scales with model size or another grouping dimension."""

    def compare(
        self,
        records: List[ProfileRecord],
        metric: str,
        group_by: str = "model",
    ) -> pd.DataFrame:
        """Group by [group_by, block_idx, sublayer] and compute mean of metric.

        Parameters
        ----------
        records:
            List of ProfileRecord objects.
        metric:
            Column name of the metric to aggregate.
        group_by:
            Primary grouping key (default ``"model"``).

        Returns a DataFrame with columns: <group_by>, block_idx, sublayer, <metric>.
        """
        df = _records_to_df(records)
        result = (
            df.groupby([group_by, "block_idx", "sublayer"])[metric]
            .mean()
            .reset_index()
        )
        return result


class FrameworkComparator:
    """Compare a metric across inference frameworks."""

    def compare(self, records: List[ProfileRecord], metric: str) -> pd.DataFrame:
        """Group by [framework, block_idx, sublayer] and compute mean of metric.

        Returns a DataFrame with columns: framework, block_idx, sublayer, <metric>.
        """
        df = _records_to_df(records)
        result = (
            df.groupby(["framework", "block_idx", "sublayer"])[metric]
            .mean()
            .reset_index()
        )
        return result
