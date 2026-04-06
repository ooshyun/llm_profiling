"""Smoke tests for the dashboard package — imports and config sanity checks."""

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for CI


class TestDashboardImports:
    def test_presentation_package_importable(self) -> None:
        import presentation  # noqa: F401

    def test_plots_module_importable(self) -> None:
        from presentation import plots  # noqa: F401

    def test_dashboard_package_importable(self) -> None:
        import presentation.dashboard  # noqa: F401


class TestDashboardConfig:
    def test_config_importable(self) -> None:
        from presentation.dashboard import config  # noqa: F401

    def test_default_db_path_is_path(self) -> None:
        from pathlib import Path
        from presentation.dashboard.config import DEFAULT_DB_PATH

        assert isinstance(DEFAULT_DB_PATH, Path)

    def test_default_parquet_dir_is_path(self) -> None:
        from pathlib import Path
        from presentation.dashboard.config import DEFAULT_PARQUET_DIR

        assert isinstance(DEFAULT_PARQUET_DIR, Path)

    def test_refresh_interval_is_positive_int(self) -> None:
        from presentation.dashboard.config import REFRESH_INTERVAL_SEC

        assert isinstance(REFRESH_INTERVAL_SEC, int)
        assert REFRESH_INTERVAL_SEC > 0

    def test_default_db_path_value(self) -> None:
        from presentation.dashboard.config import DEFAULT_DB_PATH

        assert DEFAULT_DB_PATH.name == "profiles.db"

    def test_default_parquet_dir_value(self) -> None:
        from presentation.dashboard.config import DEFAULT_PARQUET_DIR

        assert DEFAULT_PARQUET_DIR.name == "parquet"
