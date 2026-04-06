"""Experiment configuration and runner for tiny-llm-profiler."""
from __future__ import annotations

import itertools
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


# ---------------------------------------------------------------------------
# ExperimentConfig
# ---------------------------------------------------------------------------


@dataclass
class ExperimentConfig:
    """Parsed representation of an experiment YAML file."""

    # Core experiment knobs
    warmup_tokens: int = 32
    prompt_tokens: int = 128
    generate_tokens: int = 64
    repetitions: int = 5
    cooldown_sec: int = 30

    # Profiling options
    platform_monitor_interval_ms: int = 100
    overhead_calibration: bool = True

    # Target devices and models (optional; populated by the runner)
    devices: List[str] = field(default_factory=list)
    models: List[str] = field(default_factory=list)

    # Raw extra fields preserved for forward-compatibility
    extra: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: Path) -> "ExperimentConfig":
        """Load an ExperimentConfig from a YAML file.

        The YAML is expected to have at least an ``experiment`` key that maps
        to a dict of experiment parameters.  An optional ``profiling`` key is
        also consumed.  Any remaining top-level keys are stored in ``extra``.

        Example YAML::

            experiment:
              warmup_tokens: 32
              prompt_tokens: 128
              generate_tokens: 64
              repetitions: 5
              cooldown_sec: 30
            profiling:
              platform_monitor_interval_ms: 100
              overhead_calibration: true
        """
        with open(path, "r") as fh:
            raw: Dict[str, Any] = yaml.safe_load(fh) or {}

        exp_section: Dict[str, Any] = raw.pop("experiment", {})
        profiling_section: Dict[str, Any] = raw.pop("profiling", {})

        return cls(
            warmup_tokens=exp_section.get("warmup_tokens", 32),
            prompt_tokens=exp_section.get("prompt_tokens", 128),
            generate_tokens=exp_section.get("generate_tokens", 64),
            repetitions=exp_section.get("repetitions", 5),
            cooldown_sec=exp_section.get("cooldown_sec", 30),
            platform_monitor_interval_ms=profiling_section.get(
                "platform_monitor_interval_ms", 100
            ),
            overhead_calibration=profiling_section.get("overhead_calibration", True),
            devices=exp_section.get("devices", []),
            models=exp_section.get("models", []),
            extra=raw,
        )


# ---------------------------------------------------------------------------
# ExperimentRunner
# ---------------------------------------------------------------------------


class ExperimentRunner:
    """Generates and optionally executes an experiment matrix.

    Parameters
    ----------
    config:
        Parsed experiment configuration.
    devices:
        List of device identifiers to include in the matrix.
        If *None*, ``config.devices`` is used.
    models:
        List of model identifiers to include in the matrix.
        If *None*, ``config.models`` is used.
    """

    def __init__(
        self,
        config: ExperimentConfig,
        devices: Optional[List[str]] = None,
        models: Optional[List[str]] = None,
    ) -> None:
        self._config = config
        self._devices = devices if devices is not None else config.devices
        self._models = models if models is not None else config.models

    # ------------------------------------------------------------------
    # Matrix generation
    # ------------------------------------------------------------------

    def _generate_matrix(
        self,
        devices: List[str],
        models: List[str],
        repetitions: int,
    ) -> List[Dict[str, Any]]:
        """Return the full Cartesian product of devices × models × repetitions.

        Each entry in the returned list is a dict with keys:
        ``run_id``, ``device``, ``model``, ``repetition``.

        Parameters
        ----------
        devices:
            Device identifiers.
        models:
            Model identifiers.
        repetitions:
            Number of repetitions per (device, model) pair.
        """
        matrix: List[Dict[str, Any]] = []
        for device, model, rep in itertools.product(
            devices, models, range(1, repetitions + 1)
        ):
            matrix.append(
                {
                    "run_id": str(uuid.uuid4()),
                    "device": device,
                    "model": model,
                    "repetition": rep,
                }
            )
        return matrix

    def build_matrix(self) -> List[Dict[str, Any]]:
        """Return the experiment matrix using config-level devices/models."""
        return self._generate_matrix(
            self._devices,
            self._models,
            self._config.repetitions,
        )
