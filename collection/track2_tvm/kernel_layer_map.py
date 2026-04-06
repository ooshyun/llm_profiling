"""
Maps TVM kernel names to transformer layer locations (block_idx, sublayer).

Priority order for lookup:
1. Exact name match
2. Regex pattern match (in insertion order)
3. Default unknown fallback
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class KernelLayerInfo:
    block_idx: int   # -1 for non-block kernels
    sublayer: str    # canonical sublayer name


_DEFAULT_UNKNOWN = KernelLayerInfo(block_idx=-1, sublayer="unknown")


class KernelLayerMapper:
    """Maps TVM kernel names to (block_idx, sublayer) via exact or regex lookup."""

    def __init__(self) -> None:
        # Exact name → KernelLayerInfo
        self._exact: Dict[str, KernelLayerInfo] = {}
        # List of (compiled_pattern, raw_pattern, sublayer, block_idx) in insertion order
        self._patterns: List[Tuple[re.Pattern, str, str, int]] = []

    # ------------------------------------------------------------------
    # Mutation helpers
    # ------------------------------------------------------------------

    def add_mapping(self, kernel_name: str, block_idx: int, sublayer: str) -> None:
        """Register an exact-match mapping."""
        self._exact[kernel_name] = KernelLayerInfo(block_idx=block_idx, sublayer=sublayer)

    def add_pattern(
        self,
        pattern: str,
        sublayer: str,
        block_idx: int = -1,
    ) -> None:
        """Register a regex pattern match.

        If the pattern contains a named group ``blk``, the integer value of
        that capture is used as *block_idx* and the *block_idx* argument is
        ignored.
        """
        compiled = re.compile(pattern)
        self._patterns.append((compiled, pattern, sublayer, block_idx))

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def lookup(self, kernel_name: str) -> KernelLayerInfo:
        """Return KernelLayerInfo for *kernel_name*.

        Falls back to ``KernelLayerInfo(block_idx=-1, sublayer='unknown')``
        when no rule matches.
        """
        # 1. Exact match
        if kernel_name in self._exact:
            return self._exact[kernel_name]

        # 2. Regex patterns (first match wins)
        for compiled, _raw, sublayer, block_idx in self._patterns:
            m = compiled.search(kernel_name)
            if m:
                # If the pattern has a named group "blk", extract block index
                try:
                    blk_str = m.group("blk")
                    block_idx = int(blk_str)
                except IndexError:
                    pass
                return KernelLayerInfo(block_idx=block_idx, sublayer=sublayer)

        # 3. Unknown fallback
        return _DEFAULT_UNKNOWN

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Persist the mapper to a JSON file at *path*."""
        data = {
            "exact": {
                name: asdict(info)
                for name, info in self._exact.items()
            },
            "patterns": [
                {"pattern": raw, "sublayer": sublayer, "block_idx": block_idx}
                for _compiled, raw, sublayer, block_idx in self._patterns
            ],
        }
        Path(path).write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str) -> "KernelLayerMapper":
        """Reconstruct a KernelLayerMapper from a JSON file."""
        data = json.loads(Path(path).read_text())
        mapper = cls()
        for name, info in data.get("exact", {}).items():
            mapper.add_mapping(name, info["block_idx"], info["sublayer"])
        for entry in data.get("patterns", []):
            mapper.add_pattern(
                entry["pattern"],
                entry["sublayer"],
                entry.get("block_idx", -1),
            )
        return mapper
