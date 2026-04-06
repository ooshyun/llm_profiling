import shutil
import subprocess
from pathlib import Path
from typing import Union


class RemoteCollector:
    """Collect profiling artefacts from local paths, SSH hosts, or ADB devices."""

    def __init__(self, output_dir: Union[Path, str]) -> None:
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Collection methods
    # ------------------------------------------------------------------

    def collect_local(self, src_dir: Path, device_name: str) -> Path:
        """Copy files from a local directory into output_dir/device_name/.

        Returns the destination directory path.
        """
        dest = self._device_dir(device_name)
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(src_dir, dest)
        return dest

    def collect_ssh(self, host: str, remote_path: str, device_name: str) -> Path:
        """Copy files from a remote host via scp -r into output_dir/device_name/.

        Returns the destination directory path.
        """
        dest = self._device_dir(device_name)
        dest.mkdir(parents=True, exist_ok=True)

        subprocess.run(
            ["scp", "-r", f"{host}:{remote_path}", str(dest)],
            check=True,
        )
        return dest

    def collect_adb(self, serial: str, remote_path: str, device_name: str) -> Path:
        """Pull files from an Android device via adb pull into output_dir/device_name/.

        Returns the destination directory path.
        """
        dest = self._device_dir(device_name)
        dest.mkdir(parents=True, exist_ok=True)

        subprocess.run(
            ["adb", "-s", serial, "pull", remote_path, str(dest)],
            check=True,
        )
        return dest

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _device_dir(self, device_name: str) -> Path:
        return self._output_dir / device_name
