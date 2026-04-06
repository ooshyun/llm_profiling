import pytest

from collection.platform_monitor.base import PlatformMonitor
from collection.platform_monitor.jetson import JetsonMonitor
from collection.platform_monitor.rpi import RpiMonitor
from schemas.profile_schema import PlatformSnapshot


def test_base_is_abstract():
    with pytest.raises(TypeError):
        PlatformMonitor()  # type: ignore[abstract]


def test_rpi_parse_thermal():
    monitor = RpiMonitor.__new__(RpiMonitor)
    assert monitor._parse_thermal("52300\n") == pytest.approx(52.3)
    assert monitor._parse_thermal("0\n") == pytest.approx(0.0)
    assert monitor._parse_thermal("75000\n") == pytest.approx(75.0)


def test_rpi_parse_meminfo():
    meminfo_text = (
        "MemTotal:        8000000 kB\n"
        "MemFree:         2000000 kB\n"
        "MemAvailable:    3000000 kB\n"
        "Buffers:          500000 kB\n"
        "Cached:          1000000 kB\n"
    )
    monitor = RpiMonitor.__new__(RpiMonitor)
    used_bytes, available_bytes = monitor._parse_meminfo(meminfo_text)

    assert available_bytes == 3000000 * 1024
    assert used_bytes == (8000000 - 3000000) * 1024


def test_rpi_parse_cpufreq():
    monitor = RpiMonitor.__new__(RpiMonitor)
    assert monitor._parse_cpufreq("1800000\n") == pytest.approx(1800.0)
    assert monitor._parse_cpufreq("600000\n") == pytest.approx(600.0)
    assert monitor._parse_cpufreq("1500000\n") == pytest.approx(1500.0)


def test_rpi_sample_returns_snapshot(tmp_path):
    # Create fake thermal file
    thermal_file = tmp_path / "thermal_zone0_temp"
    thermal_file.write_text("52300\n")

    # Create fake meminfo file
    meminfo_file = tmp_path / "meminfo"
    meminfo_file.write_text(
        "MemTotal:        8000000 kB\n"
        "MemFree:         2000000 kB\n"
        "MemAvailable:    3000000 kB\n"
    )

    # Create fake cpufreq files for 4 CPUs
    cpu_freqs_khz = [1800000, 1600000, 1400000, 1200000]
    cpufreq_dir = tmp_path / "cpufreq"
    cpufreq_dir.mkdir()
    for i, freq in enumerate(cpu_freqs_khz):
        cpu_file = cpufreq_dir / f"cpu{i}_scaling_cur_freq"
        cpu_file.write_text(f"{freq}\n")

    cpufreq_pattern = str(cpufreq_dir / "cpu{}_scaling_cur_freq")

    monitor = RpiMonitor(
        thermal_path=str(thermal_file),
        meminfo_path=str(meminfo_file),
        cpufreq_pattern=cpufreq_pattern,
        n_cpus=4,
    )

    snapshot = monitor.sample()

    assert isinstance(snapshot, PlatformSnapshot)
    assert snapshot.cpu_temp_c == pytest.approx(52.3)
    assert snapshot.gpu_temp_c is None
    assert snapshot.soc_temp_c is None
    assert snapshot.power_mw == pytest.approx(0.0)
    assert snapshot.gpu_freq_mhz is None
    assert snapshot.cpu_freq_mhz == pytest.approx([1800.0, 1600.0, 1400.0, 1200.0])
    assert snapshot.mem_available_bytes == 3000000 * 1024
    assert snapshot.mem_used_bytes == (8000000 - 3000000) * 1024


def test_jetson_parse_tegrastats_line():
    line = (
        "04-05-2026 18:00:00 "
        "RAM 2200/62835MB (lfb 14520x4MB) "
        "CPU [25%@2201,30%@2201,20%@2201,22%@2201,0%@0,0%@0,0%@0,0%@0] "
        "GR3D_FREQ 50% "
        "VDD_GPU_SOC 4500mW "
        "VDD_CPU_CV 3200mW "
        "SOC 45000C "
        "GPU 42000C "
        "tj 48000C"
    )
    monitor = JetsonMonitor.__new__(JetsonMonitor)
    snap = monitor._parse_tegrastats(line)
    assert snap.mem_used_bytes == 2200 * 1024 * 1024
    assert snap.mem_available_bytes == (62835 - 2200) * 1024 * 1024
    assert snap.power_mw == 4500.0 + 3200.0
    assert len(snap.cpu_freq_mhz) == 8


def test_jetson_parse_tegrastats_gpu_temp():
    line = (
        "04-05-2026 18:00:00 "
        "RAM 2200/62835MB (lfb 14520x4MB) "
        "CPU [25%@2201,30%@2201,0%@0,0%@0,0%@0,0%@0,0%@0,0%@0] "
        "GR3D_FREQ 80% "
        "VDD_GPU_SOC 5000mW "
        "VDD_CPU_CV 3000mW "
        "SOC 50000C "
        "GPU 55000C "
        "tj 58000C"
    )
    monitor = JetsonMonitor.__new__(JetsonMonitor)
    snap = monitor._parse_tegrastats(line)
    assert snap.gpu_temp_c == 55.0
    assert snap.soc_temp_c == 50.0
    assert snap.cpu_temp_c == 58.0


def test_jetson_parse_real_tegrastats_format():
    """Test with actual JetPack 6.0 tegrastats output (new format: label@valueC)."""
    line = (
        "04-05-2026 22:54:16 RAM 7517/62841MB (lfb 15x4MB) "
        "SWAP 46/31421MB (cached 0MB) "
        "CPU [0%@729,3%@729,0%@729,0%@729,0%@729,0%@729,0%@729,0%@729,off,off,off,off] "
        "GR3D_FREQ 0% "
        "cpu@45.625C tboard@35.75C soc2@42.531C tdiode@36C "
        "soc0@42.843C tj@45.625C soc1@42.593C "
        "VDD_GPU_SOC 2388mW/2388mW VDD_CPU_CV 397mW/397mW "
        "VIN_SYS_5V0 3824mW/3824mW VDDQ_VDD2_1V8AO 402mW/402mW"
    )
    monitor = JetsonMonitor.__new__(JetsonMonitor)
    snap = monitor._parse_tegrastats(line)

    assert snap.mem_used_bytes == 7517 * 1024 * 1024
    assert snap.mem_available_bytes == (62841 - 7517) * 1024 * 1024
    assert len(snap.cpu_freq_mhz) == 8  # 8 active, 4 "off" skipped
    assert snap.cpu_freq_mhz[0] == 729.0
    assert snap.cpu_temp_c == 45.625  # tj@45.625C
    assert snap.soc_temp_c == 42.843  # soc0@42.843C
    assert snap.power_mw == 2388.0 + 397.0  # VDD_GPU_SOC + VDD_CPU_CV
    # VIN_SYS and VDDQ are not VDD_ prefixed, not counted
