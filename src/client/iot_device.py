"""IoT device simulator: per-device compute and bandwidth profiles.

Each ``IoTDevice`` is a lightweight bundle of capabilities that the rest of the framework
respects when scheduling work. The simulation does not actually rate-limit CPU; instead, the
profile is recorded in metrics and used to scale local epochs or batch sizes if desired.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class IoTDeviceProfile:
    """Static capabilities of a simulated IoT device."""

    device_id: int
    label: str                    # e.g., "wearable_smartwatch", "bedside_monitor"
    cpu_factor: float = 1.0       # 1.0 = baseline; <1 means slower than baseline
    bandwidth_kbps: float = 256.0 # uplink budget per round
    memory_mb: int = 64           # working RAM budget
    is_battery_powered: bool = True


def default_profiles(num_clients: int) -> list[IoTDeviceProfile]:
    """Generate a heterogeneous mix of device profiles for ``num_clients`` devices.

    Mix: 50% wearables (slow CPU, 128 kbps), 30% bedside monitors (fast CPU, 1024 kbps),
    20% gateway hubs (very fast, 4096 kbps). The exact counts are clamped to integers and the
    remainder lands in the wearable bucket so small ``num_clients`` (e.g., 4) is sensible.
    """
    if num_clients <= 0:
        raise ValueError("num_clients must be positive")

    n_bedside = max(0, int(round(num_clients * 0.3)))
    n_gateway = max(0, int(round(num_clients * 0.2)))
    n_wearable = num_clients - n_bedside - n_gateway

    profiles: list[IoTDeviceProfile] = []
    cid = 0
    for _ in range(n_wearable):
        profiles.append(IoTDeviceProfile(
            device_id=cid, label="wearable_smartwatch",
            cpu_factor=0.6, bandwidth_kbps=128.0, memory_mb=32, is_battery_powered=True,
        ))
        cid += 1
    for _ in range(n_bedside):
        profiles.append(IoTDeviceProfile(
            device_id=cid, label="bedside_monitor",
            cpu_factor=1.5, bandwidth_kbps=1024.0, memory_mb=128, is_battery_powered=False,
        ))
        cid += 1
    for _ in range(n_gateway):
        profiles.append(IoTDeviceProfile(
            device_id=cid, label="gateway_hub",
            cpu_factor=2.5, bandwidth_kbps=4096.0, memory_mb=512, is_battery_powered=False,
        ))
        cid += 1
    return profiles
