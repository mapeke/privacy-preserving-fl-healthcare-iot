"""MIT-BIH Arrhythmia Database loader with synthetic-ECG fallback.

The MIT-BIH database (PhysioNet) ships ECG records as triples ``(.dat, .hea, .atr)`` sampled at
360 Hz with two leads. We use the ``wfdb`` Python package to download and parse them.

If PhysioNet is unreachable (offline dev box, CI without network, etc.), a deterministic synthetic
ECG generator is used so the rest of the pipeline can still run end-to-end. This is flagged
clearly in returned metadata so experiments don't silently report results on synthetic data.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


# AAMI 5-class mapping from MIT-BIH annotation symbols.
# Reference: AAMI EC57 standard (also widely used in the literature).
_AAMI_MAP: dict[str, int] = {
    # N — Normal beat
    "N": 0, "L": 0, "R": 0, "e": 0, "j": 0,
    # S — Supraventricular ectopic
    "A": 1, "a": 1, "J": 1, "S": 1,
    # V — Ventricular ectopic
    "V": 2, "E": 2,
    # F — Fusion
    "F": 3,
    # Q — Unknown / paced / other
    "/": 4, "f": 4, "Q": 4,
}

DEFAULT_RECORDS = (100, 101, 103, 105, 106, 108, 109, 111, 112, 113, 114, 115)


@dataclass
class EcgDataset:
    """A flat dataset of ECG beat windows.

    Attributes:
        x: ``(N, 1, window_size)`` float32 tensor of normalized beats.
        y: ``(N,)`` int64 vector of AAMI class indices in 0..4.
        record_ids: parallel ``(N,)`` array recording which MIT-BIH record each beat came from
            (used to simulate per-patient IoT clients).
        is_synthetic: True iff produced by the fallback generator.
    """

    x: np.ndarray
    y: np.ndarray
    record_ids: np.ndarray
    is_synthetic: bool

    def __len__(self) -> int:
        return int(self.y.shape[0])


def load_mitbih(
    records: tuple[int, ...] | list[int] = DEFAULT_RECORDS,
    cache_dir: str | Path = "data/cache",
    window_size: int = 360,
    *,
    allow_synthetic_fallback: bool = True,
) -> EcgDataset:
    """Load MIT-BIH beats with windowed segmentation and AAMI labels.

    Args:
        records: PhysioNet record numbers (e.g., 100, 101, ...).
        cache_dir: Local cache where ``wfdb`` stores downloaded ``.dat/.hea/.atr`` files.
        window_size: Samples per beat window (typically 360 = 1 s @ 360 Hz).
        allow_synthetic_fallback: If True and PhysioNet is unreachable, generate synthetic data.

    Returns:
        :class:`EcgDataset`.
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    try:
        return _load_real_mitbih(records, cache_path, window_size)
    except Exception as exc:  # pragma: no cover - exercised only when offline
        if not allow_synthetic_fallback:
            raise
        logger.warning(
            "MIT-BIH download/parse failed (%s). Falling back to synthetic ECG generator.",
            exc,
        )
        return generate_synthetic_ecg(
            n_samples_per_record=600,
            records=records,
            window_size=window_size,
        )


def _load_real_mitbih(
    records: tuple[int, ...] | list[int],
    cache_dir: Path,
    window_size: int,
) -> EcgDataset:
    """Real MIT-BIH path. Imported lazily so the synthetic fallback works without ``wfdb``."""
    import wfdb  # type: ignore[import-untyped]

    half = window_size // 2
    xs: list[np.ndarray] = []
    ys: list[int] = []
    rec_ids: list[int] = []

    for rec in records:
        rec_str = f"{rec:03d}"
        local_path = cache_dir / f"{rec_str}.dat"
        if not local_path.exists():
            logger.info("Downloading MIT-BIH record %s -> %s", rec_str, cache_dir)
            wfdb.dl_database("mitdb", str(cache_dir), records=[rec_str], annotators=["atr"])
        logger.info("Loading MIT-BIH record %s", rec_str)
        record = wfdb.rdrecord(str(cache_dir / rec_str))
        ann = wfdb.rdann(str(cache_dir / rec_str), "atr")

        signal = record.p_signal[:, 0].astype(np.float32)  # lead II
        # Per-record z-score normalization so all clients see comparable amplitudes.
        signal = (signal - signal.mean()) / (signal.std() + 1e-8)

        for sample_idx, symbol in zip(ann.sample, ann.symbol):
            if symbol not in _AAMI_MAP:
                continue
            start = sample_idx - half
            end = sample_idx + (window_size - half)
            if start < 0 or end > signal.size:
                continue
            xs.append(signal[start:end])
            ys.append(_AAMI_MAP[symbol])
            rec_ids.append(rec)

    if not xs:
        raise RuntimeError("No usable beats found in the requested records.")

    x = np.stack(xs, axis=0)[:, None, :].astype(np.float32)
    y = np.asarray(ys, dtype=np.int64)
    record_ids = np.asarray(rec_ids, dtype=np.int64)
    return EcgDataset(x=x, y=y, record_ids=record_ids, is_synthetic=False)


def generate_synthetic_ecg(
    n_samples_per_record: int = 500,
    records: tuple[int, ...] | list[int] = DEFAULT_RECORDS,
    window_size: int = 360,
    *,
    seed: int = 0,
) -> EcgDataset:
    """Generate a deterministic synthetic ECG dataset for offline development.

    Each "record" produces beats from class 0..4 with a realistic skewed distribution
    (~85% normal, ~6% S, ~6% V, ~1.5% F, ~1.5% Q). Each class has a distinct waveform shape so a
    well-trained model should reach >90% accuracy — useful for sanity-checking the FL pipeline.
    """
    rng = np.random.default_rng(seed)
    class_probs = np.array([0.85, 0.06, 0.06, 0.015, 0.015])

    t = np.linspace(0.0, 1.0, window_size, dtype=np.float32)

    def beat(cls: int, jitter: float) -> np.ndarray:
        # Different class shapes: peak position, width, and harmonic content all differ.
        peak = 0.4 + 0.05 * cls + 0.02 * jitter
        width = 0.05 + 0.01 * cls
        spike = np.exp(-((t - peak) ** 2) / (2.0 * width**2))
        wave = (
            0.2 * np.sin(2 * np.pi * (1 + 0.2 * cls) * t)
            + 0.6 * spike
            + 0.05 * np.sin(2 * np.pi * 5 * t)
        )
        wave = wave + 0.05 * rng.standard_normal(window_size).astype(np.float32) * jitter
        return wave.astype(np.float32)

    xs: list[np.ndarray] = []
    ys: list[int] = []
    rec_ids: list[int] = []
    for rec in records:
        for _ in range(n_samples_per_record):
            cls = int(rng.choice(len(class_probs), p=class_probs))
            xs.append(beat(cls, jitter=float(rng.uniform(0.5, 1.5))))
            ys.append(cls)
            rec_ids.append(rec)

    x = np.stack(xs, axis=0)[:, None, :].astype(np.float32)
    y = np.asarray(ys, dtype=np.int64)
    record_ids = np.asarray(rec_ids, dtype=np.int64)
    return EcgDataset(x=x, y=y, record_ids=record_ids, is_synthetic=True)
