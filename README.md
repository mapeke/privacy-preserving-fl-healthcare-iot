# Privacy-Preserving Federated Learning Framework for IoT-Based Healthcare Systems

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Flower](https://img.shields.io/badge/FL-Flower-orange)](https://flower.ai/)
[![PyTorch](https://img.shields.io/badge/DL-PyTorch-EE4C2C)](https://pytorch.org/)
[![Opacus](https://img.shields.io/badge/DP-Opacus-7E57C2)](https://opacus.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> Undergraduate diploma project — a research-grade framework that trains an ECG arrhythmia classifier
> across simulated wearable healthcare IoT devices **without** centralizing raw patient signals,
> while layering three complementary privacy defenses on top of vanilla federated averaging.

---

## Motivation

Connected health devices — smartwatches, Holter monitors, bedside telemetry — produce streams of
physiological data that are simultaneously **clinically valuable** and **highly sensitive**. Existing
deployments centralize this data, which:

1. Creates a single point of failure for breaches.
2. Exposes patients to re-identification attacks even after pseudonymization.
3. Conflicts with HIPAA / GDPR data-minimization principles.

This project demonstrates a complete pipeline in which raw ECG signals **never leave the device**, and
even the model updates that *are* shared carry strong, layered privacy guarantees.

## Privacy stack

| Layer                      | Technique                                  | Threat addressed                                  |
|----------------------------|---------------------------------------------|--------------------------------------------------|
| Local training             | DP-SGD (Opacus) with RDP accountant         | Membership inference; gradient leakage           |
| Aggregation                | Pairwise-mask Secure Aggregation            | Honest-but-curious server; per-client traceability|
| Communication              | Top-k sparsification + 8-bit quantization   | Bandwidth budget for IoT; reconstruction attacks |

## Use case

ECG arrhythmia classification on the **MIT-BIH Arrhythmia Database** (PhysioNet). Each simulated IoT
device holds a non-IID partition of beats, performs local DP-SGD training, masks its update with the
secure-aggregation protocol, compresses it, and ships it to the server. The server unmasks, decompresses,
aggregates, and broadcasts the new global model.

A synthetic-ECG fallback ships with the repo so the entire pipeline runs offline if PhysioNet is
unreachable.

## Quick start

```bash
# 1. Install
pip install -r requirements.txt

# 2. Run unit tests
pytest tests/

# 3. Centralized baseline
python experiments/run_centralized.py --epochs 5

# 4. Vanilla federated learning (no privacy)
python experiments/run_federated.py --num-clients 4 --rounds 10

# 5. Federated + differential privacy
python experiments/run_federated_dp.py --num-clients 4 --rounds 10 --target-epsilon 5.0

# 6. Full pipeline (DP + SecAgg + compression)
python experiments/run_full_pipeline.py --num-clients 4 --rounds 10
```

## Repository layout

```
.
├── configs/        # YAML experiment configs
├── docs/           # Architecture, threat model, privacy analysis
├── src/
│   ├── data/       # MIT-BIH loader, preprocessing, IID/non-IID partitioning
│   ├── models/     # 1D CNN for ECG classification
│   ├── client/     # Flower NumPyClient + DP trainer + IoT device simulator
│   ├── server/     # Flower server, custom strategy with privacy hooks
│   ├── privacy/    # DP, secure aggregation, compression primitives
│   └── utils/      # Config, logging, metrics, seeding
├── experiments/    # Runnable experiment entry points
├── notebooks/      # Exploratory and analysis notebooks
├── tests/          # pytest unit tests
└── scripts/        # Shell helpers for server/client launch
```

## Status

This repository is built incrementally as part of the diploma writeup. Each commit corresponds
to a single conceptual unit so the git history can be cited chapter-by-chapter in the thesis.

## License

MIT — see [LICENSE](LICENSE).
