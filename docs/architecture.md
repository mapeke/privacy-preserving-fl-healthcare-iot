# System Architecture

This document describes the system-level architecture of the privacy-preserving federated learning
framework for IoT-based healthcare. It covers the actors, the data flow per round, and the layout of
the Python packages that implement them.

## Actors

```
┌───────────────────────────┐         ┌──────────────────────────────┐
│   IoT Healthcare Device   │   ...   │   Aggregation Server         │
│  (simulated client × N)   │ ◄────► │ (orchestrator + aggregator)  │
└───────────────────────────┘         └──────────────────────────────┘
        ▲                                          ▲
        │ holds raw ECG (private)                  │ holds global model state
        │ runs DP-SGD locally                      │ runs custom FedAvg strategy
        │ masks update with SecAgg                 │ unmasks via SecAgg
        │ compresses delta                         │ decompresses + aggregates
```

| Actor               | Responsibility                                                                  |
|---------------------|---------------------------------------------------------------------------------|
| **IoT client**      | Owns a partition of patient ECG beats; never shares raw data; trains locally.   |
| **Aggregation server** | Coordinates rounds; aggregates masked, compressed updates; never sees raw data. |
| **Adversary (modeled)** | Honest-but-curious server, a coalition of curious clients, or an eavesdropper.  |

## Per-round data flow

```mermaid
sequenceDiagram
    participant C as IoT Clients (i = 1..N)
    participant S as Server
    S->>C: broadcast global model w_t
    Note over C: local DP-SGD training<br/>compute Δw_i = w_i - w_t
    Note over C: compress Δw_i<br/>(top-k + 8-bit quantize)
    Note over C: mask: Δw̃_i = Δw_i + Σ mask_ij - Σ mask_ji
    C->>S: send (Δw̃_i, n_i)
    Note over S: aggregate Σ Δw̃_i = Σ Δw_i (masks cancel)
    Note over S: decompress; weighted average
    Note over S: w_{t+1} = w_t + (1/Σn_i) Σ n_i · Δw_i
```

The cryptographic property that makes this work: in the pairwise-mask scheme, every mask appears
once with `+` (at client `i`) and once with `-` (at client `j`), so summing all clients' masked
updates cancels every mask exactly. Any **strict subset** of updates remains a uniformly random
vector — i.e., the server learns nothing about an individual client's update.

## Package layout

```
src/
├── data/                       # everything that touches raw signals
│   ├── mitbih_loader.py        # downloads MIT-BIH from PhysioNet (or synthetic fallback)
│   ├── preprocessing.py        # bandpass filter, beat segmentation, AAMI 5-class labels
│   └── partitioner.py          # IID / Dirichlet non-IID splits across N IoT devices
│
├── models/                     # DL architectures
│   ├── ecg_cnn.py              # full 1D CNN (~50k params)
│   └── lightweight.py          # tiny variant (~5k params) for edge constraints
│
├── client/                     # client-side runtime
│   ├── flower_client.py        # Flower NumPyClient (fit / evaluate / get_parameters)
│   ├── dp_trainer.py           # local training loop wrapped by Opacus PrivacyEngine
│   └── iot_device.py           # simulates compute / bandwidth limits per device
│
├── privacy/                    # cryptographic + statistical defenses
│   ├── differential_privacy.py # noise calibration; DP accountant
│   ├── secure_agg.py           # PRG-based pairwise mask primitives
│   └── compression.py          # top-k sparsification + 8-bit quantization
│
├── server/                     # server-side runtime
│   ├── flower_server.py        # entry point; starts simulation
│   ├── strategies.py           # PrivacyAwareFedAvg with hooks for SecAgg + decompression
│   └── secure_aggregation.py   # server's view of the masking protocol
│
└── utils/                      # cross-cutting helpers
    ├── config.py               # YAML config loader
    ├── metrics.py              # accuracy, macro-F1, per-class P/R
    ├── logging_utils.py        # console + file handlers
    └── seed.py                 # reproducibility
```

## Round configuration knobs (see `configs/`)

- `num_clients`              — total IoT devices
- `clients_per_round`        — sampled fraction (Flower's `fraction_fit`)
- `local_epochs`             — local SGD passes per round
- `target_epsilon`, `target_delta` — DP budget
- `max_grad_norm`            — DP gradient clip threshold
- `secagg.enabled`           — turn pairwise masking on/off
- `compression.top_k_ratio`  — fraction of weights kept per layer
- `compression.quant_bits`   — quantization width (default 8)

## Where the substance lives

The most consequential files (those a thesis reader will want to inspect first) are:

- `src/server/strategies.py` — the custom `PrivacyAwareFedAvg` that orchestrates the privacy stack.
- `src/privacy/differential_privacy.py` — DP accounting and `PrivacyEngine` attachment.
- `src/privacy/secure_agg.py` — the pairwise-mask correctness argument in code.
- `src/client/flower_client.py` — the client side of every round.
