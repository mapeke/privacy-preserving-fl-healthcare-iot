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

## Table of contents

- [Motivation](#motivation)
- [Privacy stack](#privacy-stack)
- [Use case](#use-case)
- [Quick start](#quick-start)
- [Repository layout](#repository-layout)
- [Configuration](#configuration)
- [Experiments and results](#experiments-and-results)
- [Tests](#tests)
- [Documentation](#documentation)
- [Citation](#citation)
- [License](#license)

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

See [`docs/threat-model.md`](docs/threat-model.md) for the full attacker model and
[`docs/privacy-analysis.md`](docs/privacy-analysis.md) for the formal arguments.

## Use case

ECG arrhythmia classification on the **MIT-BIH Arrhythmia Database** (PhysioNet) with the AAMI
5-class label scheme (N, S, V, F, Q). Each simulated IoT device holds a non-IID partition of beats,
performs local DP-SGD training, masks its update with the secure-aggregation protocol, compresses it,
and ships it to the server. The server unmasks, decompresses, aggregates, and broadcasts the new
global model.

A **synthetic-ECG fallback** ships with the repo so the entire pipeline runs offline if PhysioNet
is unreachable. Synthetic results are flagged in run metadata.

## Quick start

```bash
# 0. Clone
git clone https://github.com/mapeke/privacy-preserving-fl-healthcare-iot.git
cd privacy-preserving-fl-healthcare-iot

# 1. Install
pip install -r requirements.txt

# 2. Run unit tests (42 tests should pass)
pytest tests/

# 3. Centralized baseline (upper bound)
python experiments/run_centralized.py --epochs 5

# 4. Vanilla federated learning (no privacy)
python experiments/run_federated.py --num-clients 4 --rounds 10

# 5. Federated + differential privacy
python experiments/run_federated_dp.py --num-clients 4 --rounds 10 --target-epsilon 5.0

# 6. Federated + secure aggregation
python experiments/run_federated_secagg.py --num-clients 4 --rounds 10

# 7. Headline pipeline: DP + SecAgg + Compression
python experiments/run_full_pipeline.py --num-clients 4 --rounds 10

# 8. Or all of the above in one shot
./scripts/run_all_experiments.sh
```

Each run writes `results/<experiment_name>/{run.log, resolved_config.yaml, history.json}` so every
number in the diploma can be reproduced from a single config.

## Repository layout

```
.
├── configs/        # YAML experiment configs (default + 3 experiment overrides)
├── docs/           # Architecture, threat model, privacy analysis
├── src/
│   ├── data/       # MIT-BIH loader, preprocessing, IID/non-IID partitioning
│   ├── models/     # 1D CNN for ECG classification (full + lightweight)
│   ├── client/     # Flower NumPyClient + DP trainer + IoT device simulator
│   ├── server/     # Flower server, custom strategy with privacy hooks
│   ├── privacy/    # DP, secure aggregation, compression primitives
│   └── utils/      # Config, logging, metrics, seeding
├── experiments/    # Runnable experiment entry points
├── notebooks/      # Exploratory and analysis notebooks
├── tests/          # pytest unit tests (42 cases)
└── scripts/        # Shell helpers
```

## Configuration

All experiments are driven by YAML files under `configs/`:

| File                            | Purpose                                                      |
|---------------------------------|--------------------------------------------------------------|
| `default.yaml`                  | Baseline FedAvg — every other config inherits from this.     |
| `experiment_dp.yaml`            | Adds DP-SGD knobs (target ε, δ, gradient clip, accountant).  |
| `experiment_secagg.yaml`        | Enables the pairwise-mask SecAgg protocol.                   |
| `experiment_full.yaml`          | DP + SecAgg + top-k/8-bit compression.                       |

Any single value can be overridden on the CLI:

```bash
python experiments/run_full_pipeline.py \
    --target-epsilon 8.0 \
    --top-k-ratio 0.05 \
    --rounds 30
```

## Experiments and results

Five experiments exercise the framework:

1. **Centralized** — pool all clients' data and train one model. The accuracy upper bound.
2. **Federated (vanilla)** — FedAvg with no privacy defenses. Tests the federation alone.
3. **Federated + DP** — adds DP-SGD per client. Measures the DP-only utility cost.
4. **Federated + SecAgg** — adds pairwise-mask aggregation. Verifies mask cancellation in code.
5. **Full pipeline** — all defenses simultaneously. The headline result.

Results from the dev-machine run on 12 MIT-BIH records (records 100–115, 25 393 beats,
4 simulated IoT clients, Dirichlet non-IID partition, single-seed run on RTX-class GPU):

| Setup                                  | Test accuracy | Final ε  | Notes                                     |
|----------------------------------------|---------------|----------|-------------------------------------------|
| Centralized baseline (5 epochs)        | **98.78 %**   | n/a      | Macro-F1 = 0.36; rare classes (S/F/Q) ≈ 0 |
| Federated FedAvg (10 rounds)           | **98.80 %**   | n/a      | Matches centralized — federation is free  |
| Federated + SecAgg (10 rounds)         | **98.60 %**   | n/a      | Within noise of vanilla; masks cancel ✓   |
| Federated + DP (20 rounds, ε≈5)        | 97.11 %       | 4.9955   | Collapses to majority-class predictor     |
| Full pipeline DP+SecAgg+Compression    | 97.11 %       | 7.9964   | Same plateau; ε relaxed to 8 didn't help  |

The dataset is **heavily imbalanced** (97.1 % class N, 2.6 % V, 0.24 % S, 0.03 % F, 0.03 % Q),
so the 97.11 % accuracy DP runs reach is exactly the proportion of class N — i.e. the noise
swamps the rare-class signal and the model collapses to predicting N for every beat. This is a
real and reportable result for the diploma: it shows the **utility cost of DP-SGD on imbalanced
healthcare data** and motivates extensions like class-weighted loss, focal loss, or per-class DP
budgets.

Reproduce all five rows with `./scripts/run_all_experiments.sh`.

The `notebooks/03_privacy_utility_tradeoff.ipynb` notebook reads each `history.json` and
plots accuracy vs ε for the diploma figure.

### Follow-up: records 200–234

The thesis's third finding (DP collapses to majority-class) is *dataset-specific*. To test
this, the same five experiments were re-run on MIT-BIH records 200–234 — 25 records,
**61 818 beats**, with substantially more arrhythmia variety:

| Class | records 100–115 | records 200–234 |
|-------|-----------------|-----------------|
| N     | 97.09 %         | 82.08 %         |
| S     | 0.24 %          | 4.19 %          |
| V     | 2.60 %          | 9.53 %          |
| F     | 0.03 %          | 1.28 %          |
| Q     | 0.03 %          | 2.93 %          |

Same hyperparameters as the original table — no per-dataset tuning:

| Setup                                  | 100–115 acc | 200–234 acc | 200–234 ε | vs majority N (82.08 %) |
|----------------------------------------|-------------|-------------|-----------|-------------------------|
| Centralized baseline (5 epochs)        | 98.78 %     | **93.89 %** | n/a       | +11.81 pp; macro-F1 0.69 (was 0.36) |
| Federated FedAvg (10 rounds)           | 98.80 %     | **96.36 %** | n/a       | +14.28 pp; federation still free |
| Federated + SecAgg (10 rounds)         | 98.60 %     | **95.58 %** | n/a       | +13.50 pp; mask-cancellation noise ~0.78 pp |
| Federated + DP (20 rounds, ε≈5)        | 97.11 %     | **88.45 %** | 4.9964    | **+6.37 pp** — DP retains signal |
| Full pipeline DP+SecAgg+Compression    | 97.11 %     | **81.76 %** | 7.9935    | −0.32 pp — collapses to majority |

Two refinements to the thesis findings come out of this:

1. **DP-only does not always collapse to majority-class.** On records 100–115 the 97.11 %
   plateau equalled class-N prevalence exactly; on records 200–234 the 88.45 % accuracy is
   6.4 pp above the majority baseline, meaning DP-SGD genuinely learned the more frequent
   rare-class signal. The "DP collapse" finding is a property of *extreme* imbalance, not of
   DP-SGD per se.
2. **Full-pipeline compression undoes that signal.** The 30-round full-pipeline run on
   records 200–234 trains noisily (accuracy oscillates 65 %–86 %) and ends at the majority
   baseline (81.76 %). Top-k+int8 compression on top of DP-SGD's already-noisy gradients
   appears to push the regime back across the cliff. ε relaxed from 5 → 8 was not enough
   to compensate.

Reproduce all five rows with the configs in [`configs/followup_records200/`](configs/followup_records200/).

## Tests

```bash
pytest tests/ -v
```

Coverage: synthetic-data shapes/labels/determinism, partitioning invariants, classification
metrics, model forward shapes, DP config validation + lazy-import, **secure-aggregation mask
cancellation correctness**, compression round-trip bounds, YAML config inheritance + overrides.

## Documentation

- [`docs/architecture.md`](docs/architecture.md) — system architecture, per-round data flow,
  package layout.
- [`docs/threat-model.md`](docs/threat-model.md) — adversaries (A1–A5) and the
  attack-surface coverage matrix.
- [`docs/privacy-analysis.md`](docs/privacy-analysis.md) — DP-SGD via RDP, SecAgg correctness
  proof sketch, composition with DP, references.

## Citation

```bibtex
@misc{mapeke2026fl-healthcare-iot,
  title  = {Privacy-Preserving Federated Learning Framework for IoT-Based Healthcare Systems},
  author = {mapeke},
  year   = {2026},
  note   = {Undergraduate diploma project},
  url    = {https://github.com/mapeke/privacy-preserving-fl-healthcare-iot}
}
```

## License

MIT — see [LICENSE](LICENSE).
