# CLAUDE.md

Working notes for any Claude (or human) session that touches this repo. Keep this file tight and pointed. If a section grows past one screen, split it out into `docs/`.

## What this project is

A privacy-preserving federated learning framework for IoT-based healthcare, instantiated on MIT-BIH ECG arrhythmia classification with four simulated wearable IoT clients. Three independently switchable privacy layers stack on top of FedAvg:

1. **Differential privacy** (DP-SGD via Opacus) — `src/privacy/differential_privacy.py`
2. **Pairwise-mask secure aggregation** — `src/privacy/secure_agg.py`
3. **Top-k + 8-bit compression** — `src/privacy/compression.py`

Stack: Python 3.12, PyTorch 2.4, Flower 1.25, Opacus 1.5, wfdb 4.3. Single GPU is enough for any experiment in this repo.

The thesis at `docs/thesis/main.pdf` is the canonical interpretation of the results. Read it before making any structural change to the privacy layers — its three findings constrain what the framework is allowed to claim.

## Working agreement

- **Commit cadence:** one logical change per commit. Push after every commit. Conventional-commit messages (`feat(client):`, `fix(data):`, `docs(thesis):`, etc.).
- **Don't skip hooks.** `--no-verify` is never the right tool here.
- **Don't tune for headline numbers.** If a reproducibility check shows a gap vs. published results, document the gap (sources, magnitude) instead of tuning until it disappears. The thesis's reproducibility-check section sets this convention; don't undo it.
- **Don't add backwards-compat shims.** Internal callers can change in lockstep.
- **Bug fixes get their own commit** with a message that explains *why*, not just *what*. See `7596fca` (wfdb API) and `40dd3f4` (Opacus + inplace ReLU) for the format.

## Test policy

- **Mandatory:** `tests/test_secagg.py::test_masks_cancel_under_aggregation`. This is the cryptographic correctness invariant from `docs/privacy-analysis.md §2`. If you change the SecAgg protocol, the test must still pass — that is the contract.
- **Mandatory:** `tests/test_dp.py` lazy-import assertion. `attach_dp` must import Opacus inside the function body, not at module top, so the rest of the test suite runs without Opacus installed.
- **Smoke coverage:** everything else in `src/`. Tests should run in under 10 seconds total (currently 4 s, 42 tests).
- Run with `pytest tests/ -q` from the repo root.

## How to run experiments

Always invoke as a module from the repo root:

```bash
PYTHONPATH=. python -m experiments.run_centralized      --config configs/default.yaml --epochs 5
PYTHONPATH=. python -m experiments.run_federated        --config configs/default.yaml
PYTHONPATH=. python -m experiments.run_federated_dp     --config configs/experiment_dp.yaml
PYTHONPATH=. python -m experiments.run_federated_secagg --config configs/experiment_secagg.yaml
PYTHONPATH=. python -m experiments.run_full_pipeline    --config configs/experiment_full.yaml
```

Each writes `results/<experiment_name>/{run.log, resolved_config.yaml, history.json}`. The resolved config is the source of truth for reproducing a run — it includes every override applied.

CLI flags override YAML keys via dotted path (e.g., `--target-epsilon 8.0` maps to `privacy.differential_privacy.target_epsilon`). See `experiments/run_*.py` for the supported flags.

## Gotchas already paid for

- **`nn.ReLU(inplace=True)` is incompatible with Opacus.** Per-sample gradient hooks attach `BackwardHookFunction` views to layer outputs; in-place ops fail with a `view+inplace` RuntimeError. In a Flower simulation Ray swallows the exception and the client returns its broadcast parameters unchanged, producing silently flat training. Models in `src/models/` use non-inplace ReLU. Don't change that back. (Fixed in `40dd3f4`.)
- **`wfdb` removed `pn_cache`** between 4.1 and 4.3. We use `wfdb.dl_database` to materialise records into our own cache directory and then read from disk. (Fixed in `7596fca`.)
- **Flower 1.25 deprecation warnings** about `client_fn(cid)` vs. `client_fn(context)` are non-blocking. The simulation works with the old signature; switching to the new Context API is a future-work item.
- **Protobuf conflicts.** Installing `flwr[simulation]` upgrades protobuf, which then conflicts with TensorFlow / `grpcio-health-checking`. Those conflicts are not relevant to this project; ignore them.

## Dataset and result gotchas

- **MIT-BIH records 100–115** are heavily imbalanced (97.1% class N, 0.03% F, 0.03% Q). Don't propose a fix to the DP class-prior collapse without also acknowledging that's what the records do.
- **Final DP test accuracy is 0.9711 = class-N prevalence to four decimals.** This is a finding, not a bug. The thesis's Chapter 6 hangs on it. If your changes accidentally make this number move while DP is still enabled, double-check you haven't disabled DP somewhere.
- **Records 200–234** contain more arrhythmia variety and would change the dataset's class balance. Listed in the thesis as future work; not currently used.

## Privacy-stack invariants worth preserving

- DP, SecAgg, and compression are independent switches. `dp_config=None`, `secagg_session=None`, `compression_method=None` (or `"none"`) each turn the corresponding layer off without touching the others.
- The `PrivacyAwareFedAvg` strategy falls through to `super().aggregate_fit` when SecAgg is off — vanilla weighted FedAvg behaviour is preserved bit-exact for that path.
- The pairwise-mask SecAgg is *not* dropout-tolerant (no Shamir secret sharing). If a client drops mid-round, aggregation produces garbage. Documented in `src/privacy/secure_agg.py` and `docs/privacy-analysis.md`. Don't deploy this prototype as-is; the simulation is OK because clients never drop.
- Compression is post-processing-immune to DP and compatible with SecAgg as long as both endpoints agree on the sparsification mask. Don't sparsify on one side without sending the indices.

## Documentation

| File | Purpose |
|---|---|
| `README.md` | Public entry point, quick start, results table |
| `docs/architecture.md` | System architecture + per-round data flow |
| `docs/threat-model.md` | A1–A5 adversaries, attack-surface coverage matrix |
| `docs/privacy-analysis.md` | DP-SGD via RDP, SecAgg correctness sketch, composition |
| `docs/thesis/main.pdf` | The diploma writeup itself (23 pages) |
| `docs/thesis/main.tex` | LaTeX source; `./build.sh` rebuilds the PDF |

## When in doubt

Read `docs/thesis/main.pdf §1.4 Scope and non-goals` before adding a feature. The non-goals list is binding; if a request lands outside it, ask before implementing.
