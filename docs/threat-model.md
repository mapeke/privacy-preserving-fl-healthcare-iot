# Threat Model

This document enumerates the adversaries the framework defends against, their capabilities, and which
component of the privacy stack mitigates each threat. It deliberately uses the precise vocabulary
expected in a privacy-research thesis.

## Assets being protected

| Asset                                  | Sensitivity                                  |
|----------------------------------------|----------------------------------------------|
| Raw ECG waveforms                      | PHI under HIPAA; highly sensitive            |
| Local labels (arrhythmia class)        | Diagnostic; highly sensitive                 |
| Per-client model updates (Δw_i)        | Leaks training data via gradient inversion   |
| Aggregated global model w_t            | Leaks via membership inference (mitigated by DP) |
| Patient identity / device identity     | Quasi-identifier; must not be re-linkable    |

## Adversaries and capabilities

### A1 — Honest-but-curious aggregation server

- **Capability:** sees every message that hits the server: per-client updates, weights, metadata.
- **Behavior:** follows the protocol faithfully; does not deviate, drop, or inject messages.
- **Goal:** infer training-data properties of any individual client from its update.
- **Mitigation:** secure aggregation (server only ever sees the masked sum, not individual updates).

### A2 — Curious-client coalition (size t < N)

- **Capability:** colludes among themselves; can pool their private masks and shares.
- **Behavior:** still trains honestly so as not to be detected by simple correctness checks.
- **Goal:** unmask a non-coalition client's update.
- **Mitigation:** pairwise masks between every pair `(i, j)` mean a coalition of size `t` learns
  nothing about non-coalition updates as long as at least one honest pair exists. With our default
  `N=4`, secagg withstands up to `t=2` without compromise.

### A3 — Network eavesdropper

- **Capability:** passive observation of the wire (e.g., on-path attacker on Wi-Fi).
- **Mitigation:** transport-layer encryption (TLS, assumed; out of scope for this prototype). The
  masked updates would also be uniformly random, providing a defense-in-depth.

### A4 — External adversary holding only the released global model

- **Capability:** queries / inspects the final global model weights (e.g., model is published).
- **Goal:** membership inference — "was patient P in the training set?"
- **Mitigation:** **differential privacy** with budget `(ε, δ)` provides a quantitative bound on how
  much any single record can change the output distribution. Our default targets are `ε ≤ 5`,
  `δ ≤ 1e-5`.

### A5 — Malicious client (Byzantine)

- **Out of scope** for this prototype. Defending against poisoning / backdoor attacks would require
  robust aggregation (Krum, median, trimmed mean). Documented as future work.

## Attack surface coverage matrix

| Attack                         | A1  | A2  | A3  | A4  | A5  |
|--------------------------------|-----|-----|-----|-----|-----|
| Gradient inversion             | ✅ SecAgg | ✅ SecAgg | ✅ TLS+SecAgg | n/a | ❌ OOS |
| Membership inference (per-update) | ✅ SecAgg | ✅ SecAgg | ✅ TLS+SecAgg | n/a | ❌ OOS |
| Membership inference (global model) | ✅ DP | ✅ DP | n/a | ✅ DP | ❌ OOS |
| Reconstruction from compressed update | ✅ Compression doesn't help adversary; SecAgg still holds | ✅ | ✅ | n/a | ❌ OOS |
| Model poisoning                | ❌ | ❌ | n/a | n/a | ❌ |

## Trust assumptions

1. **TLS is assumed** between every client and the server (standard mTLS in a real deployment).
2. **The PRG used to derive masks is secure** (we use Python's `secrets` for seed generation and a
   deterministic NumPy generator from that seed for mask material).
3. **No more than `N - 2` clients collude** for SecAgg correctness in the basic pairwise scheme.
4. **Hardware side channels** (timing, power, EM) on the IoT device are out of scope — a real
   deployment would also need hardware countermeasures.

## Out of scope

- Byzantine robustness / model poisoning defenses
- Hardware-level side channels
- Verifiable computation (proving the server aggregated correctly)
- Free-rider detection

These are flagged in the conclusions chapter as natural extensions.
