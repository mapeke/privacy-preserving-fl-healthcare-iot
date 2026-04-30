# Privacy Analysis

This note collects the formal privacy arguments behind each layer of the framework, in the form
expected in a thesis chapter.

## 1. Differential privacy of the local update (DP-SGD)

We use **DP-SGD** (Abadi et al., 2016) for local training. Each local SGD step is replaced with:

1. Compute per-sample gradients `g_i` for a minibatch `B`.
2. Clip each gradient: `ḡ_i = g_i / max(1, ‖g_i‖_2 / C)` with clipping threshold `C`.
3. Add Gaussian noise: `g̃ = (1/|B|)(Σ_i ḡ_i + 𝒩(0, σ²C²I))`.
4. Update weights: `w ← w - η · g̃`.

This step satisfies `(α, αC²σ⁻²/2)`-Rényi differential privacy at every order `α > 1` (Mironov, 2017).
Composing across `T` steps and converting to `(ε, δ)`-DP via the **RDP accountant** gives a tight bound.

### Parameter choices in this framework

| Parameter            | Default          | Justification                                     |
|----------------------|------------------|---------------------------------------------------|
| `target_epsilon`     | 5.0              | Common reasonable budget for medical models       |
| `target_delta`       | 1e-5             | `< 1/N` where `N` is dataset size                 |
| `max_grad_norm` (C)  | 1.0              | Standard; tunable per layer                        |
| `noise_multiplier` (σ) | auto-calibrated by Opacus to hit `target_epsilon` |  |

### What ε actually buys you

For the worst-case adversary A4 with auxiliary information about all *other* training records, the
posterior on whether a given record was in the training set shifts by at most a factor of `e^ε ≈ 148`
(at ε=5). This is a worst-case bound; empirical membership-inference accuracy is typically much closer
to 50% (random guess) at this budget.

## 2. Correctness of pairwise-mask Secure Aggregation

Let each client `i ∈ [N]` hold an update `Δw_i ∈ ℝ^d`. For every unordered pair `{i, j}`, both clients
derive a shared seed `s_{ij}` from a Diffie–Hellman key exchange (or from a pre-shared secret in the
prototype), expand it via a PRG into a mask vector `m_{ij} ∈ ℝ^d`, and define:

```
Δw̃_i = Δw_i + Σ_{j > i} m_{ij} − Σ_{j < i} m_{ji}      (mod p, for fixed-point arithmetic)
```

The server then computes:

```
S = Σ_i Δw̃_i
  = Σ_i Δw_i + Σ_i (Σ_{j > i} m_{ij} − Σ_{j < i} m_{ji})
  = Σ_i Δw_i + 0
  = Σ_i Δw_i
```

because every mask `m_{ij}` appears exactly once with `+` (in client `i`'s sum) and once with `−` (in
client `j`'s sum).

### Privacy

For any **strict subset** `S ⊊ [N]` of clients, the marginal distribution of `(Δw̃_i)_{i ∈ S}`
contains at least one mask `m_{ij}` with `j ∉ S`, which is uniformly random and known only to the two
endpoints. Therefore the joint distribution is uniformly random conditional on the true updates,
and the server learns no information about `(Δw_i)_{i ∈ S}` other than what it could infer from the
total sum. (Formal statement: Bonawitz et al., 2017, Theorem 4.1.)

### Limitations of the prototype

- We use a pre-shared seed per pair (set up at session start) instead of the full DH/Shamir scheme.
  This means the framework does **not** tolerate dropouts: every client that started a round must
  finish it. Real deployments use Shamir's secret sharing of the seed to recover from up to `t`
  dropouts.
- Fixed-point arithmetic is approximated in `float32` for simplicity. A production deployment would
  use modular integer arithmetic over a finite field.

These limitations are flagged in the framework's `secure_agg.py` docstrings.

## 3. Composition: DP + SecAgg

DP and secure aggregation **compose without interaction**:

- DP guarantees a bound on the final global model `w_T` regardless of what intermediate channel
  carried the updates.
- SecAgg guarantees that intermediate per-client updates were never visible to the server.

The composed system therefore satisfies both:

> An adversary observing only the released global model `w_T` learns at most `(ε, δ)` worth of
> information about any individual record, AND an adversary observing the per-round server inbox
> learns nothing about any individual client's update beyond the aggregate.

## 4. Compression and privacy

Top-k sparsification and 8-bit quantization are **lossy operators applied before mask addition**.
They do not weaken DP (post-processing immunity) and are compatible with SecAgg as long as both
sender and receiver agree on the sparsification mask (we transmit the indices alongside the values).
Compression's role here is bandwidth, not privacy; it is included because IoT bandwidth is the
binding constraint in deployment.

## References

1. Abadi, M., et al. *Deep Learning with Differential Privacy.* CCS 2016.
2. Mironov, I. *Rényi Differential Privacy.* CSF 2017.
3. Bonawitz, K., et al. *Practical Secure Aggregation for Privacy-Preserving Machine Learning.*
   CCS 2017.
4. McMahan, H. B., et al. *Communication-Efficient Learning of Deep Networks from Decentralized
   Data.* AISTATS 2017.
5. Yousefpour, A., et al. *Opacus: User-Friendly Differential Privacy Library in PyTorch.* 2021.
