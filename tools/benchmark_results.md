# CSS runtime benchmark

Captured 2026-09-19 on Chrome 153.0.0.0 (macOS 10.15.7) through a local
HTTP server. The benchmark page loads the exact stylesheet, warms the graph
twice, then measures five full 96-step CSSOM rollouts for each of two fixed
prompts. It reports nearest-rank median and p95 over ten measured rollouts.

The baseline is the checked-in artifact from `99a9d739f6ae48476b36653133dfc82aa0b9011d`.
The candidate is the generated artifact on `luna/css-coefficient-pooling`.
Both captures used cache-busting stylesheet URLs in the same browser profile.

| Metric | Baseline | Candidate | Change |
| --- | ---: | ---: | ---: |
| CSS bytes | 86,770,655 | 75,896,232 | -12.53% |
| Artifact SHA-256 | `1411db243749e7b28808b53456b5e968aa202690eabb6988284cc0049cf73577` | `304dc90ffb5b62866a97b47bf0e89c6453d768b85b00431a39727a6a3bf6ba43` | different artifact |
| Full rollout mean | 2,025.49 ms | 1,652.31 ms | -18.42% |
| Full rollout median | 2,025.20 ms | 1,653.00 ms | -18.38% |
| Full rollout p95 | 2,035.50 ms | 1,658.20 ms | -18.54% |

The baseline and candidate produced the same output hashes for both prompts:

```text
user: what can you do?       2fe52e9f
user: write a css button     887d07c0
```

With `verify=1`, the candidate page also compared the emitted stylesheet
against `model/weights.json`: all 10,389 distinct matrix values and 128 bias
values were present. A typed CSSOM probe parsed all 10,517 distinct emitted
numbers with zero mismatches under a `1e-5` relative tolerance (maximum
absolute error `4.9976e-6`, caused by CSSOM's normalized number serialization).

Chrome's optional `performance.memory` counters were also captured as
diagnostics. Used heap before/after measurement was 91,335,619/91,339,151
bytes for the baseline and 80,642,541/80,650,561 bytes for the candidate;
these counters are process-level observations and are not treated as a
normalised memory benchmark.

To reproduce, serve the repository and open:

```text
/tools/benchmark_runtime.html?label=candidate&revision=local&cache=unique
```

Use the same page against a baseline checkout or artifact copy, changing the
label, revision, and cache query values. The page prints JSON containing the
artifact hash, sample timings, output hashes, and available memory counters.
