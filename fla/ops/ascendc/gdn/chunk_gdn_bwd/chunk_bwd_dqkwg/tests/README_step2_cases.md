# Step2 DQKWG Cases

This directory contains `case_step2_01` through `case_step2_12` for `chunk_bwd_dqkwg`.

Run all cases with fixed-seed golden cache and perf:

```bash
TEST_DEVICE_ID=14 ./run_step2_cases.sh all
```

Run selected cases:

```bash
TEST_DEVICE_ID=14 ./run_step2_cases.sh case_step2_07 case_step2_08
```

Run perf only from an existing cache:

```bash
DQKWG_STEP2_SKIP_GOLDEN=1 TEST_DEVICE_ID=14 ./run_step2_cases.sh all
```

Useful environment variables:

```text
DQKWG_STEP2_CACHE_ROOT      Cache root for fixed inputs and CPU golden outputs.
DQKWG_STEP2_RESULTS_DIR     Output directory for summary, golden, and perf files.
DQKWG_STEP2_REFRESH_CACHE   Set to 1 to rebuild fixed-seed cache.
DQKWG_STEP2_SKIP_GOLDEN     Set to 1 to skip NPU golden comparison.
DQKWG_STEP2_SKIP_PERF       Set to 1 to skip perf timing.
DQKWG_STEP2_WARMUP          Perf warmup iterations, default 3.
DQKWG_STEP2_REPEAT          Perf repeat iterations, default 20.
```

The runner uses deterministic seeds `20242618..20242629` for the 12 cases and stores
the materialized `cu_seqlens` in `step2_cache_manifest.json`.
