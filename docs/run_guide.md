# Business Entity Resolution — Pipeline

Blocking + classifier pipeline for the Amazon ML Challenge 2026 Business Entity
Resolution task. Reproduces `output/matching_results.tsv` and
`output/candidate_pairs.tsv` from the raw Source-1/2/3 `.tsv` files.

## What this is

`src/run_pipeline.py` is the full pipeline, in the same order it was developed
and validated in: load → build blocking keys → build a country-partitioned
blocking index → stream train data through candidate generation + feature
computation → fit a classifier and tune a decision threshold → stream test
data through the same candidate generation + feature computation + scoring →
1-to-1 de-duplicate → write both output TSVs → run a self-check against the
organizers' structural rules.

It is a single script rather than a package because every stage streams
through shared in-process state (an id-code space, a country-partitioned
index, and per-chunk Parquet caches) built earlier in the same run — splitting
it into importable modules would not make it more reusable, since none of the
pieces are meant to run standalone against a partial pipeline state.

## Requirements

```bash
pip install -r requirements.txt
```

Python 3.10+. Developed and run on Google Colab's free tier (CPU-only,
~12.7 GB RAM) — no GPU is used or needed.

## Data & credentials

The pipeline reads the train/test `.tsv` files directly from an S3 bucket
(`s3://<bucket>/dataset/train/...`, `s3://<bucket>/dataset/test/...`), rather
than a local path, since that's how the input data was distributed for this
challenge.

Set before running:

```bash
export BUCKET_NAME=your-bucket-name
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_DEFAULT_REGION=us-east-1   # match your bucket's region
```

If running on an AWS-hosted instance with an attached IAM role (e.g.
SageMaker), the credential env vars can be omitted — `boto3`/`s3fs` pick up
the role automatically.

To point at a different region/bucket without editing the script, the env
vars above are read at the top of `src/run_pipeline.py`; everything else
(chunk size, candidate caps, key columns, model, threshold sweep range) is a
plain constant near the top of that file under "Config" — no CLI flags, by
design, since these are exactly the knobs documented and justified in the
top-level project `README.md`'s bottleneck log.

## Run

```bash
cd src
python run_pipeline.py
```

Writes `output/candidate_pairs.tsv` and `output/matching_results.tsv`
(relative to the working directory `run_pipeline.py` is launched from), plus
intermediate Parquet caches under `cache/`. Expect roughly 10 minutes for
the train pass and 50-65 minutes for the test pass on Colab's free-tier CPU,
plus one-time index-build time proportional to pool size — see the
top-level project README's "Run steps" for section-by-section expected
timings and what to watch for.

Progress and diagnostics (peak memory, per-chunk timing, blocking-recall
ceiling, validation F_0.5, per-country singleton rates) print to stdout as
the pipeline runs; the last cell's self-check prints 5 PASS/FAIL lines
against the organizers' structural rules (every S1 id present exactly once,
no duplicate matched ids, matched ids ⊆ that row's candidates, etc.) before
you'd spend a submission on the output.

## Validate before submitting

```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

(`utils/validate_submission.py` is the organizer-provided validator from the
student resources, not part of this repo.)

## Approach summary

See `Documentation_template.md` at the root of this submission for the full write-up.
In short:

- **Blocking**: a country-partitioned, 6-key blocking index. Candidates are only
  generated within the same country, and the keys include a composite
  postal-code × name-prefix key. This narrows ~15-30M records to at most 60 candidates
  per entity.
- **Scoring**: a `HistGradientBoostingClassifier` scores each candidate on 19 string,
  postal-code and entity-relative features.
- **Decision**: a two-threshold rule tuned for macro F_0.5 on a validation split
  accepts candidates with p ≥ T_HI, plus each entity's best candidate if p ≥ T_LO.
- **1-to-1 resolution**: any record claimed by several entities is kept only for its
  highest-scoring claim.
