# Business Entity Resolution at Scale

Match businesses across three noisy data sources (~15–30M records) using **only free-tier, CPU-only compute** (Google Colab, ~12.7 GB RAM). Built in 72 hours for the Amazon ML Challenge 2026.

| | |
|---|---|
| **Final leaderboard macro F₀.₅** | **0.729** (first submission: 0.612) |
| Validation macro F₀.₅ | 0.7396 (40k held-out entities, scored like the leaderboard) |
| Compute | CPU only, ~12.7 GB RAM, no GPU |
| Scale | 36.2M candidate pairs over 1.73M test entities, ~1 hour for the test pass |

## The problem

Source 1 is a deduplicated reference set of businesses. Sources 2 and 3 hold noisy records of the same businesses: legal-suffix variants (Ltd/Limited, Pvt/Private), abbreviations, typos, word-order swaps, missing or reordered address parts. For each Source-1 entity, predict which Source-2/3 records refer to it. The test set includes **France, which never appears in training**.

## Approach

```
raw TSVs ─► normalize + 6 blocking keys ─► country-partitioned index ─► candidates (≤60/entity)
        ─► 19 features ─► HistGradientBoosting ─► two-threshold rule ─► 1-to-1 resolution ─► output
```

1. **Memory-bounded, vectorized blocking.** A CSR-style index with no per-row Python loop. A per-key candidate budget keeps one coarse key from crowding out the precise ones. Country is a lossless partition (zero cross-country matches in the ~7.6M ground-truth pairs).
2. **Composite `postal × name-prefix` key.** Its buckets are small, so they never get truncated. This was the single biggest gain: blocking recall went from 47.7% to 57.0% on real data.
3. **Entity-relative features.** Besides 12 string/postal pair features, 7 features describe how a candidate compares to the entity's *other* candidates (gap to best score, rank, candidate count).
4. **Top-candidate decision rule.** Accept if `p ≥ T_HI`, or if it is the entity's best candidate and `p ≥ T_LO`. Both thresholds are tuned directly for macro F₀.₅. It exists because 94% of entities have a match, so predicting "no match" is costly.
5. **No country-specific logic.** Accent transliteration (NFKD → ASCII) rather than deletion keeps French names intact.

## Results (real data, same validation entities)

| Pipeline | Blocking recall | Val. macro F₀.₅ |
|---|---|---|
| 5 keys; LogReg, 4 features, 1 threshold | 47.7% | 0.6408 |
| + composite postal × name-prefix key | 57.0% | 0.7001 |
| + HistGradientBoosting, 19 features | 57.0% | 0.7383 |
| + top-candidate rule (**final**) | 57.0% | **0.7396** |
| *ceiling: perfect classifier on these candidates* | 57.0% | 0.7622 |

The classifier is within 0.023 of perfect on its candidates. **The bottleneck is blocking recall**: 57% of true pairs reach the candidate set, against a 94% key-overlap ceiling.

## Roadmap

- [ ] Token-level blocking: index rare name tokens, with a frequency cap
- [ ] More composite keys (postal × rare token)
- [ ] Similarity-ranked selection inside large buckets, instead of random truncation
- [ ] Investigate India's 27% predicted-singleton rate (vs. 13% France/US)
- [ ] Refactor the single script into modules (`normalize`, `blocking`, `features`, `model`, `resolve`) with tests
- [ ] Local-path data loading (currently reads from S3); Windows support (uses the Linux-only `resource` module)
- [ ] Embedding / ANN-based blocking

## Repo layout

```
src/run_pipeline.py             full pipeline (exported 1:1 from the validated notebook)
notebooks/entity_resolution_v13.ipynb
docs/technical_report.md        methodology, ablations, error analysis
docs/DATA.md                    dataset layout, schemas, how to obtain it
docs/run_guide.md               setup, credentials, expected runtimes
docs/engineering_log.md         bottleneck log from the 72h build
results/sample_output.tsv       format example (first rows only)
```

## Run it

The data is owned by the challenge organizers and is **not included**; see [docs/DATA.md](docs/DATA.md) for the layout and schemas.

```bash
pip install -r requirements.txt
export BUCKET_NAME=your-bucket AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=... AWS_DEFAULT_REGION=us-east-1
python src/run_pipeline.py     # Linux/macOS/Colab; see docs/run_guide.md
```

Expect ~10 min for the train pass and 50–65 min for the test pass on Colab free tier.

## Author

Milind Pandey. Team "Gradient Warriors", Amazon ML Challenge 2026. This is an independent project and is not affiliated with or endorsed by Amazon.

Licensed under MIT.
