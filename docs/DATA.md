# Data

The dataset was provided by the Amazon ML Challenge 2026 organizers and is **not redistributed here**. It is ~2.3 GB across 7 TSV files, and the terms of the challenge govern its use.

## Layout the pipeline expects

```
<bucket>/dataset/train/train_source1.tsv       2,206,821 rows
<bucket>/dataset/train/train_source2.tsv       5,034,616 rows
<bucket>/dataset/train/train_source3.tsv       5,285,603 rows
<bucket>/dataset/train/train_ground_truth.tsv  2,206,821 rows
<bucket>/dataset/test/test_source1.tsv         1,732,544 rows
<bucket>/dataset/test/test_source2.tsv         4,887,273 rows
<bucket>/dataset/test/test_source3.tsv         5,082,316 rows
```

`run_pipeline.py` reads these from `s3://$BUCKET_NAME/dataset/...`. To run it you need to obtain the files from the challenge materials and upload them to your own S3 bucket with this layout.

## Schemas (tab-separated, with header)

| File | Columns |
|---|---|
| `*_source{1,2,3}.tsv` | `entity_id`, `business_name`, `business_address`, `country` |
| `train_ground_truth.tsv` | `source1_entity_id`, `matched_entity_ids` (comma-separated S2/S3 ids; empty = no match) |

- Train covers US and India; test adds France.
- IDs are prefixed `S1-`, `S2-`, `S3-`.

## Output format

`output/matching_results.tsv`: `source1_entity_id`, `matched_entity_ids`. This is the scored file. See `results/sample_output.tsv`.
`output/candidate_pairs.tsv`: `source1_entity_id`, `candidate_entity_ids`.
