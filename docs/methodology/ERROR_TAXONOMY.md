# Error Taxonomy

Each failed or partially failed prediction should be assigned one or more error labels.

## Data errors

- `missing_information`
- `stale_data`
- `wrong_market_data`
- `timestamp_leakage`
- `source_conflict`

## Reasoning / model errors

- `wrong_direction`
- `magnitude_underestimate`
- `magnitude_overestimate`
- `weight_misspecification`
- `event_decay_misspecification`
- `breadth_vs_index_confusion`
- `open_vs_close_target_confusion`
- `regime_misclassification`

## Process errors

- `prediction_rewritten_after_open`
- `unversioned_model_change`
- `insufficient_sample_change`
- `benchmark_not_checked`
