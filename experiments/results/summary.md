# Results

| policy    |   runs |   requests |   errors |   hit_rate |   hit_rate_after |   moved_after_scale |   p50_latency_ms |   p95_latency_ms |   p95_prompt_ms |   imbalance |
|:----------|-------:|-----------:|---------:|-----------:|-----------------:|--------------------:|-----------------:|-----------------:|----------------:|------------:|
| bounded   |      1 |         14 |        0 |      0.801 |              nan |                 nan |             4141 |          10521.7 |         6864    |       1.208 |
| llm-aware |      1 |         14 |        0 |      0.937 |              nan |                 nan |             1844 |           6318.1 |         5118.44 |       1.136 |
