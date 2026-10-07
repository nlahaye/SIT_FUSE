---
description: Detailed steps for the context assignment of SIT_FUSE output
---

# Post-processing

In order to make sense of our trained data, we must assign labels to our output to see how well the model performed. The process is know as context assignment, and differs depending on what data you're working with. Below include examples of the context assignment process for wildfire data and harmful alagal bloom (HAB) data.

***

{% content-ref url="what-is-context-assignment.md" %}
[what-is-context-assignment.md](what-is-context-assignment.md)
{% endcontent-ref %}

## SHAP and TimeSHAP analysis

Run `python -m sit_fuse.xai.xai -y config.yml`. Spatial encoders use Kernel
SHAP with the existing zero baseline and hierarchical numeric-label target;
RTDBN uses per-cluster TimeSHAP scores. The SHAP/TimeSHAP dependency pins in
`pyproject.toml` remain required.

Optional YAML settings (CLI arguments override them):

```yaml
xai:
  max_samples_per_cluster: 200  # spatial samples; 0 explains all
  max_windows_per_cluster: 200  # RTDBN windows; 0 explains all
  batch_size: 512              # maximum model inference batch size
  seed: 42
  # max_evals: 500             # coalition samples per explanation
```

The corresponding CLI flags are `--max-samples-per-cluster`,
`--max-windows-per-cluster`, `--batch-size`, `--seed`, and `--max-evals`.
Without `max_evals`, spatial SHAP retains its automatic coalition budget and
TimeSHAP retains 500 samples. Evaluation budgets are coalition samples, **not**
a cap on total model calls. Inference runs in evaluation mode without gradients.

Sampling is deterministic within predicted clusters, retains rare clusters,
and defaults to at most 200 observations per cluster. Spatial analysis still
uses the first configured test file; RTDBN uses held-out test windows.
Existing output names and directories are retained. Spatial selected row indices
are saved in `explanation_sample_indices.npy`, with settings in
`explanation_settings.json`; RTDBN indices are in
`rtdbn_timeshap/sample_indices.npy`. Spatial caches are regenerated when settings,
indices, sampled input content, configuration, or checkpoint sizes/timestamps
differ (including legacy caches without sampling metadata).
The legacy `kmeans_background` filenames still refer to a zero baseline, and
`shap_values_kmeans_background.npz` remains pickle-serialized for compatibility.

These sampled reports are global summaries, not exhaustive local explanations.
Per-cluster sampling changes cluster proportions; population-wide aggregation
should account for original cluster prevalence. Before reducing `max_evals`,
compare feature rankings, signs, and stability against a higher-budget run on
the same selected observations. Numeric hierarchical cluster IDs remain the
spatial target, not per-cluster probabilities; their arbitrary numbering should
be considered when interpreting attributions.

Lightweight regression tests use synthetic arrays and mocked model/explanation
services: `python -m unittest discover -s tests -p 'test_xai.py'` (requires NumPy).

{% content-ref url="wildfire-example/" %}
[wildfire-example](wildfire-example/)
{% endcontent-ref %}

{% content-ref url="harmful-algal-bloom-example/" %}
[harmful-algal-bloom-example](harmful-algal-bloom-example/)
{% endcontent-ref %}
