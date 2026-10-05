# Remaining work and plugin retirement

The project, detection and segmentation annotation flows, tiled inference, TIFF batch prediction, output saving, retraining services, and headless API have usable implementations. The items below are still open or need real-data validation. The complete historical design and decisions are preserved in [the archived improvement plan](archive/IMPROVEMENT_PLAN.md).

## Later improvements

- Dataset/project browser filters, explicit review states, optional autosave drafts, and keyboard shortcuts.
- Model comparison summaries, clearer model lineage, per-class metrics, and richer training progress plots.
- Duplicate-data and quality warnings for annotation pools; optional hard-negative or uncertainty review queues.
- Performance benchmarking for lazy image input, Dask versus sequential tiling, controlled GPU batching, and out-of-memory fallback.
- Privacy controls for absolute source paths recorded in shareable run metadata.
