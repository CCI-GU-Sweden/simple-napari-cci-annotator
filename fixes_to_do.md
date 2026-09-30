# Improvement plan audit and next work

Reviewed against `IMPROVEMENT_PLAN.md` and the current source on 2026-09-30. The four earlier widget findings were addressed with regression tests. The broader plan work remains open.

## Mini plan: batch processing and output saving

The current-image full-resolution output writer and **Save Current Output** action cover the single-image part of this workflow. The first batch workflow is also implemented: top-level TIFF/OME-TIFF input folders, per-plane prediction with locked project conversion, a configurable `Prediction` output parent, cancellation, per-file error continuation, and `analysis_metadata.json` with source/model/output checksums. Follow-up enhancements from the original mini plan remain:

1. Add a preflight summary with dimension checks, disk space estimate, and destination conflict checks.
2. Stream very large TIFF series plane by plane when the TIFF layout supports it; currently one series is loaded at a time.
3. Add per-file durations and a safe resume flow that verifies saved checksums.
4. Open a saved batch result as an editable napari annotation for selective crop and canonical saving.

## Verification limits

The plugin, segmentation, output writer, and batch tests pass in the `napari-yolo` Conda environment with `pytest-qt`, `QT_QPA_PLATFORM=offscreen`, and automatic pytest plugin loading disabled.
