# Large-tile performance review

Archived after the main segmentation bottlenecks were resolved on 2026-10-07.

## Implemented improvements

- Added phase timings and detailed napari progress. Enable them with:

  ```python
  import simple_napari_cci_annotator._debug as debug
  debug.VERBOSE_PERFORMANCE_TIMING = True
  ```

- Replaced per-instance geometry scans with shared label statistics. A 2048×2048 mask with 256 IDs improved from 3.802 seconds to 0.041 seconds.
- Replaced per-instance disconnected-component scans with one label-aware pass. The synthetic 2048×2048 / 256-ID case improved from 6.003 seconds to 0.081 seconds. A production napari import that previously spent 2250.958 seconds validating disconnected instances subsequently completed all import work in 1.410 seconds on a smaller representative image.
- Replaced canonical per-ID mask relabeling with one lookup-table operation. The synthetic 1024×1024 / 256-ID case improved from 1.204 seconds to 0.030 seconds.
- Replaced tile-local per-ID remapping with one lookup-table operation per tile.

## Production result

The same 49-tile image with approximately 1,500 objects was measured before and after lookup-table relabeling:

| Phase | Before | After |
| --- | ---: | ---: |
| Tile inference and assembly | 63.080 s | 61.620 s |
| Fusion and relabeling | 29.046 s | 0.548 s |
| Total tiled segmentation | 92.765 s | 62.852 s |
| Napari result import | 1.412 s | 1.410 s |

Fusion improved by approximately 53×. Final merging and napari import are no longer significant bottlenecks for this image.

## Remaining optional work

1. **Profile tile prediction composition.** Tile inference and assembly now dominates runtime. This phase combines model execution and CPU-side `compose_predictions`; GPU acceleration only addresses model execution.
2. **Optimize `compose_predictions` if profiling justifies it.** It still contains repeated per-instance tile-mask operations.
3. **Vectorize detection NMS if large detection jobs are slow.** The current Python implementation grows approximately quadratically with the candidate count.
4. **Remove duplicate validation and metadata refreshes.** These are now relatively inexpensive but still occur in some import and output-saving paths.
5. **Reduce detailed provenance when necessary.** Large temporary-ID mappings and conflict lists may increase metadata size on exceptionally dense images.

Seam matching measured only 0.022 seconds on the 49-tile production image and does not need optimization.

Performance changes must continue to preserve deterministic IDs, lineage, class-aware fusion, confidence selection, source-shape cropping, border clearing, cancellation, and reproducibility metadata.
