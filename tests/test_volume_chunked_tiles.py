"""Chunked tile writes match the current in-memory tiling semantics."""

import numpy as np
import pytest

from simple_napari_cci_annotator._instance_mask import PredictedInstance, compose_predictions
from simple_napari_cci_annotator._segmentation_io import InstanceRecord
from simple_napari_cci_annotator._segmentation_tiling import (
    TiledSegmentationEngine, _canonical_relabel, _seam_equivalences,
)
from simple_napari_cci_annotator._tiled_inference import InferenceCancelled, InferenceSettings
from simple_napari_cci_annotator._volume_chunked_tiles import predict_tiles_to_chunks
from simple_napari_cci_annotator._volume_chunked_fusion import _chunked_seams, fuse_tile_chunks


class Predictor:
    def predict_image(self, image, settings, classes, *, component_policy="largest"):
        mask = image[..., 0] > 100
        predictions = [PredictedInstance(mask, (0, 0, mask.shape[1], mask.shape[0]), 0, 0.8)] if np.any(mask) else []
        return compose_predictions(predictions, mask.shape, classes, component_policy=component_policy)


@pytest.mark.parametrize("shape", [(100, 130), (64, 64), (33, 75)])
def test_chunked_tiles_match_existing_seam_fusion(tmp_path, shape):
    zarr = pytest.importorskip("zarr", minversion="3")
    yy, xx = np.indices(shape)
    mask = ((xx - shape[1] // 2) ** 2 + (yy - shape[0] // 2) ** 2) < 800
    image = np.repeat((mask.astype(np.uint8) * 255)[..., None], 3, axis=2)
    rgb = zarr.create_array(store=str(tmp_path / "rgb.zarr"), data=image, chunks=(20, 20, 3), zarr_format=3)
    settings = InferenceSettings(tile_size=64, overlap=16, max_detections=10)
    core = settings.tile_size - 2 * settings.overlap
    padded = tuple(size + (-size) % core for size in shape)
    output = zarr.create_array(store=str(tmp_path / "labels.zarr"), shape=padded,
                               chunks=(core, core), dtype="uint32", zarr_format=3)
    predictor = Predictor()
    registry, metadata = predict_tiles_to_chunks(rgb, output, predictor, settings, {0: "Cell"})
    temporary = output[:]
    assert np.any(temporary), (registry.completed, list(registry.records))
    equivalences, _, _ = _seam_equivalences(temporary, core, registry.records)
    actual, _, _ = _canonical_relabel(temporary, registry.records, {0: "Cell"}, equivalences)
    expected = TiledSegmentationEngine(predictor).predict(image, settings, {0: "Cell"}, component_policy="preserve")
    np.testing.assert_array_equal(actual[:shape[0], :shape[1]], expected.mask)
    assert metadata["grid"] == expected.provenance["grid"]
    assert metadata["padding"] == expected.provenance["padding"]
    fused = zarr.create_array(store=str(tmp_path / "fused.zarr"), shape=shape,
                              chunks=(17, 19), dtype="uint32", zarr_format=3)
    records, mapping, pairs, conflicts, ambiguous = fuse_tile_chunks(
        output, fused, core, registry.records, {0: "Cell"}, chunk_xy=19,
    )
    np.testing.assert_array_equal(fused[:], expected.mask)
    assert records == expected.instances
    assert pairs == equivalences
    assert {str(key): value for key, value in mapping.items()} == expected.provenance["temporary_to_final"]


def test_chunked_tiles_cancel_before_prediction():
    settings = InferenceSettings(tile_size=64, overlap=16)
    with pytest.raises(InferenceCancelled):
        predict_tiles_to_chunks(np.zeros((32, 32, 3), np.uint8), np.zeros((32, 32), np.uint32),
                                Predictor(), settings, {0: "Cell"}, cancelled=lambda: True)


def test_chunked_seams_preserve_class_conflicts_and_ambiguity():
    mask = np.array([[1, 1, 2, 2], [1, 1, 2, 2],
                     [3, 4, 5, 5], [3, 4, 5, 5]], dtype=np.uint32)
    records = {i: InstanceRecord(i, cls, str(cls), 0.8, "prediction", "predicted",
                                 (0, 0, 1, 1), 1) for i, cls in [(1, 0), (2, 1), (3, 0), (4, 0), (5, 1)]}
    expected = _seam_equivalences(mask, 2, records)
    assert _chunked_seams(mask, 2, records, 1, lambda: None) == expected
