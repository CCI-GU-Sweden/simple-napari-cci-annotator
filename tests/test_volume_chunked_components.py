"""Component extraction agrees with the existing full-plane implementation."""

import numpy as np
import pytest

from simple_napari_cci_annotator._instance_mask import ComposedInstances, split_prediction_components
from simple_napari_cci_annotator._segmentation_io import refresh_instance_records
from simple_napari_cci_annotator._volume_chunked_components import split_components_to_chunks


@pytest.mark.parametrize("chunk", [3, 5, 8])
@pytest.mark.parametrize("seed", [3, 17, 42])
def test_components_match_full_plane_across_chunk_boundaries(tmp_path, chunk, seed):
    zarr = pytest.importorskip("zarr", minversion="3")
    rng = np.random.default_rng(seed)
    mask = rng.integers(0, 4, size=(13, 17), dtype=np.uint32)
    mask[4:9, 3:13] = 2
    mask[0, 0] = mask[-1, -1] = 3
    raw = {i: {"class_id": 0, "source": "prediction", "status": "predicted",
               "confidence": 0.8, "lineage": [i]} for i in (1, 2, 3)}
    records = refresh_instance_records(mask, raw, {0: "Cell"})
    expected = split_prediction_components(ComposedInstances(mask, records, ()), {0: "Cell"})
    original = zarr.create_array(store=str(tmp_path / "original.zarr"), data=mask,
                                 chunks=(chunk, chunk), zarr_format=3)
    scratch = zarr.create_array(store=str(tmp_path / "scratch.zarr"), shape=mask.shape,
                                chunks=(chunk, chunk), dtype="uint64", zarr_format=3)
    output = zarr.create_array(store=str(tmp_path / "output.zarr"), shape=mask.shape,
                               chunks=(chunk, chunk), dtype="uint32", zarr_format=3)
    actual_records, provenance = split_components_to_chunks(original, scratch, output, records, chunk_xy=chunk)
    np.testing.assert_array_equal(output[:], expected.mask)
    assert actual_records == expected.instances
    assert provenance == expected.provenance["slice_components"]


def test_empty_slice_is_supported():
    mask = np.zeros((4, 5), dtype=np.uint32)
    scratch = np.empty(mask.shape, dtype=np.uint64)
    output = np.empty_like(mask)
    records, provenance = split_components_to_chunks(mask, scratch, output, {}, chunk_xy=3)
    assert not np.any(output)
    assert records == provenance == {}
