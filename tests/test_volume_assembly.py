"""Adjacent slice assembly preserves branches, conflicts and thresholds."""

import numpy as np
import pytest

from simple_napari_cci_annotator._instance_mask import PredictedInstance, compose_predictions
from simple_napari_cci_annotator._tiled_inference import InferenceCancelled
from simple_napari_cci_annotator._volume_assembly import (
    AssemblySettings, assemble_volume, open_assembly, save_assembly,
)
from simple_napari_cci_annotator._volume_store import VolumeMaskStore, VolumeStorageError
from simple_napari_cci_annotator import _volume_assembly as assembly_module


def make_store(tmp_path, slices, classes, *, incomplete=False):
    shape = (len(slices), *slices[0].shape)
    contract = {"volume": {"geometry": {"shape": list(shape)},
                           "selection": {"z_range": [7, 7 + len(slices)]}}}
    root = tmp_path / "run"
    with VolumeMaskStore.create(root, contract) as store:
        for index, labels in enumerate(slices):
            if incomplete and index == len(slices) - 1:
                break
            predictions = [PredictedInstance(labels == local, (0, 0, shape[2], shape[1]),
                                             classes[index][local], 0.8)
                           for local in sorted(classes[index])]
            result = compose_predictions(predictions, labels.shape, {0: "cell", 1: "other"},
                                         component_policy="preserve")
            store.write_slice(7 + index, result, {})
        if not incomplete:
            store.set_status("completed")
    return root


def test_branches_rejoin_and_class_conflicts_remain_separate(tmp_path):
    first = np.array([[1, 1, 1, 1, 0], [1, 1, 1, 1, 0]], dtype=np.uint32)
    middle = np.array([[1, 1, 2, 2, 0], [1, 1, 2, 2, 0]], dtype=np.uint32)
    last = np.array([[1, 1, 1, 1, 2], [1, 1, 1, 1, 2]], dtype=np.uint32)
    root = make_store(tmp_path, [first, middle, last],
                      [{1: 0}, {1: 0, 2: 0}, {1: 0, 2: 1}])
    result = assemble_volume(root)
    assert result.object_for_node == {(7, 1): 1, (8, 1): 1, (8, 2): 1,
                                      (9, 1): 1, (9, 2): 2}
    assert len([link for link in result.links if link.accepted]) == 4
    assert all(link.overlap_fraction == 1 for link in result.links)


def test_thresholds_classes_empty_slices_and_determinism(tmp_path):
    first = np.array([[1, 1, 0, 2], [1, 1, 0, 2]], dtype=np.uint32)
    second = np.array([[1, 0, 0, 2], [0, 0, 0, 2]], dtype=np.uint32)
    empty = np.zeros_like(first)
    root = make_store(tmp_path, [first, second, empty, first],
                      [{1: 0, 2: 0}, {1: 0, 2: 1}, {}, {1: 0, 2: 0}])
    result = assemble_volume(root, AssemblySettings(minimum_pixels=2, minimum_fraction=0.5))
    assert [link.reason for link in result.links] == ["below threshold", "class mismatch"]
    assert len(set(result.object_for_node.values())) == 6
    assert result == assemble_volume(root, AssemblySettings(minimum_pixels=2, minimum_fraction=0.5))
    accepted = assemble_volume(root, AssemblySettings(minimum_pixels=1, minimum_fraction=1, minimum_iou=0.25))
    assert accepted.links[0].accepted
    assert accepted.links[0].iou == 0.25


def test_incomplete_and_cancelled_runs_are_rejected(tmp_path):
    mask = np.ones((2, 2), dtype=np.uint32)
    root = make_store(tmp_path, [mask, mask], [{1: 0}, {1: 0}], incomplete=True)
    with pytest.raises(VolumeStorageError, match="every slice"):
        assemble_volume(root)
    with pytest.raises(InferenceCancelled):
        assemble_volume(root, cancelled=lambda: True)
    with pytest.raises(ValueError):
        assemble_volume(root, AssemblySettings(minimum_fraction=1.1))


def test_saved_assembly_reopens_lazily_without_changing_source(tmp_path):
    first = np.array([[1, 1, 0], [0, 0, 0]], dtype=np.uint32)
    second = np.array([[1, 1, 0], [0, 0, 2]], dtype=np.uint32)
    root = make_store(tmp_path, [first, second], [{1: 0}, {1: 0, 2: 1}])
    folder = save_assembly(root)
    other = save_assembly(root, AssemblySettings(minimum_pixels=3))
    assert folder != other
    _, _, graph, array = open_assembly(folder)
    np.testing.assert_array_equal(array.compute(), np.stack([first, second]))
    assert graph["settings"]["minimum_fraction"] == 0.5
    assert len(graph["links"]) == 1
    with VolumeMaskStore.open(root) as store:
        np.testing.assert_array_equal(store.read_slice(8)[0], second)
    (folder / "graph.json").write_text("{}", encoding="utf-8")
    with pytest.raises(VolumeStorageError, match="checksum"):
        open_assembly(folder)


def test_cancelled_assembly_is_not_published(tmp_path, monkeypatch):
    mask = np.ones((2, 2), dtype=np.uint32)
    root = make_store(tmp_path, [mask], [{1: 0}])
    original = assembly_module._atomic_json
    cancelled = [False]

    def cancel_after_graph(path, data):
        original(path, data)
        if path.name == "graph.json":
            cancelled[0] = True

    monkeypatch.setattr(assembly_module, "_atomic_json", cancel_after_graph)
    with pytest.raises(InferenceCancelled):
        save_assembly(root, cancelled=lambda: cancelled[0])
    folder = next((root / "assemblies").iterdir())
    assert not (folder / "assembly.json").exists()
    with pytest.raises(VolumeStorageError):
        open_assembly(folder)
