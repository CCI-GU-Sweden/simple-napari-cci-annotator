"""Layer selection behavior with napari's actual layer list."""

import numpy as np
import pytest

from simple_napari_cci_annotator._project_store import ProjectStore
from simple_napari_cci_annotator._widget import SimpleCciAnnotatorQWidget


@pytest.mark.parametrize("task", ["segment", "detect"])
def test_selecting_and_removing_image_with_project(tmp_path, qtbot, task):
    from napari.components import ViewerModel

    viewer = ViewerModel()
    widget = SimpleCciAnnotatorQWidget(viewer)
    qtbot.addWidget(widget)
    widget._set_project(ProjectStore.initialize(tmp_path / task, task=task))
    image = viewer.add_image(np.zeros((64, 64), dtype=np.uint8), name="source")
    viewer.add_image(np.ones((64, 64), dtype=np.uint8), name="other")
    viewer.layers.selection.active = image
    assert viewer.layers.selection.active is image
    viewer.layers.remove_selected()
    assert image not in viewer.layers
    assert widget._annotation_image_layer is not image


def test_stack_navigation_hides_and_restores_real_napari_labels(tmp_path, qtbot):
    from napari.components import ViewerModel

    viewer = ViewerModel()
    image = viewer.add_image(np.zeros((4, 32, 32), dtype=np.uint8), axis_labels=("Z", "Y", "X"))
    widget = SimpleCciAnnotatorQWidget(viewer)
    qtbot.addWidget(widget)
    widget._set_project(ProjectStore.initialize(tmp_path / "project", task="segment"))
    assert widget._segmentation_layer() is None
    widget._on_reload_labels()
    labels = widget._segmentation_layer()
    assert labels.visible and labels.editable
    start = viewer.dims.current_step[0]
    viewer.dims.set_current_step(0, (start + 1) % 4)
    assert not labels.visible and not labels.editable
    viewer.dims.set_current_step(0, start)
    assert labels.visible and labels.editable
