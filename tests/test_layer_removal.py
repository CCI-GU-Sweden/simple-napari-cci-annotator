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
