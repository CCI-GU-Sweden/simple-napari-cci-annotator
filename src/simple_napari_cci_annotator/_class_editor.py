from __future__ import annotations

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)


class ClassMapDialog(QDialog):
    """Editor for contiguous YOLO class IDs and human-readable names."""

    def __init__(self, classes: dict[int, str], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Edit Project Classes")
        self.resize(440, 360)
        self._table = QTableWidget(0, 2)
        self._table.setHorizontalHeaderLabels(["Class ID", "Class name"])
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.setSelectionBehavior(QTableWidget.SelectRows)
        self._table.setSelectionMode(QTableWidget.SingleSelection)
        for class_id, name in sorted(classes.items()):
            self._append_row(class_id, name)

        add_button = QPushButton("Add Class")
        add_button.clicked.connect(self._on_add)
        remove_button = QPushButton("Remove Last Class")
        remove_button.clicked.connect(self._on_remove_last)
        edit_buttons = QHBoxLayout()
        edit_buttons.addWidget(add_button)
        edit_buttons.addWidget(remove_button)
        edit_buttons.addStretch(1)

        note = QLabel(
            "IDs are stable YOLO indices. Classes can be appended or renamed. "
            "A class used by saved labels cannot be removed."
        )
        note.setWordWrap(True)
        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout()
        layout.addWidget(note)
        layout.addWidget(self._table)
        layout.addLayout(edit_buttons)
        layout.addWidget(buttons)
        self.setLayout(layout)

    def classes(self) -> dict[int, str]:
        return {
            row: self._table.item(row, 1).text().strip()
            for row in range(self._table.rowCount())
        }

    def _append_row(self, class_id: int, name: str) -> None:
        row = self._table.rowCount()
        self._table.insertRow(row)
        id_item = QTableWidgetItem(str(class_id))
        id_item.setFlags(id_item.flags() & ~Qt.ItemIsEditable)
        self._table.setItem(row, 0, id_item)
        self._table.setItem(row, 1, QTableWidgetItem(name))

    def _on_add(self) -> None:
        class_id = self._table.rowCount()
        self._append_row(class_id, f"CLASS_{class_id}")
        self._table.setCurrentCell(class_id, 1)
        self._table.editItem(self._table.item(class_id, 1))

    def _on_remove_last(self) -> None:
        if self._table.rowCount() <= 1:
            QMessageBox.warning(
                self, "Project classes", "At least one class is required."
            )
            return
        self._table.removeRow(self._table.rowCount() - 1)

    def _on_accept(self) -> None:
        classes = self.classes()
        names = list(classes.values())
        if any(not name for name in names):
            QMessageBox.warning(self, "Project classes", "Class names cannot be empty.")
            return
        if len({name.casefold() for name in names}) != len(names):
            QMessageBox.warning(self, "Project classes", "Class names must be unique.")
            return
        self.accept()
