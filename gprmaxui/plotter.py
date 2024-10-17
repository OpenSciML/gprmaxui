from pathlib import Path

from PySide6.QtGui import QContextMenuEvent
from PySide6.QtWidgets import (
    QVBoxLayout,
    QMenu,
    QFileDialog,
    QDialog,
    QDialogButtonBox,
)
from pyvistaqt import QtInteractor


class PlotterDialog(QDialog):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.resize(800, 600)

        self.plotter = QtInteractor()
        self.setWindowTitle("GPRMax Model Plotter")
        layout = QVBoxLayout(self)
        layout.addWidget(self.plotter)

        # add buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def contextMenuEvent(self, event: QContextMenuEvent) -> None:
        menu = QMenu(self)
        menu.addAction("Save screenshot", self._save_screenshot)
        menu.exec_(event.globalPos())

    def _save_screenshot(self):
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save screenshot", str(Path.home()), "PNG (*.png)"
        )
        if filename:
            self.plotter.screenshot(filename)
