"""Apple-style UI components."""

import os
from PyQt6.QtWidgets import (
    QWidget, QLabel, QVBoxLayout, QHBoxLayout, QFrame,
    QPushButton, QProgressBar, QFileDialog, QLineEdit,
    QDialog, QScrollArea,
)
from PyQt6.QtCore import Qt, pyqtSignal, QSize, QMimeData
from PyQt6.QtGui import QFont, QPixmap, QDragEnterEvent, QDropEvent

from ui.styles import COLORS


class FolderPicker(QWidget):
    """Folder selection row with label chip, path input, and drag-and-drop.

    Pass config_key to auto-persist the last-used path across sessions.
    """
    folder_changed = pyqtSignal(str)

    def __init__(self, label="", placeholder="Select folder (or drag & drop)...",
                 config_key=None, parent=None):
        super().__init__(parent)
        self._config_key = config_key
        self.setAcceptDrops(True)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        if label:
            chip = QLabel(label)
            chip.setStyleSheet(f"""
                background-color: {COLORS['bg_secondary']};
                color: {COLORS['text2']};
                border-radius: 6px;
                padding: 4px 10px;
                font-size: 11px;
                font-weight: 600;
            """)
            chip.setFixedHeight(28)
            layout.addWidget(chip)

        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText(placeholder)
        self.path_edit.setReadOnly(True)
        self.path_edit.setFixedHeight(34)
        layout.addWidget(self.path_edit, 1)

        btn = QPushButton("Browse")
        btn.setObjectName("SecondaryButton")
        btn.setFixedHeight(34)
        btn.clicked.connect(self._browse)
        layout.addWidget(btn)

        # Restore last-used path if a config key was given
        if config_key:
            from core.config import get_folder
            saved = get_folder(config_key)
            if saved and os.path.isdir(saved):
                self.path_edit.setText(saved)

    def _browse(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder:
            self._set_and_persist(folder)

    def _set_and_persist(self, folder: str):
        self.path_edit.setText(folder)
        if self._config_key:
            from core.config import set_folder
            set_folder(self._config_key, folder)
        self.folder_changed.emit(folder)

    def path(self):
        return self.path_edit.text()

    def set_path(self, p):
        self._set_and_persist(p)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if url.isLocalFile() and os.path.isdir(url.toLocalFile()):
                    event.acceptProposedAction()
                    return

    def dropEvent(self, event: QDropEvent):
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if os.path.isdir(path):
                self._set_and_persist(path)
                break


class StatusBar(QWidget):
    """Bottom status bar showing engine info."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("StatusBar")
        self.setFixedHeight(30)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 0, 16, 0)
        layout.setSpacing(0)

        self.status_label = QLabel("Ready")
        self.status_label.setObjectName("StatusLabel")
        layout.addWidget(self.status_label)

        layout.addStretch()

        self.engine_label = QLabel("")
        self.engine_label.setObjectName("StatusLabel")
        layout.addWidget(self.engine_label)

    def set_status(self, text):
        self.status_label.setText(text)

    def set_engine_info(self, text):
        self.engine_label.setText(text)


class SectionHeader(QLabel):
    """Page section title."""

    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setStyleSheet(f"""
            font-size: 17px;
            font-weight: 700;
            color: {COLORS['text']};
            letter-spacing: -0.3px;
            padding: 0;
            margin: 0;
        """)


class Badge(QLabel):
    """Small colored badge / tag."""

    def __init__(self, text="", color="accent", parent=None):
        super().__init__(text, parent)
        bg = COLORS.get(f'{color}_light', COLORS['accent_light'])
        fg = COLORS.get(color, COLORS['accent'])
        self.setStyleSheet(f"""
            background-color: {bg};
            color: {fg};
            border-radius: 10px;
            padding: 2px 10px;
            font-size: 11px;
            font-weight: 700;
        """)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)


class InfoCard(QFrame):
    """Small key-value info card."""

    def __init__(self, key="", parent=None):
        super().__init__(parent)
        self.setObjectName("Card")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(3)

        self.key_label = QLabel(key)
        self.key_label.setStyleSheet(f"""
            font-size: 10px;
            font-weight: 700;
            color: {COLORS['text3']};
            letter-spacing: 0.5px;
        """)
        layout.addWidget(self.key_label)

        self.value_label = QLabel("\u2014")
        self.value_label.setStyleSheet(f"""
            font-size: 12px;
            font-weight: 600;
            color: {COLORS['text']};
        """)
        layout.addWidget(self.value_label)

    def set_value(self, text, color=None):
        c = COLORS.get(color, COLORS['text']) if color else COLORS['text']
        self.value_label.setText(text)
        self.value_label.setStyleSheet(f"font-size: 12px; font-weight: 600; color: {c};")


class ImageThumbnail(QFrame):
    """Clickable image thumbnail card."""
    clicked = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("Card")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._path = ""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(120, 120)
        self.image_label.setStyleSheet("background-color: transparent;")
        layout.addWidget(self.image_label)

        self.name_label = QLabel()
        self.name_label.setStyleSheet(f"font-size: 11px; color: {COLORS['text2']};")
        self.name_label.setWordWrap(True)
        self.name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.name_label)

        self.score_label = QLabel()
        self.score_label.setStyleSheet(
            f"font-size: 11px; font-weight: 600; color: {COLORS['accent']};")
        self.score_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.score_label)

    def set_image(self, path, name="", score=None, thumb_size=150):
        self._path = path
        pixmap = QPixmap(path)
        if not pixmap.isNull():
            pixmap = pixmap.scaled(
                thumb_size, thumb_size,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.image_label.setPixmap(pixmap)
        else:
            self.image_label.setText("—")
        self.name_label.setText(name or "")
        if score is not None:
            self.score_label.setText(f"{score:.4f}")

    def mousePressEvent(self, event):
        if self._path:
            self.clicked.emit(self._path)


class ImagePreviewDialog(QDialog):
    """Full-size image preview dialog."""

    def __init__(self, path, parent=None):
        super().__init__(parent)
        self.setWindowTitle(os.path.basename(path))
        self.setMinimumSize(700, 550)
        layout = QVBoxLayout(self)

        label = QLabel()
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pixmap = QPixmap(path)
        if not pixmap.isNull():
            screen = self.screen().availableGeometry()
            max_w = int(screen.width() * 0.7)
            max_h = int(screen.height() * 0.7)
            pixmap = pixmap.scaled(max_w, max_h,
                                    Qt.AspectRatioMode.KeepAspectRatio,
                                    Qt.TransformationMode.SmoothTransformation)
            label.setPixmap(pixmap)
            self.resize(pixmap.width() + 40, pixmap.height() + 80)

        scroll = QScrollArea()
        scroll.setWidget(label)
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll)

        path_label = QLabel(path)
        path_label.setStyleSheet(f"font-size: 11px; color: {COLORS['text3']}; padding: 4px;")
        path_label.setWordWrap(True)
        layout.addWidget(path_label)
