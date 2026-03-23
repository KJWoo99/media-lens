"""Cache Management page — view stats and clear cached data."""

import hashlib
import os
import threading

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QMessageBox, QGridLayout,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal

from ui.styles import COLORS
from ui.components import SectionHeader
from core.cache_manager import CacheManager

# Model hashes — must match values in the engine files
_CLIP_MODEL_ID    = "apple/DFN5B-CLIP-ViT-H-14-378"
_SIGLIP2_MODEL_ID = "google/siglip2-base-patch16-224"
_CLIP_HASH    = hashlib.md5(_CLIP_MODEL_ID.encode()).hexdigest()[:8]
_SIGLIP2_HASH = hashlib.md5(_SIGLIP2_MODEL_ID.encode()).hexdigest()[:8]


class _ClearThread(QThread):
    finished = pyqtSignal(str)
    error    = pyqtSignal(str)

    def __init__(self, action, label):
        super().__init__()
        self._action = action   # callable
        self._label  = label

    def run(self):
        try:
            self._action()
            self.finished.emit(self._label)
        except Exception as e:
            self.error.emit(str(e))


class _StatCard(QFrame):
    """Small info card for a single cache stat."""

    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.setObjectName("Card")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(4)

        self._title = QLabel(title.upper())
        self._title.setStyleSheet(
            f"font-size: 10px; font-weight: 700; letter-spacing: 0.8px; color: {COLORS['text3']};")
        layout.addWidget(self._title)

        self._value = QLabel("—")
        self._value.setStyleSheet(
            f"font-size: 20px; font-weight: 700; color: {COLORS['text']};")
        layout.addWidget(self._value)

        self._sub = QLabel("")
        self._sub.setStyleSheet(f"font-size: 11px; color: {COLORS['text3']};")
        layout.addWidget(self._sub)

    def set_value(self, value: str, sub: str = ""):
        self._value.setText(value)
        self._sub.setText(sub)


class CachePage(QWidget):
    """Cache statistics and management."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._active_thread = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 24, 28, 16)
        layout.setSpacing(16)

        # ── Header ───────────────────────────────────────────────────────
        header = QHBoxLayout()
        header.addWidget(SectionHeader("Cache Management"))
        header.addStretch()

        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.setObjectName("GhostButton")
        self.refresh_btn.setFixedHeight(32)
        self.refresh_btn.clicked.connect(self._refresh_stats)
        header.addWidget(self.refresh_btn)
        layout.addLayout(header)

        # ── Stat cards ───────────────────────────────────────────────────
        cards_grid = QGridLayout()
        cards_grid.setSpacing(12)

        self.card_clip    = _StatCard("CLIP")
        self.card_siglip2 = _StatCard("SigLIP2")
        self.card_dinov2  = _StatCard("DINOv2")
        self.card_video   = _StatCard("Video")
        self.card_db      = _StatCard("DB Size")

        cards_grid.addWidget(self.card_clip,    0, 0)
        cards_grid.addWidget(self.card_siglip2, 0, 1)
        cards_grid.addWidget(self.card_dinov2,  0, 2)
        cards_grid.addWidget(self.card_video,   0, 3)
        cards_grid.addWidget(self.card_db,      0, 4)
        layout.addLayout(cards_grid)

        # ── Clear buttons ────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)

        self.btn_clip = QPushButton("Clear CLIP")
        self.btn_clip.setObjectName("GhostButton")
        self.btn_clip.setFixedHeight(34)
        self.btn_clip.clicked.connect(self._clear_clip)
        btn_row.addWidget(self.btn_clip)

        self.btn_siglip2 = QPushButton("Clear SigLIP2")
        self.btn_siglip2.setObjectName("GhostButton")
        self.btn_siglip2.setFixedHeight(34)
        self.btn_siglip2.clicked.connect(self._clear_siglip2)
        btn_row.addWidget(self.btn_siglip2)

        self.btn_dinov2 = QPushButton("Clear DINOv2")
        self.btn_dinov2.setObjectName("GhostButton")
        self.btn_dinov2.setFixedHeight(34)
        self.btn_dinov2.clicked.connect(self._clear_dinov2)
        btn_row.addWidget(self.btn_dinov2)

        self.btn_video = QPushButton("Clear Video")
        self.btn_video.setObjectName("GhostButton")
        self.btn_video.setFixedHeight(34)
        self.btn_video.clicked.connect(self._clear_video)
        btn_row.addWidget(self.btn_video)

        btn_row.addSpacing(16)

        self.btn_invalid = QPushButton("Clear Invalid")
        self.btn_invalid.setObjectName("GhostButton")
        self.btn_invalid.setFixedHeight(34)
        self.btn_invalid.clicked.connect(self._clear_invalid)
        btn_row.addWidget(self.btn_invalid)

        self.btn_all = QPushButton("Clear All")
        self.btn_all.setObjectName("DangerButton")
        self.btn_all.setFixedHeight(34)
        self.btn_all.clicked.connect(self._clear_all)
        btn_row.addWidget(self.btn_all)

        btn_row.addStretch()
        layout.addLayout(btn_row)

        # ── Status label ─────────────────────────────────────────────────
        self.status_lbl = QLabel("")
        self.status_lbl.setStyleSheet(f"font-size: 12px; color: {COLORS['text3']};")
        layout.addWidget(self.status_lbl)

        layout.addStretch()

        # Load stats on first show
        self._refresh_stats()

    def _refresh_stats(self):
        try:
            cache = CacheManager()
            stats = cache.get_stats()
            by_model = cache.get_clip_count_by_model()

            clip_count    = by_model.get(_CLIP_HASH, 0)
            siglip2_count = by_model.get(_SIGLIP2_HASH, 0)
            dinov2_count  = stats.get('image_feature_cache', 0)
            video_count   = stats.get('video_cache', 0)
            db_mb         = stats.get('db_size_mb', 0)

            self.card_clip.set_value(
                f"{clip_count:,}", "embeddings")
            self.card_siglip2.set_value(
                f"{siglip2_count:,}", "embeddings")
            self.card_dinov2.set_value(
                f"{dinov2_count:,}", "features")
            self.card_video.set_value(
                f"{video_count:,}", "hashes")
            self.card_db.set_value(
                f"{db_mb:.1f} MB", "SQLite")

            self.status_lbl.setText("Stats refreshed")
        except Exception as e:
            self.status_lbl.setText(f"Error: {e}")

    def _set_buttons_enabled(self, enabled: bool):
        for btn in (self.btn_clip, self.btn_siglip2, self.btn_dinov2,
                    self.btn_video, self.btn_invalid, self.btn_all, self.refresh_btn):
            btn.setEnabled(enabled)

    def _run_clear(self, label: str, action):
        if self._active_thread and self._active_thread.isRunning():
            return
        self._set_buttons_enabled(False)
        self.status_lbl.setText(f"Clearing {label}...")
        thread = _ClearThread(action, label)
        thread.finished.connect(self._on_clear_done)
        thread.error.connect(self._on_clear_error)
        thread.finished.connect(lambda *_: thread.deleteLater())
        thread.error.connect(lambda *_: thread.deleteLater())
        thread.start()
        self._active_thread = thread

    def _on_clear_done(self, label: str):
        self._active_thread = None
        self._set_buttons_enabled(True)
        self.status_lbl.setText(f"{label} cache cleared")
        self.status_lbl.setStyleSheet(f"font-size: 12px; color: {COLORS['green']};")
        self._refresh_stats()

    def _on_clear_error(self, msg: str):
        self._active_thread = None
        self._set_buttons_enabled(True)
        self.status_lbl.setText(f"Error: {msg}")
        self.status_lbl.setStyleSheet(f"font-size: 12px; color: {COLORS['red']};")

    def _clear_clip(self):
        self._run_clear("CLIP", lambda: CacheManager().clear_clip_cache(_CLIP_HASH))

    def _clear_siglip2(self):
        self._run_clear("SigLIP2", lambda: CacheManager().clear_clip_cache(_SIGLIP2_HASH))

    def _clear_dinov2(self):
        self._run_clear("DINOv2", lambda: CacheManager().clear_image_features())

    def _clear_video(self):
        self._run_clear("Video", lambda: CacheManager().clear_video_cache())

    def _clear_invalid(self):
        def _action():
            n = CacheManager().clear_invalid()
            return n
        self._run_clear("Invalid", lambda: CacheManager().clear_invalid())

    def _clear_all(self):
        reply = QMessageBox.question(
            self, "Confirm Clear All",
            "Delete ALL cached data (CLIP, SigLIP2, DINOv2, Video)?\n"
            "Models will need to re-process images on next use.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            self._run_clear("All", lambda: CacheManager().clear_all())
