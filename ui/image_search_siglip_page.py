"""Image Search (beta) — MetaCLIP2 text-to-image search with TensorRT image encoder.

No translation model required: 300+ languages natively supported by MetaCLIP2.
"""

import os
import time
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QProgressBar, QScrollArea, QGridLayout,
    QFrame, QSlider, QCheckBox, QMenu, QApplication, QFileDialog,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QPixmap

from ui.styles import COLORS
from ui.components import FolderPicker, SectionHeader, ImagePreviewDialog


# ── Worker threads ─────────────────────────────────────────────────────────

class ModelLoadThread(QThread):
    status   = pyqtSignal(str)        # intermediate progress messages
    finished = pyqtSignal(bool, str)  # (ok, message)

    def __init__(self, engine):
        super().__init__()
        self.engine = engine

    def run(self):
        import logging as _log
        import traceback as _tb
        try:
            self.engine.initialize(progress_callback=lambda msg: self.status.emit(msg))
            self.finished.emit(True, self.engine.backend_info)
        except BaseException as e:
            _log.getLogger(__name__).error(
                f"SigLIP2 ModelLoadThread crashed: {type(e).__name__}: {e}\n"
                + _tb.format_exc())
            try:
                self.finished.emit(False, f"{type(e).__name__}: {e}")
            except Exception:
                pass


class FolderLoadThread(QThread):
    status  = pyqtSignal(int, str)
    finished = pyqtSignal(int)
    stopped  = pyqtSignal()
    error    = pyqtSignal(str)

    def __init__(self, engine, folder, recursive=False):
        super().__init__()
        self.engine    = engine
        self.folder    = folder
        self.recursive = recursive
        self._stop_requested = False

    def run(self):
        try:
            count = self.engine.process_folder(
                self.folder,
                status_callback=lambda pct, msg: self.status.emit(pct, msg),
                recursive=self.recursive
            )
            if self._stop_requested:
                self.stopped.emit()
            else:
                self.finished.emit(count)
        except Exception as e:
            self.error.emit(str(e))


class SearchThread(QThread):
    result = pyqtSignal(list, float)
    error  = pyqtSignal(str)

    def __init__(self, engine, query, folder, top_k, recursive=False, query_image=None):
        super().__init__()
        self.engine      = engine
        self.query       = query
        self.folder      = folder
        self.top_k       = top_k
        self.recursive   = recursive
        self.query_image = query_image  # None = text mode, path = image mode

    def run(self):
        try:
            start = time.time()
            if self.query_image:
                results = self.engine.search_by_image(
                    self.query_image, self.folder, self.top_k,
                    recursive=self.recursive)
            else:
                results = self.engine.search(self.query, self.folder, self.top_k,
                                             recursive=self.recursive)
            elapsed = time.time() - start
            self.result.emit(results, elapsed)
        except Exception as e:
            self.error.emit(str(e))


# ── Main page ──────────────────────────────────────────────────────────────

class ImageSearchSigLIPPage(QWidget):
    """SigLIP2 text-to-image search — no translation model, native multilingual."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.engine = None
        self._active_thread = None
        self._last_results = []
        self._last_elapsed = 0.0
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 24, 28, 16)
        layout.setSpacing(14)

        # ── Header ───────────────────────────────────────
        header = QHBoxLayout()

        title = SectionHeader("Image Search")
        header.addWidget(title)

        # "(beta)" badge
        beta_badge = QLabel("beta")
        beta_badge.setStyleSheet(f"""
            background-color: {COLORS['purple']};
            color: white;
            border-radius: 8px;
            padding: 2px 8px;
            font-size: 10px;
            font-weight: 700;
            letter-spacing: 0.5px;
        """)
        beta_badge.setFixedHeight(20)
        header.addWidget(beta_badge)
        header.addSpacing(6)

        # native-language note
        lang_note = QLabel("300+ languages — no translation needed")
        lang_note.setStyleSheet(
            f"font-size: 11px; color: {COLORS['text3']}; padding: 0 4px;")
        header.addWidget(lang_note)

        header.addStretch()

        self.model_status = QLabel("Model not loaded")
        self.model_status.setStyleSheet(
            f"font-size: 12px; color: {COLORS['text3']};")
        header.addWidget(self.model_status)
        layout.addLayout(header)

        # ── Folder picker ─────────────────────────────────
        self.folder_picker = FolderPicker(
            label="Folder", placeholder="Select image folder...",
            config_key="siglip2_search")
        layout.addWidget(self.folder_picker)

        # ── Load row ──────────────────────────────────────
        load_row = QHBoxLayout()
        load_row.setSpacing(8)

        self.load_btn = QPushButton("Load Folder")
        self.load_btn.setObjectName("PrimaryButton")
        self.load_btn.setFixedHeight(34)
        self.load_btn.clicked.connect(self._load_folder)
        load_row.addWidget(self.load_btn)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setObjectName("DangerButton")
        self.stop_btn.setFixedHeight(34)
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop_load)
        load_row.addWidget(self.stop_btn)

        self.recursive_check = QCheckBox("Subfolders")
        load_row.addWidget(self.recursive_check)

        load_row.addSpacing(4)
        self.load_status = QLabel("")
        self.load_status.setStyleSheet(
            f"font-size: 12px; color: {COLORS['text2']};")
        load_row.addWidget(self.load_status, 1)
        layout.addLayout(load_row)

        # ── Progress ──────────────────────────────────────
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        self.progress_text = QLabel("")
        self.progress_text.setStyleSheet(
            f"font-size: 11px; color: {COLORS['text3']}; padding: 0 2px;")
        self.progress_text.setVisible(False)
        layout.addWidget(self.progress_text)

        # ── Search bar ────────────────────────────────────
        search_row = QHBoxLayout()
        search_row.setSpacing(8)

        # Mode toggle
        self.image_mode_btn = QPushButton("Image")
        self.image_mode_btn.setCheckable(True)
        self.image_mode_btn.setFixedHeight(38)
        self.image_mode_btn.setFixedWidth(68)
        self.image_mode_btn.setObjectName("GhostButton")
        self.image_mode_btn.toggled.connect(self._toggle_search_mode)
        search_row.addWidget(self.image_mode_btn)

        # Text input (visible by default)
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText(
            "Search images by text (Korean / English)...")
        self.search_input.setFixedHeight(38)
        self.search_input.setStyleSheet(
            "font-size: 14px; border-radius: 10px; padding: 8px 14px;")
        self.search_input.returnPressed.connect(self._search)
        search_row.addWidget(self.search_input, 1)

        # Image query widgets (hidden by default)
        self.image_path_edit = QLineEdit()
        self.image_path_edit.setPlaceholderText("Select query image...")
        self.image_path_edit.setFixedHeight(38)
        self.image_path_edit.setReadOnly(True)
        self.image_path_edit.setVisible(False)
        search_row.addWidget(self.image_path_edit, 1)

        self.browse_image_btn = QPushButton("Browse...")
        self.browse_image_btn.setObjectName("GhostButton")
        self.browse_image_btn.setFixedHeight(38)
        self.browse_image_btn.setVisible(False)
        self.browse_image_btn.clicked.connect(self._browse_query_image)
        search_row.addWidget(self.browse_image_btn)

        self.search_btn = QPushButton("Search")
        self.search_btn.setObjectName("PrimaryButton")
        self.search_btn.setFixedHeight(38)
        self.search_btn.clicked.connect(self._search)
        search_row.addWidget(self.search_btn)
        layout.addLayout(search_row)

        # ── Options row ───────────────────────────────────
        opt_row = QHBoxLayout()
        opt_row.setSpacing(8)

        results_lbl = QLabel("Results")
        results_lbl.setStyleSheet(f"font-size: 12px; color: {COLORS['text2']};")
        opt_row.addWidget(results_lbl)

        self.count_slider = QSlider(Qt.Orientation.Horizontal)
        self.count_slider.setRange(5, 50)
        self.count_slider.setValue(10)
        self.count_slider.setFixedWidth(130)
        opt_row.addWidget(self.count_slider)

        self.count_label = QLabel("10")
        self.count_label.setStyleSheet(
            f"font-size: 12px; font-weight: 600; color: {COLORS['text']}; min-width: 22px;")
        self.count_slider.valueChanged.connect(
            lambda v: self.count_label.setText(str(v)))
        opt_row.addWidget(self.count_label)

        opt_row.addSpacing(16)

        min_score_lbl = QLabel("Min score")
        min_score_lbl.setStyleSheet(f"font-size: 12px; color: {COLORS['text2']};")
        opt_row.addWidget(min_score_lbl)

        self.score_slider = QSlider(Qt.Orientation.Horizontal)
        self.score_slider.setRange(0, 100)
        self.score_slider.setValue(0)
        self.score_slider.setFixedWidth(110)
        opt_row.addWidget(self.score_slider)

        self.score_label = QLabel("0%")
        self.score_label.setStyleSheet(
            f"font-size: 12px; font-weight: 600; color: {COLORS['text']}; min-width: 30px;")
        self.score_slider.valueChanged.connect(self._on_score_filter_changed)
        opt_row.addWidget(self.score_label)

        opt_row.addStretch()
        self.search_info = QLabel("")
        self.search_info.setStyleSheet(
            f"font-size: 12px; color: {COLORS['text2']};")
        opt_row.addWidget(self.search_info)
        layout.addLayout(opt_row)

        # ── Results grid ──────────────────────────────────
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("background-color: transparent;")

        self.results_container = QWidget()
        self.results_grid = QGridLayout(self.results_container)
        self.results_grid.setSpacing(12)
        scroll.setWidget(self.results_container)
        layout.addWidget(scroll, 1)

    # ── Engine wiring ──────────────────────────────────────────────────

    def set_engine(self, engine):
        self.engine = engine
        self.model_status.setText("Loading MetaCLIP2...")
        self.model_status.setStyleSheet(
            f"font-size: 12px; color: {COLORS['yellow']};")

        thread = ModelLoadThread(engine)
        thread.status.connect(self._on_model_status)
        thread.finished.connect(self._on_model_loaded)
        thread.finished.connect(lambda *_: thread.deleteLater())
        thread.start()
        self._active_thread = thread

    def _on_model_status(self, msg):
        self.model_status.setText(msg)

    def _on_model_loaded(self, ok, msg):
        self._active_thread = None
        if ok:
            self.model_status.setText(msg)
            self.model_status.setStyleSheet(
                f"font-size: 12px; color: {COLORS['green']};")
        else:
            self.model_status.setText(f"Error: {msg}")
            self.model_status.setStyleSheet(
                f"font-size: 12px; color: {COLORS['red']};")

    # ── Folder loading ─────────────────────────────────────────────────

    def _load_folder(self):
        folder = self.folder_picker.path()
        if not folder or not os.path.isdir(folder):
            return
        if not self.engine or not self.engine.initialized:
            self.load_status.setText("Wait for model to load first")
            return

        if self._active_thread and self._active_thread.isRunning():
            try:
                self._active_thread.status.disconnect()
                self._active_thread.finished.disconnect()
                self._active_thread.error.disconnect()
            except Exception:
                pass
            self.engine.is_processing = False

        self.load_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.progress_text.setVisible(True)
        self.progress_text.setText("Starting...")
        self.load_status.setText("")
        self.load_status.setStyleSheet(
            f"font-size: 12px; color: {COLORS['text2']};")

        recursive = self.recursive_check.isChecked()
        thread = FolderLoadThread(self.engine, folder, recursive)
        thread.status.connect(self._on_load_status)
        thread.finished.connect(self._on_load_done)
        thread.stopped.connect(self._on_load_stopped)
        thread.error.connect(self._on_load_error)
        thread.finished.connect(lambda *_: thread.deleteLater())
        thread.stopped.connect(lambda: thread.deleteLater())
        thread.error.connect(lambda *_: thread.deleteLater())
        thread.start()
        self._active_thread = thread

    def abort_processing(self):
        """Stop active processing (called on tab switch)."""
        self._stop_load()

    def _stop_load(self):
        if self._active_thread:
            self._active_thread._stop_requested = True
        if self.engine:
            self.engine.is_processing = False
        self.stop_btn.setEnabled(False)
        self.load_status.setText("Stopping... (finishing current batch)")

    def _on_load_status(self, pct, msg):
        self.progress.setValue(pct)
        self.progress_text.setText(msg)

    def _on_load_stopped(self):
        self._active_thread = None
        self.load_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress.setVisible(False)
        self.progress_text.setVisible(False)
        self.load_status.setText("Stopped")
        self.load_status.setStyleSheet(
            f"font-size: 12px; color: {COLORS['text3']};")

    def _on_load_done(self, count):
        self._active_thread = None
        self.load_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress.setVisible(False)
        self.progress_text.setVisible(False)
        self.load_status.setText(f"Ready: {count} images loaded")
        self.load_status.setStyleSheet(
            f"font-size: 12px; color: {COLORS['green']};")

    def _on_load_error(self, msg):
        self._active_thread = None
        self.load_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress.setVisible(False)
        self.progress_text.setVisible(False)
        self.load_status.setText(f"Error: {msg}")
        self.load_status.setStyleSheet(
            f"font-size: 12px; color: {COLORS['red']};")

    # ── Search ─────────────────────────────────────────────────────────

    def _toggle_search_mode(self, image_mode: bool):
        self.search_input.setVisible(not image_mode)
        self.image_path_edit.setVisible(image_mode)
        self.browse_image_btn.setVisible(image_mode)

    def _browse_query_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Query Image", self.folder_picker.path() or "",
            "Images (*.jpg *.jpeg *.png *.bmp *.gif *.webp *.tif *.tiff)")
        if path:
            self.image_path_edit.setText(path)

    def _search(self):
        folder = self.folder_picker.path()
        if not folder or not os.path.isdir(folder):
            self.search_info.setText("Select a folder first")
            return
        if not self.engine or not self.engine.initialized:
            self.search_info.setText("Model not ready")
            return

        image_mode = self.image_mode_btn.isChecked()
        if image_mode:
            query_image = self.image_path_edit.text().strip()
            if not query_image or not os.path.isfile(query_image):
                self.search_info.setText("Select a query image first")
                return
            query = None
        else:
            query = self.search_input.text().strip()
            if not query:
                return
            query_image = None

        self.search_btn.setEnabled(False)
        self.search_info.setText("Searching...")
        self._clear_results()

        top_k     = self.count_slider.value()
        recursive = self.recursive_check.isChecked()
        thread = SearchThread(self.engine, query, folder, top_k, recursive,
                              query_image=query_image)
        thread.result.connect(self._show_results)
        thread.error.connect(self._search_error)
        thread.result.connect(lambda *_: thread.deleteLater())
        thread.error.connect(lambda *_: thread.deleteLater())
        thread.start()
        self._active_thread = thread

    def _show_results(self, results, elapsed):
        self._active_thread = None
        self.search_btn.setEnabled(True)
        self._last_results = results
        self._last_elapsed = elapsed
        self._apply_score_filter()

    def _on_score_filter_changed(self, value):
        self.score_label.setText(f"{value}%")
        self._apply_score_filter()

    def _apply_score_filter(self):
        threshold = self.score_slider.value() / 100.0
        filtered = [(p, s) for p, s in self._last_results if s >= threshold]
        elapsed = self._last_elapsed

        self._clear_results()
        if not filtered:
            self.search_info.setText(
                f"No results ≥{int(threshold * 100)}% ({elapsed:.2f}s)"
                if self._last_results else f"No results ({elapsed:.2f}s)")
            return

        self.search_info.setText(f"{len(filtered)} results in {elapsed:.2f}s")

        cols = 4
        for i, (path, score) in enumerate(filtered):
            card = QFrame()
            card.setObjectName("Card")
            card.setCursor(Qt.CursorShape.PointingHandCursor)
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(10, 10, 10, 10)
            card_layout.setSpacing(6)

            # Thumbnail
            img_label = QLabel()
            img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            img_label.setFixedSize(160, 160)
            img_label.setStyleSheet(
                f"background-color: {COLORS['surface2']}; border-radius: 8px;")
            pixmap = QPixmap(path)
            if not pixmap.isNull():
                pixmap = pixmap.scaled(
                    156, 156,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation)
                img_label.setPixmap(pixmap)
            card_layout.addWidget(img_label)

            # Score badge — MetaCLIP2 uses contrastive loss (same scale as CLIP)
            score_pct   = int(score * 100)
            score_color = (COLORS['green']  if score > 0.85
                           else COLORS['accent'] if score > 0.70
                           else COLORS['text2'])
            score_lbl = QLabel(f"{score_pct}%")
            score_lbl.setStyleSheet(
                f"font-size: 12px; font-weight: 700; color: {score_color};")
            score_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            card_layout.addWidget(score_lbl)

            # File name
            name = QLabel(os.path.basename(path))
            name.setStyleSheet(f"font-size: 10px; color: {COLORS['text3']};")
            name.setWordWrap(True)
            name.setAlignment(Qt.AlignmentFlag.AlignCenter)
            card_layout.addWidget(name)

            card.mousePressEvent = self._make_card_handler(path)
            self.results_grid.addWidget(card, i // cols, i % cols)

    def _make_card_handler(self, path):
        def handler(event):
            if event.button() == Qt.MouseButton.RightButton:
                menu = QMenu(self)
                act_explorer = menu.addAction("Open in Explorer")
                act_copy = menu.addAction("Copy Path")
                action = menu.exec(event.globalPosition().toPoint())
                if action == act_explorer:
                    os.startfile(os.path.dirname(path))
                elif action == act_copy:
                    QApplication.clipboard().setText(path)
            elif event.button() == Qt.MouseButton.LeftButton:
                self._preview(path)
        return handler

    def _search_error(self, msg):
        self._active_thread = None
        self.search_btn.setEnabled(True)
        self.search_info.setText(f"Error: {msg}")

    def _preview(self, path):
        dlg = ImagePreviewDialog(path, self)
        dlg.exec()

    def _clear_results(self):
        while self.results_grid.count():
            item = self.results_grid.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
