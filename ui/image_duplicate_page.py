"""Image Duplicate Detection page - DINOv2 based comparison with heatmap."""

import os
import logging
import numpy as np
import cv2
from PIL import Image, ImageChops, ImageStat
from send2trash import send2trash

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QProgressBar, QSplitter, QFrame, QListWidget, QListWidgetItem,
    QTabWidget, QSlider, QMessageBox, QFileDialog, QSizePolicy, QCheckBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QPixmap, QImage, QFont

from ui.styles import COLORS
from ui.components import FolderPicker, SectionHeader, InfoCard

logger = logging.getLogger(__name__)


class DetectionThread(QThread):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(list)
    error = pyqtSignal(str)

    def __init__(self, engine, mode, folder1, folder2=None, threshold=0.95, recursive=False):
        super().__init__()
        self.engine = engine
        self.mode = mode
        self.folder1 = folder1
        self.folder2 = folder2
        self.threshold = threshold
        self.recursive = recursive

    def run(self):
        try:
            if self.mode == 'one':
                def prog(done, total, extracted, total_imgs, eta_str=""):
                    pct = int(done / max(total, 1) * 100)
                    self.progress.emit(pct, f"Extracting: {extracted}/{total_imgs}{eta_str}")
                def status(pct, msg):
                    self.progress.emit(pct, msg)
                results = self.engine.find_duplicates_one_folder(
                    self.folder1, self.threshold, prog, status, recursive=self.recursive)
            else:
                def prog(pct, msg):
                    self.progress.emit(pct, msg)
                results = self.engine.find_duplicates_two_folders(
                    self.folder1, self.folder2, self.threshold, prog,
                    cancel_check=lambda: not self.engine.is_processing,
                    recursive=self.recursive)
            self.finished.emit(results)
        except Exception as e:
            self.error.emit(str(e))


class InitThread(QThread):
    status   = pyqtSignal(str)  # intermediate progress messages
    finished = pyqtSignal(str)

    def __init__(self, engine):
        super().__init__()
        self.engine = engine

    def run(self):
        try:
            backend = self.engine.initialize(progress_callback=lambda msg: self.status.emit(msg))
            self.finished.emit(backend)
        except Exception as e:
            logger.error(f"Engine init failed: {e}", exc_info=True)
            self.finished.emit("PyTorch")


class ImageDuplicatePage(QWidget):
    """Image duplicate detection with 1-folder / 2-folder modes."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.engine = None
        self._results = []
        self._filtered = []
        self._active_thread = None
        self._img_refs = {}  # prevent GC of pixmaps/QImages
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 24, 28, 16)
        layout.setSpacing(12)

        # Header
        header = QHBoxLayout()
        header.addWidget(SectionHeader("Image Duplicate Detection"))
        header.addStretch()
        self.backend_label = QLabel("Initializing...")
        self.backend_label.setStyleSheet(f"font-size: 12px; color: {COLORS['text3']};")
        header.addWidget(self.backend_label)
        layout.addLayout(header)

        # Tab for 1-folder / 2-folder
        self.tabs = QTabWidget()

        # ── One Folder tab ──
        one_tab = QWidget()
        one_layout = QVBoxLayout(one_tab)
        one_layout.setContentsMargins(0, 12, 0, 0)
        self.folder1_picker = FolderPicker(label="Folder", placeholder="Select folder to scan...",
                                            config_key="imgdup_single")
        one_layout.addWidget(self.folder1_picker)
        one_layout.addStretch()
        self.tabs.addTab(one_tab, "Single Folder")

        # ── Two Folder tab ──
        two_tab = QWidget()
        two_layout = QVBoxLayout(two_tab)
        two_layout.setContentsMargins(0, 12, 0, 0)
        self.folderA_picker = FolderPicker(label="Folder 1", placeholder="First folder...",
                                            config_key="imgdup_folder1")
        two_layout.addWidget(self.folderA_picker)
        self.folderB_picker = FolderPicker(label="Folder 2", placeholder="Second folder...",
                                            config_key="imgdup_folder2")
        two_layout.addWidget(self.folderB_picker)
        two_layout.addStretch()
        self.tabs.addTab(two_tab, "Two Folders")

        layout.addWidget(self.tabs)

        # Controls row
        ctrl = QHBoxLayout()
        ctrl.setSpacing(8)

        thr_lbl = QLabel("Threshold")
        thr_lbl.setStyleSheet(f"font-size: 12px; color: {COLORS['text2']};")
        ctrl.addWidget(thr_lbl)
        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_slider.setRange(80, 100)
        self.threshold_slider.setValue(95)
        self.threshold_slider.setFixedWidth(130)
        ctrl.addWidget(self.threshold_slider)
        self.threshold_label = QLabel("0.95")
        self.threshold_label.setStyleSheet(
            f"font-size: 12px; font-weight: 600; color: {COLORS['text']}; min-width: 32px;")
        self.threshold_slider.valueChanged.connect(
            lambda v: self.threshold_label.setText(f"{v / 100:.2f}"))
        ctrl.addWidget(self.threshold_label)

        ctrl.addSpacing(8)

        self.recursive_check = QCheckBox("Subfolders")
        ctrl.addWidget(self.recursive_check)

        ctrl.addSpacing(4)

        self.scan_btn = QPushButton("Scan")
        self.scan_btn.setObjectName("PrimaryButton")
        self.scan_btn.setFixedHeight(34)
        self.scan_btn.clicked.connect(self._start_scan)
        ctrl.addWidget(self.scan_btn)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setObjectName("DangerButton")
        self.stop_btn.setFixedHeight(34)
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop_scan)
        ctrl.addWidget(self.stop_btn)

        self.save_btn = QPushButton("Export")
        self.save_btn.setObjectName("GhostButton")
        self.save_btn.setFixedHeight(34)
        self.save_btn.clicked.connect(self._export)
        ctrl.addWidget(self.save_btn)

        ctrl.addStretch()
        self.result_count = QLabel("")
        self.result_count.setStyleSheet(
            f"font-size: 12px; font-weight: 600; color: {COLORS['accent']};")
        ctrl.addWidget(self.result_count)
        layout.addLayout(ctrl)

        # Progress
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        self.progress_text = QLabel("")
        self.progress_text.setStyleSheet(
            f"font-size: 11px; color: {COLORS['text3']}; padding: 0 2px;")
        self.progress_text.setVisible(False)
        layout.addWidget(self.progress_text)

        # Main content: list + compare panel
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Result list
        self.result_list = QListWidget()
        self.result_list.setMinimumWidth(280)
        self.result_list.setMaximumWidth(400)
        self.result_list.currentRowChanged.connect(self._on_select)
        splitter.addWidget(self.result_list)

        # Compare panel
        compare = QFrame()
        compare.setObjectName("Card")
        compare_layout = QVBoxLayout(compare)
        compare_layout.setContentsMargins(12, 12, 12, 12)
        compare_layout.setSpacing(8)

        # Image pair
        pair_row = QHBoxLayout()

        # Left image
        left_col = QVBoxLayout()
        left_header = QHBoxLayout()
        left_header.addWidget(QLabel("Original"))
        self.del_left_btn = QPushButton("Delete")
        self.del_left_btn.setObjectName("DangerButton")
        self.del_left_btn.setFixedHeight(28)
        self.del_left_btn.clicked.connect(lambda: self._delete_image('left'))
        left_header.addWidget(self.del_left_btn)
        left_col.addLayout(left_header)
        self.img1_label = QLabel()
        self.img1_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.img1_label.setMinimumSize(200, 200)
        self.img1_label.setStyleSheet(f"background-color: {COLORS['surface2']}; border-radius: 8px;")
        left_col.addWidget(self.img1_label, 1)
        self.img1_meta = QLabel()
        self.img1_meta.setStyleSheet(f"font-size: 11px; color: {COLORS['text3']};")
        left_col.addWidget(self.img1_meta)
        pair_row.addLayout(left_col, 1)

        # Right image
        right_col = QVBoxLayout()
        right_header = QHBoxLayout()
        right_header.addWidget(QLabel("Duplicate"))
        self.del_right_btn = QPushButton("Delete")
        self.del_right_btn.setObjectName("DangerButton")
        self.del_right_btn.setFixedHeight(28)
        self.del_right_btn.clicked.connect(lambda: self._delete_image('right'))
        right_header.addWidget(self.del_right_btn)
        right_col.addLayout(right_header)
        self.img2_label = QLabel()
        self.img2_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.img2_label.setMinimumSize(200, 200)
        self.img2_label.setStyleSheet(f"background-color: {COLORS['surface2']}; border-radius: 8px;")
        right_col.addWidget(self.img2_label, 1)
        self.img2_meta = QLabel()
        self.img2_meta.setStyleSheet(f"font-size: 11px; color: {COLORS['text3']};")
        right_col.addWidget(self.img2_meta)
        pair_row.addLayout(right_col, 1)

        compare_layout.addLayout(pair_row, 1)

        # Heatmap
        heat_header = QLabel("Difference Heatmap")
        heat_header.setStyleSheet(
            f"font-size: 11px; font-weight: 600; color: {COLORS['text3']}; "
            f"letter-spacing: 0.4px;")
        compare_layout.addWidget(heat_header)
        self.heatmap_label = QLabel()
        self.heatmap_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.heatmap_label.setMinimumHeight(120)
        self.heatmap_label.setStyleSheet(f"background-color: {COLORS['surface2']}; border-radius: 8px;")
        compare_layout.addWidget(self.heatmap_label)

        # Analysis cards
        cards_row = QHBoxLayout()
        self.card_format = InfoCard("FORMAT")
        self.card_res = InfoCard("RESOLUTION")
        self.card_size = InfoCard("FILE SIZE")
        self.card_pixel = InfoCard("PIXEL DIFF")
        cards_row.addWidget(self.card_format)
        cards_row.addWidget(self.card_res)
        cards_row.addWidget(self.card_size)
        cards_row.addWidget(self.card_pixel)
        compare_layout.addLayout(cards_row)

        splitter.addWidget(compare)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        layout.addWidget(splitter, 1)

    def set_engine(self, engine):
        self.engine = engine
        self.scan_btn.setEnabled(False)
        thread = InitThread(engine)
        thread.status.connect(self._on_init_status)
        thread.finished.connect(self._on_init)
        thread.finished.connect(lambda: self._cleanup_thread(thread))
        thread.start()
        self._active_thread = thread

    def _cleanup_thread(self, thread):
        """Clean up finished thread reference."""
        thread.deleteLater()
        if self._active_thread is thread:
            self._active_thread = None

    def _on_init_status(self, msg):
        self.backend_label.setText(msg)

    def _on_init(self, backend):
        self.backend_label.setText(f"DINOv2-Base - {backend} - {'CUDA' if self.engine.device.type == 'cuda' else 'CPU'}")
        self.backend_label.setStyleSheet(f"font-size: 12px; color: {COLORS['green']};")
        self.scan_btn.setEnabled(True)

    def _start_scan(self):
        if not self.engine or not self.engine.initialized:
            return

        threshold = self.threshold_slider.value() / 100.0
        mode = 'one' if self.tabs.currentIndex() == 0 else 'two'

        if mode == 'one':
            folder = self.folder1_picker.path()
            if not folder or not os.path.isdir(folder):
                QMessageBox.warning(self, "Warning", "Select a folder first")
                return
            folder2 = None
        else:
            folder = self.folderA_picker.path()
            folder2 = self.folderB_picker.path()
            if not folder or not folder2 or not os.path.isdir(folder) or not os.path.isdir(folder2):
                QMessageBox.warning(self, "Warning", "Select both folders")
                return

        # Clean up any previous thread still running
        if self._active_thread is not None:
            self.engine.is_processing = False
            try:
                self._active_thread.progress.disconnect()
                self._active_thread.finished.disconnect()
                self._active_thread.error.disconnect()
            except (TypeError, RuntimeError):
                pass
            self._active_thread = None

        self.engine.is_processing = True
        self.scan_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress.setVisible(True)
        self.progress_text.setVisible(True)
        self.progress.setValue(0)
        self.progress_text.setText("Starting...")
        self.result_list.clear()
        self._results = []
        self._filtered = []

        recursive = self.recursive_check.isChecked()
        thread = DetectionThread(self.engine, mode, folder, folder2, threshold, recursive)
        thread.progress.connect(self._on_progress)
        thread.finished.connect(self._on_done)
        thread.error.connect(self._on_error)
        thread.finished.connect(lambda: self._cleanup_thread(thread))
        thread.start()
        self._active_thread = thread

    def abort_processing(self):
        """Stop active processing (called on tab switch)."""
        self._stop_scan()

    def _stop_scan(self):
        if self.engine:
            self.engine.stop()
        # Disconnect old thread signals to prevent stale callbacks
        if self._active_thread is not None:
            try:
                self._active_thread.progress.disconnect()
                self._active_thread.finished.disconnect()
                self._active_thread.error.disconnect()
            except (TypeError, RuntimeError):
                pass
            self._active_thread = None
        self.stop_btn.setEnabled(False)
        self.scan_btn.setEnabled(True)
        self.progress.setVisible(False)
        self.progress_text.setVisible(False)

    def _on_progress(self, pct, msg):
        self.progress.setValue(pct)
        self.progress_text.setText(msg)

    def _on_done(self, results):
        self._results = results
        self._filtered = sorted(results, key=lambda x: x['similarity'], reverse=True)
        self.scan_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress.setVisible(False)
        self.progress_text.setVisible(False)

        self.result_list.clear()
        for d in self._filtered:
            self.result_list.addItem(
                f"{d['img1_name']}  |  {d['img2_name']}  ({d['similarity']:.4f})")

        self.result_count.setText(f"{len(results)} pairs")
        if results:
            self.result_list.setCurrentRow(0)

    def _on_error(self, msg):
        self.scan_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress.setVisible(False)
        self.progress_text.setText(f"Error: {msg}")

    def _on_select(self, row):
        if row < 0 or row >= len(self._filtered):
            return
        d = self._filtered[row]
        self._show_pair(d['img1_path'], d['img2_path'])

    def _show_pair(self, path1, path2):
        """Display image pair with heatmap and analysis."""
        # Load images
        for path, label, meta_label, side in [
            (path1, self.img1_label, self.img1_meta, 'left'),
            (path2, self.img2_label, self.img2_meta, 'right')
        ]:
            pixmap = QPixmap(path)
            if not pixmap.isNull():
                w = label.width() - 10
                h = label.height() - 10
                scaled = pixmap.scaled(max(w, 150), max(h, 150),
                                        Qt.AspectRatioMode.KeepAspectRatio,
                                        Qt.TransformationMode.SmoothTransformation)
                label.setPixmap(scaled)
                self._img_refs[side] = scaled
            meta_label.setText(f"{os.path.basename(path)}  ({pixmap.width()}x{pixmap.height()})")

        # Heatmap
        try:
            img1 = Image.open(path1).convert('RGB').resize((224, 224))
            img2 = Image.open(path2).convert('RGB').resize((224, 224))
            diff = ImageChops.difference(img1, img2)
            stat = ImageStat.Stat(diff)
            pixel_diff = sum(stat.mean) / (255 * 3)

            enhanced = diff.point(lambda p: min(255, p * 5))
            arr = np.array(enhanced.convert('L'))
            heatmap = cv2.applyColorMap(arr, cv2.COLORMAP_JET)
            heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

            h, w, ch = heatmap_rgb.shape
            # .copy() prevents GC of underlying numpy buffer
            qimg = QImage(heatmap_rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()
            self._img_refs['heatmap_qimg'] = qimg  # prevent GC
            hm_pixmap = QPixmap.fromImage(qimg)
            hw = self.heatmap_label.width() - 10
            hh = self.heatmap_label.height() - 10
            hm_pixmap = hm_pixmap.scaled(max(hw, 200), max(hh, 100),
                                          Qt.AspectRatioMode.KeepAspectRatio,
                                          Qt.TransformationMode.SmoothTransformation)
            self.heatmap_label.setPixmap(hm_pixmap)
            self._img_refs['heatmap'] = hm_pixmap

            # Analysis cards
            ext1 = os.path.splitext(path1)[1].upper().lstrip('.')
            ext2 = os.path.splitext(path2)[1].upper().lstrip('.')
            self.card_format.set_value(f"{ext1} / {ext2}")

            # Use PIL for resolution (fast header-only)
            with Image.open(path1) as p1_img:
                p1w, p1h = p1_img.size
            with Image.open(path2) as p2_img:
                p2w, p2h = p2_img.size
            same_res = (p1w == p2w and p1h == p2h)
            self.card_res.set_value(
                f"{p1w}x{p1h} / {p2w}x{p2h}"
                + (" (Same)" if same_res else ""),
                'green' if same_res else 'yellow')

            sz1 = os.path.getsize(path1)
            sz2 = os.path.getsize(path2)
            diff_pct = abs(sz1 - sz2) / max(sz1, sz2) * 100
            self.card_size.set_value(
                f"{sz1 / 1024:.1f}KB / {sz2 / 1024:.1f}KB ({diff_pct:.1f}%)",
                'green' if diff_pct < 1 else 'yellow')

            self.card_pixel.set_value(
                f"{pixel_diff * 100:.3f}%" + (" (Identical)" if pixel_diff < 0.0001 else ""),
                'green' if pixel_diff < 0.0001 else 'yellow')

        except Exception as e:
            logger.warning(f"Heatmap error: {e}")

    def _delete_image(self, side):
        if not self._filtered:
            return
        row = self.result_list.currentRow()
        if row < 0 or row >= len(self._filtered):
            return
        d = self._filtered[row]
        path = d['img1_path'] if side == 'left' else d['img2_path']
        name = os.path.basename(path)

        if not os.path.exists(path):
            QMessageBox.warning(self, "Warning", "File not found")
            return

        reply = QMessageBox.question(self, "Confirm Delete",
                                      f"Move '{name}' to Recycle Bin?",
                                      QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            try:
                send2trash(path)
                # Remove this pair from results and refresh list
                self._filtered.pop(row)
                self._results = [r for r in self._results
                                 if r['img1_path'] != path and r['img2_path'] != path]
                self.result_list.takeItem(row)
                self.result_count.setText(f"{len(self._filtered)} pairs")
                # Select next item
                if self._filtered:
                    next_row = min(row, len(self._filtered) - 1)
                    self.result_list.setCurrentRow(next_row)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to delete: {e}")

    def _export(self):
        if not self._filtered:
            QMessageBox.warning(self, "Warning", "No results to export")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Results", "duplicate_results.txt", "Text files (*.txt)")
        if not path:
            return
        try:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(f"Image Duplicate Detection Results\n{'=' * 60}\n")
                f.write(f"Total pairs: {len(self._filtered)}\n\n")
                for i, d in enumerate(self._filtered, 1):
                    f.write(f"[{i}] {d['img1_name']} | {d['img2_name']}\n")
                    f.write(f"    Similarity: {d['similarity']:.5f}\n")
                    f.write(f"    Same resolution: {'Yes' if d.get('same_resolution') else 'No'}\n\n")
            QMessageBox.information(self, "Exported", f"Saved to: {path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Export failed: {e}")
