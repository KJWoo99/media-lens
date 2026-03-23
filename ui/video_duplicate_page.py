"""Video Duplicate Detection page with improved analysis engine."""

import os
import json
from pathlib import Path

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QProgressBar, QSplitter, QFrame, QTreeWidget, QTreeWidgetItem,
    QTabWidget, QTextEdit, QMessageBox, QFileDialog, QCheckBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QColor, QFont

from ui.styles import COLORS
from ui.components import FolderPicker, SectionHeader


class VideoAnalysisThread(QThread):
    progress = pyqtSignal(str, int, int)
    finished_two = pyqtSignal(list, int, int)
    finished_one = pyqtSignal(list, int)
    error = pyqtSignal(str)

    def __init__(self, analyzer, mode, folder1, folder2=None):
        super().__init__()
        self.analyzer = analyzer
        self.mode = mode
        self.folder1 = folder1
        self.folder2 = folder2

    def run(self):
        try:
            def prog(stage, current, total):
                self.progress.emit(stage, current, total)

            if self.mode == 'two':
                results, c1, c2 = self.analyzer.find_duplicates(
                    self.folder1, self.folder2, prog)
                self.finished_two.emit(results, c1, c2)
            else:
                results, count = self.analyzer.find_duplicates_single_folder(
                    self.folder1, prog)
                self.finished_one.emit(results, count)
        except Exception as e:
            self.error.emit(str(e))


class VideoDuplicatePage(QWidget):
    """Video duplicate detection with 1-folder / 2-folder modes."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.analyzer = None
        self._results = []
        self._active_thread = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 12)
        layout.setSpacing(12)

        # Header
        header = QHBoxLayout()
        header.addWidget(SectionHeader("Video Duplicate Detection"))
        header.addStretch()
        self.ffmpeg_status = QLabel("")
        self.ffmpeg_status.setStyleSheet(f"font-size: 12px; color: {COLORS['text3']};")
        header.addWidget(self.ffmpeg_status)
        layout.addLayout(header)

        # Tab for modes
        self.tabs = QTabWidget()

        # Two folders tab
        two_tab = QWidget()
        two_layout = QVBoxLayout(two_tab)
        two_layout.setContentsMargins(0, 12, 0, 0)
        self.vid_folder1 = FolderPicker(label="Folder 1", placeholder="First video folder...",
                                        config_key="video_folder1")
        two_layout.addWidget(self.vid_folder1)
        self.vid_folder2 = FolderPicker(label="Folder 2", placeholder="Second video folder...",
                                        config_key="video_folder2")
        two_layout.addWidget(self.vid_folder2)
        two_layout.addStretch()
        self.tabs.addTab(two_tab, "Two Folders")

        # Single folder tab
        one_tab = QWidget()
        one_layout = QVBoxLayout(one_tab)
        one_layout.setContentsMargins(0, 12, 0, 0)
        self.vid_folder_single = FolderPicker(label="Folder", placeholder="Select video folder...",
                                              config_key="video_single")
        one_layout.addWidget(self.vid_folder_single)
        one_layout.addStretch()
        self.tabs.addTab(one_tab, "Single Folder")

        layout.addWidget(self.tabs)

        # Controls
        ctrl = QHBoxLayout()

        self.use_cache = QCheckBox("Use cache")
        self.use_cache.setChecked(True)
        self.use_cache.toggled.connect(self._on_cache_toggled)
        ctrl.addWidget(self.use_cache)

        ctrl.addSpacing(8)

        self.cache_info_btn = QPushButton("Cache Info")
        self.cache_info_btn.setObjectName("GhostButton")
        self.cache_info_btn.setFixedHeight(36)
        self.cache_info_btn.clicked.connect(self._show_cache_info)
        ctrl.addWidget(self.cache_info_btn)

        ctrl.addStretch()

        self.scan_btn = QPushButton("Analyze")
        self.scan_btn.setObjectName("PrimaryButton")
        self.scan_btn.setFixedHeight(36)
        self.scan_btn.clicked.connect(self._start)
        ctrl.addWidget(self.scan_btn)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setObjectName("DangerButton")
        self.stop_btn.setFixedHeight(36)
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop)
        ctrl.addWidget(self.stop_btn)

        self.save_btn = QPushButton("Export JSON")
        self.save_btn.setObjectName("GhostButton")
        self.save_btn.setFixedHeight(36)
        self.save_btn.clicked.connect(self._export)
        ctrl.addWidget(self.save_btn)

        layout.addLayout(ctrl)

        # Progress
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)
        self.progress_text = QLabel("")
        self.progress_text.setStyleSheet(
            f"font-size: 11px; color: {COLORS['text3']}; padding: 0 2px;")
        layout.addWidget(self.progress_text)

        # Results: tree + detail
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Result tree
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Match", "Info"])
        self.tree.setColumnWidth(0, 350)
        self.tree.itemClicked.connect(self._on_tree_click)
        self.tree.setMinimumWidth(350)
        splitter.addWidget(self.tree)

        # Detail panel
        detail_frame = QFrame()
        detail_frame.setObjectName("Card")
        detail_layout = QVBoxLayout(detail_frame)
        detail_layout.setContentsMargins(12, 12, 12, 12)

        detail_hdr = QLabel("Details")
        detail_hdr.setStyleSheet(
            f"font-size: 12px; font-weight: 600; color: {COLORS['text2']}; padding: 0 0 4px 0;")
        detail_layout.addWidget(detail_hdr)
        self.detail_text = QTextEdit()
        self.detail_text.setReadOnly(True)
        self.detail_text.setStyleSheet(
            f"font-family: 'Consolas', 'SF Mono', monospace; font-size: 12px;")
        detail_layout.addWidget(self.detail_text)
        splitter.addWidget(detail_frame)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter, 1)

    def set_analyzer(self, analyzer):
        self.analyzer = analyzer
        if analyzer.ffmpeg_ok:
            self.ffmpeg_status.setText("FFmpeg: OK")
            self.ffmpeg_status.setStyleSheet(f"font-size: 12px; color: {COLORS['green']};")
        else:
            self.ffmpeg_status.setText(f"FFmpeg: {analyzer.ffmpeg_msg}")
            self.ffmpeg_status.setStyleSheet(f"font-size: 12px; color: {COLORS['red']};")

    def _on_cache_toggled(self, checked):
        """Sync use_cache checkbox with analyzer."""
        if self.analyzer:
            self.analyzer.set_use_cache(checked)

    def _start(self):
        if not self.analyzer:
            return

        mode = 'two' if self.tabs.currentIndex() == 0 else 'one'

        if mode == 'two':
            f1 = self.vid_folder1.path()
            f2 = self.vid_folder2.path()
            if not f1 or not f2 or not os.path.isdir(f1) or not os.path.isdir(f2):
                QMessageBox.warning(self, "Warning", "Select both folders")
                return
        else:
            f1 = self.vid_folder_single.path()
            f2 = None
            if not f1 or not os.path.isdir(f1):
                QMessageBox.warning(self, "Warning", "Select a folder")
                return

        self.analyzer.is_processing = True
        self.scan_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.tree.clear()
        self._results = []

        thread = VideoAnalysisThread(self.analyzer, mode, f1, f2)
        thread.progress.connect(self._on_progress)
        thread.finished_two.connect(self._on_done_two)
        thread.finished_one.connect(self._on_done_one)
        thread.error.connect(self._on_error)
        # Clean up thread when done
        thread.finished_two.connect(lambda *_: thread.deleteLater())
        thread.finished_one.connect(lambda *_: thread.deleteLater())
        thread.error.connect(lambda *_: thread.deleteLater())
        thread.start()
        self._active_thread = thread

    def abort_processing(self):
        """Stop active processing (called on tab switch)."""
        self._stop()

    def _stop(self):
        if self.analyzer:
            self.analyzer.stop()
        self.stop_btn.setEnabled(False)
        self.scan_btn.setEnabled(True)

    def _on_progress(self, stage, current, total):
        # Weighted progress across stages to prevent 0-100% resets
        stage_weights = {
            'analyze': (0, 80),       # single folder: 0-80%
            'analyze1': (0, 35),      # two folders: folder 1
            'analyze2': (35, 65),     # two folders: folder 2
            'compare': (65, 100),     # comparison phase
        }
        start_pct, end_pct = stage_weights.get(stage, (0, 100))
        if total > 0:
            stage_progress = current / total
            overall_pct = int(start_pct + stage_progress * (end_pct - start_pct))
            self.progress.setValue(overall_pct)

        stage_names = {
            'analyze': 'Analyzing videos',
            'analyze1': 'Analyzing folder 1',
            'analyze2': 'Analyzing folder 2',
            'compare': 'Comparing videos'
        }
        self.progress_text.setText(f"{stage_names.get(stage, stage)}: {current}/{total}")

    def _on_done_two(self, results, c1, c2):
        self._results = results
        self._active_thread = None
        self._finish_ui()
        self.progress_text.setText(f"Done: F1={c1}, F2={c2} videos, {len(results)} matches")
        self._populate_tree(results)

    def _on_done_one(self, results, count):
        self._results = results
        self._active_thread = None
        self._finish_ui()
        self.progress_text.setText(f"Done: {count} videos, {len(results)} matches")
        self._populate_tree(results)

    def _on_error(self, msg):
        self._active_thread = None
        self._finish_ui()
        self.progress_text.setText(f"Error: {msg}")
        QMessageBox.critical(self, "Error", msg)

    def _finish_ui(self):
        self.scan_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress.setVisible(False)

    def _populate_tree(self, results):
        self.tree.clear()
        exact = [r for r in results if r['match_type'] == 'exact_duplicate']
        same = [r for r in results if r['match_type'] == 'same_content']
        partial = [r for r in results if r['match_type'] == 'partial_match']

        categories = [
            ("Exact Duplicates", exact,
             QColor(COLORS['green_light']), QColor(COLORS['green'])),
            ("Same Content", same,
             QColor(COLORS['accent_light']), QColor(COLORS['accent'])),
            ("Partial Matches", partial,
             QColor(COLORS['yellow_light']), QColor(COLORS['yellow'])),
        ]

        bold_font = QFont()
        bold_font.setWeight(QFont.Weight.SemiBold)
        bold_font.setPointSize(11)

        for name, items, bg_color, fg_color in categories:
            if items:
                parent = QTreeWidgetItem(self.tree, [f"{name} ({len(items)})", ""])
                parent.setBackground(0, bg_color)
                parent.setBackground(1, bg_color)
                parent.setForeground(0, fg_color)
                parent.setFont(0, bold_font)
                for i, r in enumerate(items, 1):
                    v1 = Path(r['video1']).name
                    conf = f"{r['confidence'] * 100:.1f}%"
                    child = QTreeWidgetItem(parent, [f"    {v1}", conf])
                    child.setData(0, Qt.ItemDataRole.UserRole, r)

        self.tree.expandAll()

    def _on_tree_click(self, item, column):
        data = item.data(0, Qt.ItemDataRole.UserRole)
        if not data:
            return

        lines = [
            f"{'=' * 50}",
            f"Match Type: {data['match_type']}",
            f"Confidence: {data['confidence'] * 100:.1f}%",
            f"{'=' * 50}",
            "",
            f"Video 1: {data['video1']}",
            f"Video 2: {data['video2']}",
            "",
            "Details:",
        ]
        for k, v in data['details'].items():
            lines.append(f"  {k}: {v}")

        self.detail_text.setText("\n".join(lines))

    def _show_cache_info(self):
        if not self.analyzer or not self.analyzer.cache:
            QMessageBox.information(self, "Cache Info", "Cache is disabled")
            return
        stats = self.analyzer.cache.get_stats()
        msg = (f"Video cache: {stats.get('video_cache', 0)} entries\n"
               f"Image features: {stats.get('image_feature_cache', 0)} entries\n"
               f"CLIP embeddings: {stats.get('clip_cache', 0)} entries\n"
               f"DB size: {stats.get('db_size_mb', 0):.2f} MB")
        QMessageBox.information(self, "Cache Info", msg)

    def _export(self):
        if not self._results:
            QMessageBox.warning(self, "Warning", "No results to export")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export", "video_duplicates.json", "JSON files (*.json)")
        if not path:
            return
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(self._results, f, ensure_ascii=False, indent=2)
            QMessageBox.information(self, "Exported", f"Saved to: {path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Export failed: {e}")
