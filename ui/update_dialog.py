"""Model update notification and download progress dialogs."""

import sys
import subprocess
import logging
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QProgressBar, QFrame
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont

from ui.styles import COLORS

logger = logging.getLogger(__name__)


class DownloadThread(QThread):
    progress = pyqtSignal(int, int, str)   # current, total, msg
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, updates):
        super().__init__()
        self.updates = updates

    def run(self):
        try:
            from core.model_updater import download_updates
            download_updates(
                self.updates,
                progress_callback=lambda cur, tot, msg: self.progress.emit(cur, tot, msg)
            )
            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))


class UpdateAvailableDialog(QDialog):
    """Popup: shows available model updates and asks user to update."""

    def __init__(self, updates, parent=None):
        super().__init__(parent)
        self.updates = updates
        self.setWindowTitle("모델 업데이트 알림")
        self.setMinimumWidth(420)
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(14)

        # Title
        title = QLabel("새로운 모델 업데이트가 있습니다")
        title.setFont(QFont("Segoe UI", 13, QFont.Weight.Bold))
        title.setStyleSheet(f"color: {COLORS['text']};")
        layout.addWidget(title)

        # Subtitle
        sub = QLabel("다음 모델의 새 버전이 출시되었습니다:")
        sub.setStyleSheet(f"font-size: 12px; color: {COLORS['text2']};")
        layout.addWidget(sub)

        # Model list
        for upd in self.updates:
            row = QFrame()
            row.setObjectName("Card")
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(12, 8, 12, 8)

            name_lbl = QLabel(upd["display"])
            name_lbl.setStyleSheet(f"font-size: 12px; font-weight: 600; color: {COLORS['text']};")
            row_layout.addWidget(name_lbl)
            row_layout.addStretch()

            sha_lbl = QLabel(f"{upd['local']} → {upd['remote']}")
            sha_lbl.setStyleSheet(f"font-size: 11px; color: {COLORS['text3']};")
            row_layout.addWidget(sha_lbl)

            layout.addWidget(row)

        # Note
        note = QLabel("업데이트 후 프로그램이 자동으로 재시작됩니다.")
        note.setStyleSheet(f"font-size: 11px; color: {COLORS['text3']};")
        layout.addWidget(note)

        # Buttons
        btn_row = QHBoxLayout()
        btn_row.addStretch()

        skip_btn = QPushButton("나중에")
        skip_btn.setObjectName("GhostButton")
        skip_btn.setFixedHeight(34)
        skip_btn.clicked.connect(self.reject)
        btn_row.addWidget(skip_btn)

        update_btn = QPushButton("지금 업데이트")
        update_btn.setObjectName("PrimaryButton")
        update_btn.setFixedHeight(34)
        update_btn.clicked.connect(self.accept)
        btn_row.addWidget(update_btn)

        layout.addLayout(btn_row)


class DownloadProgressDialog(QDialog):
    """Shows download progress and restarts app when done."""

    def __init__(self, updates, parent=None):
        super().__init__(parent)
        self.updates = updates
        self.setWindowTitle("모델 다운로드 중")
        self.setMinimumWidth(400)
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint)
        # Block close during download
        self.setWindowFlag(Qt.WindowType.WindowCloseButtonHint, False)
        self._setup_ui()
        self._start_download()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(12)

        title = QLabel("모델 업데이트 다운로드")
        title.setFont(QFont("Segoe UI", 13, QFont.Weight.Bold))
        title.setStyleSheet(f"color: {COLORS['text']};")
        layout.addWidget(title)

        self.status_label = QLabel("준비 중...")
        self.status_label.setStyleSheet(f"font-size: 12px; color: {COLORS['text2']};")
        layout.addWidget(self.status_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, len(self.updates))
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        self.detail_label = QLabel("")
        self.detail_label.setStyleSheet(f"font-size: 11px; color: {COLORS['text3']};")
        layout.addWidget(self.detail_label)

        self.restart_label = QLabel("")
        self.restart_label.setStyleSheet(
            f"font-size: 12px; font-weight: 600; color: {COLORS['green']};")
        self.restart_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.restart_label)

    def _start_download(self):
        self._thread = DownloadThread(self.updates)
        self._thread.progress.connect(self._on_progress)
        self._thread.finished.connect(self._on_done)
        self._thread.error.connect(self._on_error)
        self._thread.start()

    def _on_progress(self, current, total, msg):
        self.progress_bar.setValue(current)
        self.status_label.setText(msg)
        self.detail_label.setText(
            f"{current}/{total} 완료" if current < total else "")

    def _on_done(self):
        self.progress_bar.setValue(len(self.updates))
        self.status_label.setText("다운로드 완료!")
        self.restart_label.setText("잠시 후 프로그램이 재시작됩니다...")
        # Allow close button now
        self.setWindowFlag(Qt.WindowType.WindowCloseButtonHint, True)
        self.show()
        # Restart after short delay
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(2000, self._restart)

    def _on_error(self, msg):
        self.status_label.setText(f"오류: {msg}")
        self.status_label.setStyleSheet(f"font-size: 12px; color: {COLORS['red']};")
        self.setWindowFlag(Qt.WindowType.WindowCloseButtonHint, True)
        self.show()

    def _restart(self):
        """Restart the application."""
        logger.info("Restarting application after model update...")
        subprocess.Popen([sys.executable] + sys.argv)
        sys.exit(0)


def show_update_flow(updates, parent=None):
    """Full update flow: ask → download → restart."""
    dlg = UpdateAvailableDialog(updates, parent)
    if dlg.exec() == QDialog.DialogCode.Accepted:
        progress_dlg = DownloadProgressDialog(updates, parent)
        progress_dlg.exec()
