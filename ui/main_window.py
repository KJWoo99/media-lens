"""Main window with sidebar navigation."""

import logging
import torch
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QStackedWidget, QFrame, QSizePolicy,
    QMessageBox,
)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QFont

from ui.styles import COLORS, SIDEBAR_WIDTH
from ui.components import StatusBar


class SidebarButton(QPushButton):
    """Navigation button in the sidebar."""

    def __init__(self, icon, text, parent=None):
        super().__init__(parent)
        self.setObjectName("SidebarButton")
        self.setFixedHeight(40)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setCheckable(True)

        # Layout: icon + text in separate labels via stylesheet approach
        self.setText(f"  {icon}  {text}")
        font = QFont("Segoe UI", 13)
        font.setWeight(QFont.Weight.Medium)
        self.setFont(font)

    def set_active(self, active):
        self.setProperty("active", "true" if active else "false")
        self.setChecked(active)
        self.style().unpolish(self)
        self.style().polish(self)


class MainWindow(QMainWindow):
    """Main application window with sidebar and stacked pages."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Media Manager")
        self.setMinimumSize(1200, 800)
        self.resize(1440, 900)
        self._engines = []      # engine per page (None for pages without GPU models)
        self._current_page = 0

        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # Content area (sidebar + pages)
        content = QWidget()
        content_layout = QHBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)

        # ── Sidebar ──────────────────────────────────────
        sidebar = QFrame()
        sidebar.setObjectName("Sidebar")
        sidebar.setFixedWidth(SIDEBAR_WIDTH)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(10, 20, 10, 16)
        sidebar_layout.setSpacing(1)

        # App name
        app_title = QLabel("Media Manager")
        app_title.setStyleSheet(f"""
            font-size: 15px;
            font-weight: 800;
            color: {COLORS['text']};
            padding: 4px 6px 18px 6px;
            letter-spacing: -0.3px;
        """)
        sidebar_layout.addWidget(app_title)

        # Section label
        section = QLabel("TOOLS")
        section.setObjectName("SidebarTitle")
        sidebar_layout.addWidget(section)

        # Nav buttons
        self._nav_buttons = []
        self._pages = []
        self.stack = QStackedWidget()

        nav_items = [
            ("\U0001F50D", "Image Search"),
            ("\U0001F50D", "Image Search (beta)"),
            ("\U0001F5BC\uFE0F", "Image Duplicate"),
            ("\U0001F3AC", "Video Duplicate"),
            ("\U0001F5C4\uFE0F", "Cache"),
        ]

        for i, (icon, label) in enumerate(nav_items):
            btn = SidebarButton(icon, label)
            btn.clicked.connect(lambda checked, idx=i: self._switch_page(idx))
            sidebar_layout.addWidget(btn)
            self._nav_buttons.append(btn)

        sidebar_layout.addStretch()

        # Separator
        sep = QFrame()
        sep.setObjectName("SidebarSeparator")
        sep.setFixedHeight(1)
        sidebar_layout.addWidget(sep)

        content_layout.addWidget(sidebar)
        content_layout.addWidget(self.stack, 1)

        root_layout.addWidget(content, 1)

        # Status bar
        self.status_bar = StatusBar()
        root_layout.addWidget(self.status_bar)

        # GPU info
        self._update_gpu_info()

    def add_page(self, page_widget):
        self._pages.append(page_widget)
        self.stack.addWidget(page_widget)

    def set_engines(self, engines):
        """Set engine list matching page order. None for pages without GPU models."""
        self._engines = engines

    def _switch_page(self, idx):
        old = self._current_page
        if old != idx:
            # Warn user if current engine is actively processing
            old_eng = self._engines[old] if old < len(self._engines) else None
            if old_eng and getattr(old_eng, 'is_processing', False):
                msg = QMessageBox(self)
                msg.setWindowTitle("작업 중단")
                msg.setText("현재 페이지의 작업이 중단됩니다.\n그래도 페이지를 이동하시겠습니까?")
                msg.setStandardButtons(
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
                msg.setDefaultButton(QMessageBox.StandardButton.No)
                msg.button(QMessageBox.StandardButton.Yes).setText("이동")
                msg.button(QMessageBox.StandardButton.No).setText("취소")
                if msg.exec() != QMessageBox.StandardButton.Yes:
                    # Restore button state and stay on current page
                    for i, btn in enumerate(self._nav_buttons):
                        btn.set_active(i == old)
                    return
            self._swap_gpu(old, idx)
        self._current_page = idx
        for i, btn in enumerate(self._nav_buttons):
            btn.set_active(i == idx)
        self.stack.setCurrentIndex(idx)

    def _swap_gpu(self, old_idx, new_idx):
        """Offload old engine to CPU, load new engine to GPU.
        Calls page.abort_processing() so the thread sets _stop_requested=True
        and emits stopped (not finished) when it exits."""
        old_eng  = self._engines[old_idx] if old_idx < len(self._engines) else None
        old_page = self._pages[old_idx]   if old_idx < len(self._pages)   else None
        new_eng  = self._engines[new_idx] if new_idx < len(self._engines) else None
        # Always abort the old page regardless of whether it has an engine
        # (e.g. video_duplicate_page has no engine but runs its own threads)
        if old_page and hasattr(old_page, 'abort_processing'):
            old_page.abort_processing()
        if old_eng and getattr(old_eng, 'initialized', False):
            try:
                old_eng.to_cpu()
            except Exception as e:
                logging.getLogger(__name__).warning(f"GPU offload failed: {e}")
        if new_eng and getattr(new_eng, 'initialized', False):
            try:
                new_eng.to_gpu()
            except Exception as e:
                logging.getLogger(__name__).warning(f"GPU load failed: {e}")

    def notify_engine_ready(self, page_idx):
        """Called after an engine finishes initialization.
        Moves the active page's engine to GPU; offloads others to CPU."""
        if page_idx >= len(self._engines):
            return
        eng = self._engines[page_idx]
        if not eng or not getattr(eng, 'initialized', False):
            return
        if page_idx == self._current_page:
            try:
                eng.to_gpu()
            except Exception:
                pass
        else:
            try:
                eng.to_cpu()
            except Exception:
                pass

    def select_page(self, idx):
        self._switch_page(idx)

    def _update_gpu_info(self):
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            self.status_bar.set_engine_info(f"{name}  {mem:.0f} GB")
        else:
            self.status_bar.set_engine_info("CPU Mode")
