"""Media Manager - Unified media analysis application.

Combines:
- CLIP-based text-to-image search
- DINOv2 image duplicate detection (TensorRT/PyTorch)
- Video duplicate detection (perceptual hash + cosine similarity)
"""

import sys
import os
import logging
import threading
import faulthandler
import traceback

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_LOG_DIR = os.path.dirname(os.path.abspath(__file__))

# Dump C-level stack trace to file on SIGSEGV/SIGABRT (native crash diagnostics)
_crash_log = open(os.path.join(_LOG_DIR, "crash_diag.log"), "w")
faulthandler.enable(file=_crash_log, all_threads=True)

# Catch ALL unhandled Python exceptions (including from Qt slots)
def _excepthook(exc_type, exc_value, exc_tb):
    with open(os.path.join(_LOG_DIR, "unhandled_exception.log"), "a") as f:
        f.write(f"\n=== Unhandled exception ===\n")
        traceback.print_exception(exc_type, exc_value, exc_tb, file=f)
    sys.__excepthook__(exc_type, exc_value, exc_tb)

sys.excepthook = _excepthook

# Catch ALL unhandled thread exceptions (Python 3.8+)
def _thread_excepthook(args):
    with open(os.path.join(_LOG_DIR, "unhandled_exception.log"), "a") as f:
        f.write(f"\n=== Unhandled thread exception (thread: {args.thread}) ===\n")
        traceback.print_exception(args.exc_type, args.exc_value, args.exc_tb, file=f)

threading.excepthook = _thread_excepthook

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
# Suppress noisy HTTP/model download logs
for _quiet in ("httpx", "httpcore", "huggingface_hub", "urllib3", "filelock"):
    logging.getLogger(_quiet).setLevel(logging.WARNING)

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QObject, pyqtSignal, qInstallMessageHandler, QtMsgType


def _qt_msg_handler(mode, _context, message):
    """Filter known benign Qt warnings from stderr."""
    if mode == QtMsgType.QtWarningMsg and "setPixelSize" in message:
        return  # QFont pixel-size-0 on startup (DPI timing issue, harmless)
    if mode == QtMsgType.QtWarningMsg:
        logger.warning("Qt: %s", message)
    elif mode == QtMsgType.QtCriticalMsg:
        logger.error("Qt: %s", message)
    elif mode == QtMsgType.QtFatalMsg:
        # qFatal() called — log to file BEFORE Qt terminates the process
        import traceback
        with open(os.path.join(_LOG_DIR, "unhandled_exception.log"), "a") as _f:
            _f.write(f"\n=== Qt qFatal: {message} ===\n")
            traceback.print_stack(file=_f)
        logger.critical("Qt FATAL: %s", message)


qInstallMessageHandler(_qt_msg_handler)

from ui.styles import GLOBAL_STYLESHEET
from ui.main_window import MainWindow
from ui.image_search_page import ImageSearchPage
from ui.image_duplicate_page import ImageDuplicatePage
from ui.video_duplicate_page import VideoDuplicatePage
from ui.cache_page import CachePage

# Pre-import transformers before any QThreads start to prevent race conditions
# when multiple engines initialize simultaneously (Python's import lock is not thread-safe
# for first-time imports, causing "partially initialized module" errors).
import transformers  # noqa: F401

from core.clip_engine import CLIPEngine
from core.resnet_engine import ResNetEngine
from core.video_analyzer import VideoAnalyzer
from core.cache_manager import CacheManager
from core.model_updater import ModelUpdateChecker

logger = logging.getLogger(__name__)


class _UpdateBridge(QObject):
    """Thread-safe bridge: fires signal on main thread when updates are found."""
    updates_found = pyqtSignal(list)


def main():
    # High DPI support
    os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")

    app = QApplication(sys.argv)
    app.setStyleSheet(GLOBAL_STYLESHEET)
    _update_bridge = _UpdateBridge()

    # Auto-cleanup stale cache entries in background (avoids blocking startup)
    def _run_cache_cleanup():
        try:
            cache = CacheManager()
            deleted = cache.clear_invalid()
            if deleted > 0:
                logger.info(f"Startup cache cleanup: removed {deleted} stale entries")
        except Exception as e:
            logger.warning(f"Cache cleanup failed: {e}")
    threading.Thread(target=_run_cache_cleanup, daemon=True, name="CacheCleanup").start()

    # Create main window
    window = MainWindow()

    # Create engines
    clip   = CLIPEngine()
    resnet = ResNetEngine()
    video  = VideoAnalyzer(use_gpu=True)

    # Create pages
    search_page = ImageSearchPage()
    search_page.set_engine(clip)

    dup_page = ImageDuplicatePage()
    dup_page.set_engine(resnet)

    video_page = VideoDuplicatePage()
    video_page.set_analyzer(video)

    cache_page = CachePage()

    # Add pages (order must match nav_items in main_window.py)
    window.add_page(search_page)
    window.add_page(dup_page)
    window.add_page(video_page)
    window.add_page(cache_page)

    # GPU memory management: only keep active tab's model on GPU
    window.set_engines([clip, resnet, None, None])

    # After each model finishes loading, offload to CPU if not on active tab
    if search_page._active_thread:
        search_page._active_thread.finished.connect(
            lambda ok, _msg: window.notify_engine_ready(0) if ok else None)
    if dup_page._active_thread:
        dup_page._active_thread.finished.connect(
            lambda _backend: window.notify_engine_ready(1))

    # Default to first page
    window.select_page(0)
    window.show()

    # Start background model update check (runs after 5s delay)
    def _on_updates_found(updates):
        try:
            from ui.update_dialog import show_update_flow
            show_update_flow(updates, window)
        except Exception as e:
            logger.warning(f"Update dialog failed: {e}")

    _update_bridge.updates_found.connect(_on_updates_found)
    ModelUpdateChecker(
        on_updates_found=lambda u: _update_bridge.updates_found.emit(u)
    ).start()

    sys.exit(app.exec())


if __name__ == '__main__':
    main()
