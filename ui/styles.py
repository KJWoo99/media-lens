"""Apple-style theme for Media Manager."""

COLORS = {
    'bg':            '#f5f5f7',
    'bg_secondary':  '#e8e8ed',
    'surface':       '#ffffff',
    'surface2':      '#f9f9fb',
    'border':        '#d2d2d7',
    'border_light':  '#e5e5ea',

    'accent':        '#0071e3',
    'accent_hover':  '#0077ed',
    'accent_light':  '#e8f0fd',

    'green':         '#34c759',
    'green_light':   '#e8f8ed',
    'red':           '#ff3b30',
    'red_light':     '#fff0ef',
    'yellow':        '#ff9f0a',
    'yellow_light':  '#fff5e6',
    'purple':        '#af52de',

    'text':          '#1d1d1f',
    'text2':         '#6e6e73',
    'text3':         '#aeaeb2',
    'text_inverse':  '#ffffff',
}

SIDEBAR_WIDTH = 210

GLOBAL_STYLESHEET = f"""
/* ── Global ─────────────────────────────────────────── */
QMainWindow, QDialog {{
    background-color: {COLORS['bg']};
    font-family: 'Segoe UI', 'SF Pro Text', system-ui, sans-serif;
    font-size: 13px;
    color: {COLORS['text']};
}}
QWidget {{
    font-family: 'Segoe UI', 'SF Pro Text', system-ui, sans-serif;
    font-size: 13px;
    color: {COLORS['text']};
}}

/* ── Sidebar ────────────────────────────────────────── */
#Sidebar {{
    background-color: {COLORS['surface']};
    border-right: 1px solid {COLORS['border_light']};
}}
#SidebarButton {{
    background-color: transparent;
    border: none;
    border-radius: 8px;
    padding: 9px 12px;
    text-align: left;
    font-size: 13px;
    font-weight: 500;
    color: {COLORS['text2']};
}}
#SidebarButton:hover {{
    background-color: {COLORS['bg']};
    color: {COLORS['text']};
}}
#SidebarButton[active="true"] {{
    background-color: {COLORS['accent_light']};
    color: {COLORS['accent']};
    font-weight: 600;
}}
#SidebarTitle {{
    font-size: 10px;
    font-weight: 700;
    color: {COLORS['text3']};
    padding: 12px 12px 4px 12px;
    letter-spacing: 0.8px;
}}
#SidebarSeparator {{
    background-color: {COLORS['border_light']};
    max-height: 1px;
    margin: 6px 12px;
}}

/* ── Cards ──────────────────────────────────────────── */
#Card {{
    background-color: {COLORS['surface']};
    border: 1px solid {COLORS['border_light']};
    border-radius: 12px;
}}

/* ── Buttons ────────────────────────────────────────── */
#PrimaryButton {{
    background-color: {COLORS['accent']};
    color: white;
    border: none;
    border-radius: 8px;
    padding: 7px 18px;
    font-size: 13px;
    font-weight: 600;
    min-width: 72px;
}}
#PrimaryButton:hover {{
    background-color: {COLORS['accent_hover']};
}}
#PrimaryButton:pressed {{
    background-color: #005bbf;
}}
#PrimaryButton:disabled {{
    background-color: {COLORS['border_light']};
    color: {COLORS['text3']};
}}
#SecondaryButton {{
    background-color: {COLORS['surface']};
    color: {COLORS['text']};
    border: 1px solid {COLORS['border']};
    border-radius: 8px;
    padding: 7px 14px;
    font-size: 13px;
    font-weight: 500;
}}
#SecondaryButton:hover {{
    background-color: {COLORS['bg']};
    border-color: {COLORS['text3']};
}}
#SecondaryButton:pressed {{
    background-color: {COLORS['bg_secondary']};
}}
#DangerButton {{
    background-color: {COLORS['red_light']};
    color: {COLORS['red']};
    border: none;
    border-radius: 8px;
    padding: 7px 14px;
    font-size: 13px;
    font-weight: 600;
}}
#DangerButton:hover {{
    background-color: #ffe5e3;
}}
#DangerButton:pressed {{
    background-color: #ffd5d3;
}}
#DangerButton:disabled {{
    background-color: {COLORS['bg_secondary']};
    color: {COLORS['text3']};
}}
#GhostButton {{
    background-color: transparent;
    color: {COLORS['text2']};
    border: none;
    border-radius: 8px;
    padding: 7px 14px;
    font-size: 13px;
}}
#GhostButton:hover {{
    background-color: {COLORS['bg']};
    color: {COLORS['text']};
}}

/* ── Inputs ─────────────────────────────────────────── */
QLineEdit {{
    background-color: {COLORS['surface']};
    border: 1px solid {COLORS['border']};
    border-radius: 8px;
    padding: 7px 12px;
    font-size: 13px;
    color: {COLORS['text']};
    selection-background-color: {COLORS['accent_light']};
    selection-color: {COLORS['accent']};
}}
QLineEdit:focus {{
    border: 1.5px solid {COLORS['accent']};
    background-color: {COLORS['surface']};
}}
QLineEdit:read-only {{
    background-color: {COLORS['surface2']};
    color: {COLORS['text2']};
}}
QLineEdit::placeholder {{
    color: {COLORS['text3']};
}}

/* ── Text Edit ──────────────────────────────────────── */
QTextEdit, QPlainTextEdit {{
    background-color: {COLORS['surface2']};
    border: 1px solid {COLORS['border_light']};
    border-radius: 10px;
    padding: 10px;
    font-size: 12px;
    color: {COLORS['text']};
    selection-background-color: {COLORS['accent_light']};
    selection-color: {COLORS['accent']};
}}
QTextEdit:focus, QPlainTextEdit:focus {{
    border: 1.5px solid {COLORS['accent']};
}}

/* ── CheckBox ───────────────────────────────────────── */
QCheckBox {{
    spacing: 8px;
    font-size: 13px;
    color: {COLORS['text']};
}}
QCheckBox::indicator {{
    width: 17px;
    height: 17px;
    border-radius: 5px;
    border: 1.5px solid {COLORS['border']};
    background-color: {COLORS['surface']};
}}
QCheckBox::indicator:hover {{
    border-color: {COLORS['accent']};
    background-color: {COLORS['accent_light']};
}}
QCheckBox::indicator:checked {{
    background-color: {COLORS['accent']};
    border-color: {COLORS['accent']};
}}
QCheckBox::indicator:checked:hover {{
    background-color: {COLORS['accent_hover']};
    border-color: {COLORS['accent_hover']};
}}
QCheckBox::indicator:disabled {{
    background-color: {COLORS['bg_secondary']};
    border-color: {COLORS['border_light']};
}}
QCheckBox:disabled {{
    color: {COLORS['text3']};
}}

/* ── ComboBox ───────────────────────────────────────── */
QComboBox {{
    background-color: {COLORS['surface']};
    border: 1px solid {COLORS['border']};
    border-radius: 8px;
    padding: 6px 12px;
    font-size: 13px;
    color: {COLORS['text']};
    min-width: 80px;
}}
QComboBox:hover {{
    border-color: {COLORS['accent']};
}}
QComboBox:focus {{
    border: 1.5px solid {COLORS['accent']};
}}
QComboBox::drop-down {{
    border: none;
    width: 28px;
    border-radius: 0 8px 8px 0;
}}
QComboBox QAbstractItemView {{
    background-color: {COLORS['surface']};
    border: 1px solid {COLORS['border']};
    border-radius: 8px;
    padding: 4px;
    selection-background-color: {COLORS['accent_light']};
    selection-color: {COLORS['accent']};
    outline: none;
}}

/* ── Progress Bar ───────────────────────────────────── */
QProgressBar {{
    background-color: {COLORS['bg_secondary']};
    border: none;
    border-radius: 4px;
    height: 8px;
    text-align: center;
    font-size: 0px;
    max-height: 8px;
}}
QProgressBar::chunk {{
    background-color: {COLORS['accent']};
    border-radius: 4px;
}}

/* ── Scroll Area ────────────────────────────────────── */
QScrollArea {{
    border: none;
    background-color: transparent;
}}
QScrollBar:vertical {{
    background: transparent;
    width: 7px;
    margin: 0;
}}
QScrollBar::handle:vertical {{
    background: {COLORS['border']};
    border-radius: 3px;
    min-height: 28px;
}}
QScrollBar::handle:vertical:hover {{
    background: {COLORS['text3']};
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}
QScrollBar:horizontal {{
    background: transparent;
    height: 7px;
}}
QScrollBar::handle:horizontal {{
    background: {COLORS['border']};
    border-radius: 3px;
    min-width: 28px;
}}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    width: 0;
}}

/* ── Tree / List Widget ─────────────────────────────── */
QTreeWidget, QListWidget {{
    background-color: {COLORS['surface']};
    border: 1px solid {COLORS['border_light']};
    border-radius: 10px;
    padding: 4px;
    outline: none;
}}
QTreeWidget::item, QListWidget::item {{
    padding: 6px 8px;
    border-radius: 6px;
    color: {COLORS['text']};
}}
QTreeWidget::item:selected, QListWidget::item:selected {{
    background-color: {COLORS['accent_light']};
    color: {COLORS['accent']};
}}
QTreeWidget::item:hover:!selected, QListWidget::item:hover:!selected {{
    background-color: {COLORS['bg']};
}}
QHeaderView::section {{
    background-color: {COLORS['surface2']};
    border: none;
    border-bottom: 1px solid {COLORS['border_light']};
    padding: 7px 8px;
    font-size: 12px;
    font-weight: 600;
    color: {COLORS['text2']};
}}
QHeaderView::section:first {{
    border-radius: 8px 0 0 0;
}}
QHeaderView::section:last {{
    border-radius: 0 8px 0 0;
}}

/* ── Slider ─────────────────────────────────────────── */
QSlider::groove:horizontal {{
    background: {COLORS['bg_secondary']};
    height: 4px;
    border-radius: 2px;
}}
QSlider::handle:horizontal {{
    background: {COLORS['surface']};
    border: 2px solid {COLORS['accent']};
    width: 16px;
    height: 16px;
    margin: -6px 0;
    border-radius: 8px;
}}
QSlider::handle:horizontal:hover {{
    background: {COLORS['accent_light']};
}}
QSlider::sub-page:horizontal {{
    background: {COLORS['accent']};
    border-radius: 2px;
}}

/* ── Tab Widget ─────────────────────────────────────── */
QTabWidget::pane {{
    border: none;
    background-color: transparent;
}}
QTabBar {{
    background-color: {COLORS['bg_secondary']};
    border-radius: 8px;
    padding: 2px;
}}
QTabBar::tab {{
    background-color: transparent;
    border: none;
    border-radius: 6px;
    padding: 6px 18px;
    font-size: 12px;
    font-weight: 500;
    color: {COLORS['text2']};
    min-width: 80px;
}}
QTabBar::tab:selected {{
    background-color: {COLORS['surface']};
    color: {COLORS['text']};
    font-weight: 600;
}}
QTabBar::tab:hover:!selected {{
    color: {COLORS['text']};
    background-color: rgba(0,0,0,0.04);
}}

/* ── Splitter ───────────────────────────────────────── */
QSplitter::handle {{
    background-color: {COLORS['border_light']};
    width: 1px;
    height: 1px;
}}

/* ── Status Bar ─────────────────────────────────────── */
#StatusBar {{
    background-color: {COLORS['surface']};
    border-top: 1px solid {COLORS['border_light']};
    padding: 0 16px;
}}
#StatusLabel {{
    font-size: 11px;
    color: {COLORS['text3']};
}}

/* ── Tool Tip ────────────────────────────────────────── */
QToolTip {{
    background-color: {COLORS['text']};
    color: {COLORS['text_inverse']};
    border: none;
    border-radius: 6px;
    padding: 5px 9px;
    font-size: 12px;
    opacity: 240;
}}

/* ── Message Box ─────────────────────────────────────── */
QMessageBox {{
    background-color: {COLORS['surface']};
}}
QMessageBox QLabel {{
    color: {COLORS['text']};
    font-size: 13px;
}}
QMessageBox QPushButton {{
    background-color: {COLORS['surface']};
    color: {COLORS['accent']};
    border: 1px solid {COLORS['border']};
    border-radius: 8px;
    padding: 6px 18px;
    font-size: 13px;
    font-weight: 500;
    min-width: 70px;
}}
QMessageBox QPushButton:hover {{
    background-color: {COLORS['bg']};
}}
QMessageBox QPushButton:default {{
    background-color: {COLORS['accent']};
    color: white;
    border: none;
    font-weight: 600;
}}
QMessageBox QPushButton:default:hover {{
    background-color: {COLORS['accent_hover']};
}}

/* ── Group Box ──────────────────────────────────────── */
QGroupBox {{
    background-color: {COLORS['surface']};
    border: 1px solid {COLORS['border_light']};
    border-radius: 12px;
    margin-top: 18px;
    padding-top: 20px;
    font-size: 12px;
    font-weight: 600;
    color: {COLORS['text2']};
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 14px;
    padding: 0 6px;
}}
"""
