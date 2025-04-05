#!/usr/bin/env python3
"""
Main entry point for the Minitab-like application
"""

import sys

# Try importing from PyQt6 first, then fall back to PyQt5 if necessary
try:
    from PyQt6.QtWidgets import QApplication
    using_pyqt6 = True
except ImportError:
    # Fall back to PyQt5
    from PyQt5.QtWidgets import QApplication
    using_pyqt6 = False

from minitab_app.modules.ui.main_window import MinitabMainWindow


def main():
    """Main entry point for the application"""
    # Check if QApplication already exists
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    
    # Create and show the main window
    window = MinitabMainWindow()
    window.show()
    
    # Start the event loop
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
