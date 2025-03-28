"""
Main entry point for the modularized Minitab-like application.
Generated on: 2025-03-24 21:02:37
"""

import sys
from PyQt6.QtWidgets import QApplication
from src.gui.modular.base import MainWindow

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
