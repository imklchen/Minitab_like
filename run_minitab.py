#!/usr/bin/env python3
"""
Run script for the Minitab-like application
"""

import sys
from PyQt6.QtWidgets import QApplication
from minitab_app.modules.ui.main_window import MinitabMainWindow

def main():
    """Run the application"""
    app = QApplication(sys.argv)
    window = MinitabMainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main() 