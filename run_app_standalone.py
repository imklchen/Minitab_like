#!/usr/bin/env python3
"""
Standalone script to run the Minitab-like application
This bypasses the minitab_app.main module to avoid event loop issues
"""

import sys
import traceback

try:
    # Try importing from PyQt6 first
    try:
        from PyQt6.QtWidgets import QApplication
        print("Using PyQt6")
    except ImportError:
        # Fall back to PyQt5
        from PyQt5.QtWidgets import QApplication
        print("Using PyQt5")
    
    # Import the main window class
    try:
        from minitab_app.modules.ui.main_window import MinitabMainWindow
    except ImportError as e:
        print(f"Error importing main window: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Create application and window
    app = QApplication(sys.argv)
    window = MinitabMainWindow()
    window.show()
    
    # Run the application
    sys.exit(app.exec())
    
except Exception as e:
    print(f"Error running application: {e}")
    traceback.print_exc()
    sys.exit(1) 