#!/usr/bin/env python3
"""
Direct run script for the Minitab-like application using PyQt6
"""

import sys
import traceback

if __name__ == "__main__":
    try:
        # Make sure PyQt6 is installed
        try:
            from PyQt6 import QtCore
            print(f"PyQt6 version: {QtCore.QT_VERSION_STR}")
        except ImportError as e:
            print(f"ERROR: PyQt6 import issue: {e}")
            print("You may need to install PyQt6 with 'pip install PyQt6'")
            sys.exit(1)
        
        # Import and run the main function
        try:
            from minitab_app.main import main
            sys.exit(main())
        except Exception as e:
            print(f"ERROR: Failed to run the application: {e}")
            print("Detailed error information:")
            traceback.print_exc()
            sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        traceback.print_exc()
        sys.exit(1) 