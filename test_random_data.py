#!/usr/bin/env python3
"""
Test script for Random Data Generation
"""

import sys
import traceback
import matplotlib.pyplot as plt

try:
    # Try importing PyQt
    try:
        from PyQt6.QtWidgets import QApplication
        print("Using PyQt6")
    except ImportError:
        from PyQt5.QtWidgets import QApplication
        print("Using PyQt5")
    
    # Import the main application class
    from minitab_app.modules.ui.main_window import MinitabMainWindow
    
    # Create QApplication
    app = QApplication(sys.argv)
    
    # Create main window
    window = MinitabMainWindow()
    window.show()
    
    # Test instructions
    print("Random Data Generation Test Script")
    print("---------------------------------")
    print("The application should be open now.")
    print("To test random data generation, go to:")
    print("1. Calc > Random Data > Normal")
    print("2. Calc > Random Data > Uniform")
    print("3. Calc > Random Data > Binomial")
    print("4. Calc > Random Data > Poisson")
    print("\nTo test Poisson distribution calculations, go to:")
    print("Calc > Probability Distributions > Poisson")
    
    # Run the application
    sys.exit(app.exec())
    
except Exception as e:
    print(f"Error running application: {e}")
    traceback.print_exc()
    sys.exit(1) 