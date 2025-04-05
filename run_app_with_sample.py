#!/usr/bin/env python3
"""
Run the Minitab-like application with sample data to demonstrate the probability analysis feature
"""

import sys
import traceback
import pandas as pd

try:
    # Try importing PyQt
    try:
        from PyQt6.QtWidgets import QApplication
        print("Using PyQt6")
    except ImportError:
        from PyQt5.QtWidgets import QApplication
        print("Using PyQt5")
    
    # Import the main application class
    from minitab_app.core.app import MinitabApp
    from minitab_app.modules.ui.main_window import MinitabMainWindow
    
    # Create QApplication
    app = QApplication(sys.argv)
    
    # Create main window
    window = MinitabMainWindow()
    
    # Load sample data directly
    try:
        # Load sample data
        sample_data = pd.read_csv("sample_data/sample_data.csv")
        print(f"Loaded sample data with columns: {', '.join(sample_data.columns)}")
        
        # Set data in both the window and the application
        window.data = sample_data
        window.app.data = sample_data
        
        # Update the table with the loaded data
        window.update_table_from_data()
        
        print("Sample data loaded into the application table.")
        print("To test Probability Analysis, go to: Quality > Quality Tools > Probability Analysis")
        print("Then select the 'Height' column when prompted.")
    except Exception as e:
        print(f"Error loading sample data: {e}")
        traceback.print_exc()
    
    # Show the window
    window.show()
    
    # Run the application
    sys.exit(app.exec())
    
except Exception as e:
    print(f"Error running application: {e}")
    traceback.print_exc()
    sys.exit(1) 