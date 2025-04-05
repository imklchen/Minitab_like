#!/usr/bin/env python3
"""
Integration Test Script for Basic Statistics and Random Data Generation
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
        
        print("\nIntegration Test for Basic Statistics and Random Data Generation")
        print("=============================================================")
        print("\nBasic Statistics Features:")
        print("1. Stat > Basic Statistics > Descriptive Statistics")
        print("2. Stat > Basic Statistics > Correlation")
        print("3. Quality > Quality Tools > Probability Analysis")
        
        print("\nRandom Data Generation Features:")
        print("4. Calc > Random Data > Normal")
        print("5. Calc > Random Data > Uniform")
        print("6. Calc > Random Data > Binomial")
        print("7. Calc > Random Data > Poisson")
        print("8. Calc > Probability Distributions > Poisson")
        
        print("\nTest Instructions:")
        print("- For Descriptive Statistics, select 'Height' column")
        print("- For Correlation, select 'Height' and 'Weight' columns")
        print("- For Probability Analysis, select 'Height' column")
        print("- For random data generation, follow the prompts for each distribution")
        print("- For Poisson distribution, load the sample_data/poisson_test.csv file first")
        
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