"""
Core application class
"""

import sys
import pandas as pd
from PyQt6.QtWidgets import QApplication

from simplified_app.ui.main_window import MinitabMainWindow

class MinitabApp:
    """
    Main application class that initializes and manages the application
    """
    def __init__(self):
        self.app = QApplication(sys.argv)
        self.main_window = None
        self.data = pd.DataFrame()
        self.current_file = None
    
    def run(self):
        """Initialize and run the application"""
        self.main_window = MinitabMainWindow(self)
        self.main_window.show()
        return self.app.exec()
    
    def get_data(self):
        """Return the current data"""
        return self.data
    
    def set_data(self, data):
        """Set the application data"""
        self.data = data
        # Update UI if main window exists
        if self.main_window:
            self.main_window.update_table()
    
    def get_current_file(self):
        """Return the current file path"""
        return self.current_file
    
    def set_current_file(self, file_path):
        """Set the current file path"""
        self.current_file = file_path
        # Update window title if main window exists
        if self.main_window:
            self.main_window.update_title()
