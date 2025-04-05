"""
Core application module - Application main class
"""

import os
import sys
import pandas as pd
from PyQt6.QtWidgets import (QApplication, QMainWindow, QFileDialog, 
                           QMessageBox, QTableWidget)
from minitab_app.core.file_utils import load_data, save_data, data_to_table

class MinitabApp:
    """
    Core application class that manages application state and operations
    """
    
    def __init__(self):
        """Initialize the application"""
        self.data = pd.DataFrame()
        self.current_file = None
        self.app_name = "Minitab-like Application"
        self.app_version = "1.0.0"
        self.is_modified = False
        self.main_window = None  # Will be set when the main window is created
    
    def load_data_from_file(self, file_path=None):
        """
        Load data from a file
        
        Args:
            file_path (str, optional): Path to the file to load. If None, show file dialog.
            
        Returns:
            bool: True if data was loaded successfully, False otherwise
        """
        try:
            if file_path is None:
                return False
                
            data = load_data(file_path)
            if data is not None:
                self.data = data
                self.current_file = file_path
                self.is_modified = False
                return True
            return False
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def save_data_to_file(self, file_path=None):
        """
        Save data to a file
        
        Args:
            file_path (str, optional): Path to save the file to. If None, use current file.
            
        Returns:
            bool: True if data was saved successfully, False otherwise
        """
        try:
            if file_path is None and self.current_file is None:
                return False
                
            save_path = file_path if file_path else self.current_file
            save_data(self.data, save_path)
            self.current_file = save_path
            self.is_modified = False
            return True
        except Exception as e:
            print(f"Error saving data: {e}")
            return False
    
    def update_data_from_table(self, table_widget):
        """
        Update data from a table widget
        
        Args:
            table_widget (QTableWidget): Table widget to get data from
            
        Returns:
            bool: True if data was updated successfully, False otherwise
        """
        try:
            from minitab_app.core.file_utils import table_to_data
            self.data = table_to_data(table_widget)
            self.is_modified = True
            return True
        except Exception as e:
            print(f"Error updating data from table: {e}")
            return False
    
    def get_project_info(self):
        """
        Get information about the current project
        
        Returns:
            dict: Project information
        """
        return {
            "app_name": self.app_name,
            "app_version": self.app_version,
            "current_file": self.current_file,
            "data_shape": self.data.shape if self.data is not None else None,
            "is_modified": self.is_modified
        }
        
    def create_new_project(self):
        """
        Create a new empty project
        
        Returns:
            bool: True if a new project was created successfully
        """
        if self.is_modified:
            # Should ask for confirmation, but that requires GUI integration
            pass
            
        self.data = pd.DataFrame()
        self.current_file = None
        self.is_modified = False
        return True

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

    def run_probability_analysis(self, column_name):
        """
        Run probability analysis on a specific column
        
        Args:
            column_name (str): The name of the column to analyze
            
        Returns:
            bool: True if analysis was successful, False otherwise
        """
        try:
            if self.data is None or self.data.empty:
                print("No data available for analysis")
                return False
                
            if column_name not in self.data.columns:
                print(f"Column '{column_name}' not found in data")
                return False
                
            # Import necessary libraries
            import numpy as np
            import pandas as pd
            import matplotlib.pyplot as plt
            from scipy import stats
            
            # Get the data
            data = pd.to_numeric(self.data[column_name], errors='coerce').dropna()
            
            if len(data) == 0:
                print(f"No valid numeric data in column '{column_name}'")
                return False
                
            # Calculate mean and standard deviation
            mu = np.mean(data)
            sigma = np.std(data, ddof=1)
            
            # Create figure with two subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # First subplot: Normal distribution with histogram
            x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 100)
            y = stats.norm.pdf(x, mu, sigma)
            
            # Plot histogram and normal curve
            ax1.hist(data, bins='auto', density=True, alpha=0.7, color='skyblue')
            ax1.plot(x, y, 'r-', lw=2, label=f'Normal Dist (μ={mu:.2f}, σ={sigma:.4f})')
            ax1.set_title('Normal Distribution Plot')
            ax1.set_xlabel('Value')
            ax1.set_ylabel('Density')
            ax1.legend()
            
            # Second subplot: QQ plot
            stats.probplot(data, dist="norm", plot=ax2)
            ax2.set_title('Q-Q Plot')
            
            # Add diagonal line
            ax2.get_lines()[0].set_marker('o')
            ax2.get_lines()[0].set_markerfacecolor('skyblue')
            ax2.get_lines()[1].set_color('r')  # Make the diagonal line red
            
            plt.tight_layout()
            plt.show()
            
            # Add statistical test results
            stat, p_value = stats.normaltest(data)
            print(f"Probability Analysis for column: {column_name}")
            print("-" * 40)
            print(f"Mean: {mu:.4f}")
            print(f"Standard Deviation: {sigma:.4f}")
            print(f"Normality Test (D'Agostino's K^2):")
            print(f"Statistic: {stat:.4f}")
            print(f"P-value: {p_value:.4f}")
            print(f"Data {'appears' if p_value > 0.05 else 'does not appear'} to be normally distributed")
            
            return True
            
        except Exception as e:
            print(f"Error in probability analysis: {e}")
            import traceback
            traceback.print_exc()
            return False
