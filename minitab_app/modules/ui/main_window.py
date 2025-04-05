"""
Main window module - Main application window
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import traceback
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTableWidget,
                           QTextEdit, QPushButton, QFileDialog, QMessageBox,
                           QMenu, QMenuBar, QStatusBar, QToolBar, QDialog,
                           QLabel, QComboBox, QInputDialog, QTableWidgetItem)
from PyQt6.QtGui import QIcon, QAction
from PyQt6.QtCore import Qt, QSize

from minitab_app.core.app import MinitabApp
from minitab_app.core.file_utils import data_to_table, table_to_data
from minitab_app.modules.stats import basic_stats
from minitab_app.modules.quality import control_charts
from minitab_app.modules.stats import doe
from minitab_app.modules.quality.capability import process_capability, probabilityAnalysis
from minitab_app.modules.quality import msa
from minitab_app.modules.sixsigma import dmaic
from minitab_app.modules.sixsigma.metrics import dpmoCalculator, sigmaLevelCalc, yieldAnalysis
from minitab_app.modules.stats.advanced_stats import hypothesis_testing, regression_analysis
from minitab_app.modules.stats.chi_square import chi_square_tests


class MinitabMainWindow(QMainWindow):
    """
    Main application window
    """
    
    def __init__(self):
        """Initialize the main window"""
        super().__init__()
        
        # Initialize application core
        self.app = MinitabApp()
        self.data = pd.DataFrame(columns=[f"C{i+1}" for i in range(10)])  # Initial columns like original
        self.current_file = None  # Track current file
        
        # Setup window properties
        self.setWindowTitle("Custom Minitab-Like Tool")
        self.resize(900, 600)
        
        # Setup UI components
        self.initUI()
        
    def initUI(self):
        """Initialize the user interface"""
        # Create table with 50 rows and 10 columns initially
        self.table = QTableWidget(50, 10)
        self.table.setHorizontalHeaderLabels(self.data.columns)
        
        # Create session window for output
        self.sessionWindow = QTextEdit()
        self.sessionWindow.setReadOnly(True)
        
        # Add welcome message to session window
        self.sessionWindow.setText("Welcome to Custom Minitab-Like Tool!\nReady for statistical analysis and quality improvement.")
        
        # Create vertical layout for table and session window
        layout = QVBoxLayout()
        layout.addWidget(self.table, 1)  # Equal space for table
        layout.addWidget(self.sessionWindow, 1)  # Equal space for session window
        
        # Set layout to container widget
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)
        
        # Create menu bar
        self.createMenuBar()
        
        # Connect table change signals
        self.table.itemChanged.connect(self.onTableItemChanged)
    
    def createMenuBar(self):
        """Create application menus"""
        menuBar = self.menuBar()
        
        # Create main menus - match original structure
        fileMenu = menuBar.addMenu("File")
        statMenu = menuBar.addMenu("Stat")
        qualityMenu = menuBar.addMenu("Quality")
        sixSigmaMenu = menuBar.addMenu("Six Sigma")
        calcMenu = menuBar.addMenu("Calc")
        
        # File Menu
        fileMenu.addAction(self.makeAction("Open", self.open_file))
        fileMenu.addAction(self.makeAction("Save", self.save_file))
        fileMenu.addAction(self.makeAction("Save As", self.save_file_as))
        fileMenu.addSeparator()
        fileMenu.addAction(self.makeAction("Clear Table Data", self.clear_table))
        fileMenu.addSeparator()
        fileMenu.addAction(self.makeAction("Exit", self.close))
        
        # Stat Menu
        basicStatMenu = QMenu("Basic Statistics", self)
        statMenu.addMenu(basicStatMenu)
        basicStatMenu.addAction(self.makeAction("Descriptive Statistics", self.show_descriptive_stats))
        basicStatMenu.addAction(self.makeAction("Correlation", self.show_correlation))
        
        # Advanced Statistics submenu
        advancedStatMenu = QMenu("Advanced Statistics", self)
        statMenu.addMenu(advancedStatMenu)
        advancedStatMenu.addAction(self.makeAction("Hypothesis Testing", self.show_hypothesis_testing))
        advancedStatMenu.addAction(self.makeAction("ANOVA", self.placeholder_function))
        advancedStatMenu.addAction(self.makeAction("Regression Analysis", self.show_regression_analysis))
        advancedStatMenu.addAction(self.makeAction("Chi-Square Tests", self.show_chi_square_tests))
        
        # Add Design of Experiments directly to Stat menu
        statMenu.addAction(self.makeAction("Create DOE", self.show_create_doe))
        statMenu.addAction(self.makeAction("Analyze DOE", self.show_analyze_doe))
        
        # Quality Menu
        # Add Quality Tools submenu
        qualityToolsMenu = QMenu("Quality Tools", self)
        qualityMenu.addMenu(qualityToolsMenu)
        
        # Move Probability Analysis under Quality Tools
        qualityToolsMenu.addAction(self.makeAction("Probability Analysis", self.show_probability_analysis))
        qualityToolsMenu.addAction(self.makeAction("Process Capability", self.show_process_capability))
        
        # Control Charts submenu
        controlChartsMenu = QMenu("Control Charts", self)
        qualityMenu.addMenu(controlChartsMenu)
        controlChartsMenu.addAction(self.makeAction("X-bar R Chart", self.show_xbar_r_chart))
        controlChartsMenu.addAction(self.makeAction("Individual Chart", self.show_individual_chart))
        controlChartsMenu.addAction(self.makeAction("Moving Range Chart", self.placeholder_function))
        
        # Measurement System Analysis submenu
        msaMenu = QMenu("Measurement System Analysis", self)
        qualityMenu.addMenu(msaMenu)
        msaMenu.addAction(self.makeAction("Gage R&R Study", self.show_gage_rr_study))
        msaMenu.addAction(self.makeAction("Linearity Study", self.show_linearity_study))
        msaMenu.addAction(self.makeAction("Bias Study", self.show_bias_study))
        msaMenu.addAction(self.makeAction("Stability Study", self.show_stability_study))
        
        # Six Sigma Menu
        dmaicMenu = QMenu("DMAIC Tools", self)
        sixSigmaMenu.addMenu(dmaicMenu)
        dmaicMenu.addAction(self.makeAction("Pareto Chart", self.show_pareto_chart))
        dmaicMenu.addAction(self.makeAction("Fishbone Diagram", self.show_fishbone_diagram))
        dmaicMenu.addAction(self.makeAction("FMEA Template", self.show_fmea_template))
        
        metrics = QMenu("Six Sigma Metrics", self)
        sixSigmaMenu.addMenu(metrics)
        metrics.addAction(self.makeAction("DPMO Calculator", self.show_dpmo_calculator))
        metrics.addAction(self.makeAction("Sigma Level Calculator", self.show_sigma_level_calculator))
        metrics.addAction(self.makeAction("Process Yield Analysis", self.show_yield_analysis))
        
        # Calc Menu
        randomDataMenu = QMenu("Random Data", self)
        calcMenu.addMenu(randomDataMenu)
        randomDataMenu.addAction(self.makeAction("Normal", lambda: self.show_random_data("normal")))
        randomDataMenu.addAction(self.makeAction("Uniform", lambda: self.show_random_data("uniform")))
        randomDataMenu.addAction(self.makeAction("Binomial", lambda: self.show_random_data("binomial")))
        randomDataMenu.addAction(self.makeAction("Poisson", lambda: self.show_random_data("poisson")))
        
        # Add probability distributions submenu
        probDistMenu = QMenu("Probability Distributions", self)
        calcMenu.addMenu(probDistMenu)
        probDistMenu.addAction(self.makeAction("Poisson", self.show_poisson_distribution))
    
    def makeAction(self, name, func):
        """Create a QAction for menus"""
        action = QAction(name, self)
        action.triggered.connect(func)
        return action
    
    def onTableItemChanged(self, item):
        """Handle table cell changes"""
        self.app.is_modified = True
    
    def placeholder_function(self, *args):
        """Placeholder for unimplemented functions"""
        QMessageBox.information(self, "Information", "This function is not yet implemented in the modular version.")
    
    def load_data_from_table(self):
        """Load data from table widget to DataFrame"""
        try:
            # Get dimensions
            rows = self.table.rowCount()
            cols = self.table.columnCount()
            
            # Check if there's any data in the table
            has_data = False
            for i in range(rows):
                for j in range(cols):
                    item = self.table.item(i, j)
                    if item and item.text().strip():
                        has_data = True
                        break
                if has_data:
                    break
            
            if not has_data:
                self.data = pd.DataFrame()
                return False
            
            # Get headers
            headers = []
            for j in range(cols):
                header = self.table.horizontalHeaderItem(j)
                headers.append(header.text() if header else f"Column_{j+1}")
            
            # Create empty DataFrame
            data = []
            for i in range(rows):
                row_data = []
                row_has_data = False
                for j in range(cols):
                    item = self.table.item(i, j)
                    # Handle empty or null cells
                    if item is None or not item.text().strip():
                        row_data.append(np.nan)
                    else:
                        row_has_data = True
                        try:
                            # Try to convert to numeric
                            value = float(item.text().strip())
                        except ValueError:
                            # If not numeric, keep as string
                            value = item.text().strip()
                        row_data.append(value)
                if row_has_data:  # Only add rows that have data
                    data.append(row_data)
            
            if not data:  # If no rows with data were found
                self.data = pd.DataFrame()
                return False
            
            # Create DataFrame
            self.data = pd.DataFrame(data, columns=headers)
            
            # Try to convert numeric columns
            for col in self.data.columns:
                try:
                    # Try to convert to numeric, but only if it doesn't create any NaN values
                    numeric_series = pd.to_numeric(self.data[col], errors='coerce')
                    if not numeric_series.isna().any():
                        self.data[col] = numeric_series
                except:
                    continue
                    
            return True
                    
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error loading data from table: {str(e)}")
            self.data = pd.DataFrame()
            return False
    
    def update_table_from_data(self):
        """Update table widget from DataFrame"""
        try:
            if self.data is not None and not self.data.empty:
                # Clear existing table
                self.table.clear()
                
                # Set table dimensions
                rows, cols = self.data.shape
                # Ensure at least 50 rows
                display_rows = max(rows, 50)
                self.table.setRowCount(display_rows)
                self.table.setColumnCount(cols)
                
                # Set headers
                self.table.setHorizontalHeaderLabels(self.data.columns.astype(str))
                
                # Populate table with data
                for i in range(rows):
                    for j in range(cols):
                        value = str(self.data.iloc[i, j])
                        if pd.isna(self.data.iloc[i, j]):
                            value = ''
                        item = QTableWidgetItem(value)
                        self.table.setItem(i, j, item)
                
                # Adjust column widths
                self.table.resizeColumnsToContents()
            else:
                # If data is empty, create a default empty table
                self.table.clear()
                self.table.setRowCount(50)
                self.table.setColumnCount(10)
                self.table.setHorizontalHeaderLabels([f"C{i+1}" for i in range(10)])
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error updating table: {str(e)}")
    
    def selectColumnDialog(self, title="Select Column", columns=None):
        """
        Display column selection dialog
        
        Args:
            title (str): Dialog title
            columns (list): List of column names to display
            
        Returns:
            str or None: Selected column name or None if cancelled
        """
        if columns is None:
            # Load data from table and get all columns
            self.load_data_from_table()
            columns = self.data.columns.tolist()
        
        if not columns:
            QMessageBox.warning(self, "Warning", "No columns available")
            return None
        
        col, ok = QInputDialog.getItem(
            self, title, "Choose column:", columns, 0, False
        )
        
        if ok and col:
            return col
        return None
    
    def open_file(self):
        """Open a data file"""
        try:
            filename, _ = QFileDialog.getOpenFileName(
                self,
                "Open File",
                "",
                "CSV Files (*.csv);;Excel Files (*.xlsx);;All Files (*.*)"
            )
            
            if not filename:
                return  # User cancelled
            
            try:
                if filename.endswith('.xlsx'):
                    try:
                        self.data = pd.read_excel(filename)
                    except ImportError:
                        QMessageBox.critical(
                            self,
                            "Error",
                            "Excel support requires the openpyxl package.\n"
                            "Please install it using: pip install openpyxl"
                        )
                        return
                else:
                    self.data = pd.read_csv(filename)
                
                self.current_file = filename  # Update current file
                self.app.current_file = filename  # Update app's current file
                self.update_table_from_data()
                self.sessionWindow.append(f"Data loaded from file: {filename}")
                
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Error",
                    f"Failed to open file:\n{str(e)}"
                )
                self.sessionWindow.append(f"Error opening file: {str(e)}")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error selecting file: {str(e)}")
            self.sessionWindow.append(f"Error selecting file: {str(e)}")
    
    def save_file(self):
        """Save data to current file or prompt for new file"""
        try:
            # Try to load data from table first
            if self.load_data_from_table() == False:
                QMessageBox.warning(self, "Warning", "No data to save. Please enter some data in the table first.")
                return

            if self.current_file:
                try:
                    if self.current_file.endswith('.xlsx'):
                        try:
                            self.data.to_excel(self.current_file, index=False)
                            self.sessionWindow.append(f"Data saved to Excel file: {self.current_file}")
                        except ImportError:
                            QMessageBox.critical(
                                self,
                                "Error",
                                "Excel support requires the openpyxl package.\n"
                                "Please install it using: pip install openpyxl"
                            )
                            return
                    else:
                        # Default to CSV if no extension or other extension
                        if not self.current_file.endswith('.csv'):
                            self.current_file += '.csv'
                        self.data.to_csv(self.current_file, index=False)
                        self.sessionWindow.append(f"Data saved to CSV file: {self.current_file}")
                    
                    self.app.is_modified = False
                    QMessageBox.information(
                        self,
                        "Success",
                        f"Data successfully saved to {self.current_file}"
                    )
                except Exception as e:
                    QMessageBox.critical(
                        self,
                        "Error",
                        f"Failed to save file:\n{str(e)}"
                    )
                    self.sessionWindow.append(f"Error saving file: {str(e)}")
            else:
                # If no current file, do Save As
                self.save_file_as()
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error preparing data to save: {str(e)}")
            self.sessionWindow.append(f"Error preparing data to save: {str(e)}")
    
    def save_file_as(self):
        """Save data to a new file"""
        try:
            # Try to load data from table first
            if self.load_data_from_table() == False:
                QMessageBox.warning(self, "Warning", "No data to save. Please enter some data in the table first.")
                return

            filename, _ = QFileDialog.getSaveFileName(
                self,
                "Save File As",
                "",
                "CSV Files (*.csv);;Excel Files (*.xlsx);;All Files (*.*)"
            )
            
            if not filename:
                return  # User cancelled
                
            try:
                if filename.endswith('.xlsx'):
                    try:
                        self.data.to_excel(filename, index=False)
                        self.sessionWindow.append(f"Data saved to Excel file: {filename}")
                    except ImportError:
                        QMessageBox.critical(
                            self,
                            "Error",
                            "Excel support requires the openpyxl package.\n"
                            "Please install it using: pip install openpyxl"
                        )
                        return
                else:
                    # Default to CSV if no extension or other extension
                    if not filename.endswith('.csv'):
                        filename += '.csv'
                    self.data.to_csv(filename, index=False)
                    self.sessionWindow.append(f"Data saved to CSV file: {filename}")
                
                self.current_file = filename  # Update current file
                self.app.current_file = filename  # Update app's current file
                self.app.is_modified = False
                QMessageBox.information(
                    self,
                    "Success",
                    f"Data successfully saved to {filename}"
                )
                
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Error",
                    f"Failed to save file:\n{str(e)}"
                )
                self.sessionWindow.append(f"Error saving file: {str(e)}")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error preparing data to save: {str(e)}")
            self.sessionWindow.append(f"Error preparing data to save: {str(e)}")
    
    def clear_table(self):
        """Clear all data from the table without closing the application."""
        try:
            reply = QMessageBox.question(
                self, 'Clear Table Data', 
                'Are you sure you want to clear all data?',
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                # Create a new empty DataFrame
                self.data = pd.DataFrame(columns=[f"C{i+1}" for i in range(10)])
                
                # Update the table
                self.table.clearContents()
                self.table.setRowCount(50)
                self.table.setColumnCount(10)
                self.table.setHorizontalHeaderLabels(self.data.columns)
                
                # Update the session window
                self.sessionWindow.append("\nTable data cleared successfully.")
                self.app.is_modified = False
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error clearing table data: {str(e)}")
    
    # Statistical analysis methods
    def show_descriptive_stats(self):
        """Show descriptive statistics"""
        try:
            if self.load_data_from_table():
                basic_stats.calculateDescriptiveStats(self)
            else:
                QMessageBox.warning(self, "Warning", "Please load or enter data first")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error calculating descriptive statistics: {str(e)}")
            traceback.print_exc()
    
    def show_correlation(self):
        """Show correlation analysis"""
        try:
            if self.load_data_from_table():
                basic_stats.calculateCorrelation(self)
            else:
                QMessageBox.warning(self, "Warning", "Please load or enter data first")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error calculating correlation: {str(e)}")
            traceback.print_exc()
    
    # Quality control methods
    def show_xbar_r_chart(self):
        """Show X-bar and R control chart"""
        self.load_data_from_table()
        if self.data.empty:
            QMessageBox.warning(self, "Warning", "Please load or enter data first")
            return
        
        control_charts.xbarRChart(self)
    
    def show_individual_chart(self):
        """Show individual control chart"""
        self.load_data_from_table()
        if self.data.empty:
            QMessageBox.warning(self, "Warning", "Please load or enter data first")
            return
        
        control_charts.individualChart(self)
    
    # Help methods
    def show_about_dialog(self):
        """Show about dialog"""
        QMessageBox.about(
            self, "About Minitab-like Application",
            "Custom Minitab-Like Tool\n\n"
            "Version: 1.0.0\n\n"
            "A statistical analysis application with quality control tools.\n"
            "Created for educational purposes."
        )

    # Add alias methods to maintain compatibility with original module functions
    def loadDataFromTable(self):
        """Alias for load_data_from_table to maintain compatibility"""
        return self.load_data_from_table()
    
    def updateTable(self):
        """Alias for update_table_from_data to maintain compatibility"""
        self.update_table_from_data()

    def show_probability_analysis(self):
        """Show probability analysis dialog"""
        try:
            if self.load_data_from_table():
                probabilityAnalysis(self)
            else:
                QMessageBox.warning(self, "Warning", "Please load or enter data first")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error in probability analysis: {str(e)}")
            traceback.print_exc()

    def show_process_capability(self):
        """Show process capability analysis dialog"""
        try:
            process_capability(self)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error in process capability analysis: {str(e)}")
            import traceback
            traceback.print_exc()

    # Add the random data generation methods
    def show_random_data(self, dist_type):
        """Show random data generation dialog for the specified distribution type"""
        try:
            from minitab_app.modules.calc.random_data import generate_random_data
            generate_random_data(self, dist_type.lower())
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error generating random data: {str(e)}")
            traceback.print_exc()

    def show_poisson_distribution(self):
        """Show Poisson distribution calculation dialog"""
        try:
            from minitab_app.modules.calc.random_data import poisson_distribution
            poisson_distribution(self)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error in Poisson distribution calculation: {str(e)}")
            traceback.print_exc()

    # Add new methods for MSA functions
    def show_gage_rr_study(self):
        try:
            if self.load_data_from_table():
                msa.gage_rr(self)
            else:
                QMessageBox.warning(self, "Warning", "Please load or enter data first")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error in Gage R&R Study: {str(e)}")
            traceback.print_exc()
    
    def show_linearity_study(self):
        try:
            if self.load_data_from_table():
                msa.linearity_study(self)
            else:
                QMessageBox.warning(self, "Warning", "Please load or enter data first")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error in Linearity Study: {str(e)}")
            traceback.print_exc()
    
    def show_bias_study(self):
        """Show bias study dialog"""
        try:
            if self.load_data_from_table():
                msa.bias_study(self)
            else:
                QMessageBox.warning(self, "Warning", "Please load or enter data first")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error in bias study: {str(e)}")
            traceback.print_exc()
    
    def show_stability_study(self):
        """Show stability study dialog"""
        try:
            if self.load_data_from_table():
                msa.stability_study(self)
            else:
                QMessageBox.warning(self, "Warning", "Please load or enter data first")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error in stability study: {str(e)}")
            traceback.print_exc()

    def show_hypothesis_testing(self):
        """Show hypothesis testing dialog"""
        try:
            if self.load_data_from_table():
                hypothesis_testing(self)
            else:
                QMessageBox.warning(self, "Warning", "Please load or enter data first")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error in hypothesis testing: {str(e)}")
            traceback.print_exc()

    def show_regression_analysis(self):
        """Show regression analysis dialog"""
        try:
            if self.load_data_from_table():
                regression_analysis(self)
            else:
                QMessageBox.warning(self, "Warning", "Please load or enter data first")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error in regression analysis: {str(e)}")
            traceback.print_exc()

    def show_chi_square_tests(self):
        """Show Chi-Square tests dialog"""
        try:
            chi_square_tests(self)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error in Chi-Square tests: {str(e)}")
            traceback.print_exc()

    def show_dpmo_calculator(self):
        """Show DPMO calculator dialog"""
        dpmoCalculator(self)

    def show_yield_analysis(self):
        """Show process yield analysis dialog"""
        yieldAnalysis(self)

    def show_pareto_chart(self):
        """Show Pareto chart dialog"""
        dmaic.pareto_chart(self)

    def show_fishbone_diagram(self):
        """Show Fishbone diagram dialog"""
        dmaic.fishbone_diagram(self)

    def show_fmea_template(self):
        """Show FMEA template dialog"""
        dmaic.fmea_template(self)

    def show_sigma_level_calculator(self):
        """Show Sigma Level Calculator dialog"""
        sigmaLevelCalc(self)

    def show_create_doe(self):
        """Show Create DOE dialog"""
        try:
            doe.create_doe(self)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error creating DOE: {str(e)}")
            traceback.print_exc()
    
    def show_analyze_doe(self):
        """Show Analyze DOE dialog"""
        try:
            if self.load_data_from_table():
                doe.analyze_doe(self)
            else:
                QMessageBox.warning(self, "Warning", "Please load or enter data first")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error analyzing DOE: {str(e)}")
            traceback.print_exc()
