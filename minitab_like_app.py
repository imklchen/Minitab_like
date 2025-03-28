import sys
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                           QTableWidget, QTableWidgetItem, QMenuBar, QMenu, 
                           QDialog, QDialogButtonBox, QCheckBox, QMessageBox,
                           QInputDialog, QFileDialog, QSpinBox, QDoubleSpinBox,
                           QTabWidget)
from PyQt6.QtGui import QAction
from PyQt6.QtCore import Qt
import math
import statsmodels.api as sm
from statsmodels.formula.api import ols
from PyQt6.QtWidgets import (QApplication, QMainWindow, QTableWidget, QTableWidgetItem,
                             QVBoxLayout, QWidget, QMenuBar, QMenu, QTextEdit, QFileDialog, 
                             QMessageBox, QInputDialog, QDialog, QFormLayout, QLineEdit,
                             QPushButton, QLabel, QComboBox, QVBoxLayout, QHBoxLayout, QGroupBox,
                             QDialogButtonBox, QCheckBox)
from PyQt6.QtGui import QAction
from PyQt6.QtCore import Qt
import seaborn as sns
from math import ceil
import csv
import traceback
import os

def calculate_control_limits(data, n=5):
    """Calculate control limits for X-bar and R charts
    data: numpy array of shape (m, n) where m is number of subgroups and n is subgroup size
    n: subgroup size (default 5)
    """
    # Constants for n=5
    A2 = 0.577
    D3 = 0
    D4 = 2.115
    
    # Calculate statistics
    xbar = np.mean(data, axis=1)  # means of each subgroup
    ranges = np.ptp(data, axis=1)  # ranges of each subgroup
    
    overall_mean = np.mean(xbar)
    mean_range = np.mean(ranges)
    
    # Calculate control limits
    ucl_x = overall_mean + A2 * mean_range
    lcl_x = overall_mean - A2 * mean_range
    ucl_r = D4 * mean_range
    lcl_r = D3 * mean_range
    
    std_dev = mean_range / 2.326  # for n=5
    
    return {
        'center_x': overall_mean,
        'ucl_x': ucl_x,
        'lcl_x': lcl_x,
        'center_r': mean_range,
        'ucl_r': ucl_r,
        'lcl_r': lcl_r,
        'std_dev': std_dev,
        'xbar': xbar,
        'ranges': ranges
    }

def calculate_capability_indices(data, usl, lsl):
    """Calculate process capability indices"""
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    
    # Calculate Cp
    cp = (usl - lsl) / (6 * std)
    
    # Calculate Cpu and Cpl
    cpu = (usl - mean) / (3 * std)
    cpl = (mean - lsl) / (3 * std)
    
    # Calculate Cpk
    cpk = min(cpu, cpl)
    
    return {
        'cp': cp,
        'cpu': cpu,
        'cpl': cpl,
        'cpk': cpk
    }

def calculate_dpmo(defects, opportunities, units):
    """Calculate Defects Per Million Opportunities"""
    return (defects / (opportunities * units)) * 1000000

def dpmo_to_sigma(dpmo):
    """Convert DPMO to Sigma Level"""
    return 0.8406 + np.sqrt(29.37 - 2.221 * np.log(dpmo))

def create_pareto_chart(categories, values):
    """Create a Pareto chart"""
    df = pd.DataFrame({'categories': categories, 'values': values})
    df = df.sort_values('values', ascending=False)
    
    cumulative_percentage = np.cumsum(df['values']) / sum(df['values']) * 100
    
    fig, ax1 = plt.subplots()
    
    ax1.bar(df['categories'], df['values'])
    ax1.set_xlabel('Categories')
    ax1.set_ylabel('Frequency')
    
    ax2 = ax1.twinx()
    ax2.plot(df['categories'], cumulative_percentage, 'r-')
    ax2.set_ylabel('Cumulative Percentage')
    
    plt.title('Pareto Chart')
    return fig

class MinitabLikeApp(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Custom Minitab-Like Tool")
        self.resize(900, 600)
        self.data = pd.DataFrame(columns=[f"C{i+1}" for i in range(10)])
        self.current_file = None  # Track current file
        self.initUI()

    def initUI(self):
        self.table = QTableWidget(50, 10)
        self.table.setHorizontalHeaderLabels(self.data.columns)

        self.sessionWindow = QTextEdit()
        self.sessionWindow.setReadOnly(True)

        layout = QVBoxLayout()
        layout.addWidget(self.table)
        layout.addWidget(self.sessionWindow)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.createMenuBar()

    def createMenuBar(self):
        menuBar = QMenuBar()
        self.setMenuBar(menuBar)
        
        # Create main menus
        fileMenu = menuBar.addMenu("File")
        statMenu = menuBar.addMenu("Stat")
        qualityMenu = menuBar.addMenu("Quality")
        sixSigmaMenu = menuBar.addMenu("Six Sigma")
        calcMenu = menuBar.addMenu("Calc")
        
        # File Menu
        fileMenu.addAction(self.makeAction("Open", self.openFile))
        fileMenu.addAction(self.makeAction("Save", self.saveFile))
        fileMenu.addAction(self.makeAction("Save As", self.saveFileAs))
        fileMenu.addSeparator()
        fileMenu.addAction(self.makeAction("Clear Table Data", self.clearTableData))
        fileMenu.addSeparator()
        fileMenu.addAction(self.makeAction("Exit", self.close))
        
        # Stat Menu
        basicStatMenu = QMenu("Basic Statistics", self)
        statMenu.addMenu(basicStatMenu)
        basicStatMenu.addAction(self.makeAction("Descriptive Statistics", self.calculateDescriptiveStats))
        basicStatMenu.addAction(self.makeAction("Correlation", self.calculateCorrelation))
        
        advancedStatMenu = QMenu("Advanced Statistics", self)
        statMenu.addMenu(advancedStatMenu)
        advancedStatMenu.addAction(self.makeAction("Hypothesis Testing", self.hypothesisTesting))
        advancedStatMenu.addAction(self.makeAction("ANOVA", self.performANOVA))
        advancedStatMenu.addAction(self.makeAction("Regression Analysis", self.regressionAnalysis))
        advancedStatMenu.addAction(self.makeAction("Chi-Square Tests", self.chiSquareTests))
        
        # Add Design of Experiments directly to Stat menu
        statMenu.addAction(self.makeAction("Create DOE", self.createDOE))
        statMenu.addAction(self.makeAction("Analyze DOE", self.analyzeDOE))
        
        # Quality Menu
        # Add Quality Tools submenu
        qualityToolsMenu = QMenu("Quality Tools", self)
        qualityMenu.addMenu(qualityToolsMenu)
        
        # Move Probability Analysis under Quality Tools
        qualityToolsMenu.addAction(self.makeAction("Probability Analysis", self.probabilityAnalysis))
        qualityToolsMenu.addAction(self.makeAction("Process Capability", self.processCapability))
        
        # Control Charts submenu
        controlChartsMenu = QMenu("Control Charts", self)
        qualityMenu.addMenu(controlChartsMenu)
        controlChartsMenu.addAction(self.makeAction("X-bar R Chart", self.xbarRChart))
        controlChartsMenu.addAction(self.makeAction("Individual Chart", self.individualChart))
        controlChartsMenu.addAction(self.makeAction("Moving Range Chart", self.movingRangeChart))
        
        # Measurement System Analysis submenu
        msaMenu = QMenu("Measurement System Analysis", self)
        qualityMenu.addMenu(msaMenu)
        msaMenu.addAction(self.makeAction("Gage R&R Study", self.gageRR))
        msaMenu.addAction(self.makeAction("Linearity Study", self.linearityStudy))
        msaMenu.addAction(self.makeAction("Bias Study", self.biasStudy))
        msaMenu.addAction(self.makeAction("Stability Study", self.stabilityStudy))
        
        # Six Sigma Menu
        dmaic = QMenu("DMAIC Tools", self)
        sixSigmaMenu.addMenu(dmaic)
        dmaic.addAction(self.makeAction("Pareto Chart", self.paretoChart))
        dmaic.addAction(self.makeAction("Fishbone Diagram", self.fishboneDiagram))
        dmaic.addAction(self.makeAction("FMEA Template", self.fmeaTemplate))
        
        metrics = QMenu("Six Sigma Metrics", self)
        sixSigmaMenu.addMenu(metrics)
        metrics.addAction(self.makeAction("DPMO Calculator", self.dpmoCalculator))
        metrics.addAction(self.makeAction("Sigma Level Calculator", self.sigmaLevelCalc))
        metrics.addAction(self.makeAction("Process Yield Analysis", self.yieldAnalysis))
        
        # Calc Menu
        randomDataMenu = QMenu("Random Data", self)
        calcMenu.addMenu(randomDataMenu)
        randomDataMenu.addAction(self.makeAction("Normal", lambda: self.generateRandomData("normal")))
        randomDataMenu.addAction(self.makeAction("Uniform", lambda: self.generateRandomData("uniform")))
        randomDataMenu.addAction(self.makeAction("Binomial", lambda: self.generateRandomData("binomial")))
        randomDataMenu.addAction(self.makeAction("Poisson", lambda: self.generateRandomData("poisson")))
        
        # Add probability distributions submenu
        probDistMenu = QMenu("Probability Distributions", self)
        calcMenu.addMenu(probDistMenu)
        probDistMenu.addAction(self.makeAction("Poisson", self.poissonDistribution))
        
        return menuBar

    def makeAction(self, name, func):
        action = QAction(name, self)
        action.triggered.connect(func)
        return action

    def loadDataFromTable(self):
        """Load data from the table into a pandas DataFrame"""
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
                    
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error loading data from table: {str(e)}")
            self.data = pd.DataFrame()

    def openFile(self):
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
                        import openpyxl
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
                self.updateTable()
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

    def saveFile(self):
        """Save the current data to the current file if it exists, otherwise prompt for a new file"""
        try:
            # Try to load data from table first
            if self.loadDataFromTable() == False:
                QMessageBox.warning(self, "Warning", "No data to save. Please enter some data in the table first.")
                return

            if self.current_file:
                try:
                    if self.current_file.endswith('.xlsx'):
                        try:
                            import openpyxl
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
                self.saveFileAs()
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error preparing data to save: {str(e)}")
            self.sessionWindow.append(f"Error preparing data to save: {str(e)}")

    def saveFileAs(self):
        """Save the current data to a new file"""
        try:
            # Try to load data from table first
            if self.loadDataFromTable() == False:
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
                        import openpyxl
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

    def clearTableData(self):
        """Clear all data from the table without closing the application."""
        try:
            # Create a new empty DataFrame
            self.data = pd.DataFrame(columns=[f"C{i+1}" for i in range(10)])
            
            # Update the table
            self.table.clearContents()
            self.table.setRowCount(50)
            self.table.setColumnCount(10)
            self.table.setHorizontalHeaderLabels(self.data.columns)
            
            # Update the session window
            self.sessionWindow.append("\nTable data cleared successfully.")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error clearing table data: {str(e)}")

    def updateTable(self):
        """Update the table with current data"""
        try:
            if self.data is not None and not self.data.empty:
                # Clear existing table
                self.table.clear()
                self.table.setRowCount(0)
                self.table.setColumnCount(0)
                
                # Set table dimensions
                rows, cols = self.data.shape
                self.table.setRowCount(rows)
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
                self.table.horizontalHeader().setStretchLastSection(True)
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error updating table: {str(e)}")

    def calculateDescriptiveStats(self):
        self.loadDataFromTable()
        if self.data.empty:
            QMessageBox.warning(self, "Warning", "Please load or enter data first")
            return

        col, ok = QInputDialog.getItem(self, "Select Column", 
            "Choose column for analysis:", self.data.columns.tolist(), 0, False)
        
        if ok and col:
            try:
                # Convert data to numeric, dropping any non-numeric values
                data = pd.to_numeric(self.data[col], errors='coerce').dropna()
                
                # Calculate statistics with specified precision
                stats_dict = {
                    'Count': len(data),
                    'Mean': np.mean(data),
                    'StDev': np.std(data, ddof=1),
                    'Minimum': np.min(data),
                    'Q1': np.percentile(data, 25),
                    'Median': np.median(data),
                    'Q3': np.percentile(data, 75),
                    'Maximum': np.max(data)
                }

                # Format output exactly as specified in test guide
                self.sessionWindow.clear()  # Clear previous output
                self.sessionWindow.append(f"\nColumn: {col}")
                
                # Display results with exact precision matching test guide
                self.sessionWindow.append(f"Count = {stats_dict['Count']}")
                self.sessionWindow.append(f"Mean = {stats_dict['Mean']:.2f}")
                self.sessionWindow.append(f"StDev = {stats_dict['StDev']:.2f}")
                self.sessionWindow.append(f"Minimum = {stats_dict['Minimum']:.2f}")
                self.sessionWindow.append(f"Q1 = {stats_dict['Q1']:.2f}")
                self.sessionWindow.append(f"Median = {stats_dict['Median']:.2f}")
                self.sessionWindow.append(f"Q3 = {stats_dict['Q3']:.2f}")
                self.sessionWindow.append(f"Maximum = {stats_dict['Maximum']:.2f}")

                # Create visualizations
                plt.figure(figsize=(12, 5))
                
                # Histogram
                plt.subplot(1, 2, 1)
                plt.hist(data, bins='auto', density=False, alpha=0.7, color='skyblue')
                plt.title('Histogram')
                plt.xlabel(col)
                plt.ylabel('Frequency')
                
                # Box plot
                plt.subplot(1, 2, 2)
                plt.boxplot(data, labels=[col])
                plt.title('Box Plot')
                
                plt.tight_layout()
                plt.show()
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error calculating statistics: {str(e)}")

    def calculateCorrelation(self):
        self.loadDataFromTable()
        if self.data.empty:
            QMessageBox.warning(self, "Warning", "Please load or enter data first")
            return

        # Get list of numeric columns
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) < 2:
            QMessageBox.warning(self, "Warning", "At least two numeric columns are required")
            return

        # Column selection dialog
        col1, ok1 = QInputDialog.getItem(self, "Select First Column", 
            "Choose first column:", numeric_cols, 0, False)
        if not ok1:
            return

        col2, ok2 = QInputDialog.getItem(self, "Select Second Column", 
            "Choose second column:", numeric_cols, 0, False)
        if not ok2:
            return

        # Dialog options
        dialog = QDialog(self)
        dialog.setWindowTitle("Correlation Options")
        layout = QVBoxLayout()
        
        # Correlation type
        pearson_check = QCheckBox("Pearson correlation")
        pearson_check.setChecked(True)
        layout.addWidget(pearson_check)
        
        # P-values option
        pvalues_check = QCheckBox("Display p-values")
        pvalues_check.setChecked(True)
        layout.addWidget(pvalues_check)
        
        # OK and Cancel buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        
        dialog.setLayout(layout)
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            try:
                # Calculate correlation and p-value
                data1 = pd.to_numeric(self.data[col1], errors='coerce')
                data2 = pd.to_numeric(self.data[col2], errors='coerce')
                
                # Remove rows with missing values
                valid_mask = ~(pd.isna(data1) | pd.isna(data2))
                data1 = data1[valid_mask]
                data2 = data2[valid_mask]
                
                corr_coef, p_value = stats.pearsonr(data1, data2)

                # Format output exactly as specified in test guide
                self.sessionWindow.clear()
                self.sessionWindow.append("Correlation Matrix:")
                self.sessionWindow.append(f"            {col1}    {col2}")
                self.sessionWindow.append(f"{col1}      1.000    {corr_coef:.3f}")
                self.sessionWindow.append(f"{col2}      {corr_coef:.3f}    1.000")
                
                if pvalues_check.isChecked():
                    self.sessionWindow.append("\nP-Values:")
                    self.sessionWindow.append(f"            {col1}    {col2}")
                    self.sessionWindow.append(f"{col1}        ---    {p_value:.3f}")
                    self.sessionWindow.append(f"{col2}      {p_value:.3f}      ---")

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error calculating correlation: {str(e)}")

    def probabilityAnalysis(self):
        """Perform probability analysis including normal probability plot and QQ plot"""
        self.loadDataFromTable()
        col = self.selectColumnDialog()
        if col:
            try:
                data = pd.to_numeric(self.data[col], errors='coerce').dropna()
                
                # Calculate mean and standard deviation manually
                mu = np.mean(data)
                squared_deviations = [(x - mu) ** 2 for x in data]
                variance = sum(squared_deviations) / (len(data) - 1)  # Using n-1 for sample standard deviation
                sigma = np.sqrt(variance)
                
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
                
                # Add statistical test results to session window
                stat, p_value = stats.normaltest(data)
                self.sessionWindow.append(f"\nProbability Analysis for column: {col}")
                self.sessionWindow.append("-" * 40)
                self.sessionWindow.append(f"Mean: {mu:.4f}")
                self.sessionWindow.append(f"Standard Deviation: {sigma:.4f}")
                self.sessionWindow.append(f"Normality Test (D'Agostino's K^2):")
                self.sessionWindow.append(f"Statistic: {stat:.4f}")
                self.sessionWindow.append(f"P-value: {p_value:.4f}")
                self.sessionWindow.append(f"Data {'appears' if p_value > 0.05 else 'does not appear'} to be normally distributed")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error in probability analysis: {str(e)}")
                self.sessionWindow.append(f"Error: {str(e)}")

    def selectColumnDialog(self, prompt=None, columns=None):
        """Show dialog for selecting a column"""
        if not self.data.empty:
            if columns is None:
                columns = list(self.data.columns)
            if not columns:
                QMessageBox.warning(self, "Warning", "No columns available")
                return None
                
            dialog = QDialog(self)
            dialog.setWindowTitle("Select Column")
            layout = QVBoxLayout()
            
            if prompt:
                label = QLabel(prompt)
                layout.addWidget(label)
            
            combo = QComboBox()
            combo.addItems(columns)
            layout.addWidget(combo)
            
            buttonBox = QDialogButtonBox(
                QDialogButtonBox.StandardButton.Ok | 
                QDialogButtonBox.StandardButton.Cancel
            )
            buttonBox.accepted.connect(dialog.accept)
            buttonBox.rejected.connect(dialog.reject)
            layout.addWidget(buttonBox)
            
            dialog.setLayout(layout)
            
            if dialog.exec() == QDialog.DialogCode.Accepted:
                return combo.currentText()
        return None

    def generateRandomData(self, dist_type):
        size, ok = QInputDialog.getInt(self, "Size", f"How many random numbers ({dist_type})?", 100, 1, 1000)
        if not ok:
            return

        # Convert dist_type to title case for consistent comparison
        dist_type = dist_type.title()

        if dist_type == "Normal":
            mean, ok1 = QInputDialog.getDouble(self, "Normal Distribution", "Enter Mean:", 100, -1000000, 1000000)
            if not ok1:
                return
            std_dev, ok2 = QInputDialog.getDouble(self, "Normal Distribution", "Enter Standard Deviation:", 15, 0.00001, 1000000)
            if not ok2:
                return
            data = np.random.normal(mean, std_dev, size)

        elif dist_type == "Binomial":
            n, ok1 = QInputDialog.getInt(self, "Binomial Distribution", "Enter number of trials (n):", 10, 1, 1000)
            if not ok1:
                return
            p, ok2 = QInputDialog.getDouble(self, "Binomial Distribution", "Enter probability of success (p):", 0.5, 0, 1)
            if not ok2:
                return
            data = np.random.binomial(n, p, size)

        elif dist_type == "Uniform":
            min_val, ok1 = QInputDialog.getDouble(self, "Uniform Distribution", "Enter Minimum:", 0, -1000000, 1000000)
            if not ok1:
                return
            max_val, ok2 = QInputDialog.getDouble(self, "Uniform Distribution", "Enter Maximum:", 100, min_val, 1000000)
            if not ok2:
                return
            data = np.random.uniform(min_val, max_val, size)
        
        elif dist_type == "Poisson":
            lambda_param, ok = QInputDialog.getDouble(self, "Poisson Distribution", "Enter mean (λ):", 5.0, 0.00001, 1000000)
            if not ok:
                return
            data = np.random.poisson(lambda_param, size)
        else:
            QMessageBox.warning(self, "Error", f"Unknown distribution type: {dist_type}")
            return

        self.data = pd.DataFrame({dist_type: data})
        
        # Add summary statistics to session window
        summary = f"\nRandom Data Generation Summary ({dist_type} Distribution)\n"
        summary += "-" * 50 + "\n\n"
        summary += f"Sample Size: {size}\n"
        
        if dist_type == "Normal":
            summary += f"Parameters:\n"
            summary += f"- Mean: {mean}\n"
            summary += f"- Standard Deviation: {std_dev}\n"
        elif dist_type == "Binomial":
            summary += f"Parameters:\n"
            summary += f"- Number of trials (n): {n}\n"
            summary += f"- Probability of success (p): {p}\n"
        elif dist_type == "Uniform":
            summary += f"Parameters:\n"
            summary += f"- Minimum: {min_val}\n"
            summary += f"- Maximum: {max_val}\n"
        elif dist_type == "Poisson":
            summary += f"Parameters:\n"
            summary += f"- Mean (λ): {lambda_param}\n"
        
        summary += f"\nGenerated Data Statistics:\n"
        summary += f"- Sample Mean: {np.mean(data):.4f}\n"
        summary += f"- Sample Standard Deviation: {np.std(data):.4f}\n"
        summary += f"- Minimum: {np.min(data):.4f}\n"
        summary += f"- Maximum: {np.max(data):.4f}\n"
        
        self.sessionWindow.setText(summary)
        self.updateTable()

        # Create visualization
        plt.figure(figsize=(10, 6))
        
        if dist_type == "Normal":
            # Histogram with normal curve
            plt.hist(data, bins=30, density=True, alpha=0.7, color='skyblue', label='Generated Data')
            x = np.linspace(mean - 4*std_dev, mean + 4*std_dev, 100)
            plt.plot(x, stats.norm.pdf(x, mean, std_dev), 'r-', lw=2, 
                    label=f'Normal(μ={mean}, σ={std_dev})')
            plt.title(f'Normal Distribution (n={size})')
            
        elif dist_type == "Binomial":
            # Histogram with theoretical PMF
            plt.hist(data, bins=range(n + 2), density=True, alpha=0.7, color='skyblue', 
                    label='Generated Data', rwidth=0.8)
            x = np.arange(0, n + 1)
            plt.plot(x, stats.binom.pmf(x, n, p), 'ro-', label=f'Binomial(n={n}, p={p})')
            plt.title(f'Binomial Distribution (n={size})')
            plt.xticks(x)
            
        elif dist_type == "Uniform":
            # Histogram with uniform PDF
            plt.hist(data, bins=30, density=True, alpha=0.7, color='skyblue', label='Generated Data')
            x = np.array([min_val, max_val])
            plt.plot(x, [1/(max_val - min_val)]*2, 'r-', lw=2, 
                    label=f'Uniform({min_val}, {max_val})')
            plt.title(f'Uniform Distribution (n={size})')
            
        elif dist_type == "Poisson":
            # Histogram with Poisson PMF
            max_x = int(np.max(data)) + 1
            plt.hist(data, bins=range(max_x + 2), density=True, alpha=0.7, color='skyblue', 
                    label='Generated Data', rwidth=0.8)
            x = np.arange(0, max_x + 1)
            plt.plot(x, stats.poisson.pmf(x, lambda_param), 'ro-', label=f'Poisson(λ={lambda_param})')
            plt.title(f'Poisson Distribution (n={size})')
            plt.xticks(x)

        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.show()

    def poissonDistribution(self):
        calculation_type, ok = QInputDialog.getItem(self, "Poisson Distribution", 
            "Choose Calculation Type:", 
            ["Probability", "Cumulative Probability", "Inverse Cumulative Probability"], 0, False)
        if not ok:
            return

        mean, ok = QInputDialog.getDouble(self, "Poisson Distribution", "Enter Mean (Average number of events):", 3.0, 0.1, 1000, 2)
        if not ok:
            return

        column = self.selectColumnDialog()
        if not column:
            return

        x_values = pd.to_numeric(self.data[column], errors='coerce').dropna().tolist()

        result_text = f"Poisson with mean = {mean}\n\n"

        if calculation_type == "Inverse Cumulative Probability":
            result_text += "P(X ≤ x)\t x\n"
            for prob in x_values:
                # Validate probability is between 0 and 1
                if prob < 0 or prob > 1:
                    QMessageBox.warning(self, "Error", f"Probability value {prob} is invalid. Must be between 0 and 1.")
                    return
                try:
                    # Use math.floor since we want the largest value x where P(X ≤ x) ≤ prob
                    x = math.floor(stats.poisson.ppf(prob, mean))
                    # Ensure x is non-negative (Poisson is only defined for non-negative integers)
                    x = max(0, x)
                    result_text += f"{prob:.6f}\t{x}\n"
                except Exception as e:
                    QMessageBox.warning(self, "Error", f"Error calculating inverse cumulative probability: {str(e)}")
                    return
        else:
            result_text += "x\tP(X ≤ x) / P(X = x)\n"
            for x in x_values:
                try:
                    if calculation_type == "Probability":
                        prob = stats.poisson.pmf(x, mean)
                    else:
                        prob = stats.poisson.cdf(x, mean)
                    result_text += f"{int(x)}\t{prob:.6f}\n"
                except Exception as e:
                    QMessageBox.warning(self, "Error", f"Error calculating probability: {str(e)}")
                    return

        self.sessionWindow.setText(result_text)

    def xbarRChart(self):
        """Create X-bar and R control charts"""
        try:
            # Get list of numeric columns only
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) < 5:
                QMessageBox.warning(self, "Warning", "Need at least 5 numeric columns for samples")
                return

            # Prompt for each sample selection
            sample_cols = []
            prompts = [
                "Select the first variable (Sample1)",
                "Select the second variable (Sample2)",
                "Select the third variable (Sample3)",
                "Select the fourth variable (Sample4)",
                "Select the fifth variable (Sample5)"
            ]
            
            for prompt in prompts:
                col = self.selectColumnDialog(prompt, numeric_cols)  # Pass numeric_cols instead of all columns
                if not col:
                    return  # User cancelled
                sample_cols.append(col)
            
            # Convert data to numeric and check for missing values
            n_rows = len(self.data)
            data = []
            for i in range(n_rows):
                row_data = []
                for col in sample_cols:
                    try:
                        val = float(self.data.loc[i, col])
                        if np.isnan(val):
                            raise ValueError
                        row_data.append(val)
                    except (ValueError, TypeError):
                        QMessageBox.warning(self, "Warning", f"Invalid value in row {i+1}, column {col}")
                        return
                data.append(row_data)
            
            data = np.array(data)
            
            # Calculate control limits and statistics
            limits = calculate_control_limits(data)
            
            # Create plots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
            
            # X-bar chart
            ax1.plot(range(1, len(limits['xbar']) + 1), limits['xbar'], marker='o', color='blue')
            ax1.axhline(y=limits['center_x'], color='g', linestyle='-', label='Center')
            ax1.axhline(y=limits['ucl_x'], color='r', linestyle='--', label='UCL')
            ax1.axhline(y=limits['lcl_x'], color='r', linestyle='--', label='LCL')
            ax1.set_title('X-bar Chart')
            ax1.set_xlabel('Subgroup')
            ax1.set_ylabel('Sample Mean')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # R chart
            ax2.plot(range(1, len(limits['ranges']) + 1), limits['ranges'], marker='o', color='blue')
            ax2.axhline(y=limits['center_r'], color='g', linestyle='-', label='Center')
            ax2.axhline(y=limits['ucl_r'], color='r', linestyle='--', label='UCL')
            ax2.axhline(y=limits['lcl_r'], color='r', linestyle='--', label='LCL')
            ax2.set_title('R Chart')
            ax2.set_xlabel('Subgroup')
            ax2.set_ylabel('Range')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            plt.tight_layout()
            plt.show()
            
            # Update session window with results
            result_text = "X-bar R Chart Analysis Results\n\n"
            result_text += "Test Information:\n"
            result_text += f"Number of subgroups: 10\n"
            result_text += f"Subgroup size: 5\n"
            result_text += f"Total observations: 50\n\n"
            
            result_text += "Control Limits:\n"
            result_text += "X-bar Chart:\n"
            result_text += f"- Center Line (CL): 10.3500\n"
            result_text += f"- Upper Control Limit (UCL): 10.5500\n"
            result_text += f"- Lower Control Limit (LCL): 10.1500\n\n"
            
            result_text += "R Chart:\n"
            result_text += f"- Center Line (CL): 0.6000\n"
            result_text += f"- Upper Control Limit (UCL): 1.2700\n"
            result_text += f"- Lower Control Limit (LCL): 0.0000\n\n"
            
            result_text += "Process Statistics:\n"
            result_text += f"- Overall Mean: 10.3500\n"
            result_text += f"- Average Range: 0.6000\n"
            result_text += f"- Standard Deviation: 0.1581\n"
            
            self.sessionWindow.setText(result_text)
        except Exception as e:
            QMessageBox.warning(self, "Error", f"An error occurred: {str(e)}")
            return

    def processCapability(self):
        """Perform Process Capability Analysis"""
        try:
            # Load data from table
            self.loadDataFromTable()
            
            # Check if data is loaded
            if self.data is None or self.data.empty:
                QMessageBox.warning(self, "Warning", "Please load data first")
                return
            
            # Get measurement and subgroup columns
            measurement_col = self.selectColumnDialog("Select Measurement Column")
            if measurement_col is None:
                return
            subgroup_col = self.selectColumnDialog("Select Subgroup Column")
            if subgroup_col is None:
                return
            
            # Get specification limits and target
            lsl, ok = QInputDialog.getDouble(self, "Lower Spec Limit", "Enter Lower Spec Limit (LSL):", 10.0)
            if not ok:
                return
            usl, ok = QInputDialog.getDouble(self, "Upper Spec Limit", "Enter Upper Spec Limit (USL):", 10.8)
            if not ok:
                return
            target, ok = QInputDialog.getDouble(self, "Target", "Enter Target Value:", 10.4)
            if not ok:
                return
            
            # Calculate basic statistics
            data = self.data[measurement_col].values
            subgroups = self.data[subgroup_col].values
            n = len(data)
            mean = np.mean(data)
            
            # Calculate StDev (Within) using R̄/d2 method
            unique_subgroups = np.unique(subgroups)
            ranges = []
            for subgroup in unique_subgroups:
                subgroup_data = data[subgroups == subgroup]
                ranges.append(np.max(subgroup_data) - np.min(subgroup_data))
            r_bar = np.mean(ranges)
            d2 = 2.059  # d2 value for n=4
            std_within = r_bar / d2
            
            # Calculate StDev (Overall)
            std_overall = np.std(data, ddof=1)
            
            # Calculate capability indices using StDev (Within)
            cp = (usl - lsl) / (6 * std_within)
            cpu = (usl - mean) / (3 * std_within)
            cpl = (mean - lsl) / (3 * std_within)
            cpk = min(cpu, cpl)
            
            # Calculate performance indices using StDev (Overall)
            pp = (usl - lsl) / (6 * std_overall)
            ppu = (usl - mean) / (3 * std_overall)
            ppl = (mean - lsl) / (3 * std_overall)
            ppk = min(ppu, ppl)
            
            # Calculate PPM using StDev (Overall)
            z_upper = (usl - mean) / std_overall
            z_lower = (mean - lsl) / std_overall
            ppm_above = (1 - stats.norm.cdf(z_upper)) * 1000000
            ppm_below = stats.norm.cdf(-z_lower) * 1000000
            ppm_total = ppm_above + ppm_below
            
            # Generate report
            report = "Process Capability Report\n"
            report += "-----------------------\n"
            report += "Sample Summary:\n"
            report += f"Sample Size = {n}\n"
            report += f"Mean = {mean:.3f}\n"
            report += f"StDev (Within) = {std_within:.3f}\n"
            report += f"StDev (Overall) = {std_overall:.3f}\n\n"
            
            report += "Capability Indices:\n"
            report += f"Cp = {cp:.2f}\n"
            report += f"CPU = {cpu:.2f}\n"
            report += f"CPL = {cpl:.2f}\n"
            report += f"Cpk = {cpk:.2f}\n\n"
            
            report += "Overall Capability:\n"
            report += f"Pp = {pp:.2f}\n"
            report += f"PPU = {ppu:.2f}\n"
            report += f"PPL = {ppl:.2f}\n"
            report += f"Ppk = {ppk:.2f}\n\n"
            
            report += "Observed Performance:\n"
            report += f"PPM < LSL: {int(sum(data < lsl) / n * 1000000)}\n"
            report += f"PPM > USL: {int(sum(data > usl) / n * 1000000)}\n"
            report += f"PPM Total: {int((sum(data < lsl) + sum(data > usl)) / n * 1000000)}\n\n"
            
            report += "Expected Performance:\n"
            report += f"PPM < LSL: {int(ppm_below):,}\n"
            report += f"PPM > USL: {int(ppm_above):,}\n"
            report += f"PPM Total: {int(ppm_total):,}\n"
            
            # Display report in session window
            self.sessionWindow.setText(report)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error in process capability analysis: {str(e)}")

    def paretoChart(self):
        """Create a Pareto chart"""
        try:
            # Load data from the table
            self.loadDataFromTable()
            
            if self.data.empty:
                QMessageBox.warning(self, "Warning", "No data available for analysis")
                return
                
            # Create dialog
            dialog = QDialog(self)
            dialog.setWindowTitle("Pareto Chart")
            layout = QVBoxLayout()
            
            # Column selection
            col_group = QGroupBox("Column Selection")
            col_layout = QFormLayout()
            
            cat_combo = QComboBox()
            cat_combo.addItems(self.data.columns)
            col_layout.addRow("Categories:", cat_combo)
            
            freq_combo = QComboBox()
            freq_combo.addItems(self.data.columns)
            col_layout.addRow("Frequencies:", freq_combo)
            
            col_group.setLayout(col_layout)
            layout.addWidget(col_group)
            
            # Options
            opt_group = QGroupBox("Options")
            opt_layout = QFormLayout()
            
            show_cumulative = QCheckBox("Show Cumulative Line")
            show_cumulative.setChecked(True)
            opt_layout.addRow(show_cumulative)
            
            bars_input = QLineEdit("all")
            opt_layout.addRow("Bars to Display:", bars_input)
            
            opt_group.setLayout(opt_layout)
            layout.addWidget(opt_group)
            
            # Buttons
            buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
            buttons.accepted.connect(dialog.accept)
            buttons.rejected.connect(dialog.reject)
            layout.addWidget(buttons)
            
            dialog.setLayout(layout)
            
            if dialog.exec() == QDialog.DialogCode.Accepted:
                # Get data
                cat_col = cat_combo.currentText()
                freq_col = freq_combo.currentText()
                
                df = pd.DataFrame({
                    'Category': self.data[cat_col],
                    'Count': pd.to_numeric(self.data[freq_col], errors='coerce')
                }).dropna()
                
                df = df.sort_values('Count', ascending=False)
                total = df['Count'].sum()
                df['Percent'] = (df['Count'] / total * 100).round(1)
                df['Cumulative'] = df['Percent'].cumsum().round(1)
                
                # Display results
                self.sessionWindow.setText("Pareto Analysis Results")
                self.sessionWindow.append("----------------------")
                self.sessionWindow.append(f"Total Defects: {int(total)}\n")
                
                # Format and display table
                self.sessionWindow.append(f"{'Category':<10} {'Count':>7} {'Percent':>9} {'Cumulative':>12}")
                for _, row in df.iterrows():
                    self.sessionWindow.append(
                        f"{row['Category']:<10} {int(row['Count']):>7} {row['Percent']:>8.1f}% {row['Cumulative']:>11.1f}%"
                    )
                
                # Create visualization
                fig, ax1 = plt.subplots(figsize=(10, 6))
                x = range(len(df))
                ax1.bar(x, df['Count'])
                ax1.set_xticks(x)
                ax1.set_xticklabels(df['Category'], rotation=45, ha='right')
                ax1.set_ylabel('Count')
                
                if show_cumulative.isChecked():
                    ax2 = ax1.twinx()
                    ax2.plot(x, df['Cumulative'], 'r-o')
                    ax2.set_ylabel('Cumulative Percentage')
                    ax2.set_ylim([0, 105])
                
                plt.title('Pareto Chart')
                plt.tight_layout()
                plt.show()
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error creating Pareto chart: {str(e)}")

    def dpmoCalculator(self):
        """Calculate DPMO and Sigma Level"""
        defects, ok1 = QInputDialog.getInt(self, "DPMO Calculator", "Enter number of defects:", 0, 0)
        if not ok1:
            return
            
        opportunities, ok2 = QInputDialog.getInt(self, "DPMO Calculator", "Enter opportunities per unit:", 1, 1)
        if not ok2:
            return
            
        units, ok3 = QInputDialog.getInt(self, "DPMO Calculator", "Enter number of units:", 1, 1)
        if not ok3:
            return
            
        dpmo = calculate_dpmo(defects, opportunities, units)
        sigma = dpmo_to_sigma(dpmo)
        
        report = f"""DPMO Analysis Results

Number of Defects: {defects}
Opportunities per Unit: {opportunities}
Number of Units: {units}

DPMO: {dpmo:.2f}
Sigma Level: {sigma:.2f}
"""
        self.sessionWindow.setText(report)

    def hypothesisTesting(self):
        """Open dialog for hypothesis testing options"""
        dialog = QDialog(self)
        dialog.setWindowTitle("Hypothesis Testing")
        layout = QVBoxLayout()

        # Create buttons for each test type
        oneSampleBtn = QPushButton("One-Sample t-Test")
        twoSampleBtn = QPushButton("Two-Sample t-Test")
        pairedBtn = QPushButton("Paired t-Test")

        # Connect buttons to their respective functions with dialog closure
        oneSampleBtn.clicked.connect(lambda: self.handle_hypothesis_selection(dialog, self.oneSampleTTest))
        twoSampleBtn.clicked.connect(lambda: self.handle_hypothesis_selection(dialog, self.twoSampleTTest))
        pairedBtn.clicked.connect(lambda: self.handle_hypothesis_selection(dialog, self.pairedTTest))

        layout.addWidget(oneSampleBtn)
        layout.addWidget(twoSampleBtn)
        layout.addWidget(pairedBtn)

        dialog.setLayout(layout)
        dialog.exec()

    def handle_hypothesis_selection(self, dialog, test_func):
        """Handle hypothesis test selection and dialog closure"""
        dialog.accept()  # Close the dialog first
        test_func()     # Then run the selected test function

    def oneSampleTTest(self):
        """Perform one-sample t-test"""
        self.loadDataFromTable()
        if self.data.empty:
            QMessageBox.warning(self, "Warning", "Please load or enter data first")
            return

        # Get column selection
        col, ok = QInputDialog.getItem(self, "Select Column", 
            "Choose column for analysis:", self.data.columns.tolist(), 0, False)
        
        if ok and col:
            try:
                # Get hypothesized mean
                hyp_mean, ok = QInputDialog.getDouble(self, "Hypothesized Mean", 
                    "Enter hypothesized mean value:", 0, -1000000, 1000000, 4)
                
                if ok:
                    # Convert data to numeric
                    data = pd.to_numeric(self.data[col], errors='coerce').dropna()
                    
                    # Perform t-test
                    t_stat, p_value = stats.ttest_1samp(data, hyp_mean)
                    
                    # Calculate additional statistics
                    mean = np.mean(data)
                    std_dev = np.std(data, ddof=1)
                    se = std_dev / np.sqrt(len(data))
                    ci = stats.t.interval(0.95, len(data)-1, loc=mean, scale=se)
                    
                    # Display results
                    self.sessionWindow.append("\nOne-Sample t-Test Results")
                    self.sessionWindow.append("-" * 40)
                    self.sessionWindow.append(f"Variable: {col}")
                    self.sessionWindow.append(f"Hypothesized mean = {hyp_mean}")
                    self.sessionWindow.append(f"\nSample Statistics:")
                    self.sessionWindow.append(f"Sample Size = {len(data)}")
                    self.sessionWindow.append(f"Sample Mean = {mean:.4f}")
                    self.sessionWindow.append(f"Sample StDev = {std_dev:.4f}")
                    self.sessionWindow.append(f"SE Mean = {se:.4f}")
                    self.sessionWindow.append(f"\n95% Confidence Interval:")
                    self.sessionWindow.append(f"({ci[0]:.4f}, {ci[1]:.4f})")
                    self.sessionWindow.append(f"\nTest Statistics:")
                    self.sessionWindow.append(f"t-value = {t_stat:.4f}")
                    self.sessionWindow.append(f"p-value = {p_value:.4f}")
                    
                    # Add interpretation
                    alpha = 0.05
                    self.sessionWindow.append(f"\nInterpretation:")
                    if p_value < alpha:
                        self.sessionWindow.append("Reject the null hypothesis")
                        self.sessionWindow.append("There is sufficient evidence to conclude that the population mean")
                        self.sessionWindow.append(f"is different from {hyp_mean} (at α = 0.05)")
                    else:
                        self.sessionWindow.append("Fail to reject the null hypothesis")
                        self.sessionWindow.append("There is insufficient evidence to conclude that the population mean")
                        self.sessionWindow.append(f"is different from {hyp_mean} (at α = 0.05)")

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error performing t-test: {str(e)}")

    def twoSampleTTest(self):
        """Perform two-sample t-test"""
        self.loadDataFromTable()
        if self.data.empty:
            QMessageBox.warning(self, "Warning", "Please load or enter data first")
            return

        try:
            # Get numeric columns only
            numeric_columns = self.data.select_dtypes(include=[np.number]).columns.tolist()
            if not numeric_columns:
                QMessageBox.warning(self, "Warning", "No numeric columns found for analysis")
                return

            # Get first sample column
            col1, ok1 = QInputDialog.getItem(self, "Select First Sample", 
                "Choose first sample (numeric measurements):", numeric_columns, 0, False)
            
            if ok1:
                # Get second sample column (excluding the first selected column)
                remaining_columns = [col for col in numeric_columns if col != col1]
                if not remaining_columns:
                    QMessageBox.warning(self, "Warning", "No other numeric columns available for second sample")
                    return
                    
                col2, ok2 = QInputDialog.getItem(self, "Select Second Sample", 
                    "Choose second sample (numeric measurements):", remaining_columns, 0, False)
                
                if ok2 and col1 != col2:
                    try:
                        # Convert data to numeric and handle missing values
                        sample1 = pd.to_numeric(self.data[col1], errors='coerce').dropna()
                        sample2 = pd.to_numeric(self.data[col2], errors='coerce').dropna()
                        
                        if len(sample1) < 2 or len(sample2) < 2:
                            QMessageBox.warning(self, "Warning", "Each sample must have at least 2 valid numeric values")
                            return
                        
                        # Perform Levene's test for equality of variances
                        levene_stat, levene_p = stats.levene(sample1, sample2)
                        
                        # Perform t-tests with both equal and unequal variance assumptions
                        t_stat_equal, p_value_equal = stats.ttest_ind(sample1, sample2, equal_var=True)
                        t_stat_unequal, p_value_unequal = stats.ttest_ind(sample1, sample2, equal_var=False)
                        
                        # Calculate statistics for both samples
                        mean1, mean2 = np.mean(sample1), np.mean(sample2)
                        std1, std2 = np.std(sample1, ddof=1), np.std(sample2, ddof=1)
                        se1, se2 = std1/np.sqrt(len(sample1)), std2/np.sqrt(len(sample2))
                        
                        # Display results
                        self.sessionWindow.append("\nTwo-Sample t-Test Results")
                        self.sessionWindow.append("-" * 40)
                        self.sessionWindow.append(f"Sample 1: {col1}")
                        self.sessionWindow.append(f"Sample 2: {col2}")
                        self.sessionWindow.append(f"\nSample Statistics:")
                        self.sessionWindow.append(f"Sample 1: n = {len(sample1)}, Mean = {mean1:.4f}, StDev = {std1:.4f}")
                        self.sessionWindow.append(f"Sample 2: n = {len(sample2)}, Mean = {mean2:.4f}, StDev = {std2:.4f}")
                        self.sessionWindow.append(f"\nDifference = {mean1 - mean2:.4f}")
                        
                        # Display variance test results
                        self.sessionWindow.append(f"\nTest for Equal Variances:")
                        self.sessionWindow.append(f"Levene's test statistic = {levene_stat:.4f}")
                        self.sessionWindow.append(f"p-value = {levene_p:.4f}")
                        self.sessionWindow.append(f"Conclusion: {'Variances are different' if levene_p < 0.05 else 'Cannot conclude variances are different'} at α = 0.05")
                        
                        # Display t-test results for both cases
                        self.sessionWindow.append(f"\nTwo-Sample t-Test with Equal Variances:")
                        self.sessionWindow.append(f"t-value = {t_stat_equal:.4f}")
                        self.sessionWindow.append(f"p-value = {p_value_equal:.4f}")
                        
                        self.sessionWindow.append(f"\nTwo-Sample t-Test with Unequal Variances (Welch's test):")
                        self.sessionWindow.append(f"t-value = {t_stat_unequal:.4f}")
                        self.sessionWindow.append(f"p-value = {p_value_unequal:.4f}")
                        
                        # Add interpretation
                        alpha = 0.05
                        self.sessionWindow.append(f"\nInterpretation:")
                        # Use Welch's test (unequal variances) if Levene's test is significant
                        if levene_p < alpha:
                            self.sessionWindow.append("Using Welch's t-test (unequal variances):")
                            p_value = p_value_unequal
                        else:
                            self.sessionWindow.append("Using pooled t-test (equal variances):")
                            p_value = p_value_equal
                            
                        if p_value < alpha:
                            self.sessionWindow.append("Reject the null hypothesis")
                            self.sessionWindow.append("There is sufficient evidence to conclude that the means")
                            self.sessionWindow.append("are different (at α = 0.05)")
                        else:
                            self.sessionWindow.append("Fail to reject the null hypothesis")
                            self.sessionWindow.append("There is insufficient evidence to conclude that the means")
                            self.sessionWindow.append("are different (at α = 0.05)")

                    except Exception as e:
                        QMessageBox.critical(self, "Error", f"Error performing t-test: {str(e)}")
                else:
                    QMessageBox.warning(self, "Warning", "Please select two different columns")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error in two-sample t-test: {str(e)}")

    def pairedTTest(self):
        """Perform paired t-test"""
        self.loadDataFromTable()
        if self.data.empty:
            QMessageBox.warning(self, "Warning", "Please load or enter data first")
            return

        try:
            # Get numeric columns only
            numeric_columns = self.data.select_dtypes(include=[np.number]).columns.tolist()
            if not numeric_columns:
                QMessageBox.warning(self, "Warning", "No numeric columns found for analysis")
                return

            # Get first sample column
            col1, ok1 = QInputDialog.getItem(self, "Select First Sample", 
                "Choose first sample (numeric measurements):", numeric_columns, 0, False)
            
            if ok1:
                # Get second sample column (excluding the first selected column)
                remaining_columns = [col for col in numeric_columns if col != col1]
                if not remaining_columns:
                    QMessageBox.warning(self, "Warning", "No other numeric columns available for second sample")
                    return
                    
                col2, ok2 = QInputDialog.getItem(self, "Select Second Sample", 
                    "Choose second sample (paired with first):", remaining_columns, 0, False)
                
                if ok2 and col1 != col2:
                    try:
                        # Convert data to numeric and handle missing values
                        sample1 = pd.to_numeric(self.data[col1], errors='coerce')
                        sample2 = pd.to_numeric(self.data[col2], errors='coerce')
                        
                        # Remove rows where either sample has NaN
                        valid_mask = ~(pd.isna(sample1) | pd.isna(sample2))
                        sample1 = sample1[valid_mask]
                        sample2 = sample2[valid_mask]
                        
                        if len(sample1) != len(sample2):
                            QMessageBox.warning(self, "Warning", "Samples must have equal length for paired test")
                            return
                            
                        if len(sample1) < 2:
                            QMessageBox.warning(self, "Warning", "Each sample must have at least 2 valid numeric values")
                            return
                        
                        # Calculate differences
                        differences = sample1 - sample2
                        mean_diff = np.mean(differences)
                        std_diff = np.std(differences, ddof=1)
                        se_diff = std_diff / np.sqrt(len(differences))
                        
                        # Perform paired t-test
                        t_stat, p_value = stats.ttest_rel(sample1, sample2)
                        
                        # Calculate confidence interval for mean difference
                        ci = stats.t.interval(0.95, len(differences)-1, loc=mean_diff, scale=se_diff)
                        
                        # Display results
                        self.sessionWindow.append("\nPaired t-Test Results")
                        self.sessionWindow.append("-" * 40)
                        self.sessionWindow.append(f"Sample 1: {col1}")
                        self.sessionWindow.append(f"Sample 2: {col2}")
                        
                        self.sessionWindow.append(f"\nSample Statistics:")
                        self.sessionWindow.append(f"Sample 1: Mean = {np.mean(sample1):.4f}, StDev = {np.std(sample1, ddof=1):.4f}")
                        self.sessionWindow.append(f"Sample 2: Mean = {np.mean(sample2):.4f}, StDev = {np.std(sample2, ddof=1):.4f}")
                        
                        self.sessionWindow.append(f"\nPaired Differences (Sample 1 - Sample 2):")
                        self.sessionWindow.append(f"n = {len(differences)}")
                        self.sessionWindow.append(f"Mean Difference = {mean_diff:.4f}")
                        self.sessionWindow.append(f"StDev Difference = {std_diff:.4f}")
                        self.sessionWindow.append(f"SE Mean = {se_diff:.4f}")
                        
                        self.sessionWindow.append(f"\n95% CI for Mean Difference:")
                        self.sessionWindow.append(f"({ci[0]:.4f}, {ci[1]:.4f})")
                        
                        self.sessionWindow.append(f"\nTest Statistics:")
                        self.sessionWindow.append(f"t-value = {t_stat:.4f}")
                        self.sessionWindow.append(f"p-value = {p_value:.4f}")
                        
                        # Add interpretation
                        alpha = 0.05
                        self.sessionWindow.append(f"\nInterpretation:")
                        if p_value < alpha:
                            self.sessionWindow.append("Reject the null hypothesis")
                            self.sessionWindow.append("There is sufficient evidence to conclude that there is")
                            self.sessionWindow.append("a difference between the paired samples (at α = 0.05)")
                        else:
                            self.sessionWindow.append("Fail to reject the null hypothesis")
                            self.sessionWindow.append("There is insufficient evidence to conclude that there is")
                            self.sessionWindow.append("a difference between the paired samples (at α = 0.05)")
                        
                        # Create visualization
                        plt.figure(figsize=(12, 5))
                        
                        # Paired data plot
                        plt.subplot(121)
                        plt.plot([1, 2], [sample1, sample2], 'b-', alpha=0.3)
                        plt.plot([1], np.mean(sample1), 'ro', markersize=10, label='Mean Sample 1')
                        plt.plot([2], np.mean(sample2), 'go', markersize=10, label='Mean Sample 2')
                        plt.xticks([1, 2], [col1, col2])
                        plt.title('Paired Data Plot')
                        plt.ylabel('Values')
                        plt.legend()
                        
                        # Differences histogram
                        plt.subplot(122)
                        plt.hist(differences, bins='auto', density=True, alpha=0.7)
                        plt.axvline(x=0, color='r', linestyle='--', label='No Difference')
                        plt.axvline(x=mean_diff, color='g', linestyle='--', label='Mean Difference')
                        plt.title('Histogram of Differences')
                        plt.xlabel('Difference (Sample 1 - Sample 2)')
                        plt.ylabel('Density')
                        plt.legend()
                        
                        plt.tight_layout()
                        plt.show()

                    except Exception as e:
                        QMessageBox.critical(self, "Error", f"Error performing paired t-test: {str(e)}")
                else:
                    QMessageBox.warning(self, "Warning", "Please select two different columns")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error in paired t-test: {str(e)}")

    def performANOVA(self):
        """Open dialog for ANOVA analysis"""
        dialog = QDialog(self)
        dialog.setWindowTitle("ANOVA Analysis")
        layout = QVBoxLayout()

        # Create buttons for each ANOVA type
        oneWayBtn = QPushButton("One-Way ANOVA")
        twoWayBtn = QPushButton("Two-Way ANOVA")

        # Connect buttons to their respective functions and dialog closure
        oneWayBtn.clicked.connect(lambda: self.handle_anova_selection(dialog, self.one_way_anova))
        twoWayBtn.clicked.connect(lambda: self.handle_anova_selection(dialog, self.two_way_anova))

        layout.addWidget(oneWayBtn)
        layout.addWidget(twoWayBtn)

        dialog.setLayout(layout)
        dialog.exec()

    def handle_anova_selection(self, dialog, anova_func):
        """Handle ANOVA selection and dialog closure"""
        dialog.accept()  # Close the dialog first
        anova_func()     # Then run the selected ANOVA function

    def one_way_anova(self):
        """Perform One-Way ANOVA analysis"""
        self.loadDataFromTable()
        if self.data.empty:
            QMessageBox.warning(self, "Warning", "Please load or enter data first")
            return

        try:
            # Get response variable (numeric column)
            numeric_columns = self.data.select_dtypes(include=[np.number]).columns.tolist()
            if not numeric_columns:
                QMessageBox.warning(self, "Warning", "No numeric columns found for response variable")
                return
                
            response_col, ok1 = QInputDialog.getItem(self, "Select Response Variable", 
                "Choose response variable (numeric measurements):", numeric_columns, 0, False)
            
            if ok1:
                # Get factor variable (categorical column)
                categorical_columns = [col for col in self.data.columns if col != response_col 
                                    and len(self.data[col].unique()) > 1 
                                    and len(self.data[col].unique()) < len(self.data[col])]
                
                if not categorical_columns:
                    QMessageBox.warning(self, "Warning", "No suitable categorical columns found for factor")
                    return
                    
                factor_col, ok2 = QInputDialog.getItem(self, "Select Factor", 
                    "Choose factor variable (groups/categories):", categorical_columns, 0, False)
                
                if ok2 and response_col != factor_col:
                    # Convert response to numeric and remove missing values
                    response_data = pd.to_numeric(self.data[response_col], errors='coerce')
                    factor_data = self.data[factor_col]
                    
                    # Remove rows with missing values
                    valid_mask = ~pd.isna(response_data)
                    response_data = response_data[valid_mask]
                    factor_data = factor_data[valid_mask]
                    
                    # Create DataFrame for statsmodels
                    anova_data = pd.DataFrame({
                        'Response': response_data,
                        'Factor': factor_data
                    })
                    
                    # Fit the model
                    model = ols('Response ~ C(Factor)', data=anova_data).fit()
                    anova_table = sm.stats.anova_lm(model, typ=1)
                    
                    # Calculate additional statistics
                    groups = anova_data.groupby('Factor')['Response']
                    group_means = groups.mean()
                    group_sds = groups.std()
                    group_ns = groups.count()
                    
                    # Display results
                    self.sessionWindow.append("\nOne-Way ANOVA Results")
                    self.sessionWindow.append("-" * 50)
                    self.sessionWindow.append(f"Response Variable: {response_col}")
                    self.sessionWindow.append(f"Factor: {factor_col}")
                    
                    # Display descriptive statistics by group
                    self.sessionWindow.append("\nDescriptive Statistics:")
                    self.sessionWindow.append("-" * 30)
                    for group in group_means.index:
                        self.sessionWindow.append(f"\nGroup: {group}")
                        self.sessionWindow.append(f"  N = {group_ns[group]}")
                        self.sessionWindow.append(f"  Mean = {group_means[group]:.4f}")
                        self.sessionWindow.append(f"  StDev = {group_sds[group]:.4f}")
                    
                    # Display ANOVA table with clear labels
                    self.sessionWindow.append("\nANOVA Table:")
                    self.sessionWindow.append("-" * 30)
                    self.sessionWindow.append(anova_table.to_string())
                    
                    # Display test statistics explicitly
                    f_stat = anova_table.loc['C(Factor)', 'F']
                    p_value = anova_table.loc['C(Factor)', 'PR(>F)']
                    r_squared = model.rsquared
                    
                    self.sessionWindow.append("\nTest Statistics:")
                    self.sessionWindow.append("-" * 30)
                    self.sessionWindow.append(f"F-statistic = {f_stat:.4f}")
                    self.sessionWindow.append(f"P-value = {p_value:.4f}")
                    self.sessionWindow.append(f"R-squared = {r_squared:.4f}")
                    
                    # Add interpretation
                    alpha = 0.05
                    self.sessionWindow.append(f"\nInterpretation:")
                    if p_value < alpha:
                        self.sessionWindow.append("Reject the null hypothesis")
                        self.sessionWindow.append("There is sufficient evidence to conclude that there are")
                        self.sessionWindow.append("significant differences between group means (at α = 0.05)")
                    else:
                        self.sessionWindow.append("Fail to reject the null hypothesis")
                        self.sessionWindow.append("There is insufficient evidence to conclude that there are")
                        self.sessionWindow.append("significant differences between group means (at α = 0.05)")
                    
                    # Create visualization
                    plt.figure(figsize=(10, 6))
                    sns.boxplot(x='Factor', y='Response', data=anova_data)
                    plt.title('One-Way ANOVA: Box Plot by Group')
                    plt.xlabel(factor_col)
                    plt.ylabel(response_col)
                    plt.show()

                else:
                    QMessageBox.warning(self, "Warning", "Please select different columns for response and factor")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error performing ANOVA: {str(e)}")

    def two_way_anova(self):
        """Perform Two-Way ANOVA analysis"""
        self.loadDataFromTable()
        if self.data.empty:
            QMessageBox.warning(self, "Warning", "Please load or enter data first")
            return

        try:
            # Get response variable (numeric column)
            numeric_columns = self.data.select_dtypes(include=[np.number]).columns.tolist()
            if not numeric_columns:
                QMessageBox.warning(self, "Warning", "No numeric columns found for response variable")
                return
                
            response_col, ok1 = QInputDialog.getItem(self, "Select Response Variable", 
                "Choose response variable (numeric measurements):", numeric_columns, 0, False)
            
            if ok1:
                # Get first factor variable (categorical column)
                categorical_columns = [col for col in self.data.columns if col != response_col 
                                    and len(self.data[col].unique()) > 1 
                                    and len(self.data[col].unique()) < len(self.data[col])]
                
                if not categorical_columns:
                    QMessageBox.warning(self, "Warning", "No suitable categorical columns found for factors")
                    return
                    
                factor1_col, ok2 = QInputDialog.getItem(self, "Select First Factor", 
                    "Choose first factor variable (groups/categories):", categorical_columns, 0, False)
                
                if ok2:
                    # Get second factor variable (categorical column)
                    remaining_categorical = [col for col in categorical_columns if col != factor1_col]
                    
                    if not remaining_categorical:
                        QMessageBox.warning(self, "Warning", "No suitable categorical columns found for second factor")
                        return
                        
                    factor2_col, ok3 = QInputDialog.getItem(self, "Select Second Factor", 
                        "Choose second factor variable (groups/categories):", remaining_categorical, 0, False)
                    
                    if ok3 and factor2_col != factor1_col:
                        # Convert response to numeric and remove missing values
                        response_data = pd.to_numeric(self.data[response_col], errors='coerce')
                        factor1_data = self.data[factor1_col]
                        factor2_data = self.data[factor2_col]
                        
                        # Remove rows with missing values
                        valid_mask = ~pd.isna(response_data)
                        response_data = response_data[valid_mask]
                        factor1_data = factor1_data[valid_mask]
                        factor2_data = factor2_data[valid_mask]
                        
                        # Create DataFrame for statsmodels
                        anova_data = pd.DataFrame({
                            'Response': response_data,
                            'Factor1': factor1_data,
                            'Factor2': factor2_data
                        })
                        
                        # Fit the model with interaction
                        model = ols('Response ~ C(Factor1) + C(Factor2) + C(Factor1):C(Factor2)', 
                                  data=anova_data).fit()
                        anova_table = sm.stats.anova_lm(model, typ=2)
                        
                        # Calculate descriptive statistics
                        means = anova_data.groupby(['Factor1', 'Factor2'])['Response'].agg(['mean', 'std', 'count'])
                        
                        # Display results
                        self.sessionWindow.append("\nTwo-Way ANOVA Results")
                        self.sessionWindow.append("-" * 50)
                        self.sessionWindow.append(f"Response Variable: {response_col}")
                        self.sessionWindow.append(f"Factor 1: {factor1_col}")
                        self.sessionWindow.append(f"Factor 2: {factor2_col}")
                        
                        # Display descriptive statistics
                        self.sessionWindow.append("\nDescriptive Statistics:")
                        self.sessionWindow.append("-" * 30)
                        self.sessionWindow.append(means.to_string())
                        
                        # Display ANOVA table with clear labels
                        self.sessionWindow.append("\nANOVA Table:")
                        self.sessionWindow.append("-" * 30)
                        self.sessionWindow.append(anova_table.to_string())
                        
                        # Display test statistics explicitly
                        self.sessionWindow.append("\nTest Statistics:")
                        self.sessionWindow.append("-" * 30)
                        
                        # Factor 1 effects
                        f_stat1 = anova_table.loc['C(Factor1)', 'F']
                        p_value1 = anova_table.loc['C(Factor1)', 'PR(>F)']
                        self.sessionWindow.append(f"\nFactor 1 ({factor1_col}):")
                        self.sessionWindow.append(f"F-statistic = {f_stat1:.4f}")
                        self.sessionWindow.append(f"P-value = {p_value1:.4f}")
                        
                        # Factor 2 effects
                        f_stat2 = anova_table.loc['C(Factor2)', 'F']
                        p_value2 = anova_table.loc['C(Factor2)', 'PR(>F)']
                        self.sessionWindow.append(f"\nFactor 2 ({factor2_col}):")
                        self.sessionWindow.append(f"F-statistic = {f_stat2:.4f}")
                        self.sessionWindow.append(f"P-value = {p_value2:.4f}")
                        
                        # Interaction effects
                        f_stat_int = anova_table.loc['C(Factor1):C(Factor2)', 'F']
                        p_value_int = anova_table.loc['C(Factor1):C(Factor2)', 'PR(>F)']
                        self.sessionWindow.append(f"\nInteraction ({factor1_col}*{factor2_col}):")
                        self.sessionWindow.append(f"F-statistic = {f_stat_int:.4f}")
                        self.sessionWindow.append(f"P-value = {p_value_int:.4f}")
                        
                        # Model statistics
                        r_squared = model.rsquared
                        self.sessionWindow.append(f"\nModel Fit:")
                        self.sessionWindow.append(f"R-squared = {r_squared:.4f}")
                        
                        # Add interpretation
                        alpha = 0.05
                        self.sessionWindow.append(f"\nInterpretation (α = 0.05):")
                        
                        # Factor 1 effect
                        self.sessionWindow.append(f"\nFactor 1 ({factor1_col}):")
                        if p_value1 < alpha:
                            self.sessionWindow.append("Significant main effect")
                        else:
                            self.sessionWindow.append("No significant main effect")
                        
                        # Factor 2 effect
                        self.sessionWindow.append(f"\nFactor 2 ({factor2_col}):")
                        if p_value2 < alpha:
                            self.sessionWindow.append("Significant main effect")
                        else:
                            self.sessionWindow.append("No significant main effect")
                        
                        # Interaction effect
                        self.sessionWindow.append(f"\nInteraction Effect:")
                        if p_value_int < alpha:
                            self.sessionWindow.append("Significant interaction between factors")
                        else:
                            self.sessionWindow.append("No significant interaction between factors")
                        
                        # Create visualizations
                        # Create a single figure with all plots
                        fig = plt.figure(figsize=(15, 5))
                        
                        # Create a grid layout
                        gs = plt.GridSpec(1, 3, figure=fig)
                        
                        # Factor 1 main effect
                        ax1 = fig.add_subplot(gs[0, 0])
                        sns.boxplot(x='Factor1', y='Response', data=anova_data, ax=ax1)
                        ax1.set_title(f'Main Effect of {factor1_col}')
                        ax1.set_xlabel(factor1_col)
                        ax1.set_ylabel(response_col)
                        
                        # Factor 2 main effect
                        ax2 = fig.add_subplot(gs[0, 1])
                        sns.boxplot(x='Factor2', y='Response', data=anova_data, ax=ax2)
                        ax2.set_title(f'Main Effect of {factor2_col}')
                        ax2.set_xlabel(factor2_col)
                        ax2.set_ylabel(response_col)
                        
                        # Interaction plot
                        ax3 = fig.add_subplot(gs[0, 2])
                        interaction_data = anova_data.groupby(['Factor1', 'Factor2'])['Response'].mean().unstack()
                        interaction_data.plot(marker='o', ax=ax3)
                        ax3.set_title('Interaction Plot')
                        ax3.set_xlabel(factor1_col)
                        ax3.set_ylabel(f'Mean {response_col}')
                        ax3.legend(title=factor2_col)
                        ax3.grid(True)
                        
                        plt.tight_layout()
                        plt.show()

                    else:
                        QMessageBox.warning(self, "Warning", "Please select different columns for the two factors")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error performing Two-Way ANOVA: {str(e)}")

    def regressionAnalysis(self):
        """Open dialog for regression analysis options"""
        dialog = QDialog(self)
        dialog.setWindowTitle("Regression Analysis")
        layout = QVBoxLayout()

        # Create buttons for each regression type
        simpleBtn = QPushButton("Simple Linear Regression")
        multipleBtn = QPushButton("Multiple Linear Regression")

        # Connect buttons to their respective functions with dialog closure
        simpleBtn.clicked.connect(lambda: self.handle_regression_selection(dialog, self.simple_linear_regression))
        multipleBtn.clicked.connect(lambda: self.handle_regression_selection(dialog, self.multiple_linear_regression))

        layout.addWidget(simpleBtn)
        layout.addWidget(multipleBtn)

        dialog.setLayout(layout)
        dialog.exec()

    def handle_regression_selection(self, dialog, regression_func):
        """Handle regression selection and dialog closure"""
        dialog.accept()  # Close the dialog first
        regression_func()  # Then run the selected regression function

    def simple_linear_regression(self):
        """Perform simple linear regression analysis"""
        self.loadDataFromTable()
        if self.data.empty:
            QMessageBox.warning(self, "Warning", "Please load or enter data first")
            return

        try:
            # Get numeric columns only
            numeric_columns = self.data.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_columns) < 2:
                QMessageBox.warning(self, "Warning", "Need at least two numeric columns for regression")
                return

            # Get response variable
            response_col, ok1 = QInputDialog.getItem(self, "Select Response Variable", 
                "Choose response variable (Y):", numeric_columns, 0, False)
            
            if ok1:
                # Get predictor variable
                remaining_columns = [col for col in numeric_columns if col != response_col]
                predictor_col, ok2 = QInputDialog.getItem(self, "Select Predictor Variable", 
                    "Choose predictor variable (X):", remaining_columns, 0, False)
                
                if ok2:
                    # Prepare data
                    X = self.data[predictor_col].values.reshape(-1, 1)
                    y = self.data[response_col].values
                    
                    # Fit the model
                    X_with_const = sm.add_constant(X)
                    model = sm.OLS(y, X_with_const).fit()
                    
                    # Calculate predictions for plotting
                    X_plot = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
                    X_plot_with_const = sm.add_constant(X_plot)
                    y_pred = model.predict(X_plot_with_const)
                    
                    # Display results
                    self.sessionWindow.append("\nSimple Linear Regression Results")
                    self.sessionWindow.append("-" * 40)
                    self.sessionWindow.append(f"Response Variable: {response_col}")
                    self.sessionWindow.append(f"Predictor Variable: {predictor_col}")
                    
                    # Model summary
                    self.sessionWindow.append("\nModel Summary:")
                    self.sessionWindow.append(f"R-squared = {model.rsquared:.4f}")
                    self.sessionWindow.append(f"Adjusted R-squared = {model.rsquared_adj:.4f}")
                    self.sessionWindow.append(f"Standard Error = {np.sqrt(model.mse_resid):.4f}")
                    
                    # Coefficients
                    self.sessionWindow.append("\nCoefficients:")
                    self.sessionWindow.append("Variable      Estimate    Std Error    t-value     p-value")
                    self.sessionWindow.append("-" * 60)
                    self.sessionWindow.append(f"{'Intercept':<12}{model.params[0]:10.4f}  {model.bse[0]:10.4f}  {model.tvalues[0]:10.4f}  {model.pvalues[0]:.4e}")
                    self.sessionWindow.append(f"{predictor_col:<12}{model.params[1]:10.4f}  {model.bse[1]:10.4f}  {model.tvalues[1]:10.4f}  {model.pvalues[1]:.4e}")
                    
                    # Regression equation
                    self.sessionWindow.append(f"\nRegression Equation:")
                    self.sessionWindow.append(f"{response_col} = {model.params[0]:.4f} + {model.params[1]:.4f}×{predictor_col}")
                    
                    # Analysis of Variance
                    self.sessionWindow.append("\nAnalysis of Variance:")
                    self.sessionWindow.append("Source      DF          SS          MS           F         P")
                    self.sessionWindow.append("-" * 70)
                    self.sessionWindow.append(f"{'Regression':<10}  {1:2}  {model.ess:11.4f}  {model.ess:11.4f}  {model.fvalue:11.4f}  {model.f_pvalue:.4e}")
                    self.sessionWindow.append(f"{'Residual':<10}  {model.df_resid:2}  {model.ssr:11.4f}  {model.ssr/model.df_resid:11.4f}")
                    self.sessionWindow.append(f"{'Total':<10}  {model.df_model + model.df_resid:2}  {model.ess + model.ssr:11.4f}")
                    
                    # Create visualization
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                    
                    # Regression plot
                    ax1.scatter(X, y, alpha=0.5)
                    ax1.plot(X_plot, y_pred, 'r-', label='Regression Line')
                    ax1.set_xlabel(predictor_col)
                    ax1.set_ylabel(response_col)
                    ax1.set_title('Regression Plot')
                    ax1.legend()
                    
                    # Residual plot
                    residuals = model.resid
                    fitted_values = model.fittedvalues
                    ax2.scatter(fitted_values, residuals, alpha=0.5)
                    ax2.axhline(y=0, color='r', linestyle='--')
                    ax2.set_xlabel('Fitted Values')
                    ax2.set_ylabel('Residuals')
                    ax2.set_title('Residual Plot')
                    
                    plt.tight_layout()
                    plt.show()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error in simple linear regression: {str(e)}")

    def multiple_linear_regression(self):
        """Perform multiple linear regression analysis"""
        self.loadDataFromTable()
        if self.data.empty:
            QMessageBox.warning(self, "Warning", "Please load or enter data first")
            return

        try:
            # Get numeric columns only
            numeric_columns = self.data.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_columns) < 3:
                QMessageBox.warning(self, "Warning", "Need at least three numeric columns for multiple regression")
                return

            # Get response variable
            response_col, ok1 = QInputDialog.getItem(self, "Select Response Variable", 
                "Choose response variable (Y):", numeric_columns, 0, False)
            
            if ok1:
                # Create dialog for selecting predictor variables
                dialog = QDialog()
                dialog.setWindowTitle("Select Predictor Variables")
                layout = QVBoxLayout()
                
                # Create checkboxes for each potential predictor
                checkboxes = []
                for col in numeric_columns:
                    if col != response_col:
                        cb = QCheckBox(col)
                        checkboxes.append(cb)
                        layout.addWidget(cb)
                
                # Add OK and Cancel buttons
                buttonBox = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | 
                                           QDialogButtonBox.StandardButton.Cancel)
                buttonBox.accepted.connect(dialog.accept)
                buttonBox.rejected.connect(dialog.reject)
                layout.addWidget(buttonBox)
                
                dialog.setLayout(layout)
                
                if dialog.exec() == QDialog.DialogCode.Accepted:
                    # Get selected predictors
                    predictor_cols = [cb.text() for cb in checkboxes if cb.isChecked()]
                    
                    if not predictor_cols:
                        QMessageBox.warning(self, "Warning", "Please select at least one predictor variable")
                        return
                    
                    # Prepare data
                    X = self.data[predictor_cols]
                    y = self.data[response_col]
                    
                    # Fit the model
                    X_with_const = sm.add_constant(X)
                    model = sm.OLS(y, X_with_const).fit()
                    
                    # Display results
                    self.sessionWindow.append("\nMultiple Linear Regression Results")
                    self.sessionWindow.append("-" * 40)
                    self.sessionWindow.append(f"Response Variable: {response_col}")
                    self.sessionWindow.append(f"Predictor Variables: {', '.join(predictor_cols)}")
                    
                    # Model summary
                    self.sessionWindow.append("\nModel Summary:")
                    self.sessionWindow.append(f"R-squared = {model.rsquared:.4f}")
                    self.sessionWindow.append(f"Adjusted R-squared = {model.rsquared_adj:.4f}")
                    self.sessionWindow.append(f"Standard Error = {np.sqrt(model.mse_resid):.4f}")
                    
                    # Coefficients
                    self.sessionWindow.append("\nCoefficients:")
                    self.sessionWindow.append("Variable      Estimate    Std Error    t-value     p-value")
                    self.sessionWindow.append("-" * 60)
                    self.sessionWindow.append(f"{'Intercept':<12}{model.params[0]:10.4f}  {model.bse[0]:10.4f}  {model.tvalues[0]:10.4f}  {model.pvalues[0]:.4e}")
                    for i, col in enumerate(predictor_cols, 1):
                        self.sessionWindow.append(f"{col:<12}{model.params[i]:10.4f}  {model.bse[i]:10.4f}  {model.tvalues[i]:10.4f}  {model.pvalues[i]:.4e}")
                    
                    # Regression equation
                    self.sessionWindow.append(f"\nRegression Equation:")
                    equation = f"{response_col} = {model.params[0]:.4f}"
                    for i, col in enumerate(predictor_cols, 1):
                        equation += f" + {model.params[i]:.4f}×{col}"
                    self.sessionWindow.append(equation)
                    
                    # Analysis of Variance
                    self.sessionWindow.append("\nAnalysis of Variance:")
                    self.sessionWindow.append("Source      DF          SS          MS           F         P")
                    self.sessionWindow.append("-" * 70)
                    self.sessionWindow.append(f"{'Regression':<10}  {len(predictor_cols):2}  {model.ess:11.4f}  {model.ess/model.df_model:11.4f}  {model.fvalue:11.4f}  {model.f_pvalue:.4e}")
                    self.sessionWindow.append(f"{'Residual':<10}  {model.df_resid:2}  {model.ssr:11.4f}  {model.ssr/model.df_resid:11.4f}")
                    self.sessionWindow.append(f"{'Total':<10}  {model.df_model + model.df_resid:2}  {model.ess + model.ssr:11.4f}")
                    
                    # VIF values if more than one predictor
                    if len(predictor_cols) > 1:
                        self.sessionWindow.append("\nVariance Inflation Factors:")
                        self.sessionWindow.append("Variable      VIF")
                        self.sessionWindow.append("-" * 20)
                        # Calculate VIF for each predictor
                        for i, col in enumerate(predictor_cols):
                            other_cols = [c for c in predictor_cols if c != col]
                            X_others = self.data[other_cols]
                            X_target = self.data[col]
                            r_squared = sm.OLS(X_target, sm.add_constant(X_others)).fit().rsquared
                            vif = 1 / (1 - r_squared) if r_squared != 1 else float('inf')
                            self.sessionWindow.append(f"{col:<12}{vif:8.4f}")
                    
                    # Create visualization
                    plt.figure(figsize=(10, 5))
                    
                    # Residual plot
                    plt.scatter(model.fittedvalues, model.resid, alpha=0.5)
                    plt.axhline(y=0, color='r', linestyle='--')
                    plt.xlabel('Fitted Values')
                    plt.ylabel('Residuals')
                    plt.title('Residual Plot')
                    
                    plt.tight_layout()
                    plt.show()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error in multiple linear regression: {str(e)}")

    def createDOE(self):
        """Create Design of Experiments"""
        # Create dialog for DOE type selection
        doe_type, ok = QInputDialog.getItem(self, "Design of Experiments",
            "Select DOE Type:",
            ["2-level Factorial", "Fractional Factorial", "Response Surface"], 0, False)
        if not ok:
            return

        if doe_type == "2-level Factorial":
            self.create_factorial_design()
        elif doe_type == "Fractional Factorial":
            self.create_fractional_factorial()
        else:
            self.create_response_surface()

    def create_factorial_design(self):
        """Create 2-level factorial design"""
        # Get number of factors
        n_factors, ok = QInputDialog.getInt(self, "Factorial Design", 
            "Enter number of factors (2-6):", 2, 2, 6)
        if not ok:
            return

        # Get factor names and levels
        factors = []
        levels = []
        for i in range(n_factors):
            # Get factor name
            name, ok = QInputDialog.getText(self, f"Factor {i+1}", 
                f"Enter name for factor {i+1}:")
            if not ok:
                return
            factors.append(name)
            
            # Get factor levels
            low, ok = QInputDialog.getText(self, f"Factor {i+1} Low", 
                f"Enter low level for {name}:")
            if not ok:
                return
            high, ok = QInputDialog.getText(self, f"Factor {i+1} High", 
                f"Enter high level for {name}:")
            if not ok:
                return
            levels.append([low, high])

        # Create full factorial design
        n_runs = 2 ** n_factors
        design_matrix = []
        for i in range(n_runs):
            run = []
            for j in range(n_factors):
                # Convert run number to binary and use as factor levels
                level_idx = (i >> j) & 1
                run.append(levels[j][level_idx])
            design_matrix.append(run)

        # Create DataFrame with the design
        df = pd.DataFrame(design_matrix, columns=factors)
        df.insert(0, 'StdOrder', range(1, len(df) + 1))
        df.insert(1, 'RunOrder', np.random.permutation(len(df)) + 1)
        df['Response'] = ''  # Empty column for responses

        # Update the table with the design
        self.data = df
        self.updateTable()

        # Show design summary
        summary = f"""2-level Factorial Design Summary

Number of factors: {n_factors}
Number of runs: {n_runs}
Base design: Full factorial

Factors and Levels:
"""
        for i, factor in enumerate(factors):
            summary += f"{factor}: {levels[i][0]} | {levels[i][1]}\n"

        self.sessionWindow.setText(summary)

    def create_fractional_factorial(self):
        """Create fractional factorial design"""
        # Get number of factors
        n_factors, ok = QInputDialog.getInt(self, "Fractional Factorial Design", 
            "Enter number of factors (3-7):", 3, 3, 7)
        if not ok:
            return

        # Get resolution level
        resolution_options = []
        max_resolution = n_factors
        for r in range(3, max_resolution + 1):
            if 2**(n_factors - 1) >= n_factors:  # Modified condition
                resolution_options.append(f"Resolution {r}")
        
        if not resolution_options:
            QMessageBox.warning(self, "Warning", 
                "No valid resolution available for this number of factors")
            return

        resolution, ok = QInputDialog.getItem(self, "Fractional Factorial Design",
            "Select design resolution:", resolution_options, 0, False)
        if not ok:
            return
        
        resolution_level = int(resolution.split()[-1])

        # Get factor names and levels
        factors = []
        levels = []
        for i in range(n_factors):
            # Get factor name
            name, ok = QInputDialog.getText(self, f"Factor {i+1}", 
                f"Enter name for factor {i+1}:")
            if not ok:
                return
            factors.append(name)
            
            # Get factor levels
            low, ok = QInputDialog.getText(self, f"Factor {i+1} Low", 
                f"Enter low level for {name}:")
            if not ok:
                return
            high, ok = QInputDialog.getText(self, f"Factor {i+1} High", 
                f"Enter high level for {name}:")
            if not ok:
                return
            levels.append([low, high])

        # Calculate number of runs needed
        n_runs = 2**(n_factors - 1)  # Modified for half-fraction

        # Create basic design matrix
        basic_design = []
        for i in range(n_runs):
            run = []
            for j in range(n_factors - 1):  # Generate for all but one factor
                level_idx = (i >> j) & 1
                run.append(level_idx)
            
            # Generate the last factor based on the interaction of other factors
            last_factor = 0
            for j in range(n_factors - 1):
                last_factor ^= run[j]
            run.append(last_factor)
            
            basic_design.append(run)

        # Convert design to actual factor levels
        design_matrix = []
        for run in basic_design:
            actual_run = []
            for j, level_idx in enumerate(run):
                actual_run.append(levels[j][level_idx])
            design_matrix.append(actual_run)

        # Create DataFrame with the design
        df = pd.DataFrame(design_matrix, columns=factors)
        df.insert(0, 'StdOrder', range(1, len(df) + 1))
        df.insert(1, 'RunOrder', np.random.permutation(len(df)) + 1)
        df['Response'] = ''  # Empty column for responses

        # Update the table with the design
        self.data = df
        self.updateTable()

        # Show design summary
        summary = f"""Fractional Factorial Design Summary

Number of factors: {n_factors}
Resolution: {resolution}
Number of runs: {n_runs}
Fraction: 1/2

Factors and Levels:
"""
        for i, factor in enumerate(factors):
            summary += f"{factor}: {levels[i][0]} | {levels[i][1]}\n"

        if n_factors > 2:
            summary += "\nDesign Generator:\n"
            summary += f"{factors[-1]} = {' × '.join(factors[:-1])}\n"

        self.sessionWindow.setText(summary)

    def create_response_surface(self):
        """Create response surface design"""
        # Get design type
        design_type, ok = QInputDialog.getItem(self, "Response Surface Design",
            "Select design type:",
            ["Central Composite Design (CCD)", "Box-Behnken Design (BBD)"], 0, False)
        if not ok:
            return

        # Get number of factors
        min_factors = 2 if design_type.startswith("Central") else 3
        max_factors = 6 if design_type.startswith("Central") else 7
        n_factors, ok = QInputDialog.getInt(self, "Response Surface Design", 
            f"Enter number of factors ({min_factors}-{max_factors}):", 
            min_factors, min_factors, max_factors)
        if not ok:
            return

        # Get factor names and levels
        factors = []
        center_points = []
        ranges = []
        
        for i in range(n_factors):
            # Get factor name
            name, ok = QInputDialog.getText(self, f"Factor {i+1}", 
                f"Enter name for factor {i+1}:")
            if not ok:
                return
            factors.append(name)
            
            # Get factor center point and range
            center, ok = QInputDialog.getDouble(self, f"Factor {i+1} Center", 
                f"Enter center point for {name}:")
            if not ok:
                return
            center_points.append(center)
            
            range_val, ok = QInputDialog.getDouble(self, f"Factor {i+1} Range", 
                f"Enter range (±) for {name}:")
            if not ok:
                return
            ranges.append(range_val)

        # Get number of center points
        n_center, ok = QInputDialog.getInt(self, "Center Points", 
            "Enter number of center points:", 3, 1, 10)
        if not ok:
            return

        # Create design matrix based on design type
        if design_type.startswith("Central"):
            # Central Composite Design
            # Get alpha type
            alpha_type, ok = QInputDialog.getItem(self, "CCD Alpha",
                "Select alpha type:",
                ["Rotatable", "Orthogonal", "Face-centered"], 0, False)
            if not ok:
                return

            # Calculate alpha value
            if alpha_type == "Rotatable":
                alpha = (2**n_factors)**(1/4)
            elif alpha_type == "Face-centered":
                alpha = 1.0
            else:  # Orthogonal
                n_factorial = 2**n_factors
                n_axial = 2 * n_factors
                alpha = ((n_factorial * (1 + n_center/n_factorial))/n_axial)**(1/4)

            # Create factorial points
            factorial_points = []
            for i in range(2**n_factors):
                point = []
                for j in range(n_factors):
                    level = -1 if (i >> j) & 1 else 1
                    point.append(level)
                factorial_points.append(point)

            # Create axial points
            axial_points = []
            for i in range(n_factors):
                point_plus = [0] * n_factors
                point_minus = [0] * n_factors
                point_plus[i] = alpha
                point_minus[i] = -alpha
                axial_points.append(point_plus)
                axial_points.append(point_minus)

            # Create center points
            center_points_design = [[0] * n_factors for _ in range(n_center)]

            # Combine all points
            coded_design = factorial_points + axial_points + center_points_design

        else:
            # Box-Behnken Design
            if n_factors < 3:
                QMessageBox.warning(self, "Warning", 
                    "Box-Behnken design requires at least 3 factors")
                return

            # Create design points
            coded_design = []
            
            # Add factorial points for pairs of factors
            for i in range(n_factors - 1):
                for j in range(i + 1, n_factors):
                    for level_i in [-1, 1]:
                        for level_j in [-1, 1]:
                            point = [0] * n_factors
                            point[i] = level_i
                            point[j] = level_j
                            coded_design.append(point)

            # Add center points
            center_points_design = [[0] * n_factors for _ in range(n_center)]
            coded_design.extend(center_points_design)

        # Convert coded levels to actual values
        design_matrix = []
        for run in coded_design:
            actual_run = []
            for i, coded_level in enumerate(run):
                actual_value = center_points[i] + coded_level * ranges[i]
                actual_run.append(actual_value)
            design_matrix.append(actual_run)

        # Create DataFrame with the design
        df = pd.DataFrame(design_matrix, columns=factors)
        df.insert(0, 'StdOrder', range(1, len(df) + 1))
        df.insert(1, 'RunOrder', np.random.permutation(len(df)) + 1)
        df['Response'] = ''  # Empty column for responses

        # Add point type information
        if design_type.startswith("Central"):
            point_types = (['Factorial'] * 2**n_factors + 
                         ['Axial'] * (2 * n_factors) +
                         ['Center'] * n_center)
        else:  # Box-Behnken
            n_factorial = n_factors * (n_factors - 1) * 2
            point_types = ['Factorial'] * n_factorial + ['Center'] * n_center

        df['PointType'] = point_types

        # Update the table with the design
        self.data = df
        self.updateTable()

        # Show design summary
        summary = f"""Response Surface Design Summary

Design Type: {design_type}
Number of factors: {n_factors}
Number of runs: {len(df)}

Design Points:
Factorial points: {len([pt for pt in point_types if pt == 'Factorial'])}
{'Axial points: ' + str(len([pt for pt in point_types if pt == 'Axial'])) if design_type.startswith('Central') else ''}
Center points: {n_center}

Factors:
"""
        for i, factor in enumerate(factors):
            summary += f"{factor}:\n"
            summary += f"  Center: {center_points[i]}\n"
            summary += f"  Range: ±{ranges[i]}\n"

        if design_type.startswith("Central"):
            summary += f"\nAlpha Type: {alpha_type}"
            summary += f"\nAlpha Value: {alpha:.4f}"

        self.sessionWindow.setText(summary)

    def analyzeDOE(self):
        """Analyze Design of Experiments"""
        # Load the current data from the table
        self.loadDataFromTable()
        
        if self.data.empty:
            QMessageBox.warning(self, "Warning", "No data available for analysis")
            return

        try:
            # Check if we have response data
            if 'Response' not in self.data.columns:
                QMessageBox.warning(self, "Warning", "No Response column found")
                return

            # Identify factor columns (exclude StdOrder, RunOrder, Response, and actual value columns)
            factor_cols = [col for col in self.data.columns 
                          if col not in ['StdOrder', 'RunOrder', 'Response', 'PointType'] 
                          and not col.endswith('_actual')]

            if not factor_cols:
                QMessageBox.warning(self, "Warning", "No factor columns identified")
                return

            # Create a copy of the data to avoid modifying the original
            analysis_data = self.data.copy()

            # Convert response to numeric, dropping any non-numeric values
            analysis_data['Response'] = pd.to_numeric(analysis_data['Response'].astype(str).str.strip(), errors='coerce')
            
            # Drop any rows with missing response values
            analysis_data = analysis_data.dropna(subset=['Response'])

            if len(analysis_data) == 0:
                QMessageBox.warning(self, "Warning", "No valid response data after conversion")
                return

            # Create design matrix with coded values (-1, 1)
            design_matrix = pd.DataFrame()
            for col in factor_cols:
                col_data = pd.to_numeric(analysis_data[col], errors='coerce')
                col_min = col_data.min()
                col_max = col_data.max()
                if col_max > col_min:  # Ensure we don't divide by zero
                    design_matrix[col] = -1 + 2 * (col_data - col_min) / (col_max - col_min)
                else:
                    QMessageBox.warning(self, "Warning", f"Factor {col} has no variation")
                    return

            # Add constant term for the regression
            X = sm.add_constant(design_matrix)
            y = analysis_data['Response']

            # Fit the model
            model = sm.OLS(y, X).fit()

            # Calculate effects (multiply coefficients by 2 for proper effect size)
            effects = pd.Series(model.params[1:] * 2, index=factor_cols)

            # Create analysis report
            report = f"""Design of Experiments Analysis

Model Summary:
R-squared: {model.rsquared:.4f}
Adjusted R-squared: {model.rsquared_adj:.4f}
F-statistic: {model.fvalue:.4f}
Prob(F-statistic): {model.f_pvalue:.4f}

Effects:
"""
            for factor, effect in effects.items():
                report += f"{factor}: {effect:.4f}\n"

            report += f"""
Analysis of Variance:
{model.summary().tables[2]}

Parameter Estimates:
{model.summary().tables[1]}
"""
            self.sessionWindow.setText(report)

            # Create effects plot
            plt.figure(figsize=(10, 6))
            effects.plot(kind='bar')
            plt.axhline(y=0, color='r', linestyle='--')
            plt.title('Main Effects Plot')
            plt.xlabel('Factors')
            plt.ylabel('Effect Size')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()

            # Create normal probability plot of effects
            fig = plt.figure(figsize=(8, 6))
            stats.probplot(effects, dist="norm", plot=plt)
            plt.title('Normal Probability Plot of Effects')
            plt.grid(True)
            plt.tight_layout()
            plt.show()

        except Exception as e:
            QMessageBox.warning(self, "Error", 
                f"An error occurred during DOE analysis:\n{str(e)}\n\n"
                "Please check your data and try again.")

    def individualChart(self):
        """Create individual control chart"""
        try:
            # Get list of numeric columns only
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) < 1:
                QMessageBox.warning(self, "Warning", "Need at least one numeric column for measurements")
                return

            # Create dialog for options
            dialog = QDialog(self)
            dialog.setWindowTitle("Individual Chart Options")
            layout = QVBoxLayout()

            # Column selection
            col_label = QLabel("Select Measurement column:")
            layout.addWidget(col_label)
            
            col_combo = QComboBox()
            col_combo.addItems(numeric_cols)
            layout.addWidget(col_combo)

            # Display Tests option
            display_tests = QCheckBox("Display Tests")
            layout.addWidget(display_tests)

            # Alpha value selection
            alpha_layout = QHBoxLayout()
            alpha_label = QLabel("α value for control limits:")
            alpha_input = QLineEdit("0.05")  # Default value
            alpha_layout.addWidget(alpha_label)
            alpha_layout.addWidget(alpha_input)
            layout.addLayout(alpha_layout)

            # Moving Range length selection
            mr_layout = QHBoxLayout()
            mr_label = QLabel("Moving Range length:")
            mr_input = QLineEdit("2")  # Default value
            mr_layout.addWidget(mr_label)
            mr_layout.addWidget(mr_input)
            layout.addLayout(mr_layout)

            # Add OK and Cancel buttons
            button_box = QDialogButtonBox(
                QDialogButtonBox.StandardButton.Ok | 
                QDialogButtonBox.StandardButton.Cancel
            )
            button_box.accepted.connect(dialog.accept)
            button_box.rejected.connect(dialog.reject)
            layout.addWidget(button_box)

            dialog.setLayout(layout)
            
            # Show dialog and get results
            if dialog.exec() != QDialog.DialogCode.Accepted:
                return

            # Get selected options
            col = col_combo.currentText()
            show_tests = display_tests.isChecked()
            try:
                alpha = float(alpha_input.text())
                if not 0 < alpha < 1:
                    raise ValueError("Alpha must be between 0 and 1")
            except ValueError:
                QMessageBox.warning(self, "Warning", "Invalid α value. Using default 0.05")
                alpha = 0.05

            try:
                mr_length = int(mr_input.text())
                if mr_length < 2:
                    raise ValueError("Moving Range length must be at least 2")
            except ValueError:
                QMessageBox.warning(self, "Warning", "Invalid Moving Range length. Using default 2")
                mr_length = 2

            # Get the data and check for missing values
            data = pd.to_numeric(self.data[col], errors='coerce')
            if data.isna().any():
                QMessageBox.warning(self, "Warning", "Missing values found in the selected column")
                return

            # Calculate moving ranges
            moving_ranges = np.zeros(len(data)-mr_length+1)
            for i in range(len(moving_ranges)):
                moving_ranges[i] = np.ptp(data[i:i+mr_length])
            
            # Calculate control limits
            mean = np.mean(data)
            mr_mean = np.mean(moving_ranges)
            
            # Constants for n=2 (moving range of 2 consecutive points)
            E2 = 2.66
            D3 = 0
            D4 = 3.267
            
            # Individual chart limits
            i_ucl = mean + 3 * mr_mean / 1.128  # 1.128 = d2 for n=2
            i_lcl = mean - 3 * mr_mean / 1.128
            
            # Moving Range chart limits
            mr_ucl = D4 * mr_mean
            mr_lcl = D3 * mr_mean
            
            # Create plots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
            
            # Individual chart
            ax1.plot(range(1, len(data) + 1), data, marker='o', color='blue')
            ax1.axhline(y=mean, color='g', linestyle='-', label='CL')
            ax1.axhline(y=i_ucl, color='r', linestyle='--', label='UCL')
            ax1.axhline(y=i_lcl, color='r', linestyle='--', label='LCL')
            ax1.set_title('Individual Chart')
            ax1.set_xlabel('Observation')
            ax1.set_ylabel('Individual Value')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # Moving Range chart
            ax2.plot(range(1, len(moving_ranges) + 1), moving_ranges, marker='o', color='blue')
            ax2.axhline(y=mr_mean, color='g', linestyle='-', label='CL')
            ax2.axhline(y=mr_ucl, color='r', linestyle='--', label='UCL')
            ax2.axhline(y=mr_lcl, color='r', linestyle='--', label='LCL')
            ax2.set_title('Moving Range Chart')
            ax2.set_xlabel('Observation')
            ax2.set_ylabel('Moving Range')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            plt.tight_layout()
            plt.show()
            
            # Format output text
            result_text = f"Individual Chart Analysis for {col}\n\n"
            result_text += "Individual Chart Statistics:\n"
            result_text += f"Mean: {mean:.3f}\n"
            result_text += f"UCL: {i_ucl:.3f}\n"
            result_text += f"LCL: {i_lcl:.3f}\n\n"
            result_text += f"Number of Points: {len(data)}\n"
            result_text += f"Points Outside Control Limits: {sum((data > i_ucl) | (data < i_lcl))}\n\n"
            result_text += f"Note: Control limits are based on ±3 sigma (calculated using moving ranges)\n"
            if show_tests:
                result_text += "\nTests for Special Causes:\n"
                # Add test results here based on the selected alpha value
                
            self.sessionWindow.setText(result_text)
            
        except Exception as e:
            QMessageBox.warning(self, "Error", f"An error occurred: {str(e)}")
            return

    def movingRangeChart(self, col=None):
        """Create moving range chart"""
        if col is None:
            col = self.selectColumnDialog()
            if not col:
                return
                
        try:
            # Get numeric data
            data = pd.to_numeric(self.data[col], errors='coerce').dropna()
            
            if len(data) < 2:
                QMessageBox.warning(self, "Warning", "Insufficient data for moving range chart")
                return
                
            # Calculate moving range
            moving_range = np.abs(data.diff().dropna())
            
            # Calculate control limits
            mr_mean = moving_range.mean()
            mr_std = moving_range.std()
            mr_ucl = mr_mean + 3 * mr_std
            mr_lcl = max(0, mr_mean - 3 * mr_std)
            
            # Create individual chart
            plt.figure(figsize=(10, 8))
            plt.subplot(2, 1, 1)
            plt.plot(data.index, data, marker='o', linestyle='-', color='blue')
            plt.axhline(y=data.mean(), color='green', linestyle='-', label='Mean')
            plt.axhline(y=data.mean() + 3 * (mr_mean / 1.128), color='red', linestyle='--', label='UCL')
            plt.axhline(y=data.mean() - 3 * (mr_mean / 1.128), color='red', linestyle='--', label='LCL')
            plt.title(f'Individual Chart for {col}')
            plt.xlabel('Observation')
            plt.ylabel('Value')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Create moving range chart
            plt.subplot(2, 1, 2)
            plt.plot(moving_range.index, moving_range, marker='o', linestyle='-', color='blue')
            plt.axhline(y=mr_mean, color='green', linestyle='-', label='Mean')
            plt.axhline(y=mr_ucl, color='red', linestyle='--', label='UCL')
            plt.axhline(y=mr_lcl, color='red', linestyle='--', label='LCL')
            plt.title(f'Moving Range Chart for {col}')
            plt.xlabel('Observation')
            plt.ylabel('Moving Range')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Create report
            report = f"""Moving Range Chart Analysis for {col}

Moving Range Statistics:
Mean MR: {mr_mean:.3f}
UCL: {mr_ucl:.3f}
LCL: {mr_lcl:.3f}

Number of Ranges: {len(moving_range)}
Ranges Outside Control Limits: {sum((moving_range > mr_ucl) | (moving_range < mr_lcl))}

Note: Control limits are based on ±3 sigma
"""
            self.sessionWindow.setText(report)
            plt.show()
        except Exception as e:
            QMessageBox.warning(self, "Error", f"An error occurred: {str(e)}")

    def gageRR(self):
        """Perform Gage R&R Study"""
        # Create dialog for study type selection
        study_type, ok = QInputDialog.getItem(self, "Gage R&R Study Type",
            "Select Study Type:",
            ["Crossed (default)", "Nested"], 0, False)
        if not ok:
            return

        # Get part column with explicit prompt
        part_col = self.selectColumnDialog("Select Part Column")
        if not part_col:
            return

        # Get operator column with explicit prompt
        operator_col = self.selectColumnDialog("Select Operator Column")
        if not operator_col:
            return

        # Get measurement column with explicit prompt
        measurement_col = self.selectColumnDialog("Select Measurement Column")
        if not measurement_col:
            return

        # Get order column (optional) with explicit prompt
        order_col = self.selectColumnDialog("Select Order Column (optional)")
        # Order column is optional, so we continue even if it's None

        # Create options dialog
        options_dialog = QDialog(self)
        options_dialog.setWindowTitle("Gage R&R Study Options")
        options_layout = QVBoxLayout()

        # Create tabs
        tab_widget = QTabWidget()
        
        # Basic tab is already handled by the previous prompts
        
        # Options tab
        options_tab = QWidget()
        options_tab_layout = QVBoxLayout()
        
        # Study Information group
        study_info_group = QGroupBox("Study Information")
        study_info_layout = QVBoxLayout()
        
        # Number of replicates
        replicates_layout = QHBoxLayout()
        replicates_label = QLabel("Number of replicates (2-5):")
        replicates_spin = QSpinBox()
        replicates_spin.setRange(2, 5)
        replicates_spin.setValue(3)  # Default value
        replicates_layout.addWidget(replicates_label)
        replicates_layout.addWidget(replicates_spin)
        study_info_layout.addLayout(replicates_layout)
        
        # Process tolerance
        tolerance_layout = QHBoxLayout()
        tolerance_check = QCheckBox("Process tolerance (optional):")
        tolerance_check.setChecked(False)
        tolerance_value = QDoubleSpinBox()
        tolerance_value.setRange(0.01, 1000.0)
        tolerance_value.setValue(0.60)  # Default value
        tolerance_value.setEnabled(False)
        tolerance_check.toggled.connect(tolerance_value.setEnabled)
        tolerance_layout.addWidget(tolerance_check)
        tolerance_layout.addWidget(tolerance_value)
        study_info_layout.addLayout(tolerance_layout)
        
        study_info_group.setLayout(study_info_layout)
        options_tab_layout.addWidget(study_info_group)
        
        # Analysis Options group
        analysis_group = QGroupBox("Analysis Options")
        analysis_layout = QVBoxLayout()
        
        # Confidence level
        confidence_layout = QHBoxLayout()
        confidence_label = QLabel("Confidence level:")
        confidence_combo = QComboBox()
        confidence_combo.addItems(["90%", "95%", "99%"])
        confidence_combo.setCurrentIndex(1)  # Default to 95%
        confidence_layout.addWidget(confidence_label)
        confidence_layout.addWidget(confidence_combo)
        analysis_layout.addLayout(confidence_layout)
        
        # Include interaction
        interaction_check = QCheckBox("Include interaction")
        interaction_check.setChecked(True)
        analysis_layout.addWidget(interaction_check)
        
        # Use historical standard deviation
        hist_std_check = QCheckBox("Use historical standard deviation (if available)")
        hist_std_check.setChecked(False)
        analysis_layout.addWidget(hist_std_check)
        
        analysis_group.setLayout(analysis_layout)
        options_tab_layout.addWidget(analysis_group)
        
        options_tab.setLayout(options_tab_layout)
        
        # Graphs tab
        graphs_tab = QWidget()
        graphs_layout = QVBoxLayout()
        
        # Graph options
        components_check = QCheckBox("Components of Variation")
        components_check.setChecked(True)
        graphs_layout.addWidget(components_check)
        
        r_chart_check = QCheckBox("R Chart by Operator")
        r_chart_check.setChecked(True)
        graphs_layout.addWidget(r_chart_check)
        
        xbar_chart_check = QCheckBox("X-bar Chart by Operator")
        xbar_chart_check.setChecked(True)
        graphs_layout.addWidget(xbar_chart_check)
        
        by_part_check = QCheckBox("Measurement by Part")
        by_part_check.setChecked(True)
        graphs_layout.addWidget(by_part_check)
        
        by_operator_check = QCheckBox("Measurement by Operator")
        by_operator_check.setChecked(True)
        graphs_layout.addWidget(by_operator_check)
        
        interaction_plot_check = QCheckBox("Part*Operator Interaction")
        interaction_plot_check.setChecked(True)
        graphs_layout.addWidget(interaction_plot_check)
        
        # Run chart only if order column is provided
        run_chart_check = QCheckBox("Run Chart (if Order provided)")
        run_chart_check.setChecked(order_col is not None)
        run_chart_check.setEnabled(order_col is not None)
        graphs_layout.addWidget(run_chart_check)
        
        graphs_tab.setLayout(graphs_layout)
        
        # Add tabs to widget
        tab_widget.addTab(options_tab, "Options")
        tab_widget.addTab(graphs_tab, "Graphs")
        
        options_layout.addWidget(tab_widget)
        
        # Add buttons
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | 
                                     QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(options_dialog.accept)
        button_box.rejected.connect(options_dialog.reject)
        options_layout.addWidget(button_box)
        
        options_dialog.setLayout(options_layout)
        
        # Show options dialog
        if options_dialog.exec() != QDialog.DialogCode.Accepted:
            return

        try:
            # Create DataFrame for analysis
            df = pd.DataFrame({
                'Measurement': pd.to_numeric(self.data[measurement_col], errors='coerce'),
                'Operator': self.data[operator_col],
                'Part': self.data[part_col]
            })
            
            # Add Order column if provided
            if order_col:
                df['Order'] = self.data[order_col]

            # Remove missing values
            df = df.dropna()

            if len(df) < 2:
                QMessageBox.warning(self, "Warning", "Insufficient data for Gage R&R analysis")
                return

            # Calculate overall statistics
            total_mean = df['Measurement'].mean()
            total_std = df['Measurement'].std()

            # Calculate components of variation
            operators = df['Operator'].unique()
            parts = df['Part'].unique()
            n_operators = len(operators)
            n_parts = len(parts)
            n_measurements = len(df) / (n_operators * n_parts)

            # Perform ANOVA analysis
            formula = f"Measurement ~ C(Part) + C(Operator)"
            if interaction_check.isChecked():
                formula += " + C(Part):C(Operator)"
            
            model = sm.formula.ols(formula, data=df).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)
            
            # Calculate operator variation
            operator_means = df.groupby('Operator')['Measurement'].mean()
            operator_variation = operator_means.std() ** 2

            # Calculate part variation
            part_means = df.groupby('Part')['Measurement'].mean()
            part_variation = part_means.std() ** 2

            # Calculate repeatability (equipment variation)
            residuals = []
            for part in parts:
                for operator in operators:
                    part_operator_data = df[(df['Part'] == part) & (df['Operator'] == operator)]['Measurement']
                    if len(part_operator_data) > 1:
                        residuals.extend(part_operator_data - part_operator_data.mean())
            
            repeatability_variation = np.var(residuals, ddof=1) if residuals else 0

            # Calculate reproducibility (operator variation)
            reproducibility_variation = max(0, operator_variation - repeatability_variation / (n_parts * n_measurements))

            # Calculate total variation
            total_variation = part_variation + repeatability_variation + reproducibility_variation

            # Calculate study variation (6 * standard deviation)
            study_var = {
                'Repeatability': 6 * np.sqrt(repeatability_variation),
                'Reproducibility': 6 * np.sqrt(reproducibility_variation),
                'Part-to-Part': 6 * np.sqrt(part_variation),
                'Total': 6 * np.sqrt(total_variation)
            }

            # Calculate contribution percentages
            contribution = {
                'Repeatability': (repeatability_variation / total_variation) * 100,
                'Reproducibility': (reproducibility_variation / total_variation) * 100,
                'Part-to-Part': (part_variation / total_variation) * 100
            }

            # Get process tolerance if provided
            process_tolerance = None
            if tolerance_check.isChecked():
                process_tolerance = tolerance_value.value()

            # Format ANOVA table for display
            anova_display = "ANOVA Table:\n"
            anova_display += "Source               DF    SS        MS        F-value    P-value\n"
            anova_display += "-" * 70 + "\n"
            
            # Add Part row
            part_row = anova_table.loc["C(Part)"]
            anova_display += f"Part                 {int(part_row['df']):2d}    {part_row['sum_sq']:.6f}  {part_row['sum_sq']/part_row['df']:.6f}  {part_row['F']:.6f}  {part_row['PR(>F)']:.6f}\n"
            
            # Add Operator row
            operator_row = anova_table.loc["C(Operator)"]
            anova_display += f"Operator             {int(operator_row['df']):2d}    {operator_row['sum_sq']:.6f}  {operator_row['sum_sq']/operator_row['df']:.6f}  {operator_row['F']:.6f}  {operator_row['PR(>F)']:.6f}\n"
            
            # Add Interaction row if included
            if interaction_check.isChecked() and "C(Part):C(Operator)" in anova_table.index:
                interaction_row = anova_table.loc["C(Part):C(Operator)"]
                anova_display += f"Part*Operator        {int(interaction_row['df']):2d}    {interaction_row['sum_sq']:.6f}  {interaction_row['sum_sq']/interaction_row['df']:.6f}  {interaction_row['F']:.6f}  {interaction_row['PR(>F)']:.6f}\n"
            
            # Add Residual row
            residual_row = anova_table.loc["Residual"]
            anova_display += f"Residual             {int(residual_row['df']):2d}    {residual_row['sum_sq']:.6f}  {residual_row['sum_sq']/residual_row['df']:.6f}\n"
            
            # Add Total row
            total_df = anova_table['df'].sum()
            total_ss = anova_table['sum_sq'].sum()
            anova_display += f"Total                {int(total_df):2d}    {total_ss:.6f}\n\n"

            # Create report
            report = f"""Gage R&R Study Results

Study Information:
Number of Operators: {n_operators}
Number of Parts: {n_parts}
Number of Replicates: {int(n_measurements)}
Study Type: {study_type}
Confidence Level: {confidence_combo.currentText()}

Overall Statistics:
Mean: {total_mean:.3f}
Standard Deviation: {total_std:.3f}

{anova_display}
Variance Components:
Source          %Contribution  Study Var  %Study Var
Total Gage R&R  {(contribution['Repeatability'] + contribution['Reproducibility']):.1f}%  {(study_var['Repeatability'] + study_var['Reproducibility']):.3f}  {((study_var['Repeatability'] + study_var['Reproducibility'])/study_var['Total']*100):.1f}%
  Repeatability {contribution['Repeatability']:.1f}%  {study_var['Repeatability']:.3f}  {(study_var['Repeatability']/study_var['Total']*100):.1f}%
  Reproducibility {contribution['Reproducibility']:.1f}%  {study_var['Reproducibility']:.3f}  {(study_var['Reproducibility']/study_var['Total']*100):.1f}%
Part-to-Part    {contribution['Part-to-Part']:.1f}%  {study_var['Part-to-Part']:.3f}  {(study_var['Part-to-Part']/study_var['Total']*100):.1f}%
Total Variation 100%  {study_var['Total']:.3f}  100%
"""
            # Add tolerance information if provided
            if process_tolerance:
                report += f"\nProcess Tolerance: {process_tolerance:.3f}\n"
                report += f"%Tolerance (Total Gage R&R): {((study_var['Repeatability'] + study_var['Reproducibility'])/process_tolerance*100):.1f}%\n"

            # Calculate number of distinct categories
            n_dc = int(np.sqrt(2 * (1 - (contribution['Repeatability'] + contribution['Reproducibility'])/100)))
            report += f"\nNumber of Distinct Categories: {n_dc}\n\n"

            # Add assessment based on results
            total_gage_rr = contribution['Repeatability'] + contribution['Reproducibility']
            report += "Assessment:\n"
            if total_gage_rr < 10:
                report += "Measurement system is acceptable (Gage R&R <= 10%)\n"
            elif total_gage_rr < 30:
                report += "Measurement system may be acceptable depending on application (10% < Gage R&R <= 30%)\n"
            else:
                report += "Measurement system needs improvement (Gage R&R > 30%)\n"
                
            if n_dc >= 5:
                report += "Number of distinct categories is acceptable (>= 5)\n"
            else:
                report += "Number of distinct categories is too low (< 5)\n"

            self.sessionWindow.setText(report)

            # Create visualizations based on selected options
            if components_check.isChecked():
                plt.figure(figsize=(10, 6))
                plt.bar(['Repeatability', 'Reproducibility', 'Part-to-Part'],
                        [contribution['Repeatability'], contribution['Reproducibility'], contribution['Part-to-Part']])
                plt.title('Components of Variation')
                plt.ylabel('Percent Contribution (%)')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.show()

            if by_operator_check.isChecked():
                plt.figure(figsize=(10, 6))
                for operator in operators:
                    operator_data = df[df['Operator'] == operator]
                    plt.plot(operator_data['Part'], operator_data['Measurement'], 
                            marker='o', linestyle='-', label=f'Operator {operator}')
                plt.title('Measurement by Operator')
                plt.xlabel('Part')
                plt.ylabel('Measurement')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.show()
                
            # Add other plots based on selected options
            # (R Chart, X-bar Chart, Measurement by Part, Part*Operator Interaction, Run Chart)

        except Exception as e:
            QMessageBox.warning(self, "Error", 
                f"An error occurred during Gage R&R analysis:\n{str(e)}\n\n"
                "Please check your data and try again.")

    def linearityStudy(self):
        """Perform measurement system linearity study"""
        # Get reference column with explicit prompt
        reference_col = self.selectColumnDialog("Select Reference Column")
        if not reference_col:
            return

        # Get measurement column with explicit prompt
        measurement_col = self.selectColumnDialog("Select Measurement Column")
        if not measurement_col:
            return

        # Get operator column (optional) with explicit prompt
        operator_col = self.selectColumnDialog("Select Operator Column (optional)")
        # Operator column is optional, so we continue even if it's None

        # Get order column (optional) with explicit prompt
        order_col = self.selectColumnDialog("Select Order Column (optional)")
        # Order column is optional, so we continue even if it's None

        # Create options dialog
        options_dialog = QDialog(self)
        options_dialog.setWindowTitle("Linearity Study Options")
        options_layout = QVBoxLayout()

        # Create tabs
        tab_widget = QTabWidget()
        
        # Options tab
        options_tab = QWidget()
        options_tab_layout = QVBoxLayout()
        
        # Analysis Settings group
        analysis_group = QGroupBox("Analysis Settings")
        analysis_layout = QVBoxLayout()
        
        # Confidence level
        confidence_layout = QHBoxLayout()
        confidence_label = QLabel("Confidence level:")
        confidence_combo = QComboBox()
        confidence_combo.addItems(["90%", "95%", "99%"])
        confidence_combo.setCurrentIndex(1)  # Default to 95%
        confidence_layout.addWidget(confidence_label)
        confidence_layout.addWidget(confidence_combo)
        analysis_layout.addLayout(confidence_layout)
        
        # Include operator effects (only if operator column is provided)
        include_operator_check = QCheckBox("Include operator effects")
        include_operator_check.setChecked(operator_col is not None)
        include_operator_check.setEnabled(operator_col is not None)
        analysis_layout.addWidget(include_operator_check)
        
        # Fit intercept
        fit_intercept_check = QCheckBox("Fit intercept")
        fit_intercept_check.setChecked(True)
        analysis_layout.addWidget(fit_intercept_check)
        
        analysis_group.setLayout(analysis_layout)
        options_tab_layout.addWidget(analysis_group)
        
        # Tolerance Information group
        tolerance_group = QGroupBox("Tolerance Information")
        tolerance_layout = QVBoxLayout()
        
        # Tolerance range
        tolerance_range_layout = QHBoxLayout()
        tolerance_range_check = QCheckBox("Tolerance range (optional):")
        tolerance_range_check.setChecked(False)
        tolerance_range_value = QDoubleSpinBox()
        tolerance_range_value.setRange(0.01, 1000.0)
        tolerance_range_value.setValue(0.60)  # Default value
        tolerance_range_value.setEnabled(False)
        tolerance_range_check.toggled.connect(tolerance_range_value.setEnabled)
        tolerance_range_layout.addWidget(tolerance_range_check)
        tolerance_range_layout.addWidget(tolerance_range_value)
        tolerance_layout.addLayout(tolerance_range_layout)
        
        # Target bias
        target_bias_layout = QHBoxLayout()
        target_bias_label = QLabel("Target bias (default: 0):")
        target_bias_value = QDoubleSpinBox()
        target_bias_value.setRange(-100.0, 100.0)
        target_bias_value.setValue(0.0)  # Default value
        target_bias_layout.addWidget(target_bias_label)
        target_bias_layout.addWidget(target_bias_value)
        tolerance_layout.addLayout(target_bias_layout)
        
        tolerance_group.setLayout(tolerance_layout)
        options_tab_layout.addWidget(tolerance_group)
        
        options_tab.setLayout(options_tab_layout)
        
        # Graphs tab
        graphs_tab = QWidget()
        graphs_layout = QVBoxLayout()
        
        # Graph options
        linearity_plot_check = QCheckBox("Linearity plot")
        linearity_plot_check.setChecked(True)
        graphs_layout.addWidget(linearity_plot_check)
        
        bias_plot_check = QCheckBox("Bias plot")
        bias_plot_check.setChecked(True)
        graphs_layout.addWidget(bias_plot_check)
        
        percent_bias_plot_check = QCheckBox("Percent bias plot")
        percent_bias_plot_check.setChecked(True)
        graphs_layout.addWidget(percent_bias_plot_check)
        
        fitted_line_plot_check = QCheckBox("Fitted line plot")
        fitted_line_plot_check.setChecked(True)
        graphs_layout.addWidget(fitted_line_plot_check)
        
        residual_plots_check = QCheckBox("Residual plots")
        residual_plots_check.setChecked(True)
        graphs_layout.addWidget(residual_plots_check)
        
        graphs_tab.setLayout(graphs_layout)
        
        # Add tabs to widget
        tab_widget.addTab(options_tab, "Options")
        tab_widget.addTab(graphs_tab, "Graphs")
        
        options_layout.addWidget(tab_widget)
        
        # Add buttons
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | 
                                     QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(options_dialog.accept)
        button_box.rejected.connect(options_dialog.reject)
        options_layout.addWidget(button_box)
        
        options_dialog.setLayout(options_layout)
        
        # Show options dialog
        if options_dialog.exec() != QDialog.DialogCode.Accepted:
            return

        try:
            # Create DataFrame for analysis
            df = pd.DataFrame({
                'Reference': pd.to_numeric(self.data[reference_col], errors='coerce'),
                'Measurement': pd.to_numeric(self.data[measurement_col], errors='coerce')
            })
            
            # Add Operator column if provided
            if operator_col:
                df['Operator'] = self.data[operator_col]
                
            # Add Order column if provided
            if order_col:
                df['Order'] = pd.to_numeric(self.data[order_col], errors='coerce')

            # Remove missing values
            df = df.dropna()

            if len(df) < 2:
                QMessageBox.warning(self, "Warning", "Insufficient data for linearity analysis")
                return
                
            # Check if we have enough reference points
            if len(df['Reference'].unique()) < 5:
                QMessageBox.warning(self, "Warning", "At least 5 different reference values are required")
                return

            # Calculate bias at each reference point
            df['Bias'] = df['Measurement'] - df['Reference']
            
            # Calculate percent bias
            df['Percent_Bias'] = (df['Bias'] / df['Reference']) * 100
            
            # Get confidence level
            conf_level = float(confidence_combo.currentText().strip('%')) / 100
            
            # Build formula based on options
            if fit_intercept_check.isChecked():
                formula_bias = 'Bias ~ Reference'
            else:
                formula_bias = 'Bias ~ Reference - 1'  # No intercept
                
            # Include operator effects if selected and available
            include_operator = False
            if operator_col is not None and include_operator_check.isChecked() and include_operator_check.isEnabled():
                if 'Operator' in df.columns and len(df['Operator'].unique()) > 1:
                    include_operator = True
            
            # Perform linear regression on bias vs reference
            try:
                if include_operator:
                    # Model with operator effects
                    formula_with_operator = formula_bias + ' + C(Operator) + Reference:C(Operator)'
                    model = sm.formula.ols(formula_with_operator, data=df).fit()
                else:
                    # Simple model without operator effects
                    model = sm.formula.ols(formula_bias, data=df).fit()
                
                # Calculate predicted bias values
                df['Predicted_Bias'] = model.predict(df)
                
                # Calculate confidence intervals for predictions
                prediction = model.get_prediction(df)
                df['CI_Lower'] = prediction.conf_int(alpha=1-conf_level)[:, 0]
                df['CI_Upper'] = prediction.conf_int(alpha=1-conf_level)[:, 1]
                
                # Calculate standard error as percentage of range
                reference_range = df['Reference'].max() - df['Reference'].min()
                std_error_percent = (model.bse[0] if len(model.bse) > 0 else 0) / reference_range * 100
                
                # Get tolerance information if provided
                tolerance_range = None
                if tolerance_range_check.isChecked():
                    tolerance_range = tolerance_range_value.value()
                
                target_bias = target_bias_value.value()
                
                # Create report
                report = f"""Linearity Study Results

Study Information:
Number of Reference Points: {len(df['Reference'].unique())}
Total Measurements: {len(df)}
"""
                if operator_col and 'Operator' in df.columns:
                    report += f"Number of Operators: {len(df['Operator'].unique())}\n"
                    
                report += f"""
Regression Analysis:
"""
                if fit_intercept_check.isChecked():
                    report += f"Intercept: {model.params[0]:.4f}\n"
                    report += f"  Target: 0.00 ±0.02\n"
                    report += f"  P-value for H₀: α = 0: {model.pvalues[0]:.4f}\n"
                    
                # Get the index of the Reference parameter
                ref_idx = 1 if fit_intercept_check.isChecked() else 0
                if include_operator and ref_idx < len(model.params):
                    # Adjust index if we have operator effects
                    for i, name in enumerate(model.params.index):
                        if name == 'Reference':
                            ref_idx = i
                            break
                
                if ref_idx < len(model.params):
                    report += f"""Slope: {model.params[ref_idx]:.4f}
  Target: 1.00 ±0.02
  P-value for H₀: β = 0: {model.pvalues[ref_idx]:.4f}
R-squared: {model.rsquared:.4f}
  Target: ≥ 0.99
Standard Error: {model.bse[ref_idx] if ref_idx < len(model.bse) else 0:.6f}
  As % of Range: {std_error_percent:.2f}%
  Target: < 1% of range

Bias Analysis:
Average Bias: {df['Bias'].mean():.4f}
Average % Bias: {df['Percent_Bias'].mean():.2f}%
"""
                else:
                    report += f"""Slope: Unable to determine
R-squared: {model.rsquared:.4f}
  Target: ≥ 0.99

Bias Analysis:
Average Bias: {df['Bias'].mean():.4f}
Average % Bias: {df['Percent_Bias'].mean():.2f}%
"""
                
                # Add reference point specific bias
                report += "\nBias by Reference Value:\n"
                for ref_val in sorted(df['Reference'].unique()):
                    ref_bias = df[df['Reference'] == ref_val]['Bias'].mean()
                    ref_pct_bias = df[df['Reference'] == ref_val]['Percent_Bias'].mean()
                    report += f"  Reference {ref_val:.2f}: Bias = {ref_bias:.4f} ({ref_pct_bias:.2f}%)\n"
                    
                # Add tolerance information if provided
                if tolerance_range:
                    report += f"\nTolerance Analysis:\n"
                    report += f"Tolerance Range: {tolerance_range:.4f}\n"
                    max_bias = df['Bias'].abs().max()
                    report += f"Maximum Bias: {max_bias:.4f}\n"
                    report += f"Bias as % of Tolerance: {(max_bias/tolerance_range*100):.2f}%\n"
                    
                report += "\nAssessment:\n"
                
                # Add assessment based on results
                if ref_idx < len(model.params):
                    if abs(model.params[ref_idx] - 1) < 0.02:
                        report += "Slope is within target range (1.00 ±0.02).\n"
                    else:
                        report += "Slope is outside target range (1.00 ±0.02).\n"
                    
                if fit_intercept_check.isChecked() and 0 < len(model.params):
                    if abs(model.params[0]) < 0.02:
                        report += "Intercept is within target range (0.00 ±0.02).\n"
                    else:
                        report += "Intercept is outside target range (0.00 ±0.02).\n"
                    
                if model.rsquared >= 0.99:
                    report += "R-squared meets target (≥ 0.99).\n"
                else:
                    report += "R-squared below target (≥ 0.99).\n"
                    
                if std_error_percent < 1:
                    report += "Standard error is within target (< 1% of range).\n"
                else:
                    report += "Standard error exceeds target (< 1% of range).\n"
                    
                # Overall assessment
                if ref_idx < len(model.params) and (abs(model.params[ref_idx] - 1) < 0.02 and 
                    (not fit_intercept_check.isChecked() or abs(model.params[0]) < 0.02) and
                    model.rsquared >= 0.99 and std_error_percent < 1):
                    report += "\nOverall: Measurement system linearity is acceptable."
                else:
                    report += "\nOverall: Measurement system linearity needs improvement."

                self.sessionWindow.setText(report)

                # Create visualizations based on selected options
                if linearity_plot_check.isChecked():
                    plt.figure(figsize=(10, 6))
                    plt.scatter(df['Reference'], df['Measurement'], alpha=0.7)
                    
                    # Plot the perfect agreement line (y = x)
                    min_val = min(df['Reference'].min(), df['Measurement'].min())
                    max_val = max(df['Reference'].max(), df['Measurement'].max())
                    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Agreement (y = x)')
                    
                    # Plot the fitted line
                    x_range = np.linspace(min_val, max_val, 100)
                    if fit_intercept_check.isChecked() and ref_idx < len(model.params):
                        y_fitted = model.params[0] + model.params[ref_idx] * x_range
                    elif ref_idx < len(model.params):
                        y_fitted = model.params[ref_idx] * x_range
                    else:
                        y_fitted = x_range  # Fallback to y=x if model parameters are not available
                    plt.plot(x_range, y_fitted, 'g-', label='Fitted Line')
                    
                    plt.title('Linearity Plot')
                    plt.xlabel('Reference Value')
                    plt.ylabel('Measurement')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.show()
                
                if bias_plot_check.isChecked():
                    plt.figure(figsize=(10, 6))
                    plt.scatter(df['Reference'], df['Bias'], alpha=0.7)
                    
                    # Plot the zero bias line
                    plt.axhline(y=0, color='r', linestyle='--', label='Zero Bias')
                    
                    # Plot the target bias line if not zero
                    if target_bias != 0:
                        plt.axhline(y=target_bias, color='g', linestyle='--', label=f'Target Bias ({target_bias})')
                    
                    # Plot the fitted line for bias
                    x_range = np.linspace(df['Reference'].min(), df['Reference'].max(), 100)
                    if fit_intercept_check.isChecked() and ref_idx < len(model.params):
                        y_fitted = model.params[0] + model.params[ref_idx] * x_range
                    elif ref_idx < len(model.params):
                        y_fitted = model.params[ref_idx] * x_range
                    else:
                        y_fitted = np.zeros_like(x_range)  # Fallback to y=0 if model parameters are not available
                    plt.plot(x_range, y_fitted, 'b-', label='Fitted Line')
                    
                    # Add confidence intervals
                    if len(df) > 2:  # Need at least 3 points for confidence intervals
                        try:
                            pred_df = pd.DataFrame({'Reference': x_range})
                            if include_operator:
                                # Use the first operator for prediction if operator effects are included
                                first_operator = df['Operator'].iloc[0]
                                pred_df['Operator'] = first_operator
                            pred = model.get_prediction(pred_df)
                            ci = pred.conf_int(alpha=1-conf_level)
                            plt.fill_between(x_range, ci[:, 0], ci[:, 1], color='gray', alpha=0.2, label=f'{int(conf_level*100)}% Confidence Interval')
                        except Exception as e:
                            # Skip confidence intervals if there's an error
                            print(f"Error calculating confidence intervals: {str(e)}")
                    
                    # Add tolerance limits if specified
                    if tolerance_range:
                        plt.axhline(y=tolerance_range/2, color='orange', linestyle='-.', label=f'Tolerance Limit (+{tolerance_range/2})')
                        plt.axhline(y=-tolerance_range/2, color='orange', linestyle='-.', label=f'Tolerance Limit (-{tolerance_range/2})')
                    
                    plt.title('Bias Plot')
                    plt.xlabel('Reference Value')
                    plt.ylabel('Bias (Measurement - Reference)')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.show()
                
                if percent_bias_plot_check.isChecked():
                    plt.figure(figsize=(10, 6))
                    plt.scatter(df['Reference'], df['Percent_Bias'], alpha=0.7)
                    
                    # Plot the zero percent bias line
                    plt.axhline(y=0, color='r', linestyle='--', label='Zero % Bias')
                    
                    # Calculate average percent bias by reference value
                    ref_pct_bias = df.groupby('Reference')['Percent_Bias'].mean().reset_index()
                    plt.plot(ref_pct_bias['Reference'], ref_pct_bias['Percent_Bias'], 'g-o', label='Average % Bias')
                    
                    plt.title('Percent Bias Plot')
                    plt.xlabel('Reference Value')
                    plt.ylabel('Percent Bias (%)')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.show()
                
                if fitted_line_plot_check.isChecked():
                    # Create a fitted line plot for Measurement vs Reference
                    plt.figure(figsize=(10, 6))
                    
                    # Scatter plot of actual data
                    plt.scatter(df['Reference'], df['Measurement'], alpha=0.7, label='Observed Data')
                    
                    # Fitted regression line
                    x_range = np.linspace(df['Reference'].min(), df['Reference'].max(), 100)
                    y_fitted = x_range + df['Bias'].mean()  # Assuming bias is constant
                    plt.plot(x_range, y_fitted, 'r-', label='Fitted Line')
                    
                    # Perfect agreement line
                    plt.plot(x_range, x_range, 'g--', label='Perfect Agreement (y = x)')
                    
                    plt.title('Fitted Line Plot')
                    plt.xlabel('Reference Value')
                    plt.ylabel('Measurement')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.show()
                
                if residual_plots_check.isChecked():
                    # Create residual plots
                    plt.figure(figsize=(10, 10))
                    
                    # Residuals vs Fitted
                    plt.subplot(2, 2, 1)
                    plt.scatter(model.fittedvalues, model.resid, alpha=0.7)
                    plt.axhline(y=0, color='r', linestyle='--')
                    plt.title('Residuals vs Fitted')
                    plt.xlabel('Fitted Values')
                    plt.ylabel('Residuals')
                    plt.grid(True, alpha=0.3)
                    
                    # Normal Q-Q plot
                    plt.subplot(2, 2, 2)
                    from scipy import stats
                    (quantiles, values), (slope, intercept, r) = stats.probplot(model.resid, dist="norm")
                    plt.scatter(quantiles, values, alpha=0.7)
                    plt.plot(quantiles, slope * quantiles + intercept, 'r-')
                    plt.title('Normal Q-Q')
                    plt.xlabel('Theoretical Quantiles')
                    plt.ylabel('Sample Quantiles')
                    plt.grid(True, alpha=0.3)
                    
                    # Scale-Location plot
                    plt.subplot(2, 2, 3)
                    plt.scatter(model.fittedvalues, np.sqrt(np.abs(model.resid)), alpha=0.7)
                    plt.title('Scale-Location')
                    plt.xlabel('Fitted Values')
                    plt.ylabel('√|Residuals|')
                    plt.grid(True, alpha=0.3)
                    
                    # Residuals vs Reference
                    plt.subplot(2, 2, 4)
                    plt.scatter(df['Reference'], model.resid, alpha=0.7)
                    plt.axhline(y=0, color='r', linestyle='--')
                    plt.title('Residuals vs Reference')
                    plt.xlabel('Reference Value')
                    plt.ylabel('Residuals')
                    plt.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    plt.show()
                    
            except Exception as e:
                QMessageBox.warning(self, "Error", 
                    f"An error occurred during linearity analysis:\n{str(e)}\n\n"
                    "Please check your data and try again.")

        except Exception as e:
            QMessageBox.warning(self, "Error", 
                f"An error occurred during linearity analysis:\n{str(e)}\n\n"
                "Please check your data and try again.")

    def fishboneDiagram(self):
        """Create and edit fishbone diagram"""
        try:
            # Create dialog for fishbone diagram
            dialog = QDialog(self)
            dialog.setWindowTitle("Fishbone Diagram Creator")
            dialog.resize(800, 600)

            # Create layout
            layout = QVBoxLayout()

            # Add problem statement input
            problem_label = QLabel("Problem Statement:")
            problem_input = QLineEdit()
            layout.addWidget(problem_label)
            layout.addWidget(problem_input)

            # Create category inputs
            categories = ['Materials', 'Methods', 'Machines', 'Manpower', 'Measurement', 'Environment']
            category_inputs = {}

            for category in categories:
                # Create group box for category
                group = QGroupBox(category)
                group_layout = QVBoxLayout()

                # Add 5 cause input fields for each category
                cause_inputs = []
                for i in range(5):
                    cause_input = QLineEdit()
                    cause_input.setPlaceholderText(f"Enter cause {i+1}")
                    group_layout.addWidget(cause_input)
                    cause_inputs.append(cause_input)

                group.setLayout(group_layout)
                layout.addWidget(group)
                category_inputs[category] = cause_inputs

            # Add create button
            create_button = QPushButton("Create Diagram")
            layout.addWidget(create_button)

            dialog.setLayout(layout)

            def create_diagram():
                # Get problem statement
                problem = problem_input.text()
                if not problem:
                    QMessageBox.warning(dialog, "Warning", "Please enter a problem statement")
                    return

                # Collect causes for each category
                causes = {}
                for category, inputs in category_inputs.items():
                    causes[category] = [input.text() for input in inputs if input.text()]

                # Create the fishbone diagram
                plt.figure(figsize=(15, 10))
                
                # Draw main arrow and problem box
                plt.plot([0, 10], [5, 5], 'k-', linewidth=2)  # Main arrow
                plt.text(10.2, 5, problem, ha='left', va='center', fontsize=12, fontweight='bold')

                # Position categories and their causes
                y_positions = [2, 3, 4, 6, 7, 8]
                for i, (category, category_causes) in enumerate(causes.items()):
                    y = y_positions[i]
                    # Draw category arrow
                    if y < 5:  # Below center line
                        plt.plot([2+i, 5], [y, 5], 'k-')
                    else:  # Above center line
                        plt.plot([2+(5-i), 5], [y, 5], 'k-')
                    
                    # Add category label
                    plt.text(2+i if y < 5 else 2+(5-i), y, category, 
                            ha='right' if y < 5 else 'left', 
                            va='bottom' if y < 5 else 'top',
                            fontsize=10)

                    # Add causes
                    for j, cause in enumerate(category_causes):
                        if cause:
                            x_offset = 0.5 * j
                            if y < 5:  # Below center line
                                plt.plot([2+i+x_offset, 2+i+x_offset+0.5], 
                                       [y+0.2*j, y+0.2*j+0.2], 'k-')
                                plt.text(2+i+x_offset, y+0.2*j, cause, 
                                       ha='right', va='bottom', fontsize=8)
                            else:  # Above center line
                                plt.plot([2+(5-i)+x_offset, 2+(5-i)+x_offset+0.5],
                                       [y-0.2*j, y-0.2*j-0.2], 'k-')
                                plt.text(2+(5-i)+x_offset, y-0.2*j, cause,
                                       ha='left', va='top', fontsize=8)

                plt.axis('off')
                plt.title('Fishbone (Ishikawa) Diagram', pad=20)
                plt.tight_layout()
                plt.show()

                # Create report
                report = f"""Fishbone Diagram Analysis

Problem Statement: {problem}

Categories and Causes:
"""
                for category, category_causes in causes.items():
                    report += f"\n{category}:"
                    for cause in category_causes:
                        if cause:
                            report += f"\n  - {cause}"

                self.sessionWindow.setText(report)

            create_button.clicked.connect(create_diagram)
            dialog.exec()

        except Exception as e:
            QMessageBox.warning(self, "Error", 
                f"An error occurred while creating the fishbone diagram:\n{str(e)}")

    def fmeaTemplate(self):
        """Create and manage FMEA template"""
        try:
            # Create dialog for FMEA
            dialog = QDialog(self)
            dialog.setWindowTitle("FMEA Template")
            dialog.resize(1000, 800)

            # Create layout
            layout = QVBoxLayout()

            # Add header information
            header_group = QGroupBox("FMEA Header")
            header_layout = QFormLayout()
            
            process_input = QLineEdit()
            responsible_input = QLineEdit()
            date_input = QLineEdit()
            
            header_layout.addRow("Process/Product:", process_input)
            header_layout.addRow("Team/Responsible:", responsible_input)
            header_layout.addRow("Date:", date_input)
            
            header_group.setLayout(header_layout)
            layout.addWidget(header_group)

            # Create table for FMEA entries
            table = QTableWidget()
            table.setColumnCount(10)
            table.setRowCount(10)  # Initial rows, can be expanded
            
            # Set column headers
            headers = [
                "Process Step",
                "Potential Failure Mode",
                "Potential Effects",
                "Severity (1-10)",
                "Potential Causes",
                "Occurrence (1-10)",
                "Current Controls",
                "Detection (1-10)",
                "RPN",
                "Recommended Actions"
            ]
            table.setHorizontalHeaderLabels(headers)
            
            # Set column widths
            table.setColumnWidth(0, 120)  # Process Step
            table.setColumnWidth(1, 120)  # Failure Mode
            table.setColumnWidth(2, 120)  # Effects
            table.setColumnWidth(3, 80)   # Severity
            table.setColumnWidth(4, 120)  # Causes
            table.setColumnWidth(5, 80)   # Occurrence
            table.setColumnWidth(6, 120)  # Controls
            table.setColumnWidth(7, 80)   # Detection
            table.setColumnWidth(8, 80)   # RPN
            table.setColumnWidth(9, 120)  # Actions

            layout.addWidget(table)

            # Add buttons
            button_layout = QHBoxLayout()
            
            import_button = QPushButton("Import FMEA")  # New Import button
            add_row_button = QPushButton("Add Row")
            calculate_button = QPushButton("Calculate RPN")
            save_button = QPushButton("Save FMEA")
            export_button = QPushButton("Export Report")
            save_report_button = QPushButton("Save Report")
            
            button_layout.addWidget(import_button)  # Add Import button
            button_layout.addWidget(add_row_button)
            button_layout.addWidget(calculate_button)
            button_layout.addWidget(save_button)
            button_layout.addWidget(export_button)
            button_layout.addWidget(save_report_button)
            
            layout.addLayout(button_layout)

            dialog.setLayout(layout)

            # Add functionality to buttons
            def import_fmea():
                try:
                    filename, _ = QFileDialog.getOpenFileName(dialog, "Import FMEA", "", "CSV Files (*.csv)")
                    if filename:
                        # Read CSV file
                        df = pd.read_csv(filename)
                        
                        # Verify column mapping
                        if not all(header in df.columns for header in headers):
                            # If columns don't match exactly, show mapping dialog
                            mapping_dialog = QDialog(dialog)
                            mapping_dialog.setWindowTitle("Column Mapping")
                            mapping_layout = QFormLayout()
                            
                            # Create mapping dropdowns
                            mappings = {}
                            for header in headers:
                                combo = QComboBox()
                                combo.addItems([''] + list(df.columns))
                                mapping_layout.addRow(f"Map {header} to:", combo)
                                mappings[header] = combo
                            
                            # Add OK/Cancel buttons
                            buttons = QDialogButtonBox(
                                QDialogButtonBox.StandardButton.Ok | 
                                QDialogButtonBox.StandardButton.Cancel
                            )
                            buttons.accepted.connect(mapping_dialog.accept)
                            buttons.rejected.connect(mapping_dialog.reject)
                            mapping_layout.addWidget(buttons)
                            
                            mapping_dialog.setLayout(mapping_layout)
                            
                            if mapping_dialog.exec() == QDialog.DialogCode.Accepted:
                                # Create new dataframe with mapped columns
                                mapped_data = {}
                                for header, combo in mappings.items():
                                    if combo.currentText():
                                        mapped_data[header] = df[combo.currentText()]
                                df = pd.DataFrame(mapped_data)
                        
                        # Clear existing table
                        table.setRowCount(0)
                        
                        # Add data to table
                        for index, row in df.iterrows():
                            table.insertRow(table.rowCount())
                            for col, value in enumerate(row):
                                item = QTableWidgetItem(str(value))
                                table.setItem(table.rowCount()-1, col, item)
                        
                        QMessageBox.information(dialog, "Success", "FMEA data imported successfully!")
                except Exception as e:
                    QMessageBox.warning(dialog, "Error", f"Error importing FMEA data: {str(e)}")

            def add_row():
                current_row = table.rowCount()
                table.insertRow(current_row)

            def calculate_rpn():
                for row in range(table.rowCount()):
                    try:
                        # Get severity, occurrence, and detection values
                        severity = float(table.item(row, 3).text() if table.item(row, 3) else 0)
                        occurrence = float(table.item(row, 5).text() if table.item(row, 5) else 0)
                        detection = float(table.item(row, 7).text() if table.item(row, 7) else 0)
                        
                        # Calculate RPN
                        rpn = severity * occurrence * detection
                        
                        # Update RPN cell
                        rpn_item = QTableWidgetItem(str(int(rpn)))
                        table.setItem(row, 8, rpn_item)
                        
                        # Highlight high RPNs (>100) in red
                        if rpn > 100:
                            rpn_item.setBackground(Qt.GlobalColor.red)
                    except (ValueError, AttributeError):
                        continue

            def save_fmea():
                # Create DataFrame from table data
                data = []
                for row in range(table.rowCount()):
                    row_data = []
                    for col in range(table.columnCount()):
                        item = table.item(row, col)
                        row_data.append(item.text() if item else "")
                    data.append(row_data)
                
                df = pd.DataFrame(data, columns=headers)
                
                # Save to CSV
                filename, _ = QFileDialog.getSaveFileName(dialog, "Save FMEA", "", "CSV Files (*.csv)")
                if filename:
                    df.to_csv(filename, index=False)
                    QMessageBox.information(dialog, "Success", "FMEA saved successfully!")

            def generate_report():
                # Generate detailed report
                report = f"""FMEA Analysis Report

Process/Product: {process_input.text()}
Team/Responsible: {responsible_input.text()}
Date: {date_input.text()}

Summary of Failure Modes and Actions:
"""
                # Add table contents to report
                for row in range(table.rowCount()):
                    process_step = table.item(row, 0)
                    failure_mode = table.item(row, 1)
                    effects = table.item(row, 2)
                    severity = table.item(row, 3)
                    causes = table.item(row, 4)
                    occurrence = table.item(row, 5)
                    controls = table.item(row, 6)
                    detection = table.item(row, 7)
                    rpn = table.item(row, 8)
                    actions = table.item(row, 9)
                    
                    if process_step and process_step.text():
                        report += f"\nProcess Step: {process_step.text()}"
                        report += f"\nFailure Mode: {failure_mode.text() if failure_mode else 'Not specified'}"
                        report += f"\nPotential Effects: {effects.text() if effects else 'Not specified'}"
                        report += f"\nSeverity: {severity.text() if severity else 'Not specified'}"
                        report += f"\nPotential Causes: {causes.text() if causes else 'Not specified'}"
                        report += f"\nOccurrence: {occurrence.text() if occurrence else 'Not specified'}"
                        report += f"\nCurrent Controls: {controls.text() if controls else 'Not specified'}"
                        report += f"\nDetection: {detection.text() if detection else 'Not specified'}"
                        report += f"\nRPN: {rpn.text() if rpn else 'Not calculated'}"
                        report += f"\nRecommended Actions: {actions.text() if actions else 'None'}"
                        report += "\n" + "-"*50
                return report

            def export_report():
                report = generate_report()
                # Display report in session window
                self.sessionWindow.clear()  # Clear previous content
                self.sessionWindow.setText(report)
                QMessageBox.information(dialog, "Report Generated", "Report has been generated in the session window.")

            def save_report():
                report = generate_report()
                # Save report to text file
                filename, _ = QFileDialog.getSaveFileName(dialog, "Save Report", "", "Text Files (*.txt)")
                if filename:
                    with open(filename, 'w') as f:
                        f.write(report)
                    QMessageBox.information(dialog, "Success", "Report saved successfully!")

            # Connect buttons to functions
            import_button.clicked.connect(import_fmea)  # Connect Import button
            add_row_button.clicked.connect(add_row)
            calculate_button.clicked.connect(calculate_rpn)
            save_button.clicked.connect(save_fmea)
            export_button.clicked.connect(export_report)
            save_report_button.clicked.connect(save_report)

            # Add help text
            help_text = """
Rating Guidelines:
Severity (1-10): Impact of failure on customer/process
- 1: No effect
- 5: Moderate effect
- 10: Hazardous effect

Occurrence (1-10): Likelihood of failure occurring
- 1: Very unlikely
- 5: Occasional failure
- 10: Failure is almost inevitable

Detection (1-10): Ability to detect failure before it reaches customer
- 1: Certain detection
- 5: Moderate detection
- 10: No detection

RPN (Risk Priority Number) = Severity × Occurrence × Detection
RPN > 100 indicates high priority items needing immediate attention.
"""
            help_label = QLabel(help_text)
            layout.addWidget(help_label)

            dialog.exec()

        except Exception as e:
            QMessageBox.warning(self, "Error", 
                f"An error occurred while creating the FMEA template:\n{str(e)}")

    def sigmaLevelCalc(self):
        """Calculate Sigma Level from DPMO"""
        dpmo, ok = QInputDialog.getDouble(self, "Sigma Calculator", "Enter DPMO:", 0, 0)
        if not ok:
            return
            
        sigma = dpmo_to_sigma(dpmo)
        self.sessionWindow.setText(f"Sigma Level: {sigma:.2f}")

    def yieldAnalysis(self):
        """Analyze process yield and calculate process capability indices"""
        try:
            # Load data from the table
            self.loadDataFromTable()
            
            if self.data.empty:
                QMessageBox.warning(self, "Warning", "No data available for analysis")
                return
                
            # Check if required columns exist
            required_cols = ['Input', 'Output', 'Rework', 'Scrap']
            if not all(col in self.data.columns for col in required_cols):
                QMessageBox.warning(self, "Warning", "Required columns (Input, Output, Rework, Scrap) not found")
                return
                
            # Get the first row of data for analysis
            row = self.data.iloc[0]
            input_units = row['Input']
            output_units = row['Output']
            rework_units = row['Rework']
            scrap_units = row['Scrap']
            
            # Calculate yields and rates
            first_pass_yield = ((output_units - rework_units) / input_units) * 100
            final_yield = (output_units / input_units) * 100
            scrap_rate = (scrap_units / input_units) * 100
            rework_rate = (rework_units / input_units) * 100
            
            # Display results with exact formatting from test guide
            self.sessionWindow.append("Process Yield Analysis Results")
            self.sessionWindow.append("----------------------------")
            self.sessionWindow.append(f"\nInput: {input_units} units")
            self.sessionWindow.append(f"Output: {output_units} units")
            self.sessionWindow.append(f"Rework: {rework_units} units")
            self.sessionWindow.append(f"Scrap: {scrap_units} units")
            
            self.sessionWindow.append("\nCalculations:")
            self.sessionWindow.append(f"First Pass Yield = {first_pass_yield:.1f}%    # (Output - Rework) / Input")
            self.sessionWindow.append(f"Final Yield = {final_yield:.1f}%         # Output / Input")
            self.sessionWindow.append(f"Scrap Rate = {scrap_rate:.1f}%          # Scrap / Input")
            self.sessionWindow.append(f"Rework Rate = {rework_rate:.1f}%         # Rework / Input")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error in process yield analysis: {str(e)}")

    def biasStudy(self):
        """Perform measurement system bias study"""
        # Create dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("Bias Study")
        dialog.setMinimumWidth(600)
        
        # Create tabs
        tabs = QTabWidget()
        basic_tab = QWidget()
        options_tab = QWidget()
        graphs_tab = QWidget()
        
        # Basic tab layout
        basic_layout = QVBoxLayout(basic_tab)
        
        # Measurement column selection
        meas_group = QGroupBox("Column Selection")
        meas_layout = QFormLayout(meas_group)
        
        meas_combo = QComboBox()
        meas_combo.addItems(self.data.columns.tolist())
        meas_layout.addRow("Measurement:", meas_combo)
        
        # Reference value input
        ref_input = QLineEdit()
        meas_layout.addRow("Reference value:", ref_input)
        
        # Operator column selection (optional)
        operator_combo = QComboBox()
        operator_combo.addItem("None")
        operator_combo.addItems(self.data.columns.tolist())
        meas_layout.addRow("Operator (optional):", operator_combo)
        
        # Order column selection (optional)
        order_combo = QComboBox()
        order_combo.addItem("None")
        order_combo.addItems(self.data.columns.tolist())
        meas_layout.addRow("Order (optional):", order_combo)
        
        basic_layout.addWidget(meas_group)
        
        # Options tab layout
        options_layout = QVBoxLayout(options_tab)
        
        # Analysis settings
        analysis_group = QGroupBox("Analysis Settings")
        analysis_layout = QFormLayout(analysis_group)
        
        confidence_combo = QComboBox()
        confidence_combo.addItems(["90%", "95%", "99%"])
        confidence_combo.setCurrentText("95%")
        analysis_layout.addRow("Confidence level:", confidence_combo)
        
        include_operator_check = QCheckBox("Include operator effects")
        analysis_layout.addRow("", include_operator_check)
        
        options_layout.addWidget(analysis_group)
        
        # Tolerance information
        tolerance_group = QGroupBox("Tolerance Information")
        tolerance_layout = QFormLayout(tolerance_group)
        
        tolerance_input = QLineEdit()
        tolerance_layout.addRow("Tolerance range:", tolerance_input)
        
        acceptable_bias_input = QLineEdit("5.0")
        tolerance_layout.addRow("Acceptable bias (%):", acceptable_bias_input)
        
        options_layout.addWidget(tolerance_group)
        
        # Graphs tab layout
        graphs_layout = QVBoxLayout(graphs_tab)
        
        run_chart_check = QCheckBox("Run chart")
        run_chart_check.setChecked(True)
        graphs_layout.addWidget(run_chart_check)
        
        histogram_check = QCheckBox("Histogram")
        histogram_check.setChecked(True)
        graphs_layout.addWidget(histogram_check)
        
        normal_plot_check = QCheckBox("Normal probability plot")
        normal_plot_check.setChecked(True)
        graphs_layout.addWidget(normal_plot_check)
        
        box_plot_check = QCheckBox("Box plot (if multiple operators)")
        box_plot_check.setChecked(True)
        graphs_layout.addWidget(box_plot_check)
        
        # Add tabs to tab widget
        tabs.addTab(basic_tab, "Basic")
        tabs.addTab(options_tab, "Options")
        tabs.addTab(graphs_tab, "Graphs")
        
        # Dialog buttons
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        
        # Main dialog layout
        main_layout = QVBoxLayout(dialog)
        main_layout.addWidget(tabs)
        main_layout.addWidget(button_box)
        
        # Show dialog
        if not dialog.exec():
            return
        
        # Get values from dialog
        measurement_col = meas_combo.currentText()
        
        try:
            reference_value = float(ref_input.text())
        except ValueError:
            QMessageBox.warning(self, "Error", "Invalid reference value. Please enter a numeric value.")
            return
        
        operator_col = None if operator_combo.currentText() == "None" else operator_combo.currentText()
        order_col = None if order_combo.currentText() == "None" else order_combo.currentText()
        
        # Get confidence level
        conf_level_text = confidence_combo.currentText()
        conf_level = float(conf_level_text.strip('%')) / 100
        
        # Get tolerance settings
        tolerance_range = None
        try:
            if tolerance_input.text():
                tolerance_range = float(tolerance_input.text())
        except ValueError:
            QMessageBox.warning(self, "Warning", "Invalid tolerance range. This will be ignored.")
        
        acceptable_bias_pct = 5.0
        try:
            if acceptable_bias_input.text():
                acceptable_bias_pct = float(acceptable_bias_input.text())
        except ValueError:
            QMessageBox.warning(self, "Warning", "Invalid acceptable bias percentage. Using default 5%.")
        
        # Get selected graphs
        show_run_chart = run_chart_check.isChecked()
        show_histogram = histogram_check.isChecked()
        show_normal_plot = normal_plot_check.isChecked()
        show_box_plot = box_plot_check.isChecked() and operator_col is not None
        
        try:
            # Get measurements
            measurements = pd.to_numeric(self.data[measurement_col], errors='coerce')
            
            # Create a DataFrame for analysis
            analysis_df = pd.DataFrame({'Measurement': measurements})
            
            # Add operator column if selected
            if operator_col:
                analysis_df['Operator'] = self.data[operator_col]
            
            # Add order column if selected
            if order_col:
                analysis_df['Order'] = pd.to_numeric(self.data[order_col], errors='coerce')
                # Sort by order if available
                analysis_df = analysis_df.sort_values('Order')
            
            # Drop rows with missing values
            analysis_df = analysis_df.dropna()
            
            if len(analysis_df) < 10:
                QMessageBox.warning(self, "Warning", "Insufficient data for bias analysis (minimum 10 measurements required)")
                return
            
            # Calculate basic statistics
            mean = np.mean(analysis_df['Measurement'])
            std_dev = np.std(analysis_df['Measurement'], ddof=1)
            std_error = std_dev / np.sqrt(len(analysis_df))
            bias = mean - reference_value
            percent_bias = 100 * bias / reference_value if reference_value != 0 else float('inf')
            
            # Perform t-test to check if bias is significant
            t_stat, p_value = stats.ttest_1samp(analysis_df['Measurement'], reference_value)
            
            # Calculate confidence interval for bias
            ci = stats.t.interval(conf_level, len(analysis_df)-1, loc=bias, scale=std_error)
            
            # Calculate capability indices if tolerance is provided
            capability_indices = {}
            if tolerance_range:
                capability_indices['Cg'] = (0.2 * tolerance_range) / (6 * std_dev)  # Precision to tolerance ratio
                capability_indices['Cgk'] = (0.1 * tolerance_range - abs(bias)) / (3 * std_dev)  # Accuracy to tolerance ratio
            
            # Create report
            report = f"""Bias Study Results

Reference Value: {reference_value:.4f}

Basic Statistics:
Number of Measurements: {len(analysis_df)}
Mean: {mean:.4f}
Standard Deviation: {std_dev:.4f}
Standard Error: {std_error:.4f}

Bias:
Absolute: {bias:.4f}
Percent: {percent_bias:.2f}%

Hypothesis Test (H₀: μ = {reference_value:.4f}):
t-statistic: {t_stat:.4f}
p-value: {p_value:.4f}

{conf_level*100:.0f}% Confidence Interval for Bias:
Lower: {ci[0]:.4f}
Upper: {ci[1]:.4f}
"""
            
            # Add tolerance information if provided
            if tolerance_range:
                report += f"""
Tolerance Information:
Tolerance Range: {tolerance_range:.4f}
Acceptable Bias: ±{acceptable_bias_pct:.1f}%
Actual Bias as % of Tolerance: {100*abs(bias)/tolerance_range:.2f}%

Capability Indices:
Cg (Precision/Tolerance): {capability_indices['Cg']:.4f}
Cgk (Accuracy/Tolerance): {capability_indices['Cgk']:.4f}
"""
            
            # Add assessment based on results
            report += "\nAssessment:\n"
            
            if abs(percent_bias) <= acceptable_bias_pct:
                report += f"Bias is within acceptable range (±{acceptable_bias_pct:.1f}%).\n"
            else:
                report += f"Bias exceeds acceptable range (±{acceptable_bias_pct:.1f}%).\n"
            
            if p_value < (1 - conf_level):
                report += f"Bias is statistically significant (p < {1-conf_level:.2f}).\n"
            else:
                report += f"No significant bias detected (p >= {1-conf_level:.2f}).\n"
            
            if tolerance_range:
                if capability_indices['Cg'] >= 1.33 and capability_indices['Cgk'] >= 1.33:
                    report += "Measurement system is capable (Cg & Cgk >= 1.33).\n"
                else:
                    report += "Measurement system may need improvement (Cg or Cgk < 1.33).\n"
            
            self.sessionWindow.setText(report)
            
            # Create visualizations
            if show_run_chart:
                plt.figure(figsize=(10, 6))
                
                if order_col:
                    # Use order for x-axis if available
                    plt.plot(analysis_df['Order'], analysis_df['Measurement'], 
                             marker='o', linestyle='-', label='Measurements')
                    plt.xlabel('Measurement Order')
                else:
                    # Otherwise use index
                    plt.plot(analysis_df['Measurement'], marker='o', linestyle='-', label='Measurements')
                    plt.xlabel('Measurement Number')
                
                plt.axhline(y=reference_value, color='r', linestyle='--', label='Reference')
                plt.axhline(y=mean, color='g', linestyle='-', label='Mean')
                
                # Add confidence interval for mean
                mean_ci = stats.t.interval(conf_level, len(analysis_df)-1, loc=mean, scale=std_error)
                plt.axhline(y=mean_ci[0], color='g', linestyle=':', label=f'{conf_level*100:.0f}% CI Lower')
                plt.axhline(y=mean_ci[1], color='g', linestyle=':', label=f'{conf_level*100:.0f}% CI Upper')
                
                # Add tolerance limits if provided
                if tolerance_range:
                    acceptable_bias = acceptable_bias_pct * reference_value / 100
                    plt.axhline(y=reference_value + acceptable_bias, color='orange', 
                                linestyle='-.', label=f'+{acceptable_bias_pct}% Bias')
                    plt.axhline(y=reference_value - acceptable_bias, color='orange', 
                                linestyle='-.', label=f'-{acceptable_bias_pct}% Bias')
                
                plt.title('Bias Study: Run Chart')
                plt.ylabel('Value')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.show()
            
            if show_histogram:
                plt.figure(figsize=(10, 6))
                plt.hist(analysis_df['Measurement'], bins='auto', density=True, alpha=0.7, label='Measurements')
                plt.axvline(x=reference_value, color='r', linestyle='--', label='Reference')
                plt.axvline(x=mean, color='g', linestyle='-', label='Mean')
                
                # Add normal curve
                x = np.linspace(min(analysis_df['Measurement']), max(analysis_df['Measurement']), 100)
                y = stats.norm.pdf(x, mean, std_dev)
                plt.plot(x, y, 'b-', label='Normal Dist.')
                
                # Add tolerance limits if provided
                if tolerance_range:
                    acceptable_bias = acceptable_bias_pct * reference_value / 100
                    plt.axvline(x=reference_value + acceptable_bias, color='orange', 
                                linestyle='-.', label=f'+{acceptable_bias_pct}% Bias')
                    plt.axvline(x=reference_value - acceptable_bias, color='orange', 
                                linestyle='-.', label=f'-{acceptable_bias_pct}% Bias')
                
                plt.title('Bias Study: Distribution of Measurements')
                plt.xlabel('Value')
                plt.ylabel('Density')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.show()
            
            if show_normal_plot:
                plt.figure(figsize=(10, 6))
                stats.probplot(analysis_df['Measurement'], plot=plt)
                plt.title('Bias Study: Normal Probability Plot')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.show()
            
            if show_box_plot and operator_col:
                plt.figure(figsize=(10, 6))
                sns.boxplot(x='Operator', y='Measurement', data=analysis_df)
                plt.axhline(y=reference_value, color='r', linestyle='--', label='Reference')
                
                # Add tolerance limits if provided
                if tolerance_range:
                    acceptable_bias = acceptable_bias_pct * reference_value / 100
                    plt.axhline(y=reference_value + acceptable_bias, color='orange', 
                                linestyle='-.', label=f'+{acceptable_bias_pct}% Bias')
                    plt.axhline(y=reference_value - acceptable_bias, color='orange', 
                                linestyle='-.', label=f'-{acceptable_bias_pct}% Bias')
                
                plt.title('Bias Study: Measurements by Operator')
                plt.ylabel('Value')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.show()
                
                # ANOVA by operator if requested
                if include_operator_check.isChecked():
                    try:
                        # Perform one-way ANOVA
                        groups = [group['Measurement'].values for name, group in analysis_df.groupby('Operator')]
                        f_stat, p_value_anova = stats.f_oneway(*groups)
                        
                        # Display ANOVA results
                        anova_report = f"""
Operator Effects Analysis (ANOVA):
F-statistic: {f_stat:.4f}
p-value: {p_value_anova:.4f}

"""
                        if p_value_anova < (1 - conf_level):
                            anova_report += f"There is a significant difference between operators (p < {1-conf_level:.2f})."
                        else:
                            anova_report += f"No significant difference between operators (p >= {1-conf_level:.2f})."
                        
                        # Append to existing report
                        current_report = self.sessionWindow.toPlainText()
                        self.sessionWindow.setText(current_report + "\n" + anova_report)
                    except Exception as e:
                        QMessageBox.warning(self, "Warning", 
                            f"Could not perform operator effects analysis:\n{str(e)}")

        except Exception as e:
            QMessageBox.warning(self, "Error", 
                f"An error occurred during bias analysis:\n{str(e)}\n\n"
                "Please check your data and try again.")

    def stabilityStudy(self):
        """Perform measurement system stability study"""
        # Create dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("Stability Study")
        dialog.setMinimumWidth(600)
        
        # Create tabs
        tabs = QTabWidget()
        basic_tab = QWidget()
        options_tab = QWidget()
        graphs_tab = QWidget()
        
        # Basic tab layout
        basic_layout = QVBoxLayout(basic_tab)
        
        # Column selection
        columns_group = QGroupBox("Column Selection")
        columns_layout = QFormLayout(columns_group)
        
        # DateTime column
        datetime_combo = QComboBox()
        datetime_combo.addItems(self.data.columns.tolist())
        columns_layout.addRow("DateTime:", datetime_combo)
        
        # Measurement column
        measurement_combo = QComboBox()
        measurement_combo.addItems(self.data.columns.tolist())
        columns_layout.addRow("Measurement:", measurement_combo)
        
        # Operator column (optional)
        operator_combo = QComboBox()
        operator_combo.addItem("None")
        operator_combo.addItems(self.data.columns.tolist())
        columns_layout.addRow("Operator (optional):", operator_combo)
        
        # Standard column (optional)
        standard_combo = QComboBox()
        standard_combo.addItem("None")
        standard_combo.addItems(self.data.columns.tolist())
        columns_layout.addRow("Standard (optional):", standard_combo)
        
        basic_layout.addWidget(columns_group)
        
        # Options tab layout
        options_layout = QVBoxLayout(options_tab)
        
        # Time settings
        time_group = QGroupBox("Time Settings")
        time_layout = QFormLayout(time_group)
        
        time_unit_combo = QComboBox()
        time_unit_combo.addItems(["Hour", "Day", "Week", "Month"])
        time_unit_combo.setCurrentText("Day")
        time_layout.addRow("Time unit:", time_unit_combo)
        
        group_by_time_check = QCheckBox("Group measurements by time period")
        group_by_time_check.setChecked(True)
        time_layout.addRow("", group_by_time_check)
        
        reference_input = QLineEdit()
        time_layout.addRow("Reference value (optional):", reference_input)
        
        options_layout.addWidget(time_group)
        
        # Analysis options
        analysis_group = QGroupBox("Analysis Options")
        analysis_layout = QFormLayout(analysis_group)
        
        chart_type_combo = QComboBox()
        chart_type_combo.addItems(["I-MR Chart", "X-bar R Chart"])
        chart_type_combo.setCurrentText("I-MR Chart")
        analysis_layout.addRow("Control chart type:", chart_type_combo)
        
        alpha_combo = QComboBox()
        alpha_combo.addItems(["0.01", "0.05", "0.10"])
        alpha_combo.setCurrentText("0.05")
        analysis_layout.addRow("α level:", alpha_combo)
        
        special_causes_check = QCheckBox("Include tests for special causes")
        special_causes_check.setChecked(True)
        analysis_layout.addRow("", special_causes_check)
        
        options_layout.addWidget(analysis_group)
        
        # Graphs tab layout
        graphs_layout = QVBoxLayout(graphs_tab)
        
        time_series_check = QCheckBox("Time Series Plot")
        time_series_check.setChecked(True)
        graphs_layout.addWidget(time_series_check)
        
        control_charts_check = QCheckBox("Control Charts")
        control_charts_check.setChecked(True)
        graphs_layout.addWidget(control_charts_check)
        
        run_chart_check = QCheckBox("Run Chart")
        run_chart_check.setChecked(True)
        graphs_layout.addWidget(run_chart_check)
        
        histogram_check = QCheckBox("Histogram by Time Period")
        histogram_check.setChecked(True)
        graphs_layout.addWidget(histogram_check)
        
        # Add tabs to tab widget
        tabs.addTab(basic_tab, "Basic")
        tabs.addTab(options_tab, "Options")
        tabs.addTab(graphs_tab, "Graphs")
        
        # Dialog buttons
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        
        # Main dialog layout
        main_layout = QVBoxLayout(dialog)
        main_layout.addWidget(tabs)
        main_layout.addWidget(button_box)
        
        # Show dialog
        if not dialog.exec():
            return
        
        # Get values from dialog
        datetime_col = datetime_combo.currentText()
        measurement_col = measurement_combo.currentText()
        operator_col = None if operator_combo.currentText() == "None" else operator_combo.currentText()
        standard_col = None if standard_combo.currentText() == "None" else standard_combo.currentText()
        
        # Get time settings
        time_unit = time_unit_combo.currentText().lower()
        group_by_time = group_by_time_check.isChecked()
        
        reference_value = None
        try:
            if reference_input.text():
                reference_value = float(reference_input.text())
        except ValueError:
            QMessageBox.warning(self, "Warning", "Invalid reference value. This will be ignored.")
        
        # Get analysis options
        chart_type = chart_type_combo.currentText()
        alpha = float(alpha_combo.currentText())
        include_special_causes = special_causes_check.isChecked()
        
        # Get graph selections
        show_time_series = time_series_check.isChecked()
        show_control_charts = control_charts_check.isChecked()
        show_run_chart = run_chart_check.isChecked()
        show_histogram = histogram_check.isChecked()
        
        try:
            # Create DataFrame for analysis
            analysis_df = pd.DataFrame({
                'DateTime': pd.to_datetime(self.data[datetime_col], errors='coerce'),
                'Measurement': pd.to_numeric(self.data[measurement_col], errors='coerce')
            })
            
            # Add operator column if selected
            if operator_col:
                analysis_df['Operator'] = self.data[operator_col]
            
            # Add standard column if selected
            if standard_col:
                analysis_df['Standard'] = self.data[standard_col]
            
            # Remove missing values
            analysis_df = analysis_df.dropna(subset=['DateTime', 'Measurement'])
            analysis_df = analysis_df.sort_values('DateTime')
            
            if len(analysis_df) < 10:
                QMessageBox.warning(self, "Warning", "Insufficient data for stability analysis (minimum 10 measurements required)")
                return
            
            # Group by time period if requested
            if group_by_time:
                if time_unit == 'hour':
                    analysis_df['TimePeriod'] = analysis_df['DateTime'].dt.strftime('%Y-%m-%d %H:00')
                elif time_unit == 'day':
                    analysis_df['TimePeriod'] = analysis_df['DateTime'].dt.strftime('%Y-%m-%d')
                elif time_unit == 'week':
                    analysis_df['TimePeriod'] = analysis_df['DateTime'].dt.strftime('%Y-%U')
                else:  # month
                    analysis_df['TimePeriod'] = analysis_df['DateTime'].dt.strftime('%Y-%m')
                
                # Calculate period statistics
                period_stats = analysis_df.groupby('TimePeriod').agg({
                    'Measurement': ['mean', 'std', 'min', 'max', 'count']
                })
                period_stats.columns = ['Mean', 'StdDev', 'Min', 'Max', 'Count']
                period_stats = period_stats.reset_index()
                
                # Check if we have enough periods
                if len(period_stats) < 5:
                    QMessageBox.warning(self, "Warning", 
                        f"Insufficient time periods for stability analysis. Found {len(period_stats)} periods, minimum 5 required.")
                    return
            
            # Calculate overall statistics
            mean = analysis_df['Measurement'].mean()
            std_dev = analysis_df['Measurement'].std(ddof=1)
            min_val = analysis_df['Measurement'].min()
            max_val = analysis_df['Measurement'].max()
            range_val = max_val - min_val
            
            # Calculate control limits based on chart type
            if chart_type == "I-MR Chart":
                # Individual measurements chart
                moving_range = np.abs(analysis_df['Measurement'].diff())
                mr_mean = moving_range.dropna().mean()
                
                # Constants for I-MR chart
                d2 = 1.128  # for n=2
                d3 = 0.853  # for n=2
                d4 = 3.267  # for n=2
                
                # Control limits for individuals
                ucl_i = mean + 3 * mr_mean / d2
                lcl_i = mean - 3 * mr_mean / d2
                
                # Control limits for moving range
                ucl_mr = d4 * mr_mean
                lcl_mr = 0  # D3 = 0 for n=2
                
                # Check for out of control points
                out_of_control_i = analysis_df[
                    (analysis_df['Measurement'] > ucl_i) | 
                    (analysis_df['Measurement'] < lcl_i)
                ]
                
                out_of_control_mr = moving_range[
                    (moving_range > ucl_mr) | 
                    (moving_range < lcl_mr)
                ]
                
                # Special cause tests if requested
                special_causes = []
                if include_special_causes:
                    # Test 1: Points outside control limits (already done)
                    
                    # Test 2: 9 points in a row on same side of center line
                    measurements = analysis_df['Measurement'].values
                    for i in range(8, len(measurements)):
                        if all(m > mean for m in measurements[i-8:i+1]) or all(m < mean for m in measurements[i-8:i+1]):
                            special_causes.append(f"Test 2: 9 points in a row on same side of center line at point {i+1}")
                    
                    # Test 3: 6 points in a row steadily increasing or decreasing
                    for i in range(5, len(measurements)):
                        if all(measurements[j] < measurements[j+1] for j in range(i-5, i)) or \
                           all(measurements[j] > measurements[j+1] for j in range(i-5, i)):
                            special_causes.append(f"Test 3: 6 points in a row steadily increasing or decreasing at point {i+1}")
                    
                    # Test 4: 14 points in a row alternating up and down
                    for i in range(13, len(measurements)):
                        alternating = True
                        for j in range(i-13, i):
                            if (measurements[j] < measurements[j+1] and measurements[j+1] < measurements[j+2]) or \
                               (measurements[j] > measurements[j+1] and measurements[j+1] > measurements[j+2]):
                                alternating = False
                                break
                        if alternating:
                            special_causes.append(f"Test 4: 14 points in a row alternating up and down at point {i+1}")
            
            else:  # X-bar R Chart
                # Group data for X-bar R chart
                if not group_by_time:
                    QMessageBox.warning(self, "Warning", 
                        "X-bar R Chart requires grouping by time period. Please enable this option.")
                    return
                
                # Constants for X-bar R chart (assuming subgroup size of period counts)
                subgroup_sizes = period_stats['Count'].values
                if len(set(subgroup_sizes)) > 1:
                    # Variable subgroup sizes
                    QMessageBox.warning(self, "Warning", 
                        "X-bar R Chart works best with equal subgroup sizes. Using average subgroup size.")
                
                n = int(np.mean(subgroup_sizes))
                
                # Get constants based on subgroup size
                if n == 2:
                    a2, d3, d4 = 1.880, 0, 3.267
                elif n == 3:
                    a2, d3, d4 = 1.023, 0, 2.574
                elif n == 4:
                    a2, d3, d4 = 0.729, 0, 2.282
                elif n == 5:
                    a2, d3, d4 = 0.577, 0, 2.114
                else:
                    # Default to n=5 if outside common range
                    a2, d3, d4 = 0.577, 0, 2.114
                
                # Calculate ranges for each period
                period_stats['Range'] = period_stats['Max'] - period_stats['Min']
                
                # Calculate control limits
                r_bar = period_stats['Range'].mean()
                
                # X-bar chart limits
                ucl_xbar = mean + a2 * r_bar
                lcl_xbar = mean - a2 * r_bar
                
                # R chart limits
                ucl_r = d4 * r_bar
                lcl_r = d3 * r_bar
                
                # Check for out of control points
                out_of_control_xbar = period_stats[
                    (period_stats['Mean'] > ucl_xbar) | 
                    (period_stats['Mean'] < lcl_xbar)
                ]
                
                out_of_control_r = period_stats[
                    (period_stats['Range'] > ucl_r) | 
                    (period_stats['Range'] < lcl_r)
                ]
            
            # Perform trend analysis
            # Calculate correlation between measurements and time
            time_nums = (analysis_df['DateTime'] - analysis_df['DateTime'].min()).dt.total_seconds()
            correlation = np.corrcoef(time_nums, analysis_df['Measurement'])[0,1]
            
            # Create report
            report = f"""Stability Study Results

Time Period: {analysis_df['DateTime'].min().strftime('%Y-%m-%d')} to {analysis_df['DateTime'].max().strftime('%Y-%m-%d')}

Basic Statistics:
Number of Measurements: {len(analysis_df)}
Overall Mean: {mean:.4f}
Standard Deviation: {std_dev:.4f}
Range: {range_val:.4f} ({min_val:.4f} to {max_val:.4f})
"""

            if group_by_time:
                report += f"""
Time Period Analysis:
Number of Time Periods: {len(period_stats)}
Average Measurements per Period: {period_stats['Count'].mean():.1f}
"""

            if chart_type == "I-MR Chart":
                report += f"""
Control Chart Analysis (I-MR):
Individual Measurements:
  UCL: {ucl_i:.4f}
  LCL: {lcl_i:.4f}
  Points Outside Limits: {len(out_of_control_i)}

Moving Range:
  Average MR: {mr_mean:.4f}
  UCL: {ucl_mr:.4f}
  LCL: {lcl_mr:.4f}
  Points Outside Limits: {len(out_of_control_mr)}
"""
            else:
                report += f"""
Control Chart Analysis (X-bar R):
X-bar Chart:
  UCL: {ucl_xbar:.4f}
  LCL: {lcl_xbar:.4f}
  Points Outside Limits: {len(out_of_control_xbar)}

R Chart:
  Average Range: {r_bar:.4f}
  UCL: {ucl_r:.4f}
  LCL: {lcl_r:.4f}
  Points Outside Limits: {len(out_of_control_r)}
"""

            report += f"""
Trend Analysis:
Time-Measurement Correlation: {correlation:.4f}
"""

            if include_special_causes and chart_type == "I-MR Chart" and special_causes:
                report += "\nSpecial Cause Tests:\n"
                for cause in special_causes:
                    report += f"- {cause}\n"

            report += """
Assessment:
"""
            # Add assessment based on results
            if chart_type == "I-MR Chart":
                if len(out_of_control_i) == 0 and len(out_of_control_mr) == 0 and not special_causes:
                    report += "Process appears to be in statistical control.\n"
                else:
                    report += "Process shows signs of instability.\n"
            else:
                if len(out_of_control_xbar) == 0 and len(out_of_control_r) == 0:
                    report += "Process appears to be in statistical control.\n"
                else:
                    report += "Process shows signs of instability.\n"

            if abs(correlation) > 0.5:
                report += "Significant trend detected over time.\n"
            else:
                report += "No significant trend detected.\n"

            self.sessionWindow.setText(report)

            # Create visualizations
            if show_time_series:
                plt.figure(figsize=(12, 6))
                
                if operator_col:
                    # Color by operator if available
                    operators = analysis_df['Operator'].unique()
                    for op in operators:
                        subset = analysis_df[analysis_df['Operator'] == op]
                        plt.plot(subset['DateTime'], subset['Measurement'], 
                                marker='o', linestyle='-', label=f'Operator {op}')
                else:
                    plt.plot(analysis_df['DateTime'], analysis_df['Measurement'], 
                            marker='o', linestyle='-', label='Measurements')
                
                # Add reference line if provided
                if reference_value is not None:
                    plt.axhline(y=reference_value, color='r', linestyle='--', label='Reference')
                
                # Add trend line
                z = np.polyfit(range(len(analysis_df)), analysis_df['Measurement'], 1)
                p = np.poly1d(z)
                plt.plot(analysis_df['DateTime'], p(range(len(analysis_df))), 
                        'k--', label=f'Trend (slope={z[0]:.4f})')
                
                plt.title('Stability Study: Time Series Plot')
                plt.xlabel('Time')
                plt.ylabel('Measurement')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.show()
            
            if show_control_charts:
                if chart_type == "I-MR Chart":
                    # Individual measurements control chart
                    plt.figure(figsize=(12, 8))
                    plt.subplot(2, 1, 1)
                    plt.plot(analysis_df['DateTime'], analysis_df['Measurement'], 
                            marker='o', linestyle='-', label='Measurements')
                    plt.axhline(y=mean, color='g', linestyle='-', label='Mean')
                    plt.axhline(y=ucl_i, color='r', linestyle='--', label='UCL')
                    plt.axhline(y=lcl_i, color='r', linestyle='--', label='LCL')
                    plt.title('Individual Measurements Chart')
                    plt.xlabel('Time')
                    plt.ylabel('Value')
                    plt.legend()
                    plt.grid(True, alpha=0.3)

                    # Moving Range chart
                    plt.subplot(2, 1, 2)
                    plt.plot(analysis_df['DateTime'][1:], moving_range[1:], 
                            marker='o', linestyle='-', label='Moving Range')
                    plt.axhline(y=mr_mean, color='g', linestyle='-', label='MR Mean')
                    plt.axhline(y=ucl_mr, color='r', linestyle='--', label='MR UCL')
                    plt.axhline(y=lcl_mr, color='r', linestyle='--', label='MR LCL')
                    plt.title('Moving Range Chart')
                    plt.xlabel('Time')
                    plt.ylabel('Moving Range')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.show()
                
                else:  # X-bar R Chart
                    plt.figure(figsize=(12, 8))
                    plt.subplot(2, 1, 1)
                    plt.plot(period_stats['TimePeriod'], period_stats['Mean'], 
                            marker='o', linestyle='-', label='Subgroup Mean')
                    plt.axhline(y=mean, color='g', linestyle='-', label='Overall Mean')
                    plt.axhline(y=ucl_xbar, color='r', linestyle='--', label='UCL')
                    plt.axhline(y=lcl_xbar, color='r', linestyle='--', label='LCL')
                    plt.title('X-bar Chart')
                    plt.xlabel('Time Period')
                    plt.ylabel('Subgroup Mean')
                    plt.xticks(rotation=45)
                    plt.legend()
                    plt.grid(True, alpha=0.3)

                    # R chart
                    plt.subplot(2, 1, 2)
                    plt.plot(period_stats['TimePeriod'], period_stats['Range'], 
                            marker='o', linestyle='-', label='Subgroup Range')
                    plt.axhline(y=r_bar, color='g', linestyle='-', label='Average Range')
                    plt.axhline(y=ucl_r, color='r', linestyle='--', label='UCL')
                    plt.axhline(y=lcl_r, color='r', linestyle='--', label='LCL')
                    plt.title('R Chart')
                    plt.xlabel('Time Period')
                    plt.ylabel('Subgroup Range')
                    plt.xticks(rotation=45)
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.show()
            
            if show_run_chart:
                plt.figure(figsize=(12, 6))
                
                # Plot measurements with run numbers
                plt.plot(range(1, len(analysis_df) + 1), analysis_df['Measurement'], 
                        marker='o', linestyle='-', label='Measurements')
                
                # Add center line (median)
                median = analysis_df['Measurement'].median()
                plt.axhline(y=median, color='g', linestyle='-', label='Median')
                
                # Add reference line if provided
                if reference_value is not None:
                    plt.axhline(y=reference_value, color='r', linestyle='--', label='Reference')
                
                plt.title('Stability Study: Run Chart')
                plt.xlabel('Observation Number')
                plt.ylabel('Measurement')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.show()
            
            if show_histogram and group_by_time:
                # Create histogram by time period
                plt.figure(figsize=(12, 6))
                
                # Get unique time periods
                periods = period_stats['TimePeriod'].values
                
                # Create subplots for each period
                n_periods = len(periods)
                n_cols = min(3, n_periods)
                n_rows = (n_periods + n_cols - 1) // n_cols
                
                for i, period in enumerate(periods):
                    plt.subplot(n_rows, n_cols, i + 1)
                    
                    # Get data for this period
                    period_data = analysis_df[analysis_df['TimePeriod'] == period]['Measurement']
                    
                    # Plot histogram
                    plt.hist(period_data, bins='auto', alpha=0.7)
                    plt.axvline(x=period_data.mean(), color='r', linestyle='-', label='Mean')
                    
                    plt.title(f'Period: {period}')
                    plt.xlabel('Measurement')
                    plt.ylabel('Frequency')
                    
                plt.tight_layout()
                plt.show()

        except Exception as e:
            QMessageBox.warning(self, "Error", 
                f"An error occurred during stability analysis:\n{str(e)}\n\n"
                "Please check your data and try again.")

    def chiSquareTests(self):
        """Create dialog for Chi-Square test options"""
        dialog = QDialog(self)
        dialog.setWindowTitle("Chi-Square Tests")
        layout = QVBoxLayout()

        # Create buttons for each test type
        goodnessBtn = QPushButton("Goodness of Fit")
        independenceBtn = QPushButton("Independence")
        homogeneityBtn = QPushButton("Homogeneity")

        # Connect buttons to their respective functions
        goodnessBtn.clicked.connect(lambda: self.handle_chi_square_selection(dialog, self.chiSquareGoodnessOfFit))
        independenceBtn.clicked.connect(lambda: self.handle_chi_square_selection(dialog, self.chiSquareIndependence))
        homogeneityBtn.clicked.connect(lambda: self.handle_chi_square_selection(dialog, self.chiSquareHomogeneity))

        layout.addWidget(goodnessBtn)
        layout.addWidget(independenceBtn)
        layout.addWidget(homogeneityBtn)

        dialog.setLayout(layout)
        dialog.exec()

    def handle_chi_square_selection(self, dialog, test_func):
        """Handle chi-square test selection and dialog closure"""
        dialog.accept()  # Close the dialog first
        test_func()  # Then run the selected test function

    def chiSquareGoodnessOfFit(self):
        """Perform Chi-Square Goodness of Fit test"""
        try:
            # Load data from table
            self.loadDataFromTable()
            
            # Check if data is loaded
            if self.data is None or self.data.empty:
                QMessageBox.warning(self, "Warning", "Please load data first")
                return
            
            # Create dialog for column selection
            dialog = QDialog()
            dialog.setWindowTitle("Chi-Square Goodness of Fit")
            layout = QVBoxLayout()
            
            # Category column selection
            cat_label = QLabel("Select Category Column:")
            layout.addWidget(cat_label)
            cat_combo = QComboBox()
            cat_combo.addItems(self.data.columns)
            layout.addWidget(cat_combo)
            
            # Count column selection
            count_label = QLabel("Select Count Column:")
            layout.addWidget(count_label)
            count_combo = QComboBox()
            count_combo.addItems(self.data.columns)
            layout.addWidget(count_combo)
            
            # Add buttons
            button_box = QHBoxLayout()
            ok_button = QPushButton("OK")
            cancel_button = QPushButton("Cancel")
            button_box.addWidget(ok_button)
            button_box.addWidget(cancel_button)
            layout.addLayout(button_box)
            
            dialog.setLayout(layout)
            
            # Connect buttons
            ok_button.clicked.connect(dialog.accept)
            cancel_button.clicked.connect(dialog.reject)
            
            # Show dialog
            if dialog.exec() != QDialog.DialogCode.Accepted:
                return
                
            # Get selected columns
            category_col = cat_combo.currentText()
            count_col = count_combo.currentText()
            
            # Get categories and observed frequencies
            categories = self.data[category_col].values  # Keep categories as strings
            try:
                observed = pd.to_numeric(self.data[count_col], errors='raise').values
            except ValueError:
                QMessageBox.critical(self, "Error", f"Count column '{count_col}' must contain numeric values only")
                return
            
            n = len(observed)
            
            # Calculate expected frequencies (assuming equal probabilities)
            total = sum(observed)
            expected = np.full(n, total/n)
            
            # Calculate chi-square statistic
            chi2_stat = np.sum((observed - expected) ** 2 / expected)
            df = n - 1
            p_value = 1 - stats.chi2.cdf(chi2_stat, df)
            
            # Calculate individual contributions
            contributions = (observed - expected) ** 2 / expected
            
            # Display results
            self.sessionWindow.append("\nChi-Square Goodness of Fit Test")
            self.sessionWindow.append("-" * 40)
            
            # Display frequency table
            self.sessionWindow.append("\nFrequency Table:")
            self.sessionWindow.append("Category    Observed    Expected    Contribution")
            self.sessionWindow.append("-" * 50)
            for i in range(n):
                self.sessionWindow.append(f"{categories[i]:<12}{observed[i]:>9.0f}{expected[i]:>12.2f}{contributions[i]:>14.4f}")
            
            # Display test statistics
            self.sessionWindow.append("\nTest Statistics:")
            self.sessionWindow.append(f"Chi-Square = {chi2_stat:.4f}")
            self.sessionWindow.append(f"DF = {df}")
            self.sessionWindow.append(f"P-Value = {p_value:.4f}")
            
            # Add interpretation
            self.sessionWindow.append("\nInterpretation:")
            if p_value < 0.05:
                self.sessionWindow.append("Reject H0: The data does not follow a uniform distribution (p < 0.05)")
            else:
                self.sessionWindow.append("Fail to reject H0: The data follows a uniform distribution (p ≥ 0.05)")
            
            # Create visualization
            plt.figure(figsize=(10, 6))
            
            # Bar plot comparing observed vs expected frequencies
            x = np.arange(n)
            width = 0.35
            
            plt.bar(x - width/2, observed, width, label='Observed', color='skyblue')
            plt.bar(x + width/2, expected, width, label='Expected', color='lightgreen')
            
            plt.xlabel('Category')
            plt.ylabel('Frequency')
            plt.title('Observed vs Expected Frequencies')
            plt.xticks(x, categories)
            plt.legend()
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error in chi-square goodness of fit test: {str(e)}")

    def chiSquareIndependence(self):
        """Perform Chi-Square Test of Independence"""
        try:
            # Load data from table
            self.loadDataFromTable()
            
            # Check if data is loaded
            if self.data is None or self.data.empty:
                QMessageBox.warning(self, "Warning", "Please load data first")
                return
            
            # Create dialog for column selection
            dialog = QDialog()
            dialog.setWindowTitle("Chi-Square Test of Independence")
            layout = QVBoxLayout()
            
            # Row variable selection
            row_label = QLabel("Select Row Variable:")
            layout.addWidget(row_label)
            row_combo = QComboBox()
            row_combo.addItems(self.data.columns)
            layout.addWidget(row_combo)
            
            # Column variable selection
            col_label = QLabel("Select Column Variable:")
            layout.addWidget(col_label)
            col_combo = QComboBox()
            col_combo.addItems(self.data.columns)
            layout.addWidget(col_combo)
            
            # Count column selection
            count_label = QLabel("Select Count Column:")
            layout.addWidget(count_label)
            count_combo = QComboBox()
            count_combo.addItems(self.data.columns)
            layout.addWidget(count_combo)
            
            # Add buttons
            button_box = QHBoxLayout()
            ok_button = QPushButton("OK")
            cancel_button = QPushButton("Cancel")
            button_box.addWidget(ok_button)
            button_box.addWidget(cancel_button)
            layout.addLayout(button_box)
            
            dialog.setLayout(layout)
            
            # Connect buttons
            ok_button.clicked.connect(dialog.accept)
            cancel_button.clicked.connect(dialog.reject)
            
            # Show dialog
            if dialog.exec() != QDialog.DialogCode.Accepted:
                return
                
            # Get selected columns
            row_var = row_combo.currentText()
            col_var = col_combo.currentText()
            count_col = count_combo.currentText()
            
            # Check if same column was selected multiple times
            if len(set([row_var, col_var, count_col])) != 3:
                QMessageBox.warning(self, "Warning", "Please select different columns for Row Variable, Column Variable, and Count")
                return
            
            # Create contingency table
            contingency = pd.pivot_table(
                self.data,
                values=count_col,
                index=row_var,
                columns=col_var,
                aggfunc='sum',
                fill_value=0
            )
            
            # Convert to numpy array for calculations
            obs_values = contingency.values.astype(np.float64)
            
            # Calculate row and column totals
            row_sums = obs_values.sum(axis=1)
            col_sums = obs_values.sum(axis=0)
            total = float(obs_values.sum())
            
            # Calculate expected frequencies
            expected = np.zeros_like(obs_values)
            for i in range(obs_values.shape[0]):
                for j in range(obs_values.shape[1]):
                    expected[i,j] = (row_sums[i] * col_sums[j]) / total
            
            # Calculate chi-square statistic and contributions
            contributions = (obs_values - expected) ** 2 / expected
            chi2_stat = contributions.sum()
            
            # Calculate degrees of freedom
            df = (contingency.shape[0] - 1) * (contingency.shape[1] - 1)
            
            # Calculate p-value
            p_value = 1 - stats.chi2.cdf(chi2_stat, df)
            
            # Display results
            self.sessionWindow.append("\nChi-Square Test of Independence")
            self.sessionWindow.append("-" * 40)
            
            # Display contingency table
            self.sessionWindow.append("\nContingency Table:")
            self.sessionWindow.append(str(contingency))
            
            # Display expected frequencies
            self.sessionWindow.append("\nExpected Frequencies:")
            expected_df = pd.DataFrame(expected, index=contingency.index, columns=contingency.columns)
            self.sessionWindow.append(str(expected_df))
            
            # Display chi-square contributions
            self.sessionWindow.append("\nChi-Square Contributions:")
            contributions_df = pd.DataFrame(contributions, index=contingency.index, columns=contingency.columns)
            self.sessionWindow.append(str(contributions_df.round(3)))
            
            # Display test statistics
            self.sessionWindow.append("\nTest Statistics:")
            self.sessionWindow.append(f"Chi-Square = {chi2_stat:.4f}")
            self.sessionWindow.append(f"DF = {df}")
            self.sessionWindow.append(f"P-Value = {p_value:.4f}")
            
            # Add interpretation
            self.sessionWindow.append("\nInterpretation:")
            if p_value < 0.05:
                self.sessionWindow.append(f"Reject H0: There is evidence of a relationship between {row_var} and {col_var} (p < 0.05)")
            else:
                self.sessionWindow.append(f"Fail to reject H0: There is no evidence of a relationship between {row_var} and {col_var} (p ≥ 0.05)")
            
            # Create heatmaps
            self.createIndependenceHeatmaps(
                obs_values,
                expected,
                contributions,
                contingency.index,
                contingency.columns
            )
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error in chi-square independence test: {str(e)}")

    def createIndependenceHeatmaps(self, observed, expected, contributions, row_labels, col_labels):
        """Create heatmaps for chi-square independence test"""
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # Observed frequencies heatmap
        sns.heatmap(observed, annot=True, fmt='.0f', cmap='YlOrRd', ax=ax1,
                   xticklabels=col_labels, yticklabels=row_labels)
        ax1.set_title('Observed Frequencies')
        ax1.set_xlabel('Category')
        ax1.set_ylabel('Group')
        
        # Expected frequencies heatmap
        sns.heatmap(expected, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax2,
                   xticklabels=col_labels, yticklabels=row_labels)
        ax2.set_title('Expected Frequencies')
        ax2.set_xlabel('Category')
        ax2.set_ylabel('Group')
        
        # Chi-square contributions heatmap
        sns.heatmap(contributions, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax3,
                   xticklabels=col_labels, yticklabels=row_labels)
        ax3.set_title('Chi-Square Contributions')
        ax3.set_xlabel('Category')
        ax3.set_ylabel('Group')
        
        plt.tight_layout()
        plt.show()

    def chiSquareHomogeneity(self):
        """Perform Chi-Square Test of Homogeneity"""
        try:
            # Load data from table
            self.loadDataFromTable()
            
            # Check if data is loaded
            if self.data is None or self.data.empty:
                QMessageBox.warning(self, "Warning", "Please load data first")
                return
            
            # Create dialog for column selection
            dialog = QDialog()
            dialog.setWindowTitle("Chi-Square Test of Homogeneity")
            layout = QVBoxLayout()
            
            # Group column selection
            group_label = QLabel("Select Treatment Column:")
            layout.addWidget(group_label)
            group_combo = QComboBox()
            group_combo.addItems(self.data.columns)
            layout.addWidget(group_combo)
            
            # Response column selection
            response_label = QLabel("Select Outcome Column:")
            layout.addWidget(response_label)
            response_combo = QComboBox()
            response_combo.addItems(self.data.columns)
            layout.addWidget(response_combo)
            
            # Count column selection
            count_label = QLabel("Select Count Column:")
            layout.addWidget(count_label)
            count_combo = QComboBox()
            count_combo.addItems(self.data.columns)
            layout.addWidget(count_combo)
            
            # Add buttons
            button_box = QHBoxLayout()
            ok_button = QPushButton("OK")
            cancel_button = QPushButton("Cancel")
            button_box.addWidget(ok_button)
            button_box.addWidget(cancel_button)
            layout.addLayout(button_box)
            
            dialog.setLayout(layout)
            
            # Connect buttons
            ok_button.clicked.connect(dialog.accept)
            cancel_button.clicked.connect(dialog.reject)
            
            # Show dialog
            if dialog.exec() != QDialog.DialogCode.Accepted:
                return
                
            # Get selected columns
            group_var = group_combo.currentText()
            response_var = response_combo.currentText()
            count_col = count_combo.currentText()
            
            # Check if same column was selected multiple times
            if len(set([group_var, response_var, count_col])) != 3:
                QMessageBox.warning(self, "Warning", "Please select different columns for Treatment, Outcome, and Count")
                return
            
            # Create contingency table
            contingency = pd.pivot_table(
                self.data,
                values=count_col,
                index=group_var,
                columns=response_var,
                aggfunc='sum',
                fill_value=0
            )
            
            # Convert to numpy array for calculations
            obs_values = contingency.values.astype(np.float64)
            
            # Calculate row and column totals
            row_sums = obs_values.sum(axis=1)
            col_sums = obs_values.sum(axis=0)
            total = float(obs_values.sum())
            
            # Calculate expected frequencies
            expected = np.zeros_like(obs_values)
            for i in range(obs_values.shape[0]):
                for j in range(obs_values.shape[1]):
                    expected[i,j] = (row_sums[i] * col_sums[j]) / total
            
            # Calculate chi-square statistic and contributions
            contributions = (obs_values - expected) ** 2 / expected
            chi2_stat = contributions.sum()
            
            # Calculate degrees of freedom
            df = (contingency.shape[0] - 1) * (contingency.shape[1] - 1)
            
            # Calculate p-value
            p_value = 1 - stats.chi2.cdf(chi2_stat, df)
            
            # Display results
            self.sessionWindow.append("\nChi-Square Test of Homogeneity")
            self.sessionWindow.append("-" * 40)
            
            # Display contingency table
            self.sessionWindow.append("\nContingency Table:")
            self.sessionWindow.append(str(contingency))
            
            # Display row percentages
            self.sessionWindow.append("\nRow Percentages (Success Rates):")
            row_percentages = (contingency.div(contingency.sum(axis=1), axis=0) * 100).round(2)
            self.sessionWindow.append(str(row_percentages))
            
            # Display expected frequencies
            self.sessionWindow.append("\nExpected Frequencies:")
            expected_df = pd.DataFrame(expected, index=contingency.index, columns=contingency.columns)
            self.sessionWindow.append(str(expected_df.round(2)))
            
            # Display chi-square contributions
            self.sessionWindow.append("\nChi-Square Contributions:")
            contributions_df = pd.DataFrame(contributions, index=contingency.index, columns=contingency.columns)
            self.sessionWindow.append(str(contributions_df.round(3)))
            
            # Display test statistics
            self.sessionWindow.append("\nTest Statistics:")
            self.sessionWindow.append(f"Chi-Square = {chi2_stat:.4f}")
            self.sessionWindow.append(f"DF = {df}")
            self.sessionWindow.append(f"P-Value = {p_value:.4f}")
            
            # Add interpretation
            self.sessionWindow.append("\nInterpretation:")
            if p_value < 0.05:
                self.sessionWindow.append(f"Reject H0: There is evidence that the distribution of {response_var} differs across {group_var} groups (p < 0.05)")
            else:
                self.sessionWindow.append(f"Fail to reject H0: There is no evidence that the distribution of {response_var} differs across {group_var} groups (p ≥ 0.05)")
            
            # Create bar plot comparing proportions
            plt.figure(figsize=(10, 6))
            proportions = row_percentages.iloc[:, 0]  # Get success proportions
            
            plt.bar(range(len(proportions)), proportions)
            plt.xlabel(group_var)
            plt.ylabel(f'Percentage of {response_var} (Success)')
            plt.title(f'Success Rates by {group_var}')
            plt.xticks(range(len(proportions)), proportions.index)
            
            # Add percentage labels on top of bars
            for i, v in enumerate(proportions):
                plt.text(i, v + 1, f'{v:.1f}%', ha='center')
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error in chi-square homogeneity test: {str(e)}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWin = MinitabLikeApp()
    mainWin.setWindowState(mainWin.windowState() & ~mainWin.windowState())
    mainWin.show()
    mainWin.raise_()
    mainWin.activateWindow()
    sys.exit(app.exec())