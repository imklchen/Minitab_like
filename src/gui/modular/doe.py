"""
Design of Experiments functionality

This module is part of the modularized Minitab-like application.
Generated on: 2025-03-24 21:02:37
"""

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QTableWidget,
    QTableWidgetItem, QMenuBar, QMenu, QTextEdit,
    QDialogButtonBox, QDialog, QLabel, QComboBox,
    QPushButton, QMessageBox, QFileDialog, QSpinBox,
    QDoubleSpinBox, QFormLayout, QLineEdit, QListWidget,
    QHBoxLayout, QGridLayout, QCheckBox, QListWidgetItem,
    QRadioButton, QButtonGroup, QGroupBox, QStackedWidget,
    QStatusBar, QDialogButtonBox, QFormLayout, QGroupBox,
    QLineEdit, QInputDialog, QTabWidget, QDateEdit
)
from PyQt6.QtGui import QAction, QColor
from PyQt6.QtCore import Qt, QDate, QTimer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import seaborn as sns
import scipy.stats as stats
import datetime
import os
import sys
import tempfile
from PyQt6.QtGui import QTextCursor
import itertools
import random
import statsmodels.api as sm


def createDOE(self):
    """Create Design of Experiments"""
    # Show dialog to select DOE type
    dialog = QDialog(self)
    dialog.setWindowTitle("Design of Experiments")
    layout = QVBoxLayout(dialog)
    
    # Create a label
    label = QLabel("Select DOE Type:")
    layout.addWidget(label)
    
    # Create a combobox for selecting DOE type
    combo = QComboBox()
    combo.addItems(["2-level Factorial", "Fractional Factorial", "Response Surface"])
    layout.addWidget(combo)
    
    # Add buttons
    buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
    buttons.accepted.connect(dialog.accept)
    buttons.rejected.connect(dialog.reject)
    layout.addWidget(buttons)
    
    # Show dialog
    result = dialog.exec()
    if result == 1:  # Accepted
        doe_type = combo.currentText()
        if doe_type == "2-level Factorial":
            create_factorial_design(self)
        elif doe_type == "Fractional Factorial":
            create_fractional_factorial(self)
        else:  # Response Surface
            create_response_surface(self)


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
    
    # Generate all combinations of -1 and 1 for each factor
    for combo in itertools.product([-1, 1], repeat=n_factors):
        design_matrix.append(list(combo))
    
    # Convert to pandas DataFrame
    design = pd.DataFrame(design_matrix, columns=factors)
    
    # Add actual value columns
    for i, factor in enumerate(factors):
        # Create a new column with the actual values
        design[f"{factor}_actual"] = design[factor].map({-1: levels[i][0], 1: levels[i][1]})
    
    # Add StdOrder and RunOrder columns
    design.insert(0, "StdOrder", range(1, n_runs + 1))
    
    # Randomize if needed (simple approach with reindex)
    run_order = list(range(1, n_runs + 1))
    random.shuffle(run_order)
    design.insert(1, "RunOrder", run_order)
    
    # Add empty Response column
    design["Response"] = np.nan
    
    # Append to current data
    if not hasattr(self, 'data') or self.data is None or self.data.empty:
        self.data = design
    else:
        # Identify columns to rename if they already exist
        existing_cols = self.data.columns.tolist()
        for col in design.columns:
            if col in existing_cols and col not in ["StdOrder", "RunOrder", "Response"]:
                new_col = f"{col}_1"
                design = design.rename(columns={col: new_col})
        
        # Add new columns to the existing data
        for col in design.columns:
            self.data[col] = pd.Series(dtype=design[col].dtype)
        
        # Fill in the values for the new rows
        for i, row in design.iterrows():
            self.data.loc[i, design.columns] = row
    
    # Update the table widget
    self.updateTable()
    
    # Display summary in session window
    self.sessionWindow.append("\nFull Factorial Design Created")
    self.sessionWindow.append(f"Number of Factors: {n_factors}")
    self.sessionWindow.append(f"Number of Runs: {n_runs}")
    
    # Show factors and levels
    self.sessionWindow.append("\nFactors and Levels:")
    for i, factor in enumerate(factors):
        self.sessionWindow.append(f"{factor}: {levels[i][0]} | {levels[i][1]}")


def create_fractional_factorial(self):
    """Create fractional factorial design"""
    # Get number of factors
    n_factors, ok = QInputDialog.getInt(self, "Fractional Factorial Design", 
        "Enter number of factors (3-7):", 3, 3, 7)
    if not ok:
        return
    
    # Get fraction size (p in 2^(k-p))
    max_p = n_factors - 2  # Ensure at least 2^2 = 4 runs
    p, ok = QInputDialog.getInt(self, "Fraction Size", 
        f"Enter fraction size (1-{max_p}):", 1, 1, max_p)
    if not ok:
        return
    
    # Get factor names and levels (same as factorial)
    factors = []
    levels = []
    for i in range(n_factors):
        name, ok = QInputDialog.getText(self, f"Factor {i+1}", f"Enter name for factor {i+1}:")
        if not ok:
            return
        factors.append(name)
        
        # Get factor levels
        low, ok = QInputDialog.getText(self, f"Factor {i+1} Low", f"Enter low level for {name}:")
        if not ok:
            return
        high, ok = QInputDialog.getText(self, f"Factor {i+1} High", f"Enter high level for {name}:")
        if not ok:
            return
        levels.append([low, high])
    
    # Create base design for 2^(k-p) runs
    base_factors = n_factors - p
    n_runs = 2 ** base_factors
    
    # Create base factorial design
    base_design = []
    for combo in itertools.product([-1, 1], repeat=base_factors):
        base_design.append(list(combo))
    
    # Create the full design with generators
    design_matrix = []
    
    # Add base columns
    for row in base_design:
        # Start with the base factors
        new_row = row.copy()
        
        # Add the derived factors using simple generators
        # For example, X3 = X1*X2, X4 = X1*X3, etc.
        for j in range(base_factors, n_factors):
            # Simple generator: product of first (j % base_factors) + 1 columns
            generator_cols = [(j % base_factors) + 1]
            value = row[generator_cols[0] - 1]  # -1 because 0-based indexing
            new_row.append(value)
        
        design_matrix.append(new_row)
    
    # Convert to pandas DataFrame
    design = pd.DataFrame(design_matrix, columns=factors)
    
    # Add actual value columns
    for i, factor in enumerate(factors):
        design[f"{factor}_actual"] = design[factor].map({-1: levels[i][0], 1: levels[i][1]})
    
    # Add StdOrder and RunOrder columns
    design.insert(0, "StdOrder", range(1, n_runs + 1))
    
    # Randomize if needed
    run_order = list(range(1, n_runs + 1))
    random.shuffle(run_order)
    design.insert(1, "RunOrder", run_order)
    
    # Add empty Response column
    design["Response"] = np.nan
    
    # Append to current data (similar to factorial design)
    if not hasattr(self, 'data') or self.data is None or self.data.empty:
        self.data = design
    else:
        # Identify columns to rename if they already exist
        existing_cols = self.data.columns.tolist()
        for col in design.columns:
            if col in existing_cols and col not in ["StdOrder", "RunOrder", "Response"]:
                new_col = f"{col}_1"
                design = design.rename(columns={col: new_col})
        
        # Add new columns to the existing data
        for col in design.columns:
            self.data[col] = pd.Series(dtype=design[col].dtype)
        
        # Fill in the values for the new rows
        for i, row in design.iterrows():
            self.data.loc[i, design.columns] = row
    
    # Update the table widget
    self.updateTable()
    
    # Display summary in session window
    self.sessionWindow.append("\nFractional Factorial Design Created")
    self.sessionWindow.append(f"Number of Factors: {n_factors}")
    self.sessionWindow.append(f"Fraction: 1/{2**p}")
    self.sessionWindow.append(f"Number of Runs: {n_runs}")
    
    # Show factors and levels
    self.sessionWindow.append("\nFactors and Levels:")
    for i, factor in enumerate(factors):
        self.sessionWindow.append(f"{factor}: {levels[i][0]} | {levels[i][1]}")
    
    # Show generators (simplified)
    if p > 0:
        self.sessionWindow.append("\nDesign Generators (simplified):")
        for j in range(base_factors, n_factors):
            gen_col = (j % base_factors) + 1
            self.sessionWindow.append(f"{factors[j]} = {factors[gen_col-1]}")


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
    self.updateDataFromTable()
    
    if self.data.empty:
        QMessageBox.warning(self, "Warning", "No data available for analysis")
        return

    try:
        # Check if we have response data
        if 'Response' not in self.data.columns:
            QMessageBox.warning(self, "Warning", "No Response column found")
            return

        # Identify factor columns (exclude StdOrder, RunOrder, and Response)
        factor_cols = [col for col in self.data.columns 
                      if col not in ['StdOrder', 'RunOrder', 'Response', 'PointType'] and not col.endswith('_actual')]

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



