"""
Design of Experiments (DOE) module
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QFormLayout, QComboBox, 
                           QSpinBox, QDialogButtonBox, QLabel, QTableWidget,
                           QTableWidgetItem, QPushButton, QMessageBox, 
                           QFileDialog, QInputDialog, QLineEdit, QCheckBox,
                           QHBoxLayout)


def create_doe(main_window):
    """Create Design of Experiments"""
    # Create dialog for DOE type selection
    doe_type, ok = QInputDialog.getItem(main_window, "Design of Experiments",
        "Select DOE Type:",
        ["2-level Factorial", "Fractional Factorial", "Response Surface"], 0, False)
    if not ok:
        return

    if doe_type == "2-level Factorial":
        create_factorial_design(main_window)
    elif doe_type == "Fractional Factorial":
        create_fractional_factorial(main_window)
    else:
        create_response_surface(main_window)

def create_factorial_design(main_window):
    """Create 2-level factorial design"""
    # Get number of factors
    n_factors, ok = QInputDialog.getInt(main_window, "Factorial Design", 
        "Enter number of factors (2-6):", 2, 2, 6)
    if not ok:
        return

    # Get factor names and levels
    factors = []
    levels = []
    for i in range(n_factors):
        # Get factor name
        name, ok = QInputDialog.getText(main_window, f"Factor {i+1}", 
            f"Enter name for factor {i+1}:")
        if not ok:
            return
        factors.append(name)
        
        # Get factor levels
        low, ok = QInputDialog.getText(main_window, f"Factor {i+1} Low", 
            f"Enter low level for {name}:")
        if not ok:
            return
        high, ok = QInputDialog.getText(main_window, f"Factor {i+1} High", 
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
    main_window.data = df
    main_window.update_table_from_data()

    # Show design summary
    summary = f"""2-level Factorial Design Summary

Number of factors: {n_factors}
Number of runs: {n_runs}
Base design: Full factorial

Factors and Levels:
"""
    for i, factor in enumerate(factors):
        summary += f"{factor}: {levels[i][0]} | {levels[i][1]}\n"

    main_window.sessionWindow.setText(summary)

def create_fractional_factorial(main_window):
    """Create fractional factorial design"""
    # Get number of factors
    n_factors, ok = QInputDialog.getInt(main_window, "Fractional Factorial Design", 
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
        QMessageBox.warning(main_window, "Warning", 
            "No valid resolution available for this number of factors")
        return

    resolution, ok = QInputDialog.getItem(main_window, "Fractional Factorial Design",
        "Select design resolution:", resolution_options, 0, False)
    if not ok:
        return
    
    resolution_level = int(resolution.split()[-1])

    # Get factor names and levels
    factors = []
    levels = []
    for i in range(n_factors):
        # Get factor name
        name, ok = QInputDialog.getText(main_window, f"Factor {i+1}", 
            f"Enter name for factor {i+1}:")
        if not ok:
            return
        factors.append(name)
        
        # Get factor levels
        low, ok = QInputDialog.getText(main_window, f"Factor {i+1} Low", 
            f"Enter low level for {name}:")
        if not ok:
            return
        high, ok = QInputDialog.getText(main_window, f"Factor {i+1} High", 
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
    main_window.data = df
    main_window.update_table_from_data()

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

    main_window.sessionWindow.setText(summary)

def create_response_surface(main_window):
    """Create response surface design"""
    # Get design type
    design_type, ok = QInputDialog.getItem(main_window, "Response Surface Design",
        "Select design type:",
        ["Central Composite Design (CCD)", "Box-Behnken Design (BBD)"], 0, False)
    if not ok:
        return

    # Get number of factors
    min_factors = 2 if design_type.startswith("Central") else 3
    max_factors = 6 if design_type.startswith("Central") else 7
    n_factors, ok = QInputDialog.getInt(main_window, "Response Surface Design", 
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
        name, ok = QInputDialog.getText(main_window, f"Factor {i+1}", 
            f"Enter name for factor {i+1}:")
        if not ok:
            return
        factors.append(name)
        
        # Get factor center point and range
        center, ok = QInputDialog.getDouble(main_window, f"Factor {i+1} Center", 
            f"Enter center point for {name}:")
        if not ok:
            return
        center_points.append(center)
        
        range_val, ok = QInputDialog.getDouble(main_window, f"Factor {i+1} Range", 
            f"Enter range (±) for {name}:")
        if not ok:
            return
        ranges.append(range_val)

    # Get number of center points
    n_center, ok = QInputDialog.getInt(main_window, "Center Points", 
        "Enter number of center points:", 3, 1, 10)
    if not ok:
        return

    # Create design matrix based on design type
    if design_type.startswith("Central"):
        # Central Composite Design
        # Get alpha type
        alpha_type, ok = QInputDialog.getItem(main_window, "CCD Alpha",
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
            QMessageBox.warning(main_window, "Warning", 
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
    main_window.data = df
    main_window.update_table_from_data()

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

    main_window.sessionWindow.setText(summary)

def analyze_doe(main_window):
    """Analyze Design of Experiments"""
    # Load the current data from the table
    main_window.load_data_from_table()
    
    if main_window.data.empty:
        QMessageBox.warning(main_window, "Warning", "No data available for analysis")
        return

    try:
        # Check if we have response data
        if 'Response' not in main_window.data.columns:
            QMessageBox.warning(main_window, "Warning", "No Response column found")
            return

        # Identify factor columns (exclude StdOrder, RunOrder, Response, and actual value columns)
        factor_cols = [col for col in main_window.data.columns 
                      if col not in ['StdOrder', 'RunOrder', 'Response', 'PointType'] 
                      and not col.endswith('_actual')]

        if not factor_cols:
            QMessageBox.warning(main_window, "Warning", "No factor columns identified")
            return

        # Create a copy of the data to avoid modifying the original
        analysis_data = main_window.data.copy()

        # Convert response to numeric, dropping any non-numeric values
        analysis_data['Response'] = pd.to_numeric(analysis_data['Response'].astype(str).str.strip(), errors='coerce')
        
        # Drop any rows with missing response values
        analysis_data = analysis_data.dropna(subset=['Response'])

        if len(analysis_data) == 0:
            QMessageBox.warning(main_window, "Warning", "No valid response data after conversion")
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
                QMessageBox.warning(main_window, "Warning", f"Factor {col} has no variation")
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
        main_window.sessionWindow.setText(report)

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
        QMessageBox.warning(main_window, "Error", 
            f"An error occurred during DOE analysis:\n{str(e)}\n\n"
            "Please check your data and try again.")
        import traceback
        traceback.print_exc()