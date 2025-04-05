"""
Metrics module
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QFormLayout, QSpinBox, 
                           QDoubleSpinBox, QDialogButtonBox, QLabel, 
                           QComboBox, QMessageBox, QInputDialog)


def calculate_dpmo(defects, opportunities, units):
    """Calculate Defects Per Million Opportunities"""
    if opportunities <= 0 or units <= 0:
        raise ValueError("Opportunities and units must be positive numbers")
    return (defects / (opportunities * units)) * 1000000

def dpmo_to_sigma(dpmo):
    """Convert DPMO to Sigma Level"""
    if dpmo <= 0:
        return 6.0  # For perfect process (0 defects), we assign 6 sigma
    elif dpmo > 1000000:
        return 0.0  # For processes with more than 1M DPMO, we assign 0 sigma
    return 0.8406 + np.sqrt(29.37 - 2.221 * np.log(dpmo))

def dpmoCalculator(main_window):
    """Calculate DPMO and Sigma Level"""
    # Get defects through dialog
    defects, ok1 = QInputDialog.getInt(main_window, "DPMO Calculator", "Enter number of defects:", 0, 0)
    if not ok1:
        return
        
    # Get opportunities per unit
    opportunities, ok2 = QInputDialog.getInt(main_window, "DPMO Calculator", "Enter opportunities per unit:", 1, 1)
    if not ok2:
        return
        
    # Get number of units
    units, ok3 = QInputDialog.getInt(main_window, "DPMO Calculator", "Enter number of units:", 1, 1)
    if not ok3:
        return
        
    # Calculate DPMO and Sigma level
    dpmo = calculate_dpmo(defects, opportunities, units)
    sigma = dpmo_to_sigma(dpmo)
    
    # Generate report
    report = f"""DPMO Analysis Results

Number of Defects: {defects}
Opportunities per Unit: {opportunities}
Number of Units: {units}

DPMO: {dpmo:.2f}
Sigma Level: {sigma:.2f}
"""
    main_window.sessionWindow.setText(report)

def sigmaLevelCalc(main_window):
    """Calculate Sigma Level from DPMO"""
    dpmo, ok = QInputDialog.getDouble(main_window, "Sigma Calculator", "Enter DPMO:", 0, 0, 1000000, 2)
    if not ok:
        return
        
    sigma = dpmo_to_sigma(dpmo)
    
    # Generate report
    report = f"""Sigma Level Calculation Results

DPMO (Defects Per Million Opportunities): {dpmo:.2f}
Sigma Level: {sigma:.2f}

Note: The sigma calculation includes a 1.5 sigma process shift.
"""
    main_window.sessionWindow.setText(report)

def yieldAnalysis(main_window):
    """Analyze process yield and calculate process capability indices"""
    try:
        # Load data from the table
        main_window.load_data_from_table()
        
        if main_window.data.empty:
            QMessageBox.warning(main_window, "Warning", "No data available for analysis")
            return
            
        # Check if required columns exist
        required_cols = ['Input', 'Output', 'Rework', 'Scrap']
        if not all(col in main_window.data.columns for col in required_cols):
            QMessageBox.warning(main_window, "Warning", "Required columns (Input, Output, Rework, Scrap) not found")
            return
            
        # Get the first row of data for analysis
        row = main_window.data.iloc[0]
        input_units = row['Input']
        output_units = row['Output']
        rework_units = row['Rework']
        scrap_units = row['Scrap']
        
        # Calculate yields and rates
        first_pass_yield = ((output_units - rework_units) / input_units) * 100
        final_yield = (output_units / input_units) * 100
        scrap_rate = (scrap_units / input_units) * 100
        rework_rate = (rework_units / input_units) * 100
        
        # Generate report
        report = """Process Yield Analysis Results
----------------------------

Input: {} units
Output: {} units
Rework: {} units
Scrap: {} units

Calculations:
First Pass Yield = {:.1f}%    # (Output - Rework) / Input
Final Yield = {:.1f}%         # Output / Input
Scrap Rate = {:.1f}%          # Scrap / Input
Rework Rate = {:.1f}%         # Rework / Input
""".format(input_units, output_units, rework_units, scrap_units, 
           first_pass_yield, final_yield, scrap_rate, rework_rate)
        
        # Display results
        main_window.sessionWindow.setText(report)
        
    except Exception as e:
        QMessageBox.critical(main_window, "Error", f"Error in process yield analysis: {str(e)}")
        import traceback
        traceback.print_exc()