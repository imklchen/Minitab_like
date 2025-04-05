"""
Capability module
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QFormLayout, QDoubleSpinBox, 
                           QDialogButtonBox, QLabel, QMessageBox, QInputDialog)


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

def process_capability(main_window):
    """
    Perform Process Capability Analysis
    
    Args:
        main_window: The main window object with data and UI components
    """
    try:
        # Load data from table
        main_window.load_data_from_table()
        
        # Check if data is loaded
        if main_window.data is None or main_window.data.empty:
            QMessageBox.warning(main_window, "Warning", "Please load data first")
            return
        
        # Get measurement and subgroup columns
        measurement_col = main_window.selectColumnDialog("Select Measurement Column")
        if measurement_col is None:
            return
        subgroup_col = main_window.selectColumnDialog("Select Subgroup Column")
        if subgroup_col is None:
            return
        
        # Get specification limits and target
        lsl, ok = QInputDialog.getDouble(main_window, "Lower Spec Limit", "Enter Lower Spec Limit (LSL):", 10.0)
        if not ok:
            return
        usl, ok = QInputDialog.getDouble(main_window, "Upper Spec Limit", "Enter Upper Spec Limit (USL):", 10.8)
        if not ok:
            return
        target, ok = QInputDialog.getDouble(main_window, "Target", "Enter Target Value:", 10.4)
        if not ok:
            return
        
        # Calculate basic statistics
        data = main_window.data[measurement_col].values
        subgroups = main_window.data[subgroup_col].values
        n = len(data)
        mean = np.mean(data)
        
        # Calculate StDev (Within) using R̄/d2 method
        unique_subgroups = np.unique(subgroups)
        ranges = []
        for subgroup in unique_subgroups:
            subgroup_data = data[subgroups == subgroup]
            ranges.append(np.max(subgroup_data) - np.min(subgroup_data))
        r_bar = np.mean(ranges)
        
        # Determine the d2 value based on the subgroup size
        # Common subgroup sizes and their d2 values
        d2_values = {2: 1.128, 3: 1.693, 4: 2.059, 5: 2.326, 6: 2.534, 7: 2.704, 8: 2.847, 9: 2.970, 10: 3.078}
        
        # Get average subgroup size
        subgroup_sizes = []
        for subgroup in unique_subgroups:
            subgroup_size = sum(subgroups == subgroup)
            subgroup_sizes.append(subgroup_size)
        avg_subgroup_size = int(np.mean(subgroup_sizes))
        
        # Use appropriate d2 value or default to 2.059 (for n=4)
        d2 = d2_values.get(avg_subgroup_size, 2.059)
        
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
        
        # Create visualization if available
        try:
            # Create capability chart
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot histogram
            ax.hist(data, bins=30, density=True, alpha=0.6, color='skyblue')
            
            # Plot normal curve
            x = np.linspace(mean - 4*std_overall, mean + 4*std_overall, 100)
            y = stats.norm.pdf(x, mean, std_overall)
            ax.plot(x, y, 'b-', linewidth=2)
            
            # Add specification limits
            ax.axvline(lsl, color='red', linestyle='--', label=f'LSL = {lsl}')
            ax.axvline(usl, color='red', linestyle='--', label=f'USL = {usl}')
            ax.axvline(mean, color='green', linestyle='-', label=f'Mean = {mean:.3f}')
            
            # Add labels and legend
            ax.set_title('Process Capability Analysis')
            ax.set_xlabel('Measurement')
            ax.set_ylabel('Density')
            ax.legend()
            
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Error generating visualization: {e}")
        
        # Display report in session window
        main_window.sessionWindow.setText(report)
        
    except Exception as e:
        QMessageBox.critical(main_window, "Error", f"Error in process capability analysis: {str(e)}")
        import traceback
        traceback.print_exc()

def probabilityAnalysis(main_window):
    """
    Perform probability analysis including normal probability plot and QQ plot
    
    Args:
        main_window: The main window object with data and UI components
    """
    # Use the selectColumnDialog method from the main window
    col = main_window.selectColumnDialog("Select Column for Probability Analysis")
    if col:
        try:
            data = pd.to_numeric(main_window.data[col], errors='coerce').dropna()
            
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
            main_window.sessionWindow.setText(f"Probability Analysis for column: {col}")
            main_window.sessionWindow.append("-" * 40)
            main_window.sessionWindow.append(f"Mean: {mu:.4f}")
            main_window.sessionWindow.append(f"Standard Deviation: {sigma:.4f}")
            main_window.sessionWindow.append(f"Normality Test (D'Agostino's K^2):")
            main_window.sessionWindow.append(f"Statistic: {stat:.4f}")
            main_window.sessionWindow.append(f"P-value: {p_value:.4f}")
            main_window.sessionWindow.append(f"Data {'appears' if p_value > 0.05 else 'does not appear'} to be normally distributed")
            
        except Exception as e:
            QMessageBox.critical(main_window, "Error", f"Error in probability analysis: {str(e)}")
            main_window.sessionWindow.append(f"Error: {str(e)}")