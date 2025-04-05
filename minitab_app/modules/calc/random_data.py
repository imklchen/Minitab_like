"""
Random Data Generation Module
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import math

# Try importing from PyQt6 first, then fall back to PyQt5 if necessary
try:
    from PyQt6.QtWidgets import (QInputDialog, QMessageBox)
    using_pyqt6 = True
except ImportError:
    from PyQt5.QtWidgets import (QInputDialog, QMessageBox)
    using_pyqt6 = False

def generate_random_data(main_window, dist_type):
    """
    Generate random data according to the specified distribution
    
    Args:
        main_window: The main application window
        dist_type: The type of distribution (normal, uniform, binomial, poisson)
    """
    size, ok = QInputDialog.getInt(main_window, "Size", f"How many random numbers ({dist_type})?", 100, 1, 1000)
    if not ok:
        return

    # Convert dist_type to title case for consistent comparison
    dist_type = dist_type.title()

    if dist_type == "Normal":
        mean, ok1 = QInputDialog.getDouble(main_window, "Normal Distribution", "Enter Mean:", 100, -1000000, 1000000)
        if not ok1:
            return
        std_dev, ok2 = QInputDialog.getDouble(main_window, "Normal Distribution", "Enter Standard Deviation:", 15, 0.00001, 1000000)
        if not ok2:
            return
        data = np.random.normal(mean, std_dev, size)

    elif dist_type == "Binomial":
        n, ok1 = QInputDialog.getInt(main_window, "Binomial Distribution", "Enter number of trials (n):", 10, 1, 1000)
        if not ok1:
            return
        p, ok2 = QInputDialog.getDouble(main_window, "Binomial Distribution", "Enter probability of success (p):", 0.5, 0, 1)
        if not ok2:
            return
        data = np.random.binomial(n, p, size)

    elif dist_type == "Uniform":
        min_val, ok1 = QInputDialog.getDouble(main_window, "Uniform Distribution", "Enter Minimum:", 0, -1000000, 1000000)
        if not ok1:
            return
        max_val, ok2 = QInputDialog.getDouble(main_window, "Uniform Distribution", "Enter Maximum:", 100, min_val, 1000000)
        if not ok2:
            return
        data = np.random.uniform(min_val, max_val, size)
    
    elif dist_type == "Poisson":
        lambda_param, ok = QInputDialog.getDouble(main_window, "Poisson Distribution", "Enter mean (λ):", 5.0, 0.00001, 1000000)
        if not ok:
            return
        data = np.random.poisson(lambda_param, size)
    else:
        QMessageBox.warning(main_window, "Error", f"Unknown distribution type: {dist_type}")
        return

    # Create DataFrame with the generated data
    main_window.data = pd.DataFrame({dist_type: data})
    
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
    
    main_window.sessionWindow.setText(summary)
    main_window.update_table_from_data()

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

def poisson_distribution(main_window):
    """
    Calculate Poisson distribution probabilities
    
    Args:
        main_window: The main application window
    """
    calculation_type, ok = QInputDialog.getItem(main_window, "Poisson Distribution", 
        "Choose Calculation Type:", 
        ["Probability", "Cumulative Probability", "Inverse Cumulative Probability"], 0, False)
    if not ok:
        return

    mean, ok = QInputDialog.getDouble(main_window, "Poisson Distribution", "Enter Mean (Average number of events):", 3.0, 0.1, 1000, 2)
    if not ok:
        return

    column = main_window.selectColumnDialog()
    if not column:
        return

    # Load data from table before proceeding
    main_window.load_data_from_table()
    
    x_values = pd.to_numeric(main_window.data[column], errors='coerce').dropna().tolist()

    result_text = f"Poisson with mean = {mean}\n\n"

    if calculation_type == "Inverse Cumulative Probability":
        result_text += "P(X ≤ x)\t x\n"
        for prob in x_values:
            # Validate probability is between 0 and 1
            if prob < 0 or prob > 1:
                QMessageBox.warning(main_window, "Error", f"Probability value {prob} is invalid. Must be between 0 and 1.")
                return
            try:
                # Use math.floor since we want the largest value x where P(X ≤ x) ≤ prob
                x = math.floor(stats.poisson.ppf(prob, mean))
                # Ensure x is non-negative (Poisson is only defined for non-negative integers)
                x = max(0, x)
                result_text += f"{prob:.6f}\t{x}\n"
            except Exception as e:
                QMessageBox.warning(main_window, "Error", f"Error calculating inverse cumulative probability: {str(e)}")
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
                QMessageBox.warning(main_window, "Error", f"Error calculating probability: {str(e)}")
                return

    main_window.sessionWindow.setText(result_text)