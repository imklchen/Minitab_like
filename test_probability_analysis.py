#!/usr/bin/env python3
"""
Test script for the probability analysis function
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def minimal_probability_analysis(data_column):
    """
    Perform a minimal probability analysis on a data column
    
    Args:
        data_column: Pandas Series or list of numeric values
    """
    # Ensure data is numeric and dropna
    data = pd.to_numeric(data_column, errors='coerce').dropna()
    
    # Calculate mean and standard deviation
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
    
    # Add statistical test results
    stat, p_value = stats.normaltest(data)
    print(f"Probability Analysis Results")
    print("-" * 40)
    print(f"Mean: {mu:.4f}")
    print(f"Standard Deviation: {sigma:.4f}")
    print(f"Normality Test (D'Agostino's K^2):")
    print(f"Statistic: {stat:.4f}")
    print(f"P-value: {p_value:.4f}")
    print(f"Data {'appears' if p_value > 0.05 else 'does not appear'} to be normally distributed")
    
    # Show the plot
    plt.show()

if __name__ == "__main__":
    # Load sample data
    try:
        # Try to load the sample_data.csv file
        data = pd.read_csv("sample_data/sample_data.csv")
        print(f"Loaded data with columns: {', '.join(data.columns)}")
        
        # Use the 'Height' column for analysis
        if 'Height' in data.columns:
            print(f"Running probability analysis on 'Height' column...")
            minimal_probability_analysis(data['Height'])
        else:
            # Try to use the first numeric column
            numeric_cols = data.select_dtypes(include=np.number).columns
            if len(numeric_cols) > 0:
                col_name = numeric_cols[0]
                print(f"No 'Height' column found. Using '{col_name}' instead...")
                minimal_probability_analysis(data[col_name])
            else:
                print("No numeric columns found in the data.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc() 