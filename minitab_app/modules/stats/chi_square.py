"""
Chi-Square Tests Module for statistical analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                           QComboBox, QPushButton, QMessageBox)


def chi_square_tests(main_window):
    """Create dialog for Chi-Square test options"""
    dialog = QDialog(main_window)
    dialog.setWindowTitle("Chi-Square Tests")
    layout = QVBoxLayout()

    # Create buttons for each test type
    goodnessBtn = QPushButton("Goodness of Fit")
    independenceBtn = QPushButton("Independence")
    homogeneityBtn = QPushButton("Homogeneity")

    # Connect buttons to their respective functions
    goodnessBtn.clicked.connect(lambda: handle_chi_square_selection(dialog, lambda: chi_square_goodness_of_fit(main_window)))
    independenceBtn.clicked.connect(lambda: handle_chi_square_selection(dialog, lambda: chi_square_independence(main_window)))
    homogeneityBtn.clicked.connect(lambda: handle_chi_square_selection(dialog, lambda: chi_square_homogeneity(main_window)))

    layout.addWidget(goodnessBtn)
    layout.addWidget(independenceBtn)
    layout.addWidget(homogeneityBtn)

    dialog.setLayout(layout)
    dialog.exec()


def handle_chi_square_selection(dialog, test_func):
    """Handle chi-square test selection and dialog closure"""
    dialog.accept()  # Close the dialog first
    test_func()  # Then run the selected test function


def chi_square_goodness_of_fit(main_window):
    """Perform Chi-Square Goodness of Fit test"""
    try:
        # Load data from table
        main_window.load_data_from_table()
        
        # Check if data is loaded
        if main_window.data is None or main_window.data.empty:
            QMessageBox.warning(main_window, "Warning", "Please load data first")
            return
        
        # Create dialog for column selection
        dialog = QDialog()
        dialog.setWindowTitle("Chi-Square Goodness of Fit")
        layout = QVBoxLayout()
        
        # Category column selection
        cat_label = QLabel("Select Category Column:")
        layout.addWidget(cat_label)
        cat_combo = QComboBox()
        cat_combo.addItems(main_window.data.columns)
        layout.addWidget(cat_combo)
        
        # Count column selection
        count_label = QLabel("Select Count Column:")
        layout.addWidget(count_label)
        count_combo = QComboBox()
        count_combo.addItems(main_window.data.columns)
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
        categories = main_window.data[category_col].values  # Keep categories as strings
        try:
            observed = pd.to_numeric(main_window.data[count_col], errors='raise').values
        except ValueError:
            QMessageBox.critical(main_window, "Error", f"Count column '{count_col}' must contain numeric values only")
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
        main_window.sessionWindow.append("\nChi-Square Goodness of Fit Test")
        main_window.sessionWindow.append("-" * 40)
        
        # Display frequency table
        main_window.sessionWindow.append("\nFrequency Table:")
        main_window.sessionWindow.append("Category    Observed    Expected    Contribution")
        main_window.sessionWindow.append("-" * 50)
        for i in range(n):
            main_window.sessionWindow.append(f"{categories[i]:<12}{observed[i]:>9.0f}{expected[i]:>12.2f}{contributions[i]:>14.4f}")
        
        # Display test statistics
        main_window.sessionWindow.append("\nTest Statistics:")
        main_window.sessionWindow.append(f"Chi-Square = {chi2_stat:.4f}")
        main_window.sessionWindow.append(f"DF = {df}")
        main_window.sessionWindow.append(f"P-Value = {p_value:.4f}")
        
        # Add interpretation
        main_window.sessionWindow.append("\nInterpretation:")
        if p_value < 0.05:
            main_window.sessionWindow.append("Reject H0: The data does not follow a uniform distribution (p < 0.05)")
        else:
            main_window.sessionWindow.append("Fail to reject H0: The data follows a uniform distribution (p ≥ 0.05)")
        
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
        QMessageBox.critical(main_window, "Error", f"Error in chi-square goodness of fit test: {str(e)}")
        import traceback
        traceback.print_exc()


def chi_square_independence(main_window):
    """Perform Chi-Square Test of Independence"""
    try:
        # Load data from table
        main_window.load_data_from_table()
        
        # Check if data is loaded
        if main_window.data is None or main_window.data.empty:
            QMessageBox.warning(main_window, "Warning", "Please load data first")
            return
        
        # Create dialog for column selection
        dialog = QDialog()
        dialog.setWindowTitle("Chi-Square Test of Independence")
        layout = QVBoxLayout()
        
        # Row variable selection
        row_label = QLabel("Select Row Variable:")
        layout.addWidget(row_label)
        row_combo = QComboBox()
        row_combo.addItems(main_window.data.columns)
        layout.addWidget(row_combo)
        
        # Column variable selection
        col_label = QLabel("Select Column Variable:")
        layout.addWidget(col_label)
        col_combo = QComboBox()
        col_combo.addItems(main_window.data.columns)
        layout.addWidget(col_combo)
        
        # Count column selection
        count_label = QLabel("Select Count Column:")
        layout.addWidget(count_label)
        count_combo = QComboBox()
        count_combo.addItems(main_window.data.columns)
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
            QMessageBox.warning(main_window, "Warning", "Please select different columns for Row Variable, Column Variable, and Count")
            return
        
        # Create contingency table
        contingency = pd.pivot_table(
            main_window.data,
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
        main_window.sessionWindow.append("\nChi-Square Test of Independence")
        main_window.sessionWindow.append("-" * 40)
        
        # Display contingency table
        main_window.sessionWindow.append("\nContingency Table:")
        main_window.sessionWindow.append(str(contingency))
        
        # Display expected frequencies
        main_window.sessionWindow.append("\nExpected Frequencies:")
        expected_df = pd.DataFrame(expected, index=contingency.index, columns=contingency.columns)
        main_window.sessionWindow.append(str(expected_df))
        
        # Display chi-square contributions
        main_window.sessionWindow.append("\nChi-Square Contributions:")
        contributions_df = pd.DataFrame(contributions, index=contingency.index, columns=contingency.columns)
        main_window.sessionWindow.append(str(contributions_df.round(3)))
        
        # Display test statistics
        main_window.sessionWindow.append("\nTest Statistics:")
        main_window.sessionWindow.append(f"Chi-Square = {chi2_stat:.4f}")
        main_window.sessionWindow.append(f"DF = {df}")
        main_window.sessionWindow.append(f"P-Value = {p_value:.4f}")
        
        # Add interpretation
        main_window.sessionWindow.append("\nInterpretation:")
        if p_value < 0.05:
            main_window.sessionWindow.append(f"Reject H0: There is evidence of a relationship between {row_var} and {col_var} (p < 0.05)")
        else:
            main_window.sessionWindow.append(f"Fail to reject H0: There is no evidence of a relationship between {row_var} and {col_var} (p ≥ 0.05)")
        
        # Create heatmaps
        create_independence_heatmaps(
            obs_values,
            expected,
            contributions,
            contingency.index,
            contingency.columns
        )
        
    except Exception as e:
        QMessageBox.critical(main_window, "Error", f"Error in chi-square independence test: {str(e)}")
        import traceback
        traceback.print_exc()


def create_independence_heatmaps(observed, expected, contributions, row_labels, col_labels):
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


def chi_square_homogeneity(main_window):
    """Perform Chi-Square Test of Homogeneity"""
    try:
        # Load data from table
        main_window.load_data_from_table()
        
        # Check if data is loaded
        if main_window.data is None or main_window.data.empty:
            QMessageBox.warning(main_window, "Warning", "Please load data first")
            return
        
        # Create dialog for column selection
        dialog = QDialog()
        dialog.setWindowTitle("Chi-Square Test of Homogeneity")
        layout = QVBoxLayout()
        
        # Group column selection
        group_label = QLabel("Select Treatment Column:")
        layout.addWidget(group_label)
        group_combo = QComboBox()
        group_combo.addItems(main_window.data.columns)
        layout.addWidget(group_combo)
        
        # Response column selection
        response_label = QLabel("Select Outcome Column:")
        layout.addWidget(response_label)
        response_combo = QComboBox()
        response_combo.addItems(main_window.data.columns)
        layout.addWidget(response_combo)
        
        # Count column selection
        count_label = QLabel("Select Count Column:")
        layout.addWidget(count_label)
        count_combo = QComboBox()
        count_combo.addItems(main_window.data.columns)
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
            QMessageBox.warning(main_window, "Warning", "Please select different columns for Treatment, Outcome, and Count")
            return
        
        # Create contingency table
        contingency = pd.pivot_table(
            main_window.data,
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
        main_window.sessionWindow.append("\nChi-Square Test of Homogeneity")
        main_window.sessionWindow.append("-" * 40)
        
        # Display contingency table
        main_window.sessionWindow.append("\nContingency Table:")
        main_window.sessionWindow.append(str(contingency))
        
        # Display row percentages
        main_window.sessionWindow.append("\nRow Percentages (Success Rates):")
        row_percentages = (contingency.div(contingency.sum(axis=1), axis=0) * 100).round(2)
        main_window.sessionWindow.append(str(row_percentages))
        
        # Display expected frequencies
        main_window.sessionWindow.append("\nExpected Frequencies:")
        expected_df = pd.DataFrame(expected, index=contingency.index, columns=contingency.columns)
        main_window.sessionWindow.append(str(expected_df.round(2)))
        
        # Display chi-square contributions
        main_window.sessionWindow.append("\nChi-Square Contributions:")
        contributions_df = pd.DataFrame(contributions, index=contingency.index, columns=contingency.columns)
        main_window.sessionWindow.append(str(contributions_df.round(3)))
        
        # Display test statistics
        main_window.sessionWindow.append("\nTest Statistics:")
        main_window.sessionWindow.append(f"Chi-Square = {chi2_stat:.4f}")
        main_window.sessionWindow.append(f"DF = {df}")
        main_window.sessionWindow.append(f"P-Value = {p_value:.4f}")
        
        # Add interpretation
        main_window.sessionWindow.append("\nInterpretation:")
        if p_value < 0.05:
            main_window.sessionWindow.append(f"Reject H0: There is evidence that the distribution of {response_var} differs across {group_var} groups (p < 0.05)")
        else:
            main_window.sessionWindow.append(f"Fail to reject H0: There is no evidence that the distribution of {response_var} differs across {group_var} groups (p ≥ 0.05)")
        
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
        QMessageBox.critical(main_window, "Error", f"Error in chi-square homogeneity test: {str(e)}")
        import traceback
        traceback.print_exc() 