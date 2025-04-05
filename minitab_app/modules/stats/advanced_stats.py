"""
Advanced Stats module
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QPushButton, QMessageBox, 
                           QInputDialog, QFormLayout, QLineEdit, QComboBox, 
                           QDialogButtonBox, QLabel, QCheckBox)
import statsmodels.api as sm
import traceback


def hypothesis_testing(main_window):
    """Open dialog for hypothesis testing options"""
    dialog = QDialog(main_window)
    dialog.setWindowTitle("Hypothesis Testing")
    layout = QVBoxLayout()

    # Create buttons for each test type
    oneSampleBtn = QPushButton("One-Sample t-Test")
    twoSampleBtn = QPushButton("Two-Sample t-Test")
    pairedBtn = QPushButton("Paired t-Test")

    # Connect buttons to their respective functions with dialog closure
    oneSampleBtn.clicked.connect(lambda: handle_hypothesis_selection(dialog, lambda: one_sample_t_test(main_window)))
    twoSampleBtn.clicked.connect(lambda: handle_hypothesis_selection(dialog, lambda: two_sample_t_test(main_window)))
    pairedBtn.clicked.connect(lambda: handle_hypothesis_selection(dialog, lambda: paired_t_test(main_window)))

    layout.addWidget(oneSampleBtn)
    layout.addWidget(twoSampleBtn)
    layout.addWidget(pairedBtn)

    dialog.setLayout(layout)
    dialog.exec()

def handle_hypothesis_selection(dialog, test_func):
    """Handle hypothesis test selection and dialog closure"""
    dialog.accept()  # Close the dialog first
    test_func()     # Then run the selected test function

def one_sample_t_test(main_window):
    """Perform one-sample t-test"""
    main_window.load_data_from_table()
    if main_window.data.empty:
        QMessageBox.warning(main_window, "Warning", "Please load or enter data first")
        return

    # Get column selection
    col, ok = QInputDialog.getItem(main_window, "Select Column", 
        "Choose column for analysis:", main_window.data.columns.tolist(), 0, False)
    
    if ok and col:
        try:
            # Get hypothesized mean
            hyp_mean, ok = QInputDialog.getDouble(main_window, "Hypothesized Mean", 
                "Enter hypothesized mean value:", 0, -1000000, 1000000, 4)
            
            if ok:
                # Convert data to numeric
                data = pd.to_numeric(main_window.data[col], errors='coerce').dropna()
                
                # Perform t-test
                t_stat, p_value = stats.ttest_1samp(data, hyp_mean)
                
                # Calculate additional statistics
                mean = np.mean(data)
                std_dev = np.std(data, ddof=1)
                se = std_dev / np.sqrt(len(data))
                ci = stats.t.interval(0.95, len(data)-1, loc=mean, scale=se)
                
                # Display results
                main_window.sessionWindow.append("\nOne-Sample t-Test Results")
                main_window.sessionWindow.append("-" * 40)
                main_window.sessionWindow.append(f"Variable: {col}")
                main_window.sessionWindow.append(f"Hypothesized mean = {hyp_mean}")
                main_window.sessionWindow.append(f"\nSample Statistics:")
                main_window.sessionWindow.append(f"Sample Size = {len(data)}")
                main_window.sessionWindow.append(f"Sample Mean = {mean:.4f}")
                main_window.sessionWindow.append(f"Sample StDev = {std_dev:.4f}")
                main_window.sessionWindow.append(f"SE Mean = {se:.4f}")
                main_window.sessionWindow.append(f"\n95% Confidence Interval:")
                main_window.sessionWindow.append(f"({ci[0]:.4f}, {ci[1]:.4f})")
                main_window.sessionWindow.append(f"\nTest Statistics:")
                main_window.sessionWindow.append(f"t-value = {t_stat:.4f}")
                main_window.sessionWindow.append(f"p-value = {p_value:.4f}")
                
                # Add interpretation
                alpha = 0.05
                main_window.sessionWindow.append(f"\nInterpretation:")
                if p_value < alpha:
                    main_window.sessionWindow.append("Reject the null hypothesis")
                    main_window.sessionWindow.append("There is sufficient evidence to conclude that the population mean")
                    main_window.sessionWindow.append(f"is different from {hyp_mean} (at α = 0.05)")
                else:
                    main_window.sessionWindow.append("Fail to reject the null hypothesis")
                    main_window.sessionWindow.append("There is insufficient evidence to conclude that the population mean")
                    main_window.sessionWindow.append(f"is different from {hyp_mean} (at α = 0.05)")

                # Create visualization
                plt.figure(figsize=(10, 6))
                
                # Histogram with normal curve
                plt.hist(data, bins='auto', density=True, alpha=0.7, label='Data')
                
                # Plot normal distribution curve
                x = np.linspace(mean - 4*std_dev, mean + 4*std_dev, 100)
                plt.plot(x, stats.norm.pdf(x, mean, std_dev), 'r-', label='Normal Curve')
                
                # Add hypothesized mean line
                plt.axvline(x=hyp_mean, color='g', linestyle='--', label=f'Hypothesized Mean = {hyp_mean}')
                
                # Add sample mean line
                plt.axvline(x=mean, color='b', linestyle='-', label=f'Sample Mean = {mean:.4f}')
                
                plt.title(f'One-Sample t-Test for {col}')
                plt.xlabel('Value')
                plt.ylabel('Density')
                plt.legend()
                plt.tight_layout()
                plt.show()

        except Exception as e:
            QMessageBox.critical(main_window, "Error", f"Error performing t-test: {str(e)}")

def two_sample_t_test(main_window):
    """Perform two-sample t-test"""
    main_window.load_data_from_table()
    if main_window.data.empty:
        QMessageBox.warning(main_window, "Warning", "Please load or enter data first")
        return

    try:
        # Get numeric columns only
        numeric_columns = main_window.data.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_columns:
            QMessageBox.warning(main_window, "Warning", "No numeric columns found for analysis")
            return

        # Get first sample column
        col1, ok1 = QInputDialog.getItem(main_window, "Select First Sample", 
            "Choose first sample (numeric measurements):", numeric_columns, 0, False)
        
        if ok1:
            # Get second sample column (excluding the first selected column)
            remaining_columns = [col for col in numeric_columns if col != col1]
            if not remaining_columns:
                QMessageBox.warning(main_window, "Warning", "No other numeric columns available for second sample")
                return
                
            col2, ok2 = QInputDialog.getItem(main_window, "Select Second Sample", 
                "Choose second sample (numeric measurements):", remaining_columns, 0, False)
            
            if ok2 and col1 != col2:
                try:
                    # Convert data to numeric and handle missing values
                    sample1 = pd.to_numeric(main_window.data[col1], errors='coerce').dropna()
                    sample2 = pd.to_numeric(main_window.data[col2], errors='coerce').dropna()
                    
                    if len(sample1) < 2 or len(sample2) < 2:
                        QMessageBox.warning(main_window, "Warning", "Each sample must have at least 2 valid numeric values")
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
                    main_window.sessionWindow.append("\nTwo-Sample t-Test Results")
                    main_window.sessionWindow.append("-" * 40)
                    main_window.sessionWindow.append(f"Sample 1: {col1}")
                    main_window.sessionWindow.append(f"Sample 2: {col2}")
                    main_window.sessionWindow.append(f"\nSample Statistics:")
                    main_window.sessionWindow.append(f"Sample 1: n = {len(sample1)}, Mean = {mean1:.4f}, StDev = {std1:.4f}")
                    main_window.sessionWindow.append(f"Sample 2: n = {len(sample2)}, Mean = {mean2:.4f}, StDev = {std2:.4f}")
                    main_window.sessionWindow.append(f"\nDifference = {mean1 - mean2:.4f}")
                    
                    # Display variance test results
                    main_window.sessionWindow.append(f"\nTest for Equal Variances:")
                    main_window.sessionWindow.append(f"Levene's test statistic = {levene_stat:.4f}")
                    main_window.sessionWindow.append(f"p-value = {levene_p:.4f}")
                    main_window.sessionWindow.append(f"Conclusion: {'Variances are different' if levene_p < 0.05 else 'Cannot conclude variances are different'} at α = 0.05")
                    
                    # Display t-test results for both cases
                    main_window.sessionWindow.append(f"\nTwo-Sample t-Test with Equal Variances:")
                    main_window.sessionWindow.append(f"t-value = {t_stat_equal:.4f}")
                    main_window.sessionWindow.append(f"p-value = {p_value_equal:.4f}")
                    
                    main_window.sessionWindow.append(f"\nTwo-Sample t-Test with Unequal Variances (Welch's test):")
                    main_window.sessionWindow.append(f"t-value = {t_stat_unequal:.4f}")
                    main_window.sessionWindow.append(f"p-value = {p_value_unequal:.4f}")
                    
                    # Add interpretation
                    alpha = 0.05
                    main_window.sessionWindow.append(f"\nInterpretation:")
                    # Use Welch's test (unequal variances) if Levene's test is significant
                    if levene_p < alpha:
                        main_window.sessionWindow.append("Using Welch's t-test (unequal variances):")
                        p_value = p_value_unequal
                    else:
                        main_window.sessionWindow.append("Using pooled t-test (equal variances):")
                        p_value = p_value_equal
                        
                    if p_value < alpha:
                        main_window.sessionWindow.append("Reject the null hypothesis")
                        main_window.sessionWindow.append("There is sufficient evidence to conclude that the means")
                        main_window.sessionWindow.append("are different (at α = 0.05)")
                    else:
                        main_window.sessionWindow.append("Fail to reject the null hypothesis")
                        main_window.sessionWindow.append("There is insufficient evidence to conclude that the means")
                        main_window.sessionWindow.append("are different (at α = 0.05)")
                    
                    # Create visualization
                    plt.figure(figsize=(12, 6))
                    
                    # Box plots
                    plt.subplot(121)
                    plt.boxplot([sample1, sample2], labels=[col1, col2])
                    plt.title('Box Plots of Samples')
                    plt.ylabel('Values')
                    
                    # Histograms
                    plt.subplot(122)
                    plt.hist(sample1, bins='auto', alpha=0.5, label=col1)
                    plt.hist(sample2, bins='auto', alpha=0.5, label=col2)
                    plt.axvline(x=mean1, color='blue', linestyle='--', label=f'{col1} Mean')
                    plt.axvline(x=mean2, color='orange', linestyle='--', label=f'{col2} Mean')
                    plt.title('Histograms of Samples')
                    plt.xlabel('Values')
                    plt.ylabel('Frequency')
                    plt.legend()
                    
                    plt.tight_layout()
                    plt.show()

                except Exception as e:
                    QMessageBox.critical(main_window, "Error", f"Error performing t-test: {str(e)}")
            else:
                QMessageBox.warning(main_window, "Warning", "Please select two different columns")

    except Exception as e:
        QMessageBox.critical(main_window, "Error", f"Error in two-sample t-test: {str(e)}")

def paired_t_test(main_window):
    """Perform paired t-test"""
    main_window.load_data_from_table()
    if main_window.data.empty:
        QMessageBox.warning(main_window, "Warning", "Please load or enter data first")
        return

    try:
        # Get numeric columns only
        numeric_columns = main_window.data.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_columns:
            QMessageBox.warning(main_window, "Warning", "No numeric columns found for analysis")
            return

        # Get first sample column
        col1, ok1 = QInputDialog.getItem(main_window, "Select First Sample", 
            "Choose first sample (numeric measurements):", numeric_columns, 0, False)
        
        if ok1:
            # Get second sample column (excluding the first selected column)
            remaining_columns = [col for col in numeric_columns if col != col1]
            if not remaining_columns:
                QMessageBox.warning(main_window, "Warning", "No other numeric columns available for second sample")
                return
                
            col2, ok2 = QInputDialog.getItem(main_window, "Select Second Sample", 
                "Choose second sample (paired with first):", remaining_columns, 0, False)
            
            if ok2 and col1 != col2:
                try:
                    # Convert data to numeric and handle missing values
                    sample1 = pd.to_numeric(main_window.data[col1], errors='coerce')
                    sample2 = pd.to_numeric(main_window.data[col2], errors='coerce')
                    
                    # Remove rows where either sample has NaN
                    valid_mask = ~(pd.isna(sample1) | pd.isna(sample2))
                    sample1 = sample1[valid_mask]
                    sample2 = sample2[valid_mask]
                    
                    if len(sample1) != len(sample2):
                        QMessageBox.warning(main_window, "Warning", "Samples must have equal length for paired test")
                        return
                        
                    if len(sample1) < 2:
                        QMessageBox.warning(main_window, "Warning", "Each sample must have at least 2 valid numeric values")
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
                    main_window.sessionWindow.append("\nPaired t-Test Results")
                    main_window.sessionWindow.append("-" * 40)
                    main_window.sessionWindow.append(f"Sample 1: {col1}")
                    main_window.sessionWindow.append(f"Sample 2: {col2}")
                    
                    main_window.sessionWindow.append(f"\nSample Statistics:")
                    main_window.sessionWindow.append(f"Sample 1: Mean = {np.mean(sample1):.4f}, StDev = {np.std(sample1, ddof=1):.4f}")
                    main_window.sessionWindow.append(f"Sample 2: Mean = {np.mean(sample2):.4f}, StDev = {np.std(sample2, ddof=1):.4f}")
                    
                    main_window.sessionWindow.append(f"\nPaired Differences (Sample 1 - Sample 2):")
                    main_window.sessionWindow.append(f"n = {len(differences)}")
                    main_window.sessionWindow.append(f"Mean Difference = {mean_diff:.4f}")
                    main_window.sessionWindow.append(f"StDev Difference = {std_diff:.4f}")
                    main_window.sessionWindow.append(f"SE Mean = {se_diff:.4f}")
                    
                    main_window.sessionWindow.append(f"\n95% CI for Mean Difference:")
                    main_window.sessionWindow.append(f"({ci[0]:.4f}, {ci[1]:.4f})")
                    
                    main_window.sessionWindow.append(f"\nTest Statistics:")
                    main_window.sessionWindow.append(f"t-value = {t_stat:.4f}")
                    main_window.sessionWindow.append(f"p-value = {p_value:.4f}")
                    
                    # Add interpretation
                    alpha = 0.05
                    main_window.sessionWindow.append(f"\nInterpretation:")
                    if p_value < alpha:
                        main_window.sessionWindow.append("Reject the null hypothesis")
                        main_window.sessionWindow.append("There is sufficient evidence to conclude that there is")
                        main_window.sessionWindow.append("a difference between the paired samples (at α = 0.05)")
                    else:
                        main_window.sessionWindow.append("Fail to reject the null hypothesis")
                        main_window.sessionWindow.append("There is insufficient evidence to conclude that there is")
                        main_window.sessionWindow.append("a difference between the paired samples (at α = 0.05)")
                    
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
                    QMessageBox.critical(main_window, "Error", f"Error performing paired t-test: {str(e)}")
            else:
                QMessageBox.warning(main_window, "Warning", "Please select two different columns")

    except Exception as e:
        QMessageBox.critical(main_window, "Error", f"Error in paired t-test: {str(e)}")

def regression_analysis(main_window):
    """Open dialog for regression analysis options"""
    dialog = QDialog(main_window)
    dialog.setWindowTitle("Regression Analysis")
    layout = QVBoxLayout()

    # Create buttons for each regression type
    simpleBtn = QPushButton("Simple Linear Regression")
    multipleBtn = QPushButton("Multiple Linear Regression")

    # Connect buttons to their respective functions with dialog closure
    simpleBtn.clicked.connect(lambda: handle_regression_selection(dialog, lambda: simple_linear_regression(main_window)))
    multipleBtn.clicked.connect(lambda: handle_regression_selection(dialog, lambda: multiple_linear_regression(main_window)))

    layout.addWidget(simpleBtn)
    layout.addWidget(multipleBtn)

    dialog.setLayout(layout)
    dialog.exec()

def handle_regression_selection(dialog, regression_func):
    """Handle regression selection and dialog closure"""
    dialog.accept()  # Close the dialog first
    regression_func()  # Then run the selected regression function

def simple_linear_regression(main_window):
    """Perform simple linear regression analysis"""
    main_window.load_data_from_table()
    if main_window.data.empty:
        QMessageBox.warning(main_window, "Warning", "Please load or enter data first")
        return

    try:
        # Get numeric columns only
        numeric_columns = main_window.data.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_columns) < 2:
            QMessageBox.warning(main_window, "Warning", "Need at least two numeric columns for regression")
            return

        # Get response variable
        response_col, ok1 = QInputDialog.getItem(main_window, "Select Response Variable", 
            "Choose response variable (Y):", numeric_columns, 0, False)
        
        if ok1:
            # Get predictor variable
            remaining_columns = [col for col in numeric_columns if col != response_col]
            predictor_col, ok2 = QInputDialog.getItem(main_window, "Select Predictor Variable", 
                "Choose predictor variable (X):", remaining_columns, 0, False)
            
            if ok2:
                # Prepare data
                X = main_window.data[predictor_col].values.reshape(-1, 1)
                y = main_window.data[response_col].values
                
                # Fit the model
                X_with_const = sm.add_constant(X)
                model = sm.OLS(y, X_with_const).fit()
                
                # Calculate predictions for plotting
                X_plot = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
                X_plot_with_const = sm.add_constant(X_plot)
                y_pred = model.predict(X_plot_with_const)
                
                # Display results
                main_window.sessionWindow.append("\nSimple Linear Regression Results")
                main_window.sessionWindow.append("-" * 40)
                main_window.sessionWindow.append(f"Response Variable: {response_col}")
                main_window.sessionWindow.append(f"Predictor Variable: {predictor_col}")
                
                # Model summary
                main_window.sessionWindow.append("\nModel Summary:")
                main_window.sessionWindow.append(f"R-squared = {model.rsquared:.4f}")
                main_window.sessionWindow.append(f"Adjusted R-squared = {model.rsquared_adj:.4f}")
                main_window.sessionWindow.append(f"Standard Error = {np.sqrt(model.mse_resid):.4f}")
                
                # Coefficients
                main_window.sessionWindow.append("\nCoefficients:")
                main_window.sessionWindow.append("Variable      Estimate    Std Error    t-value     p-value")
                main_window.sessionWindow.append("-" * 60)
                main_window.sessionWindow.append(f"{'Intercept':<12}{model.params[0]:10.4f}  {model.bse[0]:10.4f}  {model.tvalues[0]:10.4f}  {model.pvalues[0]:.4e}")
                main_window.sessionWindow.append(f"{predictor_col:<12}{model.params[1]:10.4f}  {model.bse[1]:10.4f}  {model.tvalues[1]:10.4f}  {model.pvalues[1]:.4e}")
                
                # Regression equation
                main_window.sessionWindow.append(f"\nRegression Equation:")
                main_window.sessionWindow.append(f"{response_col} = {model.params[0]:.4f} + {model.params[1]:.4f}×{predictor_col}")
                
                # Analysis of Variance
                main_window.sessionWindow.append("\nAnalysis of Variance:")
                main_window.sessionWindow.append("Source      DF          SS          MS           F         P")
                main_window.sessionWindow.append("-" * 70)
                main_window.sessionWindow.append(f"{'Regression':<10}  {1:2}  {model.ess:11.4f}  {model.ess:11.4f}  {model.fvalue:11.4f}  {model.f_pvalue:.4e}")
                main_window.sessionWindow.append(f"{'Residual':<10}  {model.df_resid:2}  {model.ssr:11.4f}  {model.ssr/model.df_resid:11.4f}")
                main_window.sessionWindow.append(f"{'Total':<10}  {model.df_model + model.df_resid:2}  {model.ess + model.ssr:11.4f}")
                
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
        QMessageBox.critical(main_window, "Error", f"Error in simple linear regression: {str(e)}")
        traceback.print_exc()

def multiple_linear_regression(main_window):
    """Perform multiple linear regression analysis"""
    main_window.load_data_from_table()
    if main_window.data.empty:
        QMessageBox.warning(main_window, "Warning", "Please load or enter data first")
        return

    try:
        # Get numeric columns only
        numeric_columns = main_window.data.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_columns) < 3:
            QMessageBox.warning(main_window, "Warning", "Need at least three numeric columns for multiple regression")
            return

        # Get response variable
        response_col, ok1 = QInputDialog.getItem(main_window, "Select Response Variable", 
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
                    QMessageBox.warning(main_window, "Warning", "Please select at least one predictor variable")
                    return
                
                # Prepare data
                X = main_window.data[predictor_cols]
                y = main_window.data[response_col]
                
                # Fit the model
                X_with_const = sm.add_constant(X)
                model = sm.OLS(y, X_with_const).fit()
                
                # Display results
                main_window.sessionWindow.append("\nMultiple Linear Regression Results")
                main_window.sessionWindow.append("-" * 40)
                main_window.sessionWindow.append(f"Response Variable: {response_col}")
                main_window.sessionWindow.append(f"Predictor Variables: {', '.join(predictor_cols)}")
                
                # Model summary
                main_window.sessionWindow.append("\nModel Summary:")
                main_window.sessionWindow.append(f"R-squared = {model.rsquared:.4f}")
                main_window.sessionWindow.append(f"Adjusted R-squared = {model.rsquared_adj:.4f}")
                main_window.sessionWindow.append(f"Standard Error = {np.sqrt(model.mse_resid):.4f}")
                
                # Coefficients
                main_window.sessionWindow.append("\nCoefficients:")
                main_window.sessionWindow.append("Variable      Estimate    Std Error    t-value     p-value")
                main_window.sessionWindow.append("-" * 60)
                main_window.sessionWindow.append(f"{'Intercept':<12}{model.params[0]:10.4f}  {model.bse[0]:10.4f}  {model.tvalues[0]:10.4f}  {model.pvalues[0]:.4e}")
                for i, col in enumerate(predictor_cols, 1):
                    main_window.sessionWindow.append(f"{col:<12}{model.params[i]:10.4f}  {model.bse[i]:10.4f}  {model.tvalues[i]:10.4f}  {model.pvalues[i]:.4e}")
                
                # Regression equation
                main_window.sessionWindow.append(f"\nRegression Equation:")
                equation = f"{response_col} = {model.params[0]:.4f}"
                for i, col in enumerate(predictor_cols, 1):
                    equation += f" + {model.params[i]:.4f}×{col}"
                main_window.sessionWindow.append(equation)
                
                # Analysis of Variance
                main_window.sessionWindow.append("\nAnalysis of Variance:")
                main_window.sessionWindow.append("Source      DF          SS          MS           F         P")
                main_window.sessionWindow.append("-" * 70)
                main_window.sessionWindow.append(f"{'Regression':<10}  {len(predictor_cols):2}  {model.ess:11.4f}  {model.ess/model.df_model:11.4f}  {model.fvalue:11.4f}  {model.f_pvalue:.4e}")
                main_window.sessionWindow.append(f"{'Residual':<10}  {model.df_resid:2}  {model.ssr:11.4f}  {model.ssr/model.df_resid:11.4f}")
                main_window.sessionWindow.append(f"{'Total':<10}  {model.df_model + model.df_resid:2}  {model.ess + model.ssr:11.4f}")
                
                # VIF values if more than one predictor
                if len(predictor_cols) > 1:
                    main_window.sessionWindow.append("\nVariance Inflation Factors:")
                    main_window.sessionWindow.append("Variable      VIF")
                    main_window.sessionWindow.append("-" * 20)
                    # Calculate VIF for each predictor
                    for i, col in enumerate(predictor_cols):
                        other_cols = [c for c in predictor_cols if c != col]
                        X_others = main_window.data[other_cols]
                        X_target = main_window.data[col]
                        r_squared = sm.OLS(X_target, sm.add_constant(X_others)).fit().rsquared
                        vif = 1 / (1 - r_squared) if r_squared != 1 else float('inf')
                        main_window.sessionWindow.append(f"{col:<12}{vif:8.4f}")
                
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
        QMessageBox.critical(main_window, "Error", f"Error in multiple linear regression: {str(e)}")
        traceback.print_exc()

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