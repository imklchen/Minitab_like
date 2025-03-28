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
