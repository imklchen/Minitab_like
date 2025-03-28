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
