            # Get measurement and subgroup columns
            measurement_col = self.selectColumnDialog("Select Measurement Column")
            if measurement_col is None:
                return
            subgroup_col = self.selectColumnDialog("Select Subgroup Column")
            if subgroup_col is None:
                return
            
            # Get specification limits and target
            lsl, ok = QInputDialog.getDouble(self, "Lower Spec Limit", "Enter Lower Spec Limit (LSL):", 10.0)
            if not ok:
                return
            usl, ok = QInputDialog.getDouble(self, "Upper Spec Limit", "Enter Upper Spec Limit (USL):", 10.8)
            if not ok:
                return
            target, ok = QInputDialog.getDouble(self, "Target", "Enter Target Value:", 10.4)
            if not ok:
                return
            
            # Calculate basic statistics
            data = self.data[measurement_col].values
            subgroups = self.data[subgroup_col].values
            n = len(data)
            mean = np.mean(data)
            
            # Calculate StDev (Within) using R̄/d2 method
            unique_subgroups = np.unique(subgroups)
            ranges = []
            for subgroup in unique_subgroups:
                subgroup_data = data[subgroups == subgroup]
                ranges.append(np.max(subgroup_data) - np.min(subgroup_data))
            r_bar = np.mean(ranges)
            d2 = 2.059  # d2 value for n=4
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
            
            # Display report in session window
            self.sessionWindow.setText(report)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error in process capability analysis: {str(e)}")

    def paretoChart(self):
        """Create a Pareto chart"""
        try:
            # Load data from the table
            self.loadDataFromTable()
            
            if self.data.empty:
                QMessageBox.warning(self, "Warning", "No data available for analysis")
                return
                
            # Create dialog
            dialog = QDialog(self)
            dialog.setWindowTitle("Pareto Chart")
            layout = QVBoxLayout()
            
            # Column selection
            col_group = QGroupBox("Column Selection")
            col_layout = QFormLayout()
            
            cat_combo = QComboBox()
            cat_combo.addItems(self.data.columns)
            col_layout.addRow("Categories:", cat_combo)
            
            freq_combo = QComboBox()
            freq_combo.addItems(self.data.columns)
            col_layout.addRow("Frequencies:", freq_combo)
            
            col_group.setLayout(col_layout)
            layout.addWidget(col_group)
            
            # Options
            opt_group = QGroupBox("Options")
            opt_layout = QFormLayout()
            
            show_cumulative = QCheckBox("Show Cumulative Line")
            show_cumulative.setChecked(True)
            opt_layout.addRow(show_cumulative)
            
            bars_input = QLineEdit("all")
            opt_layout.addRow("Bars to Display:", bars_input)
            
            opt_group.setLayout(opt_layout)
            layout.addWidget(opt_group)
            
            # Buttons
            buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
            buttons.accepted.connect(dialog.accept)
            buttons.rejected.connect(dialog.reject)
            layout.addWidget(buttons)
            
            dialog.setLayout(layout)
            
            if dialog.exec() == QDialog.DialogCode.Accepted:
                # Get data
                cat_col = cat_combo.currentText()
                freq_col = freq_combo.currentText()
                
                df = pd.DataFrame({
                    'Category': self.data[cat_col],
                    'Count': pd.to_numeric(self.data[freq_col], errors='coerce')
                }).dropna()
                
                df = df.sort_values('Count', ascending=False)
                total = df['Count'].sum()
                df['Percent'] = (df['Count'] / total * 100).round(1)
                df['Cumulative'] = df['Percent'].cumsum().round(1)
                
                # Display results
                self.sessionWindow.setText("Pareto Analysis Results")
                self.sessionWindow.append("----------------------")
                self.sessionWindow.append(f"Total Defects: {int(total)}\n")
                
                # Format and display table
                self.sessionWindow.append(f"{'Category':<10} {'Count':>7} {'Percent':>9} {'Cumulative':>12}")
                for _, row in df.iterrows():
                    self.sessionWindow.append(
                        f"{row['Category']:<10} {int(row['Count']):>7} {row['Percent']:>8.1f}% {row['Cumulative']:>11.1f}%"
                    )
                
                # Create visualization
                fig, ax1 = plt.subplots(figsize=(10, 6))
                x = range(len(df))
                ax1.bar(x, df['Count'])
                ax1.set_xticks(x)
                ax1.set_xticklabels(df['Category'], rotation=45, ha='right')
                ax1.set_ylabel('Count')
                
                if show_cumulative.isChecked():
                    ax2 = ax1.twinx()
                    ax2.plot(x, df['Cumulative'], 'r-o')
                    ax2.set_ylabel('Cumulative Percentage')
                    ax2.set_ylim([0, 105])
                
                plt.title('Pareto Chart')
                plt.tight_layout()
                plt.show()
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error creating Pareto chart: {str(e)}")

    def dpmoCalculator(self):
        """Calculate DPMO and Sigma Level"""
        defects, ok1 = QInputDialog.getInt(self, "DPMO Calculator", "Enter number of defects:", 0, 0)
        if not ok1:
            return
            
        opportunities, ok2 = QInputDialog.getInt(self, "DPMO Calculator", "Enter opportunities per unit:", 1, 1)
        if not ok2:
            return
            
        units, ok3 = QInputDialog.getInt(self, "DPMO Calculator", "Enter number of units:", 1, 1)
        if not ok3:
            return
            
        dpmo = calculate_dpmo(defects, opportunities, units)
        sigma = dpmo_to_sigma(dpmo)
        
        report = f"""DPMO Analysis Results

Number of Defects: {defects}
Opportunities per Unit: {opportunities}
Number of Units: {units}

DPMO: {dpmo:.2f}
Sigma Level: {sigma:.2f}
"""
        self.sessionWindow.setText(report)

    def hypothesisTesting(self):
        """Open dialog for hypothesis testing options"""
        dialog = QDialog(self)
        dialog.setWindowTitle("Hypothesis Testing")
        layout = QVBoxLayout()

        # Create buttons for each test type
        oneSampleBtn = QPushButton("One-Sample t-Test")
        twoSampleBtn = QPushButton("Two-Sample t-Test")
        pairedBtn = QPushButton("Paired t-Test")

        # Connect buttons to their respective functions with dialog closure
        oneSampleBtn.clicked.connect(lambda: self.handle_hypothesis_selection(dialog, self.oneSampleTTest))
        twoSampleBtn.clicked.connect(lambda: self.handle_hypothesis_selection(dialog, self.twoSampleTTest))
        pairedBtn.clicked.connect(lambda: self.handle_hypothesis_selection(dialog, self.pairedTTest))

        layout.addWidget(oneSampleBtn)
        layout.addWidget(twoSampleBtn)
        layout.addWidget(pairedBtn)

        dialog.setLayout(layout)
        dialog.exec()

    def handle_hypothesis_selection(self, dialog, test_func):
        """Handle hypothesis test selection and dialog closure"""
        dialog.accept()  # Close the dialog first
        test_func()     # Then run the selected test function

    def oneSampleTTest(self):
        """Perform one-sample t-test"""
        self.loadDataFromTable()
        if self.data.empty:
            QMessageBox.warning(self, "Warning", "Please load or enter data first")
            return

        # Get column selection
        col, ok = QInputDialog.getItem(self, "Select Column", 
            "Choose column for analysis:", self.data.columns.tolist(), 0, False)
        
        if ok and col:
            try:
                # Get hypothesized mean
                hyp_mean, ok = QInputDialog.getDouble(self, "Hypothesized Mean", 
                    "Enter hypothesized mean value:", 0, -1000000, 1000000, 4)
                
                if ok:
                    # Convert data to numeric
                    data = pd.to_numeric(self.data[col], errors='coerce').dropna()
                    
                    # Perform t-test
                    t_stat, p_value = stats.ttest_1samp(data, hyp_mean)
                    
                    # Calculate additional statistics
                    mean = np.mean(data)
                    std_dev = np.std(data, ddof=1)
                    se = std_dev / np.sqrt(len(data))
                    ci = stats.t.interval(0.95, len(data)-1, loc=mean, scale=se)
                    
                    # Display results
                    self.sessionWindow.append("\nOne-Sample t-Test Results")
                    self.sessionWindow.append("-" * 40)
                    self.sessionWindow.append(f"Variable: {col}")
                    self.sessionWindow.append(f"Hypothesized mean = {hyp_mean}")
                    self.sessionWindow.append(f"\nSample Statistics:")
                    self.sessionWindow.append(f"Sample Size = {len(data)}")
                    self.sessionWindow.append(f"Sample Mean = {mean:.4f}")
                    self.sessionWindow.append(f"Sample StDev = {std_dev:.4f}")
                    self.sessionWindow.append(f"SE Mean = {se:.4f}")
                    self.sessionWindow.append(f"\n95% Confidence Interval:")
                    self.sessionWindow.append(f"({ci[0]:.4f}, {ci[1]:.4f})")
                    self.sessionWindow.append(f"\nTest Statistics:")
                    self.sessionWindow.append(f"t-value = {t_stat:.4f}")
                    self.sessionWindow.append(f"p-value = {p_value:.4f}")
                    
                    # Add interpretation
                    alpha = 0.05
                    self.sessionWindow.append(f"\nInterpretation:")
                    if p_value < alpha:
                        self.sessionWindow.append("Reject the null hypothesis")
                        self.sessionWindow.append("There is sufficient evidence to conclude that the population mean")
                        self.sessionWindow.append(f"is different from {hyp_mean} (at α = 0.05)")
                    else:
                        self.sessionWindow.append("Fail to reject the null hypothesis")
                        self.sessionWindow.append("There is insufficient evidence to conclude that the population mean")
                        self.sessionWindow.append(f"is different from {hyp_mean} (at α = 0.05)")

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error performing t-test: {str(e)}")

    def twoSampleTTest(self):
        """Perform two-sample t-test"""
        self.loadDataFromTable()
        if self.data.empty:
            QMessageBox.warning(self, "Warning", "Please load or enter data first")
            return

        try:
            # Get numeric columns only
            numeric_columns = self.data.select_dtypes(include=[np.number]).columns.tolist()
            if not numeric_columns:
                QMessageBox.warning(self, "Warning", "No numeric columns found for analysis")
                return

            # Get first sample column
            col1, ok1 = QInputDialog.getItem(self, "Select First Sample", 
                "Choose first sample (numeric measurements):", numeric_columns, 0, False)
            
            if ok1:
                # Get second sample column (excluding the first selected column)
                remaining_columns = [col for col in numeric_columns if col != col1]
                if not remaining_columns:
                    QMessageBox.warning(self, "Warning", "No other numeric columns available for second sample")
                    return
                    
                col2, ok2 = QInputDialog.getItem(self, "Select Second Sample", 
                    "Choose second sample (numeric measurements):", remaining_columns, 0, False)
                
                if ok2 and col1 != col2:
                    try:
                        # Convert data to numeric and handle missing values
                        sample1 = pd.to_numeric(self.data[col1], errors='coerce').dropna()
                        sample2 = pd.to_numeric(self.data[col2], errors='coerce').dropna()
                        
                        if len(sample1) < 2 or len(sample2) < 2:
                            QMessageBox.warning(self, "Warning", "Each sample must have at least 2 valid numeric values")
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
                        self.sessionWindow.append("\nTwo-Sample t-Test Results")
                        self.sessionWindow.append("-" * 40)
                        self.sessionWindow.append(f"Sample 1: {col1}")
                        self.sessionWindow.append(f"Sample 2: {col2}")
                        self.sessionWindow.append(f"\nSample Statistics:")
                        self.sessionWindow.append(f"Sample 1: n = {len(sample1)}, Mean = {mean1:.4f}, StDev = {std1:.4f}")
                        self.sessionWindow.append(f"Sample 2: n = {len(sample2)}, Mean = {mean2:.4f}, StDev = {std2:.4f}")
                        self.sessionWindow.append(f"\nDifference = {mean1 - mean2:.4f}")
                        
                        # Display variance test results
                        self.sessionWindow.append(f"\nTest for Equal Variances:")
                        self.sessionWindow.append(f"Levene's test statistic = {levene_stat:.4f}")
                        self.sessionWindow.append(f"p-value = {levene_p:.4f}")
                        self.sessionWindow.append(f"Conclusion: {'Variances are different' if levene_p < 0.05 else 'Cannot conclude variances are different'} at α = 0.05")
                        
                        # Display t-test results for both cases
                        self.sessionWindow.append(f"\nTwo-Sample t-Test with Equal Variances:")
                        self.sessionWindow.append(f"t-value = {t_stat_equal:.4f}")
                        self.sessionWindow.append(f"p-value = {p_value_equal:.4f}")
                        
                        self.sessionWindow.append(f"\nTwo-Sample t-Test with Unequal Variances (Welch's test):")
                        self.sessionWindow.append(f"t-value = {t_stat_unequal:.4f}")
                        self.sessionWindow.append(f"p-value = {p_value_unequal:.4f}")
                        
                        # Add interpretation
                        alpha = 0.05
                        self.sessionWindow.append(f"\nInterpretation:")
                        # Use Welch's test (unequal variances) if Levene's test is significant
                        if levene_p < alpha:
                            self.sessionWindow.append("Using Welch's t-test (unequal variances):")
                            p_value = p_value_unequal
                        else:
                            self.sessionWindow.append("Using pooled t-test (equal variances):")
                            p_value = p_value_equal
                            
                        if p_value < alpha:
                            self.sessionWindow.append("Reject the null hypothesis")
                            self.sessionWindow.append("There is sufficient evidence to conclude that the means")
                            self.sessionWindow.append("are different (at α = 0.05)")
                        else:
                            self.sessionWindow.append("Fail to reject the null hypothesis")
                            self.sessionWindow.append("There is insufficient evidence to conclude that the means")
                            self.sessionWindow.append("are different (at α = 0.05)")

                    except Exception as e:
                        QMessageBox.critical(self, "Error", f"Error performing t-test: {str(e)}")
                else:
                    QMessageBox.warning(self, "Warning", "Please select two different columns")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error in two-sample t-test: {str(e)}")

    def pairedTTest(self):
        """Perform paired t-test"""
        self.loadDataFromTable()
        if self.data.empty:
            QMessageBox.warning(self, "Warning", "Please load or enter data first")
            return

        try:
            # Get numeric columns only
            numeric_columns = self.data.select_dtypes(include=[np.number]).columns.tolist()
            if not numeric_columns:
                QMessageBox.warning(self, "Warning", "No numeric columns found for analysis")
                return

            # Get first sample column
            col1, ok1 = QInputDialog.getItem(self, "Select First Sample", 
                "Choose first sample (numeric measurements):", numeric_columns, 0, False)
            
            if ok1:
                # Get second sample column (excluding the first selected column)
                remaining_columns = [col for col in numeric_columns if col != col1]
                if not remaining_columns:
                    QMessageBox.warning(self, "Warning", "No other numeric columns available for second sample")
                    return
                    
                col2, ok2 = QInputDialog.getItem(self, "Select Second Sample", 
                    "Choose second sample (paired with first):", remaining_columns, 0, False)
                
                if ok2 and col1 != col2:
                    try:
                        # Convert data to numeric and handle missing values
                        sample1 = pd.to_numeric(self.data[col1], errors='coerce')
                        sample2 = pd.to_numeric(self.data[col2], errors='coerce')
                        
                        # Remove rows where either sample has NaN
                        valid_mask = ~(pd.isna(sample1) | pd.isna(sample2))
                        sample1 = sample1[valid_mask]
                        sample2 = sample2[valid_mask]
                        
                        if len(sample1) != len(sample2):
                            QMessageBox.warning(self, "Warning", "Samples must have equal length for paired test")
                            return
                            
                        if len(sample1) < 2:
                            QMessageBox.warning(self, "Warning", "Each sample must have at least 2 valid numeric values")
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
                        self.sessionWindow.append("\nPaired t-Test Results")
                        self.sessionWindow.append("-" * 40)
                        self.sessionWindow.append(f"Sample 1: {col1}")
                        self.sessionWindow.append(f"Sample 2: {col2}")
                        
                        self.sessionWindow.append(f"\nSample Statistics:")
                        self.sessionWindow.append(f"Sample 1: Mean = {np.mean(sample1):.4f}, StDev = {np.std(sample1, ddof=1):.4f}")
                        self.sessionWindow.append(f"Sample 2: Mean = {np.mean(sample2):.4f}, StDev = {np.std(sample2, ddof=1):.4f}")
                        
                        self.sessionWindow.append(f"\nPaired Differences (Sample 1 - Sample 2):")
                        self.sessionWindow.append(f"n = {len(differences)}")
                        self.sessionWindow.append(f"Mean Difference = {mean_diff:.4f}")
                        self.sessionWindow.append(f"StDev Difference = {std_diff:.4f}")
                        self.sessionWindow.append(f"SE Mean = {se_diff:.4f}")
                        
                        self.sessionWindow.append(f"\n95% CI for Mean Difference:")
                        self.sessionWindow.append(f"({ci[0]:.4f}, {ci[1]:.4f})")
                        
                        self.sessionWindow.append(f"\nTest Statistics:")
                        self.sessionWindow.append(f"t-value = {t_stat:.4f}")
                        self.sessionWindow.append(f"p-value = {p_value:.4f}")
                        
                        # Add interpretation
                        alpha = 0.05
                        self.sessionWindow.append(f"\nInterpretation:")
                        if p_value < alpha:
                            self.sessionWindow.append("Reject the null hypothesis")
                            self.sessionWindow.append("There is sufficient evidence to conclude that there is")
                            self.sessionWindow.append("a difference between the paired samples (at α = 0.05)")
                        else:
                            self.sessionWindow.append("Fail to reject the null hypothesis")
                            self.sessionWindow.append("There is insufficient evidence to conclude that there is")
                            self.sessionWindow.append("a difference between the paired samples (at α = 0.05)")
                        
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
                        QMessageBox.critical(self, "Error", f"Error performing paired t-test: {str(e)}")
                else:
                    QMessageBox.warning(self, "Warning", "Please select two different columns")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error in paired t-test: {str(e)}")

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

    def handle_anova_selection(self, dialog, anova_func):
        """Handle ANOVA selection and dialog closure"""
        dialog.accept()  # Close the dialog first
        anova_func()     # Then run the selected ANOVA function

    def one_way_anova(self):
        """Perform One-Way ANOVA analysis"""
        self.loadDataFromTable()
        if self.data.empty:
            QMessageBox.warning(self, "Warning", "Please load or enter data first")
            return

        try:
            # Get response variable (numeric column)
            numeric_columns = self.data.select_dtypes(include=[np.number]).columns.tolist()
            if not numeric_columns:
                QMessageBox.warning(self, "Warning", "No numeric columns found for response variable")
                return
                
            response_col, ok1 = QInputDialog.getItem(self, "Select Response Variable", 
                "Choose response variable (numeric measurements):", numeric_columns, 0, False)
            
            if ok1:
                # Get factor variable (categorical column)
                categorical_columns = [col for col in self.data.columns if col != response_col 
                                    and len(self.data[col].unique()) > 1 
                                    and len(self.data[col].unique()) < len(self.data[col])]
                
                if not categorical_columns:
                    QMessageBox.warning(self, "Warning", "No suitable categorical columns found for factor")
                    return
                    
                factor_col, ok2 = QInputDialog.getItem(self, "Select Factor", 
                    "Choose factor variable (groups/categories):", categorical_columns, 0, False)
                
                if ok2 and response_col != factor_col:
                    # Convert response to numeric and remove missing values
                    response_data = pd.to_numeric(self.data[response_col], errors='coerce')
                    factor_data = self.data[factor_col]
                    
                    # Remove rows with missing values
                    valid_mask = ~pd.isna(response_data)
                    response_data = response_data[valid_mask]
                    factor_data = factor_data[valid_mask]
                    
                    # Create DataFrame for statsmodels
                    anova_data = pd.DataFrame({
                        'Response': response_data,
                        'Factor': factor_data
                    })
                    
                    # Fit the model
                    model = ols('Response ~ C(Factor)', data=anova_data).fit()
                    anova_table = sm.stats.anova_lm(model, typ=1)
                    
                    # Calculate additional statistics
                    groups = anova_data.groupby('Factor')['Response']
                    group_means = groups.mean()
                    group_sds = groups.std()
                    group_ns = groups.count()
                    
                    # Display results
                    self.sessionWindow.append("\nOne-Way ANOVA Results")
                    self.sessionWindow.append("-" * 50)
                    self.sessionWindow.append(f"Response Variable: {response_col}")
                    self.sessionWindow.append(f"Factor: {factor_col}")
                    
                    # Display descriptive statistics by group
                    self.sessionWindow.append("\nDescriptive Statistics:")
                    self.sessionWindow.append("-" * 30)
                    for group in group_means.index:
                        self.sessionWindow.append(f"\nGroup: {group}")
                        self.sessionWindow.append(f"  N = {group_ns[group]}")
                        self.sessionWindow.append(f"  Mean = {group_means[group]:.4f}")
                        self.sessionWindow.append(f"  StDev = {group_sds[group]:.4f}")
                    
                    # Display ANOVA table with clear labels
                    self.sessionWindow.append("\nANOVA Table:")
                    self.sessionWindow.append("-" * 30)
                    self.sessionWindow.append(anova_table.to_string())
                    
                    # Display test statistics explicitly
                    f_stat = anova_table.loc['C(Factor)', 'F']
                    p_value = anova_table.loc['C(Factor)', 'PR(>F)']
                    r_squared = model.rsquared
                    
                    self.sessionWindow.append("\nTest Statistics:")
                    self.sessionWindow.append("-" * 30)
                    self.sessionWindow.append(f"F-statistic = {f_stat:.4f}")
                    self.sessionWindow.append(f"P-value = {p_value:.4f}")
                    self.sessionWindow.append(f"R-squared = {r_squared:.4f}")
                    
                    # Add interpretation
                    alpha = 0.05
                    self.sessionWindow.append(f"\nInterpretation:")
                    if p_value < alpha:
                        self.sessionWindow.append("Reject the null hypothesis")
                        self.sessionWindow.append("There is sufficient evidence to conclude that there are")
                        self.sessionWindow.append("significant differences between group means (at α = 0.05)")
                    else:
                        self.sessionWindow.append("Fail to reject the null hypothesis")
                        self.sessionWindow.append("There is insufficient evidence to conclude that there are")
                        self.sessionWindow.append("significant differences between group means (at α = 0.05)")
                    
                    # Create visualization
                    plt.figure(figsize=(10, 6))
                    sns.boxplot(x='Factor', y='Response', data=anova_data)
                    plt.title('One-Way ANOVA: Box Plot by Group')
                    plt.xlabel(factor_col)
                    plt.ylabel(response_col)
                    plt.show()

                else:
                    QMessageBox.warning(self, "Warning", "Please select different columns for response and factor")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error performing ANOVA: {str(e)}")

    def two_way_anova(self):
        """Perform Two-Way ANOVA analysis"""
        self.loadDataFromTable()
        if self.data.empty:
            QMessageBox.warning(self, "Warning", "Please load or enter data first")
            return

        try:
            # Get response variable (numeric column)
            numeric_columns = self.data.select_dtypes(include=[np.number]).columns.tolist()
            if not numeric_columns:
                QMessageBox.warning(self, "Warning", "No numeric columns found for response variable")
                return
                
            response_col, ok1 = QInputDialog.getItem(self, "Select Response Variable", 
                "Choose response variable (numeric measurements):", numeric_columns, 0, False)
            
            if ok1:
                # Get first factor variable (categorical column)
                categorical_columns = [col for col in self.data.columns if col != response_col 
                                    and len(self.data[col].unique()) > 1 
                                    and len(self.data[col].unique()) < len(self.data[col])]
                
                if not categorical_columns:
                    QMessageBox.warning(self, "Warning", "No suitable categorical columns found for factors")
                    return
                    
                factor1_col, ok2 = QInputDialog.getItem(self, "Select First Factor", 
                    "Choose first factor variable (groups/categories):", categorical_columns, 0, False)
                
                if ok2:
                    # Get second factor variable (categorical column)
                    remaining_categorical = [col for col in categorical_columns if col != factor1_col]
                    
                    if not remaining_categorical:
                        QMessageBox.warning(self, "Warning", "No suitable categorical columns found for second factor")
                        return
                        
                    factor2_col, ok3 = QInputDialog.getItem(self, "Select Second Factor", 
                        "Choose second factor variable (groups/categories):", remaining_categorical, 0, False)
                    
                    if ok3 and factor2_col != factor1_col:
                        # Convert response to numeric and remove missing values
                        response_data = pd.to_numeric(self.data[response_col], errors='coerce')
                        factor1_data = self.data[factor1_col]
                        factor2_data = self.data[factor2_col]
                        
                        # Remove rows with missing values
                        valid_mask = ~pd.isna(response_data)
                        response_data = response_data[valid_mask]
                        factor1_data = factor1_data[valid_mask]
                        factor2_data = factor2_data[valid_mask]
                        
                        # Create DataFrame for statsmodels
                        anova_data = pd.DataFrame({
                            'Response': response_data,
                            'Factor1': factor1_data,
                            'Factor2': factor2_data
                        })
                        
                        # Fit the model with interaction
                        model = ols('Response ~ C(Factor1) + C(Factor2) + C(Factor1):C(Factor2)', 
                                  data=anova_data).fit()
                        anova_table = sm.stats.anova_lm(model, typ=2)
                        
                        # Calculate descriptive statistics
                        means = anova_data.groupby(['Factor1', 'Factor2'])['Response'].agg(['mean', 'std', 'count'])
                        
                        # Display results
                        self.sessionWindow.append("\nTwo-Way ANOVA Results")
                        self.sessionWindow.append("-" * 50)
                        self.sessionWindow.append(f"Response Variable: {response_col}")
                        self.sessionWindow.append(f"Factor 1: {factor1_col}")
                        self.sessionWindow.append(f"Factor 2: {factor2_col}")
                        
                        # Display descriptive statistics
                        self.sessionWindow.append("\nDescriptive Statistics:")
                        self.sessionWindow.append("-" * 30)
                        self.sessionWindow.append(means.to_string())
                        
                        # Display ANOVA table with clear labels
                        self.sessionWindow.append("\nANOVA Table:")
                        self.sessionWindow.append("-" * 30)
                        self.sessionWindow.append(anova_table.to_string())
                        
                        # Display test statistics explicitly
                        self.sessionWindow.append("\nTest Statistics:")
                        self.sessionWindow.append("-" * 30)
                        
                        # Factor 1 effects
                        f_stat1 = anova_table.loc['C(Factor1)', 'F']
                        p_value1 = anova_table.loc['C(Factor1)', 'PR(>F)']
                        self.sessionWindow.append(f"\nFactor 1 ({factor1_col}):")
                        self.sessionWindow.append(f"F-statistic = {f_stat1:.4f}")
                        self.sessionWindow.append(f"P-value = {p_value1:.4f}")
                        
                        # Factor 2 effects
                        f_stat2 = anova_table.loc['C(Factor2)', 'F']
                        p_value2 = anova_table.loc['C(Factor2)', 'PR(>F)']
                        self.sessionWindow.append(f"\nFactor 2 ({factor2_col}):")
                        self.sessionWindow.append(f"F-statistic = {f_stat2:.4f}")
                        self.sessionWindow.append(f"P-value = {p_value2:.4f}")
                        
                        # Interaction effects
                        f_stat_int = anova_table.loc['C(Factor1):C(Factor2)', 'F']
                        p_value_int = anova_table.loc['C(Factor1):C(Factor2)', 'PR(>F)']
                        self.sessionWindow.append(f"\nInteraction ({factor1_col}*{factor2_col}):")
                        self.sessionWindow.append(f"F-statistic = {f_stat_int:.4f}")
                        self.sessionWindow.append(f"P-value = {p_value_int:.4f}")
                        
                        # Model statistics
                        r_squared = model.rsquared
                        self.sessionWindow.append(f"\nModel Fit:")
                        self.sessionWindow.append(f"R-squared = {r_squared:.4f}")
                        
                        # Add interpretation
                        alpha = 0.05
                        self.sessionWindow.append(f"\nInterpretation (α = 0.05):")
                        
                        # Factor 1 effect
                        self.sessionWindow.append(f"\nFactor 1 ({factor1_col}):")
                        if p_value1 < alpha:
                            self.sessionWindow.append("Significant main effect")
                        else:
                            self.sessionWindow.append("No significant main effect")
                        
                        # Factor 2 effect
                        self.sessionWindow.append(f"\nFactor 2 ({factor2_col}):")
                        if p_value2 < alpha:
                            self.sessionWindow.append("Significant main effect")
                        else:
                            self.sessionWindow.append("No significant main effect")
                        
                        # Interaction effect
                        self.sessionWindow.append(f"\nInteraction Effect:")
                        if p_value_int < alpha:
                            self.sessionWindow.append("Significant interaction between factors")
                        else:
                            self.sessionWindow.append("No significant interaction between factors")
                        
                        # Create visualizations
                        # Create a single figure with all plots
                        fig = plt.figure(figsize=(15, 5))
                        
                        # Create a grid layout
                        gs = plt.GridSpec(1, 3, figure=fig)
                        
                        # Factor 1 main effect
                        ax1 = fig.add_subplot(gs[0, 0])
                        sns.boxplot(x='Factor1', y='Response', data=anova_data, ax=ax1)
                        ax1.set_title(f'Main Effect of {factor1_col}')
                        ax1.set_xlabel(factor1_col)
                        ax1.set_ylabel(response_col)
                        
                        # Factor 2 main effect
                        ax2 = fig.add_subplot(gs[0, 1])
                        sns.boxplot(x='Factor2', y='Response', data=anova_data, ax=ax2)
                        ax2.set_title(f'Main Effect of {factor2_col}')
                        ax2.set_xlabel(factor2_col)
                        ax2.set_ylabel(response_col)
                        
                        # Interaction plot
                        ax3 = fig.add_subplot(gs[0, 2])
                        interaction_data = anova_data.groupby(['Factor1', 'Factor2'])['Response'].mean().unstack()
                        interaction_data.plot(marker='o', ax=ax3)
                        ax3.set_title('Interaction Plot')
                        ax3.set_xlabel(factor1_col)
                        ax3.set_ylabel(f'Mean {response_col}')
                        ax3.legend(title=factor2_col)
                        ax3.grid(True)
                        
                        plt.tight_layout()
                        plt.show()

                    else:
                        QMessageBox.warning(self, "Warning", "Please select different columns for the two factors")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error performing Two-Way ANOVA: {str(e)}")

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

    def handle_regression_selection(self, dialog, regression_func):
        """Handle regression selection and dialog closure"""
        dialog.accept()  # Close the dialog first
        regression_func()  # Then run the selected regression function

    def simple_linear_regression(self):
        """Perform simple linear regression analysis"""
        self.loadDataFromTable()
        if self.data.empty:
            QMessageBox.warning(self, "Warning", "Please load or enter data first")
            return

        try:
            # Get numeric columns only
            numeric_columns = self.data.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_columns) < 2:
                QMessageBox.warning(self, "Warning", "Need at least two numeric columns for regression")
                return

            # Get response variable
            response_col, ok1 = QInputDialog.getItem(self, "Select Response Variable", 
                "Choose response variable (Y):", numeric_columns, 0, False)
            
            if ok1:
                # Get predictor variable
                remaining_columns = [col for col in numeric_columns if col != response_col]
                predictor_col, ok2 = QInputDialog.getItem(self, "Select Predictor Variable", 
                    "Choose predictor variable (X):", remaining_columns, 0, False)
                
                if ok2:
                    # Prepare data
                    X = self.data[predictor_col].values.reshape(-1, 1)
                    y = self.data[response_col].values
                    
                    # Fit the model
                    X_with_const = sm.add_constant(X)
                    model = sm.OLS(y, X_with_const).fit()
                    
                    # Calculate predictions for plotting
                    X_plot = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
                    X_plot_with_const = sm.add_constant(X_plot)
                    y_pred = model.predict(X_plot_with_const)
                    
                    # Display results
                    self.sessionWindow.append("\nSimple Linear Regression Results")
                    self.sessionWindow.append("-" * 40)
                    self.sessionWindow.append(f"Response Variable: {response_col}")
                    self.sessionWindow.append(f"Predictor Variable: {predictor_col}")
                    
                    # Model summary
                    self.sessionWindow.append("\nModel Summary:")
                    self.sessionWindow.append(f"R-squared = {model.rsquared:.4f}")
                    self.sessionWindow.append(f"Adjusted R-squared = {model.rsquared_adj:.4f}")
                    self.sessionWindow.append(f"Standard Error = {np.sqrt(model.mse_resid):.4f}")
                    
                    # Coefficients
                    self.sessionWindow.append("\nCoefficients:")
                    self.sessionWindow.append("Variable      Estimate    Std Error    t-value     p-value")
                    self.sessionWindow.append("-" * 60)
                    self.sessionWindow.append(f"{'Intercept':<12}{model.params[0]:10.4f}  {model.bse[0]:10.4f}  {model.tvalues[0]:10.4f}  {model.pvalues[0]:.4e}")
                    self.sessionWindow.append(f"{predictor_col:<12}{model.params[1]:10.4f}  {model.bse[1]:10.4f}  {model.tvalues[1]:10.4f}  {model.pvalues[1]:.4e}")
                    
                    # Regression equation
                    self.sessionWindow.append(f"\nRegression Equation:")
                    self.sessionWindow.append(f"{response_col} = {model.params[0]:.4f} + {model.params[1]:.4f}×{predictor_col}")
                    
                    # Analysis of Variance
                    self.sessionWindow.append("\nAnalysis of Variance:")
                    self.sessionWindow.append("Source      DF          SS          MS           F         P")
                    self.sessionWindow.append("-" * 70)
                    self.sessionWindow.append(f"{'Regression':<10}  {1:2}  {model.ess:11.4f}  {model.ess:11.4f}  {model.fvalue:11.4f}  {model.f_pvalue:.4e}")
                    self.sessionWindow.append(f"{'Residual':<10}  {model.df_resid:2}  {model.ssr:11.4f}  {model.ssr/model.df_resid:11.4f}")
                    self.sessionWindow.append(f"{'Total':<10}  {model.df_model + model.df_resid:2}  {model.ess + model.ssr:11.4f}")
                    
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
            QMessageBox.critical(self, "Error", f"Error in simple linear regression: {str(e)}")

    def multiple_linear_regression(self):
        """Perform multiple linear regression analysis"""
        self.loadDataFromTable()
        if self.data.empty:
            QMessageBox.warning(self, "Warning", "Please load or enter data first")
            return

        try:
            # Get numeric columns only
            numeric_columns = self.data.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_columns) < 3:
                QMessageBox.warning(self, "Warning", "Need at least three numeric columns for multiple regression")
                return

            # Get response variable
            response_col, ok1 = QInputDialog.getItem(self, "Select Response Variable", 
                "Choose response variable (Y):", numeric_columns, 0, False)
            
            if ok1:
                # Create dialog for selecting predictor variables
                dialog = QDialog()
                dialog.setWindowTitle("Select Predictor Variables")
                layout = QVBoxLayout()
                