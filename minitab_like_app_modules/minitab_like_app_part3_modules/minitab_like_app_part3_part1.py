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
                        QMessageBox.warning(self, "Warning", "Please select at least one predictor variable")
                        return
                    
                    # Prepare data
                    X = self.data[predictor_cols]
                    y = self.data[response_col]
                    
                    # Fit the model
                    X_with_const = sm.add_constant(X)
                    model = sm.OLS(y, X_with_const).fit()
                    
                    # Display results
                    self.sessionWindow.append("\nMultiple Linear Regression Results")
                    self.sessionWindow.append("-" * 40)
                    self.sessionWindow.append(f"Response Variable: {response_col}")
                    self.sessionWindow.append(f"Predictor Variables: {', '.join(predictor_cols)}")
                    
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
                    for i, col in enumerate(predictor_cols, 1):
                        self.sessionWindow.append(f"{col:<12}{model.params[i]:10.4f}  {model.bse[i]:10.4f}  {model.tvalues[i]:10.4f}  {model.pvalues[i]:.4e}")
                    
                    # Regression equation
                    self.sessionWindow.append(f"\nRegression Equation:")
                    equation = f"{response_col} = {model.params[0]:.4f}"
                    for i, col in enumerate(predictor_cols, 1):
                        equation += f" + {model.params[i]:.4f}×{col}"
                    self.sessionWindow.append(equation)
                    
                    # Analysis of Variance
                    self.sessionWindow.append("\nAnalysis of Variance:")
                    self.sessionWindow.append("Source      DF          SS          MS           F         P")
                    self.sessionWindow.append("-" * 70)
                    self.sessionWindow.append(f"{'Regression':<10}  {len(predictor_cols):2}  {model.ess:11.4f}  {model.ess/model.df_model:11.4f}  {model.fvalue:11.4f}  {model.f_pvalue:.4e}")
                    self.sessionWindow.append(f"{'Residual':<10}  {model.df_resid:2}  {model.ssr:11.4f}  {model.ssr/model.df_resid:11.4f}")
                    self.sessionWindow.append(f"{'Total':<10}  {model.df_model + model.df_resid:2}  {model.ess + model.ssr:11.4f}")
                    
                    # VIF values if more than one predictor
                    if len(predictor_cols) > 1:
                        self.sessionWindow.append("\nVariance Inflation Factors:")
                        self.sessionWindow.append("Variable      VIF")
                        self.sessionWindow.append("-" * 20)
                        # Calculate VIF for each predictor
                        for i, col in enumerate(predictor_cols):
                            other_cols = [c for c in predictor_cols if c != col]
                            X_others = self.data[other_cols]
                            X_target = self.data[col]
                            r_squared = sm.OLS(X_target, sm.add_constant(X_others)).fit().rsquared
                            vif = 1 / (1 - r_squared) if r_squared != 1 else float('inf')
                            self.sessionWindow.append(f"{col:<12}{vif:8.4f}")
                    
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
            QMessageBox.critical(self, "Error", f"Error in multiple linear regression: {str(e)}")

    def createDOE(self):
        """Create Design of Experiments"""
        # Create dialog for DOE type selection
        doe_type, ok = QInputDialog.getItem(self, "Design of Experiments",
            "Select DOE Type:",
            ["2-level Factorial", "Fractional Factorial", "Response Surface"], 0, False)
        if not ok:
            return

        if doe_type == "2-level Factorial":
            self.create_factorial_design()
        elif doe_type == "Fractional Factorial":
            self.create_fractional_factorial()
        else:
            self.create_response_surface()

    def create_factorial_design(self):
        """Create 2-level factorial design"""
        # Get number of factors
        n_factors, ok = QInputDialog.getInt(self, "Factorial Design", 
            "Enter number of factors (2-6):", 2, 2, 6)
        if not ok:
            return

        # Get factor names and levels
        factors = []
        levels = []
        for i in range(n_factors):
            # Get factor name
            name, ok = QInputDialog.getText(self, f"Factor {i+1}", 
                f"Enter name for factor {i+1}:")
            if not ok:
                return
            factors.append(name)
            
            # Get factor levels
            low, ok = QInputDialog.getText(self, f"Factor {i+1} Low", 
                f"Enter low level for {name}:")
            if not ok:
                return
            high, ok = QInputDialog.getText(self, f"Factor {i+1} High", 
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
        self.data = df
        self.updateTable()

        # Show design summary
        summary = f"""2-level Factorial Design Summary

Number of factors: {n_factors}
Number of runs: {n_runs}
Base design: Full factorial

Factors and Levels:
"""
        for i, factor in enumerate(factors):
            summary += f"{factor}: {levels[i][0]} | {levels[i][1]}\n"

        self.sessionWindow.setText(summary)

    def create_fractional_factorial(self):
        """Create fractional factorial design"""
        # Get number of factors
        n_factors, ok = QInputDialog.getInt(self, "Fractional Factorial Design", 
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
            QMessageBox.warning(self, "Warning", 
                "No valid resolution available for this number of factors")
            return

        resolution, ok = QInputDialog.getItem(self, "Fractional Factorial Design",
            "Select design resolution:", resolution_options, 0, False)
        if not ok:
            return
        
        resolution_level = int(resolution.split()[-1])

        # Get factor names and levels
        factors = []
        levels = []
        for i in range(n_factors):
            # Get factor name
            name, ok = QInputDialog.getText(self, f"Factor {i+1}", 
                f"Enter name for factor {i+1}:")
            if not ok:
                return
            factors.append(name)
            
            # Get factor levels
            low, ok = QInputDialog.getText(self, f"Factor {i+1} Low", 
                f"Enter low level for {name}:")
            if not ok:
                return
            high, ok = QInputDialog.getText(self, f"Factor {i+1} High", 
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
        self.data = df
        self.updateTable()

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

        self.sessionWindow.setText(summary)

    def create_response_surface(self):
        """Create response surface design"""
        # Get design type
        design_type, ok = QInputDialog.getItem(self, "Response Surface Design",
            "Select design type:",
            ["Central Composite Design (CCD)", "Box-Behnken Design (BBD)"], 0, False)
        if not ok:
            return

        # Get number of factors
        min_factors = 2 if design_type.startswith("Central") else 3
        max_factors = 6 if design_type.startswith("Central") else 7
        n_factors, ok = QInputDialog.getInt(self, "Response Surface Design", 
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
            name, ok = QInputDialog.getText(self, f"Factor {i+1}", 
                f"Enter name for factor {i+1}:")
            if not ok:
                return
            factors.append(name)
            
            # Get factor center point and range
            center, ok = QInputDialog.getDouble(self, f"Factor {i+1} Center", 
                f"Enter center point for {name}:")
            if not ok:
                return
            center_points.append(center)
            
            range_val, ok = QInputDialog.getDouble(self, f"Factor {i+1} Range", 
                f"Enter range (±) for {name}:")
            if not ok:
                return
            ranges.append(range_val)

        # Get number of center points
        n_center, ok = QInputDialog.getInt(self, "Center Points", 
            "Enter number of center points:", 3, 1, 10)
        if not ok:
            return

        # Create design matrix based on design type
        if design_type.startswith("Central"):
            # Central Composite Design
            # Get alpha type
            alpha_type, ok = QInputDialog.getItem(self, "CCD Alpha",
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
                QMessageBox.warning(self, "Warning", 
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
        self.data = df
        self.updateTable()

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

        self.sessionWindow.setText(summary)

    def analyzeDOE(self):
        """Analyze Design of Experiments"""
        # Load the current data from the table
        self.loadDataFromTable()
        
        if self.data.empty:
            QMessageBox.warning(self, "Warning", "No data available for analysis")
            return

        try:
            # Check if we have response data
            if 'Response' not in self.data.columns:
                QMessageBox.warning(self, "Warning", "No Response column found")
                return

            # Identify factor columns (exclude StdOrder, RunOrder, and Response)
            factor_cols = [col for col in self.data.columns 
                          if col not in ['StdOrder', 'RunOrder', 'Response']]

            if not factor_cols:
                QMessageBox.warning(self, "Warning", "No factor columns identified")
                return

            # Create a copy of the data to avoid modifying the original
            analysis_data = self.data.copy()

            # Convert response to numeric, dropping any non-numeric values
            analysis_data['Response'] = pd.to_numeric(analysis_data['Response'].astype(str).str.strip(), errors='coerce')
            
            # Drop any rows with missing response values
            analysis_data = analysis_data.dropna(subset=['Response'])

            if len(analysis_data) == 0:
                QMessageBox.warning(self, "Warning", "No valid response data after conversion")
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
                    QMessageBox.warning(self, "Warning", f"Factor {col} has no variation")
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
            self.sessionWindow.setText(report)

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
            QMessageBox.warning(self, "Error", 
                f"An error occurred during DOE analysis:\n{str(e)}\n\n"
                "Please check your data and try again.")

    def individualChart(self):
        """Create individual control chart"""
        try:
            # Get list of numeric columns only
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) < 1:
                QMessageBox.warning(self, "Warning", "Need at least one numeric column for measurements")
                return

            # Create dialog for options
            dialog = QDialog(self)
            dialog.setWindowTitle("Individual Chart Options")
            layout = QVBoxLayout()

            # Column selection
            col_label = QLabel("Select Measurement column:")
            layout.addWidget(col_label)
            
            col_combo = QComboBox()
            col_combo.addItems(numeric_cols)
            layout.addWidget(col_combo)

            # Display Tests option
            display_tests = QCheckBox("Display Tests")
            layout.addWidget(display_tests)

            # Alpha value selection
            alpha_layout = QHBoxLayout()
            alpha_label = QLabel("α value for control limits:")
            alpha_input = QLineEdit("0.05")  # Default value
            alpha_layout.addWidget(alpha_label)
            alpha_layout.addWidget(alpha_input)
            layout.addLayout(alpha_layout)

            # Moving Range length selection
            mr_layout = QHBoxLayout()
            mr_label = QLabel("Moving Range length:")
            mr_input = QLineEdit("2")  # Default value
            mr_layout.addWidget(mr_label)
            mr_layout.addWidget(mr_input)
            layout.addLayout(mr_layout)

            # Add OK and Cancel buttons
            button_box = QDialogButtonBox(
                QDialogButtonBox.StandardButton.Ok | 
                QDialogButtonBox.StandardButton.Cancel
            )
            button_box.accepted.connect(dialog.accept)
            button_box.rejected.connect(dialog.reject)
            layout.addWidget(button_box)

            dialog.setLayout(layout)
            
            # Show dialog and get results
            if dialog.exec() != QDialog.DialogCode.Accepted:
                return

            # Get selected options
            col = col_combo.currentText()
            show_tests = display_tests.isChecked()
            try:
                alpha = float(alpha_input.text())
                if not 0 < alpha < 1:
                    raise ValueError("Alpha must be between 0 and 1")
            except ValueError:
                QMessageBox.warning(self, "Warning", "Invalid α value. Using default 0.05")
                alpha = 0.05

            try:
                mr_length = int(mr_input.text())
                if mr_length < 2:
                    raise ValueError("Moving Range length must be at least 2")
            except ValueError:
                QMessageBox.warning(self, "Warning", "Invalid Moving Range length. Using default 2")
                mr_length = 2

            # Get the data and check for missing values
            data = pd.to_numeric(self.data[col], errors='coerce')
            if data.isna().any():
                QMessageBox.warning(self, "Warning", "Missing values found in the selected column")
                return

            # Calculate moving ranges
            moving_ranges = np.zeros(len(data)-mr_length+1)
            for i in range(len(moving_ranges)):
                moving_ranges[i] = np.ptp(data[i:i+mr_length])
            
            # Calculate control limits
            mean = np.mean(data)
            mr_mean = np.mean(moving_ranges)
            
            # Constants for n=2 (moving range of 2 consecutive points)
            E2 = 2.66
            D3 = 0
            D4 = 3.267
            
            # Individual chart limits
            i_ucl = mean + 3 * mr_mean / 1.128  # 1.128 = d2 for n=2
            i_lcl = mean - 3 * mr_mean / 1.128
            
            # Moving Range chart limits
            mr_ucl = D4 * mr_mean
            mr_lcl = D3 * mr_mean
            
            # Create plots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
            
            # Individual chart
            ax1.plot(range(1, len(data) + 1), data, marker='o', color='blue')
            ax1.axhline(y=mean, color='g', linestyle='-', label='CL')
            ax1.axhline(y=i_ucl, color='r', linestyle='--', label='UCL')
            ax1.axhline(y=i_lcl, color='r', linestyle='--', label='LCL')
            ax1.set_title('Individual Chart')
            ax1.set_xlabel('Observation')
            ax1.set_ylabel('Individual Value')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # Moving Range chart
            ax2.plot(range(1, len(moving_ranges) + 1), moving_ranges, marker='o', color='blue')
            ax2.axhline(y=mr_mean, color='g', linestyle='-', label='CL')
            ax2.axhline(y=mr_ucl, color='r', linestyle='--', label='UCL')
            ax2.axhline(y=mr_lcl, color='r', linestyle='--', label='LCL')
            ax2.set_title('Moving Range Chart')
            ax2.set_xlabel('Observation')
            ax2.set_ylabel('Moving Range')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            plt.tight_layout()
            plt.show()
            
            # Format output text
            result_text = f"Individual Chart Analysis for {col}\n\n"
            result_text += "Individual Chart Statistics:\n"
            result_text += f"Mean: {mean:.3f}\n"
            result_text += f"UCL: {i_ucl:.3f}\n"
            result_text += f"LCL: {i_lcl:.3f}\n\n"
            result_text += f"Number of Points: {len(data)}\n"
            result_text += f"Points Outside Control Limits: {sum((data > i_ucl) | (data < i_lcl))}\n\n"
            result_text += f"Note: Control limits are based on ±3 sigma (calculated using moving ranges)\n"
            if show_tests:
                result_text += "\nTests for Special Causes:\n"
                # Add test results here based on the selected alpha value
                
            self.sessionWindow.setText(result_text)
            
        except Exception as e:
            QMessageBox.warning(self, "Error", f"An error occurred: {str(e)}")
            return

    def movingRangeChart(self, col=None):
        """Create moving range chart"""
        if col is None:
            col = self.selectColumnDialog()
            if not col:
                return
                
        try:
            # Get numeric data
            data = pd.to_numeric(self.data[col], errors='coerce').dropna()
            
            if len(data) < 2:
                QMessageBox.warning(self, "Warning", "Insufficient data for moving range chart")
                return
                
            # Calculate moving range
            moving_range = np.abs(data.diff().dropna())
            
            # Calculate control limits
            mr_mean = moving_range.mean()
            mr_std = moving_range.std()
            mr_ucl = mr_mean + 3 * mr_std
            mr_lcl = max(0, mr_mean - 3 * mr_std)
            
            # Create individual chart
            plt.figure(figsize=(10, 8))
            plt.subplot(2, 1, 1)
            plt.plot(data.index, data, marker='o', linestyle='-', color='blue')
            plt.axhline(y=data.mean(), color='green', linestyle='-', label='Mean')
            plt.axhline(y=data.mean() + 3 * (mr_mean / 1.128), color='red', linestyle='--', label='UCL')
            plt.axhline(y=data.mean() - 3 * (mr_mean / 1.128), color='red', linestyle='--', label='LCL')
            plt.title(f'Individual Chart for {col}')
            plt.xlabel('Observation')
            plt.ylabel('Value')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Create moving range chart
            plt.subplot(2, 1, 2)
            plt.plot(moving_range.index, moving_range, marker='o', linestyle='-', color='blue')
            plt.axhline(y=mr_mean, color='green', linestyle='-', label='Mean')
            plt.axhline(y=mr_ucl, color='red', linestyle='--', label='UCL')
            plt.axhline(y=mr_lcl, color='red', linestyle='--', label='LCL')
            plt.title(f'Moving Range Chart for {col}')
            plt.xlabel('Observation')
            plt.ylabel('Moving Range')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Create report
            report = f"""Moving Range Chart Analysis for {col}

Moving Range Statistics:
Mean MR: {mr_mean:.3f}
UCL: {mr_ucl:.3f}
LCL: {mr_lcl:.3f}

Number of Ranges: {len(moving_range)}
Ranges Outside Control Limits: {sum((moving_range > mr_ucl) | (moving_range < mr_lcl))}

Note: Control limits are based on ±3 sigma
"""
            self.sessionWindow.setText(report)
            plt.show()
        except Exception as e:
            QMessageBox.warning(self, "Error", f"An error occurred: {str(e)}")

    def gageRR(self):
        """Perform Gage R&R Study"""
        # Create dialog for study type selection
        study_type, ok = QInputDialog.getItem(self, "Gage R&R Study Type",
            "Select Study Type:",
            ["Crossed (default)", "Nested"], 0, False)
        if not ok:
            return

        # Get part column with explicit prompt
        part_col = self.selectColumnDialog("Select Part Column")
        if not part_col:
            return

        # Get operator column with explicit prompt
        operator_col = self.selectColumnDialog("Select Operator Column")
        if not operator_col:
            return

        # Get measurement column with explicit prompt
        measurement_col = self.selectColumnDialog("Select Measurement Column")
        if not measurement_col:
            return

        # Get order column (optional) with explicit prompt
        order_col = self.selectColumnDialog("Select Order Column (optional)")
        # Order column is optional, so we continue even if it's None

        # Create options dialog
        options_dialog = QDialog(self)
        options_dialog.setWindowTitle("Gage R&R Study Options")
        options_layout = QVBoxLayout()

        # Create tabs
        tab_widget = QTabWidget()
        
        # Basic tab is already handled by the previous prompts
        
        # Options tab
        options_tab = QWidget()
        options_tab_layout = QVBoxLayout()
        
        # Study Information group
        study_info_group = QGroupBox("Study Information")
        study_info_layout = QVBoxLayout()
        
        # Number of replicates
        replicates_layout = QHBoxLayout()
        replicates_label = QLabel("Number of replicates (2-5):")
        replicates_spin = QSpinBox()
        replicates_spin.setRange(2, 5)
        replicates_spin.setValue(3)  # Default value
        replicates_layout.addWidget(replicates_label)
        replicates_layout.addWidget(replicates_spin)
        study_info_layout.addLayout(replicates_layout)
        
        # Process tolerance
        tolerance_layout = QHBoxLayout()
        tolerance_check = QCheckBox("Process tolerance (optional):")
        tolerance_check.setChecked(False)
        tolerance_value = QDoubleSpinBox()
        tolerance_value.setRange(0.01, 1000.0)
        tolerance_value.setValue(0.60)  # Default value
        tolerance_value.setEnabled(False)
        tolerance_check.toggled.connect(tolerance_value.setEnabled)
        tolerance_layout.addWidget(tolerance_check)
        tolerance_layout.addWidget(tolerance_value)
        study_info_layout.addLayout(tolerance_layout)
        
        study_info_group.setLayout(study_info_layout)
        options_tab_layout.addWidget(study_info_group)
        
        # Analysis Options group
        analysis_group = QGroupBox("Analysis Options")
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
        
        # Include interaction
        interaction_check = QCheckBox("Include interaction")
        interaction_check.setChecked(True)
        analysis_layout.addWidget(interaction_check)
        
        # Use historical standard deviation
        hist_std_check = QCheckBox("Use historical standard deviation (if available)")
        hist_std_check.setChecked(False)
        analysis_layout.addWidget(hist_std_check)
        
        analysis_group.setLayout(analysis_layout)
        options_tab_layout.addWidget(analysis_group)
        
        options_tab.setLayout(options_tab_layout)
        
        # Graphs tab
        graphs_tab = QWidget()
        graphs_layout = QVBoxLayout()
        
        # Graph options
        components_check = QCheckBox("Components of Variation")
        components_check.setChecked(True)
        graphs_layout.addWidget(components_check)
        
        r_chart_check = QCheckBox("R Chart by Operator")
        r_chart_check.setChecked(True)
        graphs_layout.addWidget(r_chart_check)
        
        xbar_chart_check = QCheckBox("X-bar Chart by Operator")
        xbar_chart_check.setChecked(True)
        graphs_layout.addWidget(xbar_chart_check)
        
        by_part_check = QCheckBox("Measurement by Part")
        by_part_check.setChecked(True)
        graphs_layout.addWidget(by_part_check)
        
        by_operator_check = QCheckBox("Measurement by Operator")
        by_operator_check.setChecked(True)
        graphs_layout.addWidget(by_operator_check)
        
        interaction_plot_check = QCheckBox("Part*Operator Interaction")
        interaction_plot_check.setChecked(True)
        graphs_layout.addWidget(interaction_plot_check)
        
        # Run chart only if order column is provided
        run_chart_check = QCheckBox("Run Chart (if Order provided)")
        run_chart_check.setChecked(order_col is not None)
        run_chart_check.setEnabled(order_col is not None)
        graphs_layout.addWidget(run_chart_check)
        
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
                'Measurement': pd.to_numeric(self.data[measurement_col], errors='coerce'),
                'Operator': self.data[operator_col],
                'Part': self.data[part_col]
            })
            
            # Add Order column if provided
            if order_col:
                df['Order'] = self.data[order_col]

            # Remove missing values
            df = df.dropna()

            if len(df) < 2:
                QMessageBox.warning(self, "Warning", "Insufficient data for Gage R&R analysis")
                return

            # Calculate overall statistics
            total_mean = df['Measurement'].mean()
            total_std = df['Measurement'].std()

            # Calculate components of variation
            operators = df['Operator'].unique()
            parts = df['Part'].unique()
            n_operators = len(operators)
            n_parts = len(parts)
            n_measurements = len(df) / (n_operators * n_parts)

            # Perform ANOVA analysis
            formula = f"Measurement ~ C(Part) + C(Operator)"
            if interaction_check.isChecked():
                formula += " + C(Part):C(Operator)"
            
            model = sm.formula.ols(formula, data=df).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)
            
            # Calculate operator variation
            operator_means = df.groupby('Operator')['Measurement'].mean()
            operator_variation = operator_means.std() ** 2

            # Calculate part variation
            part_means = df.groupby('Part')['Measurement'].mean()
            part_variation = part_means.std() ** 2

            # Calculate repeatability (equipment variation)
            residuals = []
            for part in parts:
                for operator in operators:
                    part_operator_data = df[(df['Part'] == part) & (df['Operator'] == operator)]['Measurement']
                    if len(part_operator_data) > 1:
                        residuals.extend(part_operator_data - part_operator_data.mean())
            
            repeatability_variation = np.var(residuals, ddof=1) if residuals else 0

            # Calculate reproducibility (operator variation)
            reproducibility_variation = max(0, operator_variation - repeatability_variation / (n_parts * n_measurements))

            # Calculate total variation
            total_variation = part_variation + repeatability_variation + reproducibility_variation
