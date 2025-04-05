"""
Msa module
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QFormLayout, QGroupBox, QComboBox, 
                           QSpinBox, QDialogButtonBox, QLabel, QMessageBox,
                           QInputDialog, QCheckBox, QTabWidget, QWidget, QHBoxLayout,
                           QDoubleSpinBox, QLineEdit)


def gage_rr(main_window):
        """Perform Gage R&R Study"""
        # Create dialog for study type selection
        study_type, ok = QInputDialog.getItem(main_window, "Gage R&R Study Type",
            "Select Study Type:",
            ["Crossed (default)", "Nested"], 0, False)
        if not ok:
            return

        # Get part column with explicit prompt
        part_col = main_window.selectColumnDialog("Select Part Column")
        if not part_col:
            return

        # Get operator column with explicit prompt
        operator_col = main_window.selectColumnDialog("Select Operator Column")
        if not operator_col:
            return

        # Get measurement column with explicit prompt
        measurement_col = main_window.selectColumnDialog("Select Measurement Column")
        if not measurement_col:
            return

        # Get order column (optional) with explicit prompt
        order_col = main_window.selectColumnDialog("Select Order Column (optional)")
        # Order column is optional, so we continue even if it's None

        # Create options dialog
        options_dialog = QDialog(main_window)
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
                'Measurement': pd.to_numeric(main_window.data[measurement_col], errors='coerce'),
                'Operator': main_window.data[operator_col],
                'Part': main_window.data[part_col]
            })
            
            # Add Order column if provided
            if order_col:
                df['Order'] = main_window.data[order_col]

            # Remove missing values
            df = df.dropna()

            if len(df) < 2:
                QMessageBox.warning(main_window, "Warning", "Insufficient data for Gage R&R analysis")
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
            
            # Add Part*Operator row if included
            if interaction_check.isChecked():
                interaction_row = anova_table.loc["C(Part):C(Operator)"]
                anova_display += f"Part*Operator       {int(interaction_row['df']):2d}    {interaction_row['sum_sq']:.6f}  {interaction_row['sum_sq']/interaction_row['df']:.6f}  {interaction_row['F']:.6f}  {interaction_row['PR(>F)']:.6f}\n"
            
            # Calculate residual
            residual_df = len(df) - model.df_model - 1
            residual_ss = model.ssr
            residual_ms = residual_ss / residual_df
            
            anova_display += f"Residual            {int(residual_df):2d}    {residual_ss:.6f}  {residual_ms:.6f}\n"
            
            # Add Total row
            total_df = len(df) - 1
            total_ss = model.centered_tss
            anova_display += f"Total               {int(total_df):2d}    {total_ss:.6f}\n"
            
            # Calculate number of distinct categories (NDC)
            ndc = 1.41 * (part_variation / total_variation) ** 0.5 * n_parts ** 0.5
            
            # Create report
            report = f"""Gage R&R Study Results

Study Information:
  Parts: {n_parts}
  Operators: {n_operators}
  Replicates: {int(n_measurements)}
  Study Type: {study_type}
  Confidence Level: {confidence_combo.currentText()}

Overall Statistics:
  Mean: {total_mean:.6f}
  Standard Deviation: {total_std:.6f}

{anova_display}

Variance Components:
  Source              Contribution    Study Var    %Study Var    %Tolerance
  ------------------  -------------   ----------   -----------   -----------
  Total Gage R&R      {contribution['Repeatability'] + contribution['Reproducibility']:.2f}%    {study_var['Repeatability'] + study_var['Reproducibility']:.6f}    {np.sqrt((repeatability_variation + reproducibility_variation) / total_variation) * 100:.2f}%    """
            
            if process_tolerance:
                report += f"{(study_var['Repeatability'] + study_var['Reproducibility']) / process_tolerance * 100:.2f}%"
            else:
                report += "N/A"
                
            report += f"""
  Repeatability       {contribution['Repeatability']:.2f}%    {study_var['Repeatability']:.6f}    {np.sqrt(repeatability_variation / total_variation) * 100:.2f}%    """
            
            if process_tolerance:
                report += f"{study_var['Repeatability'] / process_tolerance * 100:.2f}%"
            else:
                report += "N/A"
                
            report += f"""
  Reproducibility     {contribution['Reproducibility']:.2f}%    {study_var['Reproducibility']:.6f}    {np.sqrt(reproducibility_variation / total_variation) * 100:.2f}%    """
            
            if process_tolerance:
                report += f"{study_var['Reproducibility'] / process_tolerance * 100:.2f}%"
            else:
                report += "N/A"
                
            report += f"""
  Part-to-Part        {contribution['Part-to-Part']:.2f}%    {study_var['Part-to-Part']:.6f}    {np.sqrt(part_variation / total_variation) * 100:.2f}%    """
            
            if process_tolerance:
                report += f"{study_var['Part-to-Part'] / process_tolerance * 100:.2f}%"
            else:
                report += "N/A"
                
            report += f"""

Total Variation       100.00%    {study_var['Total']:.6f}    100.00%    """
            
            if process_tolerance:
                report += f"{study_var['Total'] / process_tolerance * 100:.2f}%"
            else:
                report += "N/A"
            
            # Add Number of Distinct Categories
            report += f"\n\nNumber of Distinct Categories: {int(ndc)}"
            
            # Add assessment based on %Study Var for Total Gage R&R
            gage_rr_percent = np.sqrt((repeatability_variation + reproducibility_variation) / total_variation) * 100
            
            report += "\n\nAssessment:"
            if gage_rr_percent < 10:
                report += "\n  Measurement System is acceptable."
            elif gage_rr_percent < 30:
                report += "\n  Measurement System may be acceptable depending on application."
            else:
                report += "\n  Measurement System needs improvement."
                
            if ndc < 5:
                report += "\n  Number of distinct categories is too low for adequate analysis."
            
            # Generate visualizations if requested
            fig = None
            if any([components_check.isChecked(), r_chart_check.isChecked(), xbar_chart_check.isChecked(),
                   by_part_check.isChecked(), by_operator_check.isChecked(), interaction_plot_check.isChecked(),
                   run_chart_check.isChecked() and order_col is not None]):
                
                # Create figure for plots
                num_plots = sum([components_check.isChecked(), r_chart_check.isChecked(), xbar_chart_check.isChecked(),
                               by_part_check.isChecked(), by_operator_check.isChecked(), interaction_plot_check.isChecked(),
                               run_chart_check.isChecked() and order_col is not None])
                
                # 计算行列布局
                if num_plots <= 2:
                    n_rows, n_cols = num_plots, 1
                elif num_plots <= 4:
                    n_rows, n_cols = 2, 2
                elif num_plots <= 6:
                    n_rows, n_cols = 2, 3
                else:
                    n_rows, n_cols = 3, 3
                
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*7, n_rows*5))
                # 将axes转为一维数组以便于索引
                axes = axes.flatten() if num_plots > 1 else [axes]
                
                # 隐藏未使用的子图
                for i in range(num_plots, len(axes)):
                    axes[i].set_visible(False)
                
                plot_idx = 0
                
                # Components of Variation
                if components_check.isChecked():
                    ax = axes[plot_idx]
                    plot_idx += 1
                    
                    # Data for bar chart
                    sources = ['Gage R&R', 'Repeatability', 'Reproducibility', 'Part-to-Part']
                    values = [contribution['Repeatability'] + contribution['Reproducibility'],
                             contribution['Repeatability'], contribution['Reproducibility'],
                             contribution['Part-to-Part']]
                    
                    ax.bar(sources, values, color=['blue', 'green', 'red', 'purple'])
                    ax.set_ylabel('Percent Contribution (%)')
                    ax.set_title('Components of Variation')
                    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
                
                # X-bar Chart by Operator
                if xbar_chart_check.isChecked():
                    ax = axes[plot_idx]
                    plot_idx += 1
                    
                    # Calculate means for each part and operator
                    part_operator_means = df.groupby(['Part', 'Operator'])['Measurement'].mean().unstack()
                    
                    # Plot means for each operator
                    for operator in operators:
                        ax.plot(part_operator_means.index, part_operator_means[operator], 'o-', label=f'Operator {operator}')
                    
                    ax.set_xlabel('Part')
                    ax.set_ylabel('Mean Measurement')
                    ax.set_title('X-bar Chart by Operator')
                    ax.legend()
                    ax.grid(True, linestyle='--', alpha=0.7)
                
                # R Chart by Operator
                if r_chart_check.isChecked():
                    ax = axes[plot_idx]
                    plot_idx += 1
                    
                    # Calculate ranges for each part and operator
                    part_operator_ranges = df.groupby(['Part', 'Operator'])['Measurement'].apply(lambda x: x.max() - x.min()).unstack()
                    
                    # Plot ranges for each operator
                    for operator in operators:
                        ax.plot(part_operator_ranges.index, part_operator_ranges[operator], 'o-', label=f'Operator {operator}')
                    
                    ax.set_xlabel('Part')
                    ax.set_ylabel('Range')
                    ax.set_title('R Chart by Operator')
                    ax.legend()
                    ax.grid(True, linestyle='--', alpha=0.7)
                
                # Measurement by Part
                if by_part_check.isChecked():
                    ax = axes[plot_idx]
                    plot_idx += 1
                    
                    # Calculate statistics for boxplot
                    part_data = [df[df['Part'] == part]['Measurement'].values for part in parts]
                    
                    # Create boxplot
                    ax.boxplot(part_data, labels=parts)
                    ax.set_xlabel('Part')
                    ax.set_ylabel('Measurement')
                    ax.set_title('Measurement by Part')
                    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
                
                # Measurement by Operator
                if by_operator_check.isChecked():
                    ax = axes[plot_idx]
                    plot_idx += 1
                    
                    # Calculate statistics for boxplot
                    operator_data = [df[df['Operator'] == operator]['Measurement'].values for operator in operators]
                    
                    # Create boxplot
                    ax.boxplot(operator_data, labels=operators)
                    ax.set_xlabel('Operator')
                    ax.set_ylabel('Measurement')
                    ax.set_title('Measurement by Operator')
                    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
                
                # Part*Operator Interaction
                if interaction_plot_check.isChecked():
                    ax = axes[plot_idx]
                    plot_idx += 1
                    
                    # Calculate means for interaction plot
                    interaction_data = df.groupby(['Part', 'Operator'])['Measurement'].mean().reset_index()
                    
                    # Create interaction plot
                    for operator in operators:
                        op_data = interaction_data[interaction_data['Operator'] == operator]
                        ax.plot(op_data['Part'], op_data['Measurement'], 'o-', label=f'Operator {operator}')
                    
                    ax.set_xlabel('Part')
                    ax.set_ylabel('Mean Measurement')
                    ax.set_title('Part*Operator Interaction')
                    ax.legend()
                    ax.grid(True, linestyle='--', alpha=0.7)
                
                # Run Chart (if order column provided)
                if run_chart_check.isChecked() and order_col is not None:
                    ax = axes[plot_idx]
                    plot_idx += 1
                    
                    # Sort by order
                    run_data = df.sort_values('Order')
                    
                    # Create run chart
                    ax.plot(run_data['Order'], run_data['Measurement'], 'o-')
                    ax.axhline(y=total_mean, color='r', linestyle='--', label='Mean')
                    
                    ax.set_xlabel('Order')
                    ax.set_ylabel('Measurement')
                    ax.set_title('Run Chart', pad=15, fontsize=12, loc='left')
                    ax.legend(loc='best', fontsize=9)
                    ax.grid(True, linestyle='--', alpha=0.7)
                
                plt.tight_layout()
                
                # 直接显示图形
                plt.show()
                
                # 不再使用图像标签
                image_tag = ""
            else:
                image_tag = ""
            
            # Update session window with report
            html_report = f"<pre>{report}</pre>{image_tag}"
            main_window.sessionWindow.setText(html_report)
            
        except Exception as e:
            QMessageBox.critical(main_window, "Error", f"Error performing Gage R&R analysis: {str(e)}")
            import traceback
            traceback.print_exc()

def linearity_study(main_window):
    """Perform measurement system linearity study"""
    # Get reference column with explicit prompt
    reference_col = main_window.selectColumnDialog("Select Reference Column")
    if not reference_col:
        return

    # Get measurement column with explicit prompt
    measurement_col = main_window.selectColumnDialog("Select Measurement Column")
    if not measurement_col:
        return

    # Get part column with explicit prompt
    part_col = main_window.selectColumnDialog("Select Part/Sample Column")
    if not part_col:
        return

    # Get operator column (optional) with explicit prompt
    operator_col = main_window.selectColumnDialog("Select Operator Column (optional)")
    # Operator column is optional, so we continue even if it's None
    
    # Create dialog
    dialog = QDialog(main_window)
    dialog.setWindowTitle("Linearity Study")
    dialog.setMinimumWidth(600)
    dialog.setMinimumHeight(400)  # 增加最小高度以避免元素擠壓
    
    # Create dialog layout
    main_layout = QVBoxLayout(dialog)
    main_layout.setSpacing(10)  # 增加間距
    
    # Create tabs
    tabs = QTabWidget()
    options_tab = QWidget()
    graphs_tab = QWidget()
    
    # Options tab layout
    options_layout = QVBoxLayout(options_tab)
    options_layout.setSpacing(15)  # 增加控件間的間距
    options_layout.setContentsMargins(10, 20, 10, 10)  # 增加上邊距
    
    # Analysis settings
    analysis_group = QGroupBox("Analysis Settings")
    analysis_group.setMinimumHeight(80)  # 確保分組框有足夠的高度
    analysis_layout = QFormLayout(analysis_group)
    analysis_layout.setVerticalSpacing(10)  # 增加表單項目間的垂直間距
    analysis_layout.setContentsMargins(10, 20, 10, 10)  # 設置內邊距
    
    confidence_combo = QComboBox()
    confidence_combo.addItems(["90%", "95%", "99%"])
    confidence_combo.setCurrentText("95%")
    analysis_layout.addRow("Confidence level:", confidence_combo)
    
    options_layout.addWidget(analysis_group)
    
    # Tolerance information
    tolerance_group = QGroupBox("Tolerance Information")
    tolerance_group.setMinimumHeight(100)  # 確保分組框有足夠的高度
    tolerance_layout = QFormLayout(tolerance_group)
    tolerance_layout.setVerticalSpacing(10)  # 增加表單項目間的垂直間距
    tolerance_layout.setContentsMargins(10, 20, 10, 10)  # 設置內邊距
    
    tolerance_input = QLineEdit()
    tolerance_layout.addRow("Tolerance range:", tolerance_input)
    
    acceptable_percent_input = QLineEdit("5.0")
    tolerance_layout.addRow("Acceptable percent (%):", acceptable_percent_input)
    
    options_layout.addWidget(tolerance_group)
    options_layout.addStretch(1)  # 添加彈性空間
    
    # Graphs tab layout
    graphs_layout = QVBoxLayout(graphs_tab)
    graphs_layout.setSpacing(10)  # 增加間距
    graphs_layout.setContentsMargins(10, 10, 10, 10)  # 設置內邊距
    
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
    
    residuals_plot_check = QCheckBox("Residuals plot")
    residuals_plot_check.setChecked(True)
    graphs_layout.addWidget(residuals_plot_check)
    
    graphs_layout.addStretch(1)  # 添加彈性空間
    
    # Add tabs to tab widget
    tabs.addTab(options_tab, "Options")
    tabs.addTab(graphs_tab, "Graphs")
    
    # 添加标签页到主布局
    main_layout.addWidget(tabs)
    
    # Dialog buttons
    button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
    button_box.accepted.connect(dialog.accept)
    button_box.rejected.connect(dialog.reject)
    main_layout.addWidget(button_box)
    
    # Show dialog
    if not dialog.exec():
        return
    
    # Get confidence level
    conf_level_text = confidence_combo.currentText()
    conf_level = float(conf_level_text.strip('%')) / 100
    
    # Get tolerance settings
    tolerance_range = None
    try:
        if tolerance_input.text():
            tolerance_range = float(tolerance_input.text())
    except ValueError:
        QMessageBox.warning(main_window, "Warning", "Invalid tolerance range. This will be ignored.")
    
    acceptable_percent = 5.0
    try:
        if acceptable_percent_input.text():
            acceptable_percent = float(acceptable_percent_input.text())
    except ValueError:
        QMessageBox.warning(main_window, "Warning", "Invalid acceptable percentage. Using default 5%.")
    
    # Get selected graphs
    show_linearity_plot = linearity_plot_check.isChecked()
    show_bias_plot = bias_plot_check.isChecked()
    show_percent_bias_plot = percent_bias_plot_check.isChecked()
    show_fitted_line_plot = fitted_line_plot_check.isChecked()
    show_residuals_plot = residuals_plot_check.isChecked()
    
    try:
        # Create DataFrame for analysis
        analysis_df = pd.DataFrame({
            'Reference': pd.to_numeric(main_window.data[reference_col], errors='coerce'),
            'Measurement': pd.to_numeric(main_window.data[measurement_col], errors='coerce'),
            'Part': main_window.data[part_col]
        })
        
        # Add operator column if selected
        if operator_col:
            analysis_df['Operator'] = main_window.data[operator_col]
        
        # Remove missing values
        analysis_df = analysis_df.dropna(subset=['Reference', 'Measurement', 'Part'])
        
        if len(analysis_df) < 10:
            QMessageBox.warning(main_window, "Warning", "Insufficient data for linearity analysis (minimum 10 measurements required)")
            return
        
        # Calculate bias for each reference value
        analysis_df['Bias'] = analysis_df['Measurement'] - analysis_df['Reference']
        analysis_df['Percent_Bias'] = 100 * analysis_df['Bias'] / analysis_df['Reference']
        
        # Calculate average bias for each reference value
        reference_stats = analysis_df.groupby('Reference').agg({
            'Measurement': ['mean', 'std', 'count'],
            'Bias': ['mean', 'std'],
            'Percent_Bias': ['mean', 'std']
        })
        
        # Rename columns
        reference_stats.columns = ['Mean_Measurement', 'StdDev_Measurement', 'Count',
                                  'Mean_Bias', 'StdDev_Bias',
                                  'Mean_Percent_Bias', 'StdDev_Percent_Bias']
        
        reference_stats = reference_stats.reset_index()
        
        # Perform linearity analysis (linear regression of bias vs reference value)
        X = reference_stats['Reference'].values.reshape(-1, 1)
        y = reference_stats['Mean_Bias'].values
        
        # Add constant to X for intercept
        X_with_const = sm.add_constant(X)
        
        # Fit linear model
        model = sm.OLS(y, X_with_const).fit()
        
        # Get regression coefficients
        intercept = model.params[0]
        slope = model.params[1]
        
        # Calculate R-squared and p-values
        r_squared = model.rsquared
        p_value_intercept = model.pvalues[0]
        p_value_slope = model.pvalues[1]
        
        # Calculate predicted bias values
        reference_stats['Predicted_Bias'] = intercept + slope * reference_stats['Reference']
        
        # Generate report
        report = f"""Linearity Study Results

Summary Statistics by Reference Value:
Reference    Mean Measurement    StdDev    Count    Mean Bias    StdDev Bias    % Bias
{'-' * 90}
"""
        
        for _, row in reference_stats.iterrows():
            report += f"{row['Reference']:<12.4f}{row['Mean_Measurement']:<20.4f}{row['StdDev_Measurement']:<10.4f}"
            report += f"{row['Count']:<9.0f}{row['Mean_Bias']:<14.4f}{row['StdDev_Bias']:<15.4f}"
            report += f"{row['Mean_Percent_Bias']:<8.2f}%\n"
        
        report += f"""
Linearity Regression Analysis:
Bias = {intercept:.6f} + {slope:.6f} * Reference

R-squared: {r_squared:.4f}
P-value (Intercept): {p_value_intercept:.6f}
P-value (Slope): {p_value_slope:.6f}

"""
        
        # Add interpretation
        report += "Interpretation:\n"
        
        # Assess linearity (slope significance)
        if p_value_slope < 0.05:
            report += "- The measurement system exhibits significant linearity (p < 0.05).\n"
            report += f"- For each unit increase in the reference value, the bias changes by {slope:.6f} units.\n"
        else:
            report += "- The measurement system does not exhibit significant linearity (p >= 0.05).\n"
        
        # Assess bias (intercept significance)
        if p_value_intercept < 0.05:
            report += f"- There is a significant constant bias of {intercept:.6f} units (p < 0.05).\n"
        else:
            report += "- There is no significant constant bias (p >= 0.05).\n"
        
        # Overall assessment
        if abs(slope) < 0.01 and abs(intercept) < 0.01:
            report += "- The measurement system has good linearity and minimal bias.\n"
        elif abs(slope) < 0.01 and abs(intercept) >= 0.01:
            report += "- The measurement system has good linearity but shows constant bias.\n"
        elif abs(slope) >= 0.01 and abs(intercept) < 0.01:
            report += "- The measurement system has poor linearity but minimal constant bias.\n"
        else:
            report += "- The measurement system has both poor linearity and significant constant bias.\n"
        
        # Check if tolerance provided for capability assessment
        if tolerance_range:
            # Calculate percent of tolerance for linearity
            max_bias_change = abs(slope) * (reference_stats['Reference'].max() - reference_stats['Reference'].min())
            percent_of_tolerance = 100 * max_bias_change / tolerance_range
            
            report += f"\nLinearity Assessment with Tolerance:\n"
            report += f"- Maximum bias change across range: {max_bias_change:.4f} units\n"
            report += f"- Percent of tolerance: {percent_of_tolerance:.2f}%\n"
            
            if percent_of_tolerance <= acceptable_percent:
                report += f"- Linearity is acceptable (below {acceptable_percent}% of tolerance)\n"
            else:
                report += "- Linearity is NOT acceptable (above {acceptable_percent}% of tolerance)\n"
        
        # Generate visualizations if requested
        if show_linearity_plot or show_bias_plot or show_percent_bias_plot or show_fitted_line_plot or show_residuals_plot:
            # Create figure for plots
            num_plots = sum([show_linearity_plot, show_bias_plot, show_percent_bias_plot, show_fitted_line_plot, show_residuals_plot])
            
            # 计算行列布局
            if num_plots <= 2:
                n_rows, n_cols = num_plots, 1
            elif num_plots <= 4:
                n_rows, n_cols = 2, 2
            else:
                n_rows, n_cols = 3, 2
            
            # 創建圖形並增加總體高度，確保標題有足夠空間
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*9, n_rows*7))
            
            # 設置全域字型大小
            plt.rcParams.update({'font.size': 10})
            
            # 設置子圖之間的間距（增加間距）
            plt.subplots_adjust(hspace=0.5, wspace=0.5)
            
            # 将axes转为一维数组以便于索引
            axes = axes.flatten() if num_plots > 1 else [axes]
            
            # 隐藏未使用的子图
            for i in range(num_plots, len(axes)):
                axes[i].set_visible(False)
            
            plot_idx = 0
            
            # Linearity plot (Bias vs Reference)
            if show_linearity_plot:
                ax = axes[plot_idx]
                plot_idx += 1
                
                # Scatter plot of mean bias vs reference
                ax.scatter(reference_stats['Reference'], reference_stats['Mean_Bias'], 
                          s=80, color='blue', label='Mean Bias')
                
                # Add error bars for standard deviation
                ax.errorbar(reference_stats['Reference'], reference_stats['Mean_Bias'],
                           yerr=reference_stats['StdDev_Bias'], fmt='none', color='blue', alpha=0.5)
                
                # Add regression line
                x_range = np.linspace(reference_stats['Reference'].min(), reference_stats['Reference'].max(), 100)
                ax.plot(x_range, intercept + slope * x_range, color='red', linestyle='--', 
                       label=f'y = {intercept:.4f} + {slope:.4f}x')
                
                # Add zero line
                ax.axhline(y=0, color='green', linestyle='-', alpha=0.7, label='Zero Bias')
                
                ax.set_xlabel('Reference Value')
                ax.set_ylabel('Bias')
                # 縮短標題文字，增加間距，設置靠左對齊
                ax.set_title('Linearity Plot', pad=15, fontsize=12, loc='left')
                ax.legend(loc='best', fontsize=9)
                ax.grid(True, linestyle='--', alpha=0.7)
                
                # 設置較大的標記點和字體
                for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                            ax.get_xticklabels() + ax.get_yticklabels()):
                    item.set_fontsize(10)
            
            # Bias plot (% Bias vs Reference)
            if show_bias_plot:
                ax = axes[plot_idx]
                plot_idx += 1
                
                # Scatter plot of mean percent bias vs reference
                ax.scatter(reference_stats['Reference'], reference_stats['Mean_Bias'], 
                          s=80, color='purple', label='Mean Bias')
                
                # Add error bars for standard deviation
                ax.errorbar(reference_stats['Reference'], reference_stats['Mean_Bias'],
                           yerr=reference_stats['StdDev_Bias'], fmt='none', color='purple', alpha=0.5)
                
                # Add zero line
                ax.axhline(y=0, color='green', linestyle='-', alpha=0.7, label='Zero Bias')
                
                ax.set_xlabel('Reference Value')
                ax.set_ylabel('Bias')
                # 縮短標題文字，增加間距，設置靠左對齊
                ax.set_title('Bias Plot', pad=15, fontsize=12, loc='left')
                ax.legend(loc='best', fontsize=9)
                ax.grid(True, linestyle='--', alpha=0.7)
                
                # 設置較大的標記點和字體
                for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                            ax.get_xticklabels() + ax.get_yticklabels()):
                    item.set_fontsize(10)
            
            # Percent bias plot
            if show_percent_bias_plot:
                ax = axes[plot_idx]
                plot_idx += 1
                
                # Scatter plot of mean percent bias vs reference
                ax.scatter(reference_stats['Reference'], reference_stats['Mean_Percent_Bias'], 
                          s=80, color='orange', label='Mean % Bias')
                
                # Add error bars for standard deviation
                ax.errorbar(reference_stats['Reference'], reference_stats['Mean_Percent_Bias'],
                           yerr=reference_stats['StdDev_Percent_Bias'], fmt='none', color='orange', alpha=0.5)
                
                # Add zero line
                ax.axhline(y=0, color='green', linestyle='-', alpha=0.7, label='Zero % Bias')
                
                ax.set_xlabel('Reference Value')
                ax.set_ylabel('Percent Bias (%)')
                # 縮短標題文字，增加間距，設置靠左對齊
                ax.set_title('Percent Bias Plot', pad=15, fontsize=12, loc='left')
                ax.legend(loc='best', fontsize=9)
                ax.grid(True, linestyle='--', alpha=0.7)
                
                # 設置較大的標記點和字體
                for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                            ax.get_xticklabels() + ax.get_yticklabels()):
                    item.set_fontsize(10)
            
            # Fitted line plot
            if show_fitted_line_plot:
                ax = axes[plot_idx]
                plot_idx += 1
                
                # Scatter plot of measurement vs reference
                ax.scatter(analysis_df['Reference'], analysis_df['Measurement'], 
                          s=80, color='green', label='Measurements')
                
                # Add perfect agreement line (y=x)
                min_val = min(analysis_df['Reference'].min(), analysis_df['Measurement'].min())
                max_val = max(analysis_df['Reference'].max(), analysis_df['Measurement'].max())
                x_range = np.linspace(min_val, max_val, 100)
                ax.plot(x_range, x_range, color='blue', linestyle='--', 
                       label='y=x')
                
                # Add fitted line (y = x + bias)
                ax.plot(x_range, x_range + (intercept + slope * x_range), color='red', linestyle='-', 
                       label=f'Fitted Line')
                
                ax.set_xlabel('Reference Value')
                ax.set_ylabel('Measurement')
                # 縮短標題文字，增加間距，設置靠左對齊
                ax.set_title('Fitted Line Plot', pad=15, fontsize=12, loc='left')
                ax.legend(loc='best', fontsize=9)
                ax.grid(True, linestyle='--', alpha=0.7)
                
                # 設置較大的標記點和字體
                for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                            ax.get_xticklabels() + ax.get_yticklabels()):
                    item.set_fontsize(10)
            
            # Residuals plot
            if show_residuals_plot:
                ax = axes[plot_idx]
                plot_idx += 1
                
                # Calculate residuals (observed bias - predicted bias)
                reference_stats['Residuals'] = reference_stats['Mean_Bias'] - reference_stats['Predicted_Bias']
                
                # Scatter plot of residuals vs reference
                ax.scatter(reference_stats['Reference'], reference_stats['Residuals'], 
                          s=80, color='red', label='Residuals')
                
                # Add zero line
                ax.axhline(y=0, color='green', linestyle='-', alpha=0.7, label='Zero Residual')
                
                ax.set_xlabel('Reference Value')
                ax.set_ylabel('Residuals')
                # 縮短標題文字，增加間距，設置靠左對齊
                ax.set_title('Residuals Plot', pad=15, fontsize=12, loc='left')
                ax.legend(loc='best', fontsize=9)
                ax.grid(True, linestyle='--', alpha=0.7)
                
                # 設置較大的標記點和字體
                for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                            ax.get_xticklabels() + ax.get_yticklabels()):
                    item.set_fontsize(10)
            
            # 自動調整子圖布局，避免重疊，增加填充距離
            fig.tight_layout(pad=4.0)
            
            # 添加更多輸出圖表的空間，確保不會被截斷
            fig.set_size_inches(n_cols*9, n_rows*7, forward=True)
            
            # 直接显示图形
            plt.show()
            
            # 不再使用图像标签
            image_tag = ""
        else:
            image_tag = ""
        
        # Update session window with report
        html_report = f"<pre>{report}</pre>{image_tag}"
        main_window.sessionWindow.setText(html_report)
        
    except Exception as e:
        QMessageBox.critical(main_window, "Error", f"Error performing linearity analysis: {str(e)}")
        import traceback
        traceback.print_exc()

def bias_study(main_window):
    """Perform measurement system bias study"""
    # Get measurement column with explicit prompt
    measurement_col = main_window.selectColumnDialog("Select Measurement Column")
    if not measurement_col:
        return
        
    # Get reference value via input dialog
    from PyQt6.QtWidgets import QInputDialog
    reference_value, ok = QInputDialog.getDouble(
        main_window, 
        "Enter Reference Value", 
        "Reference value:", 
        0.0, -1000000, 1000000, 4
    )
    if not ok:
        return
    
    # Get operator column (optional) with explicit prompt
    operator_col = main_window.selectColumnDialog("Select Operator Column (optional)")
    # Operator column is optional
    
    # Get order column (optional) with explicit prompt
    order_col = main_window.selectColumnDialog("Select Order Column (optional)")
    # Order column is optional
    
    # Create dialog
    dialog = QDialog(main_window)
    dialog.setWindowTitle("Bias Study")
    dialog.setMinimumWidth(600)
    dialog.setMinimumHeight(400)  # 增加最小高度以避免元素擠壓
    
    # 先创建主布局
    main_layout = QVBoxLayout(dialog)
    main_layout.setSpacing(10)  # 增加間距
    
    # Create tabs
    tabs = QTabWidget()
    options_tab = QWidget()
    graphs_tab = QWidget()
    
    # Options tab layout
    options_layout = QVBoxLayout(options_tab)
    options_layout.setSpacing(15)  # 增加控件間的間距
    options_layout.setContentsMargins(10, 20, 10, 10)  # 增加上邊距
    
    # Analysis settings
    analysis_group = QGroupBox("Analysis Settings")
    analysis_group.setMinimumHeight(80)  # 確保分組框有足夠的高度
    analysis_layout = QFormLayout(analysis_group)
    analysis_layout.setVerticalSpacing(10)  # 增加表單項目間的垂直間距
    analysis_layout.setContentsMargins(10, 20, 10, 10)  # 設置內邊距
    
    confidence_combo = QComboBox()
    confidence_combo.addItems(["90%", "95%", "99%"])
    confidence_combo.setCurrentText("95%")
    analysis_layout.addRow("Confidence level:", confidence_combo)
    
    include_operator_check = QCheckBox("Include operator effects")
    analysis_layout.addRow("", include_operator_check)
    
    options_layout.addWidget(analysis_group)
    
    # Tolerance information
    tolerance_group = QGroupBox("Tolerance Information")
    tolerance_group.setMinimumHeight(100)  # 確保分組框有足夠的高度
    tolerance_layout = QFormLayout(tolerance_group)
    tolerance_layout.setVerticalSpacing(10)  # 增加表單項目間的垂直間距
    tolerance_layout.setContentsMargins(10, 20, 10, 10)  # 設置內邊距
    
    tolerance_input = QLineEdit()
    tolerance_layout.addRow("Tolerance range:", tolerance_input)
    
    acceptable_bias_input = QLineEdit("5.0")
    tolerance_layout.addRow("Acceptable bias (%):", acceptable_bias_input)
    
    options_layout.addWidget(tolerance_group)
    options_layout.addStretch(1)  # 添加彈性空間
    
    # Graphs tab layout
    graphs_layout = QVBoxLayout(graphs_tab)
    graphs_layout.setSpacing(10)  # 增加間距
    graphs_layout.setContentsMargins(10, 10, 10, 10)  # 設置內邊距
    
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
    
    graphs_layout.addStretch(1)  # 添加彈性空間
    
    # Add tabs to tab widget
    tabs.addTab(options_tab, "Options")
    tabs.addTab(graphs_tab, "Graphs")
    
    # 添加标签页到主布局
    main_layout.addWidget(tabs)
    
    # Dialog buttons
    button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
    button_box.accepted.connect(dialog.accept)
    button_box.rejected.connect(dialog.reject)
    main_layout.addWidget(button_box)
    
    # Show dialog
    if not dialog.exec():
        return
    
    # Get confidence level
    conf_level_text = confidence_combo.currentText()
    conf_level = float(conf_level_text.strip('%')) / 100
    
    # Get tolerance settings
    tolerance_range = None
    try:
        if tolerance_input.text():
            tolerance_range = float(tolerance_input.text())
    except ValueError:
        QMessageBox.warning(main_window, "Warning", "Invalid tolerance range. This will be ignored.")
    
    acceptable_bias_pct = 5.0
    try:
        if acceptable_bias_input.text():
            acceptable_bias_pct = float(acceptable_bias_input.text())
    except ValueError:
        QMessageBox.warning(main_window, "Warning", "Invalid acceptable bias percentage. Using default 5%.")
    
    # Get selected graphs
    show_run_chart = run_chart_check.isChecked()
    show_histogram = histogram_check.isChecked()
    show_normal_plot = normal_plot_check.isChecked()
    show_box_plot = box_plot_check.isChecked() and operator_col is not None
    
    try:
        # Get measurements
        measurements = pd.to_numeric(main_window.data[measurement_col], errors='coerce')
        
        # Create a DataFrame for analysis
        analysis_df = pd.DataFrame({'Measurement': measurements})
        
        # Add operator column if selected
        if operator_col:
            analysis_df['Operator'] = main_window.data[operator_col]
        
        # Add order column if selected
        if order_col:
            analysis_df['Order'] = pd.to_numeric(main_window.data[order_col], errors='coerce')
            # Sort by order if available
            analysis_df = analysis_df.sort_values('Order')
        
        # Drop rows with missing values
        analysis_df = analysis_df.dropna()
        
        if len(analysis_df) < 10:
            QMessageBox.warning(main_window, "Warning", "Insufficient data for bias analysis (minimum 10 measurements required)")
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

Bias Analysis:
Bias: {bias:.4f}
Percent Bias: {percent_bias:.2f}%
T-Statistic: {t_stat:.4f}
P-Value: {p_value:.4f}
{conf_level*100:.0f}% Confidence Interval for Bias: ({ci[0]:.4f}, {ci[1]:.4f})

"""
        
        # Add interpretation
        if p_value < 0.05:
            report += "Interpretation: There is statistically significant bias (p < 0.05).\n"
        else:
            report += "Interpretation: There is no statistically significant bias (p >= 0.05).\n"
        
        # Add capability indices if tolerance is provided
        if tolerance_range:
            report += f"""
Capability Indices:
Cg (Precision to Tolerance): {capability_indices['Cg']:.4f}
Cgk (Accuracy to Tolerance): {capability_indices['Cgk']:.4f}

"""
            
            # Add interpretation of capability indices
            if capability_indices['Cg'] >= 1.33:
                report += "Precision Capability (Cg): ACCEPTABLE (>= 1.33)\n"
            elif capability_indices['Cg'] >= 1.00:
                report += "Precision Capability (Cg): MARGINALLY ACCEPTABLE (>= 1.00)\n"
            else:
                report += "Precision Capability (Cg): NOT ACCEPTABLE (< 1.00)\n"
                
            if capability_indices['Cgk'] >= 1.33:
                report += "Accuracy Capability (Cgk): ACCEPTABLE (>= 1.33)\n"
            elif capability_indices['Cgk'] >= 1.00:
                report += "Accuracy Capability (Cgk): MARGINALLY ACCEPTABLE (>= 1.00)\n"
            else:
                report += "Accuracy Capability (Cgk): NOT ACCEPTABLE (< 1.00)\n"
        
        # Generate visualizations if requested
        if show_run_chart or show_histogram or show_normal_plot or show_box_plot:
            # Create figure for plots
            num_plots = sum([show_run_chart, show_histogram, show_normal_plot, show_box_plot])
            
            # 计算行列布局
            if num_plots <= 2:
                n_rows, n_cols = num_plots, 1
            else:
                n_rows, n_cols = 2, 2
            
            # 創建圖形並增加總體高度，確保標題有足夠空間
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*9, n_rows*7))
            
            # 設置全域字型大小
            plt.rcParams.update({'font.size': 10})
            
            # 設置子圖之間的間距（增加間距）
            plt.subplots_adjust(hspace=0.5, wspace=0.5)
            
            # 将axes转为一维数组以便于索引
            axes = axes.flatten() if num_plots > 1 else [axes]
            
            # 隐藏未使用的子图
            for i in range(num_plots, len(axes)):
                axes[i].set_visible(False)
            
            plot_idx = 0
            
            # Run Chart
            if show_run_chart:
                ax = axes[plot_idx]
                plot_idx += 1
                
                # Plot measurements vs order or index
                if order_col:
                    x = analysis_df['Order']
                    xlabel = 'Order'
                else:
                    x = range(len(analysis_df))
                    xlabel = 'Measurement Index'
                
                ax.plot(x, analysis_df['Measurement'], 'o-', color='blue')
                
                # Add reference line
                ax.axhline(y=reference_value, color='red', linestyle='--', label=f'Reference')
                
                # Add mean line
                ax.axhline(y=mean, color='green', linestyle='-', label=f'Mean')
                
                ax.set_xlabel(xlabel)
                ax.set_ylabel('Measurement')
                # 縮短標題文字，增加間距，設置靠左對齊
                ax.set_title('Run Chart', pad=15, fontsize=12, loc='left')
                ax.legend(loc='best', fontsize=9)  # 這裡確保圖例總是顯示，因為有 Reference 和 Mean 線
                ax.grid(True, linestyle='--', alpha=0.7)
                
                # 設置較大的標記點和字體
                for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                            ax.get_xticklabels() + ax.get_yticklabels()):
                    item.set_fontsize(10)
                
                # Format x-axis for better readability
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            # Histogram
            if show_histogram:
                ax = axes[plot_idx]
                plot_idx += 1
                
                # Create histogram
                ax.hist(analysis_df['Measurement'], bins='auto', alpha=0.7)
                
                # Add density curve
                x = np.linspace(mean - 4*std_dev, mean + 4*std_dev, 100)
                ax.plot(x, stats.norm.pdf(x, mean, std_dev), 'r-', 
                       label=f'Normal Curve')
                
                # Add reference line
                ax.axvline(x=reference_value, color='green', linestyle='--', 
                          label=f'Reference')
                
                ax.set_xlabel('Measurement')
                ax.set_ylabel('Density')
                # 縮短標題文字，增加間距，設置靠左對齊
                ax.set_title('Histogram', pad=15, fontsize=12, loc='left')
                ax.legend(loc='best', fontsize=9)  # 這裡確保圖例總是顯示，因為有 Normal Curve 和 Reference 線
                ax.grid(True, linestyle='--', alpha=0.7)
                
                # 設置較大的標記點和字體
                for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                            ax.get_xticklabels() + ax.get_yticklabels()):
                    item.set_fontsize(10)
            
            # Normal Probability Plot
            if show_normal_plot:
                ax = axes[plot_idx]
                plot_idx += 1
                
                # Create Q-Q plot
                stats.probplot(analysis_df['Measurement'], plot=ax)
                
                # 縮短標題文字，增加間距，設置靠左對齊
                ax.set_title('Normal Probability Plot', pad=15, fontsize=12, loc='left')
                ax.grid(True, linestyle='--', alpha=0.7)
                
                # 設置較大的標記點和字體
                for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                            ax.get_xticklabels() + ax.get_yticklabels()):
                    item.set_fontsize(10)
            
            # Box Plot by Operator (if operator column provided)
            if show_box_plot and operator_col is not None:
                ax = axes[plot_idx]
                plot_idx += 1
                
                # Group data by operator
                operators = analysis_df['Operator'].unique()
                box_data = [analysis_df[analysis_df['Operator'] == op]['Measurement'] for op in operators]
                
                # Create box plot
                ax.boxplot(box_data, labels=operators)
                
                # Add reference line
                ax.axhline(y=reference_value, color='red', linestyle='--', 
                          label=f'Reference')
                
                ax.set_xlabel('Operator')
                ax.set_ylabel('Measurement')
                # 縮短標題文字，增加間距，設置靠左對齊
                ax.set_title('Box Plot by Operator', pad=15, fontsize=12, loc='left')
                ax.legend(loc='best', fontsize=9)
                ax.grid(True, axis='y', linestyle='--', alpha=0.7)
                
                # 設置較大的標記點和字體
                for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                            ax.get_xticklabels() + ax.get_yticklabels()):
                    item.set_fontsize(10)
            
            # 自動調整子圖布局，避免重疊，增加填充距離
            fig.tight_layout(pad=4.0)
            
            # 添加更多輸出圖表的空間，確保不會被截斷
            fig.set_size_inches(n_cols*9, n_rows*7, forward=True)
            
            # 直接显示图形
            plt.show()
            
            # 不再使用图像标签
            image_tag = ""
        else:
            image_tag = ""
        
        # Update session window with report
        html_report = f"<pre>{report}</pre>{image_tag}"
        main_window.sessionWindow.setText(html_report)
        
    except Exception as e:
        QMessageBox.critical(main_window, "Error", f"Error performing bias analysis: {str(e)}")
        import traceback
        traceback.print_exc()

def stability_study(main_window):
    """Perform measurement system stability study"""
    # Get datetime column with explicit prompt
    datetime_col = main_window.selectColumnDialog("Select DateTime Column")
    if not datetime_col:
        return

    # Get measurement column with explicit prompt
    measurement_col = main_window.selectColumnDialog("Select Measurement Column")
    if not measurement_col:
        return

    # Get operator column (optional) with explicit prompt
    operator_col = main_window.selectColumnDialog("Select Operator Column (optional)")
    # Operator column is optional

    # Get standard column (optional) with explicit prompt
    standard_col = main_window.selectColumnDialog("Select Standard Column (optional)")
    # Standard column is optional
    
    # 添加 order_col 變量，用於解決 'order_col' is not defined 的錯誤
    order_col = main_window.selectColumnDialog("Select Order Column (optional)")
    # Order column is optional
    
    # Create dialog
    dialog = QDialog(main_window)
    dialog.setWindowTitle("Stability Study")
    dialog.setMinimumWidth(600)
    dialog.setMinimumHeight(450)  # 增加最小高度以避免元素擠壓
    
    # 先創建主佈局
    main_layout = QVBoxLayout(dialog)
    main_layout.setSpacing(10)  # 增加間距
    
    # Create tabs
    tabs = QTabWidget()
    options_tab = QWidget()
    graphs_tab = QWidget()
    
    # Options tab layout
    options_layout = QVBoxLayout(options_tab)
    options_layout.setSpacing(15)  # 增加控件間的間距
    options_layout.setContentsMargins(10, 20, 10, 10)  # 增加上邊距
    
    # Time settings
    time_group = QGroupBox("Time Settings")
    time_group.setMinimumHeight(120)  # 確保分組框有足夠的高度
    time_layout = QFormLayout(time_group)
    time_layout.setVerticalSpacing(10)  # 增加表單項目間的垂直間距
    time_layout.setContentsMargins(10, 20, 10, 10)  # 設置內邊距
    
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
    analysis_group.setMinimumHeight(120)  # 確保分組框有足夠的高度
    analysis_layout = QFormLayout(analysis_group)
    analysis_layout.setVerticalSpacing(10)  # 增加表單項目間的垂直間距
    analysis_layout.setContentsMargins(10, 20, 10, 10)  # 設置內邊距
    
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
    options_layout.addStretch(1)  # 添加彈性空間
    
    # Graphs tab layout
    graphs_layout = QVBoxLayout(graphs_tab)
    graphs_layout.setSpacing(10)  # 增加間距
    graphs_layout.setContentsMargins(10, 10, 10, 10)  # 設置內邊距
    
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
    
    graphs_layout.addStretch(1)  # 添加彈性空間
    
    # Add tabs to tab widget
    tabs.addTab(options_tab, "Options")
    tabs.addTab(graphs_tab, "Graphs")
    
    main_layout.addWidget(tabs)
    
    # Dialog buttons
    button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
    button_box.accepted.connect(dialog.accept)
    button_box.rejected.connect(dialog.reject)
    main_layout.addWidget(button_box)
    
    # Show dialog
    if not dialog.exec():
        return
    
    # Get time settings
    time_unit = time_unit_combo.currentText().lower()
    group_by_time = group_by_time_check.isChecked()
    
    reference_value = None
    try:
        if reference_input.text():
            reference_value = float(reference_input.text())
    except ValueError:
        QMessageBox.warning(main_window, "Warning", "Invalid reference value. This will be ignored.")
    
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
            'DateTime': pd.to_datetime(main_window.data[datetime_col], errors='coerce'),
            'Measurement': pd.to_numeric(main_window.data[measurement_col], errors='coerce')
        })
        
        # Add operator column if selected
        if operator_col:
            analysis_df['Operator'] = main_window.data[operator_col]
        
        # Add standard column if selected
        if standard_col:
            analysis_df['Standard'] = main_window.data[standard_col]
            
        # Add order column if selected
        if order_col:
            analysis_df['Order'] = pd.to_numeric(main_window.data[order_col], errors='coerce')
        
        # Remove missing values
        analysis_df = analysis_df.dropna(subset=['DateTime', 'Measurement'])
        analysis_df = analysis_df.sort_values('DateTime')
        
        if len(analysis_df) < 10:
            QMessageBox.warning(main_window, "Warning", "Insufficient data for stability analysis (minimum 10 measurements required)")
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
                QMessageBox.warning(main_window, "Warning", 
                    f"Insufficient time periods for stability analysis. Found {len(period_stats)} periods, minimum 5 required.")
                return
            
            # For control charts, use period statistics
            control_chart_data = period_stats
            x_values = period_stats['TimePeriod']
            y_values = period_stats['Mean']
            
            # For I-MR Chart with grouped data, use Rbar/d2 method
            if chart_type == "I-MR Chart":
                # Calculate moving ranges between consecutive means
                ranges = []
                for i in range(1, len(y_values)):
                    ranges.append(abs(y_values.iloc[i] - y_values.iloc[i-1]))
                
                mr_mean = np.mean(ranges) if ranges else 0
                
                # Calculate control limits
                center_line = np.mean(y_values)
                upper_cl = center_line + 3 * mr_mean / 1.128  # d2 for n=2 is 1.128
                lower_cl = center_line - 3 * mr_mean / 1.128
            
            # For Xbar-R Chart with grouped data
            else:
                # Calculate average range
                r_bar = np.mean(period_stats['StdDev'] * 
                              np.sqrt(period_stats['Count']) * 
                              np.sqrt(2/np.pi))  # Approximate range from StdDev for normal data
                
                # Find average sample size
                avg_n = np.mean(period_stats['Count'])
                
                # Get d2 value based on average sample size (approximate for fractional n)
                if avg_n <= 2:
                    d2 = 1.128
                elif avg_n <= 3:
                    d2 = 1.693
                elif avg_n <= 4:
                    d2 = 2.059
                elif avg_n <= 5:
                    d2 = 2.326
                else:
                    d2 = 2.534  # n=6
                
                # Calculate control limits
                center_line = np.mean(y_values)
                upper_cl = center_line + d2 * r_bar
                lower_cl = center_line - d2 * r_bar
        
        else:
            # Use raw data points for analysis
            control_chart_data = analysis_df
            x_values = analysis_df['DateTime']
            y_values = analysis_df['Measurement']
            
            # For I-MR Chart with individual values
            if chart_type == "I-MR Chart":
                # Calculate moving ranges
                ranges = []
                for i in range(1, len(y_values)):
                    ranges.append(abs(y_values.iloc[i] - y_values.iloc[i-1]))
                
                mr_mean = np.mean(ranges) if ranges else 0
                
                # Calculate control limits
                center_line = np.mean(y_values)
                upper_cl = center_line + 3 * mr_mean / 1.128
                lower_cl = center_line - 3 * mr_mean / 1.128
            
            # For Xbar-R Chart with individual values, use moving average
            else:
                QMessageBox.warning(main_window, "Warning", 
                                   "X-bar R Chart requires grouped data. Switching to I-MR Chart.")
                chart_type = "I-MR Chart"
                
                # Calculate moving ranges
                ranges = []
                for i in range(1, len(y_values)):
                    ranges.append(abs(y_values.iloc[i] - y_values.iloc[i-1]))
                
                mr_mean = np.mean(ranges) if ranges else 0
                
                # Calculate control limits
                center_line = np.mean(y_values)
                upper_cl = center_line + 3 * mr_mean / 1.128
                lower_cl = center_line - 3 * mr_mean / 1.128
        
        # Calculate overall statistics
        overall_mean = np.mean(y_values)
        overall_std = np.std(y_values, ddof=1)
        
        # Calculate bias if reference value is provided
        bias = None
        percent_bias = None
        if reference_value is not None:
            bias = overall_mean - reference_value
            percent_bias = 100 * bias / reference_value if reference_value != 0 else float('inf')
        
        # Perform stability tests
        stability_results = {
            'Points Outside Control Limits': [],
            'Trends': [],
            'Shifts': []
        }
        
        if include_special_causes:
            # Check for points outside control limits
            for i, val in enumerate(y_values):
                if val > upper_cl or val < lower_cl:
                    stability_results['Points Outside Control Limits'].append(i)
            
            # Check for trends (7 consecutive points trending up or down)
            for i in range(6, len(y_values)):
                trend_up = True
                trend_down = True
                for j in range(6):
                    if y_values.iloc[i-j] <= y_values.iloc[i-j-1]:
                        trend_up = False
                    if y_values.iloc[i-j] >= y_values.iloc[i-j-1]:
                        trend_down = False
                if trend_up or trend_down:
                    stability_results['Trends'].append(i)
            
            # Check for shifts (8 consecutive points above or below centerline)
            for i in range(7, len(y_values)):
                above_center = True
                below_center = True
                for j in range(8):
                    if y_values.iloc[i-j] <= center_line:
                        above_center = False
                    if y_values.iloc[i-j] >= center_line:
                        below_center = False
                if above_center or below_center:
                    stability_results['Shifts'].append(i)
        
        # Generate report
        report = f"""Stability Study Results

Time Period: {time_unit.capitalize()} {analysis_df['DateTime'].min().strftime('%Y-%m-%d')} to {analysis_df['DateTime'].max().strftime('%Y-%m-%d')}

Overall Statistics:
Mean: {overall_mean:.4f}
Standard Deviation: {overall_std:.4f}
Control Chart Type: {chart_type}
"""

        if reference_value is not None:
            report += f"""
Reference Value: {reference_value:.4f}
Bias: {bias:.4f}
Percent Bias: {percent_bias:.2f}%
"""
            
        report += f"""
Control Chart Statistics:
Center Line: {center_line:.4f}
Upper Control Limit: {upper_cl:.4f}
Lower Control Limit: {lower_cl:.4f}
"""

        # Add stability assessment
        report += "\nStability Assessment:\n"
        
        # Check if any special causes were detected
        if include_special_causes:
            if stability_results['Points Outside Control Limits']:
                report += f"- {len(stability_results['Points Outside Control Limits'])} points outside control limits\n"
            if stability_results['Trends']:
                report += f"- {len(stability_results['Trends'])} trends detected\n"
            if stability_results['Shifts']:
                report += f"- {len(stability_results['Shifts'])} shifts detected\n"
            
            if (not stability_results['Points Outside Control Limits'] and 
                not stability_results['Trends'] and 
                not stability_results['Shifts']):
                report += "- No special causes detected. Measurement system appears stable.\n"
            else:
                report += "- Special causes detected. Measurement system shows signs of instability.\n"
        
        # Generate visualizations if requested
        if show_time_series or show_control_charts or show_run_chart or show_histogram:
            # Create figure for plots
            num_plots = sum([show_time_series, show_control_charts, show_run_chart, show_histogram])
            
            # Calculate layout based on number of plots
            if num_plots <= 2:
                n_rows, n_cols = num_plots, 1
            else:
                n_rows, n_cols = 2, 2
            
            # 創建圖形並增加總體高度，確保標題有足夠空間
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*9, n_rows*7))
            
            # 設置全域字型大小
            plt.rcParams.update({'font.size': 10})
            
            # 設置子圖之間的間距（增加間距）
            plt.subplots_adjust(hspace=0.5, wspace=0.5)
            
            # Convert axes to 1D array for easier indexing
            axes = axes.flatten() if num_plots > 1 else [axes]
            
            # Hide unused subplots
            for i in range(num_plots, len(axes)):
                axes[i].set_visible(False)
            
            plot_idx = 0
            
            # Time Series Plot
            if show_time_series:
                ax = axes[plot_idx]
                plot_idx += 1
                
                # Plot measurements vs time
                ax.plot(analysis_df['DateTime'], analysis_df['Measurement'], 'o-', color='blue')
                
                ax.set_xlabel('Time')
                ax.set_ylabel('Measurement')
                # 縮短標題文字，增加間距，設置靠左對齊
                ax.set_title('Time Series Plot', pad=15, fontsize=12, loc='left')
                ax.grid(True, linestyle='--', alpha=0.7)
                
                # 設置較大的標記點和字體
                for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                            ax.get_xticklabels() + ax.get_yticklabels()):
                    item.set_fontsize(10)
                
                # Format x-axis for better readability
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            # Control Chart
            if show_control_charts:
                ax = axes[plot_idx]
                plot_idx += 1
                
                # Plot measurements
                ax.plot(range(len(analysis_df)), analysis_df['Measurement'], 'bo-')
                
                # Add mean line
                ax.axhline(y=overall_mean, color='r', linestyle='-', label='Mean')
                
                # Add control limits
                ax.axhline(y=lower_cl, color='r', linestyle='--', label=f'LCL')
                ax.axhline(y=upper_cl, color='r', linestyle='--', label=f'UCL')
                
                # Highlight points outside control limits
                outside_lcl = analysis_df['Measurement'] < lower_cl
                outside_ucl = analysis_df['Measurement'] > upper_cl
                outside_control = outside_lcl | outside_ucl
                
                if outside_control.any():
                    ax.plot(np.where(outside_control)[0], 
                           analysis_df.loc[outside_control, 'Measurement'], 
                           'ro', markersize=10, label='Out of control')
                
                ax.set_xlabel('Sample')
                ax.set_ylabel('Measurement')
                # 縮短標題文字，增加間距，設置靠左對齊
                ax.set_title('Control Chart', pad=15, fontsize=12, loc='left')
                ax.legend(loc='best', fontsize=9)
                ax.grid(True, linestyle='--', alpha=0.7)
                
                # 設置較大的標記點和字體
                for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                            ax.get_xticklabels() + ax.get_yticklabels()):
                    item.set_fontsize(10)
            
            # Run Chart
            if show_run_chart:
                ax = axes[plot_idx]
                plot_idx += 1
                
                # Plot measurements vs order
                if order_col:
                    x = analysis_df['Order']
                    xlabel = 'Order'
                else:
                    x = range(len(analysis_df))
                    xlabel = 'Measurement Index'
                
                ax.plot(x, analysis_df['Measurement'], 'o-', color='blue')
                
                # Add reference line if provided
                if reference_value is not None:
                    ax.axhline(y=reference_value, color='red', linestyle='--', 
                              label=f'Reference')
                
                ax.set_xlabel(xlabel)
                ax.set_ylabel('Measurement')
                # 縮短標題文字，增加間距，設置靠左對齊
                ax.set_title('Run Chart', pad=15, fontsize=12, loc='left')
                ax.legend(loc='best', fontsize=9)  # 這裡確保圖例總是顯示，因為有 Reference 和 Mean 線
                ax.grid(True, linestyle='--', alpha=0.7)
                
                # 設置較大的標記點和字體
                for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                            ax.get_xticklabels() + ax.get_yticklabels()):
                    item.set_fontsize(10)
                
                # Format x-axis for better readability
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            # Histogram
            if show_histogram:
                ax = axes[plot_idx]
                plot_idx += 1
                
                # Create histogram
                ax.hist(analysis_df['Measurement'], bins='auto', alpha=0.7)
                
                # Add density curve if enough data points
                if len(analysis_df) >= 5:
                    x = np.linspace(min(analysis_df['Measurement']), 
                                   max(analysis_df['Measurement']), 100)
                    ax.plot(x, stats.norm.pdf(x, overall_mean, overall_std), 'r-', 
                           label=f'Normal Curve')
                
                # Add reference line if provided
                if reference_value is not None:
                    ax.axvline(x=reference_value, color='green', linestyle='--', 
                              label=f'Reference')
                
                ax.set_xlabel('Measurement')
                ax.set_ylabel('Density')
                # 縮短標題文字，增加間距，設置靠左對齊
                ax.set_title('Histogram', pad=15, fontsize=12, loc='left')
                ax.legend(loc='best', fontsize=9)  # 這裡確保圖例總是顯示，因為有 Normal Curve 和 Reference 線
                ax.grid(True, linestyle='--', alpha=0.7)
                
                # 設置較大的標記點和字體
                for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                            ax.get_xticklabels() + ax.get_yticklabels()):
                    item.set_fontsize(10)
            
            # 自動調整子圖布局，避免重疊，增加填充距離
            fig.tight_layout(pad=4.0)
            
            # 添加更多輸出圖表的空間，確保不會被截斷
            fig.set_size_inches(n_cols*9, n_rows*7, forward=True)
            
            # Display the figure
            plt.show()
            
            # 不再使用图像标签
            image_tag = ""
        else:
            image_tag = ""
        
        # Update session window with report
        html_report = f"<pre>{report}</pre>{image_tag}"
        main_window.sessionWindow.setText(html_report)
        
    except Exception as e:
        QMessageBox.critical(main_window, "Error", f"Error performing stability analysis: {str(e)}")
        import traceback
        traceback.print_exc()