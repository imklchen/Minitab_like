"""
DMAIC tools module
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, 
                           QLineEdit, QTableWidget, QTableWidgetItem, 
                           QPushButton, QLabel, QComboBox, QDialogButtonBox,
                           QMessageBox, QSpinBox, QInputDialog, QGroupBox, 
                           QCheckBox, QFileDialog)
from PyQt6.QtCore import Qt


def create_pareto_chart(categories, values):
    """Create a Pareto chart"""
    df = pd.DataFrame({'categories': categories, 'values': values})
    df = df.sort_values('values', ascending=False)
    
    cumulative_percentage = np.cumsum(df['values']) / sum(df['values']) * 100
    
    fig, ax1 = plt.subplots()
    
    ax1.bar(df['categories'], df['values'])
    ax1.set_xlabel('Categories')
    ax1.set_ylabel('Frequency')
    
    ax2 = ax1.twinx()
    ax2.plot(df['categories'], cumulative_percentage, 'r-')
    ax2.set_ylabel('Cumulative Percentage')
    
    plt.title('Pareto Chart')
    return fig


def pareto_chart(main_window):
    """Create a Pareto chart"""
    try:
        # Load data from the table
        main_window.load_data_from_table()
        
        if main_window.data.empty:
            QMessageBox.warning(main_window, "Warning", "No data available for analysis")
            return
            
        # Create dialog
        dialog = QDialog(main_window)
        dialog.setWindowTitle("Pareto Chart")
        layout = QVBoxLayout()
        
        # Column selection
        col_group = QGroupBox("Column Selection")
        col_layout = QFormLayout()
        
        cat_combo = QComboBox()
        cat_combo.addItems(main_window.data.columns)
        col_layout.addRow("Categories:", cat_combo)
        
        freq_combo = QComboBox()
        freq_combo.addItems(main_window.data.columns)
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
                'Category': main_window.data[cat_col],
                'Count': pd.to_numeric(main_window.data[freq_col], errors='coerce')
            }).dropna()
            
            df = df.sort_values('Count', ascending=False)
            
            # Handle "Bars to Display" option
            bars_to_display = bars_input.text().strip()
            if bars_to_display.lower() != "all":
                try:
                    n_bars = int(bars_to_display)
                    if n_bars > 0 and n_bars < len(df):
                        df = df.head(n_bars)
                except ValueError:
                    pass  # If invalid number, use all bars
            
            total = df['Count'].sum()
            df['Percent'] = (df['Count'] / total * 100).round(1)
            df['Cumulative'] = df['Percent'].cumsum().round(1)
            
            # Display results
            main_window.sessionWindow.setText("Pareto Analysis Results")
            main_window.sessionWindow.append("----------------------")
            main_window.sessionWindow.append(f"Total Defects: {int(total)}\n")
            
            # Format and display table
            main_window.sessionWindow.append(f"{'Category':<15} {'Count':>7} {'Percent':>9} {'Cumulative':>12}")
            for _, row in df.iterrows():
                main_window.sessionWindow.append(
                    f"{str(row['Category']):<15} {int(row['Count']):>7} {row['Percent']:>8.1f}% {row['Cumulative']:>11.1f}%"
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
        QMessageBox.critical(main_window, "Error", f"Error creating Pareto chart: {str(e)}")
        import traceback
        traceback.print_exc()


def fishbone_diagram(main_window):
    """Create and edit fishbone diagram"""
    try:
        # Create dialog for fishbone diagram
        dialog = QDialog(main_window)
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

            main_window.sessionWindow.setText(report)

        create_button.clicked.connect(create_diagram)
        dialog.exec()

    except Exception as e:
        QMessageBox.warning(main_window, "Error", 
            f"An error occurred while creating the fishbone diagram:\n{str(e)}")
        import traceback
        traceback.print_exc()


def fmea_template(main_window):
    """Create and manage FMEA template"""
    try:
        # Create dialog for FMEA
        dialog = QDialog(main_window)
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
        
        import_button = QPushButton("Import FMEA")
        add_row_button = QPushButton("Add Row")
        calculate_button = QPushButton("Calculate RPN")
        save_button = QPushButton("Save FMEA")
        export_button = QPushButton("Export Report")
        save_report_button = QPushButton("Save Report")
        
        button_layout.addWidget(import_button)
        button_layout.addWidget(add_row_button)
        button_layout.addWidget(calculate_button)
        button_layout.addWidget(save_button)
        button_layout.addWidget(export_button)
        button_layout.addWidget(save_report_button)
        
        layout.addLayout(button_layout)

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
            main_window.sessionWindow.clear()  # Clear previous content
            main_window.sessionWindow.setText(report)
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
        import_button.clicked.connect(import_fmea)
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

        dialog.setLayout(layout)
        dialog.exec()

    except Exception as e:
        QMessageBox.warning(main_window, "Error", 
            f"An error occurred while creating the FMEA template:\n{str(e)}")
        import traceback
        traceback.print_exc()