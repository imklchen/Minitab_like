"""
Main application window
"""

import sys
import pandas as pd
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QTableWidget, QTableWidgetItem,
                           QMenuBar, QTextEdit, QFileDialog, QMessageBox, QAction)
from PyQt6.QtCore import Qt

from simplified_app.core.file_utils import load_data, save_data

class MinitabMainWindow(QMainWindow):
    """
    Main application window - Simplified version
    """
    def __init__(self, app_controller):
        super().__init__()
        self.app_controller = app_controller
        self.setWindowTitle("Minitab-Like Tool (Simplified)")
        self.resize(900, 600)
        self.initUI()
    
    def initUI(self):
        """Initialize the user interface"""
        # Create central widget with table and session window
        self.table = QTableWidget(50, 10)
        self.table.setHorizontalHeaderLabels([f"C{i+1}" for i in range(10)])
        
        self.sessionWindow = QTextEdit()
        self.sessionWindow.setReadOnly(True)
        
        layout = QVBoxLayout()
        layout.addWidget(self.table)
        layout.addWidget(self.sessionWindow)
        
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)
        
        # Create simplified menu bar
        self.createMenuBar()
    
    def createMenuBar(self):
        """Create a simplified application menu bar"""
        menuBar = QMenuBar()
        self.setMenuBar(menuBar)
        
        # Create main menus - just File menu for now
        fileMenu = menuBar.addMenu("File")
        
        # File Menu
        fileMenu.addAction(self.makeAction("Open", self.openFile))
        fileMenu.addAction(self.makeAction("Save", self.saveFile))
        fileMenu.addAction(self.makeAction("Save As", self.saveFileAs))
        fileMenu.addSeparator()
        fileMenu.addAction(self.makeAction("Clear Table Data", self.clearTableData))
        fileMenu.addSeparator()
        fileMenu.addAction(self.makeAction("Exit", self.close))
    
    def makeAction(self, name, func):
        """Create a QAction for menus"""
        action = QAction(name, self)
        action.triggered.connect(func)
        return action
    
    def loadDataFromTable(self):
        """Load data from the table into a DataFrame"""
        rows = self.table.rowCount()
        cols = self.table.columnCount()
        
        # Get headers
        headers = [self.table.horizontalHeaderItem(i).text() for i in range(cols)]
        
        # Create empty DataFrame
        data = pd.DataFrame(columns=headers)
        
        # Fill data
        for i in range(rows):
            row_data = []
            has_data = False
            
            for j in range(cols):
                item = self.table.item(i, j)
                if item and item.text().strip():
                    row_data.append(item.text())
                    has_data = True
                else:
                    row_data.append(None)
            
            if has_data:
                data.loc[len(data)] = row_data
        
        # Convert numeric columns
        for col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='ignore')
        
        return data
    
    def update_table(self):
        """Update the table with current data"""
        data = self.app_controller.get_data()
        
        # Clear table
        self.table.clearContents()
        
        # Set dimensions
        if not data.empty:
            nrows, ncols = data.shape
            self.table.setRowCount(max(nrows, 50))  # Minimum 50 rows
            self.table.setColumnCount(ncols)
            
            # Set headers
            self.table.setHorizontalHeaderLabels(data.columns)
            
            # Fill data
            for i in range(nrows):
                for j in range(ncols):
                    value = data.iloc[i, j]
                    if pd.isna(value):
                        value = ""
                    self.table.setItem(i, j, QTableWidgetItem(str(value)))
    
    def update_title(self):
        """Update window title with current file"""
        file_path = self.app_controller.get_current_file()
        if file_path:
            self.setWindowTitle(f"Minitab-Like Tool (Simplified) - {file_path}")
        else:
            self.setWindowTitle("Minitab-Like Tool (Simplified)")
    
    def log_message(self, message):
        """Add message to session window"""
        self.sessionWindow.append(message)
    
    # File operations
    def openFile(self):
        """Open a file dialog and load data"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Data File", "", 
            "CSV Files (*.csv);;Excel Files (*.xlsx *.xls);;Text Files (*.txt);;All Files (*)"
        )
        
        if file_path:
            try:
                data = load_data(file_path)
                self.app_controller.set_data(data)
                self.app_controller.set_current_file(file_path)
                self.log_message(f"Loaded data from {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", str(e))
    
    def saveFile(self):
        """Save data to current file or open Save As dialog"""
        file_path = self.app_controller.get_current_file()
        if not file_path:
            return self.saveFileAs()
        
        try:
            data = self.loadDataFromTable()
            save_data(data, file_path)
            self.log_message(f"Saved data to {file_path}")
            return True
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
            return False
    
    def saveFileAs(self):
        """Open a Save As dialog and save data"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Data File", "", 
            "CSV Files (*.csv);;Excel Files (*.xlsx *.xls);;Text Files (*.txt)"
        )
        
        if file_path:
            try:
                data = self.loadDataFromTable()
                save_data(data, file_path)
                self.app_controller.set_current_file(file_path)
                self.log_message(f"Saved data to {file_path}")
                return True
            except Exception as e:
                QMessageBox.critical(self, "Error", str(e))
                return False
        
        return False
    
    def clearTableData(self):
        """Clear all data from the table"""
        reply = QMessageBox.question(
            self, "Clear Data", 
            "Are you sure you want to clear all data?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, 
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self.app_controller.set_data(pd.DataFrame(columns=[f"C{i+1}" for i in range(10)]))
            self.app_controller.set_current_file(None)
            self.log_message("Cleared all data")
