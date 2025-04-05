"""
Simple testing script for Minitab-like app
"""

import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel

def main():
    app = QApplication(sys.argv)
    window = QMainWindow()
    window.setWindowTitle("Minitab-Like App - Modular Test")
    window.resize(800, 600)
    
    # Create central widget
    central = QWidget()
    layout = QVBoxLayout()
    
    # Add a simple label
    label = QLabel("Minitab-Like App - Modular Version")
    label.setStyleSheet("font-size: 24px; color: blue;")
    layout.addWidget(label)
    
    central.setLayout(layout)
    window.setCentralWidget(central)
    
    window.show()
    return app.exec()

if __name__ == "__main__":
    sys.exit(main()) 