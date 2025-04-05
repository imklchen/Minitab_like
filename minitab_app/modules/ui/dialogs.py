from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
                           QCheckBox, QPushButton, QDialogButtonBox, QFormLayout, 
                           QSpinBox, QDoubleSpinBox, QLineEdit, QListWidget,
                           QRadioButton, QButtonGroup, QGroupBox, QTabWidget,
                           QMessageBox, QTableWidget, QTableWidgetItem)
from PyQt6.QtCore import Qt

class SelectColumnDialog(QDialog):
    """Dialog for selecting columns from the data"""
    def __init__(self, parent=None, columns=None, prompt=None, multi_select=False):
        super().__init__(parent)
        self.setWindowTitle("Select Column(s)")
        self.columns = columns or []
        self.multi_select = multi_select
        self.selected_columns = []
        
        layout = QVBoxLayout()
        
        if prompt:
            label = QLabel(prompt)
            layout.addWidget(label)
        
        if multi_select:
            self.setupMultiSelect(layout)
        else:
            self.setupSingleSelect(layout)
        
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
        
        self.setLayout(layout)
    
    def setupSingleSelect(self, layout):
        """Setup UI for single column selection"""
        self.combo_box = QComboBox()
        for col in self.columns:
            self.combo_box.addItem(col)
        
        layout.addWidget(self.combo_box)
    
    def setupMultiSelect(self, layout):
        """Setup UI for multiple column selection"""
        self.check_boxes = []
        for col in self.columns:
            check_box = QCheckBox(col)
            self.check_boxes.append(check_box)
            layout.addWidget(check_box)
    
    def getSelectedColumns(self):
        """Get the selected column(s)"""
        if self.multi_select:
            return [cb.text() for cb in self.check_boxes if cb.isChecked()]
        else:
            return self.combo_box.currentText() if self.combo_box.currentText() else None

class InputDialog(QDialog):
    """Dialog for general input"""
    def __init__(self, parent=None, title="Input", fields=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.fields = fields or []
        self.field_widgets = {}
        
        layout = QFormLayout()
        
        for field in self.fields:
            field_name = field['name']
            field_type = field.get('type', 'text')
            field_label = field.get('label', field_name)
            field_default = field.get('default', '')
            
            if field_type == 'text':
                widget = QLineEdit()
                widget.setText(str(field_default))
            elif field_type == 'int':
                widget = QSpinBox()
                widget.setRange(field.get('min', -100000), field.get('max', 100000))
                widget.setValue(int(field_default) if field_default else 0)
            elif field_type == 'double':
                widget = QDoubleSpinBox()
                widget.setRange(field.get('min', -100000.0), field.get('max', 100000.0))
                widget.setDecimals(field.get('decimals', 2))
                widget.setValue(float(field_default) if field_default else 0.0)
            elif field_type == 'combo':
                widget = QComboBox()
                for option in field.get('options', []):
                    widget.addItem(option)
                if field_default and field_default in field.get('options', []):
                    widget.setCurrentText(field_default)
            elif field_type == 'checkbox':
                widget = QCheckBox()
                widget.setChecked(bool(field_default))
            else:
                widget = QLineEdit()
                widget.setText(str(field_default))
            
            layout.addRow(field_label, widget)
            self.field_widgets[field_name] = widget
        
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addRow(button_box)
        
        self.setLayout(layout)
    
    def getValues(self):
        """Get values from all fields"""
        result = {}
        
        for field in self.fields:
            field_name = field['name']
            field_type = field.get('type', 'text')
            widget = self.field_widgets[field_name]
            
            if field_type == 'text':
                result[field_name] = widget.text()
            elif field_type in ['int', 'double']:
                result[field_name] = widget.value()
            elif field_type == 'combo':
                result[field_name] = widget.currentText()
            elif field_type == 'checkbox':
                result[field_name] = widget.isChecked()
            else:
                result[field_name] = widget.text()
        
        return result

class OptionDialog(QDialog):
    """Dialog for selecting from multiple options"""
    def __init__(self, parent=None, title="Select Option", options=None, prompt=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.options = options or []
        self.selected_option = None
        
        layout = QVBoxLayout()
        
        if prompt:
            label = QLabel(prompt)
            layout.addWidget(label)
        
        self.radio_group = QButtonGroup(self)
        for i, option in enumerate(self.options):
            radio = QRadioButton(option)
            self.radio_group.addButton(radio, i)
            layout.addWidget(radio)
        
        # Select first option by default
        if self.options:
            self.radio_group.button(0).setChecked(True)
        
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
        
        self.setLayout(layout)
    
    def getSelectedOption(self):
        """Get the selected option"""
        if self.radio_group.checkedId() >= 0:
            return self.options[self.radio_group.checkedId()]
        return None
