# Minitab-Like 应用程序重构指南

## 重构前的准备工作

1. **创建代码备份**：
   ```powershell
   Copy-Item minitab_like_app.py minitab_like_app.py.bak
   ```

2. **安装必要的依赖**：
   确保所有依赖都已安装，可以通过以下命令检查：
   ```powershell
   pip install -r requirements.txt
   ```

3. **准备测试数据**：
   确保有充足的测试数据可用于验证功能。

## 第一步：创建基本项目结构

1. **创建模块目录结构**：
   ```powershell
   mkdir -p minitab_app/{core,modules/{stats,quality,sixsigma,calc,ui}}
   ```

2. **创建 `__init__.py` 文件**：
   ```powershell
   New-Item -ItemType File -Path minitab_app/__init__.py
   New-Item -ItemType File -Path minitab_app/core/__init__.py
   New-Item -ItemType File -Path minitab_app/modules/__init__.py
   New-Item -ItemType File -Path minitab_app/modules/stats/__init__.py
   New-Item -ItemType File -Path minitab_app/modules/quality/__init__.py
   New-Item -ItemType File -Path minitab_app/modules/sixsigma/__init__.py
   New-Item -ItemType File -Path minitab_app/modules/calc/__init__.py
   New-Item -ItemType File -Path minitab_app/modules/ui/__init__.py
   ```

## 第二步：创建核心应用组件

1. **创建文件工具模块** (`minitab_app/core/file_utils.py`):
   ```python
   import pandas as pd
   import os
   
   def load_data(file_path):
       """加载数据文件"""
       try:
           _, ext = os.path.splitext(file_path)
           
           if ext.lower() == '.csv':
               return pd.read_csv(file_path)
           elif ext.lower() in ['.xls', '.xlsx']:
               return pd.read_excel(file_path)
           elif ext.lower() == '.txt':
               return pd.read_csv(file_path, sep=None, engine='python')
           else:
               raise ValueError(f"不支持的文件格式: {ext}")
       except Exception as e:
           raise Exception(f"加载文件错误: {e}")
   
   def save_data(data, file_path):
       """保存数据到文件"""
       try:
           _, ext = os.path.splitext(file_path)
           
           if ext.lower() == '.csv':
               data.to_csv(file_path, index=False)
           elif ext.lower() in ['.xls', '.xlsx']:
               data.to_excel(file_path, index=False)
           elif ext.lower() == '.txt':
               data.to_csv(file_path, sep='\t', index=False)
           else:
               raise ValueError(f"不支持的文件格式: {ext}")
       except Exception as e:
           raise Exception(f"保存文件错误: {e}")
   ```

## 第三步：逐个模块重构

### 步骤 3.1：重构基本统计模块

1. **提取描述性统计功能** (`minitab_app/modules/stats/basic_stats.py`):
   ```python
   import numpy as np
   import pandas as pd
   from scipy import stats
   
   def descriptive_stats(ui_context, data, columns=None):
       """计算描述性统计"""
       if columns is None or len(columns) == 0:
           return None
           
       # 计算统计量
       desc_stats = data[columns].describe()
       
       # 添加偏度和峰度
       desc_stats.loc['skew'] = data[columns].skew()
       desc_stats.loc['kurtosis'] = data[columns].kurtosis()
       
       return desc_stats
   ```

2. **在主应用中添加引用**:
   ```python
   # 仅修改这一个函数，保持其余代码不变
   def calculateDescriptiveStats(self):
       # 加载数据
       self.loadDataFromTable()  # 保持原有的数据加载
       
       # 获取选择的列（保持与原有逻辑相同）
       numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
       if len(numeric_cols) == 0:
           QMessageBox.warning(self, "警告", "没有数值型列")
           return
           
       # 显示列选择对话框（保持原有代码）
       # ...原有的对话框代码...
       
       # 调用模块中的函数处理
       from minitab_app.modules.stats.basic_stats import descriptive_stats
       result = descriptive_stats(self, self.data, selected_columns)
       
       # 显示结果（保持原有代码）
       # ...原有的结果显示代码...
   ```

### 步骤 3.2：重构质量控制模块

1. **提取控制图功能** (`minitab_app/modules/quality/control_charts.py`):
   ```python
   import numpy as np
   import pandas as pd
   import matplotlib.pyplot as plt
   
   def calculate_control_limits(data, n=5):
       """计算控制限"""
       # 将原有的函数移至此处
       # ...控制限计算逻辑...
       return limits_dict
   
   def xbar_r_chart(ui_context, data, columns, subgroup_size=5):
       """创建 X-bar R 控制图"""
       # 实现控制图逻辑
       # ...控制图实现...
       
       # 返回结果或直接显示图表
       return results
   ```

2. **在主应用中添加引用**：
   ```python
   def xbarRChart(self):
       # 加载数据
       self.loadDataFromTable()
       
       # 原有的对话框和数据准备代码
       # ...
       
       # 调用模块中的函数
       from minitab_app.modules.quality.control_charts import xbar_r_chart
       xbar_r_chart(self, self.data, selected_columns, subgroup_size)
   ```

## 第四步：迁移常量和工具函数

1. **创建常量模块** (`minitab_app/core/constants.py`):
   ```python
   # 控制图常量
   CONTROL_CHART_CONSTANTS = {
       2: {'A2': 1.880, 'D3': 0, 'D4': 3.267},
       3: {'A2': 1.023, 'D3': 0, 'D4': 2.575},
       4: {'A2': 0.729, 'D3': 0, 'D4': 2.282},
       5: {'A2': 0.577, 'D3': 0, 'D4': 2.115},
       # ...其他常量...
   }
   
   # 其他常量
   # ...
   ```

2. **创建实用工具模块** (`minitab_app/core/utils.py`):
   ```python
   import numpy as np
   
   def is_normal_distributed(data, alpha=0.05):
       """检查数据是否服从正态分布"""
       from scipy import stats
       _, p_value = stats.normaltest(data)
       return p_value > alpha
   
   # 其他通用函数
   # ...
   ```

## 第五步：完成重构与测试

1. **创建主入口** (`minitab_app/main.py`):
   ```python
   import sys
   from minitab_app.core.app import MinitabApp
   
   def main():
       app = MinitabApp()
       return app.run()
   
   if __name__ == "__main__":
       sys.exit(main())
   ```

2. **创建启动脚本** (`run_minitab.py`):
   ```python
   import sys
   from minitab_app.main import main
   
   if __name__ == "__main__":
       sys.exit(main())
   ```

3. **验证重构**:
   - 运行 `python run_minitab.py`
   - 测试每个重构的功能
   - 与原始应用进行比较

## 后续步骤建议

1. **增量扩展重构**：
   - 一次只对一个功能模块进行重构
   - 每次重构后进行测试
   - 逐步替换所有功能

2. **改进 UI 结构**：
   - 将 UI 逻辑与业务逻辑分离
   - 使用 MVC 或 MVVM 模式重构 UI

3. **添加单元测试**：
   - 为每个模块添加单元测试
   - 考虑使用 `pytest` 框架

## 重构注意事项

1. **保持向后兼容**：
   - 确保重构后的应用能处理原有格式的数据文件
   - 保持用户界面和用户体验的一致性

2. **维护版本控制**：
   - 使用 Git 等版本控制系统
   - 频繁提交小型更改
   - 为每个完整功能创建分支

3. **处理依赖关系**：
   - 识别并解决模块间的依赖关系
   - 避免循环依赖

4. **文档同步更新**：
   - 更新注释和文档字符串
   - 创建模块 API 文档

通过遵循这个指南，您可以逐步将单体应用重构为模块化结构，同时保持应用程序的稳定性和功能完整性。 