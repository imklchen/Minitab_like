# Minitab-like 應用程序

這是一個類似 Minitab 的統計分析應用程序，提供各種統計工具和質量控制功能。應用程序使用 Python 和 PyQt5 開發，支持數據分析、圖表生成和報告導出。

## 功能

- **基本統計分析**：描述性統計、假設檢驗、相關性分析等
- **品質控制工具**：控制圖、工藝能力分析、測量系統分析等
- **實驗設計 (DOE)**：全因子、部分因子和響應面設計
- **六標準差工具**：FMEA、魚骨圖、失效模式分析
- **數據視覺化**：直方圖、散點圖、箱線圖等多種圖表類型
- **數據匯入/匯出**：支持 CSV 和 Excel 格式

## 安裝

1. 確保已安裝 Python 3.7 或更高版本
2. 克隆或下載此倉庫
3. 安裝所需的依賴項：

```bash
pip install -r requirements.txt
```

## 使用方法

### 運行模塊化版本（推薦）

執行以下命令啟動重構後的模塊化版本：

```bash
python run_minitab_app.py
```

或者，您也可以直接運行重構目錄中的腳本：

```bash
cd refactored
python run_minitab.py
```

### 樣本數據

樣本數據文件位於 `sample_data/` 目錄中，可用於測試應用程序的各種功能。

## 項目結構

```
minitab_like/
├── refactored/           # 重構後的模塊化代碼
│   ├── gui/              # 圖形用戶界面模塊
│   ├── quality/          # 質量控制工具
│   ├── stats/            # 統計分析模塊
│   └── run_minitab.py    # 啟動腳本
├── sample_data/          # 樣本數據文件
├── requirements.txt      # 依賴項目列表
├── run_minitab_app.py    # 主啟動腳本
└── README.md             # 說明文檔
```

## 文檔

- `test_guide.md` - 測試指南，詳細說明了如何測試各個功能
- `IMPLEMENTATION_STATUS.md` - 實現狀態，列出了已完成和待開發的功能
- `SESSION_MANAGEMENT_GUIDE.md` - 會話管理指南，說明了如何管理數據會話

## 開發

請參閱 `development_tasks.md` 了解開發任務和計劃。

## Features

### Menu Structure

1. **File**
   - Open (supports CSV and Excel files)
   - Save (exports to CSV or Excel)
   - Exit

2. **Stat**
   - **Basic Statistics**
     - Descriptive Statistics
     - Correlation Analysis
     - Probability Analysis
   
   - **Advanced Statistics**
     - Hypothesis Testing
       - One-Sample t-Test
       - Two-Sample t-Test
       - Paired t-Test
     - ANOVA
       - One-Way ANOVA
       - Two-Way ANOVA
     - Regression Analysis
       - Simple Linear Regression
       - Multiple Linear Regression
     - Chi-Square Tests
       - Goodness of Fit
       - Test of Independence
       - Test of Homogeneity
   
   - **Design of Experiments (DOE)**
     - Create DOE
       - 2-level Factorial
       - Fractional Factorial
       - Response Surface
     - Analyze DOE

3. **Quality**
   - **Control Charts**
     - X-bar R Chart
     - Individual Chart
     - Moving Range Chart
   
   - **Capability Analysis**
     - Process Capability
     - Measurement System Analysis
       - Gage R&R Study
       - Linearity Study
       - Bias Study
       - Stability Study

4. **Six Sigma**
   - **DMAIC Tools**
     - Pareto Chart
     - Fishbone Diagram
     - FMEA Template
   
   - **Six Sigma Metrics**
     - DPMO Calculator
     - Sigma Level Calculator
     - Process Yield Analysis

5. **Calc**
   - **Random Data**
     - Normal
     - Binomial
     - Uniform
   
   - **Probability Distributions**
     - Poisson

## Requirements

- Python 3.x
- Required packages:
  - PyQt6
  - pandas
  - numpy
  - scipy
  - matplotlib
  - seaborn
  - statsmodels

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/Minitab_like.git
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

Run the application:
```bash
python minitab_like_app_v6.py
```

## Features Description

### Statistical Analysis
- Comprehensive descriptive statistics
- Various hypothesis tests
- ANOVA (One-way and Two-way)
- Regression analysis
- Chi-Square analysis

### Quality Control
- Various control charts
- Process capability analysis
- Measurement system analysis

### Design of Experiments
- 2-level factorial designs
- Fractional factorial designs
- Response surface designs

### Six Sigma Tools
- DMAIC tools implementation
- Process performance metrics
- Quality improvement templates

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with PyQt6 for the graphical interface
- Statistical computations powered by SciPy and NumPy
- Visualization capabilities provided by Matplotlib and Seaborn 