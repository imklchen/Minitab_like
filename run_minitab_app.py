#!/usr/bin/env python3
"""
Minitab-like 應用程序主入口點
此腳本是應用程序的主要入口點，通過導入和執行重構後的模組化版本啟動應用程序
"""

import sys
import os
from pathlib import Path

# 確保所有必要的包和模塊可以被找到
script_dir = Path(__file__).parent.absolute()
refactored_dir = script_dir / "refactored"

if str(refactored_dir) not in sys.path:
    sys.path.append(str(refactored_dir))

try:
    # 導入並執行重構後的應用程序
    from refactored.run_minitab import main
    
    if __name__ == "__main__":
        main()
except Exception as e:
    print(f"啟動應用程序時發生錯誤: {str(e)}")
    sys.exit(1) 