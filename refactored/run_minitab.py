#!/usr/bin/env python3
"""
Minitab-like 應用程序啟動腳本
此腳本負責初始化並啟動已重構的模組化版本的 Minitab-like 應用程序
"""

import sys
import os
import logging
from pathlib import Path

# 設置日誌
logging.basicConfig(
    filename='minitab_app.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 添加必要的路徑
current_dir = Path(__file__).parent.absolute()
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

try:
    # 從 gui 模塊導入應用程序
    from gui.main_window import MinitabApp
    import PyQt5.QtWidgets as QtWidgets
    
    def main():
        """主函數，負責啟動應用程序"""
        logger.info("啟動 Minitab-like 應用程序")
        app = QtWidgets.QApplication(sys.argv)
        window = MinitabApp()
        window.show()
        sys.exit(app.exec_())

    if __name__ == "__main__":
        main()
        
except Exception as e:
    logger.error(f"啟動應用程序時發生錯誤: {str(e)}", exc_info=True)
    print(f"啟動應用程序時發生錯誤: {str(e)}")
    sys.exit(1) 