"""
最小测试脚本 - 不使用 PyQt6
"""

import pandas as pd
import os

def main():
    """最小测试程序主函数"""
    print("===== Minitab-Like App - Minimal Test =====")
    
    # 创建一些示例数据
    data = pd.DataFrame({
        'A': [1, 2, 3, 4, 5],
        'B': [10, 20, 30, 40, 50],
        'C': [100, 200, 300, 400, 500]
    })
    
    # 显示数据
    print("\n数据示例:")
    print(data)
    
    # 保存到文件
    file_path = "test_data.csv"
    try:
        data.to_csv(file_path, index=False)
        print(f"\n数据已保存到 {file_path}")
    except Exception as e:
        print(f"保存数据时出错: {e}")
    
    # 从文件加载数据
    try:
        if os.path.exists(file_path):
            loaded_data = pd.read_csv(file_path)
            print("\n从文件加载的数据:")
            print(loaded_data)
    except Exception as e:
        print(f"加载数据时出错: {e}")
    
    print("\n===== 测试完成 =====")

if __name__ == "__main__":
    main() 