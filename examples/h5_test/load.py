import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_and_analyze_h5(file_path):
    """
    加载并分析HDF5文件的内容
    
    Args:
        file_path: HDF5文件的路径
    """
    try:
        with h5py.File(file_path, 'r') as f:
            # 打印文件结构
            print("=== HDF5文件结构 ===")
            def print_structure(name, obj):
                if isinstance(obj, h5py.Dataset):
                    print(f"数据集: {name}")
                    print(f"  形状: {obj.shape}")
                    print(f"  类型: {obj.dtype}")
                    print(f"  大小: {obj.size}")
            
            f.visititems(print_structure)
            
            # 获取所有数据集的基本统计信息
            print("\n=== 数据统计信息 ===")
            for name, dataset in f.items():
                if isinstance(dataset, h5py.Dataset):
                    if dataset.dtype.kind in ['i', 'f']:  # 只分析数值类型数据
                        data = dataset[:]
                        print(f"\n{name}的统计信息:")
                        print(f"  最小值: {np.min(data)}")
                        print(f"  最大值: {np.max(data)}")
                        print(f"  平均值: {np.mean(data)}")
                        print(f"  标准差: {np.std(data)}")
    except Exception as e:
        print(f"加载HDF5文件时发生错误: {str(e)}")

def main():
    # 假设HDF5文件在当前目录
    current_dir = Path(__file__).parent
    
    # 列出目录中所有的.h5文件
    h5_files = list(current_dir.glob("*.hdf5"))
    
    if not h5_files:
        print("当前目录未找到.h5文件！")
        return
        
    for h5_file in h5_files:
        print(f"\n分析文件: {h5_file.name}")
        print("=" * 50)
        load_and_analyze_h5(h5_file)

if __name__ == "__main__":
    main()
