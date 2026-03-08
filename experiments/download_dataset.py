"""
下载MNIST到F盘
"""

import numpy as np
from sklearn.datasets import load_digits, fetch_openml
import os

# F盘路径
F_PATH = "F:/datasets/"

# 创建目录
os.makedirs(F_PATH, exist_ok=True)

print("="*60)
print("Downloading MNIST to F:/datasets/")
print("="*60)

# 1. sklearn digits (小数据集)
print("\n1. Downloading sklearn digits...")
digits = load_digits()
np.savez(F_PATH + "digits.npz", 
         data=digits.data, 
         target=digits.target)
print(f"   Saved: {F_PATH}digits.npz")
print(f"   Shape: {digits.data.shape}")

# 2. 下载MNIST
print("\n2. Downloading MNIST (may take time)...")
try:
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    np.savez(F_PATH + "mnist.npz",
             data=mnist.data,
             target=mnist.target)
    print(f"   Saved: {F_PATH}mnist.npz")
    print(f"   Shape: {mnist.data.shape}")
except Exception as e:
    print(f"   Error: {e}")

# 3. CIFAR-10
print("\n3. Downloading CIFAR-10...")
try:
    # 使用openml
    cifar = fetch_openml('CIFAR_10', version=1, as_frame=False)
    np.savez(F_PATH + "cifar10.npz",
             data=cifar.data,
             target=cifar.target)
    print(f"   Saved: {F_PATH}cifar10.npz")
except Exception as e:
    print(f"   Error: {e}")

print("\n" + "="*60)
print("Download Complete!")
print("="*60)
print(f"\nFiles in {F_PATH}:")
for f in os.listdir(F_PATH):
    size = os.path.getsize(F_PATH + f) / 1024 / 1024
    print(f"  {f}: {size:.1f} MB")
