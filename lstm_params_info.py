#!/usr/bin/env python3
"""LSTM 基準的快速參數計數"""
from lstm_baseline import LSTM
import numpy as np

np.random.seed(42)
lstm = LSTM(32, 64, 16)
params = lstm.get_params()
total = sum(p.size for p in params.values())

print("LSTM Parameter Count")
print("="*50)
print(f"Configuration: input=32, hidden=64, output=16")
print("="*50)
print(f"\nTotal parameters: {total:,}\n")
print("Parameter breakdown:")
for name, param in sorted(params.items()):
    print(f"  {name:8s}: {str(param.shape):20s} = {param.size:6,} params")
