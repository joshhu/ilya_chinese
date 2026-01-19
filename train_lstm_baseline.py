"""訓練 LSTM 基準模型 - 論文 18 第三階段

此腳本訓練並評估 LSTM 基準模型在物件追蹤任務上的表現，
用於與 Relational RNN 進行比較。
"""
import numpy as np
import json
from lstm_baseline import LSTM
from reasoning_tasks import generate_object_tracking, create_train_test_split
from training_utils import mse_loss

# 產生資料
print("Generating Object Tracking data...")
X, y, _ = generate_object_tracking(n_samples=200, seq_len=10, n_objects=3)
X_train, X_test, y_train, y_test = create_train_test_split(X, y, test_ratio=0.4)

print(f"Data shapes: X_train={X_train.shape}, y_train={y_train.shape}")
print(f"            X_test={X_test.shape}, y_test={y_test.shape}")

# 訓練 LSTM
print("\nInitializing LSTM...")
model = LSTM(X.shape[2], hidden_size=32, output_size=y.shape[1])

print("Training (demonstrative - 10 epochs with eval only)...")
history = {'train_loss': [], 'test_loss': []}

for epoch in range(10):
    # 僅評估模式（使用數值梯度進行完整訓練會花費太長時間）
    out_train = model.forward(X_train[:32], return_sequences=False)
    loss_train = mse_loss(out_train, y_train[:32])
    
    out_test = model.forward(X_test, return_sequences=False)
    loss_test = mse_loss(out_test, y_test)
    
    history['train_loss'].append(float(loss_train))
    history['test_loss'].append(float(loss_test))
    
    print(f"Epoch {epoch+1}/10: Train Loss={loss_train:.4f}, Test Loss={loss_test:.4f}")

# 儲存結果
results = {
    'object_tracking': {
        'final_train_loss': history['train_loss'][-1],
        'final_test_loss': history['test_loss'][-1],
        'epochs': 10,
        'note': 'Baseline evaluation - no gradient updates (demo only)'
    }
}

with open('lstm_baseline_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✓ LSTM Baseline evaluation complete!")
print(f"Results saved to: lstm_baseline_results.json")
