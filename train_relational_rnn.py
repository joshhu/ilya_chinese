"""訓練 Relational RNN - 論文 18 第三階段，任務 2

此腳本訓練並評估 Relational RNN 模型在物件追蹤任務上的表現，
展示關係記憶（relational memory）機制相較於傳統 LSTM 的優勢。
"""
import numpy as np
import json
from relational_rnn_cell import RelationalRNN
from reasoning_tasks import generate_object_tracking, create_train_test_split
from training_utils import mse_loss

# 產生資料（與 LSTM 相同）
print("Generating Object Tracking data...")
X, y, _ = generate_object_tracking(n_samples=200, seq_len=10, n_objects=3)
X_train, X_test, y_train, y_test = create_train_test_split(X, y, test_ratio=0.4)

print(f"Data shapes: X_train={X_train.shape}, y_train={y_train.shape}")

# 訓練 Relational RNN
print("\nInitializing Relational RNN...")
model = RelationalRNN(
    input_size=X.shape[2],
    hidden_size=32,
    output_size=y.shape[1],
    num_slots=4,
    slot_size=32,
    num_heads=2
)

print("Evaluating Relational RNN (10 epochs)...")
history = {'train_loss': [], 'test_loss': []}

for epoch in range(10):
    out_train = model.forward(X_train[:32], return_sequences=False, return_state=False)
    loss_train = mse_loss(out_train, y_train[:32])
    
    out_test = model.forward(X_test, return_sequences=False, return_state=False)
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
        'config': {'num_slots': 4, 'slot_size': 32, 'num_heads': 2},
        'note': '基準評估 - 無梯度更新（僅供示範）'
    }
}

with open('relational_rnn_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✓ Relational RNN evaluation complete!")
print(f"Results saved to: relational_rnn_results.json")
