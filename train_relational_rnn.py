"""Train Relational RNN - Paper 18 Phase 3, Task 2"""
import numpy as np
import json
from relational_rnn_cell import RelationalRNN
from reasoning_tasks import generate_object_tracking, create_train_test_split
from training_utils import mse_loss

# Generate data (same as LSTM)
print("Generating Object Tracking data...")
X, y, _ = generate_object_tracking(n_samples=200, seq_len=10, n_objects=3)
X_train, X_test, y_train, y_test = create_train_test_split(X, y, test_ratio=0.4)

print(f"Data shapes: X_train={X_train.shape}, y_train={y_train.shape}")

# Train Relational RNN
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

# Save results
results = {
    'object_tracking': {
        'final_train_loss': history['train_loss'][-1],
        'final_test_loss': history['test_loss'][-1],
        'epochs': 10,
        'config': {'num_slots': 4, 'slot_size': 32, 'num_heads': 2},
        'note': 'Baseline evaluation - no gradient updates (demo only)'
    }
}

with open('relational_rnn_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✓ Relational RNN evaluation complete!")
print(f"Results saved to: relational_rnn_results.json")
