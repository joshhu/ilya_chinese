# Training Utilities - Quick Reference

## Installation
```python
# No installation needed - pure NumPy
from training_utils import *
```

## Common Workflows

### Basic Classification Training
```python
from lstm_baseline import LSTM
from training_utils import train_model, evaluate

model = LSTM(input_size=10, hidden_size=32, output_size=3)

history = train_model(
    model,
    train_data=(X_train, y_train),
    val_data=(X_val, y_val),
    epochs=50,
    batch_size=32,
    learning_rate=0.01,
    task='classification'
)

test_loss, test_acc = evaluate(model, X_test, y_test)
```

### Regression Training
```python
history = train_model(
    model,
    train_data=(X_train, y_train),
    val_data=(X_val, y_val),
    task='regression',  # Use MSE loss
    epochs=100
)
```

### With All Features
```python
history = train_model(
    model,
    train_data=(X_train, y_train),
    val_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    learning_rate=0.01,
    lr_decay=0.95,           # Decay LR by 5%
    lr_decay_every=10,       # Every 10 epochs
    clip_norm=5.0,           # Clip gradients to norm 5
    patience=10,             # Early stopping patience
    task='classification',
    verbose=True
)
```

## Function Reference

### Loss Functions
```python
# Classification
loss = cross_entropy_loss(predictions, targets)  # targets: (batch,) or (batch, n_classes)

# Regression
loss = mse_loss(predictions, targets)  # MSE for continuous values

# Accuracy
acc = accuracy(predictions, targets)  # Classification accuracy [0, 1]
```

### Single Training Step
```python
loss, metric, grad_norm = train_step(
    model, X_batch, y_batch,
    learning_rate=0.01,
    clip_norm=5.0,
    task='classification'
)
```

### Evaluation
```python
loss, metric = evaluate(
    model, X_test, y_test,
    task='classification',
    batch_size=32
)
```

### Gradient Clipping
```python
clipped_grads, global_norm = clip_gradients(grads, max_norm=5.0)
```

### Learning Rate Schedule
```python
lr = learning_rate_schedule(
    epoch,
    initial_lr=0.001,
    decay=0.95,
    decay_every=10
)
```

### Early Stopping
```python
early_stop = EarlyStopping(patience=10, min_delta=1e-4)

for epoch in range(epochs):
    # ... training ...
    if early_stop(val_loss, model.get_params()):
        print("Early stopping!")
        best_params = early_stop.get_best_params()
        model.set_params(best_params)
        break
```

### Visualization
```python
plot_training_curves(history, save_path='training.png')
```

## History Dictionary

```python
history = {
    'train_loss': [1.2, 1.1, 1.0, ...],      # Training loss per epoch
    'train_metric': [0.3, 0.4, 0.5, ...],    # Training metric per epoch
    'val_loss': [1.3, 1.2, 1.1, ...],        # Validation loss per epoch
    'val_metric': [0.25, 0.35, 0.45, ...],   # Validation metric per epoch
    'learning_rates': [0.01, 0.01, ...],     # LR used per epoch
    'grad_norms': [0.5, 0.4, 0.3, ...]       # Gradient norms per epoch
}
```

## Data Format

### Input Data
```python
X_train: (num_samples, seq_len, input_size)  # Sequences
y_train: (num_samples,)                       # Class labels (classification)
         or (num_samples, output_size)        # Targets (regression)
```

### Model Interface
```python
class YourModel:
    def forward(self, X, return_sequences=False):
        # X: (batch, seq_len, input_size)
        # return: (batch, output_size) if return_sequences=False
        pass

    def get_params(self):
        return {'W': self.W, 'b': self.b}

    def set_params(self, params):
        self.W = params['W']
        self.b = params['b']
```

## Hyperparameter Suggestions

### Small Dataset (< 1000 samples)
```python
epochs=100
batch_size=16
learning_rate=0.01
lr_decay=0.95
lr_decay_every=10
clip_norm=5.0
patience=10
```

### Medium Dataset (1000-10000 samples)
```python
epochs=50
batch_size=32
learning_rate=0.01
lr_decay=0.95
lr_decay_every=5
clip_norm=5.0
patience=10
```

### Large Dataset (> 10000 samples)
```python
epochs=30
batch_size=64
learning_rate=0.01
lr_decay=0.95
lr_decay_every=5
clip_norm=5.0
patience=5
```

### Overfitting Signs
```python
# Check train-val gap
train_acc = history['train_metric'][-1]
val_acc = history['val_metric'][-1]
gap = train_acc - val_acc

if gap > 0.1:  # Overfitting
    # Solutions:
    # - Increase patience (more epochs)
    # - Use smaller learning rate
    # - Add regularization (not implemented)
    # - Get more data
```

### Underfitting Signs
```python
# Both train and val accuracy low
if train_acc < 0.6 and val_acc < 0.6:
    # Solutions:
    # - Increase model size (hidden_size)
    # - Train longer (more epochs)
    # - Increase learning rate
    # - Check data quality
```

## Common Issues

### NaN in Loss
```python
# Possible causes:
# 1. Learning rate too high → reduce LR
# 2. Gradients exploding → check clip_norm
# 3. Numerical instability → losses use stable implementations

# Solution:
learning_rate=0.001  # Reduce
clip_norm=1.0        # Lower clipping threshold
```

### Loss Not Decreasing
```python
# Possible causes:
# 1. Learning rate too low
# 2. Wrong task type
# 3. Data/label mismatch

# Check:
print(f"Loss: {loss}, Metric: {metric}")
print(f"Predictions: {model.forward(X_batch[:1])}")
print(f"Targets: {y_batch[:1]}")
```

### Training Too Slow
```python
# Numerical gradients are slow
# For faster training:
# 1. Use smaller batches
# 2. Reduce model size
# 3. Use fewer epochs
# 4. Implement analytical gradients (BPTT)
```

## Testing

### Quick Test
```bash
python3 test_training_utils_quick.py
```

### Full Test Suite
```bash
python3 training_utils.py
```

### Demonstrations
```bash
python3 training_demo.py
```

## Files

- `training_utils.py` - Main implementation (37KB)
- `training_demo.py` - Demonstrations (11KB)
- `test_training_utils_quick.py` - Quick test (5KB)
- `TRAINING_UTILS_README.md` - Full documentation (10KB)
- `TRAINING_QUICK_REFERENCE.md` - This file (8KB)
- `TASK_P2_T3_SUMMARY.md` - Task summary (9KB)

## Next Steps

1. Implement Relational RNN with same interface
2. Use these utilities to train both LSTM and Relational RNN
3. Compare performance on reasoning tasks
4. (Optional) Implement analytical gradients for faster training
