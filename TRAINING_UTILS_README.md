# Training Utilities - Paper 18: Relational RNN

## Task P2-T3: Training Utilities and Loss Functions

This module provides comprehensive training utilities for both LSTM and Relational RNN models using NumPy only.

## Files

- `training_utils.py` - Main utilities module with loss functions, training loops, and optimization helpers
- `training_demo.py` - Comprehensive demonstrations of all training features
- `TRAINING_UTILS_README.md` - This documentation

## Features Implemented

### 1. Loss Functions

#### Cross-Entropy Loss
```python
loss = cross_entropy_loss(predictions, targets)
```
- Supports both sparse (class indices) and one-hot encoded targets
- Numerically stable implementation using log-sum-exp trick
- Used for classification tasks

#### Mean Squared Error (MSE) Loss
```python
loss = mse_loss(predictions, targets)
```
- For regression tasks (object tracking, trajectory prediction)
- Simple squared difference averaged over all elements

#### Softmax Function
```python
probs = softmax(logits)
```
- Numerically stable softmax implementation
- Converts logits to probabilities

#### Accuracy Metric
```python
acc = accuracy(predictions, targets)
```
- Classification accuracy computation
- Works with both sparse and one-hot targets

### 2. Gradient Computation

#### Numerical Gradient (Finite Differences)
```python
gradients = compute_numerical_gradient(model, X_batch, y_batch, loss_fn)
```
- Element-by-element finite difference approximation
- Educational implementation (slow but correct)
- Uses central difference: `df/dx ≈ (f(x + ε) - f(x - ε)) / (2ε)`

#### Fast Numerical Gradient
```python
gradients = compute_numerical_gradient_fast(model, X_batch, y_batch, loss_fn)
```
- Vectorized gradient estimation (faster than element-wise)
- Still slower than analytical gradients but more practical
- Good for prototyping and testing

**Note**: For production use, implement analytical gradients via backpropagation through time (BPTT).

### 3. Optimization Utilities

#### Gradient Clipping
```python
clipped_grads, global_norm = clip_gradients(grads, max_norm=5.0)
```
- Prevents exploding gradients (critical for RNN stability)
- Clips by global norm across all parameters
- Returns both clipped gradients and original norm for monitoring

#### Learning Rate Schedule
```python
lr = learning_rate_schedule(epoch, initial_lr=0.001, decay=0.95, decay_every=10)
```
- Exponential decay schedule
- Reduces learning rate over time for fine-tuning
- Formula: `lr = initial_lr * (decay ^ (epoch // decay_every))`

#### Early Stopping
```python
early_stopping = EarlyStopping(patience=10, min_delta=1e-4)
should_stop = early_stopping(val_loss, model_params)
best_params = early_stopping.get_best_params()
```
- Prevents overfitting by monitoring validation loss
- Saves best parameters automatically
- Configurable patience (epochs to wait) and minimum improvement threshold

### 4. Training Functions

#### Single Training Step
```python
loss, metric, grad_norm = train_step(
    model, X_batch, y_batch,
    learning_rate=0.001,
    clip_norm=5.0,
    task='classification'
)
```
- Performs one gradient descent step
- Computes gradients, clips them, and updates parameters
- Returns loss, metric (accuracy or negative loss), and gradient norm
- Supports both classification and regression tasks

#### Model Evaluation
```python
avg_loss, avg_metric = evaluate(
    model, X_test, y_test,
    task='classification',
    batch_size=32
)
```
- Evaluates model without updating parameters
- Processes data in batches (handles large datasets)
- Returns average loss and metric

#### Full Training Loop
```python
history = train_model(
    model,
    train_data=(X_train, y_train),
    val_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    learning_rate=0.001,
    lr_decay=0.95,
    lr_decay_every=10,
    clip_norm=5.0,
    patience=10,
    task='classification',
    verbose=True
)
```

Features:
- Automatic batching with optional shuffling
- Learning rate decay
- Gradient clipping
- Early stopping with best model restoration
- Progress tracking and verbose output
- Returns comprehensive training history

History dictionary contains:
- `train_loss`: Training loss per epoch
- `train_metric`: Training metric per epoch
- `val_loss`: Validation loss per epoch
- `val_metric`: Validation metric per epoch
- `learning_rates`: Learning rates used
- `grad_norms`: Gradient norms (for monitoring stability)

### 5. Visualization

#### Plot Training Curves
```python
plot_training_curves(history, save_path='training_curves.png')
```
- Creates 2x2 grid of plots:
  - Loss over epochs (train & val)
  - Metric over epochs (train & val)
  - Learning rate schedule
  - Gradient norms
- Falls back to text output if matplotlib unavailable

## Usage Examples

### Basic Training
```python
from lstm_baseline import LSTM
from training_utils import train_model, evaluate

# Create model
model = LSTM(input_size=10, hidden_size=32, output_size=3)

# Prepare data
X_train, y_train = ...  # (num_samples, seq_len, input_size)
X_val, y_val = ...

# Train
history = train_model(
    model,
    train_data=(X_train, y_train),
    val_data=(X_val, y_val),
    epochs=50,
    batch_size=32,
    learning_rate=0.01,
    task='classification'
)

# Evaluate
test_loss, test_acc = evaluate(model, X_test, y_test)
print(f"Test accuracy: {test_acc:.4f}")
```

### Custom Training Loop
```python
from training_utils import train_step, clip_gradients

for epoch in range(num_epochs):
    for X_batch, y_batch in create_batches(X_train, y_train, batch_size=32):
        loss, acc, grad_norm = train_step(
            model, X_batch, y_batch,
            learning_rate=0.01,
            clip_norm=5.0
        )
        print(f"Batch loss: {loss:.4f}, acc: {acc:.4f}")
```

### Regression Task
```python
# For regression (e.g., object tracking)
history = train_model(
    model,
    train_data=(X_train, y_train),
    val_data=(X_val, y_val),
    task='regression',  # Use MSE loss
    epochs=100
)
```

## Model Compatibility

The training utilities work with any model that implements:

```python
class YourModel:
    def forward(self, X, return_sequences=False):
        """
        Args:
            X: (batch, seq_len, input_size)
            return_sequences: bool
        Returns:
            outputs: (batch, output_size) if return_sequences=False
                    (batch, seq_len, output_size) if return_sequences=True
        """
        pass

    def get_params(self):
        """Return dict of parameter names to arrays"""
        return {'W': self.W, 'b': self.b, ...}

    def set_params(self, params):
        """Set parameters from dict"""
        self.W = params['W']
        self.b = params['b']
```

Compatible models:
- LSTM (from `lstm_baseline.py`)
- Relational RNN (to be implemented)
- Any custom RNN architecture following the interface

## Test Results

All tests pass successfully:

```
✓ Loss Functions
  - Cross-entropy: Perfect predictions → near-zero loss
  - MSE: Perfect predictions → zero loss
  - Sparse and one-hot targets give identical results

✓ Optimization Utilities
  - Gradient clipping: Small gradients unchanged, large gradients clipped to max_norm
  - Learning rate schedule: Exponential decay works correctly
  - Early stopping: Stops after patience epochs without improvement

✓ Training Loop
  - Single step: Parameters update correctly
  - Evaluation: Works without parameter updates
  - Full training: Loss decreases over epochs
  - History tracking: All metrics recorded correctly
```

## Performance Characteristics

### Numerical Gradients
- **Pros**:
  - Simple to implement
  - No risk of backpropagation bugs
  - Educational value

- **Cons**:
  - Very slow (O(parameters) forward passes per step)
  - Approximate (finite difference error)
  - Not suitable for large models or production use

### Recommendations
1. **For prototyping**: Use provided numerical gradients
2. **For experiments**: Implement fast numerical gradient estimation
3. **For production**: Implement analytical gradients via BPTT

## Simplifications & Limitations

1. **Gradients**: Numerical approximation instead of analytical BPTT
   - Trade-off: Simplicity vs. speed
   - Suitable for educational purposes and small models

2. **Optimizer**: Simple SGD only (no momentum, Adam, etc.)
   - Easy to extend with more sophisticated optimizers

3. **Batching**: No parallel processing
   - Pure NumPy implementation (no GPU support)

4. **Gradient Estimation**: Fast version still approximate
   - Uses random perturbations instead of element-wise finite differences

## Future Enhancements

Potential improvements (not required for this task):
- [ ] Analytical gradient computation via BPTT
- [ ] Adam optimizer
- [ ] Momentum-based optimization
- [ ] Learning rate warmup
- [ ] Gradient accumulation for large batches
- [ ] Mixed precision training simulation
- [ ] More advanced LR schedules (cosine annealing, etc.)

## Integration with Relational RNN

These utilities are ready to use with the Relational RNN model. Simply ensure your Relational RNN implements the required interface (`forward`, `get_params`, `set_params`), and all training utilities will work seamlessly.

Example:
```python
from relational_rnn import RelationalRNN
from training_utils import train_model

# Create Relational RNN
model = RelationalRNN(input_size=10, hidden_size=32, output_size=3)

# Train exactly like LSTM
history = train_model(
    model,
    train_data=(X_train, y_train),
    val_data=(X_val, y_val),
    epochs=50
)
```

## Summary

This implementation provides a complete, NumPy-only training infrastructure for:
- **Loss computation**: Cross-entropy and MSE with numerical stability
- **Gradient computation**: Numerical approximation (finite differences)
- **Optimization**: Gradient clipping, LR scheduling, early stopping
- **Training**: Full training loop with metrics tracking
- **Monitoring**: Comprehensive history and visualization

All utilities are tested, documented, and ready for use with both LSTM and Relational RNN models.
