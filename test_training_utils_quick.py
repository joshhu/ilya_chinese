"""
Quick Test for Training Utilities
Paper 18: Relational RNN - Task P2-T3

Fast sanity check that all training utilities work correctly.
"""

import numpy as np
from lstm_baseline import LSTM
from training_utils import (
    cross_entropy_loss,
    mse_loss,
    accuracy,
    clip_gradients,
    learning_rate_schedule,
    EarlyStopping,
    train_step,
    evaluate,
    train_model
)


def quick_test():
    """Quick sanity check of all utilities."""
    print("=" * 60)
    print("Quick Test - Training Utilities")
    print("=" * 60)

    np.random.seed(42)

    # Test 1: Loss functions
    print("\n[1/6] Testing loss functions...")
    predictions = np.random.randn(10, 3)
    targets_sparse = np.random.randint(0, 3, size=10)
    targets_onehot = np.eye(3)[targets_sparse]

    ce_loss_sparse = cross_entropy_loss(predictions, targets_sparse)
    ce_loss_onehot = cross_entropy_loss(predictions, targets_onehot)
    acc = accuracy(predictions, targets_sparse)

    assert np.isclose(ce_loss_sparse, ce_loss_onehot), "CE loss mismatch"
    assert 0 <= acc <= 1, "Accuracy out of range"
    print(f"  ✓ Cross-entropy: {ce_loss_sparse:.4f}")
    print(f"  ✓ Accuracy: {acc:.4f}")

    mse = mse_loss(predictions, np.random.randn(10, 3))
    assert mse >= 0, "MSE should be non-negative"
    print(f"  ✓ MSE: {mse:.4f}")

    # Test 2: Gradient clipping
    print("\n[2/6] Testing gradient clipping...")
    grads = {
        'W': np.random.randn(100, 100) * 10,
        'b': np.random.randn(100) * 10
    }
    clipped, norm = clip_gradients(grads, max_norm=5.0)
    clipped_norm = np.sqrt(sum(np.sum(g**2) for g in clipped.values()))

    assert norm > 5.0, "Original norm should exceed threshold"
    assert np.isclose(clipped_norm, 5.0, rtol=1e-4), "Clipped norm should equal max_norm"
    print(f"  ✓ Original norm: {norm:.4f}")
    print(f"  ✓ Clipped norm: {clipped_norm:.4f}")

    # Test 3: Learning rate schedule
    print("\n[3/6] Testing learning rate schedule...")
    lr0 = learning_rate_schedule(0, initial_lr=0.1, decay=0.9, decay_every=5)
    lr10 = learning_rate_schedule(10, initial_lr=0.1, decay=0.9, decay_every=5)

    assert lr0 == 0.1, "Initial LR should be 0.1"
    assert lr10 == 0.1 * 0.9**2, "LR at epoch 10 should be decayed twice"
    print(f"  ✓ Epoch 0: {lr0:.6f}")
    print(f"  ✓ Epoch 10: {lr10:.6f}")

    # Test 4: Early stopping
    print("\n[4/6] Testing early stopping...")
    early_stop = EarlyStopping(patience=3)

    losses = [1.0, 0.9, 0.8, 0.85, 0.84, 0.83, 0.83]
    stopped = False
    for i, loss in enumerate(losses):
        stopped = early_stop(loss)
        if stopped:
            break

    assert stopped, "Early stopping should trigger"
    assert early_stop.best_loss == 0.8, "Best loss should be 0.8"
    print(f"  ✓ Stopped at epoch {i}")
    print(f"  ✓ Best loss: {early_stop.best_loss:.2f}")

    # Test 5: Train step
    print("\n[5/6] Testing train step...")
    model = LSTM(input_size=5, hidden_size=8, output_size=3)
    X_batch = np.random.randn(4, 10, 5)
    y_batch = np.random.randint(0, 3, size=4)

    params_before = {k: v.copy() for k, v in model.get_params().items()}
    loss, acc, grad_norm = train_step(
        model, X_batch, y_batch,
        learning_rate=0.01,
        task='classification'
    )
    params_after = model.get_params()

    # Check parameters changed
    changed = any(
        not np.allclose(params_before[k], params_after[k])
        for k in params_before.keys()
    )

    assert changed, "Parameters should change after training step"
    assert loss > 0, "Loss should be positive"
    assert 0 <= acc <= 1, "Accuracy should be in [0, 1]"
    print(f"  ✓ Loss: {loss:.4f}")
    print(f"  ✓ Accuracy: {acc:.4f}")
    print(f"  ✓ Grad norm: {grad_norm:.4f}")

    # Test 6: Evaluation
    print("\n[6/6] Testing evaluation...")
    X_test = np.random.randn(20, 10, 5)
    y_test = np.random.randint(0, 3, size=20)

    test_loss, test_acc = evaluate(model, X_test, y_test, task='classification')

    assert test_loss > 0, "Test loss should be positive"
    assert 0 <= test_acc <= 1, "Test accuracy should be in [0, 1]"
    print(f"  ✓ Test loss: {test_loss:.4f}")
    print(f"  ✓ Test accuracy: {test_acc:.4f}")

    # Summary
    print("\n" + "=" * 60)
    print("All Quick Tests Passed! ✓")
    print("=" * 60)
    print("\nUtilities tested:")
    print("  ✓ cross_entropy_loss")
    print("  ✓ mse_loss")
    print("  ✓ accuracy")
    print("  ✓ clip_gradients")
    print("  ✓ learning_rate_schedule")
    print("  ✓ EarlyStopping")
    print("  ✓ train_step")
    print("  ✓ evaluate")
    print("\nReady for production use!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    quick_test()
