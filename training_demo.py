"""
訓練工具示範程式
論文 18：Relational RNN - 任務 P2-T3

透過簡單的序列分類任務展示訓練工具的使用方式。
說明如何使用訓練迴圈、早停機制（early stopping）和視覺化功能。
"""

import numpy as np
from lstm_baseline import LSTM
from training_utils import (
    train_model,
    evaluate,
    plot_training_curves,
    cross_entropy_loss,
    accuracy
)


def generate_sequence_classification_data(num_samples=500, seq_len=20,
                                         input_size=10, num_classes=3):
    """
    產生合成序列分類資料。

    任務：根據序列前半部分與後半部分的特徵總和進行分類。

    參數：
        num_samples: 要產生的序列數量
        seq_len: 每個序列的長度
        input_size: 每個時間步的特徵數量
        num_classes: 輸出類別數量

    回傳：
        X: (num_samples, seq_len, input_size) - 輸入序列
        y: (num_samples,) - 類別標籤
    """
    X = np.random.randn(num_samples, seq_len, input_size)

    # 根據序列特性建立有意義的標籤
    labels = []
    for i in range(num_samples):
        # 計算前半部分與後半部分的總和
        first_half_sum = np.sum(X[i, :seq_len//2, :])
        second_half_sum = np.sum(X[i, seq_len//2:, :])

        # 根據差值進行分類
        diff = second_half_sum - first_half_sum

        if diff < -5:
            label = 0
        elif diff > 5:
            label = 2
        else:
            label = 1

        labels.append(label)

    y = np.array(labels)

    return X, y


def demo_basic_training():
    """展示使用 LSTM 進行基本訓練。"""
    print("=" * 80)
    print("Demo 1: Basic LSTM Training on Sequence Classification")
    print("=" * 80)

    # 設定隨機種子
    np.random.seed(42)

    # 產生資料
    print("\nGenerating synthetic sequence classification data...")
    X_train, y_train = generate_sequence_classification_data(
        num_samples=400, seq_len=20, input_size=8, num_classes=3
    )
    X_val, y_val = generate_sequence_classification_data(
        num_samples=100, seq_len=20, input_size=8, num_classes=3
    )
    X_test, y_test = generate_sequence_classification_data(
        num_samples=100, seq_len=20, input_size=8, num_classes=3
    )

    print(f"Train set: {X_train.shape}, {y_train.shape}")
    print(f"Val set: {X_val.shape}, {y_val.shape}")
    print(f"Test set: {X_test.shape}, {y_test.shape}")

    # 類別分布
    print("\nClass distribution (train):")
    for i in range(3):
        count = np.sum(y_train == i)
        print(f"  Class {i}: {count} samples ({count/len(y_train)*100:.1f}%)")

    # 建立模型
    print("\nCreating LSTM model...")
    input_size = 8
    hidden_size = 32
    output_size = 3

    model = LSTM(input_size, hidden_size, output_size)
    print(f"Model: LSTM(input={input_size}, hidden={hidden_size}, output={output_size})")

    # 計算參數數量
    params = model.get_params()
    total_params = sum(p.size for p in params.values())
    print(f"Total parameters: {total_params:,}")

    # 訓練模型
    print("\nTraining model...")
    print("-" * 80)

    history = train_model(
        model,
        train_data=(X_train, y_train),
        val_data=(X_val, y_val),
        epochs=20,
        batch_size=32,
        learning_rate=0.01,
        lr_decay=0.95,
        lr_decay_every=5,
        clip_norm=5.0,
        patience=10,
        task='classification',
        verbose=True
    )

    # 在測試集上評估
    print("\n" + "=" * 80)
    print("Evaluating on test set...")
    test_loss, test_acc = evaluate(model, X_test, y_test, task='classification')
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")

    # 輸出訓練摘要
    print("\n" + "=" * 80)
    print("Training Summary")
    print("=" * 80)
    print(f"Epochs trained: {len(history['train_loss'])}")
    print(f"Final train loss: {history['train_loss'][-1]:.4f}")
    print(f"Final val loss: {history['val_loss'][-1]:.4f}")
    print(f"Best val loss: {min(history['val_loss']):.4f}")
    print(f"Final train accuracy: {history['train_metric'][-1]:.4f}")
    print(f"Final val accuracy: {history['val_metric'][-1]:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")

    # 繪製訓練曲線
    print("\nGenerating training curves...")
    plot_training_curves(history)

    return model, history


def demo_overfitting_detection():
    """展示使用早停機制防止過度擬合。"""
    print("\n" + "=" * 80)
    print("Demo 2: Early Stopping - Detecting Overfitting")
    print("=" * 80)

    # 設定隨機種子
    np.random.seed(42)

    # 產生小型資料集以促進過度擬合
    print("\nGenerating small dataset (encourages overfitting)...")
    X_train, y_train = generate_sequence_classification_data(
        num_samples=50, seq_len=15, input_size=5, num_classes=3
    )
    X_val, y_val = generate_sequence_classification_data(
        num_samples=100, seq_len=15, input_size=5, num_classes=3
    )

    print(f"Train set: {X_train.shape} (small)")
    print(f"Val set: {X_val.shape}")

    # 建立較大的模型（更容易過度擬合）
    print("\nCreating large model (prone to overfitting)...")
    model = LSTM(input_size=5, hidden_size=64, output_size=3)

    # 使用早停機制進行訓練
    print("\nTraining with early stopping (patience=5)...")
    print("-" * 80)

    history = train_model(
        model,
        train_data=(X_train, y_train),
        val_data=(X_val, y_val),
        epochs=50,  # 允許較多的訓練週期
        batch_size=16,
        learning_rate=0.02,
        patience=5,  # 若連續 5 個週期無改善則停止
        task='classification',
        verbose=True
    )

    # 分析結果
    print("\n" + "=" * 80)
    print("Overfitting Analysis")
    print("=" * 80)
    print(f"Training stopped at epoch: {len(history['train_loss'])}")
    print(f"Final train loss: {history['train_loss'][-1]:.4f}")
    print(f"Final val loss: {history['val_loss'][-1]:.4f}")

    # 檢查過度擬合的跡象
    train_val_gap = history['train_metric'][-1] - history['val_metric'][-1]
    print(f"\nTrain accuracy: {history['train_metric'][-1]:.4f}")
    print(f"Val accuracy: {history['val_metric'][-1]:.4f}")
    print(f"Train-Val gap: {train_val_gap:.4f}")

    if train_val_gap > 0.1:
        print("WARNING: Significant train-val gap suggests overfitting")
    else:
        print("Model generalizes well (small train-val gap)")

    return history


def demo_learning_rate_schedule():
    """展示學習率排程的效果。"""
    print("\n" + "=" * 80)
    print("Demo 3: Learning Rate Schedule Effects")
    print("=" * 80)

    np.random.seed(42)

    # 產生資料
    X_train, y_train = generate_sequence_classification_data(
        num_samples=200, seq_len=15, input_size=6, num_classes=3
    )
    X_val, y_val = generate_sequence_classification_data(
        num_samples=50, seq_len=15, input_size=6, num_classes=3
    )

    # 使用積極的學習率衰減進行訓練
    print("\nTraining with aggressive LR decay (0.9 every 3 epochs)...")
    model1 = LSTM(input_size=6, hidden_size=24, output_size=3)

    history1 = train_model(
        model1,
        train_data=(X_train, y_train),
        val_data=(X_val, y_val),
        epochs=15,
        batch_size=32,
        learning_rate=0.05,
        lr_decay=0.9,
        lr_decay_every=3,
        patience=20,
        task='classification',
        verbose=True
    )

    print("\n" + "=" * 80)
    print("Learning Rate Schedule Analysis")
    print("=" * 80)
    print("LR values used:")
    for epoch, lr in enumerate(history1['learning_rates'][:10], 1):
        print(f"  Epoch {epoch:2d}: {lr:.6f}")

    print(f"\nInitial LR: {history1['learning_rates'][0]:.6f}")
    print(f"Final LR: {history1['learning_rates'][-1]:.6f}")
    print(f"LR reduction factor: {history1['learning_rates'][-1] / history1['learning_rates'][0]:.2f}x")


def demo_gradient_clipping():
    """展示梯度裁剪以確保訓練穩定性。"""
    print("\n" + "=" * 80)
    print("Demo 4: Gradient Clipping for Training Stability")
    print("=" * 80)

    np.random.seed(42)

    # 產生資料
    X_train, y_train = generate_sequence_classification_data(
        num_samples=100, seq_len=20, input_size=8, num_classes=3
    )
    X_val, y_val = generate_sequence_classification_data(
        num_samples=30, seq_len=20, input_size=8, num_classes=3
    )

    print("\nTraining with gradient clipping (max_norm=5.0)...")
    model = LSTM(input_size=8, hidden_size=32, output_size=3)

    history = train_model(
        model,
        train_data=(X_train, y_train),
        val_data=(X_val, y_val),
        epochs=10,
        batch_size=16,
        learning_rate=0.02,
        clip_norm=5.0,
        patience=20,
        task='classification',
        verbose=True
    )

    print("\n" + "=" * 80)
    print("Gradient Norm Analysis")
    print("=" * 80)
    print("Gradient norms during training:")
    for epoch, norm in enumerate(history['grad_norms'][:10], 1):
        status = " (CLIPPED)" if norm > 4.5 else ""
        print(f"  Epoch {epoch:2d}: {norm:.4f}{status}")

    max_norm = max(history['grad_norms'])
    avg_norm = np.mean(history['grad_norms'])
    print(f"\nMax gradient norm: {max_norm:.4f}")
    print(f"Avg gradient norm: {avg_norm:.4f}")

    if max_norm > 4.8:
        print("Note: Gradient clipping was active (some gradients were clipped)")
    else:
        print("Note: Gradients stayed within bounds (no clipping needed)")


def main():
    """執行所有示範。"""
    print("\n" + "=" * 80)
    print(" " * 20 + "TRAINING UTILITIES DEMONSTRATION")
    print(" " * 18 + "Paper 18: Relational RNN - Task P2-T3")
    print("=" * 80)
    print("\nThis demo shows:")
    print("  1. Basic training with LSTM on sequence classification")
    print("  2. Early stopping to prevent overfitting")
    print("  3. Learning rate schedule effects")
    print("  4. Gradient clipping for stability")
    print("=" * 80)

    # 執行各項示範
    demo_basic_training()
    demo_overfitting_detection()
    demo_learning_rate_schedule()
    demo_gradient_clipping()

    print("\n" + "=" * 80)
    print(" " * 25 + "ALL DEMONSTRATIONS COMPLETED")
    print("=" * 80)
    print("\nKey Takeaways:")
    print("  - Training utilities work correctly with LSTM models")
    print("  - Early stopping prevents overfitting on small datasets")
    print("  - Learning rate decay helps fine-tune in later epochs")
    print("  - Gradient clipping maintains training stability")
    print("  - All utilities are compatible with any model with get/set_params()")
    print("\nThese utilities are ready for use with Relational RNN!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
