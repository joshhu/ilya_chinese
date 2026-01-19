"""
LSTM Baseline - Usage Demonstration

This script demonstrates how to use the LSTM baseline for various tasks.
"""

import numpy as np
from lstm_baseline import LSTM, LSTMCell


def demo_sequence_classification():
    """
    Demonstrate LSTM for sequence classification.
    Task: Classify sequences based on their patterns.
    """
    print("\n" + "="*60)
    print("Demo 1: Sequence Classification")
    print("="*60)

    # Create synthetic data: sequences with different patterns
    batch_size = 4
    seq_len = 20
    input_size = 8
    hidden_size = 32
    num_classes = 3

    print(f"\nTask: Classify {num_classes} different sequence patterns")
    print(f"Sequence length: {seq_len}, Input features: {input_size}")

    # Generate sequences with different patterns
    sequences = []
    labels = []

    # Pattern 0: Increasing trend
    seq0 = np.linspace(0, 1, seq_len).reshape(-1, 1) * np.random.randn(seq_len, input_size) * 0.1
    seq0 = seq0 + np.linspace(0, 1, seq_len).reshape(-1, 1)
    sequences.append(seq0)
    labels.append(0)

    # Pattern 1: Decreasing trend
    seq1 = np.linspace(1, 0, seq_len).reshape(-1, 1) * np.random.randn(seq_len, input_size) * 0.1
    seq1 = seq1 + np.linspace(1, 0, seq_len).reshape(-1, 1)
    sequences.append(seq1)
    labels.append(1)

    # Pattern 2: Oscillating
    seq2 = np.sin(np.linspace(0, 4*np.pi, seq_len)).reshape(-1, 1) * np.ones((seq_len, input_size))
    seq2 = seq2 + np.random.randn(seq_len, input_size) * 0.1
    sequences.append(seq2)
    labels.append(2)

    # Pattern 0 again
    seq0_2 = np.linspace(0, 1, seq_len).reshape(-1, 1) * np.random.randn(seq_len, input_size) * 0.1
    seq0_2 = seq0_2 + np.linspace(0, 1, seq_len).reshape(-1, 1)
    sequences.append(seq0_2)
    labels.append(0)

    # Stack into batch
    batch = np.stack(sequences, axis=0)  # (batch_size, seq_len, input_size)

    # Create LSTM model
    lstm = LSTM(input_size, hidden_size, output_size=num_classes)

    # Forward pass - get only final output for classification
    outputs = lstm.forward(batch, return_sequences=False)

    print(f"\nInput shape: {batch.shape}")
    print(f"Output shape: {outputs.shape}")
    print(f"Expected shape: ({batch_size}, {num_classes})")

    # Apply softmax to get class probabilities
    exp_outputs = np.exp(outputs - np.max(outputs, axis=1, keepdims=True))
    probabilities = exp_outputs / np.sum(exp_outputs, axis=1, keepdims=True)

    print(f"\nPredicted class probabilities (before training):")
    for i in range(batch_size):
        pred_class = np.argmax(probabilities[i])
        true_class = labels[i]
        print(f"  Sample {i}: pred={pred_class}, true={true_class}, probs={probabilities[i]}")

    print("\nNote: Model is randomly initialized, so predictions are random.")
    print("After training, it would learn to classify these patterns correctly.")


def demo_sequence_to_sequence():
    """
    Demonstrate LSTM for sequence-to-sequence tasks.
    Task: Echo the input sequence with a transformation.
    """
    print("\n" + "="*60)
    print("Demo 2: Sequence-to-Sequence Processing")
    print("="*60)

    batch_size = 2
    seq_len = 15
    input_size = 10
    hidden_size = 24
    output_size = 10

    print(f"\nTask: Process sequences and output transformed sequences")
    print(f"Input sequence length: {seq_len}")
    print(f"Output sequence length: {seq_len}")

    # Create input sequences
    sequences = np.random.randn(batch_size, seq_len, input_size) * 0.5

    # Create LSTM
    lstm = LSTM(input_size, hidden_size, output_size=output_size)

    # Forward pass - get all time step outputs
    outputs = lstm.forward(sequences, return_sequences=True)

    print(f"\nInput shape: {sequences.shape}")
    print(f"Output shape: {outputs.shape}")
    print(f"Expected shape: ({batch_size}, {seq_len}, {output_size})")

    # Show output statistics
    print(f"\nOutput statistics:")
    print(f"  Mean: {np.mean(outputs):.4f}")
    print(f"  Std: {np.std(outputs):.4f}")
    print(f"  Min: {np.min(outputs):.4f}")
    print(f"  Max: {np.max(outputs):.4f}")


def demo_state_persistence():
    """
    Demonstrate how LSTM maintains state across time steps.
    """
    print("\n" + "="*60)
    print("Demo 3: State Persistence and Memory")
    print("="*60)

    batch_size = 1
    seq_len = 30
    input_size = 5
    hidden_size = 16

    print(f"\nDemonstrating how LSTM maintains memory over {seq_len} time steps")

    # Create a sequence with a pattern early on
    sequence = np.zeros((batch_size, seq_len, input_size))
    # Set a distinctive pattern in first 5 time steps
    sequence[:, 0:5, :] = 1.0
    # Rest is zeros

    # Create LSTM
    lstm = LSTM(input_size, hidden_size, output_size=None)

    # Get all outputs and final state
    outputs, final_h, final_c = lstm.forward(sequence, return_sequences=True, return_state=True)

    print(f"\nInput shape: {sequence.shape}")
    print(f"Output shape: {outputs.shape}")

    # Analyze how the hidden state evolves
    print(f"\nHidden state evolution:")
    print(f"  At t=5 (after pattern):  mean={np.mean(outputs[0, 5, :]):.4f}, std={np.std(outputs[0, 5, :]):.4f}")
    print(f"  At t=15 (middle):         mean={np.mean(outputs[0, 15, :]):.4f}, std={np.std(outputs[0, 15, :]):.4f}")
    print(f"  At t=29 (end):           mean={np.mean(outputs[0, 29, :]):.4f}, std={np.std(outputs[0, 29, :]):.4f}")

    print(f"\nFinal hidden state shape: {final_h.shape}")
    print(f"Final cell state shape: {final_c.shape}")

    print("\nThe LSTM maintains internal state throughout the sequence,")
    print("allowing it to remember patterns from early time steps.")


def demo_initialization_importance():
    """
    Demonstrate the importance of proper initialization.
    """
    print("\n" + "="*60)
    print("Demo 4: Importance of Initialization")
    print("="*60)

    input_size = 16
    hidden_size = 32
    seq_len = 100
    batch_size = 1

    # Create LSTM with proper initialization
    lstm = LSTM(input_size, hidden_size, output_size=None)

    # Create long sequence
    sequence = np.random.randn(batch_size, seq_len, input_size) * 0.1

    # Forward pass
    outputs = lstm.forward(sequence, return_sequences=True)

    print(f"\nProcessing long sequence (length={seq_len})")
    print(f"\nWith proper initialization:")
    print(f"  Orthogonal recurrent weights")
    print(f"  Xavier input weights")
    print(f"  Forget bias = 1.0")
    print(f"\nResults:")
    print(f"  Output mean: {np.mean(outputs):.4f}")
    print(f"  Output std: {np.std(outputs):.4f}")
    print(f"  Contains NaN: {np.isnan(outputs).any()}")
    print(f"  Contains Inf: {np.isinf(outputs).any()}")

    # Check gradient flow (approximate)
    output_start = outputs[:, 0:10, :]
    output_end = outputs[:, -10:, :]

    print(f"\nGradient flow (variance check):")
    print(f"  Early outputs variance: {np.var(output_start):.4f}")
    print(f"  Late outputs variance: {np.var(output_end):.4f}")
    print(f"  Ratio: {np.var(output_end) / (np.var(output_start) + 1e-8):.4f}")

    print("\nProper initialization helps maintain stable gradients")
    print("and prevents vanishing/exploding gradient problems.")


def demo_cell_level_usage():
    """
    Demonstrate using LSTMCell directly for custom loops.
    """
    print("\n" + "="*60)
    print("Demo 5: Using LSTMCell for Custom Processing")
    print("="*60)

    input_size = 8
    hidden_size = 16
    batch_size = 3

    print(f"\nManually stepping through time with LSTMCell")
    print(f"Useful for custom training loops or variable-length sequences")

    # Create cell
    cell = LSTMCell(input_size, hidden_size)

    # Initialize states
    h = np.zeros((hidden_size, batch_size))
    c = np.zeros((hidden_size, batch_size))

    print(f"\nInitial states:")
    print(f"  h shape: {h.shape}, all zeros: {np.allclose(h, 0)}")
    print(f"  c shape: {c.shape}, all zeros: {np.allclose(c, 0)}")

    # Process several time steps
    print(f"\nProcessing 5 time steps:")
    for t in range(5):
        # Random input
        x = np.random.randn(batch_size, input_size) * 0.1

        # Step forward
        h, c = cell.forward(x, h, c)

        print(f"  t={t}: h_mean={np.mean(h):.4f}, c_mean={np.mean(c):.4f}")

    print(f"\nFinal states:")
    print(f"  h shape: {h.shape}")
    print(f"  c shape: {c.shape}")
    print("\nThis gives you full control over the processing loop.")


if __name__ == "__main__":
    print("\n" + "="*70)
    print(" "*15 + "LSTM Baseline - Usage Demonstrations")
    print("="*70)

    np.random.seed(42)  # For reproducibility

    # Run all demos
    demo_sequence_classification()
    demo_sequence_to_sequence()
    demo_state_persistence()
    demo_initialization_importance()
    demo_cell_level_usage()

    print("\n" + "="*70)
    print(" "*20 + "All Demonstrations Complete!")
    print("="*70)
    print("\nKey Takeaways:")
    print("1. LSTM can handle various sequence tasks (classification, seq2seq)")
    print("2. It maintains internal memory across time steps")
    print("3. Proper initialization is critical for stability")
    print("4. Both LSTM and LSTMCell classes provide flexibility")
    print("5. Ready for comparison with Relational RNN")
    print("="*70 + "\n")
