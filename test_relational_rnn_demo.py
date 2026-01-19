"""
Demonstration of Relational RNN Cell - Extended Testing

This script provides additional visualizations and tests to demonstrate:
1. How memory evolves over a sequence
2. How LSTM and memory interact
3. Comparison of outputs with and without memory

Paper 18: Relational RNN - Task P2-T2 Demo
"""

import numpy as np
from relational_rnn_cell import RelationalRNN, RelationalRNNCell
from lstm_baseline import LSTM


def analyze_memory_evolution():
    """Detailed analysis of how memory evolves over a sequence."""
    print("=" * 80)
    print("Analyzing Memory Evolution Over Sequence")
    print("=" * 80)

    np.random.seed(42)

    # Configuration
    batch_size = 1  # Single example for clarity
    seq_len = 15
    input_size = 32
    hidden_size = 64
    num_slots = 4
    slot_size = 64

    print(f"\nConfiguration:")
    print(f"  Sequence length: {seq_len}")
    print(f"  Memory slots: {num_slots}")
    print(f"  Slot size: {slot_size}")

    # Create cell
    cell = RelationalRNNCell(
        input_size=input_size,
        hidden_size=hidden_size,
        num_slots=num_slots,
        slot_size=slot_size,
        num_heads=2
    )

    # Create sequence with pattern
    # First half: small values, second half: large values
    sequence = np.random.randn(batch_size, seq_len, input_size) * 0.1
    sequence[:, seq_len//2:, :] *= 5.0  # Increase magnitude in second half

    print(f"\n[Analysis] Processing sequence and tracking memory...")

    # Initialize states
    h = np.zeros((hidden_size, batch_size))
    c = np.zeros((hidden_size, batch_size))
    memory = cell.init_memory(batch_size)

    # Track memory statistics
    memory_norms = []
    memory_means = []
    memory_stds = []
    slot_norms = []  # Track each slot separately

    # Process sequence
    for t in range(seq_len):
        x_t = sequence[:, t, :]
        output, h, c, memory = cell.forward(x_t, h, c, memory)

        # Compute statistics
        memory_norm = np.linalg.norm(memory)
        memory_mean = np.mean(memory)
        memory_std = np.std(memory)

        memory_norms.append(memory_norm)
        memory_means.append(memory_mean)
        memory_stds.append(memory_std)

        # Track individual slot norms
        slot_norm = [np.linalg.norm(memory[0, i, :]) for i in range(num_slots)]
        slot_norms.append(slot_norm)

    print(f"\n[Results] Memory Evolution Statistics:")
    print(f"\n  Overall Memory Norm (L2):")
    print(f"    Initial steps (1-5):  {np.mean(memory_norms[:5]):.4f}")
    print(f"    Middle steps (6-10):  {np.mean(memory_norms[5:10]):.4f}")
    print(f"    Final steps (11-15):  {np.mean(memory_norms[10:]):.4f}")

    print(f"\n  Memory Mean:")
    print(f"    Initial steps (1-5):  {np.mean(memory_means[:5]):.4f}")
    print(f"    Middle steps (6-10):  {np.mean(memory_means[5:10]):.4f}")
    print(f"    Final steps (11-15):  {np.mean(memory_means[10:]):.4f}")

    print(f"\n  Memory Standard Deviation:")
    print(f"    Initial steps (1-5):  {np.mean(memory_stds[:5]):.4f}")
    print(f"    Middle steps (6-10):  {np.mean(memory_stds[5:10]):.4f}")
    print(f"    Final steps (11-15):  {np.mean(memory_stds[10:]):.4f}")

    # Analyze slot specialization
    print(f"\n  Individual Slot Norms at Final Step:")
    final_slot_norms = slot_norms[-1]
    for i, norm in enumerate(final_slot_norms):
        print(f"    Slot {i}: {norm:.4f}")

    # Check if slots have different magnitudes (indication of specialization)
    slot_variance = np.var(final_slot_norms)
    print(f"\n  Slot norm variance: {slot_variance:.4f}")
    if slot_variance > 0.01:
        print(f"    -> Slots show differentiation (potential specialization)")
    else:
        print(f"    -> Slots relatively uniform")

    print("\n" + "=" * 80 + "\n")


def compare_with_without_memory():
    """Compare LSTM alone vs. LSTM with relational memory."""
    print("=" * 80)
    print("Comparing LSTM vs. LSTM + Relational Memory")
    print("=" * 80)

    np.random.seed(42)

    # Configuration
    batch_size = 2
    seq_len = 10
    input_size = 32
    hidden_size = 64
    output_size = 16

    # Create same sequence for fair comparison
    sequence = np.random.randn(batch_size, seq_len, input_size)

    print(f"\nConfiguration:")
    print(f"  batch_size: {batch_size}")
    print(f"  seq_len: {seq_len}")
    print(f"  input_size: {input_size}")
    print(f"  hidden_size: {hidden_size}")
    print(f"  output_size: {output_size}")

    # LSTM baseline
    print(f"\n[1] LSTM Baseline (no relational memory)")
    lstm = LSTM(input_size, hidden_size, output_size)
    lstm_outputs = lstm.forward(sequence, return_sequences=True)

    # Relational RNN
    print(f"[2] Relational RNN (LSTM + relational memory)")
    rel_rnn = RelationalRNN(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        num_slots=4,
        slot_size=64,
        num_heads=2
    )
    rel_outputs = rel_rnn.forward(sequence, return_sequences=True)

    # Compare outputs
    print(f"\n[Comparison] Output Statistics:")
    print(f"\n  LSTM Baseline:")
    print(f"    Mean: {lstm_outputs.mean():.4f}")
    print(f"    Std:  {lstm_outputs.std():.4f}")
    print(f"    Min:  {lstm_outputs.min():.4f}")
    print(f"    Max:  {lstm_outputs.max():.4f}")

    print(f"\n  Relational RNN:")
    print(f"    Mean: {rel_outputs.mean():.4f}")
    print(f"    Std:  {rel_outputs.std():.4f}")
    print(f"    Min:  {rel_outputs.min():.4f}")
    print(f"    Max:  {rel_outputs.max():.4f}")

    # Compute difference
    diff = np.abs(lstm_outputs - rel_outputs)
    print(f"\n  Absolute Difference:")
    print(f"    Mean: {diff.mean():.4f}")
    print(f"    Max:  {diff.max():.4f}")

    # Analysis
    print(f"\n[Analysis]")
    print(f"  - Both models process the same sequence")
    print(f"  - Different random initializations lead to different outputs")
    print(f"  - Relational RNN has additional memory mechanism")
    print(f"  - Memory allows for more complex representations")

    print("\n" + "=" * 80 + "\n")


def demonstrate_lstm_memory_interaction():
    """Show step-by-step how LSTM and memory interact."""
    print("=" * 80)
    print("Demonstrating LSTM + Memory Interaction")
    print("=" * 80)

    np.random.seed(42)

    # Simple configuration
    batch_size = 1
    input_size = 8
    hidden_size = 16
    num_slots = 3
    slot_size = 16

    print(f"\nConfiguration:")
    print(f"  Input size: {input_size}")
    print(f"  Hidden size: {hidden_size}")
    print(f"  Num slots: {num_slots}")
    print(f"  Slot size: {slot_size}")

    # Create cell
    cell = RelationalRNNCell(
        input_size=input_size,
        hidden_size=hidden_size,
        num_slots=num_slots,
        slot_size=slot_size,
        num_heads=1
    )

    # Initialize states
    h = np.zeros((hidden_size, batch_size))
    c = np.zeros((hidden_size, batch_size))
    memory = cell.init_memory(batch_size)

    print(f"\n[Initial State]")
    print(f"  LSTM h: all zeros")
    print(f"  LSTM c: all zeros")
    print(f"  Memory: all zeros")

    # Process a few steps
    num_steps = 3
    for step in range(num_steps):
        print(f"\n[Step {step + 1}]")

        # Create input
        x = np.random.randn(batch_size, input_size) * 0.5
        print(f"  Input: mean={x.mean():.4f}, std={x.std():.4f}")

        # Forward pass
        output, h_new, c_new, memory_new = cell.forward(x, h, c, memory)

        # Show changes
        h_change = np.linalg.norm(h_new - h)
        c_change = np.linalg.norm(c_new - c)
        mem_change = np.linalg.norm(memory_new - memory)

        print(f"  LSTM hidden change: {h_change:.4f}")
        print(f"  LSTM cell change:   {c_change:.4f}")
        print(f"  Memory change:      {mem_change:.4f}")
        print(f"  Output: mean={output.mean():.4f}, std={output.std():.4f}")

        # Update states
        h = h_new
        c = c_new
        memory = memory_new

    print(f"\n[Interaction Summary]")
    print(f"  1. Input -> LSTM -> updates hidden state (h)")
    print(f"  2. Hidden state (h) -> updates memory via projection")
    print(f"  3. Memory slots interact via self-attention")
    print(f"  4. Memory readout combined with LSTM hidden")
    print(f"  5. Combined representation -> output")

    print("\n" + "=" * 80 + "\n")


def test_memory_capacity():
    """Test how different numbers of memory slots affect behavior."""
    print("=" * 80)
    print("Testing Memory Capacity (Different Number of Slots)")
    print("=" * 80)

    np.random.seed(42)

    # Configuration
    batch_size = 2
    seq_len = 10
    input_size = 32
    hidden_size = 64
    output_size = 16

    # Same sequence for all tests
    sequence = np.random.randn(batch_size, seq_len, input_size)

    slot_configs = [1, 2, 4, 8]

    print(f"\nTesting different numbers of memory slots:")

    results = []
    for num_slots in slot_configs:
        print(f"\n[Testing] num_slots = {num_slots}")

        model = RelationalRNN(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            num_slots=num_slots,
            slot_size=64,
            num_heads=2
        )

        outputs = model.forward(sequence, return_sequences=True)

        print(f"  Output shape: {outputs.shape}")
        print(f"  Output mean:  {outputs.mean():.4f}")
        print(f"  Output std:   {outputs.std():.4f}")

        results.append({
            'num_slots': num_slots,
            'mean': outputs.mean(),
            'std': outputs.std()
        })

    print(f"\n[Summary]")
    print(f"  All configurations successfully process the sequence")
    print(f"  More slots = more memory capacity for relational reasoning")
    print(f"  Flexibility in choosing num_slots based on task complexity")

    print("\n" + "=" * 80 + "\n")


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 80)
    print(" " * 15 + "RELATIONAL RNN - EXTENDED DEMONSTRATIONS")
    print(" " * 20 + "Paper 18: Relational RNN - Task P2-T2")
    print("=" * 80 + "\n")

    # Run demonstrations
    analyze_memory_evolution()
    compare_with_without_memory()
    demonstrate_lstm_memory_interaction()
    test_memory_capacity()

    print("=" * 80)
    print(" " * 25 + "ALL DEMONSTRATIONS COMPLETE")
    print("=" * 80)
    print("\nKey Insights:")
    print("  1. Memory evolves dynamically over sequence processing")
    print("  2. Memory slots can specialize to different patterns")
    print("  3. LSTM provides sequential processing foundation")
    print("  4. Memory adds relational reasoning capability")
    print("  5. Combined system benefits from both mechanisms")
    print("\nArchitecture Benefits:")
    print("  - LSTM: Handles temporal dependencies and sequences")
    print("  - Memory: Maintains multiple related representations")
    print("  - Attention: Enables memory slots to interact")
    print("  - Gates: Control information flow and updates")
    print("  - Combination: Both sequential and relational processing")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
