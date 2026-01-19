"""
Relational Memory Core Module
Paper 18: Relational RNN - Implementation Task P2-T1

This module implements the core innovation of the Relational RNN paper:
a set of memory slots that interact via multi-head self-attention to enable
relational reasoning across stored information.

The Relational Memory Core maintains a set of memory vectors (slots) that
can attend to each other, allowing information sharing and relational
reasoning. This is the key difference from traditional RNNs that use a
single hidden state vector.

Educational implementation for the Sutskever 30 papers project.
"""

import numpy as np
from attention_mechanism import multi_head_attention, init_attention_params


def layer_norm(x, gamma=None, beta=None, eps=1e-6):
    """
    Layer Normalization.

    Normalizes across the feature dimension for each example independently.
    This helps stabilize training and allows each layer to adapt its inputs
    to have zero mean and unit variance.

    Args:
        x: input tensor, shape (..., d_model)
        gamma: scale parameter, shape (d_model,)
        beta: shift parameter, shape (d_model,)
        eps: small constant for numerical stability

    Returns:
        normalized output, shape (..., d_model)

    Mathematical formulation:
        mean = mean(x, axis=-1, keepdims=True)
        var = var(x, axis=-1, keepdims=True)
        x_norm = (x - mean) / sqrt(var + eps)
        output = gamma * x_norm + beta
    """
    # Compute mean and variance across the last dimension (feature dimension)
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)

    # Normalize
    x_norm = (x - mean) / np.sqrt(var + eps)

    # Scale and shift if parameters provided
    if gamma is not None and beta is not None:
        # Ensure gamma and beta have the right shape for broadcasting
        assert gamma.shape[-1] == x.shape[-1], \
            f"gamma shape {gamma.shape} incompatible with x shape {x.shape}"
        assert beta.shape[-1] == x.shape[-1], \
            f"beta shape {beta.shape} incompatible with x shape {x.shape}"

        x_norm = gamma * x_norm + beta

    return x_norm


def gated_update(old_value, new_value, gate_weights=None):
    """
    Gated update mechanism for memory.

    Uses a learned gate to interpolate between old and new memory values.
    This allows the model to learn when to retain old information vs.
    incorporate new information.

    Args:
        old_value: previous memory, shape (..., d_model)
        new_value: candidate new memory, shape (..., d_model)
        gate_weights: optional gate parameters, shape (d_model * 2, d_model)
                     If None, returns new_value directly

    Returns:
        updated memory, shape (..., d_model)

    Mathematical formulation:
        gate_input = concat([old_value, new_value], axis=-1)
        gate = sigmoid(gate_input @ gate_weights)
        output = gate * new_value + (1 - gate) * old_value
    """
    if gate_weights is None:
        # No gating, just return new value
        return new_value

    assert old_value.shape == new_value.shape, \
        f"Shape mismatch: old_value {old_value.shape} vs new_value {new_value.shape}"

    d_model = old_value.shape[-1]

    # Concatenate old and new values
    gate_input = np.concatenate([old_value, new_value], axis=-1)
    # gate_input: (..., d_model * 2)

    # Compute gate values using sigmoid
    # gate_weights: (d_model * 2, d_model)
    gate_logits = np.matmul(gate_input, gate_weights)  # (..., d_model)
    gate = 1.0 / (1.0 + np.exp(-gate_logits))  # sigmoid

    # Gated combination
    output = gate * new_value + (1.0 - gate) * old_value

    return output


def init_memory(batch_size, num_slots, slot_size, init_std=0.1):
    """
    Initialize memory slots.

    Creates initial memory state for the relational memory core.
    Memory is initialized with small random values to break symmetry.

    Args:
        batch_size: number of examples in batch
        num_slots: number of memory slots
        slot_size: dimension of each memory slot
        init_std: standard deviation for initialization

    Returns:
        memory: shape (batch_size, num_slots, slot_size)
    """
    memory = np.random.randn(batch_size, num_slots, slot_size) * init_std
    return memory


class RelationalMemory:
    """
    Relational Memory Core using multi-head self-attention.

    This is the core innovation of the Relational RNN paper. Instead of
    maintaining a single hidden state vector (like traditional RNNs), the
    Relational Memory maintains multiple memory slots that can interact
    via self-attention. This allows the model to:

    1. Store multiple pieces of information simultaneously
    2. Enable relational reasoning by allowing slots to attend to each other
    3. Dynamically route information between slots based on relevance

    Architecture:
        1. Multi-head self-attention across memory slots
        2. Residual connection around attention
        3. Layer normalization for stability
        4. Optional gated update to control information flow
        5. Optional input incorporation via attention

    The memory acts as a relational reasoning module that can maintain
    and manipulate structured representations.
    """

    def __init__(self, num_slots=8, slot_size=64, num_heads=4,
                 use_gate=True, use_input_attention=True):
        """
        Initialize Relational Memory Core.

        Args:
            num_slots: number of memory slots (default: 8)
            slot_size: dimension of each memory slot (default: 64)
            num_heads: number of attention heads (default: 4)
            use_gate: whether to use gated update (default: True)
            use_input_attention: whether to incorporate input via attention (default: True)
        """
        self.num_slots = num_slots
        self.slot_size = slot_size
        self.num_heads = num_heads
        self.use_gate = use_gate
        self.use_input_attention = use_input_attention

        # Check that slot_size is divisible by num_heads
        assert slot_size % num_heads == 0, \
            f"slot_size ({slot_size}) must be divisible by num_heads ({num_heads})"

        # Initialize attention parameters for self-attention
        self.attn_params = init_attention_params(slot_size, num_heads)

        # Initialize layer normalization parameters
        self.ln1_gamma = np.ones(slot_size)
        self.ln1_beta = np.zeros(slot_size)
        self.ln2_gamma = np.ones(slot_size)
        self.ln2_beta = np.zeros(slot_size)

        # Initialize gate parameters if using gating
        if use_gate:
            # Gate takes concatenated [old, new] and outputs gate values
            std = np.sqrt(2.0 / (slot_size * 2 + slot_size))
            self.gate_weights = np.random.randn(slot_size * 2, slot_size) * std
        else:
            self.gate_weights = None

        # Initialize parameters for input incorporation if used
        if use_input_attention:
            self.ln_input_gamma = np.ones(slot_size)
            self.ln_input_beta = np.zeros(slot_size)

    def forward(self, memory, input_vec=None):
        """
        Forward pass through relational memory core.

        Args:
            memory: current memory state, shape (batch, num_slots, slot_size)
            input_vec: optional input to incorporate, shape (batch, input_size)
                      If provided and use_input_attention=True, input is attended to

        Returns:
            updated_memory: new memory state, shape (batch, num_slots, slot_size)
            attention_weights: self-attention weights, shape (batch, num_heads, num_slots, num_slots)

        Algorithm:
            1. Self-attention across memory slots
            2. Add residual connection
            3. Layer normalization
            4. Optional: attend to input
            5. Optional: gated update
        """
        # Validate input shapes
        assert memory.ndim == 3, \
            f"memory must be 3D (batch, num_slots, slot_size), got {memory.shape}"
        batch_size, num_slots, slot_size = memory.shape
        assert num_slots == self.num_slots, \
            f"Expected {self.num_slots} slots, got {num_slots}"
        assert slot_size == self.slot_size, \
            f"Expected slot_size {self.slot_size}, got {slot_size}"

        # Store original memory for residual and gating
        memory_orig = memory

        # Step 1: Multi-head self-attention across memory slots
        # Memory attends to itself: Q=K=V=memory
        attn_output, attn_weights = multi_head_attention(
            Q=memory,
            K=memory,
            V=memory,
            num_heads=self.num_heads,
            W_q=self.attn_params['W_q'],
            W_k=self.attn_params['W_k'],
            W_v=self.attn_params['W_v'],
            W_o=self.attn_params['W_o'],
            mask=None
        )
        # attn_output: (batch, num_slots, slot_size)
        # attn_weights: (batch, num_heads, num_slots, num_slots)

        # Step 2: Residual connection
        memory = memory_orig + attn_output

        # Step 3: Layer normalization
        memory = layer_norm(
            memory,
            gamma=self.ln1_gamma,
            beta=self.ln1_beta
        )

        # Step 4: Optional input attention
        # If input is provided and we're using input attention,
        # incorporate input into memory via broadcasting and gating
        if input_vec is not None and self.use_input_attention:
            # Input_vec: (batch, input_size)
            # Need to make it compatible with memory

            # Project input to slot_size if needed
            if input_vec.shape[-1] != self.slot_size:
                # Simple linear projection
                input_size = input_vec.shape[-1]
                if not hasattr(self, 'input_projection'):
                    # Initialize projection matrix
                    std = np.sqrt(2.0 / (input_size + self.slot_size))
                    self.input_projection = np.random.randn(input_size, self.slot_size) * std

                # Project input
                input_vec_proj = np.matmul(input_vec, self.input_projection)
            else:
                input_vec_proj = input_vec

            # Broadcast input to all memory slots: (batch, num_slots, slot_size)
            # Each slot gets to see the same input
            input_broadcast = np.tile(input_vec_proj[:, np.newaxis, :], (1, self.num_slots, 1))

            # Combine memory and input via a simple gating mechanism
            # This is a simplified approach compared to full cross-attention
            # which would require handling different sequence lengths

            # Concatenate memory and input, then project
            memory_input_concat = np.concatenate([memory, input_broadcast], axis=-1)
            # Shape: (batch, num_slots, slot_size * 2)

            # Project back to slot_size
            if not hasattr(self, 'input_combine_weights'):
                std = np.sqrt(2.0 / (self.slot_size * 2 + self.slot_size))
                self.input_combine_weights = np.random.randn(self.slot_size * 2, self.slot_size) * std

            input_contribution = np.matmul(memory_input_concat, self.input_combine_weights)
            # Shape: (batch, num_slots, slot_size)

            # Add residual and normalize
            memory_before_input = memory
            memory = memory_before_input + input_contribution
            memory = layer_norm(
                memory,
                gamma=self.ln_input_gamma,
                beta=self.ln_input_beta
            )

        # Step 5: Optional gated update
        if self.use_gate and self.gate_weights is not None:
            memory = gated_update(
                old_value=memory_orig,
                new_value=memory,
                gate_weights=self.gate_weights
            )

        return memory, attn_weights

    def reset_memory(self, batch_size, init_std=0.1):
        """
        Create fresh memory state.

        Args:
            batch_size: number of examples in batch
            init_std: standard deviation for initialization

        Returns:
            memory: shape (batch_size, num_slots, slot_size)
        """
        return init_memory(batch_size, self.num_slots, self.slot_size, init_std)


# ============================================================================
# Test Functions
# ============================================================================

def test_layer_norm():
    """Test layer normalization."""
    print("=" * 80)
    print("Testing Layer Normalization")
    print("=" * 80)

    np.random.seed(42)

    # Test basic layer norm
    batch_size = 2
    seq_len = 5
    d_model = 8

    x = np.random.randn(batch_size, seq_len, d_model) * 2.0 + 3.0

    print(f"\nInput shape: {x.shape}")
    print(f"Input mean (approx): {np.mean(x):.4f}")
    print(f"Input std (approx): {np.std(x):.4f}")

    # Test without gamma/beta
    print("\n[Test 1] Layer norm without scale/shift")
    x_norm = layer_norm(x)

    # Check that each example has been normalized
    for b in range(batch_size):
        for s in range(seq_len):
            vec_mean = np.mean(x_norm[b, s])
            vec_std = np.std(x_norm[b, s])
            assert np.abs(vec_mean) < 1e-6, f"Mean not close to 0: {vec_mean}"
            assert np.abs(vec_std - 1.0) < 1e-6, f"Std not close to 1: {vec_std}"

    print(f"  Output mean (approx): {np.mean(x_norm):.6f}")
    print(f"  Output std (approx): {np.std(x_norm):.4f}")
    print(f"  Each vector normalized: PASS")

    # Test with gamma/beta
    print("\n[Test 2] Layer norm with scale/shift")
    gamma = np.ones(d_model) * 2.0
    beta = np.ones(d_model) * 0.5

    x_norm_scaled = layer_norm(x, gamma=gamma, beta=beta)

    # Check that scaling is applied
    # Each normalized vector should be scaled and shifted
    for b in range(batch_size):
        for s in range(seq_len):
            vec_mean = np.mean(x_norm_scaled[b, s])
            # Mean should be close to mean(beta) = 0.5
            assert np.abs(vec_mean - 0.5) < 0.1, f"Mean not close to 0.5: {vec_mean}"

    print(f"  Output mean (approx): {np.mean(x_norm_scaled):.4f}")
    print(f"  Scaling applied: PASS")

    print("\n" + "=" * 80)
    print("Layer Normalization: ALL TESTS PASSED")
    print("=" * 80 + "\n")


def test_gated_update():
    """Test gated update mechanism."""
    print("=" * 80)
    print("Testing Gated Update")
    print("=" * 80)

    np.random.seed(42)

    batch_size = 2
    num_slots = 4
    d_model = 8

    old_value = np.random.randn(batch_size, num_slots, d_model)
    new_value = np.random.randn(batch_size, num_slots, d_model)

    print(f"\nOld value shape: {old_value.shape}")
    print(f"New value shape: {new_value.shape}")

    # Test without gate (should return new_value)
    print("\n[Test 1] Update without gate")
    result = gated_update(old_value, new_value, gate_weights=None)
    assert np.allclose(result, new_value), "Without gate should return new_value"
    print(f"  Returns new_value: PASS")

    # Test with gate
    print("\n[Test 2] Update with gate")
    std = np.sqrt(2.0 / (d_model * 2 + d_model))
    gate_weights = np.random.randn(d_model * 2, d_model) * std

    result_gated = gated_update(old_value, new_value, gate_weights=gate_weights)

    print(f"  Output shape: {result_gated.shape}")
    assert result_gated.shape == old_value.shape, "Output shape mismatch"
    print(f"  Output shape correct: PASS")

    # Check that output is a combination of old and new
    # It should be different from both old_value and new_value (in general)
    # unless gate is all 0s or all 1s
    assert not np.allclose(result_gated, old_value), "Output should differ from old_value"
    assert not np.allclose(result_gated, new_value), "Output should differ from new_value"
    print(f"  Output is combination of old and new: PASS")

    # Check numerical stability
    assert not np.any(np.isnan(result_gated)), "Output contains NaN"
    assert not np.any(np.isinf(result_gated)), "Output contains Inf"
    print(f"  No NaN/Inf in output: PASS")

    print("\n" + "=" * 80)
    print("Gated Update: ALL TESTS PASSED")
    print("=" * 80 + "\n")


def test_init_memory():
    """Test memory initialization."""
    print("=" * 80)
    print("Testing Memory Initialization")
    print("=" * 80)

    np.random.seed(42)

    batch_size = 2
    num_slots = 4
    slot_size = 64

    print(f"\nParameters:")
    print(f"  batch_size: {batch_size}")
    print(f"  num_slots: {num_slots}")
    print(f"  slot_size: {slot_size}")

    memory = init_memory(batch_size, num_slots, slot_size)

    print(f"\nMemory shape: {memory.shape}")
    assert memory.shape == (batch_size, num_slots, slot_size), \
        f"Shape mismatch: expected {(batch_size, num_slots, slot_size)}, got {memory.shape}"
    print(f"  Shape correct: PASS")

    # Check initialization statistics
    print(f"\nMemory statistics:")
    print(f"  Mean: {np.mean(memory):.6f}")
    print(f"  Std: {np.std(memory):.4f}")

    # Should be roughly zero mean, small std
    assert np.abs(np.mean(memory)) < 0.1, "Mean too far from 0"
    assert 0.05 < np.std(memory) < 0.2, "Std outside expected range"
    print(f"  Statistics reasonable: PASS")

    print("\n" + "=" * 80)
    print("Memory Initialization: ALL TESTS PASSED")
    print("=" * 80 + "\n")


def test_relational_memory():
    """Test the RelationalMemory class."""
    print("=" * 80)
    print("Testing Relational Memory Core")
    print("=" * 80)

    np.random.seed(42)

    # Test parameters (as specified in task)
    batch_size = 2
    num_slots = 4
    slot_size = 64
    num_heads = 2

    print(f"\nParameters:")
    print(f"  batch_size: {batch_size}")
    print(f"  num_slots: {num_slots}")
    print(f"  slot_size: {slot_size}")
    print(f"  num_heads: {num_heads}")

    # Initialize relational memory
    print("\n[Test 1] Initialization")
    rm = RelationalMemory(
        num_slots=num_slots,
        slot_size=slot_size,
        num_heads=num_heads,
        use_gate=True,
        use_input_attention=True
    )

    print(f"  RelationalMemory created")
    print(f"  num_slots: {rm.num_slots}")
    print(f"  slot_size: {rm.slot_size}")
    print(f"  num_heads: {rm.num_heads}")
    print(f"  use_gate: {rm.use_gate}")
    print(f"  use_input_attention: {rm.use_input_attention}")

    # Verify parameters initialized
    assert rm.attn_params is not None, "Attention params not initialized"
    assert rm.gate_weights is not None, "Gate weights not initialized"
    print(f"  All parameters initialized: PASS")

    # Test memory reset
    print("\n[Test 2] Memory reset")
    memory = rm.reset_memory(batch_size)
    print(f"  Memory shape: {memory.shape}")
    assert memory.shape == (batch_size, num_slots, slot_size), \
        f"Memory shape mismatch: expected {(batch_size, num_slots, slot_size)}, got {memory.shape}"
    print(f"  Memory shape correct: PASS")

    # Test forward pass without input
    print("\n[Test 3] Forward pass without input")
    updated_memory, attn_weights = rm.forward(memory)

    print(f"  Updated memory shape: {updated_memory.shape}")
    print(f"  Attention weights shape: {attn_weights.shape}")

    assert updated_memory.shape == (batch_size, num_slots, slot_size), \
        f"Updated memory shape mismatch"
    assert attn_weights.shape == (batch_size, num_heads, num_slots, num_slots), \
        f"Attention weights shape mismatch"
    print(f"  Output shapes correct: PASS")

    # Check attention weights sum to 1
    attn_sums = np.sum(attn_weights, axis=-1)
    assert np.allclose(attn_sums, 1.0), "Attention weights don't sum to 1"
    print(f"  Attention weights sum to 1: PASS")

    # Check for NaN/Inf
    assert not np.any(np.isnan(updated_memory)), "Memory contains NaN"
    assert not np.any(np.isinf(updated_memory)), "Memory contains Inf"
    assert not np.any(np.isnan(attn_weights)), "Attention weights contain NaN"
    assert not np.any(np.isinf(attn_weights)), "Attention weights contain Inf"
    print(f"  No NaN/Inf in outputs: PASS")

    # Test forward pass with input
    print("\n[Test 4] Forward pass with input")
    input_size = 32
    input_vec = np.random.randn(batch_size, input_size)

    updated_memory_with_input, attn_weights_with_input = rm.forward(memory, input_vec)

    print(f"  Input shape: {input_vec.shape}")
    print(f"  Updated memory shape: {updated_memory_with_input.shape}")
    print(f"  Attention weights shape: {attn_weights_with_input.shape}")

    assert updated_memory_with_input.shape == (batch_size, num_slots, slot_size), \
        f"Updated memory shape mismatch"
    print(f"  Output shape correct: PASS")

    # Memory should be different when input is provided
    assert not np.allclose(updated_memory, updated_memory_with_input), \
        "Input should affect memory"
    print(f"  Input affects memory: PASS")

    # Test multiple forward passes (simulating sequence)
    print("\n[Test 5] Multiple forward passes")
    memory_seq = rm.reset_memory(batch_size)
    memories = [memory_seq]

    for t in range(5):
        input_t = np.random.randn(batch_size, input_size)
        memory_seq, _ = rm.forward(memory_seq, input_t)
        memories.append(memory_seq)

    print(f"  Processed {len(memories)-1} timesteps")

    # Check that memory evolves over time
    for t in range(1, len(memories)):
        assert not np.allclose(memories[0], memories[t]), \
            f"Memory should change at timestep {t}"
    print(f"  Memory evolves over time: PASS")

    # Test without gating
    print("\n[Test 6] Without gating")
    rm_no_gate = RelationalMemory(
        num_slots=num_slots,
        slot_size=slot_size,
        num_heads=num_heads,
        use_gate=False,
        use_input_attention=False
    )

    memory_no_gate = rm_no_gate.reset_memory(batch_size)
    updated_no_gate, _ = rm_no_gate.forward(memory_no_gate)

    print(f"  Forward pass without gate: PASS")
    assert updated_no_gate.shape == (batch_size, num_slots, slot_size), \
        "Shape mismatch without gate"
    print(f"  Output shape correct: PASS")

    # Test different configurations
    print("\n[Test 7] Different configurations")
    configs = [
        {'num_slots': 8, 'slot_size': 64, 'num_heads': 4},
        {'num_slots': 4, 'slot_size': 128, 'num_heads': 8},
        {'num_slots': 16, 'slot_size': 32, 'num_heads': 2},
    ]

    for i, config in enumerate(configs):
        test_rm = RelationalMemory(**config)
        test_memory = test_rm.reset_memory(batch_size)
        test_updated, test_attn = test_rm.forward(test_memory)

        assert test_updated.shape == (batch_size, config['num_slots'], config['slot_size']), \
            f"Config {i} failed"
        assert test_attn.shape == (batch_size, config['num_heads'], config['num_slots'], config['num_slots']), \
            f"Config {i} attention shape failed"
        print(f"  Config {i+1} (slots={config['num_slots']}, size={config['slot_size']}, heads={config['num_heads']}): PASS")

    print("\n" + "=" * 80)
    print("Relational Memory Core: ALL TESTS PASSED")
    print("=" * 80 + "\n")


def demonstrate_relational_reasoning():
    """Demonstrate how relational memory enables reasoning."""
    print("=" * 80)
    print("Demonstrating Relational Reasoning Capabilities")
    print("=" * 80)

    np.random.seed(42)

    batch_size = 1
    num_slots = 4
    slot_size = 64
    num_heads = 2

    print("\nScenario: Memory slots representing entities that need to interact")
    print(f"  num_slots: {num_slots} (e.g., 4 objects being tracked)")
    print(f"  slot_size: {slot_size} (feature dimension)")
    print(f"  num_heads: {num_heads} (different types of relationships)")

    rm = RelationalMemory(
        num_slots=num_slots,
        slot_size=slot_size,
        num_heads=num_heads,
        use_gate=True,
        use_input_attention=True
    )

    # Initialize memory with distinct patterns for each slot
    memory = rm.reset_memory(batch_size)

    # Simulate making slots somewhat different
    for slot in range(num_slots):
        memory[0, slot, :] += np.random.randn(slot_size) * 0.5

    print("\n[Observation 1] Initial memory state")
    print(f"  Memory shape: {memory.shape}")
    print(f"  Memory initialized with distinct patterns per slot")

    # Forward pass to see attention patterns
    updated_memory, attn_weights = rm.forward(memory)

    print("\n[Observation 2] Attention patterns after one step")
    print(f"  Attention weights shape: {attn_weights.shape}")
    print(f"\n  Attention matrix (head 0):")
    print(f"  Rows = query slots, Cols = key slots")
    print(f"  Values show how much each slot attends to others\n")

    # Display attention for first head
    attn_head0 = attn_weights[0, 0]  # (num_slots, num_slots)
    for i in range(num_slots):
        row_str = "  Slot " + str(i) + " -> ["
        for j in range(num_slots):
            row_str += f"{attn_head0[i, j]:.3f}"
            if j < num_slots - 1:
                row_str += ", "
        row_str += "]"
        print(row_str)

    # Check which slots have high mutual attention
    print("\n[Observation 3] Relational interactions")
    threshold = 0.3
    for i in range(num_slots):
        for j in range(i + 1, num_slots):
            mutual_attn = attn_head0[i, j] + attn_head0[j, i]
            if mutual_attn > threshold:
                print(f"  Strong interaction between Slot {i} and Slot {j} (score: {mutual_attn:.3f})")

    # Simulate sequence of inputs
    print("\n[Observation 4] Evolution with inputs")
    memory_t = memory
    input_sequence = [np.random.randn(batch_size, 32) for _ in range(3)]

    for t, input_t in enumerate(input_sequence):
        memory_t, attn_t = rm.forward(memory_t, input_t)
        mean_attn = np.mean(attn_t[0, 0])
        print(f"  Step {t+1}: Mean attention = {mean_attn:.4f}")

    print("\n[Key Insights]")
    print("  1. Memory slots can attend to each other, enabling relational reasoning")
    print("  2. Different attention heads can capture different types of relations")
    print("  3. Memory evolves over time while maintaining multiple representations")
    print("  4. This enables reasoning about relationships between stored entities")
    print("  5. Unlike single-vector RNN states, can maintain distinct concepts simultaneously")

    print("\n" + "=" * 80 + "\n")


def main():
    """Run all tests and demonstrations."""
    print("\n" + "=" * 80)
    print(" " * 20 + "RELATIONAL MEMORY CORE TEST SUITE")
    print(" " * 25 + "Paper 18: Relational RNN - Task P2-T1")
    print("=" * 80 + "\n")

    # Run tests
    test_layer_norm()
    test_gated_update()
    test_init_memory()
    test_relational_memory()
    demonstrate_relational_reasoning()

    print("=" * 80)
    print(" " * 25 + "ALL TESTS COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print("\nSummary of Implementation:")
    print("  - layer_norm(): Normalizes activations for training stability")
    print("  - gated_update(): Controls information flow with learned gates")
    print("  - init_memory(): Initializes memory slots with small random values")
    print("  - RelationalMemory class: Core module with multi-head self-attention")
    print("\nKey Features:")
    print("  - Multi-head self-attention across memory slots")
    print("  - Residual connections for gradient flow")
    print("  - Layer normalization for stability")
    print("  - Optional gated updates for selective memory retention")
    print("  - Optional input attention for incorporating new information")
    print("\nRelational Reasoning Aspect:")
    print("  - Memory slots can attend to each other via self-attention")
    print("  - Enables modeling relationships between stored entities")
    print("  - Different attention heads capture different relational patterns")
    print("  - Maintains multiple distinct representations simultaneously")
    print("  - Superior to single-vector hidden states for complex reasoning")
    print("\nAll tests passed with batch=2, slots=4, slot_size=64, heads=2")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
