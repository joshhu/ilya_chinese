"""
Relational RNN Cell - Combining LSTM with Relational Memory

This module implements a Relational RNN that combines:
1. Standard LSTM for sequential processing
2. Relational Memory with multi-head self-attention for relational reasoning

Paper 18: Relational RNN - Implementation Task P2-T2

Architecture:
- LSTM processes sequential inputs and maintains hidden/cell states
- Relational memory maintains a set of memory slots that interact via attention
- LSTM hidden state is projected and used to update the relational memory
- Memory readout is combined with LSTM output for final predictions

Educational implementation using NumPy only.
"""

import numpy as np
from lstm_baseline import LSTMCell, xavier_initializer, orthogonal_initializer
from attention_mechanism import multi_head_attention, init_attention_params


class RelationalMemory:
    """
    Relational memory module using multi-head self-attention.

    The memory consists of a set of slots that interact via attention mechanism.
    This allows the model to maintain and reason about multiple related pieces
    of information simultaneously.

    Architecture:
        1. Memory slots interact via multi-head self-attention
        2. Gate mechanism controls memory updates
        3. Residual connections preserve information
    """

    def __init__(self, num_slots=4, slot_size=64, num_heads=2, input_size=None):
        """
        Initialize relational memory.

        Args:
            num_slots: number of memory slots
            slot_size: dimension of each memory slot
            num_heads: number of attention heads
            input_size: dimension of input to memory (if None, equals slot_size)
        """
        self.num_slots = num_slots
        self.slot_size = slot_size
        self.num_heads = num_heads
        self.input_size = input_size if input_size is not None else slot_size

        assert slot_size % num_heads == 0, \
            f"slot_size ({slot_size}) must be divisible by num_heads ({num_heads})"

        # Multi-head attention parameters for memory interaction
        self.attn_params = init_attention_params(slot_size, num_heads)

        # Input projection: project input to memory space
        if self.input_size != slot_size:
            self.W_input = xavier_initializer((slot_size, self.input_size))
            self.b_input = np.zeros((slot_size, 1))
        else:
            self.W_input = None
            self.b_input = None

        # Gate for controlling memory updates
        # Gates decide how much to update vs. preserve existing memory
        gate_input_size = slot_size + self.input_size
        self.W_gate = xavier_initializer((slot_size, gate_input_size))
        self.b_gate = np.zeros((slot_size, 1))

        # Update projection: combines attention output with input
        self.W_update = xavier_initializer((slot_size, slot_size))
        self.b_update = np.zeros((slot_size, 1))

    def forward(self, memory_prev, input_vec=None):
        """
        Update memory using self-attention and optional input.

        Args:
            memory_prev: previous memory state, shape (batch, num_slots, slot_size)
            input_vec: optional input to incorporate, shape (batch, input_size)

        Returns:
            memory_new: updated memory, shape (batch, num_slots, slot_size)

        Process:
            1. Apply multi-head self-attention to memory slots
            2. If input provided, project it and add to memory
            3. Apply gated update to control information flow
            4. Residual connection to preserve existing memory
        """
        batch_size = memory_prev.shape[0]

        # Step 1: Multi-head self-attention over memory slots
        # memory_prev: (batch, num_slots, slot_size)
        # Self-attention: each slot attends to all other slots
        attended_memory, attn_weights = multi_head_attention(
            Q=memory_prev,
            K=memory_prev,
            V=memory_prev,
            num_heads=self.num_heads,
            W_q=self.attn_params['W_q'],
            W_k=self.attn_params['W_k'],
            W_v=self.attn_params['W_v'],
            W_o=self.attn_params['W_o']
        )
        # attended_memory: (batch, num_slots, slot_size)

        # Step 2: Project and incorporate input if provided
        if input_vec is not None:
            # input_vec: (batch, input_size)
            # Project to slot_size if needed
            if self.W_input is not None:
                # Reshape for matrix multiplication
                # input_vec: (batch, input_size) -> (input_size, batch)
                input_vec_T = input_vec.T  # (input_size, batch)
                # W_input @ input_vec_T: (slot_size, batch)
                projected_input = self.W_input @ input_vec_T + self.b_input
                # projected_input: (slot_size, batch) -> (batch, slot_size)
                projected_input = projected_input.T
            else:
                projected_input = input_vec
            # projected_input: (batch, slot_size)

            # Add projected input to first memory slot
            # This is a simple way to inject external information
            attended_memory[:, 0, :] = attended_memory[:, 0, :] + projected_input

        # Step 3: Apply update projection with nonlinearity
        # Process each slot independently
        # attended_memory: (batch, num_slots, slot_size)
        # Reshape to (batch * num_slots, slot_size) for processing
        attended_flat = attended_memory.reshape(batch_size * self.num_slots, self.slot_size)
        # attended_flat: (batch * num_slots, slot_size) -> (slot_size, batch * num_slots)
        attended_flat_T = attended_flat.T

        # Apply update transformation
        # W_update @ attended_flat_T: (slot_size, batch * num_slots)
        updated_flat_T = np.tanh(self.W_update @ attended_flat_T + self.b_update)
        # updated_flat_T: (slot_size, batch * num_slots) -> (batch * num_slots, slot_size)
        updated_flat = updated_flat_T.T
        # Reshape back: (batch, num_slots, slot_size)
        updated_memory = updated_flat.reshape(batch_size, self.num_slots, self.slot_size)

        # Step 4: Gated update
        if input_vec is not None:
            # Compute gate values
            # For each slot, decide how much to update based on attended memory and input
            gates_list = []
            for slot_idx in range(self.num_slots):
                # Get attended memory for this slot: (batch, slot_size)
                slot_attended = attended_memory[:, slot_idx, :]  # (batch, slot_size)

                # Concatenate with input for gating decision
                # gate_input: (batch, slot_size + input_size)
                gate_input = np.concatenate([slot_attended, input_vec], axis=1)
                # gate_input: (batch, slot_size + input_size) -> (slot_size + input_size, batch)
                gate_input_T = gate_input.T

                # Compute gate: (slot_size, batch)
                gate_T = self._sigmoid(self.W_gate @ gate_input_T + self.b_gate)
                # gate_T: (slot_size, batch) -> (batch, slot_size)
                gate = gate_T.T
                gates_list.append(gate)

            # Stack gates: (batch, num_slots, slot_size)
            gates = np.stack(gates_list, axis=1)
        else:
            # No input, use constant gate value
            gates = np.ones((batch_size, self.num_slots, self.slot_size)) * 0.5

        # Step 5: Apply gated residual connection
        # memory_new = gate * updated + (1 - gate) * old
        memory_new = gates * updated_memory + (1 - gates) * memory_prev

        return memory_new

    @staticmethod
    def _sigmoid(x):
        """Numerically stable sigmoid function."""
        return np.where(
            x >= 0,
            1 / (1 + np.exp(-x)),
            np.exp(x) / (1 + np.exp(x))
        )


class RelationalRNNCell:
    """
    Relational RNN Cell combining LSTM with relational memory.

    This cell processes one time step by:
    1. Running LSTM on input to get hidden state
    2. Using LSTM hidden state to update relational memory
    3. Reading from memory and combining with LSTM output

    The combination allows both sequential processing (LSTM) and
    relational reasoning (memory with attention).
    """

    def __init__(self, input_size, hidden_size, num_slots=4, slot_size=64, num_heads=2):
        """
        Initialize Relational RNN Cell.

        Args:
            input_size: dimension of input features
            hidden_size: dimension of LSTM hidden state
            num_slots: number of relational memory slots
            slot_size: dimension of each memory slot
            num_heads: number of attention heads for memory
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_slots = num_slots
        self.slot_size = slot_size
        self.num_heads = num_heads

        # LSTM cell for sequential processing
        self.lstm_cell = LSTMCell(input_size, hidden_size)

        # Relational memory with attention
        self.memory = RelationalMemory(
            num_slots=num_slots,
            slot_size=slot_size,
            num_heads=num_heads,
            input_size=hidden_size  # Memory receives LSTM hidden state
        )

        # Projection from memory to output contribution
        # Read from memory by mean pooling across slots
        self.W_memory_read = xavier_initializer((hidden_size, slot_size))
        self.b_memory_read = np.zeros((hidden_size, 1))

        # Combine LSTM output and memory readout
        self.W_combine = xavier_initializer((hidden_size, hidden_size * 2))
        self.b_combine = np.zeros((hidden_size, 1))

    def forward(self, x, h_prev, c_prev, memory_prev):
        """
        Forward pass for one time step.

        Args:
            x: input, shape (batch, input_size)
            h_prev: previous LSTM hidden state, shape (hidden_size, batch) or (batch, hidden_size)
            c_prev: previous LSTM cell state, shape (hidden_size, batch) or (batch, hidden_size)
            memory_prev: previous memory, shape (batch, num_slots, slot_size)

        Returns:
            output: combined output, shape (batch, hidden_size)
            h_new: new LSTM hidden state, shape (hidden_size, batch)
            c_new: new LSTM cell state, shape (hidden_size, batch)
            memory_new: new memory state, shape (batch, num_slots, slot_size)

        Process:
            1. LSTM forward pass: x -> h_new, c_new
            2. Use h_new to update memory: h_new -> memory_new
            3. Read from memory (mean pool across slots)
            4. Combine LSTM hidden state with memory readout
        """
        batch_size = x.shape[0]

        # Handle input shape for h_prev and c_prev
        # LSTM expects (hidden_size, batch)
        if h_prev.ndim == 2 and h_prev.shape[0] == batch_size:
            # Convert (batch, hidden_size) -> (hidden_size, batch)
            h_prev = h_prev.T
        if c_prev.ndim == 2 and c_prev.shape[0] == batch_size:
            # Convert (batch, hidden_size) -> (hidden_size, batch)
            c_prev = c_prev.T

        # Step 1: LSTM forward pass
        # x: (batch, input_size)
        # h_prev, c_prev: (hidden_size, batch)
        h_new, c_new = self.lstm_cell.forward(x, h_prev, c_prev)
        # h_new, c_new: (hidden_size, batch)

        # Step 2: Update relational memory using LSTM hidden state
        # h_new: (hidden_size, batch) -> (batch, hidden_size)
        h_new_for_memory = h_new.T

        # Update memory with LSTM hidden state as input
        memory_new = self.memory.forward(memory_prev, h_new_for_memory)
        # memory_new: (batch, num_slots, slot_size)

        # Step 3: Read from memory
        # Simple strategy: mean pool across memory slots
        memory_readout = np.mean(memory_new, axis=1)  # (batch, slot_size)

        # Project memory readout to hidden_size
        # memory_readout: (batch, slot_size) -> (slot_size, batch)
        memory_readout_T = memory_readout.T
        # W_memory_read @ memory_readout_T: (hidden_size, batch)
        memory_contribution_T = self.W_memory_read @ memory_readout_T + self.b_memory_read
        # memory_contribution: (batch, hidden_size)
        memory_contribution = memory_contribution_T.T

        # Step 4: Combine LSTM hidden state with memory contribution
        # h_new: (hidden_size, batch) -> (batch, hidden_size)
        h_new_batch_first = h_new.T

        # Concatenate LSTM hidden and memory contribution
        combined_input = np.concatenate([h_new_batch_first, memory_contribution], axis=1)
        # combined_input: (batch, hidden_size * 2)

        # Apply combination layer
        # combined_input: (batch, hidden_size * 2) -> (hidden_size * 2, batch)
        combined_input_T = combined_input.T
        # W_combine @ combined_input_T: (hidden_size, batch)
        output_T = np.tanh(self.W_combine @ combined_input_T + self.b_combine)
        # output: (batch, hidden_size)
        output = output_T.T

        return output, h_new, c_new, memory_new

    def init_memory(self, batch_size):
        """
        Initialize memory to zeros.

        Args:
            batch_size: batch size

        Returns:
            memory: initialized memory, shape (batch, num_slots, slot_size)
        """
        return np.zeros((batch_size, self.num_slots, self.slot_size))


class RelationalRNN:
    """
    Full Relational RNN for sequence processing.

    Processes sequences using RelationalRNNCell and projects to output space.
    """

    def __init__(self, input_size, hidden_size, output_size, num_slots=4, slot_size=64, num_heads=2):
        """
        Initialize Relational RNN.

        Args:
            input_size: dimension of input features
            hidden_size: dimension of LSTM hidden state
            output_size: dimension of output
            num_slots: number of memory slots
            slot_size: dimension of each memory slot
            num_heads: number of attention heads
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_slots = num_slots
        self.slot_size = slot_size
        self.num_heads = num_heads

        # Relational RNN cell
        self.cell = RelationalRNNCell(
            input_size=input_size,
            hidden_size=hidden_size,
            num_slots=num_slots,
            slot_size=slot_size,
            num_heads=num_heads
        )

        # Output projection layer
        self.W_out = xavier_initializer((output_size, hidden_size))
        self.b_out = np.zeros((output_size, 1))

    def forward(self, sequence, return_sequences=True, return_state=False):
        """
        Process a sequence through the Relational RNN.

        Args:
            sequence: input sequence, shape (batch, seq_len, input_size)
            return_sequences: if True, return outputs for all time steps
            return_state: if True, return final states

        Returns:
            outputs: shape (batch, seq_len, output_size) if return_sequences
                    else (batch, output_size)
            If return_state=True, also returns (h_final, c_final, memory_final)
        """
        batch_size, seq_len, _ = sequence.shape

        # Initialize states
        h = np.zeros((self.hidden_size, batch_size))
        c = np.zeros((self.hidden_size, batch_size))
        memory = self.cell.init_memory(batch_size)

        # Store outputs
        outputs = []

        # Process sequence
        for t in range(seq_len):
            # Get input at time t
            x_t = sequence[:, t, :]  # (batch, input_size)

            # Forward pass through cell
            cell_output, h, c, memory = self.cell.forward(x_t, h, c, memory)
            # cell_output: (batch, hidden_size)
            # h, c: (hidden_size, batch)
            # memory: (batch, num_slots, slot_size)

            # Project to output space
            # cell_output: (batch, hidden_size) -> (hidden_size, batch)
            cell_output_T = cell_output.T
            # W_out @ cell_output_T: (output_size, batch)
            out_t_T = self.W_out @ cell_output_T + self.b_out
            # out_t: (batch, output_size)
            out_t = out_t_T.T

            outputs.append(out_t)

        # Prepare return values
        if return_sequences:
            result = np.stack(outputs, axis=1)  # (batch, seq_len, output_size)
        else:
            result = outputs[-1]  # (batch, output_size)

        if return_state:
            # Return states in batch-first format
            h_final = h.T  # (batch, hidden_size)
            c_final = c.T  # (batch, hidden_size)
            memory_final = memory  # (batch, num_slots, slot_size)
            return result, h_final, c_final, memory_final
        else:
            return result


# ============================================================================
# Test Functions
# ============================================================================

def test_relational_memory():
    """Test the relational memory module."""
    print("=" * 80)
    print("Testing Relational Memory Module")
    print("=" * 80)

    np.random.seed(42)

    # Test parameters
    batch_size = 2
    num_slots = 4
    slot_size = 64
    num_heads = 2
    input_size = 32

    print(f"\nParameters:")
    print(f"  batch_size: {batch_size}")
    print(f"  num_slots: {num_slots}")
    print(f"  slot_size: {slot_size}")
    print(f"  num_heads: {num_heads}")
    print(f"  input_size: {input_size}")

    # Create relational memory
    print(f"\n[Test 1] Creating RelationalMemory...")
    rel_mem = RelationalMemory(
        num_slots=num_slots,
        slot_size=slot_size,
        num_heads=num_heads,
        input_size=input_size
    )
    print(f"  RelationalMemory created successfully")

    # Test forward pass without input
    print(f"\n[Test 2] Forward pass without input...")
    memory = np.random.randn(batch_size, num_slots, slot_size) * 0.1
    memory_new = rel_mem.forward(memory, input_vec=None)

    print(f"  Input memory shape: {memory.shape}")
    print(f"  Output memory shape: {memory_new.shape}")
    assert memory_new.shape == (batch_size, num_slots, slot_size), \
        f"Shape mismatch: expected {(batch_size, num_slots, slot_size)}, got {memory_new.shape}"
    assert not np.isnan(memory_new).any(), "NaN detected in memory output"
    assert not np.isinf(memory_new).any(), "Inf detected in memory output"
    print(f"  Shape correct, no NaN/Inf")

    # Test forward pass with input
    print(f"\n[Test 3] Forward pass with input...")
    input_vec = np.random.randn(batch_size, input_size)
    memory_new_with_input = rel_mem.forward(memory, input_vec=input_vec)

    print(f"  Input vector shape: {input_vec.shape}")
    print(f"  Output memory shape: {memory_new_with_input.shape}")
    assert memory_new_with_input.shape == (batch_size, num_slots, slot_size)
    assert not np.isnan(memory_new_with_input).any(), "NaN detected"
    assert not np.isinf(memory_new_with_input).any(), "Inf detected"
    print(f"  Shape correct, no NaN/Inf")

    # Verify memory evolves
    print(f"\n[Test 4] Verifying memory evolution...")
    assert not np.allclose(memory_new_with_input, memory), \
        "Memory should change after forward pass"
    print(f"  Memory evolves correctly")

    # Test different inputs produce different outputs
    print(f"\n[Test 5] Different inputs produce different outputs...")
    input_vec_2 = np.random.randn(batch_size, input_size) * 2.0
    memory_new_2 = rel_mem.forward(memory, input_vec=input_vec_2)
    assert not np.allclose(memory_new_with_input, memory_new_2), \
        "Different inputs should produce different memory states"
    print(f"  Different inputs -> different outputs")

    print("\n" + "=" * 80)
    print("Relational Memory: ALL TESTS PASSED")
    print("=" * 80 + "\n")


def test_relational_rnn_cell():
    """Test the Relational RNN Cell."""
    print("=" * 80)
    print("Testing Relational RNN Cell")
    print("=" * 80)

    np.random.seed(42)

    # Test parameters
    batch_size = 2
    input_size = 32
    hidden_size = 64
    num_slots = 4
    slot_size = 64
    num_heads = 2

    print(f"\nParameters:")
    print(f"  batch_size: {batch_size}")
    print(f"  input_size: {input_size}")
    print(f"  hidden_size: {hidden_size}")
    print(f"  num_slots: {num_slots}")
    print(f"  slot_size: {slot_size}")
    print(f"  num_heads: {num_heads}")

    # Create cell
    print(f"\n[Test 1] Creating RelationalRNNCell...")
    cell = RelationalRNNCell(
        input_size=input_size,
        hidden_size=hidden_size,
        num_slots=num_slots,
        slot_size=slot_size,
        num_heads=num_heads
    )
    print(f"  RelationalRNNCell created successfully")

    # Test single time step
    print(f"\n[Test 2] Single time step forward pass...")
    x = np.random.randn(batch_size, input_size)
    h_prev = np.zeros((batch_size, hidden_size))
    c_prev = np.zeros((batch_size, hidden_size))
    memory_prev = cell.init_memory(batch_size)

    output, h_new, c_new, memory_new = cell.forward(x, h_prev, c_prev, memory_prev)

    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  h_new shape: {h_new.shape}")
    print(f"  c_new shape: {c_new.shape}")
    print(f"  memory_new shape: {memory_new.shape}")

    # Verify shapes
    assert output.shape == (batch_size, hidden_size), \
        f"Output shape mismatch: expected {(batch_size, hidden_size)}, got {output.shape}"
    assert h_new.shape == (hidden_size, batch_size), \
        f"h_new shape mismatch: expected {(hidden_size, batch_size)}, got {h_new.shape}"
    assert c_new.shape == (hidden_size, batch_size), \
        f"c_new shape mismatch: expected {(hidden_size, batch_size)}, got {c_new.shape}"
    assert memory_new.shape == (batch_size, num_slots, slot_size), \
        f"memory_new shape mismatch: expected {(batch_size, num_slots, slot_size)}, got {memory_new.shape}"

    # Check for NaN/Inf
    assert not np.isnan(output).any(), "NaN in output"
    assert not np.isinf(output).any(), "Inf in output"
    assert not np.isnan(h_new).any(), "NaN in h_new"
    assert not np.isnan(c_new).any(), "NaN in c_new"
    assert not np.isnan(memory_new).any(), "NaN in memory_new"

    print(f"  All shapes correct, no NaN/Inf")

    # Test state evolution
    print(f"\n[Test 3] State evolution over multiple steps...")
    h = h_prev
    c = c_prev
    memory = memory_prev

    for step in range(3):
        x_t = np.random.randn(batch_size, input_size)
        output, h, c, memory = cell.forward(x_t, h, c, memory)
        print(f"  Step {step + 1}: output range [{output.min():.3f}, {output.max():.3f}]")

    print(f"  State evolution successful")

    # Verify memory evolves
    print(f"\n[Test 4] Verifying memory evolution...")
    assert not np.allclose(memory, memory_prev), \
        "Memory should evolve over time steps"
    print(f"  Memory evolves correctly")

    print("\n" + "=" * 80)
    print("Relational RNN Cell: ALL TESTS PASSED")
    print("=" * 80 + "\n")


def test_relational_rnn():
    """Test the full Relational RNN."""
    print("=" * 80)
    print("Testing Relational RNN (Full Sequence Processor)")
    print("=" * 80)

    np.random.seed(42)

    # Test parameters (matching task specification)
    batch_size = 2
    seq_len = 10
    input_size = 32
    hidden_size = 64
    output_size = 16
    num_slots = 4
    slot_size = 64
    num_heads = 2

    print(f"\nParameters:")
    print(f"  batch_size: {batch_size}")
    print(f"  seq_len: {seq_len}")
    print(f"  input_size: {input_size}")
    print(f"  hidden_size: {hidden_size}")
    print(f"  output_size: {output_size}")
    print(f"  num_slots: {num_slots}")
    print(f"  slot_size: {slot_size}")
    print(f"  num_heads: {num_heads}")

    # Create model
    print(f"\n[Test 1] Creating RelationalRNN...")
    model = RelationalRNN(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        num_slots=num_slots,
        slot_size=slot_size,
        num_heads=num_heads
    )
    print(f"  RelationalRNN created successfully")

    # Create random sequence
    print(f"\n[Test 2] Processing sequence (return_sequences=True)...")
    sequence = np.random.randn(batch_size, seq_len, input_size)
    print(f"  Input sequence shape: {sequence.shape}")

    outputs = model.forward(sequence, return_sequences=True)
    print(f"  Output shape: {outputs.shape}")
    print(f"  Expected: ({batch_size}, {seq_len}, {output_size})")

    assert outputs.shape == (batch_size, seq_len, output_size), \
        f"Shape mismatch: expected {(batch_size, seq_len, output_size)}, got {outputs.shape}"
    assert not np.isnan(outputs).any(), "NaN detected in outputs"
    assert not np.isinf(outputs).any(), "Inf detected in outputs"
    print(f"  Shape correct, no NaN/Inf")

    # Test return_sequences=False
    print(f"\n[Test 3] Processing sequence (return_sequences=False)...")
    output_last = model.forward(sequence, return_sequences=False)
    print(f"  Output shape: {output_last.shape}")
    print(f"  Expected: ({batch_size}, {output_size})")

    assert output_last.shape == (batch_size, output_size), \
        f"Shape mismatch: expected {(batch_size, output_size)}, got {output_last.shape}"
    print(f"  Shape correct")

    # Test return_state=True
    print(f"\n[Test 4] Processing with state return...")
    outputs, h_final, c_final, memory_final = model.forward(
        sequence, return_sequences=True, return_state=True
    )

    print(f"  Outputs shape: {outputs.shape}")
    print(f"  h_final shape: {h_final.shape}")
    print(f"  c_final shape: {c_final.shape}")
    print(f"  memory_final shape: {memory_final.shape}")

    assert h_final.shape == (batch_size, hidden_size)
    assert c_final.shape == (batch_size, hidden_size)
    assert memory_final.shape == (batch_size, num_slots, slot_size)
    print(f"  All state shapes correct")

    # Test memory evolution over sequence
    print(f"\n[Test 5] Verifying memory evolution over sequence...")
    # Process same sequence again and track memory at each step
    h = np.zeros((hidden_size, batch_size))
    c = np.zeros((hidden_size, batch_size))
    memory = model.cell.init_memory(batch_size)

    memory_states = [memory.copy()]
    for t in range(seq_len):
        x_t = sequence[:, t, :]
        _, h, c, memory = model.cell.forward(x_t, h, c, memory)
        memory_states.append(memory.copy())

    # Check that memory changes over time
    memory_changes = []
    for t in range(1, len(memory_states)):
        change = np.linalg.norm(memory_states[t] - memory_states[t-1])
        memory_changes.append(change)

    print(f"  Memory change per step (first 5):")
    for t, change in enumerate(memory_changes[:5]):
        print(f"    Step {t+1}: {change:.4f}")

    assert all(change > 0 for change in memory_changes), \
        "Memory should change at each time step"
    print(f"  Memory evolves correctly over time")

    # Test different sequences produce different outputs
    print(f"\n[Test 6] Different sequences produce different outputs...")
    sequence_2 = np.random.randn(batch_size, seq_len, input_size) * 2.0
    outputs_2 = model.forward(sequence_2, return_sequences=True)

    assert not np.allclose(outputs, outputs_2), \
        "Different input sequences should produce different outputs"
    print(f"  Different inputs -> different outputs")

    print("\n" + "=" * 80)
    print("Relational RNN: ALL TESTS PASSED")
    print("=" * 80 + "\n")

    return model


def compare_with_lstm_baseline():
    """Compare Relational RNN with LSTM baseline."""
    print("=" * 80)
    print("Comparison: Relational RNN vs. LSTM Baseline")
    print("=" * 80)

    from lstm_baseline import LSTM

    np.random.seed(42)

    # Common parameters
    batch_size = 2
    seq_len = 10
    input_size = 32
    hidden_size = 64
    output_size = 16

    # Create same input sequence for fair comparison
    sequence = np.random.randn(batch_size, seq_len, input_size)

    print(f"\nTest Configuration:")
    print(f"  batch_size: {batch_size}")
    print(f"  seq_len: {seq_len}")
    print(f"  input_size: {input_size}")
    print(f"  hidden_size: {hidden_size}")
    print(f"  output_size: {output_size}")

    # LSTM Baseline
    print(f"\n[1] LSTM Baseline")
    lstm = LSTM(input_size, hidden_size, output_size)
    lstm_outputs = lstm.forward(sequence, return_sequences=True)

    print(f"  Output shape: {lstm_outputs.shape}")
    print(f"  Output range: [{lstm_outputs.min():.3f}, {lstm_outputs.max():.3f}]")
    print(f"  Output mean: {lstm_outputs.mean():.3f}")
    print(f"  Output std: {lstm_outputs.std():.3f}")

    # Count LSTM parameters
    lstm_params = lstm.get_params()
    lstm_param_count = sum(p.size for p in lstm_params.values())
    print(f"  Parameter count: {lstm_param_count:,}")

    # Relational RNN
    print(f"\n[2] Relational RNN")
    rel_rnn = RelationalRNN(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        num_slots=4,
        slot_size=64,
        num_heads=2
    )
    rel_outputs = rel_rnn.forward(sequence, return_sequences=True)

    print(f"  Output shape: {rel_outputs.shape}")
    print(f"  Output range: [{rel_outputs.min():.3f}, {rel_outputs.max():.3f}]")
    print(f"  Output mean: {rel_outputs.mean():.3f}")
    print(f"  Output std: {rel_outputs.std():.3f}")

    # Estimate Relational RNN parameters (approximate)
    # LSTM + Memory attention + projections
    print(f"  Additional components:")
    print(f"    - Relational memory with {rel_rnn.num_slots} slots")
    print(f"    - Multi-head attention ({rel_rnn.num_heads} heads)")
    print(f"    - Memory update gates and projections")

    # Architecture comparison
    print(f"\n[3] Architecture Comparison")
    print(f"\n  LSTM Baseline:")
    print(f"    - Sequential processing only")
    print(f"    - Hidden state carries all information")
    print(f"    - No explicit relational reasoning")

    print(f"\n  Relational RNN:")
    print(f"    - Sequential processing (LSTM)")
    print(f"    + Relational memory (multi-head attention)")
    print(f"    - Memory slots can interact and specialize")
    print(f"    - Explicit relational reasoning capability")

    # Integration explanation
    print(f"\n[4] LSTM + Memory Integration")
    print(f"  How they interact:")
    print(f"    1. LSTM processes input sequentially")
    print(f"    2. LSTM hidden state updates relational memory")
    print(f"    3. Memory slots interact via self-attention")
    print(f"    4. Memory readout combined with LSTM output")
    print(f"    5. Combined representation used for predictions")

    print(f"\n  Benefits:")
    print(f"    - LSTM: temporal dependencies, sequential patterns")
    print(f"    - Memory: relational reasoning, entity tracking")
    print(f"    - Combined: both sequential and relational processing")

    print("\n" + "=" * 80)
    print("Comparison Complete")
    print("=" * 80 + "\n")


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print(" " * 15 + "RELATIONAL RNN IMPLEMENTATION TEST SUITE")
    print(" " * 20 + "Paper 18: Relational RNN - Task P2-T2")
    print("=" * 80 + "\n")

    # Run all tests
    test_relational_memory()
    test_relational_rnn_cell()
    model = test_relational_rnn()
    compare_with_lstm_baseline()

    print("=" * 80)
    print(" " * 25 + "ALL TESTS COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print("\nImplementation Summary:")
    print("  - RelationalMemory: Multi-head self-attention over memory slots")
    print("  - RelationalRNNCell: Combines LSTM + relational memory")
    print("  - RelationalRNN: Full sequence processor with output projection")
    print("  - All shapes verified")
    print("  - No NaN/Inf in forward passes")
    print("  - Memory evolution confirmed")
    print("  - Comparison with LSTM baseline complete")
    print("\nIntegration Approach:")
    print("  1. LSTM processes sequential input -> hidden state")
    print("  2. Hidden state updates relational memory via attention")
    print("  3. Memory slots interact through multi-head self-attention")
    print("  4. Memory readout (mean pooling) combined with LSTM output")
    print("  5. Combined representation projected to output space")
    print("\nKey Features:")
    print("  - Gated memory updates for controlled information flow")
    print("  - Residual connections preserve existing memory")
    print("  - Separate processing streams (sequential + relational)")
    print("  - Flexible memory size and attention heads")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
