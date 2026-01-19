"""
Synthetic Sequential Reasoning Dataset Generator
Paper 18: Relational RNN (Santoro et al.)

This module generates three types of sequential reasoning tasks:
1. Object Tracking - Track multiple objects moving in a 2D grid
2. Pair Matching - Remember and retrieve paired elements
3. Simple bAbI-style QA - Answer questions based on sequential facts

All tasks require memory and relational reasoning capabilities.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict


# ============================================================================
# Task 1: Object Tracking
# ============================================================================

def generate_object_tracking(n_samples=1000, seq_len=15, n_objects=3, grid_size=5):
    """
    Track objects moving in a 2D grid.

    Task: Multiple objects move randomly in a grid. At the end, query for the
    final position of a specific object. Requires tracking object identities
    and their positions over time.

    Args:
        n_samples: Number of samples to generate
        seq_len: Length of movement sequence
        n_objects: Number of objects to track
        grid_size: Size of the grid (grid_size x grid_size)

    Returns:
        X: (n_samples, seq_len+1, input_dim) - Input sequences
           Each timestep encodes: [object_id (one-hot), x_pos, y_pos]
           Last timestep is the query: [object_id (one-hot), 0, 0]
        y: (n_samples, 2) - Final position of queried object [x, y]
        metadata: Dictionary with task information

    Input dimension: n_objects (one-hot) + 2 (x, y coordinates)
    """
    input_dim = n_objects + 2
    X = np.zeros((n_samples, seq_len + 1, input_dim))
    y = np.zeros((n_samples, 2))

    for i in range(n_samples):
        # Initialize random starting positions for each object
        positions = {}
        for obj_id in range(n_objects):
            positions[obj_id] = [
                np.random.randint(0, grid_size),
                np.random.randint(0, grid_size)
            ]

        # Generate movement sequence
        for t in range(seq_len):
            # Choose a random object to move
            obj_id = np.random.randint(0, n_objects)

            # Random walk (move in one direction or stay)
            direction = np.random.choice(['up', 'down', 'left', 'right', 'stay'])
            if direction == 'up':
                positions[obj_id][1] = min(positions[obj_id][1] + 1, grid_size - 1)
            elif direction == 'down':
                positions[obj_id][1] = max(positions[obj_id][1] - 1, 0)
            elif direction == 'left':
                positions[obj_id][0] = max(positions[obj_id][0] - 1, 0)
            elif direction == 'right':
                positions[obj_id][0] = min(positions[obj_id][0] + 1, grid_size - 1)

            # Encode: [one-hot object_id, x, y]
            X[i, t, obj_id] = 1  # One-hot encoding
            X[i, t, n_objects] = positions[obj_id][0] / grid_size  # Normalize x
            X[i, t, n_objects + 1] = positions[obj_id][1] / grid_size  # Normalize y

        # Query: Ask for position of a random object
        query_obj = np.random.randint(0, n_objects)
        X[i, seq_len, query_obj] = 1  # Query encoding (one-hot, no position)

        # Target: Final position of queried object (normalized)
        y[i, 0] = positions[query_obj][0] / grid_size
        y[i, 1] = positions[query_obj][1] / grid_size

    metadata = {
        'task': 'object_tracking',
        'n_objects': n_objects,
        'grid_size': grid_size,
        'seq_len': seq_len,
        'input_dim': input_dim,
        'output_dim': 2
    }

    return X, y, metadata


# ============================================================================
# Task 2: Pair Matching
# ============================================================================

def generate_pair_matching(n_samples=1000, seq_len=10, vocab_size=20):
    """
    Remember pairs shown earlier in sequence.

    Task: First half shows pairs (A, B), (C, D), etc. Second half queries
    one element from a pair. Model must retrieve the paired element.

    Args:
        n_samples: Number of samples to generate
        seq_len: Total sequence length (must be even)
        vocab_size: Size of vocabulary for elements

    Returns:
        X: (n_samples, seq_len, vocab_size+1) - Input sequences
           First half: pairs encoded as consecutive one-hot vectors
           Second half: query (one element with special marker)
        y: (n_samples, vocab_size) - The paired element (one-hot)
        metadata: Dictionary with task information

    Example sequence (vocab_size=5, seq_len=6):
        t=0: [1,0,0,0,0,0] (element A)
        t=1: [0,1,0,0,0,0] (element B) -> pair (A, B)
        t=2: [0,0,1,0,0,0] (element C)
        t=3: [0,0,0,1,0,0] (element D) -> pair (C, D)
        t=4: [1,0,0,0,0,1] (query A with marker)
        t=5: padding
        Output: [0,1,0,0,0] (answer: B)
    """
    if seq_len % 2 != 0:
        seq_len += 1  # Make it even

    n_pairs = seq_len // 4  # Use first half for showing pairs
    input_dim = vocab_size + 1  # +1 for query marker

    X = np.zeros((n_samples, seq_len, input_dim))
    y = np.zeros((n_samples, vocab_size))

    for i in range(n_samples):
        # Generate unique pairs
        available = list(range(vocab_size))
        np.random.shuffle(available)

        pairs = []
        for p in range(n_pairs):
            if len(available) >= 2:
                elem1 = available.pop()
                elem2 = available.pop()
                pairs.append((elem1, elem2))

        # Show pairs in first half
        for p, (elem1, elem2) in enumerate(pairs):
            t1 = p * 2
            t2 = p * 2 + 1
            X[i, t1, elem1] = 1
            X[i, t2, elem2] = 1

        # Query in second half
        if pairs:
            query_pair_idx = np.random.randint(0, len(pairs))
            elem1, elem2 = pairs[query_pair_idx]

            # Randomly query either element of the pair
            if np.random.rand() > 0.5:
                query_elem = elem1
                answer_elem = elem2
            else:
                query_elem = elem2
                answer_elem = elem1

            # Place query
            query_time = n_pairs * 2
            X[i, query_time, query_elem] = 1
            X[i, query_time, vocab_size] = 1  # Query marker

            # Set answer
            y[i, answer_elem] = 1

    metadata = {
        'task': 'pair_matching',
        'vocab_size': vocab_size,
        'n_pairs': n_pairs,
        'seq_len': seq_len,
        'input_dim': input_dim,
        'output_dim': vocab_size
    }

    return X, y, metadata


# ============================================================================
# Task 3: Simple bAbI-style QA
# ============================================================================

def generate_babi_simple(n_samples=1000, max_facts=5, n_entities=5, n_locations=4):
    """
    Simple question answering with 2-3 supporting facts.

    Task: Track entities and their properties/locations over time.
    Answer questions that require combining multiple facts.

    Args:
        n_samples: Number of samples to generate
        max_facts: Maximum number of facts before question
        n_entities: Number of entities (e.g., John, Mary, ball)
        n_locations: Number of locations (e.g., kitchen, garden)

    Returns:
        X: (n_samples, max_facts+1, input_dim) - Input sequences
           Each fact: [entity (one-hot), location (one-hot), fact_type]
           Question: [query_entity, 0s, question_marker]
        y: (n_samples, n_locations) - Answer location (one-hot)
        metadata: Dictionary with task information

    Example:
        Fact 1: John went to kitchen
        Fact 2: Mary went to garden
        Fact 3: John grabbed ball
        Q: Where is ball? A: kitchen

    Fact types:
        0: entity goes to location
        1: entity grabs object
    """
    # Input: [entity_id (one-hot n_entities), location_id (one-hot n_locations),
    #         fact_type (2 types), question_marker]
    input_dim = n_entities + n_locations + 2 + 1

    X = np.zeros((n_samples, max_facts + 1, input_dim))
    y = np.zeros((n_samples, n_locations))

    # Reserve last entity as "object" (e.g., ball)
    n_agents = n_entities - 1
    object_id = n_entities - 1

    for i in range(n_samples):
        # Track state
        entity_locations = {}  # entity_id -> location_id
        object_holder = None   # which entity has the object

        # Generate facts
        n_facts = np.random.randint(2, max_facts + 1)

        for t in range(n_facts):
            fact_type = np.random.choice([0, 1], p=[0.7, 0.3])  # More movement than grabs

            if fact_type == 0:  # Entity goes to location
                entity = np.random.randint(0, n_agents)
                location = np.random.randint(0, n_locations)
                entity_locations[entity] = location

                # Encode fact
                X[i, t, entity] = 1
                X[i, t, n_entities + location] = 1
                X[i, t, n_entities + n_locations] = 1  # fact_type = 0

            elif fact_type == 1 and len(entity_locations) > 0:  # Entity grabs object
                # Only entities that have been to locations can grab
                entity = np.random.choice(list(entity_locations.keys()))
                object_holder = entity

                # Encode fact
                X[i, t, entity] = 1
                X[i, t, n_entities + n_locations + 1] = 1  # fact_type = 1

        # Generate question: "Where is the object?"
        X[i, max_facts, object_id] = 1
        X[i, max_facts, -1] = 1  # Question marker

        # Answer: location of object
        if object_holder is not None and object_holder in entity_locations:
            answer_location = entity_locations[object_holder]
        elif len(entity_locations) > 0:
            # If object wasn't grabbed, random location where someone is
            answer_location = np.random.choice(list(entity_locations.values()))
        else:
            answer_location = 0  # Default

        y[i, answer_location] = 1

    metadata = {
        'task': 'babi_simple',
        'n_entities': n_entities,
        'n_locations': n_locations,
        'max_facts': max_facts,
        'input_dim': input_dim,
        'output_dim': n_locations
    }

    return X, y, metadata


# ============================================================================
# Data Utilities
# ============================================================================

def create_train_test_split(X, y, test_ratio=0.2, seed=42):
    """
    Split data into train and test sets.

    Args:
        X: Input data (n_samples, seq_len, input_dim)
        y: Target data (n_samples, output_dim)
        test_ratio: Fraction of data for testing
        seed: Random seed for reproducibility

    Returns:
        X_train, X_test, y_train, y_test
    """
    np.random.seed(seed)
    n_samples = X.shape[0]
    n_test = int(n_samples * test_ratio)

    # Random permutation
    indices = np.random.permutation(n_samples)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]

    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]

    return X_train, X_test, y_train, y_test


def create_batches(X, y, batch_size=32, shuffle=True):
    """
    Create mini-batches for training.

    Args:
        X: Input data (n_samples, seq_len, input_dim)
        y: Target data (n_samples, output_dim)
        batch_size: Size of each batch
        shuffle: Whether to shuffle before batching

    Yields:
        (X_batch, y_batch) tuples
    """
    n_samples = X.shape[0]
    indices = np.arange(n_samples)

    if shuffle:
        np.random.shuffle(indices)

    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_indices = indices[start_idx:end_idx]

        yield X[batch_indices], y[batch_indices]


def normalize_sequences(X, method='minmax'):
    """
    Normalize input sequences.

    Args:
        X: Input data (n_samples, seq_len, input_dim)
        method: 'minmax' or 'standard'

    Returns:
        Normalized X
    """
    if method == 'minmax':
        X_min = X.min(axis=(0, 1), keepdims=True)
        X_max = X.max(axis=(0, 1), keepdims=True)
        X_range = X_max - X_min
        X_range[X_range == 0] = 1  # Avoid division by zero
        return (X - X_min) / X_range
    elif method == 'standard':
        X_mean = X.mean(axis=(0, 1), keepdims=True)
        X_std = X.std(axis=(0, 1), keepdims=True)
        X_std[X_std == 0] = 1
        return (X - X_mean) / X_std
    else:
        return X


# ============================================================================
# Visualization
# ============================================================================

def visualize_example(X, y, metadata, sample_idx=0, task_type='tracking'):
    """
    Visualize one example from each task type.

    Args:
        X: Input data
        y: Target data
        metadata: Task metadata
        sample_idx: Which sample to visualize
        task_type: 'tracking', 'matching', or 'babi'
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    if task_type == 'tracking':
        visualize_tracking_example(X, y, metadata, sample_idx, axes)
    elif task_type == 'matching':
        visualize_matching_example(X, y, metadata, sample_idx, axes)
    elif task_type == 'babi':
        visualize_babi_example(X, y, metadata, sample_idx, axes)

    plt.tight_layout()
    return fig


def visualize_tracking_example(X, y, metadata, sample_idx, axes):
    """Visualize object tracking task."""
    seq_len = metadata['seq_len']
    n_objects = metadata['n_objects']
    grid_size = metadata['grid_size']

    # Extract sequence
    seq = X[sample_idx]
    target = y[sample_idx]

    # Plot 1: Heatmap of input sequence
    ax = axes[0]
    ax.imshow(seq.T, aspect='auto', cmap='viridis', interpolation='nearest')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Input Dimension')
    ax.set_title(f'Object Tracking Sequence (Sample {sample_idx})')
    ax.axvline(seq_len - 0.5, color='red', linestyle='--', label='Query')
    ax.legend()

    # Plot 2: Object trajectories
    ax = axes[1]

    # Track each object's position over time
    for obj_id in range(n_objects):
        positions = []
        times = []
        for t in range(seq_len):
            if seq[t, obj_id] > 0.5:  # This object moved
                x = seq[t, n_objects] * grid_size
                y = seq[t, n_objects + 1] * grid_size
                positions.append([x, y])
                times.append(t)

        if positions:
            positions = np.array(positions)
            ax.plot(positions[:, 0], positions[:, 1], 'o-',
                   label=f'Object {obj_id}', markersize=8, linewidth=2)
            ax.scatter(positions[-1, 0], positions[-1, 1],
                      s=200, marker='*', edgecolors='black', linewidths=2)

    # Show queried object's final position
    query_obj = np.argmax(seq[seq_len, :n_objects])
    target_x = target[0] * grid_size
    target_y = target[1] * grid_size
    ax.scatter(target_x, target_y, s=300, marker='X',
              color='red', edgecolors='black', linewidths=2,
              label=f'Target (Object {query_obj})', zorder=10)

    ax.set_xlim(-0.5, grid_size - 0.5)
    ax.set_ylim(-0.5, grid_size - 0.5)
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title(f'Object Trajectories (Query: Object {query_obj})')
    ax.legend()
    ax.grid(True, alpha=0.3)


def visualize_matching_example(X, y, metadata, sample_idx, axes):
    """Visualize pair matching task."""
    seq_len = metadata['seq_len']
    vocab_size = metadata['vocab_size']
    n_pairs = metadata['n_pairs']

    seq = X[sample_idx]
    target = y[sample_idx]

    # Plot 1: Input sequence heatmap
    ax = axes[0]
    ax.imshow(seq.T, aspect='auto', cmap='viridis', interpolation='nearest')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Input Dimension')
    ax.set_title(f'Pair Matching Sequence (Sample {sample_idx})')
    ax.axvline(n_pairs * 2 - 0.5, color='red', linestyle='--', label='Query Start')
    ax.legend()

    # Plot 2: Textual representation
    ax = axes[1]
    ax.axis('off')

    text_lines = ["Pair Matching Task\n" + "="*30 + "\n"]

    # Show pairs
    text_lines.append("Shown Pairs:")
    for p in range(n_pairs):
        t1 = p * 2
        t2 = p * 2 + 1
        elem1 = np.argmax(seq[t1, :vocab_size])
        elem2 = np.argmax(seq[t2, :vocab_size])
        text_lines.append(f"  Pair {p+1}: ({elem1}, {elem2})")

    # Show query
    text_lines.append("\nQuery:")
    query_time = n_pairs * 2
    query_elem = np.argmax(seq[query_time, :vocab_size])
    text_lines.append(f"  Element: {query_elem}")

    # Show answer
    text_lines.append("\nExpected Answer:")
    answer_elem = np.argmax(target)
    text_lines.append(f"  Paired Element: {answer_elem}")

    text = "\n".join(text_lines)
    ax.text(0.1, 0.5, text, transform=ax.transAxes,
           fontsize=12, verticalalignment='center',
           fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))


def visualize_babi_example(X, y, metadata, sample_idx, axes):
    """Visualize bAbI-style QA task."""
    max_facts = metadata['max_facts']
    n_entities = metadata['n_entities']
    n_locations = metadata['n_locations']

    seq = X[sample_idx]
    target = y[sample_idx]

    # Plot 1: Input sequence heatmap
    ax = axes[0]
    ax.imshow(seq.T, aspect='auto', cmap='viridis', interpolation='nearest')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Input Dimension')
    ax.set_title(f'bAbI-style QA Sequence (Sample {sample_idx})')
    ax.axvline(max_facts - 0.5, color='red', linestyle='--', label='Question')
    ax.legend()

    # Plot 2: Textual representation
    ax = axes[1]
    ax.axis('off')

    entity_names = [f"Entity{i}" for i in range(n_entities - 1)] + ["Object"]
    location_names = [f"Loc{i}" for i in range(n_locations)]

    text_lines = ["bAbI-style QA Task\n" + "="*30 + "\n"]
    text_lines.append("Facts:")

    # Parse facts
    for t in range(max_facts):
        if seq[t].sum() > 0:
            entity_id = np.argmax(seq[t, :n_entities])
            location_part = seq[t, n_entities:n_entities+n_locations]
            fact_type_part = seq[t, n_entities+n_locations:n_entities+n_locations+2]

            if fact_type_part[0] > 0.5:  # Goes to location
                location_id = np.argmax(location_part)
                text_lines.append(f"  {t+1}. {entity_names[entity_id]} went to {location_names[location_id]}")
            elif fact_type_part[1] > 0.5:  # Grabs object
                text_lines.append(f"  {t+1}. {entity_names[entity_id]} grabbed {entity_names[-1]}")

    # Parse question
    text_lines.append("\nQuestion:")
    query_entity = np.argmax(seq[max_facts, :n_entities])
    text_lines.append(f"  Where is {entity_names[query_entity]}?")

    # Show answer
    text_lines.append("\nExpected Answer:")
    answer_location = np.argmax(target)
    text_lines.append(f"  {location_names[answer_location]}")

    text = "\n".join(text_lines)
    ax.text(0.1, 0.5, text, transform=ax.transAxes,
           fontsize=11, verticalalignment='center',
           fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))


# ============================================================================
# Testing and Validation
# ============================================================================

def test_all_tasks():
    """
    Test all task generation functions.
    Verify shapes, distributions, and solvability.
    """
    print("="*60)
    print("Testing Sequential Reasoning Tasks")
    print("="*60)

    # Test 1: Object Tracking
    print("\n[Task 1: Object Tracking]")
    X1, y1, meta1 = generate_object_tracking(n_samples=100, seq_len=15, n_objects=3, grid_size=5)
    print(f"  Input shape: {X1.shape}")
    print(f"  Output shape: {y1.shape}")
    print(f"  Input dim: {meta1['input_dim']} (expected: {meta1['n_objects']+2})")
    print(f"  Output dim: {meta1['output_dim']}")
    print(f"  Value ranges - X: [{X1.min():.3f}, {X1.max():.3f}], y: [{y1.min():.3f}, {y1.max():.3f}]")
    assert X1.shape == (100, 16, 5), "Object tracking shape mismatch!"
    assert y1.shape == (100, 2), "Object tracking output shape mismatch!"
    print("  ✓ Passed shape tests")

    # Test 2: Pair Matching
    print("\n[Task 2: Pair Matching]")
    X2, y2, meta2 = generate_pair_matching(n_samples=100, seq_len=10, vocab_size=20)
    print(f"  Input shape: {X2.shape}")
    print(f"  Output shape: {y2.shape}")
    print(f"  Input dim: {meta2['input_dim']} (expected: {meta2['vocab_size']+1})")
    print(f"  Output dim: {meta2['output_dim']}")
    print(f"  Value ranges - X: [{X2.min():.3f}, {X2.max():.3f}], y: [{y2.min():.3f}, {y2.max():.3f}]")
    assert X2.shape == (100, 10, 21), "Pair matching shape mismatch!"
    assert y2.shape == (100, 20), "Pair matching output shape mismatch!"
    # Check that outputs are one-hot
    assert np.allclose(y2.sum(axis=1), 1.0), "Pair matching outputs not one-hot!"
    print("  ✓ Passed shape tests")

    # Test 3: bAbI-style QA
    print("\n[Task 3: bAbI-style QA]")
    X3, y3, meta3 = generate_babi_simple(n_samples=100, max_facts=5, n_entities=5, n_locations=4)
    print(f"  Input shape: {X3.shape}")
    print(f"  Output shape: {y3.shape}")
    print(f"  Input dim: {meta3['input_dim']}")
    print(f"  Output dim: {meta3['output_dim']}")
    print(f"  Value ranges - X: [{X3.min():.3f}, {X3.max():.3f}], y: [{y3.min():.3f}, {y3.max():.3f}]")
    # Input dim = n_entities + n_locations + 2 (fact types) + 1 (question marker) = 5 + 4 + 2 + 1 = 12
    assert X3.shape == (100, 6, 12), "bAbI shape mismatch!"
    assert y3.shape == (100, 4), "bAbI output shape mismatch!"
    assert np.allclose(y3.sum(axis=1), 1.0), "bAbI outputs not one-hot!"
    print("  ✓ Passed shape tests")

    # Test utilities
    print("\n[Testing Utilities]")
    X_train, X_test, y_train, y_test = create_train_test_split(X1, y1, test_ratio=0.2)
    print(f"  Train split: {X_train.shape}, Test split: {X_test.shape}")
    assert X_train.shape[0] == 80 and X_test.shape[0] == 20, "Split ratio incorrect!"
    print("  ✓ Train/test split works")

    batch_count = 0
    for X_batch, y_batch in create_batches(X1, y1, batch_size=32):
        batch_count += 1
        assert X_batch.shape[0] <= 32, "Batch size too large!"
    print(f"  Created {batch_count} batches")
    print("  ✓ Batching works")

    print("\n" + "="*60)
    print("All tests passed!")
    print("="*60)

    return {
        'tracking': (X1, y1, meta1),
        'matching': (X2, y2, meta2),
        'babi': (X3, y3, meta3)
    }


def visualize_all_tasks(test_results):
    """
    Visualize examples from all three tasks.
    """
    print("\nGenerating visualizations...")

    # Object Tracking
    X1, y1, meta1 = test_results['tracking']
    fig1 = visualize_example(X1, y1, meta1, sample_idx=0, task_type='tracking')
    plt.savefig('/Users/paulamerigojr.iipajo/sutskever-30-implementations/task_tracking_example.png',
                dpi=150, bbox_inches='tight')
    print("  Saved: task_tracking_example.png")

    # Pair Matching
    X2, y2, meta2 = test_results['matching']
    fig2 = visualize_example(X2, y2, meta2, sample_idx=0, task_type='matching')
    plt.savefig('/Users/paulamerigojr.iipajo/sutskever-30-implementations/task_matching_example.png',
                dpi=150, bbox_inches='tight')
    print("  Saved: task_matching_example.png")

    # bAbI QA
    X3, y3, meta3 = test_results['babi']
    fig3 = visualize_example(X3, y3, meta3, sample_idx=0, task_type='babi')
    plt.savefig('/Users/paulamerigojr.iipajo/sutskever-30-implementations/task_babi_example.png',
                dpi=150, bbox_inches='tight')
    print("  Saved: task_babi_example.png")

    plt.show()


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)

    # Test all tasks
    test_results = test_all_tasks()

    # Visualize examples
    visualize_all_tasks(test_results)

    print("\n" + "="*60)
    print("Dataset Generation Complete!")
    print("="*60)
    print("\nTask Summary:")
    print("  1. Object Tracking: Track 3 objects moving in 5x5 grid")
    print("  2. Pair Matching: Remember and retrieve paired elements")
    print("  3. bAbI-style QA: Answer questions from sequential facts")
    print("\nAll tasks require:")
    print("  - Memory of past events")
    print("  - Relational reasoning between entities")
    print("  - Temporal context understanding")
