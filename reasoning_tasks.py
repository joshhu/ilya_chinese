"""
合成序列推理資料集生成器
論文 18：Relational RNN（關係型循環神經網路）(Santoro et al.)

本模組生成三種類型的序列推理任務：
1. 物件追蹤 (Object Tracking) - 追蹤在 2D 網格中移動的多個物件
2. 配對匹配 (Pair Matching) - 記憶並檢索配對元素
3. 簡單 bAbI 風格問答 - 根據序列事實回答問題

所有任務都需要記憶和關係推理能力。
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict


# ============================================================================
# 任務 1：物件追蹤
# ============================================================================

def generate_object_tracking(n_samples=1000, seq_len=15, n_objects=3, grid_size=5):
    """
    追蹤在 2D 網格中移動的物件。

    任務：多個物件在網格中隨機移動。最後，查詢特定物件的最終位置。
    需要追蹤物件身份及其隨時間變化的位置。

    參數：
        n_samples: 要生成的樣本數量
        seq_len: 移動序列的長度
        n_objects: 要追蹤的物件數量
        grid_size: 網格大小 (grid_size x grid_size)

    回傳：
        X: (n_samples, seq_len+1, input_dim) - 輸入序列
           每個時間步編碼：[object_id (one-hot 編碼), x_pos, y_pos]
           最後一個時間步是查詢：[object_id (one-hot 編碼), 0, 0]
        y: (n_samples, 2) - 被查詢物件的最終位置 [x, y]
        metadata: 包含任務資訊的字典

    輸入維度：n_objects (one-hot 編碼) + 2 (x, y 座標)
    """
    input_dim = n_objects + 2
    X = np.zeros((n_samples, seq_len + 1, input_dim))
    y = np.zeros((n_samples, 2))

    for i in range(n_samples):
        # 為每個物件初始化隨機起始位置
        positions = {}
        for obj_id in range(n_objects):
            positions[obj_id] = [
                np.random.randint(0, grid_size),
                np.random.randint(0, grid_size)
            ]

        # 生成移動序列
        for t in range(seq_len):
            # 選擇一個隨機物件進行移動
            obj_id = np.random.randint(0, n_objects)

            # 隨機遊走（朝一個方向移動或停留）
            direction = np.random.choice(['up', 'down', 'left', 'right', 'stay'])
            if direction == 'up':
                positions[obj_id][1] = min(positions[obj_id][1] + 1, grid_size - 1)
            elif direction == 'down':
                positions[obj_id][1] = max(positions[obj_id][1] - 1, 0)
            elif direction == 'left':
                positions[obj_id][0] = max(positions[obj_id][0] - 1, 0)
            elif direction == 'right':
                positions[obj_id][0] = min(positions[obj_id][0] + 1, grid_size - 1)

            # 編碼：[one-hot object_id, x, y]
            X[i, t, obj_id] = 1  # One-hot 編碼
            X[i, t, n_objects] = positions[obj_id][0] / grid_size  # 正規化 x
            X[i, t, n_objects + 1] = positions[obj_id][1] / grid_size  # 正規化 y

        # 查詢：詢問一個隨機物件的位置
        query_obj = np.random.randint(0, n_objects)
        X[i, seq_len, query_obj] = 1  # 查詢編碼（one-hot，無位置）

        # 目標：被查詢物件的最終位置（已正規化）
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
# 任務 2：配對匹配
# ============================================================================

def generate_pair_matching(n_samples=1000, seq_len=10, vocab_size=20):
    """
    記憶序列中較早顯示的配對。

    任務：前半部分顯示配對 (A, B)、(C, D) 等。後半部分查詢配對中的一個元素。
    模型必須檢索出配對的另一個元素。

    參數：
        n_samples: 要生成的樣本數量
        seq_len: 總序列長度（必須是偶數）
        vocab_size: 元素的詞彙表大小

    回傳：
        X: (n_samples, seq_len, vocab_size+1) - 輸入序列
           前半部分：配對編碼為連續的 one-hot 向量
           後半部分：查詢（帶有特殊標記的單一元素）
        y: (n_samples, vocab_size) - 配對的元素（one-hot 編碼）
        metadata: 包含任務資訊的字典

    序列範例 (vocab_size=5, seq_len=6)：
        t=0: [1,0,0,0,0,0] (元素 A)
        t=1: [0,1,0,0,0,0] (元素 B) -> 配對 (A, B)
        t=2: [0,0,1,0,0,0] (元素 C)
        t=3: [0,0,0,1,0,0] (元素 D) -> 配對 (C, D)
        t=4: [1,0,0,0,0,1] (查詢 A 並帶有標記)
        t=5: 填充
        輸出: [0,1,0,0,0] (答案: B)
    """
    if seq_len % 2 != 0:
        seq_len += 1  # 使其為偶數

    n_pairs = seq_len // 4  # 使用前半部分來顯示配對
    input_dim = vocab_size + 1  # +1 用於查詢標記

    X = np.zeros((n_samples, seq_len, input_dim))
    y = np.zeros((n_samples, vocab_size))

    for i in range(n_samples):
        # 生成唯一的配對
        available = list(range(vocab_size))
        np.random.shuffle(available)

        pairs = []
        for p in range(n_pairs):
            if len(available) >= 2:
                elem1 = available.pop()
                elem2 = available.pop()
                pairs.append((elem1, elem2))

        # 在前半部分顯示配對
        for p, (elem1, elem2) in enumerate(pairs):
            t1 = p * 2
            t2 = p * 2 + 1
            X[i, t1, elem1] = 1
            X[i, t2, elem2] = 1

        # 在後半部分進行查詢
        if pairs:
            query_pair_idx = np.random.randint(0, len(pairs))
            elem1, elem2 = pairs[query_pair_idx]

            # 隨機查詢配對中的任一元素
            if np.random.rand() > 0.5:
                query_elem = elem1
                answer_elem = elem2
            else:
                query_elem = elem2
                answer_elem = elem1

            # 放置查詢
            query_time = n_pairs * 2
            X[i, query_time, query_elem] = 1
            X[i, query_time, vocab_size] = 1  # 查詢標記

            # 設定答案
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
# 任務 3：簡單 bAbI 風格問答
# ============================================================================

def generate_babi_simple(n_samples=1000, max_facts=5, n_entities=5, n_locations=4):
    """
    具有 2-3 個支持事實的簡單問答。

    任務：追蹤實體及其屬性/位置隨時間的變化。
    回答需要結合多個事實的問題。

    參數：
        n_samples: 要生成的樣本數量
        max_facts: 問題前的最大事實數量
        n_entities: 實體數量（例如：John、Mary、ball）
        n_locations: 位置數量（例如：kitchen、garden）

    回傳：
        X: (n_samples, max_facts+1, input_dim) - 輸入序列
           每個事實：[entity (one-hot 編碼), location (one-hot 編碼), fact_type]
           問題：[query_entity, 0s, question_marker]
        y: (n_samples, n_locations) - 答案位置（one-hot 編碼）
        metadata: 包含任務資訊的字典

    範例：
        事實 1：John went to kitchen（John 去了廚房）
        事實 2：Mary went to garden（Mary 去了花園）
        事實 3：John grabbed ball（John 拿起了球）
        問：Where is ball?（球在哪裡？）答：kitchen（廚房）

    事實類型：
        0: 實體前往某位置
        1: 實體拿取物件
    """
    # 輸入：[entity_id (one-hot n_entities), location_id (one-hot n_locations),
    #        fact_type (2 種類型), question_marker]
    input_dim = n_entities + n_locations + 2 + 1

    X = np.zeros((n_samples, max_facts + 1, input_dim))
    y = np.zeros((n_samples, n_locations))

    # 保留最後一個實體作為「物件」（例如：球）
    n_agents = n_entities - 1
    object_id = n_entities - 1

    for i in range(n_samples):
        # 追蹤狀態
        entity_locations = {}  # entity_id -> location_id
        object_holder = None   # 哪個實體持有物件

        # 生成事實
        n_facts = np.random.randint(2, max_facts + 1)

        for t in range(n_facts):
            fact_type = np.random.choice([0, 1], p=[0.7, 0.3])  # 移動比拿取更多

            if fact_type == 0:  # 實體前往某位置
                entity = np.random.randint(0, n_agents)
                location = np.random.randint(0, n_locations)
                entity_locations[entity] = location

                # 編碼事實
                X[i, t, entity] = 1
                X[i, t, n_entities + location] = 1
                X[i, t, n_entities + n_locations] = 1  # fact_type = 0

            elif fact_type == 1 and len(entity_locations) > 0:  # 實體拿取物件
                # 只有去過某位置的實體才能拿取
                entity = np.random.choice(list(entity_locations.keys()))
                object_holder = entity

                # 編碼事實
                X[i, t, entity] = 1
                X[i, t, n_entities + n_locations + 1] = 1  # fact_type = 1

        # 生成問題：「物件在哪裡？」
        X[i, max_facts, object_id] = 1
        X[i, max_facts, -1] = 1  # 問題標記

        # 答案：物件的位置
        if object_holder is not None and object_holder in entity_locations:
            answer_location = entity_locations[object_holder]
        elif len(entity_locations) > 0:
            # 如果物件沒有被拿取，隨機選擇某人所在的位置
            answer_location = np.random.choice(list(entity_locations.values()))
        else:
            answer_location = 0  # 預設值

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
# 資料工具函式
# ============================================================================

def create_train_test_split(X, y, test_ratio=0.2, seed=42):
    """
    將資料分割成訓練集和測試集。

    參數：
        X: 輸入資料 (n_samples, seq_len, input_dim)
        y: 目標資料 (n_samples, output_dim)
        test_ratio: 測試資料的比例
        seed: 用於可重現性的隨機種子

    回傳：
        X_train, X_test, y_train, y_test
    """
    np.random.seed(seed)
    n_samples = X.shape[0]
    n_test = int(n_samples * test_ratio)

    # 隨機排列
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
    建立用於訓練的小批次。

    參數：
        X: 輸入資料 (n_samples, seq_len, input_dim)
        y: 目標資料 (n_samples, output_dim)
        batch_size: 每個批次的大小
        shuffle: 是否在分批前進行洗牌

    產生：
        (X_batch, y_batch) 元組
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
    正規化輸入序列。

    參數：
        X: 輸入資料 (n_samples, seq_len, input_dim)
        method: 'minmax'（最小最大正規化）或 'standard'（標準化）

    回傳：
        正規化後的 X
    """
    if method == 'minmax':
        X_min = X.min(axis=(0, 1), keepdims=True)
        X_max = X.max(axis=(0, 1), keepdims=True)
        X_range = X_max - X_min
        X_range[X_range == 0] = 1  # 避免除以零
        return (X - X_min) / X_range
    elif method == 'standard':
        X_mean = X.mean(axis=(0, 1), keepdims=True)
        X_std = X.std(axis=(0, 1), keepdims=True)
        X_std[X_std == 0] = 1
        return (X - X_mean) / X_std
    else:
        return X


# ============================================================================
# 視覺化
# ============================================================================

def visualize_example(X, y, metadata, sample_idx=0, task_type='tracking'):
    """
    視覺化每種任務類型的一個範例。

    參數：
        X: 輸入資料
        y: 目標資料
        metadata: 任務元資料
        sample_idx: 要視覺化的樣本索引
        task_type: 'tracking'（追蹤）、'matching'（匹配）或 'babi'
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
    """視覺化物件追蹤任務。"""
    seq_len = metadata['seq_len']
    n_objects = metadata['n_objects']
    grid_size = metadata['grid_size']

    # 擷取序列
    seq = X[sample_idx]
    target = y[sample_idx]

    # 圖 1：輸入序列的熱力圖
    ax = axes[0]
    ax.imshow(seq.T, aspect='auto', cmap='viridis', interpolation='nearest')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Input Dimension')
    ax.set_title(f'Object Tracking Sequence (Sample {sample_idx})')
    ax.axvline(seq_len - 0.5, color='red', linestyle='--', label='Query')
    ax.legend()

    # 圖 2：物件軌跡
    ax = axes[1]

    # 追蹤每個物件隨時間的位置
    for obj_id in range(n_objects):
        positions = []
        times = []
        for t in range(seq_len):
            if seq[t, obj_id] > 0.5:  # 這個物件移動了
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

    # 顯示被查詢物件的最終位置
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
    """視覺化配對匹配任務。"""
    seq_len = metadata['seq_len']
    vocab_size = metadata['vocab_size']
    n_pairs = metadata['n_pairs']

    seq = X[sample_idx]
    target = y[sample_idx]

    # 圖 1：輸入序列熱力圖
    ax = axes[0]
    ax.imshow(seq.T, aspect='auto', cmap='viridis', interpolation='nearest')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Input Dimension')
    ax.set_title(f'Pair Matching Sequence (Sample {sample_idx})')
    ax.axvline(n_pairs * 2 - 0.5, color='red', linestyle='--', label='Query Start')
    ax.legend()

    # 圖 2：文字表示
    ax = axes[1]
    ax.axis('off')

    text_lines = ["Pair Matching Task\n" + "="*30 + "\n"]

    # 顯示配對
    text_lines.append("Shown Pairs:")
    for p in range(n_pairs):
        t1 = p * 2
        t2 = p * 2 + 1
        elem1 = np.argmax(seq[t1, :vocab_size])
        elem2 = np.argmax(seq[t2, :vocab_size])
        text_lines.append(f"  Pair {p+1}: ({elem1}, {elem2})")

    # 顯示查詢
    text_lines.append("\nQuery:")
    query_time = n_pairs * 2
    query_elem = np.argmax(seq[query_time, :vocab_size])
    text_lines.append(f"  Element: {query_elem}")

    # 顯示答案
    text_lines.append("\nExpected Answer:")
    answer_elem = np.argmax(target)
    text_lines.append(f"  Paired Element: {answer_elem}")

    text = "\n".join(text_lines)
    ax.text(0.1, 0.5, text, transform=ax.transAxes,
           fontsize=12, verticalalignment='center',
           fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))


def visualize_babi_example(X, y, metadata, sample_idx, axes):
    """視覺化 bAbI 風格問答任務。"""
    max_facts = metadata['max_facts']
    n_entities = metadata['n_entities']
    n_locations = metadata['n_locations']

    seq = X[sample_idx]
    target = y[sample_idx]

    # 圖 1：輸入序列熱力圖
    ax = axes[0]
    ax.imshow(seq.T, aspect='auto', cmap='viridis', interpolation='nearest')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Input Dimension')
    ax.set_title(f'bAbI-style QA Sequence (Sample {sample_idx})')
    ax.axvline(max_facts - 0.5, color='red', linestyle='--', label='Question')
    ax.legend()

    # 圖 2：文字表示
    ax = axes[1]
    ax.axis('off')

    entity_names = [f"Entity{i}" for i in range(n_entities - 1)] + ["Object"]
    location_names = [f"Loc{i}" for i in range(n_locations)]

    text_lines = ["bAbI-style QA Task\n" + "="*30 + "\n"]
    text_lines.append("Facts:")

    # 解析事實
    for t in range(max_facts):
        if seq[t].sum() > 0:
            entity_id = np.argmax(seq[t, :n_entities])
            location_part = seq[t, n_entities:n_entities+n_locations]
            fact_type_part = seq[t, n_entities+n_locations:n_entities+n_locations+2]

            if fact_type_part[0] > 0.5:  # 前往某位置
                location_id = np.argmax(location_part)
                text_lines.append(f"  {t+1}. {entity_names[entity_id]} went to {location_names[location_id]}")
            elif fact_type_part[1] > 0.5:  # 拿取物件
                text_lines.append(f"  {t+1}. {entity_names[entity_id]} grabbed {entity_names[-1]}")

    # 解析問題
    text_lines.append("\nQuestion:")
    query_entity = np.argmax(seq[max_facts, :n_entities])
    text_lines.append(f"  Where is {entity_names[query_entity]}?")

    # 顯示答案
    text_lines.append("\nExpected Answer:")
    answer_location = np.argmax(target)
    text_lines.append(f"  {location_names[answer_location]}")

    text = "\n".join(text_lines)
    ax.text(0.1, 0.5, text, transform=ax.transAxes,
           fontsize=11, verticalalignment='center',
           fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))


# ============================================================================
# 測試與驗證
# ============================================================================

def test_all_tasks():
    """
    測試所有任務生成函式。
    驗證形狀、分佈和可解性。
    """
    print("="*60)
    print("Testing Sequential Reasoning Tasks")
    print("="*60)

    # 測試 1：物件追蹤
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

    # 測試 2：配對匹配
    print("\n[Task 2: Pair Matching]")
    X2, y2, meta2 = generate_pair_matching(n_samples=100, seq_len=10, vocab_size=20)
    print(f"  Input shape: {X2.shape}")
    print(f"  Output shape: {y2.shape}")
    print(f"  Input dim: {meta2['input_dim']} (expected: {meta2['vocab_size']+1})")
    print(f"  Output dim: {meta2['output_dim']}")
    print(f"  Value ranges - X: [{X2.min():.3f}, {X2.max():.3f}], y: [{y2.min():.3f}, {y2.max():.3f}]")
    assert X2.shape == (100, 10, 21), "Pair matching shape mismatch!"
    assert y2.shape == (100, 20), "Pair matching output shape mismatch!"
    # 檢查輸出是否為 one-hot 編碼
    assert np.allclose(y2.sum(axis=1), 1.0), "Pair matching outputs not one-hot!"
    print("  ✓ Passed shape tests")

    # 測試 3：bAbI 風格問答
    print("\n[Task 3: bAbI-style QA]")
    X3, y3, meta3 = generate_babi_simple(n_samples=100, max_facts=5, n_entities=5, n_locations=4)
    print(f"  Input shape: {X3.shape}")
    print(f"  Output shape: {y3.shape}")
    print(f"  Input dim: {meta3['input_dim']}")
    print(f"  Output dim: {meta3['output_dim']}")
    print(f"  Value ranges - X: [{X3.min():.3f}, {X3.max():.3f}], y: [{y3.min():.3f}, {y3.max():.3f}]")
    # 輸入維度 = n_entities + n_locations + 2 (事實類型) + 1 (問題標記) = 5 + 4 + 2 + 1 = 12
    assert X3.shape == (100, 6, 12), "bAbI shape mismatch!"
    assert y3.shape == (100, 4), "bAbI output shape mismatch!"
    assert np.allclose(y3.sum(axis=1), 1.0), "bAbI outputs not one-hot!"
    print("  ✓ Passed shape tests")

    # 測試工具函式
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
    視覺化所有三個任務的範例。
    """
    print("\nGenerating visualizations...")

    # 物件追蹤
    X1, y1, meta1 = test_results['tracking']
    fig1 = visualize_example(X1, y1, meta1, sample_idx=0, task_type='tracking')
    plt.savefig('/Users/paulamerigojr.iipajo/sutskever-30-implementations/task_tracking_example.png',
                dpi=150, bbox_inches='tight')
    print("  Saved: task_tracking_example.png")

    # 配對匹配
    X2, y2, meta2 = test_results['matching']
    fig2 = visualize_example(X2, y2, meta2, sample_idx=0, task_type='matching')
    plt.savefig('/Users/paulamerigojr.iipajo/sutskever-30-implementations/task_matching_example.png',
                dpi=150, bbox_inches='tight')
    print("  Saved: task_matching_example.png")

    # bAbI 問答
    X3, y3, meta3 = test_results['babi']
    fig3 = visualize_example(X3, y3, meta3, sample_idx=0, task_type='babi')
    plt.savefig('/Users/paulamerigojr.iipajo/sutskever-30-implementations/task_babi_example.png',
                dpi=150, bbox_inches='tight')
    print("  Saved: task_babi_example.png")

    plt.show()


# ============================================================================
# 主程式執行
# ============================================================================

if __name__ == "__main__":
    # 設定隨機種子以確保可重現性
    np.random.seed(42)

    # 測試所有任務
    test_results = test_all_tasks()

    # 視覺化範例
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
