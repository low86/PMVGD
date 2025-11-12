import os
import random
import pickle


def split_ids_from_list_pkl(
        pkl_path,
        save_dir='./',
        train_ratio=0.75,
        val_ratio=0.1,
        test_ratio=0.15,
        seed=42,
):
    if round(train_ratio + val_ratio + test_ratio, 5) != 1.0:
        raise ValueError("train + val + test ratios must sum to 1.")

    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"Pkl file not found: {pkl_path}")

    with open(pkl_path, 'rb') as f:
        graph_list = pickle.load(f)  # List[HeteroData]

    all_ids = list(range(len(graph_list)))
    random.seed(seed)
    random.shuffle(all_ids)

    total = len(all_ids)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train_ids = all_ids[:train_end]
    val_ids = all_ids[train_end:val_end]
    test_ids = all_ids[val_end:]

    os.makedirs(save_dir, exist_ok=True)
    for split_name, split_ids in zip(["train", "val", "test"], [train_ids, val_ids, test_ids]):
        with open(os.path.join(save_dir, f"{split_name}_ids.txt"), "w") as f:
            for idx in split_ids:
                f.write(f"{idx}\n")

    print(f"Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")


def split_ids_from_list_pkl_chunk(
        pkl_path,
        save_dir='./',
        train_ratio=0.75,
        val_ratio=0.1,
        test_ratio=0.15,
        seed=42,
        chunk=-1,
):
    if round(train_ratio + val_ratio + test_ratio, 5) != 1.0:
        raise ValueError("train + val + test ratios must sum to 1.")

    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"Pkl file not found: {pkl_path}")

    with open(pkl_path, 'rb') as f:
        graph_list = pickle.load(f)  # List[HeteroData]

    all_ids = list(range(len(graph_list)))
    random.seed(seed)
    random.shuffle(all_ids)

    total = len(all_ids)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train_ids = all_ids[:train_end]
    val_ids = all_ids[train_end:val_end]
    test_ids = all_ids[val_end:]

    os.makedirs(save_dir, exist_ok=True)
    for split_name, split_ids in zip(["train", "val", "test"], [train_ids, val_ids, test_ids]):
        with open(os.path.join(save_dir, f"{split_name}_ids_chunk{chunk}.txt"), "w") as f:
            for idx in split_ids:
                f.write(f"{idx}\n")

    print(f"Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")
