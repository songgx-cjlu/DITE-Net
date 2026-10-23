"""RADIOML 2016.10A and RML22: original stratified 60/20/20 splits.

The source experiments split each (modulation, SNR) group separately, with
NumPy RandomState(2016) for RADIOML 2016.10A and RandomState(2022) for RML22.
Only open dataset files you trust: Python pickle can execute arbitrary code.
"""

from pathlib import Path
import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


SPLIT_SEEDS = {"rml2016a": 2016, "rml22": 2022}


def load_radioml(data_path, dataset="rml2016a", batch_size=64, num_workers=0):
    if dataset not in SPLIT_SEEDS:
        raise ValueError(f"Unknown dataset: {dataset}. Choose {list(SPLIT_SEEDS)}")
    path = Path(data_path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    # Both benchmark releases in this project use {(mod, snr): I/Q array}.
    with path.open("rb") as f:
        raw = pickle.load(f, encoding="latin1")

    mods = sorted({key[0] for key in raw})
    snrs = sorted({key[1] for key in raw})
    mod_to_idx = {mod: i for i, mod in enumerate(mods)}
    inputs, labels, snr_values = [], [], []

    # Preserve the original project ordering before stratified splitting.
    for mod in mods:
        for snr in snrs:
            block = np.asarray(raw[(mod, snr)], dtype=np.float32)
            if block.ndim != 3 or block.shape[1] != 2:
                raise ValueError(f"Unexpected I/Q block shape: {block.shape} for {(mod, snr)}")
            inputs.append(block)
            labels.extend([mod_to_idx[mod]] * len(block))
            snr_values.extend([snr] * len(block))

    x = np.concatenate(inputs, axis=0)
    y = np.asarray(labels, dtype=np.int64)
    s = np.asarray(snr_values, dtype=np.int32)
    rng = np.random.RandomState(SPLIT_SEEDS[dataset])
    train_idx, val_idx, test_idx = [], [], []

    for mod_index, _ in enumerate(mods):
        for snr in snrs:
            indices = np.where((y == mod_index) & (s == snr))[0]
            rng.shuffle(indices)
            n_train = int(0.6 * len(indices))
            n_val = int(0.2 * len(indices))
            n_test = len(indices) - n_train - n_val
            train_idx.extend(indices[:n_train])
            val_idx.extend(indices[n_train:n_train + n_val])
            test_idx.extend(indices[-n_test:] if n_test else [])

    def make_loader(indices, shuffle, with_snr):
        indices = np.asarray(indices, dtype=np.int64)
        tensors = [torch.from_numpy(x[indices]), torch.from_numpy(y[indices])]
        if with_snr:
            tensors.append(torch.from_numpy(s[indices]))
        return DataLoader(TensorDataset(*tensors), batch_size=batch_size,
                          shuffle=shuffle, num_workers=num_workers,
                          pin_memory=torch.cuda.is_available())

    loaders = (
        make_loader(train_idx, shuffle=True, with_snr=False),
        make_loader(val_idx, shuffle=False, with_snr=False),
        make_loader(test_idx, shuffle=False, with_snr=True),
    )
    metadata = {"classes": [str(mod) for mod in mods], "snrs": [int(t) for t in snrs],
                "input_length": int(x.shape[-1]), "counts": [len(train_idx), len(val_idx), len(test_idx)],
                "split_seed": SPLIT_SEEDS[dataset]}
    return loaders, metadata
