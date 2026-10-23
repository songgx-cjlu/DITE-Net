# DITE-Net: Automatic Modulation Recognition

Minimal source-code release for **DITE-Net: A Dendritic-Inhibitory Temporal Evidence Network for Automatic Modulation Recognition** (submitted to *Electronics*).

This repository contains **only the proposed full DITE-Net model** and the minimal training/evaluation pipeline for two public RadioML datasets. No baseline networks, ablation experiments, comparison scripts, experimental logs, dataset copies, or trained weights are included.

## Files

| File | Description |
|---|---|
| `dite_net.py` | Full network: oscillatory encoding, DCI, EIR, predictive-coding correction, sparse channel mixer, context modulation, BiGRU and classifier. |
| `datasets.py` | RadioML dictionary loading and stratified 60/20/20 splits. |
| `run.py` | Single-model training, validation, checkpoint selection and per-SNR testing. |
| `smoke_test.py` | Synthetic CPU forward/backward check. |
| `requirements.txt` | Minimal Python dependencies. |

## Setup

Install a compatible PyTorch build for your CPU/CUDA platform (see the official PyTorch installation guide), then:

```bash
pip install -r requirements.txt
python smoke_test.py
```

The smoke test checks a model with **220,959 trainable parameters** for 11-way classification and an input tensor of `(batch, 2, 128)`.

## Datasets

Download the datasets separately from their original providers; **data is not distributed with this code**.

- **RADIOML 2016.10A**: DeepSig dataset distribution (https://www.deepsig.ai/datasets/). Use the dictionary-format `.pkl` release with `(modulation, SNR)` keys and arrays of shape `(N, 2, 128)`.
- **RML22**: Official source (https://github.com/venkateshsathya/RML22). Use the corresponding dictionary-format `RML22.pkl`.

Class indices are sorted from the original dataset dictionary keys. For consistency with the source experiments, stratification is performed **independently for each modulation/SNR pair**, using 60% training, 20% validation, 20% test and split RNG seeds **2016** / **2022**, respectively. Input I/Q is provided to the network as float32 without extra STFT transformations or normalization.

**Security note:** Only load `.pkl` files from a trusted source. Python pickle files can run arbitrary code during deserialization.

## Train and test

RADIOML 2016.10A (100 epochs, for example seed 0):

```bash
python run.py --dataset rml2016a \
    --data-path /path/to/RML2016.10a_dict.pkl \
    --seed 0 --epochs 100 --output-dir outputs/rml2016a_seed0
```

RML22:

```bash
python run.py --dataset rml22 \
    --data-path /path/to/RML22.pkl \
    --seed 0 --epochs 100 --output-dir outputs/rml22_seed0
```

To select a physical GPU, set `CUDA_VISIBLE_DEVICES` before invoking the command, e.g., `CUDA_VISIBLE_DEVICES=1 python run.py ...`. By default, the script uses CUDA when available.

Training settings taken from the original project: AdamW, learning rate `1e-3`, batch size 64, weight decay 0, gradient clipping at 1.0, epoch-30 LR milestone with gamma 0.1, cross-entropy plus auxiliary prediction loss (weight 0.05), and best validation accuracy checkpoint selection. The project uses `num_workers=0` by default. Training `--seed` is independent of the fixed dataset split RNG seed.

Outputs in `--output-dir`:

- `best_model.pth` (best validation checkpoint)
- `history.csv` (epoch-wise metrics)
- `test_metrics.json` (overall accuracy, low/high SNR and per-SNR accuracies)

Evaluate an existing checkpoint:

```bash
python run.py --dataset rml2016a \
    --data-path /path/to/RML2016.10a_dict.pkl \
    --eval-only --checkpoint outputs/rml2016a_seed0/best_model.pth \
    --output-dir outputs/re_evaluation
```

Reported low-SNR scores average the per-SNR accuracies over **-20 through 0 dB**; high-SNR scores average **2 through 18 dB**. Overall accuracy uses all test samples. Individual results may vary by training environment and seed; no pretrained checkpoint is included in this archive, so the paper's reported measurements are not independently verified here.

## Model fidelity

The published model module is a cleanup of the `NCAMRNet` *full configuration* in the authors' final v6.1.6 code. Ablation/configuration switches were removed, but **module computations, trainable tensors, parameter names and initialization order were preserved**. A fixed-seed comparison against the original full model confirmed matching state-dictionary keys, initialization tensors, parameter count (220,959), and forward predictions on a synthetic input. The `DITENet` class can therefore load full-model `state_dict` weights from the original implementation (when available).

## License

No license file has been assigned in this draft. The code owner(s) should select an appropriate license before making the repository public.
"# DITE-Net" 
