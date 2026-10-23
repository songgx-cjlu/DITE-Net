# DITE-Net: A Dendritic-Inhibitory Temporal Evidence Network for Automatic Modulation Recognition

## Files

| File | Description |
|---|---|
| `dite_net.py` | Full network: oscillatory encoding, DCI, EIR, predictive-coding correction, sparse channel mixer, context modulation, BiGRU and classifier. |
| `datasets.py` | RadioML dictionary loading and stratified 60/20/20 splits. |
| `run.py` | Single-model training, validation, checkpoint selection and per-SNR testing. |
| `smoke_test.py` | Synthetic CPU forward/backward check. |
| `requirements.txt` | Minimal Python dependencies. |


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


## License

No license file has been assigned in this draft. The code owner(s) should select an appropriate license before making the repository public.
"# DITE-Net" 
