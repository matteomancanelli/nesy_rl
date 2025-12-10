# Neuro-Symbolic Offline RL

## Submodules

- `https://github.com/axelmezini/suffix-prediction`
- `https://github.com/axelmezini/nesy-suffix-prediction-dfa`
- `https://github.com/jannerm/trajectory-transformer`

## Run

```
git clone --recurse-submodules https://github.com/matteomancanelli/nesy_rl
cd nesy_rl
python train_cb.py \
    --ltl_formula 'G (! s0_bin5)' \
    --epochs 20 \
    --save_path runs/
```

## Using a Different Environment

The training pipeline is environment-independent.
To plug in a new environment, only two small components need to change:

- Replace colour_bomb.py with your own environment
- Replace cb_dataset.py with a module that interacrs with your environment and returns inputs for training
- Update the adapter configuration in train_cb.py (see function "build_adapter_and_dfa")

Everything else in the pipeline is domain-agnostic.