# GAITGen: Disentangled Motion-Pathology Impaired Gait Generative Model


<h3 align="center"> ✨ Bringing Motion Generation to the Clinical Domain ✨
<strong>(WACV 2026)
</h3>

<p align="center">
  <a href="https://arxiv.org/abs/2503.22397"><img src="https://img.shields.io/badge/arXiv-2503.22397-b31b1b.svg" alt="arXiv"></a>
  <a href="https://vadeli.github.io/GAITGen/"><img src="https://img.shields.io/badge/Project-Page-blue.svg" alt="Project Page"></a>
</p>

PyTorch implementation for training gait motion generation models with residual vector quantization and transformer-based motion priors.

GAITGen represents gait sequences as discrete motion tokens with an RVQ-VAE tokenizer, then learns transformer models over those tokens for conditional gait motion modeling. The code supports standard and disentangled VQ-VAE variants, masked transformer training, and residual transformer training.

## Contents

- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Evaluator Assets](#evaluator-assets)
- [Training](#training)
- [Citation](#citation)
- [License](#license)

## Installation

Create the Conda environment:

```bash
conda env create -f environment.yml
conda activate gaitgen
```

Alternatively, install the Python dependencies in an existing Python 3.10 environment:

```bash
pip install -r requirements.txt
```

Download the GloVe files used by the motion-language evaluator:

```bash
bash prepare/download_glove.sh
```

## Data Preparation

Set the dataset root in:

```text
data/configs/base.yaml
```

Configure the PDGaM dataset paths in:

```text
data/configs/pdgam.yaml
```

The processed dataset is expected to contain HumanML3D-style motion features:

```text
PDGaM/
├── Annotations/
│   └── Gait/
│       ├── train.csv
│       └── test.csv
└── representation_HML3D/
    ├── new_joint_vecs/
    │   ├── <sequence_id>.npy
    │   └── <sequence_id>_M.npy
    ├── Mean.npy
    ├── Std.npy
    ├── train.txt
    ├── test.txt
    ├── train_tiny.txt
    └── test_tiny.txt
```

`train_tiny.txt` and `test_tiny.txt` are optional convenience splits used when passing `--tiny`.

## Evaluator Assets

Validation uses a pretrained motion-language evaluator. Place the evaluator checkpoint at:

```text
checkpoints/pdgam/text_mot_match/model/finest.tar
```

The evaluator option metadata used by the training scripts is stored at:

```text
checkpoints/pdgam/Comp_v6_KLD005/opt.txt
```

## Training

All commands below assume the PDGaM configuration files have been updated and the evaluator assets are available.

### 1. Train The Disentangled Conditional RVQ-VAE Tokenizer

```bash
python train_vq.py \
  --name gaitgen_vq_tokenizer \
  --gpu_id 0 \
  --dataset_name '["pdgam"]'
```

The tokenizer defaults correspond to the disentangled conditional RVQ-VAE training recipe used for GAITGen. Pass explicit options to override individual hyperparameters.

This command writes the tokenizer checkpoint to:

```text
checkpoints/pdgam/gaitgen_vq_tokenizer/
```

### 2. Train The Masked Transformer

Set `--vq_name` to the trained disentangled RVQ-VAE experiment name.

```bash
python train_t2m_transformer.py \
  --name gaitgen_mask_transformer \
  --gpu_id 0 \
  --dataset_name '["pdgam"]' \
  --batch_size 64 \
  --vq_name gaitgen_vq_tokenizer \
  --latent_dim 128 \
  --n_heads 6 \
  --disentangled
```

### 3. Train The Residual Transformer

```bash
python train_res_transformer.py \
  --name gaitgen_residual_transformer \
  --gpu_id 0 \
  --dataset_name '["pdgam"]' \
  --batch_size 64 \
  --vq_name gaitgen_vq_tokenizer \
  --cond_drop_prob 0.2 \
  --share_weight \
  --disentangled
```

Training outputs are written under:

```text
checkpoints/<dataset_name>/<experiment_name>/
log/
wandb/
```

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{adeli2026gaitgen,
  title={GAITGen: Disentangled Motion-Pathology Impaired Gait Generative Model -- Bringing Motion Generation to the Clinical Domain},
  author={Vida Adeli, Soroush Mehraban, Majid Mirmehdi, Alan Whone, Benjamin Filtjens, Amirhossein Dadashzadeh, Alfonso Fasano, Andrea Iaboni, Babak Taati},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  year={2026}
}
```

## Acknowledgements

We acknowledge [MoMask](https://github.com/EricGuo5513/momask-codes) for its open-source implementation.

## License

See [LICENSE](LICENSE).
