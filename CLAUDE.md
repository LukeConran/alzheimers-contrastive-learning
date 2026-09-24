# Alzheimer's MRI Classification

3D brain MRI classification into Alzheimer's Disease, Mild Cognitive Impairment, and Normal Cognition using 3D ResNet / Swin Transformer backbones with optional supervised contrastive learning. Runs on an HPC cluster (SLURM).

## Project Structure

- `datasets/mri_dataset.py` — Dataset classes (standard, contrastive)
- `models/resnet.py` — 3D ResNet with dual-head (classification + contrastive projector)
- `models/resnet_org.py` — Original baseline ResNet without contrastive support
- `losses/supcon_loss.py` — Supervised Contrastive Loss (Khosla et al., NeurIPS 2020)
- `medicalnet_model.py` — Model factory, pretrained weight loading, param group splitting
- `train.py` — Main training loop
- `test.py` — Evaluation: per-class accuracy, AUC, precision, recall

## Data Format

Images are `.npz` files with key `"image_mr"` (3D float array, shape `(D, H, W)`).
Labels come from a JSON manifest:

```json
[{"id": "001", "label": "Normal Cognition"}, ...]
```

Label mapping: `"Alzheimer's Disease"→0`, `"Mild Cognitive Impairment"→1`, `"Normal Cognition"→2`

**Note:** The raw JSON uses `"Dementia"` instead of `"Alzheimer's Disease"` — the dataset class was already fixed to handle this. Do not change it back.

## Training Modes

**Standard classification:**

```bash
python train.py --backbone resnet --model_name resnet10 --lr 1e-4 --batch_size 8 --epochs 100
```

**With supervised contrastive loss:**

```bash
python train.py --contrastive --lambda_con 0.5 --temperature 0.1 ...
```

When `--contrastive` is enabled, `ContrastiveDataset` returns two augmented views per scan and `SupConLoss` is added alongside cross-entropy. Note: contrastive mode runs 3 forward passes per batch vs 1 for CE — roughly 3x the compute cost per epoch.

**With MedicalNet pretrained weights (recommended):**

```bash
python train.py --checkpoint_pretrain /home/lukeconran/alzheimers/MedicalNet_pytorch_files2/pretrain/resnet_10.pth ...
```

Pretrained weights are at `/home/lukeconran/alzheimers/MedicalNet_pytorch_files2/pretrain/` on the cluster. When used, the backbone trains at `lr` and the classification head trains at `lr*100`.

**Resuming a timed-out job:**

```bash
python train.py --resume /home/lukeconran/alzheimers/results/ce/model_resnet10_bs8_epoch_20.pth --epochs 30 ...
```

Periodic checkpoints (every 10 epochs) save a full dict: `model`, `optimizer`, `scheduler`, `epoch`, `best_val_acc`. The best model checkpoint saves only `state_dict` for use with `test.py`.

## Model Architecture

`ResNet.forward(x, return_embedding=False)`:

- `False` → 3-class logits (classification)
- `True` → L2-normalized 128-dim embedding via projector MLP (for SupCon)

## Augmentations (MRI-safe)

Applied only during contrastive training in `ContrastiveDataset._augment()`:

- Random axis flips (50% per axis)
- Additive Gaussian noise (σ=0.05)
- Intensity scaling (0.9–1.1×)

## Research Notes

The core experiment is **CE vs SupCon**: does supervised contrastive learning improve classification performance given the same labeled dataset? Run `bin/train_ce.slurm` and `bin/train_contrastive.slurm` in parallel and compare per-class accuracy, AUC, precision, and recall from `test.py`.

**SimCLR is not viable with ADNI alone.** The original motivation (reducing clinician labeling burden) is not addressed by SupCon since it requires labels. SimCLR would fix this but needs a large unlabeled MRI pool — ADNI is fully labeled, so there is nothing to pretrain on without a separate dataset (e.g. UK Biobank).
