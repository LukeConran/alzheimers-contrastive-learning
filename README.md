# Alzheimer's MRI Classification: CE vs. Supervised Contrastive Learning

3D brain MRI classification into **Alzheimer's Disease (AD)**, **Mild Cognitive Impairment (MCI)**, and **Normal Cognition (NC)** using 3D ResNet backbones, comparing plain cross-entropy training against supervised contrastive pretraining. Runs on an HPC cluster via SLURM.

## Motivation: Why this repo looks the way it does

The original plan was **self-supervised contrastive learning** (SimCLR-style): pretrain a 3D ResNet encoder on a large pool of *unlabeled* MRI scans until it produced features that were easy to classify, then fine-tune a lightweight classifier on a much smaller labeled subset. The appeal was practical: because clinician-labeled MRI scans are expensive to produce, if a model could learn good representations from unlabeled scans first, the amount of labeled data (and clinician time) needed downstream would shrink substantially.

That plan ran into a structural problem: **ADNI, the dataset this project uses, is fully labeled.** There is no large unlabeled pool sitting alongside it, so every scan already has an AD / MCI / NC label. Self-supervised pretraining has nothing to offer when there's nothing unlabeled to pretrain on; the labeling-burden problem SimCLR was meant to solve doesn't exist within this dataset. (It would still be viable on a separate, mostly-unlabeled corpus like UK Biobank, but that's a different project.)

So the project pivoted to a question that ADNI's data supports: **given labels, can we use them more effectively than cross-entropy alone?** Instead of self-supervised pretraining, this repo uses **Supervised Contrastive Loss** ([Khosla et al., NeurIPS 2020](https://arxiv.org/abs/2004.11362)), which pulls embeddings of the same class together and pushs different classes apart, using the labels directly. This is also trained jointly with (or ahead of) the classification head. The research question thus became:

> **Does supervised contrastive learning improve classification performance over plain cross-entropy, given the exact same labeled dataset and backbone?**

## Results

Final-epoch metrics from `results/{backbone}/{mode}/metrics.json` (50 epochs, same data/backbone across CE and SupCon runs):

| Backbone | Mode | Train Acc | Val Acc | Val Loss |
|---|---|---|---|---|
| ResNet-10 | CE | 0.692 | 0.361 | 3.779 |
| ResNet-10 | SupCon | 0.626 | **0.599** | **0.848** |
| ResNet-18 | CE | 0.958 | 0.669 | 1.016 |
| ResNet-18 | SupCon | 0.767 | **0.679** | **0.756** |

**ResNet-10** is the clearest result: CE overfits hard (val accuracy collapses to 0.36 despite 0.69 train accuracy, val loss balloons to 3.78), while SupCon generalizes far better (0.60 val accuracy, 0.85 val loss) despite lower training accuracy. **ResNet-18** has enough capacity that both modes generalize reasonably, but SupCon still edges out CE on validation accuracy and has substantially lower validation loss, being consistent with the contrastive term acting as a regularizer, not just an accuracy booster.

### Figures (ResNet-18)

Confusion matrices, ROC curves, training curves, and a direct comparison plot are in [`figures/resnet18/`](figures/resnet18/).

The metric comparison below is the clearest single view of the result: SupCon edges out CE on AUC while training to a lower validation loss.

![Comparison](figures/resnet18/comparison.png)

Training curves show how each run got there, and we can see CE and SupCon both converge without the overfitting collapse seen on ResNet-10, but SupCon's validation loss stays lower throughout training, consistent with the contrastive term acting as a regularizer.

![Training curves](figures/resnet18/training_curves.png)

ROC and confusion-matrix pairs below let you compare per-class separability and error patterns directly: look for tighter clustering along the diagonal in the SupCon confusion matrix and higher AUC in the SupCon ROC curves.

| CE | SupCon |
|---|---|
| ![ROC CE](figures/resnet18/roc_ce.png) | ![ROC SupCon](figures/resnet18/roc_supcon.png) |
| ![CM CE](figures/resnet18/cm_ce.png) | ![CM SupCon](figures/resnet18/cm_supcon.png) |

## Project structure

- `datasets/mri_dataset.py` — dataset classes: standard, contrastive (two augmented views per scan)
- `models/resnet.py` — 3D ResNet with a dual head: classification logits *and* an L2-normalized 128-dim contrastive embedding
- `models/resnet_org.py` — the original CE-only baseline ResNet, without the contrastive projector head
- `losses/supcon_loss.py` — Supervised Contrastive Loss ([Khosla et al., NeurIPS 2020](https://arxiv.org/abs/2004.11362))
- `medicalnet_model.py` — model factory, MedicalNet pretrained weight loading, param group splitting
- `train.py` — training loop (CE and/or SupCon, depending on `--contrastive`)
- `test.py` — evaluation: per-class accuracy, AUC, precision, recall
- `plot.py` — generates the comparison / ROC / confusion-matrix / training-curve figures above
- `bin/ce/`, `bin/contrastive/` — SLURM job scripts (smoke test, test, train) for each mode, both backbones

## Data format

Images are `.npz` files with key `"image_mr"` (3D float array, shape `(D, H, W)`). Labels come from a JSON manifest:

```json
[{"id": "001", "label": "Normal Cognition"}, ...]
```

Label mapping: `"Alzheimer's Disease" → 0`, `"Mild Cognitive Impairment" → 1`, `"Normal Cognition" → 2`.

> The raw ADNI JSON uses `"Dementia"` instead of `"Alzheimer's Disease"` — `mri_dataset.py` already handles this mapping.

## Environment

This project runs on an HPC cluster via SLURM, not a local install. Thus, there's no `pip install` step because the cluster provides pinned modules directly:

```bash
module load GCC/12.3.0 OpenMPI/4.1.5 PyTorch/2.1.2-CUDA-12.1.1 scikit-learn/1.3.1
```

Every script under `bin/` loads these modules before running Python. See any `.slurm` file there for the full pattern.

## Reproducing the CE vs. SupCon comparison

**Standard cross-entropy:**

```bash
python train.py \
  --train_json <path> --train_image_dir <path> \
  --valid_json <path> --valid_image_dir <path> \
  --backbone resnet --model_name resnet10 \
  --lr 1e-4 --batch_size 8 --epochs 100
```

**Supervised contrastive:**

```bash
python train.py \
  --train_json <path> --train_image_dir <path> \
  --valid_json <path> --valid_image_dir <path> \
  --backbone resnet --model_name resnet10 \
  --lr 1e-4 --batch_size 8 --epochs 100 \
  --contrastive --lambda_con 0.5 --temperature 0.1
```

With `--contrastive`, `ContrastiveDataset` returns two augmented views per scan and `SupConLoss` is added alongside cross-entropy. Each view requires its own backbone forward pass, so this costs roughly **2x the forward-pass compute per epoch** compared to CE — both the classification logits and the contrastive embedding for view1 are computed from a single shared pass (see `ResNet.forward(..., return_both=True)`), rather than the naive 3x you'd get from forwarding view1 twice.

**With MedicalNet pretrained weights (recommended):**

```bash
python train.py ... --checkpoint_pretrain /home/lukeconran/alzheimers/MedicalNet_pytorch_files2/pretrain/resnet_10.pth
```

When used, the backbone trains at `--lr` and the classification head trains at `--lr * 100`.

**Resuming a timed-out job:**

```bash
python train.py ... --resume /home/lukeconran/alzheimers/results/resnet10/ce/model_resnet10_bs8_epoch_20.pth --epochs 30
```

**End-to-end reproduction of the comparison above:** run `bin/ce/train_resnet10.slurm` and `bin/contrastive/train_resnet10.slurm` (and the `resnet18` variants) in parallel, evaluate each with `test.py` for per-class accuracy/AUC/precision/recall, then run `plot.py` to regenerate the figures in `figures/`.

**Quick smoke test on a small subset:** `bin/ce/smoke_resnet10.slurm` and `bin/contrastive/smoke_resnet10.slurm` (and the `resnet18` variants) run the same pipeline for a single epoch, useful for confirming the environment and data paths work before committing to a full 50–100 epoch run.

## MRI-safe augmentations

Applied only during contrastive training, in `ContrastiveDataset._augment()`:
- Random axis flips (50% per axis)
- Additive Gaussian noise (σ = 0.05)
- Intensity scaling (0.9–1.1×)

## Limitations

- **SimCLR is not viable with ADNI alone.** As explained above, self-supervised pretraining needs an unlabeled pool to be worth anything, and ADNI is fully labeled, so there's nothing to pretrain on without pulling in a separate corpus (e.g. UK Biobank).

## References

- Khosla et al., *Supervised Contrastive Learning*, NeurIPS 2020 — [arXiv:2004.11362](https://arxiv.org/abs/2004.11362)
- Chen et al., *A Simple Framework for Contrastive Learning of Visual Representations (SimCLR)*, ICML 2020 — [arXiv:2002.05709](https://arxiv.org/abs/2002.05709)
