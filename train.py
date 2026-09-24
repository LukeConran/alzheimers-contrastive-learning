import os
import json
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import models
import datasets
from losses import SupConLoss          # supervised contrastive loss

from medicalnet_model import generate_model

def get_3d_sincos_pos_embed(D, H, W, dim, device):
    def sincos_embedding(pos, dim_half):
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim_half, 2).float() / dim_half)).to(device)
        sinusoid_inp = torch.einsum("i,j->ij", pos.float(), inv_freq)
        emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)
        return emb

    d_pos = torch.arange(D, device=device)
    h_pos = torch.arange(H, device=device)
    w_pos = torch.arange(W, device=device)

    dim_each = dim // 3
    d_emb = sincos_embedding(d_pos, dim_each)
    h_emb = sincos_embedding(h_pos, dim_each)
    w_emb = sincos_embedding(w_pos, dim_each)

    pos_embed = torch.zeros((D, H, W, dim), device=device)
    for d in range(D):
        for h in range(H):
            for w in range(W):
                pos_embed[d, h, w] = torch.cat([d_emb[d], h_emb[h], w_emb[w]], dim=-1)

    return pos_embed.view(-1, dim)


def train(args):
    # ── Output directory ──────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Device setup ──────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Model and optimizer setup ─────────────────────────────────────────────
    if args.backbone == "resnet":
        classifier, parameters = generate_model(model_name=args.model_name, num_seg_classes=3, phase='train', pretrain_path=args.checkpoint_pretrain)
        if args.checkpoint_pretrain:
            print("Use pretrained model")
            params = [{ 'params': parameters['base_parameters'], 'lr': args.lr }, { 'params': parameters['new_parameters'], 'lr': args.lr*100 }]
            optimizer = torch.optim.Adam(params, weight_decay=1e-3)
        else:
            print("Train from scratch")
            optimizer = torch.optim.Adam(parameters, weight_decay=1e-3, lr=args.lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    # ── Loss functions ────────────────────────────────────────────────────────
    # Cross-entropy is always used for classification.
    # SupConLoss is added on top when --contrastive is enabled.
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    supcon = SupConLoss(temperature=args.temperature)

    # ── Dataset and dataloader setup ─────────────────────────────────────────
    # When contrastive mode is on, use ContrastiveDataset which returns
    # (view1, view2, label) instead of (image, label).
    # The validation set always uses the standard dataset (no augmentation needed).
    if args.contrastive:
        print(f"Contrastive mode ON  (lambda={args.lambda_con}, temperature={args.temperature})")
        train_dataset = datasets.MyDataSet.ContrastiveDataset(
            json_path=args.train_json, image_dir=args.train_image_dir
        )
    else:
        print("Contrastive mode OFF — standard cross-entropy training")
        train_dataset = datasets.MyDataSet.MyDataset(
            json_path=args.train_json, image_dir=args.train_image_dir
        )

    valid_dataset = datasets.MyDataSet.MyDataset(
        json_path=args.valid_json, image_dir=args.valid_image_dir
    )

    print(f"Length of training dataset: {len(train_dataset)}")
    print(f"Length of validation dataset: {len(valid_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,        # parallel CPU workers for loading
        pin_memory=True,      # faster CPU→GPU transfer
        persistent_workers=True,  # don't respawn workers each epoch
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
    )

    # ── Resume from checkpoint ────────────────────────────────────────────────
    start_epoch = 0
    best_val_acc = 0.0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        classifier.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch']
        best_val_acc = checkpoint.get('best_val_acc', 0.0)
        print(f"Resumed from checkpoint '{args.resume}' at epoch {start_epoch}", flush=True)

    # ── Metrics log (loaded from file if resuming) ────────────────────────────
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    if args.resume and os.path.exists(metrics_path):
        with open(metrics_path) as f:
            epoch_metrics = json.load(f)
    else:
        epoch_metrics = []

    # ── Training loop ─────────────────────────────────────────────────────────
    n_train_batches = len(train_loader)
    for epoch in range(start_epoch, args.epochs):
        classifier.train()
        total_loss = 0
        correct = 0
        total = 0
        epoch_start = time.time()

        for i, group in enumerate(optimizer.param_groups):
            print(f"Epoch {epoch + 1} - Learning Rate Group {i}: {group['lr']:.6f}", flush=True)

        # ── Per-batch training step ───────────────────────────────────────────
        for batch_idx, batch in enumerate(train_loader):

            if args.contrastive:
                # ── Contrastive + classification joint loss ───────────────────
                # Batch contains two independently-augmented views of each scan.
                # view1/view2 shape: (B, D, H, W) — channel dim added below.
                view1, view2, labels = batch
                view1   = view1.to(device)   # (B, 1, D, H, W)
                view2   = view2.to(device)   # (B, 1, D, H, W)
                labels  = labels.to(device)

                # Classification loss: use view1 through the standard fc head
                logits = classifier(view1)
                ce_loss = criterion(logits, labels)

                # Contrastive loss:
                #   1. Get L2-normalized projections for both views
                #   2. Concatenate to shape (2*B, proj_dim)
                #   3. SupConLoss uses labels to identify which pairs are positive
                emb1 = classifier(view1, return_embedding=True)   # (B, proj_dim)
                emb2 = classifier(view2, return_embedding=True)   # (B, proj_dim)
                features = torch.cat([emb1, emb2], dim=0)         # (2*B, proj_dim)
                con_loss = supcon(features, labels)

                # Combined loss: cross-entropy + lambda * contrastive
                # lambda_con controls the trade-off.
                # Start with lambda=0.5 and tune; if val_acc drops, reduce it.
                loss = ce_loss + args.lambda_con * con_loss

            else:
                # ── Standard cross-entropy only ───────────────────────────────
                images, labels = batch
                images = images.to(device)   # (B, 1, D, H, W)
                labels = labels.to(device)
                logits = classifier(images)
                loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == n_train_batches:
                running_loss = total_loss / total
                running_acc = correct / total
                print(f"  [Epoch {epoch+1}/{args.epochs}] Batch {batch_idx+1}/{n_train_batches} — "
                      f"loss: {running_loss:.4f}  acc: {running_acc:.4f}", flush=True)

        train_loss = total_loss / total
        train_acc = correct / total
        scheduler.step()

        # ── Validation (always standard, no augmentation) ─────────────────────
        classifier.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in valid_loader:
                images = images.to(device)   # (B, 1, D, H, W)
                labels = labels.to(device)
                logits = classifier(images)
                loss = criterion(logits, labels)
                val_loss += loss.item() * labels.size(0)
                val_correct += (logits.argmax(dim=1) == labels).sum().item()
                val_total += labels.size(0)

        val_avg_loss = val_loss / val_total
        val_acc = val_correct / val_total

        # ── Checkpointing ─────────────────────────────────────────────────────
        if (epoch + 1) % 10 == 0:
            ckpt_path = os.path.join(args.output_dir, f"model_{args.model_name}_bs{args.batch_size}_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model': classifier.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_val_acc': best_val_acc,
            }, ckpt_path)
            print(f"Saved checkpoint at epoch {epoch+1}.", flush=True)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(classifier.state_dict(), os.path.join(args.output_dir, f"best_model_{args.model_name}_bs{args.batch_size}.pth"))
            print(f"New best model saved at epoch {epoch+1} with val_acc: {val_acc:.4f}", flush=True)

        epoch_time = time.time() - epoch_start
        print(f"[Epoch {epoch+1}/{args.epochs}] Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_avg_loss:.4f}  Val Acc: {val_acc:.4f} | "
              f"Time: {epoch_time:.1f}s", flush=True)

        # ── Save epoch metrics ────────────────────────────────────────────────
        epoch_metrics.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_avg_loss,
            'val_acc': val_acc,
        })
        with open(metrics_path, 'w') as f:
            json.dump(epoch_metrics, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SwinViT Classifier on ADNI MRI dataset")

    parser.add_argument("--train_json", type=str, required=True, help="Path to training JSON file")
    parser.add_argument("--train_image_dir", type=str, required=True, help="Path to training image directory")
    parser.add_argument("--valid_json", type=str, required=True, help="Path to validation JSON file")
    parser.add_argument("--valid_image_dir", type=str, required=True, help="Path to validation image directory")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--backbone", type=str, default='resnet')
    parser.add_argument("--checkpoint_pretrain", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Directory to save model checkpoints")
    parser.add_argument("--model_name", type=str, default='resnet10')
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to a checkpoint (.pth) to resume training from.")

    # ── Contrastive learning arguments ────────────────────────────────────────
    parser.add_argument("--contrastive", action="store_true",
                        help="Enable supervised contrastive loss alongside cross-entropy.")
    parser.add_argument("--lambda_con", type=float, default=0.5,
                        help="Weight on the contrastive loss term. "
                             "Combined loss = CE + lambda_con * SupCon. "
                             "Tune between 0.1 and 1.0; reduce if val_acc drops.")
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="SupCon temperature. Lower = sharper similarity distribution. "
                             "0.1 is a good starting point for medical imaging.")
    parser.add_argument("--label_smoothing", type=float, default=0.0,
                        help="Label smoothing for CrossEntropyLoss (0.0 = off). "
                             "0.1 is recommended for CE runs to prevent overconfident predictions.")

    args = parser.parse_args()
    print(args)
    train(args)
