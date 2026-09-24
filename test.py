import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import datasets
from medicalnet_model import generate_model
from sklearn.metrics import roc_auc_score, precision_score, recall_score
import numpy as np

def test(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 选择模型 Select model
    if args.backbone == "resnet":
        model, _ = generate_model(model_name=args.model_name, num_seg_classes=3, phase='test')
    else:
        raise ValueError(f"Unsupported backbone: {args.backbone}")

    # 加载权重 Load weights
    if args.checkpoint is not None:
        print(f"Loading model checkpoint from {args.checkpoint}")
        state_dict = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state_dict)
    else:
        raise ValueError("Checkpoint path must be provided for testing.")

    model = model.to(device)
    model.eval()

    # 加载测试数据 Load test data
    test_cache_dir = os.path.join(args.cache_dir, "test") if args.cache_dir else None
    test_dataset = datasets.mri_dataset.MyDataset(
        json_path=args.test_json, image_dir=args.test_image_dir, cache_dir=test_cache_dir
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    all_preds = []
    all_labels = []
    all_logits = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)

            total_loss += loss.item() * labels.size(0)
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_samples += labels.size(0)

            all_preds.extend(logits.argmax(dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_logits.append(logits.cpu())

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    print(f"[Test] Loss: {avg_loss:.4f} | Accuracy: {accuracy:.4f}")

    # 统计指标 Statistical indicators
    all_logits = torch.cat(all_logits, dim=0).numpy()  # (N, C)
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    num_classes = all_logits.shape[1]

    # Overall AUC
    try:
        y_one_hot = np.eye(num_classes)[all_labels]
        auc_macro = roc_auc_score(y_one_hot, all_logits, average='macro', multi_class='ovr')
        print(f"[Test] Overall AUC (macro): {auc_macro:.4f}")
    except ValueError as e:
        print(f"[Warning] AUC calculation failed: {e}")

    # Per-class Accuracy, AUC, Precision, Recall
    print("\n[Per-Class Performance]")
    for i in range(num_classes):
        class_mask = (all_labels == i)
        class_labels = (all_labels == i).astype(int)  # binary for AUC
        class_scores = all_logits[:, i]
        acc_i = (all_preds[class_mask] == all_labels[class_mask]).mean() if class_mask.sum() > 0 else float('nan')

        try:
            auc_i = roc_auc_score(class_labels, class_scores)
        except ValueError as e:
            auc_i = float('nan')
            print(f"[Class {i}] AUC calculation failed: {e}")

        try:
            prec_i = precision_score(all_labels, all_preds, labels=[i], average='micro')
            rec_i = recall_score(all_labels, all_preds, labels=[i], average='micro')
        except ValueError:
            prec_i, rec_i = float('nan'), float('nan')

        print(f"Class {i}: Accuracy = {acc_i:.4f} | AUC = {auc_i:.4f} | "
              f"Precision = {prec_i:.4f} | Recall = {rec_i:.4f}")

    # 可选保存预测 Optional saving of predictions
    if args.save_preds:
        np.savez(args.save_preds, predictions=all_preds, labels=all_labels, logits=all_logits)
        print(f"Saved predictions to {args.save_preds}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test classifier on ADNI MRI dataset")
    parser.add_argument("--test_json", type=str, required=True, help="Path to testing JSON file")
    parser.add_argument("--test_image_dir", type=str, required=True, help="Path to testing image directory")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--backbone", type=str, default='resnet', help="Model type: 'resnet' or 'swin'")
    parser.add_argument("--model_name", type=str, default='resnet10')
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--save_preds", type=str, default=None, help="Optional path to save predictions as .npz")
    parser.add_argument("--cache_dir", type=str, default=None,
                         help="Directory to persist normalized volumes as .npy files "
                              "(e.g. a scratch path on HPC). Disabled by default.")

    args = parser.parse_args()
    print(args)
    print(f"Testing model: {args.model_name}")
    test(args)
