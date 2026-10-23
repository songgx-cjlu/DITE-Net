"""Train or evaluate the full DITE-Net on two public RadioML benchmarks.

No comparison models, architectural variants or ablation experiments.
"""

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from datasets import load_radioml
from dite_net import DITENet


def parse_args():
    parser = argparse.ArgumentParser(description="Full DITE-Net: training and evaluation")
    parser.add_argument("--dataset", choices=["rml2016a", "rml22"], required=True)
    parser.add_argument("--data-path", required=True, help="Path to benchmark .pkl file")
    parser.add_argument("--output-dir", default="outputs/dite_net")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--milestone", type=int, default=30)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--checkpoint", help="Checkpoint for --eval-only")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Match the original experimental implementation.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def loss_fn(model, criterion, logits, labels):
    return criterion(logits, labels) + model.aux_pred_weight * model.get_aux_loss()


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    correct = total = 0
    loss_sum = 0.0
    snr_correct = defaultdict(lambda: [0, 0])

    for batch in loader:
        x, labels = batch[:2]
        snrs = batch[2] if len(batch) == 3 else None
        x, labels = x.to(device), labels.to(device)
        logits = model(x)
        loss_sum += loss_fn(model, criterion, logits, labels).item()
        predicted = logits.argmax(dim=1)
        correct += (predicted == labels).sum().item()
        total += labels.numel()
        if snrs is not None:
            for prediction, truth, snr in zip(predicted.cpu(), labels.cpu(), snrs):
                key = int(snr)
                snr_correct[key][0] += int(prediction == truth)
                snr_correct[key][1] += 1

    per_snr = {str(snr): hits / count for snr, (hits, count) in sorted(snr_correct.items())}
    low = [acc for snr, acc in per_snr.items() if -20 <= int(snr) <= 0]
    high = [acc for snr, acc in per_snr.items() if 0 < int(snr) <= 18]
    return {"loss": loss_sum / max(len(loader), 1),
            "accuracy": correct / max(total, 1),
            "low_snr_accuracy": float(np.mean(low)) if low else None,
            "high_snr_accuracy": float(np.mean(high)) if high else None,
            "per_snr_accuracy": per_snr}


def load_state(path, model, device):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    state = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state)
    return checkpoint


def main():
    args = parse_args()
    if args.epochs < 1:
        raise ValueError("--epochs must be >= 1")
    set_seed(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    (train_loader, val_loader, test_loader), data_info = load_radioml(
        args.data_path, dataset=args.dataset,
        batch_size=args.batch_size, num_workers=args.num_workers)
    model = DITENet(num_classes=len(data_info["classes"]),
                    seq_len=data_info["input_length"]).to(device)
    criterion = nn.CrossEntropyLoss()
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[DITE-Net] dataset={args.dataset}, device={device}, params={params:,}, "
          f"train/val/test={data_info['counts']}")
    best_path = out / "best_model.pth"

    if args.eval_only:
        if not args.checkpoint:
            raise ValueError("--eval-only requires --checkpoint PATH")
        load_state(args.checkpoint, model, device)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[args.milestone], gamma=0.1)
        best_val_acc = -1.0
        with (out / "history.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])
            for epoch in range(1, args.epochs + 1):
                model.train()
                running_loss = 0.0
                correct = total = 0
                for inputs, labels in train_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad(set_to_none=True)
                    logits = model(inputs)
                    loss = loss_fn(model, criterion, logits, labels)
                    loss.backward()
                    if args.grad_clip > 0:
                        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    optimizer.step()
                    running_loss += loss.item()
                    correct += (logits.argmax(dim=1) == labels).sum().item()
                    total += len(labels)
                train_acc = correct / max(1, total)
                train_loss = running_loss / max(1, len(train_loader))
                val = evaluate(model, val_loader, criterion, device)
                scheduler.step()
                writer.writerow([epoch, train_loss, train_acc, val["loss"], val["accuracy"]])
                f.flush()
                if val["accuracy"] > best_val_acc:
                    best_val_acc = val["accuracy"]
                    torch.save({"state_dict": model.state_dict(), "epoch": epoch,
                                "val_accuracy": best_val_acc, "classes": data_info["classes"],
                                "dataset": args.dataset, "seed": args.seed}, best_path)
                print(f"epoch={epoch:03d}/{args.epochs} train_acc={train_acc:.4f} "
                      f"val_acc={val['accuracy']:.4f} best_val={best_val_acc:.4f}")
        load_state(best_path, model, device)

    result = evaluate(model, test_loader, criterion, device)
    result.update({"dataset": args.dataset, "seed": args.seed,
                   "parameters": params, "split": "stratified 60/20/20", "data": data_info,
                   "checkpoint": str(args.checkpoint if args.eval_only else best_path)})
    (out / "test_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[TEST] accuracy={result['accuracy']:.6f}")
    print(f"[SAVED] {out / 'test_metrics.json'}")


if __name__ == "__main__":
    main()
