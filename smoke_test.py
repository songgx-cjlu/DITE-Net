"""Minimal CPU forward/backward test without downloading a dataset."""

import torch
import torch.nn as nn
from dite_net import DITENet


def main():
    torch.set_num_threads(min(torch.get_num_threads(), 2))
    model = DITENet(input_dim=2, seq_len=128, num_classes=11)
    model.train()
    sample = torch.randn(2, 2, 128)
    targets = torch.tensor([0, 1])
    logits = model(sample)
    loss = nn.CrossEntropyLoss()(logits, targets) + model.aux_pred_weight * model.get_aux_loss()
    loss.backward()
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert tuple(logits.shape) == (2, 11)
    assert params == 220959, f"Unexpected parameter count: {params}"
    assert torch.isfinite(logits).all() and torch.isfinite(loss)
    assert any(p.grad is not None for p in model.parameters())
    print(f"PASS: logits={tuple(logits.shape)}, parameters={params:,}, backward=OK")


if __name__ == "__main__":
    main()
