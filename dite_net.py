"""
DITE-Net: A Dendritic-Inhibitory Temporal Evidence Network for AMR.

Input:  x with shape (B, 2, L), where channel 0/1 are I/Q samples.
Output: logits with shape (B, num_classes).

The design maps several neural-circuit ideas to AMC:
1) oscillatory encoding: learnable neural oscillation bands for periodic/IQ phase patterns;
2) dendritic compartments: parallel dilated branches for multi-scale temporal evidence;
3) excitation-inhibition balance: inhibitory gate suppresses noisy responses;
4) predictive coding: prediction-error correction forms an auxiliary consistency term;
5) recurrent temporal integration: GRU integrates evidence over the IQ sequence.
"""

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, padding: Optional[int] = None,
                 groups: int = 1, dropout: float = 0.0):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, groups=groups, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x: torch.Tensor, return_aux=False):
        return self.net(x)


class OscillatoryEncoding(nn.Module):
    """Learnable sinusoidal carrier bank for IQ signals.

    It projects raw IQ sequences onto multiple oscillatory carriers. This is not an FFT;
    it is a lightweight neural oscillation prior with learnable frequency scale and phase.
    """

    def __init__(self, out_channels: int = 64, num_bands: int = 8, input_dim: int = 2, dropout: float = 0.1):
        super().__init__()
        self.num_bands = num_bands
        self.input_dim = input_dim
        init_freq = torch.linspace(1.0, float(num_bands), num_bands)
        self.log_freq = nn.Parameter(torch.log(init_freq))
        self.phase = nn.Parameter(torch.zeros(num_bands))
        self.proj = nn.Sequential(
            nn.Conv1d(input_dim * 2 * num_bands, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 2, L)
        b, c, l = x.shape
        t = torch.linspace(0.0, 1.0, l, device=x.device, dtype=x.dtype).view(1, 1, l)
        freq = F.softplus(self.log_freq).view(1, self.num_bands, 1)
        phase = self.phase.view(1, self.num_bands, 1)
        angle = 2.0 * math.pi * freq * t + phase
        sin_basis = torch.sin(angle)
        cos_basis = torch.cos(angle)

        # (B, C, K, L) -> (B, C*K, L), for both sin and cos carriers
        x_exp = x.unsqueeze(2)
        osc_sin = (x_exp * sin_basis.unsqueeze(1)).reshape(b, c * self.num_bands, l)
        osc_cos = (x_exp * cos_basis.unsqueeze(1)).reshape(b, c * self.num_bands, l)
        return self.proj(torch.cat([osc_sin, osc_cos], dim=1))


class DendriticCompartmentBlock(nn.Module):
    """Parallel dilated temporal branches with adaptive routing."""

    def __init__(self, channels: int, dilations: List[int] = (1, 2, 4, 8), dropout: float = 0.1):
        super().__init__()
        self.num_branches = len(dilations)
        self.branches = nn.ModuleList()
        for d in dilations:
            self.branches.append(nn.Sequential(
                nn.Conv1d(channels, channels, kernel_size=3, padding=d, dilation=d, groups=channels, bias=False),
                nn.BatchNorm1d(channels),
                nn.GELU(),
                nn.Conv1d(channels, channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(channels),
                nn.GELU(),
                nn.Dropout(dropout),
            ))
        self.router = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, max(channels // 4, 8)),
            nn.GELU(),
            nn.Linear(max(channels // 4, 8), self.num_branches),
        )
        self.out_norm = nn.BatchNorm1d(channels)

    def forward(self, x: torch.Tensor, return_aux=False):
        gates = torch.softmax(self.router(x), dim=-1)  # (B, K)
        branch_out = torch.stack([branch(x) for branch in self.branches], dim=1)
        y = (branch_out * gates[:, :, None, None]).sum(dim=1)
        out = self.out_norm(x + y)
        if return_aux:
            return out, {"dci_weights": gates}
        return out


class ExcitationInhibitionBlock(nn.Module):
    """E/I balance: excitatory evidence is adaptively suppressed by inhibitory activity."""

    def __init__(self, channels: int, dropout: float = 0.1):
        super().__init__()
        self.excitation = ConvBNAct1D(channels, channels, kernel_size=3, dropout=dropout)
        self.inhibition = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=5, padding=2, groups=channels, bias=False),
            nn.BatchNorm1d(channels),
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.norm = nn.BatchNorm1d(channels)

    def forward(self, x: torch.Tensor, return_aux=False):
        e = self.excitation(x)
        i = self.inhibition(x)
        out = self.norm(x + e * (1.0 - i))
        if return_aux:
            return out, {"eir_gate": i}
        return out


class SoftLIFGate(nn.Module):
    """Differentiable spiking-like gate; keeps the model easy to train with CE loss."""

    def __init__(self, channels: int, init_threshold: float = 0.0, tau: float = 1.0):
        super().__init__()
        self.threshold = nn.Parameter(torch.full((1, channels, 1), init_threshold))
        self.log_tau = nn.Parameter(torch.log(torch.tensor(float(tau))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tau = F.softplus(self.log_tau) + 1e-4
        spike_prob = torch.sigmoid((x - self.threshold) / tau)
        return x * spike_prob


class PredictiveCodingBlock(nn.Module):
    """One-step local prediction and prediction-error correction."""

    def __init__(self, channels: int):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=5, padding=2, groups=channels, bias=False),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size=1),
        )
        self.error_to_state = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(channels),
            nn.GELU(),
        )
        self.gamma = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor):
        pred = self.predictor(x)
        error = x - pred
        y = x + self.gamma.tanh() * self.error_to_state(error)
        pred_loss = error.pow(2).mean()
        return y, pred_loss


class ConnectomicSparseMixer(nn.Module):
    """Sparse local-plus-long-range channel mixing inspired by cortical connectomes."""

    def __init__(self, channels: int, local_radius: int = 4, long_range_stride: int = 11, dropout: float = 0.1):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(channels, channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        idx = torch.arange(channels)
        mask = (idx[:, None] - idx[None, :]).abs() <= local_radius
        long_range = ((idx[:, None] * long_range_stride + idx[None, :]) % max(channels, 1)) == 0
        mask = (mask | long_range).float()
        self.register_buffer("mask", mask)

        self.norm = nn.BatchNorm1d(channels)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x.transpose(1, 2), self.weight * self.mask, self.bias).transpose(1, 2)
        y = self.dropout(self.act(self.norm(y)))
        return x + self.alpha.tanh() * y


class BrainWidePriorContext(nn.Module):
    """Use global prior-like context to modulate distributed temporal regions."""

    def __init__(self, channels: int, num_regions: int = 8, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.num_regions = num_regions
        heads = max(1, min(num_heads, channels))
        while channels % heads != 0 and heads > 1:
            heads -= 1
        self.region_norm = nn.LayerNorm(channels)
        self.prior_norm = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, heads, dropout=dropout, batch_first=True)
        self.to_affine = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, channels * 2),
        )
        self.dropout = nn.Dropout(dropout)
        self.gamma = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L). Region tokens act like distributed brain areas.
        regions = F.adaptive_avg_pool1d(x, self.num_regions).transpose(1, 2)
        prior = F.adaptive_avg_pool1d(x, 1).transpose(1, 2)
        context, _ = self.attn(self.prior_norm(prior), self.region_norm(regions), self.region_norm(regions))
        scale, shift = self.to_affine(context.squeeze(1)).chunk(2, dim=-1)
        scale = torch.tanh(scale).unsqueeze(-1)
        shift = shift.unsqueeze(-1)
        y = x * (1.0 + self.gamma.tanh() * scale) + self.gamma.tanh() * shift
        return self.dropout(y)


class NeuralCircuitBlock(nn.Module):
    def __init__(self, channels: int, dilations: List[int] = (1, 2, 4, 8), dropout: float = 0.1):
        super().__init__()
        self.dendritic = DendriticCompartmentBlock(channels, dilations=dilations, dropout=dropout)
        self.ei = ExcitationInhibitionBlock(channels, dropout=dropout)
        self.lif = SoftLIFGate(channels)
        self.pc = PredictiveCodingBlock(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, return_aux=False):
        aux = {}
        if return_aux:
            y, dci_aux = self.dendritic(x, return_aux=True)
            aux.update(dci_aux)
        else:
            y = self.dendritic(x)

        if return_aux:
            y, eir_aux = self.ei(y, return_aux=True)
            aux.update(eir_aux)
        else:
            y = self.ei(y)

        y = self.lif(y)
        y, pred_loss = self.pc(y)

        if return_aux:
            return self.dropout(y), pred_loss, aux
        return self.dropout(y), pred_loss


class DITENet(nn.Module):
    """Neural-Circuit-inspired network for IQ-based modulation recognition."""

    def __init__(self, input_dim: int = 2, seq_len: int = 128, num_classes: int = 11,
                 hidden_dim: int = 64, num_bands: int = 8, num_blocks: int = 3,
                 dropout: float = 0.1, aux_pred_weight: float = 0.05):
        super().__init__()
        self.seq_len = seq_len
        self.num_classes = num_classes
        self.aux_pred_weight = aux_pred_weight
        self._last_aux_loss = None

        self.osc = OscillatoryEncoding(out_channels=hidden_dim, num_bands=num_bands,
                                       input_dim=input_dim, dropout=dropout)
        self.stem = nn.Sequential(
            ConvBNAct1D(hidden_dim, hidden_dim, kernel_size=3, dropout=dropout),
            ConvBNAct1D(hidden_dim, hidden_dim, kernel_size=3, dropout=dropout),
        )
        self.blocks = nn.ModuleList([
            NeuralCircuitBlock(hidden_dim, dilations=(1, 2, 4, 8), dropout=dropout)
            for _ in range(num_blocks)
        ])
        self.connectomic_mixer = ConnectomicSparseMixer(hidden_dim, dropout=dropout)
        self.brainwide_context = BrainWidePriorContext(hidden_dim, dropout=dropout)

        # Bidirectional temporal evidence integration.
        self.temporal_integrator = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.temporal_norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor, return_aux=False) -> torch.Tensor:
        # Accept both (B, 2, L) and (B, L, 2), but internal format is (B, 2, L).
        if x.dim() != 3:
            raise ValueError(f"DITENet expects 3D input, got shape {tuple(x.shape)}")
        if x.size(1) != 2 and x.size(-1) == 2:
            x = x.transpose(1, 2).contiguous()

        y = self.osc(x)
        y = self.stem(y)

        pred_losses = []
        aux_outputs = {"dci_weights": [], "eir_gate": []}

        for block in self.blocks:
            if return_aux:
                y, pred_loss, block_aux = block(y, return_aux=True)
                if "dci_weights" in block_aux:
                    aux_outputs["dci_weights"].append(block_aux["dci_weights"])
                if "eir_gate" in block_aux:
                    aux_outputs["eir_gate"].append(block_aux["eir_gate"])
            else:
                y, pred_loss = block(y)

            pred_losses.append(pred_loss)
        y = self.connectomic_mixer(y)
        y = self.brainwide_context(y)

        # (B, C, L) -> (B, L, C)
        y = y.transpose(1, 2)
        y, _ = self.temporal_integrator(y)

        y = self.temporal_norm(y)
        y = y.transpose(1, 2)

        avg_feat = F.adaptive_avg_pool1d(y, 1).squeeze(-1)
        max_feat = F.adaptive_max_pool1d(y, 1).squeeze(-1)
        embedding = torch.cat([avg_feat, max_feat], dim=1)

        logits = self.classifier(embedding)

        self._last_aux_loss = torch.stack(pred_losses).mean() if pred_losses else logits.new_tensor(0.0)

        if return_aux:
            aux_outputs["embedding"] = embedding
            if aux_outputs["dci_weights"]:
                aux_outputs["dci_weights"] = torch.stack(aux_outputs["dci_weights"], dim=1)
            if aux_outputs["eir_gate"]:
                aux_outputs["eir_gate"] = torch.stack(aux_outputs["eir_gate"], dim=1)
            return logits, aux_outputs

        return logits

    def get_aux_loss(self) -> torch.Tensor:
        if self._last_aux_loss is None:
            return next(self.parameters()).new_tensor(0.0)
        return self._last_aux_loss
