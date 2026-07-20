"""Small, independently switchable E2E architecture candidates.

Each module is intentionally residual and zero-initialized so that the base
E2E model is recovered exactly at initialization.  The modules expose compact
diagnostic summaries; they never print tensors during normal training.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _rms(x):
    return x.detach().float().square().mean().sqrt()


class MaskConditionedInitialField(nn.Module):
    """Refine shift-back H with explicit mask-conditioned features."""

    def __init__(self, bands=28):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Conv2d(bands * 3, bands, 1, bias=True),
            nn.SiLU(),
            nn.Conv2d(bands, bands, 3, 1, 1, groups=bands, bias=False),
            nn.SiLU(),
            nn.Conv2d(bands, bands, 1, bias=True),
        )
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)
        self._last_stats = {}

    def forward(self, h, mask):
        correction = self.adapter(torch.cat([h, mask, h * mask], dim=1))
        with torch.no_grad():
            self._last_stats = {
                "input_rms": float(_rms(h).cpu()),
                "correction_rms": float(_rms(correction).cpu()),
                "correction_ratio": float((_rms(correction) / (_rms(h) + 1e-8)).cpu()),
            }
        return h + correction

    def diagnostic_stats(self):
        return dict(self._last_stats)


class WavelengthAxisReconstruction(nn.Module):
    """Low/high-frequency refinement on the explicit real 28-band axis."""

    def __init__(self, bands=28, cutoff_init=0.28):
        super().__init__()
        cutoff_init = min(max(float(cutoff_init), 0.05), 0.95)
        self.bands = int(bands)
        self.cutoff_logit = nn.Parameter(
            torch.tensor(math.log(cutoff_init / (1.0 - cutoff_init)))
        )
        self.refine = nn.Sequential(
            nn.Conv2d(bands * 4, bands, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(bands, bands, 3, 1, 1, groups=bands, bias=False),
            nn.GELU(),
            nn.Conv2d(bands, bands, 1, bias=True),
        )
        nn.init.zeros_(self.refine[-1].weight)
        nn.init.zeros_(self.refine[-1].bias)
        self._last_stats = {}

    def _split(self, h):
        spectrum = torch.fft.rfft(h, n=self.bands, dim=1)
        freq = torch.fft.rfftfreq(self.bands, device=h.device, dtype=h.dtype)
        freq = freq / freq.max().clamp_min(1e-6)
        cutoff = 0.05 + 0.90 * torch.sigmoid(self.cutoff_logit)
        low_filter = torch.exp(-0.5 * (freq / cutoff.clamp_min(1e-4)).square())
        low = torch.fft.irfft(
            spectrum * low_filter.view(1, -1, 1, 1), n=self.bands, dim=1
        )
        return low, h - low, cutoff

    def forward(self, h, mask):
        low, high, cutoff = self._split(h)
        correction = self.refine(torch.cat([low, high, mask, h * mask], dim=1))
        with torch.no_grad():
            base_rms = _rms(h)
            self._last_stats = {
                "cutoff": float(cutoff.detach().cpu()),
                "low_energy_ratio": float((_rms(low) / (base_rms + 1e-8)).cpu()),
                "high_energy_ratio": float((_rms(high) / (base_rms + 1e-8)).cpu()),
                "correction_ratio": float((_rms(correction) / (base_rms + 1e-8)).cpu()),
            }
        return h + correction

    def diagnostic_stats(self):
        return dict(self._last_stats)

