import torch
import torch.nn.functional as F
import random
import time

class WeakLatentAug:
    def __init__(self, noise_std=0.02, flip_prob=0.5):
        self.noise_std = noise_std
        self.flip_prob = flip_prob

        # self.random = random.Random()
        # self.torch_gen = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu")
        # self.torch_gen.manual_seed(int(time.time() * 1000))

    def __call__(self, h):
        # h: (B, C, H, W)
        noise = torch.randn(h.size(), dtype=h.dtype, layout=h.layout, device=h.device)
        h = h + noise * self.noise_std # light noise

        if random.random() < self.flip_prob:
            h = torch.flip(h, dims=[-1])  # horizontal flip
        if random.random() < self.flip_prob:
            h = torch.flip(h, dims=[-2])  # vertical flip

        return h

class StrongLatentAug:
    def __init__(self, noise_std=0.05, drop_prob=0.1, shift=True, flip=True):
        self.noise_std = noise_std
        self.drop_prob = drop_prob
        self.shift = shift
        self.flip = flip

        # self.random = random.Random()
        # self.torch_gen = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu")
        # self.torch_gen.manual_seed(int(time.time() * 1000))

    def __call__(self, h):
        # h: (B, C, H, W)
        noise = torch.randn(h.size(), dtype=h.dtype, layout=h.layout, device=h.device)
        h = h + noise * self.noise_std

        if self.drop_prob > 0:
            mask = (torch.rand(h.size(), dtype=h.dtype, layout=h.layout, device=h.device) > self.drop_prob).float()
            h = h * mask

        if self.flip:
            if random.random() < 0.5:
                h = torch.flip(h, dims=[-1])  # horizontal
            if random.random() < 0.5:
                h = torch.flip(h, dims=[-2])  # vertical

        if self.shift:
            h = self.random_shift(h)

        return h

    def random_shift(self, h, max_shift=1):
        # Translate via affine_grid (fast on GPU)
        B, C, H, W = h.shape
        device = h.device

        tx = random.uniform(-max_shift, max_shift) / W
        ty = random.uniform(-max_shift, max_shift) / H

        theta = torch.tensor([
            [1, 0, tx],
            [0, 1, ty]
        ], device=device, dtype=h.dtype).unsqueeze(0).repeat(B, 1, 1)

        grid = F.affine_grid(theta, h.size(), align_corners=False)
        return F.grid_sample(h, grid, align_corners=False, padding_mode='border')



