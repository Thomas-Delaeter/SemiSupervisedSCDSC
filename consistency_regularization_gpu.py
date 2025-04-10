import random
import torch
import torch.nn.functional as F

class HSIRandAugment:
    def __init__(self, apply_ops=None):
        self.apply_ops = apply_ops or ['flip', 'crop', 'rotate', 'affine']

    def __call__(self, batch):
        if 'flip' in self.apply_ops:
            batch = self.random_flip(batch)
        if 'crop' in self.apply_ops:
            batch = self.random_crop(batch)
        if 'rotate' in self.apply_ops:
            batch = self.random_rotate(batch)
        if 'affine' in self.apply_ops:
            batch = self.random_affine(batch)
        return batch

    def random_flip(self, batch):
        B = batch.shape[0]
        flip_h = torch.rand(B, device=batch.device) > 0.5
        flip_v = torch.rand(B, device=batch.device) > 0.5
        if flip_h.any():
            batch[flip_h] = torch.flip(batch[flip_h], dims=[3])
        if flip_v.any():
            batch[flip_v] = torch.flip(batch[flip_v], dims=[2])
        return batch

    def random_crop(self, batch, padding=2):
        B, C, H, W = batch.shape
        padded = F.pad(batch, (padding, padding, padding, padding), mode='reflect')
        Hp, Wp = H + 2 * padding, W + 2 * padding
        top = torch.randint(0, Hp - H + 1, (B,), device=batch.device)
        left = torch.randint(0, Wp - W + 1, (B,), device=batch.device)
        crops = torch.stack([
            padded[i, :, top[i]:top[i]+H, left[i]:left[i]+W]
            for i in range(B)
        ])
        return crops

    def random_rotate(self, batch, max_deg=15):
        B = batch.shape[0]
        angle = torch.empty(B, device=batch.device).uniform_(-max_deg, max_deg) * 3.1415 / 180
        cos = torch.cos(angle)
        sin = torch.sin(angle)

        theta = torch.zeros(B, 2, 3, device=batch.device)
        theta[:, 0, 0] = cos
        theta[:, 0, 1] = -sin
        theta[:, 1, 0] = sin
        theta[:, 1, 1] = cos

        grid = F.affine_grid(theta, batch.size(), align_corners=False)
        return F.grid_sample(batch, grid, align_corners=False, mode='bilinear', padding_mode='border')

    def random_affine(self, batch, max_deg=10, translate=0.1):
        B, _, H, W = batch.shape
        angle = torch.empty(B, device=batch.device).uniform_(-max_deg, max_deg) * 3.1415 / 180
        tx = torch.empty(B, device=batch.device).uniform_(-translate, translate)
        ty = torch.empty(B, device=batch.device).uniform_(-translate, translate)

        cos = torch.cos(angle)
        sin = torch.sin(angle)

        theta = torch.zeros(B, 2, 3, device=batch.device)
        theta[:, 0, 0] = cos
        theta[:, 0, 1] = -sin
        theta[:, 1, 0] = sin
        theta[:, 1, 1] = cos
        theta[:, 0, 2] = tx
        theta[:, 1, 2] = ty

        grid = F.affine_grid(theta, batch.size(), align_corners=False)
        return F.grid_sample(batch, grid, align_corners=False, mode='bilinear', padding_mode='border')

# augment = HSIRandAugment(num_ops=2)
# # Dataset shape: (6104, 8, 7, 7)
# X_tensor = torch.from_numpy(X).float()  # Convert to float tensor
# augmented_patch = augment(X_tensor[0])  # Apply strong augmentations

import torch
import torch.nn.functional as F
import random

class WeakAug:
    def __init__(self, crop_size=7, padding=2, flip_prob=0.5):
        self.crop_size = crop_size
        self.padding = padding
        self.flip_prob = flip_prob

    def __call__(self, batch):
        # batch: (B, C, H, W)
        B, C, H, W = batch.shape
        device = batch.device

        # Padding
        batch = F.pad(batch, (self.padding, self.padding, self.padding, self.padding), mode='reflect')
        _, _, Hp, Wp = batch.shape

        # Random crop
        top = torch.randint(0, Hp - self.crop_size + 1, (B,), device=device)
        left = torch.randint(0, Wp - self.crop_size + 1, (B,), device=device)

        crops = torch.stack([
            batch[i, :, top[i]:top[i]+self.crop_size, left[i]:left[i]+self.crop_size]
            for i in range(B)
        ])

        # Random flip
        flip_h = torch.rand(B, device=device) < self.flip_prob
        flip_v = torch.rand(B, device=device) < self.flip_prob

        if flip_h.any():
            crops[flip_h] = torch.flip(crops[flip_h], dims=[3])
        if flip_v.any():
            crops[flip_v] = torch.flip(crops[flip_v], dims=[2])

        # Translate
        do_translate = torch.rand(B, device=device) < 0.5
        if do_translate.any():
            crops[do_translate] = self.translate(crops[do_translate])

        return crops

    def translate(self, batch, shift_range=1):
        B, C, H, W = batch.shape
        device = batch.device
        tx = torch.randint(-shift_range, shift_range + 1, (B,), device=device).float() / W
        ty = torch.randint(-shift_range, shift_range + 1, (B,), device=device).float() / H

        theta = torch.eye(2, 3, device=device).unsqueeze(0).repeat(B, 1, 1)
        theta[:, 0, 2] = tx
        theta[:, 1, 2] = ty

        grid = F.affine_grid(theta, batch.size(), align_corners=False)
        return F.grid_sample(batch, grid, align_corners=False, padding_mode='border')


# weak_aug = WeakAug()
# aug_patch = weak_aug(hsi_patch_tensor)
