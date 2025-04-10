import random
import torch
import torch.nn.functional as F

class HSIRandAugment:
    def __init__(self, num_ops=2, magnitude=10, patch_size=(7, 7)):
        self.num_ops = num_ops
        self.magnitude = magnitude
        self.patch_size = patch_size  # Needed for cropping
        self.augment_ops = [
            self.random_flip,
            self.random_crop,
            self.random_rotate,
            self.random_affine,
        ]

    def __call__(self, img_tensor):
        # img_tensor shape: (C, H, W)
        ops = random.sample(self.augment_ops, self.num_ops)
        for op in ops:
            img_tensor = op(img_tensor)
        return img_tensor

    def random_flip(self, img):
        if random.random() > 0.5:
            img = torch.flip(img, dims=[2])  # Horizontal
        if random.random() > 0.5:
            img = torch.flip(img, dims=[1])  # Vertical
        return img

    def random_crop(self, img, padding=2):
        c, h, w = img.shape
        img_padded = F.pad(img.unsqueeze(0), (padding, padding, padding, padding), mode='reflect').squeeze(0)
        top = random.randint(0, padding * 2)
        left = random.randint(0, padding * 2)
        return img_padded[:, top:top+h, left:left+w]

    def random_rotate(self, img, max_deg=15):
        angle = random.uniform(-max_deg, max_deg)
        angle_rad = torch.tensor(angle * 3.1415 / 180, device=img.device)

        cos_a = torch.cos(angle_rad)
        sin_a = torch.sin(angle_rad)

        theta = torch.tensor([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0]
        ], dtype=torch.float32, device=img.device).unsqueeze(0)

        grid = F.affine_grid(theta, img.unsqueeze(0).size(), align_corners=False)
        return F.grid_sample(img.unsqueeze(0), grid, align_corners=False, mode='bilinear',
                             padding_mode='border').squeeze(0)

    def random_affine(self, img, max_deg=10, translate=0.1):
        angle = random.uniform(-max_deg, max_deg)
        angle_rad = torch.tensor(angle * 3.1415 / 180, device=img.device)

        translations = [
            random.uniform(-translate, translate),
            random.uniform(-translate, translate)
        ]

        cos_a = torch.cos(angle_rad)
        sin_a = torch.sin(angle_rad)

        theta = torch.tensor([
            [cos_a, -sin_a, translations[0]],
            [sin_a, cos_a, translations[1]]
        ], dtype=torch.float32, device=img.device).unsqueeze(0)

        grid = F.affine_grid(theta, img.unsqueeze(0).size(), align_corners=False)
        return F.grid_sample(img.unsqueeze(0), grid, align_corners=False, mode='bilinear',
                             padding_mode='border').squeeze(0)


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

    def __call__(self, img):
        # img: tensor (C, H, W)

        # Pad reflectively
        img = F.pad(img.unsqueeze(0), (self.padding, self.padding, self.padding, self.padding), mode='reflect').squeeze(0)

        # Random crop
        _, h, w = img.shape
        top = random.randint(0, h - self.crop_size)
        left = random.randint(0, w - self.crop_size)
        img = img[:, top:top + self.crop_size, left:left + self.crop_size]

        # Random horizontal/vertical flip
        if random.random() < self.flip_prob:
            img = torch.flip(img, dims=[2])  # horizontal
        if random.random() < self.flip_prob:
            img = torch.flip(img, dims=[1])  # vertical

        # Small spatial translation (simulate minor jitter)
        if random.random() < 0.5:
            img = self.translate(img)

        return img

    def translate(self, img, shift_range=1):
        _, h, w = img.shape
        tx = random.randint(-shift_range, shift_range)
        ty = random.randint(-shift_range, shift_range)
        grid = F.affine_grid(
            torch.tensor([[[1, 0, tx / w], [0, 1, ty / h]]], dtype=torch.float32, device=img.device),
            size=(1, *img.shape),
            align_corners=False
        )
        return F.grid_sample(img.unsqueeze(0), grid, align_corners=False, padding_mode='border').squeeze(0)

# weak_aug = WeakAug()
# aug_patch = weak_aug(hsi_patch_tensor)
