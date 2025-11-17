import random
import torch
import numpy as np

import argparse

base_path = '../'


def setup_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def quick_args_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=500)
    parser.add_argument('--max_device_batch_size', type=int, default=512)
    parser.add_argument('--base_learning_rate', type=float, default=1.5e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--mask_ratio', type=float, default=0.75)
    parser.add_argument('--total_epoch', type=int, default=5)
    parser.add_argument('--warmup_epoch', type=int, default=200)
    parser.add_argument('--model_path', type=str, default=base_path + 'vit-t-mae.pt')

    args, unknown = parser.parse_known_args()
    return args, unknown

def full_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--max_device_batch_size', type=int, default=512)
    parser.add_argument('--base_learning_rate', type=float, default=1.5e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--mask_ratio', type=float, default=0.75)
    parser.add_argument('--total_epoch', type=int, default=4000)
    parser.add_argument('--warmup_epoch', type=int, default=200)
    parser.add_argument('--model_path', type=str, default=base_path + 'vit-t-mae.pt')

    args, unknown = parser.parse_known_args()
    return args, unknown

def mask_image(image, patch_size:int, mask_ratio: float = 0.75, emb_dim=192):
    B, C, H, W = image.shape
    num_patches_h = H // patch_size
    num_patches_w = W // patch_size
    total_patches = num_patches_h * num_patches_w
    
    mask_area = int(total_patches * mask_ratio)
    mask_side = int(np.sqrt(mask_area))
    
    start_h = (num_patches_h - mask_side) // 2
    start_w = (num_patches_w - mask_side) // 2

    mask = torch.ones(total_patches, dtype=torch.bool, device=image.device)
    
    # Créer une liste des indices à masquer
    masked_indices = []
    for i in range(mask_side):
        for j in range(mask_side):
            patch_idx = (start_h + i) * num_patches_w + (start_w + j)
            if patch_idx < total_patches:
                masked_indices.append(patch_idx)
    
    # Utiliser indexing pour éviter les opérations in-place
    if masked_indices:
        mask[masked_indices] = False
    
    # Convertir image en patches
    patchify = torch.nn.Conv2d(3, emb_dim, patch_size, patch_size).to(image.device)
    # Copier les weights si nécessaire, ou laisser random (juste pour le format)
    patches = patchify(image)
    patches = patches.permute(2, 3, 1, 0)  # (H_patches, W_patches, emb_dim, B)
    patches = patches.contiguous().view(-1, B, emb_dim)
    
    # Séparer patches visibles et masqués
    visible_patches = patches[mask]  # (T_visible, B, C*patch_h*patch_w)
    masked_patches = patches[~mask]
    
    # Créer les indexes
    all_indexes = torch.arange(total_patches, device=image.device)
    forward_indexes = all_indexes[mask]  # Indexes des patches visibles
    backward_indexes = all_indexes  # Tous les indexes dans l'ordre original
    
    return visible_patches, forward_indexes, backward_indexes
