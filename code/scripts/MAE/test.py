from Model import MAE_Encoder, MAE_Decoder, PatchShuffle
import os
import math
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor, Compose, Normalize, Resize, CenterCrop
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from utils import base_path, mask_image



transform = Compose([
    # Resize(256),                    # Redimensionner d'abord
    # CenterCrop(224),                # Rogner au centre pour avoir 224x224
    ToTensor(),
    Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

load_batch_size = 10

train_dataset = torchvision.datasets.CIFAR10(base_path + 'data', train=True, download=True, transform=transform)
val_dataset = torchvision.datasets.CIFAR10(base_path + 'data', train=False, download=True, transform=transform)
dataloader = torch.utils.data.DataLoader(train_dataset, load_batch_size, shuffle=True, num_workers=4)
writer = SummaryWriter(os.path.join('logs', 'cifar10', 'mae-pretrain'))
device = 'cuda' if torch.cuda.is_available() else 'cpu'





def denormalize(tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    """Dénormalise un tensor d'image"""
    tensor = tensor.clone()
    for i, (m, s) in enumerate(zip(mean, std)):
        tensor[i] = tensor[i] * s + m
    return tensor


def plot_mask_from_indexes(image_shape, forward_indexes, patch_size=2, num_patches_h=16, num_patches_w=16):
    """
    Crée une visualisation du masque à partir des forward_indexes
    """
    total_patches = num_patches_h * num_patches_w
    
    mask = torch.zeros(total_patches, dtype=torch.float32)
    mask[forward_indexes] = 1
    
    mask_2d = mask.reshape(num_patches_h, num_patches_w)
    
    mask_img = mask_2d.unsqueeze(-1).repeat(1, 1, 3)  # (H_patches, W_patches, 3)
    mask_img = mask_img.repeat_interleave(patch_size, dim=0).repeat_interleave(patch_size, dim=1)
    
    return mask_img.cpu().numpy()



def test_single_image(model_path):
    # Prendre une seule image
    data_iter = iter(dataloader)
    single_image, label = next(data_iter)
    single_image = single_image[0:1].to(device)  # Garder seulement la première image
    
    print(f"Testing single image - Shape: {single_image.shape}")
    
    model = torch.load(model_path, weights_only=False)

    visible_patches, forward_indexes, backward_indexes = mask_image(single_image, 2)


    predicted_img, mask = model.forward_from_patches(visible_patches, forward_indexes)
    
    # Afficher la comparaison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor='lightgray')
    
    # Original
    orig_img = denormalize(single_image[0]).permute(1, 2, 0).cpu().numpy()
    orig_img = np.clip(orig_img, 0, 1)
    axes[0].imshow(orig_img)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Masque
    mask_img = plot_mask_from_indexes(
        image_shape=single_image.shape, 
        forward_indexes=forward_indexes.squeeze(),  # Enlever la dimension batch
        patch_size=2,
        num_patches_h=16,  # 32/2 = 16
        num_patches_w=16   # 32/2 = 16
    )
    axes[1].imshow(mask_img, cmap='gray')
    axes[1].set_title('Mask (white=kept, black=masked)')
    axes[1].axis('off')
    
    # Reconstruite
    recon_img = denormalize(predicted_img[0]).permute(1, 2, 0).detach().cpu().numpy()
    recon_img = np.clip(recon_img, 0, 1)
    axes[2].imshow(recon_img)
    axes[2].set_title('Reconstructed Image')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    loss = torch.mean((predicted_img - single_image) ** 2 * mask / 0.75)
    print(f"Single image loss: {loss.item():.4f}")


test_single_image(base_path + 'vit-t-mae.pt')