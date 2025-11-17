"""
Script de test pour vérifier les dimensions des tenseurs dans le pipeline MAE
"""
import torch
import torch.nn as nn
import sys
sys.path.insert(0, '/home/crhonopost/Bureau/projets/master/semestre3/SuperInpainter3000/code/scripts/MAE')

from Model import MAE_Encoder, MAE_Decoder, MAE_ViT
from utils import mask_image, setup_seed

def test_dimensions():
    setup_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Paramètres
    image_size = 32
    patch_size = 2
    emb_dim = 192
    encoder_layer = 12
    encoder_head = 3
    decoder_layer = 4
    decoder_head = 3
    mask_ratio = 0.75
    
    print("=" * 80)
    print("TEST 1: Créer une image de test")
    print("=" * 80)
    # Créer une image de test
    image = torch.randn(1, 3, image_size, image_size).to(device)  # (1, 3, 32, 32)
    print(f"✓ Image shape: {image.shape}")
    
    print("\n" + "=" * 80)
    print("TEST 2: mask_image - masquer l'image")
    print("=" * 80)
    visible_patches, forward_indexes, backward_indexes = mask_image(image, patch_size, mask_ratio, emb_dim)
    print(f"✓ visible_patches shape: {visible_patches.shape}")
    print(f"✓ forward_indexes shape: {forward_indexes.shape}")
    print(f"✓ backward_indexes shape: {backward_indexes.shape}")
    
    # Vérifications
    assert visible_patches.shape[0] < 256, f"visible_patches should be less than 256, got {visible_patches.shape[0]}"
    assert visible_patches.shape[1] == 1, f"Batch size should be 1, got {visible_patches.shape[1]}"
    assert visible_patches.shape[2] == emb_dim, f"Embedding dim should be {emb_dim}, got {visible_patches.shape[2]}"
    assert forward_indexes.shape[0] == visible_patches.shape[0], "forward_indexes size mismatch"
    assert backward_indexes.shape[0] == 256, "backward_indexes should contain all patches"
    print("✓ Toutes les dimensions sont correctes!")
    
    print("\n" + "=" * 80)
    print("TEST 3: MAE_Encoder.forward_from_patches")
    print("=" * 80)
    encoder = MAE_Encoder(
        image_size=image_size,
        patch_size=patch_size,
        emb_dim=emb_dim,
        num_layer=encoder_layer,
        num_head=encoder_head,
        mask_ratio=mask_ratio
    ).to(device)
    
    features, backward_indexes_out = encoder.forward_from_patches(visible_patches, forward_indexes)
    print(f"✓ features shape: {features.shape}")
    print(f"✓ backward_indexes_out shape: {backward_indexes_out.shape}")
    
    # Vérifications
    assert features.shape[0] == 256, f"features should have 256 patches, got {features.shape[0]}"
    assert features.shape[1] == 1, f"Batch size should be 1, got {features.shape[1]}"
    assert features.shape[2] == emb_dim, f"Embedding dim should be {emb_dim}, got {features.shape[2]}"
    assert backward_indexes_out.shape[0] == 256, "backward_indexes should have 256"
    assert backward_indexes_out.shape[1] == 1, "backward_indexes batch dim should be 1"
    print("✓ Toutes les dimensions sont correctes!")
    
    print("\n" + "=" * 80)
    print("TEST 4: MAE_Decoder.forward")
    print("=" * 80)
    decoder = MAE_Decoder(
        image_size=image_size,
        patch_size=patch_size,
        emb_dim=emb_dim,
        num_layer=decoder_layer,
        num_head=decoder_head
    ).to(device)
    
    reconstructed_img, mask = decoder.forward(features, backward_indexes_out)
    print(f"✓ reconstructed_img shape: {reconstructed_img.shape}")
    print(f"✓ mask shape: {mask.shape}")
    
    # Vérifications
    assert reconstructed_img.shape == image.shape, f"Image shape mismatch: {reconstructed_img.shape} vs {image.shape}"
    assert mask.shape == image.shape, f"Mask shape mismatch: {mask.shape} vs {image.shape}"
    print("✓ Toutes les dimensions sont correctes!")
    
    print("\n" + "=" * 80)
    print("TEST 5: MAE_ViT complet")
    print("=" * 80)
    model = MAE_ViT(
        image_size=image_size,
        patch_size=patch_size,
        emb_dim=emb_dim,
        encoder_layer=encoder_layer,
        encoder_head=encoder_head,
        decoder_layer=decoder_layer,
        decoder_head=decoder_head,
        mask_ratio=mask_ratio
    ).to(device)
    
    reconstructed_img_full, mask_full = model.forward_from_patches(visible_patches, forward_indexes)
    print(f"✓ reconstructed_img shape: {reconstructed_img_full.shape}")
    print(f"✓ mask shape: {mask_full.shape}")
    
    # Vérifications
    assert reconstructed_img_full.shape == image.shape, f"Image shape mismatch"
    assert mask_full.shape == image.shape, f"Mask shape mismatch"
    print("✓ Toutes les dimensions sont correctes!")
    
    # Calculer la perte
    mse_loss = torch.mean((reconstructed_img_full - image) ** 2)
    print(f"\n✓ MSE Loss: {mse_loss.item():.6f}")
    
    print("\n" + "=" * 80)
    print("✅ TOUS LES TESTS PASSÉS!")
    print("=" * 80)

if __name__ == "__main__":
    test_dimensions()
