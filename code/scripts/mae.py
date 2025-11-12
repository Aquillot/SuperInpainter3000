import torch
import torch.nn as nn
import math

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# https://medium.com/@ovularslan/masked-autoencoders-mae-the-art-of-seeing-more-by-masking-most-pytorch-implementation-4566e08c66a6


def generate_sinusoidal_positional_embeddings(num_patches, emb_dim, is_cls_token):
    """
    Generate fixed sinusoidal positional embeddings

    Args:
        num_patches (int): Number of patches (excluding cls_token)
        emb_dim (int): Embedding dimension (must be even)
        is_cls_token (bool): Whether to generate position embedding for cls_token
    """
    assert emb_dim % 2 == 0, "Embedding dimension must be even"

    half_dim = emb_dim // 2
    embeddings = torch.zeros(size=(num_patches, emb_dim))

    position = torch.arange(num_patches).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, half_dim, 1) * (-math.log(10000.0) / half_dim))
    angles = position * div_term

    # Compute all sine values first
    embeddings[:, :half_dim] = torch.sin(angles)

    # Then compute all cosine values
    embeddings[:, half_dim:] = torch.cos(angles)

    embeddings = embeddings.unsqueeze(0)
    if is_cls_token:
        embeddings = torch.cat([torch.zeros(1, 1, emb_dim), embeddings], dim=1)

    return embeddings


class PatchEmbedding(nn.Module):
    """
        Converts an input image into patch embeddings

        Args:
            img_size (int): Height and width of the input image (assumed square).
            in_channel (int): Number of input channels (e.g., 3 for RGB).
            emb_dim (int): Embedding dimension for each patch.
            patch_size (int): Size of each patch.
        """
    def __init__(self, img_size, in_channel, emb_dim, patch_size):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels=in_channel, out_channels=emb_dim, kernel_size=patch_size, stride=patch_size)
        self.flatten = nn.Flatten(start_dim=-2, end_dim=-1)
        self.num_of_patches = (img_size // patch_size) ** 2

    def forward(self, x):
        x = self.proj(x)
        x = self.flatten(x)
        x = x.permute(0, 2, 1)
        return x


class Attention(nn.Module):
    """
        Multi-head self-attention module.

        Args:
            in_dim (int): Input embedding dimension.
            num_heads (int): Number of attention heads.
            attn_drop_p (float): Dropout probability for attention scores.
            attn_proj_drop_p (float): Dropout probability for the output projection.
        """
    def __init__(self, in_dim, num_heads, attn_drop_p, attn_proj_drop_p):
        super().__init__()
        head_dim = in_dim // num_heads
        self.scale = head_dim ** (-0.5)
        self.num_heads = num_heads
        self.qkv_proj = nn.Linear(in_features=in_dim, out_features=3*in_dim)
        self.attn_proj = nn.Linear(in_features=in_dim, out_features=in_dim)
        self.attn_drop = nn.Dropout(p=attn_drop_p)
        self.attn_proj_drop = nn.Dropout(p=attn_proj_drop_p)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, C//self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1) * self.scale).softmax(dim=-1)
        attn = self.attn_drop(attn)
        attn = (attn @ v).permute(0, 2, 1, 3)
        attn = attn.reshape(B, N, -1)
        attn = self.attn_proj(attn)
        attn = self.attn_proj_drop(attn)
        return attn

class SingleTransformerLayer(nn.Module):
    """
        A single Transformer layer.

        Args:
            emb_dim (int): Input embedding dimension.
            num_heads (int): Number of attention heads.
            mlp_expansion (int): Expansion factor for MLP layers.
            mlp_drop_p (float): Dropout probability for MLP layers.
            attn_drop_p (float): Dropout probability for attention scores.
            attn_proj_drop_p (float): Dropout probability for the output projection.
            drop_path_p (float): Stochastic depth probability.
        """
    def __init__(self, emb_dim, num_heads, mlp_expansion, mlp_drop_p, attn_drop_p, attn_proj_drop_p, drop_path_p):
        super().__init__()
        self.LN1 = nn.LayerNorm(emb_dim)
        self.attn = Attention(emb_dim, num_heads, attn_drop_p, attn_proj_drop_p)
        self.LN2 = nn.LayerNorm(emb_dim)
        self.MLP = nn.Sequential(
            nn.Linear(in_features=emb_dim, out_features=mlp_expansion*emb_dim),
            nn.GELU(),
            nn.Dropout(p=mlp_drop_p),
            nn.Linear(in_features=mlp_expansion*emb_dim, out_features=emb_dim),
            nn.Dropout(p=mlp_drop_p)
        )
        self.drop_path = nn.Dropout(drop_path_p) if drop_path_p > 0.0 else nn.Identity()

    def forward(self, x):
        res = x
        x = self.LN1(x)
        x = self.attn(x)
        x = res + self.drop_path(x)

        res = x
        x = self.LN2(x)
        x = self.MLP(x)
        x = res + self.drop_path(x)

        return x

class MaskedAutoencoder(nn.Module):
    """
        Implementation of the Masked Autoencoder (MAE) architecture for self-supervised learning.

        Args:
            img_size (int): Height and width of the input image (assumed square).
            in_channel (int): Number of input channels (e.g., 3 for RGB).
            encoder_emb_dim (int): Embedding dimension of the encoder.
            encoder_depth (int): Number of transformer layers in the encoder.
            decoder_emb_dim (int): Embedding dimension of the decoder.
            decoder_depth (int): Number of transformer layers in the decoder.
            encoder_num_heads (int): Number of attention heads in the encoder.
            decoder_num_heads (int): Number of attention heads in the decoder.
            patch_size (int): Size of each patch.
            mask_ratio (float): Ratio of patches to be masked during training.
            mlp_expansion (int): Expansion factor for MLP layers.
            mlp_drop_p (float): Dropout probability for MLP layers.
            attn_drop_p (float): Dropout probability for attention scores.
            attn_proj_drop_p (float): Dropout probability for the output projection.
            drop_path_p (float): Stochastic depth probability.
        """
    def __init__(self, img_size=224, in_channel=3, encoder_emb_dim=768, encoder_depth=12, decoder_emb_dim=512, decoder_depth=8, encoder_num_heads=12, decoder_num_heads=16, patch_size=16, mask_ratio=0.75, mlp_expansion=4, mlp_drop_p=0.0, attn_drop_p=0.0, attn_proj_drop_p=0.0, drop_path_p=0.0):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.patcher = PatchEmbedding(img_size, in_channel, encoder_emb_dim, patch_size)

        # Fixed positional embeddings as stated in the paper
        self.pos_embeddings_encoder = generate_sinusoidal_positional_embeddings(self.patcher.num_of_patches, encoder_emb_dim, is_cls_token=True)
        self.pos_embeddings_decoder = generate_sinusoidal_positional_embeddings(self.patcher.num_of_patches, decoder_emb_dim, is_cls_token=True)

        self.cls_token = nn.Parameter(data=torch.zeros(size=(1, 1, encoder_emb_dim)), requires_grad=True)
        self.mask_embedding = nn.Parameter(data=torch.zeros(size=(1, 1, decoder_emb_dim)), requires_grad=True)

        self.encoder_blocks = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        self.LN_encoder = nn.LayerNorm(encoder_emb_dim)
        self.LN_decoder = nn.LayerNorm(decoder_emb_dim)
        self.proj = nn.Linear(in_features=encoder_emb_dim, out_features=decoder_emb_dim)
        self.decoder_proj = nn.Linear(in_features=decoder_emb_dim, out_features=(patch_size ** 2) * in_channel)
        drop_path_rates_encoder = torch.linspace(0.0, drop_path_p, encoder_depth).tolist()
        drop_path_rates_decoder = torch.linspace(0.0, drop_path_p, decoder_depth).tolist()

        for i in range(encoder_depth):
            self.encoder_blocks.append(SingleTransformerLayer(encoder_emb_dim, encoder_num_heads, mlp_expansion, mlp_drop_p, attn_drop_p, attn_proj_drop_p, drop_path_rates_encoder[i]))
        for j in range(decoder_depth):
            self.decoder_blocks.append(SingleTransformerLayer(decoder_emb_dim, decoder_num_heads, mlp_expansion, mlp_drop_p, attn_drop_p, attn_proj_drop_p, drop_path_rates_decoder[j]))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        # Xavier Uniform is used to initialize all Transformer blocks as stated in the paper section A.1 -> Pre-training
        nn.init.normal(self.cls_token, std=0.02)
        nn.init.normal(self.mask_embedding, std=0.02)

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        patcher_weights = self.patcher.proj.weight.data
        torch.nn.init.xavier_uniform_(patcher_weights.view([patcher_weights.shape[0], -1]))

        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        # This function is taken directly from the official PyTorch implementation of MAE "https://github.com/facebookresearch/mae"

        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))  # number of visible patches to keep

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # index order of patches from smallest to largest noise
        ids_restore = torch.argsort(ids_shuffle, dim=1)  # allows us to unshuffle (reorder back to original patch order)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]  # indices of the visible patches
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))  # Gathers the patch embeddings at ids_keep positions, these are the patch embeddings that will be sent to the encoder

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0

        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)  # Reorders the binary mask back to the original patch sequence, tracks which patches were masked

        return x_masked, mask, ids_restore

    def compute_loss(self, x_org, x_pred, mask):
        B, C, H, W = x_org.shape
        patch_size = self.patcher.patch_size
        patch_dim = H // patch_size

        # Patchify the input image
        x_org = x_org.reshape(B, C, patch_dim, patch_size, patch_dim, patch_size)
        x_org = x_org.permute(0, 2, 4, 3, 5, 1)
        x_org = x_org.reshape(-1, patch_dim**2, (patch_size**2)*C)

        # Calculate MSE only on the masked patches
        loss = torch.mean((x_org - x_pred) ** 2, dim=-1)
        loss = (loss * mask).sum() / mask.sum()
        return loss

    def forward(self, x):
        B, C, H, W = x.shape
        x_org = x
        x = self.patcher(x)
        x += self.pos_embeddings_encoder[:, 1:, :] # Add pos_embeddings w/o cls_token

        x_masked, mask, ids_restore = self.random_masking(x, self.mask_ratio)
        cls_token = self.cls_token + self.pos_embeddings_encoder[:, :1, :] # Add pos_embeddings to cls_token
        cls_token = cls_token.expand(B, -1, -1)
        x_masked = torch.cat([cls_token, x_masked], dim=1) # Concat the cls_token to non-masked patches
        for layer in self.encoder_blocks:
            x_masked = layer(x_masked)
        x_masked = self.LN_encoder(x_masked)
        x_masked = self.proj(x_masked) # Change emb_dim to match decoder_emb_dim

        num_of_masked_patches = int(mask.shape[1] * self.mask_ratio) # Number of masked patches (Assumed the mask ratio is constant for all samples in all batches)
        mask_embedding = self.mask_embedding.repeat(B, num_of_masked_patches, 1)

        x_full = torch.cat([x_masked[:, 1:, :], mask_embedding], dim=1) # Add mask_embeddings to non-masked patches w/o cls_token
        x_full = torch.gather(x_full, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x_full.shape[2])) # Unshuffle to the original patch order
        x_full = torch.cat([x_masked[:, :1, :], x_full], dim=1) # Concat the cls_token
        x_full += self.pos_embeddings_decoder # Add pos_embeddings with cls_token

        for layer in self.decoder_blocks:
            x_full = layer(x_full)
        x_full = self.LN_decoder(x_full)
        x_full = self.decoder_proj(x_full) # Change emb_dim to match (patch_size ** 2) * C

        x_pred = x_full[:, 1:, :] # Remove cls_token
        loss = self.compute_loss(x_org, x_pred, mask) # Loss computation
        return x_pred, loss




def reconstruct_from_patches(x_pred, patch_size=16, img_size=224, channels=3):
    """
    x_pred: torch.Tensor [N_patches, patch_dim=(patch_size**2 * channels)]
    """
    num_patches_per_dim = img_size // patch_size

    # Reformater les patches à leur taille (patch_size, patch_size, C)
    patches = x_pred.reshape(num_patches_per_dim, num_patches_per_dim, patch_size, patch_size, channels)

    # Réarranger les patches en grille
    img = patches.permute(0, 2, 1, 3, 4).reshape(img_size, img_size, channels)

    return img



################### implementation

transform = transforms.Compose([
    transforms.Resize(224),       # redimensionne à la taille attendue par le modèle
    transforms.ToTensor(),
])
train_data = datasets.CIFAR10(root="../data", train=True, download=True, transform=transform)

batch_size = 64
train_dataloader = DataLoader(train_data, batch_size=batch_size)


model = MaskedAutoencoder()
for X, y in train_dataloader:
    print(f"Batch X shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    preds, loss = model.forward(X)

    x_pred_single = preds[0]
    img_reconstructed = reconstruct_from_patches(x_pred_single, patch_size=16, img_size=224, channels=3)

    # C'est juste l'image d'input
    plt.imshow(X[0].permute(1, 2, 0).detach().cpu().numpy(), vmin=0, vmax=1)

    plt.axis("off")
    plt.show()
    break
