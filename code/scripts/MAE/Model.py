import torch
import timm
import numpy as np

from einops import repeat, rearrange
from einops.layers.torch import Rearrange

from timm.layers import trunc_normal_
from timm.models.vision_transformer import Block


def random_indexes(size : int):
    forward_indexes = np.arange(size)
    np.random.shuffle(forward_indexes)
    backward_indexes = np.argsort(forward_indexes)
    return forward_indexes, backward_indexes

def take_indexes(sequences, indexes):
    return torch.gather(sequences, 0, repeat(indexes, 't b -> t b c', c=sequences.shape[-1]))

def block_aware_indexes(T, mask_ratio):
    grid_size = int(T ** 0.5)

    block_size = int(grid_size * 0.6)  # 60% de la grille
    num_masked = int(T * mask_ratio)

    start_h = torch.randint(0, grid_size - block_size + 1, (1,))
    start_w = torch.randint(0, grid_size - block_size + 1, (1,))

    block_indices = []
    for h in range(start_h, start_h + block_size):
        for w in range(start_w, start_w + block_size):
            idx = h * grid_size + w
            block_indices.append(idx)

    if len(block_indices) > num_masked:
        block_indices = block_indices[:num_masked]

    block_indices = torch.tensor(block_indices)

    all_indices = torch.arange(T)

    mask = torch.isin(all_indices, block_indices)
    outside_block = all_indices[~mask]  # Indices hors du bloc
    inside_block = all_indices[mask]    # Indices dans le bloc

    outside_block = outside_block[torch.randperm(len(outside_block))]
    inside_block = inside_block[torch.randperm(len(inside_block))]

    forward_indexes = torch.cat([outside_block, inside_block])

    backward_indexes = torch.argsort(forward_indexes)

    return forward_indexes, backward_indexes




class PatchShuffle(torch.nn.Module):
    def __init__(self, ratio, mask_type: str = "random") -> None:
        super().__init__()
        self.ratio = ratio
        self.mask_type = mask_type

    def forward(self, patches : torch.Tensor):
        T, B, C = patches.shape
        remain_T = int(T * (1 - self.ratio))

        if(self.mask_type == "random"):
            indexes = [random_indexes(T) for _ in range(B)]
            forward_indexes = torch.as_tensor(
                np.stack([i[0] for i in indexes], axis=-1), dtype=torch.long).to(patches.device)
            backward_indexes = torch.as_tensor(np.stack([i[1] for i in indexes], axis=-1), dtype=torch.long).to(patches.device)
        else:
            indexes = [block_aware_indexes(T, self.ratio) for _ in range(B)]
            forward_indexes = torch.as_tensor(
                np.stack([i[0] for i in indexes], axis=-1), dtype=torch.long).to(patches.device)
            backward_indexes = torch.as_tensor(
                np.stack([i[1] for i in indexes], axis=-1), dtype=torch.long).to(patches.device)


        patches = take_indexes(patches, forward_indexes)
        patches = patches[:remain_T]

        return patches, forward_indexes, backward_indexes

class MAE_Encoder(torch.nn.Module):
    def __init__(self,
                 image_size=32,
                 patch_size=2,
                 emb_dim=192,
                 num_layer=12,
                 num_head=3,
                 mask_ratio=0.75,
                 mask_type="random"
                 ) -> None:
        super().__init__()

        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embedding = torch.nn.Parameter(torch.zeros((image_size // patch_size) ** 2, 1, emb_dim))
        self.shuffle = PatchShuffle(mask_ratio, mask_type)

        self.patchify = torch.nn.Conv2d(3, emb_dim, patch_size, patch_size)

        self.transformer = torch.nn.Sequential(*[Block(emb_dim, num_head) for _ in range(num_layer)])

        self.layer_norm = torch.nn.LayerNorm(emb_dim)

        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embedding, std=.02)

    def forward(self, img):
        patches = self.patchify(img)
        patches = rearrange(patches, 'b c h w -> (h w) b c')
        patches = patches + self.pos_embedding

        patches, forward_indexes, backward_indexes = self.shuffle(patches)

        patches = torch.cat([self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0)
        patches = rearrange(patches, 't b c -> b t c')
        features = self.layer_norm(self.transformer(patches))
        features = rearrange(features, 'b t c -> t b c')

        return features, backward_indexes
    
    def forward_from_patches(self, patches, forward_indexes=None):
        # Forward avec patches déjà préparés
        # patches shape: (num_visible, B, C)
        # forward_indexes shape: (num_visible,) - les indices des patches visibles
        B = patches.shape[1]
        C = patches.shape[2]
        total_patches = self.pos_embedding.shape[0]  # e.g., 256 pour image 32x32
        
        # Ajouter les embeddings de position aux patches visibles
        num_visible = patches.shape[0]
        patches = patches + self.pos_embedding[:num_visible]
        
        if forward_indexes is None:
            # Masquage normal aléatoire
            patches, forward_indexes, backward_indexes = self.shuffle(patches)
        else:
            # Utiliser les indexes fournis (image déjà masquée)
            # Créer un tensor de tous les patches avec des mask tokens aux positions manquantes
            full_patches = torch.zeros(total_patches, B, C, device=patches.device)
            
            # forward_indexes contient les indices des patches visibles
            if forward_indexes.dim() == 1:
                # forward_indexes est 1D: (num_visible,)
                # On met les patches visibles aux bonnes positions
                full_patches[forward_indexes] = patches
            else:
                # forward_indexes est 2D: (num_visible, B) - pas le cas ici avec mask_image
                raise ValueError(f"forward_indexes dimension non supportée: {forward_indexes.shape}")
            
            patches = full_patches
            # backward_indexes: mapping de retour pour la reconstruction
            backward_indexes = torch.arange(total_patches, device=patches.device).unsqueeze(-1).expand(-1, B)
        
        # Ajouter le CLS token et passer par le transformer
        patches = torch.cat([self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0)
        patches = rearrange(patches, 't b c -> b t c')
        features = self.layer_norm(self.transformer(patches))
        features = rearrange(features, 'b t c -> t b c')
        
        # Enlever le CLS token
        features = features[1:]

        return features, backward_indexes


class MAE_Decoder(torch.nn.Module):
    def __init__(self,
                 image_size=32,
                 patch_size=2,
                 emb_dim=192,
                 num_layer=4,
                 num_head=3,
                 ) -> None:
        super().__init__()

        self.mask_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embedding = torch.nn.Parameter(torch.zeros((image_size // patch_size) ** 2 + 1, 1, emb_dim))

        self.transformer = torch.nn.Sequential(*[Block(emb_dim, num_head) for _ in range(num_layer)])

        self.head = torch.nn.Linear(emb_dim, 3 * patch_size ** 2)
        self.patch2img = Rearrange('(h w) b (c p1 p2) -> b c (h p1) (w p2)', p1=patch_size, p2=patch_size, h=image_size//patch_size)

        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.mask_token, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embedding, std=.02)

    def forward(self, features, backward_indexes):
        T = features.shape[0]
        backward_indexes = torch.cat([torch.zeros(1, backward_indexes.shape[1]).to(backward_indexes), backward_indexes + 1], dim=0)
        features = torch.cat([features, self.mask_token.expand(backward_indexes.shape[0] - features.shape[0], features.shape[1], -1)], dim=0)
        features = take_indexes(features, backward_indexes)
        features = features + self.pos_embedding

        features = rearrange(features, 't b c -> b t c')
        features = self.transformer(features)
        features = rearrange(features, 'b t c -> t b c')
        features = features[1:] # remove global feature

        patches = self.head(features)
        mask = torch.zeros_like(patches)
        mask[T-1:] = 1
        mask = take_indexes(mask, backward_indexes[1:] - 1)
        img = self.patch2img(patches)
        mask = self.patch2img(mask)

        return img, mask


class MAE_ViT(torch.nn.Module):
    def __init__(self,
                 image_size=32,
                 patch_size=2,
                 emb_dim=192,
                 encoder_layer=12,
                 encoder_head=3,
                 decoder_layer=4,
                 decoder_head=3,
                 mask_ratio=0.75,
                 apply_mask=True
                 ) -> None:
        super().__init__()

        self.encoder = MAE_Encoder(image_size, patch_size, emb_dim, encoder_layer, encoder_head, mask_ratio, mask_type="block")
        self.decoder = MAE_Decoder(image_size, patch_size, emb_dim, decoder_layer, decoder_head)

    def forward(self, img):
        features, backward_indexes = self.encoder(img)
        predicted_img, mask = self.decoder(features,  backward_indexes)
        return predicted_img, mask

    def forward_from_patches(self, patches, forward_indexes):
        features, backward_indexes = self.encoder.forward_from_patches(patches, forward_indexes)
        predicted_img, mask = self.decoder(features, backward_indexes)
        return predicted_img, mask