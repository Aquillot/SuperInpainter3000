import torch

def to_img(x):
    x = x[0].detach().cpu()                      # remove batch
    x = (x * 0.5 + 0.5).clamp(0,1)               # [-1,1] -> [0,1]
    return x.permute(1, 2, 0)




def draw_line(mask_channel, x1, y1, x2, y2, thickness):
    """Draw a line on mask_channel by setting pixels to 0."""
    steps = max(abs(x2 - x1), abs(y2 - y1)) + 1
    xs = torch.linspace(x1, x2, steps).round().long()
    ys = torch.linspace(y1, y2, steps).round().long()

    for x, y in zip(xs, ys):
        x0 = max(0, x - thickness)
        x1_clip = min(mask_channel.shape[1], x + thickness)
        y0 = max(0, y - thickness)
        y1_clip = min(mask_channel.shape[0], y + thickness)
        mask_channel[y0:y1_clip, x0:x1_clip] = 0

def create_mask(batch_size, H, W, device):
    """Create random line masks.
    Returns: tensor (B, 1, H, W) where 1=hole, 0=valid
    """
    mask = torch.ones((batch_size, 1, H, W), device=device)

    for b in range(batch_size):
        n_lines = torch.randint(1, 4, (1,)).item()  # 1-3 lines
        for _ in range(n_lines):
            x1 = torch.randint(0, W, (1,)).item()
            y1 = torch.randint(0, H, (1,)).item()
            x2 = torch.randint(0, W, (1,)).item()
            y2 = torch.randint(0, H, (1,)).item()
            thickness = torch.randint(3, 10, (1,)).item()
            draw_line(mask[b, 0], x1, y1, x2, y2, thickness)

    # After draw_line: 0=hole, 1=valid
    # We want: 1=hole, 0=valid
    mask = 1.0 - mask
    return mask


def create_mask_fast(batch_size, H, W, device, num_lines_range=None, thickness_range=None):
    """
    Version optimisée qui traite ligne par ligne mais garde la vectorisation sur le batch.
    Utilise moins de mémoire que la version complète.
    """

    if thickness_range is None:
        thickness_range = (int((H + W) * 0.08), int((H + W) * 0.2))
    if((H+W) * 0.5 <= 64 and num_lines_range is None):
        num_lines_range = (1, 2)
    elif(num_lines_range is not None):
        num_lines_range=(1, 4)

    mask = torch.zeros((batch_size, 1, H, W), device=device)
    
    # Créer la grille une seule fois
    yy, xx = torch.meshgrid(torch.arange(H, device=device, dtype=torch.float32),
                           torch.arange(W, device=device, dtype=torch.float32),
                           indexing='ij')
    yy = yy.unsqueeze(0)  # (1, H, W)
    xx = xx.unsqueeze(0)  # (1, H, W)
    
    for b in range(batch_size):
        n_lines = torch.randint(num_lines_range[0], num_lines_range[1], (1,), device=device).item()
        
        for _ in range(n_lines):
            # Paramètres de la ligne
            x1 = torch.randint(0, W, (1,), device=device).float()
            y1 = torch.randint(0, H, (1,), device=device).float()
            x2 = torch.randint(0, W, (1,), device=device).float()
            y2 = torch.randint(0, H, (1,), device=device).float()
            thickness = torch.randint(thickness_range[0], thickness_range[1], (1,), device=device).float()
            
            # Calculer distance pour cette ligne
            dx = x2 - x1
            dy = y2 - y1
            line_length_sq = torch.clamp(dx * dx + dy * dy, min=1e-6)
            
            t = ((xx - x1) * dx + (yy - y1) * dy) / line_length_sq
            t = torch.clamp(t, 0, 1)
            
            proj_x = x1 + t * dx
            proj_y = y1 + t * dy
            
            dist = torch.sqrt((xx - proj_x) ** 2 + (yy - proj_y) ** 2)
            line_mask = dist < thickness
            
            mask[b, 0] = torch.where(line_mask[0], torch.ones_like(mask[b, 0]), mask[b, 0])
    
    return mask

def tensor_to_image(tensor):
    """Convert tensor to displayable image [0, 1] range."""
    img = tensor.detach().cpu()
    # If batch, take first image
    if img.dim() == 4:
        img = img[0]
    # Normalize to [0, 1]
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    # Convert CHW to HWC
    if img.shape[0] in [1, 3]:
        img = img.permute(1, 2, 0)
    if img.shape[-1] == 1:
        img = img.squeeze(-1)
    return img.numpy()


def normalize_images(image_range, images):
    """Ensure images are in the expected range."""
    if image_range == 'tanh':
        # Normalize to [-1, 1]
        img_min, img_max = images.min(), images.max()
        if img_min >= 0.0 and img_max <= 1.0:
            # Images are in [0, 1], convert to [-1, 1]
            images = images * 2.0 - 1.0
        return images, -1.0  # fill_value for holes
    else:  # sigmoid
        # Ensure images are in [0, 1]
        img_min, img_max = images.min(), images.max()
        if img_min < 0.0:
            # Images are in [-1, 1], convert to [0, 1]
            images = (images + 1.0) / 2.0
        return images, 1.0  # fill_value for holes