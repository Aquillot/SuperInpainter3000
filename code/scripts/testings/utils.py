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

def create_mask(batch_size, H, W, device, mask_ratio):
    """Create random line masks.
    Returns: tensor (B, 1, H, W) where 1=hole, 0=valid
    """
    mask = torch.ones((batch_size, 1, H, W), device=device)

    target_area = int(mask_ratio * H * W)

    for b in range(batch_size):
        covered_area = 0

        while covered_area < target_area:
            # Génération aléatoire d’un trait
            x1 = torch.randint(0, W, (1,)).item()
            y1 = torch.randint(0, H, (1,)).item()
            x2 = torch.randint(0, W, (1,)).item()
            y2 = torch.randint(0, H, (1,)).item()

            thickness = torch.randint(3, 10, (1,)).item()

            # Avant de dessiner, sauvegarder pour connaître ce que le trait rajoute
            old_mask = mask[b, 0].clone()

            # Dessiner le trait
            draw_line(mask[b, 0], x1, y1, x2, y2, thickness)

            # Calcul du gain de surface
            new_mask = mask[b, 0]
            added = (old_mask - new_mask).clamp(min=0).sum().item()

            covered_area += added

    # After draw_line: 0=hole, 1=valid
    # We want: 1=hole, 0=valid
    mask = 1.0 - mask
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