import torch

def draw_line(mask_channel, x1, y1, x2, y2, thickness):
    # Ligne paramétrique
    steps = max(abs(x2 - x1), abs(y2 - y1)) + 1
    xs = torch.linspace(x1, x2, steps).round().long()
    ys = torch.linspace(y1, y2, steps).round().long()

    for x, y in zip(xs, ys):
        x0 = max(0, x - thickness)
        x1 = min(mask_channel.shape[1], x + thickness)
        y0 = max(0, y - thickness)
        y1 = min(mask_channel.shape[0], y + thickness)

        mask_channel[y0:y1, x0:x1] = 0

def create_mask(batch_size, H, W, device):
    mask = torch.ones((batch_size, 1, H, W), device=device)

    for b in range(batch_size):
        n_lines = torch.randint(3, 8, (1,)).item()
        for _ in range(n_lines):
            x1 = torch.randint(0, W, (1,)).item()
            y1 = torch.randint(0, H, (1,)).item()
            x2 = torch.randint(0, W, (1,)).item()
            y2 = torch.randint(0, H, (1,)).item()

            thickness = torch.randint(6, 10, (1,)).item()
            draw_line(mask[b, 0], x1, y1, x2, y2, thickness)

    return mask

def to_img(x):
    x = x[0].detach().cpu()                      # remove batch
    x = (x * 0.5 + 0.5).clamp(0,1)               # [-1,1] -> [0,1]
    return x.permute(1, 2, 0)