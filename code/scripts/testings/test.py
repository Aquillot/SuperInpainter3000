import os
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms import ToTensor, Compose, Normalize, Resize, CenterCrop
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torch
from tqdm import tqdm
from PIL import Image, ImageDraw


from model import UNet


base_path = "../../"
load_batch_size = 64
total_epoch = 250
mask_ratio = 0.5


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
        n_lines = torch.randint(1, 3, (1,)).item()
        for _ in range(n_lines):
            x1 = torch.randint(0, W, (1,)).item()
            y1 = torch.randint(0, H, (1,)).item()
            x2 = torch.randint(0, W, (1,)).item()
            y2 = torch.randint(0, H, (1,)).item()

            thickness = torch.randint(3, 8, (1,)).item()
            draw_line(mask[b, 0], x1, y1, x2, y2, thickness)

    return mask

def to_img(x):
    x = x[0].detach().cpu()                      # remove batch
    x = (x * 0.5 + 0.5).clamp(0,1)               # [-1,1] -> [0,1]
    return x.permute(1, 2, 0)


def test_masking():
    transform = Compose([Resize(256), ToTensor()])
    train_dataset = torchvision.datasets.CIFAR10(
        root=base_path+"data",
        train=True,
        download=True,
        transform=transform
    )

    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    img_tensor, _ = next(iter(dataloader))   # shape : (1,3,256,256)
    img_tensor = img_tensor.to(device)
    mask = create_mask(1, 256, 256, device)
    
    masked = img_tensor * mask
    

    mask_vis = mask.repeat(1, 3, 1, 1)

    fig, ax = plt.subplots(1, 3, figsize=(12,4))
    ax[0].imshow(to_img(img_tensor))
    ax[0].set_title("Image originale")

    ax[1].imshow(mask_vis[0].detach().cpu().permute(1,2,0))
    ax[1].set_title("Masque")

    ax[2].imshow(to_img(masked))
    ax[2].set_title("Image masquée")

    for a in ax: a.axis("off")
    plt.show()



def train_unet():
    transform = Compose([
        Resize(512),
        # CenterCrop(224),                # Rogner au centre pour avoir 224x224
        ToTensor(),
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    train_dataset = torchvision.datasets.CIFAR10(base_path+"data", train=True, download=True, transform=Compose([ToTensor(), Normalize(0.5, 0.5)]))
    val_dataset = torchvision.datasets.CIFAR10(base_path+"data", train=False, download=True, transform=Compose([ToTensor(), Normalize(0.5, 0.5)]))
    dataloader = torch.utils.data.DataLoader(train_dataset, load_batch_size, shuffle=True, num_workers=8)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'




    model = UNet(3).to(device)
    # optim = torch.optim.AdamW(model.parameters(), lr=args.base_learning_rate * args.batch_size / 256, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    # lr_func = lambda epoch: min((epoch + 1) / (args.warmup_epoch + 1e-8), 0.5 * (math.cos(epoch / args.total_epoch * math.pi) + 1))
    # lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_func)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = torch.nn.MSELoss()

    # optim.zero_grad()
    for e in range(total_epoch):
        model.train()
        losses = []
        for img, label in tqdm(iter(dataloader)):
            img = img.to(device)

            B, C, H, W = img.shape
            mask = create_mask(B, H, W, device)
            masked_img = img * mask

            net_input = torch.cat([masked_img, mask], dim=1)

            predicted_img = model(net_input)

            loss = loss_fn(predicted_img, img)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    torch.save(model.state_dict(), base_path + "models/testing_unet.pt")


def test_model():
    transform = Compose([
        Resize(512),
        # CenterCrop(224),                # Rogner au centre pour avoir 224x224
        ToTensor(),
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    val_dataset = torchvision.datasets.CIFAR10(base_path + "data", train=False, download=True, transform=Compose([ToTensor(), Normalize(0.5, 0.5)]))
    dataloader = torch.utils.data.DataLoader(val_dataset, load_batch_size, shuffle=True, num_workers=8)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'




    model = UNet(3).to(device)
    model.load_state_dict(torch.load(base_path + "models/testing_unet.pt", weights_only=True))

    img_tensor, _ = next(iter(dataloader))
    img_tensor = img_tensor.to(device)
    B, C, H, W = img_tensor.shape
    mask = create_mask(B, H, W, device)
    
    masked = img_tensor * mask

    net_input = torch.cat([masked, mask], dim=1)

    reconstructed = model(net_input)

    fig, ax = plt.subplots(1, 3, figsize=(12,4))
    ax[0].imshow(to_img(img_tensor))
    ax[0].set_title("Image originale")

    ax[1].imshow(to_img(masked))
    ax[1].set_title("Image masquée")

    ax[2].imshow(to_img(reconstructed))
    ax[2].set_title("Image générée")


    for a in ax: a.axis("off")
    plt.show()
    



train_unet()
test_model()