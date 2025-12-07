import kagglehub
from torch.utils.data import Dataset
import os
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor, Compose, Normalize, Resize, InterpolationMode, RandomCrop, \
    RandomHorizontalFlip
import torchvision
import torch
from tqdm import tqdm
from PIL import Image
import math

#from ..pennet import InpaintGeneratorPennet
from model import PenNET, InpaintGenerator
from utils import create_mask, to_img


from model import UNet
from Trainer import Trainer

base_path = "../../"

epoch_size = 0.2

config = {
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'lr': 1e-4,
    'beta1': 0.5, # valeur classique pour Adam dans les GANs
    'beta2': 0.999, # valeur classique pour Adam dans les GANs
    'batch_size': 12,
    "total_epochs": 10,
    'd2glr': 0.05,                  # lr ratio D/G
    'num_workers': 16,             # nombre de "threads" pour le DataLoader
    'save_dir': base_path + "models",
    'current_model_name': 'epoch_1',

    'image_range': 'tanh',
    'adversarial_weight': 0.01,
    'hole_weight': 6.0, # poids de la loss dans la zone masquée plus important que dans la zone valide car on veut bien remplir les trous
    'valid_weight': 5.0,
    'pyramid_weight': 0.05,

    "mask_ratio" : 0.5,
    "train": True,
    "dataset_images_size": 512,
    "state_dict_G_path": None,
    "state_dict_D_path": None,
}




class HumanFacesDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.images = [
            os.path.join(root_dir, f)
            for f in os.listdir(root_dir)
            if f.lower().endswith(('.jpg', '.png', '.jpeg'))
        ]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, 0  # label fictif si ton trainer en attend un



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



def train_gan():
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    print("allocated:", torch.cuda.memory_allocated()/1024**2, "MiB")
    print("reserved: ", torch.cuda.memory_reserved()/1024**2, "MiB")

    # ---------- dataset / dataloader ----------
    train_transform = Compose([
        # Option A (classique pour inpainting): resize un peu plus grand puis random crop
        Resize(286, interpolation=InterpolationMode.BILINEAR),  # agrandir un peu
        RandomCrop(256),  # recadrage aléatoire -> variabilité
        RandomHorizontalFlip(p=0.5),
        ToTensor(),
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # -> range [-1,1] (tanh)
    ])

    # Changer le transform car les images sont en 178 x 218 et comme c'est des visages on va juste les recadrer en 160 x 160 au centre
    train_transform = Compose([
        RandomCrop(128),  # recadrage aléatoire -> variabilité
        ToTensor(),
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # -> range [-1,1] (tanh)
    ])


    val_transform = Compose([
        Resize(256, interpolation=InterpolationMode.BILINEAR),
        ToTensor(),
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # path = kagglehub.dataset_download("ashwingupta3012/human-faces")
    # train_dataset = HumanFacesDataset(os.path.join(path, "Humans"), transform=transform)
    # train_dataset = torchvision.datasets.CIFAR10(base_path + "data", train=True, download=True, transform=transform)
    # train_dataset = torchvision.datasets.Places365(base_path + "data", split='train-standard', small=True, download=True, transform=train_transform)
    train_dataset = ImageFolder(root=base_path + "data/DatasetCustom", transform=train_transform)
    # ensure save dir
    os.makedirs(config['save_dir'], exist_ok=True)

    # ---------- instantiate trainer and train ----------
    trainer = Trainer(config, train_dataset, UNet(3))
    # Load state dicts if provided
    new_state_dict_G = None
    new_state_dict_D = None
    print("Starting training with TrainerSimple...")
    if config["state_dict_G_path"] is not None:
        state_dict_G = torch.load(base_path+config["state_dict_G_path"], map_location=config['device'])
        # Clenaing keys if necessary
        new_state_dict_G = {}
        for k, v in state_dict_G.items():
            nk = k
            for prefix in ['module.', 'unet.', 'generator.', 'model.', 'net.']:
                if nk.startswith(prefix):
                    nk = nk[len(prefix):]
                    break
            new_state_dict_G[nk] = v
    if config["state_dict_D_path"] is not None:
        state_dict_D = torch.load(base_path+config["state_dict_D_path"], map_location=config['device'])
        # Clenaing keys if necessary
        new_state_dict_D = {}
        for k, v in state_dict_D.items():
            nk = k
            for prefix in ['module.', 'discriminator.', 'model.']:
                if nk.startswith(prefix):
                    nk = nk[len(prefix):]
                    break
            new_state_dict_D[nk] = v

    trainer.load_models(new_state_dict_G, new_state_dict_D)

    trainer.train()

    # ---------- save final models ----------
    trainer.save_models(base_path + f"models/gen_{config['current_model_name']}.pth",
                        base_path + f"models/disc_{config['current_model_name']}.pth")
    print("Saved final models in", config['save_dir'])

def test_model_gen():
    base_path = "../../"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    # Charger le checkpoint
    checkpoint = torch.load(base_path + f"models/gen_"+config["current_model_name"]+".pth", map_location=device)

    # Extraire le state_dict de la clé 'netG'
    state_dict = checkpoint['netG'] if 'netG' in checkpoint else checkpoint

    # Nettoyer les préfixes
    cleaned_state = {}
    for k, v in state_dict.items():
        nk = k
        for prefix in ['module.', 'unet.', 'generator.', 'model.', 'net.']:
            if nk.startswith(prefix):
                nk = nk[len(prefix):]
                break
        cleaned_state[nk] = v

    # Créer et charger le modèle
    model = PenNET(init_weights=False).to(device)
    model.load_state_dict(cleaned_state, strict=False)

    transform = Compose([
        Resize(config['dataset_images_size'], interpolation=InterpolationMode.NEAREST),
        ToTensor(),
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # image_size = 64
    # DATA_DIR = base_path + 'data/test-images'
    # val_dataset = torchvision.datasets.ImageFolder(DATA_DIR, transform=transform)
    #val_dataset = torchvision.datasets.Places365(base_path + "data", split='val', download=True, small=True, transform=transform)
    val_transform = Compose([
        RandomCrop(128),  # recadrage aléatoire -> variabilité
        ToTensor(),
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # -> range [-1,1] (tanh)
    ])


    val_dataset = torchvision.datasets.CIFAR10(base_path + "data", train=False, download=True, transform=transform)
    #val_dataset = ImageFolder(root=base_path + "data/DatasetCustom", transform=val_transform)
    dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=8)

    img_tensor, _ = next(iter(dataloader))
    img_tensor = img_tensor.to(device)
    B, C, H, W = img_tensor.shape
    mask = create_mask(B, H, W, device)
    # mask = 1.0 - mask  # now 1=hole, 0=valid

    # choose fill_value consistent with normalization
    fill_value = -1.0 if img_tensor.min().item() < 0.0 else 1.0

    # CORRECT : build masked image (keep outside, fill hole)
    masked = img_tensor * (1.0 - mask) + fill_value * mask

    net_input = torch.cat([masked, mask], dim=1)
    print("images:", img_tensor.min().item(), img_tensor.max().item(), "mean", img_tensor.mean().item())
    print("mask:", mask.min().item(), mask.max().item(), "mean", mask.mean().item())
    print("masked:", masked.min().item(), masked.max().item(), "mean", masked.mean().item())
    print("net_input:", net_input.min().item(), net_input.max().item(), "mean", net_input.mean().item())

    model.eval()
    with torch.no_grad():
        pyramid_imgs, reconstructed = model(net_input, mask)

    fig, ax = plt.subplots(1, 4, figsize=(12,4))
    ax[0].imshow(to_img(img_tensor))
    ax[0].set_title("Image originale")

    ax[1].imshow(to_img(masked))
    ax[1].set_title("Image masquée")

    ax[2].imshow(to_img(reconstructed))
    ax[2].set_title("Image générée")

    # Image triché avec la partie non masquée de l'originale
    cheated = reconstructed * mask + img_tensor * (1.0 - mask)
    ax[3].imshow(to_img(cheated))
    ax[3].set_title("Assemblage généré/original")

    # Calcul du psnr entre img_tensor et reconstructed
    mse = torch.mean((img_tensor - reconstructed) ** 2).item()
    psnr = 10 * math.log10(1.0 / mse) if mse > 0 else float('inf')
    print(f"PSNR entre image originale et image générée: {psnr:.2f} dB")


    for a in ax: a.axis("off")
    plt.show()


if config["train"]:
    train_gan()
for i in range(10):
   test_model_gen()
