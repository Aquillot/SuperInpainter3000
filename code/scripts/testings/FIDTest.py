# Génération et sauvegarde d'un millier d'images pour le calcul du FID score
import os
import torch
from torchvision.transforms import ToTensor, Compose, Normalize, Resize, InterpolationMode, RandomCrop, \
    RandomHorizontalFlip
from torchvision.utils import save_image
from torchvision.datasets import ImageFolder
from model import PenNET, InpaintGenerator
from model import UNet
from utils import create_mask
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision
from tqdm import tqdm


def generate_images_for_fid(generator, dataset, save_dir, num_images=1000, batch_size=50, device='cuda'):
    """
    Génère et sauvegarde paires d'images "real" et "fake" pour le calcul du FID.

    Args:
        generator: modèle générateur (PyTorch nn.Module) attendu en mode eval; prend en entrée un tenseur concaténé [masked, mask]
        dataset: instance de torch.utils.data.Dataset (retourne soit (img, ...) soit img)
        save_dir: répertoire où seront créés les sous-dossiers 'real' et 'fake'
        num_images: nombre total d'images à générer/sauver
        batch_size: taille de batch pour l'inférence
        device: 'cuda' ou 'cpu'

    Comportement:
        - Aucun affichage à l'écran
        - Sauvegarde dans save_dir/real et save_dir/fake
        - Itère le dataset dans l'ordre jusqu'à atteindre num_images
    """

    os.makedirs(save_dir, exist_ok=True)
    real_dir = os.path.join(save_dir, 'real')
    fake_dir = os.path.join(save_dir, 'fake')
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(fake_dir, exist_ok=True)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    generator = generator.to(device)
    generator.eval()

    saved = 0

    def tensor_to_pil(img_tensor):
        # img_tensor: C,H,W or B,C,H,W
        if img_tensor.dim() == 4:
            img_tensor = img_tensor[0]
        t = img_tensor.detach().cpu()
        t_min = float(t.min())
        t_max = float(t.max())
        # Detect range and denormalize if needed
        # If in [-1,1], convert to [0,1]
        if t_min >= -1.1 and t_max <= 1.1 and t_min < 0:
            t = (t + 1.0) / 2.0
        # If already in [0,1], keep
        t = t.clamp(0.0, 1.0)
        arr = (t * 255).byte().permute(1, 2, 0).numpy()
        return Image.fromarray(arr)

    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (list, tuple)):
                imgs = batch[0]
            else:
                imgs = batch

            imgs = imgs.to(device)
            B, C, H, W = imgs.shape

            # create mask: adopt la même convention que test_model_gen (mask where 1=hole)
            mask = create_mask(B, H, W, device)
            mask = 1.0 - mask

            # choose fill_value consistent with normalization range
            fill_value = -1.0 if imgs.min().item() < 0.0 else 1.0

            masked = imgs * (1.0 - mask) + fill_value * mask
            net_input = torch.cat([masked, mask], dim=1)

            recon = generator(net_input)

            # sauvegarde image par image
            for i in range(B):
                if saved >= num_images:
                    return

                real_img = imgs[i]
                fake_img = recon[i]

                real_pil = tensor_to_pil(real_img)
                fake_pil = tensor_to_pil(fake_img)

                idx_str = f"{saved:06d}"
                real_pil.save(os.path.join(real_dir, f"img_{idx_str}.png"))
                fake_pil.save(os.path.join(fake_dir, f"img_{idx_str}.png"))

                saved += 1

    # fonction termine silencieusement une fois le dataset parcouru ou num_images atteint


# A partir de deux dossiers d'images générées et réelles, calculer le FID score
from torchmetrics.image.fid import FrechetInceptionDistance

def calculate_fid_score(generated_dir, real_dir, batch_size=50, device='cuda'):
    """
    Calcule le FID score entre deux dossiers d'images.
    """
    transform = transforms.Compose([
        transforms.Resize((299)),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
    ])

    class ImageFolderDataset(Dataset):
        def __init__(self, folder, transform=None):
            self.folder = folder
            self.transform = transform
            self.image_files = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        def __len__(self):
            return len(self.image_files)

        def __getitem__(self, idx):
            img_path = os.path.join(self.folder, self.image_files[idx])
            img = Image.open(img_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img

    real_dataset = ImageFolderDataset(real_dir, transform=transform)
    fake_dataset = ImageFolderDataset(generated_dir, transform=transform)

    real_loader = DataLoader(real_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    fake_loader = DataLoader(fake_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    fid = FrechetInceptionDistance(feature=2048).to(device)

    def to_uint8(tensor):
        # tensor : B,C,H,W float in [0,1] expected -> convert to uint8
        if tensor.dtype == torch.uint8:
            return tensor
        t = (tensor * 255.0).clamp(0, 255)
        return t.to(torch.uint8)

    # Add real images
    for batch in real_loader:
        batch = batch.to(device)
        batch_uint8 = to_uint8(batch)
        fid.update(batch_uint8, real=True)

    # Add fake images
    for batch in fake_loader:
        batch = batch.to(device)
        batch_uint8 = to_uint8(batch)
        fid.update(batch_uint8, real=False)

    fid_score = fid.compute().item()
    return fid_score


base_path = "../../"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

val_transform = Compose([
        RandomCrop(128),  # recadrage aléatoire -> variabilité
        ToTensor(),
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # -> range [-1,1] (tanh)
    ])


# val_dataset = torchvision.datasets.CIFAR10(base_path + "data", train=False, download=True, transform=transform)
val_dataset = ImageFolder(root=base_path + "data/DatasetCustom", transform=val_transform)


# charger votre générateur (ex : UNet ou InpaintGenerator)
gen = InpaintGenerator(PenNET(3)).to(device)
state = torch.load('../../models/gen_best.pth', map_location=device)
# si nécessaire, adapter les clés du state_dict (comme dans test_model_gen)
new_state_dict = {k.replace('unet.', ''): v for k, v in state.items()}
gen.load_state_dict(new_state_dict)

print ("Générateur chargé pour FID computation.")
generate_images_for_fid(gen, val_dataset, save_dir='outputs/fid', num_images=500, batch_size=12, device=device)
print("Images générées pour FID.")

fid_score = calculate_fid_score('outputs/fid/fake', 'outputs/fid/real', batch_size=4, device=device)
print(f"FID score: {fid_score}")