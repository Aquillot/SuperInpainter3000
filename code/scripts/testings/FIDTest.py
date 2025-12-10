# Génération et sauvegarde d'un millier d'images pour le calcul du FID score
import os
import torch
from torchvision.transforms import ToTensor, Compose, Normalize, Resize, InterpolationMode, RandomCrop, \
    RandomHorizontalFlip
from torchvision.transforms.v2 import CenterCrop
from torchvision.utils import save_image
from torchvision.datasets import ImageFolder
from pennet import InpaintGeneratorPennet
from model import PenNET, InpaintGenerator
from model import UNet
from utils import create_mask, to_img
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision
from tqdm import tqdm


# --- Helpers robustes et fonction corrigée / combo ---
import os
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from torchmetrics.image.fid import FrechetInceptionDistance

def _ensure_batch_tensor(output, batch_size=None, device=None):
    """
    Prend la sortie du générateur qui peut être :
      - un tensor B,C,H,W  --> retourne tel quel
      - un tensor C,H,W    --> ajoute dim batch
      - une liste/tuple de tensors :
          * si chaque élément a dim==3 (C,H,W) -> stack en B,C,H,W
          * si contient un tensor dim==4 -> retourne ce tensor
          * sinon prend le premier tensor trouvé
    """
    if torch.is_tensor(output):
        if output.dim() == 3:
            return output.unsqueeze(0)
        return output
    if isinstance(output, (list, tuple)):
        # Si liste d'éléments tensoriels
        tensors = [o for o in output if torch.is_tensor(o)]
        if len(tensors) == 0:
            raise ValueError("Le générateur a renvoyé une liste sans tenseurs.")
        # Si liste d'éléments (C,H,W) -> stack
        if tensors[0].dim() == 3:
            return torch.stack(tensors, dim=0)
        # Chercher un tensor B,C,H,W
        for t in tensors:
            if t.dim() == 4:
                return t
        # Sinon retourne le premier tensor trouvé
        return tensors[0]
    raise ValueError(f"Type de sortie du générateur inattendu: {type(output)}")

# ---- remplacements à coller ----
def _denorm_to_0_1(t):
    """
    Reçoit tensor float (C,H,W) ou (B,C,H,W).
    Détecte le range probable et renvoie float tensor en [0,1].
    Détection plus robuste que juste regarder t_min<0 :
      - si t.min() < -0.05 -> probablement range [-1,1] (Normalize 0.5,0.5)
      - elif t.max() > 1.05 -> probablement range [-1,1] (au cas où)
      - sinon on suppose [0,1] et on clamp.
    """
    if not torch.is_tensor(t):
        raise ValueError("Attendu un tensor")
    t = t.detach().cpu().float()
    t_min = float(t.min())
    t_max = float(t.max())
    # heuristique robuste
    if t_min < -0.05 or t_max > 1.05:
        # On suppose que l'espace est [-1,1]
        t = (t + 1.0) / 2.0
    # clamp final
    t = t.clamp(0.0, 1.0)
    return t


def tensor_to_pil(img_tensor):
    """
    Convertit C,H,W ou B,C,H,W (Tensor) ou list/tuple -> PIL.Image.
    Si batch (B,C,H,W) on retourne la première image du batch.
    Utilise la dénormalisation robuste ci-dessus.
    """
    if isinstance(img_tensor, (list, tuple)):
        img_tensor = _ensure_batch_tensor(img_tensor)
    if torch.is_tensor(img_tensor):
        if img_tensor.dim() == 4:
            img_tensor = img_tensor[0]
        t = _denorm_to_0_1(img_tensor)
        arr = (t * 255.0).round().byte().permute(1, 2, 0).numpy()
        return Image.fromarray(arr)
    raise ValueError("tensor_to_pil: input must be tensor or list/tuple of tensors")

def tensor_batch_to_uint8(tensor_batch):
    """
    tensor_batch : B,C,H,W float, peut être en [-1,1] ou [0,1].
    Retourne uint8 B,C,H,W dans [0,255] prêt pour torchmetrics FID.
    """
    t = _denorm_to_0_1(tensor_batch)    # float [0,1]
    t_uint8 = (t * 255.0).round().to(torch.uint8)
    return t_uint8

def generate_images_for_fid(generator, dataset, save_dir, num_images=1000, batch_size=50, device='cuda'):
    """
    Corrigé : génère des images et les sauve sur disque.
    Robuste aux sorties du générateur qui peuvent être list/tuple.
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
    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (list, tuple)):
                imgs = batch[0]
            else:
                imgs = batch
            imgs = imgs.to(device)
            B = imgs.shape[0]

            # masque et préparation (même logique que ton code)
            _, C, H, W = imgs.shape
            mask = create_mask(B, H, W, device)   # ta fonction create_mask
            fill_value = -1.0 if imgs.min().item() < 0.0 else 1.0
            masked = imgs * (1.0 - mask) + fill_value * mask
            net_input = torch.cat([masked, mask], dim=1)

            feats, recon = generator(net_input, mask)
            for i in range(B):
                if saved >= num_images:
                    return
                real_pil = tensor_to_pil(imgs[i])
                fake_pil = tensor_to_pil(recon[i])
                idx_str = f"{saved:06d}"
                real_pil.save(os.path.join(real_dir, f"img_{idx_str}.png"))
                fake_pil.save(os.path.join(fake_dir, f"img_{idx_str}.png"))
                saved += 1


def generate_images_and_compute_fid(generator, dataset, num_images=1000, batch_size=50,
                                    device='cuda', return_images=False, resize_for_fid=299):
    """
    Fonction combo :
      - génère jusqu'à num_images (itération sur dataset)
      - garde images réelles + générées EN MÉMOIRE si return_images=True (liste de PIL ou tensors)
      - calcule le FID en mémoire (sans écrire sur disque) et retourne le score
    Args:
      generator: nn.Module
      dataset: torch.utils.data.Dataset
      return_images: False par défaut ; si True retourne (fid_score, real_pils, fake_pils)
      resize_for_fid: taille utilisée pour FID (Inception attend 299)
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    generator = generator.to(device)
    generator.eval()

    fid = FrechetInceptionDistance(feature=2048).to(device)

    # transform pour redimensionner + center crop avant FID (float [0,1])
    fid_transform = transforms.Compose([
        transforms.Resize((resize_for_fid, resize_for_fid)),
        transforms.ToTensor(),  # produit float [0,1]
    ])

    saved = 0
    real_images_mem = []  # si return_images True -> on stocke PIL.Image
    fake_images_mem = []

    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (list, tuple)):
                imgs = batch[0]
            else:
                imgs = batch
            imgs = imgs.to(device)
            B = imgs.shape[0]

            # masques etc.
            _, C, H, W = imgs.shape
            mask = create_mask(B, H, W, device)
            fill_value = -1.0 if imgs.min().item() < 0.0 else 1.0
            masked = imgs * (1.0 - mask) + fill_value * mask
            net_input = torch.cat([masked, mask], dim=1)

            recon = generator(net_input, mask)
            recon = _ensure_batch_tensor(recon, batch_size=B, device=device)
            if recon.shape[0] != B:
                if recon.shape[0] > B:
                    recon = recon[:B]
                else:
                    raise RuntimeError(f"Batch size mismatch: recon {recon.shape[0]} vs imgs {B}")

            # Troncature si on dépasse num_images
            remaining = num_images - saved
            take = min(B, remaining)
            real_batch = imgs[:take].detach().cpu()
            fake_batch = recon[:take].detach().cpu()

            # Denorm -> [0,1]
            real_batch = _denorm_to_0_1(real_batch)
            fake_batch = _denorm_to_0_1(fake_batch)

            # Préparer uint8 attendu par torchmetrics FID
            real_uint8 = (real_batch * 255.0).round().to(torch.uint8).to(device)
            fake_uint8 = (fake_batch * 255.0).round().to(torch.uint8).to(device)

            # Mettre à jour le FID (en mémoire)
            fid.update(real_uint8, real=True)
            fid.update(fake_uint8, real=False)

            # Optionnel : stocker en mémoire (PIL)
            if return_images:
                for r, f in zip(real_batch, fake_batch):
                    real_images_mem.append(tensor_to_pil(r))
                    fake_images_mem.append(tensor_to_pil(f))

            saved += take
            if saved >= num_images:
                break

    fid_score = fid.compute().item()
    if return_images:
        return fid_score, real_images_mem, fake_images_mem
    return fid_score

# A partir de deux dossiers d'images générées et réelles, calculer le FID score
from torchmetrics.image.fid import FrechetInceptionDistance

# -- adaptation de calculate_fid_score pour s'assurer de cohérence --
def calculate_fid_score(generated_dir, real_dir, batch_size=50, device='cuda'):
    """
    Calcule le FID score entre deux dossiers d'images.
    Lecture PIL -> ToTensor renvoie déjà [0,1] ; on s'assure ensuite de convertir
    en uint8 via tensor_batch_to_uint8 qui est cohérent par rapport à la génération.
    """
    transform = transforms.Compose([
        transforms.Resize((299)),
        transforms.CenterCrop(299),
        transforms.ToTensor(),   # PIL -> float [0,1]
    ])

    class ImageFolderDataset(Dataset):
        def __init__(self, folder, transform=None):
            self.folder = folder
            self.transform = transform
            self.image_files = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            self.image_files.sort()

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

    # Add real images
    for batch in real_loader:
        batch = batch.to(device)                       # float [0,1]
        batch_uint8 = tensor_batch_to_uint8(batch).to(device)  # uint8
        fid.update(batch_uint8, real=True)

    # Add fake images
    for batch in fake_loader:
        batch = batch.to(device)
        batch_uint8 = tensor_batch_to_uint8(batch).to(device)
        fid.update(batch_uint8, real=False)

    fid_score = fid.compute().item()
    return fid_score


base_path = "../../"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")




# Nettoyer les préfixes du state_dict
def clean_state_dict(state):
    new_state = {}
    for k, v in state.items():
        # Retirer les préfixes courants
        nk = k
        for prefix in ['module.', 'unet.', 'generator.', 'model.', 'net.']:
            if nk.startswith(prefix):
                nk = nk[len(prefix):]
                break
        new_state[nk] = v
    return new_state



val_transform = Compose([
        CenterCrop(128),  # recadrage centrer
        ToTensor(),
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # -> range [-1,1] (tanh)
    ])


# val_dataset = torchvision.datasets.CIFAR10(base_path + "data", train=False, download=True, transform=transform)
val_dataset = ImageFolder(root=base_path + "data/val_256", transform=val_transform)


# charger votre générateur (ex : UNet ou InpaintGenerator)
path = '../../models/SuperInpainter3000.pth'
state = torch.load(path, map_location=device)

# Extraire le vrai state_dict (peut être enveloppé dans différentes clés)
if isinstance(state, dict):
    # Chercher le state_dict dans les clés communes
    for key in ['netG', 'generator', 'state_dict', 'model_state_dict', 'gen_state_dict']:
        if key in state:
            state_dict = state[key]
            print(f"State_dict trouvé dans la clé: '{key}'")
            break
    else:
        # Si aucune clé connue, utiliser directement le checkpoint
        state_dict = state
else:
    state_dict = state
cleaned_state = clean_state_dict(state_dict)

# Essayer de charger avec différentes architectures
architectures = [
    ("PENNet", lambda: InpaintGeneratorPennet(init_weights=False)),
    ("PenNET/InpaintGenerator", lambda: PenNET(3)),
    ("UNet", lambda: UNet(3))
]
gen = None
for arch_name, model_builder in architectures:
    try:
        print(f"Tentative de chargement avec {arch_name}...")
        candidate = model_builder().to(device)
        candidate.load_state_dict(cleaned_state, strict=False)

        # Vérifier si le chargement a réussi (au moins 50% des paramètres chargés)
        model_keys = set(candidate.state_dict().keys())
        loaded_keys = set(cleaned_state.keys())
        match_ratio = len(model_keys & loaded_keys) / len(model_keys)

        if match_ratio > 0.5:
            gen = candidate
            model_type = arch_name
            # Ajuster la taille d'entrée selon le modèle
            model_name = path.split('/')[-1]
            selected_model_path = path
            print(f"Modèle chargé avec {arch_name} ({match_ratio * 100:.1f}% correspondance)")
            break
        else:
            print(f"{arch_name}: Trop peu de paramètres correspondent ({match_ratio * 100:.1f}%)")

    except Exception as e:
        print(f"{arch_name} échec: {str(e)[:100]}")
        continue

gen.load_state_dict(cleaned_state)

print ("Générateur chargé pour FID computation.")
generate_images_for_fid(gen, val_dataset, save_dir='outputs/fid', num_images=10000, batch_size=12, device=device)
print("Images générées pour FID.")

fid_score = calculate_fid_score('outputs/fid/fake', 'outputs/fid/real', batch_size=12, device=device)
print(f"FID score: {fid_score}")