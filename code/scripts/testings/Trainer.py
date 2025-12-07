import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import csv
import os
import math
import numpy as np
from tqdm import tqdm
from torchvision.utils import make_grid

from utils import create_mask_fast, normalize_images
from model import UNet, InpaintGenerator, Discriminator, PenNET
from AdversarialLoss import AdversarialLoss
import torch.nn.functional as F

class Trainer:
    def __init__(self, config, dataset, model=PenNET()):
        """
        config: dict contenant au minimum:
          - lr, beta1, beta2, batch_size, iterations,
          - adversarial_weight, hole_weight, valid_weight, pyramid_weight,
          - device (e.g. 'cuda' or 'cpu'), save_dir (optional)
          - image_range: 'tanh' ([-1,1]) or 'sigmoid' ([0,1])
        dataset: PyTorch Dataset yielding (image, mask, ...)
        """
        self.device = torch.device(config.get('device','cuda' if torch.cuda.is_available() else 'cpu'))
        self.config = config
        self.dataloader = DataLoader(
            dataset, 
            batch_size=config['batch_size'], 
            shuffle=True, 
            num_workers=config.get('num_workers', 4), 
            pin_memory=True
        )
        
        self.image_range = config.get('image_range', 'tanh')
        assert self.image_range in ['tanh', 'sigmoid'], "image_range must be 'tanh' or 'sigmoid'"
        
        # Models
        self.netG = InpaintGenerator(model).to(self.device)
        self.netD = Discriminator(in_channels=3, use_sn=True).to(self.device)
        
        # Losses & optimizers
        self.adv_loss = AdversarialLoss().to(self.device)
        
        # Configure pixel loss function
        pixel_loss_type = config.get('pixel_loss', 'l1')  # 'l1', 'l2', or 'smooth_l1'
        if pixel_loss_type == 'l1':
            self.pixel_loss = nn.L1Loss()
        elif pixel_loss_type == 'l2':
            self.pixel_loss = nn.MSELoss()
        elif pixel_loss_type == 'smooth_l1':
            self.pixel_loss = nn.SmoothL1Loss()
        else:
            raise ValueError(f"Unknown pixel_loss type: {pixel_loss_type}. Choose 'l1', 'l2', or 'smooth_l1'")
        
        print(f"Using {pixel_loss_type.upper()} loss for pixel reconstruction")
        
        self.optimG = torch.optim.Adam(
            self.netG.parameters(), 
            lr=config['lr'], 
            betas=(config.get('beta1', 0.5), config.get('beta2', 0.999))
        )
        self.optimD = torch.optim.Adam(
            self.netD.parameters(), 
            lr=config['lr'] * config.get('d2glr', 1.0), 
            betas=(config.get('beta1', 0.5), config.get('beta2', 0.999))
        )
        
        self.iters = 0
        self.max_iters = config["total_epochs"] * math.ceil(len(dataset) / config["batch_size"])
        self.best_loss = float('inf')
        
        # Loss tracking
        self.loss_history = {
            'iters': [],
            'loss_D': [],
            'loss_D_real': [],
            'loss_D_fake': [],
            'loss_G': [],
            'loss_G_adv': [],
            'hole_loss': [],
            'valid_loss': [],
            'pyramid_loss': []
        }

    def save_checkpoint(self, filepath, is_best=False):
        """
        Sauvegarde un checkpoint complet de l'entraînement.
        
        Args:
            filepath: Chemin où sauvegarder le checkpoint
            is_best: Si True, c'est le meilleur modèle jusqu'à présent
        """
        checkpoint = {
            'iters': self.iters,
            'max_iters': self.max_iters,
            'best_loss': self.best_loss,
            'netG_state_dict': self.netG.state_dict(),
            'netD_state_dict': self.netD.state_dict(),
            'optimG_state_dict': self.optimG.state_dict(),
            'optimD_state_dict': self.optimD.state_dict(),
            'loss_history': self.loss_history,
            'config': self.config
        }
        
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
        torch.save(checkpoint, filepath)
        
        if is_best:
            best_path = os.path.join(
                os.path.dirname(filepath), 
                'checkpoint_best.pth'
            )
            torch.save(checkpoint, best_path)
    
    def load_checkpoint(self, filepath):
        """
        Charge un checkpoint pour reprendre l'entraînement.
        
        Args:
            filepath: Chemin du checkpoint à charger
        
        Returns:
            bool: True si le chargement a réussi, False sinon
        """
        if not os.path.exists(filepath):
            print(f"Checkpoint not found at {filepath}")
            return False
        
        print(f"Loading checkpoint from {filepath}...")
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Restore training state
        self.iters = checkpoint['iters']
        self.max_iters = checkpoint.get('max_iters', self.max_iters)
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        
        # Restore models
        self.netG.load_state_dict(checkpoint['netG_state_dict'])
        self.netD.load_state_dict(checkpoint['netD_state_dict'])
        
        # Restore optimizers
        self.optimG.load_state_dict(checkpoint['optimG_state_dict'])
        self.optimD.load_state_dict(checkpoint['optimD_state_dict'])
        
        # Restore loss history
        self.loss_history = checkpoint.get('loss_history', self.loss_history)
        
        print(f"Checkpoint loaded successfully. Resuming from iteration {self.iters}/{self.max_iters}")
        return True
    
    def train_epoch(self):
        pbar = tqdm(self.dataloader, desc=f'Epoch Progress', 
                    total=min(len(self.dataloader), 
                             (self.max_iters - self.iters) // 1 + 1),
                    ncols=120)

        for images, *_ in pbar:
            if self.iters >= self.max_iters:
                break
            self.iters += 1
            
            images = images.to(self.device)
            B, C, H, W = images.shape
            
            # Normalize images
            images, fill_value = normalize_images(self.image_range, images)
            
            # Create masks
            masks = create_mask_fast(B, H, W, device=self.device, num_lines_range=self.config['mask_line_count'], thickness_range=self.config['mask_line_width'])
            
            # Build input
            images_masked = images * (1 - masks) + fill_value * masks
            inputs = torch.cat([images_masked, masks], dim=1)

            # Forward G
            feats, pred_img = self.netG(inputs, masks)
            comp_img = images * (1 - masks) + pred_img * masks

            # Train D
            self.optimD.zero_grad()
            real_score = self.netD(images)
            fake_score = self.netD(comp_img.detach())
            loss_D_real = self.adv_loss(real_score, True, True)
            loss_D_fake = self.adv_loss(fake_score, False, True)
            loss_D = 0.5 * (loss_D_real + loss_D_fake)
            loss_D.backward()
            self.optimD.step()

            # Train G
            self.optimG.zero_grad()
            fake_score_for_G = self.netD(comp_img)
            loss_G_adv = self.adv_loss(fake_score_for_G, True, False)
            loss_G = loss_G_adv * self.config.get('adversarial_weight', 1.0)
            
            # Pixel losses
            hole_loss = self.pixel_loss(pred_img * masks, images * masks) / (masks.mean() + 1e-8)
            valid_loss = self.pixel_loss(pred_img * (1 - masks), images * (1 - masks)) / ((1 - masks).mean() + 1e-8)
            
            loss_G = loss_G + hole_loss * self.config.get('hole_weight', 6.0)
            loss_G = loss_G + valid_loss * self.config.get('valid_weight', 1.0)
            
            # Pyramid loss
            pyramid_loss = torch.tensor(0.0, device=self.device)
            if feats is not None and len(feats) > 0 and self.config.get('pyramid_weight', 0.0) > 0.0:
                for f in feats:
                    target = F.interpolate(
                        images, 
                        size=f.size()[2:4], 
                        mode='bilinear', 
                        align_corners=True
                    )
                    pyramid_loss += self.pixel_loss(f, target)
                pyramid_loss = pyramid_loss / len(feats)
                loss_G = loss_G + pyramid_loss * self.config.get('pyramid_weight', 0.5)
            
            loss_G.backward()
            self.optimG.step()

            # Track losses
            log_interval = self.config.get('log_interval', 10)
            if self.iters % log_interval == 0 or self.iters == 1:
                self.loss_history['iters'].append(self.iters)
                self.loss_history['loss_D'].append(loss_D.item())
                self.loss_history['loss_D_real'].append(loss_D_real.item())
                self.loss_history['loss_D_fake'].append(loss_D_fake.item())
                self.loss_history['loss_G'].append(loss_G.item())
                self.loss_history['loss_G_adv'].append(loss_G_adv.item())
                self.loss_history['hole_loss'].append(hole_loss.item())
                self.loss_history['valid_loss'].append(valid_loss.item())
                self.loss_history['pyramid_loss'].append(pyramid_loss.item())


            pbar.set_postfix({
                'D': f'{loss_D.item():.4f}',
                'G': f'{loss_G.item():.4f}',
                'hole': f'{hole_loss.item():.4f}',
                'valid': f'{valid_loss.item():.4f}',
                'iter': f'{self.iters}/{self.max_iters}'
            })

            min_improvement = self.config.get('min_improvement', 0.01)  # Amélioration minimale de 1%
            save_cooldown = self.config.get('save_cooldown', 100)  # Attendre 100 iters entre sauvegardes
            if not hasattr(self, 'last_save_iter'):
                self.last_save_iter = 0


            # Save best model
            # Wait a bit between saves and there need to be a minimum threshold in improvements to trigger a save
            save_cooldown_ok = (self.iters - self.last_save_iter) >= save_cooldown and self.iters > 0.5 * self.max_iters
            save_diff_ok = loss_G.item() < self.best_loss * (1 - min_improvement)
            if save_cooldown_ok and save_diff_ok:
                self.best_loss = loss_G.item()
                save_dir = self.config.get('save_dir', './')
                checkpoint_path = f"{save_dir}/checkpoint_best.pth"
                self.save_checkpoint(checkpoint_path, is_best=True)
            
            # Periodic checkpoint save
            checkpoint_interval = self.config.get('checkpoint_interval', 1000)
            if self.iters % checkpoint_interval == 0:
                save_dir = self.config.get('save_dir', './')
                checkpoint_path = f"{save_dir}/checkpoint_iter_{self.iters}.pth"
                self.save_checkpoint(checkpoint_path)
        pbar.close()

    def save_models(self, pathG, pathD):
        torch.save(self.netG.state_dict(), pathG)
        torch.save(self.netD.state_dict(), pathD)

    def save_metrics(self, filepath):
        """Save training metrics to CSV."""
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            header = list(self.loss_history.keys())
            writer.writerow(header)
            num_rows = len(self.loss_history['iters'])
            for i in range(num_rows):
                row = [self.loss_history[key][i] for key in header]
                writer.writerow(row)
        print(f"Saved metrics to {filepath}")

    def train(self, resume_from=None):
        """
        Lance l'entraînement.
        
        Args:
            resume_from: Chemin vers un checkpoint pour reprendre l'entraînement.
                        Si None, commence depuis le début.
        """
        # Load checkpoint if specified
        if resume_from is not None:
            if self.load_checkpoint(resume_from):
                print(f"Resuming training from iteration {self.iters}")
            else:
                print("Starting training from scratch")
        
        try:
            while self.iters < self.max_iters:
                self.train_epoch()

                # Save periodic checkpoint (already handled in train_epoch)
                # But also save at end of epoch
                save_dir = self.config.get('save_dir', './')
                checkpoint_path = f"{save_dir}/checkpoint_latest.pth"
                self.save_checkpoint(checkpoint_path)
                
                # Save metrics
                metrics_csv = f"{save_dir}/training_metrics.csv"
                self.save_metrics(metrics_csv)
            
            # Final save
            save_dir = self.config.get('save_dir', './')
            final_checkpoint = f"{save_dir}/checkpoint_final.pth"
            self.save_checkpoint(final_checkpoint)
            
            metrics_plot = f"{save_dir}/training_curves.png"
            self.plot_metrics(metrics_plot)
            print("Training finished")
        
        except KeyboardInterrupt:
            print("\nTraining interrupted by user!")
            save_dir = self.config.get('save_dir', './')
            interrupt_checkpoint = f"{save_dir}/checkpoint_interrupted.pth"
            self.save_checkpoint(interrupt_checkpoint)
            print(f"Checkpoint saved to {interrupt_checkpoint}")
            print("You can resume training by loading this checkpoint.")

    # Keep other methods (update_realtime_viz, plot_metrics, etc.) unchanged
    def tensor_to_image(self, tensor):
        """Convert tensor to displayable image."""
        if len(tensor.shape) == 4:
            tensor = tensor[0]
        img = tensor.detach().cpu().permute(1, 2, 0).numpy()
        if self.image_range == 'tanh':
            img = (img + 1) / 2
        img = np.clip(img, 0, 1)
        return img

    @staticmethod
    def plot_checkpoint_history(checkpoint_path, save_path=None):
        """
        Affiche l'historique d'entraînement depuis un checkpoint.
        
        Args:
            checkpoint_path: Chemin vers le checkpoint
            save_path: Chemin où sauvegarder le graphique (optionnel)
        """
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint not found at {checkpoint_path}")
            return
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        loss_history = checkpoint.get('loss_history', None)
        
        if loss_history is None or len(loss_history.get('iters', [])) == 0:
            print("No loss history found in checkpoint")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Training History from Checkpoint\n(Iteration {checkpoint.get("iters", "?")})', fontsize=16)
        
        iters = loss_history['iters']
        
        # D losses
        axes[0, 0].plot(iters, loss_history['loss_D'], label='Total D Loss', linewidth=2)
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Discriminator Loss')
        axes[0, 0].grid(True)
        axes[0, 0].legend()
        
        # D real vs fake
        axes[0, 1].plot(iters, loss_history['loss_D_real'], label='D Real', linewidth=2)
        axes[0, 1].plot(iters, loss_history['loss_D_fake'], label='D Fake', linewidth=2)
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].set_title('Discriminator Real vs Fake')
        axes[0, 1].grid(True)
        axes[0, 1].legend()
        
        # G adversarial loss
        axes[0, 2].plot(iters, loss_history['loss_G_adv'], label='G Adversarial', linewidth=2, color='orange')
        axes[0, 2].set_xlabel('Iteration')
        axes[0, 2].set_ylabel('Loss')
        axes[0, 2].set_title('Generator Adversarial Loss')
        axes[0, 2].grid(True)
        axes[0, 2].legend()
        
        # Total G loss
        axes[1, 0].plot(iters, loss_history['loss_G'], label='Total G Loss', linewidth=2, color='green')
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].set_title('Generator Total Loss')
        axes[1, 0].grid(True)
        axes[1, 0].legend()
        
        # Hole and valid losses
        axes[1, 1].plot(iters, loss_history['hole_loss'], label='Hole Loss', linewidth=2)
        axes[1, 1].plot(iters, loss_history['valid_loss'], label='Valid Loss', linewidth=2)
        axes[1, 1].plot(iters, loss_history['pyramid_loss'], label='Pyramid Loss', linewidth=2)
        axes[1, 1].set_xlabel('Iteration')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].set_title('Reconstruction Losses')
        axes[1, 1].grid(True)
        axes[1, 1].legend()
        
        # Combined G loss components
        axes[1, 2].plot(iters, loss_history['loss_G_adv'], label='Adversarial', linewidth=2)
        axes[1, 2].plot(iters, loss_history['hole_loss'], label='Hole', linewidth=2)
        axes[1, 2].plot(iters, loss_history['valid_loss'], label='Valid', linewidth=2)
        axes[1, 2].set_xlabel('Iteration')
        axes[1, 2].set_ylabel('Loss')
        axes[1, 2].set_title('G Loss Components')
        axes[1, 2].grid(True)
        axes[1, 2].legend()
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            print(f"Saved plot to {save_path}")
        
        plt.show()
        return fig

    def plot_metrics(self, filepath=None):
        """Plot training metrics."""
        # Use the static method with current loss_history
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Training Metrics', fontsize=16)
        
        iters = self.loss_history['iters']
        
        # D losses
        axes[0, 0].plot(iters, self.loss_history['loss_D'], label='Total D Loss', linewidth=2)
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Discriminator Loss')
        axes[0, 0].grid(True)
        axes[0, 0].legend()
        
        # D real vs fake
        axes[0, 1].plot(iters, self.loss_history['loss_D_real'], label='D Real', linewidth=2)
        axes[0, 1].plot(iters, self.loss_history['loss_D_fake'], label='D Fake', linewidth=2)
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].set_title('Discriminator Real vs Fake')
        axes[0, 1].grid(True)
        axes[0, 1].legend()
        
        # G adversarial loss
        axes[0, 2].plot(iters, self.loss_history['loss_G_adv'], label='G Adversarial', linewidth=2, color='orange')
        axes[0, 2].set_xlabel('Iteration')
        axes[0, 2].set_ylabel('Loss')
        axes[0, 2].set_title('Generator Adversarial Loss')
        axes[0, 2].grid(True)
        axes[0, 2].legend()
        
        # Total G loss
        axes[1, 0].plot(iters, self.loss_history['loss_G'], label='Total G Loss', linewidth=2, color='green')
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].set_title('Generator Total Loss')
        axes[1, 0].grid(True)
        axes[1, 0].legend()
        
        # Hole and valid losses
        axes[1, 1].plot(iters, self.loss_history['hole_loss'], label='Hole Loss', linewidth=2)
        axes[1, 1].plot(iters, self.loss_history['valid_loss'], label='Valid Loss', linewidth=2)
        axes[1, 1].plot(iters, self.loss_history['pyramid_loss'], label='Pyramid Loss', linewidth=2)
        axes[1, 1].set_xlabel('Iteration')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].set_title('Reconstruction Losses')
        axes[1, 1].grid(True)
        axes[1, 1].legend()
        
        # Combined G loss components
        axes[1, 2].plot(iters, self.loss_history['loss_G_adv'], label='Adversarial', linewidth=2)
        axes[1, 2].plot(iters, self.loss_history['hole_loss'], label='Hole', linewidth=2)
        axes[1, 2].plot(iters, self.loss_history['valid_loss'], label='Valid', linewidth=2)
        axes[1, 2].set_xlabel('Iteration')
        axes[1, 2].set_ylabel('Loss')
        axes[1, 2].set_title('G Loss Components')
        axes[1, 2].grid(True)
        axes[1, 2].legend()
        
        plt.tight_layout()
        
        if filepath:
            os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
            plt.savefig(filepath, dpi=100, bbox_inches='tight')
            print(f"Saved plot to {filepath}")
        
        plt.show()