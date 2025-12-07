import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import csv
import os
import math
import numpy as np
from torchvision.utils import make_grid

from utils import create_mask, normalize_images

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
        
        # CRITICAL: Define expected image range
        self.image_range = config.get('image_range', 'tanh')  # 'tanh' or 'sigmoid'
        assert self.image_range in ['tanh', 'sigmoid'], "image_range must be 'tanh' or 'sigmoid'"
        
        # models
        self.netG = InpaintGenerator(model).to(self.device)
        self.netD = Discriminator(in_channels=3, use_sn=True).to(self.device)
        
        # losses & optimizers
        self.adv_loss = AdversarialLoss().to(self.device)
        self.l1 = nn.L1Loss()
        
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
        
        # loss tracking for visualization
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
        
        # Real-time visualization setup
        self.use_realtime_viz = config.get('realtime_viz', False)
        if self.use_realtime_viz:
            plt.ion()  # Enable interactive mode
            self.fig = plt.figure(figsize=(20, 10))
            self.gs = GridSpec(3, 5, figure=self.fig, hspace=0.3, wspace=0.3)
            
            # Create subplots for images
            self.ax_input = self.fig.add_subplot(self.gs[0, 0])
            self.ax_mask = self.fig.add_subplot(self.gs[0, 1])
            self.ax_pred = self.fig.add_subplot(self.gs[0, 2])
            self.ax_comp = self.fig.add_subplot(self.gs[0, 3])
            self.ax_gt = self.fig.add_subplot(self.gs[0, 4])
            
            # Create subplots for loss curves
            self.ax_loss_d = self.fig.add_subplot(self.gs[1, 0:2])
            self.ax_loss_g = self.fig.add_subplot(self.gs[1, 2:4])
            self.ax_loss_pixel = self.fig.add_subplot(self.gs[1, 4])
            self.ax_loss_components = self.fig.add_subplot(self.gs[2, 0:3])
            self.ax_loss_ratio = self.fig.add_subplot(self.gs[2, 3:5])
            
            # Configure axes
            for ax in [self.ax_input, self.ax_mask, self.ax_pred, self.ax_comp, self.ax_gt]:
                ax.axis('off')
            
            self.ax_input.set_title('Masked Input', fontsize=10, fontweight='bold')
            self.ax_mask.set_title('Mask', fontsize=10, fontweight='bold')
            self.ax_pred.set_title('Prediction', fontsize=10, fontweight='bold')
            self.ax_comp.set_title('Composite', fontsize=10, fontweight='bold')
            self.ax_gt.set_title('Ground Truth', fontsize=10, fontweight='bold')
            
            self.fig.suptitle('Training Progress', fontsize=16, fontweight='bold')
            plt.show(block=False)
        self.best_loss = float('inf')
    
    def update_realtime_viz(self, images, masks, pred_img, comp_img, images_masked):
        """Update the real-time visualization with current batch results."""
        # Update images
        self.ax_input.clear()
        self.ax_mask.clear()
        self.ax_pred.clear()
        self.ax_comp.clear()
        self.ax_gt.clear()
        
        self.ax_input.imshow(self.tensor_to_image(images_masked))
        self.ax_input.set_title('Masked Input', fontsize=10, fontweight='bold')
        self.ax_input.axis('off')
        
        self.ax_mask.imshow(self.tensor_to_image(masks), cmap='gray')
        self.ax_mask.set_title('Mask (white=hole)', fontsize=10, fontweight='bold')
        self.ax_mask.axis('off')
        
        self.ax_pred.imshow(self.tensor_to_image(pred_img))
        self.ax_pred.set_title('Prediction', fontsize=10, fontweight='bold')
        self.ax_pred.axis('off')
        
        self.ax_comp.imshow(self.tensor_to_image(comp_img))
        self.ax_comp.set_title('Composite', fontsize=10, fontweight='bold')
        self.ax_comp.axis('off')
        
        self.ax_gt.imshow(self.tensor_to_image(images))
        self.ax_gt.set_title('Ground Truth', fontsize=10, fontweight='bold')
        self.ax_gt.axis('off')
        
        # Update loss curves (only if we have enough data)
        if len(self.loss_history['iters']) > 1:
            iters = self.loss_history['iters']
            
            # D losses
            self.ax_loss_d.clear()
            self.ax_loss_d.plot(iters, self.loss_history['loss_D'], 'b-', label='Total D', linewidth=2)
            self.ax_loss_d.plot(iters, self.loss_history['loss_D_real'], 'g--', label='D Real', alpha=0.7)
            self.ax_loss_d.plot(iters, self.loss_history['loss_D_fake'], 'r--', label='D Fake', alpha=0.7)
            self.ax_loss_d.set_xlabel('Iteration')
            self.ax_loss_d.set_ylabel('Loss')
            self.ax_loss_d.set_title('Discriminator Losses')
            self.ax_loss_d.legend(loc='upper right', fontsize=8)
            self.ax_loss_d.grid(True, alpha=0.3)
            
            # G adversarial
            self.ax_loss_g.clear()
            self.ax_loss_g.plot(iters, self.loss_history['loss_G'], 'purple', label='Total G', linewidth=2)
            self.ax_loss_g.plot(iters, self.loss_history['loss_G_adv'], 'orange', label='G Adversarial', alpha=0.7)
            self.ax_loss_g.set_xlabel('Iteration')
            self.ax_loss_g.set_ylabel('Loss')
            self.ax_loss_g.set_title('Generator Losses')
            self.ax_loss_g.legend(loc='upper right', fontsize=8)
            self.ax_loss_g.grid(True, alpha=0.3)
            
            # Pixel losses
            self.ax_loss_pixel.clear()
            self.ax_loss_pixel.plot(iters, self.loss_history['hole_loss'], 'red', label='Hole', linewidth=2)
            self.ax_loss_pixel.plot(iters, self.loss_history['valid_loss'], 'blue', label='Valid', linewidth=2)
            if max(self.loss_history['pyramid_loss']) > 0:
                self.ax_loss_pixel.plot(iters, self.loss_history['pyramid_loss'], 'green', label='Pyramid', linewidth=2)
            self.ax_loss_pixel.set_xlabel('Iteration')
            self.ax_loss_pixel.set_ylabel('Loss')
            self.ax_loss_pixel.set_title('Reconstruction Losses')
            self.ax_loss_pixel.legend(loc='upper right', fontsize=8)
            self.ax_loss_pixel.grid(True, alpha=0.3)
            
            # All components
            self.ax_loss_components.clear()
            self.ax_loss_components.plot(iters, self.loss_history['loss_G_adv'], label='Adversarial', linewidth=1.5)
            self.ax_loss_components.plot(iters, self.loss_history['hole_loss'], label='Hole', linewidth=1.5)
            self.ax_loss_components.plot(iters, self.loss_history['valid_loss'], label='Valid', linewidth=1.5)
            if max(self.loss_history['pyramid_loss']) > 0:
                self.ax_loss_components.plot(iters, self.loss_history['pyramid_loss'], label='Pyramid', linewidth=1.5)
            self.ax_loss_components.set_xlabel('Iteration')
            self.ax_loss_components.set_ylabel('Loss')
            self.ax_loss_components.set_title('All Loss Components')
            self.ax_loss_components.legend(loc='upper right', fontsize=8)
            self.ax_loss_components.grid(True, alpha=0.3)
            
            # D/G ratio (indicator of training balance)
            self.ax_loss_ratio.clear()
            d_losses = np.array(self.loss_history['loss_D'])
            g_adv_losses = np.array(self.loss_history['loss_G_adv'])
            ratio = d_losses / (g_adv_losses + 1e-8)
            self.ax_loss_ratio.plot(iters, ratio, 'purple', linewidth=2)
            self.ax_loss_ratio.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Balance')
            self.ax_loss_ratio.set_xlabel('Iteration')
            self.ax_loss_ratio.set_ylabel('Ratio')
            self.ax_loss_ratio.set_title('D_loss / G_adv_loss Ratio')
            self.ax_loss_ratio.legend(loc='upper right', fontsize=8)
            self.ax_loss_ratio.grid(True, alpha=0.3)

        plt.draw()
        plt.pause(0.001)  # Small pause to update display
    
    def train_epoch(self):
        for images, *_ in self.dataloader:
            if self.iters >= self.max_iters:
                break
            self.iters += 1
            
            images = images.to(self.device)
            B, C, H, W = images.shape
            
            # Normalize images to expected range
            images, fill_value = normalize_images(self.image_range, images)
            
            # Create masks: 1=hole, 0=valid
            masks = create_mask(B, H, W, device=self.device)
            
            # Build input: masked image + mask channel
            images_masked = images * (1 - masks) + fill_value * masks
            inputs = torch.cat([images_masked, masks], dim=1)  # (N, 4, H, W)

            # ----- Forward G -----
            feats, pred_img = self.netG(inputs, masks)
            comp_img = images * (1 - masks) + pred_img * masks

            # ----- Train D -----
            self.optimD.zero_grad()
            real_score = self.netD(images)
            fake_score = self.netD(comp_img.detach())
            loss_D_real = self.adv_loss(real_score, True, True)
            loss_D_fake = self.adv_loss(fake_score, False, True)
            loss_D = 0.5 * (loss_D_real + loss_D_fake)
            loss_D.backward()
            self.optimD.step()

            # ----- Train G -----
            self.optimG.zero_grad()
            fake_score_for_G = self.netD(comp_img)
            loss_G_adv = self.adv_loss(fake_score_for_G, True, False)
            loss_G = loss_G_adv * self.config.get('adversarial_weight', 1.0)
            
            # Pixel losses (normalized by mask area)
            hole_loss = self.l1(pred_img * masks, images * masks) / (masks.mean() + 1e-8)
            valid_loss = self.l1(pred_img * (1 - masks), images * (1 - masks)) / ((1 - masks).mean() + 1e-8)
            
            loss_G = loss_G + hole_loss * self.config.get('hole_weight', 6.0)
            loss_G = loss_G + valid_loss * self.config.get('valid_weight', 1.0)
            
            # Pyramid loss (normalized by number of scales)
            pyramid_loss = torch.tensor(0.0, device=self.device)
            if feats is not None and len(feats) > 0 and self.config.get('pyramid_weight', 0.0) > 0.0:
                for f in feats:
                    target = F.interpolate(
                        images, 
                        size=f.size()[2:4], 
                        mode='bilinear', 
                        align_corners=True
                    )
                    pyramid_loss += self.l1(f, target)
                pyramid_loss = pyramid_loss / len(feats)  # NORMALIZE BY NUMBER OF SCALES
                loss_G = loss_G + pyramid_loss * self.config.get('pyramid_weight', 0.5)
            
            loss_G.backward()
            self.optimG.step()

            # Track losses
            self.loss_history['iters'].append(self.iters)
            self.loss_history['loss_D'].append(loss_D.item())
            self.loss_history['loss_D_real'].append(loss_D_real.item())
            self.loss_history['loss_D_fake'].append(loss_D_fake.item())
            self.loss_history['loss_G'].append(loss_G.item())
            self.loss_history['loss_G_adv'].append(loss_G_adv.item())
            self.loss_history['hole_loss'].append(hole_loss.item())
            self.loss_history['valid_loss'].append(valid_loss.item())
            self.loss_history['pyramid_loss'].append(pyramid_loss.item())

            # Logging
            if self.iters % 100 == 0:
                print(f"[iter {self.iters}/{self.max_iters}] "
                      f"D: {loss_D.item():.4f} G: {loss_G.item():.4f} "
                      f"adv: {loss_G_adv.item():.4f} hole: {hole_loss.item():.4f} "
                      f"valid: {valid_loss.item():.4f} pyr: {pyramid_loss.item():.4f}\n")

            if loss_G < self.best_loss:
                self.best_loss = loss_G.item()
                # Save best model
                pathG = f"{self.config.get('save_dir', './')}/gen_best.pth"
                pathD = f"{self.config.get('save_dir', './')}/disc_best.pth"
                self.save_models(pathG, pathD)
                print(f"New best model saved at iter {self.iters} with G loss {loss_G.item():.4f}\n")
            
            # Real-time visualization update
            if self.use_realtime_viz and self.iters % self.config.get('viz_update_freq', 50) == 0:
                self.update_realtime_viz(images, masks, pred_img, comp_img, images_masked)

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

    def plot_metrics(self, filepath=None):
        """Plot training metrics."""
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

    def save_models(self, pathG, pathD):
        torch.save(self.netG.state_dict(), pathG)
        torch.save(self.netD.state_dict(), pathD)

    def load_models(self, state_dict_G = None, state_dict_D= None):
        if state_dict_G is not None:
            self.netG.load_state_dict_to_model(state_dict_G)
        if state_dict_D is not None:
            self.netD.load_state_dict(state_dict_D)
        if state_dict_G or state_dict_D:
            print("Models loaded.")


    def train(self):
        try:
            while self.iters < self.max_iters:
                self.train_epoch()

                pathG = f"{self.config.get('save_dir', './')}/gen_iter_{self.iters}.pth"
                pathD = f"{self.config.get('save_dir', './')}/disc_iter_{self.iters}.pth"
                self.save_models(pathG, pathD)
                print(f"Saved models at iteration {self.iters}")

                save_dir = self.config.get('save_dir', './')
                metrics_csv = f"{save_dir}/training_metrics.csv"
                self.save_metrics(metrics_csv)
            
            # Final save
            pathG = f"{self.config.get('save_dir', './')}/gen_final.pth"
            pathD = f"{self.config.get('save_dir', './')}/disc_final.pth"
            self.save_models(pathG, pathD)
            
            save_dir = self.config.get('save_dir', './')
            metrics_plot = f"{save_dir}/training_curves.png"
            self.plot_metrics(metrics_plot)
            print("Training finished")
        finally:
            # Always close visualization window if it was opened
            print("Je devrais gérer un cas ici")
