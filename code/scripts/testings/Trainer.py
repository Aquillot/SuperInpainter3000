import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import csv
import os
from model import UNet, InpaintGenerator, Discriminator
from AdversarialLoss import AdversarialLoss

class Trainer:
    def __init__(self, config, dataset):
        """
        config: dict contenant au minimum:
          - lr, beta1, beta2, batch_size, iterations,
          - adversarial_weight, hole_weight, valid_weight, pyramid_weight,
          - device (e.g. 'cuda' or 'cpu'), save_dir (optional)
        dataset: PyTorch Dataset yielding (image, mask, ...)
        """
        self.device = torch.device(config.get('device','cuda' if torch.cuda.is_available() else 'cpu'))
        self.config = config
        self.dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config.get('num_workers',4), pin_memory=True)
        # models
        unet = UNet(n_class=3)           # ton UNet défini ailleurs
        self.netG = InpaintGenerator(unet).to(self.device)
        self.netD = Discriminator(in_channels=3, use_sn=True).to(self.device)
        # losses & optimizers
        self.adv_loss = AdversarialLoss().to(self.device)
        self.l1 = nn.L1Loss()
        self.optimG = torch.optim.Adam(self.netG.parameters(), lr=config['lr'], betas=(config.get('beta1',0.5), config.get('beta2',0.999)))
        self.optimD = torch.optim.Adam(self.netD.parameters(), lr=config['lr']*config.get('d2glr',1.0), betas=(config.get('beta1',0.5), config.get('beta2',0.999)))
        self.iters = 0
        self.max_iters = config['iterations']
        
        # loss tracking for visualization
        self.loss_history = {
            'iters': [],
            'loss_D': [],
            'loss_D_real': [],
            'loss_D_fake': [],
            'loss_G': [],
            'loss_G_adv': [],
            'hole_loss': [],
            'valid_loss': []
        }

    def draw_line(self,mask_channel, x1, y1, x2, y2, thickness):
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


    def create_mask(self,batch_size, H, W, device):
        mask = torch.ones((batch_size, 1, H, W), device=device)

        for b in range(batch_size):
            n_lines = torch.randint(1, 3, (1,)).item()
            for _ in range(n_lines):
                x1 = torch.randint(0, W, (1,)).item()
                y1 = torch.randint(0, H, (1,)).item()
                x2 = torch.randint(0, W, (1,)).item()
                y2 = torch.randint(0, H, (1,)).item()

                thickness = torch.randint(3, 8, (1,)).item()
                self.draw_line(mask[b, 0], x1, y1, x2, y2, thickness)

        return mask

    def train_epoch(self):
        for images, *_ in self.dataloader:
            if self.iters >= self.max_iters:
                break
            self.iters += 1
            images = images.to(self.device)            # shape (N,3,H,W), expected in range consistent with loss
            B, C, H, W = images.shape
            # create_mask retourne un tensor (B,1,H,W) avec 1 = hole
            masks = self.create_mask(B, H, W, device=self.device)
            masks = masks.float()                  # ensure float type
            masks = 1.0 - masks  # now 1=hole, 0=valid

            # --- detect image value range to pick fill_value ---
            img_min, img_max = images.min().item(), images.max().item()
            # if images in [0,1], use 1.0 or 0.0; if in [-1,1], use -1.0
            if img_min >= 0.0:
                fill_value = 1.0  # or 0.0 if you prefer black holes
            else:
                fill_value = -1.0

            # build input: masked image + mask channel
            images_masked = images * (1 - masks) + fill_value * masks # fill holes
            inputs = torch.cat([images_masked, masks], dim=1)  # (N,4,H,W)

            # ----- Forward G -----
            feats, pred_img = self.netG(inputs, masks)         # pred_img: (N,3,H,W)
            comp_img = images * (1 - masks) + pred_img * masks # composite image for D

            # ----- Train D -----
            self.optimD.zero_grad()
            real_score = self.netD(images)                      # (N,1,h,w)
            fake_score = self.netD(comp_img.detach())           # detach -> no grad to G
            loss_D_real = self.adv_loss(real_score, True, True)
            loss_D_fake = self.adv_loss(fake_score, False, True)
            loss_D = 0.5 * (loss_D_real + loss_D_fake)
            loss_D.backward()
            self.optimD.step()

            # ----- Train G -----
            self.optimG.zero_grad()
            fake_score_for_G = self.netD(comp_img)              # NO detach -> gradients flow to G
            loss_G_adv = self.adv_loss(fake_score_for_G, True, False)
            loss_G = loss_G_adv * self.config.get('adversarial_weight',1.0)
            # pixel losses
            hole_loss = self.l1(pred_img * masks, images * masks) / (masks.mean() + 1e-6)
            valid_loss = self.l1(pred_img * (1 - masks), images * (1 - masks)) / ((1 - masks).mean() + 1e-6)
            loss_G = loss_G + hole_loss * self.config.get('hole_weight',1.0) + valid_loss * self.config.get('valid_weight',1.0)
            # pyramid loss omitted here (feats is None). If you implement feats, add it.
            loss_G.backward()
            self.optimG.step()

            # track losses
            self.loss_history['iters'].append(self.iters)
            self.loss_history['loss_D'].append(loss_D.item())
            self.loss_history['loss_D_real'].append(loss_D_real.item())
            self.loss_history['loss_D_fake'].append(loss_D_fake.item())
            self.loss_history['loss_G'].append(loss_G.item())
            self.loss_history['loss_G_adv'].append(loss_G_adv.item())
            self.loss_history['hole_loss'].append(hole_loss.item())
            self.loss_history['valid_loss'].append(valid_loss.item())

            # minimal logging
            if self.iters % 100 == 0:
                print(f"[iter {self.iters}/{self.max_iters}] D_loss: {loss_D.item():.4f} G_loss: {loss_G.item():.4f} adv: {loss_G_adv.item():.4f}")

    def save_metrics(self, filepath):
        """Save training metrics (losses) to a CSV file."""
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            # Write header
            header = list(self.loss_history.keys())
            writer.writerow(header)
            # Write rows
            num_rows = len(self.loss_history['iters'])
            for i in range(num_rows):
                row = [self.loss_history[key][i] for key in header]
                writer.writerow(row)
        print(f"Saved metrics to {filepath}")

    def plot_metrics(self, filepath=None):
        """Plot training metrics (losses) and optionally save to file.
        
        Args:
            filepath: if provided, save the plot to this path (e.g., 'training_curves.png')
        """
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
        axes[1, 1].set_xlabel('Iteration')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].set_title('Pixel-wise Losses')
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


    def train(self):
        while self.iters < self.max_iters:
            self.train_epoch()
            
            pathG = f"{self.config.get('save_dir','./')}/gen_iter_{self.iters}.pth"
            pathD = f"{self.config.get('save_dir','./')}/disc_iter_{self.iters}.pth"
            self.save_models(pathG, pathD)
            print(f"Saved models at iteration {self.iters} to {pathG} and {pathD}")

            save_dir = self.config.get('save_dir', './')
            metrics_csv = f"{save_dir}/training_metrics.csv"
            metrics_plot = f"{save_dir}/training_curves.png"
            self.save_metrics(metrics_csv)
            self.plot_metrics(metrics_plot)
        print("Training finished")
