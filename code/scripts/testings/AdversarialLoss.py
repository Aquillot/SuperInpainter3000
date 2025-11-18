
import torch.nn as nn


# ---------- Hinge adversarial loss (essentiel) ----------
class AdversarialLoss(nn.Module):
    """Hinge loss implementation for PatchGAN outputs (scores)."""
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()

    def forward(self, outputs, is_real: bool, is_disc: bool):
        # outputs: logits (N,1,H,W) or (N,...)
        if is_disc:
            if is_real:
                out = -outputs
            else:
                out = outputs
            return self.relu(1.0 + out).mean()
        else:
            # generator: maximize D(fake) -> minimize -D(fake)
            return (-outputs).mean()