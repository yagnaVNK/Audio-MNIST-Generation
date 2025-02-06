import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.distributions import RelaxedOneHotCategorical, Categorical
import numpy as np
from torch.optim.lr_scheduler import _LRScheduler
import math

def mish(x):
    return x * torch.tanh(F.softplus(x))

class Mish(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return mish(x)

class Upsample(nn.Module):
    def __init__(self, scale_factor=2):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor)
    

class ResBlock(nn.Module):
    def __init__(self, in_channel, channel):
        super().__init__()
        self.conv_1 = nn.Conv2d(in_channel, channel, kernel_size=3, padding=1)
        self.conv_2 = nn.Conv2d(channel, in_channel, kernel_size=3, padding=1)

    def forward(self, inp):
        x = self.conv_1(inp)
        x = mish(x)
        x = self.conv_2(x)
        x = x + inp
        return mish(x)
    

class FlatCA(_LRScheduler):
    def __init__(self, optimizer, steps, eta_min=0, last_epoch=-1):
        self.steps = steps
        self.eta_min = eta_min
        super(FlatCA, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        lr_list = []
        T_max = self.steps / 3
        for base_lr in self.base_lrs:
            # flat if first 2/3
            if 0 <= self._step_count < 2 * T_max:
                lr_list.append(base_lr)
            # annealed if last 1/3
            else:
                lr_list.append(
                    self.eta_min
                    + (base_lr - self.eta_min)
                    * (1 + math.cos(math.pi * (self._step_count - 2 * T_max) / T_max))
                    / 2
                )
            return lr_list
    


class Encoder2D(nn.Module):
    """ Downsamples by a fac of 2 """

    def __init__(self, in_feat_dim, codebook_dim, hidden_dim=128, num_res_blocks=0, batch_norm=1):
        super().__init__()
        blocks = [
            nn.Conv2d(in_feat_dim, hidden_dim // 2, kernel_size=3, stride=2, padding=1),
            Mish(),
            nn.Conv2d(hidden_dim // 2, hidden_dim, kernel_size=3, padding=1),
            Mish(),
        ]

        for _ in range(num_res_blocks):
            blocks.append(ResBlock(hidden_dim, hidden_dim // 2))
            if batch_norm == 2:
                blocks.append(nn.BatchNorm2d(hidden_dim))

        blocks.append(nn.Conv2d(hidden_dim, codebook_dim, kernel_size=1))
        if(batch_norm):
            blocks.append(nn.BatchNorm2d(codebook_dim))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        x = x.float()
        return self.blocks(x)
    


class Decoder2D(nn.Module):
    def __init__(
        self, in_feat_dim, out_feat_dim, hidden_dim=128, num_res_blocks=0, very_bottom=False,
    ):
        super().__init__()
        self.very_bottom = very_bottom
        self.out_feat_dim = out_feat_dim # num channels on bottom layer
        blocks = [nn.Conv2d(in_feat_dim, hidden_dim, kernel_size=3, padding=1), Mish()]
        for _ in range(num_res_blocks):
            blocks.append(ResBlock(hidden_dim, hidden_dim // 2))

        blocks.extend([
                Upsample(),
                nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
                Mish(),
                nn.Conv2d(hidden_dim // 2, out_feat_dim, kernel_size=3, padding=1),
        ])
 
        if very_bottom is True:
            blocks.append(nn.Tanh())       
        self.blocks = nn.Sequential(*blocks)
        
    def forward(self, x):
        x = x.float()
        return self.blocks(x)

class VQVAE(pl.LightningModule):
    def __init__(
        self,
        in_channels=2,
        hidden_dim=128,
        num_res_blocks=2,
        codebook_dim=32,
        codebook_slots=512,
        KL_coeff=0.001,
        CL_coeff=0.1,
        reset_threshold=0.03
    ):
        super().__init__()
        self.save_hyperparameters()
        self.Encoder = Encoder2D(
            in_feat_dim=in_channels,
            codebook_dim=codebook_dim,
            hidden_dim=hidden_dim,
            num_res_blocks=num_res_blocks
        )
        
        self.Decoder = Decoder2D(
            in_feat_dim=codebook_dim,
            out_feat_dim=in_channels,
            hidden_dim=hidden_dim,
            num_res_blocks=num_res_blocks
        )


    def forward(self, x):
        x=x.float()
        ze = self.Encoder(x)
        x_recon = self.Decoder(ze)
        return x_recon, ze

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x_recon, ze = self(x)
        
        dims = np.prod(x.shape[1:])
        recon_loss = F.mse_loss(x_recon, x, reduction='none').sum(dim=(1,2,3)).mean()
        cos_loss = (1 - F.cosine_similarity(x_recon, x, dim=1)).mean()/dims
        recon_loss = recon_loss / dims
        loss = recon_loss #+ cos_loss 
        
        # Log losses
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_recon_loss', recon_loss, prog_bar=True)
        self.log('train_cosine_loss', cos_loss, prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        x_recon, ze  = self(x)
        
        # Calculate losses normalized by dimensions
        dims = np.prod(x.shape[1:])
        recon_loss = F.mse_loss(x_recon, x, reduction='none').sum(dim=(1,2,3)).mean()
        cos_loss = (1 - F.cosine_similarity(x_recon, x, dim=1)).mean()/dims
        recon_loss = recon_loss / dims

        # Total loss
        loss = recon_loss #+ cos_loss 
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_recon_loss', recon_loss, prog_bar=True)
        self.log('val_cosine_loss', cos_loss, prog_bar=True)
        
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=4e-4)
        lr_scheduler = FlatCA(optimizer, steps=1, eta_min=4e-5)
        return [optimizer], [lr_scheduler]

    def reconstruct(self, x):
        with torch.no_grad():
            ze = self.Encoder(x)
            x_recon = self.Decoder(ze)
            if x_recon.shape != x.shape:
                x_recon = F.interpolate(
                    x_recon,
                    size=(x.shape[2], x.shape[3]),
                    mode='bilinear',
                    align_corners=False
                )
            return x_recon

if __name__ == "__main__":
    # Create sample input
    batch_size = 32
    input_channels = 2
    freq_dim = 128
    time_dim = 160
    x = torch.randn(batch_size, input_channels, freq_dim, time_dim)
    
    # Create model
    model = VQVAE(
        in_channels=1,
        hidden_dim=128,
        num_res_blocks=2,
        codebook_dim=32,
        codebook_slots=512
    )
    
    # Forward pass
    with torch.no_grad():
        x_recon, _, _, _, _, _ = model(x)
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {x_recon.shape}")
        assert x.shape == x_recon.shape, "Input and output shapes don't match!"