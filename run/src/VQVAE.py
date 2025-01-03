import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.distributions import RelaxedOneHotCategorical, Categorical
import numpy as np

class ResBlock2D(nn.Module):
    def __init__(self, in_channel, channel):
        super().__init__()
        self.conv_1 = nn.Conv2d(in_channel, channel, kernel_size=3, padding=1)
        self.bn_1 = nn.BatchNorm2d(channel)
        self.conv_2 = nn.Conv2d(channel, in_channel, kernel_size=3, padding=1)
        self.bn_2 = nn.BatchNorm2d(in_channel)
        self.activation = nn.ReLU()

    def forward(self, inp):
        x = self.conv_1(inp)
        x = self.bn_1(x)
        x = self.activation(x)
        x = self.conv_2(x)
        x = self.bn_2(x)
        x = x + inp
        return self.activation(x)

class Encoder2D(nn.Module):
    def __init__(self, in_feat_dim, codebook_dim, hidden_dim=128, num_res_blocks=0):
        super().__init__()
        self.blocks = nn.Sequential(
            nn.Conv2d(in_feat_dim, hidden_dim // 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(),
            nn.Conv2d(hidden_dim // 2, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
        )
        
        res_blocks = []
        for _ in range(num_res_blocks):
            res_blocks.extend([
                ResBlock2D(hidden_dim, hidden_dim // 2),
                nn.BatchNorm2d(hidden_dim)
            ])
        self.res_blocks = nn.Sequential(*res_blocks)
        
        self.final = nn.Sequential(
            nn.Conv2d(hidden_dim, codebook_dim, kernel_size=1),
            nn.BatchNorm2d(codebook_dim)
        )

    def forward(self, x):
        x = self.blocks(x)
        x = self.res_blocks(x)
        x = self.final(x)
        return x

class Decoder2D(nn.Module):
    def __init__(self, in_feat_dim, out_feat_dim, hidden_dim=128, num_res_blocks=0):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(in_feat_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU()
        )
        
        res_blocks = []
        for _ in range(num_res_blocks):
            res_blocks.extend([
                ResBlock2D(hidden_dim, hidden_dim // 2),
                nn.BatchNorm2d(hidden_dim)
            ])
        self.res_blocks = nn.Sequential(*res_blocks)

        self.upsample_1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU()
        )
        
        self.upsample_2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(hidden_dim // 2, out_feat_dim, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.initial(x)
        x = self.res_blocks(x)
        x = self.upsample_1(x)
        x = self.upsample_2(x)
        return x

class VQCodebook(nn.Module):
    def __init__(self, codebook_dim=16, codebook_slots=32, temperature=0.5):
        super().__init__()
        self.codebook_dim = codebook_dim
        self.codebook_slots = codebook_slots
        self.codebook = nn.Embedding(self.codebook_slots, self.codebook_dim)
        self.temperature = temperature
        self.log_slots_const = np.log(self.codebook_slots)

    def ze_to_zq(self, ze, soft=True):
        bs, feat_dim, h, w = ze.shape
        assert feat_dim == self.codebook_dim
        
        ze = ze.permute(0, 2, 3, 1).contiguous()
        z_e_flat = ze.view(-1, feat_dim)
        
        codebook = self.codebook.weight
        codebook_sqr = torch.sum(codebook ** 2, dim=1)
        z_e_flat_sqr = torch.sum(z_e_flat ** 2, dim=1, keepdim=True)
        
        distances = torch.addmm(
            codebook_sqr + z_e_flat_sqr,
            z_e_flat,
            codebook.t(),
            alpha=-2.0,
            beta=1.0
        )
        
        if soft:
            dist = RelaxedOneHotCategorical(self.temperature, logits=-distances)
            soft_onehot = dist.rsample()
            hard_indices = torch.argmax(soft_onehot, dim=1).view(bs, h, w)
            z_q = (soft_onehot @ codebook).view(bs, h, w, feat_dim)
            
            KL = dist.probs * (dist.probs.add(1e-9).log() + self.log_slots_const)
            KL = KL.view(bs, h, w, self.codebook_slots).sum(dim=(1,2,3)).mean()
            
            commit_loss = (dist.probs.view(bs, h, w, self.codebook_slots) * 
                         distances.view(bs, h, w, self.codebook_slots)).sum(dim=(1,2,3)).mean()
        else:
            with torch.no_grad():
                dist = Categorical(logits=-distances)
                hard_indices = dist.sample().view(bs, h, w)
                hard_onehot = F.one_hot(hard_indices, num_classes=self.codebook_slots).type_as(codebook)
                z_q = (hard_onehot @ codebook).view(bs, h, w, feat_dim)
                
                KL = dist.probs * (dist.probs.add(1e-9).log() + self.log_slots_const)
                KL = KL.view(bs, h, w, self.codebook_slots).sum(dim=(1,2,3)).mean()
                commit_loss = 0.0

        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        return z_q, hard_indices, KL, commit_loss
    
    def quantize_indices(self, z_e, soft=False):
        with torch.no_grad():
            _, indices, _, _ = self.ze_to_zq(z_e, soft=soft)
        return indices

    def lookup(self, indices):
        z_q = self.codebook(indices)
        return z_q.permute(0, 3, 1, 2).contiguous()

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
        
        self.codebook = VQCodebook(
            codebook_dim=codebook_dim,
            codebook_slots=codebook_slots
        )
        
        self.Decoder = Decoder2D(
            in_feat_dim=codebook_dim,
            out_feat_dim=in_channels,
            hidden_dim=hidden_dim,
            num_res_blocks=num_res_blocks
        )
        
        self.KL_coeff = KL_coeff
        self.CL_coeff = CL_coeff
        self.reset_threshold = reset_threshold
        
    def on_train_start(self):
        # Initialize code usage tracking on CPU
        self.register_buffer('code_count', torch.zeros(self.codebook.codebook_slots, device='cpu'))
        self.codebook_resets = 0
        
    def on_train_batch_start(self, batch, batch_idx):
        # Reset code count periodically
        if batch_idx % 100 == 0:
            self.code_count.zero_()

    @torch.no_grad()
    def reset_least_used_codeword(self):
        """Reset least used codeword to a perturbation of most used codeword."""
        if len(self.code_count) == 0:
            return
            
        max_count, most_used_code = torch.max(self.code_count, dim=0)
        if max_count == 0:
            return
            
        # Calculate usage fractions
        frac_usage = self.code_count / max_count
        min_frac_usage, min_used_code = torch.min(frac_usage, dim=0)
        
        # Reset if usage below threshold
        if min_frac_usage < self.reset_threshold:
            # Get most used codeword
            z_q_most_used = self.codebook.codebook.weight[most_used_code]
            
            # Create perturbed version based on reset count
            reset_factor = 1 / (self.codebook_resets + 1)
            moved_code = z_q_most_used + torch.randn_like(z_q_most_used) * reset_factor
            
            # Update least used codeword
            self.codebook.codebook.weight.data[min_used_code] = moved_code
            self.codebook_resets += 1
            
            #print(f"Reset code {min_used_code.item()} (usage: {min_frac_usage.item():.3f})")

    def forward(self, x):
        ze = self.Encoder(x)
        zq, indices, KL_loss, commit_loss = self.codebook.ze_to_zq(ze, soft=True)
        x_recon = self.Decoder(zq)
        
        # Ensure output dimensions match input
        if x_recon.shape != x.shape:
            x_recon = F.interpolate(
                x_recon,
                size=(x.shape[2], x.shape[3]),
                mode='bilinear',
                align_corners=False
            )
        
        return x_recon, ze, zq, indices, KL_loss, commit_loss

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x = x.unsqueeze(1)
        x_recon, ze, zq, indices, KL_loss, commit_loss = self(x)
        
        # Move indices to CPU for counting
        indices_cpu = indices.cpu()
        unique_indices, counts = torch.unique(indices_cpu, return_counts=True)
        # Update code count on CPU
        self.code_count = self.code_count.cpu()
        self.code_count[unique_indices] += counts.float()
        # Move code count back to device
        self.code_count = self.code_count.to(self.device)
        
        # Check for reset every 25 batches
        if batch_idx % 25 == 0:
            self.reset_least_used_codeword()
        
        # Calculate losses normalized by dimensions
        dims = np.prod(x.shape[1:])
        recon_loss = F.mse_loss(x_recon, x, reduction='none').sum(dim=(1,2,3)).mean()
        recon_loss = recon_loss / dims
        KL_loss = KL_loss / dims
        commit_loss = commit_loss / dims
        
        # Total loss
        loss = recon_loss + self.KL_coeff * KL_loss + self.CL_coeff * commit_loss
        
        # Log losses
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_recon_loss', recon_loss, prog_bar=True)
        self.log('train_kl_loss', KL_loss, prog_bar=True)
        self.log('train_commit_loss', commit_loss, prog_bar=True)
        self.log('codebook_resets', self.codebook_resets, prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        x = x.unsqueeze(1)
        x_recon, ze, zq, indices, KL_loss, commit_loss = self(x)
        
        # Calculate losses normalized by dimensions
        dims = np.prod(x.shape[1:])
        recon_loss = F.mse_loss(x_recon, x, reduction='none').sum(dim=(1,2,3)).mean()
        recon_loss = recon_loss / dims
        KL_loss = KL_loss / dims
        commit_loss = commit_loss / dims
        
        # Total loss
        loss = recon_loss + self.KL_coeff * KL_loss + self.CL_coeff * commit_loss
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_recon_loss', recon_loss, prog_bar=True)
        self.log('val_kl_loss', KL_loss, prog_bar=True)
        self.log('val_commit_loss', commit_loss, prog_bar=True)
        
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }

    def reconstruct(self, x):
        with torch.no_grad():
            ze = self.Encoder(x)
            zq, _, _, _ = self.codebook.ze_to_zq(ze, soft=False)
            x_recon = self.Decoder(zq)
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
    input_channels = 1
    freq_dim = 128
    time_dim = 94
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