import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from generative.networks.nets import VQVAE as VQVAE_MONAI

class VQVAE(pl.LightningModule):
    def __init__(
        self,
        in_channels=2,
        out_channels=2,
        num_res_layers=2,
        downsample_parameters=((2, 4, 1, 1), (2, 4, 1, 1)),
        upsample_parameters=((2, 4, 1, 1, 0), (2, 4, 1, 1, 0)),
        num_channels=(256, 256),
        num_res_channels=(256, 256),
        num_embeddings=256,
        embedding_dim=32,
        learning_rate=1e-4,
        weight_decay=1e-5
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Initialize VQVAE model
        self.model = VQVAE_MONAI(
            spatial_dims=2,
            in_channels=in_channels,
            out_channels=out_channels,
            num_res_layers=num_res_layers,
            downsample_parameters=downsample_parameters,
            upsample_parameters=upsample_parameters,
            num_channels=num_channels,
            num_res_channels=num_res_channels,
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim
        )
        
    def forward(self, x):
        return self.model(x)
    
    
    def _get_reconstruction_loss(self, batch):
        x, _ = batch  # Ignore labels
        reconstruction, quantization_loss = self(x)
        recons_loss = F.l1_loss(reconstruction, x)
        total_loss = recons_loss + quantization_loss
        
        return {
            'reconstruction_loss': recons_loss,
            'quantization_loss': quantization_loss,
            'total_loss': total_loss
        }
    
    def training_step(self, batch, batch_idx):
        loss_dict = self._get_reconstruction_loss(batch)
        self.log('train_total_loss', loss_dict['total_loss'])
        self.log('train_reconstruction_loss', loss_dict['reconstruction_loss'])
        self.log('train_quantization_loss', loss_dict['quantization_loss'])
        return loss_dict['total_loss']
    
    def validation_step(self, batch, batch_idx):
        loss_dict = self._get_reconstruction_loss(batch)
        self.log('val_total_loss', loss_dict['total_loss'])
        self.log('val_reconstruction_loss', loss_dict['reconstruction_loss'])
        self.log('val_quantization_loss', loss_dict['quantization_loss'])
        return loss_dict['total_loss']
    
    def test_step(self, batch, batch_idx):
        loss_dict = self._get_reconstruction_loss(batch)
        self.log('test_total_loss', loss_dict['total_loss'])
        self.log('test_reconstruction_loss', loss_dict['reconstruction_loss'])
        self.log('test_quantization_loss', loss_dict['quantization_loss'])
        return loss_dict['total_loss']
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        return optimizer