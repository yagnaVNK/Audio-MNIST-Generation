import os
import torch
import librosa
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as T
from pathlib import Path
#from src.VQVAE import VQVAE
from src.VQVAE_monai import VQVAE
from src.Dataset import AudioMNIST
from src.custom_transforms import TrimSilence, FixLength
import scipy

class VQVAETrainer:
    def __init__(
        self,
        data_dir,
        batch_size=32,
        num_workers=4,
        max_epochs=20,
        accelerator='auto',
        devices=[0,1]
    ):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_epochs = max_epochs
        self.accelerator = accelerator
        self.devices = devices
        
        # Setup data
        self.setup_data()
        
        # Initialize model
        self.model = VQVAE()
        
        # Setup logging and checkpoints
        self.setup_logging()
    
    def setup_data(self):
        # Create dataset with transforms
        transforms = [TrimSilence(5), FixLength(16000)]
        dataset = AudioMNIST(
            self.data_dir,
            transform=T.Compose(transforms)
        )
        
        # Split dataset
        train_size = int(0.8 * len(dataset))
        val_size = int(0.10 * len(dataset))
        test_size = len(dataset) - train_size - val_size
        
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Create dataloaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True
        )
        
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True
        )
    
    def setup_logging(self):
        # Create directories
        self.log_dir = Path('lightning_logs')
        self.log_dir.mkdir(exist_ok=True)
        
        # Setup logger
        self.logger = TensorBoardLogger(
            save_dir=str(self.log_dir),
            name='vqvae_logs'
        )
        
        # Setup checkpoint callback
        self.checkpoint_callback = ModelCheckpoint(
            dirpath=str(self.log_dir / 'checkpoints'),
            filename='vqvae-{epoch:02d}-{val_total_loss:.2f}',
            monitor='val_total_loss',
            mode='min',
            save_top_k=3
        )
    
    def train(self):
        # Initialize trainer
        trainer = pl.Trainer(
            max_epochs=self.max_epochs,
            accelerator=self.accelerator,
            devices=self.devices,
            logger=self.logger,
        )
        
        # Train model
        trainer.fit(
            self.model,
            train_dataloaders=self.train_loader,
            val_dataloaders=self.val_loader
        )
        if not os.path.exists('saved_models'):
            os.makedirs('saved_models')

        
        torch.save(self.model, 'saved_models/vqvae_monai.pth')

        #self.model = torch.load('saved_models/vqvae.pth')

        # Test model
        trainer.test(
            self.model,
            dataloaders=self.test_loader
        )
    
    def evaluate(self, num_samples=10):
        """Evaluate model and generate audio samples"""
        # Create results directory
        results_dir = Path('VQVAE_results')
        results_dir.mkdir(exist_ok=True)
        
        # Set model to eval mode
        self.model.eval()
        
        # Create figure for spectrograms
        plt.figure(figsize=(20, 8))
        
        for i in range(num_samples):
            # Get sample
            sample, _ = self.test_dataset[i]
            
            # Get original spectrogram
            original_complex = self.test_dataset.dataset.twod_to_complex(sample)
            original_db = librosa.amplitude_to_db(
                np.abs(original_complex),
                ref=np.max
            )
            
            # Get reconstruction
            sample_batch = sample.unsqueeze(0)
            with torch.no_grad():
                sample_recon = self.model(sample_batch)[0].cpu().numpy()
            
            recon_complex = self.test_dataset.dataset.twod_to_complex(
                sample_recon[0]
            )
            recon_db = librosa.amplitude_to_db(
                np.abs(recon_complex),
                ref=np.max
            )
            
            # Plot spectrograms
            plt.subplot(2, num_samples, i + 1)
            librosa.display.specshow(original_db[0])
            plt.title(f'Original {i+1}')
            if i == 0:
                plt.ylabel('Original')
            
            plt.subplot(2, num_samples, num_samples + i + 1)
            librosa.display.specshow(recon_db[0])
            if i == 0:
                plt.ylabel('Reconstructed')
            plt.title(f'Reconstructed {i+1}')
            
            # Generate and save audio files
            _, original_audio = scipy.signal.istft(
                original_complex,
                self.test_dataset.dataset.target_sample_rate
            )
            _, recon_audio = scipy.signal.istft(
                recon_complex,
                self.test_dataset.dataset.target_sample_rate
            )
            
            # Remove extra dimensions
            original_audio = np.squeeze(original_audio)
            recon_audio = np.squeeze(recon_audio)
            
            # Save audio files
            sf.write(
                results_dir / f'original_{i}.wav',
                original_audio,
                self.test_dataset.dataset.target_sample_rate
            )
            sf.write(
                results_dir / f'reconstruction_{i}.wav',
                recon_audio,
                self.test_dataset.dataset.target_sample_rate
            )
        
        # Save spectrogram comparison
        plt.tight_layout()
        plt.savefig(results_dir / 'spectrograms_comparison.png')
        plt.close()

def main():
    # Initialize trainer
    trainer = VQVAETrainer(
        data_dir='../Data',
        batch_size=32,
        num_workers=4,
        max_epochs=200
    )
    
    # Train model
    trainer.train()
    
    # Evaluate model
    trainer.evaluate(num_samples=10)

if __name__ == "__main__":
    main()