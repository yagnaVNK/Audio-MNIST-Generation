import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import random_split, DataLoader
import matplotlib.pyplot as plt
import soundfile as sf
import librosa
import numpy as np
from datetime import datetime
from pathlib import Path
from argparse import ArgumentParser

from src.Dataset import AudioDataset
from src.VQVAE import VQVAE

def plot_spectrograms(original, reconstructed, save_path):
    plt.figure(figsize=(15, 5))
    
    plt.subplot(121)
    plt.imshow(original.cpu().numpy(), aspect='auto', origin='lower')
    plt.title('Original Mel Spectrogram')
    plt.colorbar()
    
    plt.subplot(122)
    plt.imshow(reconstructed.cpu().numpy(), aspect='auto', origin='lower')
    plt.title('Reconstructed Mel Spectrogram')
    plt.colorbar()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def mel_to_audio(mel_spectrogram, sr=48000, n_fft=2048, hop_length=512):
    mel_spectrogram = mel_spectrogram.cpu().numpy()
    S = librosa.feature.inverse_melspectrogram(mel_spectrogram)
    y = librosa.griffinlim(
        S,
        n_iter=32,
        hop_length=hop_length,
        win_length=n_fft
    )
    return y

def main(args):
    # Set float32 matmul precision for better performance
    torch.set_float32_matmul_precision('medium')
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join("results", f"vqvae_run_{timestamp}")
    samples_dir = os.path.join(results_dir, "samples")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(samples_dir, exist_ok=True)
    
    # Initialize dataset
    dataset = AudioDataset(
        root_dir=args.data_dir,
        target_length=args.target_length
    )
    
    # Split dataset
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        persistent_workers=True if args.num_workers > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=True if args.num_workers > 0 else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=True if args.num_workers > 0 else False
    )
    
    # Initialize model
    model = VQVAE(
        in_channels=1,
        hidden_dim=args.hidden_dim,
        num_res_blocks=args.num_res_blocks,
        codebook_dim=args.codebook_dim,
        codebook_slots=args.codebook_slots,
        KL_coeff=args.kl_coeff,
        CL_coeff=args.cl_coeff
    )
    
    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(results_dir, 'checkpoints'),
        filename='vqvae-{epoch:02d}-{val_loss:.2f}',
        monitor='val_loss',
        mode='min',
        save_top_k=3
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        mode='min'
    )
    
    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator='auto',
        devices=1,
        callbacks=[checkpoint_callback, early_stopping],
        gradient_clip_val=args.grad_clip
    )
    
    # Train model
    trainer.fit(model, train_loader, val_loader)
    
    # Save final model
    torch.save(model, os.path.join(results_dir, 'final_model.pt'))
    model = model.to(device='cuda')

if __name__ == "__main__":
    parser = ArgumentParser()
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, default='../Data',
                        help='Directory containing the dataset')
    parser.add_argument('--target_length', type=int, default=47998,
                        help='Target length for audio waveforms')
    parser.add_argument('--sample_rate', type=int, default=48000,
                        help='Audio sample rate')
    
    # Model parameters
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='Hidden dimension size')
    parser.add_argument('--num_res_blocks', type=int, default=2,
                        help='Number of residual blocks')
    parser.add_argument('--codebook_dim', type=int, default=32,
                        help='Dimension of codebook vectors')
    parser.add_argument('--codebook_slots', type=int, default=512,
                        help='Number of codebook entries')
    
    # Loss coefficients
    parser.add_argument('--kl_coeff', type=float, default=0.001,
                        help='KL loss coefficient')
    parser.add_argument('--cl_coeff', type=float, default=0.1,
                        help='Commitment loss coefficient')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_epochs', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    
    args = parser.parse_args()
    
    main(args)