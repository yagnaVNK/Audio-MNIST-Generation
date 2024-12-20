import os
import torch
import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from pytorch_lightning import seed_everything

from src.Dataset import AudioDataset
from src.VQVAE import VQVAE

def mel_to_audio(mel_spectrogram, sr=48000, n_fft=2048, hop_length=512):
    """
    Convert mel spectrogram back to audio using librosa
    
    Args:
        mel_spectrogram (np.ndarray or torch.Tensor): Mel spectrogram 
        sr (int): Sample rate
        n_fft (int): FFT window size
        hop_length (int): Hop length for reconstruction
    
    Returns:
        np.ndarray: Reconstructed audio waveform
    """
    # Ensure input is numpy array and handle tensor input
    if torch.is_tensor(mel_spectrogram):
        mel_spectrogram = mel_spectrogram.cpu().numpy()
    
    # Ensure input is 2D
    if mel_spectrogram.ndim == 1:
        mel_spectrogram = mel_spectrogram.reshape(1, -1)
    
    # Denormalize mel spectrogram
    mel_spectrogram = mel_spectrogram * 80 - 80
    
    # Convert back to power spectrum
    mel_spectrogram = librosa.db_to_power(mel_spectrogram)
    
    # Reconstruct linear spectrogram 
    # Use manual conversion instead of mel_to_stft
    fmin = 20
    fmax = sr // 2
    
    # Compute mel basis matrix
    mel_basis = librosa.filters.mel(
        sr=sr, 
        n_fft=n_fft, 
        n_mels=mel_spectrogram.shape[0], 
        fmin=fmin, 
        fmax=fmax
    )
    
    # Pseudo-inverse to get back linear spectrogram
    S = np.linalg.pinv(mel_basis).dot(mel_spectrogram)
    
    # Ensure non-negative
    S = np.maximum(0, S)
    
    # Griffin-Lim reconstruction
    y = librosa.griffinlim(
        S, 
        n_iter=32, 
        hop_length=hop_length, 
        win_length=n_fft
    )
    
    # Normalize audio
    y = librosa.util.normalize(y)
    
    return y

def compute_reconstruction_metrics(original, reconstructed, sr=48000):
    """
    Compute reconstruction metrics
    
    Args:
        original (np.ndarray): Original audio
        reconstructed (np.ndarray): Reconstructed audio
        sr (int): Sample rate
    
    Returns:
        dict: Reconstruction metrics
    """
    # Trim to equal lengths
    min_length = min(len(original), len(reconstructed))
    original = original[:min_length]
    reconstructed = reconstructed[:min_length]
    
    # Compute Signal-to-Noise Ratio (SNR)
    def snr(orig, recon):
        signal_power = np.mean(orig**2)
        noise_power = np.mean((orig - recon)**2)
        return 10 * np.log10(signal_power / (noise_power + 1e-10))
    
    # Compute spectral similarity
    def spectral_convergence(orig_spec, recon_spec):
        orig_mag = np.abs(librosa.stft(orig_spec))
        recon_mag = np.abs(librosa.stft(recon_spec))
        
        # Avoid division by zero
        norm_orig = np.linalg.norm(orig_mag)
        if norm_orig == 0:
            return float('inf')
        
        return np.linalg.norm(orig_mag - recon_mag) / norm_orig
    
    metrics = {
        'SNR (dB)': snr(original, reconstructed),
        'Spectral Convergence': spectral_convergence(original, reconstructed),
        'Mean Absolute Error': np.mean(np.abs(original - reconstructed)),
        'Root Mean Square Error': np.sqrt(np.mean((original - reconstructed)**2))
    }
    
    return metrics

def evaluate_vqvae(args):
    """
    Comprehensive VQVAE evaluation script
    
    Args:
        args: Argument parser containing model and evaluation parameters
    """
    # Set random seed for reproducibility
    seed_everything(42)
    
    # Create results directory
    results_dir = os.path.join("results", "vqvae_evaluation")
    os.makedirs(results_dir, exist_ok=True)
    samples_dir = os.path.join(results_dir, "samples")
    os.makedirs(samples_dir, exist_ok=True)
    
    # Load dataset
    dataset = AudioDataset(
        root_dir=args.data_dir, 
        target_length=args.target_length
    )
    
    # Create test dataloader
    test_loader = DataLoader(
        dataset, 
        batch_size=10, 
        shuffle=True, 
        num_workers=args.num_workers
    )
    
    # Load trained model
    model = torch.load(
        os.path.join("results", 'vqvae_run_20241216_133830', 'final_model.pt')
    )
    model.eval()
    model = model.to('cuda')
    
    # Tracking variables
    all_metrics = []
    
    # Evaluation loop
    with torch.no_grad():
        for batch_idx, (spectrograms, labels) in enumerate(test_loader):
            if batch_idx >= 10:  # Limit to first 10 batches
                break
            
            # Prepare input
            spectrograms = spectrograms.unsqueeze(1).to('cuda')
            
            # Reconstruct
            reconstructed = model.reconstruct(spectrograms)
            
            # Process and save samples
            for i in range(min(5, len(spectrograms))):
                # Get original and reconstructed spectrograms
                orig_spec = spectrograms[i].squeeze()
                recon_spec = reconstructed[i].squeeze()
                
                # Get the label for this sample
                label = labels[i].item()
                
                # Plot spectrograms
                plt.figure(figsize=(15, 5))
                plt.subplot(121)
                plt.title(f'Original Mel Spectrogram (Label: {label})')
                plt.imshow(orig_spec.cpu().numpy(), aspect='auto', origin='lower')
                plt.colorbar()
                
                plt.subplot(122)
                plt.title(f'Reconstructed Mel Spectrogram (Label: {label})')
                plt.imshow(recon_spec.cpu().numpy(), aspect='auto', origin='lower')
                plt.colorbar()
                
                plt.tight_layout()
                plt.savefig(os.path.join(samples_dir, f'1stspectrogram_label_{label}.png'))
                plt.close()
                
                # Convert to audio
                orig_audio = mel_to_audio(orig_spec)
                recon_audio = mel_to_audio(recon_spec)
                
                # Save audio files with label
                sf.write(
                    os.path.join(samples_dir, f'1stOriginal_label{label}.wav'),
                    orig_audio,
                    args.sample_rate
                )
                sf.write(
                    os.path.join(samples_dir, f'1stReconstruction_label{label}.wav'),
                    recon_audio,
                    args.sample_rate
                )
                
                # Compute and store metrics
                metrics = compute_reconstruction_metrics(orig_audio, recon_audio)
                all_metrics.append(metrics)
        
        # Compute average metrics
        avg_metrics = {
            key: np.mean([m[key] for m in all_metrics]) 
            for key in all_metrics[0].keys()
        }
        
        # Save metrics
        with open(os.path.join(results_dir, 'reconstruction_metrics.txt'), 'w') as f:
            f.write("Reconstruction Metrics:\n")
            for key, value in avg_metrics.items():
                f.write(f"{key}: {value}\n")
                print(f"{key}: {value}")

if __name__ == "__main__":
    from argparse import ArgumentParser
    
    parser = ArgumentParser(description="VQVAE Evaluation Script")
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, default='../Data',
                        help='Directory containing the dataset')
    parser.add_argument('--target_length', type=int, default=47998,
                        help='Target length for audio waveforms')
    parser.add_argument('--sample_rate', type=int, default=48000,
                        help='Audio sample rate')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading')
    
    args = parser.parse_args()
    
    evaluate_vqvae(args)