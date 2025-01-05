import os
import torch
import numpy as np
import scipy
import soundfile as sf
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from pytorch_lightning import seed_everything
import librosa

from IPython.display import Audio
import src.custom_transforms as CT
import torchvision.transforms as T
from src.Dataset import AudioMNIST


def twod_to_complex(tensor: np.ndarray):
    """Converts two channel representation back to complex signal"""
    tmp_tensor = tensor.reshape(2, tensor.shape[1]*tensor.shape[2])
    new_tensor = np.zeros((1, tmp_tensor.shape[1]), dtype=np.complex64)
    new_tensor[0] = tmp_tensor[0,:] + 1j * tmp_tensor[1,:]
    new_tensor = new_tensor.reshape(1, tensor.shape[1], tensor.shape[2])
    return new_tensor

def spectrogram_to_audio(spectrogram, sample_rate):
    """Convert 2D spectrogram back to audio using inverse STFT"""
    # Convert to complex representation
    Zxx_rec = twod_to_complex(spectrogram)
    
    _, x = scipy.signal.istft(Zxx_rec, sample_rate)
    
    return x.flatten()  

def compute_reconstruction_metrics(original, reconstructed):
    """Compute reconstruction metrics between original and reconstructed audio"""
    # Trim to equal lengths
    min_length = min(len(original), len(reconstructed))
    original = original[:min_length]
    reconstructed = reconstructed[:min_length]
    
    metrics = {
        'MSE': np.mean((original - reconstructed) ** 2),
        'MAE': np.mean(np.abs(original - reconstructed)),
        'Peak SNR': 20 * np.log10(np.max(np.abs(original)) / 
                                np.sqrt(np.mean((original - reconstructed) ** 2)))
    }
    
    return metrics

def evaluate_vqvae(args):
    """Evaluate VQVAE reconstruction quality"""
    # Set random seed
    seed_everything(42)
    
    # Create results directory
    results_dir = os.path.join("results", "vqvae_evaluation")
    os.makedirs(results_dir, exist_ok=True)
    samples_dir = os.path.join(results_dir, "samples")
    os.makedirs(samples_dir, exist_ok=True)
    
    dataset = AudioMNIST(
        root='../Data',
        target_sample_rate=22050,
        transform=T.Compose([
            CT.TrimSilence(5),
            CT.FixLength(22050//4)
        ]),
        output_format='spectrogram',
        spec_params={
            'n_fft': 512,
            'n_freq': 128,
            'n_time': 44,
            'complex_output': False
        }
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
        os.path.join("results", 'vqvae_run_20250104_231557', 'final_model.pt'),
        map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    )
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    
    # Tracking metrics
    all_metrics = []
    
    # Evaluation loop
    with torch.no_grad():
        for batch_idx, (spectrograms, labels) in enumerate(test_loader):
            if batch_idx >= 10:  # Limit to first 10 batches
                break
                
            # Move to GPU if available
            if torch.cuda.is_available():
                spectrograms = spectrograms.cuda()
            
            # Get reconstructions
            reconstructed = model.reconstruct(spectrograms)
            
            # Process each sample in batch
            for i in range(min(5, len(spectrograms))):
                # Get original and reconstructed spectrograms
                orig_spec = spectrograms[i].cpu().numpy()
                recon_spec = reconstructed[i].cpu().numpy()
                label = labels[i].item()
                
                # Plot spectrograms
                # Plot spectrograms
                plt.figure(figsize=(15, 5))
                
                plt.subplot(121)
                plt.title(f'Original Spectrogram (Label: {label})')
                orig_spec_complex = dataset.twod_to_complex(orig_spec)
                librosa.display.specshow(
                    librosa.amplitude_to_db(np.abs(orig_spec_complex[0]), ref=np.max),
                    sr=args.sample_rate,
                    hop_length=512//4,  # Match your STFT parameters
                    x_axis='time',
                    y_axis='hz'
                )
                plt.colorbar(format='%+2.0f dB')
                
                plt.subplot(122)
                plt.title(f'Reconstructed Spectrogram (Label: {label})')
                recon_spec_complex = dataset.twod_to_complex(recon_spec)
                librosa.display.specshow(
                    librosa.amplitude_to_db(np.abs(recon_spec_complex[0]), ref=np.max),
                    sr=args.sample_rate,
                    hop_length=512//4,  # Match your STFT parameters
                    x_axis='time',
                    y_axis='hz'
                )
                plt.colorbar(format='%+2.0f dB')
                
                plt.tight_layout()
                plt.savefig(os.path.join(samples_dir, f'spectrogram_label_{label}_{batch_idx}_{i}.png'))
                plt.close()
                
                # Convert to audio
                orig_audio = spectrogram_to_audio(orig_spec, args.sample_rate)
                recon_audio = spectrogram_to_audio(recon_spec, args.sample_rate)
                
                # Save audio files
                sf.write(
                    os.path.join(samples_dir, f'original_label{label}_{batch_idx}_{i}.wav'),
                    orig_audio,  # Now this is already a numpy array
                    args.sample_rate
                )
                sf.write(
                    os.path.join(samples_dir, f'reconstructed_label{label}_{batch_idx}_{i}.wav'),
                    recon_audio,  # Now this is already a numpy array
                    args.sample_rate
                )
                
                # Compute metrics
                metrics = compute_reconstruction_metrics(orig_audio, recon_audio)
                all_metrics.append(metrics)
        
        # Compute average metrics
        avg_metrics = {
            key: np.mean([m[key] for m in all_metrics])
            for key in all_metrics[0].keys()
        }
        
        # Save metrics
        with open(os.path.join(results_dir, 'reconstruction_metrics.txt'), 'w') as f:
            f.write("Average Reconstruction Metrics:\n")
            for key, value in avg_metrics.items():
                f.write(f"{key}: {value:.4f}\n")
                print(f"{key}: {value:.4f}")

if __name__ == "__main__":
    from argparse import ArgumentParser
    
    parser = ArgumentParser(description="VQVAE Evaluation Script")
    
    parser.add_argument('--data_dir', type=str, default='../Data',
                        help='Directory containing the dataset')
    parser.add_argument('--target_length', type=int, default=22050,
                        help='Target length for audio waveforms')
    parser.add_argument('--sample_rate', type=int, default=22050,
                        help='Audio sample rate')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading')
    
    args = parser.parse_args()
    evaluate_vqvae(args)