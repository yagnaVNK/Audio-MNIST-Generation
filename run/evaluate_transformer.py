import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import librosa
from src.VQVAE import VQVAE
from src.Transformer import TransformerModel
from src.utils import *

def generate_fake_audio(vqvae_model, transformer_model, label, num_samples=1):
    """
    Generate fake audio samples for a given label
    
    Args:
        vqvae_model (VQVAE): Trained VQVAE model
        transformer_model (TransformerModel): Trained Transformer model
        label (int): Label for audio generation
        num_samples (int): Number of samples to generate
    
    Returns:
        tuple: Generated mel spectrograms and audio waveforms
    """
    # Create results directory
    results_dir = os.path.join("results", "fake_audio")
    os.makedirs(results_dir, exist_ok=True)
    samples_dir = os.path.join(results_dir, "samples")
    os.makedirs(samples_dir, exist_ok=True)
    
    # Stores generated data
    generated_spectrograms = []
    generated_audios = []
    
    with torch.no_grad():
        for sample_idx in range(num_samples):
            # Generate 736 new tokens starting with the label
            context = torch.tensor([[label]]).to(device)
            generated_sequence = transformer_model.generate(context, max_new_tokens=736)
            
            # Reshape generated sequence to 1, 32, 23 (label + indices)
            generated_indices = generated_sequence[0, 1:].view(1, 32, 23)
            
            # Use codebook lookup to get quantized representation
            zq = vqvae_model.codebook.lookup(generated_indices)
            
            # Decode to mel spectrogram
            generated_spec = vqvae_model.Decoder(zq)
            
            # Convert to numpy for further processing
            generated_spec_np = generated_spec.squeeze().detach().cpu().numpy()
            
            # Plot and save spectrogram
            plt.figure(figsize=(10, 5))
            plt.title(f'Generated Mel Spectrogram (Label: {label})')
            plt.imshow(generated_spec_np, aspect='auto', origin='lower')
            plt.colorbar()
            plt.tight_layout()
            plt.savefig(os.path.join(samples_dir, f'{sample_idx}fake_spectrogram_label{label}.png'))
            plt.close()
            
            # Convert to audio
            # First, denormalize the spectrogram
            generated_spec_np = generated_spec_np * 80 - 80
            mel_spec_power = librosa.db_to_power(generated_spec_np)
            
            # Compute mel basis
            mel_basis = librosa.filters.mel(
                sr=48000, 
                n_fft=2048, 
                n_mels=generated_spec_np.shape[0], 
                fmin=20, 
                fmax=24000
            )
            
            # Reconstruct linear spectrogram
            S = np.linalg.pinv(mel_basis).dot(mel_spec_power)
            S = np.maximum(0, S)
            
            # Griffin-Lim reconstruction
            audio = librosa.griffinlim(
                S, 
                n_iter=32, 
                hop_length=512, 
                win_length=2048
            )
            
            # Normalize audio
            audio = librosa.util.normalize(audio)
            
            # Save audio file
            sf.write(
                os.path.join(samples_dir, f'{sample_idx}fake_audio_label{label}.wav'),
                audio,
                48000
            )
            
            # Store generated data
            generated_spectrograms.append(generated_spec_np)
            generated_audios.append(audio)
    
    return generated_spectrograms, generated_audios

def main():
    # Load models
    vqvae_model = torch.load(
        os.path.join("results", 'vqvae_run_20241216_143233', 'final_model.pt'), 
        weights_only=False
    ).to(device)
    vqvae_model.eval()
    
    transformer_model = torch.load(
        os.path.join("results", 'vqvae_run_20241216_143233', 'Transformer.pt'), 
        weights_only=False
    ).to(device)
    transformer_model.eval()
    
    # Possible labels (adjust based on your dataset)
    labels = list(range(10))  # Assuming 10 classes
    
    # Generate samples for each label
    for label in labels:
        print(f"Generating samples for label {label}")
        generate_fake_audio(vqvae_model, transformer_model, label, num_samples=5)
    
    print("Fake audio generation complete!")

if __name__ == "__main__":
    main()