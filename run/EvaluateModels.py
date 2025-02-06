import torch
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import scipy
from tqdm import tqdm
import os
from IPython.display import Audio
import soundfile as sf
from custom_transforms import TrimSilence, FixLength
from src.Dataset import AudioMNIST
import torchvision.transforms as T
from top_pr import compute_top_pr as TopPR
from src.Transformer import *
from src.TransformerMonai import *

device = 'cuda:0'

def generate_batch_samples(transformer_model, model, dataset_length, batch_size=32, 
                         output_dir="batch_generated", model_name="model"):
    """
    Generate fake samples in batches matching the dataset length and save audio/spectrograms
    Returns: Tuple of (specs, audio) as numpy arrays
    """
    import numpy as np
    import torch
    from tqdm import tqdm
    from scipy.io.wavfile import write
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/audio", exist_ok=True)
    os.makedirs(f"{output_dir}/specs", exist_ok=True)
    
    # Calculate samples per class to match dataset length
    samples_per_class = dataset_length // 10
    
    all_specs = []
    all_audio = []
    BOS_TOKEN = 256
    
    for label in range(10):
        print(f"\nGenerating batch samples for label {label}")
        
        # Calculate number of full batches and remaining samples
        num_full_batches = samples_per_class // batch_size
        remaining_samples = samples_per_class % batch_size
        
        # Generate full batches
        for batch_idx in tqdm(range(num_full_batches)):
            # Create batch of contexts
            batch_contexts = torch.tensor([[BOS_TOKEN, 257 + label] for _ in range(batch_size)], 
                                       dtype=torch.long, device=device)
            
            # Generate indices for entire batch
            generated_batch = transformer_model.generate(batch_contexts, max_new_tokens=352)
            indices_batch = generated_batch[:, 2:]
            fake_indices_batch = indices_batch.view(batch_size, 32, 11)
            
            # Get reconstructions for batch
            fake_recon_batch = model.model.decode_samples(fake_indices_batch)
            
            # Process each sample in batch
            for sample_idx, fake_recon in enumerate(fake_recon_batch):
                fake_recon_cpu = fake_recon.cpu().detach().numpy()
                
                # Convert to complex
                tmp_tensor = fake_recon_cpu.reshape(2, fake_recon_cpu.shape[1]*fake_recon_cpu.shape[2])
                complex_data = tmp_tensor[0,:] + 1j * tmp_tensor[1,:]
                fake_recon_complex = complex_data.reshape(1, fake_recon_cpu.shape[1], fake_recon_cpu.shape[2])
                
                # Generate spectrogram
                Img_fake = librosa.amplitude_to_db(np.abs(fake_recon_complex), ref=np.max)
                all_specs.append(Img_fake)
                
                # Generate audio
                _, audio = scipy.signal.istft(fake_recon_complex, 12000)
                all_audio.append(audio)
                
                # Save spectrogram
                plt.figure(figsize=(10, 4))
                librosa.display.specshow(Img_fake[0])
                plt.colorbar(format='%+2.0f dB')
                plt.title(f'{model_name} - Label {label} Batch {batch_idx} Sample {sample_idx}')
                plt.tight_layout()
                plt.savefig(f'{output_dir}/specs/{model_name}_label{label}_batch{batch_idx}_sample{sample_idx}.png')
                plt.close()
                
                # Save audio
                array = audio
                array = array / np.max(np.abs(array)) if np.max(np.abs(array)) > 0 else array
                array = (array * 32767).astype(np.int16)
                write(f'{output_dir}/audio/{model_name}_label{label}_batch{batch_idx}_sample{sample_idx}.wav', 12000, array.T)
        
        # Handle remaining samples
        if remaining_samples > 0:
            batch_contexts = torch.tensor([[BOS_TOKEN, 257 + label] for _ in range(remaining_samples)], 
                                       dtype=torch.long, device=device)
            generated_batch = transformer_model.generate(batch_contexts, max_new_tokens=352)
            indices_batch = generated_batch[:, 2:]
            fake_indices_batch = indices_batch.view(remaining_samples, 32, 11)
            
            fake_recon_batch = model.model.decode_samples(fake_indices_batch)
            
            for sample_idx, fake_recon in enumerate(fake_recon_batch):
                fake_recon_cpu = fake_recon.cpu().detach().numpy()
                tmp_tensor = fake_recon_cpu.reshape(2, fake_recon_cpu.shape[1]*fake_recon_cpu.shape[2])
                complex_data = tmp_tensor[0,:] + 1j * tmp_tensor[1,:]
                fake_recon_complex = complex_data.reshape(1, fake_recon_cpu.shape[1], fake_recon_cpu.shape[2])
                
                Img_fake = librosa.amplitude_to_db(np.abs(fake_recon_complex), ref=np.max)
                all_specs.append(Img_fake)
                
                _, audio = scipy.signal.istft(fake_recon_complex, 12000)
                all_audio.append(audio)
                
                # Save spectrogram
                plt.figure(figsize=(10, 4))
                librosa.display.specshow(Img_fake[0])
                plt.colorbar(format='%+2.0f dB')
                plt.title(f'{model_name} - Label {label} Remaining Sample {sample_idx}')
                plt.tight_layout()
                plt.savefig(f'{output_dir}/specs/{model_name}_label{label}_remaining_sample{sample_idx}.png')
                plt.close()
                
                # Save audio
                array = audio
                array = array / np.max(np.abs(array)) if np.max(np.abs(array)) > 0 else array
                array = (array * 32767).astype(np.int16)
                write(f'{output_dir}/audio/{model_name}_label{label}_remaining_sample{sample_idx}.wav', 12000, array.T)
    
    # Convert to numpy arrays
    all_specs = np.array(all_specs)
    all_audio = np.array(all_audio)
    
    # Save arrays
    np.save(f"{output_dir}/{model_name}_specs.npy", all_specs)
    np.save(f"{output_dir}/{model_name}_audio.npy", all_audio)
    
    return all_specs, all_audio

def generate_and_save_samples(transformer_model, model, dataset, num_samples_per_class=100, 
                            output_dir="generated_samples", model_name="model"):
    """
    Generate fake samples and save both audio and spectrograms
    """
    import numpy as np
    from scipy.io.wavfile import write
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/audio", exist_ok=True)
    os.makedirs(f"{output_dir}/specs", exist_ok=True)
    
    all_specs = []
    all_audio = []
    BOS_TOKEN = 256
    
    for label in range(10):
        print(f"\nGenerating samples for label {label}")
        for i in tqdm(range(num_samples_per_class)):
            # Generate indices
            context = torch.tensor([[BOS_TOKEN, 257 + label]], dtype=torch.long, device=device)
            generated = transformer_model.generate(context, max_new_tokens=352)
            indices = generated[0, 2:]
            fake_indices = indices.view(1, 32, 11)
            
            # Get reconstruction
            fake_recon = model.model.decode_samples(fake_indices)
            fake_recon_cpu = fake_recon[0].cpu().detach().numpy()
            
            # Convert to complex directly without denormalization
            tmp_tensor = fake_recon_cpu.reshape(2, fake_recon_cpu.shape[1]*fake_recon_cpu.shape[2])
            complex_data = tmp_tensor[0,:] + 1j * tmp_tensor[1,:]
            fake_recon_complex = complex_data.reshape(1, fake_recon_cpu.shape[1], fake_recon_cpu.shape[2])
            
            # Generate spectrogram
            Img_fake = librosa.amplitude_to_db(np.abs(fake_recon_complex), ref=np.max)
            all_specs.append(Img_fake)
            
            # Generate audio
            _, audio = scipy.signal.istft(fake_recon_complex, 12000)
            all_audio.append(audio)
            
            # Save individual samples
            if i < 10:
                plt.figure(figsize=(10, 4))
                librosa.display.specshow(Img_fake[0])
                plt.colorbar(format='%+2.0f dB')
                plt.title(f'{model_name} - Label {label} Sample {i}')
                plt.tight_layout()
                plt.savefig(f'{output_dir}/specs/{model_name}_label{label}_sample{i}.png')
                plt.close()
                
                # Normalize audio to 16-bit int range
                array = audio
                array = array / np.max(np.abs(array)) if np.max(np.abs(array)) > 0 else array
                array = (array * 32767).astype(np.int16)
                
                output_path = f'{output_dir}/audio/{model_name}_label{label}_sample{i}.wav'
                write(output_path, 12000, array.T)
    
    return np.array(all_specs), np.array(all_audio)

def plot_comparison(real_specs, monai_specs, gpt_specs, output_dir):
    """
    Plot comparison of real and generated spectrograms
    """
    os.makedirs(output_dir + "/comparisons", exist_ok=True)
    
    for label in range(10):
        plt.figure(figsize=(15, 5))
        
        # Plot real sample
        plt.subplot(1, 3, 1)
        librosa.display.specshow(real_specs[label][0])
        plt.title(f'Real - Label {label}')
        plt.colorbar(format='%+2.0f dB')
        
        # Plot MONAI sample
        plt.subplot(1, 3, 2)
        librosa.display.specshow(monai_specs[label][0])
        plt.title(f'MONAI - Label {label}')
        plt.colorbar(format='%+2.0f dB')
        
        # Plot NanoGPT sample
        plt.subplot(1, 3, 3)
        librosa.display.specshow(gpt_specs[label][0])
        plt.title(f'NanoGPT - Label {label}')
        plt.colorbar(format='%+2.0f dB')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/comparisons/comparison_label{label}.png')
        plt.close()

def compute_metrics(real_dataset, fake_dataset):
    """
    Compute Top-PR metrics
    """
    real_flat = real_dataset.reshape(real_dataset.shape[0], -1)
    fake_flat = fake_dataset.reshape(fake_dataset.shape[0], -1)
    
    # Compute metrics using TopPR
    metrics = TopPR(
        real_features=real_flat,
        fake_features=fake_flat,
        alpha=0.1,
        kernel="cosine",
        random_proj=True,
        f1_score=True
    )
    
    return {
        "fidelity": metrics.get("fidelity"),
        "diversity": metrics.get("diversity"),
        "top_f1": metrics.get("Top_F1")
    }

if __name__ == "__main__":
    # Load models and dataset
    data_dir = '../Data'
    transforms = [TrimSilence(5), FixLength(16000)]
    dataset = AudioMNIST(data_dir, transform=T.Compose(transforms))

    # Create output directories
    base_output_dir = "evaluation_results"
    monai_output_dir = f"{base_output_dir}/monai"
    gpt_output_dir = f"{base_output_dir}/nanogpt"
    os.makedirs(base_output_dir, exist_ok=True)

    # Load and check models
    print("Loading models...")
    VQVAE_PATH = 'saved_models/vqvae_monai.pth'
    MONAI_TRANSFORMER_MODEL_PATH = 'saved_models/MONAI_Cond_Transformer_epochs_30.pt'
    TRANSFORMER_MODEL_PATH = 'saved_models/NanoGPT_Cond2_Transformer_epochs_30.pt'

    model = torch.load(VQVAE_PATH).to(device)
    monai_model = torch.load(MONAI_TRANSFORMER_MODEL_PATH).to(device)
    gpt_model = torch.load(TRANSFORMER_MODEL_PATH).to(device)

    # Print model information
    print("\nModel states:")
    print(f"VQVAE model type: {type(model)}")
    print(f"MONAI model type: {type(monai_model)}")
    print(f"GPT model type: {type(gpt_model)}")

    # Set models to eval mode
    model.eval()
    monai_model.eval()
    gpt_model.eval()
    
    # Get dataset length for batch generation
    dataset_length = len(dataset)
    print(f"\nTotal dataset length: {dataset_length}")
    
    # Generate batch samples
    print("\nGenerating MONAI batch samples...")
    batch_monai_output_dir = f"{base_output_dir}/batch_monai"
    batch_monai_specs, batch_monai_audio = generate_batch_samples(
        monai_model, model, dataset_length,
        batch_size=32,
        output_dir=batch_monai_output_dir,
        model_name="monai"
    )
    
    print("\nGenerating NanoGPT batch samples...")
    batch_gpt_output_dir = f"{base_output_dir}/batch_nanogpt"
    batch_gpt_specs, batch_gpt_audio = generate_batch_samples(
        gpt_model, model, dataset_length,
        batch_size=32,
        output_dir=batch_gpt_output_dir,
        model_name="nanogpt"
    )
    
    # Get all real samples
    print("\nProcessing real samples...")
    real_specs = []
    real_audio = []
    for data, label in tqdm(dataset):
        # Convert to complex and get spectrogram
        complex_data = dataset.twod_to_complex(data.numpy())
        spec = librosa.amplitude_to_db(complex_data, ref=np.max)
        real_specs.append(spec)
        # Get audio
        _, audio = scipy.signal.istft(complex_data, 12000)
        real_audio.append(audio)
    
    real_specs = np.array(real_specs)
    real_audio = np.array(real_audio)
    
    # Save real arrays
    np.save(f"{base_output_dir}/real_specs.npy", real_specs)
    np.save(f"{base_output_dir}/real_audio.npy", real_audio)
    
    # Compute batch metrics
    print("\nComputing batch metrics...")
    batch_monai_metrics = compute_metrics(real_specs, batch_monai_specs)
    batch_gpt_metrics = compute_metrics(real_specs, batch_gpt_specs)
    
    print("\nBatch MONAI Metrics:")
    print(f"Fidelity: {batch_monai_metrics['fidelity']:.4f}")
    print(f"Diversity: {batch_monai_metrics['diversity']:.4f}")
    print(f"Top F1: {batch_monai_metrics['top_f1']:.4f}")
    
    print("\nBatch NanoGPT Metrics:")
    print(f"Fidelity: {batch_gpt_metrics['fidelity']:.4f}")
    print(f"Diversity: {batch_gpt_metrics['diversity']:.4f}")
    print(f"Top F1: {batch_gpt_metrics['top_f1']:.4f}")
    
    # Save metrics to file
    with open(f"{base_output_dir}/metrics.txt", "w") as f:
        f.write("MONAI Metrics:\n")
        f.write(f"Fidelity: {batch_monai_metrics['fidelity']:.4f}\n")
        f.write(f"Diversity: {batch_monai_metrics['diversity']:.4f}\n")
        f.write(f"Top F1: {batch_monai_metrics['top_f1']:.4f}\n\n")
        
        f.write("NanoGPT Metrics:\n")
        f.write(f"Fidelity: {batch_gpt_metrics['fidelity']:.4f}\n")
        f.write(f"Diversity: {batch_gpt_metrics['diversity']:.4f}\n")
        f.write(f"Top F1: {batch_gpt_metrics['top_f1']:.4f}\n")