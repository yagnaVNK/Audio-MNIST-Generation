import torch
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import scipy
from tqdm import tqdm
import os
from torchvision import transforms as T
from custom_transforms import TrimSilence, FixLength
from src.Dataset import AudioMNIST
from top_pr import compute_top_pr as TopPR

device = 'cuda:0'

def compute_metrics(real_data, fake_data):
    """
    Compute metrics using Top PR framework for real and fake datasets.

    Parameters:
    - real_dataset_path: Path to the NumPy file containing the real dataset.
    - fake_dataset_path: Path to the NumPy file containing the fake dataset.

    Returns:
    - A dictionary containing fidelity, diversity, and Top_F1 metrics.
    """
    # Load the datasets
    if isinstance(real_data, (str, bytes, os.PathLike)):
        real_data = np.load(real_data)
    if isinstance(fake_data, (str, bytes, os.PathLike)):
        fake_data = np.load(fake_data)

    # Ensure both datasets are flattened in the signal dimension
    real_data = real_data.reshape(real_data.shape[0], -1)
    fake_data = fake_data.reshape(fake_data.shape[0], -1)

    # Compute metrics using TopPR
    Top_PR = TopPR(
        real_features=real_data,
        fake_features=fake_data,
        alpha=0.1,  # Weight for fidelity and diversity tradeoff
        kernel="cosine",  # Kernel type
        random_proj=True,  # Whether to use random projection
        f1_score=True  # Whether to compute Top_F1
    )

    # Extract the required metrics
    fidelity = Top_PR.get("fidelity")
    diversity = Top_PR.get("diversity")
    top_f1 = Top_PR.get("Top_F1")

    # Print and return the metrics
    print(f"Fidelity: {fidelity}, Diversity: {diversity}, Top_F1: {top_f1}")
    return {"fidelity": fidelity, "diversity": diversity, "Top_F1": top_f1}


def plot_spectrograms(reconstructions, labels, folder, idx, save_prefix="reconstruction"):
    """
    Plot spectrograms for a batch of reconstructions.
    """
    os.makedirs(folder, exist_ok=True)
    
    # Process each reconstruction
    for i, rec in enumerate(reconstructions):
        if rec.nelement() == 0:
            continue
            
        # Convert reconstruction to complex spectrogram
        rec_cpu = rec.cpu().detach().numpy()
        tmp_tensor = rec_cpu.reshape(2, rec_cpu.shape[1] * rec_cpu.shape[2])
        complex_data = tmp_tensor[0, :] + 1j * tmp_tensor[1, :]
        complex_spec = complex_data.reshape(1, rec_cpu.shape[1], rec_cpu.shape[2])
        
        # Generate spectrogram
        spec_db = librosa.amplitude_to_db(np.abs(complex_spec), ref=np.max)
        
        # Plot spectrogram
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(spec_db[0])
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Label {labels[i]}')
        plt.tight_layout()
        
        # Save plot
        plt.savefig(os.path.join(folder, f'{save_prefix}_{idx}_label_{labels[i]}.png'))
        plt.close()

def generate_fake_dataset(transformer_model, vqvae_model, output_file, is_conditional=True, 
                         num_labels=10, BOS_TOKEN=256, samples_per_label=3000, batch_size=30):
    """
    Generate a fake dataset using transformer and VQVAE models.
    """
    all_specs = []
    all_audio = []
    fake_indices = []
    steps_per_label = samples_per_label // batch_size
    
    for label in range(num_labels):
        print(f"Generating data for label {label}...")
        for _ in tqdm(range(steps_per_label)):
            # Generate indices
            if is_conditional:
                context = torch.tensor([[BOS_TOKEN, BOS_TOKEN + 1 + label]] * batch_size).to(device)
                new_indices = transformer_model.generate(context, max_new_tokens=352)
                new_indices = new_indices[:, 2:]  # Remove BOS and label tokens
            else:
                context = torch.tensor([[BOS_TOKEN]] * batch_size).to(device)
                new_indices = transformer_model.generate(context, max_new_tokens=352)
                new_indices = new_indices[:, 1:]  # Remove only BOS token
            
            fake_indices.append(new_indices.cpu().numpy())
            
            # Reshape indices for VQVAE decoding
            reshaped_indices = new_indices.view(-1, 32, 11)
            
            # Decode the generated indices
            reconstructions = vqvae_model.model.decode_samples(reshaped_indices)
            
            # Process each reconstruction
            for rec in reconstructions:
                rec_cpu = rec.cpu().detach().numpy()
                
                # Convert to complex spectrogram
                tmp_tensor = rec_cpu.reshape(2, rec_cpu.shape[1] * rec_cpu.shape[2])
                complex_data = tmp_tensor[0, :] + 1j * tmp_tensor[1, :]
                complex_spec = complex_data.reshape(1, rec_cpu.shape[1], rec_cpu.shape[2])
                
                # Generate spectrogram
                spec_db = librosa.amplitude_to_db(np.abs(complex_spec), ref=np.max)
                all_specs.append(spec_db)
                
                # Generate audio
                _, audio = scipy.signal.istft(complex_spec, 12000)
                all_audio.append(audio)
    
    # Convert to numpy arrays and save
    final_specs = np.array(all_specs)
    final_audio = np.array(all_audio)
    
    np.save(f"{output_file}_specs.npy", final_specs)
    np.save(f"{output_file}_audio.npy", final_audio)
    
    return fake_indices, final_specs, final_audio

def plot_codebook_histograms(indices, path):
    """
    Plot histograms of the quantization indices.
    """
    os.makedirs(path, exist_ok=True)
    
    # Convert indices to numpy array and reshape
    if isinstance(indices, list):
        indices = np.concatenate(indices)
    
    if indices.ndim > 2:
        indices = indices.reshape(-1, indices.shape[-1])
    
    # Split indices by class
    samples_per_class = len(indices) // 10
    indices_per_class = np.array_split(indices, 10)
    
    # Plot histogram for each class
    for i, class_indices in enumerate(indices_per_class):
        plt.figure(figsize=(10, 6))
        flat_indices = class_indices.flatten()
        plt.hist(flat_indices, bins=min(64, len(np.unique(flat_indices))), 
                color='blue', alpha=0.7)
        plt.title(f'Histogram for Class: {i}')
        plt.xlabel('Token Index')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.savefig(os.path.join(path, f"Histogram_Class_{i}.png"))
        plt.close()

def save_real_dataset(dataset, output_file):
    """
    Process and save the real dataset.
    """
    all_specs = []
    all_audio = []
    
    print("Processing real dataset...")
    for data, _ in tqdm(dataset):
        # Convert to complex spectrogram
        complex_data = dataset.twod_to_complex(data.numpy())
        
        # Generate spectrogram
        spec_db = librosa.amplitude_to_db(np.abs(complex_data), ref=np.max)
        all_specs.append(spec_db)
        
        # Generate audio
        _, audio = scipy.signal.istft(complex_data, 12000)
        all_audio.append(audio)
    
    # Convert to numpy arrays and save
    final_specs = np.array(all_specs)
    final_audio = np.array(all_audio)
    
    np.save(f"{output_file}_specs.npy", final_specs)
    np.save(f"{output_file}_audio.npy", final_audio)
    
    return final_specs, final_audio

if __name__ == "__main__":
    # Create base evaluation folder
    base_eval_folder = 'AudioMNIST_Evaluation'
    os.makedirs(base_eval_folder, exist_ok=True)
    BOS_TOKEN = 256
    # Load models
    print("Loading models...")
    VQVAE_PATH = 'saved_models/vqvae_monai.pth'
    MONAI_COND2_PATH = 'saved_models/MONAI_Cond2_Transformer_epochs_50.pt'
    MONAI_UNCOND_PATH = 'saved_models/MONAI_Transformer_epochs_50.pt'
    NANOGPT_COND2_PATH = 'saved_models/NanoGPT_Cond2_Transformer_epochs_50.pt'
    NANOGPT_UNCOND_PATH = 'saved_models/NanoGPT_Transformer_epochs_50.pt'
    
    VQVAE = torch.load(VQVAE_PATH,weights_only=False).to(device)
    monai_cond = torch.load(MONAI_COND2_PATH,weights_only=False).to(device)
    monai_uncond = torch.load(MONAI_UNCOND_PATH,weights_only=False).to(device)
    nanogpt_cond = torch.load(NANOGPT_COND2_PATH,weights_only=False).to(device)
    nanogpt_uncond = torch.load(NANOGPT_UNCOND_PATH,weights_only=False).to(device)
    
    # Set models to eval mode
    VQVAE.eval()
    monai_cond.eval()
    monai_uncond.eval()
    nanogpt_cond.eval()
    nanogpt_uncond.eval()
    
    # Load dataset
    data_dir = '../Data'
    transforms = [TrimSilence(5), FixLength(16000)]
    dataset = AudioMNIST(data_dir, transform=T.Compose(transforms))
    samples_per_class = len(dataset) // 10
    # Save real dataset
    real_dataset_path = os.path.join(base_eval_folder, "real_dataset")
    real_specs, real_audio = save_real_dataset(dataset, real_dataset_path)
    
    # Process each model
    models = {
        'MONAI_Conditional': (monai_cond, True),
        'MONAI_Unconditional': (monai_uncond, False),
        'NanoGPT_Conditional': (nanogpt_cond, True),
        'NanoGPT_Unconditional': (nanogpt_uncond, False)
    }
    
    for model_name, (model, is_conditional) in models.items():
        print(f"\nProcessing {model_name}...")
        
        # Create model-specific folders
        model_folder = os.path.join(base_eval_folder, f'{model_name}_results')
        spec_folder = os.path.join(model_folder, 'Spectrograms')
        histogram_folder = os.path.join(model_folder, 'CodebookHistograms')
        os.makedirs(model_folder, exist_ok=True)
        
        # Generate fake dataset
        fake_dataset_path = os.path.join(model_folder, f"fake_dataset_{model_name}")
        fake_indices, fake_specs, fake_audio = generate_fake_dataset(
            transformer_model=model,
            vqvae_model=VQVAE,
            output_file=fake_dataset_path,
            is_conditional=is_conditional
        )
        
        # Plot codebook histograms
        plot_codebook_histograms(fake_indices, histogram_folder)
        
        # Generate sample spectrograms
        for j in range(5):
            all_reconstructions = []
            labels = list(range(10))
            
            for i in labels:
                if is_conditional:
                    context = torch.tensor([[BOS_TOKEN, BOS_TOKEN + 1 + i]]).to(device)
                    new_indices = model.generate(context, max_new_tokens=352)
                    new_indices = new_indices[:, 2:]
                else:
                    context = torch.tensor([[BOS_TOKEN]]).to(device)
                    new_indices = model.generate(context, max_new_tokens=352)
                    new_indices = new_indices[:, 1:]

                reshaped_indices = new_indices.view(1, 32, 11)
                
                reconstruction = VQVAE.model.decode_samples(reshaped_indices)
                all_reconstructions.append(reconstruction.squeeze())
            
            all_reconstructions = torch.stack(all_reconstructions)
            plot_spectrograms(all_reconstructions, labels, spec_folder, j, 
                            save_prefix=f"{model_name}_sample")
        
        # Compute and print metrics
        metrics = compute_metrics(real_specs,fake_specs)
        
        print(f"\nMetrics for {model_name}:")
        print(f"Fidelity: {metrics['fidelity']:.4f}")
        print(f"Diversity: {metrics['diversity']:.4f}")
        print(f"Top F1: {metrics['Top_F1']:.4f}")
        
        # Save metrics
        with open(os.path.join(model_folder, "metrics.txt"), "w") as f:
            f.write(f"Fidelity: {metrics['fidelity']:.4f}\n")
            f.write(f"Diversity: {metrics['diversity']:.4f}\n")
            f.write(f"Top F1: {metrics['Top_F1']:.4f}\n")