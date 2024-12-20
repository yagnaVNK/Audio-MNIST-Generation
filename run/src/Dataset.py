import os
import librosa
import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path

class AudioDataset(Dataset):
    def __init__(self, root_dir=None, target_length=47998, transform=None, sample_rate=48000, 
                 n_mels=128, n_fft=2048, hop_length=512):
        if root_dir is None:
            script_dir = Path(__file__).parent.absolute()
            root_dir = script_dir.parent.parent / "Data"
        else:
            root_dir = Path(root_dir)

        self.root_dir = root_dir.absolute()
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.root_dir}")

        self.transform = transform
        self.file_paths = []
        self.labels = []
        self.sample_rate = sample_rate
        self.target_length = target_length
        
        # Mel spectrogram parameters
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length

        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Scan directory for audio files
        for folder in self.root_dir.iterdir():
            if folder.is_dir():
                for file_path in folder.glob("*.wav"):
                    try:
                        label = int(file_path.name.split("_")[0])
                        self.file_paths.append(file_path)
                        self.labels.append(label)
                    except (ValueError, IndexError) as e:
                        print(f"Warning: Skipping file {file_path.name} - Invalid filename format")

        if not self.file_paths:
            raise RuntimeError(f"No valid audio files found in {self.root_dir}")

        print(f"Dataset initialized with {len(self.file_paths)} files")
        print(f"Using data directory: {self.root_dir}")
        print(f"Device: {self.device}")

    def _process_waveform(self, waveform):
        """Process waveform with robust normalization."""
        # Center the waveform
        waveform = librosa.util.normalize(waveform, axis=0)
        
        # Handle padding/truncation
        if len(waveform) < self.target_length:
            padding = self.target_length - len(waveform)
            waveform = np.pad(waveform, (0, padding), mode='constant')
        elif len(waveform) > self.target_length:
            waveform = waveform[:self.target_length]
            
        return waveform

    def _compute_melspectrogram(self, waveform):
        """Compute mel spectrogram using librosa with robust processing."""
        try:
            # Compute mel spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=waveform,
                sr=self.sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                n_mels=self.n_mels,
                fmin=20,
                fmax=self.sample_rate//2,
                power=2.0
            )
            
            # Convert to log scale with safe handling of small values
            mel_spec = librosa.power_to_db(mel_spec, ref=np.max, top_db=80)
            
            # Normalize to [0, 1] range
            mel_spec = (mel_spec - mel_spec.min()) / (mel_spec.max() - mel_spec.min() + 1e-6)
            
            # Convert to torch tensor
            mel_spec = torch.FloatTensor(mel_spec)
            
            return mel_spec

        except Exception as e:
            print(f"Error in mel spectrogram computation: {str(e)}")
            # Return a zero tensor of the expected shape if computation fails
            return torch.zeros((self.n_mels, self.target_length // self.hop_length + 1))

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]

        try:
            # Load audio
            waveform, _ = librosa.load(str(file_path), sr=self.sample_rate)
            
            # Process waveform
            waveform = self._process_waveform(waveform)
            
            # Convert to mel spectrogram
            mel_spec = self._compute_melspectrogram(waveform)
            
            # Move to appropriate device
            mel_spec = mel_spec.to(self.device)
            
            # Apply additional transforms if specified
            if self.transform:
                mel_spec = self.transform(mel_spec)

            return mel_spec, label

        except Exception as e:
            print(f"Error processing file {file_path.name}: {str(e)}")
            # Return a zero tensor and the label if processing fails
            return torch.zeros((self.n_mels, self.target_length // self.hop_length + 1)).to(self.device), label


if __name__ == "__main__":
    # Test the dataset
    try:
        dataset = AudioDataset()
        mel_spec, label = dataset[0]
        print("Success!")
        print("Mel spectrogram shape:", mel_spec.shape)
        print("Label:", label)
        print(f"Total samples: {len(dataset)}")
        print(f"Value range: [{mel_spec.min():.3f}, {mel_spec.max():.3f}]")
        
        # Test for NaN/Inf values
        print("Contains NaN:", torch.isnan(mel_spec).any())
        print("Contains Inf:", torch.isinf(mel_spec).any())
        
        # Additional statistics
        print(f"Mean value: {mel_spec.mean():.3f}")
        print(f"Std value: {mel_spec.std():.3f}")
    except Exception as e:
        print(f"Error: {str(e)}")