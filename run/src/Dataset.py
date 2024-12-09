import os
import torchaudio
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

class AudioMNISTDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): Path to the directory with the Audio MNIST data.
            transform (callable, optional): Transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.file_paths = []
        self.labels = []

        # Load all file paths and their labels
        for folder_name in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, folder_name)
            if os.path.isdir(folder_path):
                for file_name in os.listdir(folder_path):
                    if file_name.endswith(".wav"):
                        self.file_paths.append(os.path.join(folder_path, file_name))
                        # Extract label from the filename (e.g., "3_01.wav" -> label is 3)
                        label = int(file_name.split("_")[0])
                        self.labels.append(label)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        audio_path = self.file_paths[idx]
        label = self.labels[idx]

        # Load audio file
        waveform, sample_rate = torchaudio.load(audio_path)

        # Apply any transforms if provided
        if self.transform:
            waveform = self.transform(waveform)

        return waveform, label

