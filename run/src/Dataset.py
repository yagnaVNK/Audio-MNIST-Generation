import os
import librosa
import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
import scipy
import os
from glob import glob

import librosa
from torch.utils.data import Dataset

class AudioMNIST(Dataset):
    def __init__(self, 
                 root='../Data',
                 target_sample_rate = 22050,
                 duration = None,
                 transform = None):
        super(AudioMNIST, self).__init__()
        self.root = root
        self.target_sample_rate = target_sample_rate
        self.duration = duration
        self.transform = transform
        self.wav_files = glob('*/*.wav', root_dir=root, recursive=True)

    def complex_to_2d(self,tensor: np.ndarray) -> np.ndarray:
        """Converts complex IQ to two channels representing real and imaginary

        Args:
            tensor (:class:`numpy.ndarray`):
                (batch_size, vector_length, ...)-sized tensor.

        Returns:
            transformed (:class:`numpy.ndarray`):
                Expanded vectors
        """

        new_tensor = np.zeros((2, tensor.shape[0]), dtype=np.float64)
        new_tensor[0] = np.real(tensor).astype(np.float64)
        new_tensor[1] = np.imag(tensor).astype(np.float64)
        return new_tensor

    def __getitem__(self, index):
        file_name = os.path.join(self.root, self.wav_files[index])
        label = int(os.path.basename(file_name)[0])
        waveform, _ = librosa.load(file_name, sr=self.target_sample_rate, duration=self.duration)
        n_fft = 512
        if self.transform is not None:
            waveform = self.transform(waveform)
            #print(f"Shape after time stretch {waveform.shape}")
        
        f, t, Zxx = scipy.signal.stft( waveform, self.target_sample_rate/2, nperseg = n_fft/2, noverlap = n_fft/4, window='hann')
        #print(f"Shape of Zxx after melspectrogram {Zxx.shape}")
        Zxx = Zxx[0:128, :-1]
        #print(f"Shape of Zxx after cropping {Zxx.shape}")
        Zxx = self.complex_to_2d(Zxx.flatten())
        #print(f"Shape of Zxx after flattening and converting to 2d {Zxx.shape}")
        Zxx = torch.tensor(Zxx).reshape(2, 128, 173) 
        return Zxx, label
    
    def __len__(self):
        return len(self.wav_files)
