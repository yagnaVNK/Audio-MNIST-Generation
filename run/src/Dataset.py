import torch
from torch.utils.data import Dataset
import librosa
import numpy as np
import scipy
import os
from glob import glob

class AudioMNIST(Dataset):
    def __init__(self, root, target_sample_rate=12000, duration=None, transform=None):
        super(AudioMNIST, self).__init__()
        self.root = root
        self.target_sample_rate = target_sample_rate
        self.duration = duration
        self.transform = transform
        self.wav_files = glob('*/*.wav', root_dir=root, recursive=True)
        self.n_fft = 512
        self.amp_max = 0
        self.amp_min = 0

    def get_spectrogram(self, waveform):
        f, t, Zxx = scipy.signal.stft(
            waveform, 
            self.target_sample_rate,
            nperseg=self.n_fft//2,
            noverlap=self.n_fft//4,
            window='hann'
        )
        Zxx = Zxx[0:128, :-1]
        
        # Normalize amplitudes
        amplitudes = np.abs(Zxx)
        phases = np.angle(Zxx)
        
        self.amp_min = np.min(amplitudes)
        self.amp_max = np.max(amplitudes)
        
        eps = 1e-10
        normalized_amplitudes = (amplitudes - self.amp_min) / (self.amp_max - self.amp_min + eps)
        Zxx_normalized = normalized_amplitudes * np.exp(1j * phases)
        
        return Zxx_normalized

    def twod_to_complex(self, tensor):
        tmp_tensor = tensor.reshape(2, tensor.shape[1]*tensor.shape[2])
        complex_data = tmp_tensor[0,:] + 1j * tmp_tensor[1,:]
        
        # Denormalize amplitudes
        amplitudes = np.abs(complex_data)
        phases = np.angle(complex_data)
        
        denorm_amplitudes = amplitudes * (self.amp_max - self.amp_min) + self.amp_min
        new_tensor = denorm_amplitudes * np.exp(1j * phases)
        
        return new_tensor.reshape(1, tensor.shape[1], tensor.shape[2])

    def complex_to_2d(self, tensor):
        new_tensor = np.zeros((2, tensor.shape[0]), dtype=np.float64)
        new_tensor[0] = np.real(tensor).astype(np.float64)
        new_tensor[1] = np.imag(tensor).astype(np.float64)
        return new_tensor


    def __getitem__(self, index):
        file_name = os.path.join(self.root, self.wav_files[index])
        label = int(os.path.basename(file_name)[0])
        waveform, _ = librosa.load(file_name, sr=self.target_sample_rate, duration=self.duration)
        
        if self.transform is not None:
            waveform = self.transform(waveform)
            
        Zxx = self.get_spectrogram(waveform)
        datapoint = self.complex_to_2d(Zxx.flatten())
        datapoint = datapoint.reshape(2, 128, 125)
        datapoint = datapoint[:,:,:44]
        return torch.tensor(datapoint, dtype=torch.float32), label
    
    def __len__(self):
        return len(self.wav_files)