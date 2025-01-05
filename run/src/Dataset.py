import os
import librosa
import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
import scipy
from glob import glob
import matplotlib.pyplot as plt

class AudioMNIST(Dataset):
    def __init__(self, 
                 root='../Data',
                 target_sample_rate=22050,
                 duration=None,
                 transform=None,
                 output_format='waveform',  # 'waveform' or 'spectrogram'
                 spec_params=None):
        """
        Enhanced AudioMNIST dataset that can output either waveforms or spectrograms
        
        Args:
            root (str): Root directory of dataset
            target_sample_rate (int): Target sample rate for audio
            duration (float): Duration in seconds (if None, keeps original)
            transform: Transform to apply to waveform
            output_format (str): 'waveform' or 'spectrogram'
            spec_params (dict): Parameters for spectrogram generation:
                - n_fft (int): FFT window size
                - hop_length (int): Number of samples between successive frames
                - n_freq (int): Number of frequency bins to keep
                - n_time (int): Number of time frames to keep
                - complex_output (bool): If True, returns complex spectrogram split into real/imag channels
        """
        super(AudioMNIST, self).__init__()
        torch.set_default_dtype(torch.float32)
        self.root = root
        self.target_sample_rate = target_sample_rate
        self.duration = duration
        self.transform = transform
        self.output_format = output_format
        self.wav_files = glob('*/*.wav', root_dir=root, recursive=True)
        
        # Default spectrogram parameters
        self.spec_params = {
            'n_fft': 512,
            'hop_length': 128,
            'n_freq': 128,
            'n_time': 44,
            'complex_output': False
        }
        if spec_params:
            self.spec_params.update(spec_params)

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

    def process_spectrogram(self, waveform):
        """Generate and process spectrogram according to parameters"""
        f, t, Zxx = scipy.signal.stft(
            waveform, 
            self.target_sample_rate,
            nperseg=self.spec_params['n_fft']//2,
            noverlap=self.spec_params['n_fft']//4,
            window='hann'
        )
        
        # Crop frequency and time dimensions
        Zxx = Zxx[:self.spec_params['n_freq'], :-1]
        
        if self.spec_params['complex_output']:
            # Convert to 2-channel real/imag representation
            Zxx = self.complex_to_2d(Zxx.flatten())
            #Zxx = torch.tensor(Zxx).reshape(2, self.spec_params['n_freq'], self.spec_params['n_time'])
        else:
            # Convert to magnitude spectrogram in dB
            Zxx = self.complex_to_2d(Zxx.flatten())
            Zxx = torch.tensor(Zxx).reshape(2, self.spec_params['n_freq'], self.spec_params['n_time'])
            #Zxx = np.abs(Zxx)
            #Zxx = librosa.amplitude_to_db(Zxx, ref=np.max)
            #print(Zxx.shape)
            Zxx = Zxx.to(torch.float32)
        return Zxx
    
    def twod_to_complex(self,tensor: np.ndarray):
        """Converts complex IQ to two channels representing real and imaginary

        Args:
            tensor (:class:`numpy.ndarray`):
                (batch_size, vector_length, ...)-sized tensor.

        Returns:
            transformed (:class:`numpy.ndarray`):
                Expanded vectors
        """
        tmp_tensor = tensor.reshape(2, tensor.shape[1]*tensor.shape[2])
        new_tensor = np.zeros((1, tmp_tensor.shape[1]), dtype=np.complex64)
        new_tensor[0] = tmp_tensor[0,:]+ 1j * tmp_tensor[1,:]
        new_tensor=new_tensor.reshape(1, tensor.shape[1],tensor.shape[2])
        return new_tensor

    

    def __getitem__(self, index):
        file_name = os.path.join(self.root, self.wav_files[index])
        label = int(os.path.basename(file_name)[0])
        
        # Load and process waveform
        waveform, _ = librosa.load(file_name, sr=self.target_sample_rate, duration=self.duration)
        if self.transform is not None:
            waveform = self.transform(waveform)
        
        # Return either waveform or spectrogram based on output_format
        if self.output_format == 'waveform':
            return torch.tensor(waveform).to(torch.float32), label
        else:  # spectrogram
            return self.process_spectrogram(waveform), label
    
    def __len__(self):
        return len(self.wav_files)