import os
from glob import glob

import librosa
from torch.utils.data import Dataset

class AudioMNIST(Dataset):
    def __init__(self, 
                 root,
                 target_sample_rate = 22050,
                 duration = None,
                 transform = None):
        super(AudioMNIST, self).__init__()
        self.root = root
        self.target_sample_rate = target_sample_rate
        self.duration = duration
        self.transform = transform
        self.wav_files = glob('*/*.wav', root_dir=root, recursive=True)

    def __getitem__(self, index):
        file_name = os.path.join(self.root, self.wav_files[index])
        label = int(os.path.basename(file_name)[0])
        waveform, _ = librosa.load(file_name, sr=self.target_sample_rate, duration=self.duration)
        
        if self.transform is not None:
            x = self.transform(waveform)

        return x, label
    
    
        
    def __len__(self):
        return len(self.wav_files)
