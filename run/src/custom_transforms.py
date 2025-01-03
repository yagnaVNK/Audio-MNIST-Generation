"""
Custom transformations for audio preprocessing written using the torch.Callable paradigm for
easier use.
"""
import librosa
from collections.abc import Callable

import numpy as np

class TrimSilence(Callable):
    def __init__(self, top_db):
        super(TrimSilence, self).__init__()
        self.top_db = top_db
    def __call__(self, waveform):
        waveform, _ = librosa.effects.trim(waveform, top_db = self.top_db)
        return waveform

class FixLength(Callable):
    def __init__(self, length):
        super(FixLength, self).__init__()
        self.length = length
    def __call__(self, waveform):
        return librosa.util.fix_length(waveform, size=self.length)

class TimeStretchFixLength(Callable):
    def __init__(self, length):
        super(TimeStretchFixLength, self).__init__()
        self.length = int(length)

    def __call__(self, waveform):
        num_samples = waveform.shape[0]
        result = librosa.effects.time_stretch(waveform, rate = num_samples / self.length)
        return result[:self.length]

class MelSpectrogram(Callable):
    def __init__(self, sample_rate, add_channel_dim=True, **melspectrogram_args):
        super(MelSpectrogram, self).__init__()
        self.sample_rate = sample_rate
        self.add_channel_dim = add_channel_dim
        self.melspectrogram_args = melspectrogram_args
    def __call__(self, waveform):
        spectrogram = librosa.feature.melspectrogram(
                    y=waveform, 
                    sr=self.sample_rate,
                    **self.melspectrogram_args)
        if self.add_channel_dim:
            spectrogram = spectrogram[np.newaxis, ...]
        return spectrogram
    
class STFT(Callable):
    def __init__(self, **stft_args):
        super(STFT, self).__init__()
        self.stft_args = stft_args
    def __call__(self, waveform):
        stft = librosa.stft(
                    y=waveform,
                    **self.stft_args)
        return np.abs(stft)

class MinMaxNormalization(Callable):
    def __init__(self):
        super(MinMaxNormalization, self).__init__()
    def __call__(self, x):
        x_min = x.min()
        x_max = x.max()
        return ((x - x_min) / (x_max - x_min)), x_min, x_max

class AmplitudeToDB(Callable):
    def __init__(self, ref=1.0):
        super(AmplitudeToDB, self).__init__()
        self.ref = ref
    def __call__(self, waveform):
        return librosa.amplitude_to_db(waveform, ref=self.ref)

class PowerToDB(Callable):
    def __init__(self, ref=1.0):
        super(PowerToDB, self).__init__()
        self.ref = ref
    def __call__(self, waveform):
        return librosa.power_to_db(waveform, ref=self.ref)

class ZScoreNormalization(Callable):
    def __init__(self, mean, std):
        super(ZScoreNormalization, self).__init__()
        self.mean = mean
        self.std = std
    def __call__(self, x):
        return (x - self.mean) / self.std

class MinMaxNormalization(Callable):
    def __init__(self, maximum, minimum):
        super(MinMaxNormalization, self).__init__()
        self.maximum = maximum
        self.minimum = minimum
    def __call__(self, x):
        return (x - self.minimum) / (self.maximum - self.minimum)
    def inverse(self, x_norm):
        return x_norm * (self.maximum - self.minimum) + self.minimum