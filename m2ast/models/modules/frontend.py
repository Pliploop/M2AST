import torchaudio
import torch
from torch import nn

## create two torch frontends to transform audio to mel spectrogram and CQT

class Melgram(nn.Module):
    def __init__(self, n_fft=2048, hop_length=512, n_mels=128, sr=44100):
        super(Melgram, self).__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.sr = sr
        self.mel_layer = torchaudio.transforms.MelSpectrogram(n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, sample_rate=sr)
        
    def forward(self, x):
        mel = self.mel_layer(x)
        return torchaudio.functional.amplitude_to_DB(mel, multiplier=10.0, amin=1e-10, db_multiplier=0.0)
    