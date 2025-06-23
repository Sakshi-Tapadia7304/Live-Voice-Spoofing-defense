import torch
import os
import librosa
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset
import random
class ASVspoofDataset(Dataset):
    def _init_(self, file_list, label_dict, root_path, sr=16000, n_mels=64, max_len=256, augment=True):
        self.file_list = file_list
        self.label_dict = label_dict
        self.root_path = root_path
        self.sr = sr
        self.n_mels = n_mels
        self.max_len = max_len
        self.augment = augment
        self.n_fft = 512
        self.hop_length = 160
        self.win_length = 400
    def _len_(self):
        return len(self.file_list)
    def _getitem_(self, idx):
        filename = self.file_list[idx]
        label = self.label_dict[filename]
        filepath = os.path.join(self.root_path, filename + '.flac')
        y, _ = librosa.load(filepath, sr=self.sr)
        if self.augment:
            if random.random() < 0.5:
                y += np.random.normal(0, 0.005, y.shape)
            if random.random() < 0.5:
                y = np.roll(y, int(0.1 * len(y)))
        y = np.clip(y, -1.0, 1.0)
        mel = librosa.feature.melspectrogram(
            y=y, sr=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            n_mels=self.n_mels
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_tensor = torch.tensor(mel_db).unsqueeze(0).float()
        if mel_tensor.shape[-1] > self.max_len:
            mel_tensor = mel_tensor[:, :, :self.max_len]
        else:
            mel_tensor = F.pad(mel_tensor, (0, self.max_len - mel_tensor.shape[-1]))
        mel_tensor = (mel_tensor - mel_tensor.mean()) / (mel_tensor.std() + 1e-6)
        return mel_tensor, label