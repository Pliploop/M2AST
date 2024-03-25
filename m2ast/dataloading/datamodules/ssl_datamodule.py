from m2ast.dataloading.datasets.dataset import AudioDataset
import torch
import pytorch_lightning as pl
import os
import pandas as pd

class SSLDataModule(pl.LightningDataModule):
    def __init__(self, audio_dir, target_len_s = None, target_sr = 44100, target_n_samples = 2**19-1, augmentations = None, batch_size = 32, num_workers = 8, transform = False, val_split = 0.1):
        super().__init__()
        self.audio_dir = audio_dir
        self.target_len_s = target_len_s
        self.target_sr = target_sr
        self.target_n_samples = target_n_samples if target_n_samples is not None else target_len_s*target_sr
        self.augmentations = augmentations
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform
        
        self.annotations = self.fetch_annotations()
        
        self.train_annotations = self.annotations[self.annotations['split'] == 'train']
        self.val_annotations = self.annotations[self.annotations['split'] == 'val']
        
    def fetch_annotations(self):
        
        annotations = []
        
        ## scan audio_dir for any audio files in dirs or subdirs, and create annotations with file_path and split
        
        for root, dirs, files in os.walk(self.audio_dir):
            for file in files:
                if file.endswith('.wav') or file.endswith('.mp3'):
                    annotations.append({'file_path': os.path.join(root, file)})
                        
        
        annotations = pd.DataFrame(annotations, columns = ['file_path'])
        annotations['split'] = 'train'
        # shuffle and split
        annotations = annotations.sample(frac=1).reset_index(drop=True)
        val_idx = int(len(annotations)*0.1)
        annotations.loc[:val_idx, 'split'] = 'val'
        
        print(annotations.head())
        
        return annotations
            
    
    def setup(self, stage=None):
        self.train_dataset = AudioDataset(annotations = self.train_annotations, target_len_s = self.target_len_s, target_sr = self.target_sr, target_n_samples = self.target_n_samples, augmentations = self.augmentations, transform = self.transform, train = True, return_labels = False)
        self.val_dataset = AudioDataset(annotations = self.val_annotations, target_len_s = self.target_len_s, target_sr = self.target_sr, target_n_samples = self.target_n_samples, augmentations = None, transform = False, train = False, return_labels = False)
            
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
    
    def dummy_call(self):
        dummy = self.train_dataset[0]
        for key in dummy:
            print(key, dummy[key].shape)