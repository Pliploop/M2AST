from m2ast.dataloading.datasets.dataset import AudioDataset
from m2ast.dataloading.datamodules.datamodule_splitter import DataModuleSplitter
import torch
import pytorch_lightning as pl
import os
import pandas as pd

class AudioDataModule(pl.LightningDataModule):
    def __init__(self, task = None, audio_dir = None, target_len_s = None, target_sr = 44100, target_n_samples = 2**19-1, augmentations = None, batch_size = 32, num_workers = 8, transform = False, val_split = 0.1):
        super().__init__()
        self.task = task
        self.audio_dir = audio_dir
        assert self.audio_dir is not None and self.task is not None, 'task and audio_dir cannot be None at the same time'
        
        
        self.splitter = DataModuleSplitter(audio_dir,task,val_split)
        self.target_len_s = target_len_s
        self.target_sr = target_sr
        self.target_n_samples = target_n_samples if target_n_samples is not None else target_len_s*target_sr
        self.augmentations = augmentations
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform
        self.return_labels = self.task is not None
        
        self.annotations = self.splitter.annotations
        
        self.train_annotations = self.annotations[self.annotations['split'] == 'train']
        self.val_annotations = self.annotations[self.annotations['split'] == 'val']
        self.test_annotations = self.annotations[self.annotations['split'] == 'test']
            
    
    def setup(self, stage=None):
        self.train_dataset = AudioDataset(annotations = self.train_annotations, target_len_s = self.target_len_s, target_sr = self.target_sr, target_n_samples = self.target_n_samples, augmentations = self.augmentations, transform = self.transform, train = True, return_labels = self.return_labels)
        self.val_dataset = AudioDataset(annotations = self.val_annotations, target_len_s = self.target_len_s, target_sr = self.target_sr, target_n_samples = self.target_n_samples, augmentations = None, transform = False, train = False, return_labels = self.return_labels)
        if self.return_labels:
            self.test_dataset = AudioDataset(annotations = self.test_annotations, target_len_s = self.target_len_s, target_sr = self.target_sr, target_n_samples = self.target_n_samples, augmentations = None, transform = False, train = False, return_labels = self.return_labels, return_full = True)
            
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=1, num_workers=self.num_workers)
    
    def dummy_call(self):
        dummy = self.train_dataset[0]
        for key in dummy:
            print(key, dummy[key].shape)