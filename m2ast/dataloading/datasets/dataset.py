from torch.utils.data import Dataset
from m2ast.dataloading.loading_utils import load_audio_chunk, load_and_split



class AudioDataset(Dataset):
    def __init__(self, annotations, target_len_s, target_sr,target_n_samples = 2**19-1, augmentations=None,transform = False, train = True, return_labels=False, return_full = False):
        self.annotations = annotations
        self.target_len_s = target_len_s
        self.target_sr = target_sr
        self.target_n_samples = target_n_samples if target_n_samples is not None else target_len_s*target_sr
        self.transform = transform
        self.augmentations = augmentations
        self.train = train
        self.return_labels = return_labels
        self.return_full  =     return_full # return full audio file for test dataloader
        
        
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        path = self.annotations.iloc[idx]['file_path']
        if self.return_labels:
            labels = self.annotations.iloc[idx]['labels']
        
        if self.return_full:
            audio = load_and_split(path, self.target_sr, self.target_n_samples)
            audio = audio.mean(dim=1, keepdim=True)
        else:
            audio = load_audio_chunk(path, self.target_n_samples, self.target_sr)
            audio = audio.mean(dim=0, keepdim=True)
        
        if self.transform and self.train and self.augmentations is not None:
            audio = self.augmentations(audio)
        
        # mono
        
        if self.return_labels:
            return {
                'audio': audio,
                'labels': labels
            }
        else:
            return {
                'audio': audio
            }
            
        