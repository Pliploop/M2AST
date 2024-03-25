from torch.utils.data import Dataset
from m2ast.dataloading.loading_utils import load_audio_chunk



class AudioDataset(Dataset):
    def __init__(self, annotations, target_len_s, target_sr,target_n_samples = 2**19-1, augmentations=None,transform = False, train = True, return_labels=False):
        self.annotations = annotations
        self.target_len_s = target_len_s
        self.target_sr = target_sr
        self.target_n_samples = target_n_samples if target_n_samples is not None else target_len_s*target_sr
        self.transform = transform
        self.augmentations = augmentations
        self.train = train
        self.return_labels = return_labels
        
        
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        path = self.annotations.iloc[idx]['file_path']
        if self.return_labels:
            labels = self.annotations.iloc[idx]['labels']
            
        audio = load_audio_chunk(path, self.target_n_samples, self.target_sr)
        
        if self.transform and self.train and self.augmentations is not None:
            audio = self.augmentations(audio)
        
        # mono
        audio = audio.mean(dim=0, keepdim=True)
        
        if self.return_labels:
            return {
                'audio': audio,
                'labels': labels
            }
        else:
            return {
                'audio': audio
            }
            
        