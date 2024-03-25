from torch import nn
import pytorch_lightning as pl
import torch


class Probe(nn.Module):
    
    def __init__(self, encoder, layer_dims = [768], num_classes = 50, dropout = 0, activation = 'relu', freeze_encoder = True):
        super(Probe, self).__init__()
        self.encoder = encoder
        self.layer_dims = layer_dims
        self.num_classes = num_classes
        self.dropout = dropout
        self.activation = activation
        
        if freeze_encoder:
            self.encoder.freeze()
            
        self.head = self.build_head()
        
    def build_head(self):
        
        layers = []
        in_features = self.encoder.embed_dim
        for dim in self.layer_dims:
            layers.append(nn.Linear(in_features, dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=self.dropout))
            in_features = dim
            
        layers.append(nn.Linear(in_features, self.num_classes))
        
        return nn.Sequential(*layers)
    
    
    def forward(self, x):
        encoded = self.encoder.extract_features(x)['encoded']
        logits = self.head(x)
        return {
            'logits': logits,
            'encoded': encoded
        }
    
    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
    
        
                
class LightningProbe(pl.LightningModule,Probe):
    
    def __init__(self, task,loss_fn ,optimizer = None, **kwargs):
        super(LightningProbe, self).__init__(**kwargs)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.task = task
        
        self.test_agg = {
            'logits': [],
            'labels': []
        }
        
        self.get_metrics = eval(f'{self.task}_metrics')
    
    def log_metrics(self,metrics, stage = 'train'):
        # metrics is a dictionary containing the metric name and the value
        for k,v in metrics.items():
            if stage == 'train' or stage == 'val':
                self.log(f'{stage}_{k}',v, on_step = True, on_epoch = True, prog_bar = True, sync_dist = True)
            else:
                self.log(f'{stage}_{k}',v, on_step = False, on_epoch = True, prog_bar = True, sync_dist = True)
    
        
    def training_step(self, batch, batch_idx):
        x, y = batch['audio'], batch['labels']
        logits = self(x)['logits']
        loss = self.loss_fn(logits, y)
        
        metrics = self.get_metrics(logits, y, self.num_classes)
        self.log_metrics(metrics, stage = 'train')
        self.log('train_loss', loss, on_step = False, on_epoch = True, prog_bar = True, sync_dist = True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch['audio'], batch['labels']
        logits = self(x)['logits']
        loss = self.loss_fn(logits, y)
        
        metrics = self.get_metrics(logits, y, self.num_classes)
        self.log('val_loss', loss, on_step = False, on_epoch = True, prog_bar = True, sync_dist = True)
        self.log_metrics(metrics, stage = 'val')
        
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch['audio'], batch['labels']
        encoded = self(x)['encoded']
        logits = self.head(encoded.mean(0,keepdim = True))
        self.test_agg['logits'].append(logits)
        self.test_agg['labels'].append(y)
    
    def on_epoch_end(self):
        self.test_agg['logits'] = torch.cat(self.test_agg['logits'], dim=0)
        self.test_agg['labels'] = torch.cat(self.test_agg['labels'], dim=0)
        
        metrics = self.get_metrics(self.test_agg['logits'], self.test_agg['labels'], self.num_classes)
        self.log_metrics(metrics, stage = 'test')
        
        #clear test aggregation
        self.test_agg = {
            'logits': [],
            'labels': []
        }
    
    def configure_optimizers(self):
        if self.optimizer is None:
            return torch.optim.Adam(self.parameters(), lr=1e-4)
        else:
            return self.optimizer(self.parameters())
    
    # when saving a checkpoint, only save the head
    def on_save_checkpoint(self, checkpoint):
        checkpoint['state_dict'] = self.head.state_dict()