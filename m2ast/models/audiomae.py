# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn
import pytorch_lightning as pl

#from timm.models.vision_transformer import PatchEmbed, Block
from timm.models.vision_transformer import Block
from m2ast.models.modules.pos import get_2d_sincos_pos_embed, get_2d_sincos_pos_embed_flexible, get_1d_sincos_pos_embed_from_grid
from m2ast.models.modules.patch import PatchEmbed_new, PatchEmbed_org
from timm.models.swin_transformer import SwinTransformerBlock
from m2ast.models.modules.frontend import Melgram
import wandb

class AudioMAE(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=(128,1024), patch_size=16, stride=16, in_chans=1,
                 embed_dim=768, depth=12, num_heads=12,
                 decoder_embed_dim=512, decoder_depth=2, decoder_num_heads=16,
                 mlp_ratio=4., norm_pix_loss=False, 
                 audio_exp=True, alpha=0.0, temperature=.2, mode=0, contextual_depth=8,
                 use_custom_patch=False, split_pos=False, pos_trainable=False, use_nce=False, beta=4.0, decoder_mode=0,
                 mask_t_prob=0.6, mask_f_prob=0.5, mask_2d=False,
                 epoch=0, no_shift=False, mask_ratio=0.8, n_fft=2048, hop_length=512, n_mels=128, sr=44100,
                 **kwargs
                 ):
        super().__init__()

        self.frontend = Melgram(n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, sr=sr)

        self.audio_exp=audio_exp
        self.embed_dim = embed_dim
        self.decoder_embed_dim = decoder_embed_dim
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        if use_custom_patch:
            print(f'Use custom patch_emb with patch size: {patch_size}, stride: {stride}')
            self.patch_embed = PatchEmbed_new(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, stride=stride)
        else:
            self.patch_embed = PatchEmbed_org(img_size, patch_size, in_chans, embed_dim)
        self.use_custom_patch = use_custom_patch
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        #self.split_pos = split_pos # not useful
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=pos_trainable)  # fixed sin-cos embedding
        print(f'pos_embed shape: {self.pos_embed.shape}')
        norm_layer = nn.LayerNorm
        self.norm = norm_layer(embed_dim)
        self.encoder_depth = depth
        self.contextual_depth = contextual_depth
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=pos_trainable)  # fixed sin-cos embedding


        self.no_shift=no_shift
        self.mask_ratio = mask_ratio


        self.decoder_mode = decoder_mode
        if self.use_custom_patch: # overlapped patches as in AST. Similar performance yet compute heavy
            window_size= (6,6)
            feat_size = (102,12)
        else:
            window_size= (4,4)
            feat_size = (64,8)                
        if self.decoder_mode == 1:
            decoder_modules = []
            for index in range(16):
                if self.no_shift:
                    shift_size = (0,0)
                else:
                    if (index % 2) == 0:
                        shift_size = (0,0)
                    else:
                        shift_size = (2,0)
                    #shift_size = tuple([0 if ((index % 2) == 0) else w // 2 for w in window_size])
                decoder_modules.append(
                    SwinTransformerBlock(
                        dim=decoder_embed_dim,
                        num_heads=decoder_num_heads,
                        feat_size=feat_size,
                        window_size=window_size,
                        shift_size=shift_size,
                        mlp_ratio=mlp_ratio,
                        drop=0.0,
                        drop_attn=0.0,
                        drop_path=0.0,
                        extra_norm=False,
                        sequential_attn=False,
                        norm_layer=norm_layer, #nn.LayerNorm,
                    )
                )
            self.decoder_blocks = nn.ModuleList(decoder_modules)        
        else:
            # Transfomer
            self.decoder_blocks = nn.ModuleList([
                Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
                for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch

        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.patch_size=patch_size
        self.stride=stride

        # audio exps
        self.alpha = alpha
        self.T = temperature
        self.mode = mode
        self.use_nce = use_nce
        self.beta = beta

        self.mask_t_prob=mask_t_prob
        self.mask_f_prob=mask_f_prob
        self.mask_2d=mask_2d

        self.epoch = epoch

        self.initialize_weights()
        self.first_run=True
        

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        if self.audio_exp:
            pos_embed = get_2d_sincos_pos_embed_flexible(self.pos_embed.shape[-1], self.patch_embed.patch_hw, cls_token=True)    
        else:
            pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        print(f'pos_embed shape: {pos_embed.shape}')
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        if self.audio_exp:   
            decoder_pos_embed = get_2d_sincos_pos_embed_flexible(self.decoder_pos_embed.shape[-1], self.patch_embed.patch_hw, cls_token=True)
        else:
            decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        L = (H/p)*(W/p)
        """
        p = self.patch_embed.patch_size[0]
        #assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
        
        if self.audio_exp:
            if self.use_custom_patch: # overlapped patch
                h,w = self.patch_embed.patch_hw
                # todo: fixed h/w patch size and stride size. Make hw custom in the future
                x = imgs.unfold(2, self.patch_size, self.stride).unfold(3, self.patch_size, self.stride) # n,1,H,W -> n,1,h,w,p,p
                x = x.reshape(shape=(imgs.shape[0], h*w, p**2 * 1))
                #x = imgs.reshape(shape=(imgs.shape[0], 1, h, p, w, p))
                #x = torch.einsum('nchpwq->nhwpqc', x)
                #x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 1))
            else:
                h = imgs.shape[2] // p
                w = imgs.shape[3] // p
                #h,w = self.patch_embed.patch_hw
                x = imgs.reshape(shape=(imgs.shape[0], 1, h, p, w, p))
                x = torch.einsum('nchpwq->nhwpqc', x)
                x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 1))
        else:
            h = w = imgs.shape[2] // p
            x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
            x = torch.einsum('nchpwq->nhwpqc', x)
            x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))

        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        specs: (N, 1, H, W)
        """
        p = self.patch_embed.patch_size[0]    
        h = 128//p
        w = 1024//p
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 1))
        x = torch.einsum('nhwpqc->nchpwq', x)
        specs = x.reshape(shape=(x.shape[0], 1, h * p, w * p))
        return specs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def random_masking_2d(self, x, mask_t_prob, mask_f_prob):
        """
        2D: Spectrogram (msking t and f under mask_t_prob and mask_f_prob)
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        if self.use_custom_patch: # overlapped patch
            T=101
            F=12
        else:            
            T=64
            F=8
        #x = x.reshape(N, T, F, D)
        len_keep_t = int(T * (1 - mask_t_prob))
        len_keep_f = int(F * (1 - mask_f_prob))

        # noise for mask in time
        noise_t = torch.rand(N, T, device=x.device)  # noise in [0, 1]
        # sort noise for each sample aling time
        ids_shuffle_t = torch.argsort(noise_t, dim=1) # ascend: small is keep, large is remove
        ids_restore_t = torch.argsort(ids_shuffle_t, dim=1) 
        ids_keep_t = ids_shuffle_t[:,:len_keep_t]
        # noise mask in freq
        noise_f = torch.rand(N, F, device=x.device)  # noise in [0, 1]
        ids_shuffle_f = torch.argsort(noise_f, dim=1) # ascend: small is keep, large is remove
        ids_restore_f = torch.argsort(ids_shuffle_f, dim=1) 
        ids_keep_f = ids_shuffle_f[:,:len_keep_f] #

        # generate the binary mask: 0 is keep, 1 is remove
        # mask in freq
        mask_f = torch.ones(N, F, device=x.device)
        mask_f[:,:len_keep_f] = 0
        mask_f = torch.gather(mask_f, dim=1, index=ids_restore_f).unsqueeze(1).repeat(1,T,1) # N,T,F
        # mask in time
        mask_t = torch.ones(N, T, device=x.device)
        mask_t[:,:len_keep_t] = 0
        mask_t = torch.gather(mask_t, dim=1, index=ids_restore_t).unsqueeze(1).repeat(1,F,1).permute(0,2,1) # N,T,F
        mask = 1-(1-mask_t)*(1-mask_f) # N, T, F

        # get masked x
        id2res=torch.Tensor(list(range(N*T*F))).reshape(N,T,F).to(x.device)
        id2res = id2res + 999*mask # add a large value for masked elements
        id2res2 = torch.argsort(id2res.flatten(start_dim=1))
        ids_keep=id2res2.flatten(start_dim=1)[:,:len_keep_f*len_keep_t]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        ids_restore = torch.argsort(id2res2.flatten(start_dim=1))
        mask = mask.flatten(start_dim=1)

        return x_masked, mask, ids_restore


    def forward_encoder(self, x, mask_ratio, mask_2d=False):
        
        imgs = self.frontend(x)
        # embed patches
        x = self.patch_embed(imgs)
        ## pretty print some stuff when first run 
        if self.first_run:
            print(f'x shape: {x.shape}')
            print(f'pos_embed shape: {self.pos_embed.shape}')

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        if mask_2d:
            x, mask, ids_restore = self.random_masking_2d(x, mask_t_prob=self.mask_t_prob, mask_f_prob=self.mask_f_prob)
        else:
            x, mask, ids_restore = self.random_masking(x, mask_ratio)

        if self.first_run:
            print(f'x shape after masking: {x.shape}')
            print(f'mask shape after masking: {mask.shape}')
            print(f'ids_restore shape after masking: {ids_restore.shape}')
        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        #emb = self.encoder_emb(x)

        return x, mask, ids_restore, None, imgs


    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token
        

        # add pos embed
        x = x + self.decoder_pos_embed
        
        if self.decoder_mode != 0:
            B,L,D=x.shape
            x = x[:,1:,:]
            if self.use_custom_patch:
                x = x.reshape(B,101,12,D)
                x = torch.cat([x,x[:,-1,:].unsqueeze(1)],dim=1) # hack
                x = x.reshape(B,1224,D)
        if self.decoder_mode > 3: # mvit
            x = self.decoder_blocks(x)
        else:
            # apply Transformer blocks
            for blk in self.decoder_blocks:
                x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        pred = self.decoder_pred(x)
        if self.first_run:
            print(f'x shape after decoder: {x.shape}')

        # remove cls token
        if self.decoder_mode != 0:
            if self.use_custom_patch:
                pred = pred.reshape(B,102,12,256)
                pred = pred[:,:101,:,:]
                pred = pred.reshape(B,1212,256)
            else:
                pred = pred
        else:
            pred = pred[:, 1:, :]
        return pred, None, None #emb, emb_pixel

    def forward_loss(self, imgs, pred, mask, norm_pix_loss=False):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify(imgs)
        if norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2 # MSE loss
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss      

    def forward(self, imgs, mask_ratio=None):
        if mask_ratio is None:
            mask_ratio = self.mask_ratio
        emb_enc, mask, ids_restore, _, imgs = self.forward_encoder(imgs, mask_ratio, mask_2d=self.mask_2d)
        pred, _, _ = self.forward_decoder(emb_enc, ids_restore)  # [N, L, p*p*3]
        loss_recon = self.forward_loss(imgs, pred, mask, norm_pix_loss=self.norm_pix_loss)
        self.first_run=False
        return {
            'loss': loss_recon,
            'encoded': emb_enc,
            'decoded': pred,
            'mask': mask,
            'ids_restore': ids_restore,
            'imgs': imgs
        }
    
    def extract_features(self, audio, freeze=False):
        
        with torch.no_grad():
            emb_enc, _, _, _,_ = self.forward_encoder(audio, mask_ratio=0.0)
        return {
            'encoded': emb_enc,
        }

        
    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
            
    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True
        self.train()
    
    def discard_decoder(self):
        self.decoder_blocks = None



def audiomae_small(**kwargs):
    model = AudioMAE(
        patch_size=16, embed_dim=384, depth=12, num_heads=6,
        decoder_embed_dim=512, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def audiomae_base(**kwargs):
    model = AudioMAE(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def audiomae_large(**kwargs):
    model = AudioMAE(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def audiomae_huge(**kwargs):
    model = AudioMAE(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

# make a pytorch lightning module for training purposes only

class LightningAudioMAE(AudioMAE,pl.LightningModule):
    def __init__(self, optimizer = None, **kwargs):
        super().__init__(**kwargs)
        self.optimizer = optimizer

    def training_step(self, batch, batch_idx):
        audio = batch['audio']
        output = self(audio, mask_ratio=self.mask_ratio)
        
        loss_recon = output['loss']
        
        self.log('train_loss', loss_recon, on_epoch=True, on_step=True, prog_bar=True, logger=True, sync_dist=True)
        
        # for training purposes, reconstruct the original image every 1000 steps
        if self.global_step % 2000 == 0 and self.logger is not None:
            with torch.no_grad():
                sample_img, sample_pred, sample_masked_img = self.reconstruct_batch(audio)
                
                self.logger.experiment.log({
                    "reconstructed": [wandb.Image(sample_pred[::-1,:], caption="reconstructed")],
                    "original": [wandb.Image(sample_img[::-1,:], caption="original")],
                    "masked": [wandb.Image(sample_masked_img[::-1,:], caption="masked")],
                })
            
        return loss_recon

    def reconstruct_batch(self, audio):
        with torch.no_grad():
            emb_enc, mask, ids_restore, _, imgs = self.forward_encoder(audio, mask_ratio=self.mask_ratio)
            pred, _, _ = self.forward_decoder(emb_enc, ids_restore)
            unpatchified_pred = self.unpatchify(pred)
            sample_img = imgs[0].squeeze().detach().cpu().numpy()
            sample_pred = unpatchified_pred[0].squeeze().detach().cpu().numpy()
            # convert mask to shape (H, W) knowing that it is a 1D mask of unrolled patches
            mask = mask.unsqueeze(-1).repeat(1,1, self.patch_size**2)
            mask = self.unpatchify(mask)[0].squeeze().detach().cpu().numpy()
            sample_masked_img = sample_img * (1 - mask)
            sample_pred = sample_pred * mask + sample_img * (1-mask)
        return sample_img, sample_pred, sample_masked_img
        

    def validation_step(self, batch, batch_idx):
        audio = batch['audio']
        output = self(audio, mask_ratio=self.mask_ratio)
        loss_recon = output['loss']
        self.log('val_loss', loss_recon, on_epoch=True, on_step=False, prog_bar=True, logger=True, sync_dist=True)

    def configure_optimizers(self):
        if self.optimizer is not None:
            return self.optimizer(self.parameters())
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4)
        return optimizer