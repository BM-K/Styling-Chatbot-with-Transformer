import torch
import torch.nn as nn
import math
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class Transformer(nn.Module):
    def __init__(self, args, SRC_vocab, TRG_vocab):
        super(Transformer, self).__init__()
        self.d_model = args.embedding_dim
        self.n_head = args.nhead
        self.num_encoder_layers = args.nlayers
        self.num_decoder_layers = args.nlayers
        self.dim_feedforward = args.embedding_dim
        self.dropout = args.dropout

        self.SRC_vo = SRC_vocab
        self.TRG_vo = TRG_vocab

        self.pos_encoder = PositionalEncoding(self.d_model, self.dropout)

        self.src_embedding = nn.Embedding(len(self.SRC_vo.vocab), self.d_model)
        self.trg_embedding = nn.Embedding(len(self.TRG_vo.vocab), self.d_model)

        self.transfomrer = torch.nn.Transformer(d_model=self.d_model,
                                                nhead=self.n_head,
                                                num_encoder_layers=self.num_encoder_layers,
                                                num_decoder_layers=self.num_decoder_layers,
                                                dim_feedforward=self.dim_feedforward,
                                                dropout=self.dropout)
        self.proj_vocab_layer = nn.Linear(
            in_features=self.dim_feedforward, out_features=len(self.TRG_vo.vocab))

        #self.apply(self._initailze)

    def forward(self, en_input, de_input):
        x_en_embed = self.src_embedding(en_input.long()) * math.sqrt(self.d_model)
        x_de_embed = self.trg_embedding(de_input.long()) * math.sqrt(self.d_model)
        x_en_embed = self.pos_encoder(x_en_embed)
        x_de_embed = self.pos_encoder(x_de_embed)

        # Masking
        src_key_padding_mask = en_input == self.SRC_vo.vocab.stoi['<pad>']
        tgt_key_padding_mask = de_input == self.TRG_vo.vocab.stoi['<pad>']
        memory_key_padding_mask = src_key_padding_mask
        tgt_mask = self.transfomrer.generate_square_subsequent_mask(de_input.size(1))

        x_en_embed = torch.einsum('ijk->jik', x_en_embed)
        x_de_embed = torch.einsum('ijk->jik', x_de_embed)

        feature = self.transfomrer(src=x_en_embed,
                                   tgt=x_de_embed,
                                   src_key_padding_mask=src_key_padding_mask,
                                   tgt_key_padding_mask=tgt_key_padding_mask,
                                   memory_key_padding_mask=memory_key_padding_mask,
                                   tgt_mask=tgt_mask.to(device))

        logits = self.proj_vocab_layer(feature)
        logits = torch.einsum('ijk->jik', logits)

        return logits

    def _initailze(self, layer):
        if isinstance(layer, (nn.Linear)):
            nn.init.kaiming_uniform_(layer.weight)

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout, max_len=15000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau

class GradualWarmupScheduler(_LRScheduler):

    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier <= 1.:
            raise ValueError('multiplier should be greater than 1.')
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch if epoch != 0 else 1  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
        if self.last_epoch <= self.total_epoch:
            warmup_lr = [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.total_epoch)

    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)
