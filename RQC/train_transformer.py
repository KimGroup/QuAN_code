import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable

import numpy as np
import math, time, copy
import time

from collections import OrderedDict
import os
from data_transformer import *


import wandb
import argparse

class Classifier(nn.Module):
    def __init__(self, encoder, src_embed, generator, Lvh, Vsrc, Vtgt):
        super(Classifier, self).__init__()
        self.encoder = encoder
        self.src_embed = src_embed
        self.generator = generator
        self.Lvh = Lvh
        self.Vsrc = Vsrc
        self.Vtgt = Vtgt
        self.V = Vsrc + Vtgt

    def forward(self, src, src_mask=None):
        src = self.encoder(self.src_embed(src), src_mask)
        return self.generator(src)
    

class Generator(nn.Module):
    def __init__(self, size, Vtgt):
        super(Generator, self).__init__()
        self.proj = nn.Linear(size, Vtgt)

    def forward(self, x):
        return F.log_softmax(self.proj(x.flatten(start_dim=1, end_dim=-1)), dim=-1)

# Basic modules
def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class PositionalEncoding1D(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding1D, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len,dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2,dtype=torch.float) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], 
                         requires_grad=False)
        return self.dropout(x)

class PositionalEncoding2D(nn.Module):
    def __init__(self, d_model, dropout, Lx=100, Ly=100):
        super(PositionalEncoding2D, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(Lx, Ly, d_model)
        d2 = int(d_model/2)
        px = torch.arange(0, Lx,dtype=torch.float).unsqueeze(1)
        py = torch.arange(0, Ly,dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d2, 2, dtype=torch.float) *
                             -(math.log(10000.0) / d2))

        pe[:, :, 0:d2:2] = torch.sin(px * div_term).unsqueeze(1).repeat(1, Ly, 1)
        pe[:, :, 1:d2:2] = torch.cos(px * div_term).unsqueeze(1).repeat(1, Ly, 1)
        pe[:, :, d2::2] = torch.sin(py * div_term).unsqueeze(0).repeat(Lx, 1, 1)
        pe[:, :, 1+d2::2] = torch.cos(py * div_term).unsqueeze(0).repeat(Lx, 1, 1)

        pe = pe.flatten(start_dim=0, end_dim=1)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], 
                         requires_grad=False)
        return self.dropout(x)



# Encoder
class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask=None):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

# Learning modules: Attention, Feed-forward, Embedding
class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value,# mask=mask, 
                                 dropout=self.dropout)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class Embeddings(nn.Module):
    def __init__(self, d_model, Vsrc):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(Vsrc, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)




class LabelSmoothing(nn.Module):
    def __init__(self, size, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='batchmean')
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
        
    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / self.size)
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))


class SimpleLossCompute:
    def __init__(self, generator, criterion, optimizer=None):
        self.generator = generator
        self.criterion = criterion
        self.optimizer = optimizer
        
    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(x, y.contiguous().view(-1)) / norm
        loss.backward()
        if self.optimizer is not None:
            self.optimizer.step()
            self.optimizer.zero_grad()
        return loss.item() * norm


# Make model. Set hyperparameters.
def make_model(Lv, Lh, Vsrc, Vtgt, Nl=3, d_model=16, d_ff=32, h=4, dropout=0.0):
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model, dropout)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding2D(d_model, dropout, Lx=Lv, Ly=Lh)

    # Classifier(encoder, src_embed, generator, Lvh, Vsrc, Vcls)
    model = Classifier(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), Nl),
        nn.Sequential(Embeddings(d_model, Vsrc), c(position)),
        Generator(Lv*Lh*d_model, Vtgt), torch.tensor([Lv, Lh]), Vsrc, Vtgt)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model







def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-set", help = "set size", type = int, default = None)
    parser.add_argument("-n_mini", help = "# of mini set", type = int, default = 5)

    parser.add_argument("-d1", help = "depth 1", type = int, default = None)
    parser.add_argument("-d2", help = "depth 2", type = int, default = None)
    parser.add_argument("-nr", help = "height", type = int, default = None)
    parser.add_argument("-nc", help = "width", type = int, default = None)

    parser.add_argument("-prefix", help = "prefix", type = str, default = '~/QuAN/')
    parser.add_argument("-saveprefix", help = "save folder prefix", type = str, default = 'g2_saved_models_rqc')
    parser.add_argument("-epoch", help = "total epochs", type = int, default = 100)
    parser.add_argument("-batchsize", type=int, default=50) 

    parser.add_argument("-modelnum", type = str, default = 'c11')
    parser.add_argument("-minisettype", type = str, default = None)
    parser.add_argument("-hdim", type = int, default = 64)
    parser.add_argument("-nhead", type = int, default = 16)

    parser.add_argument("-ch", type = int, default = 32)
    parser.add_argument("-ker", type = int, default = 2)
    parser.add_argument("-st", type = int, default = 1)
    parser.add_argument("-dim_outputs", type=int, default=1)
    parser.add_argument("-p_outputs", type=int, default=1)
    parser.add_argument("-num_inds", type=int, default=1000)
    parser.add_argument("-run", type = int, default = None)

    parser.add_argument("-nsy", type = str, default = 'google')
    parser.add_argument("-lr", type = float, default = 1e-4)
    parser.add_argument("-lrn", type = int, default = 0)
    parser.add_argument("-lrstep", type = float, default = 0.65)
    parser.add_argument("-lrstepsize", type = int, default = 200)
    parser.add_argument("-optim", type = str, default = None)
    parser.add_argument("-prev", type = str, default = None)
    parser.add_argument("-pprev", type = str, default = None)
    parser.add_argument("-logprev", type = str, default = None)
    parser.add_argument("-circ", type = str, help='2-qubit circuit type, CZ (C) or F-sim (F)', default = 'F')
    
    parser.add_argument("-opsc", type = str, default = 'yes')

    parser.add_argument("-wandb_name", type = str)
    parser.add_argument("-debug", action='store_true', default=False)
    parser.add_argument("-testonly", action='store_true',default=False)
    
    parser.add_argument("-shuffle_epoch", type=int, default=1, help='shuffling dataset every n epoch')
    parser.add_argument("-nprandseed", action='store_true', default=False, help='random seed fix?')
    args = parser.parse_args()

    return args
def is_non_zero_file(fpath, verb=False):
    result = os.path.isfile(fpath) and os.path.getsize(fpath) > 0
    if verb: print(f'exist={result}', fpath)
    return result

def main():

    args = get_arguments()

    if args.nprandseed: 
        print('np.random.seed(42)')
        np.random.seed(42)

    Nt = args.nr * args.nc
        
    os.makedirs(f'{args.saveprefix}',exist_ok=True)
    os.makedirs(f'{args.saveprefix}/{args.wandb_name}',exist_ok=True)

    minisetsize = args.set // args.n_mini

    if args.geo=='R':
        model_name = f'{args.nr}x{args.nc}_{args.circ}_p{args.nsy}_{args.d1}vs{args.d2}'\
                    +f'-{args.modelnum}_set{args.set}_h{args.hdim}nh{args.nhead}_ch{args.ch}ker{args.ker}st{args.st}'
    if args.n_mini!=1:
        model_name += f'_#miniset{args.n_mini}'
        
    print("Storing as:", model_name)

    if args.modelnum == 'tre': model = make_model(Lv=5, Lh=5, Vsrc=2, Vtgt=1, Nl=6, d_model=16, d_ff = 32, h=4)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'total parameters: {total_params}') 

    if args.prev != None:
        print("Previous model:", args.prev)
        state_dict = torch.load(args.prev)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if 'module' not in k:
                k = 'module.'+k
            else:
                k = k.replace('module.module', 'module')
            new_state_dict[k]=v
        model.load_state_dict(new_state_dict)

        if args.pprev == None:
            argsprev_model_name = (args.prev).replace('model_vl_', 'p_').replace('model_va_', 'p_').replace('model_e_', 'p_').replace('.pth', '.txt')
            c_p = np.loadtxt(f'{argsprev_model_name}', dtype = 'float64').astype(int)
        else: c_p = np.loadtxt(args.pprev, dtype = 'float64').astype(int)
        
    if args.prev == None:
        f = open(f'{args.saveprefix}/{args.wandb_name}/log_{model_name}.txt', 'w')
        f.close()
        c_p_name = f'{args.saveprefix}/{args.wandb_name}/p_{model_name}.txt'
        if args.pprev != None:
            if is_non_zero_file(c_p_name, verb=True): 
                c_p = np.loadtxt(c_p_name, dtype = 'float64').astype(int)
        else:
            c_p = np.random.permutation(50)
            np.savetxt(c_p_name, c_p, fmt='%d')
        
        
    total_epoch = args.epoch
    if args.nr==6: stepsize = 50
    else: stepsize = 100
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr*0.65**args.lrn)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=stepsize, gamma=0.65)
    if args.opsc != None: print(scheduler.state_dict())
    print("optimizer:", args.optim)

    criterion = nn.BCELoss()

    if args.wandb_name and not args.debug and not args.testonly:
        wandb.init(project='google_rqc', name=args.wandb_name)
        wandb.log({'# of parameters': total_params})

    start = time.time()

    if args.nsy == 'google': 
        if args.d1==args.d2: x1, x2, x3v, x4v, x3t, x4t = gen_data_google_20vs20(args, Nt, c_p)
        else: x1, x2, x3v, x4v, x3t, x4t = gen_data_google(args, Nt, c_p)
    elif args.nsy == '0': 
        if args.d1==args.d2: x1, x2, x3v, x4v, x3t, x4t = gen_data_simulated_20vs20(args, Nt, c_p)
        else: x1, x2, x3v, x4v, x3t, x4t = gen_data_simulated(args, Nt, c_p)
    x1, x2, x3v, x4v, x3t, x4t = x1.astype(bool), x2.astype(bool), x3v.astype(bool), x4v.astype(bool), x3t.astype(bool), x4t.astype(bool)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device); model.to(device)
    print(model)
    
    best_acc_v = 0.0
    best_loss_v = 50
    best_acc_t = 0.0
    best_loss_t = 50
    for epoch in range (total_epoch):
        running_loss, val_loss, test_loss = 0.0, 0.0, 0.0
        running_acc, val_acc, test_acc= 0.0, 0.0, 0.0
        val_acc0, val_acc1 = 0.0, 0.0
        test_acc0, test_acc1 = 0.0, 0.0

        if args.logprev == None: f = open(f'{args.saveprefix}/{args.wandb_name}/log_{model_name}.txt', 'a')
        else: f = open(args.logprev, 'a')

        if (total_epoch-epoch)%args.shuffle_epoch==0:
            x, y = gen_data_sample(args.set, x1, x2, Nt)
            train_loader = DataLoader(dataset = training_set(x, y), batch_size=args.batchsize, shuffle=True)
            del x, y
            print('data sampled')
    
        model.train()
        for i,data in enumerate(train_loader):
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs.int())
            del inputs
            optimizer.zero_grad()
            loss = criterion(outputs.squeeze(), labels.float().squeeze())
            if torch.isnan(loss).any(): break
            loss.backward(); optimizer.step()
            loss.detach().cpu().numpy()
            
            running_loss += loss.item()
            pred_classes = torch.cat([1-outputs, outputs], axis=1).argmax(axis=1)
            running_acc += torch.count_nonzero(torch.eq(pred_classes, labels.squeeze())).item()/len(labels)
        running_loss /= len(train_loader)
        running_acc /= len(train_loader)
        torch.cuda.empty_cache()
        
        learningratearg = optimizer.param_groups[0]["lr"]
        if args.opsc != None:
            scheduler.step(); learningratearg = scheduler.get_last_lr()
        if (total_epoch-epoch)%(args.shuffle_epoch)==0:
            xv, yv = gen_testdata_sample(args.set, x3v, x4v, Nt)
            val_loader = DataLoader(dataset = testing_set(xv, yv), batch_size=args.batchsize, shuffle=True)
            del xv, yv

        model.eval()
        for j,valdata in enumerate(val_loader):
            val_inputs, val_labels = valdata[0].to(device), valdata[1].to(device)
            val_outputs = model(val_inputs.int())
            del val_inputs
            loss_ = criterion(val_outputs.squeeze(), val_labels.float().squeeze())
            loss_.detach().cpu().numpy()
            
            val_pred_classes = torch.cat([1-val_outputs, val_outputs], axis=1).argmax(axis=1)
            val_loss += loss_.item()
            val_acc += torch.count_nonzero(torch.eq(val_pred_classes, val_labels.squeeze())).item()/len(val_labels)
            del loss_, val_outputs
            m0 = val_labels.squeeze() < 0.5
            m1 = val_labels.squeeze() > 0.5
            if len(val_labels.squeeze()[m0]) == 0: val_acc0 += 0.0
            else: val_acc0 += torch.count_nonzero(torch.eq(val_pred_classes[m0], val_labels.squeeze()[m0])).item()/len(val_labels.squeeze()[m0])
            if len(val_labels.squeeze()[m1]) == 0: val_acc1 += 0.0
            else: val_acc1 += torch.count_nonzero(torch.eq(val_pred_classes[m1], val_labels.squeeze()[m1])).item()/len(val_labels.squeeze()[m1])
        val_loss /= len(val_loader)
        val_acc /= len(val_loader)
        val_acc0 /= len(val_loader)
        val_acc1 /= len(val_loader)

        if (total_epoch-epoch)%(args.shuffle_epoch)==0:
            xt, yt = gen_testdata_sample(args.set, x3t, x4t, Nt)
            test_loader = DataLoader(dataset = testing_set(xt, yt), batch_size=args.batchsize, shuffle=True)
            del xt, yt

        for j,testdata in enumerate(test_loader):
            test_inputs, test_labels = testdata[0].to(device), testdata[1].to(device)
            test_outputs = model(test_inputs.int())
            del test_inputs
            loss_ = criterion(test_outputs.squeeze(), test_labels.float().squeeze())
            loss_.detach().cpu().numpy()
            
            test_pred_classes = torch.cat([1-test_outputs, test_outputs], axis=1).argmax(axis=1)
            test_loss += loss_.item()
            test_acc += torch.count_nonzero(torch.eq(test_pred_classes, test_labels.squeeze())).item()/len(test_labels)
            del loss_, test_outputs
            m0 = test_labels.squeeze() < 0.5
            m1 = test_labels.squeeze() > 0.5
            if len(test_labels.squeeze()[m0]) == 0: test_acc0 += 0.0
            else: test_acc0 += torch.count_nonzero(torch.eq(test_pred_classes[m0], test_labels.squeeze()[m0])).item()/len(test_labels.squeeze()[m0])
            if len(test_labels.squeeze()[m1]) == 0: test_acc1 += 0.0
            else: test_acc1 += torch.count_nonzero(torch.eq(test_pred_classes[m1], test_labels.squeeze()[m1])).item()/len(test_labels.squeeze()[m1])
        test_loss /= len(test_loader)
        test_acc /= len(test_loader)
        test_acc0 /= len(test_loader)
        test_acc1 /= len(test_loader)

        if args.wandb_name:
            wandb.log({'epoch': epoch, 'train_loss': running_loss, 'train_acc': running_acc, \
                'val_loss': val_loss, 'val_acc': val_acc, "val_acc0": val_acc0, "val_acc1": val_acc1, \
                'test_loss': test_loss, 'test_acc': test_acc, "test_acc0": test_acc0, "test_acc1": test_acc1})
        if best_acc_v < val_acc:
            torch.save(model.state_dict(), f'{args.saveprefix}/{args.wandb_name}/model_va_{model_name}.pth')
            best_acc = val_acc
        if best_loss_v > val_loss:
            torch.save(model.state_dict(), f'{args.saveprefix}/{args.wandb_name}/model_vl_{model_name}.pth')
            best_loss = val_loss
        if best_acc_t < test_acc:
            torch.save(model.state_dict(), f'{args.saveprefix}/{args.wandb_name}/model_ta_{model_name}.pth')
            best_acc_t = test_acc
        if best_loss_t > test_loss:
            torch.save(model.state_dict(), f'{args.saveprefix}/{args.wandb_name}/model_tl_{model_name}.pth')
            best_loss_t = test_loss
        torch.save(model.state_dict(), f'{args.saveprefix}/{args.wandb_name}/model_e_{model_name}.pth')
        np.savetxt(f, np.array([running_loss, val_loss, test_loss, running_acc, val_acc, test_acc, val_acc0, test_acc0, val_acc1, test_acc1]), newline=" ")
        f.write("\n")
        f.close()

        print(f'{args.nsy[0]}, {args.nr}*{args.nc}, {args.d1}vs{args.d2} - [#{epoch}/{total_epoch}], lr = {learningratearg}')
        print(f'Loss// Tr: {running_loss:.4f}, {running_acc*100:.4f}')
        print(f'Accy// Vl: {val_loss:.4f}%, {val_acc*100:.4f}%')
        print(f'Accy// Vl0/1: {val_acc0*100:.4f}%, {val_acc1*100:.4f}%')
        print(f'Accy// Te: {test_loss:.4f}%, {test_acc*100:.4f}%')
        
        torch.cuda.empty_cache()

    print(f'Time: {time.time()-start} s, {(time.time()-start)/60} min, {(time.time()-start)/3600} hour \n\n')
if __name__ == "__main__":
    main()