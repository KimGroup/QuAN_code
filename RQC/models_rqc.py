import torch
import torch.nn as nn
import torch.nn.functional as F

import random

import math
from flash_attn import flash_attn_func

class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False, flash=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)
        self.flash = flash
        
    def forward(self, Q, K):
        Q, K, V = self.fc_q(Q), self.fc_k(K), self.fc_v(K)
        dim_split = self.dim_V // self.num_heads
        
        if self.flash:
            Q_ = Q.view(Q.size(0), Q.size(1), self.num_heads, dim_split)
            K_ = K.view(K.size(0), K.size(1), self.num_heads, dim_split)
            V_ = V.view(V.size(0), V.size(1), self.num_heads, dim_split)
            O = flash_attn_func(Q_, K_, V_)
            O = O.reshape(O.size(0), O.size(1), -1)
        elif not self.flash:
            Q_ = torch.cat(Q.split(dim_split, 2), 0)
            K_ = torch.cat(K.split(dim_split, 2), 0)
            V_ = torch.cat(V.split(dim_split, 2), 0)
            A = torch.softmax(Q_.bmm(K_.transpose(1,2))/math.sqrt(self.dim_V), 2)
            O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)

        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O

class miniset_MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, miniset:int, minisettype='miniset_A2', ln=False, flash=False):
        super(miniset_MAB, self).__init__()
        self.minisettype = minisettype
        self.mab1 = MAB(dim_Q, dim_K, dim_V, num_heads, ln=ln, flash=flash)
        self.mab2 = MAB(dim_V, dim_V, dim_V, num_heads, ln=ln, flash=flash)
        self.miniset = miniset

    def forward(self, X):
        # import pdb; pdb.set_trace()
        assert X.shape[1] % self.miniset == 0, "set size must be divisable by mini-set size!"
        n_mini = X.shape[1] // self.miniset
        inter_output_list = []

        if self.minisettype=='miniset_A2':
            for i in range(n_mini):
                curr = X[:,i*self.miniset: (i+1)*self.miniset, :]
                inter_output_list.append(self.mab1(curr, curr))
            del curr

            mid_output = []
            for i in range(n_mini):
                curr = inter_output_list[i]
                for j in range(1,n_mini):
                    next = inter_output_list[(j+i)%n_mini]
                    curr = self.mab2(curr, next)
                mid_output.append(curr)
            del inter_output_list
            random.shuffle(mid_output)
            output = mid_output[0]
            for i,mat in enumerate(mid_output[1:]):
                output = self.mab2(output, mat)
            del mid_output

        return output

class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=False, flash=False):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln = ln, flash=flash)
        
    def forward(self, X):
        return self.mab(X, X)
    
class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln = False, flash=False):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln = ln, flash=flash)
        
    def forward(self, X):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)

########################################################################

class QuAN4(nn.Module):
    def __init__(self, set_size, channel, dim_output, kersize, stride, \
                dim_hidden, num_heads, Nr, Nc, p_outputs,  model_loaded_enc=None, ln=True, flash=False):
        super(QuAN4,self).__init__()
        self.Ch = channel
        self.ks = kersize
        self.st = stride
        self.Nr = Nr
        self.Nc = Nc
        self.set_size = set_size
        self.rNr = int((Nr - self.ks)/self.st + 1)
        self.rNc = int((Nc - self.ks)/self.st + 1)
        n = 1

        self.C1 = nn.Conv2d(1, channel, kersize, stride, 0)
        self.bn1 = nn.BatchNorm2d(channel)
        self.sig = nn.Sigmoid()

        self.enc = nn.Sequential(
                SAB(self.Ch*self.rNr*self.rNc, dim_hidden, num_heads, ln=ln, flash=flash),
                SAB(dim_hidden, dim_hidden*n, num_heads, ln=ln, flash=flash))
        if model_loaded_enc:
            self.enc[0] = model_loaded_enc
        
        self.dec = nn.Sequential(
                PMA(dim_hidden*n, num_heads, p_outputs, ln=ln, flash=flash),
                nn.Linear(dim_hidden*n, dim_output),
                nn.Sigmoid())

    def forward(self, X):
        X = self.bn1(self.C1(X.view(-1, 1, self.Nr, self.Nc)))
        X = X.view(-1, self.set_size, self.Ch*self.rNr*self.rNc)
        X = self.sig(self.enc(X))
        return self.dec(X)[:,0]
    
class QuAN2(nn.Module):
    def __init__(self, set_size, channel, dim_output, kersize, stride, \
                dim_hidden, num_heads, Nr, Nc, p_outputs,  model_loaded=None, ln=True, flash=False):
        super(QuAN2,self).__init__()
        self.Ch = channel
        self.ks = kersize
        self.st = stride
        self.Nr = Nr
        self.Nc = Nc
        self.set_size = set_size
        self.rNr = int((Nr - self.ks)/self.st + 1)
        self.rNc = int((Nc - self.ks)/self.st + 1)

        self.C1 = nn.Conv2d(1, channel, kersize, stride, 0)
        self.bn1 = nn.BatchNorm2d(channel)
        self.sig = nn.Sigmoid()

        self.enc = SAB(self.Ch*self.rNr*self.rNc, dim_hidden, num_heads, ln=ln, flash=flash)
        if model_loaded:
            self.enc = model_loaded.enc
        
        self.dec = nn.Sequential(
                PMA(dim_hidden, num_heads, p_outputs, ln=ln, flash=flash),
                nn.Linear(dim_hidden, dim_output),
                nn.Sigmoid())

    def forward(self, X):
        X = self.bn1(self.C1(X.view(-1, 1, self.Nr, self.Nc)))
        X = X.view(-1, self.set_size, self.Ch*self.rNr*self.rNc)
        X = self.sig(self.enc(X))
        return self.dec(X)[:,0]

class QuANn(nn.Module):
    def __init__(self, set_size, channel, dim_output, kersize, stride, \
                dim_hidden, num_heads, Nr, Nc, p_outputs, miniset:int, \
                minisettype, model_loaded=None, ln=True, flash=False):
        super(QuANn,self).__init__()
        self.Ch = channel
        self.ks = kersize
        self.st = stride
        self.Nr = Nr
        self.Nc = Nc
        self.set_size = set_size
        self.miniset = miniset
        self.rNr = int((Nr - self.ks)/self.st + 1)
        self.rNc = int((Nc - self.ks)/self.st + 1)

        self.C1 = nn.Conv2d(1, channel, kersize, stride, 0)
        self.bn1 = nn.BatchNorm2d(channel)
        self.sig = nn.Sigmoid()

        self.enc = miniset_MAB(self.Ch*self.rNr*self.rNc, self.Ch*self.rNr*self.rNc, dim_hidden, \
                                num_heads, miniset, minisettype, ln=ln, flash=False)

        if model_loaded:
            self.enc = model_loaded.enc
        
        self.dec = nn.Sequential(
                PMA(dim_hidden, num_heads, p_outputs, ln=ln, flash=flash),
                nn.Linear(dim_hidden, dim_output),
                nn.Sigmoid())

    def forward(self, X):
        X = self.bn1(self.C1(X.view(-1, 1, self.Nr, self.Nc)))
        X = X.view(-1, self.set_size, self.Ch*self.rNr*self.rNc)
        X = self.sig(self.enc(X))
        return self.dec(X)[:,0]

class PAB(nn.Module):
    def __init__(self, set_size, channel, dim_output,
                 kersize, stride, dim_hidden, num_heads, Nr, Nc,  p_outputs, ln=True, flash=False):
        super(PAB, self).__init__()
        self.Ch = channel
        self.ks = kersize
        self.st = stride
        self.Nr = Nr
        self.Nc = Nc
        self.set_size = set_size
        self.rNr = int((Nr - self.ks)/self.st + 1)
        self.rNc = int((Nc - self.ks)/self.st + 1)
        self.C1 = nn.Conv2d(1, channel, kersize, stride, 0)
        self.bn1 = nn.BatchNorm2d(channel)
        self.sig = nn.Sigmoid()

        self.enc = nn.Sequential(
                nn.Linear(self.Ch*self.rNr*self.rNc, 3*dim_hidden),
                nn.Sigmoid(),
                nn.Linear(3*dim_hidden, dim_hidden),
                nn.Sigmoid())

        self.dec = nn.Sequential(
                PMA(dim_hidden, num_heads, p_outputs, ln=ln, flash=flash),
                nn.Linear(dim_hidden, dim_output),
                nn.Sigmoid())

    def forward(self, X):
        X = self.bn1(self.C1(X.view(-1, 1, self.Nr, self.Nc)))
        X = X.view(-1, self.set_size, self.Ch*self.rNr*self.rNc)
        return self.dec(self.enc(X))[:,0]

class Set_MLP(nn.Module):
        def __init__(self, set_size, channel, dim_output,
                     kersize, stride, dim_hidden, Nr, Nc, ln=True):
            super(Set_MLP, self).__init__()
            self.Ch = channel
            self.ks = kersize
            self.st = stride
            self.Nr = Nr
            self.Nc = Nc
            self.set_size = set_size
            self.rNr = int((Nr - self.ks)/self.st + 1)
            self.rNc = int((Nc - self.ks)/self.st + 1)
            self.C1 = nn.Conv2d(1, self.Ch, self.ks, self.st, 0)
            self.bn1 = nn.BatchNorm2d(self.Ch)
            self.sig = nn.Sigmoid()

            self.enc = nn.Sequential(
                    nn.Linear(self.Ch*self.rNr*self.rNc, 3*dim_hidden),
                    nn.Sigmoid(),
                    nn.Linear(3*dim_hidden, dim_hidden),
                    nn.Sigmoid())
            self.dec = nn.Sequential(
                    nn.Linear(dim_hidden, 3*dim_hidden),
                    nn.Sigmoid(),
                    nn.Linear(3*dim_hidden, dim_output))
        def forward(self, X):
            X = self.bn1(self.C1(X.view(-1, 1, self.Nr, self.Nc)))
            X = X.view(-1, self.set_size, self.Ch*self.rNr*self.rNc)
            X = self.dec(self.enc(X))
            return self.sig(X.sum(axis=1))


class MLP3d(nn.Module):
        def __init__(self, set_size, channel, dim_output,
                     kersize, stride, dim_hidden, Nr, Nc, p_outputs, ln=True):
            super(MLP3d, self).__init__()
            self.Ch = channel
            self.ks = kersize
            self.st = stride
            self.Nr = Nr
            self.Nc = Nc
            self.set_size = set_size
            self.rNr = int((Nr - self.ks)/self.st + 1)
            self.rNc = int((Nc - self.ks)/self.st + 1)
            self.C1 = nn.Conv2d(1, self.Ch, self.ks, self.st, 0)
            self.bn1 = nn.BatchNorm2d(self.Ch)
            self.sig = nn.Sigmoid()
            self.L_set = nn.Linear(self.set_size, p_outputs)

            self.enc = nn.Sequential(
                    nn.Linear(self.Ch*self.rNr*self.rNc, dim_hidden),
                    nn.Sigmoid(),
                    nn.Linear(dim_hidden, dim_hidden),
                    nn.Sigmoid())
            self.dec = nn.Sequential(
                    nn.Linear(dim_hidden, dim_hidden),
                    nn.Sigmoid(),
                    nn.Linear(dim_hidden, dim_output))
        def forward(self, X):
            X = self.bn1(self.C1(X.view(-1, 1, self.Nr, self.Nc)))
            X = X.view(-1, self.set_size, self.Ch*self.rNr*self.rNc)
            X = self.dec(self.enc(X))
            X = self.L_set(X.view(-1, self.set_size))
            return self.sig(X)

class Conv3d(nn.Module):
        def __init__(self, set_size, channel, dim_output,
                     kersize, stride, dim_hidden, Nr, Nc, p_outputs, ln=True):
            super(Conv3d, self).__init__()
            self.Ch = channel
            self.ks = kersize
            self.st = stride
            self.Nr = Nr
            self.Nc = Nc
            self.set_size = set_size
            self.rNr = Nr - 3
            self.rNc = Nc - 3
            self.sig = nn.Sigmoid()
            
            self.L = nn.Linear(16*25*self.rNr*self.rNc, p_outputs)

            self.conv = nn.Sequential(
                    nn.Conv3d(1, 4, kernel_size = (500,2,2), stride=(50,1,1), padding=0),
                    nn.BatchNorm3d(4),
                    nn.Conv3d(4, 4, kernel_size = (50,2,2), stride=(5,1,1), padding=0),
                    nn.BatchNorm3d(4),
                    nn.Conv3d(4, 16, kernel_size = (5,2,2), stride=(1,1,1), padding=0),
                    nn.BatchNorm3d(16))
        def forward(self, X):
            X = self.conv(X.view(-1,1,self.set_size, self.Nr, self.Nc))
            X = X.view(-1, 16*25*self.rNr*self.rNc)
            X = self.L(X)
            return self.sig(X)

########################################################################