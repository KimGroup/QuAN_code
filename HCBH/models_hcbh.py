import torch
import torch.nn as nn

import math

class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V, elementwise_affine=False)
            self.ln1 = nn.LayerNorm(dim_V, elementwise_affine=False)
        self.fc_o = nn.Linear(dim_V, dim_V)
        self.sig = nn.Sigmoid()
        
    def forward(self, Q, K):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)
        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)
        
        A = torch.softmax(Q_.bmm(K_.transpose(1,2))/math.sqrt(self.dim_V), 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + self.sig(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O
    
class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln = ln)
        
    def forward(self, X):
        return self.mab(X, X)
    
class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln = False):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln = ln)
        
    def forward(self, X):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)
    
    
########################################################################

class QuAN8(nn.Module):
    def __init__(self, set_size, channel, dim_output,
                 kersize, stride, dim_hidden, num_heads, Nt, p_outputs,  ln=True):
        super(QuAN8,self).__init__()
        self.Ch = channel
        self.ks = kersize
        self.st = stride
        self.set_size = set_size
        self.Nt = Nt
        self.Nr = 4
        self.Nc = 4
        self.rNr = int((self.Nr - self.ks)/self.st + 1)
        self.rNc = int((self.Nc - self.ks)/self.st + 1)
        self.C1 = nn.Conv2d(1, channel, kersize, stride, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(channel)
        self.sig = nn.Sigmoid()

        self.enc = nn.Sequential(
                SAB(self.Ch*self.rNr*self.rNc, dim_hidden, num_heads, ln=ln),
                SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
                SAB(dim_hidden, dim_hidden, num_heads, ln=ln))
        
        self.dec = nn.Sequential(
                PMA(dim_hidden, num_heads, p_outputs, ln=ln),
                nn.Linear(dim_hidden, dim_output),
                nn.Sigmoid())

    def forward(self, X):
        X = self.bn1(self.C1(X.view(-1, 1, self.Nr, self.Nc)))
        X = X.view(-1, self.set_size, self.Ch*self.rNr*self.rNc)
        X = self.sig(self.enc(X))
        return self.dec(X)[:,0]

class QuAN4(nn.Module):
    def __init__(self, set_size, channel, dim_output,
                 kersize, stride, dim_hidden, num_heads, Nt, p_outputs,  ln=True):
        super(QuAN4,self).__init__()
        self.Ch = channel
        self.ks = kersize
        self.st = stride
        self.set_size = set_size
        self.Nt = Nt
        self.Nr = 4
        self.Nc = 4
        self.rNr = int((self.Nr - self.ks)/self.st + 1)
        self.rNc = int((self.Nc - self.ks)/self.st + 1)
        self.C1 = nn.Conv2d(1, channel, kersize, stride, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(channel)
        self.sig = nn.Sigmoid()

        self.enc = nn.Sequential(
                SAB(self.Ch*self.rNr*self.rNc, dim_hidden, num_heads, ln=ln),
                SAB(dim_hidden, dim_hidden, num_heads, ln=ln))
        
        self.dec = nn.Sequential(
                PMA(dim_hidden, num_heads, p_outputs, ln=ln),
                nn.Linear(dim_hidden, dim_output),
                nn.Sigmoid())

    def forward(self, X):
        X = self.bn1(self.C1(X.view(-1, 1, self.Nr, self.Nc)))
        X = X.view(-1, self.set_size, self.Ch*self.rNr*self.rNc)
        X = self.sig(self.enc(X))
        return self.dec(X)[:,0]

class QuAN2(nn.Module):
    def __init__(self, set_size, channel, dim_output,
                 kersize, stride, dim_hidden, num_heads, Nt, p_outputs,  ln=True):
        super(QuAN2,self).__init__()
        self.Ch = channel
        self.ks = kersize
        self.st = stride
        self.set_size = set_size
        self.Nt = Nt
        self.Nr = 4
        self.Nc = 4
        self.rNr = int((self.Nr - self.ks)/self.st + 1)
        self.rNc = int((self.Nc - self.ks)/self.st + 1)
        self.C1 = nn.Conv2d(1, channel, kersize, stride, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(channel)
        self.sig = nn.Sigmoid()

        self.enc = nn.Sequential(
                SAB(self.Ch*self.rNr*self.rNc, dim_hidden, num_heads, ln=ln))
        
        self.dec = nn.Sequential(
                PMA(dim_hidden, num_heads, p_outputs, ln=ln),
                nn.Linear(dim_hidden, dim_output),
                nn.Sigmoid())

    def forward(self, X):
        X = self.bn1(self.C1(X.view(-1, 1, self.Nr, self.Nc)))
        X = X.view(-1, self.set_size, self.Ch*self.rNr*self.rNc)
        X = self.sig(self.enc(X))
        return self.dec(X)[:,0]

class PAB(nn.Module):
    def __init__(self, set_size, channel, dim_output,
                 kersize, stride, dim_hidden, num_heads, Nt,  p_outputs, ln=True):
        super(PAB, self).__init__()
        self.Ch = channel
        self.ks = kersize
        self.st = stride
        self.set_size = set_size
        self.Nt = Nt
        self.Nr = 4
        self.Nc = 4
        self.rNr = int((self.Nr - self.ks)/self.st + 1)
        self.rNc = int((self.Nc - self.ks)/self.st + 1)
        self.C1 = nn.Conv2d(1, channel, kersize, stride, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(channel)
        self.sig = nn.Sigmoid()
        n = 3

        self.enc = nn.Sequential(
                nn.Linear(self.Ch*self.rNr*self.rNc, n*dim_hidden),
                nn.Sigmoid(),
                nn.Linear(n*dim_hidden, dim_hidden),
                nn.Sigmoid())

        self.dec = nn.Sequential(
                PMA(dim_hidden, num_heads, p_outputs, ln=ln),
                nn.Linear(dim_hidden, dim_output),
                nn.Sigmoid())

    def forward(self, X):
        X = self.bn1(self.C1(X.view(-1, 1, self.Nr, self.Nc)))
        X = X.view(-1, self.set_size, self.Ch*self.rNr*self.rNc)
        return self.dec(self.enc(X))[:,0]

class Set_MLP(nn.Module):
    def __init__(self, set_size, channel, dim_output,
                     kersize, stride, dim_hidden, Nt, ln=True):
        super(Set_MLP, self).__init__()
        self.Ch = channel
        self.ks = kersize
        self.st = stride
        self.set_size = set_size
        self.Nt = Nt
        self.Nr = 4
        self.Nc = 4
        self.rNr = int((self.Nr - self.ks)/self.st + 1)
        self.rNc = int((self.Nc - self.ks)/self.st + 1)
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
        X = self.enc(X)
        X = self.dec(X)
        return self.sig(X.mean(axis=1))