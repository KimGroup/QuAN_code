import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def gen_data_loop(args, c_p):
    '''
    data loading function.
    input
        args: argument
        c_p: random permutation of different state
    output
        x1, x2: topological/trivial training data
        x3, x4: topological/trivial testing data
    '''
    num_circ = 200 # Nc = 200
    num_train_circ = 150 # 150 training state
    circnum = np.arange(num_circ)
    circnum = circnum[c_p]; circnum1, circnum2 = circnum, circnum
    testcirc1 = circnum1[num_train_circ:] # 50 testing state
    traincirc1 = np.setdiff1d(circnum1, testcirc1)
    testcirc2 = circnum2[num_train_circ:]
    traincirc2 = np.setdiff1d(circnum2, testcirc2)

    nt = args.nr * args.nc
    
    path = f'../Data_src/toric_code/data{nt}/training_pflip'

    n_bitstring = 8134//args.set * args.set # Ms = 8134
    
    data = np.load(f'{path}_{args.p1:.3f}/trainnum{traincirc1[0]}.npz')
    a = data['arr0'].astype(bool)
    p = np.random.permutation(len(a))
    a = a[p] 
    x1 = a[:n_bitstring].reshape(1, n_bitstring, nt)
    for i, circi in enumerate(traincirc1):
        if i==0: continue
        data = np.load(f'{path}_{args.p1:.3f}/trainnum{circi}.npz')
        a = data['arr0'].astype(bool)
        p = np.random.permutation(len(a))
        a = a[p]
        xx = a[:n_bitstring].reshape(1, n_bitstring, nt)
        x1 = np.concatenate([x1,xx], axis = 0)
        del a, xx
    
    data = np.load(f'{path}_{args.p2:.3f}/trainnum{traincirc2[0]}.npz')
    a = data['arr0'].astype(bool)
    p = np.random.permutation(len(a))
    a = a[p]
    x2 = a[:n_bitstring].reshape(1, n_bitstring, nt)
    for i, circi in enumerate(traincirc2):
        if i==0: continue
        data = np.load(f'{path}_{args.p2:.3f}/trainnum{circi}.npz')
        a = data['arr0'].astype(bool)
        p = np.random.permutation(len(a))
        a = a[p]
        xx = a[:n_bitstring].reshape(1, n_bitstring, nt)
        x2 = np.concatenate([x2,xx], axis = 0)
        del a, xx
    
    data = np.load(f'{path}_{args.p1:.3f}/trainnum{testcirc1[0]}.npz')
    a = data['arr0'].astype(bool)
    p = np.random.permutation(len(a))
    a = a[p]
    x3 = a[:n_bitstring].reshape(1, n_bitstring, nt)
    for i, circi in enumerate(testcirc1):
        if i==0: continue
        data = np.load(f'{path}_{args.p1:.3f}/trainnum{circi}.npz')
        a = data['arr0'].astype(bool)
        p = np.random.permutation(len(a))
        a = a[p]
        xx = a[:n_bitstring].reshape(1, n_bitstring, nt)
        x3 = np.concatenate([x3,xx], axis = 0)
        del a, xx

    data = np.load(f'{path}_{args.p2:.3f}/trainnum{testcirc2[0]}.npz')
    a = data['arr0'].astype(bool)
    p = np.random.permutation(len(a))
    a = a[p]
    x4 = a[:n_bitstring].reshape(1, n_bitstring, nt)
    for i, circi in enumerate(testcirc2):
        if i==0: continue
        data = np.load(f'{path}_{args.p2:.3f}/trainnum{circi}.npz')
        a = data['arr0'].astype(bool)
        p = np.random.permutation(len(a))
        a = a[p]
        xx = a[:n_bitstring].reshape(1, n_bitstring, nt)
        x4 = np.concatenate([x4,xx], axis = 0)
        del a, xx
        

    print(f'gen_data_loop: test on {testcirc1[:]}')
    print(f'shapes: {x1.shape}, {x3.shape}')
    return x1, x2, x3, x4

def gen_data_sample(set_size, x1, x2, Nt):
    
    n = 1
    xx1 = np.zeros((x1.shape[0]*n*x1.shape[1]//set_size, set_size, Nt), dtype=bool)
    xx2 = np.zeros((x2.shape[0]*n*x2.shape[1]//set_size, set_size, Nt), dtype=bool)
    for i in range (n):
        p = np.random.permutation(x1.shape[1])
        xx1[xx1.shape[0]//n*i:xx1.shape[0]//n*(i+1)] = x1[:,p].reshape(x1.shape[0]*x1.shape[1]//set_size, set_size, Nt) 
        xx2[xx2.shape[0]//n*i:xx2.shape[0]//n*(i+1)] = x2[:,p].reshape(x2.shape[0]*x2.shape[1]//set_size, set_size, Nt) 
    
    y1 = np.ones(xx1.shape[0], dtype=bool)
    y2 = np.zeros(xx2.shape[0], dtype=bool)

    x = np.concatenate([xx1, xx2], axis = 0).astype(bool)
    del xx1, xx2
    y = np.concatenate([y1, y2], axis = 0)
    y = np.expand_dims(y, axis=1)

    p = np.random.permutation(len(x))
    x, y = x[p], y[p]
    return x, y

def gen_testdata_sample(set_size, x1, x2, Nt):
    n = 1
    xx1 = np.zeros((x1.shape[0]*n*x1.shape[1]//set_size, set_size, Nt), dtype=bool)
    xx2 = np.zeros((x2.shape[0]*n*x2.shape[1]//set_size, set_size, Nt), dtype=bool)
    for i in range (n):
        p = np.random.permutation(x1.shape[1])
        xx1[xx1.shape[0]//n*i:xx1.shape[0]//n*(i+1)] = x1[:,p].reshape(x1.shape[0]*x1.shape[1]//set_size, set_size, Nt) 
        xx2[xx2.shape[0]//n*i:xx2.shape[0]//n*(i+1)] = x2[:,p].reshape(x2.shape[0]*x2.shape[1]//set_size, set_size, Nt)
 
    y1 = np.ones(xx1.shape[0], dtype=bool)
    y2 = np.zeros(xx2.shape[0], dtype=bool)

    x = np.concatenate([xx1, xx2], axis = 0).astype(bool)
    del xx1, xx2
    y = np.concatenate([y1, y2], axis = 0)
    y = np.expand_dims(y, axis=1)

    p = np.random.permutation(len(x))
    x, y = x[p], y[p]
    return x, y


class training_set(Dataset):
    def __init__(self, x, y):
        xx = torch.from_numpy(x)
        yy = torch.from_numpy(y)
        self.X = xx
        self.Y = yy
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return [self.X[idx], self.Y[idx]]
    
class testing_set(Dataset):
    def __init__(self, xt, yt):
        xx = torch.from_numpy(xt)
        yy = torch.from_numpy(yt)
        self.X = xx
        self.Y = yy
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return [self.X[idx], self.Y[idx]]

