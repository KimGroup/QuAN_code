import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

def gen_data_n8_entangle(args, c_p):
    '''
    data loading function.
    input
        args: argument
        c_p: random permutation of different state
    output
        x1, x2: volume/area-law training data
        x3, x4: volume/area-law testing data
    '''
    num_circ = 17 # N_c = 17
    num_train_circ = 12 
    circnum = np.arange(num_circ)+14
    circnum = circnum[c_p]
    testcirc = circnum[num_train_circ:]
    traincirc = np.setdiff1d(circnum, testcirc) 
    
    path = '../Data_src/HCBH_dataset_n8'
    add_path = f'state_{args.basis}' 
    n_bitstring = 4096//args.set * args.set # M_s = 4096
    nt = args.nr*args.nc # N_q = 16

    data = np.load(f'{path}/{add_path}_d{args.d1:d}_t{traincirc[0]}.npz')
    a = data['s'].astype(bool)
    p = np.random.permutation(len(a))
    a = a[p]
    x1 = a[:n_bitstring].reshape(1, n_bitstring, nt)
    for i, circi in enumerate(traincirc):
        if i==0: continue
        data = np.load(f'{path}/{add_path}_d{args.d1:d}_t{circi}.npz')
        a = data['s'].astype(bool)
        p = np.random.permutation(len(a))
        a = a[p]
        xx = a[:n_bitstring].reshape(1, n_bitstring, nt)
        x1 = np.concatenate([x1,xx], axis = 0)
        del a, xx

    data = np.load(f'{path}/{add_path}_d{args.d2:d}_t{traincirc[0]}.npz')
    a = data['s'].astype(bool)
    p = np.random.permutation(len(a))
    a = a[p]
    x2 = a[:n_bitstring//2].reshape(1, n_bitstring//2, nt)
    for i, circi in enumerate(traincirc):
        if i==0: continue
        data = np.load(f'{path}/{add_path}_d{args.d2:d}_t{circi}.npz')
        a = data['s'].astype(bool)
        p = np.random.permutation(len(a))
        a = a[p]
        xx = a[:n_bitstring//2].reshape(1, n_bitstring//2, nt)
        x2 = np.concatenate([x2,xx], axis = 0)
        del a, xx

    data = np.load(f'{path}/{add_path}_d{-args.d2:d}_t{traincirc[0]}.npz')
    a = data['s'].astype(bool)
    p = np.random.permutation(len(a))
    a = a[p]
    xx = a[:n_bitstring//2].reshape(1, n_bitstring//2, nt)
    x2 = np.concatenate([x2,xx], axis = 0)
    for i, circi in enumerate(traincirc):
        if i==0: continue
        data = np.load(f'{path}/{add_path}_d{-args.d2:d}_t{circi}.npz')
        a = data['s'].astype(bool)
        p = np.random.permutation(len(a))
        a = a[p]
        xx = a[:n_bitstring//2].reshape(1, n_bitstring//2, nt)
        x2 = np.concatenate([x2,xx], axis = 0)
        del a, xx
 
    data = np.load(f'{path}/{add_path}_d{args.d1:d}_t{testcirc[0]}.npz')
    a = data['s'].astype(bool)
    p = np.random.permutation(len(a))
    a = a[p]
    x3 = a[:n_bitstring].reshape(1, n_bitstring, nt)
    for i, circi in enumerate(testcirc):
        if i==0: continue
        data = np.load(f'{path}/{add_path}_d{args.d1:d}_t{circi}.npz')
        a = data['s'].astype(bool)
        p = np.random.permutation(len(a))
        a = a[p]
        xx = a[:n_bitstring].reshape(1, n_bitstring, nt)
        x3 = np.concatenate([x3,xx], axis = 0)
        del a, xx

    data = np.load(f'{path}/{add_path}_d{args.d2:d}_t{testcirc[0]}.npz')
    a = data['s'].astype(bool)
    p = np.random.permutation(len(a))
    a = a[p]
    x4 = a[:n_bitstring//2].reshape(1, n_bitstring//2, nt)
    for i, circi in enumerate(testcirc):
        if i==0: continue
        data = np.load(f'{path}/{add_path}_d{args.d2:d}_t{circi}.npz')
        a = data['s'].astype(bool)
        p = np.random.permutation(len(a))
        a = a[p]
        xx = a[:n_bitstring//2].reshape(1, n_bitstring//2, nt)
        x4 = np.concatenate([x4,xx], axis = 0)
        del a, xx

    data = np.load(f'{path}/{add_path}_d{-args.d2:d}_t{testcirc[0]}.npz')
    a = data['s'].astype(bool)
    p = np.random.permutation(len(a))
    a = a[p]
    xx = a[:n_bitstring//2].reshape(1, n_bitstring//2, nt)
    x4 = np.concatenate([x4,xx], axis = 0)
    for i, circi in enumerate(testcirc):
        if i==0: continue
        data = np.load(f'{path}/{add_path}_d{-args.d2:d}_t{circi}.npz')
        a = data['s'].astype(bool)
        p = np.random.permutation(len(a))
        a = a[p]
        xx = a[:n_bitstring//2].reshape(1, n_bitstring//2, nt)
        x4 = np.concatenate([x4,xx], axis = 0)
        del a, xx
        
    print(f'gen_data_n8_entangle: test on {testcirc[:]}')
    print(f'shapes: {x1.shape}, {x3.shape}')
    return x1, x2.reshape(-1, n_bitstring, nt), x3, x4.reshape(-1, n_bitstring, nt)

def gen_data_sample(set_size, x1, x2, Nt, shuffle=True):
    '''
    function for generating training set
    # x1.shape = # 12, Ms, Nq
    # xx1.shape = # 12*#set, setsize, Nq
    '''
    
    n = 1
    xx1 = np.zeros((x1.shape[0]*n*x1.shape[1]//set_size, set_size, Nt), dtype=bool)
    xx2 = np.zeros((x2.shape[0]*n*x2.shape[1]//set_size, set_size, Nt), dtype=bool)
    for i in range (n):
        p = np.random.permutation(x1.shape[1])
        xx1[xx1.shape[0]//n*i:xx1.shape[0]//n*(i+1)] = x1[:,p].reshape(x1.shape[0]*x1.shape[1]//set_size, set_size, Nt) 
        xx2[xx2.shape[0]//n*i:xx2.shape[0]//n*(i+1)] = x2[:,p].reshape(x2.shape[0]*x2.shape[1]//set_size, set_size, Nt) 
    
    y1 = np.zeros(xx1.shape[0], dtype=bool)
    y2 = np.ones(xx2.shape[0], dtype=bool)

    x = np.concatenate([xx1, xx2], axis = 0).astype(bool)
    del xx1, xx2
    y = np.concatenate([y1, y2], axis = 0)
    y = np.expand_dims(y, axis=1)

    if shuffle:
        p = np.random.permutation(len(x)) # mix set.
        x, y = x[p], y[p]
    return x, y

def gen_testdata_sample(set_size, x1, x2, Nt):
    '''
    function for generating testing set
    # x1.shape = # 5, Ms, Nq
    # xx1.shape = # 5*#set, setsize, Nq
    '''
    n = 1
    xx1 = np.zeros((x1.shape[0]*int(n*x1.shape[1]/set_size), set_size, Nt), dtype=bool) 
    xx2 = np.zeros((x2.shape[0]*int(n*x2.shape[1]/set_size), set_size, Nt), dtype=bool) 
    for i in range (n):
        p = np.random.permutation(x1.shape[1])
        xx1[int(xx1.shape[0]/n)*i:int(xx1.shape[0]/n)*(i+1)] = x1[:,p].reshape(x1.shape[0]*int(x1.shape[1]/set_size), set_size, Nt) 
        xx2[int(xx2.shape[0]/n)*i:int(xx2.shape[0]/n)*(i+1)] = x2[:,p].reshape(x2.shape[0]*int(x2.shape[1]/set_size), set_size, Nt)
    
    y1 = np.ones(xx1.shape[0], dtype=bool)
    y2 = np.zeros(xx2.shape[0], dtype=bool)

    x = np.concatenate([xx1, xx2], axis = 0).astype(bool)
    del xx1, xx2
    y = np.concatenate([y1, y2], axis = 0)
    y = np.expand_dims(y, axis=1)

    p = np.random.permutation(len(x)) # mix set.
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

