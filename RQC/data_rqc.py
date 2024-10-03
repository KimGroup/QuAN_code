import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import math

def gen_data_simulated(args, Nt, c_p):
    '''
    data loading function. See details of constants in SM.
    input
        args: argument
        Nt: number of qubits, N_q
        c_p: random permutation of circuit instance 
    output
        x1, x2: shallow/deep depth training data
        x3v, x4v: shallow/deep depth testing data 1
        x3t, x4t: shallow/deep depth testing data 2
    '''
    depth_s, depth_l = args.d1, args.d2
    num_circ = 50 # N_c = 50
    num_train_circ = 35 
    circnum = np.arange(num_circ) 
    circnum = circnum[c_p]
    testcirc = circnum[num_train_circ:] 
    traincirc = np.setdiff1d(circnum, testcirc) 
    
    data_path = '../Data_src' 
    path = f'{data_path}/RQC_simulated/simulated_corrected_{Nt}'
    add_path = f'size{Nt}_'
    if Nt>35: n_bitstring = 500000*4 # M_s = 2,000,000 for N_q = 36
    else: n_bitstring = 500000 # M_s = 500,000 for N_q = 20, 25, 30
    
    data = np.load(f'{path}/{add_path}depth_{depth_s}_circuitn_{traincirc[0]}.npz')
    a = data['s'].astype(bool)
    x1 = a[:n_bitstring].reshape(1, n_bitstring, a.shape[1])
    for i, circi in enumerate(traincirc):
        if i==0: continue
        data = np.load(f'{path}/{add_path}depth_{depth_s}_circuitn_{circi}.npz')
        a = data['s'].astype(bool)
        xx = a[:n_bitstring].reshape(1, n_bitstring, a.shape[1])
        x1 = np.concatenate([x1,xx], axis = 0)
        del a, xx

    data = np.load(f'{path}/{add_path}depth_{depth_l}_circuitn_{traincirc[0]}.npz')
    a = data['s'].astype(bool)
    x2 = a[:n_bitstring].reshape(1, n_bitstring, a.shape[1])
    for i, circi in enumerate(traincirc):
        if i==0: continue
        data = np.load(f'{path}/{add_path}depth_{depth_l}_circuitn_{circi}.npz')
        a = data['s'].astype(bool)
        xx = a[:n_bitstring].reshape(1, n_bitstring, a.shape[1])
        x2 = np.concatenate([x2,xx], axis = 0)
        del a, xx
    
    data = np.load(f'{path}/{add_path}depth_{depth_s}_circuitn_{testcirc[0]}.npz')
    a = data['s'].astype(bool)
    x3 = a[:n_bitstring].reshape(1, n_bitstring, a.shape[1])
    for i, circi in enumerate(testcirc):
        if i==0: continue
        data = np.load(f'{path}/{add_path}depth_{depth_s}_circuitn_{circi}.npz')
        a = data['s'].astype(bool)
        xx = a[:n_bitstring].reshape(1, n_bitstring, a.shape[1])
        x3 = np.concatenate([x3,xx], axis = 0)
        del a, xx

    data = np.load(f'{path}/{add_path}depth_{depth_l}_circuitn_{testcirc[0]}.npz')
    a = data['s'].astype(bool)
    x4 = a[:n_bitstring].reshape(1, n_bitstring, a.shape[1])
    for i, circi in enumerate(testcirc):
        if i==0: continue
        data = np.load(f'{path}/{add_path}depth_{depth_l}_circuitn_{circi}.npz')
        a = data['s'].astype(bool)
        xx = a[:n_bitstring].reshape(1, n_bitstring, a.shape[1])
        x4 = np.concatenate([x4,xx], axis = 0)
        del a, xx
        
    x3v, x3t = x3[:,:x3.shape[1]//2], x3[:,x3.shape[1]//2:]
    x4v, x4t = x4[:,:x4.shape[1]//2], x4[:,x4.shape[1]//2:]
    print(f'gen_data_simulated: {depth_s} vs {depth_l}, test on {testcirc[:]}')
    print(f'shapes: {x1.shape}, {x3v.shape}, {x3t.shape}')
    return x1, x2, x3v, x4v, x3t, x4t

def gen_data_simulated_20vs20(args, Nt, c_p):
    depth_s, depth_l = args.d1, args.d2
    circnum = np.arange(50)
    circnum = circnum[c_p]
    testcirc = circnum[35:]
    traincirc = np.setdiff1d(circnum, testcirc)
    
    data_path = '../Data_src' #args.prefix
    path = f'{data_path}/from_kv244/simulated_corrected_{Nt}'
    add_path = f'size{Nt}_'
    if Nt>35: n_bitstring = 500000*2
    else: n_bitstring = 500000//2
    
    data = np.load(f'{path}/{add_path}depth_{depth_s}_circuitn_{traincirc[0]}.npz')
    a = data['s'].astype('bool')
    x1 = a[:n_bitstring].reshape(1, n_bitstring, a.shape[1])
    x2 = a[n_bitstring:2*n_bitstring].reshape(1, n_bitstring, a.shape[1])
    for i, circi in enumerate(traincirc):
        if i==0: continue
        data = np.load(f'{path}/{add_path}depth_{depth_s}_circuitn_{circi}.npz')
        a = data['s'].astype('bool')
        xx = a[:n_bitstring].reshape(1, n_bitstring, a.shape[1])
        x1 = np.concatenate([x1,xx], axis = 0)
        xx = a[n_bitstring:2*n_bitstring].reshape(1, n_bitstring, a.shape[1])
        x2 = np.concatenate([x2,xx], axis = 0)
        del a, xx
    
    data = np.load(f'{path}/{add_path}depth_{depth_s}_circuitn_{testcirc[0]}.npz')
    a = data['s'].astype('bool')
    x3 = a[:n_bitstring].reshape(1, n_bitstring, a.shape[1])
    x4 = a[n_bitstring:2*n_bitstring].reshape(1, n_bitstring, a.shape[1])
    for i, circi in enumerate(testcirc):
        if i==0: continue
        data = np.load(f'{path}/{add_path}depth_{depth_s}_circuitn_{circi}.npz')
        a = data['s'].astype('bool')
        xx = a[:n_bitstring].reshape(1, n_bitstring, a.shape[1])
        x3 = np.concatenate([x3,xx], axis = 0)
        xx = a[n_bitstring:2*n_bitstring].reshape(1, n_bitstring, a.shape[1])
        x4 = np.concatenate([x4,xx], axis = 0)
        del a, xx

    x3v, x3t = x3[:,:x3.shape[1]//2], x3[:,x3.shape[1]//2:]
    x4v, x4t = x4[:,:x4.shape[1]//2], x4[:,x4.shape[1]//2:]
    print(f'gen_data_simulated_20vs20: test on {testcirc[:]}')
    print(f'shapes: {x1.shape}, {x3v.shape}, {x3t.shape}')
    return x1, x2, x3v, x4v, x3t, x4t

def gen_data_google(args, Nt, c_p):
    depth_s, depth_l = args.d1, args.d2
    circnum = np.arange(50)
    circnum = circnum[c_p]
    testcirc = circnum[35:]
    traincirc = np.setdiff1d(circnum, testcirc)

    if Nt>35: n_bitstring = 500000*4
    else: n_bitstring = 500000

    data_path = '../Data_src' #args.prefix
    path = f'{data_path}/Google_F'
    add_path = ''
    
    a = np.load(f'{path}{add_path}/bitstring_dict_{Nt}_{depth_s}.npz')['s']
    b = np.load(f'{path}{add_path}/bitstring_dict_{Nt}_{depth_l}.npz')['s']
    
    x1 = a[traincirc, :n_bitstring]
    x3 = a[testcirc, :n_bitstring]
    del a
    
    x2 = b[traincirc, :n_bitstring]
    x4 = b[testcirc, :n_bitstring]
    del b
    
    x3v, x3t = x3[:,:x3.shape[1]//2], x3[:,x3.shape[1]//2:]
    x4v, x4t = x4[:,:x4.shape[1]//2], x4[:,x4.shape[1]//2:]
    print(f'gen_data_google: {depth_s} vs {depth_l}, test on {testcirc[:]}')
    print(f'shapes: {x1.shape}, {x3v.shape}, {x3t.shape}')
    return x1, x2, x3v, x4v, x3t, x4t

def gen_data_google_20vs20(args, Nt, c_p):
    depth_s, depth_l = args.d1, args.d2
    circnum = np.arange(50)
    circnum = circnum[c_p]
    testcirc = circnum[35:]
    traincirc = np.setdiff1d(circnum, testcirc)
    if Nt>35: n_bitstring = 500000*2
    else: n_bitstring = 500000//2

    data_path = '../Data_src' #args.prefix
    path = f'{data_path}/Google_F'
    add_path = ''
    
    a = np.load(f'{path}{add_path}/bitstring_dict_{Nt}_{depth_s}.npz')['s']
    
    x1 = a[traincirc, :n_bitstring]
    x2 = a[traincirc, n_bitstring:2*n_bitstring]
    x3 = a[testcirc, :n_bitstring]
    x4 = a[testcirc, n_bitstring:2*n_bitstring]
    del a
    
    x3v, x3t = x3[:,:x3.shape[1]//2], x3[:,x3.shape[1]//2:]
    x4v, x4t = x4[:,:x4.shape[1]//2], x4[:,x4.shape[1]//2:]
    print(f'gen_data_google_20vs20: test on {testcirc[:]}')
    print(f'shapes: {x1.shape}, {x3v.shape}, {x3t.shape}')
    return x1, x2, x3v, x4v, x3t, x4t

def gen_data_sample(set_size, x1, x2, Nt, shuffle=True):
    '''
    function for generating training set
    # x1.shape = # N_traincirc=35, Ms, Nq
    # xx1.shape = # N_traincirc*#set, setsize, Nq
    '''
    
    n = 1 # integer number larger than 1 to sample more sets.
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

def gen_testdata_sample(set_size, x1, x2, Nt, shuffle=True):
    '''
    function for generating testing set
    # x1.shape = # N_testcirc=15, Ms, Nq
    # xx1.shape = # N_testcirc*#set, setsize, Nq
    '''
    if Nt>35: n=1
    else: n=2
    if x1.shape[1]%set_size != 0: 
        x1 = x1[:,:-(x1.shape[1]%set_size)]
        x2 = x2[:,:-(x2.shape[1]%set_size)]
    xx1 = np.zeros((x1.shape[0]*n*x1.shape[1]//set_size, set_size, Nt), dtype=bool) 
    xx2 = np.zeros((x2.shape[0]*n*x2.shape[1]//set_size, set_size, Nt), dtype=bool) 
    print(x1.shape, xx1.shape)
    for i in range (math.ceil(n)):
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

