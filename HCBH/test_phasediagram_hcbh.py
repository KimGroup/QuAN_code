import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import numpy as np

import os
import time
import argparse

from models_hcbh import *
from data_hcbh import *

def is_non_zero_file(fpath, verb=False):
    if verb: 
        if os.path.isfile(fpath)==False: print('No', fpath)
    return os.path.isfile(fpath) and os.path.getsize(fpath) > 0
def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-set", help = "set size", type = int, default = None)
    parser.add_argument("-d1", help = "volume-law, \delta=0", type = int, default = 0)
    parser.add_argument("-d2", help = "area-law, \delta=\pm2", type = int, default = 4)
    parser.add_argument("-nr", help = "height", type = int, default = 4)
    parser.add_argument("-nc", help = "width", type = int, default = 4)
    parser.add_argument("-basis", type=int, default=0) 
    parser.add_argument("-n8", action='store_true',default=False)

    parser.add_argument("-prefix", help = "prefix", type = str, default = '~/QuAN')
    parser.add_argument("-saveprefix", help = "save folder prefix", type = str, default = 'g2_saved_models_hcbh')
    parser.add_argument("-epoch", help = "total epochs", type = int, default = 500)
    parser.add_argument("-batchsize", type=int, default=50) 

    parser.add_argument("-modelnum", help='model name', type = str, default = 'c21')
    parser.add_argument("-hdim", help='hidden dimension size d_h', type = int, default = 16)
    parser.add_argument("-nhead", help='# of head n_h', type = int, default = 4)
    parser.add_argument("-ch", help='# channel n_c', type = int, default = 8)
    parser.add_argument("-ker", help='square kernel size', type = int, default = 2)
    parser.add_argument("-st", help='stride', type = int, default = 1)
    parser.add_argument("-dim_outputs", help='size of output dimension, 1 for scalar', type=int, default=1)
    parser.add_argument("-p_outputs", help='number of outputs', type=int, default=1)

    parser.add_argument("-lr", type = float, default = 1e-4)
    parser.add_argument("-lrn", type = int, default = 0)
    parser.add_argument("-prev", type = str, default = None)
    parser.add_argument("-pprev", type = str, default = None)
    parser.add_argument("-wandb_name", type = str)
    parser.add_argument("-debug", action='store_true', default=False)
    parser.add_argument("-testonly", action='store_true',default=True)

    parser.add_argument("-shuffle_epoch", type=int, default=10, help='shuffling dataset every n epoch')
    parser.add_argument("-nprandseed", action='store_true', default=False, help='random seed fix?')
    args = parser.parse_args()

    return args

def gen_data_n8_boson_all(args, state, time, verb=False):
    prefix = '~/QuAN/Data_src'
    path = f'{prefix}/HCBH_dataset_n8/state_{args.basis}'
    n_bitstring = 2048//args.set * args.set
    nt = args.nr*args.nc
    datafilename = path+f'_d{state}_t{time}.npz'
    if is_non_zero_file(datafilename):
        a = np.load(datafilename)['s'].astype(bool)
        p = np.random.permutation(len(a))
        a = a[p]
        x1 = a[:n_bitstring].reshape(1, n_bitstring, nt)
        x1 = x1.reshape(x1.shape[0]*int(x1.shape[1]/args.set), args.set, nt)
        if verb:
            print(f'gen_data_n8_boson_all, shape: {x1.shape}')
        return x1
    else: return np.zeros(1)


class testing_set2(Dataset):
    def __init__(self, xt):
        xx = torch.from_numpy(xt)
        self.X = xx
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return [self.X[idx]]

def main():
    args = get_arguments()
    start = time.time()
    howmanyruns = 10
    args.basis = 0
    kernel=2
    d2=4
    kernel=2
    Nt=args.nr*args.nc
    generated='expn8'
    for mn,channel in [('c21',7), ('c11',8), ('c01',8), ('c00',8)]:
        print('modelnum = ',mn, generated)
        args.modelnum = mn
        if generated=='simn8': d22 = 6
        else: d22 = 4

        filename = f'boson_{mn}_basis0_0vs{d22}_ch{channel}ker{kernel}_{generated}_phasediagram.npz'
        phase_diagram_out = np.ones((5,8,howmanyruns,13,17))*(-10)
        phase_diagram_pred = np.ones((5,8,howmanyruns,13,17))*(-10)
        if is_non_zero_file('~/QuAN/Figure/Data_out/g2_saved_models_hcbh/'+filename):
            data = np.load('~/QuAN/Figure/Data_out/g2_saved_models_hcbh/'+filename)
            phase_diagram_out[:,:data['arr0'].shape[1],:data['arr0'].shape[2]] = data['arr0']
            phase_diagram_pred[:,:data['arr0'].shape[1],:data['arr0'].shape[2]] = data['arr1']
        else:
            phase_diagram_out = np.ones((5,9,howmanyruns,13,17))*(-10)
            phase_diagram_pred = np.ones((5,9,howmanyruns,13,17))*(-10)
        
        for k, setsize in enumerate(2**np.arange(9)):
            for run in range(howmanyruns):
                args.set = setsize
                args.run = run
                args.dim_outputs = 1
                args.ch = channel
                args.ker = kernel
                args.batchsize = 2048
                args.nhead = 4

                args.wandb_name = f'boson_{args.modelnum}_0vs{d22}_run_{args.run}'
                modelname= f'model_e_4x4_n8d0vsn8d{d22}-{args.modelnum}_set{args.set}_h16nh4_ch{args.ch}ker{args.ker}st1.pth'
                args.prev = modelname

                if args.modelnum == 'c00': model = Set_MLP(set_size=args.set, channel=args.ch, dim_output=args.dim_outputs, \
                                    kersize=args.ker, stride=args.st, dim_hidden=args.hdim, Nt=Nt)
                elif args.modelnum == 'c01': model = PAB(set_size=args.set, channel=args.ch, dim_output=args.dim_outputs, \
                                                            kersize=args.ker, stride=args.st, dim_hidden=args.hdim, num_heads=args.nhead, \
                                                            Nt=Nt, p_outputs=args.p_outputs)
                elif args.modelnum == 'c11': model = QuAN2(set_size=args.set, channel=args.ch, dim_output=args.dim_outputs, \
                                                            kersize=args.ker, stride=args.st, dim_hidden=args.hdim, num_heads=args.nhead, \
                                                            Nt=Nt, p_outputs=args.p_outputs)
                elif args.modelnum == 'c21': model = QuAN4(set_size=args.set, channel=args.ch, dim_output=args.dim_outputs, \
                                                            kersize=args.ker, stride=args.st, dim_hidden=args.hdim, num_heads=args.nhead, \
                                                            Nt=Nt, p_outputs=args.p_outputs)
                device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
                model.to(device)
                modelloadname = f'~/QuAN/Figure/Data_out/g2_saved_models_hcbh/{args.wandb_name}/{args.prev}'
                if not is_non_zero_file(modelloadname, True): continue
                print(setsize)
                state_dict = torch.load(modelloadname, \
                                    map_location=torch.device('cpu'))
                model.load_state_dict(state_dict)
                model.eval()

                for state in range(-d2, d2+1):
                    for time_i in range(17):
                        x1_state_t = gen_data_n8_boson_all(args, state, time_i+14)
                        if x1_state_t.shape[0]>2:
                            mat_output = []
                            mat_pred = []
                            test_data2 = testing_set2(x1_state_t)
                            test_loader2 = DataLoader(dataset = test_data2, batch_size=args.batchsize, shuffle=False)
                            for testdata in (test_loader2):
                                test_inputs = testdata[0].to(device)
                                test_outputs = model(test_inputs.float())
                                mat_output.extend(test_outputs.detach().cpu().numpy())
                                test_pred_classes = torch.cat([1-test_outputs, test_outputs], axis=1).argmax(axis=1)
                                mat_pred.extend(test_pred_classes.detach().cpu().numpy())
                            mat_output = np.array(mat_output)
                            mat_pred = np.array(mat_pred)
                            phase_diagram_out[1, k, run, state+d2,time_i] = mat_output.mean(axis=0)
                            phase_diagram_pred[1, k, run, state+d2,time_i] = mat_pred.mean(axis=0)
                            del x1_state_t, test_inputs, test_outputs
                

        np.savez('~/QuAN/Figure/Data_out/g2_saved_models_hcbh/'+filename, arr0=phase_diagram_out, arr1=phase_diagram_pred)
        print(time.time()-start)

if __name__ == "__main__":
    main()
