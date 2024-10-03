import torch
from torch.utils.data import DataLoader, Dataset

import numpy as np

import os
import time
import argparse

from models_tc import *
from data_tc import *

def is_non_zero_file(fpath, verb=False):
    if verb: 
        if os.path.isfile(fpath)==False: print('No', fpath)
    return os.path.isfile(fpath) and os.path.getsize(fpath) > 0
def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-set", help = "set size", type = int, default = None)
    parser.add_argument("-p1", help = "prob 1", type = float, default = 0.000)
    parser.add_argument("-p2", help = "prob 2", type = float, default = 0.300)
    parser.add_argument("-nr", help = "height", type = int, default = 6)
    parser.add_argument("-nc", help = "width", type = int, default = 6)

    parser.add_argument("-prefix", help = "prefix", type = str, default = '~/QuAN')
    parser.add_argument("-saveprefix", help = "save folder prefix", type = str, default = 'g2_saved_models_tc')
    parser.add_argument("-epoch", help = "total epochs", type = int, default = 100)
    parser.add_argument("-batchsize", type=int, default=50) 

    parser.add_argument("-modelnum", type = str, default = 'p11')
    parser.add_argument("-hdim", type = int, default = 16)
    parser.add_argument("-nhead", type = int, default = 4)
    parser.add_argument("-ch", type = int, default = 1)
    parser.add_argument("-ker", type = int, default = 1)
    parser.add_argument("-st", type = int, default = 1)
    parser.add_argument("-dim_outputs", type=int, default=1)
    parser.add_argument("-p_outputs", type=int, default=1)

    parser.add_argument("-lr", type = float, default = 1e-4)
    parser.add_argument("-lrn", type = int, default = 0)
    parser.add_argument("-optim", type = str, default = None)
    parser.add_argument("-prev", type = str, default = None)
    
    parser.add_argument("-wandb_name", type = str)
    parser.add_argument("-debug", action='store_true', default=False)
    parser.add_argument("-testonly", action='store_true',default=True)

    parser.add_argument("-shuffle_epoch", type=int, default=1, help='shuffling dataset every n epoch')
    parser.add_argument("-nprandseed", action='store_true', default=False, help='random seed fix?')
    args = parser.parse_args(args=[])

    return args

def gen_data_loop_topvsinc_all(args, bxbz, inc, verb=False):
    path = f'~/QuAN/Data_src/toric_code/data36/testing_pflip_{inc:.3f}'
    nt = args.nr*args.nc
    n_bitstring = 8134//args.set * args.set

    bx, bz = bxbz
    if bz!=0.14: print(f'** Error! ** {bz} != 0.14')

    datafilename = path+f'/betaX={bx:.2f}_betaZ={bz:.2f}_height=300_run={1}_width=1000.npz'
    if is_non_zero_file(datafilename):
        a = np.load(datafilename)['arr0'].astype(bool)
        x1 = a[:n_bitstring].reshape(1, n_bitstring, nt)
        for run in range(1,15):
            datafilename = path+f'/betaX={bx:.2f}_betaZ={bz:.2f}_height=300_run={run}_width=1000.npz'
            if is_non_zero_file(datafilename):
                a = np.load(datafilename)['arr0'].astype(bool)
                xx = a[:n_bitstring].reshape(1, n_bitstring, nt)
                x1 = np.concatenate([x1,xx], axis = 0)
            else: continue
        x1 = x1.reshape(x1.shape[0]*int(x1.shape[1]/args.set), args.set, nt)
        if verb:
            print(f'gen_data_loop_topvsinc_all,', f'shape: {x1.shape}', \
                  f'bx, bz, pflip = {bx},{bz},{inc}')
        return x1
    else: 
        return np.zeros(1)
    

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
    howmanyruns = 10
    p2train = 0.300

    start = time.time()
    for fraction in [1]:
        nr = 6
        nc = nr

        args = get_arguments()
        args.dataformat = 'loop'
        args.nr = nr; args.nc = nc
        args.fraction = fraction
        for mn in ['p11', 'p01']:#, 'p00']:
            print('modelnum = ',mn)
            filename = f'~/QuAN/Figure/Data_out/{args.saveprefix}/TC{args.dataformat}{nr*nc}_{mn}_0.000vs{p2train:.3f}_5_confidence.npz'
            phase_diagram_out = np.zeros((9,howmanyruns,61,113876))
            phase_diagram_out[:] = np.nan

            args.modelnum = mn
            for k, setsize in enumerate(2**np.arange(7)):
                for run in range(howmanyruns):
                    args.set = setsize
                    args.modelnum = mn
                    args.run = run
                    args.dim_outputs = 1
                    args.ch = 1
                    args.ker = 1
                    args.batchsize = int(4096//setsize)  #2**12//setsize

                    args.wandb_name = f'QSL_{args.modelnum}_0.000vs{p2train:.3f}_5_run_{args.run}_sampledivide{args.fraction}'
                    modelname= f'model_e_{args.nr}x{args.nc}_0.000vs{p2train:.3f}-{args.modelnum}_set{args.set}_h16nh4_ch{args.ch}ker{args.ker}st1.pth'
                    args.prev = modelname

                    if args.modelnum=='p11':
                        model = QuAN2(set_size=args.set, channel=args.ch, dim_output=args.dim_outputs,\
                                                kersize=args.ker, stride=args.st, dim_hidden=args.hdim, num_heads=args.nhead,\
                                                Nr=args.nr, Nc=args.nc, p_outputs=args.p_outputs)

                    if args.modelnum=='p01':
                        model = PAB(set_size=args.set, channel=args.ch, dim_output=args.dim_outputs,\
                                                kersize=args.ker, stride=args.st, dim_hidden=args.hdim, num_heads=args.nhead,\
                                                Nr=args.nr, Nc=args.nc, p_outputs=args.p_outputs)
                    if args.modelnum=='p00':
                        model = Set_MLP(set_size=args.set, channel=args.ch, dim_output=args.dim_outputs,\
                                                kersize=args.ker, stride=args.st, dim_hidden=args.hdim, \
                                                Nr=args.nr, Nc=args.nc)

                    model.apply(weights_init)
                    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
                    model.to(device)
                    modelloadname = f'{args.saveprefix}/{args.wandb_name}/{args.prev}'
                    if not is_non_zero_file(modelloadname, True): continue

                    state_dict = torch.load(modelloadname, \
                                            map_location=torch.device('cpu')) 
                    model.load_state_dict(state_dict)
                    model.eval()

                    for i, bx in enumerate(np.arange(20)*0.02):
                        if i!=0: continue
                        bz = 0.14
                        for j, pflip in enumerate(np.arange(61)*0.005):
                            x1_bxbz = gen_data_loop_topvsinc_all(args, (bx,bz), pflip, verb=False)
                            if x1_bxbz.shape[0]>2:
                                mat_output, mat_pred = [], []
                                test_data2 = testing_set2(x1_bxbz)
                                test_loader2 = DataLoader(dataset = test_data2, batch_size=args.batchsize, shuffle=False)
                                for testdata in (test_loader2):
                                    test_inputs = testdata[0].to(device)
                                    test_outputs = model(test_inputs.float())
                                    test_pred_classes = torch.cat([1-test_outputs, test_outputs], axis=1).argmax(axis=1)
                                    mat_output.extend(test_outputs.detach().cpu().numpy())
                                    mat_pred.extend(test_pred_classes.detach().cpu().numpy())
                                mat_output = np.array(mat_output)
                                mat_pred = np.array(mat_pred)
                                phase_diagram_out[k, run, j, :mat_output.shape[0]] = mat_output.reshape(-1)
                                del test_inputs, test_outputs
                    print(fraction, setsize, run, modelloadname)
            np.savez(filename, arr0=phase_diagram_out)
    print(time.time()-start)


if __name__ == "__main__":
    main()
