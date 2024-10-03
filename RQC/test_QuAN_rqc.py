import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import numpy as np
import time

from collections import OrderedDict
import sys
import os

from models_rqc import *
from data_rqc import *

import wandb

import argparse

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-set", help = "set size", type = int, default = None)
    parser.add_argument("-n_mini", help = "# of mini set", type = int, default = 5)
    parser.add_argument("-d1", help = "depth 1", type = int, default = None)
    parser.add_argument("-d2", help = "depth 2", type = int, default = 20)
    parser.add_argument("-nr", help = "height", type = int, default = None)
    parser.add_argument("-nc", help = "width", type = int, default = None)

    parser.add_argument("-prefix", help = "prefix", type = str, default = '~/QuAN/Figure/Data_out/')
    parser.add_argument("-saveprefix", help = "save folder prefix", type = str, default = 'g2_saved_models_rqc')
    parser.add_argument("-epoch", help = "total epochs", type = int, default = 400)
    parser.add_argument("-batchsize", type=int, default=20)

    parser.add_argument("-modelnum", help='model name', type = str, default = 'cm1')
    parser.add_argument("-minisettype", type = str, default = 'miniset_A2')
    parser.add_argument("-hdim", help='hidden dimension size d_h', type = int, default = 16)
    parser.add_argument("-nhead", help='# of head n_h', type = int, default = 4)
    parser.add_argument("-ch", help='# channel n_c', type = int, default = 16)
    parser.add_argument("-ker", help='square kernel size', type = int, default = 2)
    parser.add_argument("-st", help='stride', type = int, default = 1)
    parser.add_argument("-dim_outputs", help='size of output dimension, 1 for scalar', type=int, default=1)
    parser.add_argument("-p_outputs", help='number of outputs', type=int, default=1)

    parser.add_argument("-nsy", help='data type = google (experimental) or 0 (simulated)', type = str, default = 'google')
    parser.add_argument("-lr", help='learning rate', type = float, default = 4e-5)
    parser.add_argument("-lrn", help='learning rate step', type = int, default = 0)
    parser.add_argument("-opsc", help='learning rate scheduler yes or no', type = str, default = None)
    parser.add_argument("-lrstep", help='learning rate step amount', type = float, default = 0.65)
    parser.add_argument("-lrstepsize", help='learning rate step size', type = int, default = 200)
    parser.add_argument("-prev", help='loading pre-trained model', type = str, default = None)
    parser.add_argument("-pprev", help='loading same circuit indices', type = str, default = None)
    parser.add_argument("-logprev", help='loading log file', type = str, default = None)
    parser.add_argument("-circ", help='2-qubit circuit type, CZ (C) or F-sim (F)', type = str, default = 'F')
    parser.add_argument("-geo", help='circuit geometry, rectangular (R) or diamond (D)', type = str, default = 'R')
    
    parser.add_argument("-wandb_name", type = str)
    parser.add_argument("-debug", action='store_true', default=False)
    parser.add_argument("-testonly", action='store_true',default=True)

    parser.add_argument("-shuffle_epoch", type=int, default=10, help='shuffling dataset every n epoch')
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
    '''
    x00: MLP
    c3d: CNN
    c00: SMLP
    c01: PAB
    c11: QuAN_2
    cm1: QuAN_50
    '''
    if args.modelnum == 'c00': model = Set_MLP(set_size=args.set, channel=args.ch, dim_output=args.dim_outputs, kersize=args.ker, stride=args.st, dim_hidden=args.hdim, Nr=args.nr, Nc=args.nc)
    elif args.modelnum == 'c01': model = PAB(set_size=args.set, channel=args.ch, dim_output=args.dim_outputs, kersize=args.ker, stride=args.st, dim_hidden=args.hdim, num_heads=args.nhead, Nr=args.nr, Nc=args.nc, p_outputs=args.p_outputs)
    elif args.modelnum == 'c11': model = QuAN2(set_size=args.set, channel=args.ch, dim_output=args.dim_outputs, kersize=args.ker, stride=args.st, dim_hidden=args.hdim, num_heads=args.nhead, Nr=args.nr, Nc=args.nc, p_outputs=args.p_outputs)
    elif args.modelnum == 'cm1': model = QuANn(set_size=args.set, channel=args.ch, dim_output=args.dim_outputs, kersize=args.ker, stride=args.st, dim_hidden=args.hdim, num_heads=args.nhead, Nr=args.nr, Nc=args.nc, p_outputs=args.p_outputs, miniset=minisetsize, minisettype=args.minisettype)

    elif args.modelnum == 'x00': model = MLP3d(set_size=args.set, channel=args.ch, dim_output=args.dim_outputs, kersize=args.ker, stride=args.st, dim_hidden=args.hdim, Nr=args.nr, Nc=args.nc, p_outputs=args.p_outputs)
    elif args.modelnum == 'c3d': model = Conv3d(set_size=args.set, channel=args.ch, dim_output=args.dim_outputs, kersize=args.ker, stride=args.st, dim_hidden=args.hdim, Nr=args.nr, Nc=args.nc, p_outputs=args.p_outputs)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'total parameters: {total_params}')

    model = torch.nn.DataParallel(model);

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
        model.load_state_dict(new_state_dict); print('model loaded.')

        if args.pprev == None:
            argsprev_model_name_p = (args.prev).replace('model_ta_', 'p_').replace('model_va_', 'p_').replace('model_tl_', 'p_').replace('model_vl_', 'p_').replace('model_e_', 'p_').replace('.pth', '.txt')
            print(argsprev_model_name_p)
            c_p = np.loadtxt(f'{argsprev_model_name_p}', dtype = 'float64').astype(int)
        else: c_p = np.loadtxt(args.pprev, dtype = 'float64').astype(int)
        
    criterion = nn.BCELoss()

    if args.wandb_name and not args.debug and not args.testonly:
        wandb.init(project='google_rqc', name=args.wandb_name)
        wandb.log({'# of parameters': total_params})

    if args.nsy == 'google': 
        if args.d1==20: x1, x2, x3v, x4v, x3t, x4t = gen_data_google_20vs20(args, Nt, c_p)
        else: x1, x2, x3v, x4v, x3t, x4t = gen_data_google(args, Nt, c_p)
    elif args.nsy == '0': 
        if args.d1==20: x1, x2, x3v, x4v, x3t, x4t = gen_data_simulated_20vs20(args, Nt, c_p)
        else: x1, x2, x3v, x4v, x3t, x4t = gen_data_simulated(args, Nt, c_p)
    x1, x2, x3v, x4v, x3t, x4t = x1.astype(bool), x2.astype(bool), x3v.astype(bool), x4v.astype(bool), x3t.astype(bool), x4t.astype(bool)
    del x1, x2 

    start = time.time()
    
    if 'model_v' in (args.prev):
        xt, yt = gen_testdata_sample(args.set, x3t, x4t, Nt)
    elif 'model_t' in (args.prev):
        xt, yt = gen_testdata_sample(args.set, x3v, x4v, Nt)
    else: sys.exit("ERROR: model name wrong")

    test_loader = DataLoader(dataset = testing_set(xt, yt), batch_size=args.batchsize, shuffle=True)
    del xt, yt

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device); model.to(device)
    model.eval()
    
    for epoch in range (1):
        test_loss = 0.0
        test_acc = 0.0
        test_acc0 = 0.0
        test_acc1 = 0.0
    
        for j,testdata in enumerate(test_loader):

            test_inputs, test_labels = testdata[0].to(device), testdata[1].to(device)
            test_outputs = model.module(test_inputs.float())
            
            del test_inputs
            loss = criterion(test_outputs.squeeze(), test_labels.float().squeeze())
            loss.detach().cpu().numpy()
            
            test_pred_classes = torch.cat([1-test_outputs, test_outputs], axis=1).argmax(axis=1)
            test_loss += loss.item()
            test_acc += torch.count_nonzero(torch.eq(test_pred_classes, test_labels.squeeze())).item()/len(test_labels)
            m0 = test_labels.squeeze() < 0.5
            m1 = test_labels.squeeze() > 0.5
            if len(test_labels.squeeze()[m0]) == 0:
                test_acc0 += 0.0
            else: test_acc0 += torch.count_nonzero(torch.eq(test_pred_classes[m0], test_labels.squeeze()[m0])).item()/len(test_labels.squeeze()[m0])
            if len(test_labels.squeeze()[m1]) == 0:
                test_acc1 += 0.0
            else: test_acc1 += torch.count_nonzero(torch.eq(test_pred_classes[m1], test_labels.squeeze()[m1])).item()/len(test_labels.squeeze()[m1])
        test_loss /= len(test_loader)
        test_acc /= len(test_loader)
        test_acc0 /= len(test_loader)
        test_acc1 /= len(test_loader)
        
        torch.cuda.empty_cache()

    argsprev_model_name = (args.prev).replace(f'{args.saveprefix}/{args.wandb_name}/', '')
    argsprev_model_name = argsprev_model_name.replace('.pth', '') 
    fff = open(f'{args.saveprefix}/{args.wandb_name}/test_acc-data_p{args.nsy}_{args.d1}vs{args.d2}-{argsprev_model_name}.txt', 'w')
    np.savetxt(fff, np.array([test_acc, test_acc0, test_acc1]), newline = " ")
    fff.close()

    print(f'{args.saveprefix}/{args.wandb_name}/test_acc-data_p{args.nsy}_{args.d1}vs{args.d2}-{argsprev_model_name}.txt', 'w')
    print(f'Accy// {test_acc*100:.2f}%, s0: {test_acc0*100:.2f}%, s1: {test_acc1*100:.2f}%')
    
    print(f'Time: {time.time()-start} s, {(time.time()-start)/60} min, {(time.time()-start)/3600} hour \n\n')

if __name__ == "__main__":
    main()






