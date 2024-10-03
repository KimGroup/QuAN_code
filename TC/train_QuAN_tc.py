import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import numpy as np
import time

from collections import OrderedDict
import os

from models_tc import *
from data_tc import *

import wandb

import argparse

class testing_set2(Dataset):
    def __init__(self, xt):
        xx = torch.from_numpy(xt)
        self.X = xx
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return [self.X[idx]]

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
    parser.add_argument("-testonly", action='store_true',default=False)

    parser.add_argument("-shuffle_epoch", type=int, default=1, help='shuffling dataset every n epoch')
    parser.add_argument("-nprandseed", action='store_true', default=False, help='random seed fix?')
    args = parser.parse_args()

    return args

def main():

    args = get_arguments()

    if args.nprandseed: 
        print('np.random.seed(42)')
        np.random.seed(42)

    Nt = args.nr * args.nc

    os.makedirs(f'{args.saveprefix}',exist_ok=True)
    os.makedirs(f'{args.saveprefix}/{args.wandb_name}',exist_ok=True)

    if args.geo=='R':
        model_name = f'{args.nr}x{args.nc}_{args.p1:.3f}vs{args.p2:.3f}-{args.modelnum}_set{args.set}_h{args.hdim}nh{args.nhead}_ch{args.ch}ker{args.ker}st{args.st}'
    if args.modelnum == 3: model_name = model_name + f'_ind{args.num_inds}'
        
    print("Storing as:", model_name)

    if args.modelnum == 'p01': model = PAB(set_size=args.set, channel=args.ch, dim_output=args.dim_outputs, \
                                                kersize=args.ker, stride=args.st, dim_hidden=args.hdim, num_heads=args.nhead, Nr=args.nr, Nc=args.nc, p_outputs=args.p_outputs)
    elif args.modelnum == 'p11': model = QuAN2(set_size=args.set, channel=args.ch, dim_output=args.dim_outputs, \
                                                kersize=args.ker, stride=args.st, dim_hidden=args.hdim, num_heads=args.nhead, Nr=args.nr, Nc=args.nc, p_outputs=args.p_outputs)
    elif args.modelnum == 'p00': model = Set_MLP(set_size=args.set, channel=args.ch, dim_output=args.dim_outputs, \
                                                kersize=args.ker, stride=args.st, dim_hidden=args.hdim, Nr=args.nr, Nc=args.nc)
    model.apply(weights_init)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'total parameters: {total_params}') 

    model = torch.nn.DataParallel(model)
    print('model loaded.')

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
        c_p = np.loadtxt(f'{args.saveprefix}/{args.wandb_name}/p_{model_name}.txt', dtype = 'float64').astype(int)
        
    if args.prev == None:
        f = open(f'{args.saveprefix}/{args.wandb_name}/log_{model_name}.txt', 'w')
        f.close()
        c_p = np.random.permutation(200)

        np.savetxt(f'{args.saveprefix}/{args.wandb_name}/p_{model_name}.txt', c_p)
        
        
    total_epoch = args.epoch
    stepsize = 100
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr*0.65**args.lrn, weight_decay=5e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=stepsize, gamma=0.65)
    print(scheduler.state_dict())

    criterion = nn.BCELoss()

    if args.wandb_name and not args.debug and not args.testonly:
        wandb.init(project='TC', name=args.wandb_name)
        wandb.log({'# of parameters': total_params})

    x1, x2, x3, x4 = gen_data_loop(args, c_p)
    x1, x2, x3, x4 = x1.astype(bool), x2.astype(bool), x3.astype(bool), x4.astype(bool)

    start = time.time()
    best_acc = 0.0
    best_loss = np.float64('inf')

    x, y = gen_data_sample(args.set, x1, x2, Nt)
    train_data = training_set(x, y)
    del x, y
    train_loader = DataLoader(dataset = train_data, batch_size=args.batchsize, shuffle=True)
    xt, yt = gen_testdata_sample(args.set, x3, x4, Nt)
    test_data = testing_set(xt, yt)
    del xt, yt
    test_loader = DataLoader(dataset = test_data, batch_size=args.batchsize, shuffle=True)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    model.to(device)
    
    
    for epoch in range (total_epoch):
                
        if (total_epoch-epoch)%args.shuffle_epoch==0:
            x, y = gen_data_sample(args.set, x1, x2, Nt)
            print('data sampled')
            train_data = training_set(x, y)
            del x, y
            train_loader = DataLoader(dataset = train_data, batch_size=args.batchsize, shuffle=True)

        f = open(f'{args.saveprefix}/{args.wandb_name}/log_{model_name}.txt', 'a')

        running_loss = 0.0
        running_acc = 0.0
        test_loss = 0.0
        test_acc = 0.0
        test_acc0 = 0.0
        test_acc1 = 0.0
        for i,data in enumerate(train_loader):
            model.train()

            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model.module(inputs.float())
            del inputs
            optimizer.zero_grad()
            loss = criterion(outputs.squeeze(), labels.float().squeeze())
            if torch.isnan(loss).any(): break
            loss.backward()
            optimizer.step()
            
            loss.detach().cpu().numpy()
            
            running_loss += loss.item()
            pred_classes = torch.cat([1-outputs, outputs], axis=1).argmax(axis=1) 
            running_acc += torch.count_nonzero(torch.eq(pred_classes, labels.squeeze())).item()/len(labels)
        running_loss /= len(train_loader)
        running_acc /= len(train_loader)
        running_acc *= 100
        
        learningratearg = optimizer.param_groups[0]["lr"]
        scheduler.step()
        learningratearg = scheduler.get_last_lr()
        
        del pred_classes, loss
        torch.cuda.empty_cache()
        
        model.eval()
        if (total_epoch-epoch)%(args.shuffle_epoch)==0:
            xt, yt = gen_testdata_sample(args.set, x3, x4, Nt)
            test_data = testing_set(xt, yt)
            del xt, yt
            test_loader = DataLoader(dataset = test_data, batch_size=args.batchsize, shuffle=True)
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
            
            if m0.sum() == 0:
                test_acc0 += 0.0
            else: 
                test_acc0 += torch.count_nonzero(torch.eq(test_pred_classes[m0], test_labels.squeeze()[m0])).item()/len(test_labels.squeeze()[m0])
            if m1.sum() == 0:
                test_acc1 += 0.0
            else: 
                test_acc1 += torch.count_nonzero(torch.eq(test_pred_classes[m1], test_labels.squeeze()[m1])).item()/len(test_labels.squeeze()[m1])
        test_loss /= len(test_loader)
        test_acc /= len(test_loader)
        test_acc *= 100
        test_acc0 /= len(test_loader)
        test_acc0 *= 100
        test_acc1 /= len(test_loader)
        test_acc1 *= 100
        
        del test_pred_classes, loss

        if args.wandb_name:
            wandb.log({'train_loss': running_loss, 'train_acc': running_acc, 'val_loss': test_loss, 'val_acc': test_acc, "recall_0": test_acc0, "recall_1": test_acc1, "epoch": epoch})
        
        if best_acc < test_acc:
            torch.save(model.module.state_dict(), f'{args.saveprefix}/{args.wandb_name}/model_a_{model_name}.pth')
            best_acc = test_acc
        if best_loss > test_loss:
            torch.save(model.module.state_dict(), f'{args.saveprefix}/{args.wandb_name}/model_l_{model_name}.pth')
            best_loss = test_loss
            
            
        print(f'p_flip: {args.p1:.3f}vs{args.p2:.3f} - [#{epoch}/{total_epoch}], lr = {learningratearg}')
        print(f'Loss// Tr: {running_loss:.4f}, Ts: {test_loss:.4f}')
        print(f'Accy// Tr: {running_acc:.4f}%, Ts: {test_acc:.4f}%')
        print(f'Accy// s0: {test_acc0:.4f}%, s1: {test_acc1:.4f}%')
        
        torch.save(model.module.state_dict(), f'{args.saveprefix}/{args.wandb_name}/model_e_{model_name}.pth')

        np.savetxt(f, np.array([running_loss, test_loss, running_acc, test_acc, test_acc0, test_acc1]), newline=" ")
        f.write("\n")
        f.close()
        torch.cuda.empty_cache()

    print(f'Time: {time.time()-start} s, {(time.time()-start)/60} min, {(time.time()-start)/3600} hour \n\n')

if __name__ == "__main__":
    main()

