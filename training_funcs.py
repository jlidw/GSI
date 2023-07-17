import os
import time
import numpy as np
from pykrige.ok import OrdinaryKriging

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

import sys
sys.path.append('..')
from .utils.pytorchtools import EarlyStopping
from .preprocess.preprocessing import numpy_to_tensor
from .networks.models import GSI, GCN


def build_model(args, nfeat, nnodes):
    """Creates and initializes the output."""
    if args.model_type == "GCNMask":
        model = GSI(nfeat=nfeat,
                    nhid=args.hidden,
                    dropout=args.dropout,
                    nheads=args.nb_heads,
                    nnodes=nnodes,
                    mask_act=args.mask_act,
                    adj_norm=args.adj_norm,
                    sym_mask=args.sym_mask,
                    mask_dropout=args.mask_dropout)
    elif args.model_type == "GCN":
        model = GCN(nfeat=nfeat,
                    nhid=args.hidden,
                    dropout=args.dropout)
    else:
        raise NotImplementedError(f'Model type '
                                  f'`{args.model_type}` is not '
                                  f'defined')
    return model


def create_optimizers(args, model):
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=args.scheduler_f, patience=args.scheduler_p, verbose=False)

    return optimizer, scheduler


def get_loss_function(args):
    loss_func_list = []
    if args.loss == "mse":
        loss_func = nn.MSELoss()
        loss_func_list.append(loss_func)
    elif args.loss == "mae":
        loss_func = nn.L1Loss()
        loss_func_list.append(loss_func)
    elif args.loss == "smooth_l1":
        loss_func = nn.SmoothL1Loss()
        loss_func_list.append(loss_func)
    elif args.loss == "mae_mse" or args.loss == "mse_mae":
        loss_func1 = nn.L1Loss()
        loss_func2 = nn.MSELoss()
        loss_func_list.append(loss_func1)
        loss_func_list.append(loss_func2)

    return loss_func_list


def train(args, model, optimizer, scheduler, loss_func_list,
          features, adj, adj_I, labels, idx_train, model_path):
    # to track the training loss as the output trains
    train_losses = []
    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=args.patience, verbose=False, path=model_path)
    for epoch in range(1, args.epochs+1):
        t = time.time()
        model.train()
        optimizer.zero_grad()
        output = model(features, adj, adj_I)

        # Set Loss Function
        loss_train = 0
        for loss_func in loss_func_list:
            loss_train += loss_func(output[idx_train], labels[idx_train])

        loss_train.backward()
        optimizer.step()
        scheduler.step(loss_train)
        # record training loss
        train_losses.append(loss_train.item())

        # early_stopping needs the validation loss to check if it has decresed,
        # and if it has, it will make a checkpoint of the current output
        early_stopping(loss_train, model)

        if early_stopping.early_stop:
            # print("Early stopping: epoch = {}".format(epoch))
            break

    # load the last checkpoint with the best output
    model.load_state_dict(torch.load(model_path))
    return model


def test(model, features, adj, adj_I, labels, idx_train, idx_test):
    model.eval()
    with torch.no_grad():
        output = model(features, adj, adj_I)
        train_mse = F.mse_loss(output[idx_train], labels[idx_train])

    med_rain_field = labels.clone().detach()
    med_rain_field[idx_test] = output[idx_test]
    med_rain_field = med_rain_field.cpu().numpy()

    preds = output.cpu().numpy()

    return train_mse.item(), med_rain_field, preds


def load_path(args):
    model_dir = "{}/model".format(args.out_dir)
    ret_dir = "{}/result".format(args.out_dir)

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(ret_dir, exist_ok=True)

    if args.partial:
        ret_path = "{}/test_ret-{}_{}.csv".format(ret_dir, args.start_idx, args.end_idx)
    else:
        ret_path = "{}/test_ret.csv".format(ret_dir)

    return model_dir, ret_path


def run_one_graph(args, timestamp, adj, adj_I, features, labels,
                  idx_train, idx_test, model_dir, round_num=None, reload=False):
    if features.ndim == 1:
        features = features[:, np.newaxis]
    else:  # only use rainfall values as feature
        features = features[:, 0:1]

    adj, adj_I, features, labels, idx_train, idx_test = numpy_to_tensor(adj, adj_I,
                                                                        features, labels, idx_train, idx_test)

    # model path
    if round_num is not None:
        model_path = model_dir + "/" + timestamp + "_checkpoint_{}.pt".format(round_num)
    else:
        model_path = model_dir + "/" + timestamp + "_checkpoint.pt"

    # Model and optimizer
    nfeat, nnodes = features.shape[1], features.shape[0]
    model = build_model(args, nfeat, nnodes)

    if args.cuda:
        model.cuda()
        features = features.cuda()
        adj = adj.cuda()
        adj_I = adj_I.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_test = idx_test.cuda()

    optimizer, scheduler = create_optimizers(args, model)
    loss_func_list = get_loss_function(args)

    # Training
    if reload:
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path))
            # print("Reloaded Model!")
        else:
            raise FileNotFoundError("Can not find model!", model_path)
    else:
        t_total = time.time()
        model = train(args, model, optimizer, scheduler, loss_func_list, features, adj, adj_I, labels, idx_train, model_path)
        # print("Training time: {:.4f}s".format(time.time() - t_total))

    # Testing
    train_mse, med_rain_field, preds = test(model, features, adj, adj_I, labels, idx_train, idx_test)
    # print(timestamp + " is done!" + "\n")

    return train_mse, med_rain_field, preds


def get_error_by_kriging(lats, lons, errors, idx_train, idx_test, variogram="spherical"):
    OK = OrdinaryKriging(lons[idx_train], lats[idx_train], errors[idx_train],  # lons, lats, data
                         variogram_model=variogram,
                         coordinates_type="geographic",
                         weight=False,
                         verbose=False,
                         enable_plotting=False)
    z_values, sigma = OK.execute('points', lons[idx_test], lats[idx_test])

    return z_values

