import torch
import random
import numpy as np
import os.path as osp
import pandas as pd


def init_seeds(args, deterministic=True):
    seed = args.seed
    args.cuda = torch.cuda.is_available()

    random.seed(seed)  # new
    np.random.seed(seed)
    torch.manual_seed(seed)  # for cpu seed
    if args.cuda:  # for gpu seed
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # new: all gpu

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def save_args(args_dict, out_dir):
    out_path = osp.join(out_dir, 'args_settings.txt')
    with open(out_path, 'w') as f:
        for arg, value in args_dict.items():
            f.writelines(arg + ': ' + str(value) + '\n')


def save_running_time(out_dir, run_time):
    out_path = osp.join(out_dir, 'args_settings.txt')
    with open(out_path, 'a') as f:
        f.writelines('\n' + f"Total running time: {run_time} hours" + '\n')


def get_gauge_timestamp_from_data(data_df, idx_test, timestamp):
    gauges = data_df["gauge"].values[idx_test]
    timestamps = np.array([timestamp]).repeat(len(idx_test))

    return gauges, timestamps


def save_csv_results(out_path, timestamp_list, gauge_list, labels_list, preds_list, multi_preds=False):
    timestamp_arr = np.concatenate(timestamp_list)
    gauge_arr = np.concatenate(gauge_list)
    real_rain = np.concatenate(labels_list)

    out_df = pd.DataFrame()
    out_df['timestamp'] = timestamp_arr
    out_df['gauge'] = gauge_arr
    out_df['rainfall'] = real_rain

    if not multi_preds:
        out_df['pred_rain'] = np.concatenate(preds_list)
    else:
        num_rets = len(preds_list)
        for i in range(num_rets):
            col_name = 'pred_rain_{}'.format(i)
            out_df[col_name] = np.concatenate(preds_list[i])

    out_df.to_csv(out_path, index=False)


