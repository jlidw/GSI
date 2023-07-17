import tqdm
import sys
sys.path.append('..')  # import the upper directory of the current file into the search path
from .preprocess.preprocessing import load_pkl_data, SelfStandardScaler
from .utils.cfg import get_default_args
from .utils.utils import *
from .training_funcs import *


def main(args):
    # Load data
    model_dir, ret_path = load_path(args)
    data_dict, data_num = load_pkl_data(args.data_path)

    feature_list = data_dict["features"]
    label_list = data_dict["labels"]
    train_idx_list = data_dict["train_idx"]
    test_idx_list = data_dict["test_idx"]
    adj_mat1_list = data_dict["adj_mat1"]
    adj_mat2_list = data_dict["adj_mat2"]

    # for generating test results
    gauge_list = data_dict["gauges"]
    timestamp_list = data_dict["timestamps"]

    if args.partial:
        idx_s = args.start_idx
        idx_e = min(args.end_idx, data_num)
    else:
        idx_s = 0
        idx_e = data_num

    i_iter = tqdm.tqdm(range(idx_s, idx_e),
                       desc="Training: ",
                       total=idx_e-idx_s,
                       bar_format="{l_bar}{r_bar}")

    test_preds_list, test_labels_list = [], []
    for i in i_iter:
        features, labels, idx_train, idx_test = feature_list[i], label_list[i], train_idx_list[i], test_idx_list[i]
        adj_mat1, adj_mat2 = adj_mat1_list[i], adj_mat2_list[i]
        timestamp = timestamp_list[i][0]

        scaler = SelfStandardScaler(mean=labels[idx_train].mean(),
                                    std=labels[idx_train].std())  # use stats of training nodes

        features = features[:, 0]  # only rain values
        nom_features = scaler.transform(features)  # standardize features
        # nom_features[idx_test] = 0  # fixed: don't need reset to 0
        nom_labels = scaler.transform(labels)  # standardize labels

        if args.ablation_mode == 0:  # no ablation; GSI model
            pass
        elif args.ablation_mode == 1:  # initial adj matrix
            adj_mat1[adj_mat1 != 0] = 1
            adj_mat2[adj_mat2 != 0] = 1
        elif args.ablation_mode == 2:
            adj_mat1 = adj_mat2.copy()

        train_mse, med_rain_field, preds = run_one_graph(args, timestamp, adj_mat1, adj_mat2, nom_features, nom_labels,
                                                         idx_train, idx_test, model_dir)

        preds = scaler.inverse_transform(preds)
        test_preds = preds[idx_test]

        test_preds_list.append(test_preds)
        test_labels_list.append(labels[idx_test])

    test_gauge_list = gauge_list[idx_s: idx_e]
    test_timestamp_list = timestamp_list[idx_s: idx_e]
    save_csv_results(ret_path, test_timestamp_list, test_gauge_list, test_labels_list, test_preds_list)


if __name__ == '__main__':
    parser = get_default_args()
    args = parser.parse_args()
    args.out_dir = "./output"

    if args.dataset.lower() == "hk":
        args.paras_num = 1
        prefix = "HK_data"
        args.adj_type = "idw_power2_50th"
        args.data_dir = f"{args.data_dir}/HK_123_Data/pkl_data"
    elif args.dataset.lower() == "bw":
        args.paras_num = 2
        prefix = "BW_data"
        args.adj_type = "idw_power2_75th"
        args.data_dir = f"{args.data_dir}/BW_132_Data/pkl_data"
    else:
        raise NotImplementedError

    if args.paras_num == 1:
        # hyper-parameters for HK dataset
        args.lr = 0.01242280373341682
        args.weight_decay = 3.0189717208257073e-06
        args.dropout = 0.3871241027778284
        args.hidden = 8
        args.nb_heads = 16
    elif args.paras_num == 2:
        # hyper-parameters for BW dataset
        args.lr = 0.0030759392298867283
        args.weight_decay = 4.540839696209309e-05
        args.dropout = 0.3514742622380771
        args.hidden = 4
        args.nb_heads = 4

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    init_seeds(args)

    args.data_path = "{}/{}_{}.pkl".format(args.data_dir, prefix, args.adj_type)
    args.out_dir = f"{args.out_dir}/{args.model_type}/{prefix}/{args.adj_type}"

    os.makedirs(args.out_dir, exist_ok=True)
    save_args(args.__dict__, args.out_dir)

    start_time = time.time()
    main(args)

    run_time = round((time.time() - start_time) / 3600, 2)  # hour
    save_running_time(args.out_dir, run_time)


