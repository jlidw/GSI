import argparse


def get_default_args():
    """
    Build Default Arguments
    """
    parser = argparse.ArgumentParser()

    # dataset dir: HK data
    parser.add_argument('--dataset', type=str, default="hk")  # hk, bw
    parser.add_argument('--data_dir', type=str, default="./data")
    parser.add_argument('--adj_type', type=str, default="idw_power2_50th")

    parser.add_argument('--paras_num', type=int, default=1)
    parser.add_argument('--ablation_mode', type=int, default=0)

    # how many data are used
    parser.add_argument('--partial', dest='partial', action='store_true', default=False)  # only use part of data in this dir
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--end_idx', type=int, default=1000)

    # output dir
    parser.add_argument('--out_dir', type=str, default="./output")

    # GPU and seed
    parser.add_argument('--gpu_id', type=str, default="0", help='CUDA Visible Devices.')
    parser.add_argument('--seed', type=int, default=1, help='Random seed.')

    # mask params
    parser.add_argument('--mask_act', type=str, default="ELU")  # ELU
    parser.add_argument('--adj_norm', type=str, default="norm_1")  # norm_1
    parser.add_argument('--sym_mask', type=int, default=1)  # sym
    parser.add_argument('--mask_dropout', type=int, default=0)  # 1: True, 0: False

    parser.add_argument('--loss', type=str, default="mse")  # mae, mse, mae_mse, smooth_l1
    parser.add_argument('--scheduler_f', type=float, default=0.5, help='scheduler factor.')
    parser.add_argument('--scheduler_p', type=int, default=10, help='scheduler patience.')

    # network settings
    parser.add_argument('--model_type', type=str, default="GSI")  # GSI, GCN
    parser.add_argument('--hidden', type=int, default=16, help='Number of hidden units.')
    parser.add_argument('--nb_heads', type=int, default=16, help='Number of head attentions.')
    parser.add_argument('--dropout', type=float, default=0.2117675730615054,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--lr', type=float, default=0.011951044119200985, help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=1.7940017919112674e-05,
                        help='Weight decay (L2 loss on parameters).')

    # training settings
    parser.add_argument('--epochs', type=int, default=500, help='Number of epochs to train.')
    parser.add_argument('--patience', type=int, default=100, help="For EarlyStop")
    return parser

