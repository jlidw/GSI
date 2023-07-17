import os.path as osp
import pandas as pd
from glob import glob
from .eval_methods import calc_scores


def print_scores(df, label_col, pred_col):
    df.loc[df[pred_col] < 0, pred_col] = 0  # if prediction < 0, reset it to zero.

    predicts = df[pred_col].values
    labels = df[label_col].values
    rmse, mae, nse, = calc_scores(predicts, labels)
    print("[Tot Result] - RMSE: {:.4f}, MAE: {:.4f}, NSE: {:.4f}".
          format(rmse, mae, nse))


def loop_results(ret_dir, thr=0):
    path_list = sorted(glob(ret_dir))

    for _path in path_list:
        if osp.isdir(_path):
            ret_path = glob(osp.join(_path, "*.csv"))[0]
        else:
            ret_path = _path

        print(ret_path)
        ret_df = pd.read_csv(ret_path)
        print("Length: ", len(ret_df))

        pred_col = "pred_rain"
        ret_df[pred_col].fillna(0, inplace=True)  # replace the invalid NaN with 0 for TIN
        ret_df[pred_col] = ret_df[pred_col].astype(float)

        if "rainfall" in ret_df.columns.to_list():
            label_col = "rainfall"
        elif "label" in ret_df.columns.to_list():
            label_col = "label"
        else:
            raise TypeError("Wrong label name!")

        ret_df = ret_df[ret_df[label_col] >= thr]
        col_names = [c for c in ret_df.columns.values.tolist() if "pred" in c]
        for _col in col_names:
            print(_col)
            print_scores(ret_df, label_col, _col)


if __name__ == "__main__":
    threshold = 0

    # HK results
    # base_dir = "../output/GSI/HK_data/*/result/test_ret.csv"

    # BW results
    base_dir = "../output/GSI/BW_data/*/result/test_ret.csv"

    print("threshold:", threshold)
    for _dir in [base_dir]:
        loop_results(_dir, thr=threshold)
