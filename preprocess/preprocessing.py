import os
import os.path as osp
import numpy as np
import pandas as pd
import torch
import scipy.sparse as sp
from glob import glob
import pickle
import tqdm


def gen_avail_mask(df):
    """ Fixed: replace str and negative_num as np.nan by a simpler method """
    df = df.apply(pd.to_numeric, errors='coerce')  # convert each element to numeric: if error, fill nan
    df[df < 0] = np.nan  # replace negative num with nan
    nan_mask = df.isna().values  # nan values
    avail_mask = np.logical_not(nan_mask)
    return avail_mask


def read_adj_mat(adj_path):
    adj_mat = np.load(adj_path)
    return adj_mat.astype(np.float32)


def read_train_test_split(file_path):
    if isinstance(file_path, pd.DataFrame):
        gauge_df = file_path
    elif osp.exists(file_path):
        gauge_df = pd.read_csv(file_path)
    else:
        raise FileNotFoundError("'{}' does not exist!".format(file_path))

    is_test = gauge_df["is_test"].values
    train_mask = np.where(is_test == 0, True, False)
    test_mask = np.where(is_test == 1, True, False)
    return train_mask, test_mask


def create_one_data(timestamp, info_df, data_path, adj_mat):
    if isinstance(data_path, pd.DataFrame):
        df = data_path
    else:
        df = pd.read_csv(data_path)

    _df = df.rename(columns={'gauge': 'station'})
    assert np.all(_df["station"].values == info_df["station"].values)

    data_df = pd.merge(info_df, _df, how="left", on=["station"])
    avail_mask = gen_avail_mask(data_df["rainfall"])

    avail_df = data_df.loc[avail_mask, :].reset_index(drop=True).copy()  # Delete the invalid nodes
    avail_df["rainfall"] = avail_df["rainfall"].astype(np.float32)

    is_test = avail_df["is_test"].values
    train_mask = np.where(is_test == 0, True, False)
    test_mask = np.where(is_test == 1, True, False)
    idx_train = np.where(train_mask)[0]
    idx_test = np.where(test_mask)[0]

    labels = avail_df["rainfall"].values.copy()
    features = avail_df[["rainfall"]].values.copy()
    if features.ndim == 1:
        features = np.expand_dims(features, axis=1)

    features[idx_test, :1] = 0  # set test nodes' rainfall to zero
    assert np.any(features != labels)

    # delete invalid nodes from adj matrix
    indexes = np.where(avail_mask)[0]
    idx_i, idx_j = np.ix_(indexes, indexes)
    adj_mat = adj_mat[idx_i, idx_j]

    adj_mat1, adj_mat2 = process_adj(adj_mat, idx_test)

    test_gauges = avail_df["station"].values[idx_test]
    test_timestamps = np.array([timestamp]).repeat(len(idx_test))

    lats = avail_df["lat"].values.astype(float)
    lons = avail_df["lon"].values.astype(float)

    return features, labels, idx_train, idx_test, adj_mat1, adj_mat2, test_gauges, test_timestamps, lats, lons


def process_adj(adj_mat, idx_test):
    diag = np.diag(adj_mat)
    adj_diag = np.diag(diag)

    adj_no_I = adj_mat - adj_diag  # adj without self-loop

    adj_mat1 = adj_no_I.copy()
    adj_mat1[:, idx_test] = 0  # cut off the edges from test nodes to training nodes
    adj_mat1 = sp.coo_matrix(adj_mat1)
    adj_mat1 = normalize(adj_mat1)  # adj = D^-1(A), asymmetric

    adj_mat2 = adj_no_I.copy()
    adj_mat2 = (adj_mat2 + adj_mat2.T) / 2  # should be symmetric based on distance
    adj_mat2 = sp.coo_matrix(adj_mat2)

    adj_mat2 = normalize(adj_mat2)  # adj = D^-1(A); should be asymmetric
    adj_mat2 = normalize(adj_mat2 + sp.eye(adj_mat2.shape[0]))  # add self-loop, then normalize the weights again

    return adj_mat1.toarray(), adj_mat2.toarray()  # convert coo_matrix to array


def numpy_to_tensor(adj_1, adj_2, features, labels, idx_train, idx_test):
    adj_1 = torch.FloatTensor(adj_1)
    adj_2 = torch.FloatTensor(adj_2)

    features = torch.FloatTensor(features)  # all features including rainfall
    labels = torch.FloatTensor(labels)  # rainfall values as labels

    idx_train = torch.LongTensor(idx_train)
    idx_test = torch.LongTensor(idx_test)

    return adj_1, adj_2, features, labels, idx_train, idx_test


def normalize(mx):  # 这里是计算D^-1A，而不是计算论文中的D^-1/2AD^-1/2
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.  # 将数组中无穷大的元素置0处理
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sym_normalize(adj):  # D^-1/2AD^-1/2
    """Symmetrically normalize adjacency matrix."""
    # adj = adj.to_dense().cpu().numpy()
    # adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return adj


class SelfStandardScaler:
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


class SelfMinMaxScaler:
    """
    Normalize the input
    """

    def __init__(self, max_value, min_value):
        self.max = max_value
        self.min = min_value

    def transform(self, data):
        return (data - self.min) / (self.max - self.min)

    def inverse_transform(self, data):
        return (data * (self.max - self.min)) + self.min


def generate_pkl_data(info_path, avail_time_path, adj_mat_path, data_dir, out_dir, out_name):
    info_df = pd.read_csv(info_path)
    avail_timestamps = np.load(avail_time_path)
    adj_mat = read_adj_mat(adj_mat_path)

    file_paths = sorted(glob(data_dir))
    feature_list, label_list, train_idx_list, test_idx_list = [], [], [], []
    adj_mat1_list, adj_mat2_list, gauge_list, timestamp_list = [], [], [], []
    lat_list, lon_list = [], []

    file_iter = tqdm.tqdm(enumerate(file_paths),
                          desc="Generating data: ",
                          total=len(file_paths),
                          bar_format="{l_bar}{r_bar}")

    for i, _ in file_iter:
        _path = file_paths[i]

        timestamp = _path.split("/")[-1][:-4]
        if timestamp not in avail_timestamps:
            continue

        cur_data = create_one_data(timestamp, info_df, _path, adj_mat)
        features, labels, idx_train, idx_test, adj_mat1, adj_mat2, test_gauges, test_timestamps, lats, lons = cur_data

        feature_list.append(features)
        label_list.append(labels)
        train_idx_list.append(idx_train)
        test_idx_list.append(idx_test)
        adj_mat1_list.append(adj_mat1)
        adj_mat2_list.append(adj_mat2)
        gauge_list.append(test_gauges)
        timestamp_list.append(test_timestamps)
        lat_list.append(lats)
        lon_list.append(lons)

    print("Data num", len(feature_list))

    data_dict = {}
    data_dict["features"] = feature_list
    data_dict["labels"] = label_list
    data_dict["train_idx"] = train_idx_list
    data_dict["test_idx"] = test_idx_list
    data_dict["adj_mat1"] = adj_mat1_list
    data_dict["adj_mat2"] = adj_mat2_list

    # for generating test results
    data_dict["gauges"] = gauge_list
    data_dict["timestamps"] = timestamp_list

    # for residual correction by Kriging
    data_dict["lats"] = lat_list
    data_dict["lons"] = lon_list

    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/{out_name}.pkl", "wb") as fp:
        pickle.dump(data_dict, fp)

    print(f"Save {out_dir}/{out_name}.pkl!")


def load_pkl_data(path):
    with open(path, "rb") as fp:
        data_dict = pickle.load(fp)

    data_num = len(data_dict["features"])

    return data_dict, data_num


if __name__ == "__main__":
    base_dir = "../data"
    dist_type = "idw"
    power = 2

    dataset = "hk"
    if dataset.lower() == "hk":
        thr_rate = 50
        dataset_dir = f"{base_dir}/HK_123_data"
        info_path = f"{dataset_dir}/hk_stations_info.csv"
        avail_time_path = f"{dataset_dir}/2008-2012_avail_used_timestamps.npy"
        data_dir = f"{dataset_dir}/rain_csv/*/*.csv"
        adj_mat_dir = f"{dataset_dir}/adjs"
        out_dir = "../data/HK_123_Data/pkl_data"
        out_name = "HK_data"
    elif dataset.lower() == "bw":
        thr_rate = 75
        dataset_dir = f"{base_dir}/BW_132_data"
        info_path = f"{dataset_dir}bw_stations_info.csv"
        avail_time_path = f"{dataset_dir}/2012-2014_avail_used_timestamps.npy"
        adj_mat_dir = f"{dataset_dir}/adjs"
        data_dir = f"{dataset_dir}/rain_csv/*/*.csv"
        out_dir = "../data/BW_132_Data/pkl_data"
        out_name = "BW_data"
    else:
        raise NotImplementedError

    adj_name = f"{dist_type.lower()}_power{power}_{thr_rate}th"
    adj_mat_path = f"{adj_mat_dir}/{adj_name}.npy"
    cur_out_name = f"{out_name}_{adj_name}"

    generate_pkl_data(info_path, avail_time_path, adj_mat_path, data_dir, out_dir, cur_out_name)

