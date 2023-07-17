import numpy as np
import os
import pandas as pd
from geographiclib.geodesic import Geodesic


def calc_adj_mat(info_path, out_dir, out_name, dist_type="idw", power=2, thr_rate=50):
    """
    Calculate adjacency matrix based on distance

    Args:
        thr_rate: percentile, determine sparsity of the graph
        dist_type: how to measure the distance, 'IDW' or 'Gaussian'
        order: power num for IDW, 1 or 2

    Returns: adjacency matrix

    """
    df = pd.read_csv(info_path)

    lons, lats = df["lon"].values, df["lat"].values
    dist_mat = np.zeros((len(lons), len(lons)))

    for i in range(len(lons)):
        for j in range(len(lons)):
            dist = Geodesic.WGS84.Inverse(lats[i], lons[i], lats[j], lons[j])
            dist_mat[i, j] = dist["s12"] / 1000.0  # km

    if dist_type.lower() == "idw":
        adj_mat = np.power(dist_mat, -1*power)
        adj_mat[np.isinf(adj_mat)] = 0.  # reset inf (self-connections) as 0
    elif dist_type.lower() == "gaussian":
        var = np.var(dist_mat)
        adj_mat = np.exp(-np.square(dist_mat) / var)  # normalize to [0, 1]
    else:
        raise NotImplementedError

    thr = np.percentile(dist_mat, thr_rate)
    print("thr_rate:", thr_rate, "Threshold distance:", thr)

    adj_mat[np.where(dist_mat > thr)] = 0  # if dist>thr, set to 0
    adj_mat = adj_mat.astype(np.float32)

    np.save(f"{out_dir}/{out_name}.npy", adj_mat)
    print(f"Save {out_dir}/{out_name}.npy!")


if __name__ == "__main__":
    base_dir = "../../../../data"

    dist_type = "idw"  # idw, Gaussian
    power = 2
    thr_rates = [50, 75]

    # HK dataset
    # dataset_dir = f"{base_dir}/HK_123_data"
    # info_path = f"{dataset_dir}/hk_stations_info.csv"

    # BW dataset
    dataset_dir = f"{base_dir}/BW_132_data"
    info_path = f"{dataset_dir}/bw_valid_stations_info.csv"

    out_dir = f"{dataset_dir}/adjs"
    os.makedirs(out_dir, exist_ok=True)
    for thr_rate in thr_rates:
        if dist_type.lower() == "idw":
            out_name = f"{dist_type.lower()}_power{power}_{thr_rate}th"
        else:
            out_name = f"{dist_type.lower()}_{thr_rate}th"

        calc_adj_mat(info_path, out_dir, out_name, dist_type=dist_type, power=power, thr_rate=thr_rate)
