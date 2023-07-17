# GSI
The code and datasets are being released...

The code is for our paper ["Rainfall Spatial Interpolation with Graph Neural Networks"](https://link.springer.com/chapter/10.1007/978-3-031-30678-5_14) and this paper has been accepted by DASFAA 2023.

##  Datasets and Baselines
**For more dataset and baseline details, please refer to our latest work and its code repository: [SSIN](https://github.com/jlidw/SSIN).**

Two real-world hourly raingauge datasets, **HK** and **BW**, are collected and used in this paper. Download the processed datasets from [Google Drive](https://drive.google.com/drive/folders/1tiS5UjcspNKcWL8RA7J3PxqhwciR5Lg3) and place them in the `data` folder.

In the `baselines` folder of [SSIN](https://github.com/jlidw/SSIN), you can find the implementation of IDW, OK, TIN, and TPS:
* **IDW**: self-implementation.
* **OK**: by using [pykrige.ok.OrdinaryKriging](https://geostat-framework.readthedocs.io/projects/pykrige/en/stable/generated/pykrige.ok.OrdinaryKriging.html).
* **TIN**: by using [matplotlib.tri](https://matplotlib.org/stable/api/tri_api.html).
* **TPS**: by using [scipy.interpolate.Rbf](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.Rbf.html).

For GNN-based baselines, please refer to their original code: [KCN](https://github.com/tufts-ml/KCN) and [IGNNK](https://github.com/Kaimaoge/IGNNK).

## Instructions
`networks`:
* Include files about the network layers and the model architecture.
  
`postprocess`:
* Calculate the RMSE, MAE, and NSE for predicted results.

`preprocess`:
* `generate_adjs.py`: generate adjacency matrix for HK/BW dataset. 
* `preprocessing.py`: preprocess HK/BW dataset and general the `pkl` data for training/testing.

`utils`:
* `cfg.py`: build the argument function.
* `pytorchtools.py`: build the EarlyStopping function.
* `utils.py`: some useful functions.

`training_funcs.py`: 
* define training-related functions which will be called by the main function of "train_gcn.py" / "train_rc_by_gcn.py" / "train_rc_by_kriging.py".

`train_gcn.py`: 
* train the GSI model to perform spatial interpolation.

`train_rc_by_gcn.py`: 
* reload the trained GSI model for spatial interpolation and perform **residual correction by GSI**.

`train_rc_by_kriging.py`: 
* reload the trained GSI model for spatial interpolation and perform **residual correction by Kriging**.

## Citation
```
@inproceedings{li2023rainfall,
  title={Rainfall Spatial Interpolation with Graph Neural Networks},
  author={Li, Jia and Shen, Yanyan and Chen, Lei and Ng, Charles Wang Wai},
  booktitle={International Conference on Database Systems for Advanced Applications},
  pages={175--191},
  year={2023},
  organization={Springer}
}
```
