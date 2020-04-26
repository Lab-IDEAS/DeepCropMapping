# DeepCropMapping: A multi-temporal deep learning approach with improved spatial generalizability for dynamic corn and soybean mapping

This responsitory is the official implementation of DeepCropMapping: A multi-temporal deep learning approach with improved spatial generalizability for dynamic corn and soybean mapping.

## Requirements

- torch
- numpy
- pandas
- scikit-learn
- jupyter

The code has been tested in the following environment:
Ubuntu 16.04.4 LTS, Python 3.5.2, PyTorch 1.2.0

## Data

The preprocessed data (`.npy` files) for model training and evaluation is directly available from the corresponding author upon requests. The preprocessed data should be stored in the `preprocessing/out` folder that has the following structure:

```
preprocessing/out
├── Site_A
│   ├── x-2015.npy
│   ├── y-2015.npy
│   ├── . . .
│   ├── x-2018.npy
│   └── y-2018.npy
├── Site_B
├── . . .
└── Site_F
```

You can also download raw Landsat Analysis Ready Data (ARD) from [EarthExplore](https://earthexplorer.usgs.gov/) and raw Cropland Data Layer (CDL) from [CropScape](https://nassgeodata.gmu.edu/CropScape/), then follow the code in the `preprocessing` folder to generate the `.npy` files. The raw Landsat ARD and CDL data should be stored in a new `data` folder that has the following structure (specific downloaded file names may change):

```
data
├── Site_A
│   ├── ARD
│   │   ├── 2015
│   │   │   ├── LC08_CU_018007_20150424_20181206_C01_V01_PIXELQA.tif
│   │   │   ├── LC08_CU_018007_20150424_20181206_C01_V01_SRB2.tif
│   │   │   └── . . .
│   │   ├── . . .
│   │   └── 2018
│   └── CDL
│       ├── CDL_2015_clip_20190409130240_375669680.tif
│       ├── . . .
│       └── CDL_2018_clip_20190409125506_12566268.tif
├── Site_B
├── . . .
└── Site_F
```

## Training and evaluation

- The PyTorch implementation of DeepCropMapping (DCM) model is located in the `models` folder.
- The `utils` folder contains some utilities that are used for data loading, normalization, training and evluation.

The specific training and evaluation process can be executed by running the `.ipynb` files in the `expriment` folder.

The hyperparameters for different sites in the paper are set as follows:

| Hyperparameter | Site A | Site B | Site C | Site D | Site E | Site F |
| --- | --- | --- | --- | --- | --- | --- |
|Dimension of LSTM hidden features | 256 | 512 | 256 | 512 | 256 | 256 |
| Number of LSTM layers | 2 | 2 | 2 | 2 | 2 | 3 |
