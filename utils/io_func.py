import os
import re
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tifffile as tiff
import torch


def _assert_suffix_match(suffix, path):
    assert re.search(r"\.{}$".format(suffix), path), "suffix mismatch"


def make_parent_dir(filepath):
    parent_path = os.path.dirname(filepath)
    if not os.path.isdir(parent_path):
        try:
            os.mkdir(parent_path)
        except FileNotFoundError:
            make_parent_dir(parent_path)
            os.mkdir(parent_path)
        print("[INFO] Make new directory: '{}'".format(parent_path))


def save_to_csv(data, path, header=None, index=None):
    _assert_suffix_match("csv", path)
    make_parent_dir(path)
    pd.DataFrame(data).to_csv(path, header=header, index=index)
    print("[INFO] Save as csv: '{}'".format(path))


def load_from_csv(path, header=None, index_col=None):
    return pd.read_csv(path, header=header, index_col=index_col)


def save_to_excel(data, writer, other_kws={}):
    if type(writer) == str:  # if path is ExcelWriter, skip this validation
        _assert_suffix_match("xlsx", writer)
        make_parent_dir(writer)
    else:
        _assert_suffix_match("xlsx", writer.path)
        make_parent_dir(writer.path)
    if not hasattr(data, "to_excel"):
        data = pd.DataFrame(data)
    data.to_excel(writer, **other_kws)
    print("[INFO] Save as excel: '{}'".format(
        writer if type(writer) == str else writer.path
    ))


def load_from_excel(path, header=[0], index_col=[0]):
    return pd.read_excel(path, header=header, index_col=index_col)


def save_to_pkl(data, path):
    _assert_suffix_match("pkl", path)
    make_parent_dir(path)
    with open(path, "wb") as f:
        pickle.dump(data, f, protocol=-1)
    print("[INFO] Save as pkl: '{}'".format(path))


def load_from_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_to_npy(data, path):
    _assert_suffix_match("npy", path)
    make_parent_dir(path)
    np.save(path, data)
    print("[INFO] Save as npy: '{}'".format(path))


def load_from_npy(path):
    return np.load(path)


def save_to_pth(data, path, model=True):
    _assert_suffix_match("pth", path)
    make_parent_dir(path)
    if model:
        if hasattr(data, "module"):
            data = data.module.state_dict()
        else:
            data = data.state_dict()
    torch.save(data, path)
    print("[INFO] Save as pth: '{}'".format(path))


def load_from_pth(path):
    return torch.load(path)


def save_to_tiff(data, path):
    _assert_suffix_match("tiff?", path)
    make_parent_dir(path)
    tiff.imsave(path, data)
    print("[INFO] Save as tiff: '{}'".format(path))


def load_from_tiff(path):
    return tiff.imread(path)


def savefig_png(path, dpi=150):
    _assert_suffix_match("png", path)
    make_parent_dir(path)
    plt.savefig(path, bbox_inches="tight", dpi=dpi)
    print("[INFO] Save figure as png: '{}'".format(path))


def savefig_eps(path):
    _assert_suffix_match("eps", path)
    make_parent_dir(path)
    plt.savefig(path, bbox_inches="tight")
    print("[INFO] Save figure as eps: '{}'".format(path))


def saveimg_png(data, path, dpi=150):
    _assert_suffix_match("png", path)
    make_parent_dir(path)
    plt.imsave(fname=path, arr=data, dpi=dpi)
    print("[INFO] Save image as png: '{}'".format(path))
