import numpy as np
import glob
import re
import h5py
from tqdm import tqdm
import os




def load_Cartesian_coord():
    '''return <N,h,w,c_coord>'''
    h = w = 64
    H = W = 640
    x_coord = np.linspace(0, H, h)
    y_coord = np.linspace(0, W, w)
    xx, yy = np.meshgrid(x_coord, y_coord, indexing='ij')
    xy_coord = np.stack((xx, yy), axis=-1)
    assert xy_coord.shape == (h, w, 2), f"Expected coord shape ({h}, {w}, 2), but got {xy_coord.shape}"
    return xy_coord.astype(np.float32)

def load_Norway_coord():
    '''return <N,h,w,c_coord>'''
    h = 64
    w = 118
    H = 3.2 * 1000
    W = 5.9 * 1000
    x_coord = np.linspace(0, H, h)
    y_coord = np.linspace(0, W, w)
    xx, yy = np.meshgrid(x_coord, y_coord, indexing='ij')
    xy_coord = np.stack((xx, yy), axis=-1)
    assert xy_coord.shape == (h, w, 2), f"Expected coord shape ({h}, {w}, 2), but got {xy_coord.shape}"
    return xy_coord.astype(np.float32)



def load_Cartesian_all():
    '''return <N,t,h,w,3>'''
    if os.path.exists("Dataset/Cartesian.npy"):
        print("- npy file found, loading from npy...")
        data = np.load("Dataset/Cartesian.npy")
        assert len(data.shape) == 5 and data.shape[-1] == 3, f"Expected data shape (N, t, h, w, 3), but got {data.shape}"
        print(f"- Loading pressure, saturation and permeability data done, shape: {data.shape}")
        return data


def load_Norway_all():
    '''return <N,t,h,w,3>'''
    if os.path.exists("Dataset/Norway.npy"):
        print("- npy file found, loading from npy...")
        data = np.load("Dataset/Norway.npy")
        assert len(data.shape) == 5 and data.shape[-1] == 3, f"Expected data shape (N, t, h, w, 3), but got {data.shape}"
        print(f"- Loading pressure, saturation and permeability data done, shape: {data.shape}")
        return data


def load_Topography_unstructured_all():
    '''return <N,t,m,3>'''
    if os.path.exists("Dataset/Topography_unstructured.npy"):
        print("- npy file found, loading from npy...")
        data = np.load("Dataset/Topography_unstructured.npy")
        assert len(data.shape) == 4 and data.shape[-1] == 4, f"Expected data shape (N, t, m, 4), but got {data.shape}"
        print(f"- Loading pressure, saturation, thickness, and z_center data done, shape: {data.shape}")
        return data


