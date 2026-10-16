import numpy as np
import glob
import re
import h5py
import gstools as gs
from gstools.random import MasterRNG
from tqdm import tqdm
import os

# def load_elbow_flow(path):
#     return np.load(f"{path}")[1:]

# def load_channel_flow(
#     path,
#     t_start=0,
#     t_end=1200,
#     t_every=1,
# ):
#     return np.load(f"{path}")[t_start:t_end:t_every]

# def load_periodic_hill_flow(path):
#     data = np.load(f"{path}")
#     return data

# def load_3d_flow(path):
#     data = np.load(f"{path}")
#     return data


def generate_random_fields(case_num):
    # generate random permeability fields
    gs_seed = MasterRNG(case_num * 123 + 321)
    color_style = 'jet'
    x = y = range(64)
    x_len, y_len = 20, 20

    model = gs.Exponential(dim=2, var=0.7, len_scale=[x_len, y_len])

    srf = gs.SRF(model, mean=np.log(100), seed=gs_seed())

    field = srf.structured([x, y])
    data_array = srf.post_field(field=field, process=False)

    data_processed_1 = np.exp(data_array)
    data_processed_2 = data_processed_1 * 1.0e-15
    return data_array, data_processed_2

def generate_random_fields_gaussian(case_num):
    # generate random permeability fields
    gs_seed = MasterRNG(case_num * 123 + 321)
    color_style = 'jet'
    x = y = range(64)
    x_len, y_len = 8, 8

    model = gs.Gaussian(dim=2, var=1.0, len_scale=[x_len, y_len])

    srf = gs.SRF(model, mean=np.log(100), seed=gs_seed())

    field = srf.structured([x, y])
    data_array = srf.post_field(field=field, process=False)

    data_processed_1 = np.exp(data_array)
    data_processed_2 = data_processed_1 * 1.0e-15
    return data_array, data_processed_2


def load_Cartesian_coord():
    '''return <N,h,w,c_coord>'''
    h = w = 64
    H = W = 640
    # h, w = 64, 63
    # H, W = 640, 630
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

def load_unstructured_coord():
    '''return <N,m,c_coord>'''
    with h5py.File("Dataset/Norway_topography_unstructured.hdf5", "r") as f:
        xy_coord = f["coordinates"][0].astype(np.float32)
    assert len(xy_coord.shape) == 2, f"Expected coord shape (m, 2), but got {xy_coord.shape}"
    return xy_coord.astype(np.float32)

def load_unstructured_coord_Egg():
    '''return <N,m,c_coord>'''
    with h5py.File("Dataset/Egg2d_channelized.hdf5", "r") as f:
        xy_coord = f["coordinates"][0].astype(np.float32)
    assert len(xy_coord.shape) == 2, f"Expected coord shape (m, 2), but got {xy_coord.shape}"
    return xy_coord.astype(np.float32)




def load_Cartesian_pre_sat():
    '''return <N,t,h,w,1>'''
    print("Loading pressure and saturation data...")
    with h5py.File("Dataset/Cartesian_simple.hdf5", "r") as f:
        data_pressure = f["pressure"][:].astype(np.float32)
        data_saturation = f["saturation"][:].astype(np.float32)
    data_pressure = data_pressure[:500, :, :, :-1, None]
    data_saturation = data_saturation[:500, :, :, :-1, None]
    data = np.concatenate((data_pressure, data_saturation), axis=-1)
    assert len(data.shape) == 5 and data.shape[-1] == 2, f"Expected data shape (N, t, h, w, 2), but got {data.shape}"
    print(f"Loding pressure and saturation data done, shape: {data.shape}")
    return data

def load_Cartesian_pre():
    '''return <N,t,h,w,1>'''
    with h5py.File("Dataset/Cartesian_simple.hdf5", "r") as f:
        data = f["pressure"][:].astype(np.float32)
    data = data[:500, :32, :, :-1, None]
    assert len(data.shape) == 5, f"Expected data shape (N, t, h, w, 1), but got {data.shape}"
    return data


def load_Cartesian_sat():
    '''return <N,t,h,w,1>'''
    with h5py.File("Dataset/Cartesian_simple.hdf5", "r") as f:
        data = f["saturation"][:].astype(np.float32)
    data = data[:500, :32, :, :-1, None]
    assert len(data.shape) == 5, f"Expected data shape (N, t, h, w, 1), but got {data.shape}"
    return data


def load_Cartesian_all():
    '''return <N,t,h,w,3>'''
    if os.path.exists("Dataset/Cartesian_all3_small.npy"):
        print("- npy file found, loading from npy...")
        data = np.load("Dataset/Cartesian_all3_small.npy")
        assert len(data.shape) == 5 and data.shape[-1] == 3, f"Expected data shape (N, t, h, w, 3), but got {data.shape}"
        print(f"- Loading pressure, saturation and permeability data done, shape: {data.shape}")
        return data
    else:
        print("- Loading pressure, saturation and permeability data...")
        with h5py.File("Dataset/Cartesian_simple.hdf5", "r") as f:
            data1 = f["pressure"][:3000].astype(np.float32)
            data2 = f["saturation"][:3000].astype(np.float32)
            data3 = f["permeability_log"][:3000].astype(np.float32)
        print("-HDF5 data loaded, start reshaping...")
        # data1 = data1[:, :, :, :, None]
        # data2 = data2[:, :, :, :, None]
        data3 = data3[:, None, :, :]
        print("-HDF5 data reshaped, start broadcasting...")
        data3 = np.broadcast_to(data3, data1.shape)
        print("-HDF5 data broadcasted, start concatenating...")
        # data = np.empty((*data1.shape[:-1], 3), dtype=np.float32)
        # data[..., 0] = data1[..., 0]
        # data[..., 1] = data2[..., 0]
        # data[..., 2] = data3[..., 0]
        # data = np.concatenate((data1, data2, data3), axis=-1)
        data = np.stack((data1, data2, data3), axis=-1)
        np.save("Dataset/Cartesian_all3_med.npy", data)
        assert len(data.shape) == 5 and data.shape[-1] == 3, f"Expected data shape (N, t, h, w, 3), but got {data.shape}"
        print(f"- Loding pressure, saturation and permeability data done, shape: {data.shape}")
        return data

def load_Cartesian_all_test():
    '''return <N,t,h,w,3>'''
    if os.path.exists("Dataset/Cartesian_all3_small_test.npy"):
        print("- npy file found, loading from npy...")
        data = np.load("Dataset/Cartesian_all3_small_test.npy")
        assert len(data.shape) == 5 and data.shape[-1] == 3, f"Expected data shape (N, t, h, w, 3), but got {data.shape}"
        print(f"- Loading pressure, saturation and permeability data done, shape: {data.shape}")
        return data
    else:
        print("- Loading pressure, saturation and permeability data...")
        with h5py.File("Dataset/Cartesian_simple.hdf5", "r") as f:
            data1 = f["pressure"][5000:5100].astype(np.float32)
            data2 = f["saturation"][5000:5100].astype(np.float32)
            data3 = f["permeability_log"][5000:5100].astype(np.float32)
        print("-HDF5 data loaded, start reshaping...")
        # data1 = data1[:, :, :, :, None]
        # data2 = data2[:, :, :, :, None]
        data3 = data3[:, None, :, :]
        print("-HDF5 data reshaped, start broadcasting...")
        data3 = np.broadcast_to(data3, data1.shape)
        print("-HDF5 data broadcasted, start concatenating...")
        # data = np.empty((*data1.shape[:-1], 3), dtype=np.float32)
        # data[..., 0] = data1[..., 0]
        # data[..., 1] = data2[..., 0]
        # data[..., 2] = data3[..., 0]
        # data = np.concatenate((data1, data2, data3), axis=-1)
        data = np.stack((data1, data2, data3), axis=-1)
        np.save("Dataset/Cartesian_all3_small_test.npy", data)
        assert len(data.shape) == 5 and data.shape[-1] == 3, f"Expected data shape (N, t, h, w, 3), but got {data.shape}"
        print(f"- Loding pressure, saturation and permeability data done, shape: {data.shape}")
        return data
# def load_Cartesian_sat_film():
#     '''return <N,t,h,w,1>'''
#     with h5py.File("Dataset/Cartesian.hdf5", "r") as f:
#         data = f["saturation"][:].astype(np.float32)
#     data = data[:6500, ..., None]
#     assert len(data.shape) == 5, f"Expected data shape (N, t, h, w, 1), but got {data.shape}"
#     return data

def load_Norway_all():
    '''return <N,t,h,w,3>'''
    if os.path.exists("Dataset/Norway_all3.npy"):
        print("- npy file found, loading from npy...")
        data = np.load("Dataset/Norway_all3.npy")
        assert len(data.shape) == 5 and data.shape[-1] == 3, f"Expected data shape (N, t, h, w, 3), but got {data.shape}"
        print(f"- Loading pressure, saturation and permeability data done, shape: {data.shape}")
        return data
    else:
        print("- Loading pressure, saturation and permeability data...")
        with h5py.File("Dataset/Norway_3D.hdf5", "r") as f:
            data1 = f["pressure"][:2000].astype(np.float32)
            data2 = f["saturation"][:2000].astype(np.float32)
            data3 = f["permeability_log"][:2000].astype(np.float32)
        print("-HDF5 data loaded, start reshaping...")
        # data1 = data1[:, :, :, :, None]
        # data2 = data2[:, :, :, :, None]
        data3 = data3[:, None, :, :]
        print("-HDF5 data reshaped, start broadcasting...")
        data3 = np.broadcast_to(data3, data1.shape)
        print("-HDF5 data broadcasted, start concatenating...")
        # data = np.empty((*data1.shape[:-1], 3), dtype=np.float32)
        # data[..., 0] = data1[..., 0]
        # data[..., 1] = data2[..., 0]
        # data[..., 2] = data3[..., 0]
        # data = np.concatenate((data1, data2, data3), axis=-1)
        data = np.stack((data1, data2, data3), axis=-1)
        np.save("Dataset/Norway_all3.npy", data)
        assert len(data.shape) == 5 and data.shape[-1] == 3, f"Expected data shape (N, t, h, w, 3), but got {data.shape}"
        print(f"- Loding pressure, saturation and permeability data done, shape: {data.shape}")
        return data

def load_Topography_all():
    '''return <N,t,h,w,3>'''
    if os.path.exists("Dataset/Topography_all3.npy"):
        print("- npy file found, loading from npy...")
        data = np.load("Dataset/Topography_all3.npy")
        assert len(data.shape) == 5 and data.shape[-1] == 4, f"Expected data shape (N, t, h, w, 4), but got {data.shape}"
        print(f"- Loading pressure, saturation, thickness, and z_center data done, shape: {data.shape}")
        return data
    else:
        print("- Loading pressure, saturation, thickness, and z_center data...")
        with h5py.File("Dataset/Norway_topography.hdf5", "r") as f:
            data1 = f["pressure"][:2000].astype(np.float32)
            data2 = f["saturation"][:2000].astype(np.float32)
            data3 = f["thickness"][:2000].astype(np.float32)
            data4 = f["z_center"][:2000].astype(np.float32)
        print("-HDF5 data loaded, start reshaping...")
        data3 = data3[:, None, :, :]
        data4 = data4[:, None, :, :]
        print("-HDF5 data reshaped, start broadcasting...")
        data3 = np.broadcast_to(data3, data1.shape)
        data4 = np.broadcast_to(data4, data1.shape)
        print("-HDF5 data broadcasted, start concatenating...")
        data = np.stack((data1, data2, data3, data4), axis=-1)
        np.save("Dataset/Topography_all3.npy", data)
        assert len(data.shape) == 5 and data.shape[-1] == 4, f"Expected data shape (N, t, h, w, 4), but got {data.shape}"
        print(f"- Loding pressure, saturation, thickness, and z_center data done, shape: {data.shape}")
        return data


def load_Topography_unstructured_all():
    '''return <N,t,m,3>'''
    if os.path.exists("Dataset/Topography_unstructured_all3.npy"):
        print("- npy file found, loading from npy...")
        data = np.load("Dataset/Topography_unstructured_all3.npy")
        assert len(data.shape) == 4 and data.shape[-1] == 4, f"Expected data shape (N, t, m, 4), but got {data.shape}"
        print(f"- Loading pressure, saturation, thickness, and z_center data done, shape: {data.shape}")
        return data
    else:
        print("- Loading pressure, saturation, thickness, and z_center data...")
        with h5py.File("Dataset/Norway_topography_unstructured.hdf5", "r") as f:
            data1 = f["pressure"][:2000].astype(np.float32)
            data2 = f["saturation"][:2000].astype(np.float32)
            data3 = f["thickness"][:2000].astype(np.float32)
            data4 = f["z_center"][:2000].astype(np.float32)
        print("-HDF5 data loaded, start reshaping...")
        data3 = data3[:, None, ...]
        data4 = data4[:, None, ...]
        print("-HDF5 data reshaped, start broadcasting...")
        data3 = np.broadcast_to(data3, data1.shape)
        data4 = np.broadcast_to(data4, data1.shape)
        print("-HDF5 data broadcasted, start concatenating...")
        data = np.stack((data1, data2, data3, data4), axis=-1)
        np.save("Dataset/Topography_unstructured_all3.npy", data)
        assert len(data.shape) == 4 and data.shape[-1] == 4, f"Expected data shape (N, t, m, 4), but got {data.shape}"
        print(f"- Loding pressure, saturation, thickness, and z_center data done, shape: {data.shape}")
        return data


def load_Egg2d_channelized_all():
    '''return <N,t,m,3>'''
    if os.path.exists("Dataset/Egg2d_channelized_all3.npy"):
        print("- npy file found, loading from npy...")
        data = np.load("Dataset/Egg2d_channelized_all3.npy")
        assert len(data.shape) == 4 and data.shape[-1] == 3, f"Expected data shape (N, t, m, 3), but got {data.shape}"
        print(f"- Loading pressure, saturation, permeability data done, shape: {data.shape}")
        return data
    else:
        print("- Loading pressure, saturation, permeability data...")
        with h5py.File("Dataset/Egg2d_channelized.hdf5", "r") as f:
            data1 = f["pressure"][:2000].astype(np.float32)
            data2 = f["saturation"][:2000].astype(np.float32)
            data3 = f["permeability"][:2000].astype(np.float32) * 1.e15  # convert m2 to mD
        print("-HDF5 data loaded, start reshaping...")
        data3 = data3[:, None, :, 0]
        print("-HDF5 data reshaped, start broadcasting...")
        data3 = np.broadcast_to(data3, data1.shape)
        print("-HDF5 data broadcasted, start concatenating...")
        data = np.stack((data1, data2, data3), axis=-1)
        np.save("Dataset/Egg2d_channelized_all3.npy", data)
        assert len(data.shape) == 4 and data.shape[-1] == 3, f"Expected data shape (N, t, m, 3), but got {data.shape}"
        print(f"- Loding pressure, saturation, permeability data done, shape: {data.shape}")
        return data




